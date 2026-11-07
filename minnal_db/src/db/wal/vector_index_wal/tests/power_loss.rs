//! Power loss: what an fsync covered survives, and nothing else counts.

use super::*;

/// `lose_unsynced` and the power-loss image cut each segment at what an
/// fsync covered, not at the durable tail: an append that no sync covered
/// is gone, a synced one stays, and the file keeps its size.
#[test]
fn a_power_loss_keeps_exactly_what_an_fsync_covered() {
    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), 256);
    let appended = append_n(&wal, 4); // segment 0 sealed (fsynced) with 3
    wal.sync().unwrap();
    let lost = wal.append(1, &key(50), &value(50), 51).unwrap().appended();
    assert!(!wal.covered(lost.end));
    wal.lose_unsynced();
    drop(wal);
    let (wal, report) = reopen(dir.path(), 256).unwrap();
    assert_eq!(report.tail, lost.start);
    assert_eq!(rows_from(&wal, 0), expected(&appended, 0..4));
    assert!(wal.segments().unwrap().iter().all(|&(_, len)| len == 256));
}

/// Writers append same-size records and sync from several threads, on
/// small segments; at a random moment the power fails, keeping or losing
/// each 4 KiB page that no fsync covered at random. Every synced record
/// survives and every record read is one that was appended there. Then
/// more records are appended and synced over the gap, the log is reopened
/// with nothing unsynced, and it must read as exactly the surviving
/// records and then the new ones: nothing from before the power loss comes
/// back after them.
fn vector_wal_power_loss_round(seed: u64) {
    use std::sync::atomic::AtomicBool;
    const WRITERS: usize = 6;
    const SEGMENT: u32 = 16 * 1024;
    const PAGE: u64 = 4096;
    struct Rng(u64);
    impl Rng {
        fn below(&mut self, n: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % n
        }
    }
    // Same-size records: 9-byte keys, 300-byte values.
    let rec_key = |s: u64| format!("k{s:08}").into_bytes();
    let rec_value = |s: u64| vec![(s % 251) as u8; 300];

    let dir = TempDir::new().unwrap();
    let (wal, _) = open_at(dir.path(), SEGMENT);
    let appended: Mutex<BTreeMap<u64, (u32, u64)>> = Mutex::new(BTreeMap::new());
    let synced: Mutex<Vec<u64>> = Mutex::new(Vec::new());
    let synced_at_loss: Mutex<Vec<u64>> = Mutex::new(Vec::new());
    let next_seq = AtomicU64::new(1);
    let stop = AtomicBool::new(false);
    let image = TempDir::new().unwrap();
    let mut rng = Rng(seed);

    std::thread::scope(|s| {
        for w in 0..WRITERS {
            let (wal, appended, synced, next_seq, stop) = (&wal, &appended, &synced, &next_seq, &stop);
            s.spawn(move || {
                let mut rng = Rng(seed ^ (w as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                while !stop.load(Ordering::SeqCst) {
                    let mut mine = Vec::new();
                    for _ in 0..1 + rng.below(12) {
                        let seq = next_seq.fetch_add(1, Ordering::SeqCst);
                        let at = wal.append(w as u32, &rec_key(seq), &rec_value(seq), seq).unwrap();
                        appended.lock().insert(at.start, (w as u32, seq));
                        mine.push(at.start);
                    }
                    wal.sync().unwrap();
                    synced.lock().extend(mine);
                }
            });
        }

        // If a check fails, release the writers so the scope can end.
        struct Release<'a>(&'a AtomicBool, &'a VectorIndexWal);
        impl Drop for Release<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
                self.1.log.rounds.freeze(false);
            }
        }
        let _release = Release(&stop, &wal);

        std::thread::sleep(std::time::Duration::from_millis(2 + rng.below(30)));
        // Stop the world: every writer waits inside a sync, its records
        // written but not yet covered.
        wal.log.rounds.freeze(true);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while wal.log.rounds.frozen_callers() < WRITERS {
            assert!(std::time::Instant::now() < deadline, "seed {seed}: the writers never stopped");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        // What was synced when the power failed: writers stopped inside a
        // sync finish it after the image is taken.
        *synced_at_loss.lock() = synced.lock().clone();
        for entry in std::fs::read_dir(dir.path()).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), image.path().join(entry.file_name())).unwrap();
        }
        for (path, durable) in wal.power_loss_extents() {
            let copy = image.path().join(path.file_name().unwrap());
            let f = OpenOptions::new().write(true).open(&copy).unwrap();
            let len = f.metadata().unwrap().len();
            let mut at = durable;
            while at < len {
                let page_end = ((at / PAGE) + 1) * PAGE;
                let end = page_end.min(len);
                if rng.below(2) == 0 {
                    f.write_all_at(&vec![0u8; (end - at) as usize], at).unwrap();
                }
                at = end;
            }
        }
    });
    drop(wal);

    let appended = appended.into_inner();
    let (wal, _) = reopen(image.path(), SEGMENT).unwrap();
    let rows = rows_from(&wal, 0);
    for (at, ns, seq, key, value) in &rows {
        assert_eq!(
            appended.get(at),
            Some(&(*ns, *seq)),
            "seed {seed}: a record at {at:#x} that was never appended there"
        );
        assert_eq!(
            (key, value),
            (&rec_key(*seq), &Some(rec_value(*seq))),
            "seed {seed}: record {seq} damaged"
        );
    }
    let read: HashSet<u64> = rows.iter().map(|r| r.0).collect();
    for at in synced_at_loss.into_inner() {
        assert!(read.contains(&at), "seed {seed}: the synced record at {at:#x} was lost");
    }

    // Over the gap: more records, synced; then a reopen with nothing
    // unsynced reads the survivors, then exactly these.
    let first = appended.values().map(|v| v.1).max().unwrap_or(0) + 1;
    let mut want = rows;
    for seq in first..first + 1 + rng.below(200) {
        let at = wal.append(7, &rec_key(seq), &rec_value(seq), seq).unwrap().appended();
        want.push((at.start, 7, seq, rec_key(seq), Some(rec_value(seq))));
    }
    wal.sync().unwrap();
    drop(wal);
    let (wal, _) = reopen(image.path(), SEGMENT).unwrap();
    assert_eq!(rows_from(&wal, 0), want, "seed {seed}: the log after the power loss and new records");
}

#[test]
fn every_synced_record_survives_random_power_losses() {
    for seed in 1..=12u64 {
        vector_wal_power_loss_round(0xF00D_0000 + seed);
    }
}

/// The long variant: `MINNAL_POWER_LOSS_SEEDS` rounds (default 300).
#[test]
#[ignore = "long; run explicitly with --ignored"]
fn every_synced_record_survives_many_random_power_losses() {
    let rounds: u64 = std::env::var("MINNAL_POWER_LOSS_SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(300);
    for seed in 1..=rounds {
        vector_wal_power_loss_round(0xBEEF_0000 + seed);
    }
}

/// Measurement, not a check: one caller appending a batch of 8 records of
/// 1 KiB, then syncing, on the real disk (the repository's `target/`).
/// Past the first segment, so the log is on prepared segments.
#[test]
#[ignore = "measurement; run explicitly with --release --ignored --nocapture"]
fn measure_batch_sync_latency() {
    const SEGMENT: u32 = 4 << 20;
    let dir = tempfile::tempdir_in(concat!(env!("CARGO_MANIFEST_DIR"), "/../target")).unwrap();
    let (wal, _) = open_at(dir.path(), SEGMENT);
    let value = vec![7u8; 1024];
    let mut seq = 0u64;
    let mut batch = |wal: &VectorIndexWal| {
        for _ in 0..8 {
            seq += 1;
            wal.append(1, format!("k{seq:09}").as_bytes(), &value, seq).unwrap();
        }
        wal.sync().unwrap();
    };
    while split(wal.tail()).0 < 2 {
        batch(&wal);
    }
    let n = 1500u32;
    let started = std::time::Instant::now();
    for _ in 0..n {
        batch(&wal);
    }
    println!("vector WAL batch of 8 x 1 KiB + sync: {:?} each", started.elapsed() / n);
}
