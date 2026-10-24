//! Power-loss and crash tests for the main WAL's write path: group commit,
//! retirement by saved progress, checksummed frames.
//!
//! A process kill leaves the page cache intact, so it cannot show a write that
//! returned before an fsync covered it. These tests use a power-loss model
//! instead (`Database::power_loss_image`): every writer and worker is stopped
//! at one moment, the database directory is copied, and in the copy each log
//! is cut back to what an fsync actually covered. Writers stopped inside a
//! commit (`Wal::freeze_commits`) have appended but not been acknowledged —
//! the state a power loss would catch them in. The copy is then opened, which
//! runs recovery, and checked: every acknowledged write is there, and every
//! unacknowledged one is either whole or absent.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use parking_lot::Mutex;
use tempfile::TempDir;

use crate::db::database::Database;
use crate::db::error::KVError;
use crate::db::index_checkpoint_worker::IndexCheckpointTarget;
use crate::db::test_support::*;

/// Open the power-loss image of `db` in a fresh directory.
fn open_image(db: &Database) -> (TempDir, Database) {
    let dir = TempDir::new().unwrap();
    let image = dir.path().join("db");
    db.power_loss_image(&image).unwrap();
    let reopened = Database::open(&image, create_db_config()).unwrap();
    (dir, reopened)
}

/// Wait until `cond` holds, failing the test after 10 s.
fn wait_for(what: &str, cond: impl Fn() -> bool) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !cond() {
        assert!(std::time::Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn an_acknowledged_write_survives_a_power_loss_and_an_unacknowledged_one_does_not() {
    let dir = TempDir::new().unwrap();
    let db = Arc::new(Database::open(dir.path(), create_db_config()).unwrap());
    let ns = db.create_namespace("ns").unwrap();
    db.put_ns(ns, b"acked", b"v").unwrap();

    db.wal.freeze_commits(true);
    let writer = {
        let db = Arc::clone(&db);
        std::thread::spawn(move || db.put_ns(ns, b"in-flight", b"v"))
    };
    wait_for("the writer to reach its commit", || db.wal.frozen_writers() == 1);
    assert_eq!(
        db.get_ns(ns, b"in-flight").unwrap(),
        None,
        "a write was readable before an fsync covered it"
    );
    let (_image_dir, image) = open_image(&db);
    db.wal.freeze_commits(false);
    writer.join().unwrap().unwrap();

    let ns_image = image.get_namespace_id("ns").unwrap();
    assert_eq!(image.get_ns(ns_image, b"acked").unwrap(), Some(b"v".to_vec()));
    assert_eq!(
        image.get_ns(ns_image, b"in-flight").unwrap(),
        None,
        "an unsynced append survived the power loss"
    );
}

/// Concurrent writers share fsyncs, and every one of them is durable once it
/// returns.
#[test]
fn concurrent_writers_share_fsyncs_and_all_survive_a_power_loss() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("ns").unwrap();
    let before = db.metrics_snapshot();
    std::thread::scope(|s| {
        for t in 0..16u32 {
            let db = &db;
            s.spawn(move || {
                for i in 0..100u32 {
                    db.put_ns(ns, format!("t{t}-{i}").as_bytes(), b"v").unwrap();
                }
            });
        }
    });
    let after = db.metrics_snapshot();
    let (rounds, writes) = (
        after.wal_commit_rounds - before.wal_commit_rounds,
        after.wal_commit_writes - before.wal_commit_writes,
    );
    assert_eq!(writes, 1600);
    assert!(
        rounds < writes,
        "16 concurrent writers never shared an fsync ({rounds} rounds for {writes} writes)"
    );

    let (_image_dir, image) = open_image(&db);
    let ns_image = image.get_namespace_id("ns").unwrap();
    for t in 0..16u32 {
        for i in 0..100u32 {
            assert_eq!(
                image.get_ns(ns_image, format!("t{t}-{i}").as_bytes()).unwrap(),
                Some(b"v".to_vec()),
                "t{t}-{i} lost"
            );
        }
    }
}

#[test]
fn a_lone_writer_issues_one_fsync_per_write() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("ns").unwrap();
    let before = db.metrics_snapshot();
    for i in 0..100u32 {
        db.put_ns(ns, format!("k{i}").as_bytes(), b"v").unwrap();
    }
    let after = db.metrics_snapshot();
    assert_eq!(after.wal_commit_rounds - before.wal_commit_rounds, 100);
    assert_eq!(after.wal_commit_writes - before.wal_commit_writes, 100);
}

/// The commit delay makes a round wait for more writers only when others are
/// already waiting; a lone writer never waits for it.
#[test]
fn the_commit_delay_never_slows_a_lone_writer() {
    let dir = TempDir::new().unwrap();
    let mut config = create_db_config();
    config.sync_config = config.sync_config.with_commit_delay_us(200_000);
    let db = Database::open(dir.path(), config).unwrap();
    let ns = db.create_namespace("ns").unwrap();
    let started = std::time::Instant::now();
    for i in 0..10u32 {
        db.put_ns(ns, format!("k{i}").as_bytes(), b"v").unwrap();
    }
    assert!(started.elapsed() < Duration::from_secs(1), "a lone writer waited for the commit delay");
}

/// On the real disk (the repository's `target/`), not a RAM-backed `/tmp`:
/// there an fsync costs nothing, so how many writers share a round depends on
/// lock timing, not on what the delay buys.
#[test]
fn the_commit_delay_groups_concurrent_writers_into_larger_rounds() {
    fn rounds_for(delay_us: u64) -> u64 {
        let dir = tempfile::tempdir_in(concat!(env!("CARGO_MANIFEST_DIR"), "/../target")).unwrap();
        let mut config = create_db_config();
        config.sync_config = config.sync_config.with_commit_delay_us(delay_us);
        let db = Database::open(dir.path(), config).unwrap();
        let ns = db.create_namespace("ns").unwrap();
        let before = db.metrics_snapshot().wal_commit_rounds;
        std::thread::scope(|s| {
            for t in 0..8u32 {
                let db = &db;
                s.spawn(move || {
                    for i in 0..40u32 {
                        db.put_ns(ns, format!("t{t}-{i}").as_bytes(), b"v").unwrap();
                    }
                });
            }
        });
        db.metrics_snapshot().wal_commit_rounds - before
    }
    let (without, with) = (rounds_for(0), rounds_for(2_000));
    assert!(with < without, "a 2 ms delay should form fewer, larger rounds ({with} vs {without})");
}

#[test]
fn a_failed_fsync_fails_the_write_and_stops_the_wal() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("ns").unwrap();
    db.put_ns(ns, b"before", b"v").unwrap();

    db.wal.fail_next_sync();
    assert!(matches!(db.put_ns(ns, b"failed", b"v"), Err(KVError::WalPoisoned(_))));
    assert_eq!(db.get_ns(ns, b"failed").unwrap(), None, "a write whose fsync failed must not be readable");
    assert!(
        matches!(db.put_ns(ns, b"after", b"v"), Err(KVError::WalPoisoned(_))),
        "later writes must be refused"
    );
    assert!(matches!(db.delete_ns(ns, b"before"), Err(KVError::WalPoisoned(_))));
    assert_eq!(db.get_ns(ns, b"before").unwrap(), Some(b"v".to_vec()));
}

/// Every writer waiting on a round whose fsync fails gets the error, and none
/// of their writes becomes readable.
#[test]
fn a_failed_round_fails_every_writer_in_it() {
    let dir = TempDir::new().unwrap();
    let db = Arc::new(Database::open(dir.path(), create_db_config()).unwrap());
    let ns = db.create_namespace("ns").unwrap();

    db.wal.freeze_commits(true);
    let writers: Vec<_> = (0..4u32)
        .map(|i| {
            let db = Arc::clone(&db);
            std::thread::spawn(move || db.put_ns(ns, format!("k{i}").as_bytes(), b"v"))
        })
        .collect();
    wait_for("four writers to reach their commit", || db.wal.frozen_writers() == 4);
    db.wal.fail_next_sync();
    db.wal.freeze_commits(false);
    for w in writers {
        assert!(
            matches!(w.join().unwrap(), Err(KVError::WalPoisoned(_))),
            "a writer in the failed round returned Ok"
        );
    }
    for i in 0..4u32 {
        assert_eq!(db.get_ns(ns, format!("k{i}").as_bytes()).unwrap(), None);
    }
}

/// A segment is fsynced as it is left. Commit rounds fsync only the active
/// segment, so without that, writes acknowledged into a segment that filled up
/// before the next round would be lost.
#[test]
fn leaving_a_segment_makes_it_durable() {
    let dir = TempDir::new().unwrap();
    let db = Database::open_with_wal_segment_size(dir.path(), create_db_config(), 512).unwrap();
    let ns = db.create_namespace("ns").unwrap();
    for i in 0..200u32 {
        db.put_ns(ns, format!("k{i:03}").as_bytes(), &[b'v'; 40]).unwrap();
    }
    assert!(db.wal.active_segment() > 5, "setup: the writes should span segments");
    let (_image_dir, image) = open_image(&db);
    let ns_image = image.get_namespace_id("ns").unwrap();
    for i in 0..200u32 {
        assert_eq!(
            image.get_ns(ns_image, format!("k{i:03}").as_bytes()).unwrap(),
            Some(vec![b'v'; 40]),
            "k{i:03} lost"
        );
    }
}

/// A frame that fills a segment up to the room kept for its end marker leaves
/// nothing for the next one, which rotates first. The rotation must fsync the
/// full segment: a commit round fsyncs only the active segment, so a write in
/// the full segment, acknowledged by a round that ran after the move, would
/// otherwise have no fsync behind it. Found by the power-loss stress below.
#[test]
fn a_segment_filled_to_its_last_byte_is_durable_when_left() {
    use crate::db::segmented_log::{END_MARKER_LEN, FRAME_HEADER_LEN, split};
    const SEGMENT: u64 = 512;
    let dir = TempDir::new().unwrap();
    let db = Arc::new(Database::open_with_wal_segment_size(dir.path(), create_db_config(), SEGMENT).unwrap());
    let ns = db.create_namespace("ns").unwrap();
    // Find a value length whose frame ends exactly at the segment's end, padding
    // with small writes until one exists (frames are padded for alignment).
    let fill = loop {
        let remaining = SEGMENT - u64::from(END_MARKER_LEN) - u64::from(split(db.wal_metadata.read().tail).1);
        let fits = (0..remaining as usize).find(|&len| {
            let entry = crate::db::wal::WalEntry::new_upsert_ns(ns, b"full".to_vec(), vec![b'f'; len]).with_sequence(1);
            u64::from(FRAME_HEADER_LEN) + entry.to_bytes().unwrap().len() as u64 == remaining
        });
        match fits {
            Some(len) => break len,
            None => db.put_ns(ns, b"pad", b"p").unwrap(),
        }
    };

    db.wal.freeze_commits(true);
    let first = {
        let db = Arc::clone(&db);
        std::thread::spawn(move || db.put_ns(ns, b"full", &vec![b'f'; fill]))
    };
    wait_for("the first writer to reach its commit", || db.wal.frozen_writers() == 1);
    assert_eq!(
        u64::from(split(db.wal_metadata.read().tail).1),
        SEGMENT - u64::from(END_MARKER_LEN),
        "setup: the first frame must fill the segment"
    );
    let second = {
        let db = Arc::clone(&db);
        std::thread::spawn(move || db.put_ns(ns, b"next", b"n"))
    };
    wait_for("the second writer to reach its commit", || db.wal.frozen_writers() == 2);
    db.wal.freeze_commits(false);
    first.join().unwrap().unwrap();
    second.join().unwrap().unwrap();

    let (_image_dir, image) = open_image(&db);
    let ns_image = image.get_namespace_id("ns").unwrap();
    assert_eq!(
        image.get_ns(ns_image, b"full").unwrap(),
        Some(vec![b'f'; fill]),
        "the write that filled the segment was lost"
    );
    assert_eq!(image.get_ns(ns_image, b"next").unwrap(), Some(b"n".to_vec()));
}

/// The saved metadata never records a tail beyond what is fsynced. If it did,
/// a power loss would leave the saved tail past the log's real end, the next
/// open would append after a hole, and the following recovery would read the
/// hole as the end of the segment and skip the acknowledged writes after it.
#[test]
fn the_saved_tail_never_passes_the_durable_tail() {
    let dir = TempDir::new().unwrap();
    let db = Arc::new(Database::open(dir.path(), create_db_config()).unwrap());
    let ns = db.create_namespace("ns").unwrap();
    db.put_ns(ns, b"acked", b"v").unwrap();

    db.wal.freeze_commits(true);
    let writer = {
        let db = Arc::clone(&db);
        std::thread::spawn(move || db.put_ns(ns, b"in-flight", &[b'x'; 200]))
    };
    wait_for("the writer to reach its commit", || db.wal.frozen_writers() == 1);
    db.flush_wal_metadata_internal().unwrap();
    let saved = crate::db::wal::WalMetadata::from_file_bytes(&std::fs::read(wal_metadata_path(dir.path())).unwrap()).unwrap();
    assert!(
        saved.tail <= db.wal.durable_tail(),
        "saved tail {} passed the durable tail {}",
        saved.tail,
        db.wal.durable_tail()
    );

    // End to end: lose the unsynced bytes, write after reopening, crash again.
    let image_dir = TempDir::new().unwrap();
    let image_path = image_dir.path().join("db");
    db.power_loss_image(&image_path).unwrap();
    db.wal.freeze_commits(false);
    writer.join().unwrap().unwrap();
    {
        let image = Database::open(&image_path, create_db_config()).unwrap();
        let ns_image = image.get_namespace_id("ns").unwrap();
        image.put_ns(ns_image, b"after-reopen", b"v").unwrap();
        image.mark_crashed();
    }
    let image = Database::open(&image_path, create_db_config()).unwrap();
    let ns_image = image.get_namespace_id("ns").unwrap();
    assert_eq!(image.get_ns(ns_image, b"acked").unwrap(), Some(b"v".to_vec()));
    assert_eq!(
        image.get_ns(ns_image, b"after-reopen").unwrap(),
        Some(b"v".to_vec()),
        "a write acknowledged after the reopen was lost behind a hole"
    );
}

// ── Randomised power-loss stress ─────────────────────────────────────────────

/// A tiny deterministic generator (no extra dependency).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// What a key should read after a power loss: its last acknowledged value,
/// or — if a write to it was in flight — possibly that write's value.
/// A key in a namespace, and a value (`None`: deleted).
type NsKey = (u32, Vec<u8>);
type Value = Option<Vec<u8>>;

#[derive(Default)]
struct Expected {
    acked: HashMap<NsKey, Value>,
    /// Per writer, the operation it has started and not yet seen return.
    in_flight: HashMap<usize, (NsKey, Value)>,
}

/// Writers, a background thread doing what the workers do (flush, WAL GC,
/// field-index checkpoints), small memtables and small WAL segments, so
/// flushes, rotations, retirement and GC all happen often. At random moments
/// every thread is stopped — writers either between operations or inside a
/// commit — and the power-loss image is checked.
fn power_loss_stress(seed: u64, images: usize) {
    const WRITERS: usize = 8;
    const KEYS_PER_WRITER: u64 = 40;
    let dir = TempDir::new().unwrap();
    let mut config = create_db_config();
    config.lsm_config.skip_list_capacity = 64;
    let db = Database::open_with_wal_segment_size(dir.path(), config, 4096).unwrap();
    let docs = db.create_namespace("docs").unwrap();
    let other = db.create_namespace("other").unwrap();
    activate_status_index(&db, docs);

    // Each writer owns its keys, and no two writers' keys share a key-stripe
    // lock: a writer stopped inside a commit holds its key's stripe, and a
    // second writer waiting on that stripe could never reach a stopping point.
    let mut taken = std::collections::HashSet::new();
    let keys: Vec<Vec<(u32, Vec<u8>)>> = (0..WRITERS)
        .map(|w| {
            let mut mine = Vec::new();
            let mut i = 0u64;
            while (mine.len() as u64) < KEYS_PER_WRITER {
                let ns = if i.is_multiple_of(3) { other } else { docs };
                let key = format!("w{w}/k{i}").into_bytes();
                if taken.insert(db.stripe_of(ns, &key)) {
                    mine.push((ns, key));
                }
                i += 1;
            }
            mine
        })
        .collect();
    let expected = Mutex::new(Expected::default());
    let pause = AtomicBool::new(false);
    let parked = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let park = |pause: &AtomicBool, parked: &AtomicUsize| {
        if pause.load(Ordering::SeqCst) {
            parked.fetch_add(1, Ordering::SeqCst);
            while pause.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_micros(200));
            }
            parked.fetch_sub(1, Ordering::SeqCst);
        }
    };

    std::thread::scope(|s| {
        for (w, mine) in keys.iter().enumerate() {
            let (db, expected, pause, parked, stop) = (&db, &expected, &pause, &parked, &stop);
            s.spawn(move || {
                let mut rng = Rng(seed ^ (w as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                while !stop.load(Ordering::SeqCst) {
                    park(pause, parked);
                    let (ns, key) = mine[rng.below(KEYS_PER_WRITER) as usize].clone();
                    let value = if rng.below(5) == 0 {
                        None
                    } else {
                        let status = if rng.below(2) == 0 { "active" } else { "idle" };
                        Some(format!(r#"{{"status":"{status}","n":{}}}"#, rng.next()).into_bytes())
                    };
                    expected.lock().in_flight.insert(w, ((ns, key.clone()), value.clone()));
                    let result = match (&value, rng.below(4)) {
                        (Some(v), 0) => db.merge_ns(ns, &key, v, |_, op| Ok(Some(op.to_vec()))).map(|_| ()),
                        (Some(v), _) => db.put_ns(ns, &key, v),
                        (None, _) => db.delete_ns(ns, &key),
                    };
                    result.unwrap();
                    let mut e = expected.lock();
                    e.acked.insert((ns, key), value);
                    e.in_flight.remove(&w);
                }
            });
        }
        {
            let (db, pause, parked, stop) = (&db, &pause, &parked, &stop);
            s.spawn(move || {
                let mut i = 0u64;
                while !stop.load(Ordering::SeqCst) {
                    park(pause, parked);
                    match i % 3 {
                        0 => {
                            db.flush_namespaces_pinning_wal();
                        }
                        1 => {
                            db.garbage_collect_wal().unwrap();
                        }
                        _ => {
                            db.run_index_checkpoint().unwrap();
                        }
                    }
                    i += 1;
                    std::thread::sleep(Duration::from_millis(2));
                }
            });
        }

        // If a check fails, release every thread so the scope can end and the
        // panic surface, instead of leaving them parked for ever.
        struct Release<'a>(&'a AtomicBool, &'a AtomicBool, &'a Database);
        impl Drop for Release<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
                self.1.store(false, Ordering::SeqCst);
                self.2.wal.freeze_commits(false);
            }
        }
        let _release = Release(&stop, &pause, &db);

        let mut rng = Rng(seed);
        for round in 0..images {
            std::thread::sleep(Duration::from_millis(5 + rng.below(40)));
            // Stop the world: writers park between operations or wait inside
            // a commit (appended, not acknowledged); the maintenance thread
            // parks between passes.
            db.wal.freeze_commits(true);
            pause.store(true, Ordering::SeqCst);
            wait_for("every thread to stop", || {
                parked.load(Ordering::SeqCst) + db.wal.frozen_writers() == WRITERS + 1
            });
            let (_image_dir, image) = open_image(&db);
            check_image(&image, &expected.lock(), seed, round);
            drop(image);
            pause.store(false, Ordering::SeqCst);
            db.wal.freeze_commits(false);
        }
        stop.store(true, Ordering::SeqCst);
    });
}

fn check_image(image: &Database, expected: &Expected, seed: u64, round: usize) {
    let in_flight: HashMap<&(u32, Vec<u8>), &Option<Vec<u8>>> = expected.in_flight.values().map(|(k, v)| (k, v)).collect();
    for ((ns, key), want) in &expected.acked {
        // Namespace ids are kept in the copied registry.
        let got = image.get_ns(*ns, key).unwrap();
        let ok = got == *want || in_flight.get(&(*ns, key.clone())).is_some_and(|v| got == **v);
        let text = String::from_utf8_lossy(key);
        assert!(ok, "seed {seed} image {round}: ns {ns} {text} reads {got:?}, acknowledged {want:?}");
    }
    // The field index, rebuilt from its checkpoint and the WAL, agrees with the
    // documents themselves.
    let docs = image.get_namespace_id("docs").unwrap();
    activate_status_index(image, docs);
    let mut indexed = image.query_keys(docs, "status = \"active\"").unwrap().keys;
    indexed.sort();
    let mut stored: Vec<Vec<u8>> = image
        .get_store(docs)
        .unwrap()
        .scan_prefix_batch(b"")
        .unwrap()
        .into_iter()
        .filter(|(_, v)| v.windows(17).any(|w| w == br#""status":"active""#))
        .map(|(k, _)| k)
        .collect();
    stored.sort();
    assert_eq!(indexed, stored, "seed {seed} image {round}: the field index disagrees with the documents");
}

#[test]
fn every_acknowledged_write_survives_random_power_losses() {
    for seed in [0x5EED_0001u64, 0x5EED_0002] {
        power_loss_stress(seed, 6);
    }
}

/// The long variant: `MINNAL_POWER_LOSS_SEEDS` seeds (default 20) of 20
/// images each.
#[test]
#[ignore = "long power-loss stress; run with --ignored"]
fn every_acknowledged_write_survives_many_random_power_losses() {
    let seeds: u64 = std::env::var("MINNAL_POWER_LOSS_SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    for seed in 0..seeds {
        power_loss_stress(0xC0FFEE ^ seed.wrapping_mul(0x2545_F491_4F6C_DD1D), 20);
    }
}

// ── Process kill ─────────────────────────────────────────────────────────────
//
// A real child process writes concurrently (with flushes, WAL GC and index
// checkpoints running) and is killed with SIGKILL at a random moment; the parent
// reopens the database and checks it, over several kill-and-recover cycles on
// the same directory. A kill keeps the page cache, so this checks recovery and
// retirement end to end rather than fsync coverage (the power-loss tests above
// do that).

const KILL_DIR_VAR: &str = "MINNAL_KILL_TEST_DIR";
const KILL_WRITERS: usize = 6;

/// The child half of [`acknowledged_writes_survive_repeated_kills`]: writes
/// until killed. Each writer logs `I <key> <value|->` before an operation and
/// `A <key> <value|->` once it returned, in its own file.
#[test]
#[ignore = "spawned as a child process by acknowledged_writes_survive_repeated_kills"]
fn kill_child_writes_until_killed() {
    use std::io::Write;
    let Ok(dir) = std::env::var(KILL_DIR_VAR) else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let mut config = create_db_config();
    config.lsm_config.skip_list_capacity = 64;
    let db = Database::open_with_wal_segment_size(&dir.join("db"), config, 4096).unwrap();
    let ns = db.get_namespace_id("docs").unwrap_or_else(|| db.create_namespace("docs").unwrap());
    activate_status_index(&db, ns);
    let round: u64 = std::env::var("MINNAL_KILL_ROUND").unwrap().parse().unwrap();

    std::thread::scope(|s| {
        for w in 0..KILL_WRITERS {
            let (db, dir) = (&db, &dir);
            s.spawn(move || {
                let mut log = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(dir.join(format!("w{w}.log")))
                    .unwrap();
                let mut rng = Rng((round + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (w as u64 + 1));
                let mut n = 0u64;
                loop {
                    n += 1;
                    let key = format!("w{w}-k{}", rng.below(30));
                    let value = (rng.below(5) != 0).then(|| {
                        format!(
                            r#"{{"status":"{}","r":{round},"n":{n}}}"#,
                            if rng.below(2) == 0 { "active" } else { "idle" }
                        )
                    });
                    let shown = value.clone().unwrap_or_else(|| "-".into());
                    log.write_all(format!("I {key} {shown}\n").as_bytes()).unwrap();
                    match (&value, rng.below(4)) {
                        (Some(v), 0) => db.merge_ns(ns, key.as_bytes(), v.as_bytes(), |_, op| Ok(Some(op.to_vec()))).map(|_| ()),
                        (Some(v), _) => db.put_ns(ns, key.as_bytes(), v.as_bytes()),
                        (None, _) => db.delete_ns(ns, key.as_bytes()),
                    }
                    .unwrap();
                    log.write_all(format!("A {key} {shown}\n").as_bytes()).unwrap();
                }
            });
        }
        s.spawn(|| {
            let mut i = 0u64;
            loop {
                i += 1;
                match i % 3 {
                    0 => {
                        db.flush_namespaces_pinning_wal();
                    }
                    1 => {
                        db.garbage_collect_wal().unwrap();
                    }
                    _ => {
                        db.run_index_checkpoint().unwrap();
                    }
                }
                std::thread::sleep(Duration::from_millis(3));
            }
        });
    });
}

/// For one key: its last acknowledged value (`None` if never acknowledged;
/// `Some(None)` for a delete), and the values of operations that started after
/// it (killed before they returned; each may or may not be in).
type KeyLog = (Option<Option<String>>, Vec<Option<String>>);

/// Each key's [`KeyLog`], from the child's logs.
fn read_kill_logs(dir: &std::path::Path) -> HashMap<String, KeyLog> {
    let mut keys: HashMap<String, KeyLog> = HashMap::new();
    for w in 0..KILL_WRITERS {
        let Ok(text) = std::fs::read_to_string(dir.join(format!("w{w}.log"))) else {
            continue;
        };
        // Only complete lines: a kill can land in the middle of writing one, and
        // a torn `A` line would read as a truncated acknowledged value.
        for line in text.split_inclusive('\n').filter(|l| l.ends_with('\n')).map(str::trim_end) {
            let mut parts = line.splitn(3, ' ');
            let (Some(kind), Some(key), Some(value)) = (parts.next(), parts.next(), parts.next()) else {
                continue;
            };
            let value = (value != "-").then(|| value.to_string());
            let entry = keys.entry(key.to_string()).or_default();
            match kind {
                "I" => entry.1.push(value),
                "A" => {
                    entry.0 = Some(value);
                    entry.1.clear();
                }
                _ => {}
            }
        }
    }
    keys
}

#[test]
fn acknowledged_writes_survive_repeated_kills() {
    use std::os::unix::process::ExitStatusExt;
    let dir = TempDir::new().unwrap();
    let mut rng = Rng(0xD1CE);
    for round in 0..4u64 {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "db::power_loss_tests::kill_child_writes_until_killed",
                "--exact",
                "--ignored",
                "--nocapture",
            ])
            .env(KILL_DIR_VAR, dir.path())
            .env("MINNAL_KILL_ROUND", round.to_string())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        std::thread::sleep(Duration::from_millis(300 + rng.below(700)));
        child.kill().unwrap();
        let status = child.wait().unwrap();
        assert_eq!(
            status.signal(),
            Some(libc::SIGKILL),
            "round {round}: the child stopped by itself: {status:?}"
        );

        let db = Database::open(&dir.path().join("db"), create_db_config()).unwrap();
        let ns = db.get_namespace_id("docs").expect("the namespace survives");
        let logs = read_kill_logs(dir.path());
        assert!(!logs.is_empty(), "round {round}: the child wrote nothing before it was killed");
        for (key, (acked, pending)) in &logs {
            let got = db.get_ns(ns, key.as_bytes()).unwrap().map(|v| String::from_utf8(v).unwrap());
            let ok = acked.as_ref().is_some_and(|a| *a == got) || pending.contains(&got) || (acked.is_none() && got.is_none());
            if !ok {
                // Keep the evidence: the database and the child's logs.
                let kept = dir.keep();
                panic!(
                    "round {round}: {key} reads {got:?}; acknowledged {acked:?}, in flight {pending:?}; kept {}",
                    kept.display()
                );
            }
        }
        // The field index agrees with the documents after recovery.
        activate_status_index(&db, ns);
        let mut indexed = db.query_keys(ns, "status = \"active\"").unwrap().keys;
        indexed.sort();
        let mut stored: Vec<Vec<u8>> = db
            .get_store(ns)
            .unwrap()
            .scan_prefix_batch(b"")
            .unwrap()
            .into_iter()
            .filter(|(_, v)| v.windows(17).any(|w| w == br#""status":"active""#))
            .map(|(k, _)| k)
            .collect();
        stored.sort();
        assert_eq!(indexed, stored, "round {round}: the field index disagrees with the documents");
        db.shutdown().unwrap();
    }
}

// ── Preallocated segments ────────────────────────────────────────────────────

/// Every segment is created at its full size, so an fsync never has to commit
/// a new file size (design note §15).
#[test]
fn wal_segments_are_created_at_full_size() {
    const SEGMENT: u64 = 4096;
    let dir = TempDir::new().unwrap();
    let db = Database::open_with_wal_segment_size(dir.path(), create_db_config(), SEGMENT).unwrap();
    let ns = db.create_namespace("ns").unwrap();
    assert_eq!(std::fs::metadata(wal_segment_path(dir.path(), 0)).unwrap().len(), SEGMENT);
    for i in 0..200u32 {
        db.put_ns(ns, format!("k{i:03}").as_bytes(), &[b'v'; 40]).unwrap();
    }
    let last = db.wal.active_segment();
    assert!(last >= 2, "setup: the writes should span segments");
    for id in 1..=last {
        let path = wal_segment_path(dir.path(), id);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), SEGMENT, "segment {id} is not preallocated");
    }
}

/// Reopening zeroes everything past the recovered tail (so a stale frame there
/// cannot replay later) and leaves the segment at its full size.
#[test]
fn reopening_zeroes_past_the_tail_and_keeps_the_segment_preallocated() {
    use std::os::unix::fs::FileExt;
    const SEGMENT: u64 = 4096;
    let dir = TempDir::new().unwrap();
    let tail = {
        let db = Database::open_with_wal_segment_size(dir.path(), create_db_config(), SEGMENT).unwrap();
        let ns = db.create_namespace("ns").unwrap();
        db.put_ns(ns, b"k", b"v").unwrap();
        let tail = db.wal_metadata.read().tail;
        db.mark_crashed();
        tail
    };
    // Garbage past the tail, as a torn write would leave.
    let (segment, byte) = crate::db::segmented_log::split(tail);
    let wal = wal_segment_path(dir.path(), segment);
    std::fs::OpenOptions::new()
        .write(true)
        .open(&wal)
        .unwrap()
        .write_at(&[0xAB; 64], u64::from(byte))
        .unwrap();

    let db = Database::open_with_wal_segment_size(dir.path(), create_db_config(), SEGMENT).unwrap();
    assert_eq!(db.wal_metadata.read().tail, tail);
    let bytes = std::fs::read(&wal).unwrap();
    assert_eq!(bytes.len() as u64, SEGMENT, "the segment lost its preallocation");
    assert!(bytes[byte as usize..].iter().all(|&b| b == 0), "bytes past the tail were not zeroed");
}

/// Measurement, not a check: one writer's `put` latency on the real disk
/// (the repository's `target/` directory), per value size, once the log has
/// moved past its first (sparse) segment onto zero-filled ones (design note
/// §17). 2 MiB segments so that happens quickly.
#[test]
#[ignore = "measurement; run explicitly with --release --ignored --nocapture"]
fn measure_one_writer_put_latency() {
    const SEGMENT: u64 = 2 << 20;
    for size in [128usize, 2048, 8192] {
        let dir = tempfile::tempdir_in(concat!(env!("CARGO_MANIFEST_DIR"), "/../target")).unwrap();
        let db = Database::open_with_wal_segment_size(dir.path(), create_db_config(), SEGMENT).unwrap();
        let ns = db.create_namespace("ns").unwrap();
        let mut i = 0u64;
        while db.wal_metadata.read().tail < SEGMENT * 2 + SEGMENT / 4 {
            db.put_ns(ns, format!("warm{i}").as_bytes(), &vec![b'v'; size]).unwrap();
            i += 1;
        }
        let n = 2000u32;
        let started = std::time::Instant::now();
        for k in 0..n {
            db.put_ns(ns, format!("k{k:05}").as_bytes(), &vec![b'v'; size]).unwrap();
        }
        let per = started.elapsed() / n;
        println!("one-writer put, {size} B: {per:?} each, {:.0}/s", 1.0 / per.as_secs_f64());
    }
}

/// A write whose fsync fails stops holding the cut back, but its frame stays
/// in the live tail, so a flush after it saves a cut past the durable tail. If
/// a power loss then takes that frame, the reopened log must still put new
/// writes above the saved cut, or the next recovery skips them: an
/// acknowledged write lost.
#[test]
fn progress_saved_past_a_failed_fsync_does_not_hide_later_writes() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.create_namespace("ns").unwrap();
    db.put_ns(ns, b"before", b"v").unwrap();
    db.wal.fail_next_sync();
    assert!(matches!(db.put_ns(ns, b"failed", &[b'x'; 4096]), Err(KVError::WalPoisoned(_))));
    db.get_store(ns).unwrap().flush_memtable_to_level0().unwrap();
    let saved = db.wal_metadata.read().clone();
    assert!(
        saved.cut > db.wal.durable_tail(),
        "setup: the flush should save a cut past the durable tail"
    );

    // Power loss: the failed frame is gone.
    let (_first, reopened) = open_image(&db);
    let ns1 = reopened.get_namespace_id("ns").unwrap();
    reopened.put_ns(ns1, b"after", b"acknowledged").unwrap();
    // A second power loss, before anything is flushed.
    let (_second, again) = open_image(&reopened);
    let ns2 = again.get_namespace_id("ns").unwrap();
    assert_eq!(again.get_ns(ns2, b"before").unwrap(), Some(b"v".to_vec()));
    assert_eq!(again.get_ns(ns2, b"failed").unwrap(), None);
    assert_eq!(
        again.get_ns(ns2, b"after").unwrap(),
        Some(b"acknowledged".to_vec()),
        "a write acknowledged after the reopen was skipped by the next recovery"
    );
}
