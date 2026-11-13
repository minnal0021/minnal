//! Opening after a crash: WAL replay, damaged or missing WAL metadata and segments, torn tails.

use super::*;

/// The WAL segment size is not fixed at creation: positions do not encode it,
/// so a reopen with another size reads the old segments back and creates new
/// ones at the new size.
#[test]
fn test_wal_segment_size_can_change_between_opens() {
    let dir = TempDir::new().unwrap();
    let small = crate::db::wal::MIN_CONFIGURED_SEGMENT_SIZE;
    let large = 4 * small;
    let value = vec![b'v'; 1000];
    {
        let mut cfg = create_db_config();
        cfg.wal_segment_size = small;
        let db = Database::open(dir.path(), cfg).unwrap();
        assert_eq!(db.wal.segment_size(), small, "a fresh WAL honours the config");
        for i in 0..200u32 {
            db.put(format!("k{i:03}").as_bytes(), &value).unwrap();
        }
        assert!(db.wal.active_segment() >= 2, "setup: the writes should span segments");
        db.mark_crashed();
    }
    let mut cfg = create_db_config();
    cfg.wal_segment_size = large;
    let db = Database::open(dir.path(), cfg).unwrap();
    assert_eq!(db.wal.segment_size(), large, "new segments take the configured size");
    for i in 0..200u32 {
        assert_eq!(
            db.get(format!("k{i:03}").as_bytes()).unwrap(),
            Some(value.clone()),
            "k{i:03} after the reopen"
        );
    }
    let first_new = db.wal.active_segment() + 1;
    for i in 200..800u32 {
        db.put(format!("k{i:04}").as_bytes(), &value).unwrap();
    }
    let last = db.wal.active_segment();
    assert!(last >= first_new, "setup: the new writes should reach a new segment");
    assert_eq!(std::fs::metadata(wal_segment_path(dir.path(), last)).unwrap().len(), large);
    db.shutdown().unwrap();
}

#[test]
fn recovery_replay_then_crash_before_metadata_flush_loses_nothing() {
    // A crash can strike right after WAL replay re-appended values to the log but
    // before the value-log metadata was flushed — leaving segment files longer than
    // their recorded totals. The next open must reconcile that (rebuild stats from
    // the LSM) and lose nothing, even across repeated crash/replay cycles. `crash()`
    // skips Database::drop's clean flush.
    let dir = TempDir::new().unwrap();
    let mut config = create_db_config();
    config.num_buckets = 1;
    let keys: Vec<Vec<u8>> = (0..30u32).map(|i| format!("k{i:03}").into_bytes()).collect();

    // Session 1: write, then crash before any clean flush.
    {
        let db = Database::open(dir.path(), config.clone()).unwrap();
        for (i, k) in keys.iter().enumerate() {
            db.put(k, format!("v{i}").as_bytes()).unwrap();
        }
        db.crash();
    }

    // Session 2: reopen replays the WAL (re-appending to the log), then crash again
    // before the post-replay metadata is flushed.
    {
        let db = Database::open(dir.path(), config.clone()).unwrap();
        assert_eq!(
            db.get(b"k000").unwrap().as_deref(),
            Some(b"v0".as_slice()),
            "replay must restore the data"
        );
        db.crash();
    }

    // Session 3: open cleanly. Everything is still readable, and the value-log
    // accounting is self-consistent despite the stale metadata it started from.
    let db = Database::open(dir.path(), config).unwrap();
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(
            db.get(k).unwrap().as_deref(),
            Some(format!("v{i}").as_bytes()),
            "key {i} lost across replay+crash cycles"
        );
    }
    for (_bucket, segs) in db.value_log_segment_stats("default").unwrap() {
        for s in segs {
            assert_eq!(
                s.live_bytes + s.garbage_bytes,
                s.total_bytes,
                "segment {} broke live+garbage==total after recovery",
                s.id
            );
        }
    }
    db.shutdown().unwrap();
}

#[test]
fn a_torn_tail_survives_reopen_append_and_gc() {
    // The full torn-tail hazard end to end: a crash leaves a partial record at the
    // active tail; reopen truncates it and replays; we append past it, churn to make
    // the segment collectable, then GC seals and relocates it. Records from BEFORE
    // the torn tail and those appended AFTER it must all survive GC relocation.
    let dir = TempDir::new().unwrap();
    let mut config = create_db_config();
    config.num_buckets = 1; // one bucket → one active-tail segment to tear
    let value = vec![0x5Au8; 300];

    // Session 1: write, crash before clean flush.
    {
        let db = Database::open(dir.path(), config.clone()).unwrap();
        for i in 0..40u32 {
            db.put(format!("k{i:03}").as_bytes(), &value).unwrap();
        }
        db.crash();
    }

    // Inject a torn write at the tail of bucket 0's active segment.
    let seg = dir.path().join("ns_default").join("value_logs").join("value_log_0.seg000001");
    let mut bytes = std::fs::read(&seg).unwrap();
    bytes.extend_from_slice(&[0xAB; 20]); // partial, not a valid header
    std::fs::write(&seg, &bytes).unwrap();

    // Session 2: reopen (truncates the torn tail, replays), append MORE past it,
    // overwrite the originals to pile up garbage, then GC.
    let db = Database::open(dir.path(), config).unwrap();
    for i in 40..60u32 {
        db.put(format!("k{i:03}").as_bytes(), &value).unwrap();
    }
    for i in 0..40u32 {
        db.put(format!("k{i:03}").as_bytes(), &value).unwrap();
    }
    db.garbage_collect().unwrap();

    // All 60 keys survive — including those appended after the torn tail.
    for i in 0..60u32 {
        assert_eq!(
            db.get(format!("k{i:03}").as_bytes()).unwrap(),
            Some(value.clone()),
            "k{i:03} lost across torn-tail reopen + GC"
        );
    }
    for (_bucket, segs) in db.value_log_segment_stats("default").unwrap() {
        for s in segs {
            assert_eq!(
                s.live_bytes + s.garbage_bytes,
                s.total_bytes,
                "segment {} broke the invariant after GC",
                s.id
            );
        }
    }
    db.shutdown().unwrap();
}

/// Item 1 regression: several single-op writes to one key, laid down in the
/// WAL with sequences *out of physical order*, must resolve to the
/// highest-sequence value. This directly exercises the sequence sort in
/// recovery (a naive scan-order replay would pick the physically-last entry,
/// which here has a lower sequence).
#[test]
fn test_recovery_applies_same_key_writes_in_sequence_order() {
    use crate::db::wal::{Wal, WalEntry, WalMetadata};

    let dir = TempDir::new().unwrap();
    let wal_dir = crate::layout::main_wal_dir(dir.path());
    let wal_meta_path = wal_dir.join(crate::db::wal::METADATA_FILE);
    {
        let (wal, report) = Wal::open(&wal_dir, crate::defaults::wal::SEGMENT_SIZE_BYTES as u32, FRESH_WAL).unwrap();
        let mut tail = report.tail;

        // Physical order: seq 5, seq 9, seq 7. Sequence order says seq 9 wins.
        for (seq, val) in [(5u64, b"v5".to_vec()), (9, b"v9".to_vec()), (7, b"v7".to_vec())] {
            wal.append_entry(&WalEntry::new_upsert(b"k".to_vec(), val).with_sequence(seq), &mut tail)
                .unwrap();
        }
        wal.sync().unwrap();

        let mut meta = WalMetadata::new();
        meta.tail = tail;
        std::fs::write(&wal_meta_path, meta.to_file_bytes()).unwrap();
    }

    let db = Database::open(dir.path(), create_db_config()).unwrap();
    assert_eq!(
        db.get(b"k").unwrap(),
        Some(b"v9".to_vec()),
        "recovery must apply same-key writes in sequence order (seq 9 is newest)"
    );
    db.shutdown().unwrap();
}

/// WAL entries fsynced *after* the last metadata save must not be lost. The
/// metadata does not record where the log ends: open finds the tail by walking
/// the log's frames, so a save older than the newest entries cannot hide them.
/// We craft that post-crash state, three durable entries and a metadata file
/// saved before any of them, and assert recovery replays all three.
#[test]
fn test_recovery_reconstructs_wal_tail_past_stale_metadata() {
    use crate::db::wal::{Wal, WalEntry, WalMetadata};

    let dir = TempDir::new().unwrap();
    let wal_dir = crate::layout::main_wal_dir(dir.path());
    let wal_meta_path = wal_dir.join(crate::db::wal::METADATA_FILE);

    {
        let (wal, report) = Wal::open(&wal_dir, crate::defaults::wal::SEGMENT_SIZE_BYTES as u32, FRESH_WAL).unwrap();
        let mut tail = report.tail;

        wal.append_entry(&WalEntry::new_upsert(b"k1".to_vec(), b"v1".to_vec()).with_sequence(1), &mut tail)
            .unwrap();
        wal.append_entry(&WalEntry::new_upsert(b"k2".to_vec(), b"v2".to_vec()).with_sequence(2), &mut tail)
            .unwrap();
        wal.append_entry(&WalEntry::new_upsert(b"k3".to_vec(), b"v3".to_vec()).with_sequence(3), &mut tail)
            .unwrap();
        wal.sync().unwrap();

        // The metadata as saved before any of the entries.
        std::fs::write(&wal_meta_path, WalMetadata::new().to_file_bytes()).unwrap();
    }

    let db = Database::open(dir.path(), create_db_config()).unwrap();
    assert_eq!(db.get(b"k1").unwrap(), Some(b"v1".to_vec()));
    assert_eq!(
        db.get(b"k2").unwrap(),
        Some(b"v2".to_vec()),
        "an entry written after the last metadata save must be replayed"
    );
    assert_eq!(
        db.get(b"k3").unwrap(),
        Some(b"v3".to_vec()),
        "an entry written after the last metadata save must be replayed"
    );
    db.shutdown().unwrap();
}

/// Coordinator-level companion to
/// [`test_recovery_reconstructs_wal_tail_past_stale_metadata`]: rather than
/// hand-crafting the WAL, it drives the **real** `Database::put` write path. It
/// guards replay: `recover_from_wal` must apply every entry from the saved cut
/// to the tail open found in the log, so it fails if replay stops early or does
/// not run.
///
/// Scenario (the "crash just after a metadata save" window):
///   1. open Db, `put` key A,
///   2. save the WAL metadata, and keep those bytes,
///   3. `put` key B, fsynced into the WAL after that save,
///   4. `crash()`: the memtable holding A and B is lost, leaving both only in
///      the WAL. (A plain `drop` would run `Database::drop`, whose clean flush
///      moves them to an SSTable, and replay would not be needed.)
///   5. restore the saved metadata bytes,
///   6. reopen and assert BOTH A and B are recovered.
#[test]
fn test_recovery_from_stale_metadata_via_real_write_path() {
    let dir = TempDir::new().unwrap();
    let wal_meta_path = wal_metadata_path(dir.path());

    let stale_metadata: Vec<u8>;
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        db.put(b"A", b"va").unwrap();

        // The on-disk metadata as it stood at the last save before the crash.
        db.flush_wal_metadata_internal().unwrap();
        stale_metadata = std::fs::read(&wal_meta_path).unwrap();

        // B is durable in the WAL (every put fsyncs it), after that save.
        db.put(b"B", b"vb").unwrap();

        // `crash()` skips the clean flush of the memtable and metadata that
        // `Database::drop` would run. The WAL is already fsynced.
        db.crash();
    }

    // Put the saved bytes back, in case anything rewrote the file since.
    std::fs::write(&wal_meta_path, &stale_metadata).unwrap();

    // Reopen: open finds the tail by walking the log, and recover_from_wal must
    // replay both A and B, which were lost from the memtable.
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    assert_eq!(
        db.get(b"A").unwrap(),
        Some(b"va".to_vec()),
        "A lived only in the WAL and must be replayed"
    );
    assert_eq!(
        db.get(b"B").unwrap(),
        Some(b"vb".to_vec()),
        "B was written after the last metadata save and must be replayed"
    );
    db.shutdown().unwrap();
}

/// A missing segment between two that exist is not something WAL GC can
/// leave (it deletes from the oldest up): the log is damaged, and reading
/// past the hole would replay a log with acknowledged writes missing. The open
/// fails and leaves the files as they are.
#[test]
fn a_missing_middle_wal_segment_fails_the_open() -> Result<()> {
    let temp_dir = TempDir::new()?;
    {
        let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 256)?;
        let entries: Vec<WalEntry> = (0..60u32)
            .map(|i| WalEntry::new_upsert(format!("rk{:03}", i).into_bytes(), vec![b'v'; 32]))
            .collect();
        db.simulate_crash_with_wal_entries(entries)?;
        assert!(wal_segment_path(temp_dir.path(), 2).exists(), "need >= 3 segments to have a middle one");
        db.mark_crashed();
    }
    std::fs::remove_file(wal_segment_path(temp_dir.path(), 1))?;
    match Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 256) {
        Err(KVError::WalError(crate::db::wal::WalError::Log(crate::db::wal::segmented_log::LogError::Corruption(why)))) => {
            assert!(why.contains("segment 1 is missing"), "{why}");
        }
        Err(e) => panic!("expected a corruption error, got {e}"),
        Ok(_) => panic!("the open must fail on a missing middle segment"),
    }
    assert!(wal_segment_path(temp_dir.path(), 0).exists() && wal_segment_path(temp_dir.path(), 2).exists());
    Ok(())
}

/// Flip a byte of the saved WAL metadata, so its checksum fails.
fn damage_wal_metadata(db_path: &std::path::Path) {
    let metadata_path = wal_metadata_path(db_path);
    let mut data = std::fs::read(&metadata_path).unwrap();
    data[16] ^= 0xFF;
    std::fs::write(&metadata_path, data).unwrap();
}

/// Unreadable WAL metadata on a database with L1 data: the cut is unknown,
/// and replaying the whole log could bring back a key whose delete reached L1
/// (where its tombstone is dropped). The open fails, naming the file and the
/// way out, and leaves the file where it is. Removing the WAL directory (the
/// documented way out once every write is flushed) opens the database with
/// its data, and the deleted key stays deleted.
#[test]
fn unreadable_wal_metadata_with_l1_data_fails_the_open() {
    let dir = TempDir::new().unwrap();
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        db.put(b"kept", b"v").unwrap();
        db.put(b"gone", b"v").unwrap();
        db.delete(b"gone").unwrap();
        db.shutdown().unwrap(); // flushes and compacts into L1
    }
    assert!(any_level1_data_for_test(dir.path()), "setup: shutdown should leave L1 data");
    damage_wal_metadata(dir.path());
    match Database::open(dir.path(), create_db_config()) {
        Err(KVError::WalError(crate::db::wal::WalError::Log(crate::db::wal::segmented_log::LogError::Corruption(why)))) => {
            assert!(
                why.contains("unreadable") && why.contains("L1 data") && why.contains("remove the `wal` directory"),
                "{why}"
            );
        }
        Err(e) => panic!("expected a corruption error, got {e}"),
        Ok(_) => panic!("the open must fail when the WAL metadata is unreadable and L1 holds data"),
    }
    assert!(
        wal_metadata_path(dir.path()).exists(),
        "a failed open must leave the metadata where it is"
    );
    std::fs::remove_file(wal_metadata_path(dir.path())).unwrap();
    assert!(
        Database::open(dir.path(), create_db_config()).is_err(),
        "missing metadata is refused the same way"
    );

    std::fs::remove_dir_all(crate::layout::main_wal_dir(dir.path())).unwrap();
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    assert_eq!(db.get(b"kept").unwrap(), Some(b"v".to_vec()));
    assert_eq!(db.get(b"gone").unwrap(), None);
    db.shutdown().unwrap();
}

/// Unreadable WAL metadata with no L1 data (a crash during a database's first
/// flushes, before any compaction): no tombstone can have been dropped, so
/// replaying the whole log is safe, and the open does it, keeping the damaged
/// file as `metadata.corrupt`.
#[test]
fn unreadable_wal_metadata_without_l1_data_replays_the_log() {
    let dir = TempDir::new().unwrap();
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        db.put(b"flushed", b"v").unwrap();
        db.get_store(crate::namespace::DEFAULT_NAMESPACE_ID)
            .unwrap()
            .flush_memtable_to_level0()
            .unwrap();
        db.put(b"unflushed", b"v").unwrap();
        db.mark_crashed();
    }
    assert!(!any_level1_data_for_test(dir.path()), "setup: nothing should have reached L1");
    damage_wal_metadata(dir.path());
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    assert!(wal_metadata_path(dir.path()).with_extension("corrupt").exists());
    assert_eq!(db.get(b"flushed").unwrap(), Some(b"v".to_vec()));
    assert_eq!(db.get(b"unflushed").unwrap(), Some(b"v".to_vec()));
    db.shutdown().unwrap();
}

/// Whether any namespace under `db_path` holds L1 data.
fn any_level1_data_for_test(db_path: &std::path::Path) -> bool {
    crate::layout::namespace_data_dirs(db_path)
        .unwrap()
        .iter()
        .any(|d| crate::kv_store::KVStore::has_level1_data(d).unwrap())
}

/// The head is derived from the log at open (the oldest segment on disk), not
/// read from the metadata, so it is right even when WAL GC deleted segments
/// and no metadata save followed.
#[test]
fn the_head_is_the_oldest_segment_on_disk_after_a_reopen() {
    let dir = TempDir::new().unwrap();
    let oldest = {
        let db = Database::open_with_wal_segment_size(dir.path(), create_db_config(), 512).unwrap();
        for i in 0..100u32 {
            db.put(format!("k{i:03}").as_bytes(), &[b'v'; 40]).unwrap();
        }
        db.flush_all_namespaces().unwrap();
        // Delete retired segments behind the metadata's back, as a crash after
        // GC's deletes and before its save would leave them.
        let cut_segment = db.wal.segment_of(db.wal_metadata.read().cut);
        db.wal.delete_segments_below(cut_segment).unwrap();
        db.mark_crashed();
        cut_segment
    };
    assert!(oldest >= 2, "setup: some segments should be deleted");
    let db = Database::open_with_wal_segment_size(dir.path(), create_db_config(), 512).unwrap();
    assert_eq!(db.wal_metadata.read().head, crate::db::wal::segmented_log::segment_start(oldest));
    assert_eq!(db.wal_stats().head.segment_id, oldest);
    assert_eq!(db.get(b"k099").unwrap(), Some(vec![b'v'; 40]));
}

/// A configured WAL segment size must lie in [64 KiB, 4 GiB): a tiny one would
/// refuse every write, and a larger one cannot be addressed by a position.
#[test]
fn an_out_of_range_wal_segment_size_is_refused_at_open() {
    for size in [0u64, 1024, 64 * 1024 - 1, u64::from(u32::MAX) + 1, 8 << 30] {
        let dir = TempDir::new().unwrap();
        let mut config = create_db_config();
        config.wal_segment_size = size;
        match Database::open(dir.path(), config) {
            Err(KVError::WalError(crate::db::wal::WalError::InvalidSegmentSize(s))) => assert_eq!(s, size),
            Err(e) => panic!("size {size}: expected InvalidSegmentSize, got {e}"),
            Ok(_) => panic!("size {size} should be refused"),
        }
    }
    for size in [64 * 1024u64, u64::from(u32::MAX)] {
        let dir = TempDir::new().unwrap();
        let mut config = create_db_config();
        config.wal_segment_size = size;
        Database::open(dir.path(), config).unwrap().shutdown().unwrap();
    }
}

/// Why the offsets are per namespace. A key's put is in the main WAL and its
/// later delete in the vector index WAL; both reach L1, which drops the
/// tombstone. Another namespace with older unflushed writes holds the global
/// cut below the put, so replaying from the cut would bring the key back with
/// nothing newer to beat it.
#[test]
fn main_wal_replay_does_not_bring_back_a_key_whose_logged_delete_reached_l1() {
    let dir = TempDir::new().unwrap();
    let k: &[u8] = b"key";
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let holder = db.create_namespace("holder").unwrap();
        let ns = db.create_namespace("ns").unwrap();
        // `holder` writes first and never flushes: the cut stays below the put.
        db.put_ns(holder, b"h", b"v").unwrap();
        db.put_ns(ns, k, b"old").unwrap();
        let put_end = db.wal_metadata.read().tail;
        let store = db.get_store(ns).unwrap();
        store.flush_memtable_to_level0().unwrap();
        db.delete_ns_logged(ns, k).unwrap();
        db.sync_vector_index_wal().unwrap();
        store.flush_memtable_to_level0().unwrap();
        store.compact_lsm().unwrap();
        assert_eq!(db.get_ns(ns, k).unwrap(), None);
        // A second pass also removes the L0 files the first merge made
        // obsolete, tombstone included: now nothing on disk mentions the key.
        db.put_ns(ns, b"other", b"v").unwrap();
        store.flush_memtable_to_level0().unwrap();
        store.compact_lsm().unwrap();
        let meta = db.wal_metadata.read().clone();
        assert!(meta.cut < put_end, "setup: the cut must stay below the put");
        assert!(meta.replay_from(ns) >= put_end, "setup: ns flushed past its put");
        db.mark_crashed();
    }
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let ns = db.get_namespace_id("ns").unwrap();
    assert_eq!(db.get_ns(ns, k).unwrap(), None, "a deleted key came back after recovery");
    let holder = db.get_namespace_id("holder").unwrap();
    assert_eq!(
        db.get_ns(holder, b"h").unwrap(),
        Some(b"v".to_vec()),
        "holder's unflushed write must replay"
    );
}

/// Recovery skips a namespace's entries below its saved offset and replays
/// the rest; afterwards the whole log is retired.
#[test]
fn recovery_replays_each_namespace_from_its_own_offset_then_retires_the_log() {
    let dir = TempDir::new().unwrap();
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let a = db.create_namespace("a").unwrap();
        let b = db.create_namespace("b").unwrap();
        db.put_ns(a, b"a0", b"v").unwrap();
        for i in 0..10u32 {
            db.put_ns(b, format!("b{i}").as_bytes(), b"v").unwrap();
        }
        db.get_store(b).unwrap().flush_memtable_to_level0().unwrap();
        db.put_ns(b, b"b-late", b"v").unwrap();
        let saved = saved_wal_metadata(&dir);
        assert!(saved.replay_from(b) > saved.cut, "b's offset should be saved above the cut");
        db.mark_crashed();
    }
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    let (a, b) = (db.get_namespace_id("a").unwrap(), db.get_namespace_id("b").unwrap());
    assert_eq!(db.get_ns(a, b"a0").unwrap(), Some(b"v".to_vec()));
    assert_eq!(db.get_ns(b, b"b-late").unwrap(), Some(b"v".to_vec()));
    for i in 0..10u32 {
        assert_eq!(db.get_ns(b, format!("b{i}").as_bytes()).unwrap(), Some(b"v".to_vec()));
    }
    let meta = db.wal_metadata.read().clone();
    assert_eq!((meta.cut, meta.safe.len()), (meta.tail, 0), "recovery should retire the whole log");
}

/// Open cuts the log off at its last valid frame. Without that, a later record
/// the same size as a torn one ends exactly where an old, still-valid frame
/// begins, and the next recovery replays that stale frame as if it were new.
#[test]
fn open_truncates_a_torn_tail_so_stale_frames_cannot_replay() {
    let dir = TempDir::new().unwrap();
    let (second, ns) = {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = db.create_namespace("ns").unwrap();
        db.put_ns(ns, b"k1", b"v").unwrap();
        let second = db.wal_metadata.read().tail;
        db.put_ns(ns, b"k2", b"v").unwrap();
        db.put_ns(ns, b"k3", b"v").unwrap();
        db.mark_crashed();
        (second, ns)
    };
    // Tear the second frame: change its CRC, so the log's valid frames end at k1.
    {
        use std::os::unix::fs::FileExt;
        let (segment, byte) = crate::db::wal::segmented_log::split(second);
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(wal_segment_path(dir.path(), segment))
            .unwrap();
        let at = u64::from(byte) + 4;
        let mut crc = [0u8; 4];
        f.read_exact_at(&mut crc, at).unwrap();
        crc[0] ^= 0xFF;
        f.write_at(&crc, at).unwrap();
    }
    {
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        assert_eq!(db.get_ns(ns, b"k1").unwrap(), Some(b"v".to_vec()));
        assert_eq!(db.get_ns(ns, b"k3").unwrap(), None, "k3 lies after the torn frame");
        // Same size as k2's frame, so it ends where k3's stale frame began.
        db.put_ns(ns, b"k4", b"v").unwrap();
        db.mark_crashed();
    }
    let db = Database::open(dir.path(), create_db_config()).unwrap();
    assert_eq!(db.get_ns(ns, b"k4").unwrap(), Some(b"v".to_vec()));
    assert_eq!(db.get_ns(ns, b"k3").unwrap(), None, "a stale frame past the old torn tail was replayed");
}

// ── The main WAL on the shared segmented log ─────────────────────────────────

/// Open reads the main WAL only from the cut, so the sequence counter must not
/// depend on the entries below it: after a reopen, a new write to a key beats
/// every earlier one, flushed or not.
#[test]
fn a_write_after_a_reopen_beats_entries_below_the_cut() {
    let dir = TempDir::new().unwrap();
    {
        let db = Database::open_with_wal_segment_size(dir.path(), create_db_config(), 512).unwrap();
        for i in 0..100u32 {
            db.put(b"k", format!("v{i}").as_bytes()).unwrap();
        }
        db.flush_all_namespaces().unwrap();
        db.put(b"other", b"x").unwrap();
        let m = db.wal_metadata.read().clone();
        assert!(db.wal.segment_of(m.cut) >= 2, "setup: the cut should be several segments in");
        db.mark_crashed();
    }
    {
        let db = Database::open_with_wal_segment_size(dir.path(), create_db_config(), 512).unwrap();
        assert_eq!(db.get(b"k").unwrap(), Some(b"v99".to_vec()));
        db.put(b"k", b"after").unwrap();
        db.mark_crashed();
    }
    let db = Database::open_with_wal_segment_size(dir.path(), create_db_config(), 512).unwrap();
    assert_eq!(
        db.get(b"k").unwrap(),
        Some(b"after".to_vec()),
        "a write after the reopen lost to an older one"
    );
    assert_eq!(db.get(b"other").unwrap(), Some(b"x".to_vec()));
}

/// Lost metadata with segments already reclaimed: the cut is unknown, and
/// replaying whatever is left could bring back retired entries or deleted
/// keys, so the open fails instead of guessing.
#[test]
fn lost_wal_metadata_after_gc_fails_the_open() {
    let dir = TempDir::new().unwrap();
    {
        let db = Database::open_with_wal_segment_size(dir.path(), create_db_config(), 512).unwrap();
        for i in 0..100u32 {
            db.put(format!("k{i:03}").as_bytes(), &[b'v'; 40]).unwrap();
        }
        db.flush_all_namespaces().unwrap();
        db.garbage_collect_wal().unwrap();
        assert!(!wal_segment_path(dir.path(), 0).exists(), "setup: GC should have reclaimed segment 0");
        db.shutdown().unwrap();
    }
    std::fs::remove_file(wal_metadata_path(dir.path())).unwrap();
    match Database::open_with_wal_segment_size(dir.path(), create_db_config(), 512) {
        Err(KVError::WalError(crate::db::wal::WalError::Log(crate::db::wal::segmented_log::LogError::Corruption(why)))) => {
            assert!(why.contains("missing") && why.contains("L1 data"), "{why}")
        }
        Err(e) => panic!("expected a corruption error, got {e}"),
        Ok(_) => panic!("the open must fail when the WAL's progress is lost after GC"),
    }
}
