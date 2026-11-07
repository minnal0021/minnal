//! A crash at each step of GC, and value-log accounting after one.

use super::*;

// ── GC race regression tests ──────────────────────────────────────────────
//
// These tests guard against the bug where GC scanned the LSM *before*
// acquiring the bucket lock.  Any write that completed its value-log write
// but had not yet updated the LSM in that window was permanently lost:
// GC deleted the old file without copying the record, leaving a stale
// LSM pointer into a reclaimed segment (→ CorruptedLog on the next read).
//
// The fix: writes hold the bucket lock across the value-log write AND
// the LSM insert; GC scans the LSM inside the same lock, guaranteeing a
// consistent snapshot.

// ── GC crash-safety ordering ────────────────────────────────────────────
//
// The load-bearing invariant of the segmented value log:
//
//     relocate survivors → CAS re-point → FLUSH to L0 → unlink the old segment
//
// Unlinking before the flush is silent data loss: the durable LSM would point
// into a file that no longer exists, and the WAL entries that could replay those
// writes are long gone. These tests pin both halves of that ordering.

/// Value-log metadata older than the log must not hide garbage: a segment's size
/// disagreeing with its saved count triggers the exact rebuild from the LSM at open.
/// Without it every byte counts as live, and GC, which triggers on garbage, would
/// leave the segment alone.
#[test]
fn stale_value_log_metadata_is_rebuilt_from_the_lsm_at_open() {
    let dir = TempDir::new().unwrap();
    let mut cfg = default_lsm_config();
    cfg.num_buckets = 1;
    let value = vec![0x5Au8; 1024];
    let rec = crate::storage::value_log::ValueRecordHeader::record_len(b"k0".len(), value.len());
    let open = || KVStore::open(0, "default", dir.path(), cfg.clone(), SyncConfig::default(), 1024 * 1024).unwrap();

    let metadata = dir.path().join("value_logs").join("value_log_0.metadata");
    let saved = {
        let store = open();
        store.put_to_storage(b"k0", &value).unwrap();
        store.put_to_storage(b"k1", &value).unwrap();
        store.lsm.flush_memtable_to_level0().unwrap();
        store.value_log.flush_all_metadata().unwrap();
        let saved = std::fs::read(&metadata).unwrap();
        // Ten overwrites the saved metadata does not know about.
        for _ in 0..10 {
            store.put_to_storage(b"k1", &value).unwrap();
        }
        store.lsm.flush_memtable_to_level0().unwrap();
        store.crash();
        saved
    };
    std::fs::write(&metadata, &saved).unwrap();

    let store = open();
    use crate::storage::value_log::SegmentStats;
    let stats: Vec<SegmentStats> = store.value_log.all_segment_stats().into_iter().flat_map(|(_, s)| s).collect();
    let sum = |f: fn(&SegmentStats) -> u64| stats.iter().map(f).sum::<u64>();
    assert_eq!(sum(|s| s.total_bytes), 12 * rec);
    assert_eq!(sum(|s| s.live_bytes), 2 * rec, "only k0 and the newest k1 are live");
    assert_eq!(sum(|s| s.garbage_bytes), 10 * rec, "the overwritten copies are garbage");
}

/// A crash between GC's re-point and its unlink must lose nothing: the old segment
/// is still on disk, and the durable LSM still points into it.
///
/// `compact_bucket` deliberately does NOT unlink — it only hands back the segments
/// that *become* safe to unlink once the re-point is durable. So calling it and then
/// dropping the store without flushing simulates exactly that crash window.
#[test]
fn test_gc_crash_between_repoint_and_unlink_loses_nothing() {
    let dir = TempDir::new().unwrap();
    // Small segments so a few values roll several of them.
    let segment_size = 64 * 1024;
    let value = vec![0xC3u8; 8 * 1024];

    let pending = {
        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default(), segment_size).unwrap();
        for i in 0..12 {
            store.put_to_storage(format!("k{i}").as_bytes(), &value).unwrap();
        }
        // Churn one key so its old records become garbage worth collecting.
        for _ in 0..12 {
            store.put_to_storage(b"hot", &value).unwrap();
        }
        // Make the WRITES durable first. (A standalone KVStore has no WAL — the
        // Database coordinator owns that — so without this the crash below would
        // simply lose the writes themselves, which is not what we are testing.)
        store.lsm.flush_memtable_to_level0().unwrap();

        // Phases 1-2 only: survivors relocated and keys re-pointed in the MEMTABLE,
        // but nothing flushed and nothing unlinked.
        let mut pending = Vec::new();
        for bucket in 0..store.value_log.num_buckets() as u32 {
            let result = store.compact_bucket(bucket, 10.0, TAIL_GC_MIN_GARBAGE_PCT).unwrap();
            pending.extend(result.pending_unlink.iter().map(|id| (bucket, *id)));
        }
        assert!(!pending.is_empty(), "expected GC to have relocated at least one segment");

        // The old segments MUST still be on disk at this point.
        for (bucket, segment_id) in &pending {
            let log = store.value_log.get_bucket_log(*bucket).unwrap();
            assert!(
                log.segment_stats().iter().any(|s| s.id == *segment_id),
                "segment {segment_id} was unlinked before the re-point was durable — that is data loss"
            );
        }
        // A faithful crash: a clean `Drop` would flush the LSM and make this a
        // graceful close instead of the crash we mean to test.
        store.crash();
        pending
    };
    assert!(!pending.is_empty());

    // Reopen. The re-point may have been lost with the memtable, in which case the
    // keys still point at the OLD segments — which is exactly why they must not have
    // been unlinked. Either way every value must still read back.
    let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default(), segment_size).unwrap();
    for i in 0..12 {
        assert_eq!(
            store.get(format!("k{i}").as_bytes()).unwrap(),
            Some(value.clone()),
            "k{i} lost across a crash between GC's re-point and its unlink"
        );
    }
    assert_eq!(store.get(b"hot").unwrap(), Some(value), "hot key lost across the same crash");
}

/// **The data-loss guard.** Crash at the worst possible instant: the moment GC has
/// unlinked the old segments.
///
/// With the correct order (flush to L0, *then* unlink), the re-point is already
/// durable when the segments go, so this crash is harmless. Swap the two — unlink
/// first, flush second — and this same crash loses the re-point *and* the segments
/// it pointed into: the durable LSM would reference files that no longer exist, and
/// the WAL entries that could replay those writes are long gone. This test fails
/// outright in that world, which is the whole point of it.
#[test]
fn test_gc_crash_the_instant_segments_are_unlinked_loses_nothing() {
    let dir = TempDir::new().unwrap();
    let segment_size = 64 * 1024;
    let value = vec![0xE5u8; 8 * 1024];

    {
        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default(), segment_size).unwrap();
        for i in 0..12 {
            store.put_to_storage(format!("k{i}").as_bytes(), &value).unwrap();
        }
        for _ in 0..12 {
            store.put_to_storage(b"hot", &value).unwrap();
        }

        // Arm the crash point: GC will return the instant its segments are unlinked.
        store.gc_crash_after_unlink.store(true, Ordering::SeqCst);
        let stats = store.garbage_collect_with_threshold(10.0).unwrap();
        assert!(stats.bytes_reclaimed > 0, "expected GC to have unlinked at least one segment");

        // Crash: no flush, so nothing beyond what GC itself made durable survives.
        store.crash();
    }

    let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default(), segment_size).unwrap();
    for i in 0..12 {
        assert_eq!(
            store.get(format!("k{i}").as_bytes()).unwrap(),
            Some(value.clone()),
            "k{i} lost when GC crashed the instant it unlinked — the re-point was not durable first"
        );
    }
    assert_eq!(store.get(b"hot").unwrap(), Some(value), "hot key lost to the same crash");
}

/// The full pass, then a crash: after `garbage_collect_with_threshold` returns, the
/// re-point IS durable (it flushed to L0 before unlinking), so a crash immediately
/// afterwards must still resolve every key — now through the *new* segments.
#[test]
fn test_gc_survives_a_crash_immediately_after_the_pass() {
    let dir = TempDir::new().unwrap();
    let segment_size = 64 * 1024;
    let value = vec![0xD4u8; 8 * 1024];

    {
        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default(), segment_size).unwrap();
        for i in 0..12 {
            store.put_to_storage(format!("k{i}").as_bytes(), &value).unwrap();
        }
        for _ in 0..12 {
            store.put_to_storage(b"hot", &value).unwrap();
        }
        let stats = store.garbage_collect_with_threshold(10.0).unwrap();
        assert!(stats.bytes_reclaimed > 0, "expected GC to reclaim at least one segment");
        // A faithful crash right after the pass: no flush of the LSM, so only what
        // GC itself made durable survives.
        store.crash();
    }

    let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default(), segment_size).unwrap();
    for i in 0..12 {
        assert_eq!(
            store.get(format!("k{i}").as_bytes()).unwrap(),
            Some(value.clone()),
            "k{i} lost across a crash right after a GC pass — the re-point was not durable before the unlink"
        );
    }
    assert_eq!(store.get(b"hot").unwrap(), Some(value));
}

/// GC appends its relocated copies without an fsync and then unlinks the
/// segments they came from. The copies must be durable before that, or a power
/// loss after the unlink loses both. A standalone store has no flush observer,
/// so this exercises GC's own fsync rather than the observer's.
#[test]
fn test_gc_makes_its_relocated_copies_durable() {
    let dir = TempDir::new().unwrap();
    let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::new(0), 64 * 1024).unwrap();
    for round in 0..4u8 {
        for i in 0u32..200 {
            store.put_to_storage(&format!("key:{i:04}").into_bytes(), &[round; 300]).unwrap();
        }
    }
    store.value_log.sync_all().unwrap(); // start clean: only GC's own writes count
    let stats = store.garbage_collect_with_threshold(0.0).unwrap();
    assert!(stats.bytes_reclaimed > 0, "setup: GC must collect segments whose survivors it relocated");
    assert_eq!(store.value_log.unsynced_bytes(), 0, "relocated copies were left unsynced");
}
