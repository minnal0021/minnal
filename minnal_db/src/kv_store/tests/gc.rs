//! Value-log GC on its own.

use super::*;
use crate::namespace::NamespaceId;

// ── GC swap-commit recovery (recover_gc_swaps) ──────────────────────────
//
// These exercise the FILE-level pass directly with distinct "OLD"/"NEW"

/// Value-log GC must resolve liveness in BATCHES, not one point-get per
/// record.
///
/// A Level-0 file carries no bloom, no min/max and no sparse index, so a
/// point-get reaching L0 scans every entry of every L0 file in the bucket.
/// Done per record, GC becomes O(records × total L0 entries) — quadratic in
/// churn. Measured before the fix at ~165 µs per record *per L0 file*, dead
/// flat from 2 to 16 files: 8 000 records against 16 L0 files took 22 s, and
/// a live run spent 484 s moving 9 MB on the vector-index queue.
///
/// Asserted on `l0_probes` rather than on elapsed time, so it is a statement
/// about the algorithm and cannot flake on a slow machine. That counter is
/// bumped only by the per-key `search_level0_files` path, so the batched
/// path leaves it untouched while the per-record path bumps it once per
/// record.
#[test]
fn gc_resolves_liveness_in_batches_not_per_record() {
    let lsm = LSMConfig {
        // One bucket, so every key shares one L0 set — the shape a churning
        // namespace degenerates into.
        num_buckets: 1,
        skip_list_capacity: 100_000,
        ..LSMConfig::default()
    };
    let dir = TempDir::new().unwrap();
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "queue",
        dir.path(),
        lsm,
        SyncConfig::default(),
        1024 * 1024, // small segments so they seal and become GC candidates
    )
    .unwrap();
    store.set_metrics(Arc::new(crate::support::metrics::Metrics::default()));

    // Four L0 files, each round overwriting the previous round's keys, so
    // the log fills with garbage exactly like an enqueue/delete queue.
    const ROUNDS: usize = 4;
    const PER_ROUND: usize = 500;
    let value = vec![b'x'; 256];
    for round in 0..ROUNDS {
        for i in 0..PER_ROUND {
            store.put_to_storage(format!("k{:07}", round * PER_ROUND + i).as_bytes(), &value).unwrap();
        }
        if round > 0 {
            for i in 0..PER_ROUND {
                store
                    .delete_from_storage(format!("k{:07}", (round - 1) * PER_ROUND + i).as_bytes())
                    .unwrap();
            }
        }
        store.flush_memtable_to_level0().unwrap();
    }

    let before = store.metrics().unwrap().snapshot();
    store.garbage_collect_with_thresholds(0.01, 0.01).unwrap();
    let probes = store.metrics().unwrap().snapshot().l0_probes - before.l0_probes;

    // The per-record path issued one probe per record scanned (thousands
    // here); the batched path issues none at all. A tenth of one round is a
    // wide margin either side.
    assert!(
        probes < (PER_ROUND / 10) as u64,
        "GC issued {probes} per-key L0 probes — it is resolving liveness per record again, \
         which is O(records × L0 entries) and made a live run take 484 s to move 9 MB"
    );

    // The batching must not change any liveness decision: the last round's
    // keys are live, every earlier round's were deleted.
    for i in 0..PER_ROUND {
        let live = format!("k{:07}", (ROUNDS - 1) * PER_ROUND + i);
        assert_eq!(
            store.get(live.as_bytes()).unwrap().as_deref(),
            Some(value.as_slice()),
            "{live} must survive GC"
        );
    }
    for round in 0..ROUNDS - 1 {
        for i in 0..PER_ROUND {
            let dead = format!("k{:07}", round * PER_ROUND + i);
            assert_eq!(store.get(dead.as_bytes()).unwrap(), None, "{dead} was deleted and must stay deleted");
        }
    }
}

#[test]
fn a_never_filled_tail_over_the_trigger_is_sealed_and_collected() {
    // A small store keeps all its data in one active-tail segment per bucket that
    // never fills a 64 MiB segment, so tail-sealing is its only path to collection.
    // With the tail-seal bar tracking the trigger, garbage in the 30-50% band — which
    // sat stranded below the old hardcoded 50% bar — is reclaimed.
    let dir = TempDir::new().unwrap();
    let lsm_config = LSMConfig {
        num_buckets: 1, // one bucket → one active tail holds everything
        ..LSMConfig::default()
    };
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "default",
        dir.path(),
        lsm_config,
        SyncConfig::default(),
        crate::defaults::value_log::SEGMENT_SIZE_BYTES,
    )
    .unwrap();

    let value = vec![0x5Au8; 400];
    for i in 0..100u32 {
        store.put_to_storage(format!("k{i:03}").as_bytes(), &value).unwrap();
    }
    // Overwrite 70 keys: their old records become garbage — ~41% of the single tail
    // segment (70 dead of 170 total), squarely in the old dead zone.
    for i in 0..70u32 {
        store.put_to_storage(format!("k{i:03}").as_bytes(), &value).unwrap();
    }
    let waste_before = store.get_waste_ratio();
    assert!(
        (30.0..50.0).contains(&waste_before),
        "test needs tail garbage in the 30-50% band, got {waste_before:.1}%"
    );

    // Old behavior (50% tail bar): the sub-50% tail is not sealed, nothing collectable.
    let stats = store.garbage_collect_with_thresholds(10.0, 50.0).unwrap();
    assert_eq!(stats.bytes_reclaimed, 0, "at a 50% tail bar a 41%-garbage tail must stay uncollected");

    // Bar tracking the 30% trigger: the tail is sealed and its garbage reclaimed.
    let stats = store.garbage_collect_with_thresholds(10.0, 30.0).unwrap();
    assert!(stats.bytes_reclaimed > 0, "at a 30% tail bar the tail must be sealed and collected");
    assert!(
        store.get_waste_ratio() < waste_before,
        "waste must drop after collection (was {waste_before:.1}%, now {:.1}%)",
        store.get_waste_ratio()
    );

    // Every live key survived the relocation into the fresh tail.
    for i in 0..100u32 {
        assert_eq!(
            store.get(format!("k{i:03}").as_bytes()).unwrap(),
            Some(value.clone()),
            "k{i:03} lost during tail GC"
        );
    }
}

#[test]
fn test_kvstore_gc() {
    let dir = TempDir::new().unwrap();
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "default",
        dir.path(),
        default_lsm_config(),
        SyncConfig::default(),
        crate::defaults::value_log::SEGMENT_SIZE_BYTES,
    )
    .unwrap();

    // Write and overwrite to create garbage
    for i in 0..5 {
        let key = format!("key{}", i).into_bytes();
        store.put_to_storage(&key, &[0u8; 50]).unwrap();
    }
    for round in 0..3 {
        for i in 0..5 {
            let key = format!("key{}", i).into_bytes();
            store.put_to_storage(&key, &[(round + 1) as u8; 50]).unwrap();
        }
    }

    let _gc_stats = store.garbage_collect().unwrap();

    // Verify data still readable
    for i in 0..5 {
        let key = format!("key{}", i).into_bytes();
        assert!(store.get(&key).unwrap().is_some());
    }
}

#[test]
fn test_gc_write_before_gc_survives() {
    // A write that completes immediately before GC starts must not be lost.
    let dir = TempDir::new().unwrap();
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "default",
        dir.path(),
        default_lsm_config(),
        SyncConfig::default(),
        crate::defaults::value_log::SEGMENT_SIZE_BYTES,
    )
    .unwrap();

    // Create enough garbage to trigger GC (75% garbage after 4 overwrites).
    for round in 0..4u8 {
        for i in 0u32..50 {
            store.put_to_storage(&format!("base:{i:04}").into_bytes(), &[round; 128]).unwrap();
        }
    }

    // Write a sentinel key, then immediately run GC.
    // GC must scan LSM inside the bucket lock and include this key.
    store.put_to_storage(b"sentinel", b"sentinel_value").unwrap();
    store.garbage_collect_with_threshold(0.0).unwrap();

    assert_eq!(
        store.get(b"sentinel").unwrap(),
        Some(b"sentinel_value".to_vec()),
        "sentinel key lost after GC"
    );
}

#[test]
fn test_gc_writes_after_swap_go_to_new_file() {
    // After GC swaps the value-log file, subsequent writes must go to the
    // new file and be readable.
    let dir = TempDir::new().unwrap();
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "default",
        dir.path(),
        default_lsm_config(),
        SyncConfig::default(),
        crate::defaults::value_log::SEGMENT_SIZE_BYTES,
    )
    .unwrap();

    for round in 0..4u8 {
        for i in 0u32..30 {
            store.put_to_storage(&format!("k{i}").into_bytes(), &[round; 128]).unwrap();
        }
    }

    store.garbage_collect_with_threshold(0.0).unwrap();

    for i in 0u32..30 {
        let key = format!("post_gc:{i}").into_bytes();
        store.put_to_storage(&key, &[0xFFu8; 64]).unwrap();
    }

    for i in 0u32..30 {
        let key = format!("post_gc:{i}").into_bytes();
        assert_eq!(store.get(&key).unwrap(), Some(vec![0xFFu8; 64]));
    }
}

#[test]
fn test_gc_multi_round_all_values_survive() {
    // Multiple sequential GC runs must not lose any values.
    let dir = TempDir::new().unwrap();
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "default",
        dir.path(),
        default_lsm_config(),
        SyncConfig::default(),
        crate::defaults::value_log::SEGMENT_SIZE_BYTES,
    )
    .unwrap();

    for round in 0..6u8 {
        for i in 0u32..40 {
            store.put_to_storage(&format!("key:{i:04}").into_bytes(), &[round; 64]).unwrap();
        }
        store.garbage_collect_with_threshold(0.0).unwrap();
    }

    for i in 0u32..40 {
        let key = format!("key:{i:04}").into_bytes();
        // Last written value was round=5
        assert_eq!(store.get(&key).unwrap(), Some(vec![5u8; 64]), "key {i} has wrong value or is missing");
    }
}

#[test]
fn test_gc_after_delete_sequential_stays_deleted() {
    let dir = TempDir::new().unwrap();
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "default",
        dir.path(),
        default_lsm_config(),
        SyncConfig::default(),
        crate::defaults::value_log::SEGMENT_SIZE_BYTES,
    )
    .unwrap();
    const N: u32 = 120;
    for round in 0..4u8 {
        for i in 0..N {
            store.put_to_storage(&format!("del:{i:04}").into_bytes(), &[round; 96]).unwrap();
        }
    }
    for i in 0..N {
        store.delete_from_storage(&format!("del:{i:04}").into_bytes()).unwrap();
    }
    store.garbage_collect_with_threshold(0.0).unwrap();
    let mut survivors = 0;
    for i in 0..N {
        if store.get(&format!("del:{i:04}").into_bytes()).unwrap().is_some() {
            survivors += 1;
        }
    }
    assert_eq!(survivors, 0, "{survivors} keys resurrected by sequential GC-after-delete");
}
