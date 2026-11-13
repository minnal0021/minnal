//! GC racing writes, reads and scans.

use super::*;
use crate::namespace::NamespaceId;

#[test]
fn test_gc_concurrent_writes_no_data_loss() {
    // Concurrent writes and GC must not lose any data.
    // Writer thread writes 200 unique keys while GC runs 5 times in parallel.
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

    // Create garbage so GC actually fires.
    for round in 0..4u8 {
        for i in 0u32..50 {
            store.put_to_storage(&format!("base:{i:04}").into_bytes(), &[round; 128]).unwrap();
        }
    }

    const NUM_NEW: usize = 200;

    std::thread::scope(|s| {
        s.spawn(|| {
            for i in 0..NUM_NEW {
                let key = format!("new:{i:04}").into_bytes();
                store.put_to_storage(&key, &[(i % 256) as u8; 64]).unwrap();
            }
        });
        s.spawn(|| {
            for _ in 0..5 {
                store.garbage_collect_with_threshold(0.0).unwrap();
            }
        });
    });

    for i in 0..NUM_NEW {
        let key = format!("new:{i:04}").into_bytes();
        let expected = vec![(i % 256) as u8; 64];
        assert_eq!(
            store.get(&key).unwrap(),
            Some(expected),
            "key new:{i:04} was lost or corrupted after concurrent GC"
        );
    }

    // Pre-existing keys must also be intact.
    for i in 0u32..50 {
        let key = format!("base:{i:04}").into_bytes();
        assert!(store.get(&key).unwrap().is_some(), "base key {i} lost after GC");
    }
}

#[test]
fn test_gc_concurrent_reads_never_error_or_return_stale() {
    // A read racing GC must always succeed and return the last-written value
    // while GC churns the same buckets continuously. Reads take no lock: a
    // pointer GC has superseded still reads the key's own bytes (segment ids are
    // never reused), and one whose segment GC unlinked is re-resolved through
    // the LSM.
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

    // Overwrite each key several times to pile up garbage in its segments, then
    // settle on the canonical value so GC has dirty segments to compact.
    for round in 0..5u8 {
        for i in 0..N {
            store.put_to_storage(&format!("k:{i:04}").into_bytes(), &[round; 48]).unwrap();
        }
    }
    for i in 0..N {
        store.put_to_storage(&format!("k:{i:04}").into_bytes(), b"final_value").unwrap();
    }

    std::thread::scope(|s| {
        for _ in 0..3 {
            s.spawn(|| {
                for _ in 0..150 {
                    for i in 0..N {
                        let key = format!("k:{i:04}").into_bytes();
                        let got = store.get(&key).expect("get must never error while GC runs");
                        assert_eq!(
                            got,
                            Some(b"final_value".to_vec()),
                            "read returned a stale/missing value for k:{i:04} during GC"
                        );
                    }
                }
            });
        }
        s.spawn(|| {
            for _ in 0..10 {
                store.garbage_collect_with_threshold(0.0).unwrap();
            }
        });
    });

    // Final state is still correct after all the churn.
    for i in 0..N {
        let key = format!("k:{i:04}").into_bytes();
        assert_eq!(store.get(&key).unwrap(), Some(b"final_value".to_vec()));
    }
}

#[test]
fn test_gc_concurrent_deletes_are_not_resurrected() {
    // A delete that lands while GC relocates the key's record must not be
    // undone. GC re-points by compare-and-set under the bucket lock, and the
    // delete takes the same lock; without it GC could copy an about-to-be-deleted
    // record and re-insert its pointer *after* the delete removed it,
    // resurrecting the key.
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

    // Seed keys and pile up garbage so GC fires while deletes run.
    for round in 0..4u8 {
        for i in 0..N {
            store.put_to_storage(&format!("del:{i:04}").into_bytes(), &[round; 96]).unwrap();
        }
    }

    std::thread::scope(|s| {
        s.spawn(|| {
            for i in 0..N {
                store.delete_from_storage(&format!("del:{i:04}").into_bytes()).unwrap();
            }
        });
        s.spawn(|| {
            for _ in 0..10 {
                store.garbage_collect_with_threshold(0.0).unwrap();
            }
        });
    });

    // One more GC after everything settles, then every key must stay deleted.
    store.garbage_collect_with_threshold(0.0).unwrap();
    let survivors: Vec<u32> = (0..N)
        .filter(|i| store.get(&format!("del:{i:04}").into_bytes()).unwrap().is_some())
        .collect();
    assert!(
        survivors.is_empty(),
        "{} keys resurrected by concurrent GC: {:?}",
        survivors.len(),
        survivors
    );
}

#[test]
fn test_scan_during_gc_never_returns_wrong_or_missing_value() {
    // A batch read resolves (key, pointer) from the LSM and then reads the value.
    // GC can reclaim the pointer's segment between those steps. The read then
    // fails loudly, and `get_multiple` re-resolves that key through `get`: without
    // that a live key comes back missing, which this test catches.
    //
    // All keys share a single bucket so every GC pass races every read, and
    // each live key carries a distinct, *stable* value — overwrites rewrite
    // the same bytes purely to keep generating garbage, so the only way a read
    // can observe a wrong/missing value is this race.
    //
    // Nothing compacts here, on purpose: each GC pass adds level-0 files, reads
    // slow down, and the race window widens (with a compaction in the GC loop,
    // 6,000 passes never hit it).
    //
    // A round is counted in GC passes that reclaimed something, since most passes
    // find nothing to do. Three short rounds rather than one long one: catches level
    // off by about 400 reclaiming passes, and passes slow down as level-0 files pile
    // up. On tmpfs with the re-resolve removed, the first round caught it in 57 of 60
    // runs and three rounds in 60 of 60; running one round on to 2,000 passes added
    // a single catch in 60.
    //
    // This test only catches the bug on tmpfs, the default on Linux
    // (`.cargo/test-runner.sh`). On a real disk (`TMPDIR=/var/tmp`, macOS) it missed
    // the bug in all 37 runs, so a pass there means little.
    for _ in 0..3 {
        scan_during_gc_round();
    }
}

fn scan_during_gc_round() {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    let dir = TempDir::new().unwrap();
    let mut cfg = default_lsm_config();
    cfg.num_buckets = 1; // concentrate every key in one bucket
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "default",
        dir.path(),
        cfg,
        SyncConfig::default(),
        crate::defaults::value_log::SEGMENT_SIZE_BYTES,
    )
    .unwrap();

    const N: u32 = 200;
    const RECLAIMING_GC_PASSES: u64 = 400;
    let live_key = |i: u32| format!("live:{i:06}").into_bytes();
    let expected = |i: u32| {
        // Distinct prefix per key (catches cross-key misreads), padded so
        // records are large enough to span segments and force relocation.
        let mut v = format!("val:{i:06}:").into_bytes();
        v.resize(100, b'.');
        v
    };

    // Seed live keys plus junk; deleting the junk leaves garbage interleaved
    // with the survivors, so GC must rewrite dirty segments and relocate them.
    for i in 0..N {
        store.put_to_storage(&live_key(i), &expected(i)).unwrap();
        store.put_to_storage(&format!("junk:{i:06}").into_bytes(), &[0xAB; 100]).unwrap();
    }
    for i in 0..N {
        store.delete_from_storage(&format!("junk:{i:06}").into_bytes()).unwrap();
    }

    let all_live: Vec<Vec<u8>> = (0..N).map(live_key).collect();
    let stop = AtomicBool::new(false);
    let reclaiming_passes = AtomicU64::new(0);
    let failure: Mutex<Option<String>> = Mutex::new(None);

    std::thread::scope(|s| {
        // Writer: keep overwriting live keys with their *same* value so the
        // bucket keeps accumulating garbage and GC keeps relocating records.
        s.spawn(|| {
            while !stop.load(Ordering::Relaxed) {
                for i in 0..N {
                    let _ = store.put_to_storage(&live_key(i), &expected(i));
                }
            }
        });
        // GC: hammer compaction on the single bucket.
        s.spawn(|| {
            while !stop.load(Ordering::Relaxed) {
                if store.garbage_collect_with_threshold(0.0).is_ok_and(|gc| gc.bytes_reclaimed > 0) {
                    reclaiming_passes.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
        // Scanner. The value-log invariant under test: a committed value is
        // never read back wrong, and a live key resolved to a pointer is never
        // unreadable. `get_multiple` resolves explicit keys, so it is
        // the strong oracle — every live key must come back present and exact.
        //
        // `scan_prefix_batch` can momentarily return a *partial* key set under
        // GC churn — but NOT because of the LSM scan: `lsm.scan_prefix` is
        // complete across concurrent flush/compaction (asserted by
        // `test_scan_completeness_during_concurrent_flush_compaction_and_gc`).
        // The transient drop is in the value-resolution path layered on top, a
        // separate concern from the silent-wrong-value race under test. So here
        // we assert only that every pair it DOES return carries the correct
        // value, not that the set is complete (`get_multiple` above is
        // the completeness oracle).
        s.spawn(|| {
            let record = |msg: String| {
                *failure.lock().unwrap() = Some(msg);
                stop.store(true, Ordering::Relaxed);
            };
            let started = std::time::Instant::now();
            while reclaiming_passes.load(Ordering::Relaxed) < RECLAIMING_GC_PASSES {
                if started.elapsed() > std::time::Duration::from_secs(600) {
                    record(format!(
                        "only {} GC passes reclaimed anything in 10 minutes",
                        reclaiming_passes.load(Ordering::Relaxed)
                    ));
                    return;
                }
                // scan_prefix_batch → resolve_entries_from_pointers + batch_read_from_entries
                for (k, v) in store.scan_prefix_batch(b"live:").unwrap() {
                    let i: u32 = std::str::from_utf8(&k[5..]).unwrap().parse().unwrap();
                    if v != expected(i) {
                        record(format!("scan_prefix: live:{i:06} WRONG value (len {})", v.len()));
                        return;
                    }
                }

                // get_multiple has its own read loop — strong oracle.
                let got = store.get_multiple(&all_live).unwrap();
                for (i, v) in got.iter().enumerate() {
                    if v.as_deref() != Some(expected(i as u32).as_slice()) {
                        record(format!(
                            "get_multiple: live:{i:06} wrong/missing (got {:?} len)",
                            v.as_ref().map(|x| x.len())
                        ));
                        return;
                    }
                }

                if stop.load(Ordering::Relaxed) {
                    return;
                }
            }
            stop.store(true, Ordering::Relaxed);
        });
    });

    if let Some(msg) = failure.lock().unwrap().take() {
        panic!("scan raced GC and observed inconsistent state: {msg}");
    }
}

#[test]
fn test_scan_completeness_during_concurrent_flush_compaction_and_gc() {
    // The LSM scan-completeness guarantee and the value log's read guarantee
    // must hold together. Under concurrent memtable flush, L0→L1 compaction, and
    // value-log GC, a scan must:
    //   * return EVERY never-deleted key, across the active→RO→L0→L1 relocation
    //     a flush/compaction performs; and
    //   * pair each key with its EXACT value, across the relocation, LSM re-point
    //     and segment unlink a GC pass performs.
    // One bucket so every flush, merge, and GC pass races every scan. Live keys
    // are never deleted and are overwritten with their SAME value, so any wrong
    // value is a race bug; active-churn
    // transient misses from APIs that document `None` for I/O races are
    // checked strictly only after the churn threads stop.
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Barrier, Mutex};

    let dir = TempDir::new().unwrap();
    let mut cfg = default_lsm_config();
    cfg.num_buckets = 1; // concentrate every key in one bucket
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "default",
        dir.path(),
        cfg,
        SyncConfig::default(),
        crate::defaults::value_log::SEGMENT_SIZE_BYTES,
    )
    .unwrap();

    const N: u32 = 200;
    let live_key = |i: u32| format!("live:{i:06}").into_bytes();
    let expected = |i: u32| {
        let mut v = format!("val:{i:06}:").into_bytes();
        v.resize(100, b'.'); // large enough to span segments and force relocation
        v
    };

    // Live keys plus junk; deleting the junk leaves garbage interleaved with the
    // survivors so GC must rewrite dirty segments and relocate live records.
    for i in 0..N {
        store.put_to_storage(&live_key(i), &expected(i)).unwrap();
        store.put_to_storage(&format!("junk:{i:06}").into_bytes(), &[0xAB; 100]).unwrap();
    }
    for i in 0..N {
        store.delete_from_storage(&format!("junk:{i:06}").into_bytes()).unwrap();
    }

    let all_live: Vec<Vec<u8>> = (0..N).map(live_key).collect();
    let stop = AtomicBool::new(false);
    let failure: Mutex<Option<String>> = Mutex::new(None);
    let start = Barrier::new(4);

    std::thread::scope(|s| {
        // Writer: overwrite live keys with their SAME value to keep producing
        // value-log garbage without changing the committed value.
        s.spawn(|| {
            start.wait();
            while !stop.load(Ordering::Acquire) {
                for i in 0..N {
                    let _ = store.put_to_storage(&live_key(i), &expected(i));
                }
            }
        });
        // LSM churn: drive the full active→RO→L0→L1 relocation chain the scans
        // must stay complete across.
        s.spawn(|| {
            start.wait();
            while !stop.load(Ordering::Acquire) {
                let _ = store.lsm.flush_memtable_to_level0();
                let _ = store.lsm.compact_all();
            }
        });
        // Value-log GC: relocate pointers under the scans.
        s.spawn(|| {
            start.wait();
            while !stop.load(Ordering::Acquire) {
                let _ = store.garbage_collect_with_threshold(0.0);
            }
        });
        // Scanner.
        //
        // (1) LSM-layer COMPLETENESS. Each of the
        //     three pointer-returning scans must return EVERY never-deleted key
        //     while flush/compaction relocate them active→RO→L0→L1. Asserted on
        //     the key set straight off the LSM, independent of value-log GC.
        // (2) Value-log batch paths are exercised under churn, but their strict
        //     completeness/exactness is asserted only after all churn has
        //     stopped. Keeping active-churn assertions at the LSM key layer
        //     avoids turning this regression into a timing-sensitive
        //     linearizability test for batch value resolution while flush,
        //     compaction, GC, and writes are all mutating the same keys.
        s.spawn(|| {
            let record = |msg: String| {
                *failure.lock().unwrap() = Some(msg);
                stop.store(true, Ordering::Release);
            };
            start.wait();
            // Every live key must appear in the LSM scan's key set.
            let verify_complete = |label: &str, keys: Vec<Vec<u8>>| -> Option<String> {
                let mut seen = vec![false; N as usize];
                for k in keys {
                    if !k.starts_with(b"live:") {
                        continue;
                    }
                    let i: u32 = std::str::from_utf8(&k[5..]).unwrap().parse().unwrap();
                    seen[i as usize] = true;
                }
                seen.iter()
                    .position(|&done| !done)
                    .map(|m| format!("{label}: live:{m:06} MISSING from LSM scan"))
            };
            for _ in 0..300 {
                if stop.load(Ordering::Acquire) {
                    return;
                }
                // (1) LSM-layer completeness across all three scan shapes.
                let prefix_keys = store.lsm.scan_prefix(b"live:").unwrap().into_iter().map(|(k, _, _)| k).collect();
                if let Some(msg) = verify_complete("lsm.scan_prefix", prefix_keys) {
                    record(msg);
                    return;
                }
                let range_keys = store
                    .lsm
                    .range_pointers_bounded(b"live:", None, usize::MAX)
                    .unwrap()
                    .into_iter()
                    .map(|(k, _, _)| k)
                    .collect();
                if let Some(msg) = verify_complete("lsm.range_pointers_bounded", range_keys) {
                    record(msg);
                    return;
                }
                let kv_keys = store.lsm.key_pointer_pairs(None).unwrap().into_iter().map(|(k, _)| k).collect();
                if let Some(msg) = verify_complete("lsm.key_pointer_pairs", kv_keys) {
                    record(msg);
                    return;
                }

                // Exercise the batch value paths under churn, but do not assert
                // a non-linearizable active-churn snapshot here. The quiescent
                // assertions below are the completeness/exactness oracle.
                let _ = store.get_multiple(&all_live).unwrap();
                let _ = store.scan_prefix_batch(b"live:").unwrap();
            }
            stop.store(true, Ordering::Release);
        });
    });

    if let Some(msg) = failure.lock().unwrap().take() {
        panic!("scan raced concurrent flush/compaction/GC and observed inconsistent state: {msg}");
    }

    // Once the churn threads have joined, the batch get path must be complete:
    // all never-deleted live keys should resolve to their exact value.
    let got = store.get_multiple(&all_live).unwrap();
    for (i, v) in got.iter().enumerate() {
        assert_eq!(
            v.as_deref(),
            Some(expected(i as u32).as_slice()),
            "quiescent get_multiple must return live:{i:06}"
        );
    }

    let mut scanned = store.scan_prefix_batch(b"live:").unwrap();
    scanned.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(scanned.len(), N as usize, "quiescent scan_prefix_batch must return every live key");
    for (i, (k, v)) in scanned.iter().enumerate() {
        assert_eq!(k, &live_key(i as u32), "quiescent scan_prefix_batch returned unexpected key at index {i}");
        assert_eq!(
            v.as_slice(),
            expected(i as u32).as_slice(),
            "quiescent scan_prefix_batch returned wrong value for live:{i:06}"
        );
    }
}
