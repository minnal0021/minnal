//! Value-log garbage accounting: counted when the last LSM entry pointing at a
//! record goes (a memtable write replacing it, or the L0→L1 merge dropping it),
//! never by looking the key up on the write path.

use super::*;
use crate::namespace::NamespaceId;
use crate::storage::lsm::tree::Displaced;
use crate::support::metrics::Metrics;
use std::collections::HashMap;

fn open_store(dir: &TempDir, num_buckets: usize, segment_size: u64) -> KVStore {
    KVStore::open(
        NamespaceId::from_u64(0),
        "default",
        dir.path(),
        LSMConfig {
            num_buckets,
            ..LSMConfig::default()
        },
        SyncConfig::default(),
        segment_size,
    )
    .unwrap()
}

/// Every segment's running counters against an exact recount: a record is live
/// iff some LSM entry, in any layer, points at it.
fn assert_counters_exact(store: &KVStore, context: &str) {
    let mut live: HashMap<(u32, u32), u64> = HashMap::new();
    for Displaced { key_len, pointer } in store.lsm.referenced_pointers(None).unwrap() {
        let p = ShardedValuePointer::from_u128(pointer);
        *live.entry((p.bucket, p.location.segment_id)).or_default() += p.record_len(key_len);
    }
    for (bucket, segments) in store.value_log.all_segment_stats() {
        for s in segments {
            let exact = live.get(&(bucket, s.id)).copied().unwrap_or(0);
            assert_eq!(s.live_bytes, exact, "{context}: bucket {bucket} segment {}", s.id);
            assert_eq!(
                s.live_bytes + s.garbage_bytes,
                s.total_bytes,
                "{context}: bucket {bucket} segment {}",
                s.id
            );
        }
    }
}

/// Random puts and deletes, some carrying an older seq than the key's current
/// copy (as replay can), with flushes, merges and GC passes in between: after
/// every step the counters equal an exact recount. Fails if any place that stops
/// referencing a record (memtable replace, stale write, merge drop, GC re-point)
/// does not count it exactly once.
#[test]
fn the_counters_always_match_an_exact_recount() {
    let dir = TempDir::new().unwrap();
    let store = open_store(&dir, 2, 64 * 1024);
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    };
    let mut clock = 1_000u64;
    let (mut merges, mut gc_passes, mut stale) = (0, 0, 0);
    for step in 0..3_000 {
        clock += 1;
        let key = format!("key{:02}", next(40));
        let key = key.as_bytes();
        let seq = if next(10) == 0 {
            stale += 1;
            clock - 1 - next(30)
        } else {
            clock
        };
        match next(100) {
            0..=59 => {
                let value = vec![step as u8; 100 + next(1_900) as usize];
                store.put_to_storage_seq(key, &value, seq).unwrap();
            }
            60..=84 => store.delete_from_storage_seq(key, seq).unwrap(),
            85..=91 => store.lsm.flush_memtable_to_level0().unwrap(),
            92..=96 => {
                store.lsm.compact_all().unwrap();
                merges += 1;
            }
            _ => {
                store.garbage_collect_with_threshold(10.0).unwrap();
                gc_passes += 1;
            }
        }
        assert_counters_exact(&store, &format!("step {step}"));
    }
    assert!(merges > 50 && gc_passes > 50 && stale > 200, "setup: every path must run");
    assert!(store.value_log.total_garbage_bytes() > 0, "setup: garbage must have been counted");
}

/// A namespace without field indexes writes and deletes blind: no LSM lookup.
/// The record an overwrite or a delete replaces is still counted, at once if
/// the memtable held it, else when the merge drops its entry.
#[test]
fn writes_to_an_unindexed_namespace_look_nothing_up() {
    let dir = TempDir::new().unwrap();
    let store = open_store(&dir, 1, crate::defaults::value_log::SEGMENT_SIZE_BYTES);
    let metrics = Arc::new(Metrics::default());
    store.set_metrics(Arc::clone(&metrics));
    let rec = |v: &[u8]| crate::storage::value_log::ValueRecordHeader::record_len(1, v.len());
    let (a, b, c) = (vec![1u8; 100], vec![2u8; 200], vec![3u8; 300]);

    store.put_to_storage(b"k", &a).unwrap();
    store.put_to_storage(b"k", &b).unwrap();
    assert_eq!(
        store.value_log.total_garbage_bytes(),
        rec(&a),
        "replaced in the memtable: counted at once"
    );

    store.lsm.flush_memtable_to_level0().unwrap();
    store.put_to_storage(b"k", &c).unwrap();
    store.delete_from_storage(b"j").unwrap();
    assert_eq!(store.value_log.total_garbage_bytes(), rec(&a), "b is still referenced from L0");
    store.delete_from_storage(b"k").unwrap();
    assert_eq!(store.value_log.total_garbage_bytes(), rec(&a) + rec(&c), "c was in the memtable");

    assert_eq!(metrics.lookups.load(Ordering::Relaxed), 0, "no write looked the key up");

    store.lsm.flush_memtable_to_level0().unwrap();
    store.lsm.compact_all().unwrap();
    assert_eq!(
        store.value_log.total_garbage_bytes(),
        rec(&a) + rec(&b) + rec(&c),
        "the merge drops b's entry under the tombstone and counts it"
    );
    assert_eq!(store.get(b"k").unwrap(), None);
}
