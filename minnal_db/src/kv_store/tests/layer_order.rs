//! The layer-order invariant: a newer layer always holds a newer version of a
//! key. GC writes its copies as new versions under the key's stripe, replay
//! drops a record that is not newer than what is stored, and memtables reach
//! Level 0 in the order they were frozen.

use super::*;
use crate::namespace::NamespaceId;

fn small_segment_store(dir: &TempDir) -> KVStore {
    KVStore::open(
        NamespaceId::from_u64(0),
        "default",
        dir.path(),
        default_lsm_config(),
        SyncConfig::default(),
        64 * 1024, // small segments, so they seal and become GC candidates
    )
    .unwrap()
}

fn assert_layers_in_order(store: &KVStore, when: &str) {
    let violations = store.lsm.layer_order_violations().unwrap();
    assert!(
        violations.is_empty(),
        "{when}: a newer layer holds an older version:\n{}",
        violations.join("\n")
    );
}

fn seq_of(store: &KVStore, key: &[u8]) -> u64 {
    store.lsm.get_with_seq(key).unwrap().expect("key is live").1
}

/// Overwrite `hot` keys `rounds` times so their old records fill sealed
/// segments with garbage, next to `cold` keys written once.
fn fill_with_garbage(store: &KVStore, cold: usize, hot: usize, rounds: usize) {
    let value = vec![0xC0u8; 512];
    for i in 0..cold {
        store.put_to_storage(format!("cold{i:04}").as_bytes(), &value).unwrap();
    }
    for round in 0..rounds {
        for i in 0..hot {
            store.put_to_storage(format!("hot{i:04}").as_bytes(), &[round as u8; 512]).unwrap();
        }
    }
}

/// A GC copy is a new version of its key: a fresh seq, newer than the copy it
/// replaces. Written at the old seq instead, the copy would sit in the memtable
/// (and then a newer L0 file) beside the original at the same seq.
#[test]
fn gc_writes_its_copy_as_the_newest_version() {
    let dir = TempDir::new().unwrap();
    let store = small_segment_store(&dir);
    fill_with_garbage(&store, 200, 50, 6);
    store.flush_memtable_to_level0().unwrap();

    let before: Vec<(Vec<u8>, u64)> = (0..200)
        .map(|i| {
            let key = format!("cold{i:04}").into_bytes();
            let seq = seq_of(&store, &key);
            (key, seq)
        })
        .collect();

    let stats = store.garbage_collect_with_thresholds(1.0, 1.0).unwrap();
    assert!(stats.bytes_reclaimed > 0, "the test needs GC to reclaim segments holding cold keys");

    let mut moved = 0;
    for (key, old_seq) in &before {
        assert_eq!(store.get(key).unwrap(), Some(vec![0xC0u8; 512]), "GC lost a live value");
        let seq = seq_of(&store, key);
        assert!(seq >= *old_seq, "a key's version went backwards across GC");
        if seq > *old_seq {
            moved += 1;
        }
    }
    assert!(moved > 0, "GC must have relocated some cold keys, each at a fresh seq");
    assert_layers_in_order(&store, "after GC");
    store.lsm.compact_all().unwrap();
    assert_layers_in_order(&store, "after GC and a merge");
}

/// A writer holds its key's stripe from drawing its seq to applying the write.
/// GC must not re-point that key in between: its copy would carry a fresh,
/// higher seq and beat the write, which is newer. So GC waits for the stripe,
/// then sees the key no longer points at the record it copied, and skips it.
#[test]
fn a_write_that_drew_its_seq_before_gc_still_wins() {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(small_segment_store(&dir));
    fill_with_garbage(&store, 200, 50, 6);
    let key = b"cold0000".as_slice();

    // A writer that has drawn its seq and not yet applied its write.
    let stripe = store.lock_key(key);
    let seq = store.alloc_seq();

    let gc = {
        let store = Arc::clone(&store);
        std::thread::spawn(move || store.garbage_collect_with_thresholds(1.0, 1.0).unwrap())
    };
    // Let GC reach the key and find its stripe taken. Were GC to re-point the
    // key without the stripe, it would do so now, at a seq above `seq`.
    std::thread::sleep(std::time::Duration::from_millis(200));
    store.put_to_storage_seq(key, b"the writer's value", seq).unwrap();
    drop(stripe);
    let stats = gc.join().expect("GC panicked");
    assert!(stats.bytes_reclaimed > 0, "the test needs GC to reclaim segments");

    assert_eq!(
        store.get(key).unwrap().as_deref(),
        Some(b"the writer's value".as_slice()),
        "GC's copy beat a write that was newer than it"
    );
    assert_eq!(seq_of(&store, key), seq);
    assert_layers_in_order(&store, "after GC raced a writer");
}

/// A replayed record older than (or the same as) a stored copy of its key is
/// dropped: applied, it would put an older version in the memtable above a
/// newer one in L0. It costs nothing either: no value-log record, no garbage.
#[test]
fn a_replayed_record_not_newer_than_a_stored_copy_is_dropped() {
    let dir = TempDir::new().unwrap();
    let store = small_segment_store(&dir);
    store.put_to_storage_seq(b"k", b"flushed", 170).unwrap();
    store.flush_memtable_to_level0().unwrap();
    let bytes_before = store.value_log.total_live_bytes() + store.value_log.total_garbage_bytes();

    // Older than the flushed copy: a main-WAL delete behind a vector-WAL put,
    // or a WAL put behind a no-WAL write, each already in L0.
    store.replay_delete(b"k", 160).unwrap();
    store.replay_upsert(b"k", b"older", 165).unwrap();
    // The same seq: a record past its namespace's safe offset whose memtable
    // was flushed anyway.
    store.replay_upsert(b"k", b"flushed", 170).unwrap();

    assert_eq!(store.get(b"k").unwrap().as_deref(), Some(b"flushed".as_slice()));
    assert_eq!(
        store.value_log.total_live_bytes() + store.value_log.total_garbage_bytes(),
        bytes_before,
        "a dropped replay must append nothing"
    );
    assert_eq!(store.value_log.total_garbage_bytes(), 0, "a dropped replay must charge nothing");
    assert_layers_in_order(&store, "after replaying older records");

    // A newer record is applied as usual.
    store.replay_upsert(b"k", b"newer", 180).unwrap();
    assert_eq!(store.get(b"k").unwrap().as_deref(), Some(b"newer".as_slice()));
    store.replay_delete(b"k", 190).unwrap();
    assert_eq!(store.get(b"k").unwrap(), None);
    assert_layers_in_order(&store, "after replaying newer records");
}

/// Every writer at once: live puts and deletes, replays of records at old seqs,
/// GC passes, flushes and merges, in a random order. After every step, each
/// layer holds only versions older than the layer above it.
#[test]
fn the_layer_order_holds_under_every_writer() {
    let dir = TempDir::new().unwrap();
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "default",
        dir.path(),
        LSMConfig {
            num_buckets: 2,
            skip_list_capacity: 256,
            ..LSMConfig::default()
        },
        SyncConfig::default(),
        64 * 1024,
    )
    .unwrap();

    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    // Replayed records reuse seqs drawn earlier, as a log's records do.
    let mut drawn: Vec<(Vec<u8>, u64)> = Vec::new();
    for step in 0..3_000 {
        let key = format!("k{:03}", next() % 64).into_bytes();
        match next() % 100 {
            0..=49 => {
                let seq = store.alloc_seq();
                store.put_to_storage_seq(&key, &[step as u8; 2048], seq).unwrap();
                drawn.push((key, seq));
            }
            50..=64 => {
                let seq = store.alloc_seq();
                store.delete_from_storage_seq(&key, seq).unwrap();
                drawn.push((key, seq));
            }
            65..=84 if !drawn.is_empty() => {
                let (key, seq) = drawn[(next() as usize) % drawn.len()].clone();
                if next() % 2 == 0 {
                    store.replay_upsert(&key, b"replayed", seq).unwrap();
                } else {
                    store.replay_delete(&key, seq).unwrap();
                }
            }
            85..=89 => {
                store.garbage_collect_with_thresholds(1.0, 1.0).unwrap();
            }
            90..=94 => store.flush_memtable_to_level0().unwrap(),
            95..=97 => store.compact_lsm_due().unwrap(),
            _ => store.lsm.compact_all().unwrap(),
        }
        if step % 50 == 0 {
            assert_layers_in_order(&store, &format!("step {step}"));
        }
    }
    assert_layers_in_order(&store, "at the end");
}
