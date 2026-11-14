//! Puts, gets, deletes and sequence ordering.

use super::*;
use crate::namespace::NamespaceId;

#[test]
fn test_kvstore_basic_operations() {
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

    store.put_to_storage(b"key1", b"value1").unwrap();
    store.put_to_storage(b"key2", b"value2").unwrap();

    assert_eq!(store.get(b"key1").unwrap(), Some(b"value1".to_vec()));
    assert_eq!(store.get(b"key2").unwrap(), Some(b"value2".to_vec()));

    store.delete_from_storage(b"key2").unwrap();
    assert_eq!(store.get(b"key2").unwrap(), None);
}

#[test]
fn test_kvstore_update() {
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

    store.put_to_storage(b"key1", b"v1").unwrap();
    assert_eq!(store.get(b"key1").unwrap(), Some(b"v1".to_vec()));

    store.put_to_storage(b"key1", b"v2").unwrap();
    assert_eq!(store.get(b"key1").unwrap(), Some(b"v2".to_vec()));
}

#[test]
fn a_lower_seq_put_charges_its_own_record_not_the_live_one() {
    // A put that loses to a higher-sequence write for the same key (the LSM drops
    // its insert under highest-sequence-wins) must account ITS OWN freshly-appended
    // record as garbage, not displace the record that is still live. This is the
    // deterministic form of the concurrent-put accounting race: a put that read the
    // displaced pointer before the bucket lock and always marked it displaced would
    // mark the LIVE record garbage here (accounting would call the live value dead,
    // and GC could reclaim it, losing data) and leave its own dead record counted
    // live.
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

    let winner = vec![0x11u8; 400];
    let loser = vec![0x22u8; 900]; // a different size, so a mis-charge can't cancel

    // Higher sequence lands first and wins; the lower sequence arrives after and is
    // dropped by the LSM.
    store.put_to_storage_seq(b"k", &winner, 10).unwrap();
    store.put_to_storage_seq(b"k", &loser, 5).unwrap();

    assert_eq!(store.get(b"k").unwrap(), Some(winner.clone()), "highest-seq write must win the read");

    let winner_len = crate::storage::value_log::ValueRecordHeader::record_len(b"k".len(), winner.len());
    let loser_len = crate::storage::value_log::ValueRecordHeader::record_len(b"k".len(), loser.len());
    let (mut live, mut garbage) = (0u64, 0u64);
    for (_bucket, segs) in store.value_log.all_segment_stats() {
        for s in segs {
            assert_eq!(
                s.live_bytes + s.garbage_bytes,
                s.total_bytes,
                "segment {} broke live+garbage==total",
                s.id
            );
            live += s.live_bytes;
            garbage += s.garbage_bytes;
        }
    }
    assert_eq!(live, winner_len, "only the winning record's bytes may be counted live");
    assert_eq!(garbage, loser_len, "the dropped lower-seq record's bytes are the garbage");
}

/// Every appended record is either what the LSM points at or counted as garbage.
/// A put whose LSM insert fails after its value-log append leaves a record that
/// nothing points at, and the apply is retried (`apply_with_retry`, then
/// recovery), each attempt appending afresh. Left counted as live, those bytes
/// make their segment look cleaner than it is, so GC may never select it.
#[test]
fn a_put_whose_lsm_insert_fails_counts_its_record_as_garbage() {
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

    let value = vec![0x33u8; 500];
    store.lsm.fail_next_insert();
    assert!(store.put_to_storage_seq(b"k", &value, 10).is_err(), "setup: the insert fails");
    store.put_to_storage_seq(b"k", &value, 10).unwrap(); // the retry
    assert_eq!(store.get(b"k").unwrap(), Some(value.clone()));

    let record_len = crate::storage::value_log::ValueRecordHeader::record_len(b"k".len(), value.len());
    let (mut live, mut garbage) = (0u64, 0u64);
    for (_bucket, segs) in store.value_log.all_segment_stats() {
        for s in segs {
            live += s.live_bytes;
            garbage += s.garbage_bytes;
        }
    }
    assert_eq!(live, record_len, "only the retry's record is live");
    assert_eq!(garbage, record_len, "the failed attempt's record, which nothing points at, is garbage");
}

/// Replay applies an entry exactly as the live write path does, including when
/// it loses. The vector index WAL replays before the main WAL, so a main-WAL
/// delete can be replayed after a newer put of its key from the vector WAL;
/// that delete loses, and must not count the live record as garbage.
#[test]
fn a_replayed_write_that_loses_is_dropped_and_charges_nothing() {
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

    let winner = vec![0x11u8; 400];
    let loser = vec![0x22u8; 900];
    store.replay_upsert(b"k", &winner, 10).unwrap();
    store.replay_delete(b"k", 5).unwrap();
    assert_eq!(store.get(b"k").unwrap(), Some(winner.clone()), "the losing replayed delete is dropped");
    assert_eq!(
        store.value_log.total_garbage_bytes(),
        0,
        "a losing delete must not charge the live record"
    );

    store.replay_upsert(b"k", &loser, 7).unwrap();
    assert_eq!(store.get(b"k").unwrap(), Some(winner), "the losing replayed put is dropped");
    assert_eq!(
        store.value_log.total_garbage_bytes(),
        0,
        "a losing put is dropped before its value is appended, so it leaves no garbage"
    );
}

#[test]
fn get_and_get_multiple_surface_corruption_instead_of_hiding_it_as_a_miss() {
    // A value-log read error must not read as a missing key. `get` surfaces a hard
    // error (only a reclaimed segment retries), and so does `get_multiple`: the
    // batch fails rather than reporting the corrupt key absent.
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

    let keys: Vec<Vec<u8>> = (0..5u32).map(|i| format!("k{i}").into_bytes()).collect();
    for (i, k) in keys.iter().enumerate() {
        store.put_to_storage(k, format!("value-{i}-payload").as_bytes()).unwrap();
    }

    // Flip a byte in k2's value payload, in place (behind the log's back — the cached
    // fd reads the modified file, and the in-memory LSM keeps k2's pointer).
    let ptr = decode_sharded_pointer(store.lsm.get(b"k2").unwrap().unwrap()).unwrap();
    let seg_path = dir
        .path()
        .join("value_logs")
        .join(format!("value_log_{}.seg{:06}", ptr.bucket, ptr.location.segment_id));
    let mut bytes = std::fs::read(&seg_path).unwrap();
    bytes[ptr.location.rec_offset as usize + crate::storage::value_log::ValueRecordHeader::SIZE] ^= 0xFF;
    std::fs::write(&seg_path, &bytes).unwrap();

    // With checksum verification on, the flipped payload is detected.
    store.set_verify_checksums_on_read(true);

    // Single-key get surfaces the corruption rather than looping or returning None.
    assert!(
        matches!(store.get(b"k2"), Err(KVError::ShardedValueLogError(_))),
        "corruption must surface as an error, not a missing key"
    );

    // get_multiple reports the corruption too: a corrupt key is not "absent",
    // so the whole batch fails rather than returning `None` for it.
    assert!(
        matches!(store.get_multiple(&keys), Err(KVError::ShardedValueLogError(_))),
        "a batch containing a corrupt key must fail, not report the key missing"
    );

    // The same batch without the corrupt key still returns every value.
    let readable: Vec<Vec<u8>> = keys.iter().filter(|k| k.as_slice() != b"k2").cloned().collect();
    let got = store.get_multiple(&readable).unwrap();
    for (k, v) in readable.iter().zip(&got) {
        let i = keys.iter().position(|x| x == k).unwrap();
        assert_eq!(v.as_deref(), Some(format!("value-{i}-payload").as_bytes()));
    }
}

#[test]
fn test_kvstore_recovery_replay() {
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

    store.replay_upsert(b"rkey1", b"rval1", 1).unwrap();
    store.replay_upsert(b"rkey2", b"rval2", 2).unwrap();
    store.replay_delete(b"rkey1", 3).unwrap();

    assert_eq!(store.get(b"rkey1").unwrap(), None);
    assert_eq!(store.get(b"rkey2").unwrap(), Some(b"rval2".to_vec()));
}

#[test]
fn test_active_memtable_tombstone_shadows_sstable() {
    // A key flushed to an SSTable, then deleted (tombstone lands in the
    // active memtable), must read as absent. The active-memtable layer must
    // treat a tombstone as authoritative and not fall through to the live
    // SSTable value — otherwise a delete is silently resurrected on read
    // until the next compaction.
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

    store.put_to_storage(b"k", b"v").unwrap();
    store.lsm.flush_and_compact_all().unwrap(); // k now lives only in L1
    assert_eq!(store.get(b"k").unwrap(), Some(b"v".to_vec()));

    store.delete_from_storage(b"k").unwrap(); // tombstone in the fresh active memtable
    assert_eq!(store.get(b"k").unwrap(), None, "active-memtable tombstone must shadow the L1 value");
}

#[test]
fn test_put_to_storage_seq_keeps_highest_sequence() {
    // The live winner for two same-key writes is the higher sequence, even
    // when the lower-sequence write is applied last — matching recovery's
    // sequence-ordered replay (live == recovery for racing same-key writes).
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

    store.put_to_storage_seq(b"k", b"newer", 10).unwrap();
    store.put_to_storage_seq(b"k", b"older", 5).unwrap(); // applied later, lower seq → must lose
    assert_eq!(store.get(b"k").unwrap(), Some(b"newer".to_vec()));
}

#[test]
fn test_delete_from_storage_seq_respects_sequence() {
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

    store.put_to_storage_seq(b"k", b"v", 10).unwrap();
    store.delete_from_storage_seq(b"k", 5).unwrap(); // older delete → no-op
    assert_eq!(store.get(b"k").unwrap(), Some(b"v".to_vec()));
    store.delete_from_storage_seq(b"k", 15).unwrap(); // newer delete → removed
    assert_eq!(store.get(b"k").unwrap(), None);
}
