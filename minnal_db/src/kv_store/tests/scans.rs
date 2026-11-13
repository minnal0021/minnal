//! Batch gets and scans.

use super::*;
use crate::namespace::NamespaceId;

#[test]
fn test_get_multiple_matches_individual_get() {
    let dir = TempDir::new().unwrap();
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "ns",
        dir.path(),
        default_lsm_config(),
        SyncConfig::default(),
        crate::defaults::value_log::SEGMENT_SIZE_BYTES,
    )
    .unwrap();

    let pairs: Vec<(Vec<u8>, Vec<u8>)> = (0u8..20).map(|i| (format!("doc:{i:03}").into_bytes(), vec![i; 64])).collect();
    for (k, v) in &pairs {
        store.put_to_storage(k, v).unwrap();
    }

    let keys: Vec<Vec<u8>> = pairs.iter().map(|(k, _)| k.clone()).collect();
    let batch = store.get_multiple(&keys).unwrap();

    assert_eq!(batch.len(), keys.len());
    for (i, result) in batch.iter().enumerate() {
        let expected = store.get(&keys[i]).unwrap();
        assert_eq!(result, &expected, "mismatch at index {i}");
    }
}

#[test]
fn test_get_multiple_missing_keys_are_none() {
    let dir = TempDir::new().unwrap();
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "ns",
        dir.path(),
        default_lsm_config(),
        SyncConfig::default(),
        crate::defaults::value_log::SEGMENT_SIZE_BYTES,
    )
    .unwrap();

    store.put_to_storage(b"exists", b"val").unwrap();

    let keys = vec![b"exists".to_vec(), b"no_such_key".to_vec()];
    let batch = store.get_multiple(&keys).unwrap();

    assert_eq!(batch[0], Some(b"val".to_vec()));
    assert_eq!(batch[1], None);
}

#[test]
fn test_get_multiple_after_flush_to_sstable() {
    let dir = TempDir::new().unwrap();
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "ns",
        dir.path(),
        default_lsm_config(),
        SyncConfig::default(),
        crate::defaults::value_log::SEGMENT_SIZE_BYTES,
    )
    .unwrap();

    for i in 0u8..10 {
        store.put_to_storage(&[i], &[i; 32]).unwrap();
    }
    // Force a flush so keys live in the SSTable, not the active memtable.
    store.flush_and_compact_all().unwrap();

    let keys: Vec<Vec<u8>> = (0u8..10).map(|i| vec![i]).collect();
    let batch = store.get_multiple(&keys).unwrap();

    for (i, result) in batch.iter().enumerate() {
        assert_eq!(result.as_deref(), Some(vec![i as u8; 32].as_slice()), "missing key {i}");
    }
}

// ── scan_prefix_batch ─────────────────────────────────────────────────────

fn u32_prefixed_key(prefix: u32, suffix: u32) -> Vec<u8> {
    let mut k = prefix.to_be_bytes().to_vec();
    k.extend_from_slice(&suffix.to_be_bytes());
    k
}

#[test]
fn test_scan_prefixes_batch_basic() {
    let dir = TempDir::new().unwrap();
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "test_ns",
        dir.path(),
        default_lsm_config(),
        SyncConfig::default(),
        crate::defaults::value_log::SEGMENT_SIZE_BYTES,
    )
    .unwrap();

    let entries = [
        (7u32, 1u32, b"val_7_1".as_slice()),
        (7, 2, b"val_7_2"),
        (7, 3, b"val_7_3"),
        (8, 1, b"val_8_1"),
    ];
    for (pid, sid, val) in entries {
        store.put_to_storage(&u32_prefixed_key(pid, sid), val).unwrap();
    }

    let result = store.scan_prefixes_batch(&[7, 8]).unwrap();

    assert_eq!(result.iter().filter(|(p, _, _)| *p == 7).count(), 3);
    assert_eq!(result.iter().filter(|(p, _, _)| *p == 8).count(), 1);

    let vals_7: Vec<&[u8]> = result.iter().filter(|(p, _, _)| *p == 7).map(|(_, _, v)| v).collect();
    assert!(vals_7.contains(&b"val_7_1".as_slice()));
    assert!(vals_7.contains(&b"val_7_2".as_slice()));
    assert!(vals_7.contains(&b"val_7_3".as_slice()));
}

/// Following `next_cursor` must always make progress. A `limit` of 0 returned
/// an empty page whose cursor was the one passed in, so a paging loop never
/// ended.
#[test]
fn test_scan_page_batch_limit_zero_still_advances() {
    let dir = TempDir::new().unwrap();
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "t",
        dir.path(),
        default_lsm_config(),
        SyncConfig::default(),
        crate::defaults::value_log::SEGMENT_SIZE_BYTES,
    )
    .unwrap();
    for i in 0..3u8 {
        store.put_to_storage(&[b'k', i], b"v").unwrap();
    }
    let mut cursor: Option<Vec<u8>> = None;
    let mut seen = Vec::new();
    for _ in 0..10 {
        let (page, next) = store.scan_page_batch(cursor.as_deref(), None, 0).unwrap();
        seen.extend(page.into_iter().map(|(k, _)| k));
        match next {
            Some(c) => cursor = Some(c),
            None => break,
        }
    }
    assert_eq!(
        seen,
        vec![vec![b'k', 0], vec![b'k', 1], vec![b'k', 2]],
        "the walk must advance and finish"
    );
}

#[test]
fn test_scan_page_batch_end_bound_excludes_upper() {
    let dir = TempDir::new().unwrap();
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "test_ns",
        dir.path(),
        default_lsm_config(),
        SyncConfig::default(),
        crate::defaults::value_log::SEGMENT_SIZE_BYTES,
    )
    .unwrap();
    for i in 0u32..10 {
        store.put_to_storage(format!("k{i:02}").as_bytes(), format!("v{i}").as_bytes()).unwrap();
    }

    // [None, "k05") must yield k00..k04 and stop *before* the exclusive bound.
    let (pairs, next) = store.scan_page_batch(None, Some(b"k05"), 100).unwrap();
    let keys: Vec<Vec<u8>> = pairs.into_iter().map(|(k, _)| k).collect();
    assert_eq!(keys, (0u32..5).map(|i| format!("k{i:02}").into_bytes()).collect::<Vec<_>>());
    assert!(next.is_none(), "the whole [start, end) window fit in one page");
}

#[test]
fn test_scan_page_batch_cursor_walk_is_complete_and_ordered() {
    let dir = TempDir::new().unwrap();
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "test_ns",
        dir.path(),
        default_lsm_config(),
        SyncConfig::default(),
        crate::defaults::value_log::SEGMENT_SIZE_BYTES,
    )
    .unwrap();
    for i in 0u32..10 {
        store.put_to_storage(format!("k{i:02}").as_bytes(), format!("v{i}").as_bytes()).unwrap();
    }

    // Walk the whole keyspace in pages of 3 and confirm no key is dropped or
    // duplicated and global order is preserved across page boundaries.
    let mut cursor: Option<Vec<u8>> = None;
    let mut seen: Vec<Vec<u8>> = Vec::new();
    loop {
        let (pairs, next) = store.scan_page_batch(cursor.as_deref(), None, 3).unwrap();
        assert!(pairs.len() <= 3);
        seen.extend(pairs.into_iter().map(|(k, _)| k));
        match next {
            Some(c) => cursor = Some(c),
            None => break,
        }
    }
    assert_eq!(seen, (0u32..10).map(|i| format!("k{i:02}").into_bytes()).collect::<Vec<_>>());
}

#[test]
fn test_scan_prefixes_batch_missing_prefix_absent() {
    let dir = TempDir::new().unwrap();
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "test_ns",
        dir.path(),
        default_lsm_config(),
        SyncConfig::default(),
        crate::defaults::value_log::SEGMENT_SIZE_BYTES,
    )
    .unwrap();

    store.put_to_storage(&u32_prefixed_key(1, 1), b"v").unwrap();

    let result = store.scan_prefixes_batch(&[99]).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_scan_prefixes_batch_after_flush() {
    let dir = TempDir::new().unwrap();
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "test_ns",
        dir.path(),
        default_lsm_config(),
        SyncConfig::default(),
        crate::defaults::value_log::SEGMENT_SIZE_BYTES,
    )
    .unwrap();

    for sid in 0u32..5 {
        store.put_to_storage(&u32_prefixed_key(2, sid), &[sid as u8; 16]).unwrap();
    }
    store.flush_and_compact_all().unwrap();

    // Add more entries in a fresh memtable to test the multi-source merge
    for sid in 5u32..8 {
        store.put_to_storage(&u32_prefixed_key(2, sid), &[sid as u8; 16]).unwrap();
    }

    let result = store.scan_prefixes_batch(&[2]).unwrap();
    assert_eq!(result.len(), 8);
    // Key order, and every value read back whole.
    let found: Vec<(Vec<u8>, Vec<u8>)> = result.iter().map(|(_, k, v)| (k.to_vec(), v.to_vec())).collect();
    let expected: Vec<(Vec<u8>, Vec<u8>)> = (0u32..8).map(|sid| (u32_prefixed_key(2, sid), vec![sid as u8; 16])).collect();
    assert_eq!(found, expected);
}

/// A whole-namespace pass reads the keys a page at a time. Every live key must
/// be visited exactly once, in order, across page boundaries, whichever layer it
/// is in, and a deleted key not at all. The keys include `k` followed by `k\0`:
/// the next page starts at the last key plus a zero byte, so a page ending at `k`
/// must still visit `k\0`.
#[test]
fn key_pages_visit_every_live_key_once() {
    use crate::kv_store::KEY_PAGE;
    use std::ops::ControlFlow;

    let dir = TempDir::new().unwrap();
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "ns",
        dir.path(),
        default_lsm_config(),
        SyncConfig::default(),
        crate::defaults::value_log::SEGMENT_SIZE_BYTES,
    )
    .unwrap();

    let n = KEY_PAGE * 3;
    let key = |i: usize| format!("key{i:06}").into_bytes();
    // Half on disk, half in the memtable; every seventh key deleted afterwards.
    for i in 0..n / 2 {
        store.put_to_storage(&key(i), b"v").unwrap();
    }
    store.flush_memtable_to_level0().unwrap();
    for i in n / 2..n {
        store.put_to_storage(&key(i), b"v").unwrap();
    }
    for i in (0..n).step_by(7) {
        store.delete_from_storage(&key(i)).unwrap();
    }
    // The first page ends at the `KEY_PAGE`th live key; its zero-byte successor
    // sorts straight after it, so it is the first key of the second page.
    let mut zero_successor = store.keys().unwrap()[KEY_PAGE - 1].clone();
    zero_successor.push(0);
    store.put_to_storage(&zero_successor, b"v").unwrap();

    let expected = store.keys().unwrap();
    assert!(expected.len() > 2 * KEY_PAGE, "the test needs at least three pages");
    assert!(expected.contains(&zero_successor));

    let mut visited = Vec::new();
    let mut pages = 0;
    store
        .for_each_key_page(&[], None, |page| {
            assert!(page.len() <= KEY_PAGE);
            pages += 1;
            visited.extend(page.into_iter().map(|(k, ..)| k));
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(visited, expected, "a paged pass must visit every live key once, in order");
    assert_eq!(pages, expected.len().div_ceil(KEY_PAGE));
    assert_eq!(store.count_keys().unwrap(), expected.len());

    // A visitor that breaks stops the pass after that page.
    let mut pages = 0;
    store
        .for_each_key_page(&[], None, |_| {
            pages += 1;
            Ok(ControlFlow::Break(()))
        })
        .unwrap();
    assert_eq!(pages, 1);

    // The cursor-paginated form returns the same keys.
    let mut walked = Vec::new();
    let mut cursor = None;
    loop {
        let (keys, next) = store.key_page(cursor.as_deref(), None, 100).unwrap();
        assert!(keys.len() <= 100);
        walked.extend(keys);
        match next {
            Some(next) => cursor = Some(next),
            None => break,
        }
    }
    assert_eq!(walked, expected);
}

/// A key whose segment GC reclaimed mid-scan is re-read through `get`, and must
/// go back in its place: a scan's result is in key order.
#[test]
fn keys_re_read_after_a_reclaimed_segment_keep_the_scan_in_key_order() {
    let dir = TempDir::new().unwrap();
    let store = KVStore::open(
        NamespaceId::from_u64(0),
        "ns",
        dir.path(),
        default_lsm_config(),
        SyncConfig::default(),
        crate::defaults::value_log::SEGMENT_SIZE_BYTES,
    )
    .unwrap();
    for key in [b"a", b"b", b"c", b"d"] {
        store.put_to_storage(key, key).unwrap();
    }

    // The batch read returned `a` and `d`; `b` and `c` lost their segment.
    let mut pairs = vec![(b"a".to_vec(), b"a".to_vec()), (b"d".to_vec(), b"d".to_vec())];
    store.refetch_dropped(&mut pairs, vec![b"b".to_vec(), b"c".to_vec()]).unwrap();

    let keys: Vec<&[u8]> = pairs.iter().map(|(k, _)| k.as_slice()).collect();
    assert_eq!(keys, [b"a", b"b", b"c", b"d"]);
}
