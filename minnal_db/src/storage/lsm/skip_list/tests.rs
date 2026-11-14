//! The skip list's unit tests.

use super::{Applied, KeyValueRecord, SkipList, Version};
use std::collections::BTreeMap;

/// Every outcome the garbage accounting relies on: a replaced live value
/// is reported once, a stale write is reported as stale, and a write over
/// a tombstone or into an absent key displaces nothing.
#[test]
fn writes_report_what_they_displaced() {
    let mut sl = SkipList::new();
    assert_eq!(sl.try_insert_with_seq(b"k", 100, 5).unwrap(), Applied::Won { displaced: None });
    assert_eq!(sl.try_insert_with_seq(b"k", 200, 6).unwrap(), Applied::Won { displaced: Some(100) });
    assert_eq!(sl.try_insert_with_seq(b"k", 300, 4).unwrap(), Applied::Stale);
    assert_eq!(sl.insert_tombstone_with_seq(b"k", 3).unwrap(), Applied::Stale);
    assert_eq!(sl.insert_tombstone_with_seq(b"k", 7).unwrap(), Applied::Won { displaced: Some(200) });
    assert_eq!(sl.insert_tombstone_with_seq(b"k", 8).unwrap(), Applied::Won { displaced: None });
    assert_eq!(sl.try_insert_with_seq(b"k", 400, 9).unwrap(), Applied::Won { displaced: None });
    assert_eq!(sl.insert_tombstone_with_seq(b"absent", 10).unwrap(), Applied::Won { displaced: None });
}

#[test]
fn try_insert_with_seq_keeps_highest_sequence() {
    // Two writes to one key: the higher sequence must win regardless of
    // the order they are applied in. This is what makes the in-memory
    // winner for racing same-key writes match recovery (which replays in
    // sequence order).
    let mut sl = SkipList::new();
    sl.try_insert_with_seq(b"k", 100, 5).unwrap();
    // A lower-sequence write applied *later* must be dropped.
    sl.try_insert_with_seq(b"k", 200, 3).unwrap();
    assert_eq!(sl.get_value(b"k"), Some(100), "older-sequence write must not overwrite");
    // A higher-sequence write wins.
    sl.try_insert_with_seq(b"k", 300, 9).unwrap();
    assert_eq!(sl.get_value(b"k"), Some(300));
}

#[test]
fn insert_tombstone_for_absent_key_creates_tombstone_node() {
    // The production delete path must be able to tombstone a key that is
    // not in the memtable at all (it may be live in a lower LSM layer):
    // the node is created directly as a tombstone, never live.
    let mut sl = SkipList::new();
    sl.insert_tombstone_with_seq(b"k", 4).unwrap();
    assert_eq!(
        sl.entry(b"k"),
        Some((0, Version::user(4), true)),
        "tombstone node carrying the delete's seq"
    );
    assert_eq!(sl.get_value(b"k"), None);
    assert_eq!(sl.number_of_live_nodes(), 0);
    assert_eq!(sl.number_of_tombstone_nodes(), 1);
    sl.validate_insert();
}

#[test]
fn insert_tombstone_flips_live_node_in_place_and_preserves_value_bits() {
    let mut sl = SkipList::new();
    sl.try_insert_with_seq(b"k", 100, 5).unwrap();
    sl.insert_tombstone_with_seq(b"k", 7).unwrap();
    // Tombstoned in place: the stored value bits are untouched (they are
    // meaningless for a tombstone), the sequence is the delete's.
    assert_eq!(sl.entry(b"k"), Some((100, Version::user(7), true)));
    assert_eq!(sl.get_value(b"k"), None);
    assert_eq!(sl.number_of_live_nodes(), 0);
    assert_eq!(sl.number_of_tombstone_nodes(), 1);
}

#[test]
fn insert_tombstone_older_than_live_write_is_dropped() {
    // A delete racing a newer write to the same key must lose, matching
    // recovery's sequence-ordered replay.
    let mut sl = SkipList::new();
    sl.try_insert_with_seq(b"k", 100, 5).unwrap();
    sl.insert_tombstone_with_seq(b"k", 3).unwrap();
    assert_eq!(sl.entry(b"k"), Some((100, Version::user(5), false)), "stale delete must not tombstone");
    assert_eq!(sl.number_of_live_nodes(), 1);
    assert_eq!(sl.number_of_tombstone_nodes(), 0);
}

#[test]
fn write_older_than_tombstone_is_dropped() {
    // The shadow-protection case: the delete wins the sequence race, so a
    // stale write must not resurrect the key.
    let mut sl = SkipList::new();
    sl.insert_tombstone_with_seq(b"k", 10).unwrap();
    sl.try_insert_with_seq(b"k", 999, 5).unwrap();
    assert_eq!(
        sl.entry(b"k"),
        Some((0, Version::user(10), true)),
        "stale write must not resurrect a tombstone"
    );
    assert_eq!(sl.get_value(b"k"), None);
}

#[test]
fn write_newer_than_tombstone_resurrects() {
    let mut sl = SkipList::new();
    sl.insert_tombstone_with_seq(b"k", 3).unwrap();
    sl.try_insert_with_seq(b"k", 42, 5).unwrap();
    assert_eq!(sl.entry(b"k"), Some((42, Version::user(5), false)));
    assert_eq!(sl.number_of_live_nodes(), 1);
    assert_eq!(sl.number_of_tombstone_nodes(), 0);
}

#[test]
fn re_tombstone_bumps_sequence_without_touching_counters() {
    let mut sl = SkipList::new();
    sl.insert_tombstone_with_seq(b"k", 2).unwrap();
    // Deleting an already-deleted key advances the tombstone's sequence...
    sl.insert_tombstone_with_seq(b"k", 6).unwrap();
    assert_eq!(sl.entry(b"k"), Some((0, Version::user(6), true)));
    // ...and an older re-delete is dropped.
    sl.insert_tombstone_with_seq(b"k", 4).unwrap();
    assert_eq!(sl.entry(b"k"), Some((0, Version::user(6), true)));
    assert_eq!(sl.number_of_live_nodes(), 0);
    assert_eq!(sl.number_of_tombstone_nodes(), 1);
    // The seq bump matters: a write between the two delete seqs stays dead.
    sl.try_insert_with_seq(b"k", 7, 5).unwrap();
    assert_eq!(sl.get_value(b"k"), None);
}

#[test]
fn insert_tombstone_seq_comparison_spans_the_whole_sequence_space() {
    // Sequences are compared at full u64 width, so an arbitrarily large
    // gap between a tombstone and a later write still resolves correctly.
    // A truncated, wrapping comparison would let the newer write lose.
    let mut sl = SkipList::new();
    let old = 1_000u64;
    sl.insert_tombstone_with_seq(b"k", old).unwrap();
    // Further apart than 2^31, where a wrapping 32-bit comparison inverts.
    sl.try_insert_with_seq(b"k", 1, old + 3_000_000_000).unwrap();
    assert_eq!(sl.get_value(b"k"), Some(1), "a much-later write must beat the tombstone");
    sl.insert_tombstone_with_seq(b"k", old + 6_000_000_000).unwrap();
    assert_eq!(sl.get_value(b"k"), None);
    // And a stale write from before the gap must still lose.
    sl.try_insert_with_seq(b"k", 9, old + 1).unwrap();
    assert_eq!(sl.get_value(b"k"), None, "a stale low-seq write must not resurrect the key");
}

#[test]
fn absent_key_tombstones_are_ordered_and_invisible_to_iter() {
    // Tombstones created from nothing must still be properly spliced at
    // every level: ordering, linkage, and counters all hold.
    let mut sl = SkipList::new();
    assert!(sl.insert(b"b", 2));
    assert!(sl.insert(b"d", 4));
    sl.insert_tombstone_with_seq(b"a", 100).unwrap();
    sl.insert_tombstone_with_seq(b"c", 101).unwrap();
    sl.insert_tombstone_with_seq(b"e", 102).unwrap();
    sl.validate_insert();

    // iter() skips tombstones...
    let items: Vec<(Vec<u8>, u128)> = sl.iter().map(|(k, v)| (k.to_vec(), v)).collect();
    assert_eq!(items, vec![(b"b".to_vec(), 2), (b"d".to_vec(), 4)]);
    // ...but entry() reports them (the LSM shadow contract).
    assert_eq!(sl.entry(b"a").map(|(_, _, t)| t), Some(true));
    assert_eq!(sl.entry(b"c").map(|(_, _, t)| t), Some(true));
    assert_eq!(sl.entry(b"e").map(|(_, _, t)| t), Some(true));
    assert_eq!(sl.number_of_live_nodes(), 2);
    assert_eq!(sl.number_of_tombstone_nodes(), 3);
}

#[test]
fn randomized_tombstone_inserts_match_model() {
    // Interleave live writes and direct tombstones on a colliding key
    // space and check the final live set + tombstone set against a model.
    // All ops use strictly increasing sequences, so last-op-wins is the
    // expected outcome for every key.
    let mut sl = SkipList::new();
    let mut model = BTreeMap::<Vec<u8>, Option<u128>>::new(); // None = tombstone

    let mut s: u64 = 0xDEAD_BEEF_0BAD_F00D;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        s
    };

    for seq in 1..=4000u64 {
        let r = next();
        let key = format!("key{:03}", r % 500).into_bytes(); // heavy collisions
        if r % 3 == 0 {
            sl.insert_tombstone_with_seq(&key, seq).unwrap();
            model.insert(key, None);
        } else {
            let val = r as u128;
            sl.try_insert_with_seq(&key, val, seq).unwrap();
            model.insert(key, Some(val));
        }
    }
    sl.validate_insert();

    let live_expected: Vec<(Vec<u8>, u128)> = model.iter().filter_map(|(k, v)| v.map(|v| (k.clone(), v))).collect();
    let live_actual: Vec<(Vec<u8>, u128)> = sl.iter().map(|(k, v)| (k.to_vec(), v)).collect();
    assert_eq!(live_actual, live_expected, "live set must match the model in key order");

    for (k, v) in &model {
        let (_, _, tomb) = sl.entry(k).expect("every touched key must have a node");
        assert_eq!(tomb, v.is_none(), "tombstone flag mismatch for {:?}", String::from_utf8_lossy(k));
    }
    assert_eq!(sl.number_of_live_nodes(), live_expected.len());
    assert_eq!(sl.number_of_tombstone_nodes(), model.len() - live_expected.len());
}

#[test]
fn seq_comparison_spans_the_whole_sequence_space() {
    // The comparison must hold across a gap far wider than 2^31, the most
    // a wrapping 32-bit comparison can order.
    let mut sl = SkipList::new();
    sl.try_insert_with_seq(b"k", 1, 1_000).unwrap();
    sl.try_insert_with_seq(b"k", 2, 1_000 + 3_000_000_000).unwrap();
    assert_eq!(sl.get_value(b"k"), Some(2), "a much-later write must win");
    // Past u32::MAX entirely — a truncating store would alias these.
    sl.try_insert_with_seq(b"k", 3, u64::from(u32::MAX) + 1_001).unwrap();
    assert_eq!(sl.get_value(b"k"), Some(3), "sequences beyond u32::MAX must not alias");
}

#[test]
fn entry_distinguishes_tombstone_from_absent() {
    let mut sl = SkipList::new();
    assert_eq!(sl.entry(b"k"), None, "absent key");
    sl.try_insert_with_seq(b"k", 7, 1).unwrap();
    assert_eq!(sl.entry(b"k"), Some((7, Version::user(1), false)));
    sl.insert_tombstone_with_seq(b"k", 2).unwrap();
    assert_eq!(sl.entry(b"k"), Some((7, Version::user(2), true)), "tombstone must be reported, not None");
}

#[test]
fn basic_insert_get_remove() {
    let mut sl = SkipList::new();
    assert!(sl.is_empty());

    assert_eq!(sl.try_insert(b"a", 1).unwrap(), None);
    assert_eq!(sl.try_insert(b"b", 2).unwrap(), None);
    assert_eq!(sl.try_insert(b"a", 3).unwrap(), Some(1));

    assert_eq!(sl.get_value(b"a"), Some(3));
    assert_eq!(sl.get_value(b"b"), Some(2));
    assert_eq!(sl.get_value(b"c"), None);

    sl.validate_insert();

    assert_eq!(sl.remove(b"b"), Some(2));
    assert_eq!(sl.remove(b"b"), None);
    assert_eq!(sl.get_value(b"b"), None);

    sl.validate_insert();
    assert_eq!(sl.number_of_live_nodes(), 1);
}

#[test]
fn iteration_is_sorted_and_skips_tombstones() {
    let mut sl = SkipList::new();
    assert!(sl.insert(b"b", 2));
    assert!(sl.insert(b"a", 1));
    assert!(sl.insert(b"c", 3));
    sl.remove(b"b");

    let items: Vec<(Vec<u8>, u128)> = sl.iter().map(|(k, v)| (k.to_vec(), v)).collect();
    assert_eq!(items, vec![(b"a".to_vec(), 1), (b"c".to_vec(), 3)]);
}

#[test]
fn randomized_against_btreemap() {
    let max_capacity = crate::defaults::memtable::MAX_CAPACITY;
    let mut sl = SkipList::new();
    let mut model = BTreeMap::<Vec<u8>, u128>::new();

    // deterministic LCG for test-data generation
    let mut s: u64 = 0x1234_5678_9ABC_DEF0;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        s
    };

    for i in 0..max_capacity {
        let r = next();
        let mut key = [0u8; 16];
        key[..8].copy_from_slice(&r.to_le_bytes());
        key[8..].copy_from_slice(&next().to_le_bytes());

        match r % 3 {
            0 => {
                let val = r as u128;
                let a = sl.try_insert(&key, val).unwrap();
                let b = model.insert(key.to_vec(), val);
                assert_eq!(a, b);
            }
            1 => {
                let a = sl.remove(&key);
                let b = model.remove(key.as_slice());
                assert_eq!(a, b);
            }
            _ => {
                let a = sl.get_value(&key);
                let b = model.get(key.as_slice()).copied();
                assert_eq!(a, b);
            }
        }

        if i % 257 == 0 {
            sl.validate_insert();
            let sl_items: Vec<(Vec<u8>, u128)> = sl.iter().map(|(k, v)| (k.to_vec(), v)).collect();
            let model_items: Vec<(Vec<u8>, u128)> = model.iter().map(|(k, v)| (k.clone(), *v)).collect();
            assert_eq!(sl_items, model_items);
        }
    }

    sl.validate_insert();
}

#[test]
fn zero_value_is_valid() {
    let mut sl = SkipList::new();

    // 0 is a valid value: a tombstone is a flag, not a reserved value.
    assert!(sl.insert(b"a", 0));
    assert_eq!(sl.get_value(b"a"), Some(0));

    // u128::MAX is also allowed; tombstones are explicit.
    assert!(sl.insert(b"b", u128::MAX));
    assert_eq!(sl.get_value(b"b"), Some(u128::MAX));

    assert!(sl.try_insert(b"c", 0).is_ok());
}

#[test]
fn iter_raw_yields_all_entries_including_tombstones() {
    let mut sl = SkipList::new();
    assert!(sl.insert(b"a", 1));
    assert!(sl.insert(b"b", 2));
    assert!(sl.insert(b"c", 3));
    sl.remove(b"b");

    let raw: Vec<_> = sl.iter_raw().map(|(k, v, _s, t)| (k.to_vec(), v, t)).collect();
    assert_eq!(raw.len(), 3);

    // "a" — live
    assert_eq!(raw[0].0, b"a");
    assert_eq!(raw[0].1, 1);
    assert!(!raw[0].2);

    // "b" — tombstoned
    assert_eq!(raw[1].0, b"b");
    assert!(raw[1].2);

    // "c" — live
    assert_eq!(raw[2].0, b"c");
    assert_eq!(raw[2].1, 3);
    assert!(!raw[2].2);
}

#[test]
fn prefix_iter_borrowed_prefix() {
    let mut sl = SkipList::new();
    assert!(sl.insert(b"foo:1", 1));
    assert!(sl.insert(b"foo:2", 2));
    assert!(sl.insert(b"bar:1", 3));
    // tombstone one matching key
    sl.remove(b"foo:2");

    let items: Vec<_> = sl.scan_prefix(b"foo:" as &[u8]).map(|(k, v)| (k.to_vec(), v)).collect();
    assert_eq!(items, vec![(b"foo:1".to_vec(), 1)]);
}

#[test]
fn prefix_iter_owned_prefix() {
    let mut sl = SkipList::new();
    assert!(sl.insert(b"ns:a", 10));
    assert!(sl.insert(b"ns:b", 20));
    assert!(sl.insert(b"other", 99));

    let owned_prefix: Vec<u8> = b"ns:".to_vec();
    let items: Vec<_> = sl.scan_prefix(owned_prefix).map(|(k, v)| (k.to_vec(), v)).collect();
    assert_eq!(items, vec![(b"ns:a".to_vec(), 10), (b"ns:b".to_vec(), 20)]);
}

#[test]
fn prefix_iter_empty_prefix_yields_all_live() {
    let mut sl = SkipList::new();
    assert!(sl.insert(b"a", 1));
    assert!(sl.insert(b"b", 2));
    sl.remove(b"b");

    let items: Vec<_> = sl.scan_prefix(b"" as &[u8]).map(|(k, _)| k.to_vec()).collect();
    assert_eq!(items, vec![b"a".to_vec()]);
}

#[test]
fn prefix_iter_no_match_returns_empty() {
    let mut sl = SkipList::new();
    assert!(sl.insert(b"abc", 1));

    let items: Vec<_> = sl.scan_prefix(b"xyz" as &[u8]).collect();
    assert!(items.is_empty());
}

#[test]
fn collect_entries_sorted_is_sorted_and_excludes_sentinel() {
    let mut sl = SkipList::new();
    assert!(sl.insert(b"b", 2));
    assert!(sl.insert(b"a", 1));
    sl.remove(b"b");

    let v = sl.collect_key_value_records();
    assert_eq!(v.len(), 2);

    // Sorted ascending by key.
    assert_eq!(v[0].key.as_slice(), b"a");
    assert!(!v[0].tombstone);
    assert_eq!(v[0].value, 1);

    assert_eq!(v[1].key.as_slice(), b"b");
    assert!(v[1].tombstone);

    // Ensure no empty key entry leaked.
    assert!(v.iter().all(|e| !e.key.is_empty()));

    // Touch KeyValueRecord so the import isn't unused if cfg differs.
    let _ = KeyValueRecord {
        key: vec![],
        value: 0,
        tombstone: true,
        version: Version::default(),
    };
}
