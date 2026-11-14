//! Bounded range pages (`range_pointers_bounded`) across layers and tombstones.

use super::*;

/// A page reads about its own size from each layer, however many keys lie
/// after the cursor: with 20,000 keys in Level 1 and 20,000 more in the
/// memtable, a page of 10 from the start reads a few dozen entries.
#[test]
fn a_range_page_reads_about_its_own_size() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
    let key = |i: u32| format!("r:{i:06}").into_bytes();
    for i in 0..20_000 {
        lsm.insert_with_seq(&key(2 * i), u128::from(i) + 1, u64::from(i) + 1)?;
    }
    lsm.flush_memtable_to_level0()?;
    lsm.compact_all()?;
    for i in 0..20_000 {
        lsm.insert_with_seq(&key(2 * i + 1), u128::from(i) + 1, 100_000 + u64::from(i))?;
    }

    range_scan::ENTRIES_READ.with(|n| n.set(0));
    let page = lsm.range_pointers_bounded(b"r:", None, 10)?;
    let read = range_scan::ENTRIES_READ.with(|n| n.get());
    assert_eq!(page.iter().map(|e| e.0.clone()).collect::<Vec<_>>(), (0..10).map(key).collect::<Vec<_>>());
    assert!(read <= 100, "a page of 10 read {read} entries");
    Ok(())
}

/// When newer layers delete most of a window, the page is read again with
/// a wider one until it fills: 5,000 keys deleted in the memtable ahead of
/// the live ones still give the first 10 live keys.
#[test]
fn a_range_page_past_many_deletes_widens_until_full() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
    let key = |i: u32| format!("d:{i:06}").into_bytes();
    for i in 0..6_000 {
        lsm.insert_with_seq(&key(i), u128::from(i) + 1, u64::from(i) + 1)?;
    }
    lsm.flush_memtable_to_level0()?;
    lsm.compact_all()?;
    for i in 0..5_000 {
        lsm.delete_with_seq(&key(i), 10_000 + u64::from(i))?;
    }

    range_scan::ENTRIES_READ.with(|n| n.set(0));
    let page = lsm.range_pointers_bounded(b"d:", None, 10)?;
    let read = range_scan::ENTRIES_READ.with(|n| n.get());
    assert_eq!(
        page.iter().map(|e| e.0.clone()).collect::<Vec<_>>(),
        (5_000..5_010).map(key).collect::<Vec<_>>()
    );
    // Windows of 11, 44, ..., 11,264 entries per layer: bounded by a few
    // times what the page had to pass over, not by the namespace.
    assert!(read < 60_000, "read {read} entries");
    Ok(())
}

// ── range pages and tombstones ────────────────────────────────────────────

#[test]
fn test_range_keys_deleted_after_flush() -> Result<()> {
    // A key deleted after a flush must not appear in a range page.
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"r:1", 1)?;
    lsm.insert(b"r:2", 2)?;
    lsm.insert(b"r:3", 3)?;
    lsm.flush_memtable()?;

    lsm.delete(b"r:2")?; // tombstone in active memtable

    let keys = range_keys(&lsm, b"r:", 100)?;
    assert!(!keys.contains(&b"r:2".to_vec()), "deleted key must not appear");
    assert!(keys.contains(&b"r:1".to_vec()));
    assert!(keys.contains(&b"r:3".to_vec()));
    assert_eq!(keys.len(), 2);
    Ok(())
}

#[test]
fn test_range_keys_tombstone_in_ro_memtable() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"m:1", 10)?;
    lsm.insert(b"m:2", 20)?;
    lsm.flush_memtable()?; // both in SSTable
    lsm.delete(b"m:1")?;
    lsm.flush_memtable()?; // tombstone in ro_memtable

    let keys = range_keys(&lsm, b"m:", 100)?;
    assert!(!keys.contains(&b"m:1".to_vec()), "key tombstoned in ro_memtable must not appear");
    assert!(keys.contains(&b"m:2".to_vec()));
    assert_eq!(keys.len(), 1);
    Ok(())
}

#[test]
fn test_range_keys_delete_then_reinsert() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"q:1", 1)?;
    lsm.flush_memtable()?;
    lsm.delete(b"q:1")?;
    lsm.flush_memtable()?; // tombstone in ro_memtable
    lsm.insert(b"q:1", 2)?; // re-insert in active memtable

    let keys = range_keys(&lsm, b"q:", 100)?;
    assert_eq!(keys, vec![b"q:1".to_vec()], "re-inserted key must appear exactly once");
    Ok(())
}

// ── range_pointers_bounded and tombstones ────────────────────────────

#[test]
fn test_range_pointers_bounded_deleted_after_flush() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"p:1", 10)?;
    lsm.insert(b"p:2", 20)?;
    lsm.flush_memtable()?;

    lsm.delete(b"p:2")?; // tombstone in active memtable

    let pairs = lsm.range_pointers_bounded(b"p:", None, 100)?;
    assert_eq!(pairs.len(), 1, "deleted key must not appear");
    assert_eq!(pairs[0].0, b"p:1");
    assert_eq!(pairs[0].1, 10);
    Ok(())
}

#[test]
fn test_range_pointers_bounded_tombstone_in_ro_memtable() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"w:1", 100)?;
    lsm.insert(b"w:2", 200)?;
    lsm.flush_memtable()?;
    lsm.delete(b"w:1")?;
    lsm.flush_memtable()?; // tombstone in ro_memtable

    let pairs = lsm.range_pointers_bounded(b"w:", None, 100)?;
    assert!(!pairs.iter().any(|(k, _, _)| k == b"w:1"), "key deleted in ro_memtable must not appear");
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].0, b"w:2");
    Ok(())
}

// ── L0 visibility tests (buffered L1 + L0 scan correctness) ───────────────

// Flush data to L0 SSTable files and then purge the in-memory ro_memtable so
// the only copy of the data is on disk in L0.  Both range functions must still
// return the data by reading L0 files directly.

#[test]
fn test_range_pointers_bounded_sees_l0_only_data() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"l0:1", 11)?;
    lsm.insert(b"l0:2", 22)?;
    lsm.insert(b"l0:3", 33)?;
    // Flush active memtable → ro_memtable, then flush ro_memtable → L0 files.
    lsm.flush_memtable_to_level0()?;
    // Evict the ro_memtable: data is now only in L0 SSTable files.
    lsm.purge_ro_memtables_for_test();

    let pairs = lsm.range_pointers_bounded(b"l0:", None, 100)?;
    assert_eq!(pairs.len(), 3, "all three keys must be visible from L0");
    let keys: Vec<&[u8]> = pairs.iter().map(|(k, _, _)| k.as_slice()).collect();
    assert!(keys.contains(&b"l0:1".as_ref()));
    assert!(keys.contains(&b"l0:2".as_ref()));
    assert!(keys.contains(&b"l0:3".as_ref()));
    assert_eq!(pairs.iter().find(|(k, _, _)| k == b"l0:1").unwrap().1, 11);
    assert_eq!(pairs.iter().find(|(k, _, _)| k == b"l0:2").unwrap().1, 22);
    assert_eq!(pairs.iter().find(|(k, _, _)| k == b"l0:3").unwrap().1, 33);
    Ok(())
}

#[test]
fn test_range_keys_sees_l0_only_data() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"lk:a", 1)?;
    lsm.insert(b"lk:b", 2)?;
    lsm.flush_memtable_to_level0()?;
    lsm.purge_ro_memtables_for_test();

    let keys = range_keys(&lsm, b"lk:", 100)?;
    assert_eq!(keys.len(), 2, "both keys must be visible from L0");
    assert!(keys.contains(&b"lk:a".to_vec()));
    assert!(keys.contains(&b"lk:b".to_vec()));
    Ok(())
}

#[test]
fn test_range_pointers_bounded_l0_tombstone_suppresses_l1() -> Result<()> {
    // Write a key → compact to L1 → delete it (write tombstone to L0) → verify
    // the range scan returns nothing.
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"lt:1", 99)?;
    lsm.flush_memtable()?;
    lsm.compact_all()?; // data is now in L1, ro_memtable dropped

    lsm.delete(b"lt:1")?;
    lsm.flush_memtable_to_level0()?; // tombstone written to L0 file
    lsm.purge_ro_memtables_for_test(); // only L0 tombstone + L1 data remain

    let pairs = lsm.range_pointers_bounded(b"lt:", None, 100)?;
    assert!(pairs.is_empty(), "L0 tombstone must suppress L1 data");
    Ok(())
}

#[test]
fn test_range_keys_l0_tombstone_suppresses_l1() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"lkt:x", 5)?;
    lsm.flush_memtable()?;
    lsm.compact_all()?;

    lsm.delete(b"lkt:x")?;
    lsm.flush_memtable_to_level0()?;
    lsm.purge_ro_memtables_for_test();

    let keys = range_keys(&lsm, b"lkt:", 100)?;
    assert!(keys.is_empty(), "L0 tombstone must suppress L1 data");
    Ok(())
}

#[test]
fn test_range_pointers_bounded_l0_reinsert_after_l1_delete() -> Result<()> {
    // Delete a key (tombstone in L1 after compact), then re-insert it in L0.
    // The re-insert in L0 must win.
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"lr:1", 10)?;
    lsm.flush_memtable()?;
    lsm.delete(b"lr:1")?;
    lsm.flush_memtable()?;
    lsm.compact_all()?; // tombstone in L1, ro_memtable dropped

    lsm.insert(b"lr:1", 20)?; // re-insert
    lsm.flush_memtable_to_level0()?;
    lsm.purge_ro_memtables_for_test(); // re-insert only in L0

    let pairs = lsm.range_pointers_bounded(b"lr:", None, 100)?;
    assert_eq!(pairs.len(), 1, "re-inserted key must be visible");
    assert_eq!(pairs[0].0, b"lr:1");
    assert_eq!(pairs[0].1, 20);
    Ok(())
}

#[test]
fn test_range_pointers_bounded_after_compact_all() -> Result<()> {
    // Verify that after a full compact_all cycle, data is visible via L1.
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"ca:1", 1)?;
    lsm.insert(b"ca:2", 2)?;
    lsm.flush_memtable()?;
    lsm.compact_all()?; // data in L1, ro_memtable dropped, L0 obsolete

    let pairs = lsm.range_pointers_bounded(b"ca:", None, 100)?;
    assert_eq!(pairs.len(), 2);
    assert!(pairs.iter().any(|(k, v, _)| k == b"ca:1" && *v == 1));
    assert!(pairs.iter().any(|(k, v, _)| k == b"ca:2" && *v == 2));
    Ok(())
}

#[test]
fn test_range_pointers_bounded_limit_respected_with_l0() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    for i in 0u128..10 {
        lsm.insert(format!("lim:{:02}", i).as_bytes(), i)?;
    }
    lsm.flush_memtable_to_level0()?;
    lsm.purge_ro_memtables_for_test();

    let pairs = lsm.range_pointers_bounded(b"lim:", None, 3)?;
    assert_eq!(pairs.len(), 3, "limit must be respected");
    // BTreeMap ordering guarantees the first 3 lexicographic keys.
    assert_eq!(pairs[0].0, b"lim:00");
    assert_eq!(pairs[1].0, b"lim:01");
    assert_eq!(pairs[2].0, b"lim:02");
    Ok(())
}

#[test]
fn test_range_keys_limit_respected_with_l0() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    for i in 0u128..8 {
        lsm.insert(format!("klim:{:02}", i).as_bytes(), i)?;
    }
    lsm.flush_memtable_to_level0()?;
    lsm.purge_ro_memtables_for_test();

    let keys = range_keys(&lsm, b"klim:", 4)?;
    assert_eq!(keys.len(), 4, "limit must be respected");
    assert_eq!(keys[0], b"klim:00");
    assert_eq!(keys[3], b"klim:03");
    Ok(())
}

// ── bounded range pages: newer tombstones never short a page ───────────────

/// A newer-layer (memtable) tombstone shadowing an L1 key *inside* the first `limit`
/// keys must not short the page: the window is widened until `limit` live keys come
/// back. num_buckets=1 puts every key in one L1 run. Reading L1 only up to
/// `limit`+1 entries, without widening, would return about 11 keys.
#[test]
fn test_range_pointers_bounded_page_not_shorted_by_newer_tombstones() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(
        temp_dir.path(),
        LSMConfig {
            num_buckets: 1,
            ..test_lsm_config()
        },
    )?;
    for i in 0u128..50 {
        lsm.insert(format!("k{:02}", i).as_bytes(), i)?;
    }
    lsm.flush_memtable()?;
    lsm.compact_all()?; // k00..k49 all live in L1
    // Delete the first 10 keys — tombstones live in the active memtable (newer than L1).
    for i in 0u128..10 {
        lsm.delete(format!("k{:02}", i).as_bytes())?;
    }
    let pairs = lsm.range_pointers_bounded(b"k", None, 20)?;
    let keys: Vec<String> = pairs.iter().map(|(k, _, _)| String::from_utf8_lossy(k).into_owned()).collect();
    assert_eq!(
        keys.len(),
        20,
        "page must return a full `limit` of live keys despite newer tombstones; got {keys:?}"
    );
    assert_eq!(keys[0], "k10", "deleted head keys must be skipped");
    assert_eq!(keys[19], "k29");
    Ok(())
}

/// Keys-only twin of `test_range_pointers_bounded_page_not_shorted_by_newer_tombstones`.
#[test]
fn test_range_keys_page_not_shorted_by_newer_tombstones() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(
        temp_dir.path(),
        LSMConfig {
            num_buckets: 1,
            ..test_lsm_config()
        },
    )?;
    for i in 0u128..50 {
        lsm.insert(format!("k{:02}", i).as_bytes(), i)?;
    }
    lsm.flush_memtable()?;
    lsm.compact_all()?;
    for i in 0u128..10 {
        lsm.delete(format!("k{:02}", i).as_bytes())?;
    }
    let keys = range_keys(&lsm, b"k", 20)?;
    assert_eq!(keys.len(), 20, "keys page must be full despite newer tombstones");
    assert_eq!(keys[0], b"k10");
    assert_eq!(keys[19], b"k29");
    Ok(())
}

/// Tombstones clustered in one bucket: 20 of them, all hashing to a single bucket,
/// sit at the page head. That bucket's window must grow past them for the page to
/// come back full; a window that stopped growing would read only tombstones there
/// and short the page.
#[test]
fn test_range_pointers_bounded_skip_clustered_tombstones() -> Result<()> {
    let num_buckets = 8;
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(
        temp_dir.path(),
        LSMConfig {
            num_buckets,
            ..test_lsm_config()
        },
    )?;

    // 40 keys that all hash to bucket 0 (prefix "a", so they sort ahead of everything else).
    let mut bucket0: Vec<Vec<u8>> = Vec::new();
    let mut i = 0u64;
    while bucket0.len() < 40 {
        let k = format!("a{i:06}").into_bytes();
        if get_bucket_for_key(&k, num_buckets) == 0 {
            bucket0.push(k);
        }
        i += 1;
    }
    // Filler spread across the OTHER buckets, at higher keys (prefix "b").
    let mut others: Vec<Vec<u8>> = Vec::new();
    let mut j = 0u64;
    while others.len() < 40 {
        let k = format!("b{j:06}").into_bytes();
        if get_bucket_for_key(&k, num_buckets) != 0 {
            others.push(k);
        }
        j += 1;
    }
    for (v, k) in bucket0.iter().chain(others.iter()).enumerate() {
        lsm.insert(k, v as u128)?;
    }
    lsm.flush_memtable()?;
    lsm.compact_all()?; // all live in L1

    // Delete the first 20 bucket-0 keys — all in ONE bucket, all at the page head.
    for k in &bucket0[..20] {
        lsm.delete(k)?;
    }

    // Scan the "a" range: only bucket 0 has in-range keys, so its window is what
    // gates the page. limit=15 needs the window to reach past all 20 head tombstones.
    let pairs = lsm.range_pointers_bounded(b"a", Some(b"b"), 15)?;
    let keys: Vec<Vec<u8>> = pairs.iter().map(|(k, _, _)| k.clone()).collect();
    assert_eq!(
        keys.len(),
        15,
        "page must be full despite 20 clustered tombstones in one bucket; got {}",
        keys.len()
    );
    assert_eq!(
        keys.as_slice(),
        &bucket0[20..35],
        "page must be the first live keys after the deleted head"
    );
    Ok(())
}

/// Keys-only twin of `test_range_pointers_bounded_skip_clustered_tombstones`.
#[test]
fn test_range_keys_skip_clustered_tombstones() -> Result<()> {
    let num_buckets = 8;
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(
        temp_dir.path(),
        LSMConfig {
            num_buckets,
            ..test_lsm_config()
        },
    )?;

    let mut bucket0: Vec<Vec<u8>> = Vec::new();
    let mut i = 0u64;
    while bucket0.len() < 40 {
        let k = format!("a{i:06}").into_bytes();
        if get_bucket_for_key(&k, num_buckets) == 0 {
            bucket0.push(k);
        }
        i += 1;
    }
    let mut others: Vec<Vec<u8>> = Vec::new();
    let mut j = 0u64;
    while others.len() < 40 {
        let k = format!("b{j:06}").into_bytes();
        if get_bucket_for_key(&k, num_buckets) != 0 {
            others.push(k);
        }
        j += 1;
    }
    for (v, k) in bucket0.iter().chain(others.iter()).enumerate() {
        lsm.insert(k, v as u128)?;
    }
    lsm.flush_memtable()?;
    lsm.compact_all()?;

    for k in &bucket0[..20] {
        lsm.delete(k)?;
    }

    let keys = range_keys(&lsm, b"a", 15)?;
    assert_eq!(keys.len(), 15, "keys page must be full despite clustered tombstones; got {}", keys.len());
    assert_eq!(keys.as_slice(), &bucket0[20..35]);
    Ok(())
}

/// A run of tombstones **in L1 itself** at the page head must be skipped without
/// shorting the page (the L1 early-stop counts LIVE entries, not raw entries). The
/// tombstones reach L1 by compacting a delete-over-an-L0-value into level 1.
#[test]
fn test_range_pointers_bounded_l1_tombstones_at_head_skipped() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(
        temp_dir.path(),
        LSMConfig {
            num_buckets: 1,
            ..test_lsm_config()
        },
    )?;
    for i in 0u128..40 {
        lsm.insert(format!("k{:02}", i).as_bytes(), i)?;
    }
    lsm.flush_memtable_to_level0()?; // values land in L0
    // Delete the first 8, flush the tombstones to L0, then compact everything to L1 so
    // the L1 file physically carries tombstones for k00..k07 ahead of the live keys.
    for i in 0u128..8 {
        lsm.delete(format!("k{:02}", i).as_bytes())?;
    }
    lsm.flush_and_compact_all()?;
    lsm.purge_ro_memtables_for_test();
    let pairs = lsm.range_pointers_bounded(b"k", None, 20)?;
    let keys: Vec<String> = pairs.iter().map(|(k, _, _)| String::from_utf8_lossy(k).into_owned()).collect();
    assert_eq!(keys.len(), 20, "page must be full past the L1-tombstone head; got {keys:?}");
    assert_eq!(keys[0], "k08");
    assert_eq!(keys[19], "k27");
    Ok(())
}

/// A full cursor walk with small pages, over multiple buckets with scattered deletes,
/// must return exactly the same ordered set as one unbounded scan: the windowed
/// read must never drop or duplicate a live key across page boundaries.
#[test]
fn test_bounded_page_walk_matches_unbounded_reference() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?; // default multi-bucket
    for i in 0u128..500 {
        lsm.insert(format!("row:{:04}", i).as_bytes(), i)?;
    }
    lsm.flush_memtable()?;
    lsm.compact_all()?;
    // Delete every 7th key (scattered across buckets); tombstones live in the memtable.
    for i in (0u128..500).filter(|i| i % 7 == 0) {
        lsm.delete(format!("row:{:04}", i).as_bytes())?;
    }
    // Reference: one unbounded scan (uses the parallel read-everything path).
    let reference: Vec<Vec<u8>> = lsm
        .range_pointers_bounded(b"row:", None, usize::MAX)?
        .into_iter()
        .map(|(k, _, _)| k)
        .collect();
    assert!(reference.len() >= 400, "sanity: most keys still live");
    assert!(!reference.iter().any(|k| k == b"row:0000"), "row:0000 was deleted");

    // Paginated walk with a small page (limit 20; +1 sentinel for has_more).
    let mut walked: Vec<Vec<u8>> = Vec::new();
    let mut cursor: Vec<u8> = b"row:".to_vec();
    loop {
        let page = lsm.range_pointers_bounded(&cursor, None, 21)?;
        let has_more = page.len() > 20;
        for (k, _, _) in page.iter().take(20) {
            walked.push(k.clone());
        }
        if !has_more {
            break;
        }
        cursor = page[20].0.clone();
    }
    assert_eq!(walked, reference, "bounded paginated walk must equal the unbounded reference");
    Ok(())
}
