//! Prefix scans across every layer, against references.

use super::*;

// `merge_scan_layers` avoids a map insert per L1 entry, but must give exactly
// what folding L1 and then every newer layer into one map by `fold_newest`
// gives: the later (newer) copy of a key wins. Checked over many generated
// layer sets with a tiny key space, so duplicate keys within L1 and tombstones
// on both sides are common.
#[test]
fn test_merge_scan_layers_matches_seq_fold() {
    let mut state = 0x9E37_79B9_7F4A_7C15u64; // deterministic LCG, no RNG dependency
    let mut next = |bound: u64| {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % bound
    };
    for case in 0..2_000 {
        // L1 as the scan reads it: several buckets, each run key-sorted.
        let mut l1: Vec<ScanEntry> = Vec::new();
        for _bucket in 0..next(4) {
            let mut run: Vec<ScanEntry> = (0..next(12))
                .map(|_| {
                    (
                        vec![next(3) as u8, next(6) as u8],
                        (next(4) != 0).then(|| u128::from(next(1000))),
                        Version::user(next(8)),
                    )
                })
                .collect();
            run.sort_by(|a, b| a.0.cmp(&b.0));
            l1.extend(run);
        }
        let mut newer = ScanMap::new();
        for _ in 0..next(6) {
            fold_newest(
                &mut newer,
                vec![next(3) as u8, next(6) as u8],
                (next(3) != 0).then(|| u128::from(next(1000))),
                Version::user(next(8)),
            );
        }

        let mut reference = ScanMap::new();
        for (k, v, seq) in l1.iter().cloned() {
            fold_newest(&mut reference, k, v, seq);
        }
        for (k, (v, seq)) in newer.clone() {
            fold_newest(&mut reference, k, v, seq);
        }
        let expected: Vec<(Vec<u8>, u128)> = reference.into_iter().filter_map(|(k, (v, _))| v.map(|v| (k, v))).collect();

        let l1_keys = vec![l1.iter().flat_map(|(k, _, _)| k.iter().copied()).collect::<Vec<u8>>()];
        let mut at = 0;
        let packed: Vec<PackedScanEntry> = l1
            .iter()
            .map(|(k, v, seq)| {
                at += k.len();
                (0, at - k.len(), at, *v, *seq)
            })
            .collect();
        assert_eq!(
            merge_scan_layers(&l1_keys, packed, newer.clone()).to_pairs(),
            expected,
            "case {case}: l1={l1:?} newer={newer:?}"
        );
    }
}

#[test]
fn test_lsm_prefix_scan() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    // Insert keys with various prefixes
    lsm.insert(b"user:1:name", 100)?;
    lsm.insert(b"user:1:email", 101)?;
    lsm.insert(b"user:2:name", 200)?;
    lsm.insert(b"user:2:email", 201)?;
    lsm.insert(b"product:1:name", 300)?;
    lsm.insert(b"product:1:price", 301)?;

    // Test prefix scan for "user:1"
    let results = lsm.scan_prefix(b"user:1")?;
    assert_eq!(results.len(), 2);
    assert!(results.iter().any(|(k, v, _)| k == b"user:1:name" && *v == 100));
    assert!(results.iter().any(|(k, v, _)| k == b"user:1:email" && *v == 101));

    // Test prefix scan for "user:"
    let results = lsm.scan_prefix(b"user:")?;
    assert_eq!(results.len(), 4);

    // Test prefix scan for "product:"
    let results = lsm.scan_prefix(b"product:")?;
    assert_eq!(results.len(), 2);
    assert!(results.iter().any(|(k, v, _)| k == b"product:1:name" && *v == 300));
    assert!(results.iter().any(|(k, v, _)| k == b"product:1:price" && *v == 301));

    // Test prefix scan with no matches
    let results = lsm.scan_prefix(b"order:")?;
    assert_eq!(results.len(), 0);

    Ok(())
}

#[test]
fn test_lsm_prefix_scan_with_flush() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    // Insert and flush some data
    lsm.insert(b"app:config:version", 100)?;
    lsm.insert(b"app:config:debug", 101)?;
    lsm.flush_memtable()?;

    // Insert more data in new memtable
    lsm.insert(b"app:config:logging", 102)?;
    lsm.insert(b"app:state:active", 200)?;

    // Test prefix scan should find data from both memtable and read-only layer
    let results = lsm.scan_prefix(b"app:config:")?;
    assert_eq!(results.len(), 3);
    assert!(results.iter().any(|(k, v, _)| k == b"app:config:version" && *v == 100));
    assert!(results.iter().any(|(k, v, _)| k == b"app:config:debug" && *v == 101));
    assert!(results.iter().any(|(k, v, _)| k == b"app:config:logging" && *v == 102));

    Ok(())
}

#[test]
fn test_lsm_prefix_scan_with_compaction() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    // Insert data and compact to SSTable
    lsm.insert(b"session:123:user", 1000)?;
    lsm.insert(b"session:123:timestamp", 1001)?;
    lsm.insert(b"session:456:user", 2000)?;
    lsm.flush_memtable()?;
    lsm.compact_all()?;

    // Insert more data in new memtable
    lsm.insert(b"session:123:active", 1002)?;

    // Test prefix scan should find data from both SSTable and memtable
    let results = lsm.scan_prefix(b"session:123:")?;
    assert_eq!(results.len(), 3);
    assert!(results.iter().any(|(k, v, _)| k == b"session:123:user" && *v == 1000));
    assert!(results.iter().any(|(k, v, _)| k == b"session:123:timestamp" && *v == 1001));
    assert!(results.iter().any(|(k, v, _)| k == b"session:123:active" && *v == 1002));

    Ok(())
}

// The L0 sparse index is built twice: from offsets tracked while the flush
// writes the file, and by rescanning the file at open. A wrong offset is
// invisible to correctness tests — `valid_scan_start` rejects the hint and the
// lookup silently falls back to a full scan — so pin it directly: both builds
// must agree, and every hint must be a valid place to start scanning.
#[test]
fn test_l0_sparse_index_hints_are_valid_and_survive_reopen() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config = LSMConfig {
        num_buckets: 1,
        ..LSMConfig::default()
    };
    let keys: Vec<Vec<u8>> = (0..500u32).map(|i| format!("key{:05}", i * 2).into_bytes()).collect();
    // Probe present keys and the gaps between them (odd numbers are absent).
    let probes: Vec<Vec<u8>> = (0..1000u32).map(|i| format!("key{i:05}").into_bytes()).collect();

    let hints_at_flush: Vec<u64> = {
        let lsm = LSMTree::open(temp_dir.path(), config.clone())?;
        for (i, key) in keys.iter().enumerate() {
            // Values of varying width so frames differ in length.
            lsm.insert_with_seq(key, u128::from(i as u32) << (i % 96), i as u64 + 1)?;
        }
        lsm.flush_memtable_to_level0()?;
        let entries = lsm.level0_entries_snapshot(0);
        assert_eq!(entries.len(), 1, "setup: one L0 file");
        let entry = &entries[0];
        assert!(entry.summary.as_ref().is_some_and(|s| s.index.is_some()), "the flush must build an index");
        probes.iter().map(|k| entry.scan_start(k)).collect()
    };

    let lsm = LSMTree::open(temp_dir.path(), config)?;
    let entry = lsm.level0_entries_snapshot(0).remove(0);
    let hints_at_open: Vec<u64> = probes.iter().map(|k| entry.scan_start(k)).collect();
    if let Some(i) = (0..probes.len()).find(|&i| hints_at_flush[i] != hints_at_open[i]) {
        panic!(
            "flush-time and open-time indexes disagree for {:?}: {} at flush, {} at open",
            String::from_utf8_lossy(&probes[i]),
            hints_at_flush[i],
            hints_at_open[i]
        );
    }

    let file = File::open(&entry.path)?;
    let file_len = file.metadata()?.len();
    let mut hinted = 0;
    for (key, &hint) in probes.iter().zip(&hints_at_open) {
        if hint != 0 {
            hinted += 1;
            assert!(
                LSMTree::valid_scan_start(&file, hint, file_len, key),
                "hint {hint} for {:?} is not a valid scan start",
                String::from_utf8_lossy(key)
            );
        }
    }
    assert!(hinted > 900, "almost every probe should get a hint, got {hinted}");
    for (i, key) in keys.iter().enumerate() {
        assert_eq!(lsm.get(key)?, Some(u128::from(i as u32) << (i % 96)), "key {i} after reopen");
    }
    Ok(())
}

// Every multi-key scan must resolve each key exactly as a point read does: the
// copy in the newest layer that holds it. Random user writes (in seq order, as
// the key stripe applies them) and GC-style system copies (the current copy's
// next system version) are mixed with flushes, compactions and reopens. After
// every step the tree must keep newer versions in newer layers, and each scan
// must equal the brute-force view built from `reference_get` (highest version
// across every layer), including bounded pages small enough that their window
// cuts layers short.
#[test]
fn test_scans_agree_with_brute_force_resolution() -> Result<()> {
    let mut state = 0xD1B5_4A32_D192_ED03u64;
    let mut next = |bound: u64| {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % bound
    };
    // Two 4-byte prefixes ("key0", "key1") so `scan_prefixes` has ids to probe.
    let keys: Vec<Vec<u8>> = (0..40).map(|i| format!("key{}_{:02}", i % 2, i).into_bytes()).collect();

    for seed in 0..8u64 {
        let temp_dir = TempDir::new()?;
        let mut lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
        let mut clock: u64 = 1_000 + seed;

        for step in 0..250 {
            clock += 8;
            let key = &keys[next(keys.len() as u64) as usize];
            let seq = clock;
            match next(100) {
                0..=44 => {
                    lsm.insert_with_seq(key, u128::from(next(1 << 30)) + 1, seq)?;
                }
                45..=71 => {
                    lsm.delete_with_seq(key, seq)?;
                }
                72..=81 => {
                    // A GC relocation: same user seq, next system seq, new pointer.
                    if let Some((_, version)) = lsm.get_with_seq(key)? {
                        lsm.insert_with_seq(key, u128::from(next(1 << 30)) + 1, version.next_system().unwrap())?;
                    }
                }
                82..=88 => lsm.flush_memtable()?,
                89..=94 => lsm.flush_memtable_to_level0()?,
                95..=98 => lsm.compact_all()?,
                _ => {
                    lsm.flush_memtable_to_level0()?;
                    drop(lsm);
                    lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
                }
            }

            let label = format!("seed {seed} step {step}");
            let violations = lsm.layer_order_violations()?;
            assert!(violations.is_empty(), "{label}: {violations:?}");
            // Truth: every key's point-read answer, in key order.
            let mut truth: Vec<(Vec<u8>, u128, Version)> = Vec::new();
            for k in &keys {
                if let Some((ptr, s)) = reference_get(&lsm, k)? {
                    truth.push((k.clone(), ptr, s));
                }
            }
            truth.sort();
            truth.dedup();
            let with_prefix = |p: &[u8]| truth.iter().filter(|(k, _, _)| k.starts_with(p)).cloned().collect::<Vec<_>>();

            assert_eq!(lsm.scan_prefix(b"key1")?, with_prefix(b"key1"), "{label}: scan_prefix");
            let ids: std::collections::HashSet<u32> = [u32::from_be_bytes(*b"key0"), u32::from_be_bytes(*b"key1")].into_iter().collect();
            let pairs = |v: Vec<(Vec<u8>, u128, Version)>| v.into_iter().map(|(k, p, _)| (k, p)).collect::<Vec<_>>();
            assert_eq!(lsm.scan_prefixes(&ids)?.to_pairs(), pairs(truth.clone()), "{label}: scan_prefixes");
            assert_eq!(lsm.key_pointer_pairs(None)?, pairs(truth.clone()), "{label}: key_pointer_pairs");
            assert_eq!(
                lsm.range_pointers_bounded(b"", None, usize::MAX)?,
                truth,
                "{label}: range_pointers_bounded, unbounded"
            );
            // A small page from the middle: the window cuts every layer short.
            let start = b"key0_10".as_slice();
            let page: Vec<_> = truth.iter().filter(|(k, _, _)| k.as_slice() >= start).take(3).cloned().collect();
            assert_eq!(
                lsm.range_pointers_bounded(start, None, 3)?,
                page,
                "{label}: range_pointers_bounded, limit 3"
            );
            let page_keys: Vec<Vec<u8>> = page.iter().map(|(k, _, _)| k.clone()).collect();
            assert_eq!(range_keys(&lsm, start, 3)?, page_keys, "{label}: range_keys_bounded, limit 3");
            // Every page size, from several starts, with and without an end,
            // and a walk through the whole range a page at a time.
            for limit in [1usize, 2, 7, 50] {
                for start in [b"".as_slice(), b"key0_10", b"key1_"] {
                    for end in [None, Some(b"key1_25".as_slice())] {
                        let want: Vec<_> = truth
                            .iter()
                            .filter(|(k, _, _)| k.as_slice() >= start && end.is_none_or(|e| k.as_slice() < e))
                            .take(limit)
                            .cloned()
                            .collect();
                        assert_eq!(
                            lsm.range_pointers_bounded(start, end, limit)?,
                            want,
                            "{label}: limit {limit} from {start:?} to {end:?}"
                        );
                    }
                }
                let mut cursor = Vec::new();
                let mut walked = Vec::new();
                while let Some(last) = lsm
                    .range_pointers_bounded(&cursor, None, limit)?
                    .into_iter()
                    .inspect(|e| walked.push(e.clone()))
                    .last()
                {
                    cursor = last.0;
                    cursor.push(0);
                }
                assert_eq!(walked, truth, "{label}: walk in pages of {limit}");
            }
        }
    }
    Ok(())
}

// The L0 prefix filter compares a key's zero-padded 8-byte `key_prefix`, so a
// key shorter than a prefix that ends in 0x00 bytes would match it unless the
// filter checks the length too; the memtable and L1 paths compare real key
// bytes. All three layers must agree that "ab" does not start with "ab\0".
#[test]
fn test_scan_prefix_rejects_a_key_shorter_than_the_prefix_in_every_layer() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
    lsm.insert_with_seq(b"ab", 1, 1)?;
    lsm.insert_with_seq(b"ab\0x", 2, 2)?; // a real match, to show the filter still admits one
    let expected = vec![(b"ab\0x".to_vec(), 2u128, Version::user(2))];

    assert_eq!(lsm.scan_prefix(b"ab\0")?, expected, "memtable");
    lsm.flush_memtable_to_level0()?;
    assert_eq!(lsm.scan_prefix(b"ab\0")?, expected, "L0");
    lsm.compact_all()?;
    assert_eq!(lsm.scan_prefix(b"ab\0")?, expected, "L1");
    Ok(())
}

// ── scan_prefix and tombstones ─────────────────────────────────────────────

#[test]
fn test_scan_prefix_deleted_after_flush_returns_none() -> Result<()> {
    // A key deleted after a flush must not appear: the active memtable's
    // tombstone must hide its SSTable entry, which a memtable iterator that
    // skips tombstones would miss.
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"doc:1", 10)?;
    lsm.insert(b"doc:2", 20)?;
    lsm.insert(b"other:1", 99)?;
    lsm.flush_memtable()?; // both doc: keys now in SSTable

    lsm.delete(b"doc:2")?; // tombstone in active memtable

    let results = lsm.scan_prefix(b"doc:")?;

    assert_eq!(results.len(), 1, "deleted key must not appear");
    assert_eq!(results[0].0, b"doc:1");
    assert_eq!(results[0].1, 10);
    Ok(())
}

#[test]
fn test_scan_prefix_deleted_key_then_reinserted() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"k:1", 1)?;
    lsm.flush_memtable()?;
    lsm.delete(b"k:1")?;
    lsm.flush_memtable()?; // tombstone now in read-only memtable
    lsm.insert(b"k:1", 2)?; // re-insert in fresh active memtable

    let results = lsm.scan_prefix(b"k:")?;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].1, 2, "re-inserted value must win");
    Ok(())
}

// ── scan_prefix across L0 and L1 ──────────────────────────────────────────
// A newer L0 entry or tombstone must win over the older L1 copy of its key,
// which needs both layers' copies, tombstones included, resolved by seq. These
// cover the L0/L1 cases the memtable tests above do not reach.

#[test]
fn test_scan_prefix_sees_l0_only_data() -> Result<()> {
    // Data lives only in L0 (memtable + read-only memtables empty).
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"p:1", 100)?;
    lsm.insert(b"p:2", 200)?;
    lsm.flush_memtable_to_level0()?; // data only in L0
    lsm.purge_ro_memtables_for_test();

    let map: std::collections::HashMap<Vec<u8>, u128> = lsm.scan_prefix(b"p:")?.into_iter().map(|(k, v, _)| (k, v)).collect();
    assert_eq!(map.get(b"p:1".as_slice()), Some(&100), "L0-only key p:1 must appear");
    assert_eq!(map.get(b"p:2".as_slice()), Some(&200), "L0-only key p:2 must appear");
    Ok(())
}

#[test]
fn test_scan_prefix_l0_value_shadows_stale_l1() -> Result<()> {
    // Updated value: new in L0, old in L1 — the newer L0 value must win.
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"p:a", 111)?;
    lsm.flush_and_compact_all()?; // p:a=111 -> L1
    lsm.insert(b"p:a", 222)?;
    lsm.flush_memtable_to_level0()?; // p:a=222 -> L0
    lsm.purge_ro_memtables_for_test(); // only L0 value + stale L1 value remain

    let map: std::collections::HashMap<Vec<u8>, u128> = lsm.scan_prefix(b"p:")?.into_iter().map(|(k, v, _)| (k, v)).collect();
    assert_eq!(map.get(b"p:a".as_slice()), Some(&222), "newer L0 value must shadow stale L1 value");
    Ok(())
}

#[test]
fn test_scan_prefix_l0_tombstone_suppresses_l1() -> Result<()> {
    // Deleted key: tombstone in L0, live value in L1 — the tombstone must win.
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"p:1", 100)?;
    lsm.flush_and_compact_all()?; // p:1 -> L1
    lsm.insert(b"p:2", 200)?;
    lsm.delete(b"p:1")?; // tombstone in active memtable
    lsm.flush_memtable_to_level0()?; // tombstone + p:2 -> L0
    lsm.purge_ro_memtables_for_test(); // only L0 tombstone + L1 value remain

    let map: std::collections::HashMap<Vec<u8>, u128> = lsm.scan_prefix(b"p:")?.into_iter().map(|(k, v, _)| (k, v)).collect();
    assert_eq!(map.get(b"p:1".as_slice()), None, "L0 tombstone must suppress live L1 value");
    assert_eq!(map.get(b"p:2".as_slice()), Some(&200), "live key p:2 must appear");
    Ok(())
}

#[test]
fn test_scan_prefix_l0_reinsert_after_l1_delete() -> Result<()> {
    // Re-insert into L0 after the key was deleted (tombstone) in L1.
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"p:1", 100)?;
    lsm.flush_and_compact_all()?;
    lsm.delete(b"p:1")?;
    lsm.compact_all()?; // tombstone in L1
    lsm.insert(b"p:1", 999)?; // re-insert into memtable
    lsm.flush_memtable_to_level0()?; // new value to L0
    lsm.purge_ro_memtables_for_test();

    let map: std::collections::HashMap<Vec<u8>, u128> = lsm.scan_prefix(b"p:")?.into_iter().map(|(k, v, _)| (k, v)).collect();
    assert_eq!(map.get(b"p:1".as_slice()), Some(&999), "re-inserted L0 value must win over L1 tombstone");
    Ok(())
}

// ── min/max skip + sparse-index seek over real L1 SSTables ──────────────

#[test]
fn prefix_upper_bound_cases() {
    assert_eq!(prefix_upper_bound(b"abc"), Some(b"abd".to_vec()));
    assert_eq!(prefix_upper_bound(b"a"), Some(b"b".to_vec()));
    // A trailing 0xFF is dropped and the previous byte incremented.
    assert_eq!(prefix_upper_bound(b"ab\xff"), Some(b"ac".to_vec()));
    assert_eq!(prefix_upper_bound(b"ab\xff\xff"), Some(b"ac".to_vec()));
    // No finite upper bound.
    assert_eq!(prefix_upper_bound(b"\xff\xff"), None);
    assert_eq!(prefix_upper_bound(b""), None);
}

/// Sort scan output to `(key, pointer)` for order-insensitive comparison.
fn kv_sorted(v: Vec<(Vec<u8>, u128, Version)>) -> Vec<(Vec<u8>, u128)> {
    let mut r: Vec<(Vec<u8>, u128)> = v.into_iter().map(|(k, val, _)| (k, val)).collect();
    r.sort();
    r
}

#[test]
fn scan_over_l1_matches_reference_with_seek_and_skip() -> Result<()> {
    // 1000 keys, all compacted into L1 (well past SAMPLE_INTERVAL per bucket, so
    // the sparse index has multiple samples and block_start is exercised).
    let temp = TempDir::new()?;
    let lsm = LSMTree::open(temp.path(), test_lsm_config())?;
    let n = 1000u64;
    for seq in 0..n {
        lsm.insert(format!("key:{seq:06}").as_bytes(), seq as u128)?;
    }
    lsm.flush_and_compact_all()?;
    lsm.purge_ro_memtables_for_test(); // data lives only in L1 now

    // Full prefix → every key, values intact.
    let all = kv_sorted(lsm.scan_prefix(b"key:")?);
    assert_eq!(all.len(), n as usize);
    for (seq, (k, v)) in all.iter().enumerate() {
        assert_eq!((k.clone(), *v), (format!("key:{seq:06}").into_bytes(), seq as u128));
    }

    // Middle range → sparse-index seek near start + early stop at end.
    let mid = kv_sorted(lsm.range_pointers_bounded(b"key:000100", Some(b"key:000200"), usize::MAX)?);
    assert_eq!(mid.len(), 100);
    assert_eq!(mid.first().unwrap().1, 100);
    assert_eq!(mid.last().unwrap().1, 199);

    // Sub-prefix selecting a middle slice (key:000100..000199).
    assert_eq!(kv_sorted(lsm.scan_prefix(b"key:0001")?).len(), 100);

    // Ranges/prefixes that fall entirely outside [min,max] → min/max skip → empty.
    assert!(lsm.range_pointers_bounded(b"aaa", Some(b"aab"), usize::MAX)?.is_empty());
    assert!(lsm.range_pointers_bounded(b"zzz", None, usize::MAX)?.is_empty());
    assert!(lsm.scan_prefix(b"nomatch:")?.is_empty());

    // A bounded page still honours `limit` after the seek.
    assert_eq!(lsm.range_pointers_bounded(b"key:", Some(b"key;"), 10)?.len(), 10);
    Ok(())
}

#[test]
fn scan_over_l1_suppresses_tombstone() -> Result<()> {
    let temp = TempDir::new()?;
    let lsm = LSMTree::open(temp.path(), test_lsm_config())?;
    for seq in 0..200u64 {
        lsm.insert(format!("k:{seq:04}").as_bytes(), seq as u128)?;
    }
    lsm.flush_and_compact_all()?;
    lsm.delete(b"k:0100")?;
    lsm.flush_and_compact_all()?; // tombstone merged into L1
    lsm.purge_ro_memtables_for_test();

    let keys: Vec<Vec<u8>> = lsm
        .range_pointers_bounded(b"k:0050", Some(b"k:0150"), usize::MAX)?
        .into_iter()
        .map(|(k, _, _)| k)
        .collect();
    assert!(!keys.contains(&b"k:0100".to_vec()), "deleted key must be excluded after min/max seek");
    assert_eq!(keys.len(), 99, "50..149 minus the one deleted key");
    Ok(())
}

#[test]
fn scan_across_l0_and_l1_after_partial_flush() -> Result<()> {
    // Some keys in L1, a newer overlapping batch in L0: the scan must merge
    // both, with the L0 overwrite winning.
    let temp = TempDir::new()?;
    let lsm = LSMTree::open(temp.path(), test_lsm_config())?;
    for seq in 0..100u64 {
        lsm.insert(format!("m:{seq:04}").as_bytes(), seq as u128)?;
    }
    lsm.flush_and_compact_all()?; // → L1
    // Overwrite a slice with new values, land them in L0 only.
    for seq in 40..60u64 {
        lsm.insert(format!("m:{seq:04}").as_bytes(), 1000 + seq as u128)?;
    }
    lsm.flush_memtable_to_level0()?; // → L0
    lsm.purge_ro_memtables_for_test();

    let map: std::collections::HashMap<Vec<u8>, u128> = lsm
        .range_pointers_bounded(b"m:", Some(b"m;"), usize::MAX)?
        .into_iter()
        .map(|(k, v, _)| (k, v))
        .collect();
    assert_eq!(map.len(), 100);
    assert_eq!(map[b"m:0005".as_slice()], 5, "untouched L1 value");
    assert_eq!(map[b"m:0050".as_slice()], 1050, "L0 overwrite must win over L1");
    Ok(())
}
