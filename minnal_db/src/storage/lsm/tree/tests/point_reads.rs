//! Point reads: sparse-index hints, seq resolution across layers, and the read shortcuts.

use super::*;

#[test]
fn test_sparse_index_lookup_after_compaction() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    // Enough keys that some bucket's L1 file spans many SAMPLE_INTERVAL
    // blocks, so `block_start` returns non-zero offsets and the validated
    // scan-from-hint path is exercised (not just scan-from-0).
    let n: u128 = 3000;
    for i in 0..n {
        lsm.insert(format!("k:{i:08}").as_bytes(), i)?;
    }
    lsm.flush_and_compact_all()?; // push everything to L1; memtable now empty

    // Every present key must still be found via the sparse-index hint.
    for i in 0..n {
        assert_eq!(lsm.get(format!("k:{i:08}").as_bytes())?, Some(i), "missing key {i}");
    }

    // Misses: an in-range gap and keys below min / above max.
    assert_eq!(lsm.get(b"k:00000000x")?, None); // between existing keys
    assert_eq!(lsm.get(b"j:00000000")?, None); // below min ("j" < "k")
    assert_eq!(lsm.get(b"zzzzzzzz")?, None); // above max
    Ok(())
}

// The sparse index hint must actually be USED (non-zero, valid) — a buggy
// index that always produced unusable offsets would still pass the lookup
// tests (they'd silently fall back to a full scan), so assert the hints are
// real frame boundaries with key <= target.
#[test]
fn test_sparse_index_hint_is_valid_and_engaged() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
    let n: u128 = 3000;
    for i in 0..n {
        lsm.insert(format!("k:{i:08}").as_bytes(), i)?;
    }
    lsm.flush_and_compact_all()?;

    let mut saw_nonzero_hint = false;
    for i in 0..n {
        let key = format!("k:{i:08}").into_bytes();
        let bucket = get_bucket_for_key(&key, lsm.config.num_buckets) as usize;
        let file = lsm.sstable_files[bucket].read().clone();
        let file_len = file.metadata()?.len();
        let hint = lsm.sstable_indexes[bucket].read().as_ref().map(|x| x.block_start(&key)).unwrap_or(0);
        if hint != 0 {
            saw_nonzero_hint = true;
            assert!(LSMTree::valid_scan_start(&file, hint, file_len, &key), "hint {hint} invalid for key {i}");
        }
    }
    assert!(saw_nonzero_hint, "expected some non-zero index hints across 3000 keys");
    Ok(())
}

// THE correctness guarantee: a bogus/stale start offset (as if a concurrent
// compaction swapped the L1 file under the in-memory index) must never change
// the result. The validated scan falls back to a full scan from 0.
#[test]
fn test_sparse_index_stale_hint_falls_back_correctly() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
    let metrics = Arc::new(Metrics::default());
    lsm.set_metrics(Arc::clone(&metrics));
    let n: u128 = 2000;
    for i in 0..n {
        lsm.insert(format!("k:{i:08}").as_bytes(), i + 1)?; // values are 1-based (non-zero)
    }
    lsm.flush_and_compact_all()?;

    let present = format!("k:{:08}", 1234).into_bytes();
    let bucket = get_bucket_for_key(&present, lsm.config.num_buckets) as usize;
    let file = lsm.sstable_files[bucket].read().clone();
    let file_len = file.metadata()?.len();

    // Ground truth from a full scan.
    let truth = lsm.lookup_in_sstable_file_from(&file, &present, 0)?;
    let truth_ptr = match truth {
        SsLookup::Found(p, _) => p,
        other => panic!("expected present key to be Found, got {other:?}"),
    };
    assert_eq!(metrics.snapshot().sparse_hint_rejects, 0, "a zero hint is no hint, not a rejection");

    // Every bogus offset must yield the same Found pointer: past EOF (overflow
    // guard), mid-frame, near/at EOF, and a hint whose frame key > target.
    let big_key_hint = lsm.sstable_indexes[bucket]
        .read()
        .as_ref()
        .map(|x| x.block_start(b"k:99999999"))
        .unwrap_or(0);
    for bad in [u64::MAX, 1, 3, 7, file_len.saturating_sub(1), file_len, file_len / 2 + 1, big_key_hint] {
        match lsm.lookup_in_sstable_file_from(&file, &present, bad)? {
            SsLookup::Found(p, _) => assert_eq!(p, truth_ptr, "bogus hint {bad} changed the value"),
            other => panic!("bogus hint {bad} lost a present key: {other:?}"),
        }
    }
    // A fallback is correct but slow, so it must be counted: that counter is
    // the only production signal that hints are going wrong. (At least 7 of 8:
    // the mid-file offset could in principle land on a real frame boundary.)
    assert!(
        metrics.snapshot().sparse_hint_rejects >= 7,
        "rejected hints were not counted: {}",
        metrics.snapshot().sparse_hint_rejects
    );

    // An in-range absent key must stay Missing regardless of the hint.
    let absent = format!("k:{:08}x", 1234).into_bytes();
    let abucket = get_bucket_for_key(&absent, lsm.config.num_buckets) as usize;
    let afile = lsm.sstable_files[abucket].read().clone();
    for bad in [0u64, u64::MAX, 5, afile.metadata()?.len()] {
        assert!(
            matches!(lsm.lookup_in_sstable_file_from(&afile, &absent, bad)?, SsLookup::Missing),
            "bogus hint {bad} fabricated a key"
        );
    }
    Ok(())
}

// The index built during the open-time scan (`load_metadata_from_file`) is a
// distinct code path from the one built during compaction; exercise it.
#[test]
fn test_sparse_index_rebuilt_on_reopen() -> Result<()> {
    let temp_dir = TempDir::new()?;
    {
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
        for i in 0..2000u128 {
            lsm.insert(format!("k:{i:08}").as_bytes(), i + 1)?;
        }
        lsm.flush_and_compact_all()?;
    }
    // Reopen: indexes are rebuilt from the L1 files during the open scan.
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
    for i in 0..2000u128 {
        assert_eq!(lsm.get(format!("k:{i:08}").as_bytes())?, Some(i + 1), "key {i} after reopen");
    }
    assert_eq!(lsm.get(b"k:00000005x")?, None);
    Ok(())
}

// Deletes must resolve correctly through the sparse-index hint.
#[test]
fn test_sparse_index_lookup_with_deletes() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
    let n: u128 = 1000;
    for i in 0..n {
        lsm.insert(format!("k:{i:08}").as_bytes(), i + 1)?;
    }
    for i in (0..n).step_by(2) {
        lsm.delete(format!("k:{i:08}").as_bytes())?; // delete even keys
    }
    lsm.flush_and_compact_all()?;

    for i in 0..n {
        let got = lsm.get(format!("k:{i:08}").as_bytes())?;
        if i % 2 == 0 {
            assert_eq!(got, None, "deleted key {i} should be absent");
        } else {
            assert_eq!(got, Some(i + 1), "live key {i} has wrong value");
        }
    }
    Ok(())
}

#[test]
fn test_lsm_basic_operations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    // Insert a value
    lsm.insert(b"key1", 12345)?;

    // Retrieve the value
    assert_eq!(lsm.get(b"key1")?, Some(12345));

    // Insert another value
    lsm.insert(b"key2", 67890)?;
    assert_eq!(lsm.get(b"key2")?, Some(67890));

    // Update existing value
    lsm.insert(b"key1", 54321)?;
    assert_eq!(lsm.get(b"key1")?, Some(54321));

    Ok(())
}

// A relocated value re-inserted at the key's OLD sequence into the newest
// layer must NOT shadow a higher-sequence tombstone that has flushed to an
// older layer. A `get` that returned at the first layer holding the key would
// resolve this to the value and resurrect a deleted key. The test builds the
// inverted state directly; production also prevents it at the GC re-point
// (seq-aware `get_with_seq` under the bucket lock), so this guards the read
// path as a second line of defence.
#[test]
fn test_repoint_at_old_seq_does_not_resurrect_across_layers() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert_with_seq(b"k", 0xAAAA, 1)?; // value @ seq 1
    lsm.delete_with_seq(b"k", 5)?; // tombstone @ seq 5 (newer)
    assert_eq!(lsm.get(b"k")?, None, "deleted before flush");

    lsm.flush_memtable_to_level0()?; // tombstone @ seq 5 → L0
    assert_eq!(lsm.get(b"k")?, None, "deleted after flush");

    // GC re-point: re-insert the relocated value at the OLD seq into the
    // now-empty active memtable. The key is still logically deleted.
    lsm.insert_with_seq(b"k", 0xBBBB, 1)?;
    assert_eq!(lsm.get(b"k")?, None, "GC re-point at old seq resurrected a deleted key");
    Ok(())
}

// A cold key's L1 entry keeps its original seq forever, so the gap between it
// and a much-later write to the same key is unbounded. The sequence width must
// therefore cover the database's whole lifetime: a 32-bit seq with wrapping
// comparison inverts once the gap exceeds 2^31 (~2.1B writes, ~2.5 days at
// 10k writes/s), making the *newer* write compare as older. `get` would then
// return the stale L1 value, and `merge_level0_to_level1` would keep it in L1
// for good.
#[test]
fn test_write_after_huge_seq_gap_beats_cold_l1_entry() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    // A cold key written early in the database's life, compacted down to L1
    // where it keeps seq 1000 indefinitely.
    const OLD_SEQ: u64 = 1_000;
    lsm.insert_with_seq(b"cold", 0xAAAA, OLD_SEQ)?;
    // `compact_all` only drains read-only memtables, so seal the active one
    // first — otherwise the entry never leaves the memtable and this exercises
    // the skip list instead of the L1 path it is meant to cover.
    lsm.flush_memtable_to_level0()?;
    lsm.compact_all()?;
    let bucket = get_bucket_for_key(b"cold", crate::support::TEST_NUM_BUCKETS);
    let l1_len = std::fs::metadata(LSMTree::level1_path_from(temp_dir.path(), bucket))?.len();
    assert!(
        l1_len > SSTABLE_DATA_START,
        "setup: the cold entry must actually be in L1, not still in memory"
    );
    assert_eq!(lsm.get(b"cold")?, Some(0xAAAA), "setup: cold value should read back from L1");

    // ~3B writes later the same key is updated. The gap exceeds 2^31, which
    // is exactly what a 32-bit serial comparison cannot represent.
    const NEW_SEQ: u64 = OLD_SEQ + 3_000_000_000;
    assert!(NEW_SEQ - OLD_SEQ > u64::from(u32::MAX) / 2, "test premise: gap must exceed 2^31");
    lsm.insert_with_seq(b"cold", 0xBBBB, NEW_SEQ)?;

    assert_eq!(
        lsm.get(b"cold")?,
        Some(0xBBBB),
        "a newer write lost to a cold L1 entry — sequence comparison inverted across the gap"
    );

    // The inversion is permanent once the merge resolves the conflict, so
    // check the post-compaction state too.
    lsm.flush_memtable_to_level0()?;
    lsm.compact_all()?;
    assert_eq!(
        lsm.get(b"cold")?,
        Some(0xBBBB),
        "the stale value was baked into L1 — the merge resolved the conflict backwards"
    );
    Ok(())
}

// The read fast-path bound `max_lower_seq` must be initialized
// from L0 files (not just L1) at open. L0 is newer than L1, so an L0-only
// tombstone's seq would otherwise be missing from the bound after a restart,
// letting a low-seq active entry wrongly short-circuit above it. This is the
// restart variant of test_repoint_at_old_seq_does_not_resurrect_across_layers.
#[test]
fn test_max_lower_seq_folds_l0_at_open_so_fast_path_is_safe() -> Result<()> {
    let temp_dir = TempDir::new()?;
    {
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
        lsm.insert_with_seq(b"k", 0xAAAA, 1)?; // value @ seq 1
        lsm.delete_with_seq(b"k", 5)?; // tombstone @ seq 5 (newer)
        lsm.flush_memtable_to_level0()?; // tombstone @ seq 5 → an L0 file (no L1)
        assert_eq!(lsm.get(b"k")?, None, "deleted before restart");
    }

    // Reopen: max_lower_seq must be folded from the L0 file (seq 5), not left
    // at 0 (which only L1 would have contributed).
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
    assert_eq!(lsm.get(b"k")?, None, "still deleted after restart");

    // Re-point at the old seq into the fresh (empty) active memtable. With the
    // bound correctly at 5, the fast path must NOT short-circuit on seq 1 — it
    // falls through to the full scan that sees the higher-seq L0 tombstone.
    lsm.insert_with_seq(b"k", 0xBBBB, 1)?;
    assert_eq!(
        lsm.get(b"k")?,
        None,
        "stale max_lower_seq let the fast path resurrect a deleted key after restart"
    );
    Ok(())
}

#[test]
fn test_lsm_delete() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"key1", 12345)?;
    assert_eq!(lsm.get(b"key1")?, Some(12345));

    // Delete the key
    lsm.delete(b"key1")?;

    // Key should not be found
    assert_eq!(lsm.get(b"key1")?, None);

    Ok(())
}

// A delete tombstone in the active memtable must hide the key's entry in an
// SSTable. If `keys()` missed it, deleted keys would reappear after a flush:
// after a restart, WAL recovery flushes the embed queue's entries to an
// SSTable, and the queue worker would keep finding entries already cancelled.
#[test]
fn test_keys_excludes_sstable_entry_tombstoned_in_memtable() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"keep", 1)?;
    lsm.insert(b"delete_me", 2)?;

    // Flush to SSTable — simulates WAL recovery flushing the queue entries.
    lsm.flush_memtable()?;
    lsm.compact_all()?;

    // Delete one key — tombstone lands in the fresh active memtable.
    lsm.delete(b"delete_me")?;

    // keys() must not return the deleted key even though it is still in the SSTable.
    let keys = lsm.keys()?;
    assert!(!keys.contains(&b"delete_me".to_vec()), "deleted SSTable key must not appear in keys()");
    assert!(keys.contains(&b"keep".to_vec()));
    assert_eq!(keys.len(), 1);

    Ok(())
}

// Differential guard for the read path's shortcuts — the active-memtable fast
// path, RO binary search, L0 summaries (range/bloom/max-seq) and L1 max-seq
// pruning — in both `get_with_seq` and `get_multiple`. Random writes carry
// OUT-OF-ORDER sequences (so lower-seq copies land in newer layers, the case
// every shortcut must respect) plus GC-style re-points at an existing seq, with
// flushes, compactions and reopens mixed in. After every step both lookup paths
// must agree with `reference_get` for every key, including never-written ones.
#[test]
fn test_read_shortcuts_agree_with_brute_force_resolution() -> Result<()> {
    // splitmix64: tiny, seedable, reproducible across runs.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    // Keys share 8-byte prefixes in groups so several land in each bucket and
    // L0 files hold overlapping key ranges; "absent_*" keys are never written.
    let keys: Vec<Vec<u8>> = (0..48).map(|i| format!("key{:02}_{}", i % 12, i).into_bytes()).collect();
    let probes: Vec<Vec<u8>> = keys
        .iter()
        .cloned()
        .chain((0..6).map(|i| format!("absent_{i}").into_bytes()))
        .chain(std::iter::once(keys[0].clone())) // a duplicated input key
        .collect();

    // Shared across every tree so the end-of-test check can prove each
    // shortcut actually fired — a guard that never exercises them proves nothing.
    let metrics = Arc::new(Metrics::default());
    for seed in 0..16u64 {
        let temp_dir = TempDir::new()?;
        let mut lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
        lsm.set_metrics(Arc::clone(&metrics));
        let mut rng = Rng(seed);
        let mut clock: u64 = 1_000;

        for step in 0..300 {
            clock += 8;
            let key = &keys[rng.below(keys.len() as u64) as usize];
            // Jitter the sequence backwards by up to 5 writes' worth, so later
            // writes often carry lower seqs than earlier ones.
            let seq = clock - rng.below(40);
            match rng.below(100) {
                0..=54 => {
                    lsm.insert_with_seq(key, u128::from(rng.next()), seq)?;
                }
                55..=79 => {
                    lsm.delete_with_seq(key, seq)?;
                }
                80..=86 => {
                    // GC re-point: same seq as the current winner, new pointer.
                    if let Some((_, cur)) = lsm.get_with_seq(key)? {
                        lsm.insert_with_seq(key, u128::from(rng.next()), cur)?;
                    }
                }
                87..=91 => lsm.flush_memtable()?,
                92..=95 => lsm.flush_memtable_to_level0()?,
                96..=98 => lsm.compact_all()?,
                _ => {
                    // Reopen. A bare LSM has no WAL, so make everything durable first.
                    lsm.flush_memtable_to_level0()?;
                    drop(lsm);
                    lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
                    lsm.set_metrics(Arc::clone(&metrics));
                }
            }

            let batch = lsm.get_multiple(&probes)?;
            for (probe, batched) in probes.iter().zip(batch) {
                let expected = reference_get(&lsm, probe)?;
                let label = format!("seed {seed} step {step} key {:?}", String::from_utf8_lossy(probe));
                assert_eq!(lsm.get_with_seq(probe)?, expected, "{label}: get_with_seq");
                assert_eq!(batched, expected, "{label}: get_multiple");
            }
        }
    }
    let m = metrics.snapshot();
    assert!(m.fast_path_hits > 0, "the active-memtable fast path never fired");
    assert!(m.l0_bloom_rejects > 0, "no L0 file was ever skipped by range/bloom");
    assert!(m.seq_prunes > 0, "no SSTable was ever skipped by max-seq");
    assert!(m.l0_probes > 0 && m.l1_probes > 0, "the slow path never read L0 and L1");
    Ok(())
}

/// A miss on a bucket whose L1 file is empty is settled by its (empty) bloom
/// filter, without touching the file: at open, after a compaction that
/// empties the bucket, and after a reopen. A compaction that fills the
/// bucket again replaces the filter, so its keys are found.
#[test]
fn a_miss_on_an_empty_l1_never_reads_the_file() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
    let metrics = Arc::new(Metrics::default());
    lsm.set_metrics(Arc::clone(&metrics));
    let probes = |lsm: &LSMTree| -> Result<u64> {
        for i in 0..20 {
            assert_eq!(lsm.get(format!("absent_{i}").as_bytes())?, None);
        }
        Ok(metrics.snapshot().l1_probes)
    };

    lsm.insert_with_seq(b"key_a", 7, 10)?;
    assert_eq!(probes(&lsm)?, 0, "a fresh store read its empty L1 files");

    lsm.flush_memtable_to_level0()?;
    lsm.compact_all()?;
    assert_eq!(lsm.get(b"key_a")?, Some(7), "the compacted key is found through the new filter");
    lsm.delete_with_seq(b"key_a", 11)?;
    lsm.flush_memtable_to_level0()?;
    lsm.compact_all()?;
    assert_eq!(lsm.get(b"key_a")?, None);
    assert!(
        lsm.sstable_metadata.iter().all(|m| m.read().is_empty()),
        "the delete did not empty L1, so this does not test an emptied bucket"
    );
    let before = metrics.snapshot().l1_probes;
    assert_eq!(probes(&lsm)?, before, "an L1 emptied by compaction was read");

    drop(lsm);
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
    lsm.set_metrics(Arc::clone(&metrics));
    let before = metrics.snapshot().l1_probes;
    assert_eq!(probes(&lsm)?, before, "a reopened empty L1 was read");
    Ok(())
}
