//! Batch reads (`get_multiple`) against point reads.

use super::*;

#[test]
fn test_get_multiple_matches_individual_get() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    for i in 0u128..20 {
        lsm.insert(format!("k{i}").as_bytes(), i * 10)?;
    }

    let keys: Vec<Vec<u8>> = (0u128..20).map(|i| format!("k{i}").into_bytes()).collect();
    let batch: Vec<Option<u128>> = lsm.get_multiple(&keys)?.into_iter().map(|o| o.map(|(p, _)| p)).collect();

    assert_eq!(batch.len(), keys.len());
    for (i, result) in batch.iter().enumerate() {
        assert_eq!(*result, lsm.get(&keys[i])?, "mismatch at index {i}");
    }
    Ok(())
}

#[test]
fn test_get_multiple_missing_keys_return_none() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"present", 42)?;

    let keys = vec![b"present".to_vec(), b"absent".to_vec()];
    let batch: Vec<Option<u128>> = lsm.get_multiple(&keys)?.into_iter().map(|o| o.map(|(p, _)| p)).collect();

    assert_eq!(batch[0], Some(42));
    assert_eq!(batch[1], None);
    Ok(())
}

#[test]
fn test_get_multiple_after_flush_to_sstable() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    // Write to memtable, flush to SSTable, write more to a fresh memtable
    for i in 0u128..10 {
        lsm.insert(format!("flushed:{i}").as_bytes(), i)?;
    }
    lsm.flush_memtable()?;
    for i in 10u128..15 {
        lsm.insert(format!("flushed:{i}").as_bytes(), i)?;
    }

    let keys: Vec<Vec<u8>> = (0u128..15).map(|i| format!("flushed:{i}").into_bytes()).collect();
    let batch: Vec<Option<u128>> = lsm.get_multiple(&keys)?.into_iter().map(|o| o.map(|(p, _)| p)).collect();

    for (i, result) in batch.iter().enumerate() {
        assert_eq!(*result, Some(i as u128), "missing key at index {i}");
    }
    Ok(())
}

#[test]
fn test_get_multiple_after_compaction() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    for i in 0u128..20 {
        lsm.insert(format!("compact:{i}").as_bytes(), i + 100)?;
    }
    lsm.flush_memtable()?;
    lsm.compact_all()?;

    let keys: Vec<Vec<u8>> = (0u128..20).map(|i| format!("compact:{i}").into_bytes()).collect();
    let batch: Vec<Option<u128>> = lsm.get_multiple(&keys)?.into_iter().map(|o| o.map(|(p, _)| p)).collect();

    for (i, result) in batch.iter().enumerate() {
        assert_eq!(*result, Some(i as u128 + 100), "missing key at index {i}");
    }
    Ok(())
}

#[test]
fn test_get_multiple_l1_prefilter_skips_absent_keys() -> Result<()> {
    // Single bucket so every key shares one L1 file with a known [min, max]; the
    // batch mixes present keys, keys below/above the L1 range (min/max reject), and
    // an in-range key never inserted (bloom reject / L1 miss). The L1 pre-filter
    // must never drop a live key and must return None for every absent one.
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(
        temp_dir.path(),
        LSMConfig {
            num_buckets: 1,
            ..test_lsm_config()
        },
    )?;

    for i in 30u128..50 {
        lsm.insert(format!("m{i}").as_bytes(), i)?;
    }
    lsm.flush_memtable()?;
    lsm.compact_all()?; // land everything in L1 (min "m30", max "m49") with a bloom + min/max

    let keys = vec![
        b"m30".to_vec(),  // present, == min
        b"m49".to_vec(),  // present, == max
        b"m40".to_vec(),  // present, interior
        b"a00".to_vec(),  // absent, below min       -> min/max reject
        b"z99".to_vec(),  // absent, above max       -> min/max reject
        b"m300".to_vec(), // absent, inside [min,max] -> bloom reject / L1 miss
    ];
    let batch: Vec<Option<u128>> = lsm.get_multiple(&keys)?.into_iter().map(|o| o.map(|(p, _)| p)).collect();
    assert_eq!(batch[0], Some(30));
    assert_eq!(batch[1], Some(49));
    assert_eq!(batch[2], Some(40));
    assert_eq!(batch[3], None);
    assert_eq!(batch[4], None);
    assert_eq!(batch[5], None);

    // A batch of only out-of-range keys exercises the full skip (no L1 read): all None.
    let absent_only = vec![b"a00".to_vec(), b"b11".to_vec(), b"z99".to_vec()];
    let batch2: Vec<Option<u128>> = lsm.get_multiple(&absent_only)?.into_iter().map(|o| o.map(|(p, _)| p)).collect();
    assert!(batch2.iter().all(|r| r.is_none()));

    // The pre-filtered batch path must agree with the single-key point-lookup path.
    for k in &keys {
        let single = lsm.get(k)?;
        let multi = lsm.get_multiple(std::slice::from_ref(k))?[0].map(|(p, _)| p);
        assert_eq!(single, multi, "get vs get_multiple mismatch for {:?}", String::from_utf8_lossy(k));
    }
    Ok(())
}

#[test]
fn test_get_multiple_deleted_keys_return_none() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"alive", 1)?;
    lsm.insert(b"dead", 2)?;
    lsm.delete(b"dead")?;

    let keys = vec![b"alive".to_vec(), b"dead".to_vec()];
    let batch: Vec<Option<u128>> = lsm.get_multiple(&keys)?.into_iter().map(|o| o.map(|(p, _)| p)).collect();

    assert_eq!(batch[0], Some(1));
    assert_eq!(batch[1], None);
    Ok(())
}

#[test]
fn test_get_multiple_deleted_after_flush() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"kept", 10)?;
    lsm.insert(b"gone", 20)?;
    lsm.flush_memtable()?;
    lsm.delete(b"gone")?;

    let keys = vec![b"kept".to_vec(), b"gone".to_vec()];
    let batch: Vec<Option<u128>> = lsm.get_multiple(&keys)?.into_iter().map(|o| o.map(|(p, _)| p)).collect();

    assert_eq!(batch[0], Some(10));
    assert_eq!(batch[1], None);
    Ok(())
}

// `get_multiple` must find the same copy as `get_with_seq`: the one in the
// newest layer that holds the key. Value-log GC decides liveness with it, so a
// "deleted" answer for a live key would make GC drop its record and unlink the
// segment: permanent loss.
fn assert_batch_matches_point_gets(lsm: &LSMTree, keys: &[&[u8]], label: &str) -> Result<()> {
    let owned: Vec<Vec<u8>> = keys.iter().map(|k| k.to_vec()).collect();
    let batch = lsm.get_multiple(&owned)?;
    for (key, got) in keys.iter().zip(batch) {
        assert_eq!(
            got,
            lsm.get_with_seq(key)?,
            "{label}: get_multiple disagrees with get_with_seq for {:?}",
            String::from_utf8_lossy(key)
        );
    }
    Ok(())
}

// At every layer boundary the reads walk, a newer layer's copy wins: a user
// write's tombstone over a value, a user write's value over a tombstone, and a
// system action's copy (same user seq, next system seq, as GC writes) over the
// copy it replaced.
#[test]
fn test_get_multiple_takes_the_newer_layers_copy_at_every_boundary() -> Result<()> {
    #[derive(Debug, Clone, Copy)]
    enum Newer {
        ActiveMemtable,
        ReadOnlyMemtable,
        Level0,
    }
    #[derive(Debug, Clone, Copy)]
    enum Older {
        ReadOnlyMemtable,
        Level0,
        Level1,
    }
    let cases = [
        (Older::ReadOnlyMemtable, Newer::ActiveMemtable),
        (Older::Level0, Newer::ActiveMemtable),
        (Older::Level1, Newer::ActiveMemtable),
        (Older::ReadOnlyMemtable, Newer::ReadOnlyMemtable),
        (Older::Level0, Newer::ReadOnlyMemtable),
        (Older::Level1, Newer::ReadOnlyMemtable),
        (Older::Level0, Newer::Level0),
        (Older::Level1, Newer::Level0),
    ];
    for (older, newer) in cases {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        // Older layer: three live values.
        lsm.insert_with_seq(b"deleted", 0xAAAA, 10)?;
        lsm.insert_with_seq(b"rewritten", 0xBBBB, 11)?;
        lsm.insert_with_seq(b"moved", 0xCCCC, 12)?;
        match older {
            Older::ReadOnlyMemtable => lsm.flush_memtable()?,
            Older::Level0 => lsm.flush_memtable_to_level0()?,
            Older::Level1 => {
                lsm.flush_memtable_to_level0()?;
                lsm.compact_all()?;
            }
        }

        // Newer layer: a delete, an overwrite, and a GC copy of `moved`.
        lsm.delete_with_seq(b"deleted", 20)?;
        lsm.insert_with_seq(b"rewritten", 0xDDDD, 21)?;
        let moved = Version::user(12).next_system().unwrap();
        lsm.insert_with_seq(b"moved", 0xEEEE, moved)?;
        match newer {
            Newer::ActiveMemtable => {}
            Newer::ReadOnlyMemtable => lsm.flush_memtable()?,
            Newer::Level0 => lsm.flush_memtable_to_level0()?,
        }

        let label = format!("{older:?}/{newer:?}");
        assert!(lsm.layer_order_violations()?.is_empty(), "{label}: the test must build an ordered tree");
        assert_eq!(lsm.get_with_seq(b"deleted")?, None, "{label}");
        assert_eq!(lsm.get_with_seq(b"rewritten")?, Some((0xDDDD, Version::user(21))), "{label}");
        assert_eq!(lsm.get_with_seq(b"moved")?, Some((0xEEEE, moved)), "{label}");
        // A duplicated input key must resolve at every position.
        assert_batch_matches_point_gets(&lsm, &[b"deleted", b"rewritten", b"moved", b"never", b"moved"], &label)?;
    }
    Ok(())
}

/// A batch read must read near its keys, not whole SSTables: a one-key batch
/// on a large L1 file reads a few KiB, and every batch size agrees with the
/// single-key `get` across L1, several overlapping L0 files and the memtable
/// (overwrites, tombstones, absent and duplicated keys).
#[test]
fn test_get_multiple_reads_near_its_keys_and_agrees_with_get() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config = LSMConfig {
        num_buckets: 2,
        ..LSMConfig::default()
    };
    let lsm = LSMTree::open(temp_dir.path(), config)?;
    let key = |i: u64| format!("doc:{i:08}").into_bytes();
    let mut seq = 1u64;
    for i in 0..20_000u64 {
        lsm.insert_with_seq(&key(i), u128::from(i) + 1, seq)?;
        seq += 1;
    }
    lsm.flush_and_compact_all()?;
    // Two L0 files of overwrites and deletes, then unflushed memtable writes.
    for round in 0..2u64 {
        for i in (round..20_000).step_by(7) {
            if i % 3 == 0 {
                lsm.delete_with_seq(&key(i), seq)?;
            } else {
                lsm.insert_with_seq(&key(i), u128::from(i + 1_000_000 * (round + 1)), seq)?;
            }
            seq += 1;
        }
        lsm.flush_memtable_to_level0()?;
    }
    for i in (0..20_000u64).step_by(101) {
        lsm.insert_with_seq(&key(i), 7, seq)?;
        seq += 1;
    }
    let l1_bytes: u64 = (0..2).map(|b| lsm.sstable_files[b].read().metadata().map(|m| m.len()).unwrap_or(0)).sum();
    assert!(l1_bytes > 500_000, "setup: L1 should be large, is {l1_bytes} bytes");

    let mut probes: Vec<Vec<u8>> = (0..20_000u64).step_by(13).map(key).collect();
    probes.extend((0..50u64).map(|i| format!("doc:{i:08}x").into_bytes())); // absent, between keys
    probes.push(key(4_000)); // duplicated
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut pick = |n: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % n as u64) as usize
    };
    for size in [1usize, 2, 5, 20, 200, probes.len()] {
        for _ in 0..10 {
            let batch: Vec<Vec<u8>> = (0..size).map(|_| probes[pick(probes.len())].clone()).collect();
            let (got, read) = lsm.get_multiple_counted(&batch)?;
            for (k, g) in batch.iter().zip(&got) {
                assert_eq!(*g, lsm.get_with_seq(k)?, "batch of {size}: key {:?}", String::from_utf8_lossy(k));
            }
            if size == 1 {
                assert!(read < 64 * 1024, "a one-key batch read {read} bytes (L1 is {l1_bytes})");
            }
            assert!(read <= 2 * l1_bytes, "a batch of {size} read {read} bytes, more than the files");
        }
    }
    Ok(())
}

/// A stale or bogus sparse-index hint (an L1 file swapped by a compaction
/// under the index) must never lose a candidate in a batch read: it is
/// rejected and the cursor keeps scanning from where it is.
#[test]
fn test_batch_scan_survives_bogus_hints() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config = LSMConfig {
        num_buckets: 1,
        ..LSMConfig::default()
    };
    let lsm = LSMTree::open(temp_dir.path(), config)?;
    for i in 0..3_000u64 {
        lsm.insert_with_seq(format!("k:{i:08}").as_bytes(), u128::from(i) + 1, i + 1)?;
    }
    lsm.flush_and_compact_all()?;
    let file = lsm.sstable_files[0].read().clone();
    let file_len = file.metadata()?.len();
    let keys: Vec<Vec<u8>> = [5u64, 900, 901, 2_999].iter().map(|i| format!("k:{i:08}").into_bytes()).collect();
    let late = lsm.sstable_indexes[0].read().as_ref().map(|x| x.block_start(b"k:99999999")).unwrap_or(0);
    for bad in [u64::MAX, 1, 3, file_len / 2 + 1, file_len, late] {
        let candidates: Vec<(&[u8], u64)> = keys.iter().map(|k| (k.as_slice(), bad)).collect();
        let mut seen = Vec::new();
        let (_, rejected) = lsm.scan_sorted_candidates(&file, file_len, &candidates, |a| {
            if keys.iter().any(|k| k.as_slice() == a.key.as_slice()) {
                seen.push(a.key.to_vec());
            }
        })?;
        assert_eq!(seen, keys, "bogus hint {bad} lost a candidate");
        // A hint at or below the cursor's start is never needed, so never checked.
        assert_eq!(rejected, bad > SSTABLE_DATA_START, "bogus hint {bad}: rejection not reported as expected");
    }
    Ok(())
}
