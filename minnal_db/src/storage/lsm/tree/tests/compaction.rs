//! Flushes, merges into Level 1, persistence, and when a merge is due.

use super::*;

#[test]
fn test_lsm_stats() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"key1", 12345)?;
    lsm.insert(b"key2", 67890)?;

    let stats = lsm.stats();
    assert_eq!(stats.memtable_entries, 2);

    Ok(())
}

#[test]
fn test_lsm_flush_and_readonly() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    // Insert data
    lsm.insert(b"key1", 100)?;
    lsm.insert(b"key2", 200)?;

    // Manually flush to create read-only memtable
    lsm.flush_memtable()?;

    // Check stats
    let stats = lsm.stats();
    assert_eq!(stats.memtable_entries, 0); // New memtable is empty
    assert_eq!(stats.read_only_count, 1); // One read-only snapshot

    // Should still be able to read from read-only layer
    assert_eq!(lsm.get(b"key1")?, Some(100));
    assert_eq!(lsm.get(b"key2")?, Some(200));

    Ok(())
}

#[test]
fn test_lsm_compaction() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    // Insert and flush to create read-only memtable
    lsm.insert(b"apple", 1)?;
    lsm.insert(b"banana", 2)?;
    lsm.insert(b"cherry", 3)?;
    lsm.flush_memtable()?;

    // Compact all buckets
    lsm.compact_all()?;

    // After compaction, read-only memtables should be cleared
    let stats = lsm.stats();
    assert_eq!(stats.read_only_count, 0);

    // Data should still be accessible from SSTable
    assert_eq!(lsm.get(b"apple")?, Some(1));
    assert_eq!(lsm.get(b"banana")?, Some(2));
    assert_eq!(lsm.get(b"cherry")?, Some(3));

    Ok(())
}

/// A failed `merge_level0_to_level1` must not leave a bucket's
/// `compaction_in_progress` flag stuck at `true` (which would make every
/// future compaction of that bucket a silent no-op until restart). The merge
/// is made to fail by corrupting the bucket's registered L0 files, then the
/// test asserts the flag is cleared and a retry re-enters the merge.
#[test]
fn test_compact_bucket_resets_flag_on_merge_error() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(b"key1", 12345)?;
    lsm.flush_memtable_to_level0()?; // writes L0 SSTable file(s) without compacting

    // Corrupt every L0 file: a size header claiming more bytes than actually
    // follow, so the merge's read_sstable_entries_from_path fails on
    // read_exact. (A *missing* file is treated as empty, not an error, so the
    // file must exist but be malformed.)
    //
    // The corruption must keep a **valid SSTable header** — these files stay in
    // the L0 registry, and registered files are opened without re-validating
    // (see `open_registered_l0`). A headerless body would just read as EOF past
    // `SSTABLE_DATA_START`, i.e. an empty file, and the merge would succeed.
    let mut corrupted = 0;
    for bucket in 0..lsm.config.num_buckets as u32 {
        let bucket_dir = LSMTree::level0_bucket_dir_from(&lsm.base_path, bucket);
        let Ok(rd) = std::fs::read_dir(&bucket_dir) else { continue };
        for entry in rd {
            let path = entry?.path();
            if path.is_file() {
                let mut bad = sstable_header_bytes().to_vec();
                bad.extend_from_slice(&100u32.to_le_bytes()); // claims a 100-byte entry…
                bad.extend_from_slice(&[0xABu8; 10]); // …but only 10 bytes follow
                std::fs::write(&path, &bad)?;
                corrupted += 1;
            }
        }
    }
    assert!(corrupted > 0, "expected at least one L0 file to corrupt");

    // Compact the bucket holding the orphaned entry — the merge must error.
    let errored_bucket = (0..lsm.config.num_buckets as u32)
        .find(|&b| lsm.compact_bucket(b).is_err())
        .expect("compacting the bucket with the missing L0 file must error");

    // The flag must be reset (not wedged at true) after the failed merge…
    assert!(
        !lsm.compaction_in_progress[errored_bucket as usize].load(Ordering::SeqCst),
        "compaction_in_progress must be cleared after a failed merge"
    );
    // …and a retry must actually re-enter the merge (error again), not
    // silently no-op via a stuck flag.
    assert!(
        lsm.compact_bucket(errored_bucket).is_err(),
        "retry must re-attempt the merge, not short-circuit on a stuck flag"
    );

    Ok(())
}

#[test]
fn test_lsm_two_way_merge() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    // First batch: insert initial data
    lsm.insert(b"key1", 100)?;
    lsm.insert(b"key2", 200)?;
    lsm.insert(b"key3", 300)?;
    lsm.flush_memtable()?;

    // Compact to SSTable
    lsm.compact_all()?;

    // Second batch: update and add new keys
    lsm.insert(b"key1", 111)?; // Update
    lsm.insert(b"key4", 400)?; // New
    lsm.flush_memtable()?;

    // Compact again - should merge
    lsm.compact_all()?;

    // Verify merged results
    assert_eq!(lsm.get(b"key1")?, Some(111)); // Updated value
    assert_eq!(lsm.get(b"key2")?, Some(200)); // Original value
    assert_eq!(lsm.get(b"key3")?, Some(300)); // Original value
    assert_eq!(lsm.get(b"key4")?, Some(400)); // New value

    Ok(())
}

#[test]
fn test_lsm_delete_with_compaction() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    // Insert data
    lsm.insert(b"key1", 100)?;
    lsm.insert(b"key2", 200)?;
    lsm.insert(b"key3", 300)?;
    lsm.flush_memtable()?;

    // Compact to SSTable
    lsm.compact_all()?;

    // Delete a key
    lsm.delete(b"key2")?;
    lsm.flush_memtable()?;

    // Compact again - tombstone should remove the entry
    lsm.compact_all()?;

    // Verify deletion
    assert_eq!(lsm.get(b"key1")?, Some(100));
    assert_eq!(lsm.get(b"key2")?, None); // Deleted
    assert_eq!(lsm.get(b"key3")?, Some(300));

    Ok(())
}

#[test]
fn test_lsm_persistence() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let path = temp_dir.path().to_path_buf();

    // First session: insert and compact
    {
        let lsm = LSMTree::open(&path, test_lsm_config())?;
        lsm.insert(b"persistent_key1", 1000)?;
        lsm.insert(b"persistent_key2", 2000)?;
        lsm.flush_memtable()?;

        lsm.compact_all()?;
    }

    // Second session: reopen and verify
    {
        let lsm = LSMTree::open(&path, test_lsm_config())?;
        assert_eq!(lsm.get(b"persistent_key1")?, Some(1000));
        assert_eq!(lsm.get(b"persistent_key2")?, Some(2000));
    }

    Ok(())
}

#[test]
fn test_lsm_compaction_stress() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = Arc::new(LSMTree::open(temp_dir.path(), test_lsm_config())?);

    let writer_threads = 4usize;
    let writes_per_thread = 200usize;
    let stop_compaction = Arc::new(AtomicBool::new(false));

    let compaction_lsm = lsm.clone();
    let compaction_stop = stop_compaction.clone();
    let compaction_handle = std::thread::spawn(move || -> Result<()> {
        while !compaction_stop.load(Ordering::SeqCst) {
            if let Err(err) = compaction_lsm.compact_all() {
                if let LSMError::Io(io_err) = &err
                    && io_err.kind() == std::io::ErrorKind::NotFound
                {
                    std::thread::yield_now();
                    continue;
                }
                return Err(err);
            }
            std::thread::yield_now();
        }
        Ok(())
    });

    let mut handles = Vec::with_capacity(writer_threads);
    for thread_id in 0..writer_threads {
        let writer_lsm = lsm.clone();
        let handle = std::thread::spawn(move || -> Result<()> {
            for i in 0..writes_per_thread {
                let key = format!("stress:key:{}:{}", thread_id, i).into_bytes();
                let value = (((thread_id as u128) << 32) | i as u128).saturating_add(1);
                let mut attempts = 0u8;
                loop {
                    match writer_lsm.insert(&key, value) {
                        Ok(()) => break,
                        Err(LSMError::CapacityExceeded) => {
                            attempts += 1;
                            writer_lsm.flush_memtable()?;
                            std::thread::sleep(Duration::from_millis(1));
                            if attempts >= 10 {
                                return Err(LSMError::CapacityExceeded);
                            }
                        }
                        Err(err) => return Err(err),
                    }
                }
                if i % 25 == 0 {
                    writer_lsm.delete(&key)?;
                }
            }
            Ok(())
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap()?;
    }

    stop_compaction.store(true, Ordering::SeqCst);
    compaction_handle.join().unwrap()?;

    lsm.flush_and_compact_all()?;

    for thread_id in 0..writer_threads {
        let key = format!("stress:key:{}:{}", thread_id, writes_per_thread - 1).into_bytes();
        let value = lsm.get(&key)?;
        assert!(value.is_some());
    }

    Ok(())
}

/// Live (not yet merged) Level-0 files in `bucket`.
fn live_l0(lsm: &LSMTree, bucket: usize) -> usize {
    lsm.level0_files[bucket].read().iter().filter(|f| !f.is_obsolete()).count()
}

/// The worker merges a bucket's L0 into L1 only when the L0 files hold
/// `L0_COMPACTION_RATIO_PERCENT` of L1's entries, not after every flush: a
/// merge rewrites the whole L1 file. Into an empty L1 any L0 is due.
#[test]
fn small_flushes_wait_until_level0_is_a_share_of_level1() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config = LSMConfig {
        num_buckets: 1,
        ..LSMConfig::default()
    };
    let lsm = LSMTree::open(temp_dir.path(), config)?;
    let key = |i: u64| format!("doc:{i:08}").into_bytes();
    let mut seq = 1u64;
    lsm.insert_with_seq(&key(0), 1, seq)?;
    seq += 1;
    lsm.flush_memtable_to_level0()?;
    assert!(lsm.compaction_due(0), "an empty L1 takes any L0");
    lsm.compact_due()?;
    assert_eq!(live_l0(&lsm, 0), 0);

    for i in 1..1_000u64 {
        lsm.insert_with_seq(&key(i), 1, seq)?;
        seq += 1;
    }
    lsm.flush_and_compact_all()?;
    // 1,000 entries in L1: flushes of 30 wait until L0 holds 100 (10%).
    let mut flushed = 0u64;
    for round in 0..3u64 {
        for i in 0..30u64 {
            lsm.insert_with_seq(&key(2_000 + round * 30 + i), 1, seq)?;
            seq += 1;
        }
        lsm.flush_memtable_to_level0()?;
        flushed += 30;
        assert!(!lsm.compaction_due(0), "{flushed} L0 entries against 1,000 in L1 is not due");
        lsm.compact_due()?;
        assert_eq!(live_l0(&lsm, 0), round as usize + 1, "nothing merged yet");
        assert!(!lsm.has_due_compaction_work(), "the tick released the flushed memtable");
    }
    for i in 0..10u64 {
        lsm.insert_with_seq(&key(3_000 + i), 1, seq)?;
        seq += 1;
    }
    lsm.flush_memtable_to_level0()?;
    assert!(lsm.compaction_due(0), "100 L0 entries is 10% of L1");
    lsm.compact_due()?;
    assert_eq!(live_l0(&lsm, 0), 0);
    for i in (0..1_000u64).chain(2_000..2_090).chain(3_000..3_010) {
        assert!(lsm.get_with_seq(&key(i))?.is_some(), "key {i} survives the merge");
    }
    Ok(())
}

/// However small, `L0_COMPACTION_MAX_FILES` L0 files make a merge due: each
/// one a point read cannot rule out is a file it reads.
#[test]
fn many_tiny_flushes_are_merged_at_the_file_cap() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config = LSMConfig {
        num_buckets: 1,
        ..LSMConfig::default()
    };
    let lsm = LSMTree::open(temp_dir.path(), config)?;
    let key = |i: u64| format!("doc:{i:08}").into_bytes();
    let mut seq = 1u64;
    for i in 0..10_000u64 {
        lsm.insert_with_seq(&key(i), 1, seq)?;
        seq += 1;
    }
    lsm.flush_and_compact_all()?;
    for n in 1..=L0_COMPACTION_MAX_FILES {
        lsm.insert_with_seq(&key(n as u64), 2, seq)?;
        seq += 1;
        lsm.flush_memtable_to_level0()?;
        assert_eq!(lsm.compaction_due(0), n == L0_COMPACTION_MAX_FILES, "{n} one-entry files");
    }
    lsm.compact_due()?;
    assert_eq!(live_l0(&lsm, 0), 0);
    Ok(())
}

/// The explicit and shutdown path, `compact_all`, still merges every bucket
/// that has any L0 file; only the worker's `compact_due` waits.
#[test]
fn compact_all_merges_what_compact_due_leaves() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config = LSMConfig {
        num_buckets: 1,
        ..LSMConfig::default()
    };
    let lsm = LSMTree::open(temp_dir.path(), config)?;
    let key = |i: u64| format!("doc:{i:08}").into_bytes();
    let mut seq = 1u64;
    for i in 0..1_000u64 {
        lsm.insert_with_seq(&key(i), 1, seq)?;
        seq += 1;
    }
    lsm.flush_and_compact_all()?;
    lsm.delete_with_seq(&key(5), seq)?;
    lsm.flush_memtable_to_level0()?;
    lsm.compact_due()?;
    assert_eq!(live_l0(&lsm, 0), 1, "one tombstone is not due");
    assert!(lsm.get_with_seq(&key(5))?.is_none(), "the L0 tombstone shadows L1");
    lsm.compact_all()?;
    assert_eq!(live_l0(&lsm, 0), 0);
    assert!(lsm.get_with_seq(&key(5))?.is_none(), "the merge drops the tombstone and the key");
    assert!(lsm.get_with_seq(&key(6))?.is_some());
    Ok(())
}
