//! Multi-prefix scans, the Level-1 run merge, and the scan latency benchmark.

use super::*;

fn u32_prefixed_key(prefix: u32, suffix: u32) -> Vec<u8> {
    let mut k = prefix.to_be_bytes().to_vec();
    k.extend_from_slice(&suffix.to_be_bytes());
    k
}

#[test]
fn scan_prefixes_over_l1_seeks_per_prefix() -> Result<()> {
    // Many cluster prefixes flushed to L1; probing a few must return exactly
    // those clusters' keys (per-prefix sparse seek + bucket min/max skip).
    let temp = TempDir::new()?;
    let lsm = LSMTree::open(temp.path(), test_lsm_config())?;
    let n_clusters = 200u32;
    let per = 30u32;
    for cid in 0..n_clusters {
        for did in 0..per {
            lsm.insert(&u32_prefixed_key(cid, did), (cid as u128) * 1000 + did as u128)?;
        }
    }
    lsm.flush_and_compact_all()?;
    lsm.purge_ro_memtables_for_test();

    // Probe three scattered clusters out of 200.
    let ids: std::collections::HashSet<u32> = [3u32, 100, 199].into_iter().collect();
    let results = lsm.scan_prefixes(&ids)?.to_pairs();
    assert_eq!(results.len(), (ids.len() as u32 * per) as usize);
    for (k, v) in &results {
        let cid = u32::from_be_bytes(k[..4].try_into().unwrap());
        let did = u32::from_be_bytes(k[4..8].try_into().unwrap());
        assert!(ids.contains(&cid), "returned a non-probed cluster {cid}");
        assert_eq!(*v, (cid as u128) * 1000 + did as u128);
    }
    // A cluster id past every key's range → min/max skip → empty.
    let none: std::collections::HashSet<u32> = [u32::MAX].into_iter().collect();
    assert!(lsm.scan_prefixes(&none)?.is_empty());
    Ok(())
}

/// `merge_scan_layers` orders by `key_head` and then the key; that must be the
/// keys' byte order, including keys that are prefixes of others, zero bytes
/// where padding sits, and keys on both sides of 16 bytes.
#[test]
fn key_head_then_key_is_byte_order() {
    let mut s = 0x1d87_2b41_ee3a_9f05_u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut keys: Vec<Vec<u8>> = (0..5_000).map(|_| (0..next() % 24).map(|_| (next() % 3) as u8).collect()).collect();
    let mut by_head = keys.clone();
    by_head.sort_by(|a, b| key_head(a).cmp(&key_head(b)).then_with(|| a.cmp(b)));
    keys.sort();
    assert_eq!(by_head, keys);
}

/// The run merge must equal sorting every entry and keeping the last copy of
/// each key, in input order: keys of 0–23 bytes (many sharing a 16-byte head)
/// in 8 runs, with the same key in several runs, and the same entries shuffled.
#[test]
fn merging_l1_runs_equals_sorting_and_keeping_the_last() {
    let mut s = 0x5851_f42d_4c95_7f2d_u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for total in [1_000usize, 60_000] {
        let mut buffers: Vec<Vec<u8>> = Vec::new();
        let mut entries: Vec<PackedScanEntry> = Vec::new();
        for run in 0..8usize {
            let mut keys: Vec<Vec<u8>> = (0..total / 8)
                .map(|_| {
                    let len = (next() % 24) as usize;
                    let mut k: Vec<u8> = (0..len).map(|_| (next() % 4) as u8).collect();
                    if next() % 2 == 0 && len > 16 {
                        k[..16].fill(1);
                    }
                    k
                })
                .collect();
            keys.sort();
            keys.dedup();
            let mut buffer = Vec::new();
            for k in keys {
                let start = buffer.len();
                buffer.extend_from_slice(&k);
                let seq = next() % 1_000;
                entries.push((run, start, buffer.len(), Some(seq as u128), Version::user(seq)));
            }
            buffers.push(buffer);
        }
        let key = |e: &PackedScanEntry| &buffers[e.0][e.1..e.2];
        let last_of_each = |input: &[PackedScanEntry]| -> Vec<(Vec<u8>, Version)> {
            let mut reference = std::collections::BTreeMap::new();
            for e in input {
                reference.insert(key(e).to_vec(), e.4);
            }
            reference.into_iter().collect()
        };
        let merged: Vec<(Vec<u8>, Version)> = merge_l1_runs(&entries, key).iter().map(|e| (key(e).to_vec(), e.4)).collect();
        assert_eq!(merged, last_of_each(&entries), "{total} entries");

        // In random order the input breaks into far more than `MAX_MERGED_RUNS`
        // runs, which takes the sorting path; it must keep input order too.
        let mut shuffled = entries.clone();
        for i in (1..shuffled.len()).rev() {
            shuffled.swap(i, (next() % (i as u64 + 1)) as usize);
        }
        let merged: Vec<(Vec<u8>, Version)> = merge_l1_runs(&shuffled, key).iter().map(|e| (key(e).to_vec(), e.4)).collect();
        assert_eq!(merged, last_of_each(&shuffled), "{total} entries, shuffled");
    }
}

/// The per-prefix cursors must return exactly each probed prefix's keys for
/// any probe set: sparse or dense, adjacent or far apart, prefixes with no
/// keys, postings that span several sparse-index blocks, and keys too short to
/// have a prefix (which no layer returns). Eight buckets, so the scan splits
/// the prefixes over several workers that each read every bucket. Checked with
/// everything in L1, then again with rewrites, deletes and new keys in several
/// L0 files above it, which the L0 cursor must seek through the same way.
#[test]
fn scan_prefixes_over_l1_and_l0_matches_a_reference_for_any_probe_set() -> Result<()> {
    let temp = TempDir::new()?;
    let lsm = LSMTree::open(
        temp.path(),
        LSMConfig {
            num_buckets: 8,
            ..LSMConfig::default()
        },
    )?;
    let mut s = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let n_clusters = 300u32;
    let mut truth: std::collections::BTreeMap<Vec<u8>, u128> = std::collections::BTreeMap::new();
    for cid in 0..n_clusters {
        // 0–80 keys per cluster, so a cluster can span up to five
        // `SAMPLE_INTERVAL`-entry blocks per bucket or have no keys at all.
        for did in 0..(next() % 81) as u32 {
            let key = u32_prefixed_key(cid, did);
            let value = (cid as u128) << 32 | did as u128;
            lsm.insert(&key, value)?;
            truth.insert(key, value);
        }
    }
    // Three bytes: no prefix id, never returned. Four bytes: prefix 5 alone.
    lsm.insert(&[0u8, 0, 1], 1)?;
    lsm.insert(&[0u8, 0, 0, 5], 2)?;
    truth.insert(vec![0, 0, 0, 5], 2);
    lsm.flush_and_compact_all()?;
    lsm.purge_ro_memtables_for_test();

    let mut check = |lsm: &LSMTree, truth: &std::collections::BTreeMap<Vec<u8>, u128>, layers: &str| -> Result<()> {
        for density in [1u64, 5, 50, 90, 100] {
            for _ in 0..5 {
                let ids: std::collections::HashSet<u32> = (0..n_clusters + 5).filter(|_| next() % 100 < density).collect();
                if ids.is_empty() {
                    continue;
                }
                let expected: Vec<(Vec<u8>, u128)> = truth
                    .iter()
                    .filter(|(k, _)| ids.contains(&u32::from_be_bytes(k[..4].try_into().unwrap())))
                    .map(|(k, v)| (k.clone(), *v))
                    .collect();
                assert_eq!(lsm.scan_prefixes(&ids)?.to_pairs(), expected, "{layers}, density {density}%");
            }
        }
        Ok(())
    };
    check(&lsm, &truth, "L1 only")?;

    // Three L0 files, each over every other cluster from a different start, so
    // a file holds some probed prefixes and not others and spans many
    // sparse-index blocks. A quarter of the writes delete (an L0 tombstone
    // over an L1 key, or over nothing).
    let mut wave_rng = 0x9e37_79b9_7f4a_7c15_u64;
    let mut wave_next = move || {
        wave_rng ^= wave_rng << 13;
        wave_rng ^= wave_rng >> 7;
        wave_rng ^= wave_rng << 17;
        wave_rng
    };
    for wave in 0..3u32 {
        for cid in (wave..n_clusters).step_by(2) {
            for did in 0..(wave_next() % 120) as u32 {
                let key = u32_prefixed_key(cid, did);
                if wave_next() % 4 == 0 {
                    lsm.delete(&key)?;
                    truth.remove(&key);
                } else {
                    let value = ((cid as u128) << 32 | did as u128) + (u128::from(wave) + 1) * 1_000_000;
                    lsm.insert(&key, value)?;
                    truth.insert(key, value);
                }
            }
        }
        lsm.flush_memtable_to_level0()?;
    }
    // Flushed read-only memtables still answer scans until released; drop them
    // so the second wave is read from L0 alone.
    lsm.purge_ro_memtables_for_test();
    let l0_files: usize = lsm
        .level0_files
        .iter()
        .map(|b| b.read().iter().filter(|e| !e.is_obsolete()).count())
        .sum();
    assert!(l0_files >= 3, "the second wave must stay in L0 ({l0_files} files)");
    check(&lsm, &truth, "L1 and L0")?;
    Ok(())
}

/// A scan probing one posting of an L0 file must seek to it through the
/// file's sparse index, not read the file from its head: the result alone
/// cannot show this, because a rejected or missing hint falls back to reading
/// in order and still finds every key.
#[test]
fn scan_prefixes_seeks_through_level0_files() -> Result<()> {
    let temp = TempDir::new()?;
    let lsm = LSMTree::open(
        temp.path(),
        LSMConfig {
            num_buckets: 1,
            ..LSMConfig::default()
        },
    )?;
    let (n_clusters, per) = (300u32, 40u32);
    for cid in 0..n_clusters {
        for did in 0..per {
            lsm.insert(&u32_prefixed_key(cid, did), u128::from(cid) << 32 | u128::from(did))?;
        }
    }
    lsm.flush_memtable_to_level0()?;
    lsm.purge_ro_memtables_for_test();

    let probed = [n_clusters / 2];
    let (mut found, mut read, mut file_bytes) = (0u32, 0u64, 0u64);
    for entry in lsm.level0_files[0].read().iter() {
        let file = open_registered_l0(&entry.path)?;
        let starts: Vec<u64> = probed.iter().map(|id| entry.scan_start(&id.to_be_bytes())).collect();
        read += LSMTree::for_each_entry_in_prefixes(&file, &starts, &probed, |_| found += 1)?;
        file_bytes += file.metadata()?.len();
    }
    assert_eq!(found, per);
    assert!(
        read * 10 < file_bytes,
        "read {read} of {file_bytes} bytes for one posting of {n_clusters}"
    );

    let ids: std::collections::HashSet<u32> = probed.into_iter().collect();
    assert_eq!(lsm.scan_prefixes(&ids)?.len(), per as usize);
    Ok(())
}

/// `scan_prefix` reads an L0 file only near the prefix: a file whose key
/// range misses it is not read at all, and one that holds it is read from
/// the sparse-index block holding the prefix to the first key past it.
#[test]
fn scan_prefix_seeks_through_level0_files() -> Result<()> {
    let temp = TempDir::new()?;
    let lsm = LSMTree::open(
        temp.path(),
        LSMConfig {
            num_buckets: 1,
            ..LSMConfig::default()
        },
    )?;
    // One L0 file of `a:` keys, then one of `b:` keys.
    for i in 0..5_000u32 {
        lsm.insert(format!("a:{i:05}").as_bytes(), u128::from(i))?;
    }
    lsm.flush_memtable_to_level0()?;
    for i in 0..100u32 {
        lsm.insert(format!("b:{i:05}").as_bytes(), u128::from(i))?;
    }
    lsm.flush_memtable_to_level0()?;
    lsm.purge_ro_memtables_for_test();

    let frames_read = |lsm: &LSMTree| -> Vec<u64> {
        let mut files: Vec<_> = lsm.level0_files[0].read().iter().cloned().collect();
        files.sort_by_key(|f| f.created_at_ms);
        files.iter().map(|f| f.frames_read.load(Ordering::Relaxed)).collect()
    };

    let hits = lsm.scan_prefix(b"a:0250")?;
    assert_eq!(hits.len(), 10, "a:02500..a:02509");
    let [a_file, b_file] = frames_read(&lsm)[..] else {
        panic!("two L0 files")
    };
    assert!(
        a_file <= 2 * SparseIndex::SAMPLE_INTERVAL + 10,
        "read {a_file} of 5,000 frames for ten keys"
    );
    assert_eq!(b_file, 0, "a file whose key range misses the prefix is not read");

    // Every key still resolves when the prefix spans the whole file.
    assert_eq!(lsm.scan_prefix(b"a:")?.len(), 5_000);
    assert_eq!(lsm.scan_prefix(b"b:")?.len(), 100);
    Ok(())
}

#[test]
fn range_keys_bounded_over_l1_seeks_and_skips() -> Result<()> {
    let temp = TempDir::new()?;
    let lsm = LSMTree::open(temp.path(), test_lsm_config())?;
    for seq in 0..1000u64 {
        lsm.insert(format!("r:{seq:06}").as_bytes(), seq as u128)?;
    }
    lsm.flush_and_compact_all()?;
    lsm.purge_ro_memtables_for_test();

    // Unbounded from a mid start, limited page — seeks to start, min/max keeps
    // only buckets reaching it.
    let keys = range_keys(&lsm, b"r:000900", 50)?;
    assert_eq!(keys.len(), 50);
    assert_eq!(keys[0], b"r:000900".to_vec());
    assert!(keys.iter().all(|k| k.as_slice() >= b"r:000900".as_slice()));

    // Start past the max key → every bucket skipped → empty.
    assert!(range_keys(&lsm, b"zzz", usize::MAX)?.is_empty());
    // Full range from the front returns everything.
    assert_eq!(range_keys(&lsm, b"r:", usize::MAX)?.len(), 1000);
    Ok(())
}

#[test]
fn test_scan_prefixes_basic() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    // cluster 1: docs 10, 11, 12; cluster 2: doc 20; cluster 3: doc 30
    let entries = [(1u32, 10u32, 100u128), (1, 11, 101), (1, 12, 102), (2, 20, 200), (3, 30, 300)];
    for (cid, did, val) in entries {
        lsm.insert(&u32_prefixed_key(cid, did), val)?;
    }

    let ids: std::collections::HashSet<u32> = [1, 2].iter().copied().collect();
    let results = lsm.scan_prefixes(&ids)?.to_pairs();

    // Should return all keys for clusters 1 and 2, not cluster 3
    assert_eq!(results.len(), 4);
    let other_key = u32_prefixed_key(3, 30);
    assert!(!results.iter().any(|(k, _)| k == &other_key));
    assert!(results.iter().any(|(k, v)| k == &u32_prefixed_key(1, 10) && *v == 100));
    assert!(results.iter().any(|(k, v)| k == &u32_prefixed_key(2, 20) && *v == 200));
    Ok(())
}

#[test]
fn test_scan_prefixes_after_flush_and_compaction() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    // Write to memtable and flush
    lsm.insert(&u32_prefixed_key(5, 1), 501)?;
    lsm.insert(&u32_prefixed_key(5, 2), 502)?;
    lsm.flush_memtable()?;
    lsm.compact_all()?;

    // Add more in fresh memtable
    lsm.insert(&u32_prefixed_key(5, 3), 503)?;
    lsm.insert(&u32_prefixed_key(6, 1), 601)?;

    let ids: std::collections::HashSet<u32> = [5].iter().copied().collect();
    let results = lsm.scan_prefixes(&ids)?.to_pairs();

    assert_eq!(results.len(), 3);
    assert!(results.iter().any(|(k, v)| k == &u32_prefixed_key(5, 1) && *v == 501));
    assert!(results.iter().any(|(k, v)| k == &u32_prefixed_key(5, 2) && *v == 502));
    assert!(results.iter().any(|(k, v)| k == &u32_prefixed_key(5, 3) && *v == 503));
    Ok(())
}

#[test]
fn test_scan_prefixes_empty_result() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(&u32_prefixed_key(1, 1), 10)?;

    let ids: std::collections::HashSet<u32> = [99].iter().copied().collect();
    let results = lsm.scan_prefixes(&ids)?.to_pairs();
    assert!(results.is_empty());
    Ok(())
}

// ── Scan latency benchmark (memtable / L1 / L0+L1) ──────────────────────
//
// A kept, deterministic micro-benchmark for the scan paths across LSM layer
// states. `#[ignore]`d so it never runs in normal `cargo test`; run it on demand:
//
//   cargo test -p minnal_db --lib scan_latency_benchmark -- --ignored --nocapture
//
// It lives here (not in `benches/`) on purpose: building an exact L0+L1 state is
// only possible through the crate-internal `LSMTree` flush methods: the public
// `AsyncDb` flushes to SSTables only through its background worker, so a criterion
// bench cannot construct these states deterministically. The full scans show
// throughput per layer; the *selective* scans show the L1 min/max skip +
// sparse-index seek (a bounded slice out of a large L1 should not read the whole
// file), which memtable-resident timings cannot show.

/// Insert `k:{seq:06}` for `seq` in `0..total`, all left in the active memtable.
fn bench_state_memtable(total: u64) -> (TempDir, LSMTree) {
    let temp = TempDir::new().unwrap();
    let lsm = LSMTree::open(temp.path(), test_lsm_config()).unwrap();
    for seq in 0..total {
        lsm.insert(format!("k:{seq:06}").as_bytes(), seq as u128).unwrap();
    }
    (temp, lsm)
}

/// Same keys, all compacted into L1 (memtable/RO drained).
fn bench_state_l1(total: u64) -> (TempDir, LSMTree) {
    let temp = TempDir::new().unwrap();
    let lsm = LSMTree::open(temp.path(), test_lsm_config()).unwrap();
    for seq in 0..total {
        lsm.insert(format!("k:{seq:06}").as_bytes(), seq as u128).unwrap();
    }
    lsm.flush_and_compact_all().unwrap();
    lsm.purge_ro_memtables_for_test();
    (temp, lsm)
}

/// Same keys spread across all three layers: ~80% in L1, ~15% in an uncompacted
/// L0 file, ~5% left in the memtable — the realistic "recently written over a
/// compacted base" shape a scan must merge.
fn bench_state_l0_l1(total: u64) -> (TempDir, LSMTree) {
    let temp = TempDir::new().unwrap();
    let lsm = LSMTree::open(temp.path(), test_lsm_config()).unwrap();
    let l1_cut = total * 80 / 100;
    let l0_cut = total * 95 / 100;
    for seq in 0..l1_cut {
        lsm.insert(format!("k:{seq:06}").as_bytes(), seq as u128).unwrap();
    }
    lsm.flush_and_compact_all().unwrap(); // → L1
    for seq in l1_cut..l0_cut {
        lsm.insert(format!("k:{seq:06}").as_bytes(), seq as u128).unwrap();
    }
    lsm.flush_memtable_to_level0().unwrap(); // → L0 (uncompacted)
    lsm.purge_ro_memtables_for_test();
    for seq in l0_cut..total {
        lsm.insert(format!("k:{seq:06}").as_bytes(), seq as u128).unwrap(); // → memtable
    }
    (temp, lsm)
}

#[test]
#[ignore]
fn scan_latency_benchmark() {
    use std::time::Instant;
    let total = 5_000u64;

    // Warm up, run `iters`, report median-ish average µs/call. Asserts the result
    // count so a broken scan can't silently post a fast time.
    fn time_it(label: &str, iters: u32, mut f: impl FnMut() -> usize, expect: usize) {
        for _ in 0..5 {
            assert_eq!(f(), expect);
        }
        let t = Instant::now();
        for _ in 0..iters {
            assert_eq!(f(), expect);
        }
        let per = t.elapsed().as_nanos() as f64 / iters as f64 / 1e3;
        eprintln!("    {label:34} {expect:>5} results  {per:>9.2} µs/call");
    }

    // A 50-key slice in the middle of the keyspace, for the selective scans.
    let mid = total / 2;
    let sel_start = format!("k:{mid:06}").into_bytes();
    let sel_end = format!("k:{:06}", mid + 50).into_bytes();
    let sel_prefix = format!("k:{:04}", mid / 100).into_bytes(); // matches k:{mid/100}XX → 100 keys

    eprintln!("\n=== scan latency: {total} keys, 128-bit values, per LSM layer state ===");
    for (state, (_temp, lsm)) in [
        ("memtable", bench_state_memtable(total)),
        ("L1-only", bench_state_l1(total)),
        ("L0+L1+mem", bench_state_l0_l1(total)),
    ] {
        eprintln!("  [{state}]");
        let l = &lsm;
        time_it("prefix(k:) full", 60, || l.scan_prefix(b"k:").unwrap().len(), total as usize);
        time_it(
            "range(k:..) full",
            60,
            || l.range_pointers_bounded(b"k:", Some(b"k;"), usize::MAX).unwrap().len(),
            total as usize,
        );
        let (s, e) = (&sel_start, &sel_end);
        time_it(
            "range selective [50]",
            500,
            || l.range_pointers_bounded(s, Some(e), usize::MAX).unwrap().len(),
            50,
        );
        let p = &sel_prefix;
        time_it("prefix selective [100]", 500, || l.scan_prefix(p).unwrap().len(), 100);
    }
}

// ── scan_prefixes and tombstones ──────────────────────────────

#[test]
fn test_scan_prefixes_deleted_after_flush() -> Result<()> {
    // A key deleted after a flush must not appear: the active memtable's
    // tombstone must hide its SSTable entry.
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(&u32_prefixed_key(1, 10), 100)?;
    lsm.insert(&u32_prefixed_key(1, 11), 101)?;
    lsm.flush_memtable()?; // both keys now in SSTable

    lsm.delete(&u32_prefixed_key(1, 11))?; // tombstone in active memtable

    let ids: std::collections::HashSet<u32> = [1].iter().copied().collect();
    let results = lsm.scan_prefixes(&ids)?.to_pairs();

    assert_eq!(results.len(), 1, "deleted key must not appear");
    assert!(results.iter().any(|(k, v)| k == &u32_prefixed_key(1, 10) && *v == 100));
    Ok(())
}

#[test]
fn test_scan_prefixes_tombstone_in_ro_memtable() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(&u32_prefixed_key(2, 1), 201)?;
    lsm.flush_memtable()?; // key in SSTable
    lsm.delete(&u32_prefixed_key(2, 1))?;
    lsm.flush_memtable()?; // tombstone now in ro_memtable

    let ids: std::collections::HashSet<u32> = [2].iter().copied().collect();
    let results = lsm.scan_prefixes(&ids)?.to_pairs();
    assert!(results.is_empty(), "key deleted in ro_memtable must not appear");
    Ok(())
}

#[test]
fn test_scan_prefixes_delete_then_reinsert() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

    lsm.insert(&u32_prefixed_key(3, 5), 300)?;
    lsm.flush_memtable()?;
    lsm.delete(&u32_prefixed_key(3, 5))?;
    lsm.flush_memtable()?; // tombstone in ro_memtable
    lsm.insert(&u32_prefixed_key(3, 5), 999)?; // re-insert in active memtable

    let ids: std::collections::HashSet<u32> = [3].iter().copied().collect();
    let results = lsm.scan_prefixes(&ids)?.to_pairs();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].1, 999, "re-inserted value must win");
    Ok(())
}
