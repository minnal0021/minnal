//! Measures what resolving a point read by highest seq across every layer costs
//! against stopping at the first layer that holds the key, on a tree with copies
//! of keys in several layers, per class of key (where its newest copy is).
//! Measured 2026-11-14 (release, tmpfs): first hit saves about 1% on L0 keys,
//! L1-only keys and misses, and 21–25% on keys in the memtables, all of it on
//! the half in a read-only memtable (the active memtable takes the fast path
//! either way) — a state that lasts only until the sealed memtable is flushed.
//! So reads stay seq-resolved (`docs/Layer-Order-Design.md`). `#[ignore]`d: run by hand, in
//! release:
//!
//! `cargo test --release -p minnal_db --lib -- --ignored first_hit --nocapture`

use super::*;
use std::time::Instant;

impl LSMTree {
    /// A point read that stops at the first layer holding the key: the active
    /// memtable, the unflushed read-only memtables (newest first), the bucket's
    /// live L0 files (newest first), then L1. Correct only while every newer
    /// layer holds newer versions (`docs/Layer-Order-Design.md`).
    fn first_hit_for_measurement(&self, key: &[u8]) -> Result<Candidate> {
        if let Some((value, seq, tombstone)) = self.memtable.read().skip_list.entry(key) {
            return Ok(Some(((!tombstone).then_some(value), seq)));
        }
        {
            let ro_memtables = self.read_only_memtables.read();
            for ro in ro_memtables.iter().rev() {
                if let Some(r) = ro.find(key) {
                    return Ok(Some(((!r.tombstone).then_some(r.value), r.seq)));
                }
            }
        }
        let bucket = get_bucket_for_key(key, self.config.num_buckets);
        let mut entries = self.level0_entries_snapshot(bucket);
        entries.sort_by_key(|b| std::cmp::Reverse(b.created_at_ms));
        for entry in entries {
            if entry.is_obsolete() || !entry.may_contain(key) {
                continue;
            }
            let _guard = L0ReadGuard::new(Arc::clone(&entry));
            match self.lookup_in_sstable_file_path(entry.path.clone(), key, entry.scan_start(key))? {
                SsLookup::Found(ptr, seq) => return Ok(Some((Some(ptr), seq))),
                SsLookup::Deleted(seq) => return Ok(Some((None, seq))),
                SsLookup::Missing => {}
            }
        }
        Ok(match self.lookup_level1(bucket, key, None)? {
            SsLookup::Found(ptr, seq) => Some((Some(ptr), seq)),
            SsLookup::Deleted(seq) => Some((None, seq)),
            SsLookup::Missing => None,
        })
    }
}

/// L1 with `KEYS` keys, six L0 files each overwriting (or deleting) a tenth of
/// them, a read-only memtable and an active memtable each overwriting a
/// twentieth: every layer holds copies of keys that are also below it.
fn layered_tree(dir: &TempDir) -> Result<LSMTree> {
    const KEYS: u64 = 200_000;
    let lsm = LSMTree::open(
        dir.path(),
        LSMConfig {
            num_buckets: 8,
            skip_list_capacity: 1_000_000,
            ..LSMConfig::default()
        },
    )?;
    let mut seq = 0u64;
    let mut write = |lsm: &LSMTree, i: u64, delete: bool| -> Result<()> {
        seq += 1;
        let key = format!("{:016x}", i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        if delete {
            lsm.delete_with_seq(key.as_bytes(), seq)?;
        } else {
            lsm.insert_with_seq(key.as_bytes(), u128::from(seq), seq)?;
        }
        Ok(())
    };
    for i in 0..KEYS {
        write(&lsm, i, false)?;
    }
    lsm.flush_memtable_to_level0()?;
    lsm.compact_all()?;
    for file in 0..6u64 {
        for i in (file..KEYS).step_by(10) {
            write(&lsm, i, i % 70 == 0)?;
        }
        lsm.flush_memtable_to_level0()?;
    }
    for i in (3..KEYS).step_by(20) {
        write(&lsm, i, false)?;
    }
    // Take the flushed memtables off the read path, as the compaction worker
    // does on its next tick; left in place, a read would find the L0 files'
    // keys in memory.
    lsm.release_flushed_memtables();
    lsm.flush_memtable()?; // read-only, not flushed
    for i in (7..KEYS).step_by(20) {
        write(&lsm, i, false)?;
    }
    Ok(lsm)
}

#[test]
#[ignore]
fn first_hit_against_seq_resolution() -> Result<()> {
    let dir = TempDir::new()?;
    let lsm = layered_tree(&dir)?;
    assert!(lsm.layer_order_violations()?.is_empty(), "the measurement needs an ordered tree");
    let keys: Vec<Vec<u8>> = (0..400_000u64)
        .map(|i| format!("{:016x}", i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).into_bytes())
        .collect(); // half hits, half misses
    for key in keys.iter().step_by(97) {
        assert_eq!(lsm.newest_copy(key)?, lsm.first_hit_for_measurement(key)?, "the two reads disagree");
    }

    let key_of = |i: u64| format!("{:016x}", i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).into_bytes();
    let classes: [(&str, Vec<Vec<u8>>); 4] = [
        ("memtables", (0..200_000u64).filter(|i| i % 20 == 3 || i % 20 == 7).map(key_of).collect()),
        (
            "L0",
            (0..200_000u64).filter(|i| i % 10 < 6 && i % 20 != 3 && i % 20 != 7).map(key_of).collect(),
        ),
        ("L1 only", (0..200_000u64).filter(|i| i % 10 >= 6 && i % 20 != 7).map(key_of).collect()),
        ("misses", (200_000..300_000u64).map(key_of).collect()),
    ];
    for (name, keys) in &classes {
        let time = |first_hit_read: bool| -> Result<f64> {
            let started = Instant::now();
            for key in keys {
                if first_hit_read {
                    std::hint::black_box(lsm.first_hit_for_measurement(key)?);
                } else {
                    std::hint::black_box(lsm.newest_copy(key)?);
                }
            }
            Ok(started.elapsed().as_nanos() as f64 / keys.len() as f64)
        };
        time(false)?; // warm-up
        time(true)?;
        let mut seq_resolved = f64::MAX;
        let mut first_hit = f64::MAX;
        for round in 0..6 {
            // Alternate which goes first, so neither always reads warm caches.
            let order = if round % 2 == 0 { [false, true] } else { [true, false] };
            for first_hit_read in order {
                let t = time(first_hit_read)?;
                if first_hit_read {
                    first_hit = first_hit.min(t);
                } else {
                    seq_resolved = seq_resolved.min(t);
                }
            }
        }
        println!(
            "{name:>9}: {} keys, seq-resolved {seq_resolved:.0} ns/read, first hit {first_hit:.0} ns/read ({:+.1}%)",
            keys.len(),
            (first_hit / seq_resolved - 1.0) * 100.0
        );
    }
    Ok(())
}
