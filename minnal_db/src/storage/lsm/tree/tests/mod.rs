//! Unit tests for the LSM tree, one file per topic, with the fixtures they share.

mod batch_reads;
mod compaction;
mod flush;
mod format;
mod point_reads;
mod prefix_scan;
mod prefix_sets;
mod range_pages;

use super::*;
use std::time::Duration;
use tempfile::TempDir;

/// The first `limit` live keys `>= start`, through `range_pointers_bounded`.
fn range_keys(lsm: &LSMTree, start: &[u8], limit: usize) -> Result<Vec<Vec<u8>>> {
    Ok(lsm.range_pointers_bounded(start, None, limit)?.into_iter().map(|(k, _, _)| k).collect())
}

/// LSM config for tests: a small bucket count keeps the eager per-tree fd
/// footprint low (one L1 file per bucket) so the suite survives high
/// `cargo test` parallelism. See [`crate::support::TEST_NUM_BUCKETS`]. None
/// of these tests depend on the specific bucket count, only on correctness.
fn test_lsm_config() -> LSMConfig {
    LSMConfig {
        num_buckets: crate::support::TEST_NUM_BUCKETS,
        ..LSMConfig::default()
    }
}

/// Keep `(value, version)` in `best` if it is newer than what `best` holds;
/// fed newest layer first, an exact tie stays with the newer copy.
fn keep_newest(best: &mut Candidate, value: Option<u128>, version: Version) {
    if best.is_none_or(|(_, b)| version > b) {
        *best = Some((value, version));
    }
}

/// Brute-force point lookup for the differential tests (`point_reads`,
/// `prefix_scan`): reads every layer in full — no summaries, no stopping at the
/// first hit — and keeps the highest version. On any tree the writers can
/// produce (newer layers hold newer versions) this is the first copy found, so
/// the production reads, which stop there, must agree with it.
fn reference_get(lsm: &LSMTree, key: &[u8]) -> Result<Option<(u128, Version)>> {
    let mut best: Candidate = None;
    if let Some((value, version, tombstone)) = lsm.memtable.read().skip_list.entry(key) {
        keep_newest(&mut best, (!tombstone).then_some(value), version);
    }
    for ro in lsm.read_only_memtables.read().iter().rev() {
        if let Some(r) = ro.records().iter().find(|r| r.key == key) {
            keep_newest(&mut best, (!r.tombstone).then_some(r.value), r.version);
        }
    }
    let bucket = get_bucket_for_key(key, lsm.config.num_buckets);
    let mut files = lsm.level0_entries_snapshot(bucket);
    files.retain(|f| !f.is_obsolete());
    files.sort_by_key(|f| std::cmp::Reverse(f.created_at_ms));
    for file in files {
        for e in lsm.read_sstable_entries_from_path(&file.path)? {
            if e.key == key {
                keep_newest(&mut best, (!e.tombstone).then_some(e.value), e.version());
            }
        }
    }
    for e in lsm.read_sstable_entries(bucket)? {
        if e.key == key {
            keep_newest(&mut best, (!e.tombstone).then_some(e.value), e.version());
        }
    }
    Ok(best.and_then(|(value, version)| value.map(|ptr| (ptr, version))))
}
