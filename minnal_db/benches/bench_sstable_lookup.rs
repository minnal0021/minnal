// Point-lookup benchmarks across storage tiers: active memtable vs. on-disk
// L1 SSTable.
//   cargo bench --bench bench_sstable_lookup
//
// PURPOSE: compare `get()` latency for keys resident in the active memtable
// against keys pushed down to L1, using the same key/value shape and the
// same `num_keys` sweep for both, so the two tiers are directly comparable
// in one report/chart rather than scattered across separate benchmarks with
// different setups. This also isolates the cost of `LSMTree::lookup_level1`,
// the L1 lookup path originally flagged by the review item "LSM point
// lookups scan SSTables linearly". That review item has since been
// addressed: `lookup_level1` now fast-rejects via the L1 min/max key range,
// then a bloom filter (exact negative), then a sparse-index binary search
// that bounds the residual linear scan to ~SAMPLE_INTERVAL entries instead
// of the whole file — see its doc comment in `lsm_tree.rs`. This benchmark
// is no longer measuring a naive O(N) scan; it's the regression baseline
// for the accelerated path.
//
// L0 is deliberately NOT measured as its own tier: there is no public API
// path to observe an on-disk "flushed but not yet compacted" L0 state — the
// only way to write an L0 file (`compact_all`, reached via `Db::compact()`
// or the background `LsmCompactionWorker`) unconditionally merges every L0
// file into L1 in that same call. So L0 residency is not something a caller
// can hold open long enough to benchmark through the public API; memtable
// and L1 are the two tiers actually reachable and worth comparing.
//
// `bench_memtable_lookup` uses `open_store_with_capacity` (common.rs) to set
// the memtable capacity well above the largest `num_keys` tested, so the
// whole sweep stays inside a single active memtable — without this, the
// default 100_000 capacity's 95% flush threshold would seal (though not yet
// disk-flush) the memtable partway through the 100_000-key case, mixing in
// a different in-memory lookup path (`ReadOnlyMemTable`) and confounding
// the comparison.
//
// `bench_memtable_lookup` pushes nothing to disk, so it's what `bench_read`
// also measures (memtable-resident `get()`) — this benchmark exists
// alongside `bench_read` to hold key count, key shape, and value size fixed
// across both tiers for a like-for-like comparison; `bench_read` instead
// varies value size at a fixed (small) key count.
//
// `bench_l1_lookup` pushes all keys down to the on-disk L1 SSTable via
// compact + shutdown + reopen, so each `get()` runs the real (accelerated)
// L1 lookup.
//
// Keys are written at EVEN indices and we query:
//   * "hit"  — an even key that exists,
//   * "miss" — an odd key that falls *between* two existing keys,
// exercising the bloom-filter exact-negative path on the L1 side.
//
// Expected shape for L1 post-acceleration: latency should be roughly FLAT
// across `num_keys` (bloom/sparse-index cost doesn't scale with file size
// the way a full linear scan would) rather than growing 10x per decade of
// `num_keys`. Measured on one dev machine (see `minnal_db/benchmark.md`):
// L1 hit latency moved only ~10% from 1,000 to 100,000 keys (2.96 µs ->
// 3.25 µs), and L1 miss latency was flat throughout (~0.19-0.21 µs) —
// consistent with the fast-reject paths dominating over any residual scan.
// Memtable hit/miss is expected to be substantially cheaper than L1 (no
// disk I/O, no bloom/sparse-index indirection) — quantifying that gap is
// the point of measuring both tiers side by side here. If L1 latency starts
// climbing noticeably with `num_keys` in a future run, that's a signal the
// acceleration path regressed (e.g. bloom filter or sparse index not being
// consulted, or a false-positive rate high enough to fall through to a real
// scan often).
//
// `bench_mixed_tier_lookup` covers a third, more realistic shape: a
// read-only workload (no concurrent writes) against a store whose working
// set is split across both tiers at once — half the keys pushed to L1
// (written first, then compacted), half left resident in the active
// memtable (written after reopening, never flushed). Every other benchmark
// here tests a *pure* single-tier population; this one approximates
// steady state, where older data has aged onto disk and newer writes are
// still hot in memory, and a real read stream hits both. Each iteration
// alternates between an L1-resident key and a memtable-resident key, so the
// reported latency is the blended cost of a 50/50 tier mix, not either
// extreme.

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use minnal_db::Db;
use std::hint::black_box;
use std::time::Duration;
use tempfile::TempDir;

const VALUE_SIZE: usize = 16; // small, so the SSTable scan / memtable lookup dominates over value reads

// Shared num_keys sweep for both tiers, so hit/miss latency at a given
// num_keys is directly comparable across memtable and L1.
const NUM_KEYS_SWEEP: [u64; 4] = [1_000, 10_000, 50_000, 100_000];

// Capacity for the memtable-tier store: comfortably above the largest sweep
// value so the whole sweep stays inside one active memtable (default
// capacity's 95% flush threshold would otherwise seal partway through the
// 100_000 case — see module header).
const MEMTABLE_TIER_CAPACITY: usize = 200_000;

/// Populate a fresh store with `num_keys` keys at even indices, then
/// shutdown+reopen so the active memtable is flushed and compacted into L1.
/// The returned store serves every key from the on-disk SSTable.
fn populated_l1_store(num_keys: u64) -> (Db, TempDir) {
    let temp = bench_tempdir();
    let value = vec![0x5Au8; VALUE_SIZE];
    {
        let store = open_store(temp.path());
        for seq in 0..num_keys {
            store.put(&make_key("k:", seq * 2), &value).unwrap();
        }
        store.compact().unwrap();
        store.shutdown().unwrap();
    }
    // Clean shutdown marked the WAL entries persisted, so recovery does not
    // replay them into the memtable — all keys are served from the L1 SSTable.
    (open_store(temp.path()), temp)
}

/// Populate a fresh store with `num_keys` keys at even indices and leave them
/// resident in the active memtable — no flush, no compaction, no reopen.
fn populated_memtable_store(num_keys: u64) -> (Db, TempDir) {
    let temp = bench_tempdir();
    let value = vec![0x5Au8; VALUE_SIZE];
    let store = open_store_with_capacity(temp.path(), MEMTABLE_TIER_CAPACITY);
    for seq in 0..num_keys {
        store.put(&make_key("k:", seq * 2), &value).unwrap();
    }
    (store, temp)
}

/// Populate a store whose working set is split across both tiers: `num_keys`
/// "old:" keys written first and pushed to L1 (compact + shutdown + reopen),
/// then `num_keys` "new:" keys written afterward that stay resident in the
/// freshly-reopened active memtable. No further writes happen after this —
/// the resulting store approximates read-only steady state: some data has
/// aged onto disk, some is still hot in memory.
fn populated_mixed_store(num_keys: u64) -> (Db, TempDir) {
    let temp = bench_tempdir();
    let value = vec![0x5Au8; VALUE_SIZE];
    {
        let store = open_store(temp.path());
        for seq in 0..num_keys {
            store.put(&make_key("old:", seq * 2), &value).unwrap();
        }
        store.compact().unwrap();
        store.shutdown().unwrap();
    }
    let store = open_store_with_capacity(temp.path(), MEMTABLE_TIER_CAPACITY);
    for seq in 0..num_keys {
        store.put(&make_key("new:", seq * 2), &value).unwrap();
    }
    (store, temp)
}

fn bench_l1_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("sstable_lookup");
    group.measurement_time(Duration::from_secs(10));

    // NOTE: sizes are capped at 100k because `put` fsyncs the WAL on every write
    // (deliberate per-write durability, no group commit). At ~6 ms/fsync on WSL2
    // that's ~6 ms/key, so populate is fsync-bound: ~10 min at 100k, ~1.9 h at 1M
    // (the apparent "hang" we first saw was just this, not a deadlock). 1k-100k
    // is enough range to see the lookup's real shape: latency is roughly FLAT
    // across this range (bloom filter + sparse index bound the scan — see the
    // module header), not O(N). To probe larger datasets, populate via the
    // async no-WAL path (AsyncDb::put_no_wal).
    for num_keys in NUM_KEYS_SWEEP {
        let (store, _temp) = populated_l1_store(num_keys);

        let mut idx: u64 = 0;
        group.bench_with_input(BenchmarkId::new("l1_hit", num_keys), &num_keys, |b, &n| {
            b.iter(|| {
                idx = idx.wrapping_add(104_729) % n;
                black_box(store.get(&make_key("k:", idx * 2))).unwrap()
            });
        });

        let mut midx: u64 = 0;
        group.bench_with_input(BenchmarkId::new("l1_miss", num_keys), &num_keys, |b, &n| {
            b.iter(|| {
                midx = midx.wrapping_add(104_729) % n;
                black_box(store.get(&make_key("k:", midx * 2 + 1))).unwrap()
            });
        });

        let _ = store.shutdown();
    }
    group.finish();
}

fn bench_memtable_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("sstable_lookup");
    group.measurement_time(Duration::from_secs(10));

    for num_keys in NUM_KEYS_SWEEP {
        let (store, _temp) = populated_memtable_store(num_keys);

        let mut idx: u64 = 0;
        group.bench_with_input(BenchmarkId::new("memtable_hit", num_keys), &num_keys, |b, &n| {
            b.iter(|| {
                idx = idx.wrapping_add(104_729) % n;
                black_box(store.get(&make_key("k:", idx * 2))).unwrap()
            });
        });

        let mut midx: u64 = 0;
        group.bench_with_input(BenchmarkId::new("memtable_miss", num_keys), &num_keys, |b, &n| {
            b.iter(|| {
                midx = midx.wrapping_add(104_729) % n;
                black_box(store.get(&make_key("k:", midx * 2 + 1))).unwrap()
            });
        });

        let _ = store.shutdown();
    }
    group.finish();
}

/// Read-only, no-concurrent-write lookup against a store split across both
/// tiers (see `populated_mixed_store`). Each iteration alternates between an
/// L1-resident ("old:") key and a memtable-resident ("new:") key, so the
/// reported latency is the blended cost of a 50/50 tier mix — the shape a
/// real read stream sees against a steady-state store, as opposed to the
/// pure-tier extremes `bench_l1_lookup`/`bench_memtable_lookup` measure.
fn bench_mixed_tier_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("sstable_lookup");
    group.measurement_time(Duration::from_secs(10));

    for num_keys in NUM_KEYS_SWEEP {
        let (store, _temp) = populated_mixed_store(num_keys);

        let mut idx: u64 = 0;
        let mut tier_toggle = false;
        group.bench_with_input(BenchmarkId::new("mixed_hit", num_keys), &num_keys, |b, &n| {
            b.iter(|| {
                idx = idx.wrapping_add(104_729) % n;
                tier_toggle = !tier_toggle;
                let key = if tier_toggle { make_key("old:", idx * 2) } else { make_key("new:", idx * 2) };
                black_box(store.get(&key)).unwrap()
            });
        });

        let mut midx: u64 = 0;
        let mut miss_tier_toggle = false;
        group.bench_with_input(BenchmarkId::new("mixed_miss", num_keys), &num_keys, |b, &n| {
            b.iter(|| {
                midx = midx.wrapping_add(104_729) % n;
                miss_tier_toggle = !miss_tier_toggle;
                let key = if miss_tier_toggle {
                    make_key("old:", midx * 2 + 1)
                } else {
                    make_key("new:", midx * 2 + 1)
                };
                black_box(store.get(&key)).unwrap()
            });
        });

        let _ = store.shutdown();
    }
    group.finish();
}

criterion_group!(
    name    = sstable_benches;
    config  = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = bench_memtable_lookup, bench_l1_lookup, bench_mixed_tier_lookup
);
criterion_main!(sstable_benches);
