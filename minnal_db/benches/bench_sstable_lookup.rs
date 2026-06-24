// SSTable point-lookup benchmarks
//   cargo bench --bench bench_sstable_lookup
//
// PURPOSE: isolate the cost of `LSMTree::lookup_in_sstable_file`, the linear
// scan flagged by the review item "LSM point lookups scan SSTables linearly".
//
// `bench_read` does NOT exercise this path: its 2_000-key working set fits
// entirely in the memtable (capacity 100_000), so every read is a memtable hit.
// Here we push all keys down to the on-disk L1 SSTable via shutdown+reopen, so
// each `get()` runs the real linear scan. This is the BASELINE for evaluating
// the acceleration options (bloom filter / sparse block index / binary search)
// — see memory `project_lsm_lookup_acceleration`.
//
// Keys are written at EVEN indices and we query:
//   * "hit"  — an even key that exists,
//   * "miss" — an odd key that falls *between* two existing keys,
// so both stop the scan at roughly the key's sorted position (≈ rank/2 entries
// on average). If lookup time grows with `num_keys`, the scan is O(N) as flagged;
// a flat curve would mean reads are (wrongly) still hitting the memtable.

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use minnal_db::Db;
use std::hint::black_box;
use std::time::Duration;
use tempfile::TempDir;

const VALUE_SIZE: usize = 16; // small, so the SSTable scan dominates over value reads

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

fn bench_sstable_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("sstable_lookup");
    group.measurement_time(Duration::from_secs(10));

    // NOTE: sizes are capped at 100k because `put` fsyncs the WAL on every write
    // (deliberate per-write durability, no group commit). At ~6 ms/fsync on WSL2
    // that's ~6 ms/key, so populate is fsync-bound: ~10 min at 100k, ~1.9 h at 1M
    // (the apparent "hang" we first saw was just this, not a deadlock). 1k–100k
    // already shows the O(N) scan cleanly (10× keys ⇒ ~10× latency). To probe
    // larger datasets, populate via the async no-WAL path (AsyncDb::put_no_wal).
    for num_keys in [1_000u64, 10_000, 50_000, 100_000] {
        let (store, _temp) = populated_l1_store(num_keys);

        let mut idx: u64 = 0;
        group.bench_with_input(BenchmarkId::new("hit", num_keys), &num_keys, |b, &n| {
            b.iter(|| {
                idx = idx.wrapping_add(104_729) % n;
                black_box(store.get(&make_key("k:", idx * 2))).unwrap()
            });
        });

        let mut midx: u64 = 0;
        group.bench_with_input(BenchmarkId::new("miss", num_keys), &num_keys, |b, &n| {
            b.iter(|| {
                midx = midx.wrapping_add(104_729) % n;
                black_box(store.get(&make_key("k:", midx * 2 + 1))).unwrap()
            });
        });

        let _ = store.shutdown();
    }
    group.finish();
}

criterion_group!(
    name    = sstable_benches;
    config  = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = bench_sstable_lookup
);
criterion_main!(sstable_benches);
