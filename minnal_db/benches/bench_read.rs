// Read throughput benchmarks — active memtable vs. on-disk L1 SSTable.
//   cargo bench --bench bench_read
//
// Every case is measured on both tiers (`memtable_*` / `l1_*`, see `TIERS`
// in common.rs). Memtable-only latency is cheap but not representative of
// steady-state reads once data ages out of the memtable and onto disk — see
// `bench_sstable_lookup.rs` for a deeper, key-count-swept comparison of the
// two tiers in isolation. This file instead holds the two tiers alongside
// bench_read's original axis (value size), so the "reads are misleadingly
// fast if you only ever measure the memtable" point is visible on every
// case here, not just in one dedicated benchmark.

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;

const NUM_KEYS: u64 = 2_000;

fn bench_get_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("read/get");
    group.measurement_time(Duration::from_secs(10));

    for value_size in [128usize, 4_096, 65_536] {
        let value = vec![0xCDu8; value_size];

        for tier in TIERS {
            let temp = bench_tempdir();
            let mut store = open_store(temp.path());

            for seq in 0..NUM_KEYS {
                store.put(&make_key("get:", seq), &value).unwrap();
            }
            if tier == "l1" {
                store = push_to_l1(store, temp.path());
            }

            group.throughput(Throughput::Bytes(value_size as u64));
            let mut idx: u64 = 0;
            let label = format!("{tier}_value_size");
            group.bench_with_input(BenchmarkId::new(label, format!("{value_size}B")), &value_size, |b, _| {
                b.iter(|| {
                    // Pseudo-random walk using a large prime step
                    idx = idx.wrapping_add(104_729) % NUM_KEYS;
                    let key = make_key("get:", idx);
                    black_box(store.get(&key)).unwrap()
                });
            });
        }
    }
    group.finish();
}

fn bench_get_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("read/get_miss");

    for tier in TIERS {
        let temp = bench_tempdir();
        let mut store = open_store(temp.path());
        // Populate "hit" keys
        for seq in 0..1_000u64 {
            store.put(&make_key("real:", seq), b"v").unwrap();
        }
        if tier == "l1" {
            store = push_to_l1(store, temp.path());
        }

        let mut idx: u64 = 0;
        group.bench_function(tier, |b| {
            b.iter(|| {
                idx = idx.wrapping_add(104_729); // "miss:" prefix never inserted
                let key = make_key("miss:", idx);
                black_box(store.get(&key)).unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(
    name    = read_benches;
    config  = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = bench_get_throughput, bench_get_miss
);
criterion_main!(read_benches);
