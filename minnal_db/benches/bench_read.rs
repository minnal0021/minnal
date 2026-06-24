// Read throughput benchmarks
//   cargo bench --bench bench_read

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;

fn bench_get_throughput(c: &mut Criterion) {
    const NUM_KEYS: u64 = 2_000;
    let mut group = c.benchmark_group("read/get");
    group.measurement_time(Duration::from_secs(10));

    for value_size in [128usize, 4_096, 65_536] {
        let temp = bench_tempdir();
        let store = open_store(temp.path());
        let value = vec![0xCDu8; value_size];

        // Pre-populate
        for seq in 0..NUM_KEYS {
            store.put(&make_key("get:", seq), &value).unwrap();
        }

        group.throughput(Throughput::Bytes(value_size as u64));
        let mut idx: u64 = 0;
        group.bench_with_input(BenchmarkId::new("value_size", format!("{value_size}B")), &value_size, |b, _| {
            b.iter(|| {
                // Pseudo-random walk using a large prime step
                idx = idx.wrapping_add(104_729) % NUM_KEYS;
                let key = make_key("get:", idx);
                black_box(store.get(&key)).unwrap()
            });
        });
    }
    group.finish();
}

fn bench_get_miss(c: &mut Criterion) {
    let temp = bench_tempdir();
    let store = open_store(temp.path());
    // Populate "hit" keys
    for seq in 0..1_000u64 {
        store.put(&make_key("real:", seq), b"v").unwrap();
    }

    let mut idx: u64 = 0;
    c.bench_function("read/get_miss", |b| {
        b.iter(|| {
            idx = idx.wrapping_add(104_729);
            let key = make_key("miss:", idx); // "miss:" prefix never inserted
            black_box(store.get(&key)).unwrap()
        });
    });
}

criterion_group!(
    name    = read_benches;
    config  = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = bench_get_throughput, bench_get_miss
);
criterion_main!(read_benches);
