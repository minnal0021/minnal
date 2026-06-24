// Write throughput benchmarks
//   cargo bench --bench bench_write

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

fn bench_put_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("write/put");
    group.measurement_time(Duration::from_secs(10));

    const BATCH: usize = 2_000;

    for value_size in [128usize, 4_096, 65_536] {
        let value = vec![0xABu8; value_size];

        group.throughput(Throughput::Bytes(value_size as u64));
        group.bench_with_input(BenchmarkId::new("value_size", format!("{value_size}B")), &value_size, |b, _| {
            b.iter_batched_ref(
                || {
                    let temp = bench_tempdir();
                    let store = AutoCloseStore::open(temp.path());
                    (store, AtomicU64::new(0), temp)
                },
                |(store, counter, _temp)| {
                    let seq = counter.fetch_add(1, Ordering::Relaxed);
                    let key = make_key("bench:", seq);
                    black_box(store.store().put(&key, &value)).unwrap();
                },
                criterion::BatchSize::NumIterations(BATCH as u64),
            );
        });
    }
    group.finish();
}

criterion_group!(
    name    = write_benches;
    config  = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = bench_put_throughput
);
criterion_main!(write_benches);
