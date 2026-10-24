// Write throughput benchmarks
//   cargo bench --bench bench_write

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;

/// One writer's `put` of a fresh key, in the steady state: one store per value
/// size, its WAL already on zero-filled segments (see `steady_store`).
fn bench_put_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("write/put");
    group.measurement_time(Duration::from_secs(10));

    for value_size in [128usize, 4_096, 65_536] {
        let value = vec![0xABu8; value_size];
        let temp = bench_tempdir();
        let store = steady_store(temp.path(), value_size);

        group.throughput(Throughput::Bytes(value_size as u64));
        group.bench_with_input(BenchmarkId::new("value_size", format!("{value_size}B")), &value_size, |b, _| {
            let mut seq = 0u64;
            b.iter(|| {
                seq += 1;
                black_box(store.store().put(&make_key("bench:", seq), &value)).unwrap();
            });
        });
    }
    group.finish();
}

/// The first `put` into a brand-new database, on purpose: a new store per
/// iteration (criterion calls the setup once per iteration), so every write
/// lands in new file blocks. Startup cost, not the steady state.
fn bench_first_put_into_new_db(c: &mut Criterion) {
    let mut group = c.benchmark_group("write/first_put_into_new_db");
    group.measurement_time(Duration::from_secs(10));
    let value = vec![0xABu8; 128];
    group.bench_function("128B", |b| {
        b.iter_batched_ref(
            || {
                let temp = bench_tempdir();
                let store = AutoCloseStore::open(temp.path());
                (store, temp)
            },
            |(store, _temp)| black_box(store.store().put(&make_key("bench:", 0), &value)).unwrap(),
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    name    = write_benches;
    config  = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = bench_put_throughput, bench_first_put_into_new_db
);
criterion_main!(write_benches);
