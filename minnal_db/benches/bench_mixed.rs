// Mixed read/write workload benchmarks
//   cargo bench --bench bench_mixed

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

fn bench_mixed_workload(c: &mut Criterion) {
    const SEED_KEYS: u64 = 5_000;
    const VALUE_SIZE: usize = 512;
    const BATCH: usize = 2_000;
    let mut group = c.benchmark_group("mixed");
    group.measurement_time(Duration::from_secs(12));

    // (label, write_every_n_ops)
    // write_every = 5 → 20% writes; write_every = 2 → 50% writes
    let cases: &[(&str, u64)] = &[("80pct_read_20pct_write", 5), ("50pct_read_50pct_write", 2)];

    for &(label, write_every) in cases {
        let value = vec![0u8; VALUE_SIZE];

        group.throughput(Throughput::Elements(1));
        group.bench_function(label, |b| {
            b.iter_batched_ref(
                || {
                    let temp = bench_tempdir();
                    let store = AutoCloseStore::open(temp.path());
                    // Pre-seed readable keys
                    for seq in 0..SEED_KEYS {
                        store.store().put(&make_key("mix:", seq), &vec![0u8; VALUE_SIZE]).unwrap();
                    }
                    (store, AtomicU64::new(SEED_KEYS), AtomicU64::new(0), AtomicU64::new(0), temp)
                },
                |(store, write_counter, read_idx, op_counter, _temp)| {
                    let op = op_counter.fetch_add(1, Ordering::Relaxed);
                    if op % write_every == 0 {
                        let seq = write_counter.fetch_add(1, Ordering::Relaxed);
                        let key = make_key("mix:", seq);
                        black_box(store.store().put(&key, &value)).unwrap();
                    } else {
                        let idx = read_idx.fetch_add(104_729, Ordering::Relaxed) % SEED_KEYS;
                        let key = make_key("mix:", idx);
                        black_box(store.store().get(&key)).unwrap();
                    }
                },
                criterion::BatchSize::NumIterations(BATCH as u64),
            );
        });
    }
    group.finish();
}

criterion_group!(
    name    = mixed_benches;
    config  = Criterion::default().measurement_time(Duration::from_secs(12));
    targets = bench_mixed_workload
);
criterion_main!(mixed_benches);
