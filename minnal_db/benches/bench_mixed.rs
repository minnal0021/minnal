// Mixed read/write workload benchmarks
//   cargo bench --bench bench_mixed
//
// Each case is measured on both tiers for the pre-seeded, read-only working
// set (`memtable_*` / `l1_*`, see `TIERS` in common.rs) — writes always land
// in the active memtable regardless of tier (there's no "write directly to
// L1" path), only the read side's tier varies. Memtable-only latency is
// cheap but not representative of steady-state reads once data ages onto
// disk.

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
    // `iter_batched_ref` reseeds SEED_KEYS fresh (fsync'd) keys before every
    // measured batch, since the routine's writes must not leak between
    // samples — at ~2.3 ms/fsync that's ~11.5s of setup alone per sample,
    // before the L1 tier's extra compact+shutdown+reopen. At the default
    // 100 samples this took ~20-45 minutes per case for the memtable tier
    // alone; a reduced sample size keeps total runtime for both tiers x
    // both mix ratios in the tens-of-minutes range rather than hours.
    group.sample_size(20);

    // (label, write_every_n_ops)
    // write_every = 5 → 20% writes; write_every = 2 → 50% writes
    let cases: &[(&str, u64)] = &[("80pct_read_20pct_write", 5), ("50pct_read_50pct_write", 2)];

    for &(label, write_every) in cases {
        let value = vec![0u8; VALUE_SIZE];

        for tier in TIERS {
            group.throughput(Throughput::Elements(1));
            let bench_label = format!("{tier}_{label}");
            group.bench_function(&bench_label, |b| {
                b.iter_batched_ref(
                    || {
                        let temp = bench_tempdir();
                        let mut store = open_store(temp.path());
                        // Pre-seed readable keys
                        for seq in 0..SEED_KEYS {
                            store.put(&make_key("mix:", seq), &vec![0u8; VALUE_SIZE]).unwrap();
                        }
                        if tier == "l1" {
                            store = push_to_l1(store, temp.path());
                        }
                        (AutoCloseStore(Some(store)), AtomicU64::new(SEED_KEYS), AtomicU64::new(0), AtomicU64::new(0), temp)
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
    }
    group.finish();
}

criterion_group!(
    name    = mixed_benches;
    config  = Criterion::default().measurement_time(Duration::from_secs(12));
    targets = bench_mixed_workload
);
criterion_main!(mixed_benches);
