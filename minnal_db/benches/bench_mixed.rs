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
use std::time::Duration;

/// Reads and writes interleaved on one store per case, in the steady state:
/// opened once, its WAL already on zero-filled segments (`steady_store`), with
/// `SEED_KEYS` keys to read, pushed to L1 for the `l1` tier. Writes go to fresh
/// keys and accumulate across samples, as they would in a running database.
///
/// The store is opened once per case, not per iteration: criterion calls an
/// `iter_batched*` setup once per iteration, which would make every operation
/// the first one into a new database (and, with the operation counter in the
/// setup, always a write).
fn bench_mixed_workload(c: &mut Criterion) {
    const SEED_KEYS: u64 = 5_000;
    const VALUE_SIZE: usize = 512;
    let mut group = c.benchmark_group("mixed");
    group.measurement_time(Duration::from_secs(12));

    // (label, write_every_n_ops)
    // write_every = 5 → 20% writes; write_every = 2 → 50% writes
    let cases: &[(&str, u64)] = &[("80pct_read_20pct_write", 5), ("50pct_read_50pct_write", 2)];

    for &(label, write_every) in cases {
        let value = vec![0u8; VALUE_SIZE];

        for tier in TIERS {
            let temp = bench_tempdir();
            let mut steady = steady_store(temp.path(), VALUE_SIZE);
            let mut store = steady.0.take().unwrap();
            for seq in 0..SEED_KEYS {
                store.put(&make_key("mix:", seq), &value).unwrap();
            }
            if tier == "l1" {
                store = push_to_l1(store, temp.path());
            }
            let store = AutoCloseStore(Some(store));

            group.throughput(Throughput::Elements(1));
            let bench_label = format!("{tier}_{label}");
            group.bench_function(&bench_label, |b| {
                let (mut op, mut write_seq, mut read_idx) = (0u64, SEED_KEYS, 0u64);
                b.iter(|| {
                    op += 1;
                    if op % write_every == 0 {
                        write_seq += 1;
                        black_box(store.store().put(&make_key("mix:", write_seq), &value)).unwrap();
                    } else {
                        read_idx = (read_idx + 104_729) % SEED_KEYS;
                        black_box(store.store().get(&make_key("mix:", read_idx))).unwrap();
                    }
                });
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
