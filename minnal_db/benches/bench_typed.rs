// Typed API (rkyv ser/de) benchmarks
//   cargo bench --bench bench_typed
//
// Measures the overhead of the *_typed() convenience layer which adds rkyv
// serialization on write and deserialization on read, compared to raw bytes.
//
// Read-side cases (get/iter/keys/range/scan_prefix) are measured on both the
// active memtable and the on-disk L1 SSTable tier (`TIERS` in common.rs) —
// memtable-only numbers are cheap but not representative of steady-state
// reads once data ages onto disk. Write-side cases (put/delete) are NOT
// tiered: a write always lands in the active memtable regardless of what's
// already on disk, so tier doesn't apply to their cost the same way.

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use minnal_db::{
    Db,
    rkyv_derives::{Archive, Deserialize, Serialize},
};
use std::hint::black_box;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Wrapper that calls `Db::shutdown()` on drop to release file handles promptly.
/// Without this, `iter_batched_ref` accumulates open SSTable/value-log fds
/// across batches and eventually hits the OS open-file limit.
struct AutoCloseDb(Option<Db>);

impl AutoCloseDb {
    fn open(path: &Path) -> Self {
        Self(Some(Db::open_with_config(path, bench_config()).unwrap()))
    }

    fn db(&self) -> &Db {
        self.0.as_ref().unwrap()
    }
}

impl Drop for AutoCloseDb {
    fn drop(&mut self) {
        if let Some(db) = self.0.take() {
            let _ = db.shutdown();
        }
    }
}

impl AutoCloseDb {
    /// Compact + shutdown + reopen so subsequent reads hit the on-disk L1
    /// SSTable rather than the active memtable. Consumes `self` to match
    /// `push_to_l1`'s ownership shape (the old handle is closed as part of
    /// the push, not left dangling).
    fn push_to_l1(mut self, dir: &Path) -> Self {
        let db = self.0.take().expect("AutoCloseDb already closed");
        Self(Some(push_to_l1(db, dir)))
    }
}

// `prefix` is declared FIRST: `put_typed` uses the raw rkyv-archived bytes of
// the key directly as the on-disk key, and archived struct layout follows
// declaration order — so `prefix` must lead for `scan_prefix_typed(b"bench_kk")`
// to actually match anything. With `id` first (the original field order),
// the archived key started with `id`'s bytes and "bench_kk" landed at
// offset 8, so every prefix scan matched zero entries and reported a
// near-instant, meaningless latency instead of a real scan cost.
#[derive(Archive, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[rkyv(compare(PartialEq), derive(Debug))]
struct BenchKey {
    prefix: [u8; 8],
    id: u64,
}

#[derive(Archive, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[rkyv(compare(PartialEq), derive(Debug))]
struct BenchValue {
    payload: Vec<u8>,
    version: u32,
}

impl BenchKey {
    fn new(seq: u64) -> Self {
        let mut prefix = [0u8; 8];
        prefix.copy_from_slice(b"bench_kk");
        Self { prefix, id: seq }
    }
}

impl BenchValue {
    fn new(size: usize) -> Self {
        Self {
            payload: vec![0xABu8; size],
            version: 1,
        }
    }
}

fn bench_put_typed(c: &mut Criterion) {
    let mut group = c.benchmark_group("typed/put");
    group.measurement_time(Duration::from_secs(10));

    const BATCH: usize = 2_000;

    for value_size in [128usize, 4_096] {
        let value = BenchValue::new(value_size);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("value_size", format!("{value_size}B")), &value_size, |b, _| {
            b.iter_batched_ref(
                || {
                    let temp = bench_tempdir();
                    let db = AutoCloseDb::open(temp.path());
                    (db, AtomicU64::new(0), temp)
                },
                |(db, counter, _temp)| {
                    let seq = counter.fetch_add(1, Ordering::Relaxed);
                    let key = BenchKey::new(seq);
                    black_box(db.db().put_typed(&key, &value)).unwrap();
                },
                criterion::BatchSize::NumIterations(BATCH as u64),
            );
        });
    }
    group.finish();
}

fn bench_get_typed(c: &mut Criterion) {
    const NUM_KEYS: u64 = 2_000;
    let mut group = c.benchmark_group("typed/get");
    group.measurement_time(Duration::from_secs(10));

    for value_size in [128usize, 4_096] {
        let value = BenchValue::new(value_size);

        for tier in TIERS {
            let temp = bench_tempdir();
            let mut db = AutoCloseDb::open(temp.path());

            for seq in 0..NUM_KEYS {
                db.db().put_typed(&BenchKey::new(seq), &value).unwrap();
            }
            if tier == "l1" {
                db = db.push_to_l1(temp.path());
            }

            group.throughput(Throughput::Elements(1));
            let mut idx: u64 = 0;
            let label = format!("{tier}_value_size");
            group.bench_with_input(BenchmarkId::new(label, format!("{value_size}B")), &value_size, |b, _| {
                b.iter(|| {
                    idx = idx.wrapping_add(104_729) % NUM_KEYS;
                    let key = BenchKey::new(idx);
                    black_box(db.db().get_typed::<BenchKey, BenchValue>(&key)).unwrap()
                });
            });
        }
    }
    group.finish();
}

fn bench_delete_typed(c: &mut Criterion) {
    let mut group = c.benchmark_group("typed/delete");
    group.measurement_time(Duration::from_secs(10));

    const BATCH: usize = 2_000;

    group.throughput(Throughput::Elements(1));
    group.bench_function("delete", |b| {
        b.iter_batched_ref(
            || {
                let temp = bench_tempdir();
                let db = AutoCloseDb::open(temp.path());
                let value = BenchValue::new(128);
                for seq in 0..BATCH as u64 {
                    db.db().put_typed(&BenchKey::new(seq), &value).unwrap();
                }
                (db, AtomicU64::new(0), temp)
            },
            |(db, counter, _temp)| {
                let seq = counter.fetch_add(1, Ordering::Relaxed);
                let key = BenchKey::new(seq);
                black_box(db.db().delete_typed(&key)).unwrap();
            },
            criterion::BatchSize::NumIterations(BATCH as u64),
        );
    });
    group.finish();
}

fn bench_iter_typed(c: &mut Criterion) {
    let mut group = c.benchmark_group("typed/iter");
    group.measurement_time(Duration::from_secs(10));

    for num_keys in [100u64, 1_000] {
        let value = BenchValue::new(128);

        for tier in TIERS {
            let temp = bench_tempdir();
            let mut db = AutoCloseDb::open(temp.path());

            for seq in 0..num_keys {
                db.db().put_typed(&BenchKey::new(seq), &value).unwrap();
            }
            if tier == "l1" {
                db = db.push_to_l1(temp.path());
            }

            group.throughput(Throughput::Elements(num_keys));
            let label = format!("{tier}_keys");
            group.bench_with_input(BenchmarkId::new(label, num_keys), &num_keys, |b, _| {
                b.iter(|| {
                    let pairs = db.db().iter_typed::<BenchKey, BenchValue>().unwrap();
                    black_box(pairs.len())
                });
            });
        }
    }
    group.finish();
}

fn bench_keys_typed(c: &mut Criterion) {
    let mut group = c.benchmark_group("typed/keys");
    group.measurement_time(Duration::from_secs(10));

    for num_keys in [100u64, 1_000] {
        let value = BenchValue::new(128);

        for tier in TIERS {
            let temp = bench_tempdir();
            let mut db = AutoCloseDb::open(temp.path());

            for seq in 0..num_keys {
                db.db().put_typed(&BenchKey::new(seq), &value).unwrap();
            }
            if tier == "l1" {
                db = db.push_to_l1(temp.path());
            }

            group.throughput(Throughput::Elements(num_keys));
            let label = format!("{tier}_keys");
            group.bench_with_input(BenchmarkId::new(label, num_keys), &num_keys, |b, _| {
                b.iter(|| {
                    let keys = db.db().keys_typed::<BenchKey>().unwrap();
                    black_box(keys.len())
                });
            });
        }
    }
    group.finish();
}

fn bench_range_typed(c: &mut Criterion) {
    const TOTAL_KEYS: u64 = 2_000;
    let mut group = c.benchmark_group("typed/range");
    group.measurement_time(Duration::from_secs(10));

    for tier in TIERS {
        let temp = bench_tempdir();
        let mut db = AutoCloseDb::open(temp.path());
        let value = BenchValue::new(128);

        for seq in 0..TOTAL_KEYS {
            db.db().put_typed(&BenchKey::new(seq), &value).unwrap();
        }
        if tier == "l1" {
            db = db.push_to_l1(temp.path());
        }

        for result_count in [100u64, 500, 1_000] {
            let start = BenchKey::new(0);
            let end = BenchKey::new(result_count);

            group.throughput(Throughput::Elements(result_count));
            let label = format!("{tier}_result_count");
            group.bench_with_input(BenchmarkId::new(label, result_count), &result_count, |b, _| {
                b.iter(|| {
                    let pairs = db.db().range_typed::<BenchKey, BenchValue>(&start, Some(&end)).unwrap();
                    black_box(pairs.len())
                });
            });
        }
    }
    group.finish();
}

fn bench_scan_prefix_typed(c: &mut Criterion) {
    let mut group = c.benchmark_group("typed/scan_prefix");
    group.measurement_time(Duration::from_secs(10));

    for num_keys in [100u64, 1_000] {
        let value = BenchValue::new(128);

        for tier in TIERS {
            let temp = bench_tempdir();
            let mut db = AutoCloseDb::open(temp.path());

            for seq in 0..num_keys {
                db.db().put_typed(&BenchKey::new(seq), &value).unwrap();
            }
            if tier == "l1" {
                db = db.push_to_l1(temp.path());
            }

            group.throughput(Throughput::Elements(num_keys));
            let label = format!("{tier}_keys");
            group.bench_with_input(BenchmarkId::new(label, num_keys), &num_keys, |b, _| {
                b.iter(|| {
                    let pairs = db.db().scan_prefix_typed::<BenchKey, BenchValue>(b"bench_kk").unwrap();
                    black_box(pairs.len())
                });
            });
        }
    }
    group.finish();
}

criterion_group!(
    name    = typed_benches;
    config  = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = bench_put_typed,
              bench_get_typed,
              bench_delete_typed,
              bench_iter_typed,
              bench_keys_typed,
              bench_range_typed,
              bench_scan_prefix_typed
);
criterion_main!(typed_benches);
