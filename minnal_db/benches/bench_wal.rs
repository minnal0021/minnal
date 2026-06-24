// WAL performance benchmarks
//   cargo bench --bench bench_wal
//
// Covers:
//   a) WAL append latency with fsync (crash-safe path)
//   b) WAL append latency without fsync (throughput ceiling)
//   c) WAL entry serialization (rkyv to_bytes / from_bytes roundtrip)
//   d) WAL scan throughput (models recovery cost)
//   e) End-to-end throughput of the full put path (WAL + value log + LSM)
//   f) WAL append with namespace_id tagging

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use minnal_db::db::wal::{Wal, WalEntry};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// WAL append with fsync (the default crash-safe path).
///
/// Uses `iter_batched_ref` to create a fresh WAL every `BATCH` iterations,
/// preventing disk exhaustion during long measurement runs. The batch size
/// is chosen so that setup cost is amortised while disk usage stays bounded
/// (worst case: BATCH * 65 KB ≈ 64 MB per batch).
fn bench_wal_append_fsync(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal/append_fsync");
    group.measurement_time(Duration::from_secs(10));

    const BATCH: usize = 1_000;

    for value_size in [128usize, 4_096, 65_536] {
        let value = vec![0xABu8; value_size];

        group.throughput(Throughput::Bytes(value_size as u64));
        group.bench_with_input(BenchmarkId::new("value_size", format!("{value_size}B")), &value_size, |b, _| {
            b.iter_batched_ref(
                || {
                    let temp = bench_tempdir();
                    let wal_path = temp.path().join("wal.log");
                    let wal = Wal::open(&wal_path).unwrap();
                    (wal, 0u64, temp)
                },
                |(wal, tail, _temp)| {
                    let entry = WalEntry::new_upsert(b"bench_key_00001".to_vec(), value.clone());
                    black_box(wal.append_entry(&entry, tail, true)).unwrap();
                },
                criterion::BatchSize::NumIterations(BATCH as u64),
            );
        });
    }
    group.finish();
}

/// WAL append without fsync — shows the throughput ceiling when durability
/// guarantees are relaxed (e.g. for batched-sync configurations).
fn bench_wal_append_no_fsync(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal/append_no_fsync");
    group.measurement_time(Duration::from_secs(10));

    const BATCH: usize = 5_000;

    for value_size in [128usize, 4_096, 65_536] {
        let value = vec![0xABu8; value_size];

        group.throughput(Throughput::Bytes(value_size as u64));
        group.bench_with_input(BenchmarkId::new("value_size", format!("{value_size}B")), &value_size, |b, _| {
            b.iter_batched_ref(
                || {
                    let temp = bench_tempdir();
                    let wal_path = temp.path().join("wal.log");
                    let wal = Wal::open(&wal_path).unwrap();
                    (wal, 0u64, temp)
                },
                |(wal, tail, _temp)| {
                    let entry = WalEntry::new_upsert(b"bench_key_00001".to_vec(), value.clone());
                    black_box(wal.append_entry(&entry, tail, false)).unwrap();
                },
                criterion::BatchSize::NumIterations(BATCH as u64),
            );
        });
    }
    group.finish();
}

fn bench_wal_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal/serialization");
    group.measurement_time(Duration::from_secs(10));

    for value_size in [128usize, 4_096, 65_536] {
        let value = vec![0xCDu8; value_size];
        let entry = WalEntry::new_upsert(b"bench_key_00001".to_vec(), value);

        group.throughput(Throughput::Bytes(value_size as u64));
        group.bench_with_input(BenchmarkId::new("to_bytes", format!("{value_size}B")), &value_size, |b, _| {
            b.iter(|| {
                black_box(entry.to_bytes()).unwrap();
            });
        });

        let bytes = entry.to_bytes().unwrap();
        group.bench_with_input(BenchmarkId::new("from_bytes", format!("{value_size}B")), &value_size, |b, _| {
            b.iter(|| {
                black_box(WalEntry::from_bytes(&bytes)).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_wal_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal/scan");
    group.measurement_time(Duration::from_secs(10));

    for &(num_entries, value_size) in &[(1_000u64, 128usize), (1_000, 4_096), (5_000, 128)] {
        let temp = bench_tempdir();
        let wal = Wal::open(temp.path().join("wal.log")).expect("failed to open WAL");
        let value = vec![0xEFu8; value_size];
        let mut tail = 0u64;

        // Pre-populate
        for i in 0..num_entries {
            let key = format!("scan_key_{:010}", i).into_bytes();
            let entry = WalEntry::new_upsert(key, value.clone());
            wal.append_entry(&entry, &mut tail, false).unwrap();
        }
        wal.sync().unwrap();

        let label = format!("{num_entries}_entries_{}B", value_size);
        group.throughput(Throughput::Elements(num_entries));
        group.bench_function(&label, |b| {
            b.iter(|| {
                let entries = wal.scan_entries(0, tail).unwrap();
                black_box(entries.len())
            });
        });
    }
    group.finish();
}

/// Measures end-to-end throughput of the full write path: `Db::put`
/// (WAL + value log + LSM).
///
/// The WAL's share of this cost (serialization + pwrite + fsync) is measured
/// in isolation by `bench_wal_append_fsync` / `bench_wal_append_no_fsync`.
fn bench_full_put_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal/full_put_throughput");
    group.measurement_time(Duration::from_secs(10));

    let value_size = 512usize;
    let value = vec![0xABu8; value_size];
    const BATCH: usize = 1_000;

    // Full path: Db::put (WAL + value log + LSM)
    group.throughput(Throughput::Elements(1));
    group.bench_function("db_put_with_wal", |b| {
        b.iter_batched_ref(
            || {
                let temp = bench_tempdir();
                let store = AutoCloseStore::open(temp.path());
                (store, AtomicU64::new(0), temp)
            },
            |(store, counter, _temp)| {
                let seq = counter.fetch_add(1, Ordering::Relaxed);
                let key = make_key("wal_put:", seq);
                black_box(store.store().put(&key, &value)).unwrap();
            },
            criterion::BatchSize::NumIterations(BATCH as u64),
        );
    });

    group.finish();
}

/// Measures WAL append with namespace_id tagging to detect
/// any overhead from the namespace field.
fn bench_wal_namespace_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal/namespace_overhead");
    group.measurement_time(Duration::from_secs(10));

    const BATCH: usize = 1_000;
    let value = vec![0xABu8; 512];

    // Append with default namespace (id=0)
    {
        let value = value.clone();
        group.bench_function("ns_id_0", |b| {
            b.iter_batched_ref(
                || {
                    let temp = bench_tempdir();
                    let wal = Wal::open(temp.path().join("wal.log")).unwrap();
                    (wal, 0u64, temp)
                },
                |(wal, tail, _temp)| {
                    let entry = WalEntry::new_upsert_ns(0, b"key_00001".to_vec(), value.clone());
                    black_box(wal.append_entry(&entry, tail, true)).unwrap();
                },
                criterion::BatchSize::NumIterations(BATCH as u64),
            );
        });
    }

    // Append with a non-default namespace (id=42)
    {
        let value = value.clone();
        group.bench_function("ns_id_42", |b| {
            b.iter_batched_ref(
                || {
                    let temp = bench_tempdir();
                    let wal = Wal::open(temp.path().join("wal.log")).unwrap();
                    (wal, 0u64, temp)
                },
                |(wal, tail, _temp)| {
                    let entry = WalEntry::new_upsert_ns(42, b"key_00001".to_vec(), value.clone());
                    black_box(wal.append_entry(&entry, tail, true)).unwrap();
                },
                criterion::BatchSize::NumIterations(BATCH as u64),
            );
        });
    }

    group.finish();
}

criterion_group!(
    name    = wal_benches;
    config  = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = bench_wal_append_fsync,
              bench_wal_append_no_fsync,
              bench_wal_serialization,
              bench_wal_scan,
              bench_full_put_throughput,
              bench_wal_namespace_overhead
);
criterion_main!(wal_benches);
