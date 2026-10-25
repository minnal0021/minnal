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
use minnal_db::db::wal::{DEFAULT_SEGMENT_SIZE, Wal, WalEntry};
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Bytes of appends one WAL receives before the next is opened. Under half
/// the default 64 MiB segment (less the warm-up's first record), so no case
/// rotates mid-run or overlaps the background zero-fill of the next segment,
/// which starts once a segment is half full.
const WAL_BYTES_PER_RUN: usize = 24 * 1024 * 1024;

/// Open a WAL in `dir` and move it past its first segment, so the timed
/// appends land where a running database's do: in a segment zero-filled ahead
/// of use. A new log's first segment is sparse, so there a write that starts
/// a new 4 KiB block also pays for allocating it (a database's first writes;
/// `write/first_put_into_new_db` measures that). Returns the WAL and its tail.
fn warm_wal(dir: &std::path::Path) -> (Wal, u64) {
    let (wal, mut tail) = Wal::open_standalone(dir, DEFAULT_SEGMENT_SIZE).unwrap();
    let filler = WalEntry::new_upsert(b"warm".to_vec(), vec![0u8; 1 << 20]);
    while wal.segment_of(tail) == 0 {
        wal.append_entry(&filler, &mut tail).unwrap();
    }
    wal.commit(tail, Duration::ZERO).unwrap();
    (wal, tail)
}

/// Time `iters` appends of `make_entry()` with Criterion's `iter_custom`; with
/// `sync`, each append is committed (one fsync each: a lone writer's rounds).
///
/// Appends go into an already-open WAL, so the number is the append alone.
/// Every `per_wal` appends the WAL is dropped and a fresh one opened, outside
/// the timer, which bounds disk use on long runs. Opening a WAL per
/// *iteration* instead (`iter_batched_ref` with one WAL per input) makes the
/// untimed setup create thousands of files per batch; since creating a WAL
/// fsyncs its directory, that leaves journal work in flight that slows the
/// timed appends, and the unsynced case measured file creation, not appends.
fn time_appends(iters: u64, per_wal: u64, sync: bool, make_entry: impl Fn() -> WalEntry) -> Duration {
    let mut elapsed = Duration::ZERO;
    let mut done = 0;
    while done < iters {
        let n = per_wal.min(iters - done);
        let temp = bench_tempdir();
        let (wal, mut tail) = warm_wal(&temp.path().join("wal"));
        let start = Instant::now();
        for _ in 0..n {
            let entry = make_entry();
            black_box(wal.append_entry(&entry, &mut tail)).unwrap();
            if sync {
                wal.commit(tail, Duration::ZERO).unwrap();
            }
        }
        elapsed += start.elapsed();
        done += n;
    }
    elapsed
}

/// Appends per WAL for `value_size`-byte values, capped at `max`.
fn appends_per_wal(value_size: usize, max: usize) -> u64 {
    (WAL_BYTES_PER_RUN / value_size).clamp(1, max) as u64
}

/// WAL append with fsync (the default crash-safe path).
fn bench_wal_append_fsync(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal/append_fsync");
    group.measurement_time(Duration::from_secs(10));

    for value_size in [128usize, 4_096, 65_536] {
        let value = vec![0xABu8; value_size];
        let per_wal = appends_per_wal(value_size, 1_000);

        group.throughput(Throughput::Bytes(value_size as u64));
        group.bench_with_input(BenchmarkId::new("value_size", format!("{value_size}B")), &value_size, |b, _| {
            b.iter_custom(|iters| time_appends(iters, per_wal, true, || WalEntry::new_upsert(b"bench_key_00001".to_vec(), value.clone())));
        });
    }
    group.finish();
}

/// WAL append without fsync — shows the throughput ceiling when durability
/// guarantees are relaxed (e.g. for batched-sync configurations).
fn bench_wal_append_no_fsync(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal/append_no_fsync");
    group.measurement_time(Duration::from_secs(10));

    for value_size in [128usize, 4_096, 65_536] {
        let value = vec![0xABu8; value_size];
        let per_wal = appends_per_wal(value_size, 5_000);

        group.throughput(Throughput::Bytes(value_size as u64));
        group.bench_with_input(BenchmarkId::new("value_size", format!("{value_size}B")), &value_size, |b, _| {
            b.iter_custom(|iters| time_appends(iters, per_wal, false, || WalEntry::new_upsert(b"bench_key_00001".to_vec(), value.clone())));
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
        let (wal, mut tail) = Wal::open_standalone(&temp.path().join("wal"), DEFAULT_SEGMENT_SIZE).expect("failed to open WAL");
        let value = vec![0xEFu8; value_size];

        // Pre-populate
        for i in 0..num_entries {
            let key = format!("scan_key_{:010}", i).into_bytes();
            let entry = WalEntry::new_upsert(key, value.clone());
            wal.append_entry(&entry, &mut tail).unwrap();
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
    // One store, already in its steady state (see `steady_store`): a store
    // per iteration would time a first write into a new database instead.
    let temp = bench_tempdir();
    let store = steady_store(temp.path(), value_size);

    // Full path: Db::put (WAL + value log + LSM)
    group.throughput(Throughput::Elements(1));
    group.bench_function("db_put_with_wal", |b| {
        let mut seq = 0u64;
        b.iter(|| {
            seq += 1;
            black_box(store.store().put(&make_key("wal_put:", seq), &value)).unwrap();
        });
    });

    group.finish();
}

/// Measures WAL append with namespace_id tagging to detect
/// any overhead from the namespace field.
fn bench_wal_namespace_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal/namespace_overhead");
    group.measurement_time(Duration::from_secs(10));

    const BATCH: u64 = 1_000;
    let value = vec![0xABu8; 512];

    // Append with default namespace (id=0)
    {
        let value = value.clone();
        group.bench_function("ns_id_0", |b| {
            b.iter_custom(|iters| time_appends(iters, BATCH, true, || WalEntry::new_upsert_ns(0, b"key_00001".to_vec(), value.clone())));
        });
    }

    // Append with a non-default namespace (id=42)
    {
        let value = value.clone();
        group.bench_function("ns_id_42", |b| {
            b.iter_custom(|iters| time_appends(iters, BATCH, true, || WalEntry::new_upsert_ns(42, b"key_00001".to_vec(), value.clone())));
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
