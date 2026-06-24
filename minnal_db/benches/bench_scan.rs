// Scan benchmarks: prefix scan, range scan, cursor pagination, iter_stream
//   cargo bench --bench bench_scan

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use minnal_db::AsyncDb;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

// ── Prefix scan ───────────────────────────────────────────────────────────────
//
// This is the most SIMD-sensitive group. The skip-list's `scan_prefix`
// compares each candidate key against the prefix bytes; with AVX-512 each
// comparison processes 64 bytes at once. Long keys amplify the benefit.

fn bench_prefix_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan/prefix");
    group.measurement_time(Duration::from_secs(12));

    // (matching_key_count, key_length_bytes)
    let cases: &[(u64, usize)] = &[
        (100, 32),
        (1_000, 32),
        (100, 128), // longer keys → SIMD churns more
        (1_000, 128),
        (5_000, 128),
    ];

    for &(n_keys, key_len) in cases {
        let temp = bench_tempdir();
        let store = open_store(temp.path());
        let value = vec![0u8; 64];

        // Fill target prefix
        for seq in 0..n_keys {
            store.put(&make_long_key("pfx:", seq, key_len), &value).unwrap();
        }
        // Fill noise keys with a different prefix
        for seq in 0..n_keys {
            store.put(&make_long_key("other:", seq, key_len), &value).unwrap();
        }

        let label = format!("{n_keys}_keys_{}B", key_len);
        group.throughput(Throughput::Elements(n_keys));
        group.bench_function(&label, |b| {
            b.iter(|| {
                let results = store.scan_prefix(b"pfx:").unwrap();
                black_box(results.len())
            });
        });
    }
    group.finish();
}

// ── Range scan ────────────────────────────────────────────────────────────────
//
// Note: range() loads all keys >= start upfront before returning the iterator,
// so TOTAL_KEYS is kept modest to avoid per-sample O(N) cost at large N.

fn bench_range_scan(c: &mut Criterion) {
    const TOTAL_KEYS: u64 = 2_000;
    let mut group = c.benchmark_group("scan/range");
    group.measurement_time(Duration::from_secs(10));

    let temp = bench_tempdir();
    let store = open_store(temp.path());
    let value = vec![0u8; 256];

    for seq in 0..TOTAL_KEYS {
        store.put(&make_key("rng:", seq), &value).unwrap();
    }

    for result_count in [100u64, 500, 1_000] {
        let start = make_key("rng:", 0);
        let end = make_key("rng:", result_count);

        group.throughput(Throughput::Elements(result_count));
        group.bench_with_input(BenchmarkId::new("result_count", result_count), &result_count, |b, _| {
            b.iter(|| {
                let results = store.range(start.as_slice(), Some(end.as_slice())).unwrap();
                black_box(results.len())
            });
        });
    }
    group.finish();
}

// ── Cursor-based pagination (async) ───────────────────────────────────────────
//
// Measures the latency of a single page fetch (scan from the beginning).
// A multi-thread runtime is required so that spawn_blocking has a thread pool
// available and does not deadlock when block_on is called from within it.

fn bench_cursor_scan(c: &mut Criterion) {
    const TOTAL_KEYS: u64 = 1_000;
    let mut group = c.benchmark_group("scan/cursor");
    group.measurement_time(Duration::from_secs(10));

    let rt = tokio::runtime::Runtime::new().unwrap();

    let temp = bench_tempdir();
    let sync_store = open_store(temp.path());
    let value = vec![0u8; 256];
    for seq in 0..TOTAL_KEYS {
        sync_store.put(&make_key("cur:", seq), &value).unwrap();
    }
    sync_store.shutdown().unwrap();

    // `AsyncDb::open_with_config` requires an owned (`'static`) path because it
    // offloads the open via `spawn_blocking`; hand it an owned copy of the temp
    // dir path (the `TempDir` itself still owns/cleans up the directory).
    let db_path = temp.path().to_path_buf();
    let store = Arc::new(rt.block_on(async move { AsyncDb::open_with_config(db_path, bench_config()).await.unwrap() }));

    for page_size in [100usize, 500, 1_000] {
        let s = store.clone();
        group.throughput(Throughput::Elements(page_size as u64));
        group.bench_with_input(BenchmarkId::new("page_size", page_size), &page_size, |b, &ps| {
            b.iter(|| {
                rt.block_on(async {
                    let (page, next) = s.scan(None, None, ps).await.unwrap();
                    black_box((page.len(), next.is_some()))
                })
            });
        });
    }

    rt.block_on(async { store.shutdown().await.unwrap() });
    group.finish();
}

// ── Async iter (full collection) ──────────────────────────────────────────────
//
// Measures end-to-end cost of fetching all key-value pairs via the async API.

fn bench_iter(c: &mut Criterion) {
    const NUM_KEYS: u64 = 2_000;
    let mut group = c.benchmark_group("scan/iter");
    group.measurement_time(Duration::from_secs(10));

    let rt = tokio::runtime::Runtime::new().unwrap();

    for value_size in [128usize, 4_096] {
        let temp = bench_tempdir();
        let sync_store = open_store(temp.path());
        let value = vec![0u8; value_size];

        for seq in 0..NUM_KEYS {
            sync_store.put(&make_key("st:", seq), &value).unwrap();
        }
        sync_store.shutdown().unwrap();

        let db_path = temp.path().to_path_buf();
        let store = Arc::new(rt.block_on(async move { AsyncDb::open_with_config(db_path, bench_config()).await.unwrap() }));

        let s = store.clone();
        group.throughput(Throughput::Elements(NUM_KEYS));
        group.bench_with_input(BenchmarkId::new("value_size", format!("{value_size}B")), &value_size, |b, _| {
            b.iter(|| {
                rt.block_on(async {
                    let count = s.iter().await.unwrap().len();
                    black_box(count)
                })
            });
        });

        rt.block_on(async { store.shutdown().await.unwrap() });
    }
    group.finish();
}

criterion_group!(
    name    = scan_benches;
    config  = Criterion::default().measurement_time(Duration::from_secs(12));
    targets = bench_prefix_scan, bench_range_scan, bench_cursor_scan, bench_iter
);
criterion_main!(scan_benches);
