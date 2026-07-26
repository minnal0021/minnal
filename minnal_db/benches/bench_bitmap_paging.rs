//! Offset paging over a `RoaringBitmap`: `iter().skip(n)` vs `iter_page(n)`.
//!
//! `query_keys_paginated` resolves a page by windowing the bitmap that query
//! evaluation produced. It used to do that with `iter().skip(offset)`, which is
//! `O(offset)` — and worse than it sounds, because `iter` deserialises and
//! collects a `Vec<u16>` for every container it passes over. `iter_page`
//! skips whole containers on their cached cardinality instead, and materialises
//! at most `limit` values from each container it opens.
//!
//! Two shapes are measured:
//!
//! - **single page** at increasing offsets — the old path degrades linearly with
//!   the offset.
//! - **full paged walk** — every page of a result set, which is what a client
//!   paginating through query results actually does. Quadratic vs linear.
//!
//! Both bounds on `iter_page` are load-bearing, and this bench is what proved
//! it: with only the outer `take(limit)`, offset 0 landed at the start of a full
//! 65 536-value container and materialised all of it to return 50 values —
//! measured *slower* than the `iter().skip()` being replaced (74 µs vs 65 µs).
//!
//! Watch the offset-50 000 case in particular. It lands mid-container, so it
//! pays the intra-container walk that `iter_page` does **not** eliminate — that
//! skip is linear in the landing container's cardinality (≤ 65 536), just not in
//! `offset`. It is the weakest case and the one to check if the inner skip is
//! ever made rank-based per container type.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use minnal_db::index::RoaringBitmap;

/// Dense row IDs, as the field index assigns them. 200k spans several
/// containers (one per 2^16 of ID space), so container skipping has something
/// to skip.
const N_ROWS: u128 = 200_000;

/// A realistic REST page size.
const PAGE: usize = 50;

fn build_bitmap() -> RoaringBitmap {
    RoaringBitmap::from_sorted_iter(0..N_ROWS)
}

/// The pre-change path, kept verbatim so the comparison is honest.
fn page_via_iter_skip(bm: &RoaringBitmap, offset: usize, limit: usize) -> Vec<u128> {
    bm.iter().skip(offset).take(limit).collect()
}

fn page_via_rank(bm: &RoaringBitmap, offset: usize, limit: usize) -> Vec<u128> {
    bm.iter_page(offset, limit).collect()
}

fn bench_single_page(c: &mut Criterion) {
    let bm = build_bitmap();
    let mut group = c.benchmark_group("bitmap_page_at_offset");

    for offset in [0usize, 1_000, 50_000, 199_000] {
        group.bench_with_input(BenchmarkId::new("iter_skip", offset), &offset, |b, &off| {
            b.iter(|| black_box(page_via_iter_skip(&bm, off, PAGE)));
        });
        group.bench_with_input(BenchmarkId::new("iter_page", offset), &offset, |b, &off| {
            b.iter(|| black_box(page_via_rank(&bm, off, PAGE)));
        });
    }
    group.finish();
}

fn bench_full_walk(c: &mut Criterion) {
    // Smaller set: the old path is quadratic here, so a full-size walk would
    // dominate the whole bench run for no extra signal.
    let bm = RoaringBitmap::from_sorted_iter(0..20_000u128);
    let total = 20_000usize;
    let page = 200usize;

    let mut group = c.benchmark_group("bitmap_full_paged_walk");
    group.bench_function("iter_skip", |b| {
        b.iter(|| {
            let mut seen = 0usize;
            let mut offset = 0usize;
            while offset < total {
                seen += black_box(page_via_iter_skip(&bm, offset, page)).len();
                offset += page;
            }
            black_box(seen)
        });
    });
    group.bench_function("iter_page", |b| {
        b.iter(|| {
            let mut seen = 0usize;
            let mut offset = 0usize;
            while offset < total {
                seen += black_box(page_via_rank(&bm, offset, page)).len();
                offset += page;
            }
            black_box(seen)
        });
    });
    group.finish();
}

criterion_group!(benches, bench_single_page, bench_full_walk);
criterion_main!(benches);
