use std::collections::HashMap;
use std::sync::Arc;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use parking_lot::RwLock;

use minnal_db::index::query::{evaluate, parse, parse_and_evaluate};
use minnal_db::index::{DynFieldIndex, IndexValue, IndexValueType};

// ── Setup ──────────────────────────────────────────────────────────────────

const N_ROWS: u128 = 100_000;

struct Indices {
    schema: HashMap<String, u32>,
    status: Arc<RwLock<DynFieldIndex>>,
    age: Arc<RwLock<DynFieldIndex>>,
    active: Arc<RwLock<DynFieldIndex>>,
}

impl Indices {
    fn build() -> Self {
        let mut status_idx = DynFieldIndex::new(IndexValueType::Str);
        let mut age_idx = DynFieldIndex::new(IndexValueType::Int);
        let mut active_idx = DynFieldIndex::new(IndexValueType::Bool);

        let statuses = ["active", "inactive", "trial", "banned"];
        for row in 0..N_ROWS {
            let status = statuses[(row % 4) as usize];
            let age = 18 + (row % 63) as i64; // 18–80
            let active = row % 2 == 0;

            status_idx.insert(&IndexValue::Str(status.into()), row).unwrap();
            age_idx.insert(&IndexValue::Int(age), row).unwrap();
            active_idx.insert(&IndexValue::Bool(active), row).unwrap();
        }

        let schema: HashMap<String, u32> = [("status".into(), 0u32), ("age".into(), 1u32), ("active".into(), 2u32)].into();

        Self {
            schema,
            status: Arc::new(RwLock::new(status_idx)),
            age: Arc::new(RwLock::new(age_idx)),
            active: Arc::new(RwLock::new(active_idx)),
        }
    }

    fn get(&self, id: u32) -> Option<Arc<RwLock<DynFieldIndex>>> {
        match id {
            0 => Some(Arc::clone(&self.status)),
            1 => Some(Arc::clone(&self.age)),
            2 => Some(Arc::clone(&self.active)),
            _ => None,
        }
    }
}

// ── Benchmarks ─────────────────────────────────────────────────────────────

fn bench_str_eq(c: &mut Criterion) {
    let idx = Indices::build();
    c.bench_function("predicate/str_eq", |b| {
        b.iter(|| parse_and_evaluate(black_box("status = 'active'"), &idx.schema, &|id| idx.get(id)).unwrap())
    });
}

fn bench_int_range(c: &mut Criterion) {
    let idx = Indices::build();
    c.bench_function("predicate/int_range", |b| {
        b.iter(|| parse_and_evaluate(black_box("age >= 30 AND age <= 50"), &idx.schema, &|id| idx.get(id)).unwrap())
    });
}

fn bench_compound_and(c: &mut Criterion) {
    let idx = Indices::build();
    c.bench_function("predicate/compound_and", |b| {
        b.iter(|| parse_and_evaluate(black_box("status = 'active' AND active = true"), &idx.schema, &|id| idx.get(id)).unwrap())
    });
}

fn bench_compound_or(c: &mut Criterion) {
    let idx = Indices::build();
    c.bench_function("predicate/compound_or", |b| {
        b.iter(|| parse_and_evaluate(black_box("status = 'active' OR status = 'trial'"), &idx.schema, &|id| idx.get(id)).unwrap())
    });
}

fn bench_three_way_and(c: &mut Criterion) {
    let idx = Indices::build();
    c.bench_function("predicate/three_way_and", |b| {
        b.iter(|| {
            parse_and_evaluate(black_box("status = 'active' AND age >= 25 AND active = true"), &idx.schema, &|id| {
                idx.get(id)
            })
            .unwrap()
        })
    });
}

/// Parse overhead: compare parse+eval vs eval against a pre-parsed expression.
fn bench_parse_vs_eval(c: &mut Criterion) {
    let idx = Indices::build();
    let query = "status = 'active' AND age >= 30 AND active = true";
    let parsed = parse(query).unwrap();

    let mut group = c.benchmark_group("predicate/parse_vs_eval");
    group.bench_function("parse_and_eval", |b| {
        b.iter(|| parse_and_evaluate(black_box(query), &idx.schema, &|id| idx.get(id)).unwrap())
    });
    group.bench_function("eval_only", |b| {
        b.iter(|| evaluate(black_box(&parsed), &idx.schema, &|id| idx.get(id)).unwrap())
    });
    group.finish();
}

/// Selectivity sweep: same AND structure, narrowing result set.
fn bench_selectivity(c: &mut Criterion) {
    let idx = Indices::build();
    let mut group = c.benchmark_group("predicate/selectivity");

    // ~25 % of rows
    group.bench_with_input(BenchmarkId::new("AND", "25pct"), &(), |b, _| {
        b.iter(|| parse_and_evaluate(black_box("status = 'active'"), &idx.schema, &|id| idx.get(id)).unwrap())
    });

    // ~12.5 % of rows
    group.bench_with_input(BenchmarkId::new("AND", "12pct"), &(), |b, _| {
        b.iter(|| parse_and_evaluate(black_box("status = 'active' AND active = true"), &idx.schema, &|id| idx.get(id)).unwrap())
    });

    // ~6 % of rows
    group.bench_with_input(BenchmarkId::new("AND", "6pct"), &(), |b, _| {
        b.iter(|| {
            parse_and_evaluate(black_box("status = 'active' AND active = true AND age >= 50"), &idx.schema, &|id| {
                idx.get(id)
            })
            .unwrap()
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_str_eq,
    bench_int_range,
    bench_compound_and,
    bench_compound_or,
    bench_three_way_and,
    bench_parse_vs_eval,
    bench_selectivity,
);
criterion_main!(benches);
