// What `merge` costs, next to the other CRUD operations.
//   cargo bench --bench bench_merge
//
// `merge` is `get` + a caller closure + `put`, held together by a per-key lock.
// This measures the whole thing against its parts, so the three questions a
// reader actually has can be answered separately:
//
//   1. What does a merge cost against a blind `put`?        crud/{put,merge_*}
//   2. What does *atomicity* cost against the `get` + `put`
//      a caller would otherwise hand-roll?                  crud/get_then_put
//   3. How much of a merge is the caller's own closure?     closure/*
//
// **Every write here pays a WAL fsync.** `bench_config()` sets
// `SyncConfig::default()`, but that knob is `records_per_sync`, which governs
// the *value log* only — `Database::put_ns` passes `sync = true` to the WAL
// append unconditionally, by design (see `minnal_db/CLAUDE.md` → "Per-write WAL
// fsync is deliberate"). So every `crud/*` write number is fsync latency plus a
// rounding error.
//
// ## Read `crud/*` write numbers across runs, never within one
//
// On a fsync-noisy host the `crud/*` write benchmarks cannot resolve anything
// `merge` adds, and criterion's confidence interval will not tell you that —
// its CI describes spread *within* a run, while the real error is slow drift
// *between* runs. Measured on WSL2, three consecutive runs of this file with no
// code change in between:
//
// ```text
//                 run 1     run 2     run 3
//   put/2KiB     3.04 ms   3.86 ms   5.07 ms     (1.67x spread)
//   put/8B       4.86 ms   3.62 ms   5.22 ms     (1.44x spread)
// ```
//
// Each run's own CI was ±5-10%. The tell that the numbers are meaningless is
// internal, and worth knowing how to spot: an 8 B put measured *slower* than a
// 2 KiB put, which no amount of real work can produce. If you see that ordering,
// stop reading the write rows.
//
// A quieter host shrinks the problem without removing it. On the bare-metal
// Linux box of the 2026-08-22 report (`benchmark.md`), all six `crud/*` write
// rows landed inside a 1.6% band — but the 8 B put was *still* the slowest of
// them, 1.5% above the 2 KiB put. Same inversion, two orders of magnitude
// smaller, and still larger than anything a merge adds.
//
// This is why the decomposition below exists. `crud/get`, `closure/*` and
// `stripe/*` touch no fsync, reproduce to within ~2% run-to-run, and between
// them account for **everything a merge does that a put does not**. Sum those
// rather than differencing two millisecond write numbers.

#[path = "common.rs"]
mod common;
use common::*;

use criterion::{Criterion, criterion_group, criterion_main};
use minnal_db::KVError;
use std::hint::black_box;
use std::time::Duration;

type MergeResult = std::result::Result<Option<Vec<u8>>, KVError>;

/// Elements in the "medium" closure's sorted set — 256 × 8 B = 2 KiB of value,
/// which is also the value size used by every 2 KiB benchmark below so the
/// operations stay comparable.
const SET_LEN: usize = 256;
/// Keys pre-written for `crud/delete` to consume. Sized above the iteration
/// count a 13 s (3 s warm-up + 10 s measurement) run reaches at these
/// latencies; past it the benchmark wraps and deletes already-absent keys,
/// which still writes a tombstone and a WAL entry but skips the displaced-record
/// accounting, so a wrap makes the number *optimistic* rather than invalid.
const DELETE_POOL: u64 = 8_000;
/// Keys cycled by the non-consuming benchmarks (get, merge, get-then-put).
const CYCLE_POOL: u64 = 1_024;

// ── The closures under test ────────────────────────────────────────────

/// **Simple.** Decode a `u64` counter, add the operand, re-encode. The floor:
/// almost nothing but the merge machinery itself.
fn counter_merge(existing: Option<&[u8]>, operand: &[u8]) -> MergeResult {
    let current = existing.map_or(0u64, |b| u64::from_le_bytes(b.try_into().unwrap()));
    let step = u64::from_le_bytes(operand.try_into().unwrap());
    Ok(Some(current.wrapping_add(step).to_le_bytes().to_vec()))
}

/// **Medium.** Maintain a sorted, de-duplicated set of `u64`s capped at
/// [`SET_LEN`]: decode the whole value, binary-search the insertion point,
/// splice, drop the smallest if full, re-encode.
///
/// This is the shape that motivates a merge operator in the first place —
/// accumulating into a collection, where the new value genuinely depends on the
/// old one — and unlike the counter it does real work proportional to the value:
/// a 2 KiB decode, a memmove, and a 2 KiB re-encode. The cap keeps the value size
/// constant across a benchmark run so the storage cost does not drift.
fn sorted_set_merge(existing: Option<&[u8]>, operand: &[u8]) -> MergeResult {
    let mut set: Vec<u64> = match existing {
        Some(bytes) => bytes.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect(),
        None => Vec::with_capacity(SET_LEN),
    };
    let candidate = u64::from_le_bytes(operand.try_into().unwrap());

    if let Err(pos) = set.binary_search(&candidate) {
        set.insert(pos, candidate);
        if set.len() > SET_LEN {
            set.remove(0); // bounded window: evict the smallest
        }
    }

    let mut out = Vec::with_capacity(set.len() * 8);
    for v in &set {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(Some(out))
}

/// A full sorted set, as the pre-populated value for the 2 KiB benchmarks.
fn seed_set(seed: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(SET_LEN * 8);
    for i in 0..SET_LEN as u64 {
        out.extend_from_slice(&(seed * SET_LEN as u64 + i).to_le_bytes());
    }
    out
}

// ── Group 1: merge against the other CRUD operations ───────────────────
//
// **Each benchmark gets its own freshly-populated store.** Sharing one store
// across the group made the results incoherent — `merge` came out *faster* than
// `put` and an 8 B merge slower than a 2 KiB one — because every benchmark
// inherited the value-log, WAL and memtable growth of the ones before it, on top
// of already-noisy fsync latency. State must not leak between the things being
// compared.
//
// Every operation here also **overwrites an existing key** from a pool of the
// same size. A `put` to a fresh key and a `merge` (which is only meaningful on a
// key that exists) are different amounts of work; matching them is what makes the
// comparison mean anything. `bench_write.rs` covers put-to-a-fresh-key.

/// A fresh store with `CYCLE_POOL` keys under `prefix`, each holding `value`.
/// The `TempDir` is returned so it outlives the store.
fn populated(prefix: &str, value: &[u8], count: u64) -> (tempfile::TempDir, AutoCloseStore) {
    let temp = bench_tempdir();
    let store = AutoCloseStore::open(temp.path());
    for i in 0..count {
        store.store().put(&make_key(prefix, i), value).unwrap();
    }
    (temp, store)
}

fn bench_crud(c: &mut Criterion) {
    let seeded = seed_set(1);
    let mut group = c.benchmark_group("crud");
    group.measurement_time(Duration::from_secs(10));

    // ── 2 KiB values ───────────────────────────────────────────────────
    {
        let (_t, s) = populated("k:", &seeded, CYCLE_POOL);
        let db = s.store();
        group.bench_function("put/2KiB", |b| {
            let mut i = 0u64;
            b.iter(|| {
                i = (i + 1) % CYCLE_POOL;
                black_box(db.put(&make_key("k:", i), &seeded)).unwrap();
            });
        });
    }
    {
        let (_t, s) = populated("k:", &seeded, CYCLE_POOL);
        let db = s.store();
        group.bench_function("get/2KiB", |b| {
            let mut i = 0u64;
            b.iter(|| {
                i = (i + 1) % CYCLE_POOL;
                black_box(db.get(&make_key("k:", i))).unwrap();
            });
        });
    }
    {
        // Delete consumes keys, so this pool is sized above the iteration count
        // a 13 s run (3 s warm-up + 10 s measurement) reaches at these
        // latencies. Past it the benchmark wraps onto already-absent keys, which
        // still writes a tombstone and a WAL entry but skips the displaced-record
        // accounting — a wrap makes the number optimistic, not invalid.
        let (_t, s) = populated("k:", &seeded, DELETE_POOL);
        let db = s.store();
        group.bench_function("delete/2KiB", |b| {
            let mut i = 0u64;
            b.iter(|| {
                i += 1;
                black_box(db.delete(&make_key("k:", i % DELETE_POOL))).unwrap();
            });
        });
    }
    {
        let (_t, s) = populated("k:", &seeded, CYCLE_POOL);
        let db = s.store();
        group.bench_function("merge_sorted_set/2KiB", |b| {
            let mut i = 0u64;
            b.iter(|| {
                i += 1;
                let key = make_key("k:", i % CYCLE_POOL);
                black_box(db.merge(&key, &i.to_le_bytes(), sorted_set_merge)).unwrap();
            });
        });
    }
    {
        // What a caller writes today without `merge`: the same work, no
        // atomicity. The difference between this and the row above is the price
        // of the guarantee.
        let (_t, s) = populated("k:", &seeded, CYCLE_POOL);
        let db = s.store();
        group.bench_function("get_then_put_sorted_set/2KiB", |b| {
            let mut i = 0u64;
            b.iter(|| {
                i += 1;
                let key = make_key("k:", i % CYCLE_POOL);
                let existing = db.get(&key).unwrap();
                let merged = sorted_set_merge(existing.as_deref(), &i.to_le_bytes()).unwrap().unwrap();
                black_box(db.put(&key, &merged)).unwrap();
            });
        });
    }

    // ── 8 B values: the cheap-closure pair ─────────────────────────────
    {
        let (_t, s) = populated("k:", &0u64.to_le_bytes(), CYCLE_POOL);
        let db = s.store();
        group.bench_function("put/8B", |b| {
            let mut i = 0u64;
            b.iter(|| {
                i = (i + 1) % CYCLE_POOL;
                black_box(db.put(&make_key("k:", i), &i.to_le_bytes())).unwrap();
            });
        });
    }
    {
        let (_t, s) = populated("k:", &0u64.to_le_bytes(), CYCLE_POOL);
        let db = s.store();
        group.bench_function("merge_counter/8B", |b| {
            let mut i = 0u64;
            b.iter(|| {
                i += 1;
                let key = make_key("k:", i % CYCLE_POOL);
                black_box(db.merge(&key, &1u64.to_le_bytes(), counter_merge)).unwrap();
            });
        });
    }

    group.finish();
}

// ── Group 2: the closures alone, with no database underneath ───────────
//
// How much of a merge is the caller's own processing. Subtract these from the
// `crud/merge_*` numbers to see what the engine contributes.

fn bench_closure(c: &mut Criterion) {
    let mut group = c.benchmark_group("closure");
    let full = seed_set(1);

    group.bench_function("counter", |b| {
        let current = 41u64.to_le_bytes();
        let operand = 1u64.to_le_bytes();
        b.iter(|| black_box(counter_merge(black_box(Some(&current[..])), black_box(&operand[..]))).unwrap());
    });

    group.bench_function("sorted_set", |b| {
        let mut i = 0u64;
        b.iter(|| {
            i += 1;
            black_box(sorted_set_merge(black_box(Some(&full[..])), black_box(&i.to_le_bytes()[..]))).unwrap()
        });
    });

    group.finish();
}

// ── Group 3: what the key stripe added to the existing write path ──────
//
// `merge` is made atomic by a striped per-key lock that `put` and `delete` now
// take as well. `db::key_locks::KeyLocks` is crate-private, so a bench (a
// separate compilation unit) cannot call it — this **reproduces** its body
// against the same primitives to size the cost. Treat it as a proxy for
// `KeyLocks::guard`, not a measurement of it.

fn bench_key_stripe(c: &mut Criterion) {
    use mm3h::Murmur3Hasher;
    use parking_lot::Mutex;
    use std::hash::Hasher;

    const STRIPES: usize = 1024;
    let stripes: Vec<Mutex<()>> = (0..STRIPES).map(|_| Mutex::new(())).collect();

    let mut group = c.benchmark_group("stripe");
    group.bench_function("hash_and_lock_uncontended", |b| {
        let key = make_key("bench:", 42);
        b.iter(|| {
            let mut hasher = Murmur3Hasher::new_with_seed(0x5D9E_2A17);
            hasher.write(&0u32.to_le_bytes());
            hasher.write(black_box(&key));
            let idx = (hasher.finish() as usize) & (STRIPES - 1);
            let guard = stripes[idx].lock();
            black_box(&guard);
        });
    });
    group.finish();
}

criterion_group!(
    name    = merge_benches;
    config  = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = bench_crud, bench_closure, bench_key_stripe
);
criterion_main!(merge_benches);
