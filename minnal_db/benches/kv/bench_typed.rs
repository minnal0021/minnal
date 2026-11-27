// Typed API (rkyv ser/de) benchmarks
//   cargo bench --bench bench_typed
//
// Measures the overhead of the *_typed() convenience layer which adds rkyv
// serialization on write and deserialization on read, compared to raw bytes.
//
// Read-side cases (get) are measured on both the
// active memtable and the on-disk L1 SSTable tier (`TIERS` in common.rs) —
// memtable-only numbers are cheap but not representative of steady-state
// reads once data ages onto disk. Write-side cases (put/delete) are NOT
// tiered: a write always lands in the active memtable regardless of what's
// already on disk, so tier doesn't apply to their cost the same way.

#[path = "../common.rs"]
mod common;
use common::*;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use minnal_db::bench_internals::{Db, rkyv_deserialize, rkyv_serialize};
use minnal_db::rkyv_derives::{Archive, Deserialize, Serialize};
use minnal_db::{KVError, RkyvDecode, RkyvEncode};
use std::hint::black_box;
use std::path::Path;
use std::time::Duration;

/// The typed layer over the engine handle, built from the same codec helpers
/// and in the same order as `AsyncNamespace`'s typed methods, so these cases
/// time the typed layer without `AsyncDb`'s blocking-pool hop.
trait Typed {
    fn put_typed<K: RkyvEncode, V: RkyvEncode>(&self, key: &K, value: &V) -> Result<(), KVError>;
    fn get_typed<K: RkyvEncode, V: RkyvDecode>(&self, key: &K) -> Result<Option<V>, KVError>;
    fn delete_typed<K: RkyvEncode>(&self, key: &K) -> Result<(), KVError>;
}

impl Typed for Db {
    fn put_typed<K: RkyvEncode, V: RkyvEncode>(&self, key: &K, value: &V) -> Result<(), KVError> {
        self.put(&rkyv_serialize(key)?, &rkyv_serialize(value)?)
    }
    fn get_typed<K: RkyvEncode, V: RkyvDecode>(&self, key: &K) -> Result<Option<V>, KVError> {
        match self.get(&rkyv_serialize(key)?)? {
            Some(vb) => Ok(Some(rkyv_deserialize::<V>(&vb)?)),
            None => Ok(None),
        }
    }
    fn delete_typed<K: RkyvEncode>(&self, key: &K) -> Result<(), KVError> {
        self.delete(&rkyv_serialize(key)?)
    }
}

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

// A fixed 8-byte prefix and a big-endian id: `put_typed` uses the key's rkyv
// bytes as the stored key, and rkyv archives integers little-endian, so the
// id is kept as bytes for key order to match id order.
#[derive(Archive, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[rkyv(compare(PartialEq), derive(Debug))]
struct BenchKey {
    prefix: [u8; 8],
    id: [u8; 8],
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
        Self {
            prefix,
            id: seq.to_be_bytes(),
        }
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

/// One writer's typed `put` of a fresh key, in the steady state: one store per
/// value size, its WAL already on zero-filled segments (see `steady_store`).
fn bench_put_typed(c: &mut Criterion) {
    let mut group = c.benchmark_group("typed/put");
    group.measurement_time(Duration::from_secs(10));

    for value_size in [128usize, 4_096] {
        let value = BenchValue::new(value_size);
        let temp = bench_tempdir();
        let store = steady_store(temp.path(), value_size);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("value_size", format!("{value_size}B")), &value_size, |b, _| {
            let mut seq = 0u64;
            b.iter(|| {
                seq += 1;
                black_box(store.store().put_typed(&BenchKey::new(seq), &value)).unwrap();
            });
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

/// One writer's typed `delete`, in the steady state: one store, its WAL already
/// on zero-filled segments, with `POOL` keys written first. Past the first
/// `POOL` deletes the keys are already gone; a delete still writes its WAL
/// entry and tombstone, which is what this measures.
fn bench_delete_typed(c: &mut Criterion) {
    const POOL: u64 = 2_000;
    let mut group = c.benchmark_group("typed/delete");
    group.measurement_time(Duration::from_secs(10));

    let temp = bench_tempdir();
    let store = steady_store(temp.path(), 128);
    let value = BenchValue::new(128);
    for seq in 0..POOL {
        store.store().put_typed(&BenchKey::new(seq), &value).unwrap();
    }

    group.throughput(Throughput::Elements(1));
    group.bench_function("delete", |b| {
        let mut seq = 0u64;
        b.iter(|| {
            seq = (seq + 1) % POOL;
            black_box(store.store().delete_typed(&BenchKey::new(seq))).unwrap();
        });
    });
    group.finish();
}

criterion_group!(
    name    = typed_benches;
    config  = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = bench_put_typed,
              bench_get_typed,
              bench_delete_typed
);
criterion_main!(typed_benches);
