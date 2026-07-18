// Shared benchmark utilities — imported by each bench file via:
//   #[path = "common.rs"] mod common;
//   use common::*;
//
// Each bench binary uses only a subset of these helpers, so items unused by a
// given target would otherwise trip `dead_code` when that target is compiled.
#![allow(dead_code)]

use minnal_db::lsm::LSMConfig;
use minnal_db::{AsyncDb, Db, DbConfig, ScheduledTaskConfig, SyncConfig, ThresholdConfig};
use std::path::Path;
use std::time::Duration;
use tempfile::TempDir;

/// The two storage tiers most benchmarks in this suite compare. Memtable-only
/// numbers are cheap but not representative of steady-state reads once data
/// ages out of the memtable; L1 exercises the real on-disk lookup path
/// (`lookup_level1`: min/max range check, bloom filter, sparse-index bounded
/// scan — see `bench_sstable_lookup.rs`). L0 is intentionally not a separate
/// tier here: there is no public API path to observe an on-disk "flushed but
/// not yet compacted" L0 state — the only way to write an L0 file
/// (`compact_all`, reached via `compact()` or the background
/// `LsmCompactionWorker`) unconditionally merges every L0 file into L1 in
/// that same call, so it can't be held open long enough to benchmark.
pub const TIERS: [&str; 2] = ["memtable", "l1"];

/// Create a TempDir under `target/bench_tmp/` instead of `/tmp`.
///
/// `/tmp` is often a size-limited tmpfs (RAM-backed). Benchmarks that
/// create thousands of batched stores can exhaust it. The project's
/// `target/` directory lives on the real filesystem with much more space.
pub fn bench_tempdir() -> TempDir {
    let bench_tmp = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bench_tmp");
    std::fs::create_dir_all(&bench_tmp).expect("failed to create target/bench_tmp");
    tempfile::tempdir_in(&bench_tmp).expect("failed to create temp dir")
}

/// Config tuned for benchmarking: background workers effectively disabled,
/// no fsync (SyncConfig::default() is already Never/Never).
pub fn bench_config() -> DbConfig {
    DbConfig::new(
        ThresholdConfig::new(99.9), // never auto-trigger GC during bench
        ScheduledTaskConfig::new(
            Duration::from_secs(86_400), // GC interval — 24 h
            Duration::from_secs(86_400), // WAL GC interval
            Duration::from_secs(86_400), // LSM compaction interval
        ),
        SyncConfig::default(), // wal: Never, value_log: Never
        LSMConfig::default(),
    )
}

pub fn open_store(dir: &Path) -> Db {
    Db::open_with_config(dir, bench_config()).expect("failed to open Db")
}

/// Like `bench_config()` but with a custom memtable (skip-list) capacity.
/// `DbConfig::skip_list_capacity` is public and is propagated into the
/// internal `LSMConfig` on `Db::open` — this is the only way a bench (an
/// external compilation unit) can control memtable size, since `LSMConfig`
/// itself only exposes it as `pub(crate)`. Useful for keeping a `num_keys`
/// sweep entirely inside one active memtable (set capacity comfortably above
/// the largest `num_keys`) so the benchmark doesn't cross the default 100k
/// threshold and spill into a sealed read-only memtable mid-sweep.
pub fn bench_config_with_capacity(capacity: usize) -> DbConfig {
    let mut config = bench_config();
    config.skip_list_capacity = capacity;
    config
}

pub fn open_store_with_capacity(dir: &Path, capacity: usize) -> Db {
    Db::open_with_config(dir, bench_config_with_capacity(capacity)).expect("failed to open Db")
}

/// Push an already-populated store's data down to the on-disk L1 SSTable
/// (compact + shutdown + reopen), so subsequent reads hit the real,
/// accelerated `lookup_level1` path instead of the active memtable. Consumes
/// `store` and returns a fresh handle reopened over the same directory.
///
/// The explicit `drop(store)` before reopening is load-bearing: `KVStore`'s
/// `Drop` impl re-runs `flush_and_compact_all()` (harmless on its own, but it
/// touches on-disk L0/L1 files). Reopening a second `Db` over the same
/// directory before the first one drops leaves both instances alive at once
/// — the old instance's deferred Drop cleanup can then race the new
/// instance's first reads and manifest load, surfacing as a spurious
/// `NotFound` on a file the new instance expected to still be there. Always
/// drop the old handle first.
pub fn push_to_l1(store: Db, dir: &Path) -> Db {
    store.compact().expect("compact failed");
    store.shutdown().expect("shutdown failed");
    drop(store);
    open_store(dir)
}

/// Async counterpart of `push_to_l1` — same drop-before-reopen requirement.
pub async fn push_to_l1_async(store: AsyncDb, dir: &Path) -> AsyncDb {
    store.compact().await.expect("compact failed");
    store.shutdown().await.expect("shutdown failed");
    drop(store);
    AsyncDb::open_with_config(dir.to_path_buf(), bench_config()).await.expect("failed to reopen Db")
}

/// Wrapper that calls `Db::shutdown()` on drop to release file handles
/// promptly. Without this, `iter_batched_ref` accumulates open SSTable/value-log
/// fds across batches and eventually hits the OS open-file limit.
pub struct AutoCloseStore(pub Option<Db>);

impl AutoCloseStore {
    pub fn open(path: &Path) -> Self {
        Self(Some(open_store(path)))
    }

    pub fn store(&self) -> &Db {
        self.0.as_ref().unwrap()
    }
}

impl Drop for AutoCloseStore {
    fn drop(&mut self) {
        if let Some(store) = self.0.take() {
            let _ = store.shutdown();
        }
    }
}

/// Generate a key with a given prefix and zero-padded sequence number.
/// Using a fixed-width number keeps keys lexicographically ordered.
#[inline]
pub fn make_key(prefix: &str, seq: u64) -> Vec<u8> {
    format!("{}{:015}", prefix, seq).into_bytes()
}

/// Generate a key padded to `total_len` bytes for testing long-key comparisons.
#[inline]
pub fn make_long_key(prefix: &str, seq: u64, total_len: usize) -> Vec<u8> {
    let base = format!("{}{:015}", prefix, seq);
    let mut key = base.into_bytes();
    key.resize(total_len, b'x');
    key
}
