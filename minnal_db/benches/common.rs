// Shared benchmark utilities — imported by each bench file via:
//   #[path = "common.rs"] mod common;
//   use common::*;
//
// Each bench binary uses only a subset of these helpers, so items unused by a
// given target would otherwise trip `dead_code` when that target is compiled.
#![allow(dead_code)]

use minnal_db::lsm::LSMConfig;
use minnal_db::{Db, DbConfig, ScheduledTaskConfig, SyncConfig, ThresholdConfig};
use std::path::Path;
use std::time::Duration;
use tempfile::TempDir;

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
