//! Progress observer trait for index build operations.
//!
//! Both the field-index builder (`rebuild_index_for_namespace`) and the
//! vector-index reindex path (`index_all`) implement build work in terms of
//! this trait so that persistence, metrics, and in-memory state tracking are
//! composable rather than baked into the build loop.
//!
//! # Built-in implementations
//!
//! - [`InMemoryProgress`] — atomics-backed snapshot for live API queries.
//! - [`DiskProgress`] — throttled atomic-JSON tmp-then-rename for crash
//!   recovery (replaces the ad-hoc `write_disk_progress` calls).
//! - [`ChainedObserver`] — fan-out to multiple observers.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::hex::bytes_to_hex;
use crate::index_progress::BuildStatus;
use crate::store::DiskBuildProgress;

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Receives progress notifications from an index builder.
///
/// Implementations must be cheap and non-blocking because they are called from
/// inside the build task.  Heavy work (e.g. flushing to disk, writing metrics)
/// must be throttled or done on a side channel.
pub trait IndexProgressObserver: Send + Sync + 'static {
    /// One document was attempted.  `failed` is `true` when that attempt failed.
    /// `last_key` is the raw KV key of the document just processed.
    fn on_progress(&self, indexed: u64, total: u64, failed: bool, last_key: Option<&[u8]>);

    /// The build transitioned to a new lifecycle status.
    ///
    /// Called at least twice: once with `Running` when the build begins, and
    /// once with `Complete` or `Failed` when it terminates.
    fn on_status(&self, status: BuildStatus, error: Option<&str>);
}

// ── ChainedObserver ───────────────────────────────────────────────────────────

/// Fan-out adapter: delegates all calls to every contained observer in order.
pub struct ChainedObserver(pub Vec<Arc<dyn IndexProgressObserver>>);

impl IndexProgressObserver for ChainedObserver {
    fn on_progress(&self, indexed: u64, total: u64, failed: bool, last_key: Option<&[u8]>) {
        for obs in &self.0 {
            obs.on_progress(indexed, total, failed, last_key);
        }
    }

    fn on_status(&self, status: BuildStatus, error: Option<&str>) {
        for obs in &self.0 {
            obs.on_status(status, error);
        }
    }
}

// ── InMemoryProgress ──────────────────────────────────────────────────────────

/// Atomics-backed progress snapshot suitable for live polling from API handlers.
///
/// This is the replacement for the ad-hoc `BuildState` struct that was
/// embedded directly in `IndexBuildHandle`.
pub struct InMemoryProgress {
    pub total: AtomicU64,
    pub indexed: AtomicU64,
    pub done: AtomicBool,
    pub error: std::sync::Mutex<Option<String>>,
}

impl Default for InMemoryProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryProgress {
    pub fn new() -> Self {
        Self {
            total: AtomicU64::new(0),
            indexed: AtomicU64::new(0),
            done: AtomicBool::new(false),
            error: std::sync::Mutex::new(None),
        }
    }

    pub fn with_initial(total: u64, indexed: u64) -> Self {
        Self {
            total: AtomicU64::new(total),
            indexed: AtomicU64::new(indexed),
            done: AtomicBool::new(false),
            error: std::sync::Mutex::new(None),
        }
    }
}

impl IndexProgressObserver for InMemoryProgress {
    fn on_progress(&self, indexed: u64, total: u64, _failed: bool, _last_key: Option<&[u8]>) {
        self.total.store(total, Ordering::Relaxed);
        self.indexed.store(indexed, Ordering::Relaxed);
    }

    fn on_status(&self, status: BuildStatus, error: Option<&str>) {
        if status.is_terminal() {
            self.done.store(true, Ordering::Relaxed);
            if let Some(msg) = error {
                *self.error.lock().unwrap() = Some(msg.to_owned());
            }
        }
    }
}

// ── DiskProgress ─────────────────────────────────────────────────────────────

/// Throttled disk-backed observer that atomically writes `build_progress.json`
/// every `every_n` documents (tmp-then-rename for crash-safety).
///
/// Replaces the ad-hoc `write_disk_progress` calls scattered through
/// `rebuild_index_for_namespace`.
pub struct DiskProgress {
    pub path: PathBuf,
    /// Flush interval in documents.  Flush on every `indexed % every_n == 0`.
    pub every_n: u64,
}

impl DiskProgress {
    pub fn new(path: impl AsRef<Path>, every_n: u64) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            every_n,
        }
    }

    fn write(&self, progress: &DiskBuildProgress) {
        let tmp = self.path.with_extension("tmp");
        if let Ok(bytes) = serde_json::to_vec(progress) {
            let _ = std::fs::write(&tmp, &bytes);
            let _ = std::fs::rename(&tmp, &self.path);
        }
    }
}

impl IndexProgressObserver for DiskProgress {
    fn on_progress(&self, indexed: u64, total: u64, _failed: bool, last_key: Option<&[u8]>) {
        if indexed.is_multiple_of(self.every_n) {
            self.write(&DiskBuildProgress {
                status: "in_progress".to_owned(),
                total,
                indexed,
                last_key_hex: last_key.map(bytes_to_hex),
                error: None,
            });
        }
    }

    fn on_status(&self, status: BuildStatus, error: Option<&str>) {
        let status_str = match status {
            BuildStatus::Running => "in_progress",
            BuildStatus::Complete => "complete",
            BuildStatus::Failed => "failed",
            BuildStatus::Pending => "pending",
        };
        self.write(&DiskBuildProgress {
            status: status_str.to_owned(),
            total: 0,
            indexed: 0,
            last_key_hex: None,
            error: error.map(str::to_owned),
        });
    }
}
