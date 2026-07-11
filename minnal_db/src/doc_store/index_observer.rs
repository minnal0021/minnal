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

use crate::doc_store::hex::bytes_to_hex;
use crate::doc_store::index_progress::BuildStatus;
use crate::doc_store::store::DiskBuildProgress;

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
/// (tmp-then-rename for crash-safety).  This is the **single source of truth**
/// for on-disk build progress — it fully replaces the ad-hoc `write_disk_progress`
/// calls that used to be scattered through `rebuild_index_for_namespace`.
///
/// `on_progress` keeps the latest `(total, indexed, last_key)` current on every
/// call (cheap atomics) and flushes to disk every `every_n` documents.
/// `on_status` writes a record built from that remembered snapshot, so a
/// terminal `complete`/`failed` record preserves the real progress instead of
/// zeroing it.
pub struct DiskProgress {
    pub path: PathBuf,
    /// Flush interval in documents.  Flush on every `indexed % every_n == 0`.
    pub every_n: u64,
    /// Latest observed counters, so status/terminal writes carry the most recent
    /// progress rather than starting from zero.
    total: AtomicU64,
    indexed: AtomicU64,
    last_key_hex: std::sync::Mutex<Option<String>>,
}

impl DiskProgress {
    pub fn new(path: impl AsRef<Path>, every_n: u64) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            every_n,
            total: AtomicU64::new(0),
            indexed: AtomicU64::new(0),
            last_key_hex: std::sync::Mutex::new(None),
        }
    }

    /// Build a record from the latest remembered counters.
    fn snapshot(&self, status: &str, error: Option<&str>) -> DiskBuildProgress {
        DiskBuildProgress {
            status: status.to_owned(),
            total: self.total.load(Ordering::Relaxed),
            indexed: self.indexed.load(Ordering::Relaxed),
            last_key_hex: self.last_key_hex.lock().unwrap().clone(),
            error: error.map(str::to_owned),
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
        self.total.store(total, Ordering::Relaxed);
        self.indexed.store(indexed, Ordering::Relaxed);
        if let Some(k) = last_key {
            *self.last_key_hex.lock().unwrap() = Some(bytes_to_hex(k));
        }
        if indexed.is_multiple_of(self.every_n) {
            self.write(&self.snapshot("in_progress", None));
        }
    }

    fn on_status(&self, status: BuildStatus, error: Option<&str>) {
        let status_str = match status {
            BuildStatus::Running => "in_progress",
            BuildStatus::Complete => "complete",
            BuildStatus::Failed => "failed",
            BuildStatus::Pending => "pending",
        };
        self.write(&self.snapshot(status_str, error));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc_store::store::DiskBuildProgress;
    use std::sync::atomic::Ordering;

    fn read_back(path: &Path) -> DiskBuildProgress {
        let bytes = std::fs::read(path).unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// A terminal record must preserve the latest `(total, indexed, last_key)`
    /// fed via `on_progress` rather than zeroing them — the bug this guards.
    #[test]
    fn disk_progress_terminal_preserves_latest_snapshot() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("build_progress.json");
        let disk = DiskProgress::new(&path, 1_000);

        // A non-multiple of `every_n`, so no flush happens here — the terminal
        // write must still carry these values.
        disk.on_progress(523, 1_000, false, Some(b"row-523"));
        disk.on_status(BuildStatus::Complete, None);

        let rec = read_back(&path);
        assert_eq!(rec.status, "complete");
        assert_eq!(rec.total, 1_000);
        assert_eq!(rec.indexed, 523);
        assert_eq!(rec.last_key_hex, Some(bytes_to_hex(b"row-523")));
        assert_eq!(rec.error, None);
    }

    /// On failure the snapshot is preserved *and* the error message recorded.
    #[test]
    fn disk_progress_failed_preserves_snapshot_and_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("build_progress.json");
        let disk = DiskProgress::new(&path, 1_000);

        disk.on_progress(42, 100, false, Some(b"row-42"));
        disk.on_status(BuildStatus::Failed, Some("boom"));

        let rec = read_back(&path);
        assert_eq!(rec.status, "failed");
        assert_eq!(rec.total, 100);
        assert_eq!(rec.indexed, 42);
        assert_eq!(rec.last_key_hex, Some(bytes_to_hex(b"row-42")));
        assert_eq!(rec.error.as_deref(), Some("boom"));
    }

    /// `on_progress` flushes on the `every_n` boundary with current counters.
    #[test]
    fn disk_progress_flushes_on_interval() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("build_progress.json");
        let disk = DiskProgress::new(&path, 10);

        disk.on_progress(7, 50, false, Some(b"a")); // not a multiple → no flush
        assert!(!path.exists());

        disk.on_progress(10, 50, false, Some(b"b")); // multiple → flush
        let rec = read_back(&path);
        assert_eq!(rec.status, "in_progress");
        assert_eq!(rec.indexed, 10);
        assert_eq!(rec.total, 50);
        assert_eq!(rec.last_key_hex, Some(bytes_to_hex(b"b")));
    }

    /// `ChainedObserver` must fan a terminal status out to *every* member, so a
    /// failure reaches the disk observer and not just the in-memory one.
    #[test]
    fn chained_observer_fans_status_to_all_members() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("build_progress.json");
        let mem = Arc::new(InMemoryProgress::new());
        let disk = Arc::new(DiskProgress::new(&path, 1_000));
        let chain = ChainedObserver(vec![
            Arc::clone(&mem) as Arc<dyn IndexProgressObserver>,
            Arc::clone(&disk) as Arc<dyn IndexProgressObserver>,
        ]);

        chain.on_progress(5, 10, false, Some(b"k"));
        chain.on_status(BuildStatus::Failed, Some("nope"));

        // In-memory link saw the failure.
        assert!(mem.done.load(Ordering::Relaxed));
        assert_eq!(mem.error.lock().unwrap().as_deref(), Some("nope"));

        // Disk link persisted it with the snapshot intact.
        let rec = read_back(&path);
        assert_eq!(rec.status, "failed");
        assert_eq!(rec.indexed, 5);
        assert_eq!(rec.error.as_deref(), Some("nope"));
    }
}
