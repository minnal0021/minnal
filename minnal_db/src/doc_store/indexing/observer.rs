//! Progress observer trait for index build operations.
//!
//! The field-index builder (`rebuild_index_for_namespace`) reports through this
//! trait so that persistence and in-memory state tracking are composable rather
//! than baked into the build loop.
//!
//! # Built-in implementations
//!
//! - [`InMemoryProgress`] — atomics-backed snapshot for live API queries.
//! - [`RecordProgress`] — the build's record in the maintenance namespace,
//!   saved every `every_n` documents and at each status change, for a restart.
//! - [`ChainedObserver`] — fan-out to multiple observers.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::namespace_lifecycle::{BuildStatus, FieldBuildRecord};
use crate::namespace_lifecycle::{JobKey, MaintenanceStore};
use crate::support::hex::bytes_to_hex;

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

// ── RecordProgress ───────────────────────────────────────────────────────────

/// Saves a field-index build's [`FieldBuildRecord`] in the maintenance
/// namespace: every `every_n` documents, and at each status change.
///
/// `on_progress` keeps the latest `(total, indexed, resume point)` (cheap
/// atomics) and saves every `every_n` documents. `on_status` saves a record
/// built from that remembered snapshot, so a terminal record keeps the real
/// progress instead of zeroing it.
///
/// A save is a WAL-backed write, durable when it returns. One that fails is
/// logged and the build goes on: the record it would have replaced still names
/// an earlier resume point, which only costs work on a restart.
pub struct RecordProgress {
    store: Arc<MaintenanceStore>,
    key: JobKey,
    /// Save interval in documents: on every `indexed % every_n == 0`.
    every_n: u64,
    total: AtomicU64,
    indexed: AtomicU64,
    resume_after_hex: std::sync::Mutex<Option<String>>,
}

impl RecordProgress {
    /// Saves to `key` in `store`, starting from `start` (the record a resumed
    /// build continues, or a new one).
    pub(crate) fn new(store: Arc<MaintenanceStore>, key: JobKey, every_n: u64, start: &FieldBuildRecord) -> Self {
        Self {
            store,
            key,
            every_n,
            total: AtomicU64::new(start.total),
            indexed: AtomicU64::new(start.indexed),
            resume_after_hex: std::sync::Mutex::new(start.resume_after_hex.clone()),
        }
    }

    fn snapshot(&self, status: BuildStatus, error: Option<&str>) -> FieldBuildRecord {
        FieldBuildRecord {
            status,
            total: self.total.load(Ordering::Relaxed),
            indexed: self.indexed.load(Ordering::Relaxed),
            resume_after_hex: self.resume_after_hex.lock().unwrap().clone(),
            error: error.map(str::to_owned),
        }
    }

    fn save(&self, record: &FieldBuildRecord) {
        if let Err(e) = self.store.put(self.key, record) {
            log::warn!("index build progress: could not save {:?}: {e}", self.key);
        }
    }
}

impl IndexProgressObserver for RecordProgress {
    fn on_progress(&self, indexed: u64, total: u64, _failed: bool, last_key: Option<&[u8]>) {
        self.total.store(total, Ordering::Relaxed);
        self.indexed.store(indexed, Ordering::Relaxed);
        if let Some(k) = last_key {
            *self.resume_after_hex.lock().unwrap() = Some(bytes_to_hex(k));
        }
        if indexed.is_multiple_of(self.every_n) {
            self.save(&self.snapshot(BuildStatus::Running, None));
        }
    }

    fn on_status(&self, status: BuildStatus, error: Option<&str>) {
        self.save(&self.snapshot(status, error));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AsyncDb;
    use crate::namespace::NamespaceId;
    use std::sync::atomic::Ordering;

    struct Fixture {
        _dir: tempfile::TempDir,
        _db: AsyncDb,
        store: Arc<MaintenanceStore>,
        key: JobKey,
    }

    async fn fixture() -> Fixture {
        let dir = tempfile::TempDir::new().unwrap();
        let db = AsyncDb::open_with_config(dir.path().to_path_buf(), crate::support::test_db_config())
            .await
            .unwrap();
        let store = Arc::new(MaintenanceStore::open(&db).await.unwrap());
        let key = JobKey::field_build(NamespaceId::random().unwrap(), 3);
        Fixture {
            _dir: dir,
            _db: db,
            store,
            key,
        }
    }

    impl Fixture {
        fn observer(&self, every_n: u64) -> RecordProgress {
            RecordProgress::new(Arc::clone(&self.store), self.key, every_n, &FieldBuildRecord::started())
        }

        fn record(&self) -> Option<FieldBuildRecord> {
            self.store.get(self.key).unwrap()
        }
    }

    /// A terminal record must preserve the latest `(total, indexed, resume
    /// point)` fed via `on_progress` rather than zeroing them.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_terminal_record_keeps_the_latest_progress() {
        let f = fixture().await;
        let obs = f.observer(1_000);
        // Not a multiple of `every_n`, so nothing is saved here: the terminal
        // record must still carry these values.
        obs.on_progress(523, 1_000, false, Some(b"row-523"));
        obs.on_status(BuildStatus::Complete, None);
        let rec = f.record().unwrap();
        assert_eq!(rec.status, BuildStatus::Complete);
        assert_eq!((rec.total, rec.indexed), (1_000, 523));
        assert_eq!(rec.resume_after_hex, Some(bytes_to_hex(b"row-523")));
        assert_eq!(rec.error, None);
    }

    /// On failure the progress is preserved *and* the error recorded.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_failed_record_keeps_the_progress_and_the_error() {
        let f = fixture().await;
        let obs = f.observer(1_000);
        obs.on_progress(42, 100, false, Some(b"row-42"));
        obs.on_status(BuildStatus::Failed, Some("boom"));
        let rec = f.record().unwrap();
        assert_eq!(rec.status, BuildStatus::Failed);
        assert_eq!((rec.total, rec.indexed), (100, 42));
        assert_eq!(rec.error.as_deref(), Some("boom"));
    }

    /// `on_progress` saves on the `every_n` boundary with the current counters.
    #[tokio::test(flavor = "multi_thread")]
    async fn progress_is_saved_every_n_documents() {
        let f = fixture().await;
        let obs = f.observer(10);
        obs.on_progress(7, 50, false, Some(b"a"));
        assert_eq!(f.record(), None, "not a multiple of every_n: nothing saved");
        obs.on_progress(10, 50, false, Some(b"b"));
        let rec = f.record().unwrap();
        assert!(rec.is_running());
        assert_eq!((rec.total, rec.indexed), (50, 10));
        assert_eq!(rec.resume_after_hex, Some(bytes_to_hex(b"b")));
    }

    /// A progress call with no new resume point keeps the last one: the build
    /// names a key only once an index checkpoint covers it.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_save_without_a_new_resume_point_keeps_the_last_one() {
        let f = fixture().await;
        let obs = f.observer(10);
        obs.on_progress(10, 50, false, Some(b"covered"));
        obs.on_progress(20, 50, false, None);
        assert_eq!(f.record().unwrap().resume_after_hex, Some(bytes_to_hex(b"covered")));
    }

    /// `ChainedObserver` must fan a terminal status out to *every* member, so a
    /// failure reaches the record and not just the in-memory snapshot.
    #[tokio::test(flavor = "multi_thread")]
    async fn chained_observer_fans_status_to_all_members() {
        let f = fixture().await;
        let mem = Arc::new(InMemoryProgress::new());
        let rec = Arc::new(f.observer(1_000));
        let chain = ChainedObserver(vec![
            Arc::clone(&mem) as Arc<dyn IndexProgressObserver>,
            rec as Arc<dyn IndexProgressObserver>,
        ]);
        chain.on_progress(5, 10, false, Some(b"k"));
        chain.on_status(BuildStatus::Failed, Some("nope"));
        assert!(mem.done.load(Ordering::Relaxed));
        assert_eq!(mem.error.lock().unwrap().as_deref(), Some("nope"));
        let rec = f.record().unwrap();
        assert_eq!(rec.status, BuildStatus::Failed);
        assert_eq!(rec.indexed, 5);
        assert_eq!(rec.error.as_deref(), Some("nope"));
    }
}
