//! TTL Cleanup Worker
//!
//! A single background worker that periodically scans every TTL-enabled
//! namespace and tombstones records whose creation epoch has exceeded that
//! namespace's configured TTL. The value-log GC routines then reclaim the
//! physical space in subsequent runs.
//!
//! Like [`GCWorker`](crate::store::gc_value_log_worker::GCWorker) and the LSM
//! compaction worker, this is **one global task** that fans out over the
//! namespaces on each tick — not a task per namespace. The set of namespaces to
//! process (and each one's TTL and per-run delete cap) is owned by the target
//! via [`TtlTarget::run_ttl_pass`].

use log::{debug, error, info};
use std::sync::{Arc, Weak};
use std::time::Duration;
use tokio::sync::{Notify, mpsc};
use tokio::time;

/// Trait for the coordinator that knows which namespaces have a TTL and how to
/// expire them. Mirrors [`ValueLogGcTarget`](crate::store::gc_value_log_worker::ValueLogGcTarget).
pub trait TtlTarget: Send + Sync + 'static {
    /// Whether the database has been closed (the worker stops doing work).
    fn is_closed(&self) -> bool;
    /// Run one TTL pass across every TTL-enabled namespace.
    fn run_ttl_pass(&self);
}

/// Commands that can be sent to the TTL worker.
pub enum TtlCommand {
    /// Trigger an immediate TTL cleanup pass.
    Trigger,
    /// Shutdown the worker gracefully.
    Shutdown,
}

/// The single background worker that expires records across all TTL-enabled
/// namespaces.
pub struct TtlWorker {
    tx: mpsc::UnboundedSender<TtlCommand>,
    shutdown_notify: Arc<Notify>,
}

impl TtlWorker {
    /// Spawn the TTL cleanup worker.
    ///
    /// # Arguments
    /// * `target`         – The coordinator whose TTL-enabled namespaces are scanned.
    /// * `check_interval` – How often the worker wakes up (e.g. 1 hour).
    pub fn new<T: TtlTarget>(target: Arc<T>, check_interval: Duration) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let shutdown_notify = Arc::new(Notify::new());

        tokio::spawn(Self::worker_loop(Arc::downgrade(&target), rx, shutdown_notify.clone(), check_interval));

        Self { tx, shutdown_notify }
    }

    /// Send a command to trigger an immediate TTL cleanup pass.
    pub fn trigger(&self) -> Result<(), mpsc::error::SendError<TtlCommand>> {
        self.tx.send(TtlCommand::Trigger)
    }

    /// Shutdown the worker gracefully.
    pub async fn shutdown(&self) {
        let _ = self.tx.send(TtlCommand::Shutdown);
        self.shutdown_notify.notified().await;
    }

    /// Main worker loop.
    async fn worker_loop<T: TtlTarget>(
        target: Weak<T>,
        mut rx: mpsc::UnboundedReceiver<TtlCommand>,
        shutdown_notify: Arc<Notify>,
        check_interval: Duration,
    ) {
        info!("[TtlWorker] started (interval={}s)", check_interval.as_secs());
        let mut interval = time::interval(check_interval);
        interval.set_missed_tick_behavior(time::MissedTickBehavior::Skip);
        // The first tick completes immediately; skip it so the worker waits a
        // full interval before the first periodic cleanup.
        interval.tick().await;

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    match target.upgrade() {
                        Some(t) => Self::run_pass(t).await,
                        None => break,
                    }
                }

                Some(cmd) = rx.recv() => {
                    match cmd {
                        TtlCommand::Trigger => {
                            debug!("[TtlWorker] Triggered immediate TTL cleanup");
                            match target.upgrade() {
                                Some(t) => Self::run_pass(t).await,
                                None => break,
                            }
                        }
                        TtlCommand::Shutdown => {
                            info!("[TtlWorker] Shutting down");
                            break;
                        }
                    }
                }

                else => {
                    info!("[TtlWorker] Channel closed, shutting down");
                    break;
                }
            }
        }

        info!("[TtlWorker] stopped");
        shutdown_notify.notify_one();
    }

    /// Run a single TTL pass on a blocking thread (the scan/delete work can be
    /// heavy, so it must not run on the async worker thread).
    async fn run_pass<T: TtlTarget>(target: Arc<T>) {
        if target.is_closed() {
            return;
        }
        if let Err(e) = tokio::task::spawn_blocking(move || target.run_ttl_pass()).await {
            error!("[TtlWorker] TTL pass task panicked: {:?}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{TtlTarget, TtlWorker};
    use crate::db::config::SyncConfig;
    use crate::db::kv_store::KVStore;
    use crate::store::lsm::lsm_tree::LSMConfig;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tempfile::TempDir;

    /// Minimal target that just counts how many TTL passes it was asked to run,
    /// so the worker's lifecycle (spawn / trigger / shutdown) can be tested
    /// without standing up a full `Database`.
    struct CountingTarget {
        passes: AtomicUsize,
        closed: bool,
    }

    impl TtlTarget for CountingTarget {
        fn is_closed(&self) -> bool {
            self.closed
        }
        fn run_ttl_pass(&self) {
            self.passes.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn test_ttl_worker_creation_and_shutdown() {
        let target = Arc::new(CountingTarget {
            passes: AtomicUsize::new(0),
            closed: false,
        });
        let worker = TtlWorker::new(target, Duration::from_secs(3600));
        worker.shutdown().await;
    }

    #[tokio::test]
    async fn test_ttl_worker_trigger_runs_a_pass() {
        let target = Arc::new(CountingTarget {
            passes: AtomicUsize::new(0),
            closed: false,
        });
        // Long interval so only the explicit trigger fires a pass.
        let worker = TtlWorker::new(Arc::clone(&target), Duration::from_secs(3600));

        worker.trigger().unwrap();
        // FIFO mpsc: the awaited pass completes before Shutdown is processed, so
        // shutdown().await is a sufficient synchronization point.
        worker.shutdown().await;

        assert_eq!(target.passes.load(Ordering::SeqCst), 1);
    }

    fn create_test_store(dir: &TempDir) -> Arc<KVStore> {
        let lsm_config = LSMConfig {
            num_buckets: crate::support::TEST_NUM_BUCKETS,
            ..LSMConfig::default()
        };
        let sync_config = SyncConfig::default();
        let path = dir.path().join("ns_test");
        Arc::new(KVStore::open(0, "test", &path, lsm_config, sync_config).unwrap())
    }

    #[test]
    fn test_expire_records_removes_old_records() {
        let dir = TempDir::new().unwrap();
        let store = create_test_store(&dir);

        store.put_to_storage(b"key1", b"value1").unwrap();
        store.put_to_storage(b"key2", b"value2").unwrap();
        store.put_to_storage(b"key3", b"value3").unwrap();

        // TTL of 0 means everything is already expired.
        let deleted = store.expire_records(Duration::from_millis(0), 1000).unwrap();
        assert_eq!(deleted, 3);

        assert_eq!(store.get(b"key1").unwrap(), None);
        assert_eq!(store.get(b"key2").unwrap(), None);
        assert_eq!(store.get(b"key3").unwrap(), None);
    }

    #[test]
    fn test_expire_records_respects_max_deletes_cap() {
        let dir = TempDir::new().unwrap();
        let store = create_test_store(&dir);

        for i in 0..10 {
            store.put_to_storage(format!("key{:02}", i).as_bytes(), b"value").unwrap();
        }

        // Cap at 3 deletes per run.
        let deleted = store.expire_records(Duration::from_millis(0), 3).unwrap();
        assert_eq!(deleted, 3);

        let remaining = store.keys().unwrap();
        assert_eq!(remaining.len(), 7);
    }

    #[test]
    fn test_expire_records_keeps_fresh_records() {
        let dir = TempDir::new().unwrap();
        let store = create_test_store(&dir);

        store.put_to_storage(b"key1", b"value1").unwrap();

        // TTL of 1 hour — records are fresh, nothing expires.
        let deleted = store.expire_records(Duration::from_secs(3600), 1000).unwrap();
        assert_eq!(deleted, 0);

        assert_eq!(store.get(b"key1").unwrap(), Some(b"value1".to_vec()));
    }
}
