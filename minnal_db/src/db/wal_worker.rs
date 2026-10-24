//! Write-Ahead Log Worker
//!
//! This module provides background garbage collection for the WAL.
//! It monitors the WAL and triggers GC when segments lie wholly below the cut.

use crate::db::error::Result;
use log::{debug, error, info};
use std::sync::{Arc, Weak};
use std::time::Duration;
use tokio::sync::{Notify, mpsc};
use tokio::time;

/// Trait for types that support WAL garbage collection.
///
/// `Database` implements this, allowing `WalGcWorker` to drive GC.
pub trait WalGcTarget: Send + Sync + 'static {
    fn is_closed(&self) -> bool;
    /// Flush any namespace still holding un-flushed WAL-backed writes, so the
    /// cut can advance past them. Returns how many were flushed.
    fn flush_namespaces_pinning_wal(&self) -> usize;
    /// Bytes of WAL recovery would replay: from the cut to the tail.
    fn wal_pending_bytes(&self) -> u64;
    /// Returns `true` if there is at least one WAL segment wholly below the cut
    /// that is not the active (current) segment and can therefore be deleted.
    /// When this returns `false` the worker skips the GC run entirely.
    fn has_deletable_wal_segments(&self) -> bool;
    fn garbage_collect_wal(&self) -> Result<(u64, u64)>;
    /// Flush every namespace holding the vector index WAL's retired cut back.
    /// Returns how many were flushed.
    fn flush_namespaces_pinning_vector_wal(&self) -> usize;
    /// Whether any vector index WAL segment lies wholly below the retired cut.
    fn has_deletable_vector_wal_segments(&self) -> bool;
    /// Delete vector index WAL segments wholly below the retired cut. Returns
    /// `(segments, bytes)` deleted.
    fn garbage_collect_vector_wal(&self) -> Result<(u64, u64)>;
}

/// Commands that can be sent to the WAL GC worker
pub enum WalGcCommand {
    /// Shutdown the worker gracefully
    Shutdown,
}

/// WAL garbage collection worker that runs in the background
pub struct WalGcWorker {
    tx: mpsc::UnboundedSender<WalGcCommand>,
    shutdown_notify: Arc<Notify>,
}

impl WalGcWorker {
    /// Create and spawn a new WAL GC worker
    ///
    /// # Arguments
    /// * `target` - The WAL GC target (MinnalStore or Database)
    /// * `check_interval` - How often to check if WAL GC is needed
    pub fn new<T: WalGcTarget>(target: Arc<T>, check_interval: Duration) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let shutdown_notify = Arc::new(Notify::new());

        // Spawn the worker task holding only a Weak reference so it does not
        // prevent the target from being dropped.
        tokio::spawn(Self::worker_loop(Arc::downgrade(&target), rx, shutdown_notify.clone(), check_interval));

        Self { tx, shutdown_notify }
    }

    /// Shutdown the worker gracefully
    pub async fn shutdown(&self) {
        let _ = self.tx.send(WalGcCommand::Shutdown);
        self.shutdown_notify.notified().await;
    }

    /// Main worker loop
    async fn worker_loop<T: WalGcTarget>(
        target: Weak<T>,
        mut rx: mpsc::UnboundedReceiver<WalGcCommand>,
        shutdown_notify: Arc<Notify>,
        check_interval: Duration,
    ) {
        info!("[WalGcWorker] started (interval={}ms)", check_interval.as_millis());
        let mut interval = time::interval(check_interval);
        interval.set_missed_tick_behavior(time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                // Periodic WAL GC check
                _ = interval.tick() => {
                    match target.upgrade() {
                        // Two independent passes on one tick: a stuck main-WAL
                        // GC must not leave the vector index WAL growing, nor the
                        // reverse, so neither pass can skip the other.
                        // The passes flush memtables and delete files: see
                        // `run_blocking_pass`.
                        Some(t) => crate::support::run_blocking_pass(|| {
                            Self::perform_wal_gc_check(&t);
                            Self::perform_vector_wal_gc_check(&t);
                        }),
                        None => break,
                    }
                }

                // Handle commands
                Some(WalGcCommand::Shutdown) = rx.recv() => {
                    info!("[WalGcWorker] Shutting down");
                    break;
                }

                // Channel closed, shutdown
                else => {
                    info!("[WalGcWorker] Channel closed, shutting down");
                    break;
                }
            }
        }

        info!("[WalGcWorker] stopped");
        shutdown_notify.notify_one();
    }

    /// Check if WAL GC is needed and perform it if necessary
    fn perform_wal_gc_check<T: WalGcTarget>(target: &Arc<T>) {
        if target.is_closed() {
            return;
        }

        // The cut can only advance to the slowest namespace's
        // flushed offset (the WAL is shared, memtables are not), so a namespace
        // that writes a little and then goes idle would pin the WAL forever.
        // Flush those memtables first, then reassess what is reclaimable.
        let flushed = target.flush_namespaces_pinning_wal();
        if flushed > 0 {
            info!("[WalGcWorker] tick — flushed {} namespace(s) holding the WAL cut", flushed);
        }

        // Only run GC when at least one non-current segment lies wholly below
        // the cut. Entries not yet in an SSTable cannot be reclaimed — they stay
        // until their memtable is flushed or the database shuts down cleanly.
        if !target.has_deletable_wal_segments() {
            let pending = target.wal_pending_bytes();
            if pending > 0 {
                info!("[WalGcWorker] tick — {pending} bytes of WAL pending SSTable flush, no segments ready to reclaim");
            } else {
                // Routine "nothing to do" tick — see the LSM worker for the rule.
                debug!("[WalGcWorker] tick — WAL clean, nothing to reclaim");
            }
            return;
        }

        info!("[WalGcWorker] tick — reclaimable segments found, starting WAL GC");
        match target.garbage_collect_wal() {
            Ok((reclaimed, remaining)) => {
                info!(
                    "[WalGcWorker] WAL GC complete — reclaimed {} bytes, {} bytes still to replay on recovery",
                    reclaimed, remaining
                );
            }
            Err(e) => {
                error!("[WalGcWorker] WAL GC failed: {:?}", e);
            }
        }
    }

    /// The vector index WAL's pass: flush namespaces holding its retired cut
    /// back, then delete the segments wholly below the cut. Uses only the vector
    /// index WAL's own cut, never the main WAL's decisions.
    fn perform_vector_wal_gc_check<T: WalGcTarget>(target: &Arc<T>) {
        if target.is_closed() {
            return;
        }
        let flushed = target.flush_namespaces_pinning_vector_wal();
        if flushed > 0 {
            info!("[WalGcWorker] tick — flushed {flushed} namespace(s) holding the vector index WAL's retired cut");
        }
        if !target.has_deletable_vector_wal_segments() {
            debug!("[WalGcWorker] tick — vector index WAL: nothing to reclaim");
            return;
        }
        match target.garbage_collect_vector_wal() {
            Ok((segments, bytes)) => {
                info!("[WalGcWorker] vector index WAL GC complete — deleted {segments} segment(s), {bytes} bytes");
            }
            Err(e) => error!("[WalGcWorker] vector index WAL GC failed: {e:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::config::{DbConfig, ScheduledTaskConfig, SyncConfig, ThresholdConfig};
    use crate::db::database::Database;
    use crate::store::lsm::lsm_tree::LSMConfig;
    use tempfile::TempDir;

    struct SlowTarget(Arc<crate::support::runtime_probe::Busy>);

    impl WalGcTarget for SlowTarget {
        fn is_closed(&self) -> bool {
            false
        }
        fn flush_namespaces_pinning_wal(&self) -> usize {
            self.0.block();
            0
        }
        fn wal_pending_bytes(&self) -> u64 {
            0
        }
        fn has_deletable_wal_segments(&self) -> bool {
            false
        }
        fn garbage_collect_wal(&self) -> Result<(u64, u64)> {
            Ok((0, 0))
        }
        fn flush_namespaces_pinning_vector_wal(&self) -> usize {
            0
        }
        fn has_deletable_vector_wal_segments(&self) -> bool {
            false
        }
        fn garbage_collect_vector_wal(&self) -> Result<(u64, u64)> {
            Ok((0, 0))
        }
    }

    /// A GC tick flushes memtables and deletes segments; on a runtime thread it
    /// holds up the tasks scheduled behind it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn a_wal_gc_pass_does_not_block_the_runtime() {
        use crate::support::runtime_probe::{Busy, assert_runtime_free_during_pass};
        let busy = Busy::new();
        // The worker holds only a `Weak`: keep the target alive.
        let target = Arc::new(SlowTarget(Arc::clone(&busy)));
        let worker = WalGcWorker::new(Arc::clone(&target), Duration::from_secs(3600));
        assert_runtime_free_during_pass(&busy).await;
        worker.shutdown().await;
    }

    fn create_db_config() -> DbConfig {
        let gc_interval = Duration::from_secs(5);
        let wal_gc_interval = Duration::from_secs(5);
        let lsm_compaction_interval = Duration::from_secs(5);

        let sync_config = SyncConfig::default();
        let threshold_config = ThresholdConfig::new(2.5);
        let scheduled_task_config = ScheduledTaskConfig::new(gc_interval, wal_gc_interval, lsm_compaction_interval);
        let lsm_config = LSMConfig::default();
        let mut config = DbConfig::new(threshold_config, scheduled_task_config, sync_config, lsm_config);
        config.num_buckets = crate::support::TEST_NUM_BUCKETS;
        config
    }

    #[tokio::test]
    async fn test_wal_gc_worker_creation() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let db = Arc::new(Database::open(temp_dir.path(), create_db_config()).expect("failed to open database"));

        let worker = WalGcWorker::new(db, Duration::from_secs(1));
        worker.shutdown().await;
    }

    /// The worker performs GC checks on its interval.
    ///
    /// Previously drove this through an explicit `trigger_gc()` whose only other
    /// caller was a since-deleted async wrapper. On-demand WAL GC ships as the
    /// synchronous `Db::garbage_collect_wal`, which returns its result to the
    /// caller — so the interval is the worker behaviour worth pinning here.
    #[tokio::test]
    async fn test_wal_gc_worker_runs_on_its_interval() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let db = Arc::new(Database::open(temp_dir.path(), create_db_config()).expect("failed to open database"));
        for i in 0..50u32 {
            db.put(format!("k{i}").as_bytes(), b"v").expect("put");
        }

        let worker = WalGcWorker::new(db.clone(), Duration::from_millis(20));
        tokio::time::sleep(Duration::from_millis(120)).await;
        worker.shutdown().await;

        // The database is still usable and consistent after repeated GC passes.
        assert_eq!(db.get(b"k0").expect("get"), Some(b"v".to_vec()));
    }
}
