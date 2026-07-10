//! Write-Ahead Log Worker
//!
//! This module provides background garbage collection for the WAL.
//! It monitors the WAL and triggers GC when needed based on persisted entries.

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
    fn get_wal_gc_stats(&self) -> (u64, u64);
    /// Returns `true` if there is at least one fully-persisted WAL segment
    /// that is not the active (current) segment and can therefore be deleted.
    /// When this returns `false` the worker skips the GC run entirely.
    fn has_deletable_wal_segments(&self) -> bool;
    fn garbage_collect_wal(&self) -> Result<(u64, u64)>;
}

/// Commands that can be sent to the WAL GC worker
pub enum WalGcCommand {
    /// Trigger an immediate WAL GC check
    Trigger,
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

    /// Send a command to trigger immediate WAL GC
    pub fn trigger_gc(&self) -> std::result::Result<(), mpsc::error::SendError<WalGcCommand>> {
        self.tx.send(WalGcCommand::Trigger)
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
                        Some(t) => Self::perform_wal_gc_check(&t),
                        None => break,
                    }
                }

                // Handle commands
                Some(cmd) = rx.recv() => {
                    match cmd {
                        WalGcCommand::Trigger => {
                            debug!("[WalGcWorker] Triggered immediate WAL GC check");
                            match target.upgrade() {
                                Some(t) => Self::perform_wal_gc_check(&t),
                                None => break,
                            }
                        }
                        WalGcCommand::Shutdown => {
                            info!("[WalGcWorker] Shutting down");
                            break;
                        }
                    }
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

        // Only run GC when there is at least one fully-persisted non-current segment
        // to delete.  Entries that are in the active segment (not yet flushed to
        // SSTable) cannot be reclaimed by GC — they stay until the next memtable
        // flush or shutdown marks them persisted.
        if !target.has_deletable_wal_segments() {
            let (total, persisted) = target.get_wal_gc_stats();
            let pending = total.saturating_sub(persisted);
            if pending > 0 {
                info!(
                    "[WalGcWorker] tick — {} entries pending SSTable flush, no segments ready to reclaim",
                    pending
                );
            } else {
                info!("[WalGcWorker] tick — WAL clean, nothing to reclaim");
            }
            return;
        }

        info!("[WalGcWorker] tick — reclaimable segments found, starting WAL GC");
        match target.garbage_collect_wal() {
            Ok((reclaimed, remaining)) => {
                info!(
                    "[WalGcWorker] WAL GC complete — reclaimed {} bytes, {} entries remaining",
                    reclaimed, remaining
                );
            }
            Err(e) => {
                error!("[WalGcWorker] WAL GC failed: {:?}", e);
            }
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

    #[tokio::test]
    async fn test_wal_gc_worker_trigger() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let db = Arc::new(Database::open(temp_dir.path(), create_db_config()).expect("failed to open database"));

        let worker = WalGcWorker::new(db.clone(), Duration::from_secs(10));

        worker.trigger_gc().expect("failed to trigger wal gc");

        tokio::time::sleep(Duration::from_millis(100)).await;

        worker.shutdown().await;
    }
}
