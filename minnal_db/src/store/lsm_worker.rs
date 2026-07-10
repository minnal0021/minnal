//! LSM Compaction Worker
//!
//! This module provides background compaction for the LSM tree.
//! It triggers compaction on a schedule or on demand.

use crate::db::error::Result;
use log::{debug, error, info};
use std::sync::{Arc, Weak};
use std::time::Duration;
use tokio::sync::{Notify, mpsc};
use tokio::time;

/// Trait for types that support LSM compaction.
pub trait LsmCompactionTarget: Send + Sync + 'static {
    fn is_closed(&self) -> bool;
    fn has_lsm_compaction_work(&self) -> bool;
    fn compact_lsm(&self) -> Result<()>;
}

/// Commands that can be sent to the LSM compaction worker
pub enum LsmCompactionCommand {
    /// Trigger an immediate compaction check
    Trigger,
    /// Shutdown the worker gracefully
    Shutdown,
}

/// LSM compaction worker that runs in the background
pub struct LsmCompactionWorker {
    tx: mpsc::UnboundedSender<LsmCompactionCommand>,
    shutdown_notify: Arc<Notify>,
}

impl LsmCompactionWorker {
    /// Create and spawn a new LSM compaction worker
    pub fn new<T: LsmCompactionTarget>(target: Arc<T>, check_interval: Duration) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let shutdown_notify = Arc::new(Notify::new());

        tokio::spawn(Self::worker_loop(Arc::downgrade(&target), rx, shutdown_notify.clone(), check_interval));

        Self { tx, shutdown_notify }
    }

    /// Send a command to trigger immediate compaction
    #[allow(dead_code)]
    pub fn trigger_compaction(&self) -> std::result::Result<(), mpsc::error::SendError<LsmCompactionCommand>> {
        self.tx.send(LsmCompactionCommand::Trigger)
    }

    /// Clone the internal sender for external triggers
    pub fn sender(&self) -> mpsc::UnboundedSender<LsmCompactionCommand> {
        self.tx.clone()
    }

    /// Shutdown the worker gracefully
    pub async fn shutdown(&self) {
        let _ = self.tx.send(LsmCompactionCommand::Shutdown);
        self.shutdown_notify.notified().await;
    }

    async fn worker_loop<T: LsmCompactionTarget>(
        target: Weak<T>,
        mut rx: mpsc::UnboundedReceiver<LsmCompactionCommand>,
        shutdown_notify: Arc<Notify>,
        check_interval: Duration,
    ) {
        info!("[LsmCompactionWorker] started (interval={}ms)", check_interval.as_millis());
        let mut interval = time::interval(check_interval);
        interval.set_missed_tick_behavior(time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    match target.upgrade() {
                        Some(t) => Self::perform_compaction_check(&t),
                        None => break,
                    }
                }
                Some(cmd) = rx.recv() => {
                    match cmd {
                        LsmCompactionCommand::Trigger => {
                            debug!("[LsmCompactionWorker] Triggered immediate compaction check");
                            match target.upgrade() {
                                Some(t) => Self::perform_compaction_check(&t),
                                None => break,
                            }
                        }
                        LsmCompactionCommand::Shutdown => {
                            info!("[LsmCompactionWorker] Shutting down");
                            break;
                        }
                    }
                }
                else => {
                    info!("[LsmCompactionWorker] Channel closed, shutting down");
                    break;
                }
            }
        }

        info!("[LsmCompactionWorker] stopped");
        shutdown_notify.notify_one();
    }

    fn perform_compaction_check<T: LsmCompactionTarget>(target: &Arc<T>) {
        if target.is_closed() {
            return;
        }

        if !target.has_lsm_compaction_work() {
            info!("[LsmCompactionWorker] tick — memtable below flush threshold, no level-0 files; nothing to compact");
            return;
        }

        info!("[LsmCompactionWorker] tick — starting compaction");
        let start = std::time::Instant::now();
        match target.compact_lsm() {
            Ok(()) => info!("[LsmCompactionWorker] compaction complete in {:?}", start.elapsed()),
            Err(err) => error!("[LsmCompactionWorker] compaction failed: {:?}", err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::config::{DbConfig, ScheduledTaskConfig, SyncConfig, ThresholdConfig};
    use crate::db::facade::Db;
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
    async fn test_lsm_compaction_worker_creation() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let db = Arc::new(Db::open_with_config(temp_dir.path(), create_db_config()).expect("failed to open db"));

        let worker = LsmCompactionWorker::new(db, Duration::from_secs(1));
        worker.shutdown().await;
    }

    #[tokio::test]
    async fn test_lsm_compaction_worker_trigger() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let db = Arc::new(Db::open_with_config(temp_dir.path(), create_db_config()).expect("failed to open db"));

        db.put(b"key1", b"value1").expect("failed to put key1");

        let worker = LsmCompactionWorker::new(db, Duration::from_secs(10));
        worker.trigger_compaction().expect("failed to trigger compaction");
        tokio::time::sleep(Duration::from_millis(100)).await;
        worker.shutdown().await;
    }
}
