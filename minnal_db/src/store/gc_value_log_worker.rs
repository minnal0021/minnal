//! Garbage Collection Worker
//!
//! This module provides background garbage collection for the value log.
//! It monitors the value log and triggers GC when needed based on waste ratio.

use log::{debug, info};
use std::sync::{Arc, Weak};
use std::time::Duration;
use tokio::sync::{Notify, mpsc};
use tokio::time;

/// Trait for types that support value-log garbage collection.
pub trait ValueLogGcTarget: Send + Sync + 'static {
    fn is_closed(&self) -> bool;
    /// Run GC on any namespace whose waste ratio exceeds `waste_threshold`.
    fn run_gc_if_needed(&self, waste_threshold: f64);
}

/// Commands that can be sent to the GC worker
pub enum GCCommand {
    /// Trigger an immediate GC check
    #[allow(dead_code)]
    Trigger,
    /// Shutdown the worker gracefully
    Shutdown,
}

/// Garbage collection worker that runs in the background
pub struct GCWorker {
    tx: mpsc::UnboundedSender<GCCommand>,
    shutdown_notify: Arc<Notify>,
}

impl GCWorker {
    /// Create and spawn a new GC worker
    ///
    /// # Arguments
    /// * `target` - The GC target
    /// * `check_interval` - How often to check if GC is needed
    /// * `waste_threshold` - GC triggers when waste ratio exceeds this (0-100)
    pub fn new<T: ValueLogGcTarget>(target: Arc<T>, check_interval: Duration, waste_threshold: f64) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let shutdown_notify = Arc::new(Notify::new());

        tokio::spawn(Self::worker_loop(
            Arc::downgrade(&target),
            rx,
            shutdown_notify.clone(),
            check_interval,
            waste_threshold,
        ));

        Self { tx, shutdown_notify }
    }

    /// Send a command to trigger immediate GC
    #[allow(dead_code)]
    pub fn trigger_gc(&self) -> Result<(), mpsc::error::SendError<GCCommand>> {
        self.tx.send(GCCommand::Trigger)
    }

    /// Shutdown the worker gracefully
    pub async fn shutdown(&self) {
        let _ = self.tx.send(GCCommand::Shutdown);
        self.shutdown_notify.notified().await;
    }

    async fn worker_loop<T: ValueLogGcTarget>(
        target: Weak<T>,
        mut rx: mpsc::UnboundedReceiver<GCCommand>,
        shutdown_notify: Arc<Notify>,
        check_interval: Duration,
        waste_threshold: f64,
    ) {
        info!(
            "[GCWorker] started (interval={}s, threshold={:.1}%)",
            check_interval.as_secs(),
            waste_threshold
        );
        let mut interval = time::interval(check_interval);
        interval.set_missed_tick_behavior(time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    match target.upgrade() {
                        Some(t) => Self::perform_gc_check(&t, waste_threshold),
                        None => break,
                    }
                }

                Some(cmd) = rx.recv() => {
                    match cmd {
                        GCCommand::Trigger => {
                            debug!("[GCWorker] Triggered immediate GC check");
                            match target.upgrade() {
                                Some(t) => Self::perform_gc_check(&t, waste_threshold),
                                None => break,
                            }
                        }
                        GCCommand::Shutdown => {
                            info!("[GCWorker] Shutting down");
                            break;
                        }
                    }
                }

                else => {
                    info!("[GCWorker] Channel closed, shutting down");
                    break;
                }
            }
        }

        info!("[GCWorker] stopped");
        shutdown_notify.notify_one();
    }

    fn perform_gc_check<T: ValueLogGcTarget>(target: &Arc<T>, waste_threshold: f64) {
        if target.is_closed() {
            return;
        }
        target.run_gc_if_needed(waste_threshold);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::config::{DbConfig, ScheduledTaskConfig, SyncConfig, ThresholdConfig};
    use crate::db::facade::Db;
    use crate::store::lsm::lsm_tree::LSMConfig;
    use std::time::Duration;
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
    async fn test_gc_worker_creation() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let db = Arc::new(Db::open_with_config(temp_dir.path(), create_db_config()).expect("failed to open db"));

        let worker = GCWorker::new(db, Duration::from_secs(1), 50.0);
        worker.shutdown().await;
    }

    #[tokio::test]
    async fn test_gc_worker_trigger() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let db = Arc::new(Db::open_with_config(temp_dir.path(), create_db_config()).expect("failed to open db"));

        db.put(b"key1", b"value1").expect("failed to put key1");
        db.put(b"key2", b"value2").expect("failed to put key2");

        let worker = GCWorker::new(db.clone(), Duration::from_secs(10), 50.0);

        worker.trigger_gc().expect("failed to trigger gc");

        tokio::time::sleep(Duration::from_millis(100)).await;

        worker.shutdown().await;
    }

    #[tokio::test]
    async fn test_gc_worker_waste_detection() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let db = Arc::new(Db::open_with_config(temp_dir.path(), create_db_config()).expect("failed to open db"));

        for i in 0..10 {
            let key = format!("key{}", i).into_bytes();
            let value = vec![0u8; 100];
            db.put(&key, &value).expect("failed to put");
        }

        for i in 0..10 {
            let key = format!("key{}", i).into_bytes();
            let value = vec![1u8; 100];
            db.put(&key, &value).expect("failed to update");
        }

        let waste_ratio = db.waste_ratio();
        debug!("Waste ratio: {:.2}%", waste_ratio);

        let worker = GCWorker::new(db.clone(), Duration::from_secs(1), 10.0);
        worker.trigger_gc().expect("failed to trigger gc");

        tokio::time::sleep(Duration::from_millis(100)).await;
        worker.shutdown().await;
    }
}
