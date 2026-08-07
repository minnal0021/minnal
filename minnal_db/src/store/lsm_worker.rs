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
    /// Flush namespaces whose no-WAL writes exist only in memory, bounding what
    /// a crash destroys to one tick. Returns how many were flushed.
    fn flush_no_wal_memtables(&self) -> usize;
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
                    if Self::drain_self_triggers(&mut rx) {
                        break;
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
                            if Self::drain_self_triggers(&mut rx) {
                                break;
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

    /// Discard `Trigger`s that this tick's own work produced. Returns whether a
    /// `Shutdown` was seen, which must never be dropped.
    ///
    /// [`perform_compaction_check`] flushes no-WAL memtables, and sealing a
    /// memtable fires the flush observer — which sends a `Trigger`. Left in the
    /// channel, that makes the worker immediately re-run on its own side effect,
    /// and while no-WAL writes keep arriving (the vector index, continuously) it
    /// never stops: measured **147 ticks in 158 s against a 60 s interval**, each
    /// one rewriting every bucket's whole L1.
    ///
    /// Dropping them is safe because the tick has just compacted: anything queued
    /// during it is already done. A genuine seal that lands after this drain
    /// still has its own `Trigger`, and the interval tick remains the backstop.
    ///
    /// [`perform_compaction_check`]: Self::perform_compaction_check
    fn drain_self_triggers(rx: &mut mpsc::UnboundedReceiver<LsmCompactionCommand>) -> bool {
        let mut shutdown = false;
        while let Ok(cmd) = rx.try_recv() {
            if matches!(cmd, LsmCompactionCommand::Shutdown) {
                shutdown = true;
            }
        }
        shutdown
    }

    fn perform_compaction_check<T: LsmCompactionTarget>(target: &Arc<T>) {
        if target.is_closed() {
            return;
        }

        // Before deciding there is nothing to do: no-WAL writes are unrecoverable
        // once the process dies, so they must not be left sitting in a memtable
        // until it happens to fill. Flushing here also means the L0 files this
        // produces are compacted by the same tick.
        let flushed = target.flush_no_wal_memtables();
        if flushed > 0 {
            info!("[LsmCompactionWorker] tick — flushed {flushed} namespace(s) holding un-flushed no-WAL writes");
        }

        if !target.has_lsm_compaction_work() {
            // Routine "nothing to do" tick: DEBUG, not INFO. INFO is reserved
            // for ticks that actually compact or flush something.
            debug!("[LsmCompactionWorker] tick — memtable below flush threshold, no level-0 files; nothing to compact");
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

    /// Regression: the worker must not re-run on its own side effects.
    ///
    /// `perform_compaction_check` flushes no-WAL memtables; sealing a memtable
    /// fires the flush observer, and the observer sends a `Trigger` — to this
    /// very worker. Left in the channel, the worker picks it straight back up
    /// and runs again, and while no-WAL writes keep arriving (the vector index
    /// writes continuously) it never settles. Caught on a live server: **147
    /// compaction ticks in 158 s against a 60 s interval**, each one rewriting
    /// every bucket's whole L1.
    ///
    /// The target here does exactly what a real seal does — send a `Trigger`
    /// back — so the loop is reproduced rather than imitated.
    #[tokio::test]
    async fn worker_does_not_retrigger_itself_after_a_no_wal_flush() {
        use std::sync::atomic::{AtomicU64, Ordering};

        struct SelfTriggering {
            compactions: AtomicU64,
            tx: mpsc::UnboundedSender<LsmCompactionCommand>,
        }

        impl LsmCompactionTarget for SelfTriggering {
            fn is_closed(&self) -> bool {
                false
            }
            fn has_lsm_compaction_work(&self) -> bool {
                true
            }
            fn compact_lsm(&self) -> Result<()> {
                self.compactions.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn flush_no_wal_memtables(&self) -> usize {
                // What sealing a memtable really does: notify the observer,
                // which triggers this worker.
                let _ = self.tx.send(LsmCompactionCommand::Trigger);
                1
            }
        }

        let (tx, rx) = mpsc::unbounded_channel();
        let target = Arc::new(SelfTriggering {
            compactions: AtomicU64::new(0),
            tx: tx.clone(),
        });
        let notify = Arc::new(Notify::new());
        let interval = Duration::from_millis(50);
        tokio::spawn(LsmCompactionWorker::worker_loop(
            Arc::downgrade(&target),
            rx,
            Arc::clone(&notify),
            interval,
        ));

        // Prime it once, as a write-path memtable seal would.
        tx.send(LsmCompactionCommand::Trigger).unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;
        let ran = target.compactions.load(Ordering::SeqCst);

        tx.send(LsmCompactionCommand::Shutdown).unwrap();
        notify.notified().await;

        // 500 ms at a 50 ms interval is ~10 ticks, plus the one explicit
        // trigger. Self-retriggering makes this unbounded — thousands.
        assert!(
            ran <= 25,
            "worker ran {ran} compactions in 500 ms at a 50 ms interval — it is re-running on \
             its own side effects, which on a live server meant 147 ticks in 158 s"
        );
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
