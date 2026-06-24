//! Index Checkpoint Worker
//!
//! Periodically snapshots in-memory field indices to disk so that crash
//! recovery only needs to replay a bounded tail of the WAL.
//!
//! Uses the same trait-target pattern as [`WalGcWorker`]: the worker knows
//! nothing about [`IndexManager`] or [`NamespaceRegistry`] directly — it
//! calls through [`IndexCheckpointTarget`], which [`Database`] implements.
//!
//! Default interval: 15 minutes.

use std::sync::{Arc, Weak};
use std::time::Duration;

use log::{error, info};
use tokio::sync::{Notify, mpsc};
use tokio::time;

use crate::db::error::Result;

/// Default snapshot interval: 15 minutes.
pub const DEFAULT_CHECKPOINT_INTERVAL: Duration = Duration::from_secs(15 * 60);

/// Trait implemented by the database coordinator to perform an index checkpoint.
///
/// Separates the worker's scheduling logic from the mechanics of collecting
/// field metadata and writing checkpoint files.
pub trait IndexCheckpointTarget: Send + Sync + 'static {
    /// Returns `true` if the database has been closed and no further
    /// checkpoints should be attempted.
    fn is_closed(&self) -> bool;

    /// Run a full index checkpoint: collect all registered fields from the
    /// namespace registry and write their checkpoint markers via the index
    /// manager.  Returns the number of fields checkpointed.
    fn run_index_checkpoint(&self) -> Result<usize>;
}

/// Commands accepted by the checkpoint worker.
pub enum IndexCheckpointCommand {
    /// Trigger an immediate checkpoint without waiting for the next tick.
    TriggerNow,
    /// Shut down the worker gracefully.
    Shutdown,
}

/// Background worker that periodically checkpoints index state to disk.
///
/// Spawns a Tokio task on construction.  Call [`shutdown`] to stop it cleanly.
pub struct IndexCheckpointWorker {
    tx: mpsc::UnboundedSender<IndexCheckpointCommand>,
    shutdown_notify: Arc<Notify>,
}

impl IndexCheckpointWorker {
    /// Spawn a new checkpoint worker.
    ///
    /// # Arguments
    /// * `target`   - The checkpoint target (typically `Arc<Database>`).
    /// * `interval` - How often to run checkpoints (use [`DEFAULT_CHECKPOINT_INTERVAL`]).
    pub fn new<T: IndexCheckpointTarget>(target: Arc<T>, interval: Duration) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let shutdown_notify = Arc::new(Notify::new());

        tokio::spawn(Self::worker_loop(Arc::downgrade(&target), rx, shutdown_notify.clone(), interval));

        Self { tx, shutdown_notify }
    }

    /// Trigger an immediate checkpoint outside of the normal schedule.
    pub fn trigger(&self) -> std::result::Result<(), mpsc::error::SendError<IndexCheckpointCommand>> {
        self.tx.send(IndexCheckpointCommand::TriggerNow)
    }

    /// Shut down the worker and wait for it to exit.
    pub async fn shutdown(&self) {
        let _ = self.tx.send(IndexCheckpointCommand::Shutdown);
        self.shutdown_notify.notified().await;
    }

    // ── internals ─────────────────────────────────────────────────────────

    async fn worker_loop<T: IndexCheckpointTarget>(
        target: Weak<T>,
        mut rx: mpsc::UnboundedReceiver<IndexCheckpointCommand>,
        shutdown_notify: Arc<Notify>,
        interval: Duration,
    ) {
        info!("[IndexCheckpointWorker] started (interval={}s)", interval.as_secs());
        let mut ticker = time::interval(interval);
        ticker.set_missed_tick_behavior(time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    match target.upgrade() {
                        Some(t) => Self::perform_checkpoint(&t),
                        None => break,
                    }
                }

                Some(cmd) = rx.recv() => {
                    match cmd {
                        IndexCheckpointCommand::TriggerNow => {
                            info!("[IndexCheckpointWorker] immediate checkpoint triggered");
                            match target.upgrade() {
                                Some(t) => Self::perform_checkpoint(&t),
                                None => break,
                            }
                        }
                        IndexCheckpointCommand::Shutdown => {
                            info!("[IndexCheckpointWorker] Shutting down");
                            break;
                        }
                    }
                }

                else => {
                    info!("[IndexCheckpointWorker] Channel closed, shutting down");
                    break;
                }
            }
        }

        info!("[IndexCheckpointWorker] stopped");
        shutdown_notify.notify_one();
    }

    fn perform_checkpoint<T: IndexCheckpointTarget>(target: &Arc<T>) {
        if target.is_closed() {
            return;
        }
        info!("[IndexCheckpointWorker] tick — starting index checkpoint");
        let start = std::time::Instant::now();
        match target.run_index_checkpoint() {
            Ok(count) => info!(
                "[IndexCheckpointWorker] checkpoint complete in {:?} — {} field(s) flushed",
                start.elapsed(),
                count
            ),
            Err(e) => error!("[IndexCheckpointWorker] checkpoint failed: {:?}", e),
        }
    }
}

// ── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    struct FakeTarget {
        closed: AtomicBool,
        checkpoint_count: AtomicU64,
    }

    impl FakeTarget {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                closed: AtomicBool::new(false),
                checkpoint_count: AtomicU64::new(0),
            })
        }
    }

    impl IndexCheckpointTarget for FakeTarget {
        fn is_closed(&self) -> bool {
            self.closed.load(Ordering::SeqCst)
        }
        fn run_index_checkpoint(&self) -> Result<usize> {
            self.checkpoint_count.fetch_add(1, Ordering::SeqCst);
            Ok(0)
        }
    }

    #[tokio::test]
    async fn test_worker_starts_and_shuts_down() {
        let target = FakeTarget::new();
        let worker = IndexCheckpointWorker::new(target, Duration::from_secs(3600));
        worker.shutdown().await;
    }

    #[tokio::test]
    async fn test_trigger_calls_checkpoint() {
        let target = FakeTarget::new();
        let worker = IndexCheckpointWorker::new(Arc::clone(&target), Duration::from_secs(3600));

        worker.trigger().unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        worker.shutdown().await;

        assert!(target.checkpoint_count.load(Ordering::SeqCst) >= 1);
    }

    #[tokio::test]
    async fn test_skips_checkpoint_when_closed() {
        let target = FakeTarget::new();
        target.closed.store(true, Ordering::SeqCst);
        let worker = IndexCheckpointWorker::new(Arc::clone(&target), Duration::from_secs(3600));

        worker.trigger().unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        worker.shutdown().await;

        assert_eq!(target.checkpoint_count.load(Ordering::SeqCst), 0);
    }
}
