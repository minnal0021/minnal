//! LSM Compaction Worker
//!
//! This module provides background compaction for the LSM tree.
//! It triggers compaction on a schedule or on demand.

use crate::db::error::Result;
use log::{debug, error, info};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};
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

/// How often the worker reports what it did, when it did anything.
const SUMMARY_EVERY: Duration = Duration::from_secs(60);

/// What the worker did since its last INFO summary.
///
/// A store being written continuously (an embedding job, partition
/// maintenance) seals a memtable about every second, and each seal triggers a
/// compaction: logging every one at INFO measured **1,649 compactions plus 208
/// no-WAL flushes in 46 minutes** of FiQA indexing, two lines each. They are
/// logged at DEBUG and rolled up here into at most one INFO line per
/// [`SUMMARY_EVERY`]; a quiet worker logs nothing.
#[derive(Debug)]
struct CompactionTally {
    since: Instant,
    compactions: u32,
    total: Duration,
    longest: Duration,
    no_wal_flushes: usize,
}

impl CompactionTally {
    fn new(now: Instant) -> Self {
        Self {
            since: now,
            compactions: 0,
            total: Duration::ZERO,
            longest: Duration::ZERO,
            no_wal_flushes: 0,
        }
    }

    fn record_compaction(&mut self, took: Duration) {
        self.compactions += 1;
        self.total += took;
        self.longest = self.longest.max(took);
    }

    /// The summary line once [`SUMMARY_EVERY`] has passed (or `force`, at
    /// shutdown), resetting the tally; `None` when there is nothing to report.
    fn take_summary(&mut self, now: Instant, force: bool) -> Option<String> {
        let window = now.saturating_duration_since(self.since);
        if !force && window < SUMMARY_EVERY {
            return None;
        }
        let tally = std::mem::replace(self, Self::new(now));
        if tally.compactions == 0 && tally.no_wal_flushes == 0 {
            return None;
        }
        let mut line = format!(
            "{} compaction(s) in the last {}s, {:.1?} in total, longest {:.1?}",
            tally.compactions,
            window.as_secs(),
            tally.total,
            tally.longest
        );
        if tally.no_wal_flushes > 0 {
            line.push_str(&format!("; {} no-WAL memtable flush(es)", tally.no_wal_flushes));
        }
        Some(line)
    }
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
        let mut tally = CompactionTally::new(Instant::now());

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    match target.upgrade() {
                        Some(t) => Self::perform_compaction_check(&t, &mut tally),
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
                                Some(t) => Self::perform_compaction_check(&t, &mut tally),
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

        if let Some(line) = tally.take_summary(Instant::now(), true) {
            info!("[LsmCompactionWorker] {line}");
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

    fn perform_compaction_check<T: LsmCompactionTarget>(target: &Arc<T>, tally: &mut CompactionTally) {
        if target.is_closed() {
            return;
        }
        Self::compact_once(target, tally);
        if let Some(line) = tally.take_summary(Instant::now(), false) {
            info!("[LsmCompactionWorker] {line}");
        }
    }

    /// One check: flush no-WAL memtables, then compact if there is work.
    /// Routine outcomes go to DEBUG and into `tally`; failures are logged at
    /// ERROR straight away.
    fn compact_once<T: LsmCompactionTarget>(target: &Arc<T>, tally: &mut CompactionTally) {
        // Before deciding there is nothing to do: no-WAL writes are unrecoverable
        // once the process dies, so they must not be left sitting in a memtable
        // until it happens to fill. Flushing here also means the L0 files this
        // produces are compacted by the same tick.
        let flushed = target.flush_no_wal_memtables();
        if flushed > 0 {
            debug!("[LsmCompactionWorker] tick — flushed {flushed} namespace(s) holding un-flushed no-WAL writes");
            tally.no_wal_flushes += flushed;
        }

        if !target.has_lsm_compaction_work() {
            debug!("[LsmCompactionWorker] tick — memtable below flush threshold, no level-0 files; nothing to compact");
            return;
        }

        debug!("[LsmCompactionWorker] tick — starting compaction");
        let start = Instant::now();
        match target.compact_lsm() {
            Ok(()) => {
                let took = start.elapsed();
                debug!("[LsmCompactionWorker] compaction complete in {took:?}");
                tally.record_compaction(took);
            }
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

    /// Compactions roll up into one summary per window, and an idle window
    /// reports nothing.
    #[test]
    fn compactions_are_summarised_once_per_window() {
        let t0 = Instant::now();
        let mut tally = CompactionTally::new(t0);
        tally.record_compaction(Duration::from_millis(200));
        tally.record_compaction(Duration::from_millis(300));
        tally.no_wal_flushes += 2;
        assert_eq!(tally.take_summary(t0 + Duration::from_secs(30), false), None, "too early to report");

        let line = tally.take_summary(t0 + SUMMARY_EVERY, false).expect("a window with work reports it");
        assert!(line.starts_with("2 compaction(s) in the last 60s"), "{line}");
        assert!(line.contains("longest 300.0ms"), "{line}");
        assert!(line.contains("2 no-WAL memtable flush(es)"), "{line}");

        // The tally was reset: an idle window says nothing, even when forced.
        let later = t0 + SUMMARY_EVERY * 2;
        assert_eq!(tally.take_summary(later, false), None);
        assert_eq!(tally.take_summary(later, true), None);

        // Shutdown flushes a partial window.
        tally.record_compaction(Duration::from_millis(5));
        let line = tally.take_summary(later + Duration::from_secs(1), true).expect("forced summary");
        assert!(line.starts_with("1 compaction(s) in the last 1s"), "{line}");
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
