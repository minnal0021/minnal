//! Field-Index Worker
//!
//! Applies the field-index changes writes leave queued (`kv_store/index_queue.rs`),
//! so that a write does no index work: it queues its change under its bucket
//! lock and returns. The worker is woken by the first write that queues after
//! it last started a pass ([`IndexApplyWaker`], debounced), and a periodic
//! backstop covers a missed wake. Each pass applies, per field, the changes
//! queued when it began, at most `APPLY_BATCH` per hold of the field's write
//! lock.
//!
//! Unlike the other workers it is a **dedicated thread**, not a Tokio task:
//! it can wake as often as writes arrive, and running each pass through
//! `block_in_place` would hand the runtime thread's queued tasks to another
//! thread on every wake. Uses the same trait-target pattern as the other
//! workers: the worker calls through [`IndexApplyTarget`], which `Database`
//! implements, and holds it weakly, so a dropped database stops it.
//!
//! Queries read only what has been applied, so they lag the writes the worker
//! has not reached yet; each field's queue progress (pending changes, the
//! oldest one's age, the highest applied seq) is in the index health report.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::thread::JoinHandle;
use std::time::Duration;

use log::{debug, info};
use parking_lot::{Condvar, Mutex};

/// The backstop: a pass at least this often, should a wake ever be missed.
pub(crate) const BACKSTOP_INTERVAL: Duration = Duration::from_millis(250);

/// Implemented by the database coordinator: one pass over every store.
pub trait IndexApplyTarget: Send + Sync + 'static {
    /// `true` once the database is closed: no further passes.
    fn is_closed(&self) -> bool;

    /// Apply what every field of every store had queued when the pass began.
    /// Returns the changes applied.
    fn apply_field_index_queues(&self) -> usize;
}

/// What the writers and the worker thread share.
struct Signal {
    /// A write queued changes since the worker last started a pass.
    pending: AtomicBool,
    /// The worker should exit.
    stop: AtomicBool,
    /// Guards the wait, so a wake cannot fall between the worker's check of
    /// `pending` and its wait.
    lock: Mutex<()>,
    cv: Condvar,
}

impl Signal {
    fn notify(&self) {
        let _g = self.lock.lock();
        self.cv.notify_one();
    }
}

/// The write path's handle on the worker: wakes it when a write queues a
/// change. Debounced: only the first wake after the worker starts a pass
/// signals, so a stream of writes costs one atomic swap each.
pub struct IndexApplyWaker {
    signal: Arc<Signal>,
}

impl crate::kv_store::index_set::IndexApplyWake for IndexApplyWaker {
    fn wake(&self) {
        if !self.signal.pending.swap(true, Ordering::AcqRel) {
            self.signal.notify();
        }
    }
}

/// The background field-index worker: a thread started on construction; call
/// [`shutdown`](Self::shutdown) to stop it.
pub struct IndexApplyWorker {
    signal: Arc<Signal>,
    thread: Mutex<Option<JoinHandle<()>>>,
    waker: Arc<IndexApplyWaker>,
}

impl IndexApplyWorker {
    /// Start the worker over `target`. Once woken it waits `delay`
    /// (`scheduled_tasks.index_apply_delay_ms`) before each pass, so the
    /// writes of a burst are applied together.
    pub fn new<T: IndexApplyTarget>(target: Arc<T>, delay: Duration) -> Self {
        let signal = Arc::new(Signal {
            pending: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            lock: Mutex::new(()),
            cv: Condvar::new(),
        });
        let target = Arc::downgrade(&target);
        let thread = {
            let signal = Arc::clone(&signal);
            std::thread::Builder::new()
                .name("minnal-field-index".into())
                .spawn(move || Self::run(target, signal, delay))
                .expect("spawn the field-index worker thread")
        };
        Self {
            waker: Arc::new(IndexApplyWaker { signal: Arc::clone(&signal) }),
            signal,
            thread: Mutex::new(Some(thread)),
        }
    }

    /// The handle every store wakes the worker through.
    pub fn waker(&self) -> Arc<IndexApplyWaker> {
        Arc::clone(&self.waker)
    }

    /// Stop the worker and wait for its thread to exit (after the pass it may
    /// be in). What is still queued stays queued: the shutdown's checkpoint
    /// applies it.
    pub async fn shutdown(&self) {
        self.signal.stop.store(true, Ordering::Release);
        self.signal.notify();
        let thread = self.thread.lock().take();
        if let Some(thread) = thread {
            let _ = tokio::task::spawn_blocking(move || thread.join()).await;
        }
    }

    fn run<T: IndexApplyTarget>(target: Weak<T>, signal: Arc<Signal>, delay: Duration) {
        info!("[IndexApplyWorker] started");
        loop {
            {
                let mut g = signal.lock.lock();
                if !signal.pending.load(Ordering::Acquire) && !signal.stop.load(Ordering::Acquire) {
                    signal.cv.wait_for(&mut g, BACKSTOP_INTERVAL);
                }
                // Let the burst that woke us finish queueing. Writes meanwhile
                // find `pending` set and do not signal; a shutdown does.
                if !delay.is_zero() && signal.pending.load(Ordering::Acquire) {
                    let deadline = std::time::Instant::now() + delay;
                    // Only a shutdown notifies meanwhile; the wait ends at the
                    // deadline or on it.
                    while !signal.stop.load(Ordering::Acquire) {
                        if signal.cv.wait_until(&mut g, deadline).timed_out() {
                            break;
                        }
                    }
                }
            }
            if signal.stop.load(Ordering::Acquire) {
                break;
            }
            let Some(t) = target.upgrade() else { break };
            if t.is_closed() {
                break;
            }
            // Cleared before the pass: a write that queues after this point
            // wakes the worker again, so its change cannot be left behind.
            signal.pending.store(false, Ordering::Release);
            let applied = t.apply_field_index_queues();
            drop(t);
            if applied > 0 {
                debug!("[IndexApplyWorker] applied {applied} field-index change(s)");
            }
        }
        info!("[IndexApplyWorker] stopped");
    }
}

impl Drop for IndexApplyWorker {
    /// A worker dropped without [`shutdown`](Self::shutdown) still stops its
    /// thread; it is not joined.
    fn drop(&mut self) {
        self.signal.stop.store(true, Ordering::Release);
        self.signal.notify();
    }
}
