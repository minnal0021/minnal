//! Test helper: does a background worker's blocking pass stall the async runtime?
//!
//! A worker that runs file I/O (fsyncs, compactions) directly inside its Tokio
//! task occupies a runtime thread for the whole pass, and tasks scheduled on that
//! thread wait for it. M5 measured this on the index checkpoint: every 1.75 s,
//! ~55 ms of fsyncs held up the requests queued behind it (get p99 4-10 ms
//! against a 1.3 ms median). Workers must run such passes with `spawn_blocking`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// How long a slow test target blocks inside its pass.
pub(crate) const BLOCK_FOR: Duration = Duration::from_millis(300);

/// Marks a slow test target as inside its blocking pass.
#[derive(Default)]
pub(crate) struct Busy(AtomicBool);

impl Busy {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Call from the target's pass: blocks the calling thread for [`BLOCK_FOR`],
    /// the way a pass of fsyncs does.
    pub(crate) fn block(&self) {
        self.0.store(true, Ordering::SeqCst);
        std::thread::sleep(BLOCK_FOR);
        self.0.store(false, Ordering::SeqCst);
    }
}

/// Wait until the target is inside its pass, then spawn a task and assert it
/// ran promptly. Use on a runtime with one worker thread
/// (`#[tokio::test(flavor = "multi_thread", worker_threads = 1)]`): a pass run on
/// that thread holds the task for most of [`BLOCK_FOR`].
pub(crate) async fn assert_runtime_free_during_pass(busy: &Busy) {
    // Polled with a thread sleep: the test body runs on the `block_on` thread,
    // not a runtime worker, and a Tokio timer needs a free worker to fire, which
    // a blocking pass would deny it.
    let deadline = Instant::now() + Duration::from_secs(10);
    while !busy.0.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "the worker never started its pass");
        std::thread::sleep(Duration::from_millis(1));
    }
    let spawned = Instant::now();
    let waited = tokio::spawn(async move { spawned.elapsed() }).await.unwrap();
    assert!(
        waited < BLOCK_FOR / 3,
        "a task spawned during the worker's pass waited {waited:?}: the pass blocks a runtime thread"
    );
}
