//! The lifecycle governor: one task per [`LifecycleHost`] that runs every
//! namespace's creation and drop, supervised.
//!
//! A create or drop first writes the namespace's new state (`Creating`,
//! `Deleting`), then hands it to the governor, which runs its task
//! ([`Task::Provision`] or [`Task::Teardown`], through
//! [`LifecycleHost::run`]) in a coordinator task of its own. The governor is single-threaded: it only starts coordinators and
//! collects their results; the work is theirs.
//!
//! **Supervision.** A coordinator whose task fails or panics is run again,
//! after `restart_backoff`, up to `max_restarts` times: every task is
//! idempotent, so a run again finishes what the last one left. Past that the
//! namespace is *parked*: a CRITICAL error is logged, its state stays as it is,
//! and later requests to create or drop it fail at once, until the process
//! restarts (the escape hatch; the count is kept in memory only). The governor
//! itself is supervised the same way; if it gives up, its channel closes and
//! every create and drop fails with `GovernorUnavailable`, while namespaces that
//! exist keep working.
//!
//! At start the governor picks up every namespace that is still `Creating` or
//! `Deleting`, so a crash or a parked namespace from before a restart is finished.

use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinSet;

use crate::namespace_lifecycle::{LifecycleError, LifecycleHost, NamespaceState};

/// How the governor restarts what fails.
#[derive(Debug, Clone)]
pub struct LifecycleConfig {
    /// Runs again of a namespace's failed task before it is parked.
    pub max_restarts: u32,
    /// Wait before each run again.
    pub restart_backoff: Duration,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            max_restarts: 3,
            restart_backoff: Duration::from_secs(1),
        }
    }
}

/// What a namespace's lifecycle task does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// Set up a `Creating` namespace, then move it to `Created`.
    Provision,
    /// Remove a `Deleting` namespace, its record last.
    Teardown,
}

/// A task that ended without finishing: what to tell every caller waiting for
/// it.
#[derive(Debug, Clone)]
pub(crate) struct TaskFailure {
    attempts: u32,
    error: String,
}

type Outcome = Result<(), TaskFailure>;

struct Command {
    namespace: String,
    task: Task,
    done: Option<oneshot::Sender<Outcome>>,
}

/// The handle the host keeps: submits tasks, and stops the governor with it.
pub(crate) struct Governor {
    tx: mpsc::UnboundedSender<Command>,
    /// Namespaces parked after too many failed runs, with the last error.
    parked: Arc<parking_lot::Mutex<HashMap<String, TaskFailure>>>,
    supervisor: tokio::task::JoinHandle<()>,
}

impl Governor {
    /// Start the governor for `host`. It holds the host weakly: when the last
    /// `Arc` of it goes, so does the governor.
    pub(crate) fn start<H: LifecycleHost>(host: Weak<H>, config: LifecycleConfig) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let parked = Arc::new(parking_lot::Mutex::new(HashMap::new()));
        let rx = Arc::new(tokio::sync::Mutex::new(rx));
        let supervisor = tokio::spawn(supervise(host, rx, config, Arc::clone(&parked)));
        Self { tx, parked, supervisor }
    }

    /// Whether the governor takes tasks.
    pub(crate) fn is_running(&self) -> bool {
        !self.tx.is_closed()
    }

    /// Hand `namespace`'s `task` to the governor; the receiver says how it
    /// ended. A task already running for the namespace is joined, not started
    /// again.
    pub(crate) fn submit(&self, namespace: &str, task: Task) -> Result<oneshot::Receiver<Outcome>, LifecycleError> {
        let (done, rx) = oneshot::channel();
        self.tx
            .send(Command {
                namespace: namespace.to_owned(),
                task,
                done: Some(done),
            })
            .map_err(|_| LifecycleError::GovernorUnavailable)?;
        Ok(rx)
    }

    /// Hand `namespace`'s `task` to the governor and wait for it to end.
    pub(crate) async fn run(&self, namespace: &str, task: Task) -> Result<(), LifecycleError> {
        let outcome = self.submit(namespace, task)?.await.map_err(|_| LifecycleError::GovernorUnavailable)?;
        outcome.map_err(|f| LifecycleError::TaskFailed {
            namespace: namespace.to_owned(),
            attempts: f.attempts,
            error: f.error,
        })
    }

    /// Namespaces parked after too many failed runs, with the last error.
    pub(crate) fn parked(&self) -> Vec<(String, String)> {
        let mut parked: Vec<(String, String)> = self.parked.lock().iter().map(|(n, f)| (n.clone(), f.error.clone())).collect();
        parked.sort();
        parked
    }

    /// Stop the governor and every task it runs. A task cut short is finished
    /// at the next start: every task is idempotent.
    pub(crate) fn stop(&self) {
        self.supervisor.abort();
    }
}

impl Drop for Governor {
    fn drop(&mut self) {
        self.supervisor.abort();
    }
}

/// Run the governor, and run it again if it panics, up to `max_restarts`.
async fn supervise<H: LifecycleHost>(
    host: Weak<H>,
    rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<Command>>>,
    config: LifecycleConfig,
    parked: Arc<parking_lot::Mutex<HashMap<String, TaskFailure>>>,
) {
    let mut restarts = 0u32;
    loop {
        // Its own task, so a panic in it is a `JoinError` here, not ours; and
        // aborting this future aborts it with every coordinator it holds.
        let mut run = AbortOnDrop(tokio::spawn(govern(host.clone(), Arc::clone(&rx), config.clone(), Arc::clone(&parked))));
        match run.await_join().await {
            Ok(()) => return,
            Err(e) => {
                restarts += 1;
                if restarts > config.max_restarts {
                    log::error!(
                        "CRITICAL: the namespace lifecycle governor failed {restarts} times ({e}); no namespace can be created or dropped until the process restarts"
                    );
                    rx.lock().await.close();
                    return;
                }
                log::warn!(
                    "the namespace lifecycle governor failed ({e}); restarting it ({restarts} of {})",
                    config.max_restarts
                );
                tokio::time::sleep(config.restart_backoff).await;
            }
        }
    }
}

/// A namespace's task in progress: who waits for it.
struct Running {
    task: Task,
    waiters: Vec<oneshot::Sender<Outcome>>,
}

/// The governor proper: take commands, start coordinators, hand out results.
async fn govern<H: LifecycleHost>(
    host: Weak<H>,
    rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<Command>>>,
    config: LifecycleConfig,
    parked: Arc<parking_lot::Mutex<HashMap<String, TaskFailure>>>,
) {
    let mut rx = rx.lock().await;
    let mut running: HashMap<String, Running> = HashMap::new();
    let mut coordinators: JoinSet<(String, Outcome)> = JoinSet::new();

    // Every namespace a crash, or a parked task, left part-way.
    if let Some(h) = host.upgrade() {
        for (namespace, state) in h.states() {
            let task = match state {
                NamespaceState::Creating => Task::Provision,
                NamespaceState::Deleting => Task::Teardown,
                NamespaceState::Created => continue,
            };
            running.insert(namespace.clone(), Running { task, waiters: Vec::new() });
            coordinators.spawn(coordinate(host.clone(), namespace, task, config.clone()));
        }
    }

    loop {
        tokio::select! {
            command = rx.recv() => {
                let Some(Command { namespace, task, done }) = command else { break };
                if let Some(failure) = parked.lock().get(&namespace) {
                    if let Some(done) = done {
                        let _ = done.send(Err(failure.clone()));
                    }
                    continue;
                }
                match running.get_mut(&namespace) {
                    // The same task already runs: wait for it. (Another task
                    // cannot: a namespace is in one state at a time, and each state
                    // has one task.)
                    Some(run) if run.task == task => run.waiters.extend(done),
                    Some(_) => {
                        if let Some(done) = done {
                            let _ = done.send(Err(TaskFailure { attempts: 0, error: "another lifecycle task of this namespace is running".to_owned() }));
                        }
                    }
                    None => {
                        running.insert(namespace.clone(), Running { task, waiters: done.into_iter().collect() });
                        coordinators.spawn(coordinate(host.clone(), namespace, task, config.clone()));
                    }
                }
            }
            Some(joined) = coordinators.join_next(), if !coordinators.is_empty() => {
                // A coordinator never panics (its task runs in a task of its own).
                let Ok((namespace, outcome)) = joined else { continue };
                if let Err(failure) = &outcome {
                    parked.lock().insert(namespace.clone(), failure.clone());
                }
                if let Some(run) = running.remove(&namespace) {
                    for waiter in run.waiters {
                        let _ = waiter.send(outcome.clone());
                    }
                }
            }
        }
    }
}

/// Run `namespace`'s `task`, and run it again after a failure or a panic, up
/// to `max_restarts` times; then give up (the namespace is parked).
async fn coordinate<H: LifecycleHost>(host: Weak<H>, namespace: String, task: Task, config: LifecycleConfig) -> (String, Outcome) {
    let mut attempts = 0u32;
    loop {
        attempts += 1;
        let Some(h) = host.upgrade() else {
            return (
                namespace,
                Err(TaskFailure {
                    attempts,
                    error: "the host was closed".to_owned(),
                }),
            );
        };
        let ns = namespace.clone();
        let mut run = AbortOnDrop(tokio::spawn(async move { h.run(&ns, task).await.map_err(|e| e.to_string()) }));
        let error = match run.await_join().await {
            Ok(Ok(())) => return (namespace, Ok(())),
            Ok(Err(e)) => e,
            Err(e) => format!("the task panicked: {e}"),
        };
        if attempts > config.max_restarts {
            log::error!(
                "CRITICAL: namespace '{namespace}': its {task:?} task failed {attempts} times, last: {error}; it is left {} until the process restarts",
                host.upgrade()
                    .and_then(|h| h.state(&namespace))
                    .map_or("as it was", NamespaceState::as_str)
            );
            return (namespace, Err(TaskFailure { attempts, error }));
        }
        log::warn!(
            "namespace '{namespace}': its {task:?} task failed ({error}); running it again ({attempts} of {})",
            config.max_restarts
        );
        tokio::time::sleep(config.restart_backoff).await;
    }
}

/// A task handle that aborts its task when dropped, so stopping the governor
/// stops every task under it.
struct AbortOnDrop<T>(tokio::task::JoinHandle<T>);

impl<T> AbortOnDrop<T> {
    async fn await_join(&mut self) -> Result<T, tokio::task::JoinError> {
        (&mut self.0).await
    }
}

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}
