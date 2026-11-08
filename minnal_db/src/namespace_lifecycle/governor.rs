//! The lifecycle governor: one task per [`LifecycleHost`] that runs every
//! namespace's creation, drop and index job, supervised.
//!
//! A create or drop first writes the namespace's new state (`Creating`,
//! `Deleting`), then hands it to the governor, which runs its task
//! ([`Task::Provision`] or [`Task::Teardown`], through
//! [`LifecycleHost::run`]) in a coordinator task of its own. An index job is
//! first recorded and its namespace moved to `IndexMaintenance`
//! ([`Jobs::start`]), then handed over the same way as a [`Task::Job`]; when
//! it ends, its coordinator records how ([`Jobs::finish`]). A namespace runs
//! one creation or drop at a time and nothing beside it; its jobs may run
//! together (`Jobs` refuses the ones that conflict). The governor is
//! single-threaded: it only starts coordinators and collects their results;
//! the work is theirs.
//!
//! **Supervision.** A coordinator whose task fails or panics is run again,
//! after `restart_backoff`, up to `max_restarts` times: every task is
//! idempotent, so a run again finishes what the last one left. Past that a
//! job is recorded failed, with a CRITICAL error logged, and its namespace
//! goes on. A creation or a drop past that *parks* its namespace: a CRITICAL
//! error is logged, its state stays as it is, and later tasks for it fail at
//! once, until the process restarts (the escape hatch; the count is kept in
//! memory only). The governor itself is supervised the same way; if it gives
//! up, its channel closes and every create, drop and job fails with
//! `GovernorUnavailable`, while namespaces that exist keep working.
//!
//! At start the governor picks up every namespace that is still `Creating` or
//! `Deleting`, and every live job ([`Jobs::repair`]), so what a crash or a
//! parked namespace left before a restart is finished.

use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinSet;

use crate::namespace_lifecycle::{AuditEvent, JobKey, Jobs, LifecycleError, LifecycleHost, NamespaceState};

/// How the governor restarts what fails.
#[derive(Debug, Clone)]
pub struct LifecycleConfig {
    /// Runs again of a failed task before it is given up on.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Task {
    /// Set up a `Creating` namespace, then move it to `Created`.
    Provision,
    /// Remove a `Deleting` namespace, its record last.
    Teardown,
    /// Run (or resume) a started index job; its record says how far it got.
    Job(JobKey),
}

impl Task {
    fn is_job(self) -> bool {
        matches!(self, Self::Job(_))
    }
}

/// `creation`, `drop`, or `job field_build:7`.
impl std::fmt::Display for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Provision => f.write_str("creation"),
            Self::Teardown => f.write_str("drop"),
            Self::Job(key) => write!(f, "job {}", key.kind),
        }
    }
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
    /// Start the governor for `host`, whose jobs `jobs` records. It holds the
    /// host weakly: when the last `Arc` of it goes, so does the governor.
    pub(crate) fn start<H: LifecycleHost>(host: Weak<H>, jobs: Arc<Jobs>, config: LifecycleConfig) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let parked = Arc::new(parking_lot::Mutex::new(HashMap::new()));
        let rx = Arc::new(tokio::sync::Mutex::new(rx));
        let supervisor = tokio::spawn(supervise(host, jobs, rx, config, Arc::clone(&parked)));
        Self { tx, parked, supervisor }
    }

    /// Whether the governor takes tasks.
    pub(crate) fn is_running(&self) -> bool {
        !self.tx.is_closed()
    }

    /// Hand `namespace`'s `task` to the governor; the receiver says how it
    /// ended. The same task already running is joined, not started again. A
    /// job must have been started with [`Jobs::start`] first.
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
    jobs: Arc<Jobs>,
    rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<Command>>>,
    config: LifecycleConfig,
    parked: Arc<parking_lot::Mutex<HashMap<String, TaskFailure>>>,
) {
    let mut restarts = 0u32;
    loop {
        // Its own task, so a panic in it is a `JoinError` here, not ours; and
        // aborting this future aborts it with every coordinator it holds.
        let governor = govern(host.clone(), Arc::clone(&jobs), Arc::clone(&rx), config.clone(), Arc::clone(&parked));
        let mut run = AbortOnDrop(tokio::spawn(governor));
        match run.await_join().await {
            Ok(()) => return,
            Err(e) => {
                restarts += 1;
                if restarts > config.max_restarts {
                    log::error!(
                        "CRITICAL: the namespace lifecycle governor failed {restarts} times ({e}); no namespace can be created, dropped or have an index job until the process restarts"
                    );
                    jobs.audit().record("", AuditEvent::GovernorStopped, None, None, Some(&e.to_string()));
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

/// Every task in progress, by namespace and task: who waits for each.
#[derive(Default)]
struct Running(HashMap<(String, Task), Vec<oneshot::Sender<Outcome>>>);

impl Running {
    /// Why `task` cannot start on `namespace` now, if it cannot: a creation or
    /// drop runs with nothing else beside it.
    fn refusal(&self, namespace: &str, task: Task) -> Option<String> {
        let other = self
            .0
            .keys()
            .find(|(n, t)| n == namespace && *t != task && (!t.is_job() || !task.is_job()))?;
        Some(format!("the {} of this namespace is running", other.1))
    }
}

/// The governor proper: take commands, start coordinators, hand out results.
async fn govern<H: LifecycleHost>(
    host: Weak<H>,
    jobs: Arc<Jobs>,
    rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<Command>>>,
    config: LifecycleConfig,
    parked: Arc<parking_lot::Mutex<HashMap<String, TaskFailure>>>,
) {
    let mut rx = rx.lock().await;
    let mut running = Running::default();
    let mut coordinators: JoinSet<(String, Task, Outcome)> = JoinSet::new();
    let spawn = |running: &mut Running,
                 coordinators: &mut JoinSet<(String, Task, Outcome)>,
                 namespace: String,
                 task: Task,
                 waiters: Vec<oneshot::Sender<Outcome>>| {
        running.0.insert((namespace.clone(), task), waiters);
        coordinators.spawn(coordinate(host.clone(), Arc::clone(&jobs), namespace, task, config.clone()));
    };

    // Everything a crash, or a parked task, left part-way.
    if let Some(h) = host.upgrade() {
        for (namespace, task) in leftover_tasks(&*h, &jobs) {
            let name = match task {
                Task::Job(key) => key.kind.name().to_owned(),
                other => other.to_string(),
            };
            jobs.audit().record(&namespace, AuditEvent::Resumed, Some(&name), None, None);
            spawn(&mut running, &mut coordinators, namespace, task, Vec::new());
        }
    }

    loop {
        tokio::select! {
            command = rx.recv() => {
                let Some(Command { namespace, task, done }) = command else { break };
                let refusal = match parked.lock().get(&namespace) {
                    Some(failure) => Some(failure.clone()),
                    None => running.refusal(&namespace, task).map(|error| TaskFailure { attempts: 0, error }),
                };
                if let Some(failure) = refusal {
                    if let Some(done) = done {
                        let _ = done.send(Err(failure));
                    }
                    continue;
                }
                match running.0.get_mut(&(namespace.clone(), task)) {
                    // The same task already runs: wait for it.
                    Some(waiters) => waiters.extend(done),
                    None => spawn(&mut running, &mut coordinators, namespace, task, done.into_iter().collect()),
                }
            }
            Some(joined) = coordinators.join_next(), if !coordinators.is_empty() => {
                // A coordinator never panics (its task runs in a task of its own).
                let Ok((namespace, task, outcome)) = joined else { continue };
                if let Err(failure) = &outcome
                    && !task.is_job()
                {
                    parked.lock().insert(namespace.clone(), failure.clone());
                    jobs.audit().record(&namespace, AuditEvent::TaskParked, Some(&task.to_string()), None, Some(&failure.error));
                }
                for waiter in running.0.remove(&(namespace, task)).unwrap_or_default() {
                    let _ = waiter.send(outcome.clone());
                }
            }
        }
    }
}

/// The tasks to pick up at start: the creation or drop of every namespace
/// still `Creating` or `Deleting`, and every live job of the others (their
/// states put in step with their records first).
fn leftover_tasks<H: LifecycleHost>(host: &H, jobs: &Jobs) -> Vec<(String, Task)> {
    let mut tasks = Vec::new();
    for (namespace, state) in host.states() {
        match state {
            NamespaceState::Creating => tasks.push((namespace, Task::Provision)),
            NamespaceState::Deleting => tasks.push((namespace, Task::Teardown)),
            NamespaceState::Created | NamespaceState::IndexMaintenance => {
                let Some(id) = host.namespace_id(&namespace) else { continue };
                match jobs.repair(host, &namespace, id) {
                    Ok(live) => tasks.extend(live.into_iter().map(|key| (namespace.clone(), Task::Job(key)))),
                    Err(e) => log::error!("namespace '{namespace}': cannot pick up its index jobs: {e}"),
                }
            }
        }
    }
    tasks
}

/// Run `namespace`'s `task`, and run it again after a failure or a panic, up
/// to `max_restarts` times; then give up (a job is recorded failed, any other
/// task parks the namespace). A job's end is recorded here either way.
async fn coordinate<H: LifecycleHost>(
    host: Weak<H>,
    jobs: Arc<Jobs>,
    namespace: String,
    task: Task,
    config: LifecycleConfig,
) -> (String, Task, Outcome) {
    let outcome = run_with_restarts(&host, &namespace, task, &config).await;
    if let Task::Job(key) = task
        && let Some(h) = host.upgrade()
        && let Err(e) = jobs.finish(&*h, &namespace, key, outcome.clone().map_err(|f| f.error))
    {
        log::error!("namespace '{namespace}': cannot record the end of its {task}: {e}");
    }
    (namespace, task, outcome)
}

/// [`coordinate`]'s loop: run, and run again, until done or out of runs.
async fn run_with_restarts<H: LifecycleHost>(host: &Weak<H>, namespace: &str, task: Task, config: &LifecycleConfig) -> Outcome {
    let mut attempts = 0u32;
    loop {
        attempts += 1;
        let Some(h) = host.upgrade() else {
            return Err(TaskFailure {
                attempts,
                error: "the host was closed".to_owned(),
            });
        };
        let ns = namespace.to_owned();
        let mut run = AbortOnDrop(tokio::spawn(async move { h.run(&ns, task).await.map_err(|e| e.to_string()) }));
        let error = match run.await_join().await {
            Ok(Ok(())) => return Ok(()),
            Ok(Err(e)) => e,
            Err(e) => format!("the task panicked: {e}"),
        };
        if attempts > config.max_restarts {
            if task.is_job() {
                log::error!("CRITICAL: namespace '{namespace}': its {task} failed {attempts} times, last: {error}; the job is recorded failed");
            } else {
                log::error!(
                    "CRITICAL: namespace '{namespace}': its {task} failed {attempts} times, last: {error}; it is left {} until the process restarts",
                    host.upgrade()
                        .and_then(|h| h.state(namespace))
                        .map_or("as it was", NamespaceState::as_str)
                );
            }
            return Err(TaskFailure { attempts, error });
        }
        log::warn!(
            "namespace '{namespace}': its {task} failed ({error}); running it again ({attempts} of {})",
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
