//! The lifecycle governor: one task per [`LifecycleHost`] that runs every
//! namespace's creation, drop and index job, supervised.
//!
//! A create or drop first writes the namespace's new state (`Creating`,
//! `Deleting`), then hands it to the governor, which runs its task
//! ([`Task::Provision`] or [`Task::Teardown`], through
//! [`LifecycleHost::run`]) in a coordinator task of its own. An index job is
//! first recorded and its namespace moved to `IndexMaintenance`
//! ([`Jobs::start`]), then handed over the same way as a [`Task::Job`]; when
//! it ends, its coordinator records how ([`Jobs::finish`]). The governor is
//! single-threaded: it only starts coordinators and collects their results;
//! the work, and every write to disk, is theirs.
//!
//! **What runs together.** A namespace runs one creation or drop at a time
//! and nothing beside it; its jobs may run together (`Jobs` refuses the ones
//! that conflict). The namespace's state already keeps them apart, but a task
//! changes the state *before* the governor collects it: a job's end moves its
//! namespace back to `Created`, and a drop asked for at that moment is
//! accepted while the job is still counted as running. So a task that may not
//! run yet beside what runs is **queued**, in order, and started when what
//! blocks it has been collected; it is never refused for it. The same holds
//! for a task asked for again while it runs: that run may already have done
//! its work (a job started again under the same key after its last run ended),
//! so the new request waits for the next run rather than share the old one's
//! result. Two requests of a task that has not started yet share its run.
//!
//! **Supervision.** A coordinator whose task fails or panics is run again,
//! after `restart_backoff`, up to `max_restarts` times: every task is
//! idempotent, so a run again finishes what the last one left. Past that a
//! job is recorded failed, with a CRITICAL error logged, and its namespace
//! goes on. A creation or a drop past that *parks* its namespace: a CRITICAL
//! error is logged, its state stays as it is, and later tasks for it fail at
//! once, until the process restarts (the escape hatch; the count is kept in
//! memory only). A namespace parked in its creation can still be dropped: the
//! drop removes whatever the creation made, and frees its name. The governor
//! itself is supervised the same way (failures more than `restart_window`
//! apart are not counted together); if it gives up, its channel closes and
//! every create, drop and job fails with `GovernorUnavailable`, while
//! namespaces that exist keep working.
//!
//! **Stopping.** [`Governor::stop`] asks every coordinator to stop its task
//! and waits until each has: a task is stopped at its next `.await`, and is
//! finished at the next start. A job stopped this way is left live, not
//! recorded failed. The governor holds its host weakly between tasks; a
//! running task holds it.
//!
//! At start the governor picks up every namespace that is still `Creating` or
//! `Deleting`, and every live job ([`Jobs::repair`]), so what a crash or a
//! parked namespace left before a restart is finished. It finds them before
//! [`Governor::start`] returns, so nothing handed over later is mistaken for
//! one of them.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinSet;

use crate::namespace_lifecycle::{AuditEvent, JobKey, Jobs, LifecycleError, LifecycleHost, NamespaceState};

/// How the governor restarts what fails.
#[derive(Debug, Clone, PartialEq)]
pub struct LifecycleConfig {
    /// Runs again of a failed task before it is given up on.
    pub max_restarts: u32,
    /// Wait before each run again.
    pub restart_backoff: Duration,
    /// Failures of the governor itself further apart than this are not
    /// counted together: one that ran this long before failing starts its
    /// count again.
    pub restart_window: Duration,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            max_restarts: crate::defaults::lifecycle::MAX_RESTARTS,
            restart_backoff: crate::defaults::lifecycle::RESTART_BACKOFF,
            restart_window: crate::defaults::lifecycle::RESTART_WINDOW,
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

    /// The task's name in the audit log: `creation`, `drop`, or a job's kind
    /// (`field_build`).
    fn audit_name(self) -> String {
        match self {
            Self::Job(key) => key.kind.name().to_owned(),
            other => other.to_string(),
        }
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

/// Says how a task handed to the governor ended ([`Governor::wait`]).
pub(crate) type TaskDone = oneshot::Receiver<Outcome>;

/// Namespaces parked after too many failed runs: the task given up on, and
/// its last failure.
type Parked = Arc<parking_lot::Mutex<HashMap<String, (Task, TaskFailure)>>>;

/// The namespace of a [`Governor::crash`] command.
#[cfg(test)]
const CRASH: &str = "\0crash";

struct Command {
    namespace: String,
    task: Task,
    done: Option<oneshot::Sender<Outcome>>,
    /// Left part-way by a crash or a parked task, found when the governor
    /// started: recorded as resumed in the audit log.
    resumed: bool,
}

/// The handle the host keeps: submits tasks, and stops the governor with it.
pub(crate) struct Governor {
    tx: mpsc::UnboundedSender<Command>,
    parked: Parked,
    /// Set by [`stop`](Self::stop); every coordinator watches it.
    shutdown: watch::Sender<bool>,
    supervisor: parking_lot::Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl Governor {
    /// Start the governor for `host`, whose jobs `jobs` records. It holds the
    /// host weakly: when the last `Arc` of it goes, so does the governor.
    ///
    /// What a crash or a parked task left part-way is found here, before this
    /// returns, and queued ahead of anything a caller hands over: found later,
    /// by the task, a create handed over meanwhile would be taken for one a
    /// crash left (and logged as resumed).
    pub(crate) fn start<H: LifecycleHost>(host: Weak<H>, jobs: Arc<Jobs>, config: LifecycleConfig) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        if let Some(h) = host.upgrade() {
            for (namespace, task) in leftover_tasks(&*h, &jobs) {
                let _ = tx.send(Command {
                    namespace,
                    task,
                    done: None,
                    resumed: true,
                });
            }
        }
        let parked = Parked::default();
        let (shutdown, _) = watch::channel(false);
        let rx = Arc::new(tokio::sync::Mutex::new(rx));
        let supervisor = tokio::spawn(supervise(
            Context {
                host,
                jobs,
                config,
                parked: Arc::clone(&parked),
                shutdown: shutdown.subscribe(),
            },
            rx,
        ));
        Self {
            tx,
            parked,
            shutdown,
            supervisor: parking_lot::Mutex::new(Some(supervisor)),
        }
    }

    /// Whether the governor takes tasks.
    pub(crate) fn is_running(&self) -> bool {
        !self.tx.is_closed() && !*self.shutdown.borrow()
    }

    /// Hand `namespace`'s `task` to the governor; the receiver says how it
    /// ended. A task that may not run yet beside what runs on the namespace
    /// waits for it (see the module docs). A job must have been started with
    /// [`Jobs::start`] first.
    pub(crate) fn submit(&self, namespace: &str, task: Task) -> Result<TaskDone, LifecycleError> {
        if !self.is_running() {
            return Err(LifecycleError::GovernorUnavailable);
        }
        let (done, rx) = oneshot::channel();
        self.tx
            .send(Command {
                namespace: namespace.to_owned(),
                task,
                done: Some(done),
                resumed: false,
            })
            .map_err(|_| LifecycleError::GovernorUnavailable)?;
        Ok(rx)
    }

    /// Hand `namespace`'s `task` to the governor and wait for it to end.
    pub(crate) async fn run(&self, namespace: &str, task: Task) -> Result<(), LifecycleError> {
        Self::wait(namespace, self.submit(namespace, task)?).await
    }

    /// Wait for `namespace`'s task handed over as `done` to end.
    pub(crate) async fn wait(namespace: &str, done: TaskDone) -> Result<(), LifecycleError> {
        let outcome = done.await.map_err(|_| LifecycleError::GovernorUnavailable)?;
        outcome.map_err(|f| LifecycleError::TaskFailed {
            namespace: namespace.to_owned(),
            attempts: f.attempts,
            error: f.error,
        })
    }

    /// Namespaces parked after too many failed runs, with the last error.
    pub(crate) fn parked(&self) -> Vec<(String, String)> {
        let mut parked: Vec<(String, String)> = self.parked.lock().iter().map(|(n, (_, f))| (n.clone(), f.error.clone())).collect();
        parked.sort();
        parked
    }

    /// The last error of `namespace`'s task, if it was given up on (parked).
    pub(crate) fn parked_error(&self, namespace: &str) -> Option<String> {
        self.parked.lock().get(namespace).map(|(_, f)| f.error.clone())
    }

    /// Whether `namespace`'s creation was given up on: it may then be dropped
    /// while still `Creating`.
    pub(crate) fn parked_in_creation(&self, namespace: &str) -> bool {
        matches!(self.parked.lock().get(namespace), Some((Task::Provision, _)))
    }

    /// Test hook: make the governor itself panic, as a bug in it would.
    #[cfg(test)]
    pub(crate) fn crash(&self) {
        let _ = self.tx.send(Command {
            namespace: CRASH.to_owned(),
            task: Task::Provision,
            done: None,
            resumed: false,
        });
    }

    /// Stop the governor and every task it runs, and wait until each has
    /// stopped (at its next `.await`). A task cut short is finished at the
    /// next start: every task is idempotent, and a job stopped this way stays
    /// live. Tasks handed over later are refused.
    pub(crate) async fn stop(&self) {
        let _ = self.shutdown.send(true);
        let supervisor = self.supervisor.lock().take();
        if let Some(supervisor) = supervisor {
            let _ = supervisor.await;
        }
    }
}

impl Drop for Governor {
    fn drop(&mut self) {
        if let Some(supervisor) = self.supervisor.lock().take() {
            supervisor.abort();
        }
    }
}

/// What the supervisor, the governor and every coordinator share.
struct Context<H> {
    host: Weak<H>,
    jobs: Arc<Jobs>,
    config: LifecycleConfig,
    parked: Parked,
    shutdown: watch::Receiver<bool>,
}

impl<H> Clone for Context<H> {
    fn clone(&self) -> Self {
        Self {
            host: self.host.clone(),
            jobs: Arc::clone(&self.jobs),
            config: self.config.clone(),
            parked: Arc::clone(&self.parked),
            shutdown: self.shutdown.clone(),
        }
    }
}

impl<H> Context<H> {
    /// Resolves once the governor is told to stop (or its handle is gone).
    async fn stopped(&self) {
        let mut shutdown = self.shutdown.clone();
        let _ = shutdown.wait_for(|stop| *stop).await;
    }
}

/// Run the governor, and run it again if it panics, up to `max_restarts`
/// failures within `restart_window` of each other.
async fn supervise<H: LifecycleHost>(cx: Context<H>, rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<Command>>>) {
    let mut restarts = 0u32;
    let mut first = true;
    loop {
        // Its own task, so a panic in it is a `JoinError` here, not ours; and
        // aborting this future aborts it with every coordinator it holds. A
        // restart finds again what the failed run left part-way (`start`
        // found what was left before the first).
        let started = Instant::now();
        let mut run = AbortOnDrop(tokio::spawn(govern(cx.clone(), Arc::clone(&rx), !first)));
        first = false;
        match run.await_join().await {
            Ok(()) => return,
            Err(e) => {
                if started.elapsed() >= cx.config.restart_window {
                    restarts = 0;
                }
                restarts += 1;
                if restarts > cx.config.max_restarts {
                    log::error!(
                        "CRITICAL: the namespace lifecycle governor failed {restarts} times ({e}); no namespace can be created, dropped or have an index job until the process restarts"
                    );
                    cx.jobs.audit().record("", AuditEvent::GovernorStopped, None, None, Some(&e.to_string()));
                    rx.lock().await.close();
                    return;
                }
                log::warn!(
                    "the namespace lifecycle governor failed ({e}); restarting it ({restarts} of {})",
                    cx.config.max_restarts
                );
                tokio::select! {
                    () = tokio::time::sleep(cx.config.restart_backoff) => {}
                    () = cx.stopped() => return,
                }
            }
        }
    }
}

/// A task waiting for what runs on its namespace, and who waits for it.
struct Pending {
    task: Task,
    waiters: Vec<oneshot::Sender<Outcome>>,
    resumed: bool,
}

/// Every task in progress, by namespace and task, with who waits for each;
/// and, per namespace, the tasks queued behind them, in order.
#[derive(Default)]
struct Running {
    tasks: HashMap<(String, Task), Vec<oneshot::Sender<Outcome>>>,
    queued: HashMap<String, VecDeque<Pending>>,
}

impl Running {
    /// Whether `task` must wait for what runs on `namespace`: a creation or
    /// drop runs with nothing beside it, and a task asked for again while it
    /// runs waits for its next run.
    fn must_wait(&self, namespace: &str, task: Task) -> bool {
        self.tasks
            .keys()
            .any(|(n, t)| n == namespace && (*t == task || !t.is_job() || !task.is_job()))
    }

    /// Queue `pending` behind `namespace`'s tasks. The same task queued
    /// already is not queued twice: its waiters share that run.
    fn queue(&mut self, namespace: String, pending: Pending) {
        let queue = self.queued.entry(namespace).or_default();
        match queue.iter_mut().find(|p| p.task == pending.task) {
            Some(queued) => {
                queued.waiters.extend(pending.waiters);
                queued.resumed |= pending.resumed;
            }
            None => queue.push_back(pending),
        }
    }

    /// The first task queued on `namespace`, if it may start now.
    fn next_ready(&mut self, namespace: &str) -> Option<Pending> {
        let front = self.queued.get(namespace)?.front()?.task;
        if self.must_wait(namespace, front) {
            return None;
        }
        let queue = self.queued.get_mut(namespace)?;
        let next = queue.pop_front();
        if queue.is_empty() {
            self.queued.remove(namespace);
        }
        next
    }
}

/// Why `task` cannot run on `namespace` at all, if its namespace is parked:
/// the failure it was parked for. A drop of a namespace parked in its
/// creation is let through, and unparks it.
fn parked_refusal(parked: &Parked, namespace: &str, task: Task) -> Option<TaskFailure> {
    let mut parked = parked.lock();
    let (parked_task, failure) = parked.get(namespace)?;
    if task == Task::Teardown && *parked_task == Task::Provision {
        parked.remove(namespace);
        return None;
    }
    Some(failure.clone())
}

/// The governor proper: take commands, start coordinators, hand out results.
async fn govern<H: LifecycleHost>(cx: Context<H>, rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<Command>>>, rescan: bool) {
    let mut rx = rx.lock().await;
    let mut running = Running::default();
    let mut coordinators: JoinSet<(String, Task, Option<Outcome>)> = JoinSet::new();
    let start = |running: &mut Running, coordinators: &mut JoinSet<(String, Task, Option<Outcome>)>, namespace: String, pending: Pending| {
        if let Some(failure) = parked_refusal(&cx.parked, &namespace, pending.task) {
            for waiter in pending.waiters {
                let _ = waiter.send(Err(failure.clone()));
            }
            return;
        }
        running.tasks.insert((namespace.clone(), pending.task), pending.waiters);
        coordinators.spawn(coordinate(cx.clone(), namespace, pending.task, pending.resumed));
    };

    // After a restart: everything the failed run left part-way.
    if rescan && let Some(h) = cx.host.upgrade() {
        for (namespace, task) in leftover_tasks(&*h, &cx.jobs) {
            let pending = Pending {
                task,
                waiters: Vec::new(),
                resumed: true,
            };
            start(&mut running, &mut coordinators, namespace, pending);
        }
    }

    loop {
        tokio::select! {
            () = cx.stopped() => break,
            command = rx.recv() => {
                let Some(Command { namespace, task, done, resumed }) = command else { break };
                #[cfg(test)]
                if namespace == CRASH {
                    panic!("the governor was told to crash");
                }
                let pending = Pending { task, waiters: done.into_iter().collect(), resumed };
                if running.must_wait(&namespace, task) || running.queued.contains_key(&namespace) {
                    running.queue(namespace, pending);
                } else {
                    start(&mut running, &mut coordinators, namespace, pending);
                }
            }
            Some(joined) = coordinators.join_next(), if !coordinators.is_empty() => {
                // A coordinator never panics (its task runs in a task of its
                // own), and is aborted only when the governor is.
                let Ok((namespace, task, outcome)) = joined else { continue };
                let waiters = running.tasks.remove(&(namespace.clone(), task)).unwrap_or_default();
                // No outcome: the task was stopped, and its waiters are told
                // the governor is gone (their senders dropped).
                if let Some(outcome) = outcome {
                    for waiter in waiters {
                        let _ = waiter.send(outcome.clone());
                    }
                }
                while let Some(next) = running.next_ready(&namespace) {
                    start(&mut running, &mut coordinators, namespace.clone(), next);
                }
            }
        }
    }
    // Stopping: every coordinator stops its task, and is waited for.
    while coordinators.join_next().await.is_some() {}
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
/// task parks the namespace). A job's end is recorded here either way. Every
/// write to disk is done here, off the governor's loop. `None`: the governor
/// stopped the task, which is left as it is, to be resumed.
async fn coordinate<H: LifecycleHost>(cx: Context<H>, namespace: String, task: Task, resumed: bool) -> (String, Task, Option<Outcome>) {
    if resumed {
        cx.jobs
            .audit()
            .record(&namespace, AuditEvent::Resumed, Some(&task.audit_name()), None, None);
    }
    let Some(outcome) = run_with_restarts(&cx, &namespace, task).await else {
        return (namespace, task, None);
    };
    match (task, &outcome) {
        (Task::Job(key), _) => {
            if let Some(h) = cx.host.upgrade()
                && let Err(e) = cx.jobs.finish(&*h, &namespace, key, outcome.clone().map_err(|f| f.error))
            {
                log::error!("namespace '{namespace}': cannot record the end of its {task}: {e}");
            }
        }
        (_, Err(failure)) => {
            // Parked before the governor collects this task, so nothing
            // queued behind it starts first.
            cx.parked.lock().insert(namespace.clone(), (task, failure.clone()));
            cx.jobs
                .audit()
                .record(&namespace, AuditEvent::TaskParked, Some(&task.to_string()), None, Some(&failure.error));
        }
        (_, Ok(())) => {}
    }
    (namespace, task, Some(outcome))
}

/// [`coordinate`]'s loop: run, and run again, until done or out of runs;
/// `None` if the governor stopped it meanwhile.
async fn run_with_restarts<H: LifecycleHost>(cx: &Context<H>, namespace: &str, task: Task) -> Option<Outcome> {
    let mut attempts = 0u32;
    loop {
        attempts += 1;
        let Some(h) = cx.host.upgrade() else {
            return Some(Err(TaskFailure {
                attempts,
                error: "the host was closed".to_owned(),
            }));
        };
        let ns = namespace.to_owned();
        let mut run = AbortOnDrop(tokio::spawn(async move {
            h.run(&ns, task).await.map_err(|e| (H::is_final(&e), e.to_string()))
        }));
        let joined = tokio::select! {
            joined = run.await_join() => Some(joined),
            () = cx.stopped() => None,
        };
        let Some(joined) = joined else {
            // Stopped at its next `.await`; wait for that, so nothing of it
            // runs once the governor has stopped.
            run.0.abort();
            let _ = run.await_join().await;
            return None;
        };
        let (last, error) = match joined {
            Ok(Ok(())) => return Some(Ok(())),
            Ok(Err((is_final, e))) => (is_final, e),
            Err(e) => (false, format!("the task panicked: {e}")),
        };
        if last {
            log::error!("namespace '{namespace}': its {task} failed, and cannot be run again: {error}");
            return Some(Err(TaskFailure { attempts, error }));
        }
        if attempts > cx.config.max_restarts {
            if task.is_job() {
                log::error!("CRITICAL: namespace '{namespace}': its {task} failed {attempts} times, last: {error}; the job is recorded failed");
            } else {
                log::error!(
                    "CRITICAL: namespace '{namespace}': its {task} failed {attempts} times, last: {error}; it is left {} until the process restarts",
                    cx.host
                        .upgrade()
                        .and_then(|h| h.state(namespace))
                        .map_or("as it was", NamespaceState::as_str)
                );
            }
            return Some(Err(TaskFailure { attempts, error }));
        }
        log::warn!(
            "namespace '{namespace}': its {task} failed ({error}); running it again ({attempts} of {})",
            cx.config.max_restarts
        );
        tokio::select! {
            () = tokio::time::sleep(cx.config.restart_backoff) => {}
            () = cx.stopped() => return None,
        }
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
