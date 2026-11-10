//! The governor queueing a task behind what runs on its namespace: a drop or
//! job asked for the moment the namespace's last task ended, and a task asked
//! for again while it runs.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Semaphore;

use super::fake_host::{FakeHost, Script};
use crate::namespace_lifecycle::{BuildStatus, Governor, JobKind, LifecycleConfig, LifecycleHost, NamespaceState, Task};

fn governor(host: &Arc<FakeHost>) -> Governor {
    let config = LifecycleConfig {
        max_restarts: 0,
        restart_backoff: Duration::from_millis(5),
        ..LifecycleConfig::default()
    };
    Governor::start(Arc::downgrade(host), Arc::clone(&host.jobs), config)
}

/// A job's end moves its namespace back to `Created` before the governor
/// collects the job. A drop accepted at that moment (the namespace is
/// `Created`) is run once the job is collected, not refused, which would
/// leave the namespace `Deleting` with nothing removing it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_drop_the_moment_a_job_ends_is_run() {
    for round in 0..100 {
        let host = FakeHost::new().await;
        host.add("s", NamespaceState::Created);
        let gov = governor(&host);
        let key = host.start("s", JobKind::FieldBuild(7)).unwrap();
        let _job = gov.submit("s", Task::Job(key)).unwrap();
        host.wait_until("the job ended", |h| h.state("s") == Some(NamespaceState::Created)).await;
        host.transition("s", &[NamespaceState::Created], NamespaceState::Deleting).unwrap();
        gov.run("s", Task::Teardown).await.unwrap_or_else(|e| panic!("round {round}: {e}"));
        assert_eq!(host.state("s"), None, "round {round}");
    }
}

/// Likewise a job started the moment a creation ends runs, not refused (which
/// would leave its record live and the namespace in maintenance).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_job_the_moment_a_creation_ends_is_run() {
    for round in 0..100 {
        let host = FakeHost::new().await;
        let gov = governor(&host);
        host.add("s", NamespaceState::Creating);
        let _created = gov.submit("s", Task::Provision).unwrap();
        host.wait_until("created", |h| h.state("s") == Some(NamespaceState::Created)).await;
        let key = host.start("s", JobKind::FieldBuild(7)).unwrap();
        gov.run("s", Task::Job(key)).await.unwrap_or_else(|e| panic!("round {round}: {e}"));
        assert_eq!(host.jobs.get(key).unwrap().unwrap().status, BuildStatus::Complete, "round {round}");
        assert_eq!(host.state("s"), Some(NamespaceState::Created), "round {round}");
    }
}

/// A task asked for again while it runs waits for a run of its own: the
/// running one may have done its work before the second request (a job
/// started again under the same key after its last run ended), so sharing
/// its result would leave the new request's work undone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_task_asked_for_again_while_it_runs_runs_again() {
    let host = FakeHost::new().await;
    host.add("s", NamespaceState::Created);
    let gov = governor(&host);
    let key = host.start("s", JobKind::VectorReindex).unwrap();
    let gate = Arc::new(Semaphore::new(0));
    host.script(
        Task::Job(key),
        Script {
            gate: Some(Arc::clone(&gate)),
            ..Script::default()
        },
    );
    let first = gov.submit("s", Task::Job(key)).unwrap();
    host.wait_until("the first run started", |h| h.runs(Task::Job(key)) == 1).await;
    // Two more requests while it runs: they share the one run queued.
    let second = gov.submit("s", Task::Job(key)).unwrap();
    let third = gov.submit("s", Task::Job(key)).unwrap();
    gate.add_permits(2);
    first.await.unwrap().unwrap();
    second.await.unwrap().unwrap();
    third.await.unwrap().unwrap();
    assert_eq!(host.runs(Task::Job(key)), 2);
}

/// Queued tasks start in order, each once what blocks it has been collected:
/// a creation, then a job, then a drop.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn queued_tasks_start_in_order() {
    let host = FakeHost::new().await;
    let gov = governor(&host);
    host.add("s", NamespaceState::Creating);
    let gate = Arc::new(Semaphore::new(0));
    host.script(
        Task::Provision,
        Script {
            gate: Some(Arc::clone(&gate)),
            ..Script::default()
        },
    );
    let created = gov.submit("s", Task::Provision).unwrap();
    host.wait_until("the creation runs", |h| h.runs(Task::Provision) == 1).await;
    // Handed over while the creation runs, as the states would not let a
    // caller: the governor alone keeps them apart.
    let key = crate::namespace_lifecycle::JobKey::new(host.namespace_id("s").unwrap(), JobKind::FieldBuild(1));
    let job_gate = Arc::new(Semaphore::new(0));
    host.script(
        Task::Job(key),
        Script {
            gate: Some(Arc::clone(&job_gate)),
            ..Script::default()
        },
    );
    let job = gov.submit("s", Task::Job(key)).unwrap();
    let dropped = gov.submit("s", Task::Teardown).unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert_eq!(
        (host.runs(Task::Job(key)), host.runs(Task::Teardown)),
        (0, 0),
        "both wait for the creation"
    );
    gate.add_permits(1);
    created.await.unwrap().unwrap();
    host.wait_until("the job runs", |h| h.runs(Task::Job(key)) == 1).await;
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert_eq!(host.runs(Task::Teardown), 0, "the drop waits for the job");
    job_gate.add_permits(1);
    job.await.unwrap().unwrap();
    dropped.await.unwrap().unwrap();
    assert_eq!(host.runs(Task::Teardown), 1);
}

/// A creation given up on parks its namespace, but a drop of it is let
/// through, and unparks it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_namespace_parked_in_its_creation_can_be_dropped() {
    let host = FakeHost::new().await;
    let gov = governor(&host);
    host.add("s", NamespaceState::Creating);
    host.script(
        Task::Provision,
        Script {
            fail: u32::MAX,
            ..Script::default()
        },
    );
    assert!(gov.run("s", Task::Provision).await.is_err());
    assert!(gov.parked_in_creation("s"));
    assert!(gov.run("s", Task::Provision).await.is_err(), "a parked creation is not run again");

    host.set_state("s", NamespaceState::Deleting);
    gov.run("s", Task::Teardown).await.unwrap();
    assert_eq!(host.state("s"), None);
    assert!(gov.parked().is_empty());
}
