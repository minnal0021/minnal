//! The governor running jobs, creations and drops for a fake host: runs again
//! after a failure, picking up at start, cancelling, and what runs together.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Semaphore;

use super::fake_host::{FakeHost, Script};
use crate::namespace_lifecycle::{BuildStatus, Governor, JobKey, JobKind, LifecycleConfig, LifecycleError, LifecycleHost, NamespaceState, Task};

fn quick(max_restarts: u32) -> LifecycleConfig {
    LifecycleConfig {
        max_restarts,
        restart_backoff: Duration::from_millis(5),
    }
}

fn governor(host: &Arc<FakeHost>, max_restarts: u32) -> Governor {
    Governor::start(Arc::downgrade(host), Arc::clone(&host.jobs), quick(max_restarts))
}

fn status(host: &FakeHost, key: JobKey) -> Option<BuildStatus> {
    host.jobs.get(key).unwrap().map(|r| r.status)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_job_runs_and_its_namespace_returns_to_created() {
    let host = FakeHost::new().await;
    host.add("s", NamespaceState::Created);
    let gov = governor(&host, 3);
    let key = host.start("s", JobKind::FieldBuild(7)).unwrap();
    gov.run("s", Task::Job(key)).await.unwrap();
    assert_eq!(status(&host, key), Some(BuildStatus::Complete));
    assert_eq!(host.state("s"), Some(NamespaceState::Created));
}

/// A job that keeps failing is recorded failed, and its namespace goes on:
/// it is not parked, and the next job starts.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_job_that_keeps_failing_is_recorded_failed_without_parking() {
    let host = FakeHost::new().await;
    host.add("s", NamespaceState::Created);
    let gov = governor(&host, 2);
    let key = host.start("s", JobKind::FieldBuild(7)).unwrap();
    host.script(
        Task::Job(key),
        Script {
            fail: u32::MAX,
            ..Script::default()
        },
    );
    match gov.run("s", Task::Job(key)).await {
        Err(LifecycleError::TaskFailed { attempts: 3, .. }) => {}
        other => panic!("expected a failure after 3 runs, got {other:?}"),
    }
    assert_eq!(status(&host, key), Some(BuildStatus::Failed));
    assert_eq!(host.state("s"), Some(NamespaceState::Created));
    assert!(gov.parked().is_empty());
    let next = host.start("s", JobKind::FieldBuild(7)).unwrap();
    host.script(Task::Job(next), Script::default());
    gov.run("s", Task::Job(next)).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_or_panicked_job_is_run_again() {
    for panic in [false, true] {
        let host = FakeHost::new().await;
        host.add("s", NamespaceState::Created);
        let gov = governor(&host, 3);
        let key = host.start("s", JobKind::ReindexAll).unwrap();
        host.script(Task::Job(key), Script { fail: 2, panic, gate: None });
        gov.run("s", Task::Job(key)).await.unwrap();
        assert_eq!(host.runs(Task::Job(key)), 3, "panic={panic}");
        assert_eq!(status(&host, key), Some(BuildStatus::Complete));
    }
}

/// Jobs a crash left running are resumed when the governor starts, and the
/// creations and drops left part-way are finished.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn what_a_crash_left_is_picked_up_at_start() {
    let host = FakeHost::new().await;
    host.add("busy", NamespaceState::Created);
    let key = host.start("busy", JobKind::VectorReindex).unwrap();
    host.add("new", NamespaceState::Creating);
    host.add("gone", NamespaceState::Deleting);
    let _gov = governor(&host, 3);
    host.wait_until("all picked up", |h| {
        h.state("busy") == Some(NamespaceState::Created) && h.state("new") == Some(NamespaceState::Created) && h.state("gone").is_none()
    })
    .await;
    assert_eq!(status(&host, key), Some(BuildStatus::Complete));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cancel_ends_a_running_job_cancelled() {
    let host = FakeHost::new().await;
    host.add("s", NamespaceState::Created);
    let gov = governor(&host, 3);
    let key = host.start("s", JobKind::FieldBuild(7)).unwrap();
    let gate = Arc::new(Semaphore::new(0));
    host.script(
        Task::Job(key),
        Script {
            gate: Some(Arc::clone(&gate)),
            ..Script::default()
        },
    );
    let done = gov.submit("s", Task::Job(key)).unwrap();
    host.jobs.cancel("s", key).unwrap();
    gate.add_permits(1);
    done.await.unwrap().unwrap();
    assert_eq!(status(&host, key), Some(BuildStatus::Cancelled));
    assert_eq!(host.state("s"), Some(NamespaceState::Created));
}

/// A namespace's jobs run together; a drop or creation does not run beside
/// them, and is refused at once rather than queued.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn jobs_run_together_but_not_beside_a_drop() {
    let host = FakeHost::new().await;
    host.add("s", NamespaceState::Created);
    let gov = governor(&host, 0);
    let held = host.start("s", JobKind::FieldBuild(7)).unwrap();
    let gate = Arc::new(Semaphore::new(0));
    host.script(
        Task::Job(held),
        Script {
            gate: Some(Arc::clone(&gate)),
            ..Script::default()
        },
    );
    let held_done = gov.submit("s", Task::Job(held)).unwrap();
    host.wait_until("the held job is running", |h| h.runs(Task::Job(held)) == 1).await;

    let other = host.start("s", JobKind::FieldBuild(8)).unwrap();
    gov.run("s", Task::Job(other)).await.unwrap();
    match gov.run("s", Task::Teardown).await {
        Err(LifecycleError::TaskFailed { error, .. }) => assert!(error.contains("job field_build:7"), "{error}"),
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert!(gov.parked().is_empty(), "a refusal parks nothing");

    gate.add_permits(1);
    held_done.await.unwrap().unwrap();
    assert_eq!(host.state("s"), Some(NamespaceState::Created));
}
