//! The governor supervised: restarted after a panic, picking up what the
//! failed run left; stopped for good past its restarts; and stopped on
//! request, waiting for its tasks.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Semaphore;

use super::fake_host::{FakeHost, Script};
use crate::namespace_lifecycle::{AuditEvent, BuildStatus, Governor, JobKind, LifecycleConfig, LifecycleError, LifecycleHost, NamespaceState, Task};

fn governor(host: &Arc<FakeHost>, max_restarts: u32, restart_window: Duration) -> Governor {
    let config = LifecycleConfig {
        max_restarts,
        restart_backoff: Duration::from_millis(5),
        restart_window,
    };
    Governor::start(Arc::downgrade(host), Arc::clone(&host.jobs), config)
}

async fn events(host: &FakeHost, event: AuditEvent) -> usize {
    let (audit, _) = host.jobs.audit().read(None, None, 1000).await.unwrap();
    audit.iter().filter(|r| r.event == event).count()
}

/// A governor that panics is restarted, and the restart picks up what was
/// left: here a job recorded but never handed over.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_crashed_governor_restarts_and_picks_up_what_was_left() {
    let host = FakeHost::new().await;
    host.add("s", NamespaceState::Created);
    let gov = governor(&host, 3, Duration::from_secs(600));
    let key = host.start("s", JobKind::ReindexAll).unwrap();
    gov.crash();
    host.wait_until("the job ran", |h| h.runs(Task::Job(key)) == 1).await;
    host.wait_until("the job ended", |h| h.jobs.get(key).unwrap().unwrap().status == BuildStatus::Complete)
        .await;
    assert!(gov.is_running());
    assert_eq!(events(&host, AuditEvent::Resumed).await, 1);
    // It takes tasks again.
    let next = host.start("s", JobKind::FieldBuild(1)).unwrap();
    gov.run("s", Task::Job(next)).await.unwrap();
}

/// Past `max_restarts` failures the governor stops for good: every task is
/// refused, and the stop is in the audit log.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_governor_that_keeps_crashing_stops() {
    let host = FakeHost::new().await;
    host.add("s", NamespaceState::Creating);
    let gov = governor(&host, 2, Duration::from_secs(600));
    for _ in 0..3 {
        gov.crash();
    }
    host.wait_until("stopped", |_| !gov.is_running()).await;
    assert!(matches!(gov.run("s", Task::Provision).await, Err(LifecycleError::GovernorUnavailable)));
    assert_eq!(events(&host, AuditEvent::GovernorStopped).await, 1);
}

/// Failures further apart than `restart_window` are not counted together.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn crashes_far_apart_do_not_stop_the_governor() {
    let host = FakeHost::new().await;
    host.add("s", NamespaceState::Created);
    let gov = governor(&host, 1, Duration::ZERO);
    for _ in 0..5 {
        gov.crash();
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(gov.is_running());
    let key = host.start("s", JobKind::FieldBuild(1)).unwrap();
    gov.run("s", Task::Job(key)).await.unwrap();
}

/// `stop` waits for the running tasks to stop, and a job stopped this way is
/// left live, to be resumed, not recorded failed. Tasks handed over after are
/// refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stop_waits_for_the_tasks_and_leaves_a_job_live() {
    let host = FakeHost::new().await;
    host.add("s", NamespaceState::Created);
    let gov = governor(&host, 3, Duration::from_secs(600));
    let key = host.start("s", JobKind::FieldBuild(7)).unwrap();
    host.script(
        Task::Job(key),
        Script {
            gate: Some(Arc::new(Semaphore::new(0))),
            ..Script::default()
        },
    );
    let done = gov.submit("s", Task::Job(key)).unwrap();
    host.wait_until("the job runs", |h| h.runs(Task::Job(key)) == 1).await;
    tokio::time::timeout(Duration::from_secs(5), gov.stop()).await.expect("stop returns");
    assert!(done.await.is_err(), "the caller hears the governor is gone");
    assert_eq!(host.jobs.get(key).unwrap().unwrap().status, BuildStatus::Running);
    assert_eq!(host.state("s"), Some(NamespaceState::IndexMaintenance));
    assert!(!gov.is_running());
    assert!(matches!(gov.submit("s", Task::Job(key)), Err(LifecycleError::GovernorUnavailable)));
}
