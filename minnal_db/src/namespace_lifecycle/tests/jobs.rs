//! The jobs registry: starting, finishing and cancelling jobs, and the
//! namespace's state following them.

use super::fake_host::FakeHost;
use crate::namespace_lifecycle::{BuildStatus, JobKey, JobKind, LifecycleError, LifecycleHost, NamespaceState};

const BUILD_7: JobKind = JobKind::FieldBuild(7);
const BUILD_8: JobKind = JobKind::FieldBuild(8);

fn status(host: &FakeHost, key: JobKey) -> BuildStatus {
    host.jobs.get(key).unwrap().unwrap().status
}

#[tokio::test(flavor = "multi_thread")]
async fn a_job_moves_its_namespace_to_maintenance_and_back() {
    let host = FakeHost::new().await;
    host.add("s", NamespaceState::Created);
    let key = host.start("s", BUILD_7).unwrap();
    assert_eq!(host.state("s"), Some(NamespaceState::IndexMaintenance));
    assert_eq!(status(&host, key), BuildStatus::Running);

    host.jobs.finish(&*host, "s", key, Ok(())).unwrap();
    assert_eq!(host.state("s"), Some(NamespaceState::Created));
    assert_eq!(status(&host, key), BuildStatus::Complete, "the record is kept for the views");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_namespace_stays_in_maintenance_until_its_last_job_ends() {
    let host = FakeHost::new().await;
    host.add("s", NamespaceState::Created);
    let a = host.start("s", BUILD_7).unwrap();
    let b = host.start("s", BUILD_8).unwrap();
    host.jobs.finish(&*host, "s", a, Ok(())).unwrap();
    assert_eq!(host.state("s"), Some(NamespaceState::IndexMaintenance));
    host.jobs.finish(&*host, "s", b, Err("boom".to_owned())).unwrap();
    assert_eq!(host.state("s"), Some(NamespaceState::Created), "a failed job frees it too");
    let failed = host.jobs.get(b).unwrap().unwrap();
    assert_eq!((failed.status, failed.error.as_deref()), (BuildStatus::Failed, Some("boom")));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_conflicting_job_is_refused_and_writes_nothing() {
    let host = FakeHost::new().await;
    let id = host.add("s", NamespaceState::Created);
    host.start("s", BUILD_7).unwrap();
    for kind in [BUILD_7, JobKind::FieldDrop(7), JobKind::ReindexAll] {
        match host.start("s", kind) {
            Err(LifecycleError::JobConflict { running, .. }) => assert_eq!(running, "field_build:7"),
            other => panic!("{kind}: expected a conflict, got {other:?}"),
        }
    }
    assert_eq!(host.jobs.list(id).unwrap().len(), 1, "no record for a refused job");
    // Jobs on other indexes start.
    host.start("s", BUILD_8).unwrap();
    host.start("s", JobKind::VectorReindex).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_job_is_refused_unless_its_namespace_is_ready() {
    let host = FakeHost::new().await;
    let creating = host.add("c", NamespaceState::Creating);
    let deleting = host.add("d", NamespaceState::Deleting);
    assert!(matches!(
        host.jobs.start(&*host, "c", JobKey::new(creating, BUILD_7), None),
        Err(LifecycleError::Creating { .. })
    ));
    assert!(matches!(
        host.jobs.start(&*host, "d", JobKey::new(deleting, BUILD_7), None),
        Err(LifecycleError::Deleting { .. })
    ));
    assert!(matches!(
        host.jobs.start(&*host, "nobody", JobKey::new(creating, BUILD_7), None),
        Err(LifecycleError::NotFound { .. })
    ));
    assert!(host.jobs.list(creating).unwrap().is_empty() && host.jobs.list(deleting).unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_drop_is_refused_while_a_job_runs() {
    let host = FakeHost::new().await;
    host.add("s", NamespaceState::Created);
    let key = host.start("s", BUILD_7).unwrap();
    let drop = |host: &FakeHost| host.transition("s", &[NamespaceState::Created], NamespaceState::Deleting);
    assert!(matches!(drop(&host), Err(LifecycleError::InMaintenance { .. })));
    host.jobs.finish(&*host, "s", key, Ok(())).unwrap();
    drop(&host).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_cancelled_job_ends_cancelled() {
    let host = FakeHost::new().await;
    host.add("s", NamespaceState::Created);
    let key = host.start("s", BUILD_7).unwrap();
    assert!(!host.jobs.is_cancelling(key));
    host.jobs.cancel("s", key).unwrap();
    assert!(host.jobs.is_cancelling(key));
    // Progress the body saves meanwhile leaves the request standing.
    host.jobs.save_progress(key, serde_json::json!({"indexed": 5})).unwrap();
    assert!(host.jobs.is_cancelling(key));

    host.jobs.finish(&*host, "s", key, Ok(())).unwrap();
    let record = host.jobs.get(key).unwrap().unwrap();
    assert_eq!((record.status, record.progress["indexed"].as_u64()), (BuildStatus::Cancelled, Some(5)));
    assert_eq!(host.state("s"), Some(NamespaceState::Created));
    assert!(matches!(host.jobs.cancel("s", key), Err(LifecycleError::JobNotFound { .. })), "ended");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_vector_cleanup_cannot_be_cancelled() {
    let host = FakeHost::new().await;
    let id = host.add("s", NamespaceState::Created);
    let key = host.start("s", JobKind::VectorCleanup).unwrap();
    assert!(matches!(host.jobs.cancel("s", key), Err(LifecycleError::JobNotCancellable { .. })));
    assert!(matches!(
        host.jobs.cancel("s", JobKey::new(id, JobKind::VectorReindex)),
        Err(LifecycleError::JobNotFound { .. })
    ));
}

/// A crash between a job's record and its namespace's state leaves them
/// disagreeing; the records win.
#[tokio::test(flavor = "multi_thread")]
async fn repair_puts_the_state_in_step_with_the_records() {
    let host = FakeHost::new().await;
    // In maintenance with no live job: its last job ended, the move back did not.
    let idle = host.add("idle", NamespaceState::Created);
    let key = host.start("idle", BUILD_7).unwrap();
    host.jobs.finish(&*host, "idle", key, Ok(())).unwrap();
    host.set_state("idle", NamespaceState::IndexMaintenance);
    assert!(host.jobs.repair(&*host, "idle", idle).unwrap().is_empty());
    assert_eq!(host.state("idle"), Some(NamespaceState::Created));

    // Created with a live job: its record was written, the move was not.
    let busy = host.add("busy", NamespaceState::Created);
    let key = host.start("busy", BUILD_7).unwrap();
    host.set_state("busy", NamespaceState::Created);
    assert_eq!(host.jobs.repair(&*host, "busy", busy).unwrap(), vec![key]);
    assert_eq!(host.state("busy"), Some(NamespaceState::IndexMaintenance));
}
