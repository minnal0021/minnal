//! The lifecycle governor: background creates and drops, runs again after a
//! failure or a panic, parked stores, and picking up what a crash left.

use std::sync::Arc;
use std::time::Duration;

use crate::doc_store::tests::test_support::*;
use crate::doc_store::*;

fn quick(max_restarts: u32) -> LifecycleConfig {
    LifecycleConfig {
        max_restarts,
        restart_backoff: Duration::from_millis(5),
    }
}

async fn governed(db_dir: &TempDir, max_restarts: u32) -> Arc<DocStore> {
    let store = Arc::new(open_fresh(db_dir.path()).await);
    store.start_lifecycle(quick(max_restarts));
    store
}

/// Wait up to 10 s for `namespace` to reach `state` (`None`: gone).
async fn wait_for(store: &DocStore, namespace: &str, state: Option<NamespaceState>) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while store.schemas.state(namespace) != state {
        assert!(tokio::time::Instant::now() < deadline, "'{namespace}' never reached {state:?}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// `begin_create` returns once the record is written; the governor sets the
/// store up, and it takes requests once `Created`. `begin_remove` likewise.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_background_create_and_drop_finish() {
    let db_dir = TempDir::new().unwrap();
    let store = governed(&db_dir, 3).await;
    store.begin_create(make_schema("docs", vec![])).await.unwrap();
    assert!(store.schemas.contains("docs"), "listed from the start");
    wait_for(&store, "docs", Some(NamespaceState::Created)).await;
    store.put("docs", DocId::U64(1), serde_json::json!({})).await.unwrap();

    store.begin_remove("docs").await.unwrap();
    assert!(matches!(
        store.get("docs", DocId::U64(1)).await,
        Err(DocStoreError::StoreDeleting { .. }) | Err(DocStoreError::NotFound { .. })
    ));
    wait_for(&store, "docs", None).await;
    assert!(!store.db.list_namespaces().iter().any(|(n, _)| n == "docs"));
    store.shutdown().await.unwrap();
}

/// Without a running governor the background forms refuse before writing
/// anything; the waiting forms run inline.
#[tokio::test]
async fn without_a_governor_the_background_forms_refuse() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    assert!(!store.lifecycle_running());
    assert!(matches!(
        store.begin_create(make_schema("docs", vec![])).await,
        Err(DocStoreError::GovernorUnavailable)
    ));
    assert!(!store.schemas.contains("docs"), "nothing written");
    store.create(make_schema("docs", vec![])).await.unwrap();
    assert!(matches!(store.begin_remove("docs").await, Err(DocStoreError::GovernorUnavailable)));
    assert_eq!(store.schemas.state("docs"), Some(NamespaceState::Created), "nothing changed");
    store.shutdown().await.unwrap();
}

/// A task that fails, or panics, is run again; it is idempotent, so the run
/// again finishes the store.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_or_panicked_task_is_run_again() {
    for panic in [false, true] {
        let db_dir = TempDir::new().unwrap();
        let store = governed(&db_dir, 3).await;
        store.lifecycle_faults.lock().insert("docs".to_owned(), (2, panic));
        store.create(make_schema("docs", vec![])).await.unwrap();
        assert_eq!(store.schemas.state("docs"), Some(NamespaceState::Created), "panic={panic}");
        store.lifecycle_faults.lock().insert("docs".to_owned(), (1, panic));
        store.remove("docs").await.unwrap();
        assert!(!store.schemas.contains("docs"), "panic={panic}");
        assert!(store.lifecycle_parked().is_empty());
        store.shutdown().await.unwrap();
    }
}

/// Past its runs again the store is parked: left in its state, listed as
/// parked, and later creates or drops of it fail at once. A restart of the
/// process (here: a new open and governor) is the way out.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_task_that_keeps_failing_parks_its_store_until_a_restart() {
    let db_dir = TempDir::new().unwrap();
    {
        let store = governed(&db_dir, 2).await;
        store.lifecycle_faults.lock().insert("docs".to_owned(), (u32::MAX, false));
        match store.create(make_schema("docs", vec![])).await {
            Err(DocStoreError::LifecycleTaskFailed { attempts: 3, .. }) => {}
            other => panic!("expected a parked store after 3 runs, got {other:?}"),
        }
        assert_eq!(store.schemas.state("docs"), Some(NamespaceState::Creating), "left as it was");
        assert_eq!(store.lifecycle_parked().iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(), ["docs"]);
        assert!(matches!(store.remove("docs").await, Err(DocStoreError::StoreCreating { .. })));
        assert!(matches!(
            store.put("docs", DocId::U64(1), serde_json::json!({})).await,
            Err(DocStoreError::StoreCreating { .. })
        ));
        store.shutdown().await.unwrap();
    }
    // Restarted, with the cause gone: the governor picks the store up.
    let store = governed(&db_dir, 2).await;
    wait_for(&store, "docs", Some(NamespaceState::Created)).await;
    store.put("docs", DocId::U64(1), serde_json::json!({})).await.unwrap();
    store.shutdown().await.unwrap();
}

/// A store whose drop was cut short is finished by the governor as soon as it
/// starts, with no caller asking.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_governor_finishes_what_it_finds_at_start() {
    let db_dir = TempDir::new().unwrap();
    let store = Arc::new(open_fresh(db_dir.path()).await);
    store.create(make_schema("gone", vec![])).await.unwrap();
    store
        .schemas
        .transition("gone", &[NamespaceState::Created], NamespaceState::Deleting)
        .unwrap();
    store
        .schemas
        .insert(StoredSchema::Doc(Arc::new(make_schema("new", vec![]))), NamespaceState::Creating)
        .unwrap();
    store.start_lifecycle(quick(3));
    wait_for(&store, "gone", None).await;
    wait_for(&store, "new", Some(NamespaceState::Created)).await;
    store.shutdown().await.unwrap();
}

/// A store left in maintenance with no live job (its last job ended, a crash
/// came before the move back) is made `Created` when the governor starts, and
/// can then be dropped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_governor_frees_a_store_left_in_maintenance_with_no_job() {
    let db_dir = TempDir::new().unwrap();
    let store = Arc::new(open_fresh(db_dir.path()).await);
    store.create(make_schema("docs", vec![])).await.unwrap();
    store
        .schemas
        .transition("docs", &[NamespaceState::Created], NamespaceState::IndexMaintenance)
        .unwrap();
    store.start_lifecycle(quick(3));
    wait_for(&store, "docs", Some(NamespaceState::Created)).await;
    store.remove("docs").await.unwrap();
    store.shutdown().await.unwrap();
}
