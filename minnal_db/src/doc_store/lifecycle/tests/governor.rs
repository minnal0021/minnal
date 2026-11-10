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
        ..LifecycleConfig::default()
    }
}

async fn governed(db_dir: &TempDir, max_restarts: u32) -> Arc<DocStore> {
    open_with(db_dir.path(), |b| b.lifecycle(quick(max_restarts))).await
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

    store.begin_remove("docs").unwrap();
    assert!(matches!(
        store.get("docs", DocId::U64(1)).await,
        Err(DocStoreError::StoreDeleting { .. }) | Err(DocStoreError::NotFound { .. })
    ));
    wait_for(&store, "docs", None).await;
    assert!(!store.db.list_namespaces().iter().any(|(n, _)| n == "docs"));
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

/// A store whose create or drop was cut short is finished at the next open,
/// with no caller asking.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn what_a_crash_left_is_finished_at_the_next_open() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    store.create(make_schema("gone", vec![])).await.unwrap();
    store
        .schemas
        .transition("gone", &[NamespaceState::Created], NamespaceState::Deleting)
        .unwrap();
    store
        .schemas
        .insert(StoredSchema::Doc(Arc::new(make_schema("new", vec![]))), NamespaceState::Creating)
        .unwrap();
    store.shutdown().await.unwrap();
    drop(store);

    let store = governed(&db_dir, 3).await;
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
    let store = open_fresh(db_dir.path()).await;
    store.create(make_schema("docs", vec![])).await.unwrap();
    store
        .schemas
        .transition("docs", &[NamespaceState::Created], NamespaceState::IndexMaintenance)
        .unwrap();
    store.shutdown().await.unwrap();
    drop(store);

    let store = governed(&db_dir, 3).await;
    wait_for(&store, "docs", Some(NamespaceState::Created)).await;
    store.remove("docs").await.unwrap();
    store.shutdown().await.unwrap();
}

/// A store whose creation was given up on can be dropped while `Creating`:
/// the drop removes what the creation made and frees the name.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_store_parked_in_its_creation_can_be_dropped() {
    let db_dir = TempDir::new().unwrap();
    let store = governed(&db_dir, 1).await;
    store.lifecycle_faults.lock().insert("docs".to_owned(), (u32::MAX, false));
    assert!(store.create(make_schema("docs", vec![])).await.is_err());
    assert_eq!(store.schemas.state("docs"), Some(NamespaceState::Creating));
    let listed = store.list().unwrap();
    let parked = listed[0]["parked"].as_str().unwrap_or_default();
    assert!(parked.contains("injected lifecycle failure"), "the listing says why: {listed:?}");
    store.lifecycle_faults.lock().clear();
    store.remove("docs").await.unwrap();
    assert!(!store.schemas.contains("docs"));
    assert!(store.lifecycle_parked().is_empty());
    store.create(make_schema("docs", vec![])).await.unwrap();
    store.put("docs", DocId::U64(1), serde_json::json!({})).await.unwrap();
    store.shutdown().await.unwrap();
}

/// With the governor not running, a create, a drop and an index job are
/// refused before anything is written: no store is left `Creating`,
/// `Deleting` or in maintenance with nobody to finish it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nothing_is_written_while_the_governor_is_not_running() {
    let db_dir = TempDir::new().unwrap();
    let store = governed(&db_dir, 3).await;
    store.create(make_schema("kept", vec![])).await.unwrap();
    store.governor.get().unwrap().stop().await;

    assert!(matches!(
        store.create(make_schema("new", vec![])).await,
        Err(DocStoreError::GovernorUnavailable)
    ));
    assert!(matches!(
        store.begin_create(make_schema("new", vec![])).await,
        Err(DocStoreError::GovernorUnavailable)
    ));
    assert!(!store.schemas.contains("new"));
    assert!(matches!(store.remove("kept").await, Err(DocStoreError::GovernorUnavailable)));
    assert!(matches!(store.begin_remove("kept"), Err(DocStoreError::GovernorUnavailable)));
    let spec = IndexSpec {
        field: "status".to_owned(),
        index_type: IndexType::Str,
    };
    assert!(matches!(store.add_index("kept", spec).await, Err(DocStoreError::GovernorUnavailable)));
    assert_eq!(store.schemas.state("kept"), Some(NamespaceState::Created));
    store.put("kept", DocId::U64(1), serde_json::json!({})).await.unwrap();
    store.shutdown().await.unwrap();
}

/// A store dropped the moment its index build ends is dropped: the drop is
/// not refused because the governor has not yet collected the build.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_drop_the_moment_a_build_ends_finishes() {
    let db_dir = TempDir::new().unwrap();
    let store = governed(&db_dir, 0).await;
    for round in 0..20 {
        let name = format!("docs{round}");
        store.create(make_schema(&name, vec![])).await.unwrap();
        let spec = IndexSpec {
            field: "status".to_owned(),
            index_type: IndexType::Str,
        };
        store.begin_add_index(&name, spec).unwrap();
        wait_for(&store, &name, Some(NamespaceState::Created)).await;
        store.remove(&name).await.unwrap_or_else(|e| panic!("round {round}: {e}"));
        assert!(!store.schemas.contains(&name));
    }
    store.shutdown().await.unwrap();
}
