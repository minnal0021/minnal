//! The lifecycle states: what a store takes in each, and how a creation or a
//! drop a crash interrupted is finished at the next open.

use crate::doc_store::tests::test_support::*;
use crate::doc_store::*;

fn index() -> IndexSpec {
    IndexSpec {
        field: "active".to_owned(),
        index_type: IndexType::Bool,
    }
}

/// A store whose record is `Creating` (a crash before its setup finished) is
/// listed, refuses reads, writes, queries and drops, and the next open sets it
/// up and makes it `Created`.
#[tokio::test]
async fn a_store_being_created_refuses_requests_and_the_next_open_finishes_it() {
    let db_dir = TempDir::new().unwrap();
    {
        let store = open_fresh(db_dir.path()).await;
        // What `create` writes before any setup: the record alone.
        let schema = make_schema("docs", vec![index()]);
        store
            .schemas
            .insert(StoredSchema::Doc(std::sync::Arc::new(schema)), NamespaceState::Creating)
            .unwrap();

        let creating = |r: Result<_, DocStoreError>| matches!(r, Err(DocStoreError::StoreCreating { .. }));
        assert!(creating(
            store.put("docs", DocId::U64(1), serde_json::json!({"active": true})).await.map(|_| ())
        ));
        assert!(creating(store.get("docs", DocId::U64(1)).await.map(|_| ())));
        assert!(creating(store.query("docs", "active = true", Pagination::default()).await.map(|_| ())));
        assert!(creating(store.remove("docs").await), "a store being created cannot be dropped");
        assert!(matches!(
            store.create(make_schema("docs", vec![])).await,
            Err(DocStoreError::AlreadyExists { .. })
        ));
        // The admin views still answer.
        assert_eq!(store.get_schema("docs").unwrap().indices.len(), 1);
        assert_eq!(store.list().unwrap()[0]["state"], "creating");
        store.shutdown().await.unwrap();
    }
    let store = open_fresh(db_dir.path()).await;
    assert_eq!(store.schemas.state("docs"), Some(NamespaceState::Created));
    assert_eq!(store.list().unwrap()[0]["state"], "created");
    store.put("docs", DocId::U64(1), serde_json::json!({"active": true})).await.unwrap();
    assert_eq!(store.query("docs", "active = true", Pagination::default()).await.unwrap().total, 1);
    store.shutdown().await.unwrap();
}

/// A creation run again in the same process (a restarted task) finds the
/// namespace and the active index its first run made, and keeps them.
#[tokio::test]
async fn a_creation_run_twice_keeps_what_the_first_run_made() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    store.create(make_schema("docs", vec![index()])).await.unwrap();
    store.put("docs", DocId::U64(1), serde_json::json!({"active": true})).await.unwrap();
    let id = store.schemas.get("docs").unwrap().ns_id();
    store
        .schemas
        .transition("docs", &[NamespaceState::Created], NamespaceState::Creating)
        .unwrap();
    store.provision("docs").await.unwrap();
    assert_eq!(store.schemas.get("docs").unwrap().ns_id(), id, "the same namespace");
    assert_eq!(store.query("docs", "active = true", Pagination::default()).await.unwrap().total, 1);
    store.shutdown().await.unwrap();
}

/// A store being dropped refuses every request but the admin views; its name
/// cannot be taken until it is gone; the next open finishes the drop.
#[tokio::test]
async fn a_store_being_deleted_refuses_requests_and_the_next_open_finishes_it() {
    let db_dir = TempDir::new().unwrap();
    {
        let store = open_fresh(db_dir.path()).await;
        store.create(make_schema("docs", vec![index()])).await.unwrap();
        store.put("docs", DocId::U64(1), serde_json::json!({"active": true})).await.unwrap();
        // A drop a crash cut short right after it began.
        store
            .schemas
            .transition("docs", &[NamespaceState::Created], NamespaceState::Deleting)
            .unwrap();

        let deleting = |r: Result<_, DocStoreError>| matches!(r, Err(DocStoreError::StoreDeleting { .. }));
        assert!(deleting(store.get("docs", DocId::U64(1)).await.map(|_| ())));
        assert!(deleting(store.put("docs", DocId::U64(2), serde_json::json!({})).await.map(|_| ())));
        assert!(deleting(store.query("docs", "active = true", Pagination::default()).await.map(|_| ())));
        assert!(deleting(store.scan_range("docs", DocId::U64(0), None, None, 10).await.map(|_| ())));
        assert!(deleting(store.create(make_schema("docs", vec![])).await), "the name is not free yet");
        assert_eq!(store.get_schema("docs").unwrap().namespace, "docs");
        assert_eq!(store.list().unwrap()[0]["state"], "deleting");
        store.shutdown().await.unwrap();
    }
    let store = open_fresh(db_dir.path()).await;
    assert!(!store.schemas.contains("docs"));
    assert!(!store.db.list_namespaces().iter().any(|(n, _)| n == "docs"));
    store.create(make_schema("docs", vec![index()])).await.unwrap();
    assert_eq!(
        store.query("docs", "active = true", Pagination::default()).await.unwrap().total,
        0,
        "nothing of the old store"
    );
    store.shutdown().await.unwrap();
}

/// Two drops of one store at once both end with it gone, and a store created
/// under the name afterwards is untouched by either. (Before the states, the
/// second drop could remove a store created between the two.)
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_drops_at_once_remove_the_store_once() {
    let db_dir = TempDir::new().unwrap();
    let store = std::sync::Arc::new(open_fresh(db_dir.path()).await);
    store.create(make_schema("docs", vec![])).await.unwrap();
    let (a, b) = tokio::join!(store.remove("docs"), store.remove("docs"));
    for r in [a, b] {
        match r {
            Ok(()) | Err(DocStoreError::NotFound { .. }) => {}
            Err(e) => panic!("unexpected: {e:?}"),
        }
    }
    assert!(!store.schemas.contains("docs"));
    store.create(make_schema("docs", vec![])).await.unwrap();
    store.put("docs", DocId::U64(1), serde_json::json!({})).await.unwrap();
    assert!(store.get("docs", DocId::U64(1)).await.unwrap().is_some());
    store.shutdown().await.unwrap();
}

/// A store with index jobs running takes every request and cannot be
/// dropped. Left in maintenance with no live job, it comes back from a
/// restart `Created` (the governor puts it in step with its job records before
/// the open returns), its indexes active.
#[tokio::test]
async fn a_store_in_index_maintenance_takes_requests_but_cannot_be_dropped() {
    let db_dir = TempDir::new().unwrap();
    {
        let store = open_fresh(db_dir.path()).await;
        store.create(make_schema("docs", vec![index()])).await.unwrap();
        store
            .schemas
            .transition("docs", &[NamespaceState::Created], NamespaceState::IndexMaintenance)
            .unwrap();

        store.put("docs", DocId::U64(1), serde_json::json!({"active": true})).await.unwrap();
        assert!(store.get("docs", DocId::U64(1)).await.unwrap().is_some());
        assert_eq!(store.query("docs", "active = true", Pagination::default()).await.unwrap().total, 1);
        assert!(matches!(store.remove("docs").await, Err(DocStoreError::StoreInMaintenance { .. })));
        assert_eq!(store.list().unwrap()[0]["state"], "index_maintenance");
        store.shutdown().await.unwrap();
    }
    let store = open_fresh(db_dir.path()).await;
    assert_eq!(store.schemas.state("docs"), Some(NamespaceState::Created), "no live job");
    assert_eq!(store.query("docs", "active = true", Pagination::default()).await.unwrap().total, 1);
    store.shutdown().await.unwrap();
}
