//! Partitions across drops and enables: none handed out while its store
//! drops, a semantic store always has one, and a drop holds up only its
//! own store.

use super::support::*;
use super::*;

/// A drop can only stop a partition that is installed. One still loading
/// when the drop begins must not be handed out afterwards unmarked: the
/// worker trusts the mark, and its write would recreate what the drop removed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_partition_loading_during_a_vector_index_drop_is_not_handed_out() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store
        .create(semantic_schema("pause_vector_drop", Some(spec(r#"{"embedding_dim":8}"#))))
        .await
        .unwrap();
    store
        .put("pause_vector_drop", DocId::U64(1), serde_json::json!({"text": TEXT}))
        .await
        .unwrap();
    wait_for_empty_queue(&store).await;
    store.shutdown_vec_index_worker().await;
    let store = std::sync::Arc::new(store);
    store.semantic_ctx.as_ref().unwrap().evict("pause_vector_drop"); // the next use loads

    let (reached, release) = crate::semantic_search::context::load_pause::arm("pause_vector_drop");
    let (settings, ns_id) = store.schemas.vector_settings("pause_vector_drop").unwrap();
    let loading = tokio::spawn({
        let store = std::sync::Arc::clone(&store);
        async move {
            let ctx = store.semantic_ctx.as_ref().unwrap();
            ctx.for_namespace(&store.db, "pause_vector_drop", ns_id, &settings)
                .await
                .map(|ns| ns.partition)
        }
    });
    reached.notified().await; // read, not yet installed
    store.drop_vector_index("pause_vector_drop").await.unwrap();
    release.notify_one();
    let loaded = loading.await.unwrap();
    assert!(
        matches!(loaded, Err(crate::semantic_search::context::ContextError::VectorIndexNotSeeded { .. })),
        "the load read a partition the drop removed: is_dropped={:?}",
        loaded.as_ref().map(|p| p.is_dropped()).ok()
    );
    assert!(companions_left(&store, "pause_vector_drop").is_empty());
    store.shutdown().await.unwrap();
}

/// As above for a whole store's drop.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_partition_loading_during_a_store_drop_is_not_handed_out() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store
        .create(semantic_schema("pause_store_drop", Some(spec(r#"{"embedding_dim":8}"#))))
        .await
        .unwrap();
    store
        .put("pause_store_drop", DocId::U64(1), serde_json::json!({"text": TEXT}))
        .await
        .unwrap();
    wait_for_empty_queue(&store).await;
    store.shutdown_vec_index_worker().await;
    let store = std::sync::Arc::new(store);
    store.semantic_ctx.as_ref().unwrap().evict("pause_store_drop");

    let (reached, release) = crate::semantic_search::context::load_pause::arm("pause_store_drop");
    let (settings, ns_id) = store.schemas.vector_settings("pause_store_drop").unwrap();
    let loading = tokio::spawn({
        let store = std::sync::Arc::clone(&store);
        async move {
            let ctx = store.semantic_ctx.as_ref().unwrap();
            ctx.for_namespace(&store.db, "pause_store_drop", ns_id, &settings)
                .await
                .map(|ns| ns.partition)
        }
    });
    reached.notified().await;
    store.remove("pause_store_drop").await.unwrap();
    release.notify_one();
    assert!(loading.await.unwrap().is_err(), "a dropped store's partition was handed out");
    assert!(companions_left(&store, "pause_store_drop").is_empty());
    store.shutdown().await.unwrap();
}

/// A vector-index drop does not stop writes. One that read the schema before
/// semantic search went off still clears the document's vectors after the
/// drop, and that must not bring the namespaces back.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_write_without_text_after_a_vector_index_drop_recreates_nothing() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
    store.drop_vector_index("g").await.unwrap();
    assert!(companions_left(&store, "g").is_empty(), "set-up: the drop removed everything");
    store.clear_doc_vectors_if_any("g", &DocId::U64(1).to_bytes()).await.unwrap();
    let left = companions_left(&store, "g");
    assert!(left.is_empty(), "recreated: {left:?}");
    store.shutdown().await.unwrap();
}

/// Every companion exists from the moment a store is semantic, however the
/// store got there: writers rely on it and never create one. A store saved
/// without them gets them at open rather than silently indexing nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_semantic_store_has_its_vector_namespaces_from_the_start() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
    let mut left = companions_left(&store, "g");
    left.sort();
    let mut all = vector_kv::companion_namespaces("g");
    all.sort();
    assert_eq!(left, all, "seeding creates them all");
    store.db.remove_namespace(vector_kv::sparse_vectors_ns("g")).await.unwrap();
    store.shutdown().await.unwrap();
    drop(store);

    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    let names = companions_left(&store, "g");
    assert!(names.contains(&vector_kv::sparse_vectors_ns("g")), "open restores it: {names:?}");
    store.put("g", DocId::U64(1), serde_json::json!({"text": TEXT})).await.unwrap();
    wait_for_empty_queue(&store).await;
    assert_eq!(store.count_indexed_docs("g").await.unwrap(), 1);
    store.shutdown().await.unwrap();
}

/// Enables and drops racing in any order: one of them is refused or both
/// run one after the other, and whenever the schema ends with semantic
/// search on, the partition is there.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn enables_racing_drops_always_leave_a_partition() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
    let enable = || SchemaAmendment::EnableVectorIndex {
        fields: vec!["text".into()],
        vector_index: None,
    };
    let expected = |e: &DocStoreError| {
        matches!(
            e,
            DocStoreError::JobConflict { .. }
                | DocStoreError::SemanticSearchNotEnabled { .. }
                | DocStoreError::Schema(SchemaError::SemanticSearchAlreadyEnabled { .. })
        )
    };
    for round in 0..30 {
        if !store.get_schema("g").unwrap().semantic_search_enabled {
            store.amend("g", enable()).await.unwrap();
        }
        let dropping = tokio::spawn({
            let store = std::sync::Arc::clone(&store);
            async move { store.drop_vector_index("g").await }
        });
        if let Err(e) = store.amend("g", enable()).await {
            assert!(expected(&e), "round {round}: enable: {e}");
        }
        if let Err(e) = dropping.await.unwrap() {
            assert!(expected(&e), "round {round}: drop: {e}");
        }
        if store.get_schema("g").unwrap().semantic_search_enabled {
            let (settings, ns_id) = store.schemas.vector_settings("g").unwrap();
            let loaded = store.semantic_ctx.as_ref().unwrap().for_namespace(&store.db, "g", ns_id, &settings).await;
            assert!(loaded.is_ok(), "round {round}: semantic search is on, but: {:?}", loaded.err());
        }
        assert_eq!(store.schemas.state("g"), Some(NamespaceState::Created), "round {round}");
    }
    store.shutdown().await.unwrap();
}

/// A caller that gives up on a store drop while it waits for maintenance
/// (axum drops the handler when its client disconnects) does not stop it:
/// the store stays `Deleting`, refusing requests, and the governor finishes
/// the drop, partition and all, once maintenance lets go.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_drop_its_caller_gave_up_on_is_finished() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
    let loaded = partition(&store, "g").await;
    let running = loaded.lock_maintenance().await;
    let ctx = store.semantic_ctx.as_ref().unwrap();
    let mut drop_store = Box::pin(store.remove("g"));
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(100), drop_store.as_mut())
            .await
            .is_err()
    );
    assert!(ctx.is_dropping("g"), "set-up: the drop reached its wait for maintenance");
    drop(drop_store); // the caller gives up
    assert_eq!(store.schemas.state("g"), Some(NamespaceState::Deleting));
    assert!(matches!(
        store.put("g", DocId::U64(1), serde_json::json!({"text": TEXT})).await,
        Err(DocStoreError::StoreDeleting { .. })
    ));
    drop(running);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while store.schemas.contains("g") {
        assert!(std::time::Instant::now() < deadline, "the drop never finished");
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert!(!ctx.is_dropping("g"));
    assert!(!store.schemas.contains("g"));
    let left: Vec<String> = store
        .db
        .list_namespaces()
        .into_iter()
        .map(|(n, _)| n)
        .filter(|n| n.starts_with("g_") || n == "g")
        .collect();
    assert!(left.is_empty(), "{left:?}");
    store.shutdown().await.unwrap();
}

/// A drop waiting out one store's maintenance (a rebuild's k-means can take
/// minutes) holds up that store's admin operations and no other's.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_waiting_drop_holds_up_only_its_own_store() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
    store.create(make_schema("other", vec![])).await.unwrap();
    let add = |name: &str| SchemaAmendment::AddAttribute {
        name: name.to_owned(),
        attr_type: AttributeType::Str,
        description: None,
    };
    let loaded = partition(&store, "g").await;
    let running = loaded.lock_maintenance().await;
    let dropping = tokio::spawn({
        let store = std::sync::Arc::clone(&store);
        async move { store.remove("g").await.unwrap() }
    });
    let ctx = store.semantic_ctx.as_ref().unwrap();
    while !ctx.is_dropping("g") {
        tokio::task::yield_now().await;
    }

    let within = std::time::Duration::from_secs(5);
    tokio::time::timeout(within, store.create(make_schema("new", vec![])))
        .await
        .expect("a create of another store waited for the drop")
        .unwrap();
    tokio::time::timeout(within, store.amend("other", add("a")))
        .await
        .expect("an amendment of another store waited for the drop")
        .unwrap();
    let own = tokio::time::timeout(std::time::Duration::from_millis(200), store.amend("g", add("b"))).await;
    assert!(own.is_err(), "an amendment of the store being dropped must wait");

    drop(running);
    dropping.await.unwrap();
    store.shutdown().await.unwrap();
}

/// A semantic store whose vector namespaces are missing at open (nothing
/// in the lifecycle leaves that, but a partition-less schema would refuse
/// every write and enable) is given its partition by the open.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_semantic_store_missing_its_partition_gets_one_at_open() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
    for name in vector_kv::companion_namespaces("g") {
        store.db.remove_namespace(name).await.unwrap();
    }
    store.shutdown().await.unwrap();
    drop(store);

    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    partition(&store, "g").await;
    store.put("g", DocId::U64(1), serde_json::json!({"text": TEXT})).await.unwrap();
    wait_for_empty_queue(&store).await;
    assert_eq!(store.count_indexed_docs("g").await.unwrap(), 1);
    store.shutdown().await.unwrap();
}

/// Counting indexed documents only reads, so asking about a store that does
/// not exist must not create anything.
#[tokio::test]
async fn counting_indexed_documents_of_an_unknown_store_creates_nothing() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    assert_eq!(store.count_indexed_docs("nope").await.unwrap(), 0);
    let left = companions_left(&store, "nope");
    assert!(left.is_empty(), "{left:?}");
    store.shutdown().await.unwrap();
}
