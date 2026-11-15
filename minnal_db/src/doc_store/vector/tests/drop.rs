//! Dropping a store's vector index.

use super::*;

/// Dropping a KV store's vector index turns semantic search off and removes its
/// queue entries, vector namespaces and reindex record; its values stay.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn a_kv_store_can_drop_its_vector_index() {
    let db_dir = TempDir::new().unwrap();
    let store = open_kv_semantic(&db_dir).await;
    let (key, value) = (serde_json::json!("x"), serde_json::json!("some text"));
    store.kv_put("sem_kv", &key, &value).await.unwrap();
    vector_kv::create_vector_namespaces(&store.db, "sem_kv").await.unwrap();
    assert!(vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().is_some());
    let ns_id = store.load_kv_schema("sem_kv").unwrap().ns_id.unwrap();
    store.kv_index_all("sem_kv").await.unwrap();
    assert!(store.jobs.get(JobKey::vector_reindex(ns_id)).unwrap().is_some());

    store.drop_vector_index("sem_kv").await.unwrap();
    assert!(!store.load_kv_schema("sem_kv").unwrap().is_semantic_search_enabled());
    assert!(matches!(
        store.drop_vector_index("sem_kv").await,
        Err(DocStoreError::SemanticSearchNotEnabled { .. })
    ));

    assert!(vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().is_none());
    let companions: Vec<_> = store
        .db
        .list_namespaces()
        .into_iter()
        .filter(|(n, _)| vector_kv::vector_data_base(n).is_some())
        .collect();
    assert!(companions.is_empty(), "{companions:?}");
    let jobs: Vec<String> = store.jobs("sem_kv").unwrap().into_iter().map(|(n, _)| n).collect();
    assert_eq!(jobs, vec!["vector_cleanup".to_owned()], "the reindex record is removed");
    assert_eq!(store.kv_get("sem_kv", &key).await.unwrap(), Some(value.clone()));

    // New writes queue nothing.
    store.kv_put("sem_kv", &serde_json::json!("y"), &value).await.unwrap();
    assert!(vector_kv::get_queue_entry(&store.db, "sem_kv", b"y").await.unwrap().is_none());
}

/// A crash after semantic search went off and before the cleanup ran leaves
/// the `vector_cleanup` job live; the next start finishes it.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn an_interrupted_kv_vector_drop_is_finished_at_the_next_start() {
    let db_dir = TempDir::new().unwrap();
    let store = open_kv_semantic(&db_dir).await;
    let (key, value) = (serde_json::json!("x"), serde_json::json!("some text"));
    store.kv_put("sem_kv", &key, &value).await.unwrap();
    vector_kv::create_vector_namespaces(&store.db, "sem_kv").await.unwrap();
    store.start_vector_cleanup("sem_kv", || store.turn_semantic_search_off("sem_kv")).unwrap();
    store.shutdown().await.unwrap();
    drop(store);

    // The governor resumes the cleanup as soon as the store opens.
    let store = open_fresh(db_dir.path()).await;
    wait_for_jobs(&store, "sem_kv").await;
    assert!(vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().is_none());
    assert!(!store.db.list_namespaces().iter().any(|(n, _)| vector_kv::vector_data_base(n).is_some()));
    assert!(!store.load_kv_schema("sem_kv").unwrap().is_semantic_search_enabled());
    assert_eq!(store.kv_get("sem_kv", &key).await.unwrap(), Some(value));
    assert_eq!(store.schemas.state("sem_kv"), Some(NamespaceState::Created));
}

/// A store that does not exist is `NotFound`, for the route's 404.
#[tokio::test]
async fn dropping_the_vector_index_of_an_unknown_store_is_not_found() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    assert!(matches!(store.drop_vector_index("missing").await, Err(DocStoreError::NotFound { .. })));
}

/// Through the real delete path: deleting a key whose older value is being
/// embedded leaves no vectors once that embed completes (the delete leaves a
/// `Clear` entry the completing embed finds).
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn kv_delete_during_embed_leaves_no_vectors() {
    let db_dir = TempDir::new().unwrap();
    let store = open_kv_semantic(&db_dir).await;
    store
        .kv_put("sem_kv", &serde_json::json!("x"), &serde_json::json!("some text"))
        .await
        .unwrap();
    let in_flight = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().expect("queued");

    store.kv_delete("sem_kv", "x").await.unwrap();
    vector_kv::finish_embed(&store.db, &in_flight, &race_vectors()).await.unwrap();

    assert!(!vector_kv::has_any_vector_state(&store.db, "sem_kv", b"x").await.unwrap());
}

/// A KV value updated to empty text must stop matching its old text —
/// the vectors and any pending embed of the older value are cleared.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn kv_put_empty_text_clears_old_vectors() {
    let db_dir = TempDir::new().unwrap();
    let store = open_kv_semantic(&db_dir).await;
    store
        .kv_put("sem_kv", &serde_json::json!("x"), &serde_json::json!("some text"))
        .await
        .unwrap();
    let entry = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().unwrap();
    vector_kv::finish_embed(&store.db, &entry, &race_vectors()).await.unwrap();
    assert!(dense_entry(&store, b"x").await.is_some());

    store.kv_put("sem_kv", &serde_json::json!("x"), &serde_json::json!("")).await.unwrap();

    assert!(dense_entry(&store, b"x").await.is_none(), "vectors of the old text must be gone");
    let pending = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().expect("tombstone");
    assert_eq!(pending.kind, vector_kv::QueueEntryKind::Clear);
}

/// A document whose embedding field is emptied is cleared the
/// same way, including a pending embed of its older text.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn doc_put_empty_embedding_text_clears_pending_embed() {
    let db_dir = TempDir::new().unwrap();
    let store = open_without_vector_worker(db_dir.path()).await;
    create_semantic_schema(&store, "sem").await;
    store.put("sem", DocId::U64(1), serde_json::json!({"title": "old title"})).await.unwrap();
    let key = DocId::U64(1).to_bytes();
    assert_eq!(
        vector_kv::get_queue_entry(&store.db, "sem", &key).await.unwrap().unwrap().kind,
        vector_kv::QueueEntryKind::Embed
    );

    store.put("sem", DocId::U64(1), serde_json::json!({})).await.unwrap();

    let pending = vector_kv::get_queue_entry(&store.db, "sem", &key).await.unwrap().expect("tombstone");
    assert_eq!(pending.kind, vector_kv::QueueEntryKind::Clear, "the stale embed must not survive");
}

/// Writing empty text for a document that never had vectors costs nothing:
/// no tombstone, so bulk loads of documents without embedding text pay no
/// extra writes.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn empty_text_on_unindexed_doc_writes_no_queue_entry() {
    let db_dir = TempDir::new().unwrap();
    let store = open_kv_semantic(&db_dir).await;
    store.kv_put("sem_kv", &serde_json::json!("y"), &serde_json::json!("")).await.unwrap();
    assert!(vector_kv::get_queue_entry(&store.db, "sem_kv", b"y").await.unwrap().is_none());
}
