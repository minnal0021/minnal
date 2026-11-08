//! The embed-queue methods and single-document reindex.

use super::*;

// ── Vector-index queue methods ──────────────────────────────────────────

#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn test_vector_index_max_retries_default() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    assert_eq!(store.vector_index_max_retries(), 5);
}

#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn test_with_vector_index_config_sets_max_retries() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await.with_vector_index_config(VectorIndexConfig {
        max_retries: 10,
        retry_wait_secs: 1,
        concurrency: 2,
    });
    assert_eq!(store.vector_index_max_retries(), 10);
}

#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn test_pending_vector_index_count_empty() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    assert_eq!(store.pending_vector_index_count().await, 0);
}

#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn test_list_pending_queue_entries_empty() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    assert!(store.list_queue_entries().await.is_empty());
}

#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn test_pending_queue_count_and_list_after_enqueue() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    // Enqueue two entries directly via vector_kv primitives.
    vector_kv::enqueue_embed(&store.db, "ns_a", b"doc1", "text a").await.unwrap();
    vector_kv::enqueue_embed(&store.db, "ns_b", b"doc2", "text b").await.unwrap();

    assert_eq!(store.pending_vector_index_count().await, 2);

    let entries = store.list_queue_entries().await;
    assert_eq!(entries.len(), 2);

    let namespaces: std::collections::BTreeSet<_> = entries.iter().map(|e| e.namespace.as_str()).collect();
    assert!(namespaces.contains("ns_a"));
    assert!(namespaces.contains("ns_b"));
}

#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn test_delete_pending_queue_entry_removes_entry() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    vector_kv::enqueue_embed(&store.db, "ns", b"doc1", "text").await.unwrap();

    assert_eq!(store.pending_vector_index_count().await, 1);

    store.delete_queue_entry("ns", b"doc1").await.unwrap();

    assert_eq!(store.pending_vector_index_count().await, 0);
}

#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn test_delete_pending_queue_entry_noop_for_missing() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    // Deleting a non-existent entry should not error.
    store.delete_queue_entry("ns", b"ghost").await.unwrap();
    assert_eq!(store.pending_vector_index_count().await, 0);
}

#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn test_delete_pending_queue_entry_only_removes_target() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;

    vector_kv::enqueue_embed(&store.db, "ns", b"doc1", "t1").await.unwrap();
    vector_kv::enqueue_embed(&store.db, "ns", b"doc2", "t2").await.unwrap();

    store.delete_queue_entry("ns", b"doc1").await.unwrap();

    let entries = store.list_queue_entries().await;
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].doc_id_bytes, b"doc2");
}

/// Without the `semantic-search` feature, creating a semantic-search-enabled
/// store must be rejected at runtime rather than silently ignored.
#[cfg(not(feature = "semantic-search"))]
#[tokio::test]
async fn test_semantic_create_rejected_without_feature() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    let schema = DocStoreSchema {
        store_type: StoreType::Doc,
        namespace: "sem".to_owned(),
        ns_id: None,
        key_type: KeyType::U64,
        attributes: vec![crate::doc_store::schema::AttributeDef {
            name: "title".to_owned(),
            attr_type: AttributeType::Str,
            description: None,
        }],
        indices: vec![],
        semantic_search_enabled: true,
        embedding_fields: vec!["title".to_owned()],
        vector_index: None,
    };
    let err = store.create(schema).await.unwrap_err();
    assert!(matches!(err, DocStoreError::SemanticSearchNotCompiled), "got {err:?}");
}

/// A single-document vector reindex enqueues exactly that document, reports
/// `NotFound` for a missing id, and rejects a non-semantic namespace.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn test_reindex_doc_vector_enqueues_single_doc() {
    let db_dir = TempDir::new().unwrap();
    let store = open_fresh(db_dir.path()).await;
    create_semantic_schema(&store, "sem").await;

    // No SemanticSearchContext attached, so put() does not enqueue.
    for i in 1u64..=3 {
        store
            .put("sem", DocId::U64(i), serde_json::json!({"title": format!("doc {i}")}))
            .await
            .unwrap();
    }
    assert_eq!(store.pending_vector_index_count().await, 0);

    // Reindex just doc 2 → exactly one enqueue.
    assert_eq!(
        store.reindex_doc_vector("sem", DocId::U64(2)).await.unwrap(),
        VectorReindexOutcome::Enqueued
    );
    assert_eq!(store.pending_vector_index_count().await, 1);

    // A missing document reports NotFound and enqueues nothing more.
    assert_eq!(
        store.reindex_doc_vector("sem", DocId::U64(99)).await.unwrap(),
        VectorReindexOutcome::NotFound
    );
    assert_eq!(store.pending_vector_index_count().await, 1);

    // A namespace without semantic search is rejected.
    store
        .create(DocStoreSchema {
            store_type: StoreType::Doc,
            namespace: "plain".to_owned(),
            ns_id: None,
            key_type: KeyType::U64,
            attributes: vec![],
            indices: vec![],
            semantic_search_enabled: false,
            embedding_fields: vec![],
            vector_index: None,
        })
        .await
        .unwrap();
    store.put("plain", DocId::U64(1), serde_json::json!({"title": "x"})).await.unwrap();
    assert!(matches!(
        store.reindex_doc_vector("plain", DocId::U64(1)).await,
        Err(DocStoreError::SemanticSearchNotEnabled { .. })
    ));
}
