//! Vector jobs cut short or refused: a rebuild closed with what its journal
//! did, a cancelled rebuild, an enable finished at the next start, and an
//! enable refused while a cleanup runs.

use super::support::*;
use super::*;

/// A rebuild cut short is not run again: its job is closed with what the
/// partition's journal did. Abandoned (it never published), it is failed;
/// finished, it is complete.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rebuild_cut_short_is_closed_with_what_the_journal_did() {
    use crate::namespace_lifecycle::JobKind;
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
    for i in 0..10u64 {
        store
            .put("g", DocId::U64(i), serde_json::json!({"text": format!("{TEXT} Number {i}.")}))
            .await
            .unwrap();
    }
    wait_for_empty_queue(&store).await;
    let ns_id = store.load_schema("g").unwrap().ns_id.unwrap();
    let key = crate::doc_store::JobKey::new(ns_id, JobKind::PartitionRebuild);
    let above = partition(&store, "g").await.snapshot().max_ids().0;
    // As a crash before the plan was recorded leaves it: started, nothing done.
    let started = |store: &DocStore| {
        store.start_partition_rebuild("g", None).unwrap();
        store
            .jobs
            .save_progress(key, serde_json::json!({"postings": null, "above": above, "report": null}))
            .unwrap();
    };
    started(&store);
    store.run_live_jobs().await.unwrap();
    let record = store.jobs.get(key).unwrap().unwrap();
    assert_eq!(record.status, BuildStatus::Failed, "{record:?}");
    assert!(record.error.unwrap().contains("abandoned"));
    assert_eq!(store.schemas.state("g"), Some(NamespaceState::Created));

    // A rebuild that finished, then the same record left live.
    store.rebuild_vector_partition("g", None).await.unwrap();
    started(&store);
    store.run_live_jobs().await.unwrap();
    assert_eq!(store.jobs.get(key).unwrap().unwrap().status, BuildStatus::Complete);
    store.shutdown().await.unwrap();
}

/// A cancel that comes before the rebuild's plan is recorded stops it with
/// the partition as it was.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rebuild_cancelled_before_its_plan_changes_nothing() {
    use crate::semantic_search::ivf_split::rebuild::RebuildState;
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
    store.put("g", DocId::U64(1), serde_json::json!({"text": TEXT})).await.unwrap();
    wait_for_empty_queue(&store).await;
    let handle = partition(&store, "g").await;
    let before = handle.snapshot().max_ids();
    let busy = handle.lock_maintenance().await; // holds the rebuild back
    let job = store.start_vector_rebuild("g", None).unwrap();
    store.cancel_job("g", &job).unwrap();
    drop(busy);
    let record = store.wait_for_job("g", &job, std::time::Duration::from_secs(30)).await.unwrap();
    assert_eq!(record.status, BuildStatus::Cancelled, "{record:?}");
    assert_eq!(store.vector_rebuild_status("g").unwrap().state, RebuildState::Cancelled);
    assert_eq!(handle.snapshot().max_ids(), before, "no posting or centre was added");
    store.shutdown().await.unwrap();
}

/// An enable cut short by a crash (its job recorded, the schema not yet
/// saved) is finished at the next start, partition first.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_enable_cut_short_is_finished_at_the_next_start() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    {
        let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
        let mut schema = semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)));
        schema.semantic_search_enabled = false;
        schema.embedding_fields.clear();
        store.create(schema).await.unwrap();
        let current = store.load_schema("g").unwrap();
        let amendment = SchemaAmendment::EnableVectorIndex {
            fields: vec!["text".into()],
            vector_index: current.vector_index.clone(),
        };
        store.start_vector_enable("g", amendment).unwrap();
        store.shutdown().await.unwrap();
    }
    // The governor resumes the enable as soon as the store opens.
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    wait_for_jobs(&store, "g").await;
    assert!(store.get_schema("g").unwrap().semantic_search_enabled);
    partition(&store, "g").await;
    store.put("g", DocId::U64(1), serde_json::json!({"text": TEXT})).await.unwrap();
    wait_for_empty_queue(&store).await;
    assert_eq!(store.count_indexed_docs("g").await, 1);
    store.shutdown().await.unwrap();
}

/// While a vector cleanup runs, an enable is refused (FR-010: writes between
/// the enable and the cleanup lost their queue entries); once it is done the
/// store can enable again, and indexes from a fresh partition.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_enable_is_refused_while_a_vector_cleanup_runs() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
    store.put("g", DocId::U64(1), serde_json::json!({"text": TEXT})).await.unwrap();
    wait_for_empty_queue(&store).await;

    // Semantic search off and the cleanup recorded, not yet run.
    store.start_vector_cleanup("g", || store.turn_semantic_search_off("g")).unwrap();
    let enable = || SchemaAmendment::EnableVectorIndex {
        fields: vec!["text".into()],
        vector_index: None,
    };
    assert!(matches!(store.amend("g", enable()).await, Err(DocStoreError::JobConflict { running, .. }) if running == "vector_cleanup"));
    assert!(!store.get_schema("g").unwrap().semantic_search_enabled);
    store.run_live_jobs().await.unwrap();
    assert_eq!(store.count_indexed_docs("g").await, 0, "the cleanup removed the old vectors");

    store.amend("g", enable()).await.unwrap();
    assert!(store.get_schema("g").unwrap().semantic_search_enabled);
    partition(&store, "g").await;
    store.put("g", DocId::U64(2), serde_json::json!({"text": TEXT})).await.unwrap();
    wait_for_empty_queue(&store).await;
    assert_eq!(store.count_indexed_docs("g").await, 1, "the store indexes again");
    store.shutdown().await.unwrap();
}
