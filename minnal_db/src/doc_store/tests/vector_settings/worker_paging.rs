//! The vector worker reads the queue a page of each namespace per pass (four
//! entries in tests), with a cursor per namespace for the round.

use super::support::*;
use super::*;

/// Entries that have used up their retries stay in the queue, ahead of newer
/// ones in key order. Nine of them fill more than two pages, so a worker that
/// read each namespace from its start on every pass would never reach the live
/// entry behind them: the cursor has to move past them.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn exhausted_entries_do_not_hold_up_the_entries_behind_them() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
    // No worker while the queue is set up.
    store.shutdown_vec_index_worker().await;
    for id in 1..=10 {
        store.put("g", DocId::U64(id), serde_json::json!({"text": TEXT})).await.unwrap();
    }
    let max_retries = store.vector_index_config.max_retries;
    for id in 1..=9u64 {
        for _ in 0..max_retries {
            let entry = vector_kv::get_queue_entry(&store.db, "g", &id.to_be_bytes()).await.unwrap().unwrap();
            vector_kv::record_queue_failure(&store.db, &entry, "boom").await.unwrap();
        }
    }

    let worker = crate::semantic_search::workers::vec_index::VecIndexWorker::start(
        std::sync::Arc::clone(&store.db),
        std::sync::Arc::clone(store.semantic_ctx.as_ref().unwrap()),
        store.worker_host(),
        std::sync::Arc::new(tokio::sync::Notify::new()),
        store.vector_index_config.clone(),
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while vector_kv::get_queue_entry(&store.db, "g", &10u64.to_be_bytes()).await.unwrap().is_some() {
        assert!(
            std::time::Instant::now() < deadline,
            "the entry behind the exhausted ones was never carried out"
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    worker.shutdown().await;

    let left = vector_kv::list_queue_entries(&store.db).await.unwrap();
    assert_eq!(left.len(), 9, "{left:?}");
    assert!(left.iter().all(|e| e.retry_count >= max_retries), "{left:?}");
    assert_eq!(store.count_indexed_docs("g").await.unwrap(), 1);
    store.shutdown().await.unwrap();
}
