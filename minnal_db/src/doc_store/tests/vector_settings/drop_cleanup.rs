//! What a drop leaves behind: a reindex refused and cancelled, worker passes
//! and writes during a drop, exact counts, journals recovered first, and
//! files closed.

use super::support::*;
use super::*;

/// While a reindex enqueues, neither the store nor its vector index can be
/// dropped (an enqueue after the drop cleared the queue would outlive the
/// store). A cancel stops the reindex at its next page; then the store
/// drops and leaves none of its entries in the queue.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_store_cannot_be_dropped_while_a_reindex_runs_and_a_cancel_stops_it() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
    // No worker: the queue keeps what the reindex enqueues.
    store.shutdown_vec_index_worker().await;
    for id in 1..=3000 {
        store.put("g", DocId::U64(id), serde_json::json!({"text": TEXT})).await.unwrap();
    }
    let store = std::sync::Arc::new(store);
    // The puts' own entries, so the reindex's are the ones counted below.
    store.delete_all_queue_entries("g").await.unwrap();
    let reindex = tokio::spawn({
        let store = std::sync::Arc::clone(&store);
        async move { store.index_all("g").await }
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while queued_for(&store, "g").await < 50 {
        assert!(std::time::Instant::now() < deadline, "the reindex never enqueued");
        assert!(!reindex.is_finished(), "the reindex finished before the drop; make it longer");
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert!(matches!(store.remove("g").await, Err(DocStoreError::StoreInMaintenance { .. })));
    assert!(matches!(store.drop_vector_index("g").await, Err(DocStoreError::JobConflict { .. })));
    store.cancel_job("g", "vector_reindex").unwrap();
    let stats = reindex.await.unwrap().unwrap();
    assert!(
        stats.enqueued < 3000,
        "the cancel stopped the reindex part-way ({} enqueued)",
        stats.enqueued
    );
    let jobs = store.jobs("g").unwrap();
    assert_eq!(jobs[0].1.status, BuildStatus::Cancelled, "{jobs:?}");
    store.remove("g").await.unwrap();
    assert_eq!(queued_for(&store, "g").await, 0, "queue entries outlived the store");
    store.shutdown().await.unwrap();
}

/// A worker pass that starts while a store is being dropped must leave the
/// store's entries alone: their settings no longer resolve, and each one
/// failed with a warning and a retry (a log line per queued document of a
/// large store, and a retry wait after the pass).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pass_during_a_drop_skips_the_dropped_store() {
    let (url, seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
    // No worker while the entries are queued.
    store.shutdown_vec_index_worker().await;
    for id in 1..=20 {
        store.put("g", DocId::U64(id), serde_json::json!({"text": TEXT})).await.unwrap();
    }
    let ctx = std::sync::Arc::clone(store.semantic_ctx.as_ref().unwrap());
    ctx.begin_drop("g").await;
    let at_drop = seen.lock().len();
    let worker = crate::semantic_search::workers::vec_index::VecIndexWorker::start(
        std::sync::Arc::clone(&store.db),
        std::sync::Arc::clone(&ctx),
        store.worker_host(),
        std::sync::Arc::new(tokio::sync::Notify::new()),
        store.vector_index_config.clone(),
    );
    // A worker always makes its startup pass before it can see the
    // shutdown, so the pass has run once `shutdown` returns.
    worker.shutdown().await;
    ctx.end_drop("g");
    let entries: Vec<_> = vector_kv::list_queue_entries(&store.db)
        .await
        .unwrap()
        .into_iter()
        .filter(|e| e.namespace == "g")
        .collect();
    assert_eq!(entries.len(), 20, "the pass removed entries the drop owns");
    assert!(
        entries.iter().all(|e| e.retry_count == 0),
        "the pass failed entries of a store being dropped"
    );
    assert_eq!(seen.lock().len(), at_drop, "embedding calls for a store being dropped");
    store.shutdown().await.unwrap();
}

/// Writes that keep arriving while a store is dropped must neither bring
/// its namespace back nor leave embed-queue entries behind: a `put` read the
/// schema (removed last), then resolved the namespace with get-or-create
/// after the drop had removed it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writes_during_a_drop_leave_nothing_behind() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
    store.shutdown_vec_index_worker().await;
    for id in 1..=500 {
        store.put("g", DocId::U64(id), serde_json::json!({"text": TEXT})).await.unwrap();
    }
    let store = std::sync::Arc::new(store);
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let writers: Vec<_> = (0..4u64)
        .map(|w| {
            let (store, stop) = (std::sync::Arc::clone(&store), std::sync::Arc::clone(&stop));
            tokio::spawn(async move {
                let mut id = 1_000 + w * 1_000_000;
                while !stop.load(std::sync::atomic::Ordering::Acquire) {
                    let _ = store.put("g", DocId::U64(id), serde_json::json!({"text": TEXT})).await;
                    let _ = store.delete("g", DocId::U64(id - 1)).await;
                    id += 1;
                    // A refused write returns at once: let the drop run.
                    tokio::task::yield_now().await;
                }
            })
        })
        .collect();
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    store.remove("g").await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    stop.store(true, std::sync::atomic::Ordering::Release);
    for w in writers {
        w.await.unwrap();
    }
    let names: Vec<String> = store.db.list_namespaces().into_iter().map(|(n, _)| n).collect();
    assert!(!names.iter().any(|n| n == "g"), "the dropped store's namespace is back");
    let left = companions_left(&store, "g");
    assert!(left.is_empty(), "dropped namespaces are back: {left:?}");
    assert_eq!(queued_for(&store, "g").await, 0, "queue entries outlived the store");
    assert!(store.get("g", DocId::U64(1)).await.is_err(), "a read of the dropped store found it");
    store.shutdown().await.unwrap();
}

/// A document deleted while its namespace's partition is first loaded
/// (after the load counted the store, before it was installed) must not
/// leave the counts one too high. A delete's count change is dropped when no
/// partition is loaded, so unless the delete reaches the partition first, the
/// load's count, which saw the document, would stay.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_delete_during_the_first_load_keeps_the_counts_exact() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store
        .create(semantic_schema("pause_delete", Some(spec(r#"{"embedding_dim":8}"#))))
        .await
        .unwrap();
    for id in 1..=3 {
        store
            .put("pause_delete", DocId::U64(id), serde_json::json!({"text": TEXT}))
            .await
            .unwrap();
    }
    wait_for_empty_queue(&store).await;
    // No worker from here: its own first load would install a partition
    // counted after the delete and hide the foreground path under test.
    store.shutdown_vec_index_worker().await;
    let store = std::sync::Arc::new(store);
    store.semantic_ctx.as_ref().unwrap().evict("pause_delete"); // the next use loads and counts

    let (reached, release) = crate::semantic_search::context::load_pause::arm("pause_delete");
    let loading = tokio::spawn({
        let store = std::sync::Arc::clone(&store);
        async move { partition(&store, "pause_delete").await }
    });
    reached.notified().await; // counted, not yet installed
    let deleting = tokio::spawn({
        let store = std::sync::Arc::clone(&store);
        async move { store.delete("pause_delete", DocId::U64(2)).await.unwrap() }
    });
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    release.notify_one();
    deleting.await.unwrap();
    let handle = loading.await.unwrap();

    let truth: std::collections::HashMap<u32, (u64, u64)> = vector_kv::count_postings(&store.db, "pause_delete").await.unwrap();
    let counted: std::collections::HashMap<u32, (u64, u64)> = handle.snapshot().counts().snapshot().into_iter().filter(|(_, c)| c.0 > 0).collect();
    assert_eq!(counted, truth, "the loaded counts differ from the store's");
    store.shutdown().await.unwrap();
}

/// Every user of a partition gets it only after its journal is recovered:
/// a write before recovery could route to a split's children that recovery
/// then retires, chunks and all.
#[tokio::test]
async fn a_partition_is_handed_out_only_after_its_journal_is_recovered() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
    for id in 1..=5 {
        store.put("g", DocId::U64(id), serde_json::json!({"text": TEXT})).await.unwrap();
    }
    wait_for_empty_queue(&store).await;
    let handle = partition(&store, "g").await;
    // A split cut short once it is published (the copy fails), as a crash
    // would leave it.
    crate::semantic_search::vector_kv::crash_point::arm(7);
    let split = crate::semantic_search::ivf_split::split_posting(
        &store.db,
        "g",
        &handle,
        0,
        &crate::semantic_search::ivf_split::MaintenanceOps::split_only(1),
    )
    .await;
    crate::semantic_search::vector_kv::crash_point::disarm();
    assert!(split.is_err(), "set-up: the split was cut short");
    let journal = || async {
        store
            .db
            .namespace(vector_kv::ivf_journal_ns("g"))
            .await
            .unwrap()
            .scan_prefix(Vec::new())
            .await
            .unwrap()
            .len()
    };
    assert_eq!(journal().await, 1, "set-up: the split is in the journal");

    // A restart: the partition is loaded again, by its next user.
    store.semantic_ctx.as_ref().unwrap().evict("g");
    let handle = partition(&store, "g").await;
    assert_eq!(journal().await, 0, "the partition was handed out with its journal unfinished");
    assert!(!handle.needs_recovery());
    assert_eq!(
        handle.snapshot().posting_infos()[&0].state,
        crate::semantic_search::PostingState::Retired,
        "the split was finished"
    );
    store.shutdown_vec_index_worker().await;
    store.shutdown().await.unwrap();
}

/// A store with semantic search on, its vector worker running and its
/// partition split, so maintenance has run too.
#[cfg(target_os = "linux")]
async fn open_busy_semantic_store(dir: &std::path::Path, url: &str) -> std::sync::Arc<DocStore> {
    let store = open_with(dir, |b| b.semantic_search(context(url))).await;
    store
        .create(semantic_schema(
            "g",
            Some(spec(r#"{"embedding_dim":8,"maintenance":{"target_posting_size":16}}"#)),
        ))
        .await
        .unwrap();
    for i in 0..40u64 {
        store.put("g", DocId::U64(i), serde_json::json!({"text": TEXT})).await.unwrap();
    }
    wait_for_empty_queue(&store).await;
    store
}

/// However a doc store goes away, nothing keeps its database alive: not the
/// vector worker, not maintenance, not a partition. Their tasks hold the
/// database, so one that failed to exit would hold every file open.
#[cfg(target_os = "linux")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_doc_store_closes_every_file_when_it_goes() {
    let (url, _seen) = spawn_service();

    let db_dir = TempDir::new().unwrap();
    let store = open_busy_semantic_store(db_dir.path(), &url).await;
    store.shutdown().await.unwrap();
    drop(store);
    crate::support::assert_no_open_files_under(db_dir.path(), "after shutdown").await;

    let db_dir = TempDir::new().unwrap();
    drop(open_busy_semantic_store(db_dir.path(), &url).await);
    crate::support::assert_no_open_files_under(db_dir.path(), "dropped without shutdown").await;
}

/// Dropping a store closes the files of its namespace and its vector
/// companions while the rest of the database stays open.
#[cfg(target_os = "linux")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn removing_a_store_closes_its_files() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_busy_semantic_store(db_dir.path(), &url).await;
    let is_the_stores = |file: &String| file.contains("/ns_g/") || file.contains("/ns_g_");
    let held = || {
        crate::support::open_files_under(db_dir.path())
            .into_iter()
            .filter(is_the_stores)
            .collect::<Vec<_>>()
    };
    assert!(!held().is_empty(), "set-up: the store holds files");
    store.remove("g").await.unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while !held().is_empty() {
        assert!(std::time::Instant::now() < deadline, "after the drop, still open: {:#?}", held());
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    store.shutdown().await.unwrap();
}
