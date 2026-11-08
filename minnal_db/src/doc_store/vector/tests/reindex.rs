//! Store reindexes, and what an interrupted one or an interrupted drop leaves.

use super::*;
#[cfg(feature = "semantic-search")]
use crate::doc_store::vector::reindex::REINDEX_PAGE;

/// Cost of the open-time drop sweep's queue scan when a crash mid bulk-load
/// left the whole corpus queued (57,600 entries of ~1.2 KB text): reading
/// every entry against reading only the keys, and a full `DocStore` open.
/// `MINNAL_QUEUE_DOCS` overrides the size. Run with `--ignored --nocapture`.
#[cfg(feature = "semantic-search")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn open_cost_with_a_full_vector_queue() {
    let n: usize = std::env::var("MINNAL_QUEUE_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(57_600);
    let db_dir = TempDir::new().unwrap();
    {
        let store = open_kv_semantic(&db_dir).await;
        let body = "Interest rates, bond yields and portfolio risk over the market cycle. ".repeat(17);
        for i in 0..n {
            vector_kv::enqueue_embed(&store.db, "sem_kv", format!("{i:08}").as_bytes(), &format!("{i} {body}"))
                .await
                .unwrap();
        }
        store.db.shutdown().await.unwrap();
    }
    let t = std::time::Instant::now();
    let store = open_fresh(db_dir.path()).await;
    let open = t.elapsed();
    let t = std::time::Instant::now();
    let entries = vector_kv::list_queue_entries(&store.db).await.unwrap();
    let full = t.elapsed();
    let t = std::time::Instant::now();
    let namespaces = vector_kv::queued_namespaces(&store.db).await.unwrap();
    let keys = t.elapsed();
    assert_eq!((entries.len(), namespaces.len()), (n, 1));
    eprintln!(
        "\n=== open with {n} queued entries ===\n  DocStore open (keys-only sweep included): {:.3} s\n  queue scan reading every entry (old sweep): {:.3} s\n  queue scan reading keys only (new sweep):   {:.3} s",
        open.as_secs_f64(),
        full.as_secs_f64(),
        keys.as_secs_f64()
    );
}

/// A vector-index drop cut short by a crash (schema saved with semantic
/// search off, data and queue not yet cleared) is finished at the next start,
/// before a worker could process the leftover queue entries.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn an_interrupted_vector_index_drop_is_finished_at_the_next_start() {
    let db_dir = TempDir::new().unwrap();
    {
        let store = open_kv_semantic(&db_dir).await;
        store
            .kv_put("sem_kv", &serde_json::json!("x"), &serde_json::json!("some text"))
            .await
            .unwrap(); // queued
        let entry = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().unwrap();
        vector_kv::upsert_vectors(&store.db, "sem_kv", b"x", &entry.text, &race_vectors())
            .await
            .unwrap();
        store
            .kv_put("sem_kv", &serde_json::json!("y"), &serde_json::json!("other text"))
            .await
            .unwrap(); // still queued
        // The drop's first step only: the job and the schema, then a crash.
        store.start_vector_cleanup("sem_kv", || store.turn_semantic_search_off("sem_kv")).unwrap();
        store.db.shutdown().await.unwrap();
    }
    let store = open_fresh(db_dir.path()).await;
    store.run_pending_jobs().await.unwrap();
    let names: Vec<String> = store.db.list_namespaces().into_iter().map(|(n, _)| n).collect();
    assert!(
        !names.iter().any(|n| vector_kv::companion_base(n) == Some("sem_kv")),
        "vector namespaces of the dropped index remain: {names:?}"
    );
    assert!(store.list_queue_entries().await.is_empty(), "queue entries of the dropped index remain");
}

/// A finished reindex must not block the next one. Its record used to stay
/// `"running"` for good, so every later `reindex-all` was refused as already
/// in progress.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn a_finished_reindex_does_not_block_the_next() {
    let db_dir = TempDir::new().unwrap();
    let store = open_kv_semantic(&db_dir).await;
    store
        .kv_put("sem_kv", &serde_json::json!("x"), &serde_json::json!("some text"))
        .await
        .unwrap();
    store.kv_index_all("sem_kv").await.unwrap();
    // The worker finishes the work: the queue empties.
    for e in store.list_queue_entries().await {
        vector_kv::remove_queue_entry(&store.db, &e.namespace, &e.doc_id_bytes).await.unwrap();
    }
    assert_eq!(store.vec_reindex_progress("sem_kv").await.unwrap().status, BuildStatus::Complete);
    store.kv_index_all("sem_kv").await.expect("a second reindex must be allowed");
    // While its work is still queued, it is running.
    assert_eq!(store.vec_reindex_progress("sem_kv").await.unwrap().status, BuildStatus::Running);
}

/// A reindex reads the store a page at a time: every entry is enqueued across
/// page boundaries, and exhausted entries are cleared from every page of the
/// queue, without touching another store's entries.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn a_reindex_covers_every_page() {
    let db_dir = TempDir::new().unwrap();
    let store = open_kv_semantic(&db_dir).await;
    let parent = store.db.namespace("sem_kv".to_string()).await.unwrap();
    let n = REINDEX_PAGE * 2 + 7;
    for i in 0..n {
        parent
            .put(format!("{i:06}").into_bytes(), format!("text {i}").into_bytes())
            .await
            .unwrap();
    }
    parent.put(b"empty".to_vec(), Vec::new()).await.unwrap(); // nothing to embed
    // Exhausted entries spread over more than one queue page, and one of
    // another store, which must stay.
    let max = store.vector_index_max_retries();
    for (ns, key) in [("sem_kv", "000000"), ("sem_kv", "001500"), ("other", "x")] {
        vector_kv::enqueue_embed(&store.db, ns, key.as_bytes(), "stale").await.unwrap();
        for _ in 0..max {
            let e = vector_kv::get_queue_entry(&store.db, ns, key.as_bytes()).await.unwrap().unwrap();
            vector_kv::record_queue_failure(&store.db, &e, "boom").await.unwrap();
        }
    }

    let stats = store.kv_index_all("sem_kv").await.unwrap();
    assert_eq!((stats.enqueued, stats.exhausted_cleared), (n, 2));
    let queued = vector_kv::list_queue_entries(&store.db).await.unwrap();
    assert_eq!(queued.iter().filter(|e| e.namespace == "sem_kv" && e.retry_count == 0).count(), n);
    assert!(queued.iter().any(|e| e.namespace == "other"), "another store's entry was removed");
    let progress = store.vec_reindex_progress("sem_kv").await.unwrap();
    assert_eq!((progress.total_enqueued, progress.exhausted_cleared), (n, 2));
}

/// A reindex cut short by a crash leaves its job live; the next start
/// resumes it and enqueues what it had not, and a new one can follow.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn a_reindex_interrupted_by_a_crash_is_resumed_at_the_next_start() {
    let db_dir = TempDir::new().unwrap();
    {
        let store = open_kv_semantic(&db_dir).await;
        store
            .kv_put("sem_kv", &serde_json::json!("x"), &serde_json::json!("some text"))
            .await
            .unwrap();
        // The worker embedded it: nothing queued.
        for e in store.list_queue_entries().await {
            vector_kv::remove_queue_entry(&store.db, &e.namespace, &e.doc_id_bytes).await.unwrap();
        }
        // What a crash right after the reindex started leaves behind.
        store.start_vector_reindex("sem_kv").unwrap();
        store.db.shutdown().await.unwrap();
    }
    let store = with_worker_notify(open_fresh(db_dir.path()).await);
    assert_eq!(store.vec_reindex_progress("sem_kv").await.unwrap().status, BuildStatus::Running);
    store.run_pending_jobs().await.unwrap();
    assert!(
        vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().is_some(),
        "the resumed reindex enqueued it"
    );
    let progress = store.vec_reindex_progress("sem_kv").await.unwrap();
    assert!(progress.enqueue_done && progress.total_enqueued == 1, "{progress:?}");
    store.kv_index_all("sem_kv").await.expect("a new reindex must be allowed");
}
