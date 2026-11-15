//! A store or vector-index drop racing maintenance, the worker, reads,
//! searches and writes: the drop waits or stops them, and nothing is
//! recreated.

use super::support::*;
use super::*;

// ── Drops racing maintenance and writes ──

/// Dropping a vector index, or its whole store, while a maintenance
/// operation runs must wait for the operation and stop maintenance on that
/// partition. A write it made after the drop would land in a store being
/// removed, or leave stray postings in a re-enabled partition.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_drop_waits_for_running_maintenance_and_stops_it() {
    for drop_store in [false, true] {
        let (url, _seen) = spawn_service();
        let db_dir = TempDir::new().unwrap();
        let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
        store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
        store.put("g", DocId::U64(1), serde_json::json!({"text": TEXT})).await.unwrap();
        wait_for_empty_queue(&store).await;
        let handle = partition(&store, "g").await;
        let ops = store.schemas.vector_settings("g").unwrap().0.maintenance.ops();
        let store = std::sync::Arc::new(store);

        // An operation in progress: it holds the maintenance lock and writes.
        let running = handle.lock_maintenance().await;
        let dropping = tokio::spawn({
            let store = std::sync::Arc::clone(&store);
            async move {
                if drop_store {
                    store.remove("g").await.unwrap();
                } else {
                    store.drop_vector_index("g").await.unwrap();
                }
            }
        });
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        store
            .db
            .namespace(vector_kv::ivf_journal_ns("g"))
            .await
            .unwrap()
            .put(b"op".to_vec(), b"record".to_vec())
            .await
            .unwrap();
        drop(running);
        dropping.await.unwrap();
        // Maintenance on the partition it held, as a queued request would run it.
        assert!(
            crate::semantic_search::ivf_split::maintain_partition(&store.db, "g", &handle, &ops)
                .await
                .is_err(),
            "drop_store={drop_store}: maintenance went on after the drop"
        );
        let left = companions_left(&store, "g");
        assert!(left.is_empty(), "drop_store={drop_store}: dropped namespaces are back: {left:?}");
        store.shutdown_vec_index_worker().await;
        store.shutdown().await.unwrap();
    }
}

/// Dropping a store while the vector worker is part-way through a pass of
/// its entries must stop the pass on that store: the worker resolved the
/// store's partition at the start of the pass and went on embedding the
/// rest of its entries after the drop (embedding calls for a store that is
/// gone, maintenance requests that failed, vector writes racing the
/// removal of the namespaces they recreate).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_drop_stops_the_worker_pass_on_its_store() {
    let (url, seen) = spawn_slow_service(std::time::Duration::from_millis(20));
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    store.create(semantic_schema("g", Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
    for id in 1..=60 {
        store.put("g", DocId::U64(id), serde_json::json!({"text": TEXT})).await.unwrap();
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while seen.lock().len() < 4 {
        assert!(std::time::Instant::now() < deadline, "the worker never started embedding");
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    store.remove("g").await.unwrap();
    let at_drop = seen.lock().len();
    tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
    let after = seen.lock().len() - at_drop;
    // Requests already sent when the drop ended may still be answered: at
    // most two (whole document, chunks) per concurrent entry.
    assert!(after <= 8, "{after} embedding request(s) after the drop");
    assert!(!store.db.list_namespaces().iter().any(|(n, _)| n == "g"), "the dropped store is back");
    let left = companions_left(&store, "g");
    assert!(left.is_empty(), "dropped namespaces are back: {left:?}");
    assert_eq!(queued_for(&store, "g").await, 0, "queue entries outlived the store");
    store.shutdown_vec_index_worker().await;
    store.shutdown().await.unwrap();
}

/// The worker reads a document to check its queue entry, outside any
/// partition's epoch. That read racing a drop must not bring the store back:
/// the drop's sweep of leftover directories then deleted the new store's
/// files, and the next metadata write, at shutdown at the latest, failed with
/// `NotFound`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_document_read_racing_a_drop_does_not_bring_the_store_back() {
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    for round in 0..40 {
        let name = format!("g{round}");
        store.create(semantic_schema(&name, Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let (store, name, stop) = (Arc::clone(&store), name.clone(), Arc::clone(&stop));
                tokio::spawn(async move {
                    while !stop.load(std::sync::atomic::Ordering::Acquire) {
                        let _ = crate::doc_store::vector::document_text(&store.db, &store.schemas, &name, b"k").await;
                        tokio::task::yield_now().await;
                    }
                })
            })
            .collect();
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        store.remove(&name).await.unwrap();
        stop.store(true, std::sync::atomic::Ordering::Release);
        for r in readers {
            r.await.unwrap();
        }
        let names: Vec<String> = store.db.list_namespaces().into_iter().map(|(n, _)| n).collect();
        assert!(!names.contains(&name), "round {round}: the dropped store is back");
    }
    store.shutdown().await.unwrap();
}

/// Create a store, run `call` from several tasks while it is dropped, and
/// check that nothing of the store came back. Repeated, since the window is
/// small, and shut down last: a store left without its files fails there.
async fn assert_call_racing_a_drop_recreates_nothing<F, Fut>(rounds: u32, with_docs: bool, call: F)
where
    F: Fn(Arc<DocStore>, String, u64) -> Fut + Clone + Send + Sync + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let (url, _seen) = spawn_service();
    let db_dir = TempDir::new().unwrap();
    let store = open_with(db_dir.path(), |b| b.semantic_search(context(&url))).await;
    for round in 0..rounds {
        let name = format!("g{round}");
        store.create(semantic_schema(&name, Some(spec(r#"{"embedding_dim":8}"#)))).await.unwrap();
        if with_docs {
            store.put(&name, DocId::U64(0), serde_json::json!({"text": TEXT})).await.unwrap();
            wait_for_empty_queue(&store).await;
        }
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let tasks: Vec<_> = (0..4u64)
            .map(|task| {
                let (store, name, stop, call) = (Arc::clone(&store), name.clone(), Arc::clone(&stop), call.clone());
                tokio::spawn(async move {
                    let mut i = task * 1_000_000 + 1;
                    while !stop.load(std::sync::atomic::Ordering::Acquire) {
                        call(Arc::clone(&store), name.clone(), i).await;
                        i += 1;
                        tokio::task::yield_now().await;
                    }
                })
            })
            .collect();
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        store.remove(&name).await.unwrap();
        stop.store(true, std::sync::atomic::Ordering::Release);
        for t in tasks {
            t.await.unwrap();
        }
        let names: Vec<String> = store.db.list_namespaces().into_iter().map(|(n, _)| n).collect();
        assert!(!names.contains(&name), "round {round}: the dropped store is back");
        let left = companions_left(&store, &name);
        assert!(left.is_empty(), "round {round}: dropped namespaces are back: {left:?}");
    }
    store.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_search_racing_a_drop_recreates_nothing() {
    assert_call_racing_a_drop_recreates_nothing(15, true, |store, name, _| async move {
        let _ = store
            .search_semantic(&name, "hello", &SearchSpec::default(), Pagination::new(1, 10))
            .await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn counting_indexed_documents_racing_a_drop_recreates_nothing() {
    assert_call_racing_a_drop_recreates_nothing(5, false, |store, name, _| async move {
        let _ = store.count_indexed_docs(&name).await;
    })
    .await;
}

/// A document with no embedding text checks for vector state to clear
/// (`has_any_vector_state`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_write_without_embedding_text_racing_a_drop_recreates_nothing() {
    // No `text` at all: `{"text": ""}` still builds the embedding text "text: ".
    assert_call_racing_a_drop_recreates_nothing(10, false, |store, name, i| async move {
        let _ = store.put(&name, DocId::U64(i), serde_json::json!({})).await;
    })
    .await;
}
