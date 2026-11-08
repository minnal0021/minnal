//! Writes cut short by a crash, and writes in progress the worker skips.

use super::*;

/// A delete interrupted by a crash leaves the document, no vectors and a
/// `Clear` tombstone. The worker compares the tombstone with the document,
/// finds the document still there, and turns the tombstone into an embed of
/// its text, so the surviving document is indexed again.
#[cfg(feature = "semantic-search")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_cut_short_by_a_crash_is_redone_as_an_embed() {
    use crate::semantic_search::service::SemanticSearchConfig;
    let db_dir = TempDir::new().unwrap();
    let store = open_kv_semantic(&db_dir).await;
    store
        .db
        .namespace("sem_kv".to_string())
        .await
        .unwrap()
        .put(b"x".to_vec(), b"hello".to_vec())
        .await
        .unwrap();
    vector_kv::remove_queue_entry(&store.db, "sem_kv", b"x").await.unwrap();
    vector_kv::clear_vectors(&store.db, "sem_kv", b"x").await.unwrap(); // the delete, cut short by a crash

    let ctx = SemanticSearchContext::new(SemanticSearchConfig {
        embedding_service_url: "http://127.0.0.1:9".into(), // nothing listens: embeds fail and stay queued
        embedding_connect_timeout: std::time::Duration::from_millis(200),
        ..SemanticSearchConfig::default()
    });
    let store = store.with_semantic_search(ctx);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        let entry = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap();
        if entry
            .as_ref()
            .is_some_and(|e| e.kind == vector_kv::QueueEntryKind::Embed && e.text == "hello")
        {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the surviving document was never queued again: {entry:?}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    store.shutdown_vec_index_worker().await;
}

/// An upsert interrupted by a crash after its queue entry leaves an entry for
/// text the store never received. The worker rewrites it to the text the
/// store holds, so the vectors end up matching the value, not the lost write.
#[cfg(feature = "semantic-search")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_upsert_cut_short_by_a_crash_embeds_what_the_store_holds() {
    use crate::semantic_search::service::SemanticSearchConfig;
    let db_dir = TempDir::new().unwrap();
    let store = open_kv_semantic(&db_dir).await;
    store
        .db
        .namespace("sem_kv".to_string())
        .await
        .unwrap()
        .put(b"x".to_vec(), b"on disk".to_vec())
        .await
        .unwrap();
    vector_kv::enqueue_embed(&store.db, "sem_kv", b"x", "never written").await.unwrap(); // its value write was lost

    let ctx = SemanticSearchContext::new(SemanticSearchConfig {
        embedding_service_url: "http://127.0.0.1:9".into(), // nothing listens: embeds fail and stay queued
        embedding_connect_timeout: std::time::Duration::from_millis(200),
        ..SemanticSearchConfig::default()
    });
    let store = store.with_semantic_search(ctx);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        let entry = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap();
        if entry.as_ref().is_some_and(|e| e.text == "on disk") {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "the entry was never rewritten: {entry:?}");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    store.shutdown_vec_index_worker().await;
}

/// A write in progress is recorded until its guard drops, however the write
/// ends: counted per concurrent write, and removed when a request's future is
/// dropped mid-write (a client disconnect). A leaked entry would make the
/// worker skip its document for good.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn a_write_in_progress_is_forgotten_however_it_ends() {
    let in_flight = Arc::new(InFlightWrites::default());
    let a = in_flight.begin("ns", b"x", None);
    let b = in_flight.begin("ns", b"x", None);
    assert!(in_flight.contains("ns", b"x"));
    assert!(!in_flight.contains("ns", b"y") && !in_flight.contains("other", b"x"));
    drop(a);
    assert!(in_flight.contains("ns", b"x"), "the second write is still in progress");
    drop(b);
    assert!(!in_flight.contains("ns", b"x"));

    // A request cancelled mid-write: its future is dropped at an await.
    let tracker = Arc::clone(&in_flight);
    let write = async move {
        let _writing = tracker.begin("ns", b"x", None);
        std::future::pending::<()>().await;
    };
    assert!(tokio::time::timeout(std::time::Duration::from_millis(20), write).await.is_err());
    assert!(!in_flight.contains("ns", b"x"), "a cancelled write left its entry behind");
}

#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn a_store_with_no_schema_reads_as_gone_to_the_workers() {
    use crate::semantic_search::workers::{WorkerError, WorkerHost};
    let db_dir = TempDir::new().unwrap();
    let store = open_kv_semantic(&db_dir).await;
    // Partition maintenance logs this at debug level: a drop is not a failure.
    assert!(matches!(
        store.worker_host().vector_settings("never_created"),
        Err(WorkerError::StoreGone(_))
    ));
    store.shutdown().await.unwrap();
}

/// The race the in-progress check closes. A writer queues new text and the
/// worker embeds and completes it before the writer's document write lands;
/// a reindex holding an older snapshot then queues the old text. Checked
/// while the write is in progress, the old text would match the document
/// and be embedded, and the new document would be left with stale vectors.
#[cfg(feature = "semantic-search")]
#[tokio::test]
async fn a_stale_entry_waits_for_a_write_in_progress() {
    use crate::semantic_search::workers::vec_index::{EntryCheck, check_entry};
    let db_dir = TempDir::new().unwrap();
    let store = open_kv_semantic(&db_dir).await;
    let parent = store.db.namespace("sem_kv".to_string()).await.unwrap();
    parent.put(b"x".to_vec(), b"old".to_vec()).await.unwrap();

    let writing = store.begin_vector_write("sem_kv", b"x");
    vector_kv::enqueue_embed(&store.db, "sem_kv", b"x", "new").await.unwrap();
    let entry = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().unwrap();
    vector_kv::finish_embed(&store.db, &entry, &race_vectors()).await.unwrap(); // completed early
    vector_kv::enqueue_embed(&store.db, "sem_kv", b"x", "old").await.unwrap(); // the reindex's snapshot
    let stale = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().unwrap();
    assert_eq!(
        check_entry(&store.db, &*store.worker_host(), &stale).await.unwrap(),
        EntryCheck::InProgress
    );

    parent.put(b"x".to_vec(), b"new".to_vec()).await.unwrap(); // the write lands
    drop(writing);
    assert_eq!(
        check_entry(&store.db, &*store.worker_host(), &stale).await.unwrap(),
        EntryCheck::Rewritten
    );
    let fixed = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().unwrap();
    assert_eq!(fixed.text, "new");
    assert_eq!(check_entry(&store.db, &*store.worker_host(), &fixed).await.unwrap(), EntryCheck::Proceed);
}

/// The running worker leaves an entry alone while its document has a write
/// in progress, and checks it once the write ends (the guard's drop wakes it).
#[cfg(feature = "semantic-search")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_worker_skips_a_document_with_a_write_in_progress() {
    use crate::semantic_search::service::SemanticSearchConfig;
    let db_dir = TempDir::new().unwrap();
    let store = open_kv_semantic(&db_dir).await;
    store
        .db
        .namespace("sem_kv".to_string())
        .await
        .unwrap()
        .put(b"x".to_vec(), b"on disk".to_vec())
        .await
        .unwrap();

    let ctx = SemanticSearchContext::new(SemanticSearchConfig {
        embedding_service_url: "http://127.0.0.1:9".into(), // nothing listens: embeds fail and stay queued
        embedding_connect_timeout: std::time::Duration::from_millis(200),
        ..SemanticSearchConfig::default()
    });
    let store = store.with_semantic_search(ctx);
    let writing = store.begin_vector_write("sem_kv", b"x");
    vector_kv::enqueue_embed(&store.db, "sem_kv", b"x", "being written").await.unwrap();
    store.notify.as_ref().unwrap().notify_one(); // a pass while the write is in progress
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let entry = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap().unwrap();
    assert_eq!(
        (entry.text.as_str(), entry.retry_count),
        ("being written", 0),
        "touched while in progress"
    );

    drop(writing); // the write ends without its document write landing
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        let entry = vector_kv::get_queue_entry(&store.db, "sem_kv", b"x").await.unwrap();
        if entry.as_ref().is_some_and(|e| e.text == "on disk") {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "never checked after the write ended: {entry:?}");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    store.shutdown_vec_index_worker().await;
}
