//! Queue race tests: the conditional rules R1–R3 in `vector_kv/queue.rs`.
//!
//! The worker snapshots the whole queue at the start of a pass, then embeds each
//! snapshot entry. These tests stand in for the embedding step: they take the
//! snapshot entry, apply a racing operation, then run the worker's post-embed step
//! with vectors "embedded" from the snapshot text.

use super::super::*;
use tempfile::TempDir;

async fn open_db(dir: &TempDir) -> AsyncDb {
    let db = AsyncDb::open_with_config(dir.path().to_owned(), crate::support::test_db_config())
        .await
        .unwrap();
    db.namespace("docs".to_string()).await.unwrap();
    create_vector_namespaces(&db, "docs").await.unwrap();
    db
}

/// A doc's vectors as embedded from some text: one dense + one sparse chunk.
/// `tag` makes different texts produce distinguishable dense entries.
fn vectors_for(tag: f32) -> Vec<VectorIndex> {
    vec![
        VectorIndex::new(1, QuantisationStyle::MultiBit { number_of_bits: 8 }, tag, 0.0, 0.01, vec![]),
        VectorIndex::new(3, QuantisationStyle::SingleBit, tag, 0.0, 0.01, vec![]),
    ]
}

async fn snapshot(db: &AsyncDb, doc: &[u8]) -> QueueEntry {
    get_queue_entry(db, "docs", doc).await.unwrap().expect("entry queued")
}

async fn dense_tag(db: &AsyncDb, doc: &[u8]) -> Option<f32> {
    let store = DbVectorStore::new(db, "docs").await.unwrap();
    let raw = store.get_dense_entry(doc).await.unwrap()?;
    let list = VectorIndex::access_list(&raw).unwrap();
    Some(list.iter().next().unwrap().addition_factor())
}

/// R1: an upsert that lands while its doc's older text is being embedded must
/// not be lost: a worker that deleted the queue key unconditionally after
/// writing the stale vectors would destroy the newer entry.
#[tokio::test]
async fn r1_upsert_during_embed_is_not_lost() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    enqueue_embed(&db, "docs", b"x", "old text").await.unwrap();
    let in_flight = snapshot(&db, b"x").await;

    enqueue_embed(&db, "docs", b"x", "new text").await.unwrap(); // racing upsert
    finish_embed(&db, &in_flight, &vectors_for(1.0)).await.unwrap();

    let pending = get_queue_entry(&db, "docs", b"x").await.unwrap();
    assert_eq!(
        pending.map(|e| e.text).as_deref(),
        Some("new text"),
        "the newer upsert's entry must survive the older embed's completion"
    );
}

/// A store dropped and recreated while the worker discards the old store's
/// entry: the new write's entry must survive, or its document is never embedded.
#[tokio::test]
async fn discarding_an_orphaned_entry_keeps_a_newer_one() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    enqueue_embed(&db, "docs", b"x", "old store").await.unwrap();
    let orphaned = snapshot(&db, b"x").await;
    enqueue_embed(&db, "docs", b"x", "new store").await.unwrap();

    discard_orphaned_entry(&db, &orphaned).await.unwrap();
    assert_eq!(
        get_queue_entry(&db, "docs", b"x").await.unwrap().map(|e| e.text).as_deref(),
        Some("new store")
    );

    let current = snapshot(&db, b"x").await;
    discard_orphaned_entry(&db, &current).await.unwrap();
    assert!(get_queue_entry(&db, "docs", b"x").await.unwrap().is_none(), "an unchanged entry goes");
}

/// R2: a failed embed of the older text must not rewrite the entry back to it.
#[tokio::test]
async fn r2_failure_does_not_overwrite_newer_text() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    enqueue_embed(&db, "docs", b"x", "old text").await.unwrap();
    let in_flight = snapshot(&db, b"x").await;

    enqueue_embed(&db, "docs", b"x", "new text").await.unwrap(); // racing upsert
    record_queue_failure(&db, &in_flight, "embedding service timeout").await.unwrap();

    let pending = get_queue_entry(&db, "docs", b"x").await.unwrap().expect("entry kept");
    assert_eq!(pending.text, "new text", "a failure must not resurrect the older text");
    assert_eq!(pending.retry_count, 0, "the newer entry has not failed yet");
}

/// R3: a delete that lands while the doc is being embedded must leave no
/// vectors behind once the embed completes.
#[tokio::test]
async fn r3_delete_during_embed_leaves_no_vectors() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    enqueue_embed(&db, "docs", b"x", "text").await.unwrap();
    let in_flight = snapshot(&db, b"x").await;

    clear_vectors(&db, "docs", b"x").await.unwrap(); // racing delete
    finish_embed(&db, &in_flight, &vectors_for(1.0)).await.unwrap();

    assert!(
        !has_any_vector_state(&db, "docs", b"x").await.unwrap() && dense_tag(&db, b"x").await.is_none(),
        "a deleted doc must not keep vectors written by an embed that was in flight"
    );
}

/// R3, continued: the in-flight worker also retires the tombstone it found.
#[tokio::test]
async fn r3_inflight_worker_retires_the_tombstone() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    enqueue_embed(&db, "docs", b"x", "text").await.unwrap();
    let in_flight = snapshot(&db, b"x").await;
    clear_vectors(&db, "docs", b"x").await.unwrap();
    finish_embed(&db, &in_flight, &vectors_for(1.0)).await.unwrap();
    assert!(get_queue_entry(&db, "docs", b"x").await.unwrap().is_none());
}

/// A tombstone with no worker in flight: processing it removes the vectors and itself.
#[tokio::test]
async fn clear_tombstone_removes_vectors_and_itself() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    upsert_vectors(&db, "docs", b"x", "text", &vectors_for(1.0)).await.unwrap();
    clear_vectors(&db, "docs", b"x").await.unwrap();
    let tombstone = snapshot(&db, b"x").await;
    assert_eq!(tombstone.kind, QueueEntryKind::Clear);
    assert!(dense_tag(&db, b"x").await.is_none(), "clear_vectors deletes synchronously");

    process_clear(&db, "docs", b"x").await.unwrap();
    assert!(get_queue_entry(&db, "docs", b"x").await.unwrap().is_none());
    assert!(!has_any_vector_state(&db, "docs", b"x").await.unwrap());
}

/// Delete, then re-create before the worker gets to the tombstone: the
/// re-created document's embed must survive the tombstone's processing.
#[tokio::test]
async fn reupsert_after_clear_is_kept() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    clear_vectors(&db, "docs", b"x").await.unwrap();
    let tombstone = snapshot(&db, b"x").await;
    enqueue_embed(&db, "docs", b"x", "reborn").await.unwrap();

    process_clear(&db, &tombstone.namespace, &tombstone.doc_id_bytes).await.unwrap();
    let pending = get_queue_entry(&db, "docs", b"x").await.unwrap().expect("embed kept");
    assert_eq!((pending.kind, pending.text.as_str()), (QueueEntryKind::Embed, "reborn"));
}

/// The same text re-enqueued while it is being embedded completes normally:
/// the vectors just written are for the latest text.
#[tokio::test]
async fn same_text_reenqueued_mid_embed_completes() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    enqueue_embed(&db, "docs", b"x", "same").await.unwrap();
    let in_flight = snapshot(&db, b"x").await;
    enqueue_embed(&db, "docs", b"x", "same").await.unwrap();
    finish_embed(&db, &in_flight, &vectors_for(1.0)).await.unwrap();
    assert!(get_queue_entry(&db, "docs", b"x").await.unwrap().is_none());
}

/// A failure is recorded against an unchanged entry, and never re-creates one
/// that was removed meanwhile.
#[tokio::test]
async fn failure_is_recorded_only_against_the_same_entry() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    enqueue_embed(&db, "docs", b"x", "text").await.unwrap();
    let in_flight = snapshot(&db, b"x").await;
    assert_eq!(record_queue_failure(&db, &in_flight, "boom").await.unwrap(), Some(1));
    let failed = snapshot(&db, b"x").await;
    assert_eq!((failed.retry_count, failed.last_error.as_deref()), (1, Some("boom")));

    remove_queue_entry(&db, "docs", b"x").await.unwrap();
    assert_eq!(record_queue_failure(&db, &failed, "again").await.unwrap(), None);
    assert!(
        get_queue_entry(&db, "docs", b"x").await.unwrap().is_none(),
        "must not re-create the entry"
    );
}

/// An admin retry resets the count but keeps what the entry asks for: an
/// exhausted tombstone stays a tombstone.
#[tokio::test]
async fn reset_keeps_kind_and_text() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    clear_vectors(&db, "docs", b"x").await.unwrap();
    let tombstone = snapshot(&db, b"x").await;
    record_queue_failure(&db, &tombstone, "io").await.unwrap();

    let before = reset_queue_entry(&db, "docs", b"x").await.unwrap().expect("entry existed");
    assert_eq!(before.retry_count, 1);
    let after = snapshot(&db, b"x").await;
    assert_eq!((after.kind, after.retry_count, after.last_error), (QueueEntryKind::Clear, 0, None));
    assert!(reset_queue_entry(&db, "docs", b"missing").await.unwrap().is_none());
    assert!(
        get_queue_entry(&db, "docs", b"missing").await.unwrap().is_none(),
        "reset must not create entries"
    );
}

// ── Crash + WAL replay ────────────────────────────────────────────────────
//
// Queue writes (enqueues, tombstones, and every conditional update, which the
// WAL records as its resulting Upsert/Delete) are WAL-backed; vector writes
// are no-WAL. A crash is `db.crash()` (no shutdown, no Drop-time flush, but
// every file closed), so reopening replays the WAL.

/// The worst case for R3: a delete races an embed, the worker's no-WAL vector
/// writes reach disk, and the process dies before the worker can clean up.
/// Only the **vector** namespaces are flushed — as a size-triggered flush of
/// the busy vector memtables would, while the small queue memtable is not —
/// so the tombstone survives only because it is WAL-backed. Replay restores
/// it, and processing it (as the worker's startup drain does) removes the
/// orphaned vectors.
#[tokio::test]
async fn crash_after_racing_delete_keeps_tombstone_that_removes_orphans() {
    let dir = TempDir::new().unwrap();
    {
        let db = open_db(&dir).await;
        enqueue_embed(&db, "docs", b"x", "text").await.unwrap();
        let _in_flight = snapshot(&db, b"x").await;
        clear_vectors(&db, "docs", b"x").await.unwrap(); // racing delete
        upsert_vectors(&db, "docs", b"x", "text", &vectors_for(1.0)).await.unwrap(); // worker's write lands after it
        for ns in [sparse_vectors_ns("docs"), sparse_vectors_meta_ns("docs"), dense_vectors_ns("docs")] {
            db.coordinator_for_test()
                .get_store_by_name(&ns)
                .unwrap()
                .flush_memtable_to_level0()
                .unwrap();
        }
        db.crash().await; // crash before the worker completes
    }
    let db = open_db(&dir).await;
    let pending = get_queue_entry(&db, "docs", b"x").await.unwrap().expect("tombstone survives replay");
    assert_eq!(pending.kind, QueueEntryKind::Clear);
    assert!(
        has_any_vector_state(&db, "docs", b"x").await.unwrap(),
        "orphan is on disk until the tombstone runs"
    );

    process_clear(&db, "docs", b"x").await.unwrap();
    assert!(!has_any_vector_state(&db, "docs", b"x").await.unwrap());
}

/// R1 across a crash: the newer upsert's entry, left in place by the
/// conditional completion, is still queued after replay.
#[tokio::test]
async fn crash_after_superseded_completion_keeps_newer_entry() {
    let dir = TempDir::new().unwrap();
    {
        let db = open_db(&dir).await;
        enqueue_embed(&db, "docs", b"x", "old text").await.unwrap();
        let in_flight = snapshot(&db, b"x").await;
        enqueue_embed(&db, "docs", b"x", "new text").await.unwrap();
        finish_embed(&db, &in_flight, &vectors_for(1.0)).await.unwrap();
        db.crash().await;
    }
    let db = open_db(&dir).await;
    let pending = get_queue_entry(&db, "docs", b"x").await.unwrap().expect("newer entry survives replay");
    assert_eq!(pending.text, "new text");
}

/// A completion skips the WAL: after a crash it is kept only if the queue
/// was flushed first, and otherwise the entry comes back exactly as it was
/// queued, to be embedded again. A failure's bookkeeping is WAL-backed and
/// replayed as written either way — replay applies the merge's recorded
/// result and never re-runs the conditional closure.
#[tokio::test]
async fn crash_replays_completion_and_failure_results() {
    for flushed in [false, true] {
        let dir = TempDir::new().unwrap();
        {
            let db = open_db(&dir).await;
            enqueue_embed(&db, "docs", b"done", "a").await.unwrap();
            let e = snapshot(&db, b"done").await;
            finish_embed(&db, &e, &vectors_for(1.0)).await.unwrap();
            enqueue_embed(&db, "docs", b"failed", "b").await.unwrap();
            let f = snapshot(&db, b"failed").await;
            record_queue_failure(&db, &f, "timeout").await.unwrap();
            if flushed {
                db.flush_namespaces(vec![PENDING_VEC_INDEX_NS.to_string()]).await.unwrap();
            }
            db.crash().await;
        }
        let db = open_db(&dir).await;
        let done = get_queue_entry(&db, "docs", b"done").await.unwrap();
        if flushed {
            assert!(done.is_none(), "a flushed completion survives replay");
        } else {
            let done = done.expect("an unflushed completion is lost, so the entry comes back");
            assert_eq!((done.kind, done.text.as_str(), done.retry_count), (QueueEntryKind::Embed, "a", 0));
        }
        let failed = get_queue_entry(&db, "docs", b"failed").await.unwrap().expect("failed entry kept");
        assert_eq!(
            (failed.text.as_str(), failed.retry_count, failed.last_error.as_deref()),
            ("b", 1, Some("timeout")),
            "flushed: {flushed}"
        );
    }
}

/// A completion the crash loses must not hide a newer upsert: the upsert's
/// entry is WAL-backed and later by seq, so it is what replay leaves queued,
/// flushed completion or not.
#[tokio::test]
async fn a_lost_completion_never_hides_a_newer_upsert() {
    for flushed in [false, true] {
        let dir = TempDir::new().unwrap();
        {
            let db = open_db(&dir).await;
            put_doc(&db, b"x").await;
            enqueue_embed(&db, "docs", b"x", "old text").await.unwrap();
            let e = snapshot(&db, b"x").await;
            finish_embed(&db, &e, &vectors_for(1.0)).await.unwrap();
            if flushed {
                db.flush_namespaces(vec![PENDING_VEC_INDEX_NS.to_string()]).await.unwrap();
            }
            enqueue_embed(&db, "docs", b"x", "new text").await.unwrap();
            db.crash().await;
        }
        let db = open_db(&dir).await;
        let entry = get_queue_entry(&db, "docs", b"x").await.unwrap().expect("the newer upsert stays queued");
        assert_eq!(
            (entry.kind, entry.text.as_str()),
            (QueueEntryKind::Embed, "new text"),
            "flushed: {flushed}"
        );
    }
}

/// No race: the ordinary path still indexes and empties the queue.
#[tokio::test]
async fn uncontended_embed_indexes_and_completes() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    db.namespace("docs".to_string())
        .await
        .unwrap()
        .put(b"x".to_vec(), b"{}".to_vec())
        .await
        .unwrap();
    enqueue_embed(&db, "docs", b"x", "text").await.unwrap();
    let in_flight = snapshot(&db, b"x").await;
    finish_embed(&db, &in_flight, &vectors_for(2.0)).await.unwrap();
    assert_eq!(dense_tag(&db, b"x").await, Some(2.0));
    assert!(get_queue_entry(&db, "docs", b"x").await.unwrap().is_none());
}

/// An entry for a document that no longer exists (a bulk re-index enqueued it
/// from an older snapshot, or an upsert's document write never landed) is
/// rewritten to a `Clear`, before anything is embedded.
#[tokio::test]
async fn an_entry_for_a_missing_document_becomes_a_clear() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await; // "docs" exists, but no document "gone" in it
    enqueue_embed(&db, "docs", b"gone", "stale text").await.unwrap();
    let entry = snapshot(&db, b"gone").await;
    assert!(!check_entry_against_document(&db, &entry, None).await.unwrap());
    let rewritten = snapshot(&db, b"gone").await;
    assert_eq!(rewritten.kind, QueueEntryKind::Clear);
    assert!(check_entry_against_document(&db, &rewritten, None).await.unwrap());
    process_clear(&db, "docs", b"gone").await.unwrap();
    assert!(!has_any_vector_state(&db, "docs", b"gone").await.unwrap(), "no vector state may remain");
}

/// An entry that describes the document is left alone; one that does not is
/// rewritten to the document's state, with a fresh retry count.
#[tokio::test]
async fn an_entry_is_rewritten_to_what_the_document_holds() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;

    enqueue_embed(&db, "docs", b"a", "text").await.unwrap();
    let entry = snapshot(&db, b"a").await;
    assert!(check_entry_against_document(&db, &entry, Some("text")).await.unwrap());
    assert_eq!(snapshot(&db, b"a").await.text, "text");

    // An upsert whose document write never landed: the entry holds new text.
    enqueue_embed(&db, "docs", b"b", "never written").await.unwrap();
    let entry = snapshot(&db, b"b").await;
    record_queue_failure(&db, &entry, "boom").await.unwrap();
    let entry = snapshot(&db, b"b").await;
    assert!(!check_entry_against_document(&db, &entry, Some("on disk")).await.unwrap());
    let rewritten = snapshot(&db, b"b").await;
    assert_eq!(
        (rewritten.kind, rewritten.text.as_str(), rewritten.retry_count),
        (QueueEntryKind::Embed, "on disk", 0)
    );

    // A delete whose document delete never ran: a `Clear` for a live document.
    clear_vectors(&db, "docs", b"c").await.unwrap();
    let entry = snapshot(&db, b"c").await;
    assert!(!check_entry_against_document(&db, &entry, Some("still here")).await.unwrap());
    let rewritten = snapshot(&db, b"c").await;
    assert_eq!((rewritten.kind, rewritten.text.as_str()), (QueueEntryKind::Embed, "still here"));
}

/// The rewrite is conditional: an entry replaced since it was read is newer
/// than the check, and is left alone.
#[tokio::test]
async fn a_newer_entry_is_never_rewritten() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    enqueue_embed(&db, "docs", b"a", "old").await.unwrap();
    let stale = snapshot(&db, b"a").await;
    enqueue_embed(&db, "docs", b"a", "newer").await.unwrap();
    assert!(!check_entry_against_document(&db, &stale, None).await.unwrap());
    let kept = snapshot(&db, b"a").await;
    assert_eq!((kept.kind, kept.text.as_str()), (QueueEntryKind::Embed, "newer"));
}

// ── Completion waits for durability (`Dynamic-RaBitQ-Index-Design.md` → M0-1) ──

/// Like [`vectors_for`], but the sparse chunk lands in cluster 4 instead of 3,
/// as a re-embedded document's chunks can.
fn moved_vectors_for(tag: f32) -> Vec<VectorIndex> {
    vec![
        VectorIndex::new(1, QuantisationStyle::MultiBit { number_of_bits: 8 }, tag, 0.0, 0.01, vec![]),
        VectorIndex::new(4, QuantisationStyle::SingleBit, tag, 0.0, 0.01, vec![]),
    ]
}

async fn put_doc(db: &AsyncDb, doc: &[u8]) {
    db.namespace("docs".to_string())
        .await
        .unwrap()
        .put(doc.to_vec(), b"{}".to_vec())
        .await
        .unwrap();
}

/// The doc's sparse cluster keys actually present, and the clusters its meta lists.
async fn sparse_state(db: &AsyncDb, doc: &[u8]) -> (Vec<u32>, Option<Vec<u32>>) {
    let sparse = db.namespace(sparse_vectors_ns("docs")).await.unwrap();
    let mut present = Vec::new();
    for cluster in [3u32, 4] {
        if sparse.get(composite_key::encode(cluster, doc)).await.unwrap().is_some() {
            present.push(cluster);
        }
    }
    let meta = db
        .namespace(sparse_vectors_meta_ns("docs"))
        .await
        .unwrap()
        .get(doc.to_vec())
        .await
        .unwrap();
    (present, meta.as_deref().and_then(decode_sparse_meta).map(|m| m.cluster_ids()))
}

/// A re-embed must be durable before its queue entry is completed. Completed
/// while the new vectors were still unsynced, a crash could keep the
/// completion and the stale-key deletes but lose the new vectors: the document
/// would keep its old dense entry and lose its moved chunk, with nothing queued
/// to repair it.
///
/// The completion now skips the WAL, so it survives the crash only if the
/// queue was flushed; both cases are checked. The new vectors must be on disk
/// in both, and if the entry came back it must ask for the new text.
#[tokio::test]
async fn crash_before_reembed_is_synced_keeps_new_vectors() {
    for flushed in [false, true] {
        let dir = TempDir::new().unwrap();
        {
            let db = open_db(&dir).await;
            put_doc(&db, b"x").await;
            enqueue_embed(&db, "docs", b"x", "old text").await.unwrap();
            let first = snapshot(&db, b"x").await;
            finish_embed(&db, &first, &vectors_for(1.0)).await.unwrap();
            make_vector_writes_durable(&db).await.unwrap(); // old index on disk
            enqueue_embed(&db, "docs", b"x", "new text").await.unwrap();
            let second = snapshot(&db, b"x").await;
            finish_embed(&db, &second, &moved_vectors_for(2.0)).await.unwrap();
            if flushed {
                // The queue only: the completion reaches disk, the vectors
                // stay in the vector index WAL.
                db.flush_namespaces(vec![PENDING_VEC_INDEX_NS.to_string()]).await.unwrap();
            }
            db.crash().await; // crash: no vector namespace flushed after finish_embed
        }
        let db = open_db(&dir).await;
        match get_queue_entry(&db, "docs", b"x").await.unwrap() {
            None => assert!(flushed, "an unflushed completion cannot survive the crash"),
            Some(entry) => {
                assert!(!flushed, "the flushed completion was lost");
                assert_eq!(entry.text, "new text", "the entry that came back must ask for the new text");
            }
        }
        assert_eq!(
            dense_tag(&db, b"x").await,
            Some(2.0),
            "the completed re-embed's dense vector was lost (flushed: {flushed})"
        );
        assert_eq!(
            sparse_state(&db, b"x").await,
            (vec![4], Some(vec![4])),
            "the moved chunk must be present and listed (flushed: {flushed})"
        );
    }
}

/// The worker writes a batch, flushes, then completes. A crash after the flush
/// but before the completions leaves the entry queued; redoing it writes the
/// same vectors and completes it.
#[tokio::test]
async fn crash_after_flush_before_completion_redoes_the_entry() {
    let dir = TempDir::new().unwrap();
    {
        let db = open_db(&dir).await;
        put_doc(&db, b"x").await;
        enqueue_embed(&db, "docs", b"x", "text").await.unwrap();
        upsert_vectors(&db, "docs", b"x", "text", &vectors_for(1.0)).await.unwrap(); // the worker's write
        make_vector_writes_durable(&db).await.unwrap(); // the batch flush
        db.crash().await; // crash before the completions
    }
    let db = open_db(&dir).await;
    let pending = get_queue_entry(&db, "docs", b"x").await.unwrap().expect("entry still queued");
    assert_eq!(dense_tag(&db, b"x").await, Some(1.0), "flushed vectors survive");
    finish_embed(&db, &pending, &vectors_for(1.0)).await.unwrap();
    assert!(get_queue_entry(&db, "docs", b"x").await.unwrap().is_none());
    assert_eq!(dense_tag(&db, b"x").await, Some(1.0));
    assert_eq!(sparse_state(&db, b"x").await, (vec![3], Some(vec![3])));
}

/// If the vectors cannot be made durable, the entry is not completed.
#[tokio::test]
async fn a_failed_sync_completes_nothing() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    put_doc(&db, b"x").await;
    enqueue_embed(&db, "docs", b"x", "text").await.unwrap();
    let entry = snapshot(&db, b"x").await;
    db.coordinator_for_test().vector_index_wal.fail_next_sync();
    let result = finish_embed(&db, &entry, &vectors_for(1.0)).await;
    assert!(result.is_err(), "the sync should have failed");
    assert!(
        get_queue_entry(&db, "docs", b"x").await.unwrap().is_some(),
        "an entry whose vectors are not durable was completed"
    );
}
