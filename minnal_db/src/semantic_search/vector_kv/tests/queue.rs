//! Queue tests.

use super::super::*;
use crate::AsyncDb;
use tempfile::TempDir;

async fn open_db(dir: &TempDir) -> AsyncDb {
    let path = dir.path().to_owned();
    AsyncDb::open_with_config(path, crate::support::test_db_config()).await.unwrap()
}

#[tokio::test]
async fn test_enqueue_and_scan() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;

    enqueue_embed(&db, "users", b"doc1", "hello world").await.unwrap();

    let entries = list_queue_entries(&db).await.unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].namespace, "users");
    assert_eq!(entries[0].doc_id_bytes, b"doc1");
    assert_eq!(entries[0].text, "hello world");
    assert_eq!(entries[0].retry_count, 0);
}

#[tokio::test]
async fn test_enqueue_deduplicates_same_doc() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;

    enqueue_embed(&db, "ns", b"key", "old text").await.unwrap();

    enqueue_embed(&db, "ns", b"key", "new text").await.unwrap();

    let entries = list_queue_entries(&db).await.unwrap();
    assert_eq!(entries.len(), 1, "second write must overwrite first");
    assert_eq!(entries[0].text, "new text");
}

#[tokio::test]
async fn test_different_docs_in_same_namespace() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;

    enqueue_embed(&db, "ns", b"doc1", "text one").await.unwrap();
    enqueue_embed(&db, "ns", b"doc2", "text two").await.unwrap();

    let entries = list_queue_entries(&db).await.unwrap();
    assert_eq!(entries.len(), 2);
}

#[tokio::test]
async fn test_different_namespaces_do_not_collide() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;

    enqueue_embed(&db, "ns_a", b"docX", "text a").await.unwrap();
    enqueue_embed(&db, "ns_b", b"docX", "text b").await.unwrap();

    let entries = list_queue_entries(&db).await.unwrap();
    assert_eq!(entries.len(), 2, "same doc_id in different namespaces are separate entries");
    let namespaces: Vec<&str> = entries.iter().map(|e| e.namespace.as_str()).collect();
    assert!(namespaces.contains(&"ns_a"));
    assert!(namespaces.contains(&"ns_b"));
}

#[tokio::test]
async fn test_cancel_removes_entry() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;

    enqueue_embed(&db, "ns", b"doc1", "some text").await.unwrap();

    remove_queue_entry(&db, "ns", b"doc1").await.unwrap();

    let entries = list_queue_entries(&db).await.unwrap();
    assert!(entries.is_empty(), "cancel must remove the entry");
}

#[tokio::test]
async fn test_cancel_on_missing_entry_is_noop() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;

    remove_queue_entry(&db, "ns", b"ghost").await.unwrap();
    // Should not panic or error.

    let entries = list_queue_entries(&db).await.unwrap();
    assert!(entries.is_empty());
}

#[tokio::test]
async fn test_scan_empty_queue() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;
    let entries = list_queue_entries(&db).await.unwrap();
    assert!(entries.is_empty());
}

#[tokio::test]
async fn test_queue_key_roundtrip() {
    let namespace = "my_store";
    let doc_id = b"\x00\x01\x02\x03\x04\x05\x06\x07";
    let key = queue_key(namespace, doc_id);
    let (ns_out, id_out) = decode_queue_key(&key).unwrap();
    assert_eq!(ns_out, namespace);
    assert_eq!(id_out, doc_id);
}

// ── encode / decode roundtrip ─────────────────────────────────────────────

const E: QueueEntryKind = QueueEntryKind::Embed;

#[test]
fn test_value_encoding_roundtrip_zero_retries() {
    let v = decode_queue_value(&encode_queue_value(E, "hello", 0, None)).unwrap();
    assert_eq!((v.kind, v.text.as_str(), v.retry_count, v.last_error), (E, "hello", 0, None));
}

#[test]
fn test_value_encoding_roundtrip_nonzero_retries() {
    let v = decode_queue_value(&encode_queue_value(E, "embed this", 3, None)).unwrap();
    assert_eq!((v.text.as_str(), v.retry_count), ("embed this", 3));
}

#[test]
fn test_value_encoding_with_last_error() {
    let v = decode_queue_value(&encode_queue_value(E, "some text", 2, Some("connection refused"))).unwrap();
    assert_eq!((v.text.as_str(), v.retry_count), ("some text", 2));
    assert_eq!(v.last_error.as_deref(), Some("connection refused"));
}

#[test]
fn test_value_encoding_empty_text() {
    let v = decode_queue_value(&encode_queue_value(E, "", 1, None)).unwrap();
    assert_eq!((v.text.as_str(), v.retry_count), ("", 1));
}

/// A tombstone keeps its kind through encode/decode, and its bookkeeping too.
#[test]
fn test_clear_entry_roundtrip() {
    let v = decode_queue_value(&encode_queue_value(QueueEntryKind::Clear, "", 2, Some("io"))).unwrap();
    assert_eq!(v.kind, QueueEntryKind::Clear);
    assert_eq!((v.text.as_str(), v.retry_count, v.last_error.as_deref()), ("", 2, Some("io")));
}

#[test]
fn test_value_decoding_rejects_unknown_version() {
    for version in [0x01u8, 0x02, 0x04] {
        let mut bad = encode_queue_value(QueueEntryKind::Embed, "hello", 0, None);
        bad[0] = version;
        assert!(decode_queue_value(&bad).is_none(), "version {version}");
    }
}

#[test]
fn test_value_decoding_rejects_unknown_kind() {
    let bad = b"\x03\x07\x00\x00\x00\x00\x00\x00\x00\x00hello";
    assert!(decode_queue_value(bad).is_none());
}

#[test]
fn test_value_decoding_rejects_truncated() {
    // Only the version byte — retry_count (4 bytes) is missing.
    let bad = b"\x01";
    assert!(decode_queue_value(bad).is_none());
}

#[test]
fn test_value_decoding_rejects_empty_slice() {
    assert!(decode_queue_value(&[]).is_none());
}

// ── update_queue_retry ────────────────────────────────────────────────────

#[tokio::test]
async fn test_update_retry_increments_count() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;

    enqueue_embed(&db, "ns", b"doc1", "text").await.unwrap();

    update_queue_retry(&db, "ns", b"doc1", "text", 1, None).await.unwrap();

    let entries = list_queue_entries(&db).await.unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].retry_count, 1);
}

#[tokio::test]
async fn test_update_retry_preserves_text() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;

    enqueue_embed(&db, "ns", b"doc1", "original text").await.unwrap();

    update_queue_retry(&db, "ns", b"doc1", "original text", 2, None).await.unwrap();

    let entries = list_queue_entries(&db).await.unwrap();
    assert_eq!(entries[0].text, "original text");
    assert_eq!(entries[0].retry_count, 2);
}

#[tokio::test]
async fn test_update_retry_multiple_times() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;

    enqueue_embed(&db, "ns", b"doc1", "text").await.unwrap();

    for expected in 1u32..=4 {
        update_queue_retry(&db, "ns", b"doc1", "text", expected, None).await.unwrap();

        let entries = list_queue_entries(&db).await.unwrap();
        assert_eq!(entries[0].retry_count, expected, "retry_count should be {expected}");
    }
}

// ── re-enqueue resets retry_count ─────────────────────────────────────────

#[tokio::test]
async fn test_reenqueue_after_retries_resets_count_to_zero() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir).await;

    // First enqueue and simulate 3 retries.
    enqueue_embed(&db, "ns", b"doc1", "v1").await.unwrap();

    update_queue_retry(&db, "ns", b"doc1", "v1", 3, None).await.unwrap();

    // A fresh write to the same doc overwrites the queue entry with retry_count=0.
    enqueue_embed(&db, "ns", b"doc1", "v2 updated").await.unwrap();

    let entries = list_queue_entries(&db).await.unwrap();
    assert_eq!(entries.len(), 1, "still one entry for the same doc");
    assert_eq!(entries[0].text, "v2 updated");
    assert_eq!(entries[0].retry_count, 0, "re-enqueue must reset retry_count");
}

// ── restart regression ────────────────────────────────────────────────────

// Regression test for the vec-index queue infinite-loop bug after restart.
//
// Before the fix, WAL recovery called flush_and_compact_all(), pushing queue
// entries from the memtable into SSTables.  remove_queue_entry() then wrote
// a tombstone into the *active* memtable.  LsmTree::keys() (used by iter())
// collected SSTable keys last — without tombstone suppression — so the deleted
// entries re-appeared in every subsequent list_queue_entries() call, causing
// the worker to loop forever.
//
// This test reproduces that exact sequence: enqueue → graceful shutdown →
// reopen (triggers WAL recovery + SSTable flush) → cancel → assert queue empty.
#[tokio::test]
async fn test_queue_empty_after_cancel_following_restart() {
    let dir = TempDir::new().unwrap();

    // Phase 1: enqueue entries, then shut down cleanly.
    {
        let db = open_db(&dir).await;
        enqueue_embed(&db, "ns", b"doc1", "hello").await.unwrap();
        enqueue_embed(&db, "ns", b"doc2", "world").await.unwrap();
        db.shutdown().await.unwrap();
    }

    // Phase 2: reopen — WAL recovery replays the enqueues and
    // flush_and_compact_all() pushes them from the memtable into SSTables.
    let db = open_db(&dir).await;

    let entries = list_queue_entries(&db).await.unwrap();
    assert_eq!(entries.len(), 2, "both entries must survive the restart");

    // Phase 3: cancel each entry — tombstones land in the active memtable
    // while the originals are still in the SSTable.
    for entry in &entries {
        remove_queue_entry(&db, &entry.namespace, &entry.doc_id_bytes).await.unwrap();
    }

    // Phase 4: the queue must now be empty.  Before the fix, list_queue_entries
    // returned the SSTable entries ignoring the memtable tombstones.
    let remaining = list_queue_entries(&db).await.unwrap();
    assert!(
        remaining.is_empty(),
        "queue must be empty after cancellation; {} entries remain",
        remaining.len()
    );
}
