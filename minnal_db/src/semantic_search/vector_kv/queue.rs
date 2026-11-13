//! Async vector-index queue: the embed markers and their conditional updates.

use super::{delete_vector, upsert_vectors};
use crate::AsyncDb;
use crate::semantic_search::{PostingDelta, VectorIndex};

/// System-wide namespace used as the async vector-index embedding queue.
///
/// Each entry represents one pending embedding job:
///
/// | Part    | Encoding                                          |
/// |---------|---------------------------------------------------|
/// | Key     | `u16_be(namespace_len) ‖ namespace_bytes ‖ doc_id_bytes` |
/// | Value   | `0x01 ‖ retry_count (4 B BE) ‖ text_bytes`       |
///
/// The composite key means writes for the same `(namespace, doc_id)` pair
/// overwrite each other — rapid successive writes produce exactly one
/// embedding call for the most-recent text (natural deduplication).
pub const PENDING_VEC_INDEX_NS: &str = "system_pending_vec_index";

/// What a queue entry asks the worker to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueEntryKind {
    /// Embed `text` and write the document's vectors.
    Embed,
    /// Tombstone: the document must end up with **no** vectors — it was deleted,
    /// or its embedding text became empty. Deletes write this instead of removing
    /// the entry, so a worker that was already embedding the document's older text
    /// finds it when it completes and removes what it wrote (see [`finish_embed`]).
    Clear,
}

/// One entry in the async vector-index embedding queue.
#[derive(Clone, Debug)]
pub struct QueueEntry {
    /// The document store namespace this entry belongs to.
    pub namespace: String,
    /// Raw document key bytes (big-endian encoded ID).
    pub doc_id_bytes: Vec<u8>,
    /// What the worker must do for this document.
    pub kind: QueueEntryKind,
    /// Text to embed (empty for [`QueueEntryKind::Clear`]).
    pub text: String,
    /// Number of failed embedding attempts so far.
    pub retry_count: u32,
    /// Error message from the most recent failed embedding attempt.
    pub last_error: Option<String>,
}

/// The queue value's format version, its first byte. The value is
/// `[version] [kind: 1B] [retry_count: 4B BE] [error_len: 4B BE] [error bytes] [text bytes]`,
/// where `kind` is `0` (embed) or `1` (clear).
const QUEUE_VALUE_VERSION: u8 = 0x03;

const KIND_EMBED: u8 = 0;
const KIND_CLEAR: u8 = 1;

pub(super) fn encode_queue_value(kind: QueueEntryKind, text: &str, retry_count: u32, last_error: Option<&str>) -> Vec<u8> {
    let error_bytes = last_error.unwrap_or("").as_bytes();
    let mut value = Vec::with_capacity(1 + 1 + 4 + 4 + error_bytes.len() + text.len());
    value.push(QUEUE_VALUE_VERSION);
    value.push(match kind {
        QueueEntryKind::Embed => KIND_EMBED,
        QueueEntryKind::Clear => KIND_CLEAR,
    });
    value.extend_from_slice(&retry_count.to_be_bytes());
    value.extend_from_slice(&(error_bytes.len() as u32).to_be_bytes());
    value.extend_from_slice(error_bytes);
    value.extend_from_slice(text.as_bytes());
    value
}

/// A decoded queue value.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct QueueValue {
    pub(super) kind: QueueEntryKind,
    pub(super) text: String,
    pub(super) retry_count: u32,
    pub(super) last_error: Option<String>,
}

impl QueueValue {
    /// Whether this value is still the entry `entry` was taken from: same kind and
    /// same text. Retry bookkeeping is ignored — only the worker and admin retries
    /// change it, and neither changes what the entry asks for.
    fn is(&self, entry: &QueueEntry) -> bool {
        self.kind == entry.kind && self.text == entry.text
    }

    fn encode(&self) -> Vec<u8> {
        encode_queue_value(self.kind, &self.text, self.retry_count, self.last_error.as_deref())
    }

    fn into_entry(self, namespace: &str, doc_id_bytes: &[u8]) -> QueueEntry {
        QueueEntry {
            namespace: namespace.to_owned(),
            doc_id_bytes: doc_id_bytes.to_vec(),
            kind: self.kind,
            text: self.text,
            retry_count: self.retry_count,
            last_error: self.last_error,
        }
    }
}

/// Decode a queue value; `None` for an unknown version or a damaged value.
pub(super) fn decode_queue_value(value: &[u8]) -> Option<QueueValue> {
    if *value.first()? != QUEUE_VALUE_VERSION {
        return None;
    }
    let kind = match *value.get(1)? {
        KIND_EMBED => QueueEntryKind::Embed,
        KIND_CLEAR => QueueEntryKind::Clear,
        _ => return None,
    };
    let tail = &value[2..];
    if tail.len() < 8 {
        return None;
    }
    let retry_count = u32::from_be_bytes(tail[0..4].try_into().ok()?);
    let error_len = u32::from_be_bytes(tail[4..8].try_into().ok()?) as usize;
    if tail.len() < 8 + error_len {
        return None;
    }
    let last_error = if error_len > 0 {
        Some(std::str::from_utf8(&tail[8..8 + error_len]).ok()?.to_owned())
    } else {
        None
    };
    let text = std::str::from_utf8(&tail[8 + error_len..]).ok()?.to_owned();
    Some(QueueValue {
        kind,
        text,
        retry_count,
        last_error,
    })
}

pub(super) fn queue_key(namespace: &str, doc_id_bytes: &[u8]) -> Vec<u8> {
    let ns = namespace.as_bytes();
    let ns_len = ns.len() as u16;
    let mut key = Vec::with_capacity(2 + ns.len() + doc_id_bytes.len());
    key.extend_from_slice(&ns_len.to_be_bytes());
    key.extend_from_slice(ns);
    key.extend_from_slice(doc_id_bytes);
    key
}

pub(super) fn decode_queue_key(key: &[u8]) -> Option<(&str, &[u8])> {
    if key.len() < 2 {
        return None;
    }
    let ns_len = u16::from_be_bytes([key[0], key[1]]) as usize;
    if key.len() < 2 + ns_len {
        return None;
    }
    let namespace = std::str::from_utf8(&key[2..2 + ns_len]).ok()?;
    let doc_id = &key[2 + ns_len..];
    Some((namespace, doc_id))
}

/// Write a pending-embed queue entry (durable single-op write).
///
/// The queue namespace is created on first use.  Writing the same
/// `(namespace, doc_id_bytes)` key twice overwrites the earlier entry so
/// that only the latest `text` is embedded (deduplication).  The entry is
/// written with `retry_count = 0`.
///
/// Every enqueue is WAL-backed, including the bulk re-index paths: the queue is
/// the durable record of what still needs (re-)indexing and must survive a crash
/// on every path. A document write calls this **before** writing the document
/// (see *Queue first* in the module docs).
pub async fn enqueue_embed(db: &AsyncDb, namespace: &str, doc_id_bytes: &[u8], text: &str) -> Result<(), crate::KVError> {
    let queue_ns = db.namespace(PENDING_VEC_INDEX_NS.to_string()).await?;
    crash_point!();
    queue_ns
        .put(
            queue_key(namespace, doc_id_bytes),
            encode_queue_value(QueueEntryKind::Embed, text, 0, None),
        )
        .await?;
    Ok(())
}

/// Update an existing queue entry's retry count and last error (durable single-op write).
///
/// The entry's text is preserved unchanged.
pub async fn update_queue_retry(
    db: &AsyncDb,
    namespace: &str,
    doc_id_bytes: &[u8],
    text: &str,
    retry_count: u32,
    last_error: Option<&str>,
) -> Result<(), crate::KVError> {
    let queue_ns = db.namespace(PENDING_VEC_INDEX_NS.to_string()).await?;
    queue_ns
        .put(
            queue_key(namespace, doc_id_bytes),
            encode_queue_value(QueueEntryKind::Embed, text, retry_count, last_error),
        )
        .await?;
    Ok(())
}

/// Remove a pending-embed queue entry (durable single-op write).
///
/// The single queue-delete primitive, shared by every caller that needs to
/// drop an entry: the worker removing a completed entry after a successful
/// embed, the document delete/upsert paths cancelling a pending embed, admin
/// queue clears, and re-index clearing exhausted entries.
///
/// Safe to call when no queue entry exists for `(namespace, doc_id_bytes)` —
/// the delete is a no-op in that case.
pub async fn remove_queue_entry(db: &AsyncDb, namespace: &str, doc_id_bytes: &[u8]) -> Result<(), crate::KVError> {
    let queue_ns = db.namespace(PENDING_VEC_INDEX_NS.to_string()).await?;
    queue_ns.delete(queue_key(namespace, doc_id_bytes)).await?;
    Ok(())
}

/// Remove a queue entry whose namespace was gone when the worker looked, unless
/// it has changed since `entry` was read: the store may have been recreated under
/// the same name, and the entry may now be for a new write to it.
pub async fn discard_orphaned_entry(db: &AsyncDb, entry: &QueueEntry) -> Result<(), crate::KVError> {
    complete_entry(db, entry).await.map(|_| ())
}

/// Remove every queue entry of `namespace` (a store being dropped), durable on
/// return; returns how many there were.
///
/// The keys start with the namespace, so one prefix scan finds them. They are
/// deleted without the WAL and made durable by one flush of the queue's
/// memtable: a WAL-backed delete costs an fsync each, which made dropping a
/// store with 20 000 queued entries take 48 s.
pub async fn clear_queue_for(db: &AsyncDb, namespace: &str) -> Result<usize, crate::KVError> {
    let queue_ns = db.namespace(PENDING_VEC_INDEX_NS.to_string()).await?;
    let entries = queue_ns.scan_prefix(queue_key(namespace, &[])).await?;
    if entries.is_empty() {
        return Ok(0);
    }
    for (key, _) in &entries {
        queue_ns.delete_no_wal(key.clone()).await?;
    }
    db.flush_namespaces(vec![PENDING_VEC_INDEX_NS.to_string()]).await?;
    Ok(entries.len())
}

/// Return all pending embedding queue entries.
///
/// Called by the `VecIndexWorker` on startup (crash recovery) and after
/// each wake-up signal.  Returns an empty list when the queue is empty.
pub async fn list_queue_entries(db: &AsyncDb) -> Result<Vec<QueueEntry>, crate::KVError> {
    let queue_ns = db.namespace(PENDING_VEC_INDEX_NS.to_string()).await?;
    let entries = queue_ns.iter().await?;
    let mut result = Vec::with_capacity(entries.len());
    for (key, value) in entries {
        if let Some((namespace, doc_id_bytes)) = decode_queue_key(&key)
            && let Some(value) = decode_queue_value(&value)
        {
            result.push(value.into_entry(namespace, doc_id_bytes));
        }
    }
    Ok(result)
}

/// How many entries the embedding queue holds, counted from the keys alone.
pub async fn count_queue_entries(db: &AsyncDb) -> Result<usize, crate::KVError> {
    let queue_ns = db.namespace(PENDING_VEC_INDEX_NS.to_string()).await?;
    Ok(queue_ns.keys().await?.len())
}

/// The namespaces that have entries in the embedding queue, read from the keys
/// alone. The queue key carries the namespace and the value carries the text, so
/// this never reads a value: after a crash mid bulk-load the queue can hold every
/// document's text, and the startup drop sweep needs only the namespaces.
pub async fn queued_namespaces(db: &AsyncDb) -> Result<std::collections::BTreeSet<String>, crate::KVError> {
    let queue_ns = db.namespace(PENDING_VEC_INDEX_NS.to_string()).await?;
    Ok(queue_ns
        .keys()
        .await?
        .iter()
        .filter_map(|key| decode_queue_key(key).map(|(namespace, _)| namespace.to_owned()))
        .collect())
}

/// The key prefix every queue entry of `namespace` starts with, for paging
/// through one namespace's entries with [`queue_page`].
pub fn queue_prefix(namespace: &str) -> Vec<u8> {
    queue_key(namespace, &[])
}

/// One page of at most `limit` queue entries with keys in `[start, end)`, and the
/// key the next page starts at (`None` after the last page). Unlike
/// [`list_queue_entries`], memory is bounded by the page.
pub async fn queue_page(
    db: &AsyncDb,
    start: Vec<u8>,
    end: Option<Vec<u8>>,
    limit: usize,
) -> Result<(Vec<QueueEntry>, Option<Vec<u8>>), crate::KVError> {
    let queue_ns = db.namespace(PENDING_VEC_INDEX_NS.to_string()).await?;
    let (pairs, next) = queue_ns.scan(Some(start), end, limit).await?;
    let entries = pairs
        .into_iter()
        .filter_map(|(key, value)| {
            let (namespace, doc_id_bytes) = decode_queue_key(&key)?;
            Some(decode_queue_value(&value)?.into_entry(namespace, doc_id_bytes))
        })
        .collect();
    Ok((entries, next))
}

/// Look up a single pending embedding queue entry by namespace and document ID.
///
/// Returns `None` when no entry exists for the given `(namespace, doc_id_bytes)` pair.
/// This is an O(1) key lookup and does not scan the whole queue.
pub async fn get_queue_entry(db: &AsyncDb, namespace: &str, doc_id_bytes: &[u8]) -> Result<Option<QueueEntry>, crate::KVError> {
    let queue_ns = db.namespace(PENDING_VEC_INDEX_NS.to_string()).await?;
    let key = queue_key(namespace, doc_id_bytes);
    match queue_ns.get(key).await? {
        None => Ok(None),
        Some(value) => Ok(decode_queue_value(&value).map(|v| v.into_entry(namespace, doc_id_bytes))),
    }
}

// ── Conditional queue updates ─────────────────────────────────────────────────
//
// The worker snapshots the whole queue at the start of a pass, then embeds the
// snapshot entries one by one, so an upsert or delete of a document can land at
// any point between the snapshot and that entry's completion — minutes, during a
// bulk load. Every write the worker makes *after* embedding is therefore
// conditional on the entry still being the one it took, checked atomically
// under the key's stripe lock via `merge`:
//
// * R1 — completing: only delete the entry if it still asks for this text;
//   a newer upsert's entry is left for the next pass.
// * R2 — failing: only bump the retry count if the entry is unchanged; never
//   rewrite a newer entry with the older text.
// * R3 — a delete that lands mid-embed leaves a `Clear` tombstone (it no longer
//   removes the entry), so the completing worker sees it and removes the vectors
//   it just wrote.

/// `MergeAborted` reasons used to report why a conditional update wrote nothing.
const ABORT_SUPERSEDED: &str = "queue entry superseded";
const ABORT_CLEARED: &str = "queue entry is a clear tombstone";

/// What completing a queue entry found.
enum Completion {
    /// The entry was still `entry` and has been removed (or was already gone).
    Done,
    /// A newer entry replaced it; that entry stays queued for the next pass.
    Superseded,
    /// The document was cleared meanwhile: a `Clear` tombstone replaced it.
    Cleared,
}

/// Remove `entry` from the queue only if the stored entry is still `entry`.
///
/// The removal skips the WAL (`merge_no_wal`), so it is durable only once the
/// queue's memtable is flushed, on the LSM worker's next tick. Losing it is
/// harmless: the entry comes back after the crash and the document is embedded
/// again with the same text. Making it durable cost a WAL fsync per document,
/// on the lock every client write waits for — a quarter of the worker's share
/// of the WAL under ingestion. The vectors are durable before this runs
/// ([`make_vector_writes_durable`]), so a removal that does reach disk never
/// outlives the vectors it vouches for. Design note:
/// `semantic_search/docs/Fewer-WAL-Fsyncs-Design.md`, Part 1.
async fn complete_entry(db: &AsyncDb, entry: &QueueEntry) -> Result<Completion, crate::KVError> {
    let queue_ns = db.namespace(PENDING_VEC_INDEX_NS.to_string()).await?;
    crash_point!();
    let expected = entry.clone();
    let result = queue_ns
        .merge_no_wal(queue_key(&entry.namespace, &entry.doc_id_bytes), Vec::new(), move |existing, _| {
            let Some(existing) = existing else {
                return Ok(None); // already gone (an admin removed it): nothing to do
            };
            match decode_queue_value(existing) {
                Some(v) if v.is(&expected) => Ok(None), // still ours: delete it
                Some(v) if v.kind == QueueEntryKind::Clear => Err(crate::KVError::MergeAborted(ABORT_CLEARED.into())),
                _ => Err(crate::KVError::MergeAborted(ABORT_SUPERSEDED.into())),
            }
        })
        .await;
    match result {
        Ok(_) => Ok(Completion::Done),
        Err(crate::KVError::MergeAborted(reason)) if reason == ABORT_CLEARED => Ok(Completion::Cleared),
        Err(crate::KVError::MergeAborted(_)) => Ok(Completion::Superseded),
        Err(e) => Err(e),
    }
}

/// The worker's step after embedding an [`QueueEntryKind::Embed`] entry, for one
/// entry: write the document's vectors, make them durable, then complete the
/// entry. The worker does the same over a batch, with one
/// [`make_vector_writes_durable`] for the whole batch.
pub async fn finish_embed(db: &AsyncDb, entry: &QueueEntry, vector_indexes: &[VectorIndex]) -> Result<PostingDelta, crate::KVError> {
    let mut delta = upsert_vectors(db, &entry.namespace, &entry.doc_id_bytes, &entry.text, vector_indexes).await?;
    crash_point!();
    make_vector_writes_durable(db).await?;
    delta.extend(complete_embed(db, entry).await?);
    Ok(delta)
}

/// Make every vector write already done durable: one fsync of the vector index
/// WAL, which holds the chunk keys and dense entries [`upsert_vectors`] wrote
/// (`put_logged`). A barrier: see below. It covers every namespace at once, so
/// a batch spanning several stores costs one fsync.
///
/// **Why completion waits for this.** [`upsert_vectors`] deletes stale cluster keys
/// and writes the meta WAL-backed (durable at once), and writes the new keys and
/// dense entry to the vector index WAL (durable only after this sync), and
/// [`complete_embed`] removes the queue entry (no-WAL: durable at the queue's
/// next flush, which can come at any moment). Completing before the
/// sync would let a crash keep the completion and the deletes but lose the new
/// vectors. A re-embedded document would then keep its *old* dense entry with
/// nothing queued to replace it: it would stay indexed under its old text, minus
/// the chunks whose clusters changed. With
/// completion after this barrier, a crash before it leaves the queue entry in
/// place and the restarted worker redoes it. Regression test:
/// `crash_before_reembed_is_synced_keeps_new_vectors`.
pub async fn make_vector_writes_durable(db: &AsyncDb) -> Result<(), crate::KVError> {
    db.sync_vector_index_wal().await
}

/// Complete an embed entry whose vectors are written **and durable** (see
/// [`make_vector_writes_durable`]) — conditionally (see the section comment). If
/// the document was cleared while it was being embedded, the vectors just written
/// are removed again.
pub async fn complete_embed(db: &AsyncDb, entry: &QueueEntry) -> Result<PostingDelta, crate::KVError> {
    match complete_entry(db, entry).await? {
        Completion::Done | Completion::Superseded => Ok(PostingDelta::default()),
        Completion::Cleared => process_clear(db, &entry.namespace, &entry.doc_id_bytes).await,
    }
}

/// Process a [`QueueEntryKind::Clear`] tombstone: delete the document's vectors,
/// then remove the tombstone unless a newer upsert has replaced it (in which case
/// the next pass embeds that).
pub async fn process_clear(db: &AsyncDb, namespace: &str, doc_id_bytes: &[u8]) -> Result<PostingDelta, crate::KVError> {
    let delta = delete_vector(db, namespace, doc_id_bytes).await?;
    let tombstone = QueueEntry {
        namespace: namespace.to_owned(),
        doc_id_bytes: doc_id_bytes.to_vec(),
        kind: QueueEntryKind::Clear,
        text: String::new(),
        retry_count: 0,
        last_error: None,
    };
    complete_entry(db, &tombstone).await.map(|_| delta)
}

/// Record a failed attempt at `entry`: bump its retry count and store `error` —
/// only if the stored entry is still `entry`. Returns the new retry count, or
/// `None` when the entry was replaced or removed meanwhile (nothing written).
pub async fn record_queue_failure(db: &AsyncDb, entry: &QueueEntry, error: &str) -> Result<Option<u32>, crate::KVError> {
    let queue_ns = db.namespace(PENDING_VEC_INDEX_NS.to_string()).await?;
    let expected = entry.clone();
    let error = error.to_owned();
    let result = queue_ns
        .merge(queue_key(&entry.namespace, &entry.doc_id_bytes), Vec::new(), move |existing, _| {
            match existing.and_then(decode_queue_value) {
                Some(v) if v.is(&expected) => Ok(Some(
                    QueueValue {
                        retry_count: v.retry_count + 1,
                        last_error: Some(error),
                        ..v
                    }
                    .encode(),
                )),
                // Replaced or removed: never write the older entry back.
                _ => Err(crate::KVError::MergeAborted(ABORT_SUPERSEDED.into())),
            }
        })
        .await;
    match result {
        Ok(written) => Ok(written.as_deref().and_then(decode_queue_value).map(|v| v.retry_count)),
        Err(crate::KVError::MergeAborted(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Compare a queue entry with what its document holds now, and rewrite the entry
/// when it does not describe the document. Returns whether the entry describes
/// the document; when it does not, the worker leaves it for the next pass, which
/// finds the rewritten entry.
///
/// `current` is the document's embedding text, read by the caller **after**
/// seeing that no write to the document is in progress: `None` when the document
/// is absent or has no embedding text. An `Embed` entry describes the document
/// when its text is `current`; a `Clear` entry when `current` is `None`.
///
/// With no write in progress, none is half done, so a mismatch is a write cut
/// short by a crash or an error after its queue entry: an upsert whose document never landed
/// (the entry holds text the document does not), or a delete whose document
/// delete never ran (a `Clear` for a document that is still there). The entry is
/// rewritten to the document's state — an `Embed` of `current`, or a `Clear` —
/// with a fresh retry count. The rewrite is conditional on the entry still being
/// `entry`, so a newer entry is never replaced. It also corrects an entry
/// enqueued from an older snapshot of the document (a bulk re-index).
pub async fn check_entry_against_document(db: &AsyncDb, entry: &QueueEntry, current: Option<&str>) -> Result<bool, crate::KVError> {
    let describes = match (entry.kind, current) {
        (QueueEntryKind::Embed, Some(text)) => text == entry.text,
        (QueueEntryKind::Clear, None) => true,
        _ => false,
    };
    if describes {
        return Ok(true);
    }
    let rewritten = match current {
        Some(text) => encode_queue_value(QueueEntryKind::Embed, text, 0, None),
        None => encode_queue_value(QueueEntryKind::Clear, "", 0, None),
    };
    let queue_ns = db.namespace(PENDING_VEC_INDEX_NS.to_string()).await?;
    let expected = entry.clone();
    crash_point!();
    let result = queue_ns
        .merge(queue_key(&entry.namespace, &entry.doc_id_bytes), Vec::new(), move |existing, _| {
            match existing.and_then(decode_queue_value) {
                Some(v) if v.is(&expected) => Ok(Some(rewritten)),
                // Replaced or removed meanwhile: leave the newer state alone.
                _ => Err(crate::KVError::MergeAborted(ABORT_SUPERSEDED.into())),
            }
        })
        .await;
    match result {
        Ok(_) | Err(crate::KVError::MergeAborted(_)) => Ok(false),
        Err(e) => Err(e),
    }
}

/// Write a `Clear` tombstone for a document, then delete its vectors.
///
/// Used by document deletes and by upserts whose embedding text is now empty. The
/// tombstone is written **first**, so a worker already embedding the document's
/// older text finds it when it completes and removes what it wrote (R3). The
/// vectors are also deleted here, synchronously, so they stop matching
/// immediately; the worker's later [`process_clear`] is an idempotent repeat that
/// also removes the tombstone. Both writes are WAL-backed (see [`delete_vector`]).
pub async fn clear_vectors(db: &AsyncDb, namespace: &str, doc_id_bytes: &[u8]) -> Result<PostingDelta, crate::KVError> {
    let queue_ns = db.namespace(PENDING_VEC_INDEX_NS.to_string()).await?;
    crash_point!();
    queue_ns
        .put(queue_key(namespace, doc_id_bytes), encode_queue_value(QueueEntryKind::Clear, "", 0, None))
        .await?;
    delete_vector(db, namespace, doc_id_bytes).await
}

/// Reset a queue entry's retry count to zero (and clear its error) so the worker
/// tries it again, keeping its kind and text. Returns the entry as it was before
/// the reset, or `None` if there is no entry.
///
/// Atomic, unlike reading the entry and re-enqueueing its text: an upsert landing
/// between those two steps would be overwritten with the older text, and a `Clear`
/// tombstone would turn back into an embed.
pub async fn reset_queue_entry(db: &AsyncDb, namespace: &str, doc_id_bytes: &[u8]) -> Result<Option<QueueEntry>, crate::KVError> {
    let queue_ns = db.namespace(PENDING_VEC_INDEX_NS.to_string()).await?;
    let before = std::sync::Arc::new(parking_lot::Mutex::new(None));
    let slot = std::sync::Arc::clone(&before);
    let result = queue_ns
        .merge(queue_key(namespace, doc_id_bytes), Vec::new(), move |existing, _| {
            let Some(v) = existing.and_then(decode_queue_value) else {
                return Err(crate::KVError::MergeAborted("no queue entry".into()));
            };
            let reset = QueueValue {
                retry_count: 0,
                last_error: None,
                ..v.clone()
            };
            *slot.lock() = Some(v);
            Ok(Some(reset.encode()))
        })
        .await;
    match result {
        Ok(_) => Ok(before.lock().take().map(|v| v.into_entry(namespace, doc_id_bytes))),
        Err(crate::KVError::MergeAborted(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Reset the retry count of every entry of `namespace` that has reached
/// `max_retries`, so the worker picks them up again. Returns how many were reset.
pub async fn reset_exhausted_queue_entries(db: &AsyncDb, namespace: &str, max_retries: u32) -> Result<usize, crate::KVError> {
    let entries = list_queue_entries(db).await?;
    let exhausted: Vec<_> = entries
        .into_iter()
        .filter(|e| e.namespace == namespace && e.retry_count >= max_retries)
        .collect();
    for entry in &exhausted {
        reset_queue_entry(db, &entry.namespace, &entry.doc_id_bytes).await?;
    }
    Ok(exhausted.len())
}
