//! Vector-index queue administration and bulk (re)indexing.

#[cfg(feature = "semantic-search")]
use crate::doc_store::*;
use crate::namespace_lifecycle::BuildStatus;

mod lifecycle;
#[cfg(feature = "semantic-search")]
pub(crate) mod worker_host;

mod partition;
mod queue;
mod reindex;

// ── What a document holds now ─────────────────────────────────────────────────

/// Read what `namespace`'s document `key` holds now: its embedding text, built
/// as a write builds it (the schema's embedding fields for a document store, the
/// value itself for a KV store). The caller has seen no write to it in progress
/// ([`InFlightWrites`]).
#[cfg(feature = "semantic-search")]
pub(crate) async fn document_text(db: &AsyncDb, schemas: &SchemaStore, namespace: &str, key: &[u8]) -> Result<DocumentText, DocStoreError> {
    let Some(schema) = schemas.get(namespace) else {
        return Ok(DocumentText::StoreGone);
    };
    if schemas.state(namespace) == Some(NamespaceState::Deleting) {
        return Ok(DocumentText::StoreGone);
    }
    // Never get-or-create: this runs outside any partition's epoch, so a drop can
    // land at any moment, and resolving the name after it would recreate the store.
    let Some(store_ns) = db.existing_namespace(namespace.to_owned()).await? else {
        return Ok(DocumentText::StoreGone);
    };
    let Some(value) = store_ns.get(key.to_vec()).await? else {
        return Ok(DocumentText::Nothing);
    };
    let text = match schema {
        StoredSchema::Doc(schema) => match serde_json::from_slice::<serde_json::Value>(&value) {
            Ok(doc) => build_embedding_text(&doc, &schema.embedding_fields),
            Err(_) => String::new(),
        },
        StoredSchema::Kv(_) => String::from_utf8(value).unwrap_or_default(),
    };
    Ok(if text.is_empty() {
        DocumentText::Nothing
    } else {
        DocumentText::Text(text)
    })
}

/// Persistent record for one `index_all` reindex, kept in the maintenance
/// namespace (`TaskKey::vector_reindex`).
///
/// A reindex is the unit of work created by a single `index_all` call: it
/// tracks the total documents enqueued, how many have been indexed, and the
/// lifecycle status.  The record survives restarts so that the progress API can
/// show historical reindexs even after the server restarts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VecReindexProgress {
    /// `Running` until the worker has embedded everything enqueued, then
    /// `Complete`; `Failed` if the enqueue failed or a restart cut it short.
    pub status: BuildStatus,
    /// Unix milliseconds when `index_all` was called.
    pub started_at_ms: u64,
    /// Unix milliseconds when the reindex finished (terminal states only).
    pub completed_at_ms: Option<u64>,
    /// Number of documents enqueued by this reindex.
    pub total_enqueued: usize,
    /// Number of exhausted entries cleared before enqueueing.
    pub exhausted_cleared: usize,
    /// Error message when `status == Failed`.
    pub error: Option<String>,
    /// Whether every document was enqueued. A `Running` record without it, and
    /// with no reindex enqueueing in this process, was cut short by a crash.
    #[serde(default)]
    pub enqueue_done: bool,
}

/// Outcome of a single-document vector reindex
/// ([`DocStore::reindex_doc_vector`](crate::doc_store::DocStore::reindex_doc_vector) /
/// [`DocStore::kv_reindex_doc_vector`](crate::doc_store::DocStore::kv_reindex_doc_vector)).
#[cfg(feature = "semantic-search")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorReindexOutcome {
    /// The document was (re-)enqueued for embedding.
    Enqueued,
    /// No document/value exists for the given id.
    NotFound,
    /// The document produced no embedding text, so nothing was enqueued.
    SkippedEmptyText,
}

/// Result of a `DocStore::index_all` call.
#[derive(Debug, Clone, Copy)]
pub struct ReindexStats {
    /// Number of exhausted queue entries (retry_count ≥ max_retries) that were
    /// removed before re-enqueueing.
    pub exhausted_cleared: usize,
    /// Number of documents enqueued for re-embedding.
    pub enqueued: usize,
}
#[cfg(test)]
mod tests;
