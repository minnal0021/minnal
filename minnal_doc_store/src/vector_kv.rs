//! Vector KV store integration for semantic-search-enabled namespaces.
//!
//! Each document namespace that has `semantic_search_enabled = true` gets three
//! companion minnal_db namespaces managed entirely by this module:
//!
//! | Internal namespace        | Key                            | Value                                    |
//! |---------------------------|--------------------------------|------------------------------------------|
//! | `{ns}_sparse_vector`      | `cluster_id (4B BE) ‖ doc_id`  | rkyv `Vec<VectorIndex>` (SingleBit only) |
//! | `{ns}_sparse_vector_meta` | `doc_id`                       | `count (2B BE u16) ‖ [cluster_id (4B BE)]×N` |
//! | `{ns}_dense_vector`       | `doc_id`                       | rkyv `Vec<VectorIndex>` (MultiBit only)  |
//!
//! The sparse namespace stores chunked single-bit embeddings keyed by IVF cluster, enabling
//! fast first-pass scans by cluster prefix.  The meta namespace tracks which clusters each
//! document's chunks were assigned to, used only by delete/upsert to clean up stale keys.
//! The dense namespace stores the full-precision multi-bit embedding keyed directly by
//! document ID, enabling O(1) lookups in the re-ranking pass without a meta indirection.
//!
//! Query-embedding caching uses a single system-wide TTL namespace
//! (`system_qemb_cache`) shared across all doc-store namespaces — see
//! [`SYSTEM_QUERY_EMB_CACHE_NS`].
//!
//! # Failure & idempotency
//!
//! No transactions span the three namespaces.  Any partial failure leaves the
//! store in an inconsistent state, but every operation is idempotent: a plain
//! retry always converges to a correct final state.
//!
//! ## Write (upsert) sequence
//! 1. Caller writes the document to the doc store.
//! 2. Caller obtains fresh `VectorIndex` entries from the embedding service.
//! 3. [`upsert_vectors`] is called:
//!    a. Sparse: reads old cluster IDs from `_sparse_vector_meta`, deletes stale keys.
//!    b. Sparse: writes new composite keys to `_sparse_vector`, updates meta.
//!    c. Dense: overwrites `doc_id` key in `_dense_vector`.
//!
//! ## Delete sequence
//! 1. Caller deletes the document from the doc store.
//! 2. [`delete_vector`] is called:
//!    a. Reads cluster IDs from `_sparse_vector_meta`, deletes all sparse composite keys.
//!    b. Deletes `doc_id` from `_dense_vector`.

use std::time::Duration;

use minnal_db::{AsyncDb, AsyncNamespace};
use semantic_search::index::vector_index::{QuantisationStyle, VectorKvStore};
use semantic_search::{VectorIndex, composite_key};

use crate::error::DocStoreError;

// ── Namespace name helpers ────────────────────────────────────────────────────

/// Internal namespace for SingleBit vectors keyed by `[cluster_id (4B) ‖ doc_id]`.
pub fn sparse_vectors_ns(namespace: &str) -> String {
    format!("{}_sparse_vector", namespace)
}

/// Internal namespace tracking which clusters each document's sparse chunks belong to.
/// Used only by delete/upsert — not read during search.
pub fn sparse_vectors_meta_ns(namespace: &str) -> String {
    format!("{}_sparse_vector_meta", namespace)
}

/// Internal namespace for MultiBit vectors keyed directly by `doc_id`.
pub fn dense_vectors_ns(namespace: &str) -> String {
    format!("{}_dense_vector", namespace)
}

/// System-wide namespace for caching query embeddings.
///
/// A single shared TTL-enabled store under the `system` namespace; all doc-store
/// namespaces with semantic search enabled write to and read from this cache
/// regardless of which namespace the query originated from.
///
/// Keys are raw UTF-8 query strings; values are big-endian `f32` vectors.
/// The namespace is TTL-enabled; the expiry duration is supplied by the caller
/// (configurable via `[semantic_search] query_embedding_cache_ttl_secs`), so
/// stale embeddings are evicted automatically.
///
/// **Durability:** all writes to this namespace (cache populate and clear) use
/// the **no-WAL** path (`put_no_wal` / `delete_no_wal`). The cache is
/// best-effort and fully regenerable — every entry can be recomputed by calling
/// the embedding service — so paying a per-write WAL fsync would add query
/// latency for no durability benefit. A crash that drops un-flushed cache writes
/// just produces future cache misses.
const SYSTEM_QUERY_EMB_CACHE_NS: &str = "system_qemb_cache";

// ── Query embedding cache ─────────────────────────────────────────────────────

/// Fallback TTL for cached query embeddings (1 day) when no configured value is
/// available — e.g. clearing the cache on a store without semantic search
/// configured.
pub const DEFAULT_QUERY_EMBEDDING_CACHE_TTL: Duration = Duration::from_secs(86_400);

/// Maximum records the TTL worker will delete per pass.
const QUERY_EMBEDDING_CACHE_MAX_DELETES: usize = 10_000;

/// Look up cached query embeddings from the system-wide TTL cache.
///
/// Returns `(dense, sparse)` where `dense` is the single whole-query embedding
/// (Pass 2) and `sparse` is the list of sliding-window chunk embeddings (Pass 1).
/// The two are stored together as one encoded list with the dense vector first.
///
/// Returns `None` on cache miss, dimension mismatch, or any I/O error so that
/// the caller always falls back to the embedding service transparently.
pub async fn get_cached_query_embedding(db: &AsyncDb, query_text: &str, expected_dim: usize, ttl: Duration) -> Option<(Vec<f32>, Vec<Vec<f32>>)> {
    let cache_ns = db
        .namespace_with_ttl(SYSTEM_QUERY_EMB_CACHE_NS.to_string(), ttl, QUERY_EMBEDDING_CACHE_MAX_DELETES)
        .await
        .ok()?;

    let bytes = cache_ns.get(query_text.as_bytes().to_vec()).await.ok()??;
    let list = bytes_to_f32_vec_list(&bytes, expected_dim)?;
    let mut it = list.into_iter();
    let dense = it.next()?; // first element is the whole-query dense embedding
    Some((dense, it.collect()))
}

/// Store query embeddings in the system-wide TTL cache.
///
/// `dense` (the whole-query embedding) is stored as the first element of the
/// encoded list, followed by the `sparse` chunk embeddings; [`get_cached_query_embedding`]
/// splits them back out. Failures are silently ignored — the cache is
/// best-effort and must never block or fail a query.
pub async fn put_cached_query_embedding(db: &AsyncDb, query_text: &str, dense: &[f32], sparse: &[Vec<f32>], ttl: Duration) {
    let Ok(cache_ns) = db
        .namespace_with_ttl(SYSTEM_QUERY_EMB_CACHE_NS.to_string(), ttl, QUERY_EMBEDDING_CACHE_MAX_DELETES)
        .await
    else {
        return;
    };

    let mut list = Vec::with_capacity(1 + sparse.len());
    list.push(dense.to_vec());
    list.extend_from_slice(sparse);
    // No-WAL: the cache is best-effort, TTL-bounded and fully regenerable by
    // re-calling the embedding service on a miss, so a WAL fsync per populate
    // would be pure query-path latency with no durability benefit. A crash that
    // drops the entry simply turns into a future cache miss.
    let _ = cache_ns.put_no_wal(query_text.as_bytes().to_vec(), f32_vec_list_to_bytes(&list)).await;
}

/// Delete every entry from the system-wide query-embedding cache.
///
/// Returns the number of cached entries removed. This is an explicit
/// administrative operation: the cache is keyed only by query text, so after a
/// change to the chunking parameters (`window_size` / `sliding_size`) the cached
/// sparse vectors no longer match freshly-indexed documents — the cache must be
/// cleared (alongside a corpus re-index) or stale entries will silently degrade
/// recall until the configured TTL expires.
pub async fn clear_cached_query_embeddings(db: &AsyncDb, ttl: Duration) -> Result<usize, DocStoreError> {
    let cache_ns = db
        .namespace_with_ttl(SYSTEM_QUERY_EMB_CACHE_NS.to_string(), ttl, QUERY_EMBEDDING_CACHE_MAX_DELETES)
        .await
        .map_err(|e| DocStoreError::EmbeddingFailed(e.to_string()))?;

    let keys = cache_ns.keys().await.map_err(|e| DocStoreError::EmbeddingFailed(e.to_string()))?;
    let mut deleted = 0usize;
    for key in keys {
        // No-WAL: matches the no-WAL populate path. A lost clear-delete just
        // leaves a regenerable entry that the TTL worker evicts anyway.
        cache_ns.delete_no_wal(key).await.map_err(|e| DocStoreError::EmbeddingFailed(e.to_string()))?;
        deleted += 1;
    }
    Ok(deleted)
}

/// Encode a list of `Vec<f32>` embeddings as bytes.
///
/// Format: `[count: 4B BE u32][f32s for embedding 0][f32s for embedding 1]...`
/// Each f32 is encoded as 4 big-endian bytes.
fn f32_vec_list_to_bytes(list: &[Vec<f32>]) -> Vec<u8> {
    let count = list.len() as u32;
    let mut out = Vec::with_capacity(4 + list.iter().map(|v| v.len() * 4).sum::<usize>());
    out.extend_from_slice(&count.to_be_bytes());
    for v in list {
        out.extend(v.iter().flat_map(|f| f.to_be_bytes()));
    }
    out
}

/// Decode bytes back into a list of `Vec<f32>` embeddings.
///
/// Returns `None` if the bytes are malformed or any embedding has the wrong dimension.
fn bytes_to_f32_vec_list(bytes: &[u8], expected_dim: usize) -> Option<Vec<Vec<f32>>> {
    if bytes.len() < 4 {
        return None;
    }
    let count = u32::from_be_bytes(bytes[..4].try_into().ok()?) as usize;
    let rest = &bytes[4..];
    if rest.len() != count * expected_dim * 4 {
        return None;
    }
    Some(
        rest.chunks_exact(expected_dim * 4)
            .map(|chunk| chunk.chunks_exact(4).map(|c| f32::from_be_bytes(c.try_into().unwrap())).collect())
            .collect(),
    )
}

// ── Sparse-meta encoding ──────────────────────────────────────────────────────
//
// Format: `count (2B BE u16) ‖ [cluster_id (4B BE)] × count`

fn encode_sparse_meta(cluster_ids: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + cluster_ids.len() * 4);
    out.extend_from_slice(&(cluster_ids.len() as u16).to_be_bytes());
    for &id in cluster_ids {
        out.extend_from_slice(&id.to_be_bytes());
    }
    out
}

fn decode_sparse_meta(bytes: &[u8]) -> Option<Vec<u32>> {
    if bytes.len() < 2 {
        return None;
    }
    let count = u16::from_be_bytes(bytes[..2].try_into().ok()?) as usize;
    if bytes.len() != 2 + count * 4 {
        return None;
    }
    let mut ids = Vec::with_capacity(count);
    for i in 0..count {
        let off = 2 + i * 4;
        ids.push(u32::from_be_bytes(bytes[off..off + 4].try_into().ok()?));
    }
    Some(ids)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Write the quantised vector index for a document to the vector KV store.
///
/// `vector_indexes` is the combined list of SingleBit and MultiBit quantised entries
/// for the document, as returned by `embed_document`.
///
/// SingleBit entries are written to `{ns}_sparse_vector` (keyed by cluster prefix),
/// with stale cluster keys cleaned up via `{ns}_sparse_vector_meta`.
/// The MultiBit entry (at most one) is written directly to `{ns}_dense_vector` by doc_id.
///
/// Each write is a standalone single-op write — the vector index is a derived
/// structure that the [`VecIndexWorker`] (re)builds idempotently from the queue,
/// so it needs no cross-namespace atomicity with the document write.
///
/// The quantised payloads are written **without** the WAL (`put_no_wal`): they
/// are bulky and reconstructable by re-running the embed/re-index, so paying a
/// per-chunk WAL fsync (and storing a second copy of every quantised payload in
/// the WAL) is pure overhead.  A crash before the memtable flushes loses these
/// entries — possibly flushing the sparse side but losing the dense write (or
/// vice versa), leaving a partially committed index; the affected docs are
/// re-enqueued by vector-index reconciliation, which treats a partial index as
/// not-indexed (see [`has_complete_vector_index`]) and runs automatically as a
/// background task on startup (and on demand via
/// `POST /admin/indices/vector/reconcile`).  The stale-cluster `delete`s remain
/// WAL-backed — they are tiny key-only tombstones and do not fire during a fresh
/// bulk load.
///
/// [`VecIndexWorker`]: crate::vec_index_worker::VecIndexWorker
pub async fn upsert_vectors(db: &AsyncDb, namespace: &str, doc_id_bytes: &[u8], vector_indexes: &[VectorIndex]) -> Result<(), DocStoreError> {
    if vector_indexes.is_empty() {
        return Ok(());
    }

    // ── Sparse (SingleBit) ────────────────────────────────────────────────────
    let sparse_vis: Vec<&VectorIndex> = vector_indexes
        .iter()
        .filter(|vi| vi.quantisation_style == QuantisationStyle::SingleBit)
        .collect();
    if !sparse_vis.is_empty() {
        let sparse_ns = db.namespace(sparse_vectors_ns(namespace)).await?;
        let sparse_meta_ns = db.namespace(sparse_vectors_meta_ns(namespace)).await?;

        let mut new_cluster_groups: std::collections::HashMap<u32, Vec<VectorIndex>> = std::collections::HashMap::new();
        for vi in &sparse_vis {
            new_cluster_groups.entry(vi.cluster_id).or_default().push((*vi).clone());
        }
        let new_cluster_ids: Vec<u32> = new_cluster_groups.keys().copied().collect();

        // Delete stale cluster keys no longer present.
        if let Some(old_bytes) = sparse_meta_ns.get(doc_id_bytes.to_vec()).await?
            && let Some(old_ids) = decode_sparse_meta(&old_bytes)
        {
            for old_id in old_ids {
                if !new_cluster_ids.contains(&old_id) {
                    sparse_ns.delete(composite_key::encode(old_id, doc_id_bytes)).await?;
                }
            }
        }

        for (&cluster_id, chunks) in &new_cluster_groups {
            let key = composite_key::encode(cluster_id, doc_id_bytes);
            sparse_ns.put_no_wal(key, VectorIndex::list_to_bytes(chunks)).await?;
        }
        sparse_meta_ns
            .put_no_wal(doc_id_bytes.to_vec(), encode_sparse_meta(&new_cluster_ids))
            .await?;
    }

    // ── Dense (MultiBit) ─────────────────────────────────────────────────────
    let dense_vis: Vec<&VectorIndex> = vector_indexes
        .iter()
        .filter(|vi| matches!(vi.quantisation_style, QuantisationStyle::MultiBit { .. }))
        .collect();
    if !dense_vis.is_empty() {
        let dense_ns = db.namespace(dense_vectors_ns(namespace)).await?;
        let owned: Vec<VectorIndex> = dense_vis.into_iter().cloned().collect();
        dense_ns.put_no_wal(doc_id_bytes.to_vec(), VectorIndex::list_to_bytes(&owned)).await?;
    }

    Ok(())
}

/// Delete the quantised vector for a document from the vector KV store.
///
/// No-op if no vector entry exists for `doc_id_bytes` (the document was never
/// indexed, or was already cleaned up).
pub async fn delete_vector(db: &AsyncDb, namespace: &str, doc_id_bytes: &[u8]) -> Result<(), DocStoreError> {
    // Sparse: read cluster IDs from meta, delete each composite key, then delete meta.
    let sparse_meta_ns = db.namespace(sparse_vectors_meta_ns(namespace)).await?;
    if let Some(meta_bytes) = sparse_meta_ns.get(doc_id_bytes.to_vec()).await? {
        if let Some(cluster_ids) = decode_sparse_meta(&meta_bytes) {
            let sparse_ns = db.namespace(sparse_vectors_ns(namespace)).await?;
            for cluster_id in cluster_ids {
                sparse_ns.delete(composite_key::encode(cluster_id, doc_id_bytes)).await?;
            }
        }
        sparse_meta_ns.delete(doc_id_bytes.to_vec()).await?;
    }

    // Dense: delete by doc_id directly (no meta needed).
    let dense_ns = db.namespace(dense_vectors_ns(namespace)).await?;
    dense_ns.delete(doc_id_bytes.to_vec()).await?;

    Ok(())
}

/// Return `true` only if a document has a **complete** committed vector index —
/// both the sparse meta record **and** the dense entry.
///
/// A normally-indexed document always has both: the embed pipeline emits chunked
/// SingleBit entries (sparse) *and* a whole-doc MultiBit entry (dense). Requiring
/// both — rather than either — means reconciliation re-enqueues a document whose
/// index is only *partially* committed (e.g. a crash that flushed the sparse side
/// but lost the no-WAL dense write, or vice versa) instead of skipping it as
/// "indexed". The re-embed regenerates the missing half idempotently.
///
/// Used by startup reconciliation to skip documents that are already fully indexed.
pub async fn has_complete_vector_index(db: &AsyncDb, namespace: &str, doc_id_bytes: &[u8]) -> Result<bool, DocStoreError> {
    let sparse_meta_ns = db.namespace(sparse_vectors_meta_ns(namespace)).await?;
    if sparse_meta_ns.get(doc_id_bytes.to_vec()).await?.is_none() {
        return Ok(false);
    }
    let dense_ns = db.namespace(dense_vectors_ns(namespace)).await?;
    Ok(dense_ns.get(doc_id_bytes.to_vec()).await?.is_some())
}

/// Like [`has_complete_vector_index`], but also verifies the committed bytes
/// **deserialize** — catching entries that are present but corrupt, which the
/// presence-only check counts as indexed.
///
/// Reads and deserializes the sparse-meta, every sparse composite entry, and the
/// dense entry for `doc_id_bytes`. Returns `false` if any is missing, undecodable,
/// or fails [`VectorIndex::list_from_bytes`], so the *validating* reconcile pass
/// re-enqueues the document (a re-embed regenerates both halves idempotently).
///
/// Substantially more expensive than [`has_complete_vector_index`] — it value-reads
/// and deserializes per entry rather than checking key presence — so only the
/// on-demand validating reconcile (`reconcile_all_vector_indexes(.., check_bytes=true)`)
/// uses it; the cheap startup pass keeps the presence check.
pub async fn has_valid_vector_index(db: &AsyncDb, namespace: &str, doc_id_bytes: &[u8]) -> Result<bool, DocStoreError> {
    // Sparse meta: present and decodable into cluster IDs.
    let sparse_meta_ns = db.namespace(sparse_vectors_meta_ns(namespace)).await?;
    let Some(meta_bytes) = sparse_meta_ns.get(doc_id_bytes.to_vec()).await? else {
        return Ok(false);
    };
    let Some(cluster_ids) = decode_sparse_meta(&meta_bytes) else {
        return Ok(false);
    };

    // Each sparse composite entry: present and deserializes.
    let sparse_ns = db.namespace(sparse_vectors_ns(namespace)).await?;
    for cluster_id in &cluster_ids {
        let Some(bytes) = sparse_ns.get(composite_key::encode(*cluster_id, doc_id_bytes)).await? else {
            return Ok(false);
        };
        if VectorIndex::list_from_bytes(&bytes).is_err() {
            return Ok(false);
        }
    }

    // Dense: present and deserializes.
    let dense_ns = db.namespace(dense_vectors_ns(namespace)).await?;
    let Some(dense_bytes) = dense_ns.get(doc_id_bytes.to_vec()).await? else {
        return Ok(false);
    };
    Ok(VectorIndex::list_from_bytes(&dense_bytes).is_ok())
}

// ── KV-backed VectorKvStore ───────────────────────────────────────────────────

/// A [`VectorKvStore`] implementation backed directly by minnal_db namespaces.
///
/// `scan_sparse_cluster` performs a 4-byte cluster prefix scan on `{ns}_sparse_vector`.
/// `get_dense_entry` performs a direct key lookup on `{ns}_dense_vector`.
///
/// Both namespaces are resolved once at construction time so that
/// `scan_sparse_cluster` and `get_dense_entry` never pay a namespace-lookup
/// cost per call.
pub struct DbVectorStore {
    sparse_ns: AsyncNamespace,
    dense_ns: AsyncNamespace,
}

impl DbVectorStore {
    pub async fn new(db: &AsyncDb, namespace: &str) -> Result<Self, minnal_db::KVError> {
        let sparse_ns = db.namespace(sparse_vectors_ns(namespace)).await?;
        let dense_ns = db.namespace(dense_vectors_ns(namespace)).await?;
        Ok(Self { sparse_ns, dense_ns })
    }
}

impl VectorKvStore for DbVectorStore {
    async fn scan_sparse_cluster(&self, cluster_id: u32) -> Vec<(Vec<u8>, Vec<u8>)> {
        // Prefix = cluster_id (4B BE) — only entries for this cluster are returned.
        let prefix = cluster_id.to_be_bytes().to_vec();
        let Ok(entries) = self.sparse_ns.scan_prefix(prefix).await else {
            return vec![];
        };

        entries
            .into_iter()
            .filter_map(|(key, value)| {
                let (_, doc_id_bytes) = composite_key::decode(&key)?;
                Some((doc_id_bytes.to_vec(), value))
            })
            .collect()
    }

    async fn get_dense_entry(&self, doc_id_bytes: &[u8]) -> Option<Vec<u8>> {
        self.dense_ns.get(doc_id_bytes.to_vec()).await.ok()?
    }

    async fn get_dense_entries_batch(&self, doc_ids: &[Vec<u8>]) -> Vec<Option<Vec<u8>>> {
        self.dense_ns.get_multiple(doc_ids.to_vec()).await
    }

    async fn scan_sparse_clusters_batch(&self, cluster_ids: &[u32]) -> std::collections::HashMap<u32, Vec<(Vec<u8>, Vec<u8>)>> {
        let raw = self.sparse_ns.scan_prefixes_batch(cluster_ids.to_vec()).await;
        // Decode composite keys: strip the 4-byte cluster_id prefix, keep only doc_id.
        raw.into_iter()
            .map(|(cluster_id, entries)| {
                let decoded = entries
                    .into_iter()
                    .filter_map(|(key, value)| {
                        let (_, doc_id_bytes) = composite_key::decode(&key)?;
                        Some((doc_id_bytes.to_vec(), value))
                    })
                    .collect();
                (cluster_id, decoded)
            })
            .collect()
    }
}

// ── Async vector-index queue ──────────────────────────────────────────────────

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

/// One entry in the async vector-index embedding queue.
#[derive(Clone, Debug)]
pub struct QueueEntry {
    /// The document store namespace this entry belongs to.
    pub namespace: String,
    /// Raw document key bytes (big-endian encoded ID).
    pub doc_id_bytes: Vec<u8>,
    /// Text to embed.
    pub text: String,
    /// Number of failed embedding attempts so far.
    pub retry_count: u32,
    /// Error message from the most recent failed embedding attempt.
    pub last_error: Option<String>,
}

/// Version `0x01`: `[version] [retry_count: 4B BE] [text bytes]`
const QUEUE_VALUE_V1: u8 = 0x01;
/// Version `0x02`: `[version] [retry_count: 4B BE] [error_len: 4B BE] [error bytes] [text bytes]`
const QUEUE_VALUE_VERSION: u8 = 0x02;

fn encode_queue_value(text: &str, retry_count: u32, last_error: Option<&str>) -> Vec<u8> {
    let error_bytes = last_error.unwrap_or("").as_bytes();
    let mut value = Vec::with_capacity(1 + 4 + 4 + error_bytes.len() + text.len());
    value.push(QUEUE_VALUE_VERSION);
    value.extend_from_slice(&retry_count.to_be_bytes());
    value.extend_from_slice(&(error_bytes.len() as u32).to_be_bytes());
    value.extend_from_slice(error_bytes);
    value.extend_from_slice(text.as_bytes());
    value
}

/// Returns `(text, retry_count, last_error)`.  Handles both v1 and v2 on-disk formats.
fn decode_queue_value(value: &[u8]) -> Option<(String, u32, Option<String>)> {
    let version = *value.first()?;
    match version {
        v if v == QUEUE_VALUE_V1 => {
            if value.len() < 5 {
                return None;
            }
            let retry_count = u32::from_be_bytes(value[1..5].try_into().ok()?);
            let text = std::str::from_utf8(&value[5..]).ok()?.to_owned();
            Some((text, retry_count, None))
        }
        v if v == QUEUE_VALUE_VERSION => {
            if value.len() < 9 {
                return None;
            }
            let retry_count = u32::from_be_bytes(value[1..5].try_into().ok()?);
            let error_len = u32::from_be_bytes(value[5..9].try_into().ok()?) as usize;
            if value.len() < 9 + error_len {
                return None;
            }
            let last_error = if error_len > 0 {
                Some(std::str::from_utf8(&value[9..9 + error_len]).ok()?.to_owned())
            } else {
                None
            };
            let text = std::str::from_utf8(&value[9 + error_len..]).ok()?.to_owned();
            Some((text, retry_count, last_error))
        }
        _ => None,
    }
}

fn queue_key(namespace: &str, doc_id_bytes: &[u8]) -> Vec<u8> {
    let ns = namespace.as_bytes();
    let ns_len = ns.len() as u16;
    let mut key = Vec::with_capacity(2 + ns.len() + doc_id_bytes.len());
    key.extend_from_slice(&ns_len.to_be_bytes());
    key.extend_from_slice(ns);
    key.extend_from_slice(doc_id_bytes);
    key
}

fn decode_queue_key(key: &[u8]) -> Option<(&str, &[u8])> {
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
/// Every enqueue is WAL-backed, including the bulk re-index paths: the vector
/// index itself is written no-WAL ([`upsert_vectors`]), so the queue is the
/// durable source of truth for what still needs (re-)indexing and must survive
/// a crash on every path.
pub async fn enqueue_embed(db: &AsyncDb, namespace: &str, doc_id_bytes: &[u8], text: &str) -> Result<(), crate::error::DocStoreError> {
    let queue_ns = db.namespace(PENDING_VEC_INDEX_NS.to_string()).await?;
    queue_ns
        .put(queue_key(namespace, doc_id_bytes), encode_queue_value(text, 0, None))
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
) -> Result<(), crate::error::DocStoreError> {
    let queue_ns = db.namespace(PENDING_VEC_INDEX_NS.to_string()).await?;
    queue_ns
        .put(queue_key(namespace, doc_id_bytes), encode_queue_value(text, retry_count, last_error))
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
pub async fn remove_queue_entry(db: &AsyncDb, namespace: &str, doc_id_bytes: &[u8]) -> Result<(), crate::error::DocStoreError> {
    let queue_ns = db.namespace(PENDING_VEC_INDEX_NS.to_string()).await?;
    queue_ns.delete(queue_key(namespace, doc_id_bytes)).await?;
    Ok(())
}

/// Return all pending embedding queue entries.
///
/// Called by the [`VecIndexWorker`] on startup (crash recovery) and after
/// each wake-up signal.  Returns an empty list when the queue is empty.
///
/// [`VecIndexWorker`]: crate::vec_index_worker::VecIndexWorker
pub async fn list_queue_entries(db: &AsyncDb) -> Result<Vec<QueueEntry>, crate::error::DocStoreError> {
    let queue_ns = db.namespace(PENDING_VEC_INDEX_NS.to_string()).await?;
    let entries = queue_ns.iter().await?;
    let mut result = Vec::with_capacity(entries.len());
    for (key, value) in entries {
        if let Some((namespace, doc_id_bytes)) = decode_queue_key(&key)
            && let Some((text, retry_count, last_error)) = decode_queue_value(&value)
        {
            result.push(QueueEntry {
                namespace: namespace.to_owned(),
                doc_id_bytes: doc_id_bytes.to_vec(),
                text,
                retry_count,
                last_error,
            });
        }
    }
    Ok(result)
}

/// Look up a single pending embedding queue entry by namespace and document ID.
///
/// Returns `None` when no entry exists for the given `(namespace, doc_id_bytes)` pair.
/// This is an O(1) key lookup and does not scan the whole queue.
pub async fn get_queue_entry(db: &AsyncDb, namespace: &str, doc_id_bytes: &[u8]) -> Result<Option<QueueEntry>, crate::error::DocStoreError> {
    let queue_ns = db.namespace(PENDING_VEC_INDEX_NS.to_string()).await?;
    let key = queue_key(namespace, doc_id_bytes);
    match queue_ns.get(key).await? {
        None => Ok(None),
        Some(value) => match decode_queue_value(&value) {
            Some((text, retry_count, last_error)) => Ok(Some(QueueEntry {
                namespace: namespace.to_owned(),
                doc_id_bytes: doc_id_bytes.to_vec(),
                text,
                retry_count,
                last_error,
            })),
            None => Ok(None),
        },
    }
}

// ── Vector upsert tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod vector_upsert_tests {
    use super::*;
    use minnal_db::AsyncDb;
    use semantic_search::index::distance_estimator::{DotProductEstimatorState, MultiBitQuanDotProductEstimator};
    use semantic_search::index::vector_index::{QuantisationStyle, VectorKvStore, score_rkyv_bytes};
    use tempfile::TempDir;

    async fn open_db(dir: &TempDir) -> AsyncDb {
        AsyncDb::open(dir.path().to_owned()).await.unwrap()
    }

    const MULTI8: QuantisationStyle = QuantisationStyle::MultiBit { number_of_bits: 8 };

    // With zero estimator state and empty packed_vector, estimated_distance = 1.0 - addition_factor.
    fn zero_estimator() -> MultiBitQuanDotProductEstimator {
        MultiBitQuanDotProductEstimator(DotProductEstimatorState {
            cluster_id: 1,
            query_to_centroid_dot_product: 0.0,
            scaled_query_sum: 0.0,
        })
    }

    // SingleBit entry → goes to sparse namespace, scannable by cluster.
    #[tokio::test]
    async fn test_upsert_single_bit_entry_is_scannable_by_cluster() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        let ns = "docs";
        let doc_id = b"doc1";

        let vi = VectorIndex::new(1, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
        upsert_vectors(&db, ns, doc_id, &[vi]).await.unwrap();

        let store = DbVectorStore::new(&db, ns).await.unwrap();
        let entries = store.scan_sparse_cluster(1).await;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, doc_id);
    }

    // MultiBit entry → goes to dense namespace, fetchable by doc_id.
    #[tokio::test]
    async fn test_upsert_multi_bit_entry_is_fetchable_by_doc_id() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        let ns = "docs";
        let doc_id = b"doc2";

        let vi = VectorIndex::new(1, MULTI8, 0.3, 0.0, 0.05, vec![]);
        upsert_vectors(&db, ns, doc_id, &[vi]).await.unwrap();

        let store = DbVectorStore::new(&db, ns).await.unwrap();
        let bytes = store.get_dense_entry(doc_id).await.unwrap();
        let vis = VectorIndex::list_from_bytes(&bytes).unwrap();
        assert_eq!(vis.len(), 1);

        let (score, _) = score_rkyv_bytes(&bytes, &[], &zero_estimator()).unwrap();
        assert!((score - 0.7).abs() < 1e-6);
    }

    // Multiple MultiBit chunks stored in dense namespace — SimMax picks the best.
    #[tokio::test]
    async fn test_multi_bit_multi_chunk_simmax() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        let ns = "docs";
        let doc_id = b"doc3";

        // Three chunks with scores 0.7, 0.9, 0.5 — SimMax must return 0.9.
        let chunks = vec![
            VectorIndex::new(1, MULTI8, 0.3, 0.0, 0.01, vec![]), // 1 - 0.3 = 0.7
            VectorIndex::new(1, MULTI8, 0.1, 0.0, 0.05, vec![]), // 1 - 0.1 = 0.9 ← winner
            VectorIndex::new(1, MULTI8, 0.5, 0.0, 0.02, vec![]), // 1 - 0.5 = 0.5
        ];

        upsert_vectors(&db, ns, doc_id, &chunks).await.unwrap();

        let store = DbVectorStore::new(&db, ns).await.unwrap();
        let bytes = store.get_dense_entry(doc_id).await.unwrap();

        let (score, error_bound) = score_rkyv_bytes(&bytes, &[], &zero_estimator()).unwrap();
        assert!((score - 0.9).abs() < 1e-6, "SimMax should pick 0.9, got {score}");
        assert!((error_bound - 0.05).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_upsert_vectors_empty_list_is_noop() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        let ns = "docs";

        upsert_vectors(&db, ns, b"doc4", &[]).await.unwrap();

        // An empty list must write nothing to either vector namespace.
        let store = DbVectorStore::new(&db, ns).await.unwrap();
        assert!(store.get_dense_entry(b"doc4").await.is_none(), "empty list must write nothing");
    }

    // Cluster reassignment for SingleBit deletes the stale sparse composite key.
    #[tokio::test]
    async fn test_sparse_cluster_change_deletes_stale_key() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        let ns = "docs";
        let doc_id = b"doc5";

        let vi1 = VectorIndex::new(1, QuantisationStyle::SingleBit, 0.4, 0.0, 0.01, vec![]);
        upsert_vectors(&db, ns, doc_id, &[vi1]).await.unwrap();

        let store = DbVectorStore::new(&db, ns).await.unwrap();
        assert_eq!(store.scan_sparse_cluster(1).await.len(), 1);
        assert_eq!(store.scan_sparse_cluster(2).await.len(), 0);

        let vi2 = VectorIndex::new(2, QuantisationStyle::SingleBit, 0.35, 0.0, 0.02, vec![]);
        upsert_vectors(&db, ns, doc_id, &[vi2]).await.unwrap();

        assert_eq!(store.scan_sparse_cluster(1).await.len(), 0, "stale cluster-1 key must be deleted");
        assert_eq!(store.scan_sparse_cluster(2).await.len(), 1, "new cluster-2 key must exist");
    }

    // ── Sparse-meta encoding round-trip ──────────────────────────────────────

    #[test]
    fn test_encode_decode_sparse_meta_roundtrip() {
        let ids = vec![10u32, 20u32, 30u32];
        let encoded = encode_sparse_meta(&ids);
        let decoded = decode_sparse_meta(&encoded).unwrap();
        assert_eq!(decoded, ids);
    }

    #[test]
    fn test_encode_decode_sparse_meta_empty() {
        let ids: Vec<u32> = vec![];
        let encoded = encode_sparse_meta(&ids);
        let decoded = decode_sparse_meta(&encoded).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_decode_sparse_meta_rejects_truncated() {
        // 3 bytes is shorter than the minimum 2-byte header + 4 bytes per entry.
        assert!(decode_sparse_meta(&[0x00, 0x01, 0x02]).is_none());
    }

    #[test]
    fn test_decode_sparse_meta_rejects_wrong_length() {
        // Header claims count=1 but only 3 bytes of payload instead of 4.
        let mut bad = vec![0x00u8, 0x01]; // count = 1
        bad.extend_from_slice(&[0x01, 0x00, 0x00]); // only 3 bytes instead of 4
        assert!(decode_sparse_meta(&bad).is_none());
    }

    // ── delete_vector ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_delete_vector_removes_sparse_and_dense() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        let ns = "docs";
        let doc_id = b"del_doc";

        let vi_sb = VectorIndex::new(5, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
        let vi_mb = VectorIndex::new(5, MULTI8, 0.3, 0.0, 0.01, vec![]);
        upsert_vectors(&db, ns, doc_id, &[vi_sb, vi_mb]).await.unwrap();

        let store = DbVectorStore::new(&db, ns).await.unwrap();
        assert_eq!(store.scan_sparse_cluster(5).await.len(), 1, "sparse must exist before delete");
        assert!(store.get_dense_entry(doc_id).await.is_some(), "dense must exist before delete");

        delete_vector(&db, ns, doc_id).await.unwrap();

        assert_eq!(store.scan_sparse_cluster(5).await.len(), 0, "sparse must be gone after delete");
        assert!(store.get_dense_entry(doc_id).await.is_none(), "dense must be gone after delete");

        // Second delete is a no-op (meta is gone).
        delete_vector(&db, ns, doc_id).await.unwrap();
    }

    #[tokio::test]
    async fn test_delete_vector_noop_when_never_indexed() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        // Should not panic or return an error.
        delete_vector(&db, "docs", b"ghost").await.unwrap();
    }

    // ── Namespace isolation ───────────────────────────────────────────────────

    // MultiBit-only upsert must not write to the sparse namespace or sparse meta.
    #[tokio::test]
    async fn test_dense_only_upsert_leaves_sparse_namespace_empty() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        let ns = "docs";
        let doc_id = b"dense_only_doc";

        let vi = VectorIndex::new(3, MULTI8, 0.3, 0.0, 0.01, vec![]);
        upsert_vectors(&db, ns, doc_id, &[vi]).await.unwrap();

        let store = DbVectorStore::new(&db, ns).await.unwrap();
        assert!(store.get_dense_entry(doc_id).await.is_some(), "dense entry must exist");
        assert!(store.scan_sparse_cluster(3).await.is_empty(), "no sparse entry must be written");

        // Sparse meta must also be absent.
        let meta_ns = db.namespace(sparse_vectors_meta_ns(ns)).await.unwrap();
        assert!(meta_ns.get(doc_id.to_vec()).await.unwrap().is_none(), "sparse meta must not be written");
    }

    // SingleBit-only upsert must not write to the dense namespace.
    #[tokio::test]
    async fn test_sparse_only_upsert_leaves_dense_namespace_empty() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        let ns = "docs";
        let doc_id = b"sparse_only_doc";

        let vi = VectorIndex::new(7, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
        upsert_vectors(&db, ns, doc_id, &[vi]).await.unwrap();

        let store = DbVectorStore::new(&db, ns).await.unwrap();
        assert_eq!(store.scan_sparse_cluster(7).await.len(), 1, "sparse entry must exist");
        assert!(store.get_dense_entry(doc_id).await.is_none(), "no dense entry must be written");
    }

    // ── delete_vector with multiple sparse clusters ───────────────────────────

    #[tokio::test]
    async fn test_delete_vector_removes_all_sparse_clusters() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        let ns = "docs";
        let doc_id = b"multi_cluster_doc";

        let vi_sb1 = VectorIndex::new(1, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
        let vi_sb2 = VectorIndex::new(2, QuantisationStyle::SingleBit, 0.35, 0.0, 0.03, vec![]);

        upsert_vectors(&db, ns, doc_id, &[vi_sb1, vi_sb2]).await.unwrap();

        let store = DbVectorStore::new(&db, ns).await.unwrap();
        assert_eq!(store.scan_sparse_cluster(1).await.len(), 1);
        assert_eq!(store.scan_sparse_cluster(2).await.len(), 1);

        delete_vector(&db, ns, doc_id).await.unwrap();

        assert_eq!(store.scan_sparse_cluster(1).await.len(), 0, "cluster-1 must be deleted");
        assert_eq!(store.scan_sparse_cluster(2).await.len(), 0, "cluster-2 must be deleted");
    }
}

// ── Queue tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod queue_tests {
    use super::*;
    use minnal_db::AsyncDb;
    use tempfile::TempDir;

    async fn open_db(dir: &TempDir) -> AsyncDb {
        let path = dir.path().to_owned();
        AsyncDb::open(path).await.unwrap()
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

    #[test]
    fn test_value_encoding_roundtrip_zero_retries() {
        let encoded = encode_queue_value("hello", 0, None);
        let (text, retry_count, last_error) = decode_queue_value(&encoded).unwrap();
        assert_eq!(text, "hello");
        assert_eq!(retry_count, 0);
        assert!(last_error.is_none());
    }

    #[test]
    fn test_value_encoding_roundtrip_nonzero_retries() {
        let encoded = encode_queue_value("embed this", 3, None);
        let (text, retry_count, _) = decode_queue_value(&encoded).unwrap();
        assert_eq!(text, "embed this");
        assert_eq!(retry_count, 3);
    }

    #[test]
    fn test_value_encoding_with_last_error() {
        let encoded = encode_queue_value("some text", 2, Some("connection refused"));
        let (text, retry_count, last_error) = decode_queue_value(&encoded).unwrap();
        assert_eq!(text, "some text");
        assert_eq!(retry_count, 2);
        assert_eq!(last_error.as_deref(), Some("connection refused"));
    }

    #[test]
    fn test_value_encoding_empty_text() {
        let encoded = encode_queue_value("", 1, None);
        let (text, retry_count, _) = decode_queue_value(&encoded).unwrap();
        assert_eq!(text, "");
        assert_eq!(retry_count, 1);
    }

    #[test]
    fn test_v1_backward_compat() {
        // Simulate an on-disk v1 value: [0x01] [retry_count: 4B] [text bytes]
        let mut v1 = vec![0x01u8, 0x00, 0x00, 0x00, 0x02];
        v1.extend_from_slice(b"old text");
        let (text, retry_count, last_error) = decode_queue_value(&v1).unwrap();
        assert_eq!(text, "old text");
        assert_eq!(retry_count, 2);
        assert!(last_error.is_none());
    }

    #[test]
    fn test_value_decoding_rejects_unknown_version() {
        // 0x03 is not a known version.
        let bad = b"\x03\x00\x00\x00\x00\x00\x00\x00\x00hello";
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
}

// ── Dual-style (multi-bit + single-bit) upsert tests ─────────────────────────

#[cfg(test)]
mod dual_style_tests {
    use super::*;
    use minnal_db::AsyncDb;
    use semantic_search::index::vector_index::{QuantisationStyle, VectorKvStore};
    use tempfile::TempDir;

    const MULTI8: QuantisationStyle = QuantisationStyle::MultiBit { number_of_bits: 8 };

    async fn open_db(dir: &TempDir) -> AsyncDb {
        AsyncDb::open(dir.path().to_owned()).await.unwrap()
    }

    /// One MultiBit + two SingleBit entries stored together. MultiBit goes to
    /// dense namespace (fetchable by doc_id); SingleBit goes to sparse namespace
    /// (scannable by cluster).
    #[tokio::test]
    async fn test_dual_style_upsert_routes_to_correct_namespace() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        let ns = "docs";
        let doc_id = b"dual_doc";

        let vi_mb = VectorIndex::new(1, MULTI8, 0.3, 0.0, 0.01, vec![]);
        let vi_sb1 = VectorIndex::new(1, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
        let vi_sb2 = VectorIndex::new(2, QuantisationStyle::SingleBit, 0.35, 0.0, 0.03, vec![]);

        upsert_vectors(&db, ns, doc_id, &[vi_mb, vi_sb1, vi_sb2]).await.unwrap();

        let store = DbVectorStore::new(&db, ns).await.unwrap();

        let dense = store.get_dense_entry(doc_id).await;
        assert!(dense.is_some(), "MultiBit entry must be in dense namespace");
        let vis = VectorIndex::list_from_bytes(&dense.unwrap()).unwrap();
        assert_eq!(vis.len(), 1);
        assert!((vis[0].addition_factor - 0.3).abs() < 1e-6);

        let sb1 = store.scan_sparse_cluster(1).await;
        assert_eq!(sb1.len(), 1, "SingleBit cluster 1: exactly 1 entry");
        assert_eq!(sb1[0].0, doc_id);

        let sb2 = store.scan_sparse_cluster(2).await;
        assert_eq!(sb2.len(), 1, "SingleBit cluster 2: exactly 1 entry");
        assert_eq!(sb2[0].0, doc_id);
    }

    /// delete_vector removes all three entries (MB + 2 SB) and the sparse meta record.
    #[tokio::test]
    async fn test_dual_style_delete_removes_all_entries() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        let ns = "docs";
        let doc_id = b"dual_del_doc";

        let vi_mb = VectorIndex::new(1, MULTI8, 0.3, 0.0, 0.01, vec![]);
        let vi_sb1 = VectorIndex::new(1, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
        let vi_sb2 = VectorIndex::new(2, QuantisationStyle::SingleBit, 0.35, 0.0, 0.03, vec![]);

        upsert_vectors(&db, ns, doc_id, &[vi_mb, vi_sb1, vi_sb2]).await.unwrap();

        let store = DbVectorStore::new(&db, ns).await.unwrap();
        assert!(store.get_dense_entry(doc_id).await.is_some(), "MB entry exists before delete");
        assert_eq!(store.scan_sparse_cluster(1).await.len(), 1, "SB cluster-1 exists before delete");
        assert_eq!(store.scan_sparse_cluster(2).await.len(), 1, "SB cluster-2 exists before delete");

        delete_vector(&db, ns, doc_id).await.unwrap();

        assert!(store.get_dense_entry(doc_id).await.is_none(), "MB entry must be deleted");
        assert!(store.scan_sparse_cluster(1).await.is_empty(), "SB cluster-1 must be deleted");
        assert!(store.scan_sparse_cluster(2).await.is_empty(), "SB cluster-2 must be deleted");

        // Second delete is a no-op (sparse meta is gone).
        delete_vector(&db, ns, doc_id).await.unwrap();
    }

    /// Upserting with a new set replaces stale sparse entries; dense is overwritten.
    #[tokio::test]
    async fn test_dual_style_upsert_replaces_stale_sparse_entries_on_update() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        let ns = "docs";
        let doc_id = b"dual_update_doc";

        // First write: MB cluster 1, SB cluster 1, SB cluster 2.
        let vi_mb = VectorIndex::new(1, MULTI8, 0.3, 0.0, 0.01, vec![]);
        let vi_sb1 = VectorIndex::new(1, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
        let vi_sb2 = VectorIndex::new(2, QuantisationStyle::SingleBit, 0.35, 0.0, 0.03, vec![]);

        upsert_vectors(&db, ns, doc_id, &[vi_mb, vi_sb1, vi_sb2]).await.unwrap();

        let store = DbVectorStore::new(&db, ns).await.unwrap();
        assert!(store.get_dense_entry(doc_id).await.is_some());
        assert_eq!(store.scan_sparse_cluster(2).await.len(), 1);

        // Second write: MB updated (overwrites dense), SB now in cluster 3 only (cluster 2 stale).
        let vi_mb2 = VectorIndex::new(1, MULTI8, 0.25, 0.0, 0.01, vec![]);
        let vi_sb3 = VectorIndex::new(3, QuantisationStyle::SingleBit, 0.5, 0.0, 0.04, vec![]);

        upsert_vectors(&db, ns, doc_id, &[vi_mb2, vi_sb3]).await.unwrap();

        let dense = store.get_dense_entry(doc_id).await.unwrap();
        let vis = VectorIndex::list_from_bytes(&dense).unwrap();
        assert!((vis[0].addition_factor - 0.25).abs() < 1e-6, "dense must have updated addition_factor");

        assert!(store.scan_sparse_cluster(2).await.is_empty(), "stale SB cluster 2 removed");
        assert_eq!(store.scan_sparse_cluster(3).await.len(), 1, "new SB cluster 3 present");
        assert!(store.scan_sparse_cluster(1).await.is_empty(), "stale SB cluster 1 removed");
    }

    /// Multiple documents can share the same sparse cluster without interfering.
    #[tokio::test]
    async fn test_dual_style_multiple_docs_same_cluster_no_interference() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        let ns = "docs";
        let doc_a = b"doc_alpha";
        let doc_b = b"doc_beta";

        for doc in [&doc_a[..], &doc_b[..]] {
            let vi_mb = VectorIndex::new(1, MULTI8, 0.3, 0.0, 0.01, vec![]);
            let vi_sb = VectorIndex::new(1, QuantisationStyle::SingleBit, 0.4, 0.0, 0.02, vec![]);
            upsert_vectors(&db, ns, doc, &[vi_mb, vi_sb]).await.unwrap();
        }

        let store = DbVectorStore::new(&db, ns).await.unwrap();
        assert!(store.get_dense_entry(doc_a).await.is_some(), "doc_a dense entry exists");
        assert!(store.get_dense_entry(doc_b).await.is_some(), "doc_b dense entry exists");
        assert_eq!(store.scan_sparse_cluster(1).await.len(), 2, "both docs in SB cluster 1");

        // Delete doc_a; doc_b must still be present.
        delete_vector(&db, ns, doc_a).await.unwrap();

        assert!(store.get_dense_entry(doc_a).await.is_none(), "doc_a dense entry deleted");
        assert!(store.get_dense_entry(doc_b).await.is_some(), "doc_b dense entry preserved");
        assert_eq!(store.scan_sparse_cluster(1).await.len(), 1, "only doc_b remains in SB cluster 1");

        let sb_remaining = store.scan_sparse_cluster(1).await;
        assert_eq!(sb_remaining[0].0, doc_b, "remaining sparse entry belongs to doc_b");
    }
}

// ── Query embedding cache tests ───────────────────────────────────────────────

#[cfg(test)]
mod query_embedding_cache_tests {
    use super::*;
    use minnal_db::AsyncDb;
    use tempfile::TempDir;

    async fn open_db(dir: &TempDir) -> AsyncDb {
        AsyncDb::open(dir.path().to_owned()).await.unwrap()
    }

    /// TTL used by the cache round-trip tests.
    const TEST_TTL: Duration = DEFAULT_QUERY_EMBEDDING_CACHE_TTL;

    // ── Encoding helpers ──────────────────────────────────────────────────────

    #[test]
    fn test_f32_vec_list_roundtrip_single() {
        let embeddings = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
        let bytes = f32_vec_list_to_bytes(&embeddings);
        let decoded = bytes_to_f32_vec_list(&bytes, 4).unwrap();
        assert_eq!(decoded, embeddings);
    }

    #[test]
    fn test_f32_vec_list_roundtrip_multiple() {
        let embeddings = vec![vec![1.0f32, 2.0], vec![3.0f32, 4.0], vec![5.0f32, 6.0]];
        let bytes = f32_vec_list_to_bytes(&embeddings);
        let decoded = bytes_to_f32_vec_list(&bytes, 2).unwrap();
        assert_eq!(decoded, embeddings);
    }

    #[test]
    fn test_f32_vec_list_roundtrip_empty_list() {
        let embeddings: Vec<Vec<f32>> = vec![];
        let bytes = f32_vec_list_to_bytes(&embeddings);
        let decoded = bytes_to_f32_vec_list(&bytes, 4).unwrap();
        assert_eq!(decoded, embeddings);
    }

    #[test]
    fn test_f32_vec_list_rejects_wrong_dim() {
        let embeddings = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
        let bytes = f32_vec_list_to_bytes(&embeddings);
        // encoded as dim=4, but we decode claiming dim=3
        assert!(bytes_to_f32_vec_list(&bytes, 3).is_none());
    }

    #[test]
    fn test_f32_vec_list_rejects_too_short() {
        assert!(bytes_to_f32_vec_list(&[0, 1], 4).is_none());
    }

    #[test]
    fn test_f32_vec_list_rejects_truncated_payload() {
        // count=1 but only 8 bytes of payload instead of the required 4*4=16
        let mut bytes = 1u32.to_be_bytes().to_vec();
        bytes.extend_from_slice(&[0u8; 8]);
        assert!(bytes_to_f32_vec_list(&bytes, 4).is_none());
    }

    // ── Cache round-trip ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_cache_miss_returns_none() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        assert!(get_cached_query_embedding(&db, "unknown", 4, TEST_TTL).await.is_none());
    }

    #[tokio::test]
    async fn test_cache_roundtrip_dense_and_sparse() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        let dense = vec![0.1f32, 0.2, 0.3, 0.4];
        let sparse = vec![vec![0.5f32, 0.6, 0.7, 0.8], vec![0.9f32, 1.0, 1.1, 1.2]];
        put_cached_query_embedding(&db, "hello", &dense, &sparse, TEST_TTL).await;
        let (cached_dense, cached_sparse) = get_cached_query_embedding(&db, "hello", 4, TEST_TTL).await.unwrap();
        for (a, b) in cached_dense.iter().zip(dense.iter()) {
            assert!((a - b).abs() < 1e-6, "dense f32 mismatch after cache round-trip");
        }
        assert_eq!(cached_sparse.len(), 2);
        assert!((cached_sparse[0][0] - 0.5f32).abs() < 1e-6);
        assert!((cached_sparse[1][3] - 1.2f32).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_cache_roundtrip_empty_sparse() {
        // A whitespace-only query yields no chunks; only the dense vector is cached.
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        let dense = vec![1.0f32, 2.0];
        put_cached_query_embedding(&db, "dense-only", &dense, &[], TEST_TTL).await;
        let (cached_dense, cached_sparse) = get_cached_query_embedding(&db, "dense-only", 2, TEST_TTL).await.unwrap();
        assert_eq!(cached_dense, dense);
        assert!(cached_sparse.is_empty());
    }

    #[tokio::test]
    async fn test_cache_miss_on_dim_mismatch() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        let dense = vec![1.0f32, 2.0, 3.0, 4.0];
        put_cached_query_embedding(&db, "query", &dense, &[], TEST_TTL).await;
        // stored as dim=4; ask for dim=3 — should be a cache miss
        assert!(get_cached_query_embedding(&db, "query", 3, TEST_TTL).await.is_none());
    }

    #[tokio::test]
    async fn test_cache_different_queries_are_independent() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        let dense_a = vec![1.0f32, 0.0];
        let dense_b = vec![0.0f32, 1.0];
        put_cached_query_embedding(&db, "query-a", &dense_a, &[], TEST_TTL).await;
        put_cached_query_embedding(&db, "query-b", &dense_b, &[], TEST_TTL).await;
        let (cached_a, _) = get_cached_query_embedding(&db, "query-a", 2, TEST_TTL).await.unwrap();
        let (cached_b, _) = get_cached_query_embedding(&db, "query-b", 2, TEST_TTL).await.unwrap();
        assert!((cached_a[0] - 1.0f32).abs() < 1e-6);
        assert!((cached_b[1] - 1.0f32).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_clear_removes_all_entries() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        put_cached_query_embedding(&db, "query-a", &[1.0f32, 0.0], &[], TEST_TTL).await;
        put_cached_query_embedding(&db, "query-b", &[0.0f32, 1.0], &[], TEST_TTL).await;

        let cleared = clear_cached_query_embeddings(&db, TEST_TTL).await.unwrap();
        assert_eq!(cleared, 2);
        assert!(get_cached_query_embedding(&db, "query-a", 2, TEST_TTL).await.is_none());
        assert!(get_cached_query_embedding(&db, "query-b", 2, TEST_TTL).await.is_none());

        // Clearing an already-empty cache is a no-op that reports zero.
        assert_eq!(clear_cached_query_embeddings(&db, TEST_TTL).await.unwrap(), 0);
    }

    /// The cache must stay off the WAL: populating and clearing it should drive
    /// the *no-WAL* counters and add no WAL fsync to the query path. This pins
    /// the latency-motivated durability choice for the TTL cache namespace.
    #[tokio::test]
    async fn test_cache_writes_bypass_wal() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;

        // Baseline after namespace setup, so we only measure the cache writes.
        let before = db.ops_metrics();

        put_cached_query_embedding(&db, "q1", &[1.0f32, 0.0], &[], TEST_TTL).await;
        put_cached_query_embedding(&db, "q2", &[0.0f32, 1.0], &[], TEST_TTL).await;
        let cleared = clear_cached_query_embeddings(&db, TEST_TTL).await.unwrap();
        assert_eq!(cleared, 2);

        let after = db.ops_metrics();
        assert_eq!(after.no_wal_puts - before.no_wal_puts, 2, "two cache populates use put_no_wal");
        assert_eq!(after.no_wal_deletes - before.no_wal_deletes, 2, "two cache clears use delete_no_wal");
        assert_eq!(after.puts - before.puts, 0, "no WAL-backed puts from the cache");
        assert_eq!(after.deletes - before.deletes, 0, "no WAL-backed deletes from the cache");
        assert_eq!(after.wal_fsyncs - before.wal_fsyncs, 0, "cache writes add no WAL fsync");
    }

    /// The cache is keyed by query text **only** — it carries no chunk-config
    /// version, so a change to `window_size` / `sliding_size` does not invalidate
    /// it automatically (the documented limitation; remediation is an explicit
    /// `clear_cached_query_embeddings` + corpus re-index, or TTL expiry). This pins
    /// the remediation half: once the same query is re-embedded under the new
    /// chunking, the fresh entry overwrites the stale one (different chunk count).
    #[tokio::test]
    async fn test_cache_reembed_overwrites_with_new_chunking() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir).await;
        let dense = vec![1.0f32, 0.0];

        // Old chunking: 2 sparse chunks.
        let old_sparse = vec![vec![0.1f32, 0.2], vec![0.3f32, 0.4]];
        put_cached_query_embedding(&db, "q", &dense, &old_sparse, TEST_TTL).await;
        let (_, cached) = get_cached_query_embedding(&db, "q", 2, TEST_TTL).await.unwrap();
        assert_eq!(cached.len(), 2, "stale chunking is served until re-embedded (keyed by text only)");

        // New chunking for the SAME query text (e.g. after a window_size change +
        // re-embed): 4 sparse chunks. The fresh put overwrites the stale entry.
        let new_sparse = vec![vec![0.1f32, 0.2], vec![0.3f32, 0.4], vec![0.5f32, 0.6], vec![0.7f32, 0.8]];
        put_cached_query_embedding(&db, "q", &dense, &new_sparse, TEST_TTL).await;
        let (_, cached) = get_cached_query_embedding(&db, "q", 2, TEST_TTL).await.unwrap();
        assert_eq!(cached, new_sparse, "re-embedding the same query must overwrite with the new chunking");
    }
}
