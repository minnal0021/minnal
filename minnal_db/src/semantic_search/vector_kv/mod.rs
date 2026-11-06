//! Vector KV store integration for semantic-search-enabled namespaces.
//!
//! Each document namespace that has `semantic_search_enabled = true` gets these
//! companion minnal_db namespaces, plus a maintenance journal, managed entirely by
//! this module. [`create_vector_namespaces`] creates them, when the namespace is
//! seeded; everything else resolves them without creating, so a write that
//! outlives a drop cannot bring them back.
//!
//! | Internal namespace        | Key                            | Value                                    |
//! |---------------------------|--------------------------------|------------------------------------------|
//! | `{ns}_sparse_vector`      | `cluster_id (4B BE) ‖ doc_id`  | rkyv `Vec<VectorIndex>` (chunk codes: SingleBit or BitPlanes) |
//! | `{ns}_sparse_vector_meta` | `doc_id`                       | `count (2B BE u16) ‖ [cluster_id (4B BE)]×N` |
//! | `{ns}_dense_vector`       | `doc_id`                       | rkyv `Vec<VectorIndex>` (MultiBit only)  |
//! | `{ns}_ivf_centres`        | `centre_id (4B BE)`            | `f32 LE × D` (seeded once, append-only)  |
//! | `{ns}_ivf_postings`       | `posting_id (4B BE)`           | `centre_id ‖ state (1B) ‖ routing f32 LE × D` |
//!
//! The sparse namespace stores the chunk codes (the namespace's Pass-1 width: 1, 2 or 4 bits
//! per dimension) keyed by IVF cluster, enabling
//! fast first-pass scans by cluster prefix.  The meta namespace tracks which clusters each
//! document's chunks were assigned to, used only by delete/upsert to clean up stale keys.
//! The dense namespace stores the full-precision multi-bit embedding keyed directly by
//! document ID, enabling O(1) lookups in the re-ranking pass without a meta indirection.
//!
//! Query-embedding caching uses a single system-wide TTL namespace
//! (`system_qemb_cache`) shared across all doc-store namespaces — see
//! `SYSTEM_QUERY_EMB_CACHE_NS`.
//!
//! # Failure & idempotency
//!
//! No transactions span the three namespaces.  Any partial failure leaves the
//! store in an inconsistent state, but every operation is idempotent: a plain
//! retry always converges to a correct final state.
//!
//! ## Write (upsert) sequence
//! 1. Caller queues the embed ([`enqueue_embed`]), then writes the document
//!    (see *Queue first* below).
//! 2. The worker obtains fresh `VectorIndex` entries from the embedding service.
//! 3. [`upsert_vectors`] is called:
//!    a. Sparse: reads old cluster IDs from `_sparse_vector_meta`, deletes stale keys.
//!    b. Sparse: writes new composite keys to `_sparse_vector`, updates meta.
//!    c. Dense: overwrites `doc_id` key in `_dense_vector`.
//!
//! ## Delete sequence
//! 1. Caller writes a `Clear` tombstone to the queue ([`clear_vectors`]).
//! 2. [`delete_vector`] is called:
//!    a. Reads cluster IDs from `_sparse_vector_meta`, deletes all sparse composite keys.
//!    b. Deletes `doc_id` from `_dense_vector`.
//! 3. Caller deletes the document from the doc store.
//!
//! ## Queue first
//!
//! The queue entry and the document are separate writes, so every writer writes
//! the **queue entry first** and the document second. A crash between them leaves
//! a queue entry describing a write that never landed, never a changed document
//! with nothing queued. Before acting on an entry the worker compares it with the
//! document ([`check_entry_against_document`]), but only when no write to that
//! document is in progress (the document store tracks them, without locking:
//! `doc_store::store::InFlightWrites`). An entry that does not describe the
//! document is then a write cut short, or a re-index entry from an older
//! snapshot, and is rewritten to describe it. So the queue alone lists every
//! document whose vectors may be out of date, and recovery after a crash is
//! draining it; nothing scans the documents.

use crate::semantic_search::index::vector_index::{QuantisationStyle, SparseScan, VectorKvStore};
use crate::semantic_search::{PostingDelta, VectorIndex, composite_key};
use crate::{AsyncDb, AsyncNamespace};

/// A point where the crash audit can stop the process: before a write.
macro_rules! crash_point {
    () => {
        #[cfg(test)]
        $crate::semantic_search::vector_kv::crash_point::check()?;
    };
}

#[cfg(test)]
pub(crate) mod crash_point;
mod partition;
mod query_cache;
mod queue;
#[cfg(test)]
mod tests;

pub use partition::*;
pub use query_cache::*;
pub use queue::*;

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

/// Internal namespace for a namespace's centres: `centre_id (4B BE) → f32 LE × D`.
/// What codes are encoded against; append-only, and removed with the vector index.
pub fn ivf_centres_ns(namespace: &str) -> String {
    format!("{}_ivf_centres", namespace)
}

/// Internal namespace for a namespace's postings:
/// `posting_id (4B BE) → centre_id (4B BE) ‖ state (1B) ‖ routing centroid f32 LE × D`.
pub fn ivf_postings_ns(namespace: &str) -> String {
    format!("{}_ivf_postings", namespace)
}

/// Internal namespace for a namespace's maintenance journal (design doc M3a):
/// `op_id (8B BE) → record`, WAL-backed.
pub fn ivf_journal_ns(namespace: &str) -> String {
    format!("{}_ivf_journal", namespace)
}

/// The suffixes that mark a namespace as a companion of another
/// ([`VECTOR_COMPANION_SUFFIXES`](crate::db::namespace::VECTOR_COMPANION_SUFFIXES)).
pub const COMPANION_SUFFIXES: [&str; 6] = crate::db::namespace::VECTOR_COMPANION_SUFFIXES;

/// The companions removed when the vector index is dropped: all of them. The
/// partition is derived from the data since M3 (a namespace grows it by
/// splitting), so it goes with the data and re-enabling starts from the root.
pub const VECTOR_DATA_SUFFIXES: [&str; 6] = COMPANION_SUFFIXES;

/// Every companion namespace name of `namespace`.
pub fn companion_namespaces(namespace: &str) -> Vec<String> {
    COMPANION_SUFFIXES.iter().map(|suffix| format!("{namespace}{suffix}")).collect()
}

/// Create every companion namespace of `namespace`; those that exist are left
/// as they are. The one place they are created: vector writes and partition
/// maintenance resolve them without creating, so once a drop has removed them,
/// nothing that outlives it can bring them back. Call it before writing vectors
/// to a raw namespace; [`init_ivf`] does.
pub async fn create_vector_namespaces(db: &AsyncDb, namespace: &str) -> Result<(), crate::KVError> {
    for name in companion_namespaces(namespace) {
        db.namespace(name).await?;
    }
    Ok(())
}

/// If `name` holds vector-index data of another namespace (see
/// [`VECTOR_DATA_SUFFIXES`]), the base namespace it belongs to.
pub fn vector_data_base(name: &str) -> Option<&str> {
    VECTOR_DATA_SUFFIXES.iter().find_map(|suffix| name.strip_suffix(suffix))
}

/// If `name` is a vector companion namespace, the base namespace it belongs to.
///
/// The inverse of [`sparse_vectors_ns`] and friends. It lives here so the naming
/// rule has exactly one owner: the admin API used to strip its own hard-coded
/// copy of these suffixes, so renaming one would have silently stopped companion
/// namespaces being recognised as such.
pub fn companion_base(name: &str) -> Option<&str> {
    COMPANION_SUFFIXES.iter().find_map(|suffix| name.strip_suffix(suffix))
}

// ── Sparse-meta encoding ──────────────────────────────────────────────────────
//
// Format: `text_hash (8B BE u64) ‖ count (2B BE u16) ‖ [cluster_id (4B BE)] × count`
//
// The meta is a document's index record: which clusters hold its chunks (the only
// way to find them again for a delete) and a hash of the text its vectors were
// embedded from.

/// Stable 64-bit FNV-1a hash of the text a document's vectors were embedded from.
/// Stored in the meta; never `std`'s `DefaultHasher`, whose output may change
/// between Rust releases and would make every stored index look stale.
pub fn text_hash(text: &str) -> u64 {
    text.bytes()
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

/// A decoded `{ns}_sparse_vector_meta` record: the hash of the text the vectors
/// came from, and each posting holding the document's chunks with how many.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SparseMeta {
    pub(crate) text_hash: u64,
    pub(crate) postings: Vec<(u32, u32)>,
}

impl SparseMeta {
    fn posting_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.postings.iter().map(|&(id, _)| id)
    }

    #[cfg(test)]
    fn cluster_ids(&self) -> Vec<u32> {
        self.posting_ids().collect()
    }
}

/// `text_hash (8B) ‖ n (2B) ‖ n × (posting_id (4B) ‖ chunks (4B))`, all big-endian.
/// The chunk counts make every posting's size exact (deletes have only the meta
/// to go by), which the split trigger needs. Four bytes, so a document with
/// more than 65,535 chunks in one posting is counted exactly.
pub(crate) fn encode_sparse_meta(text_hash: u64, postings: &[(u32, u32)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(10 + postings.len() * 8);
    out.extend_from_slice(&text_hash.to_be_bytes());
    out.extend_from_slice(&(postings.len() as u16).to_be_bytes());
    for &(id, chunks) in postings {
        out.extend_from_slice(&id.to_be_bytes());
        out.extend_from_slice(&chunks.to_be_bytes());
    }
    out
}

/// Decodes the current form and two earlier ones: two-byte chunk counts, and
/// no counts at all (`n × posting_id (4B)`; its counts read as 0). They cannot
/// be confused: `n` fixes the length of each.
pub(crate) fn decode_sparse_meta(bytes: &[u8]) -> Option<SparseMeta> {
    if bytes.len() < 10 {
        return None;
    }
    let text_hash = u64::from_be_bytes(bytes[..8].try_into().ok()?);
    let count = u16::from_be_bytes(bytes[8..10].try_into().ok()?) as usize;
    let width = match bytes.len() - 10 {
        l if l == count * 8 => 8,
        l if l == count * 6 => 6,
        l if l == count * 4 => 4,
        _ => return None,
    };
    let mut postings = Vec::with_capacity(count);
    for i in 0..count {
        let off = 10 + i * width;
        let id = u32::from_be_bytes(bytes[off..off + 4].try_into().ok()?);
        let chunks = match width {
            8 => u32::from_be_bytes(bytes[off + 4..off + 8].try_into().ok()?),
            6 => u32::from(u16::from_be_bytes(bytes[off + 4..off + 6].try_into().ok()?)),
            _ => 0,
        };
        postings.push((id, chunks));
    }
    Some(SparseMeta { text_hash, postings })
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Write the quantised vector index for a document to the vector KV store.
///
/// `vector_indexes` is the combined list of chunk (Pass-1: `SingleBit` or
/// `BitPlanes`) and whole-document (`MultiBit`) quantised entries
/// for the document, as returned by `embed_document`, and `text` is the text they
/// were embedded from (its hash goes in the meta).
///
/// **Write order is load-bearing (crash safety).** The meta is the only record of
/// which clusters hold the document's chunks, so it must list every key that can be
/// on disk, at every moment. The writes therefore go: (1) delete the stale cluster
/// keys (WAL), (2) write the new meta (**WAL-backed**, durable on return), (3) write
/// the new chunk keys and the dense entry (vector index WAL). Each namespace's
/// memtable flushes on its own schedule, so a meta outside the main WAL could be
/// lost while a key written after it was flushed: the key would then be listed nowhere, and a later
/// delete would leave it behind for good (crash-audit test
/// `delete_racing_a_reembed_survives_a_crash_anywhere`). The meta is one small
/// record per document, so its fsync is negligible beside the embedding call.
///
/// Chunk entries are written to `{ns}_sparse_vector` (keyed by cluster prefix),
/// with stale cluster keys cleaned up via `{ns}_sparse_vector_meta`.
/// The MultiBit entry (at most one) is written directly to `{ns}_dense_vector` by doc_id.
///
/// Each write is a standalone single-op write — the vector index is a derived
/// structure that the `VecIndexWorker` (re)builds idempotently from the queue,
/// so it needs no cross-namespace atomicity with the document write.
///
/// The quantised payloads skip the main WAL: they are bulky and reconstructable
/// by re-running the embed, so a per-chunk main-WAL fsync (and a second copy of
/// every payload there) is pure overhead. They are logged in the vector index
/// WAL instead (`put_logged`, no fsync), and are durable once the caller's
/// [`make_vector_writes_durable`] returns; the worker completes an entry only
/// after that. A crash before it can lose them, but the entry is then still
/// queued and redone. The stale-cluster `delete`s remain WAL-backed — they are
/// tiny key-only tombstones.
///
/// Returns how the document's entries moved between postings, for the
/// namespace's entry counts ([`crate::semantic_search::NamespaceIvf::apply_delta`]).
pub async fn upsert_vectors(
    db: &AsyncDb,
    namespace: &str,
    doc_id_bytes: &[u8],
    text: &str,
    vector_indexes: &[VectorIndex],
) -> Result<PostingDelta, crate::KVError> {
    if vector_indexes.is_empty() {
        return Ok(PostingDelta::default());
    }
    let _doc = lock_doc_vectors(namespace, doc_id_bytes).await;

    // Never create these: only `create_vector_namespaces` does, so a write that
    // outlives a drop cannot bring back what the drop removed. Absent means the
    // vector index is gone, and the work with it.
    let (Some(sparse_ns), Some(sparse_meta_ns), Some(dense_ns)) = (
        db.existing_namespace(sparse_vectors_ns(namespace)).await?,
        db.existing_namespace(sparse_vectors_meta_ns(namespace)).await?,
        db.existing_namespace(dense_vectors_ns(namespace)).await?,
    ) else {
        log::debug!("upsert_vectors: skipping '{namespace}': its vector namespaces do not exist");
        return Ok(PostingDelta::default());
    };

    // ── Sparse (Pass-1 chunk codes) and the meta ─────────────────────────────
    // The meta is written even when there are no chunks: it also records which
    // text the vectors came from.
    let mut new_cluster_groups: std::collections::BTreeMap<u32, Vec<VectorIndex>> = std::collections::BTreeMap::new();
    for vi in vector_indexes.iter().filter(|vi| vi.quantisation_style.is_pass1()) {
        new_cluster_groups.entry(vi.cluster_id).or_default().push(vi.clone());
    }
    let new_postings: Vec<(u32, u32)> = new_cluster_groups.iter().map(|(&id, v)| (id, v.len() as u32)).collect();

    // (1) Delete stale cluster keys no longer present (WAL).
    let mut old_postings = Vec::new();
    if let Some(old_bytes) = sparse_meta_ns.get(doc_id_bytes.to_vec()).await?
        && let Some(old) = decode_sparse_meta(&old_bytes)
    {
        for old_id in old.posting_ids() {
            if !new_cluster_groups.contains_key(&old_id) {
                crash_point!();
                sparse_ns.delete(composite_key::encode(old_id, doc_id_bytes)).await?;
            }
        }
        old_postings = old.postings;
    }
    // Every old entry is replaced or deleted, every new one written: exact counts
    // even where a posting keeps the document with a different number of chunks.
    let delta = PostingDelta {
        added: new_postings.clone(),
        removed: old_postings,
    };
    // (2) The new meta, durable before any key it lists is written (WAL).
    crash_point!();
    sparse_meta_ns
        .put(doc_id_bytes.to_vec(), encode_sparse_meta(text_hash(text), &new_postings))
        .await?;
    // (3) The new chunk keys (vector index WAL; durable at the batch's sync).
    for (&cluster_id, chunks) in &new_cluster_groups {
        let key = composite_key::encode(cluster_id, doc_id_bytes);
        crash_point!();
        sparse_ns.put_logged(key, VectorIndex::list_to_bytes(chunks)).await?;
    }

    // ── Dense (MultiBit) ─────────────────────────────────────────────────────
    let dense_vis: Vec<&VectorIndex> = vector_indexes
        .iter()
        .filter(|vi| matches!(vi.quantisation_style, QuantisationStyle::MultiBit { .. }))
        .collect();
    if !dense_vis.is_empty() {
        let owned: Vec<VectorIndex> = dense_vis.into_iter().cloned().collect();
        crash_point!();
        dense_ns.put_logged(doc_id_bytes.to_vec(), VectorIndex::list_to_bytes(&owned)).await?;
    }

    Ok(delta)
}

/// Number of stripes in [`lock_doc_vectors`]; a power of two. Partition
/// maintenance locks a batch of documents at once ([`lock_docs_vectors`]), and
/// every foreground write or delete whose document shares a stripe with the
/// batch waits for it: with 256 stripes a batch covered a large share of them.
/// At 4,096 a batch of 16 covers under 0.4%. A stripe is one small mutex.
const DOC_VECTOR_LOCK_STRIPES: usize = 4096;

/// Serialise every write to one document's vectors: [`upsert_vectors`] and
/// [`delete_vector`] each hold this across their whole body.
///
/// Both are multi-key read-modify-writes over the document's sparse cluster keys,
/// its cluster list (`{ns}_sparse_vector_meta`) and its dense entry, and the list
/// is the only way to find the cluster keys again. Interleaved, a delete could read
/// the old list, then delete the list the upsert had just replaced — leaving the
/// upsert's new cluster keys with nothing recording them. Every later clean-up
/// (the worker's `process_clear`, the next delete) finds keys through the list, so
/// those keys were orphaned for good: measured 260 orphans in 300 racing rounds.
/// A document delete racing the vector worker is exactly this shape. The database
/// has a single owning process (the directory lock), so an in-process lock
/// suffices; the stripes are shared across namespaces and databases, which only
/// costs an occasional false conflict.
pub(crate) async fn lock_doc_vectors(namespace: &str, doc_id_bytes: &[u8]) -> tokio::sync::MutexGuard<'static, ()> {
    DOC_VECTOR_STRIPES[doc_vector_stripe(namespace, doc_id_bytes)].lock().await
}

static DOC_VECTOR_STRIPES: std::sync::LazyLock<Vec<tokio::sync::Mutex<()>>> =
    std::sync::LazyLock::new(|| (0..DOC_VECTOR_LOCK_STRIPES).map(|_| tokio::sync::Mutex::new(())).collect());

fn doc_vector_stripe(namespace: &str, doc_id_bytes: &[u8]) -> usize {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    namespace.hash(&mut hasher);
    doc_id_bytes.hash(&mut hasher);
    (hasher.finish() as usize) & (DOC_VECTOR_LOCK_STRIPES - 1)
}

/// [`lock_doc_vectors`] for several documents at once (partition maintenance
/// works in batches). Two documents can share a stripe, so the stripes are
/// deduplicated, and they are taken in ascending order: two batches locking at
/// once then cannot deadlock, and a writer holding one stripe never waits on a
/// stripe while holding another. Don't take a document's lock again while
/// holding these: the stripes are not reentrant.
pub(crate) async fn lock_docs_vectors<'d>(namespace: &str, docs: impl IntoIterator<Item = &'d [u8]>) -> Vec<tokio::sync::MutexGuard<'static, ()>> {
    let mut stripes: Vec<usize> = docs.into_iter().map(|d| doc_vector_stripe(namespace, d)).collect();
    stripes.sort_unstable();
    stripes.dedup();
    let mut guards = Vec::with_capacity(stripes.len());
    for s in stripes {
        guards.push(DOC_VECTOR_STRIPES[s].lock().await);
    }
    guards
}

/// Delete the quantised vector for a document from the vector KV store.
///
/// No-op if no vector entry exists for `doc_id_bytes` (the document was never
/// indexed, or was already cleaned up). Returns the postings that lost an entry.
pub async fn delete_vector(db: &AsyncDb, namespace: &str, doc_id_bytes: &[u8]) -> Result<PostingDelta, crate::KVError> {
    let _doc = lock_doc_vectors(namespace, doc_id_bytes).await;
    let mut delta = PostingDelta::default();
    // A namespace that does not exist holds nothing to delete, which is the normal
    // case after a drop: the worker clears documents after its pass, outside any
    // partition's epoch.
    // Sparse: read cluster IDs from meta, delete each composite key, then delete meta.
    if let Some(sparse_meta_ns) = db.existing_namespace(sparse_vectors_meta_ns(namespace)).await?
        && let Some(meta_bytes) = sparse_meta_ns.get(doc_id_bytes.to_vec()).await?
    {
        if let Some(meta) = decode_sparse_meta(&meta_bytes) {
            if let Some(sparse_ns) = db.existing_namespace(sparse_vectors_ns(namespace)).await? {
                for cluster_id in meta.posting_ids() {
                    crash_point!();
                    sparse_ns.delete(composite_key::encode(cluster_id, doc_id_bytes)).await?;
                }
            }
            delta.removed = meta.postings;
        }
        crash_point!();
        sparse_meta_ns.delete(doc_id_bytes.to_vec()).await?;
    }

    // Dense: delete by doc_id directly (no meta needed).
    if let Some(dense_ns) = db.existing_namespace(dense_vectors_ns(namespace)).await? {
        crash_point!();
        dense_ns.delete(doc_id_bytes.to_vec()).await?;
    }

    Ok(delta)
}

/// Return `true` if a document has **any** vector-index state that clearing it
/// would have to remove: a sparse meta record, a dense entry, or a queue entry.
///
/// Sound as a "nothing to clear" test even while the worker is running: the
/// worker writes vectors before it removes the queue entry, so at every moment an
/// in-flight document shows either its queue entry or its vectors.
pub async fn has_any_vector_state(db: &AsyncDb, namespace: &str, doc_id_bytes: &[u8]) -> Result<bool, crate::KVError> {
    if get_queue_entry(db, namespace, doc_id_bytes).await?.is_some() {
        return Ok(true);
    }
    // Without creating them: a vector-index drop does not stop writes, so this
    // can run after it removed them.
    if let Some(sparse_meta_ns) = db.existing_namespace(sparse_vectors_meta_ns(namespace)).await?
        && sparse_meta_ns.get(doc_id_bytes.to_vec()).await?.is_some()
    {
        return Ok(true);
    }
    match db.existing_namespace(dense_vectors_ns(namespace)).await? {
        Some(dense_ns) => Ok(dense_ns.get(doc_id_bytes.to_vec()).await?.is_some()),
        None => Ok(false),
    }
}

// ── KV-backed VectorKvStore ───────────────────────────────────────────────────

/// A [`VectorKvStore`] implementation backed directly by minnal_db namespaces.
///
/// `scan_sparse_cluster` performs a 4-byte cluster prefix scan on `{ns}_sparse_vector`.
/// `get_dense_entry` performs a direct key lookup on `{ns}_dense_vector`.
///
/// Both namespaces are resolved once at construction time so that
/// `scan_sparse_cluster` and `get_dense_entry` never pay a namespace-lookup
/// cost per call. Resolved without creating them: a search that outlives its
/// store's drop must not bring them back, and what is absent reads as empty.
pub struct DbVectorStore {
    sparse_ns: Option<AsyncNamespace>,
    dense_ns: Option<AsyncNamespace>,
}

impl DbVectorStore {
    pub async fn new(db: &AsyncDb, namespace: &str) -> Result<Self, crate::KVError> {
        let sparse_ns = db.existing_namespace(sparse_vectors_ns(namespace)).await?;
        let dense_ns = db.existing_namespace(dense_vectors_ns(namespace)).await?;
        Ok(Self { sparse_ns, dense_ns })
    }
}

impl VectorKvStore for DbVectorStore {
    async fn scan_sparse_cluster(&self, cluster_id: u32) -> Result<Vec<(Vec<u8>, Vec<u8>)>, crate::KVError> {
        // Prefix = cluster_id (4B BE) — only entries for this cluster are returned.
        let prefix = cluster_id.to_be_bytes().to_vec();
        let Some(sparse_ns) = &self.sparse_ns else {
            return Ok(Vec::new());
        };
        let entries = sparse_ns.scan_prefix(prefix).await?;

        Ok(entries
            .into_iter()
            .filter_map(|(key, value)| {
                let (_, doc_id_bytes) = composite_key::decode(&key)?;
                Some((doc_id_bytes.to_vec(), value))
            })
            .collect())
    }

    async fn get_dense_entry(&self, doc_id_bytes: &[u8]) -> Result<Option<Vec<u8>>, crate::KVError> {
        match &self.dense_ns {
            Some(dense_ns) => dense_ns.get(doc_id_bytes.to_vec()).await,
            None => Ok(None),
        }
    }

    async fn get_dense_entries_batch(&self, doc_ids: &[Vec<u8>]) -> Result<Vec<Option<Vec<u8>>>, crate::KVError> {
        match &self.dense_ns {
            Some(dense_ns) => dense_ns.get_multiple(doc_ids.to_vec()).await,
            None => Ok(vec![None; doc_ids.len()]),
        }
    }

    async fn scan_sparse_clusters_batch(&self, cluster_ids: &[u32]) -> Result<SparseScan, crate::KVError> {
        match &self.sparse_ns {
            Some(sparse_ns) => Ok(SparseScan::new(sparse_ns.scan_prefixes_batch(cluster_ids.to_vec()).await?)),
            None => Ok(SparseScan::default()),
        }
    }
}
