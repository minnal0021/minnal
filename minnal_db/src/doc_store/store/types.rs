//! Types surfaced by the document store: document IDs, index-build progress,
//! and the semantic-search context.
//!
//! Re-exported from the parent module, so these stay at `doc_store::store::*`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::FieldId;
use crate::doc_store::error::DocStoreError;
use crate::doc_store::index_observer::InMemoryProgress;
use crate::doc_store::key::StrKey;
use crate::doc_store::schema::KeyType;
#[cfg(feature = "semantic-search")]
use crate::doc_store::vector_settings::{SearchSettings, SearchSpec, SeededFrom, VectorIndexSettings};
#[cfg(feature = "semantic-search")]
use crate::semantic_search::ClusterIndex;
#[cfg(feature = "semantic-search")]
use crate::semantic_search::NamespaceIvf;
#[cfg(feature = "semantic-search")]
use crate::semantic_search::service::SemanticSearchConfig;

// ── ID type ───────────────────────────────────────────────────────────────────

/// A document identifier, typed to match the [`KeyType`] of the store.
///
/// Integer keys are stored in big-endian byte order and string keys verbatim,
/// so in both cases lexicographic byte order corresponds to the natural order
/// of the ID — which is what makes range scans over IDs work.
///
/// `DocId` is `Copy`: [`StrKey`] stores its bytes inline rather than in a
/// `String`, so passing an ID by value never allocates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DocId {
    /// 128-bit UUID represented as a `u128`.
    Uuid(u128),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Unsigned 128-bit integer.
    U128(u128),
    /// UTF-8 string key, validated to at most
    /// [`MAX_STR_KEY_LEN`](crate::doc_store::key::MAX_STR_KEY_LEN) bytes.
    Str(StrKey),
}

impl DocId {
    /// Serialize this ID to its raw database key bytes — big-endian for the
    /// integer types, the UTF-8 bytes themselves for [`DocId::Str`].
    pub fn to_bytes(self) -> Vec<u8> {
        match self {
            DocId::Uuid(v) | DocId::U128(v) => v.to_be_bytes().to_vec(),
            DocId::U64(v) => v.to_be_bytes().to_vec(),
            DocId::Str(k) => k.as_bytes().to_vec(),
        }
    }

    /// The store [`KeyType`] this ID belongs to.
    pub fn key_type(self) -> KeyType {
        match self {
            DocId::Uuid(_) => KeyType::Uuid,
            DocId::U64(_) => KeyType::U64,
            DocId::U128(_) => KeyType::U128,
            DocId::Str(_) => KeyType::Str,
        }
    }

    /// Deserialize bytes back to a `DocId` given the store's [`KeyType`].
    pub fn from_bytes(bytes: &[u8], key_type: KeyType) -> Result<Self, DocStoreError> {
        match key_type {
            KeyType::Str => Ok(DocId::Str(StrKey::from_bytes(bytes)?)),
            KeyType::Uuid => {
                let arr: [u8; 16] = bytes
                    .try_into()
                    .map_err(|_| DocStoreError::InvalidId(format!("expected 16 bytes for UUID key, got {}", bytes.len())))?;
                Ok(DocId::Uuid(u128::from_be_bytes(arr)))
            }
            KeyType::U64 => {
                let arr: [u8; 8] = bytes
                    .try_into()
                    .map_err(|_| DocStoreError::InvalidId(format!("expected 8 bytes for u64 key, got {}", bytes.len())))?;
                Ok(DocId::U64(u64::from_be_bytes(arr)))
            }
            KeyType::U128 => {
                let arr: [u8; 16] = bytes
                    .try_into()
                    .map_err(|_| DocStoreError::InvalidId(format!("expected 16 bytes for u128 key, got {}", bytes.len())))?;
                Ok(DocId::U128(u128::from_be_bytes(arr)))
            }
        }
    }
}

// ── Index build progress ──────────────────────────────────────────────────────

/// Persistent state for a background index build, written to
/// `{db_path}/index/{ns_id}/{field_id}/build_progress.json`.
///
/// Survives server restarts — on startup the store uses this to resume
/// interrupted builds instead of restarting from scratch.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiskBuildProgress {
    /// `"in_progress"`, `"complete"`, or `"failed"`.
    pub status: String,
    /// Total document count (0 until the initial scan finishes).
    pub total: u64,
    /// Documents processed so far.
    pub indexed: u64,
    /// Hex-encoded bytes of the last successfully processed key.
    /// `None` if no key has been processed yet.
    pub last_key_hex: Option<String>,
    /// Error message if `status == "failed"`.
    pub error: Option<String>,
}

/// How much of a document store's row map is spent on deleted documents
/// ([`DocStore::rowmap_stats`](super::DocStore::rowmap_stats)).
///
/// The row map gives every document a dense row ID for the field indexes and
/// never frees one, so it grows with every document ever written. `dead_ids`
/// counts the IDs of documents that no longer exist; `reindex-all` frees them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct RowMapStats {
    /// Row IDs allocated: every document ever indexed, live or deleted.
    pub ids_allocated: u64,
    /// Documents in the store now.
    pub live_docs: u64,
    /// `ids_allocated - live_docs`: IDs held by deleted documents.
    pub dead_ids: u64,
    /// Bytes of the row map's files on disk.
    pub bytes_on_disk: u64,
}

/// Path to the build-progress file for `(ns_id, field_id)`.
///
/// The directory comes from [`crate::db::layout`] — the engine owns where index
/// files live; the doc store only owns the filename it puts there.
pub(super) fn build_progress_path(db_path: &Path, ns_id: u32, field_id: FieldId) -> PathBuf {
    crate::db::layout::namespace_index_dir(&crate::db::layout::index_root(db_path), ns_id)
        .join(field_id.to_string())
        .join("build_progress.json")
}

/// Read the persisted build progress for a field.  Returns `None` if the file
/// does not exist or cannot be parsed.
pub(super) fn read_disk_progress(path: &Path) -> Option<DiskBuildProgress> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

// ── Vector-index reindex ─────────────────────────────────────────────────────

/// Persistent record for one `index_all` reindex, written to
/// `{db_path}/index/{ns_id}/vector_reindex.json`.
///
/// A reindex is the unit of work created by a single `index_all` call: it
/// tracks the total documents enqueued, how many have been indexed, and the
/// lifecycle status.  The record survives restarts so that the progress API can
/// show historical reindexs even after the server restarts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VecReindexProgress {
    /// `"running"`, `"complete"`, or `"failed"`.
    pub status: String,
    /// Unix milliseconds when `index_all` was called.
    pub started_at_ms: u64,
    /// Unix milliseconds when the reindex finished (terminal states only).
    pub completed_at_ms: Option<u64>,
    /// Number of documents enqueued by this reindex.
    pub total_enqueued: usize,
    /// Number of exhausted entries cleared before enqueueing.
    pub exhausted_cleared: usize,
    /// Error message when `status == "failed"`.
    pub error: Option<String>,
    /// Whether every document was enqueued. A `"running"` record without it, and
    /// with no reindex enqueueing in this process, was cut short by a crash.
    #[serde(default)]
    pub enqueue_done: bool,
}

#[cfg(feature = "semantic-search")]
pub(super) fn vec_reindex_path(db_path: &Path, ns_id: u32) -> PathBuf {
    crate::db::layout::namespace_index_dir(&crate::db::layout::index_root(db_path), ns_id).join("vector_reindex.json")
}

#[cfg(feature = "semantic-search")]
pub(super) fn read_vec_reindex(path: &Path) -> Option<VecReindexProgress> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

#[cfg(feature = "semantic-search")]
pub(super) fn write_vec_reindex(path: &Path, reindex: &VecReindexProgress) {
    let tmp = path.with_extension("tmp");
    if let Ok(bytes) = serde_json::to_vec(reindex) {
        let _ = std::fs::write(&tmp, &bytes);
        let _ = std::fs::rename(&tmp, path);
    }
}

/// A snapshot of the progress of a background index build.
#[derive(Debug, Clone)]
pub struct IndexBuildProgress {
    /// Total number of documents to be indexed (0 until the scan begins).
    pub total: u64,
    /// Number of documents processed so far.
    pub indexed: u64,
    /// `true` once the build has finished (successfully or with an error).
    pub done: bool,
    /// Non-`None` if the build failed.
    pub error: Option<String>,
}

/// Handle returned by [`DocStore::add_index`] or [`DocStore::resume_pending_builds`].
///
/// Use [`progress`] to poll status or [`wait`] to block until complete.
///
/// [`progress`]: IndexBuildHandle::progress
/// [`wait`]: IndexBuildHandle::wait
pub struct IndexBuildHandle {
    /// The namespace this build belongs to.
    pub namespace: String,
    /// The field being indexed.
    pub field: String,
    /// Live progress counters shared with the observer inside the build task.
    pub mem: Arc<InMemoryProgress>,
    /// `pub(super)` rather than private: the build is spawned from a sibling
    /// module, which before the split was the same file.
    pub(super) task: tokio::task::JoinHandle<Result<(), DocStoreError>>,
}

impl IndexBuildHandle {
    /// Return a snapshot of the current build progress.
    pub fn progress(&self) -> IndexBuildProgress {
        IndexBuildProgress {
            total: self.mem.total.load(Ordering::Relaxed),
            indexed: self.mem.indexed.load(Ordering::Relaxed),
            done: self.mem.done.load(Ordering::Relaxed),
            error: self.mem.error.lock().unwrap().clone(),
        }
    }

    /// Await the build task and return its result.
    pub async fn wait(self) -> Result<(), DocStoreError> {
        self.task.await.map_err(|e| DocStoreError::BuildFailed(e.to_string()))?
    }
}

// ── SemanticSearchContext ─────────────────────────────────────────────────────

/// Semantic-search configuration attached to a [`DocStore`].
///
/// When present, writes to namespaces with `semantic_search_enabled = true`
/// enqueue a pending embedding job instead of calling the embedding service
/// inline.  A background `VecIndexWorker` processes the queue
/// asynchronously, making the write path independent of embedding service
/// availability.
///
/// Construct once at startup and attach via [`DocStore::with_semantic_search`].
///
/// It holds what is engine-wide: the embedding service settings and one set of
/// IVF cluster centroids per supported model, used only to **seed** a namespace
/// when it first enables semantic search ([`seed_namespace`]). After that a
/// namespace runs on its own stored centres and postings, loaded once into a
/// registry. Everything that shapes a namespace's index comes from its schema;
/// [`for_namespace`] combines the two into the per-call [`SemanticSearchConfig`]
/// and the namespace's [`NamespaceIvf`].
///
/// [`seed_namespace`]: SemanticSearchContext::seed_namespace
///
/// [`for_namespace`]: SemanticSearchContext::for_namespace
#[cfg(feature = "semantic-search")]
pub struct SemanticSearchContext {
    /// Embedding service settings: URL, timeouts and the query-cache TTL. Its
    /// per-namespace fields are ignored; [`for_namespace`](Self::for_namespace)
    /// overwrites them from the namespace's settings.
    pub config: SemanticSearchConfig,
    /// Seed sources: IVF cluster centroids per supported model (lower-cased name).
    pub cluster_indexes: std::collections::HashMap<String, Arc<ClusterIndex>>,
    /// The file each seed source was loaded from, when known (recorded in a
    /// namespace's `seeded_from`).
    seed_files: std::collections::HashMap<String, std::path::PathBuf>,
    /// Each namespace's partition, loaded once from its stores. Keyed by name and
    /// `ns_id`, so a namespace dropped and recreated under the same name never
    /// reuses the old one.
    ivfs: parking_lot::RwLock<std::collections::HashMap<(String, u32), Arc<NamespaceIvf>>>,
}

/// What one namespace's semantic operations run with: the per-call config built
/// from its settings, and its model's centroids.
#[cfg(feature = "semantic-search")]
pub struct NamespaceSemantics {
    /// Service settings plus the namespace's model, dimension, chunking, code
    /// widths and search defaults.
    pub config: SemanticSearchConfig,
    /// The namespace's own partition (its centres, postings and rotation).
    pub ivf: Arc<NamespaceIvf>,
}

#[cfg(feature = "semantic-search")]
impl SemanticSearchContext {
    /// A context with the given service settings and centroids per model
    /// (names are lower-cased).
    pub fn new(config: SemanticSearchConfig, cluster_indexes: impl IntoIterator<Item = (String, Arc<ClusterIndex>)>) -> Self {
        Self {
            config,
            cluster_indexes: cluster_indexes.into_iter().map(|(m, c)| (m.to_lowercase(), c)).collect(),
            seed_files: std::collections::HashMap::new(),
            ivfs: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Record the file `model`'s seed source was loaded from.
    pub fn with_seed_file(mut self, model: &str, path: impl Into<std::path::PathBuf>) -> Self {
        self.seed_files.insert(model.to_lowercase(), path.into());
        self
    }

    /// Seed `namespace`'s centres and postings from its model's centroid set,
    /// unless it already holds a complete partition (a re-enable after a
    /// vector-index drop), and return the record for its schema. Durable on
    /// return.
    pub async fn seed_namespace(&self, db: &crate::AsyncDb, namespace: &str, settings: &VectorIndexSettings) -> Result<SeededFrom, DocStoreError> {
        let source = self.cluster_index_for(settings)?;
        let centroids: std::collections::HashMap<u32, Vec<f32>> = source.clusters.iter().map(|(&id, c)| (id, c.centroid.clone())).collect();
        if let Some(existing) = &settings.seeded_from
            && crate::vector_kv::ivf_is_seeded(db, namespace).await?
        {
            return Ok(existing.clone());
        }
        crate::vector_kv::seed_ivf(db, namespace, &centroids).await?;
        self.ivfs.write().retain(|(name, _), _| name != namespace);
        let (file, digest) = match self.seed_files.get(&settings.embedding_model) {
            Some(path) => {
                let bytes = std::fs::read(path).map_err(DocStoreError::Io)?;
                (path.display().to_string(), mm3h::murmurhash3_128(&bytes))
            }
            None => {
                let mut ids: Vec<&u32> = centroids.keys().collect();
                ids.sort_unstable();
                let bytes: Vec<u8> = ids
                    .into_iter()
                    .flat_map(|id| id.to_be_bytes().into_iter().chain(centroids[id].iter().flat_map(|x| x.to_le_bytes())))
                    .collect();
                ("<in-memory>".to_string(), mm3h::murmurhash3_128(&bytes))
            }
        };
        Ok(SeededFrom {
            file,
            murmur3_128: format!("{digest:032x}"),
            centres: centroids.len() as u32,
        })
    }

    /// Forget every loaded partition of `namespace` (it was dropped).
    pub fn evict(&self, namespace: &str) {
        self.ivfs.write().retain(|(name, _), _| name != namespace);
    }

    /// Apply one document's [`PostingDelta`] to `namespace`'s entry counts, if
    /// its partition is loaded. When it is not, there is nothing to update: the
    /// load counts from the store.
    ///
    /// [`PostingDelta`]: crate::semantic_search::PostingDelta
    pub fn apply_posting_delta(&self, namespace: &str, delta: &crate::semantic_search::PostingDelta) {
        if delta.added.is_empty() && delta.removed.is_empty() {
            return;
        }
        for ((name, _), ivf) in self.ivfs.read().iter() {
            if name == namespace {
                ivf.apply_delta(delta);
            }
        }
    }

    /// Check that `settings` can run here: its model has centroids, and they
    /// have its dimension.
    pub fn check_model(&self, settings: &VectorIndexSettings) -> Result<(), DocStoreError> {
        self.cluster_index_for(settings).map(|_| ())
    }

    fn cluster_index_for(&self, settings: &VectorIndexSettings) -> Result<&Arc<ClusterIndex>, DocStoreError> {
        let index = self
            .cluster_indexes
            .get(&settings.embedding_model)
            .ok_or_else(|| DocStoreError::UnsupportedEmbeddingModel {
                model: settings.embedding_model.clone(),
                supported: {
                    let mut m: Vec<&str> = self.cluster_indexes.keys().map(String::as_str).collect();
                    m.sort_unstable();
                    m.join(", ")
                },
            })?;
        if index.dim() != settings.embedding_dim as usize {
            return Err(DocStoreError::EmbeddingDimMismatch {
                model: settings.embedding_model.clone(),
                dim: settings.embedding_dim,
                centroid_dim: index.dim(),
            });
        }
        Ok(index)
    }

    /// The per-call config and partition for `namespace` (`ns_id`) with `settings`,
    /// loading the partition from its stores on first use.
    pub async fn for_namespace(
        &self,
        db: &crate::AsyncDb,
        namespace: &str,
        ns_id: u32,
        settings: &VectorIndexSettings,
    ) -> Result<NamespaceSemantics, DocStoreError> {
        let key = (namespace.to_owned(), ns_id);
        let cached = self.ivfs.read().get(&key).cloned();
        let ivf = match cached {
            Some(ivf) => ivf,
            None => {
                let ivf =
                    crate::vector_kv::load_ivf(db, namespace, settings.rotation_seed)
                        .await?
                        .ok_or_else(|| DocStoreError::VectorIndexNotSeeded {
                            namespace: namespace.to_owned(),
                        })?;
                // Two first uses can race to load; keep whichever landed first,
                // so every count update goes to the one partition in use.
                Arc::clone(self.ivfs.write().entry(key).or_insert_with(|| Arc::new(ivf)))
            }
        };
        if crate::semantic_search::IvfLayout::dim(ivf.as_ref()) != settings.embedding_dim as usize {
            return Err(DocStoreError::EmbeddingDimMismatch {
                model: settings.embedding_model.clone(),
                dim: settings.embedding_dim,
                centroid_dim: crate::semantic_search::IvfLayout::dim(ivf.as_ref()),
            });
        }
        let config = SemanticSearchConfig {
            model_name: settings.embedding_model.clone(),
            embedding_dim: settings.embedding_dim as usize,
            window_size: settings.window_size as usize,
            sliding_size: settings.sliding_size as usize,
            number_of_bits_for_dense_quantisation: settings.pass2_bits as usize,
            probe: probe_settings(&settings.search),
            first_pass_sparse_search_top_k: settings.search.first_pass_top_k as usize,
            top_k_results: settings.search.top_k as usize,
            ..self.config.clone()
        };
        Ok(NamespaceSemantics { config, ivf })
    }
}

#[cfg(feature = "semantic-search")]
impl NamespaceSemantics {
    /// This namespace's config with one request's search overrides applied
    /// (validated against the same ranges as the schema).
    pub fn with_overrides(mut self, overrides: &SearchSpec) -> Result<Self, DocStoreError> {
        let base = SearchSettings {
            probe_budget_entries: self.config.probe.budget_entries as u32,
            min_probes: self.config.probe.min_probes as u32,
            max_probes: self.config.probe.max_probes as u32,
            first_pass_top_k: self.config.first_pass_sparse_search_top_k as u32,
            top_k: self.config.top_k_results as u32,
        };
        let s = overrides.apply(base)?;
        self.config.probe = probe_settings(&s);
        self.config.first_pass_sparse_search_top_k = s.first_pass_top_k as usize;
        self.config.top_k_results = s.top_k as usize;
        Ok(self)
    }
}

// ── ReindexStats ─────────────────────────────────────────────────────────────

/// Outcome of a single-document vector reindex
/// ([`DocStore::reindex_doc_vector`] / [`DocStore::kv_reindex_doc_vector`]).
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

/// Result of a [`DocStore::index_all`] call.
#[derive(Debug, Clone, Copy)]
pub struct ReindexStats {
    /// Number of exhausted queue entries (retry_count ≥ max_retries) that were
    /// removed before re-enqueueing.
    pub exhausted_cleared: usize,
    /// Number of documents enqueued for re-embedding.
    pub enqueued: usize,
}

/// The probe settings a namespace's search settings describe.
#[cfg(feature = "semantic-search")]
fn probe_settings(s: &SearchSettings) -> crate::semantic_search::ProbeSettings {
    crate::semantic_search::ProbeSettings {
        budget_entries: u64::from(s.probe_budget_entries),
        min_probes: s.min_probes as usize,
        max_probes: s.max_probes as usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc_store::store::test_support::*;

    // ── DocId serialization ─────────────────────────────────────────────────

    #[test]
    fn test_doc_id_u64_roundtrip() {
        let id = DocId::U64(12345);
        let bytes = id.to_bytes();
        let restored = DocId::from_bytes(&bytes, KeyType::U64).unwrap();
        assert_eq!(id, restored);
    }

    #[test]
    fn test_doc_id_u128_roundtrip() {
        let id = DocId::U128(u128::MAX / 2);
        let bytes = id.to_bytes();
        let restored = DocId::from_bytes(&bytes, KeyType::U128).unwrap();
        assert_eq!(id, restored);
    }

    #[test]
    fn test_doc_id_uuid_roundtrip() {
        let id = DocId::Uuid(0xdeadbeef_cafebabe_12345678_9abcdef0);
        let bytes = id.to_bytes();
        let restored = DocId::from_bytes(&bytes, KeyType::Uuid).unwrap();
        assert_eq!(id, restored);
    }

    #[test]
    fn test_doc_id_ordering() {
        // big-endian encoding means byte-level ordering == numeric ordering
        let ids: Vec<DocId> = (0u64..5).map(DocId::U64).collect();
        let encoded: Vec<Vec<u8>> = ids.iter().map(|id| id.to_bytes()).collect();
        let sorted = {
            let mut c = encoded.clone();
            c.sort();
            c
        };
        assert_eq!(encoded, sorted);
    }

    #[test]
    fn test_invalid_key_size_rejected() {
        assert!(DocId::from_bytes(&[0u8; 3], KeyType::U64).is_err());
        assert!(DocId::from_bytes(&[0u8; 5], KeyType::U128).is_err());
    }

    #[test]
    fn test_doc_id_str_roundtrip() {
        let id = DocId::Str(StrKey::new("acme-corp-2026").unwrap());
        let bytes = id.to_bytes();
        assert_eq!(bytes, b"acme-corp-2026", "a str key is stored verbatim, not encoded");
        let restored = DocId::from_bytes(&bytes, KeyType::Str).unwrap();
        assert_eq!(id, restored);
    }

    /// The same property the integer types get from big-endian encoding: the
    /// order of the stored key bytes is the order of the IDs, which is what
    /// makes `scan_range` over document IDs meaningful.
    #[test]
    fn test_doc_id_str_ordering_matches_byte_ordering() {
        let ids: Vec<DocId> = ["aa", "acme", "acme-corp", "b", "z"]
            .into_iter()
            .map(|s| DocId::Str(StrKey::new(s).unwrap()))
            .collect();

        let encoded: Vec<Vec<u8>> = ids.iter().map(|id| id.to_bytes()).collect();
        let sorted = {
            let mut c = encoded.clone();
            c.sort();
            c
        };
        assert_eq!(encoded, sorted, "str ids must sort lexicographically by their key bytes");

        // And the `DocId` ordering itself agrees — a derived `Ord` on the
        // inline buffer would compare length first and put "z" before "aa".
        let mut by_doc_id = ids.clone();
        by_doc_id.sort();
        assert_eq!(by_doc_id, ids);
    }

    #[test]
    fn test_invalid_str_key_rejected() {
        // Over the cap, empty, and non-UTF-8 all fail to decode rather than
        // producing a truncated or lossy id.
        assert!(DocId::from_bytes(&[b'x'; crate::doc_store::key::MAX_STR_KEY_LEN + 1], KeyType::Str).is_err());
        assert!(DocId::from_bytes(&[], KeyType::Str).is_err());
        assert!(DocId::from_bytes(&[0xff, 0xfe], KeyType::Str).is_err());
    }
}
