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
use crate::semantic_search::ClusterIndex;
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
}

pub(super) fn vec_reindex_path(db_path: &Path, ns_id: u32) -> PathBuf {
    crate::db::layout::namespace_index_dir(&crate::db::layout::index_root(db_path), ns_id).join("vector_reindex.json")
}

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
#[cfg(feature = "semantic-search")]
pub struct SemanticSearchContext {
    /// Embedding service configuration (URL, model, dimensions).
    pub config: SemanticSearchConfig,
    /// IVF cluster centroids, probed by exact nearest-centroid distance.
    pub cluster_index: Arc<ClusterIndex>,
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
