//! Uniform progress and status types shared by both the field-index (RoaringBitmap)
//! and the vector-index (IVF + RaBitQ) subsystems.
//!
//! Previously each subsystem had its own ad-hoc progress representation:
//! - Field index: `DiskBuildProgress` with string statuses + `IndexBuildProgress`
//! - Vector index: derived live from queue depth + `{ns}_sparse_vector_meta` count
//!
//! This module provides a single set of types that both systems populate so the
//! REST API can return a consistent shape regardless of index kind.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use serde::{Deserialize, Serialize};

use crate::FieldId;
use crate::doc_store::error::DocStoreError;
use crate::doc_store::indexing::observer::InMemoryProgress;

use crate::doc_store::schema::IndexKind;

// ── BuildStatus ───────────────────────────────────────────────────────────────

/// Lifecycle status of an index build or vector-index reindex.
///
/// Stored as a snake_case string in JSON and in the on-disk progress files so
/// that old files remain readable after a version upgrade that adds new variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuildStatus {
    /// Queued but not yet started.
    Pending,
    /// Actively building.
    Running,
    /// Finished successfully.
    Complete,
    /// Terminated with an error.
    Failed,
}

impl BuildStatus {
    /// Returns `true` when the build has reached a terminal state.
    pub fn is_terminal(self) -> bool {
        matches!(self, BuildStatus::Complete | BuildStatus::Failed)
    }
}

// ── IndexId ───────────────────────────────────────────────────────────────────

/// Identifies any index in the system uniformly so that progress snapshots,
/// build handles, and API responses all use the same discriminant.
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum IndexId {
    /// RoaringBitmap field index (one per field per namespace).
    Field { namespace: String, field: String },
    /// Vector (IVF + RaBitQ) index (one per namespace).
    Vector { namespace: String },
}

impl IndexId {
    pub fn namespace(&self) -> &str {
        match self {
            IndexId::Field { namespace, .. } | IndexId::Vector { namespace } => namespace,
        }
    }

    /// Returns the [`IndexKind`] for this index identity.
    pub fn kind(&self) -> IndexKind {
        match self {
            IndexId::Field { .. } => IndexKind::Attribute,
            IndexId::Vector { .. } => IndexKind::Vector,
        }
    }
}

// ── IndexBuildSnapshot ────────────────────────────────────────────────────────

/// Uniform progress snapshot for any index type, returned by the REST API and
/// stored on disk to survive restarts.
///
/// `extra` carries type-specific data (e.g. `last_key_hex` for field builds,
/// `actionable`/`exhausted` counts for vector reindexs) without forcing the
/// common model to know about index-specific details.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexBuildSnapshot {
    /// Which index this snapshot belongs to.
    pub id: IndexId,
    /// The technology backing this index.
    pub kind: IndexKind,
    /// Current lifecycle phase.
    pub status: BuildStatus,
    /// Total documents in scope (0 until the initial scan completes).
    pub total: u64,
    /// Documents successfully indexed so far.
    pub indexed: u64,
    /// Documents that failed indexing (field builds abort on first failure so
    /// this is always ≤ 1 for field builds; per-doc for vector builds).
    pub failed: u64,
    /// Unix epoch milliseconds when the build started.
    pub started_at_ms: u64,
    /// Unix epoch milliseconds of the most recent progress update.
    pub updated_at_ms: u64,
    /// Unix epoch milliseconds when the build reached a terminal state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at_ms: Option<u64>,
    /// Error message when `status == Failed`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    /// Index-type-specific extension data (optional, serialised inline).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

impl IndexBuildSnapshot {
    /// Completion percentage in the range `[0.0, 100.0]`.
    pub fn percent(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.indexed as f64 / self.total as f64) * 100.0
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Current wall-clock time as milliseconds since the Unix epoch.
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Persistent state for a background index build, written to
/// `{db_path}/field-index/{ns_id}/{field_id}/build_progress.json`.
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
/// ([`DocStore::rowmap_stats`](crate::doc_store::DocStore::rowmap_stats)).
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
/// The directory comes from [`crate::layout`] — the engine owns where index
/// files live; the doc store only owns the filename it puts there.
pub(crate) fn build_progress_path(db_path: &Path, ns_id: u32, field_id: FieldId) -> PathBuf {
    crate::layout::namespace_index_dir(&crate::layout::index_root(db_path), ns_id)
        .join(field_id.to_string())
        .join("build_progress.json")
}

/// Read the persisted build progress for a field.  Returns `None` if the file
/// does not exist or cannot be parsed.
pub(crate) fn read_disk_progress(path: &Path) -> Option<DiskBuildProgress> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
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

/// Handle returned by [`DocStore::add_index`](crate::doc_store::DocStore::add_index) or
/// [`DocStore::resume_pending_builds`](crate::doc_store::DocStore::resume_pending_builds).
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
