//! Progress and status types for field-index builds and vector-index reindexes, so
//! the REST API can return one shape whichever kind of index it reports.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use serde::{Deserialize, Serialize};

use crate::doc_store::error::DocStoreError;
use crate::doc_store::indexing::observer::InMemoryProgress;

use crate::doc_store::schema::IndexKind;

// ── BuildStatus ───────────────────────────────────────────────────────────────

/// Lifecycle status of an index build or vector-index reindex.
///
/// Stored as a snake_case string in JSON, in API responses and in the
/// maintenance records.
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

/// Uniform progress snapshot for any index type, returned by the REST API.
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

pub use crate::support::now_ms;

/// One field-index build's progress, kept in the maintenance namespace
/// (`doc_store::maintenance`, key `TaskKey::field_build`).
///
/// Written when the build is set up (before the schema lists the index), every
/// 1,000 documents while it runs, and when it ends. A restart resumes a
/// `Running` build from `resume_after`. Each write is WAL-backed and durable on
/// return, so a record is either there whole or not there.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FieldBuildRecord {
    /// `Running` until the build ends, then `Complete` or `Failed`.
    pub status: BuildStatus,
    /// Total document count (0 until the initial scan finishes).
    pub total: u64,
    /// Documents processed so far.
    pub indexed: u64,
    /// Hex-encoded key a restart resumes after. Never ahead of what the
    /// field index has made durable: the build names a key only once an index
    /// checkpoint covers it. `None`: start from the first key.
    pub resume_after_hex: Option<String>,
    /// Why the build failed, when `status == Failed`.
    pub error: Option<String>,
}

impl FieldBuildRecord {
    /// A build that has started and processed nothing yet.
    pub fn started() -> Self {
        Self {
            status: BuildStatus::Running,
            total: 0,
            indexed: 0,
            resume_after_hex: None,
            error: None,
        }
    }

    /// Whether the build is still to finish (running, or cut short by a crash).
    pub fn is_running(&self) -> bool {
        self.status == BuildStatus::Running
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
