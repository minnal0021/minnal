//! Uniform progress and status types shared by both the field-index (RoaringBitmap)
//! and the vector-index (IVF + RaBitQ) subsystems.
//!
//! Previously each subsystem had its own ad-hoc progress representation:
//! - Field index: `DiskBuildProgress` with string statuses + `IndexBuildProgress`
//! - Vector index: derived live from queue depth + `{ns}_sparse_vector_meta` count
//!
//! This module provides a single set of types that both systems populate so the
//! REST API can return a consistent shape regardless of index kind.

use serde::{Deserialize, Serialize};

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
