//! Progress and status types for field-index builds and vector-index reindexes, so
//! the REST API can return one shape whichever kind of index it reports, and
//! the progress a field-index build saves in its job's record.

use serde::{Deserialize, Serialize};

use crate::doc_store::schema::IndexKind;
use crate::namespace_lifecycle::BuildStatus;

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

/// How far a field-index build got: the progress a `field_build` job keeps in
/// its record (and a `reindex-all`, for the field it is building).
///
/// Saved when the job starts, every 1,000 documents, and when the scan ends.
/// A restart resumes the build after `resume_after_hex`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldBuildProgress {
    /// The index being built.
    pub spec: crate::doc_store::IndexSpec,
    /// Total document count (0 until the first scan has counted them).
    pub total: u64,
    /// Documents indexed so far.
    pub indexed: u64,
    /// Hex-encoded key a restart resumes after. Never ahead of what the field
    /// index has made durable: the build names a key only once an index
    /// checkpoint covers it. `None`: start from the first key.
    pub resume_after_hex: Option<String>,
}

impl FieldBuildProgress {
    /// A build of `spec` that has done nothing yet.
    pub fn new(spec: crate::doc_store::IndexSpec) -> Self {
        Self {
            spec,
            total: 0,
            indexed: 0,
            resume_after_hex: None,
        }
    }
}

impl crate::doc_store::DocStore {
    /// The field-index builds of every document store (or only `namespace`)
    /// as progress snapshots, read from their jobs' records: every
    /// `field_build` job with a record, live or ended, and the fields a live
    /// `reindex-all` is building or has still to build.
    pub fn field_index_builds(&self, namespace: Option<&str>) -> Vec<IndexBuildSnapshot> {
        let mut snapshots = Vec::new();
        let mut schemas = self.schemas.docs();
        schemas.sort_by(|a, b| a.namespace.cmp(&b.namespace));
        for schema in schemas.iter().filter(|s| namespace.is_none_or(|n| s.namespace == n)) {
            let Some(ns_id) = schema.ns_id else { continue };
            let snapshot = |field: &str, status: BuildStatus, p: Option<&FieldBuildProgress>, record: &crate::namespace_lifecycle::JobRecord| {
                let id = IndexId::Field {
                    namespace: schema.namespace.clone(),
                    field: field.to_owned(),
                };
                IndexBuildSnapshot {
                    kind: id.kind(),
                    id,
                    status,
                    total: p.map_or(0, |p| p.total),
                    indexed: p.map_or(0, |p| p.indexed),
                    failed: 0,
                    started_at_ms: record.started_at_ms,
                    updated_at_ms: record.updated_at_ms,
                    completed_at_ms: status.is_terminal().then_some(record.updated_at_ms),
                    last_error: record.error.clone(),
                    extra: None,
                }
            };
            for (key, record) in self.jobs.list(ns_id).unwrap_or_default() {
                if let crate::namespace_lifecycle::JobKind::FieldBuild(_) = key.kind
                    && let Some(p) = record.progress::<FieldBuildProgress>()
                {
                    snapshots.push(snapshot(&p.spec.field, record.status, Some(&p), &record));
                }
            }
            if let Some((record, current, waiting)) = self.reindex_all_builds(ns_id) {
                if let Some(p) = &current {
                    snapshots.push(snapshot(&p.spec.field, record.status, Some(p), &record));
                }
                for field in &waiting {
                    snapshots.push(snapshot(field, BuildStatus::Pending, None, &record));
                }
            }
        }
        snapshots
    }
}
