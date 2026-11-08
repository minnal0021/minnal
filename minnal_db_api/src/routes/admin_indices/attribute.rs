//! Field (attribute) index operations: reindex-all, drop-all, one-document reindex, blob stats, health, row-map stats and repair.

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use minnal_db::DocStoreError;
use serde::Serialize;
use tracing::{error, info, warn};

use crate::{AppState, error::AppError, id::parse_doc_id};

// ── Attribute index operations ────────────────────────────────────────────────

/// `POST /admin/indices/{ns}/attribute/reindex-all`
///
/// Drops every field index for `{ns}`, frees its row IDs (the row map is reset,
/// so the rebuilt indexes number only the live documents), and rebuilds every
/// index, as one `reindex_all` job. Crash-safe: a run interrupted by a crash
/// is finished at the next start. Returns `202 Accepted` with the job's name;
/// progress is visible via `GET /admin/indices/{ns}/progress`. `409` while
/// any other index job of the store runs, `422` for a store with no field
/// index.
pub async fn attribute_reindex_all(State(state): State<AppState>, Path(ns): Path<String>) -> Result<impl IntoResponse, AppError> {
    match state.store.begin_reindex_all(&ns).map_err(|e| AppError::from(e).with_ns(&ns))? {
        Some(job) => {
            info!(namespace = %ns, "attribute reindex-all accepted — running in background");
            Ok(crate::routes::indices::job_accepted(&ns, &job).into_response())
        }
        None => Ok(no_attribute_indices(&ns, "reindex")),
    }
}

/// `DELETE /admin/indices/{ns}/attribute/drop-all`
///
/// Drops every field index for `{ns}` (no rebuild), as one `reindex_all` job
/// that stops after the drops. Returns `202 Accepted` with the job's name;
/// `409` while any other index job of the store runs.
pub async fn attribute_drop_all(State(state): State<AppState>, Path(ns): Path<String>) -> Result<impl IntoResponse, AppError> {
    match state
        .store
        .begin_drop_all_attribute_indices(&ns)
        .map_err(|e| AppError::from(e).with_ns(&ns))?
    {
        Some(job) => {
            info!(namespace = %ns, "attribute drop-all accepted — running in background");
            Ok(crate::routes::indices::job_accepted(&ns, &job).into_response())
        }
        None => Ok(no_attribute_indices(&ns, "drop")),
    }
}

/// `422` for a store with no field index to reindex or drop.
fn no_attribute_indices(ns: &str, what: &str) -> axum::response::Response {
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(serde_json::json!({
            "error": format!("namespace '{ns}' has no attribute indices to {what}")
        })),
    )
        .into_response()
}

// ── GET /admin/indices/{ns}/{field}/blob-stats ────────────────────────────────

/// One field index's on-disk blob growth/waste, with the compaction threshold.
#[derive(Serialize)]
pub struct FieldBlobStatsResponse {
    namespace: String,
    field: String,
    /// Compaction threshold as a fraction (`0.0..1.0`): a store is compacted at
    /// the next index checkpoint once its waste reaches this.
    waste_threshold: f64,
    /// True if the bitmap or keymap store has reached `waste_threshold`.
    over_threshold: bool,
    #[serde(flatten)]
    stats: minnal_db::IndexBlobStats,
}

/// `GET /admin/indices/{ns}/{field}/blob-stats` — on-disk blob growth for one
/// field index: bitmap and keymap **logical** bytes (everything ever appended =
/// live + stale) vs. **live** bytes (what survives compaction), their waste
/// ratios, and the distinct-value count, alongside the compaction threshold.
///
/// Complements the fleet-wide [`GET /admin/storage/index-waste`](super::admin_storage::index_waste),
/// which reports only waste *ratios*: this surfaces the absolute blob *growth*
/// between compactions that a ratio hides: every spill appends each changed
/// container plus a new directory for its bitmap, leaving the old copies as dead
/// space, so fields with many values that change often grow fastest. A
/// large, high-waste `bitmap_logical_bytes` is the signal to force a
/// [`POST /admin/storage/index-checkpoint`](super::admin_storage::trigger_index_checkpoint).
///
/// Returns `404 Not Found` when the field has no active index (unknown field, or
/// still building).
pub async fn field_blob_stats(
    State(state): State<AppState>,
    Path((ns, field)): Path<(String, String)>,
) -> Result<Json<FieldBlobStatsResponse>, (StatusCode, Json<serde_json::Value>)> {
    match state.store.field_index_blob_stats(&ns, &field) {
        Some(stats) => {
            let waste_threshold = state.db.index_blob_waste_threshold();
            let over_threshold = stats.bitmap_waste_ratio >= waste_threshold || stats.keymap_waste_ratio >= waste_threshold;
            Ok(Json(FieldBlobStatsResponse {
                namespace: ns,
                field,
                waste_threshold,
                over_threshold,
                stats,
            }))
        }
        None => Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("no active field index '{field}' in namespace '{ns}' (unknown field, or still building)")
            })),
        )),
    }
}

// ── POST /admin/indices/{ns}/attribute/{field}/reindex/{doc_id} ────────────────

/// `POST /admin/indices/{ns}/attribute/{field}/reindex/{doc_id}` — reindex a
/// **single document's** entry in **one** field index. Re-derives the field value
/// from the document's current stored bytes using the same logic as the write
/// path (clear the row's old buckets, re-extract, insert); only the named field
/// is touched — the document is not rewritten and no other field or vector index
/// is affected. Runs synchronously (it is O(1)) and returns `200`.
///
/// Doc-store namespaces only — field indices do not exist on KV stores. `404`
/// when the namespace, document, or indexed field is unknown; `409` when the
/// field index exists but is not yet active (still building).
pub async fn attribute_reindex_doc(
    State(state): State<AppState>,
    Path((ns, field, doc_id)): Path<(String, String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let err = |status: StatusCode, msg: String| (status, Json(serde_json::json!({ "error": msg })));

    let key_type = {
        let schemas = state.doc_schemas();
        match schemas.get(&ns).map(|s| s.key_type) {
            Some(kt) => kt,
            None => return Err(err(StatusCode::NOT_FOUND, format!("document store '{ns}' not found"))),
        }
    };
    let id =
        parse_doc_id(&doc_id, key_type).map_err(|_| err(StatusCode::BAD_REQUEST, format!("invalid document id '{doc_id}' for namespace '{ns}'")))?;

    match state.store.reindex_doc_field(&ns, id, &field).await {
        Ok(minnal_db::FieldReindexOutcome::Reindexed) => {
            info!(namespace = %ns, field = %field, doc_id = %doc_id, "field reindex (single doc) complete");
            Ok(Json(
                serde_json::json!({ "status": "reindexed", "namespace": ns, "field": field, "doc_id": doc_id }),
            ))
        }
        Ok(minnal_db::FieldReindexOutcome::KeyNotFound) => Err(err(StatusCode::NOT_FOUND, format!("document '{doc_id}' not found in '{ns}'"))),
        Ok(minnal_db::FieldReindexOutcome::FieldNotActive) => Err(err(
            StatusCode::CONFLICT,
            format!("field index '{field}' in '{ns}' is registered but not active (still building?)"),
        )),
        Err(DocStoreError::IndexNotFound { .. }) => Err(err(StatusCode::NOT_FOUND, format!("'{field}' is not an indexed field of '{ns}'"))),
        Err(DocStoreError::NotFound { .. }) => Err(err(StatusCode::NOT_FOUND, format!("document store '{ns}' not found"))),
        Err(e) => {
            error!(namespace = %ns, field = %field, doc_id = %doc_id, error = %e, "field reindex (single doc) failed");
            Err(err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

/// `GET /admin/indices/{ns}/health`
///
/// Report every field index in the namespace: where its persisted state reaches,
/// whether it is active, and any outstanding gap (what is missing and what
/// repair it needs).
///
/// This is the operator-facing half of FR-001. The per-query `degraded_fields`
/// field tells a *caller* their answer may be short; this tells an operator
/// which indices to repair, and is what makes the condition alertable rather
/// than something buried in a log line.
pub async fn index_health(
    State(state): State<AppState>,
    Path(ns): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let err = |status: StatusCode, msg: String| (status, Json(serde_json::json!({ "error": msg })));

    let health = state.store.index_health(&ns).await.map_err(|e| match e {
        DocStoreError::NotFound { .. } | DocStoreError::MissingNsId { .. } => err(StatusCode::NOT_FOUND, format!("document store '{ns}' not found")),
        other => err(StatusCode::INTERNAL_SERVER_ERROR, other.to_string()),
    })?;

    let degraded: Vec<&str> = health.iter().filter(|h| h.is_degraded()).map(|h| h.field_name.as_str()).collect();
    if !degraded.is_empty() {
        warn!(namespace = %ns, fields = ?degraded, "namespace has incomplete field indices");
    }
    Ok(Json(serde_json::json!({
        "namespace": ns,
        "degraded": !degraded.is_empty(),
        "degraded_fields": degraded,
        "fields": health,
    })))
}

/// `GET /admin/indices/{ns}/rowmap`
///
/// How much of the store's row map is spent on deleted documents. The row map
/// gives every document a dense row ID for the field indexes and never frees
/// one; `dead_ids` counts the IDs of documents that no longer exist, which
/// [`reindex-all`](attribute_reindex_all) frees. Counts live documents with a
/// key scan, so it is not free on a large store.
///
/// `uses_row_map: false` (and no counts) for a store without one: `u64` keys
/// take their row IDs from the key, and a store whose indexes were never built
/// has none yet. `404` for an unknown document store.
pub async fn rowmap_stats(
    State(state): State<AppState>,
    Path(ns): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let err = |status: StatusCode, msg: String| (status, Json(serde_json::json!({ "error": msg })));
    let stats = state.store.rowmap_stats(&ns).await.map_err(|e| match e {
        DocStoreError::NotFound { .. } | DocStoreError::MissingNsId { .. } => err(StatusCode::NOT_FOUND, format!("document store '{ns}' not found")),
        // A KV store has no field indexes, so no row map: as for the doc-only
        // row count, this store is the wrong kind.
        DocStoreError::Schema(minnal_db::SchemaError::WrongStoreType { .. }) => err(StatusCode::CONFLICT, e.to_string()),
        other => err(StatusCode::INTERNAL_SERVER_ERROR, other.to_string()),
    })?;
    Ok(Json(match stats {
        Some(s) => serde_json::json!({
            "namespace": ns,
            "uses_row_map": true,
            "ids_allocated": s.ids_allocated,
            "live_docs": s.live_docs,
            "dead_ids": s.dead_ids,
            "bytes_on_disk": s.bytes_on_disk,
        }),
        None => serde_json::json!({ "namespace": ns, "uses_row_map": false }),
    }))
}

/// `POST /admin/indices/{ns}/attribute/{field}/repair`
///
/// Repair a degraded field index and clear its gap record.
///
/// Row-scoped when the gap named the affected keys (work proportional to the
/// damage), a full rebuild when they could not be captured. Either way the
/// documents themselves are not re-put, so this generates no WAL traffic and
/// triggers no vector re-embedding.
///
/// Synchronous: row-scoped repair is bounded by the recorded worklist. A full
/// rebuild scans the namespace, so it can take a while on a large store.
pub async fn attribute_repair(
    State(state): State<AppState>,
    Path((ns, field)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let err = |status: StatusCode, msg: String| (status, Json(serde_json::json!({ "error": msg })));

    match state.store.repair_index(&ns, &field).await {
        Ok(minnal_db::FieldRepairOutcome::NotDegraded) => Ok(Json(serde_json::json!({
            "status": "not_degraded",
            "namespace": ns,
            "field": field,
            "message": "the index has no outstanding gap; nothing to repair",
        }))),
        Ok(outcome) => {
            info!(namespace = %ns, field = %field, outcome = ?outcome, "field index repaired");
            Ok(Json(serde_json::json!({
                "status": "repaired",
                "namespace": ns,
                "field": field,
                "result": outcome,
            })))
        }
        Err(DocStoreError::IndexNotFound { .. }) => Err(err(StatusCode::NOT_FOUND, format!("'{field}' is not an indexed field of '{ns}'"))),
        Err(DocStoreError::NotFound { .. }) => Err(err(StatusCode::NOT_FOUND, format!("document store '{ns}' not found"))),
        Err(e) => {
            error!(namespace = %ns, field = %field, error = %e, "field index repair failed");
            Err(err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}
