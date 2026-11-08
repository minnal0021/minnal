//! Index management endpoints:
//!
//! ```text
//! GET    /stores/{ns}/indices            → list all indices for namespace
//! POST   /stores/{ns}/indices            → add index (a field_build job, returns 202)
//! DELETE /stores/{ns}/indices/vector     → drop vector index (background cleanup, returns 202)
//! DELETE /stores/{ns}/indices/{field}    → drop field index (a field_drop job, returns 202)
//! ```

use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use minnal_db::{
    DocStoreError, IndexSpec, NamespaceState,
    doc_store::indexing::progress::{IndexBuildSnapshot, IndexId},
    namespace_lifecycle::BuildStatus,
};
use tracing::{error, info};

use crate::{AppState, error::AppError};

/// `GET /stores/{ns}/indices` — list all indices for a namespace.
pub async fn list_indices(State(state): State<AppState>, Path(ns): Path<String>) -> impl IntoResponse {
    let mut snaps: Vec<IndexBuildSnapshot> = state.store.field_index_builds(Some(&ns));

    if let Some(c) = state.store.vec_reindex_progress(&ns).await {
        use minnal_db::doc_store::indexing::progress::now_ms;
        let status = c.status;
        let id = IndexId::Vector { namespace: ns.clone() };
        snaps.push(IndexBuildSnapshot {
            kind: id.kind(),
            id,
            status,
            total: c.total_enqueued as u64,
            indexed: if status == BuildStatus::Complete { c.total_enqueued as u64 } else { 0 },
            failed: 0,
            started_at_ms: c.started_at_ms,
            updated_at_ms: now_ms(),
            completed_at_ms: c.completed_at_ms,
            last_error: c.error,
            extra: None,
        });
    }

    Json(snaps)
}

/// `202 Accepted` for an index job just started: `{namespace, job, state}`,
/// the job's name (for `GET /admin/stores/{ns}/jobs`) and the store's state.
pub(crate) fn job_accepted(namespace: &str, job: &str) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "namespace": namespace,
            "job": job,
            "state": NamespaceState::IndexMaintenance.as_str(),
        })),
    )
}

/// `POST /stores/{ns}/indices` — add an index. The index is activated at once
/// and built by a `field_build` job in the background: `202 Accepted`, with the
/// job's name. `409` while another job on the field (or a reindex-all) runs.
pub async fn add_index(State(state): State<AppState>, Path(ns): Path<String>, Json(spec): Json<IndexSpec>) -> Result<impl IntoResponse, AppError> {
    info!(namespace = %ns, field = %spec.field, index_type = ?spec.index_type, "adding index");
    let job = state.store.begin_add_index(&ns, spec).await?;
    info!(namespace = %ns, job = %job, "index activated, build started");
    Ok(job_accepted(&ns, &job))
}

/// `DELETE /stores/{ns}/indices/{field}` — drop a field index, as a
/// `field_drop` job in the background: `202 Accepted`, with the job's name.
/// `404` when the store does not exist or `field` is not indexed, and `409`
/// while another job on the field (its build, say: cancel it first) runs.
pub async fn drop_index(State(state): State<AppState>, Path((ns, field)): Path<(String, String)>) -> Result<impl IntoResponse, AppError> {
    info!(namespace = %ns, field = %field, "dropping index");
    let job = state.store.begin_drop_index(&ns, &field)?;
    Ok(job_accepted(&ns, &job))
}

/// `DELETE /stores/{ns}/indices/vector` — disable semantic search and drop all vector data.
///
/// Behaves identically to `DELETE /admin/indices/{ns}/vector/drop-all`.
/// The schema is updated synchronously before the background cleanup runs.
/// Returns `202 Accepted`.
pub async fn drop_vector_index(State(state): State<AppState>, Path(ns): Path<String>) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    // Claim the cleanup before semantic search goes off, atomically with the check:
    // an enable landing in between would see nothing running.
    if !state.vec_index_cleanup.lock().insert(ns.clone()) {
        return Err((
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": format!("vector index cleanup already in progress for '{ns}'")
            })),
        ));
    }
    let release = || {
        state.vec_index_cleanup.lock().remove(&ns);
    };

    // Block if a reindex is running.
    if crate::routes::admin_indices::vector_reindex_running(&state, &ns) {
        release();
        return Err((
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": format!("a vector index reindex is already running for '{ns}' — wait for it to complete")
            })),
        ));
    }

    // Disable SS in schema synchronously so new writes stop enqueuing.
    state.store.disable_semantic_search(&ns).map_err(|e| {
        release();
        let status = if matches!(e, DocStoreError::SemanticSearchNotEnabled { .. }) {
            StatusCode::UNPROCESSABLE_ENTITY
        } else if matches!(e, DocStoreError::NotFound { .. }) {
            StatusCode::NOT_FOUND
        } else {
            error!(namespace = %ns, error = %e, "vector index drop: failed to disable semantic search");
            StatusCode::INTERNAL_SERVER_ERROR
        };
        (status, Json(serde_json::json!({ "error": e.to_string() })))
    })?;

    info!(namespace = %ns, "vector index drop accepted — running in background");

    let store = Arc::clone(&state.store);
    let ops_ref = Arc::clone(&state.vec_index_cleanup);

    tokio::spawn(async move {
        match store.drop_vector_index_data(&ns).await {
            Ok(()) => info!(namespace = %ns, "vector index cleanup complete"),
            Err(e) => error!(namespace = %ns, error = %e, "vector index cleanup failed"),
        }
        ops_ref.lock().remove(&ns);
    });

    Ok(StatusCode::ACCEPTED)
}
