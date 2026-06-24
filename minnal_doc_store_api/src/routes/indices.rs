//! Index management endpoints:
//!
//! ```text
//! GET    /stores/{ns}/indices            → list all indices for namespace
//! POST   /stores/{ns}/indices            → add index (background build, returns 202)
//! DELETE /stores/{ns}/indices/vector     → drop vector index (background cleanup, returns 202)
//! DELETE /stores/{ns}/indices/{field}    → drop field index (background cleanup, returns 202)
//! ```

use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use minnal_doc_store::{
    DocStoreError, IndexSpec,
    index_progress::{BuildStatus, IndexBuildSnapshot, IndexId},
};
use tracing::{error, info};

use crate::{AppState, error::AppError, routes::stores::reload_schema};

/// `GET /stores/{ns}/indices` — list all indices for a namespace.
pub async fn list_indices(State(state): State<AppState>, Path(ns): Path<String>) -> impl IntoResponse {
    let mut snaps: Vec<IndexBuildSnapshot> = state.index_manager.list().into_iter().filter(|s| s.id.namespace() == ns).collect();

    if let Some(c) = state.store.vec_reindex_progress(&ns) {
        use minnal_doc_store::index_progress::now_ms;
        let status = match c.status.as_str() {
            "complete" => BuildStatus::Complete,
            "failed" => BuildStatus::Failed,
            _ => BuildStatus::Running,
        };
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

pub async fn add_index(State(state): State<AppState>, Path(ns): Path<String>, Json(spec): Json<IndexSpec>) -> Result<impl IntoResponse, AppError> {
    info!(namespace = %ns, field = %spec.field, index_type = ?spec.index_type, "adding index");
    let handle = state.store.add_index(&ns, spec).await?;
    info!(namespace = %ns, field = %handle.field, "index activated, background build started");
    state.index_manager.insert_field_build(handle);
    reload_schema(&state, &ns).await;
    Ok(StatusCode::ACCEPTED)
}

/// `DELETE /stores/{ns}/indices/{field}` — drop a field index.
///
/// The bitmap files are deleted in a background task.  Returns `202 Accepted`.
/// Returns `409` when an attribute index operation is already active for this namespace.
pub async fn drop_index(State(state): State<AppState>, Path((ns, field)): Path<(String, String)>) -> Result<impl IntoResponse, AppError> {
    {
        let ops = state.attr_index_ops.lock().unwrap();
        if ops.contains(&ns) {
            return Err(DocStoreError::AttrIndexOpInProgress { namespace: ns }.into());
        }
    }

    // Verify the index exists (fail fast before spawning).
    state.store.get_schema(&ns).map_err(AppError::from)?;

    info!(namespace = %ns, field = %field, "dropping index — background cleanup");
    state.attr_index_ops.lock().unwrap().insert(ns.clone());

    let state_c = state.clone();
    let store = Arc::clone(&state.store);
    let ops_ref = Arc::clone(&state.attr_index_ops);
    let ns_c = ns.clone();
    let field_c = field.clone();

    tokio::spawn(async move {
        // `drop_index` is what mutates and persists the schema (demoting the
        // index to a plain attribute) and deactivates the in-memory index, so
        // the schema cache must be reloaded *after* it succeeds — not before, or
        // the cache keeps showing the dropped index while the in-memory index is
        // already gone.
        match store.drop_index(&ns_c, &field_c) {
            Ok(()) => {
                reload_schema(&state_c, &ns_c).await;
                info!(namespace = %ns_c, field = %field_c, "index drop complete");
            }
            Err(e) => error!(namespace = %ns_c, field = %field_c, error = %e, "index drop failed"),
        }
        ops_ref.lock().unwrap().remove(&ns_c);
    });

    Ok(StatusCode::ACCEPTED)
}

/// `DELETE /stores/{ns}/indices/vector` — disable semantic search and drop all vector data.
///
/// Behaves identically to `DELETE /admin/indices/{ns}/vector/drop-all`.
/// The schema is updated synchronously before the background cleanup runs.
/// Returns `202 Accepted`.
pub async fn drop_vector_index(State(state): State<AppState>, Path(ns): Path<String>) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    // Block if cleanup already running.
    {
        let ops = state.vec_index_cleanup.lock().unwrap();
        if ops.contains(&ns) {
            return Err((
                StatusCode::CONFLICT,
                Json(serde_json::json!({
                    "error": format!("vector index cleanup already in progress for '{ns}'")
                })),
            ));
        }
    }

    // Block if a reindex reindex is running.
    if let Err(DocStoreError::VecReindexInProgress { .. }) = state.store.check_index_all_preconditions(&ns) {
        return Err((
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": format!("a vector index reindex is already running for '{ns}' — wait for it to complete")
            })),
        ));
    }

    // Disable SS in schema synchronously so new writes stop enqueuing.
    state.store.disable_semantic_search(&ns).map_err(|e| {
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

    reload_schema(&state, &ns).await;
    state.vec_index_cleanup.lock().unwrap().insert(ns.clone());
    info!(namespace = %ns, "vector index drop accepted — running in background");

    let store = Arc::clone(&state.store);
    let ops_ref = Arc::clone(&state.vec_index_cleanup);

    tokio::spawn(async move {
        match store.drop_vector_index_data(&ns).await {
            Ok(()) => info!(namespace = %ns, "vector index cleanup complete"),
            Err(e) => error!(namespace = %ns, error = %e, "vector index cleanup failed"),
        }
        ops_ref.lock().unwrap().remove(&ns);
    });

    Ok(StatusCode::ACCEPTED)
}
