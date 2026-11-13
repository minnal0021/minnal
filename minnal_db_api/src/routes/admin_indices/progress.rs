//! Index build progress across field and vector indexes.

use std::collections::HashMap;

use axum::{
    Json,
    extract::{Path, State},
    response::IntoResponse,
};
use minnal_db::doc_store::indexing::progress::IndexBuildSnapshot;
use serde::Serialize;

use crate::AppState;

// ── Progress — unified view ───────────────────────────────────────────────────

/// Combined index progress response across all or one namespace.
#[derive(Serialize)]
pub struct IndexProgressResponse {
    /// Active or recently completed field-index builds.
    attribute_builds: Vec<IndexBuildSnapshot>,
    /// Vector-index progress per semantic-search-enabled namespace.
    vector_progress: Vec<NsVectorProgress>,
}

/// Vector-index progress for one namespace (queue-based view).
#[derive(Serialize)]
pub struct NsVectorProgress {
    namespace: String,
    /// Documents with a committed vector-index entry.
    indexed_approx: u64,
    /// Actionable queue entries (`retry_count < max_retries`).
    pending: u64,
    /// Exhausted queue entries needing manual attention.
    exhausted: u64,
    /// Approximate indexing completion (0–100).
    progress_pct: f64,
}

/// `GET /admin/indices/progress` — all active index builds across every namespace.
pub async fn progress_all(State(state): State<AppState>) -> Result<impl IntoResponse, crate::error::AppError> {
    Ok(Json(build_progress_response(&state, None).await?))
}

/// `GET /admin/indices/{ns}/progress` — index progress for one namespace.
pub async fn progress_ns(State(state): State<AppState>, Path(ns): Path<String>) -> Result<impl IntoResponse, crate::error::AppError> {
    // Unknown stores are a 404, as on every other per-store route, rather than an
    // empty report that reads like a store with nothing running.
    state.store.store_type(&ns).map_err(|e| crate::error::AppError::from(e).with_ns(&ns))?;
    Ok(Json(build_progress_response(&state, Some(&ns)).await?))
}

async fn build_progress_response(state: &AppState, ns_filter: Option<&str>) -> Result<IndexProgressResponse, crate::error::AppError> {
    let max_retries = state.store.vector_index_max_retries();

    let attribute_builds: Vec<IndexBuildSnapshot> = state.store.field_index_builds(ns_filter);

    // Build vector reindex snapshots for SS-enabled namespaces.
    let ss_namespaces: Vec<String> = {
        let schemas = state.doc_schemas();
        let kv_schemas = state.kv_schemas();
        let mut v: Vec<String> = schemas
            .values()
            .filter(|s| s.semantic_search_enabled)
            .map(|s| s.namespace.clone())
            .chain(kv_schemas.values().filter(|s| s.semantic_search_enabled).map(|s| s.namespace.clone()))
            .filter(|ns| ns_filter.is_none_or(|f| ns == f))
            .collect();
        v.sort();
        v.dedup();
        v
    };

    let queue_entries = state.store.list_queue_entries().await;
    let mut pending_by_ns: HashMap<String, u64> = HashMap::new();
    let mut exhausted_by_ns: HashMap<String, u64> = HashMap::new();
    for e in &queue_entries {
        if ns_filter.is_none_or(|f| e.namespace == f) {
            *pending_by_ns.entry(e.namespace.clone()).or_default() += 1;
            if e.retry_count >= max_retries {
                *exhausted_by_ns.entry(e.namespace.clone()).or_default() += 1;
            }
        }
    }

    let mut vector_progress = Vec::with_capacity(ss_namespaces.len());
    for ns in ss_namespaces {
        let indexed_count = state
            .store
            .count_indexed_docs(&ns)
            .await
            .map_err(|e| crate::error::AppError::from(e).with_ns(&ns))?;
        let total_pending = pending_by_ns.get(&ns).copied().unwrap_or(0);
        let exhausted = exhausted_by_ns.get(&ns).copied().unwrap_or(0);
        let actionable = total_pending.saturating_sub(exhausted);
        let denominator = indexed_count + actionable;
        let progress_pct = if denominator == 0 {
            100.0
        } else {
            (indexed_count as f64 / denominator as f64) * 100.0
        };
        vector_progress.push(NsVectorProgress {
            namespace: ns,
            indexed_approx: indexed_count,
            pending: actionable,
            exhausted,
            progress_pct,
        });
    }

    Ok(IndexProgressResponse {
        attribute_builds,
        vector_progress,
    })
}
