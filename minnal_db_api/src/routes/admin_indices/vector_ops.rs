//! Vector index operations: drop, reindex-all, one-document reindex, retrying failed entries, the query-embedding cache, corruption metrics.

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use minnal_db::{DocStoreError, VectorReindexOutcome};
use tracing::{error, info};

use crate::{AppState, error::AppError, id::parse_doc_id};

// ── Vector index operations ───────────────────────────────────────────────────

/// `DELETE /admin/indices/{ns}/vector/drop-all`
///
/// Turns semantic search off for `{ns}`, a document or KV store, and removes
/// its vector index (embedding queue entries, the vector namespaces and the
/// partition), as a `vector_cleanup` job. Semantic search is off before the
/// reply, so writes stop enqueueing. Returns `202 Accepted` with the job's
/// name; `409` while another vector job of the store runs; `422` when
/// semantic search is not enabled.
pub async fn vector_drop_all(State(state): State<AppState>, Path(ns): Path<String>) -> Result<impl IntoResponse, AppError> {
    let job = state.store.begin_drop_vector_index(&ns).map_err(|e| AppError::from(e).with_ns(&ns))?;
    info!(namespace = %ns, "vector drop-all accepted — running in background");
    Ok(crate::routes::indices::job_accepted(&ns, &job))
}

/// `POST /admin/indices/{ns}/vector/reindex-all`
///
/// Re-enqueues every document of `{ns}` (document or KV store) for embedding,
/// as a `vector_reindex` job. Returns `202 Accepted` with the job's name;
/// `409` while another vector job of the store runs; `422` when semantic
/// search is not enabled; `404` when the store does not exist.
pub async fn vector_reindex_all(State(state): State<AppState>, Path(ns): Path<String>) -> Result<impl IntoResponse, AppError> {
    let job = state.store.begin_vector_reindex(&ns).map_err(|e| AppError::from(e).with_ns(&ns))?;
    info!(namespace = %ns, "vector reindex-all accepted — running in background");
    Ok(crate::routes::indices::job_accepted(&ns, &job))
}

// ── POST /admin/indices/{ns}/vector/reindex/{doc_id} ──────────────────────────

/// `POST /admin/indices/{ns}/vector/reindex/{doc_id}` — re-enqueue a **single**
/// document for vector (re-)embedding, the same enqueue the write path and
/// `vector/reindex-all` use, scoped to one document. The async worker embeds and
/// indexes it on its next pass, so this returns `200` immediately (not `202` — the
/// enqueue itself is synchronous and durable).
///
/// Works for both document stores and semantic-search KV stores; for a KV store
/// `{doc_id}` is the key string. The JSON `status` is `enqueued`, `not_found`
/// (no such document), or `skipped_empty_text` (the document has no embedding
/// text). `422` when the namespace is not semantic-search-enabled; `404` when it
/// does not exist.
pub async fn vector_reindex_doc(
    State(state): State<AppState>,
    Path((ns, doc_id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let err = |status: StatusCode, msg: String| (status, Json(serde_json::json!({ "error": msg })));

    // KV namespace? (mirrors vector_reindex_all's doc-vs-kv split).
    let is_kv = state.kv_schemas().contains_key(&ns);

    let outcome = if is_kv {
        // KV key is the raw {doc_id} string; the store encodes it per key_type.
        state.store.kv_reindex_doc_vector(&ns, &doc_id).await
    } else {
        let key_type = {
            let schemas = state.doc_schemas();
            match schemas.get(&ns).map(|s| s.key_type) {
                Some(kt) => kt,
                None => return Err(err(StatusCode::NOT_FOUND, format!("namespace '{ns}' not found"))),
            }
        };
        let id = parse_doc_id(&doc_id, key_type)
            .map_err(|_| err(StatusCode::BAD_REQUEST, format!("invalid document id '{doc_id}' for namespace '{ns}'")))?;
        state.store.reindex_doc_vector(&ns, id).await
    };

    match outcome {
        Ok(VectorReindexOutcome::Enqueued) => {
            info!(namespace = %ns, doc_id = %doc_id, "vector reindex (single doc) enqueued");
            Ok(Json(serde_json::json!({ "status": "enqueued", "namespace": ns, "doc_id": doc_id })))
        }
        Ok(VectorReindexOutcome::NotFound) => Err(err(StatusCode::NOT_FOUND, format!("document '{doc_id}' not found in '{ns}'"))),
        Ok(VectorReindexOutcome::SkippedEmptyText) => Ok(Json(
            serde_json::json!({ "status": "skipped_empty_text", "namespace": ns, "doc_id": doc_id }),
        )),
        Err(e @ DocStoreError::SemanticSearchNotEnabled { .. }) => Err(err(StatusCode::UNPROCESSABLE_ENTITY, e.to_string())),
        Err(e @ DocStoreError::NotFound { .. }) => Err(err(StatusCode::NOT_FOUND, e.to_string())),
        Err(e) => {
            error!(namespace = %ns, doc_id = %doc_id, error = %e, "vector reindex (single doc) failed");
            Err(err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

/// `POST /admin/indices/{ns}/vector/reindex-failed`
///
/// Resets `retry_count` to zero for every exhausted queue entry in `{ns}`,
/// making them actionable again.  Returns `200` with `{ "retried": N }`.
pub async fn vector_reindex_failed(
    State(state): State<AppState>,
    Path(ns): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    info!(namespace = %ns, "vector reindex-failed — resetting exhausted queue entries");
    let retried = state.store.retry_all_failed_queue_entries(&ns).await.map_err(|e| {
        error!(namespace = %ns, error = %e, "vector reindex-failed: failed to reset queue entries");
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() })))
    })?;
    info!(namespace = %ns, retried, "vector reindex-failed complete");
    Ok(Json(serde_json::json!({ "retried": retried })))
}

// ── DELETE /admin/indices/vector/query-cache ──────────────────────────────────

/// `DELETE /admin/indices/vector/query-cache` — clear the system-wide query
/// embedding cache.
///
/// The cache is shared across all semantic-search namespaces and keyed by
/// embedding model and query text, so neither switching `semantic_search.model`
/// nor changing the chunking parameters (queries are not chunked) needs a
/// clear. Clear it when the service starts producing different vectors under
/// the same model name — otherwise stale cached vectors silently degrade recall
/// until the configured TTL (`query_embedding_cache_ttl_secs`, default 1 day)
/// expires.
pub async fn vector_query_cache_clear(State(state): State<AppState>) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    info!("clearing system-wide query embedding cache");
    let cleared = state.store.clear_query_embedding_cache().await.map_err(|e| {
        error!(error = %e, "failed to clear query embedding cache");
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() })))
    })?;
    info!(cleared, "query embedding cache cleared");
    Ok(Json(serde_json::json!({ "cleared": cleared })))
}

// ── GET /admin/indices/vector/corruption-metrics ──────────────────────────────

/// `GET /admin/indices/vector/corruption-metrics` — cumulative counts of vector
/// entries skipped during search because their bytes failed to deserialize,
/// **broken down per namespace** (a namespace that has never recorded a
/// corruption is omitted; an empty object means none have).
///
/// Counters are in-memory and monotonically increasing since startup (sample
/// twice for a rate), reset on restart. A non-zero/rising value means stored
/// vectors are corrupt and queries are silently degraded — re-index the affected
/// namespace (`POST /admin/indices/{ns}/vector/reindex-all`) to re-embed its
/// documents. Split by pass (`sparse` = Pass 1, `dense` =
/// Pass 2). For a single namespace use
/// [`GET /admin/indices/{ns}/vector/corruption-metrics`](vector_corruption_metrics_ns).
pub async fn vector_corruption_metrics() -> Json<std::collections::BTreeMap<String, minnal_db::semantic_search::metrics::VectorMetricsSnapshot>> {
    Json(minnal_db::semantic_search::metrics::snapshot_all())
}

// ── GET /admin/indices/{ns}/vector/corruption-metrics ─────────────────────────

/// `GET /admin/indices/{ns}/vector/corruption-metrics` — corruption counts for a
/// single namespace (all-zero if the namespace has never recorded a corruption).
/// See [`vector_corruption_metrics`] for the cross-namespace view and semantics.
pub async fn vector_corruption_metrics_ns(Path(ns): Path<String>) -> Json<minnal_db::semantic_search::metrics::VectorMetricsSnapshot> {
    Json(minnal_db::semantic_search::metrics::snapshot(&ns))
}
