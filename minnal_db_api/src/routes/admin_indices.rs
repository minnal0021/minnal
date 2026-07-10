//! Index management admin endpoints — `/admin/indices/*`
//!
//! ```text
//! GET    /admin/indices/progress                          → all active index builds (field + vector)
//! GET    /admin/indices/vector/queue/summary              → global queue depth / lag
//! GET    /admin/indices/vector/queue/retried              → globally retried entries (all namespaces)
//! DELETE /admin/indices/vector/query-cache                → clear the system-wide query embedding cache
//! GET    /admin/indices/vector/corruption-metrics         → corrupt-skip counts per namespace (all namespaces)
//! GET    /admin/indices/{ns}/progress                     → per-namespace index progress
//! GET    /admin/indices/{ns}/vector/corruption-metrics    → corrupt-skip counts for one namespace
//! POST   /admin/indices/{ns}/attribute/reindex-all        → drop + rebuild all field indices (202)
//! DELETE /admin/indices/{ns}/attribute/drop-all           → drop all field indices (202)
//! POST   /admin/indices/{ns}/attribute/{field}/reindex/{doc_id} → reindex one doc in one field index (200)
//! GET    /admin/indices/{ns}/{field}/blob-stats           → one field's on-disk blob growth/waste (404 if not active)
//! POST   /admin/indices/{ns}/vector/reindex-all           → re-enqueue all docs for embedding (202)
//! POST   /admin/indices/{ns}/vector/reindex/{doc_id}      → re-enqueue one doc for embedding (doc + kv) (200)
//! POST   /admin/indices/{ns}/vector/reindex-failed        → reset exhausted queue entries (200)
//! DELETE /admin/indices/{ns}/vector/drop-all              → clear all vector index data (202)
//! GET    /admin/indices/{ns}/vector/queue                 → all queue entries for namespace
//! GET    /admin/indices/{ns}/vector/queue/retried         → retried entries for namespace
//! GET    /admin/indices/{ns}/vector/queue/{doc_id}        → look up one queue entry
//! DELETE /admin/indices/{ns}/vector/queue/{doc_id}        → remove one queue entry
//! POST   /admin/indices/{ns}/vector/queue/{doc_id}/retry  → retry one exhausted entry
//! ```

use std::{collections::HashMap, sync::Arc};

use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use minnal_db::doc_store::hex::hex_to_bytes;
use minnal_db::doc_store::index_progress::IndexBuildSnapshot;
use minnal_db::{DocStoreError, Page, Pagination, QueueEntry, VectorReindexOutcome};
use serde::{Deserialize, Serialize};
use tracing::{error, info};

use crate::{AppState, id::parse_doc_id, routes::stores::reload_schema};

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
pub async fn progress_all(State(state): State<AppState>) -> impl IntoResponse {
    Json(build_progress_response(&state, None).await)
}

/// `GET /admin/indices/{ns}/progress` — index progress for one namespace.
pub async fn progress_ns(State(state): State<AppState>, Path(ns): Path<String>) -> impl IntoResponse {
    Json(build_progress_response(&state, Some(&ns)).await)
}

async fn build_progress_response(state: &AppState, ns_filter: Option<&str>) -> IndexProgressResponse {
    let max_retries = state.store.vector_index_max_retries();

    let attribute_builds: Vec<IndexBuildSnapshot> = state
        .index_manager
        .list()
        .into_iter()
        .filter(|s| ns_filter.is_none_or(|ns| s.id.namespace() == ns))
        .collect();

    // Build vector reindex snapshots for SS-enabled namespaces.
    let ss_namespaces: Vec<String> = {
        let schemas = state.schemas.read().await;
        let kv_schemas = state.kv_schemas.read().await;
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
        let indexed_count = state.store.count_indexed_docs(&ns).await;
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

    IndexProgressResponse {
        attribute_builds,
        vector_progress,
    }
}

// ── Attribute index operations ────────────────────────────────────────────────

/// `POST /admin/indices/{ns}/attribute/reindex-all`
///
/// Drops every field index for `{ns}` and rebuilds them all from scratch.
/// Returns `202 Accepted`; progress is visible via `GET /admin/indices/{ns}/progress`.
/// Returns `409 Conflict` when any attribute or vector index operation is already active
/// for this namespace.
pub async fn attribute_reindex_all(
    State(state): State<AppState>,
    Path(ns): Path<String>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    {
        let ops = state.attr_index_ops.lock().unwrap();
        if ops.contains(&ns) {
            return Err((
                StatusCode::CONFLICT,
                Json(serde_json::json!({
                    "error": format!("an attribute index operation is already in progress for '{ns}'")
                })),
            ));
        }
    }

    let schema = state.store.get_schema(&ns).map_err(|e| {
        let status = if matches!(e, DocStoreError::NotFound { .. }) {
            StatusCode::NOT_FOUND
        } else {
            error!(namespace = %ns, error = %e, "attribute reindex-all: failed to load schema");
            StatusCode::INTERNAL_SERVER_ERROR
        };
        (status, Json(serde_json::json!({ "error": e.to_string() })))
    })?;

    if schema.indices.is_empty() {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "error": format!("namespace '{ns}' has no attribute indices to reindex")
            })),
        ));
    }

    state.attr_index_ops.lock().unwrap().insert(ns.clone());
    info!(namespace = %ns, "attribute reindex-all accepted — running in background");

    let store = Arc::clone(&state.store);
    let index_manager = Arc::clone(&state.index_manager);
    let ops_ref = Arc::clone(&state.attr_index_ops);

    tokio::spawn(async move {
        let result: Result<(), DocStoreError> = async {
            let specs = store.drop_all_attribute_indices(&ns)?;
            for spec in specs {
                let handle = store.add_index(&ns, spec).await?;
                index_manager.insert_field_build(handle);
            }
            Ok(())
        }
        .await;
        match result {
            Ok(()) => info!(namespace = %ns, "attribute reindex-all complete"),
            Err(e) => error!(namespace = %ns, error = %e, "attribute reindex-all failed"),
        }
        ops_ref.lock().unwrap().remove(&ns);
    });

    Ok(StatusCode::ACCEPTED)
}

/// `DELETE /admin/indices/{ns}/attribute/drop-all`
///
/// Drops every field index for `{ns}` (no rebuild).
/// Returns `202 Accepted`.
/// Returns `409` when an operation is already active.
pub async fn attribute_drop_all(State(state): State<AppState>, Path(ns): Path<String>) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    {
        let ops = state.attr_index_ops.lock().unwrap();
        if ops.contains(&ns) {
            return Err((
                StatusCode::CONFLICT,
                Json(serde_json::json!({
                    "error": format!("an attribute index operation is already in progress for '{ns}'")
                })),
            ));
        }
    }

    let schema = state.store.get_schema(&ns).map_err(|e| {
        let status = if matches!(e, DocStoreError::NotFound { .. }) {
            StatusCode::NOT_FOUND
        } else {
            error!(namespace = %ns, error = %e, "attribute drop-all: failed to load schema");
            StatusCode::INTERNAL_SERVER_ERROR
        };
        (status, Json(serde_json::json!({ "error": e.to_string() })))
    })?;

    if schema.indices.is_empty() {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "error": format!("namespace '{ns}' has no attribute indices to drop")
            })),
        ));
    }

    state.attr_index_ops.lock().unwrap().insert(ns.clone());
    info!(namespace = %ns, "attribute drop-all accepted — running in background");

    let state_c = state.clone();
    let store = Arc::clone(&state.store);
    let ops_ref = Arc::clone(&state.attr_index_ops);

    tokio::spawn(async move {
        // drop_all_attribute_indices mutates and persists the schema (demoting
        // every index to a plain attribute), so the cache must be reloaded after
        // it succeeds — otherwise it keeps showing the dropped indices.
        match store.drop_all_attribute_indices(&ns) {
            Ok(dropped) => {
                reload_schema(&state_c, &ns).await;
                info!(namespace = %ns, count = dropped.len(), "attribute drop-all complete");
            }
            Err(e) => error!(namespace = %ns, error = %e, "attribute drop-all failed"),
        }
        ops_ref.lock().unwrap().remove(&ns);
    });

    Ok(StatusCode::ACCEPTED)
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
/// between compactions that a ratio hides — worst for low-cardinality, high-churn
/// fields under the append-only whole-bitmap rewrite. A large, high-waste
/// `bitmap_logical_bytes` is the signal to force a
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
            let waste_threshold = state.store.index_blob_waste_threshold();
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
        let schemas = state.schemas.read().await;
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

// ── Vector index operations ───────────────────────────────────────────────────

/// `DELETE /admin/indices/{ns}/vector/drop-all`
///
/// Clears all vector index data for `{ns}`: embedding queue entries and the
/// `{ns}_sparse_vector`, `{ns}_dense_vector`, and `{ns}_sparse_vector_meta` companion stores.
/// The schema is updated synchronously (semantic search disabled) before the
/// background cleanup runs, preventing new embeddings from being enqueued.
/// Returns `202 Accepted`.
/// Returns `409` when a cleanup or reindex is already in progress.
/// Returns `422` when semantic search is not enabled for `{ns}`.
pub async fn vector_drop_all(State(state): State<AppState>, Path(ns): Path<String>) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
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

    // Disable semantic search in the schema immediately so new writes stop enqueuing.
    state.store.disable_semantic_search(&ns).map_err(|e| {
        let status = if matches!(e, DocStoreError::SemanticSearchNotEnabled { .. }) {
            StatusCode::UNPROCESSABLE_ENTITY
        } else if matches!(e, DocStoreError::NotFound { .. }) {
            StatusCode::NOT_FOUND
        } else {
            error!(namespace = %ns, error = %e, "vector drop-all: failed to disable semantic search");
            StatusCode::INTERNAL_SERVER_ERROR
        };
        (status, Json(serde_json::json!({ "error": e.to_string() })))
    })?;

    reload_schema(&state, &ns).await;
    state.vec_index_cleanup.lock().unwrap().insert(ns.clone());
    info!(namespace = %ns, "vector drop-all accepted — running in background");

    let store = Arc::clone(&state.store);
    let ops_ref = Arc::clone(&state.vec_index_cleanup);

    tokio::spawn(async move {
        match store.drop_vector_index_data(&ns).await {
            Ok(()) => info!(namespace = %ns, "vector drop-all cleanup complete"),
            Err(e) => error!(namespace = %ns, error = %e, "vector drop-all cleanup failed"),
        }
        ops_ref.lock().unwrap().remove(&ns);
    });

    Ok(StatusCode::ACCEPTED)
}

/// `POST /admin/indices/{ns}/vector/reindex-all`
///
/// Re-enqueues every document in `{ns}` for embedding (equivalent to a fresh
/// full index build).  Returns `202 Accepted`.
/// Returns `409` when a reindex is already running or a cleanup is active.
/// Returns `422` when semantic search is not enabled.
/// Returns `404` when the namespace does not exist.
pub async fn vector_reindex_all(State(state): State<AppState>, Path(ns): Path<String>) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    // Block if vector cleanup is in progress.
    {
        let ops = state.vec_index_cleanup.lock().unwrap();
        if ops.contains(&ns) {
            return Err((
                StatusCode::CONFLICT,
                Json(serde_json::json!({
                    "error": format!("vector index cleanup is in progress for '{ns}' — wait for it to complete")
                })),
            ));
        }
    }

    let is_kv = {
        let kv_schemas = state.kv_schemas.read().await;
        kv_schemas.contains_key(&ns)
    };

    let map_precondition_err = |label: &'static str| {
        move |e: DocStoreError| {
            let status = match &e {
                DocStoreError::NotFound { .. } => StatusCode::NOT_FOUND,
                DocStoreError::SemanticSearchNotEnabled { .. } => StatusCode::UNPROCESSABLE_ENTITY,
                DocStoreError::VecReindexInProgress { .. } => StatusCode::CONFLICT,
                _ => {
                    error!(error = %e, "{label}: unexpected error");
                    StatusCode::INTERNAL_SERVER_ERROR
                }
            };
            (status, Json(serde_json::json!({ "error": e.to_string() })))
        }
    };

    if is_kv {
        state
            .store
            .check_kv_index_all_preconditions(&ns)
            .map_err(map_precondition_err("vector reindex-all (kv)"))?;
        info!(namespace = %ns, "vector reindex-all (kv) accepted — running in background");
        tokio::spawn(async move {
            match state.store.kv_index_all(&ns).await {
                Ok(stats) => info!(
                    namespace = %ns,
                    enqueued = stats.enqueued,
                    exhausted_cleared = stats.exhausted_cleared,
                    "vector reindex-all (kv) complete",
                ),
                Err(e) => error!(namespace = %ns, error = %e, "vector reindex-all (kv) failed"),
            }
        });
    } else {
        state
            .store
            .check_index_all_preconditions(&ns)
            .map_err(map_precondition_err("vector reindex-all"))?;
        info!(namespace = %ns, "vector reindex-all accepted — running in background");
        tokio::spawn(async move {
            match state.store.index_all(&ns).await {
                Ok(stats) => info!(
                    namespace = %ns,
                    enqueued = stats.enqueued,
                    exhausted_cleared = stats.exhausted_cleared,
                    "vector reindex-all complete",
                ),
                Err(e) => error!(namespace = %ns, error = %e, "vector reindex-all failed"),
            }
        });
    }

    Ok(StatusCode::ACCEPTED)
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
    let is_kv = state.kv_schemas.read().await.contains_key(&ns);

    let outcome = if is_kv {
        // KV key is the raw {doc_id} string; the store encodes it per key_type.
        state.store.kv_reindex_doc_vector(&ns, &doc_id).await
    } else {
        let key_type = {
            let schemas = state.schemas.read().await;
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
    // Block if vector cleanup is in progress.
    {
        let ops = state.vec_index_cleanup.lock().unwrap();
        if ops.contains(&ns) {
            return Err((
                StatusCode::CONFLICT,
                Json(serde_json::json!({
                    "error": format!("vector index cleanup is in progress for '{ns}'")
                })),
            ));
        }
    }

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
/// The cache is shared across all semantic-search namespaces and keyed only by
/// query text. Clear it after changing the chunking parameters
/// (`window_size` / `sliding_size`), in tandem with a corpus re-index — otherwise
/// stale cached vectors silently degrade recall until the configured TTL
/// (`query_embedding_cache_ttl_secs`, default 1 day) expires.
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
/// vectors are corrupt and queries are silently degraded — run the validating
/// reconcile ([`POST /admin/indices/vector/reconcile`](vector_reconcile)) to
/// re-embed the affected documents. Split by pass (`sparse` = Pass 1, `dense` =
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

// ── POST /admin/indices/vector/reconcile ──────────────────────────────────────

/// `POST /admin/indices/vector/reconcile` — **validating** vector-index reconciliation.
///
/// Scans every semantic-search-enabled namespace and re-enqueues any document whose
/// committed vector index is missing a half **or present-but-corrupt** (the bytes
/// fail to deserialize), as well as documents with no pending queue entry left by a
/// crash. Unlike the cheap presence-only pass that runs at startup, this **validates
/// the bytes** — so it reads and deserializes every entry and cannot use the count
/// short-circuit, making it a full value-reading scan.
///
/// Because that scan can take a long time on a large corpus, this **returns
/// immediately with `202 Accepted`** and runs the pass in the **background**; the
/// re-enqueued count and any per-namespace failures are written to the log
/// (`info!` on completion, `warn!`/`error!` on failure). Overlapping runs are
/// rejected with `409 Conflict` so expensive scans cannot stack.
///
/// A presence-only reconcile still runs automatically on store startup.
pub async fn vector_reconcile(State(state): State<AppState>) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    use std::sync::atomic::Ordering;

    // Reject overlapping runs — a validating reconcile is a full value-reading scan.
    if state
        .vec_reconcile_running
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err((
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": "a vector reconcile is already running" })),
        ));
    }

    info!("vector reconcile (validating) accepted — running in background");
    let store = Arc::clone(&state.store);
    let running = Arc::clone(&state.vec_reconcile_running);
    tokio::spawn(async move {
        // Clear the running flag however the task exits (including a panic), so a
        // failure can never wedge the endpoint into a permanent 409.
        struct ResetOnDrop(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for ResetOnDrop {
            fn drop(&mut self) {
                self.0.store(false, std::sync::atomic::Ordering::Release);
            }
        }
        let _reset = ResetOnDrop(running);

        let reenqueued = store.validate_and_reconcile_vector_indexes().await;
        info!(reenqueued, "vector reconcile (validating) complete");
    });

    Ok(StatusCode::ACCEPTED)
}

// ── Vector queue monitoring ───────────────────────────────────────────────────

#[derive(Serialize)]
pub struct QueueEntryInfo {
    namespace: String,
    doc_id_hex: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    doc_id_str: Option<String>,
    retry_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_error: Option<String>,
    text_preview: String,
}

impl From<QueueEntry> for QueueEntryInfo {
    fn from(e: QueueEntry) -> Self {
        let doc_id_hex = minnal_db::doc_store::hex::bytes_to_hex(&e.doc_id_bytes);
        let doc_id_str = std::str::from_utf8(&e.doc_id_bytes)
            .ok()
            .filter(|s| s.chars().all(|c| c.is_ascii_graphic() || c == ' '))
            .map(str::to_owned);
        let text_preview: String = e.text.chars().take(120).collect();
        QueueEntryInfo {
            namespace: e.namespace,
            doc_id_hex,
            doc_id_str,
            retry_count: e.retry_count,
            last_error: e.last_error,
            text_preview,
        }
    }
}

#[derive(Serialize)]
pub struct QueueListResponse {
    total: usize,
    page_no: usize,
    page_size: usize,
    entries: Vec<QueueEntryInfo>,
}

#[derive(Deserialize)]
pub struct QueuePaginationParams {
    #[serde(default = "default_page_no")]
    page_no: usize,
    #[serde(default = "default_page_size")]
    page_size: usize,
}

fn default_page_no() -> usize {
    1
}
fn default_page_size() -> usize {
    20
}

fn build_queue_response(entries: Vec<QueueEntry>, pagination: Pagination) -> QueueListResponse {
    let page: Page<QueueEntry> = Page::from_vec(entries, pagination);
    QueueListResponse {
        total: page.total,
        page_no: page.page_no,
        page_size: page.page_size,
        entries: page.results.into_iter().map(QueueEntryInfo::from).collect(),
    }
}

/// `GET /admin/indices/vector/queue/retried` — entries retried at least once (all namespaces).
pub async fn vector_queue_retried(State(state): State<AppState>, Query(params): Query<QueuePaginationParams>) -> impl IntoResponse {
    let entries: Vec<_> = state.store.list_queue_entries().await.into_iter().filter(|e| e.retry_count > 0).collect();
    Json(build_queue_response(entries, Pagination::new(params.page_no, params.page_size)))
}

/// `GET /admin/indices/vector/queue/summary` — global queue depth and lag.
#[derive(Serialize)]
pub struct NsQueueSummary {
    namespace: String,
    pending: u32,
    actionable: u32,
    retrying: u32,
    exhausted: u32,
}

#[derive(Serialize)]
pub struct QueueSummaryResponse {
    max_retries_configured: u32,
    total_pending: u32,
    total_actionable: u32,
    total_retrying: u32,
    total_exhausted: u32,
    by_namespace: Vec<NsQueueSummary>,
}

pub async fn vector_queue_summary(State(state): State<AppState>) -> impl IntoResponse {
    let entries = state.store.list_queue_entries().await;
    let max_retries = state.store.vector_index_max_retries();

    let mut by_ns: std::collections::BTreeMap<String, NsQueueSummary> = std::collections::BTreeMap::new();

    for e in &entries {
        let s = by_ns.entry(e.namespace.clone()).or_insert_with(|| NsQueueSummary {
            namespace: e.namespace.clone(),
            pending: 0,
            actionable: 0,
            retrying: 0,
            exhausted: 0,
        });
        s.pending += 1;
        if e.retry_count >= max_retries {
            s.exhausted += 1;
        } else {
            s.actionable += 1;
            if e.retry_count > 0 {
                s.retrying += 1;
            }
        }
    }

    let by_namespace: Vec<NsQueueSummary> = by_ns.into_values().collect();
    let total_pending = entries.len() as u32;
    let total_actionable: u32 = by_namespace.iter().map(|s| s.actionable).sum();
    let total_retrying: u32 = by_namespace.iter().map(|s| s.retrying).sum();
    let total_exhausted: u32 = by_namespace.iter().map(|s| s.exhausted).sum();

    Json(QueueSummaryResponse {
        max_retries_configured: max_retries,
        total_pending,
        total_actionable,
        total_retrying,
        total_exhausted,
        by_namespace,
    })
}

/// `GET /admin/indices/{ns}/vector/queue` — queue entries for one namespace.
pub async fn vector_queue_by_namespace(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    Query(params): Query<QueuePaginationParams>,
) -> impl IntoResponse {
    let entries: Vec<_> = state.store.list_queue_entries().await.into_iter().filter(|e| e.namespace == ns).collect();
    Json(build_queue_response(entries, Pagination::new(params.page_no, params.page_size)))
}

/// `GET /admin/indices/{ns}/vector/queue/retried` — retried entries for one namespace.
pub async fn vector_queue_retried_by_namespace(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    Query(params): Query<QueuePaginationParams>,
) -> impl IntoResponse {
    let entries: Vec<_> = state
        .store
        .list_queue_entries()
        .await
        .into_iter()
        .filter(|e| e.namespace == ns && e.retry_count > 0)
        .collect();
    Json(build_queue_response(entries, Pagination::new(params.page_no, params.page_size)))
}

/// `GET /admin/indices/{ns}/vector/queue/{doc_id}` — look up one queue entry.
pub async fn vector_queue_get_entry(
    State(state): State<AppState>,
    Path((ns, doc_id_hex)): Path<(String, String)>,
) -> Result<Json<QueueEntryInfo>, (StatusCode, Json<serde_json::Value>)> {
    let doc_id_bytes = hex_to_bytes(&doc_id_hex).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": format!("'{doc_id_hex}' is not valid hex") })),
        )
    })?;
    match state.store.get_queue_entry(&ns, &doc_id_bytes).await {
        Some(entry) => Ok(Json(QueueEntryInfo::from(entry))),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("no queue entry for namespace='{ns}' doc_id_hex='{doc_id_hex}'")
            })),
        )),
    }
}

/// `DELETE /admin/indices/{ns}/vector/queue/{doc_id}` — remove one queue entry.
pub async fn vector_queue_delete_entry(
    State(state): State<AppState>,
    Path((ns, doc_id_hex)): Path<(String, String)>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    let doc_id_bytes = hex_to_bytes(&doc_id_hex).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": format!("'{doc_id_hex}' is not valid hex") })),
        )
    })?;

    let exists = state
        .store
        .list_queue_entries()
        .await
        .into_iter()
        .any(|e| e.namespace == ns && e.doc_id_bytes == doc_id_bytes);

    if !exists {
        return Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("no queue entry for namespace='{ns}' doc_id_hex='{doc_id_hex}'")
            })),
        ));
    }

    state.store.delete_queue_entry(&ns, &doc_id_bytes).await.map_err(|e| {
        error!(namespace = %ns, doc_id_hex = %doc_id_hex, error = %e, "failed to delete vector queue entry");
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() })))
    })?;

    info!(namespace = %ns, doc_id_hex = %doc_id_hex, "deleted vector queue entry");
    Ok(StatusCode::NO_CONTENT)
}

/// `POST /admin/indices/{ns}/vector/queue/{doc_id}/retry` — retry one exhausted entry.
pub async fn vector_queue_retry_entry(
    State(state): State<AppState>,
    Path((ns, doc_id_hex)): Path<(String, String)>,
) -> Result<Json<QueueEntryInfo>, (StatusCode, Json<serde_json::Value>)> {
    let doc_id_bytes = hex_to_bytes(&doc_id_hex).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": format!("'{doc_id_hex}' is not valid hex") })),
        )
    })?;

    let max_retries = state.store.vector_index_max_retries();
    let entry = state.store.get_queue_entry(&ns, &doc_id_bytes).await.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("no queue entry for namespace='{ns}' doc_id_hex='{doc_id_hex}'")
            })),
        )
    })?;

    if entry.retry_count < max_retries {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "error": format!(
                    "entry has not exhausted its retry budget (retry_count={} max_retries={})",
                    entry.retry_count, max_retries
                )
            })),
        ));
    }

    state.store.retry_queue_entry(&ns, &doc_id_bytes).await.map_err(|e| {
        error!(namespace = %ns, doc_id_hex = %doc_id_hex, error = %e, "failed to retry vector queue entry");
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() })))
    })?;

    info!(namespace = %ns, doc_id_hex = %doc_id_hex, "reset retry count for exhausted queue entry");
    Ok(Json(QueueEntryInfo::from(entry)))
}
