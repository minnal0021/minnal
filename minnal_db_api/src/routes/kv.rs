//! KV store CRUD, scan, and semantic search endpoints:
//!
//! ```text
//! GET    /stores/{ns}/kv/{key}                → get value by key
//! PUT    /stores/{ns}/kv/{key}[?skip_wal=]    → set value (JSON body = value)
//! DELETE /stores/{ns}/kv/{key}                → delete key
//! GET    /stores/{ns}/kv?start=&end=          → range scan (end optional)
//! GET    /stores/{ns}/kv/prefix?prefix=       → prefix scan
//! POST   /stores/{ns}/kv/semantic-search      → ANN search (value_type = str only)
//! ```
//!
//! Keys are URL path segments:
//! - `key_type = str`  → the segment is the string as-is
//! - `key_type = int`  → the segment is the decimal integer
//!
//! Values are raw JSON bodies matching the namespace `value_type`:
//! - `int`    → JSON number (integer)
//! - `str`    → JSON string
//! - `f32`    → JSON number (float)
//! - `vec_f32`→ JSON array of numbers

use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use minnal_db::{DocStoreError, KvKeyType, Pagination};
use serde::{Deserialize, Serialize};
use tokio::task::JoinSet;
use tracing::debug;

use crate::{
    AppState,
    error::AppError,
    routes::{decode_cursor, encode_cursor},
};

// ── GET /stores/{ns}/kv/{key} ────────────────────────────────────────────────

pub async fn get_kv(State(state): State<AppState>, Path((ns, key)): Path<(String, String)>) -> Result<impl IntoResponse, AppError> {
    debug!(namespace = %ns, key = %key, "kv get");
    match state.store.kv_get_by_str(&ns, &key).await.map_err(|e| AppError::from(e).with_ns(&ns))? {
        Some(value) => Ok((StatusCode::OK, Json(value)).into_response()),
        None => Ok(StatusCode::NOT_FOUND.into_response()),
    }
}

// ── PUT /stores/{ns}/kv/{key} ────────────────────────────────────────────────

pub async fn put_kv(
    State(state): State<AppState>,
    Path((ns, key)): Path<(String, String)>,
    Query(params): Query<KvPutParams>,
    Json(value): Json<serde_json::Value>,
) -> Result<impl IntoResponse, AppError> {
    debug!(namespace = %ns, key = %key, skip_wal = params.skip_wal, "kv put");
    let key_json = key_to_json(&state, &ns, &key).await?;
    if params.skip_wal {
        state
            .store
            .kv_put_no_wal(&ns, &key_json, &value)
            .await
            .map_err(|e| AppError::from(e).with_ns(&ns))?;
    } else {
        state
            .store
            .kv_put(&ns, &key_json, &value)
            .await
            .map_err(|e| AppError::from(e).with_ns(&ns))?;
    }
    Ok(StatusCode::NO_CONTENT)
}

/// Query parameters accepted by the KV PUT endpoint.
#[derive(Deserialize)]
pub struct KvPutParams {
    /// When `true`, the write bypasses the WAL for maximum throughput.
    /// Data written this way is unrecoverable on a crash — only use during
    /// bulk loading where re-running the load is acceptable.
    #[serde(default)]
    skip_wal: bool,
}

// ── DELETE /stores/{ns}/kv/{key} ─────────────────────────────────────────────

pub async fn delete_kv(State(state): State<AppState>, Path((ns, key)): Path<(String, String)>) -> Result<impl IntoResponse, AppError> {
    debug!(namespace = %ns, key = %key, "kv delete");
    state.store.kv_delete(&ns, &key).await.map_err(|e| AppError::from(e).with_ns(&ns))?;
    Ok(StatusCode::NO_CONTENT)
}

// ── GET /stores/{ns}/kv?start=&end= ──────────────────────────────────────────

#[derive(Deserialize)]
pub struct KvRangeParams {
    start: String,
    end: Option<String>,
    #[serde(default = "default_limit")]
    limit: usize,
    /// Opaque cursor from a prior page's `next_cursor`; absent for the first page.
    cursor: Option<String>,
}

pub async fn range_kv(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    Query(params): Query<KvRangeParams>,
) -> Result<impl IntoResponse, AppError> {
    debug!(namespace = %ns, start = %params.start, end = ?params.end, "kv range scan");
    let cursor = params.cursor.as_deref().map(decode_cursor).transpose()?;
    let page = state
        .store
        .kv_scan_range(&ns, &params.start, params.end.as_deref(), cursor, params.limit)
        .await
        .map_err(|e| AppError::from(e).with_ns(&ns))?;
    let results: Vec<serde_json::Value> = page
        .results
        .into_iter()
        .map(|(k, v)| serde_json::json!({ "key": k, "value": v }))
        .collect();
    Ok(Json(serde_json::json!({
        "results": results,
        "next_cursor": page.next_cursor.as_deref().map(encode_cursor),
    })))
}

// ── GET /stores/{ns}/kv/prefix?prefix= ───────────────────────────────────────

#[derive(Deserialize)]
pub struct KvPrefixScanParams {
    prefix: String,
    #[serde(default = "default_limit")]
    limit: usize,
    /// Opaque cursor from a prior page's `next_cursor`; absent for the first page.
    cursor: Option<String>,
}

pub async fn prefix_scan_kv(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    Query(params): Query<KvPrefixScanParams>,
) -> Result<impl IntoResponse, AppError> {
    debug!(namespace = %ns, prefix = %params.prefix, "kv prefix scan");
    let cursor = params.cursor.as_deref().map(decode_cursor).transpose()?;
    let page = state
        .store
        .kv_scan_prefix(&ns, &params.prefix, cursor, params.limit)
        .await
        .map_err(|e| AppError::from(e).with_ns(&ns))?;
    let results: Vec<serde_json::Value> = page
        .results
        .into_iter()
        .map(|(k, v)| serde_json::json!({ "key": k, "value": v }))
        .collect();
    Ok(Json(serde_json::json!({
        "results": results,
        "next_cursor": page.next_cursor.as_deref().map(encode_cursor),
    })))
}

// ── POST /stores/{ns}/kv/semantic-search ─────────────────────────────────────

fn default_page_size() -> usize {
    20
}
fn default_page_no() -> usize {
    1
}
fn default_limit() -> usize {
    20
}

#[derive(Deserialize)]
pub struct SemanticSearchParams {
    page_no: Option<usize>,
    page_size: Option<usize>,
    /// Alias for `page_size` so `limit` works uniformly with the cursor-paginated
    /// scan endpoints. `page_size` wins if both are given.
    limit: Option<usize>,
}

#[derive(Deserialize)]
pub struct KvSemanticSearchRequest {
    pub query: String,
    pub top_k: Option<usize>,
    #[serde(default = "default_page_size")]
    pub page_size: usize,
    #[serde(default = "default_page_no")]
    pub page_no: usize,
}

#[derive(Serialize)]
pub struct KvSemanticSearchResult {
    /// Key, rendered as a JSON string or number according to `key_type`.
    pub key: serde_json::Value,
    pub dot_product: f32,
    pub error_bound: f32,
    /// The stored value. Always present: candidates whose entry no longer exists
    /// (orphaned vector-index entries) are filtered out of the results, so a
    /// search hit always resolves to a live KV entry.
    pub value: Option<serde_json::Value>,
}

pub async fn search_kv_semantic(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    Query(qp): Query<SemanticSearchParams>,
    Json(req): Json<KvSemanticSearchRequest>,
) -> Result<impl IntoResponse, AppError> {
    debug!(namespace = %ns, top_k = ?req.top_k, "KV semantic search");

    let key_type = kv_key_type_for(&state, &ns).await?;
    let pagination = Pagination::new(qp.page_no.unwrap_or(req.page_no), qp.page_size.or(qp.limit).unwrap_or(req.page_size));

    let page = state
        .store
        .kv_search_semantic(&ns, &req.query, req.top_k, pagination)
        .await
        .map_err(|e| AppError::from(e).with_ns(&ns))?;

    let total = page.total;
    let results = hydrate_kv_results(page.results, key_type, &state, &ns).await?;

    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "results": results,
            "page_no": pagination.page_no,
            "page_size": pagination.page_size,
            "total": total,
        })),
    ))
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Look up the `KvKeyType` for `ns` from the in-memory schema cache.
async fn kv_key_type_for(state: &AppState, ns: &str) -> Result<KvKeyType, AppError> {
    state
        .kv_schemas
        .read()
        .await
        .get(ns)
        .map(|s| s.key_type)
        .ok_or_else(|| DocStoreError::NotFound { namespace: ns.to_owned() }.into())
}

/// Convert the URL key string to a typed JSON value using the schema cache.
///
/// Used by `put_kv` so that `kv_put` receives a properly typed JSON key.
async fn key_to_json(state: &AppState, ns: &str, raw: &str) -> Result<serde_json::Value, AppError> {
    let key_type = kv_key_type_for(state, ns).await?;
    match key_type {
        KvKeyType::Str => Ok(serde_json::Value::String(raw.to_owned())),
        KvKeyType::Int => raw
            .parse::<i64>()
            .map(serde_json::Value::from)
            .map_err(|_| AppError::from(DocStoreError::InvalidId(format!("key '{raw}' is not a valid integer")))),
    }
}

/// Fetch stored values in parallel and zip them with the ANN result metadata.
async fn hydrate_kv_results(
    raw: Vec<minnal_db::semantic_search::index::vector_index::QueryResult>,
    key_type: KvKeyType,
    state: &AppState,
    ns: &str,
) -> Result<Vec<KvSemanticSearchResult>, AppError> {
    let n = raw.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Deserialize key bytes from each result.
    let keys: Vec<serde_json::Value> = raw
        .iter()
        .map(|r| {
            key_type
                .deserialize_key(&r.document_id)
                .map_err(|e| AppError::from(DocStoreError::Schema(e)))
        })
        .collect::<Result<_, _>>()?;

    // Fetch values in parallel using the typed JSON key.
    let mut join_set: JoinSet<(usize, Result<Option<serde_json::Value>, DocStoreError>)> = JoinSet::new();
    let store = Arc::clone(&state.store);
    let ns_owned = ns.to_string();
    for (idx, key_json) in keys.iter().cloned().enumerate() {
        let store = Arc::clone(&store);
        let ns = ns_owned.clone();
        join_set.spawn(async move { (idx, store.kv_get(&ns, &key_json).await) });
    }

    let mut values: Vec<Option<serde_json::Value>> = vec![None; n];
    while let Some(res) = join_set.join_next().await {
        let (idx, result) = res.map_err(|e| AppError::from(DocStoreError::BuildFailed(e.to_string())))?;
        values[idx] = result.map_err(AppError::from)?;
    }

    // Drop candidates whose entry no longer exists (orphaned vector-index
    // entries) so the search result is robust against index/entry drift.
    let results = raw
        .into_iter()
        .zip(keys)
        .zip(values)
        .filter_map(|((r, key), value)| {
            value.map(|v| KvSemanticSearchResult {
                key,
                dot_product: r.dot_product,
                error_bound: r.error_bound,
                value: Some(v),
            })
        })
        .collect();

    Ok(results)
}
