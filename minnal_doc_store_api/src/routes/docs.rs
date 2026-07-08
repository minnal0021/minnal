//! Document CRUD and query endpoints:
//!
//! ```text
//! GET    /stores/{ns}/docs/{id}                → find by id
//! PUT    /stores/{ns}/docs/{id}                → put (upsert) document
//! DELETE /stores/{ns}/docs/{id}                → delete document
//! GET    /stores/{ns}/docs?start=&end=         → range scan (end is optional)
//! GET    /stores/{ns}/docs/prefix?prefix=<hex> → prefix scan by document-id bytes
//! POST   /stores/{ns}/query                    → index predicate query
//! ```

use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use minnal_doc_store::{DocStoreError, Pagination, SchemaError, StoreType};
use serde::Deserialize;
use tracing::debug;

use crate::{
    AppState,
    error::AppError,
    id::{doc_id_to_value, parse_doc_id},
    routes::{decode_cursor, encode_cursor},
};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Error for a namespace absent from the doc-schema cache: a kind-mismatch
/// (409) when it exists as a KV store, otherwise not-found (404). Since doc and
/// KV stores now share the `/stores/{ns}` prefix, hitting a doc endpoint on a KV
/// namespace should say so rather than masquerade as 404.
fn doc_cache_miss(state: &AppState, ns: &str) -> AppError {
    match state.store.store_type(ns) {
        Ok(StoreType::Kv) => DocStoreError::Schema(SchemaError::WrongStoreType {
            namespace: ns.to_owned(),
            expected: "doc",
            found: "kv",
        })
        .into(),
        _ => DocStoreError::NotFound { namespace: ns.to_owned() }.into(),
    }
}

/// Resolve the key type for `ns` from the schema cache, or the appropriate error.
async fn key_type_for(state: &AppState, ns: &str) -> Result<minnal_doc_store::KeyType, AppError> {
    state.schemas.read().await.get(ns).map(|s| s.key_type).ok_or_else(|| doc_cache_miss(state, ns))
}

/// Resolve `(ns_id, key_type)` for `ns` from the schema cache, or the
/// appropriate error.
async fn ns_schema_for(state: &AppState, ns: &str) -> Result<(u32, minnal_doc_store::KeyType), AppError> {
    let schemas = state.schemas.read().await;
    let schema = schemas.get(ns).ok_or_else(|| doc_cache_miss(state, ns))?;
    let ns_id = schema
        .ns_id
        .ok_or_else(|| AppError::from(DocStoreError::MissingNsId { namespace: ns.to_owned() }))?;
    Ok((ns_id, schema.key_type))
}

// ── Handlers ──────────────────────────────────────────────────────────────────

pub async fn get_doc(State(state): State<AppState>, Path((ns, id_str)): Path<(String, String)>) -> Result<impl IntoResponse, AppError> {
    debug!(namespace = %ns, id = %id_str, "get doc");
    let key_type = key_type_for(&state, &ns).await?;
    let id = parse_doc_id(&id_str, key_type)?;
    match state
        .store
        .get(&ns, id)
        .await
        .map_err(|e| AppError::from(e).with_ns(&ns).with_id(&id_str))?
    {
        Some(doc) => {
            debug!(namespace = %ns, id = %id_str, "doc found");
            Ok((StatusCode::OK, Json(doc)).into_response())
        }
        None => {
            debug!(namespace = %ns, id = %id_str, "doc not found");
            Ok(StatusCode::NOT_FOUND.into_response())
        }
    }
}

/// Query parameters accepted by the PUT endpoint.
#[derive(Deserialize)]
pub struct PutParams {
    /// When `true`, the write bypasses the WAL for maximum throughput.
    /// Data written this way is unrecoverable on a crash — only use during
    /// bulk loading where re-running the load is acceptable.
    #[serde(default)]
    skip_wal: bool,
}

pub async fn put_doc(
    State(state): State<AppState>,
    Path((ns, id_str)): Path<(String, String)>,
    Query(params): Query<PutParams>,
    Json(doc): Json<serde_json::Value>,
) -> Result<impl IntoResponse, AppError> {
    debug!(namespace = %ns, id = %id_str, skip_wal = params.skip_wal, "put doc");
    let key_type = key_type_for(&state, &ns).await?;
    let id = parse_doc_id(&id_str, key_type)?;
    if params.skip_wal {
        state
            .store
            .put_no_wal(&ns, id, doc)
            .await
            .map_err(|e| AppError::from(e).with_ns(&ns).with_id(&id_str))?;
    } else {
        state
            .store
            .put(&ns, id, doc)
            .await
            .map_err(|e| AppError::from(e).with_ns(&ns).with_id(&id_str))?;
    }
    Ok(StatusCode::NO_CONTENT)
}

pub async fn delete_doc(State(state): State<AppState>, Path((ns, id_str)): Path<(String, String)>) -> Result<impl IntoResponse, AppError> {
    debug!(namespace = %ns, id = %id_str, "delete doc");
    let key_type = key_type_for(&state, &ns).await?;
    let id = parse_doc_id(&id_str, key_type)?;
    state
        .store
        .delete(&ns, id)
        .await
        .map_err(|e| AppError::from(e).with_ns(&ns).with_id(&id_str))?;
    Ok(StatusCode::NO_CONTENT)
}

fn default_page_size() -> usize {
    20
}
fn default_page_no() -> usize {
    1
}
fn default_limit() -> usize {
    20
}

/// Query parameters for the range scan endpoint.
#[derive(Deserialize)]
pub struct RangeParams {
    start: String,
    end: Option<String>,
    #[serde(default = "default_limit")]
    limit: usize,
    /// Opaque cursor from a prior page's `next_cursor`; absent for the first page.
    cursor: Option<String>,
}

pub async fn range_query(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    Query(params): Query<RangeParams>,
) -> Result<impl IntoResponse, AppError> {
    debug!(namespace = %ns, start = %params.start, end = ?params.end, limit = params.limit, "range query");
    let key_type = key_type_for(&state, &ns).await?;
    let start = parse_doc_id(&params.start, key_type)?;
    let end = params.end.as_deref().map(|s| parse_doc_id(s, key_type)).transpose()?;
    let cursor = params.cursor.as_deref().map(decode_cursor).transpose()?;
    let page = state.store.scan_range(&ns, start, end, cursor, params.limit).await?;
    let results: Vec<serde_json::Value> = page
        .results
        .into_iter()
        .map(|(id, doc)| serde_json::json!({ "id": doc_id_to_value(id), "doc": doc }))
        .collect();
    Ok(Json(serde_json::json!({
        "results": results,
        "next_cursor": page.next_cursor.as_deref().map(encode_cursor),
    })))
}

/// Decode a hex string (hyphens ignored) into raw bytes.
///
/// Accepts UUID-style strings (`550e8400-e29b-41d4`) as well as plain hex
/// (`550e8400e29b41d4`).  Returns [`DocStoreError::InvalidId`] on invalid input.
fn parse_hex_prefix(s: &str) -> Result<Vec<u8>, AppError> {
    let clean: String = s.chars().filter(|&c| c != '-').collect();
    if !clean.len().is_multiple_of(2) {
        return Err(DocStoreError::InvalidId("prefix must be an even-length hex string (hyphens are ignored)".into()).into());
    }
    (0..clean.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&clean[i..i + 2], 16)
                .map_err(|_| AppError::from(DocStoreError::InvalidId(format!("invalid hex byte '{}' in prefix", &clean[i..i + 2]))))
        })
        .collect()
}

/// Query parameters for `GET /stores/{ns}/docs/prefix`.
#[derive(Deserialize)]
pub struct PrefixScanParams {
    /// Hex-encoded byte prefix of the document key (hyphens ignored).
    ///
    /// For UUID stores, `550e8400-e29b-41d4` matches every document whose UUID
    /// starts with those 8 bytes.  For U64/U128 stores, supply the big-endian
    /// hex representation of the desired prefix.
    prefix: String,
    #[serde(default = "default_limit")]
    limit: usize,
    /// Opaque cursor from a prior page's `next_cursor`; absent for the first page.
    cursor: Option<String>,
}

pub async fn prefix_scan(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    Query(params): Query<PrefixScanParams>,
) -> Result<impl IntoResponse, AppError> {
    debug!(namespace = %ns, prefix = %params.prefix, limit = params.limit, "prefix scan");
    let prefix_bytes = parse_hex_prefix(&params.prefix)?;
    let cursor = params.cursor.as_deref().map(decode_cursor).transpose()?;
    let page = state.store.scan_prefix(&ns, prefix_bytes, cursor, params.limit).await?;
    let results: Vec<serde_json::Value> = page
        .results
        .into_iter()
        .map(|(id, doc)| serde_json::json!({ "id": doc_id_to_value(id), "doc": doc }))
        .collect();
    Ok(Json(serde_json::json!({
        "results": results,
        "next_cursor": page.next_cursor.as_deref().map(encode_cursor),
    })))
}

/// URL query parameters accepted by `POST /stores/{ns}/query`.
///
/// When provided, these override the corresponding fields in the request body.
#[derive(Deserialize)]
pub struct QueryPaginationParams {
    page_no: Option<usize>,
    page_size: Option<usize>,
    /// Alias for `page_size` so `limit` works uniformly with the cursor-paginated
    /// scan endpoints. `page_size` wins if both are given.
    limit: Option<usize>,
}

/// Request body for `POST /stores/{ns}/query`.
#[derive(Deserialize)]
pub struct QueryRequest {
    predicate: String,
    #[serde(default = "default_page_size")]
    page_size: usize,
    #[serde(default = "default_page_no")]
    page_no: usize,
}

pub async fn index_query(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    Query(qp): Query<QueryPaginationParams>,
    Json(req): Json<QueryRequest>,
) -> Result<impl IntoResponse, AppError> {
    debug!(namespace = %ns, predicate = %req.predicate, "index query");
    let (ns_id, key_type) = ns_schema_for(&state, &ns).await?;
    let page_no = qp.page_no.unwrap_or(req.page_no);
    let page_size = qp.page_size.or(qp.limit).unwrap_or(req.page_size);
    let pagination = Pagination::new(page_no, page_size);
    let page = state.store.query_resolved(&ns, &req.predicate, pagination, ns_id, key_type).await?;
    let results: Vec<serde_json::Value> = page
        .results
        .into_iter()
        .map(|(id, doc)| serde_json::json!({ "id": doc_id_to_value(id), "doc": doc }))
        .collect();
    Ok(Json(serde_json::json!({
        "results": results,
        "page_no": page.page_no,
        "page_size": page.page_size,
        "total": page.total,
    })))
}
