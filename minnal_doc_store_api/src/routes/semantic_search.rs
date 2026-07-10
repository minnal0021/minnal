//! Semantic search endpoints:
//!
//! ```text
//! POST /stores/{ns}/semantic-search          → ANN query, no predicate filter
//! POST /stores/{ns}/semantic-search/filtered → ANN query restricted by index predicate
//! ```
//!
//! # Request bodies
//!
//! **Unfiltered**
//! ```json
//! { "query": "senior Rust engineer with distributed systems experience" }
//! ```
//!
//! **Filtered** — only candidates that pass the predicate are scored and returned:
//! ```json
//! {
//!   "query": "senior Rust engineer with distributed systems experience",
//!   "predicate": "status = \"active\""
//! }
//! ```
//!
//! # Response (both endpoints)
//!
//! An ordered array of results, highest similarity first:
//! ```json
//! [
//!   {
//!     "id": "550e8400-e29b-41d4-a716-446655440000",
//!     "dot_product": 0.94,
//!     "error_bound": 0.02,
//!     "document": { "id": 1, "text": "Senior Rust engineer with distributed systems experience." }
//!   }
//! ]
//! ```
//!
//! `document` is the full stored document object for the result.
//! It is `null` when the document could not be found (e.g. deleted since indexing).
//!
//! `id` is rendered according to the namespace's `key_type`:
//! - `uuid`  → hyphenated UUID string
//! - `u64`   → JSON number
//! - `u128`  → decimal string
//!
//! # Error responses
//!
//! | Condition                                          | Status |
//! |----------------------------------------------------|--------|
//! | Namespace not found                                | 404    |
//! | `semantic_search_enabled` is false for namespace   | 500    |
//! | Cluster index not loaded at startup                | 500    |
//! | Embedding service unreachable / returned an error  | 500    |
//! | Predicate references an un-indexed field (filtered)| 500    |

use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use minnal_db::{DocId, DocStoreError, Pagination};
use serde::{Deserialize, Serialize};
use tokio::task::JoinSet;
use tracing::debug;

use crate::{AppState, error::AppError, id::doc_id_to_value};

// ── Request / response types ──────────────────────────────────────────────────

fn default_page_size() -> usize {
    20
}
fn default_page_no() -> usize {
    1
}

/// URL query parameters for pagination — override body fields when present.
#[derive(serde::Deserialize)]
pub struct PaginationParams {
    page_no: Option<usize>,
    page_size: Option<usize>,
    /// Alias for `page_size` so `limit` works uniformly with the cursor-paginated
    /// scan endpoints. `page_size` wins if both are given.
    limit: Option<usize>,
}

/// Request body for `POST /stores/{ns}/semantic-search`.
#[derive(Deserialize)]
pub struct SemanticSearchRequest {
    /// Free-text query to embed and search against the document vectors.
    pub query: String,
    /// Override the number of results returned for this request only.
    /// When `None`, the value from the server-side TOML config is used.
    pub top_k: Option<usize>,
    #[serde(default = "default_page_size")]
    pub page_size: usize,
    #[serde(default = "default_page_no")]
    pub page_no: usize,
}

/// Request body for `POST /stores/{ns}/semantic-search/filtered`.
#[derive(Deserialize)]
pub struct SemanticSearchFilteredRequest {
    /// Free-text query to embed and search against the document vectors.
    pub query: String,
    /// Index predicate that candidates must satisfy (same syntax as
    /// `POST /stores/{ns}/query`).  Only documents that pass the predicate
    /// *and* score in the top-k by dot-product are returned.
    pub predicate: String,
    /// Override the number of results returned for this request only.
    /// When `None`, the value from the server-side TOML config is used.
    pub top_k: Option<usize>,
    #[serde(default = "default_page_size")]
    pub page_size: usize,
    #[serde(default = "default_page_no")]
    pub page_no: usize,
}

/// A single ranked result returned by either semantic search endpoint.
#[derive(Serialize)]
pub struct SemanticSearchResult {
    /// Document identifier serialised according to the namespace's `key_type`.
    pub id: serde_json::Value,
    /// Estimated dot-product similarity to the query (higher = more similar).
    pub dot_product: f32,
    /// Per-document error bound from the quantised vector index.
    pub error_bound: f32,
    /// The stored document value. Always present: candidates whose document no
    /// longer exists in the store (orphaned vector-index entries) are filtered
    /// out of the results, so a search hit always resolves to a live document.
    pub document: Option<serde_json::Value>,
}

// ── Handlers ──────────────────────────────────────────────────────────────────

/// `POST /stores/{ns}/semantic-search`
///
/// Embed `query` and return the top-k most similar documents in `ns`.
pub async fn query(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    Query(qp): Query<PaginationParams>,
    Json(req): Json<SemanticSearchRequest>,
) -> Result<impl IntoResponse, AppError> {
    debug!(namespace = %ns, top_k = ?req.top_k, "semantic search");
    let key_type = key_type_for(&state, &ns).await?;
    let pagination = Pagination::new(qp.page_no.unwrap_or(req.page_no), qp.page_size.or(qp.limit).unwrap_or(req.page_size));
    let page = state
        .store
        .search_semantic(&ns, &req.query, req.top_k, pagination)
        .await
        .map_err(|e| AppError::from(e).with_ns(&ns))?;
    let total = page.total;
    debug!(namespace = %ns, total = total, "semantic search complete");
    let results = decode_results(page.results, key_type, &state, &ns).await?;
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

/// `POST /stores/{ns}/semantic-search/filtered`
///
/// Embed `query` and return the top-k most similar documents in `ns` that
/// also satisfy `predicate`.
pub async fn query_filtered(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    Query(qp): Query<PaginationParams>,
    Json(req): Json<SemanticSearchFilteredRequest>,
) -> Result<impl IntoResponse, AppError> {
    debug!(namespace = %ns, predicate = %req.predicate, top_k = ?req.top_k, "filtered semantic search");
    let key_type = key_type_for(&state, &ns).await?;
    let pagination = Pagination::new(qp.page_no.unwrap_or(req.page_no), qp.page_size.or(qp.limit).unwrap_or(req.page_size));
    let page = state
        .store
        .search_semantic_filtered(&ns, &req.query, &req.predicate, req.top_k, pagination)
        .await
        .map_err(|e| AppError::from(e).with_ns(&ns))?;
    let total = page.total;
    debug!(namespace = %ns, total = total, "filtered semantic search complete");
    let results = decode_results(page.results, key_type, &state, &ns).await?;
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

async fn key_type_for(state: &AppState, ns: &str) -> Result<minnal_db::KeyType, AppError> {
    state
        .schemas
        .read()
        .await
        .get(ns)
        .map(|s| s.key_type)
        .ok_or_else(|| DocStoreError::NotFound { namespace: ns.to_owned() }.into())
}

/// Decode raw `QueryResult` bytes into API-friendly [`SemanticSearchResult`]s,
/// fetching all stored document values in parallel.
async fn decode_results(
    raw: Vec<minnal_db::semantic_search::index::vector_index::QueryResult>,
    key_type: minnal_db::KeyType,
    state: &AppState,
    ns: &str,
) -> Result<Vec<SemanticSearchResult>, AppError> {
    let n = raw.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Parse all doc IDs upfront so we can fail fast before spawning tasks.
    let doc_ids: Vec<DocId> = raw
        .iter()
        .map(|r| DocId::from_bytes(&r.document_id, key_type).map_err(AppError::from))
        .collect::<Result<_, _>>()?;

    // Fetch all documents in parallel.
    let mut join_set: JoinSet<(usize, Result<Option<serde_json::Value>, DocStoreError>)> = JoinSet::new();
    let store = Arc::clone(&state.store);
    let ns_owned = ns.to_string();
    for (idx, doc_id) in doc_ids.iter().copied().enumerate() {
        let store = Arc::clone(&store);
        let ns = ns_owned.clone();
        join_set.spawn(async move { (idx, store.get(&ns, doc_id).await) });
    }

    let mut documents: Vec<Option<serde_json::Value>> = vec![None; n];
    while let Some(res) = join_set.join_next().await {
        let (idx, doc_result) = res.map_err(|e| AppError::from(DocStoreError::BuildFailed(e.to_string())))?;
        documents[idx] = doc_result.map_err(AppError::from)?;
    }

    // Drop candidates whose document no longer exists (orphaned vector-index
    // entries): an ANN hit that doesn't resolve to a live document is filtered
    // out so the search result is robust against index/document drift.
    let results = raw
        .into_iter()
        .zip(doc_ids)
        .zip(documents)
        .filter_map(|((r, doc_id), document)| {
            document.map(|doc| SemanticSearchResult {
                id: doc_id_to_value(doc_id),
                dot_product: r.dot_product,
                error_bound: r.error_bound,
                document: Some(doc),
            })
        })
        .collect();

    Ok(results)
}
