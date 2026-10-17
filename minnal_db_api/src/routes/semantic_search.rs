//! Semantic search endpoints:
//!
//! ```text
//! POST /stores/{ns}/semantic-search          → ANN query, no predicate filter
//! POST /stores/{ns}/semantic-search/filtered → ANN query restricted by an index predicate
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
//! Both also take optional `top_k`, `page_no` and `page_size` (`limit` is accepted
//! as a query-parameter alias for `page_size`).
//!
//! # Response
//!
//! A page of results, highest similarity first:
//! ```json
//! {
//!   "results": [
//!     {
//!       "id": "550e8400-e29b-41d4-a716-446655440000",
//!       "dot_product": 0.94,
//!       "error_bound": 0.02,
//!       "document": { "text": "Senior Rust engineer with distributed systems experience." }
//!     }
//!   ],
//!   "page_no": 1,
//!   "page_size": 20,
//!   "total": 1
//! }
//! ```
//!
//! The filtered endpoint adds `degraded_fields`: the predicate's fields whose
//! index is known to be missing updates (empty when the result is complete).
//!
//! `document` is the stored document. Candidates whose document has been deleted
//! since indexing are dropped, so it is never `null`. `id` is rendered according
//! to the namespace's `key_type`: a hyphenated string for `uuid`, a number for
//! `u64`, a decimal string for `u128`, the key itself for `str`.
//!
//! # Error responses
//!
//! | Condition                                             | Status |
//! |-------------------------------------------------------|--------|
//! | Namespace not found, or it is a KV store              | 404    |
//! | `semantic_search_enabled` is false for the namespace  | 422    |
//! | Predicate is malformed or names an un-indexed field   | 400    |
//! | Embedding service unreachable or returned an error    | 500    |

use crate::limits::Limit;
use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use minnal_db::doc_store::vector_settings::SearchSpec;
use minnal_db::{DocId, DocStoreError, Pagination};
use serde::{Deserialize, Serialize};
use tokio::task::JoinSet;
use tracing::{debug, warn};

use crate::{AppState, error::AppError, id::doc_id_to_value};

/// One request's search overrides. Validated by the store against the same
/// ranges as the namespace's settings (out of range is a 400). A request that
/// gives `probe_budget_entries` reads exactly that budget, so it also turns a
/// namespace's scaled budget off for that search.
pub(crate) fn search_overrides(top_k: Option<Limit>, probe: ProbeOverrides, first_pass_top_k: Option<u32>) -> SearchSpec {
    SearchSpec {
        probe_budget_entries: probe.probe_budget_entries,
        min_probes: probe.min_probes,
        max_probes: probe.max_probes,
        first_pass_top_k,
        top_k: top_k.map(|l| l.get() as u32),
        probe_budget_fraction: probe.probe_budget_entries.map(|_| minnal_db::doc_store::vector_settings::Ratio(0.0)),
        probe_budget_floor: None,
    }
}

/// A request's overrides of how many postings Pass 1 probes.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ProbeOverrides {
    pub probe_budget_entries: Option<u32>,
    pub min_probes: Option<u32>,
    pub max_probes: Option<u32>,
}

// ── Request / response types ──────────────────────────────────────────────────

/// Requests that carry probe overrides.
macro_rules! impl_probe_overrides {
    ($($t:ty),*) => {$(
        impl $t {
            pub(crate) fn probe(&self) -> ProbeOverrides {
                ProbeOverrides {
                    probe_budget_entries: self.probe_budget_entries,
                    min_probes: self.min_probes,
                    max_probes: self.max_probes,
                }
            }
        }
    )*};
}
impl_probe_overrides!(SemanticSearchRequest, SemanticSearchFilteredRequest, super::kv::KvSemanticSearchRequest);

fn default_page_no() -> usize {
    1
}

/// URL query parameters for pagination — override body fields when present.
#[derive(serde::Deserialize)]
pub struct PaginationParams {
    page_no: Option<usize>,
    page_size: Option<Limit>,
    /// Alias for `page_size` so `limit` works uniformly with the cursor-paginated
    /// scan endpoints. `page_size` wins if both are given.
    limit: Option<Limit>,
}

/// Request body for `POST /stores/{ns}/semantic-search`.
#[derive(Deserialize)]
pub struct SemanticSearchRequest {
    /// Free-text query to embed and search against the document vectors.
    pub query: String,
    /// Override the number of results returned for this request only
    /// (clamped to 1,000). When `None`, the namespace's `search.top_k` is used.
    pub top_k: Option<Limit>,
    /// Override the entries Pass 1 reads (1 to 10⁹), for this request only.
    /// When `None`, the namespace's `search.probe_budget_entries` is used.
    pub probe_budget_entries: Option<u32>,
    /// Override the postings Pass 1 always probes (1 to 4096, at most
    /// `max_probes`), for this request only. When `None`, the namespace's
    /// `search.min_probes` is used.
    pub min_probes: Option<u32>,
    /// Override the postings Pass 1 never exceeds (1 to 4096), for this
    /// request only. When `None`, the namespace's `search.max_probes` is used.
    pub max_probes: Option<u32>,
    /// Override the number of candidates Pass 1 hands to Pass 2, for this
    /// request only (`top_k` to 10,000). When `None`, the namespace's
    /// `search.first_pass_top_k` is used.
    pub first_pass_top_k: Option<u32>,
    #[serde(default)]
    pub page_size: Limit,
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
    /// Override the number of results returned for this request only
    /// (clamped to 1,000). When `None`, the namespace's `search.top_k` is used.
    pub top_k: Option<Limit>,
    /// Override the entries Pass 1 reads (1 to 10⁹), for this request only.
    /// When `None`, the namespace's `search.probe_budget_entries` is used.
    pub probe_budget_entries: Option<u32>,
    /// Override the postings Pass 1 always probes (1 to 4096, at most
    /// `max_probes`), for this request only. When `None`, the namespace's
    /// `search.min_probes` is used.
    pub min_probes: Option<u32>,
    /// Override the postings Pass 1 never exceeds (1 to 4096), for this
    /// request only. When `None`, the namespace's `search.max_probes` is used.
    pub max_probes: Option<u32>,
    /// Override the number of candidates Pass 1 hands to Pass 2, for this
    /// request only (`top_k` to 10,000). When `None`, the namespace's
    /// `search.first_pass_top_k` is used.
    pub first_pass_top_k: Option<u32>,
    #[serde(default)]
    pub page_size: Limit,
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
    let pagination = Pagination::new(
        qp.page_no.unwrap_or(req.page_no),
        qp.page_size.or(qp.limit).unwrap_or(req.page_size).get(),
    );
    let page = state
        .store
        .search_semantic(
            &ns,
            &req.query,
            &search_overrides(req.top_k, req.probe(), req.first_pass_top_k),
            pagination,
        )
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
    let pagination = Pagination::new(
        qp.page_no.unwrap_or(req.page_no),
        qp.page_size.or(qp.limit).unwrap_or(req.page_size).get(),
    );
    let page = state
        .store
        .search_semantic_filtered(
            &ns,
            &req.query,
            &req.predicate,
            &search_overrides(req.top_k, req.probe(), req.first_pass_top_k),
            pagination,
        )
        .await
        .map_err(|e| AppError::from(e).with_ns(&ns))?;
    let total = page.total;
    let degraded_fields = page.degraded_fields.clone();
    if !degraded_fields.is_empty() {
        warn!(namespace = %ns, fields = ?degraded_fields, "filtered semantic search used an incomplete index");
    }
    debug!(namespace = %ns, total = total, "filtered semantic search complete");
    let results = decode_results(page.results, key_type, &state, &ns).await?;
    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "results": results,
            "page_no": pagination.page_no,
            "page_size": pagination.page_size,
            "total": total,
            // The predicate narrows the ANN candidate set, so an incomplete
            // predicate index means candidates were filtered against a short
            // allow-list — the results may be missing documents.
            "degraded_fields": degraded_fields,
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
