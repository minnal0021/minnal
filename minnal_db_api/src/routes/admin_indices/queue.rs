//! The embedding queue: summaries, listings and single-entry operations.

use crate::limits::Limit;

use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use minnal_db::hex_to_bytes;
use minnal_db::{Page, Pagination, QueueEntry, QueueEntryKind};
use serde::{Deserialize, Serialize};
use tracing::{error, info};

use crate::AppState;

// ── Vector queue monitoring ───────────────────────────────────────────────────

#[derive(Serialize)]
pub struct QueueEntryInfo {
    namespace: String,
    doc_id_hex: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    doc_id_str: Option<String>,
    /// `"embed"` (index `text`) or `"clear"` (a tombstone: remove the document's
    /// vectors — written by deletes and by upserts whose embedding text is empty).
    kind: &'static str,
    retry_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_error: Option<String>,
    text_preview: String,
}

impl From<QueueEntry> for QueueEntryInfo {
    fn from(e: QueueEntry) -> Self {
        let doc_id_hex = minnal_db::bytes_to_hex(&e.doc_id_bytes);
        let doc_id_str = std::str::from_utf8(&e.doc_id_bytes)
            .ok()
            .filter(|s| s.chars().all(|c| c.is_ascii_graphic() || c == ' '))
            .map(str::to_owned);
        let text_preview: String = e.text.chars().take(120).collect();
        QueueEntryInfo {
            namespace: e.namespace,
            doc_id_hex,
            doc_id_str,
            kind: match e.kind {
                QueueEntryKind::Embed => "embed",
                QueueEntryKind::Clear => "clear",
            },
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
    #[serde(default)]
    page_size: Limit,
}

fn default_page_no() -> usize {
    1
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
    Json(build_queue_response(entries, Pagination::new(params.page_no, params.page_size.get())))
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
    Json(build_queue_response(entries, Pagination::new(params.page_no, params.page_size.get())))
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
    Json(build_queue_response(entries, Pagination::new(params.page_no, params.page_size.get())))
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
