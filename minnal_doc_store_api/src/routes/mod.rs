pub mod admin_indices;
pub mod admin_storage;
pub mod admin_stores;
pub mod docs;
pub mod indices;
pub mod kv;
pub mod semantic_search;
pub mod stores;

use axum::{
    Router,
    routing::{delete, get, post},
};

use crate::{AppState, error::AppError};

/// Encode an opaque scan cursor (raw key bytes) as a hex string for round-trip
/// through a request query parameter.
pub(crate) fn encode_cursor(key: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(key.len() * 2);
    for b in key {
        let _ = write!(s, "{:02x}", b);
    }
    s
}

/// Decode a hex scan cursor (as produced by [`encode_cursor`]) back into raw key
/// bytes. Returns a 400-class error on malformed input.
pub(crate) fn decode_cursor(s: &str) -> Result<Vec<u8>, AppError> {
    if !s.len().is_multiple_of(2) {
        return Err(minnal_doc_store::DocStoreError::InvalidId("cursor must be an even-length hex string".into()).into());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| {
                AppError::from(minnal_doc_store::DocStoreError::InvalidId(format!(
                    "invalid hex byte '{}' in cursor",
                    &s[i..i + 2]
                )))
            })
        })
        .collect()
}

pub fn router() -> Router<AppState> {
    Router::new()
        // ── Admin / storage diagnostics (read) ───────────────────────────────
        .route("/admin/storage/health", get(admin_storage::health))
        .route("/admin/storage/stats", get(admin_storage::stats))
        .route("/admin/storage/ops-metrics", get(admin_storage::ops_metrics))
        .route("/admin/storage/ops-metrics/by-namespace", get(admin_storage::ops_metrics_by_namespace))
        .route("/admin/storage/stores/{ns}/ops-metrics", get(admin_storage::ops_metrics_ns))
        .route("/admin/storage/wal", get(admin_storage::wal))
        .route("/admin/storage/lsm", get(admin_storage::lsm))
        .route("/admin/storage/value-log", get(admin_storage::value_log))
        .route("/admin/storage/value-log/{ns}/pages", get(admin_storage::value_log_pages))
        .route("/admin/storage/namespaces", get(admin_storage::namespaces))
        .route("/admin/storage/namespaces/physical", get(admin_storage::physical_namespaces))
        // Admin / storage — per-namespace engine metadata (doc or KV, unified)
        .route("/admin/storage/stores/{ns}/kv-meta", get(admin_storage::store_kv_meta))
        // Admin / storage — system namespace
        .route("/admin/storage/system/stores", get(admin_storage::system_stores))
        .route("/admin/storage/system/stores/{ns}/meta", get(admin_storage::system_store_meta))
        .route("/admin/storage/index-waste", get(admin_storage::index_waste))
        // Admin / storage — operations (write)
        .route("/admin/storage/gc", post(admin_storage::trigger_gc))
        .route("/admin/storage/gc/wal", post(admin_storage::trigger_gc_wal))
        .route("/admin/storage/compact", post(admin_storage::trigger_compact))
        .route("/admin/storage/index-checkpoint", post(admin_storage::trigger_index_checkpoint))
        // ── Admin / indices — global monitoring ──────────────────────────────
        .route("/admin/indices/progress", get(admin_indices::progress_all))
        .route("/admin/indices/vector/queue/summary", get(admin_indices::vector_queue_summary))
        .route("/admin/indices/vector/queue/retried", get(admin_indices::vector_queue_retried))
        .route("/admin/indices/vector/query-cache", delete(admin_indices::vector_query_cache_clear))
        .route("/admin/indices/vector/corruption-metrics", get(admin_indices::vector_corruption_metrics))
        .route("/admin/indices/vector/reconcile", post(admin_indices::vector_reconcile))
        // Admin / indices — per-namespace monitoring & operations
        .route("/admin/indices/{ns}/progress", get(admin_indices::progress_ns))
        .route(
            "/admin/indices/{ns}/vector/corruption-metrics",
            get(admin_indices::vector_corruption_metrics_ns),
        )
        .route("/admin/indices/{ns}/attribute/reindex-all", post(admin_indices::attribute_reindex_all))
        .route("/admin/indices/{ns}/attribute/drop-all", delete(admin_indices::attribute_drop_all))
        .route(
            "/admin/indices/{ns}/attribute/{field}/reindex/{doc_id}",
            post(admin_indices::attribute_reindex_doc),
        )
        .route("/admin/indices/{ns}/{field}/blob-stats", get(admin_indices::field_blob_stats))
        .route("/admin/indices/{ns}/vector/reindex-all", post(admin_indices::vector_reindex_all))
        .route("/admin/indices/{ns}/vector/reindex/{doc_id}", post(admin_indices::vector_reindex_doc))
        .route("/admin/indices/{ns}/vector/reindex-failed", post(admin_indices::vector_reindex_failed))
        .route("/admin/indices/{ns}/vector/drop-all", delete(admin_indices::vector_drop_all))
        // Admin / indices — vector queue inspection
        .route("/admin/indices/{ns}/vector/queue", get(admin_indices::vector_queue_by_namespace))
        .route(
            "/admin/indices/{ns}/vector/queue/retried",
            get(admin_indices::vector_queue_retried_by_namespace),
        )
        .route(
            "/admin/indices/{ns}/vector/queue/{doc_id}",
            get(admin_indices::vector_queue_get_entry).delete(admin_indices::vector_queue_delete_entry),
        )
        .route(
            "/admin/indices/{ns}/vector/queue/{doc_id}/retry",
            post(admin_indices::vector_queue_retry_entry),
        )
        // ── Admin / store schema & data management ────────────────────────────
        .route("/admin/stores/{ns}/schema/export", get(admin_stores::export_schema))
        .route("/admin/stores/import", post(admin_stores::import_schema))
        .route("/admin/stores/{ns}/row-count", get(admin_stores::row_count))
        // ── Store lifecycle (unified: doc + KV, dispatched on store_type) ──────
        .route("/stores", get(stores::list).post(stores::create))
        .route("/stores/{ns}", delete(stores::drop_store))
        .route("/stores/{ns}/schema", get(stores::get_schema).patch(stores::amend_schema))
        // ── Index management (doc stores) ─────────────────────────────────────
        .route("/stores/{ns}/indices", get(indices::list_indices).post(indices::add_index))
        .route("/stores/{ns}/indices/vector", delete(indices::drop_vector_index))
        .route("/stores/{ns}/indices/{field}", delete(indices::drop_index))
        // ── Document CRUD (doc stores) ────────────────────────────────────────
        .route("/stores/{ns}/docs/{id}", get(docs::get_doc).put(docs::put_doc).delete(docs::delete_doc))
        .route("/stores/{ns}/docs", get(docs::range_query))
        .route("/stores/{ns}/docs/prefix", get(docs::prefix_scan))
        .route("/stores/{ns}/query", post(docs::index_query))
        // ── Semantic search (doc stores) ──────────────────────────────────────
        .route("/stores/{ns}/semantic-search", post(semantic_search::query))
        .route("/stores/{ns}/semantic-search/filtered", post(semantic_search::query_filtered))
        // ── KV CRUD (KV stores) ───────────────────────────────────────────────
        .route("/stores/{ns}/kv/{key}", get(kv::get_kv).put(kv::put_kv).delete(kv::delete_kv))
        .route("/stores/{ns}/kv", get(kv::range_kv))
        .route("/stores/{ns}/kv/prefix", get(kv::prefix_scan_kv))
        // ── KV semantic search (KV stores) ────────────────────────────────────
        .route("/stores/{ns}/kv/semantic-search", post(kv::search_kv_semantic))
}
