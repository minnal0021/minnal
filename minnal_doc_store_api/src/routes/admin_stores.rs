//! Admin endpoints for doc-store and KV-store schema and data management:
//!
//! ```text
//! GET  /admin/stores/{ns}/schema/export     → download doc-store schema as JSON attachment
//! POST /admin/stores/import                 → create doc store from an exported schema
//! GET  /admin/stores/{ns}/row-count         → number of documents in the namespace
//! GET  /admin/kv-stores/{ns}/schema/export  → download KV-store schema as JSON attachment
//! POST /admin/kv-stores/import              → create KV store from an exported schema
//! ```

use axum::{
    Json,
    extract::{Path, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::IntoResponse,
};
use minnal_doc_store::DocStoreError;
use tracing::info;

use crate::{AppState, error::AppError};

use super::stores::{create, create_kv};

// ── GET /admin/stores/{ns}/schema/export ─────────────────────────────────────

pub async fn export_schema(State(state): State<AppState>, Path(ns): Path<String>) -> Result<impl IntoResponse, AppError> {
    let schema = state.store.get_schema(&ns).map_err(|e| AppError::from(e).with_ns(&ns))?;
    let json = serde_json::to_vec_pretty(&schema).map_err(|e| AppError::from(DocStoreError::from(e)))?;

    let filename = format!("{ns}-schema.json");
    let disposition = format!("attachment; filename=\"{filename}\"");

    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&disposition).map_err(|_| AppError::from(DocStoreError::InvalidId(disposition)))?,
    );

    Ok((StatusCode::OK, headers, json))
}

// ── POST /admin/stores/import ─────────────────────────────────────────────────

pub async fn import_schema(state: State<AppState>, Json(mut schema): Json<minnal_doc_store::DocStoreSchema>) -> Result<impl IntoResponse, AppError> {
    let ns = schema.namespace.clone();
    info!(namespace = %ns, "importing schema");
    // ns_id is an internal assignment made at creation time — strip any value
    // carried in the exported file so the store assigns a fresh one.
    schema.ns_id = None;
    // A field that has been indexed must not also appear in attributes (attributes
    // is the non-indexed list).  Schemas written before this invariant was enforced
    // may carry the field in both; drop it from attributes so validate() passes.
    let indexed: std::collections::HashSet<&str> = schema.indices.iter().map(|i| i.field.as_str()).collect();
    schema.attributes.retain(|a| !indexed.contains(a.name.as_str()));
    // Delegate to the existing create handler — identical validation and lifecycle.
    let result = create(state, Json(schema)).await?;
    info!(namespace = %ns, "schema imported");
    Ok(result)
}

// ── GET /admin/kv-stores/{ns}/schema/export ──────────────────────────────────

pub async fn export_kv_schema(State(state): State<AppState>, Path(ns): Path<String>) -> Result<impl IntoResponse, AppError> {
    let schema = state.store.get_kv_schema(&ns).map_err(|e| AppError::from(e).with_ns(&ns))?;
    let json = serde_json::to_vec_pretty(&schema).map_err(|e| AppError::from(DocStoreError::from(e)))?;

    let filename = format!("{ns}-kv-schema.json");
    let disposition = format!("attachment; filename=\"{filename}\"");

    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&disposition).map_err(|_| AppError::from(DocStoreError::InvalidId(disposition)))?,
    );

    Ok((StatusCode::OK, headers, json))
}

// ── POST /admin/kv-stores/import ──────────────────────────────────────────────

pub async fn import_kv_schema(
    state: State<AppState>,
    Json(mut schema): Json<minnal_doc_store::KvStoreSchema>,
) -> Result<impl IntoResponse, AppError> {
    let ns = schema.namespace.clone();
    info!(namespace = %ns, "importing KV schema");
    // ns_id is an internal assignment made at creation time — strip any value
    // carried in the exported file so the store assigns a fresh one.
    schema.ns_id = None;
    // Delegate to the existing create_kv handler — identical validation and lifecycle.
    let result = create_kv(state, Json(schema)).await?;
    info!(namespace = %ns, "KV schema imported");
    Ok(result)
}

// ── GET /admin/stores/{ns}/row-count ─────────────────────────────────────────

pub async fn row_count(State(state): State<AppState>, Path(ns): Path<String>) -> Result<impl IntoResponse, AppError> {
    let count = state.store.count_docs(&ns).await.map_err(|e| AppError::from(e).with_ns(&ns))?;
    Ok(Json(serde_json::json!({ "namespace": ns, "count": count })))
}
