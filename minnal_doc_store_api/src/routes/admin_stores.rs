//! Admin endpoints for store schema and data management (doc and KV, unified):
//!
//! ```text
//! GET  /admin/stores/{ns}/schema/export  → download a store's schema as JSON attachment
//! POST /admin/stores/import              → create a store from an exported schema (store_type in payload)
//! GET  /admin/stores/{ns}/row-count      → number of documents in the namespace
//! ```

use axum::{
    Json,
    extract::{Path, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::IntoResponse,
};
use minnal_db::{DocStoreError, DocStoreSchema, KvStoreSchema, StoreType};
use tracing::info;

use crate::{AppState, error::AppError};

use super::stores::{create_doc_schema, create_kv_schema, store_type_from_value};

/// Build a JSON attachment response for an exported schema.
fn schema_attachment(ns: &str, json: Vec<u8>) -> Result<(StatusCode, HeaderMap, Vec<u8>), AppError> {
    let disposition = format!("attachment; filename=\"{ns}-schema.json\"");
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&disposition).map_err(|_| AppError::from(DocStoreError::InvalidId(disposition)))?,
    );
    Ok((StatusCode::OK, headers, json))
}

// ── GET /admin/stores/{ns}/schema/export ─────────────────────────────────────

pub async fn export_schema(State(state): State<AppState>, Path(ns): Path<String>) -> Result<impl IntoResponse, AppError> {
    let json = match state.store.store_type(&ns).map_err(|e| AppError::from(e).with_ns(&ns))? {
        StoreType::Doc => serde_json::to_vec_pretty(&state.store.get_schema(&ns)?).map_err(|e| AppError::from(DocStoreError::from(e)))?,
        StoreType::Kv => serde_json::to_vec_pretty(&state.store.get_kv_schema(&ns).map_err(|e| AppError::from(e).with_ns(&ns))?)
            .map_err(|e| AppError::from(DocStoreError::from(e)))?,
    };
    schema_attachment(&ns, json)
}

// ── POST /admin/stores/import ─────────────────────────────────────────────────

pub async fn import_schema(State(state): State<AppState>, Json(body): Json<serde_json::Value>) -> Result<impl IntoResponse, AppError> {
    let result = match store_type_from_value(&body)? {
        StoreType::Doc => {
            let mut schema: DocStoreSchema = serde_json::from_value(body).map_err(|e| AppError::from(DocStoreError::from(e)))?;
            let ns = schema.namespace.clone();
            info!(namespace = %ns, "importing doc schema");
            // ns_id is an internal assignment made at creation time — strip any value
            // carried in the exported file so the store assigns a fresh one.
            schema.ns_id = None;
            // A field that has been indexed must not also appear in attributes
            // (attributes is the non-indexed list). Schemas written before this
            // invariant was enforced may carry the field in both; drop it so
            // validate() passes.
            let indexed: std::collections::HashSet<&str> = schema.indices.iter().map(|i| i.field.as_str()).collect();
            schema.attributes.retain(|a| !indexed.contains(a.name.as_str()));
            create_doc_schema(&state, schema).await?
        }
        StoreType::Kv => {
            let mut schema: KvStoreSchema = serde_json::from_value(body).map_err(|e| AppError::from(DocStoreError::from(e)))?;
            info!(namespace = %schema.namespace, "importing KV schema");
            schema.ns_id = None;
            create_kv_schema(&state, schema).await?
        }
    };
    Ok(result)
}

// ── GET /admin/stores/{ns}/row-count ─────────────────────────────────────────

pub async fn row_count(State(state): State<AppState>, Path(ns): Path<String>) -> Result<impl IntoResponse, AppError> {
    let count = state.store.count_docs(&ns).await.map_err(|e| AppError::from(e).with_ns(&ns))?;
    Ok(Json(serde_json::json!({ "namespace": ns, "count": count })))
}
