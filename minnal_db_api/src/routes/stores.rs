//! Store lifecycle endpoints — a single unified `/stores` path for both
//! document stores and KV stores, dispatched on the schema's `store_type`:
//!
//! ```text
//! GET    /stores                 → list all stores (doc and KV)
//! POST   /stores                 → create a store (store_type in payload picks doc vs kv)
//! DELETE /stores/{ns}            → drop a store (kind resolved from its schema)
//! GET    /stores/{ns}/schema     → fetch a store's schema (doc or KV)
//! PATCH  /stores/{ns}/schema     → amend schema (doc stores only; KV → 409)
//! ```

use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use minnal_db::{AttributeType, DocStoreError, DocStoreSchema, KvStoreSchema, SchemaAmendment, StoreType};
use serde::Deserialize;
use tracing::{debug, error, info};

use crate::{AppState, error::AppError};

/// Read the mandatory `store_type` discriminant from a raw create/import
/// payload, returning a 400 if it is missing or unparseable.
pub(crate) fn store_type_from_value(body: &serde_json::Value) -> Result<StoreType, AppError> {
    body.get("store_type")
        .and_then(|v| serde_json::from_value::<StoreType>(v.clone()).ok())
        .ok_or_else(|| {
            AppError::from(DocStoreError::InvalidId(
                "payload is missing a valid 'store_type' (expected \"doc\" or \"kv\")".into(),
            ))
        })
}

pub async fn list(State(state): State<AppState>) -> Result<impl IntoResponse, AppError> {
    debug!("listing stores");
    // Combine both kinds; each schema carries its own `store_type` so callers
    // can tell them apart.
    let mut stores = state.store.list()?;
    stores.extend(state.store.list_kv()?);
    Ok(Json(stores))
}

/// `POST /stores` — create a doc or KV store, dispatched on the payload's
/// `store_type`.
pub async fn create(State(state): State<AppState>, Json(body): Json<serde_json::Value>) -> Result<impl IntoResponse, AppError> {
    match store_type_from_value(&body)? {
        StoreType::Doc => {
            let schema: DocStoreSchema = serde_json::from_value(body).map_err(|e| AppError::from(DocStoreError::from(e)))?;
            create_doc_schema(&state, schema).await
        }
        StoreType::Kv => {
            let schema: KvStoreSchema = serde_json::from_value(body).map_err(|e| AppError::from(DocStoreError::from(e)))?;
            create_kv_schema(&state, schema).await
        }
    }
}

/// Core doc-store creation, shared by `POST /stores` and schema import.
pub(crate) async fn create_doc_schema(state: &AppState, schema: DocStoreSchema) -> Result<StatusCode, AppError> {
    // Reject semantic-search-enabled stores when no cluster index is loaded.
    // Without the index, writes would attempt (and fail) to quantise embeddings,
    // leaving the namespace in a permanently broken state.
    if schema.semantic_search_enabled && state.cluster_index.is_none() {
        return Err(DocStoreError::EmbeddingFailed(
            "cannot create a semantic-search-enabled store: \
             cluster index is not loaded (check semantic_search.cluster_path in config)"
                .into(),
        )
        .into());
    }
    let ns = schema.namespace.clone();
    info!(namespace = %ns, semantic_search = schema.semantic_search_enabled, "creating store");
    state.store.create(schema).await?;
    // Reload from the persisted copy so the cache has the store-assigned ns_id.
    reload_schema(state, &ns).await;
    info!(namespace = %ns, "store created");
    Ok(StatusCode::CREATED)
}

pub async fn drop_store(State(state): State<AppState>, Path(ns): Path<String>) -> Result<impl IntoResponse, AppError> {
    info!(namespace = %ns, "dropping store");
    match state.store.store_type(&ns).map_err(|e| AppError::from(e).with_ns(&ns))? {
        StoreType::Doc => {
            state.store.remove(&ns).await?;
            state.schemas.write().await.remove(&ns);
        }
        StoreType::Kv => {
            state.store.remove_kv(&ns).await?;
            state.kv_schemas.write().await.remove(&ns);
        }
    }
    info!(namespace = %ns, "store dropped");
    Ok(StatusCode::NO_CONTENT)
}

pub async fn get_schema(State(state): State<AppState>, Path(ns): Path<String>) -> Result<impl IntoResponse, AppError> {
    match state.store.store_type(&ns).map_err(|e| AppError::from(e).with_ns(&ns))? {
        StoreType::Doc => Ok(Json(
            serde_json::to_value(state.store.get_schema(&ns)?).map_err(|e| AppError::from(DocStoreError::from(e)))?,
        )),
        StoreType::Kv => Ok(Json(
            serde_json::to_value(state.store.get_kv_schema(&ns).map_err(|e| AppError::from(e).with_ns(&ns))?)
                .map_err(|e| AppError::from(DocStoreError::from(e)))?,
        )),
    }
}

/// Request body for `PATCH /stores/{ns}/schema`.
///
/// Uses a `"op"` discriminant to select the amendment type:
/// ```json
/// {"op": "add_attribute",    "name": "email", "attr_type": "str", "description": "..."}
/// {"op": "remove_attribute", "name": "email"}
/// {"op": "update_attribute", "name": "email", "attr_type": "int"}
/// ```
#[allow(clippy::enum_variant_names)]
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub(crate) enum AmendRequest {
    AddAttribute {
        name: String,
        attr_type: AttributeType,
        description: Option<String>,
    },
    RemoveAttribute {
        name: String,
    },
    UpdateAttribute {
        name: String,
        attr_type: AttributeType,
        description: Option<String>,
    },
    AddEmbeddingAttribute {
        name: String,
        description: Option<String>,
    },
    EnableVectorIndex {
        fields: Vec<String>,
    },
}

impl From<AmendRequest> for SchemaAmendment {
    fn from(r: AmendRequest) -> Self {
        match r {
            AmendRequest::AddAttribute {
                name,
                attr_type,
                description,
            } => SchemaAmendment::AddAttribute {
                name,
                attr_type,
                description,
            },
            AmendRequest::RemoveAttribute { name } => SchemaAmendment::RemoveAttribute { name },
            AmendRequest::UpdateAttribute {
                name,
                attr_type,
                description,
            } => SchemaAmendment::UpdateAttribute {
                name,
                attr_type,
                description,
            },
            AmendRequest::AddEmbeddingAttribute { name, description } => SchemaAmendment::AddEmbeddingAttribute { name, description },
            AmendRequest::EnableVectorIndex { fields } => SchemaAmendment::EnableVectorIndex { fields },
        }
    }
}

pub async fn amend_schema(
    State(state): State<AppState>,
    Path(ns): Path<String>,
    Json(req): Json<AmendRequest>,
) -> Result<impl IntoResponse, AppError> {
    info!(namespace = %ns, "amending schema");

    // Enabling the vector index (single- or multi-field) requires semantic
    // search infrastructure.
    if matches!(&req, AmendRequest::AddEmbeddingAttribute { .. } | AmendRequest::EnableVectorIndex { .. }) && state.cluster_index.is_none() {
        return Err(DocStoreError::EmbeddingFailed(
            "cannot enable the vector index: cluster index is not loaded \
                 (check semantic_search.cluster_path in config)"
                .into(),
        )
        .into());
    }

    // For RemoveAttribute we use the dedicated path that returns whether the
    // last embedding field was removed (triggering background vector cleanup).
    if let AmendRequest::RemoveAttribute { name } = &req {
        let last_embedding_removed = state.store.remove_attribute(&ns, name)?;
        reload_schema(&state, &ns).await;

        if last_embedding_removed {
            // Check whether a cleanup is already running before spawning another.
            let already_running = state.vec_index_cleanup.lock().unwrap().contains(&ns);
            if !already_running {
                state.vec_index_cleanup.lock().unwrap().insert(ns.clone());
                info!(namespace = %ns, "last embedding field removed — cleaning up vector index data in background");
                let store = Arc::clone(&state.store);
                let ops_ref = Arc::clone(&state.vec_index_cleanup);
                let ns_clone = ns.clone();
                tokio::spawn(async move {
                    match store.drop_vector_index_data(&ns_clone).await {
                        Ok(()) => info!(namespace = %ns_clone, "vector index data cleanup complete"),
                        Err(e) => error!(namespace = %ns_clone, error = %e, "vector index data cleanup failed"),
                    }
                    ops_ref.lock().unwrap().remove(&ns_clone);
                });
            }
        }

        return Ok(StatusCode::NO_CONTENT);
    }

    state.store.amend(&ns, req.into())?;
    reload_schema(&state, &ns).await;
    Ok(StatusCode::NO_CONTENT)
}

/// Re-read one namespace's schema from the store and update the in-memory cache.
pub(crate) async fn reload_schema(state: &AppState, ns: &str) {
    if let Ok(list) = state.store.list() {
        for val in list {
            if let Ok(s) = serde_json::from_value::<DocStoreSchema>(val)
                && s.namespace == ns
            {
                state.schemas.write().await.insert(ns.to_owned(), s);
                return;
            }
        }
    }
}

// ── KV store creation (shared core) ─────────────────────────────────────────────

/// Core KV-store creation, shared by `POST /stores` and schema import.
pub(crate) async fn create_kv_schema(state: &AppState, schema: KvStoreSchema) -> Result<StatusCode, AppError> {
    if schema.semantic_search_enabled && state.cluster_index.is_none() {
        return Err(DocStoreError::EmbeddingFailed(
            "cannot create a semantic-search-enabled KV store: \
             cluster index is not loaded (check semantic_search.cluster_path in config)"
                .into(),
        )
        .into());
    }
    let ns = schema.namespace.clone();
    info!(namespace = %ns, value_type = ?schema.value_type, "creating KV store");
    state.store.create_kv(schema).await?;
    reload_kv_schema(state, &ns).await;
    info!(namespace = %ns, "KV store created");
    Ok(StatusCode::CREATED)
}

/// Re-read one KV namespace's schema and update the in-memory cache.
pub(crate) async fn reload_kv_schema(state: &AppState, ns: &str) {
    if let Ok(list) = state.store.list_kv() {
        for val in list {
            if let Ok(s) = serde_json::from_value::<KvStoreSchema>(val)
                && s.namespace == ns
            {
                state.kv_schemas.write().await.insert(ns.to_owned(), s);
                return;
            }
        }
    }
}
