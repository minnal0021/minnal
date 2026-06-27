//! Store lifecycle endpoints:
//!
//! ```text
//! GET    /stores                 → list all doc stores
//! POST   /stores                 → create doc store
//! DELETE /stores/{ns}            → drop doc store
//! GET    /stores/{ns}/schema     → fetch current doc-store schema
//! PATCH  /stores/{ns}/schema     → amend doc-store schema
//! GET    /kv-stores              → list all KV stores
//! POST   /kv-stores              → create KV store
//! DELETE /kv-stores/{ns}         → drop KV store
//! GET    /kv-stores/{ns}/schema  → fetch current KV-store schema
//! ```

use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use minnal_doc_store::{AttributeType, DocStoreError, DocStoreSchema, KvStoreSchema, SchemaAmendment};
use serde::Deserialize;
use tracing::{debug, error, info};

use crate::{AppState, error::AppError};

pub async fn list(State(state): State<AppState>) -> Result<impl IntoResponse, AppError> {
    debug!("listing stores");
    let stores = state.store.list()?;
    Ok(Json(stores))
}

pub async fn create(State(state): State<AppState>, Json(schema): Json<DocStoreSchema>) -> Result<impl IntoResponse, AppError> {
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
    reload_schema(&state, &ns).await;
    info!(namespace = %ns, "store created");
    Ok(StatusCode::CREATED)
}

pub async fn drop_store(State(state): State<AppState>, Path(ns): Path<String>) -> Result<impl IntoResponse, AppError> {
    info!(namespace = %ns, "dropping store");
    state.store.remove(&ns).await?;
    state.schemas.write().await.remove(&ns);
    info!(namespace = %ns, "store dropped");
    Ok(StatusCode::NO_CONTENT)
}

pub async fn get_schema(State(state): State<AppState>, Path(ns): Path<String>) -> Result<impl IntoResponse, AppError> {
    let schema = state.store.get_schema(&ns)?;
    Ok(Json(schema))
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

// ── KV store lifecycle ────────────────────────────────────────────────────────

pub async fn list_kv(State(state): State<AppState>) -> Result<impl IntoResponse, AppError> {
    debug!("listing KV stores");
    let stores = state.store.list_kv()?;
    Ok(Json(stores))
}

pub async fn create_kv(State(state): State<AppState>, Json(schema): Json<KvStoreSchema>) -> Result<impl IntoResponse, AppError> {
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
    reload_kv_schema(&state, &ns).await;
    info!(namespace = %ns, "KV store created");
    Ok(StatusCode::CREATED)
}

pub async fn get_kv_schema(State(state): State<AppState>, Path(ns): Path<String>) -> Result<impl IntoResponse, AppError> {
    let schema = state.store.get_kv_schema(&ns).map_err(|e| AppError::from(e).with_ns(&ns))?;
    Ok(Json(schema))
}

pub async fn drop_kv_store(State(state): State<AppState>, Path(ns): Path<String>) -> Result<impl IntoResponse, AppError> {
    info!(namespace = %ns, "dropping KV store");
    state.store.remove_kv(&ns).await?;
    state.kv_schemas.write().await.remove(&ns);
    info!(namespace = %ns, "KV store dropped");
    Ok(StatusCode::NO_CONTENT)
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
