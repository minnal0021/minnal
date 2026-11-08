//! Creating and listing document and KV stores.

use super::super::*;

impl DocStore {
    // ── Admin: create / list / drop ───────────────────────────────────────

    /// Create a new document store from the given schema, and return once it
    /// is created.
    ///
    /// Checks what the caller should hear about first (the schema, the
    /// embedding model, a namespace of that name holding keys), then writes the
    /// store's record in [`StoreState::Creating`]: from then on the store is
    /// listed, and refuses requests until it is created. Its namespace, field
    /// indexes and, with semantic search, its vector index are then set up, and
    /// it moves to [`StoreState::Created`].
    ///
    /// Returns [`DocStoreError::AlreadyExists`] if a store with that namespace
    /// exists, [`DocStoreError::StoreDeleting`] while one of that name is being
    /// dropped. A setup that fails part-way leaves the store `Creating`; it is
    /// finished at the next open.
    pub async fn create(&self, mut schema: DocStoreSchema) -> Result<(), DocStoreError> {
        schema.validate()?;
        schema.settle_vector_index(&self.index_defaults())?;
        if schema.semantic_search_enabled {
            self.check_vector_model(&schema.vector_settings()?)?;
        }

        #[cfg(not(feature = "semantic-search"))]
        if schema.semantic_search_enabled {
            return Err(DocStoreError::SemanticSearchNotCompiled);
        }

        let ns_name = schema.namespace.clone();
        info!(
            "creating namespace '{}' (key_type={:?}, indices={}, semantic_search={})",
            ns_name,
            schema.key_type,
            schema.indices.len(),
            schema.semantic_search_enabled
        );
        {
            let _lifecycle = self.lifecycle.lock(&ns_name).await;
            if !self.schemas.contains(&ns_name) {
                self.refuse_a_namespace_holding_keys(&ns_name).await?;
            }
            // An imported schema may carry the id of another database's namespace.
            schema.ns_id = None;
            self.schemas.insert(StoredSchema::Doc(Arc::new(schema)), StoreState::Creating)?;
        }
        self.provision(&ns_name).await
    }

    /// List all known document stores.
    ///
    /// Returns each store's full schema as a [`serde_json::Value`] so callers
    /// can forward it to an HTTP response or log it without an intermediate
    /// struct conversion, with its lifecycle state under `"state"`. Every store
    /// is listed, whatever its state.
    pub fn list(&self) -> Result<Vec<serde_json::Value>, DocStoreError> {
        self.load_all_schemas()?
            .into_iter()
            .map(|s| self.with_state(&s.namespace, serde_json::to_value(&*s)?))
            .collect()
    }

    /// `value` with `namespace`'s lifecycle state added under `"state"`.
    fn with_state(&self, namespace: &str, mut value: serde_json::Value) -> Result<serde_json::Value, DocStoreError> {
        if let (serde_json::Value::Object(map), Some(state)) = (&mut value, self.schemas.state(namespace)) {
            map.insert(StoreState::KEY.to_owned(), serde_json::to_value(state)?);
        }
        Ok(value)
    }

    // ── KV store lifecycle ─────────────────────────────────────────────────

    /// Create a new KV store namespace from the given schema, and return once
    /// it is created; see [`create`] for the steps and the errors.
    ///
    /// Unlike [`create`], no field indices are registered.  The namespace
    /// stores raw bytes keyed by the declared `key_type`.
    ///
    /// [`create`]: DocStore::create
    pub async fn create_kv(&self, mut schema: KvStoreSchema) -> Result<(), DocStoreError> {
        schema.validate()?;
        schema.settle_vector_index(&self.index_defaults())?;
        if schema.is_semantic_search_enabled() {
            self.check_vector_model(&schema.vector_settings()?)?;
        }

        #[cfg(not(feature = "semantic-search"))]
        if schema.semantic_search_enabled {
            return Err(DocStoreError::SemanticSearchNotCompiled);
        }

        let ns_name = schema.namespace.clone();
        info!(
            "creating KV namespace '{}' (key_type={:?}, value_type={:?}, semantic_search={})",
            ns_name, schema.key_type, schema.value_type, schema.semantic_search_enabled
        );
        {
            let _lifecycle = self.lifecycle.lock(&ns_name).await;
            if !self.schemas.contains(&ns_name) {
                self.refuse_a_namespace_holding_keys(&ns_name).await?;
            }
            schema.ns_id = None;
            self.schemas.insert(StoredSchema::Kv(Arc::new(schema)), StoreState::Creating)?;
        }
        self.provision(&ns_name).await
    }

    /// List all known KV stores, each with its lifecycle state; see [`list`](Self::list).
    pub fn list_kv(&self) -> Result<Vec<serde_json::Value>, DocStoreError> {
        self.load_all_kv_schemas()?
            .into_iter()
            .map(|s| self.with_state(&s.namespace, serde_json::to_value(&*s)?))
            .collect()
    }
}
