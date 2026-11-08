//! Creating and listing document and KV stores.

use super::super::*;
use crate::namespace_lifecycle::Task;

impl DocStore {
    // ── Admin: create / list / drop ───────────────────────────────────────

    /// Create a new document store from the given schema, and return once it
    /// is created.
    ///
    /// Checks what the caller should hear about first (the schema, the
    /// embedding model, a namespace of that name holding keys), then writes the
    /// store's record in [`NamespaceState::Creating`]: from then on the store is
    /// listed, and refuses requests until it is created. Its namespace, field
    /// indexes and, with semantic search, its vector index are then set up, and
    /// it moves to [`NamespaceState::Created`].
    ///
    /// Returns [`DocStoreError::AlreadyExists`] if a store with that namespace
    /// exists, [`DocStoreError::StoreDeleting`] while one of that name is being
    /// dropped. A setup that fails part-way leaves the store `Creating`; the
    /// lifecycle governor runs it again (or, without one, the next open does).
    pub async fn create(&self, schema: DocStoreSchema) -> Result<(), DocStoreError> {
        let namespace = self.write_creating_doc(schema).await?;
        self.run_lifecycle_task(&namespace, Task::Provision).await
    }

    /// [`create`](Self::create), returning once the store's record is written
    /// in [`NamespaceState::Creating`]: the lifecycle governor sets it up in the
    /// background, and the store is ready once its state is `Created`. Fails
    /// with [`DocStoreError::GovernorUnavailable`] when the governor is not
    /// running, before anything is written.
    pub async fn begin_create(&self, schema: DocStoreSchema) -> Result<(), DocStoreError> {
        let governor = self.running_governor()?;
        let namespace = self.write_creating_doc(schema).await?;
        governor.submit(&namespace, Task::Provision)?;
        Ok(())
    }

    /// The checks and the `Creating` record of a document store's creation;
    /// returns its name.
    async fn write_creating_doc(&self, mut schema: DocStoreSchema) -> Result<String, DocStoreError> {
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
            self.schemas.insert(StoredSchema::Doc(Arc::new(schema)), NamespaceState::Creating)?;
        }
        Ok(ns_name)
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
            map.insert(NamespaceState::KEY.to_owned(), serde_json::to_value(state)?);
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
    pub async fn create_kv(&self, schema: KvStoreSchema) -> Result<(), DocStoreError> {
        let namespace = self.write_creating_kv(schema).await?;
        self.run_lifecycle_task(&namespace, Task::Provision).await
    }

    /// [`create_kv`](Self::create_kv), returning once the record is written;
    /// see [`begin_create`](Self::begin_create).
    pub async fn begin_create_kv(&self, schema: KvStoreSchema) -> Result<(), DocStoreError> {
        let governor = self.running_governor()?;
        let namespace = self.write_creating_kv(schema).await?;
        governor.submit(&namespace, Task::Provision)?;
        Ok(())
    }

    /// The checks and the `Creating` record of a KV store's creation; returns
    /// its name.
    async fn write_creating_kv(&self, mut schema: KvStoreSchema) -> Result<String, DocStoreError> {
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
            self.schemas.insert(StoredSchema::Kv(Arc::new(schema)), NamespaceState::Creating)?;
        }
        Ok(ns_name)
    }

    /// List all known KV stores, each with its lifecycle state; see [`list`](Self::list).
    pub fn list_kv(&self) -> Result<Vec<serde_json::Value>, DocStoreError> {
        self.load_all_kv_schemas()?
            .into_iter()
            .map(|s| self.with_state(&s.namespace, serde_json::to_value(&*s)?))
            .collect()
    }
}
