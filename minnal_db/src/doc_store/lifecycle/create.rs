//! Creating and listing document and KV stores.

use super::super::*;

impl DocStore {
    // ── Admin: create / list / drop ───────────────────────────────────────

    /// Create a new document store from the given schema.
    ///
    /// - Validates the schema.
    /// - Creates the namespace in minnal_db and records the `ns_id`.
    /// - Registers and activates all declared field indices.
    /// - Persists the schema (with `ns_id`) in `system_schema`.
    ///
    /// Returns [`DocStoreError::AlreadyExists`] if a store with that namespace
    /// is already registered.
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
        let _lifecycle = self.lifecycle.lock(&ns_name).await;
        if self.schemas.contains(&ns_name) {
            return Err(DocStoreError::AlreadyExists { namespace: ns_name });
        }

        info!(
            "creating namespace '{}' (key_type={:?}, indices={}, semantic_search={})",
            ns_name,
            schema.key_type,
            schema.indices.len(),
            schema.semantic_search_enabled
        );

        self.clear_orphaned_namespace(&ns_name).await?;
        // Create namespace — `namespace()` creates it if absent
        let ns_handle = self.db.namespace(ns_name.clone()).await?;
        let ns_id = ns_handle.id();
        schema.ns_id = Some(ns_id);

        // Register + activate field indices
        activate_indices(&self.db, ns_id, &schema).await?;

        // Seed the namespace's centres before the schema says it is semantic.
        if schema.semantic_search_enabled {
            let settings = schema.vector_settings()?;
            self.seed_vector_index(&ns_name, &settings, &mut schema.vector_index).await?;
        }

        // Persist schema (last: until then the store does not exist)
        self.schemas.insert(StoredSchema::Doc(Arc::new(schema)))?;
        info!("namespace '{}' created (ns_id={})", ns_name, ns_id);
        Ok(())
    }

    /// Deal with a namespace called `name` that has no schema, before a store takes
    /// the name. An empty one goes: it is all a create cut short leaves, and it may
    /// hold field-index registrations the new store must not inherit. One holding
    /// keys is refused with [`DocStoreError::NamespaceNotEmpty`]: it was written
    /// directly through [`Db`](crate::Db), and those keys are not this store's to
    /// delete or adopt. The caller holds the lifecycle lock and has seen no schema
    /// under this name. The default namespace always exists, so it is left alone.
    async fn clear_orphaned_namespace(&self, name: &str) -> Result<(), DocStoreError> {
        if name == crate::namespace::DEFAULT_NAMESPACE {
            return Ok(());
        }
        let Some(orphan) = self.db.existing_namespace(name.to_owned()).await? else {
            return Ok(());
        };
        if !orphan.scan(None, None, 1).await?.0.is_empty() {
            return Err(DocStoreError::NamespaceNotEmpty { namespace: name.to_owned() });
        }
        log::warn!("removing empty namespace '{name}', left without a schema, before creating a store with its name");
        self.db.remove_namespace(name.to_owned()).await?;
        Ok(())
    }

    /// List all known document stores.
    ///
    /// Returns each store's full schema as a [`serde_json::Value`] so callers
    /// can forward it to an HTTP response or log it without an intermediate
    /// struct conversion.
    pub fn list(&self) -> Result<Vec<serde_json::Value>, DocStoreError> {
        self.load_all_schemas()?
            .into_iter()
            .map(|s| serde_json::to_value(&*s).map_err(DocStoreError::from))
            .collect()
    }

    // ── KV store lifecycle ─────────────────────────────────────────────────

    /// Create a new KV store namespace from the given schema.
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
        let _lifecycle = self.lifecycle.lock(&ns_name).await;
        if self.schemas.contains(&ns_name) {
            return Err(DocStoreError::AlreadyExists { namespace: ns_name });
        }

        info!(
            "creating KV namespace '{}' (key_type={:?}, value_type={:?}, semantic_search={})",
            ns_name, schema.key_type, schema.value_type, schema.semantic_search_enabled
        );

        self.clear_orphaned_namespace(&ns_name).await?;
        let ns_handle = self.db.namespace(ns_name.clone()).await?;
        schema.ns_id = Some(ns_handle.id());
        if schema.is_semantic_search_enabled() {
            let settings = schema.vector_settings()?;
            self.seed_vector_index(&ns_name, &settings, &mut schema.vector_index).await?;
        }
        let ns_id = schema.ns_id;
        self.schemas.insert(StoredSchema::Kv(Arc::new(schema)))?;
        info!("KV namespace '{}' created (ns_id={:?})", ns_name, ns_id);
        Ok(())
    }

    /// List all known KV stores.
    pub fn list_kv(&self) -> Result<Vec<serde_json::Value>, DocStoreError> {
        self.load_all_kv_schemas()?
            .into_iter()
            .map(|s| serde_json::to_value(&*s).map_err(DocStoreError::from))
            .collect()
    }
}
