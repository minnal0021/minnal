//! Amending a store's schema, exporting it, and removing attributes and
//! indexes from it.

use super::super::*;

impl DocStore {
    // ── Admin: schema amendment ────────────────────────────────────────────

    /// Amend the non-indexed attribute declarations of an existing store.
    ///
    /// Only [`SchemaAmendment::AddAttribute`], [`SchemaAmendment::RemoveAttribute`],
    /// and [`SchemaAmendment::UpdateAttribute`] are supported.  Attempting to
    /// remove or update an attribute that is used by an active index returns
    /// [`DocStoreError::AttributeIsIndexed`] — drop the index first.
    pub async fn amend(&self, namespace: &str, mut amendment: SchemaAmendment) -> Result<(), DocStoreError> {
        let _lifecycle = self.lifecycle.lock(namespace).await;
        let current = self.load_schema(namespace)?;

        // A namespace without settings yet takes the server's defaults; one that
        // has them keeps its own, whatever the defaults are now.
        if current.vector_index.is_none()
            && let SchemaAmendment::AddEmbeddingAttribute { vector_index, .. } | SchemaAmendment::EnableVectorIndex { vector_index, .. } =
                &mut amendment
        {
            *vector_index = Some(crate::doc_store::schema::vector_settings::settle(
                None,
                vector_index.as_ref(),
                &self.index_defaults(),
            )?);
        }

        // Re-map SchemaError::AttributeIsIndexed to DocStoreError with namespace context
        let apply = |schema: &mut DocStoreSchema, amendment: SchemaAmendment| {
            schema.apply_amendment(amendment).map_err(|e| match e {
                SchemaError::AttributeIsIndexed { name } => DocStoreError::AttributeIsIndexed {
                    namespace: namespace.to_owned(),
                    field: name,
                },
                other => DocStoreError::Schema(other),
            })
        };

        // Try it on a copy first: an invalid amendment fails before anything is
        // written, and one that turns semantic search on needs its partition
        // seeded before the schema says it is semantic.
        let mut trial = DocStoreSchema::clone(&current);
        apply(&mut trial, amendment.clone())?;

        // An amendment that turns on semantic search (e.g. EnableVectorIndex, or an
        // embedding attribute) is unsupported without the `semantic-search` feature.
        #[cfg(not(feature = "semantic-search"))]
        if trial.semantic_search_enabled {
            return Err(DocStoreError::SemanticSearchNotCompiled);
        }
        // Only the amendment that turns semantic search on picks a model; others
        // must keep working even if this server no longer serves it.
        if trial.semantic_search_enabled && !current.semantic_search_enabled {
            let settings = trial.vector_settings()?;
            self.check_vector_model(&settings)?;
            // Idempotent: a namespace that already has a partition keeps it.
            self.seed_vector_index(namespace, &settings, &mut trial.vector_index).await?;
        }

        // Apply it to the schema as it is now, atomically with every other
        // schema write: a concurrent amendment cannot be lost.
        self.schemas.update_doc(namespace, |schema| apply(schema, amendment))
    }

    /// Return the schema for a namespace without the JSON round-trip overhead of [`list`].
    ///
    /// [`list`]: DocStore::list
    pub fn get_schema(&self, namespace: &str) -> Result<DocStoreSchema, DocStoreError> {
        let schema = self.view_schema(namespace)?;
        Ok(DocStoreSchema::clone(&schema))
    }

    /// Return the KV schema for a namespace without the JSON round-trip overhead of [`list_kv`].
    ///
    /// [`list_kv`]: DocStore::list_kv
    pub fn get_kv_schema(&self, namespace: &str) -> Result<KvStoreSchema, DocStoreError> {
        let schema = self.view_kv_schema(namespace)?;
        Ok(KvStoreSchema::clone(&schema))
    }

    /// Remove an attribute from the schema, and, if it is an embedding field,
    /// also remove it from `embedding_fields`.  When the removal empties
    /// `embedding_fields`, `semantic_search_enabled` is set to `false` and the
    /// updated schema is persisted in one atomic write.
    ///
    /// Returns `true` when the operation disabled semantic search (all embedding
    /// fields are now gone), signalling that the caller should trigger a
    /// background vector-index cleanup.
    ///
    /// Returns `Err(AttributeIsIndexed)` if the attribute is used by a field
    /// index — drop the index first.
    pub fn remove_attribute(&self, namespace: &str, field_name: &str) -> Result<bool, DocStoreError> {
        self.schemas.update_doc(namespace, |schema| {
            let was_embedding_field = schema.embedding_fields.contains(&field_name.to_owned());
            let will_disable_ss = was_embedding_field && schema.embedding_fields.len() == 1;

            schema
                .apply_amendment(SchemaAmendment::RemoveAttribute { name: field_name.to_owned() })
                .map_err(|e| match e {
                    SchemaError::AttributeIsIndexed { name } => DocStoreError::AttributeIsIndexed {
                        namespace: namespace.to_owned(),
                        field: name,
                    },
                    other => DocStoreError::Schema(other),
                })?;

            if was_embedding_field {
                schema.embedding_fields.retain(|f| f != field_name);
                if schema.embedding_fields.is_empty() {
                    schema.semantic_search_enabled = false;
                }
            }
            Ok(will_disable_ss)
        })
    }

    /// Drop every field index for a namespace and return their specs.
    ///
    /// Each field is demoted to a plain attribute (its data stays in stored
    /// documents) and its on-disk index files are deleted. With no field index
    /// left, the namespace's row map is reset, freeing every row ID. The caller
    /// can pass the returned specs back to [`add_index`] to rebuild them; to
    /// drop and rebuild in one crash-safe step, use
    /// [`reindex_all_attribute_indices`](DocStore::reindex_all_attribute_indices).
    ///
    /// [`add_index`]: DocStore::add_index
    pub fn drop_all_attribute_indices(&self, namespace: &str) -> Result<Vec<IndexSpec>, DocStoreError> {
        let schema = self.load_schema(namespace)?;
        let specs = schema.indices.clone();
        for spec in &specs {
            self.drop_index(namespace, &spec.field)?;
        }
        // No field index is left to reference a row ID: free them all (FR-006).
        if !specs.is_empty()
            && let Some(ns_id) = schema.ns_id
        {
            self.db.reset_rowmap(ns_id)?;
        }
        Ok(specs)
    }
}
