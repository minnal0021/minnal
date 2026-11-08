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
        let lifecycle = self.lifecycle.lock(namespace).await;
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

        let apply = |schema: &mut DocStoreSchema, amendment: SchemaAmendment| apply_amendment(namespace, schema, amendment);

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
        // must keep working even if this server no longer serves it. It is a
        // `vector_enable` job, which seeds the partition before the schema says
        // the store is semantic.
        if trial.semantic_search_enabled && !current.semantic_search_enabled {
            let settings = trial.vector_settings()?;
            self.check_vector_model(&settings)?;
            drop(lifecycle);
            return self.enable_semantic_search(namespace, amendment).await;
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
    /// from `embedding_fields` too. When that empties `embedding_fields`,
    /// semantic search goes off and the vector index is removed, as a
    /// `vector_cleanup` job; this returns once it is gone.
    ///
    /// Returns `Err(AttributeIsIndexed)` if the attribute is used by a field
    /// index — drop the index first.
    pub async fn remove_attribute(&self, namespace: &str, field_name: &str) -> Result<(), DocStoreError> {
        match self.start_remove_attribute(namespace, field_name)? {
            Some(key) => self.run_job(namespace, key).await,
            None => Ok(()),
        }
    }

    /// [`remove_attribute`](Self::remove_attribute), returning the name of the
    /// `vector_cleanup` job it started, if it started one, once it is handed
    /// to the governor.
    pub fn begin_remove_attribute(&self, namespace: &str, field_name: &str) -> Result<Option<String>, DocStoreError> {
        if self.removal_disables_semantic_search(namespace, field_name)? {
            self.running_governor()?;
        }
        let Some(key) = self.start_remove_attribute(namespace, field_name)? else {
            return Ok(None);
        };
        self.submit_job(namespace, key)?;
        Ok(Some(key.kind.name().to_owned()))
    }

    /// Whether removing `field_name` takes `namespace`'s last embedding field.
    fn removal_disables_semantic_search(&self, namespace: &str, field_name: &str) -> Result<bool, DocStoreError> {
        let schema = self.load_schema(namespace)?;
        Ok(schema.embedding_fields.len() == 1 && schema.embedding_fields[0] == field_name)
    }

    fn start_remove_attribute(&self, namespace: &str, field_name: &str) -> Result<Option<JobKey>, DocStoreError> {
        let remove = || {
            self.schemas.update_doc(namespace, |schema| {
                schema
                    .apply_amendment(SchemaAmendment::RemoveAttribute { name: field_name.to_owned() })
                    .map_err(|e| match e {
                        SchemaError::AttributeIsIndexed { name } => DocStoreError::AttributeIsIndexed {
                            namespace: namespace.to_owned(),
                            field: name,
                        },
                        other => DocStoreError::Schema(other),
                    })?;
                if schema.embedding_fields.iter().any(|f| f == field_name) {
                    schema.embedding_fields.retain(|f| f != field_name);
                    if schema.embedding_fields.is_empty() {
                        schema.semantic_search_enabled = false;
                    }
                }
                Ok(())
            })
        };
        if self.removal_disables_semantic_search(namespace, field_name)? {
            Ok(Some(self.start_vector_cleanup(namespace, remove)?))
        } else {
            remove()?;
            Ok(None)
        }
    }
}

/// Apply `amendment` to `schema` of store `namespace`, naming the store in an
/// [`DocStoreError::AttributeIsIndexed`].
pub(in crate::doc_store) fn apply_amendment(namespace: &str, schema: &mut DocStoreSchema, amendment: SchemaAmendment) -> Result<(), DocStoreError> {
    schema.apply_amendment(amendment).map_err(|e| match e {
        SchemaError::AttributeIsIndexed { name } => DocStoreError::AttributeIsIndexed {
            namespace: namespace.to_owned(),
            field: name,
        },
        other => DocStoreError::Schema(other),
    })
}
