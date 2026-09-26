//! Document CRUD: `put`, `put_no_wal`, `delete`, `get`.

use super::*;

impl DocStore {
    // ── CRUD ──────────────────────────────────────────────────────────────
    //
    // Vector-index writes are decoupled from the write path.  When semantic
    // search is configured (`self.notify` is Some) and the namespace schema
    // has `semantic_search_enabled = true`, a pending-embed queue entry is
    // written atomically with the document — the background VecIndexWorker
    // processes it asynchronously.

    /// Insert or replace a document.
    ///
    /// The `id` must match the store's [`KeyType`].  The `doc` is serialized
    /// to JSON bytes before storage.
    ///
    /// When semantic search is configured and the namespace has
    /// `semantic_search_enabled = true`, the document is written first, then a
    /// pending embedding queue entry is enqueued.  The background
    /// `VecIndexWorker` processes the queue entry asynchronously — the vector
    /// index is eventually consistent with the document store.  A crash between
    /// the two writes leaves the document un-indexed until reconciliation
    /// re-enqueues it.
    pub async fn put(&self, namespace: &str, id: DocId, doc: serde_json::Value) -> Result<(), DocStoreError> {
        debug!("put namespace='{}' id={:?}", namespace, id);
        let schema = self.load_schema(namespace)?;
        schema.validate_doc(&doc)?;

        let key = id.to_bytes();
        let value = serde_json::to_vec(&doc).map_err(|e| DocStoreError::InvalidId(e.to_string()))?;

        #[cfg(feature = "semantic-search")]
        if let Some(notify) = &self.notify
            && schema.is_semantic_search_enabled()
        {
            let text = build_embedding_text(&doc, &schema.embedding_fields);
            // Write the document first, then the vector-queue op as a separate
            // single op (no cross-namespace atomicity needed).
            let ns = self.db.namespace(namespace.to_owned()).await?;
            ns.put(key.clone(), value).await?;
            if text.is_empty() {
                // No embedding text any more: drop the vectors, and any pending
                // embed, of the document's older text.
                self.clear_doc_vectors_if_any(namespace, &key).await?;
            } else {
                vector_kv::enqueue_embed(&self.db, namespace, &key, &text).await?;
                notify.notify_one();
            }
            return Ok(());
        }

        let ns = self.db.namespace(namespace.to_owned()).await?;
        ns.put(key, value).await?;
        Ok(())
    }

    /// Store a document without writing to the WAL (bulk-load path).
    ///
    /// The document write skips the WAL for throughput.  When semantic search
    /// is configured, the pending embedding queue entry is **WAL-backed** even
    /// on this path, so the worker can recover pending jobs after a crash and
    /// index any documents that survived the no-WAL write.
    pub async fn put_no_wal(&self, namespace: &str, id: DocId, doc: serde_json::Value) -> Result<(), DocStoreError> {
        let schema = self.load_schema(namespace)?;
        schema.validate_doc(&doc)?;

        let key = id.to_bytes();
        let value = serde_json::to_vec(&doc).map_err(|e| DocStoreError::InvalidId(e.to_string()))?;

        let ns = self.db.namespace(namespace.to_owned()).await?;
        ns.put_no_wal(key.clone(), value).await?;

        #[cfg(feature = "semantic-search")]
        if let Some(notify) = &self.notify
            && schema.is_semantic_search_enabled()
        {
            let text = build_embedding_text(&doc, &schema.embedding_fields);
            if text.is_empty() {
                self.clear_doc_vectors_if_any(namespace, &key).await?;
            } else {
                vector_kv::enqueue_embed(&self.db, namespace, &key, &text).await?;
                notify.notify_one();
            }
        }

        Ok(())
    }

    /// Delete a document by ID.  No-op if the document does not exist.
    ///
    /// When semantic search is configured and the namespace has
    /// `semantic_search_enabled = true`, the document is removed from the vector
    /// index first (a `Clear` tombstone in the queue, then the vectors — see
    /// [`DocStore::clear_doc_vectors`]), then the document is deleted.  Each is a
    /// separate single-op write; ordering derived data before the document means a
    /// crash between them leaves an un-indexed document (reconciliation cleans it
    /// up), never an orphaned vector.
    pub async fn delete(&self, namespace: &str, id: DocId) -> Result<(), DocStoreError> {
        debug!("delete namespace='{}' id={:?}", namespace, id);
        let key = id.to_bytes();

        #[cfg(feature = "semantic-search")]
        let schema = self.load_schema(namespace)?;
        #[cfg(feature = "semantic-search")]
        if schema.is_semantic_search_enabled() {
            self.clear_doc_vectors(namespace, &key).await?;
            let ns = self.db.namespace(namespace.to_owned()).await?;
            ns.delete(key).await?;
            return Ok(());
        }

        let ns = self.db.namespace(namespace.to_owned()).await?;
        ns.delete(key).await?;
        Ok(())
    }

    /// Retrieve a single document by its primary key.
    ///
    /// Returns `None` if no document with that ID exists.
    pub async fn get(&self, namespace: &str, id: DocId) -> Result<Option<serde_json::Value>, DocStoreError> {
        let ns = self.db.namespace(namespace.to_owned()).await?;
        match ns.get(id.to_bytes()).await? {
            None => Ok(None),
            Some(bytes) => {
                let doc = serde_json::from_slice(&bytes).map_err(|e| DocStoreError::InvalidId(e.to_string()))?;
                Ok(Some(doc))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc_store::store::test_support::*;

    // ── CRUD ────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_put_and_find_by_id() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create(make_schema("docs", vec![])).await.unwrap();

        let id = DocId::U64(42);
        let doc = serde_json::json!({"name": "Alice", "age": 30});
        store.put("docs", id, doc.clone()).await.unwrap();

        let found = store.get("docs", id).await.unwrap();
        assert_eq!(found, Some(doc));
    }

    #[tokio::test]
    async fn test_find_by_id_missing_returns_none() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create(make_schema("docs", vec![])).await.unwrap();

        let found = store.get("docs", DocId::U64(99)).await.unwrap();
        assert_eq!(found, None);
    }

    #[tokio::test]
    async fn test_delete_removes_document() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create(make_schema("docs", vec![])).await.unwrap();

        let id = DocId::U64(1);
        store.put("docs", id, serde_json::json!({"x": 1})).await.unwrap();
        store.delete("docs", id).await.unwrap();
        assert_eq!(store.get("docs", id).await.unwrap(), None);
    }
}
