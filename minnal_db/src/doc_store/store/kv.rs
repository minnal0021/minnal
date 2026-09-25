//! KV-store CRUD, scans, and KV semantic search.

use super::*;

impl DocStore {
    // ── KV CRUD ────────────────────────────────────────────────────────────

    /// Insert or replace a value in a KV namespace.
    ///
    /// `key` and `value` are JSON values typed according to the namespace schema.
    /// When semantic search is configured and the namespace has
    /// `semantic_search_enabled = true`, the KV value is written first, then a
    /// pending embedding queue entry is enqueued.  The background
    /// `VecIndexWorker` processes the queue asynchronously — the vector index
    /// is eventually consistent with the KV store.  A crash between the two
    /// writes leaves the value un-indexed until reconciliation re-enqueues it.
    pub async fn kv_put(&self, namespace: &str, key: &serde_json::Value, value: &serde_json::Value) -> Result<(), DocStoreError> {
        self.kv_put_inner(namespace, key, value, false).await
    }

    /// Insert or replace a value in a KV namespace, **bypassing the WAL**.
    ///
    /// Identical to [`kv_put`](Self::kv_put) except the value write skips the WAL
    /// for maximum throughput.  Data written this way is unrecoverable on a crash
    /// — only use during bulk loading where re-running the load is acceptable.
    /// The embed marker (when semantic search is enabled) is still enqueued
    /// through the WAL, matching the document-store `put_no_wal` behaviour.
    pub async fn kv_put_no_wal(&self, namespace: &str, key: &serde_json::Value, value: &serde_json::Value) -> Result<(), DocStoreError> {
        self.kv_put_inner(namespace, key, value, true).await
    }

    /// Shared body for [`kv_put`](Self::kv_put) and
    /// [`kv_put_no_wal`](Self::kv_put_no_wal); `skip_wal` selects the value-write
    /// durability path.
    async fn kv_put_inner(&self, namespace: &str, key: &serde_json::Value, value: &serde_json::Value, skip_wal: bool) -> Result<(), DocStoreError> {
        let schema = self.load_kv_schema(namespace)?;
        let key_bytes = schema.key_type.serialize_key(key)?;
        let value_bytes = schema.value_type.serialize_value(value)?;
        let ns = self.db.namespace(namespace.to_owned()).await?;

        if skip_wal {
            ns.put_no_wal(key_bytes.clone(), value_bytes).await?;
        } else {
            ns.put(key_bytes.clone(), value_bytes).await?;
        }

        #[cfg(feature = "semantic-search")]
        if let Some(notify) = &self.notify
            && schema.is_semantic_search_enabled()
            && let Some(text) = value.as_str()
            && !text.is_empty()
        {
            // Enqueue the embed marker as a separate single op (no cross-namespace
            // atomicity needed — see kv_put docs).
            vector_kv::enqueue_embed(&self.db, namespace, &key_bytes, text).await?;
            notify.notify_one();
        }

        Ok(())
    }

    /// Retrieve a value by key from a KV namespace.
    ///
    /// Returns `None` when the key does not exist.
    pub async fn kv_get(&self, namespace: &str, key: &serde_json::Value) -> Result<Option<serde_json::Value>, DocStoreError> {
        let schema = self.load_kv_schema(namespace)?;
        let key_bytes = schema.key_type.serialize_key(key)?;
        let ns = self.db.namespace(namespace.to_owned()).await?;
        match ns.get(key_bytes).await? {
            Some(bytes) => Ok(Some(schema.value_type.deserialize_value(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Retrieve a value by its raw URL-path key string from a KV namespace.
    ///
    /// Convenience wrapper around [`kv_get`] for HTTP handlers that receive
    /// the key as a URL path segment.
    ///
    /// [`kv_get`]: DocStore::kv_get
    pub async fn kv_get_by_str(&self, namespace: &str, raw_key: &str) -> Result<Option<serde_json::Value>, DocStoreError> {
        let schema = self.load_kv_schema(namespace)?;
        let key_bytes = schema.key_type.serialize_key_from_str(raw_key)?;
        let ns = self.db.namespace(namespace.to_owned()).await?;
        match ns.get(key_bytes).await? {
            Some(bytes) => Ok(Some(schema.value_type.deserialize_value(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Delete a key from a KV namespace.  No-op when the key does not exist.
    ///
    /// When semantic search is configured and the namespace has
    /// `semantic_search_enabled = true`, the pending queue entry and the vector
    /// index are removed first, then the KV value is deleted.  Each is a separate
    /// single-op write; ordering derived data before the value means a crash
    /// between them leaves an un-indexed value (reconciliation cleans it up),
    /// never an orphaned vector.
    pub async fn kv_delete(&self, namespace: &str, raw_key: &str) -> Result<(), DocStoreError> {
        let schema = self.load_kv_schema(namespace)?;
        let key_bytes = schema.key_type.serialize_key_from_str(raw_key)?;

        #[cfg(feature = "semantic-search")]
        if schema.is_semantic_search_enabled() {
            vector_kv::remove_queue_entry(&self.db, namespace, &key_bytes).await?;
            vector_kv::delete_vector(&self.db, namespace, &key_bytes).await?;
            let ns = self.db.namespace(namespace.to_owned()).await?;
            ns.delete(key_bytes).await?;
            return Ok(());
        }

        let ns = self.db.namespace(namespace.to_owned()).await?;
        ns.delete(key_bytes).await?;
        Ok(())
    }

    /// Return entries in `[start, end)` from a KV namespace, one cursor page at a time.
    ///
    /// Pass `end = None` for an open-ended scan to the last key, and `cursor = None`
    /// for the first page; thereafter pass back the previous page's `next_cursor`.
    /// Only the page's values are resolved from the value log — memory stays O(limit),
    /// not O(total matches). Results are ordered by key ascending.
    pub async fn kv_scan_range(
        &self,
        namespace: &str,
        start: &str,
        end: Option<&str>,
        cursor: Option<Vec<u8>>,
        limit: usize,
    ) -> Result<CursorPage<(serde_json::Value, serde_json::Value)>, DocStoreError> {
        let schema = self.load_kv_schema(namespace)?;
        let start_bytes = schema.key_type.serialize_key_from_str(start)?;
        let end_bytes = end.map(|e| schema.key_type.serialize_key_from_str(e)).transpose()?;
        let scan_start = cursor.unwrap_or(start_bytes);
        let ns = self.db.namespace(namespace.to_owned()).await?;
        let (pairs, next_cursor) = ns.scan(Some(scan_start), end_bytes, limit).await?;

        let results = pairs
            .into_iter()
            .map(|(k, v)| -> Result<_, DocStoreError> {
                let key = schema.key_type.deserialize_key(&k)?;
                let value = schema.value_type.deserialize_value(&v)?;
                Ok((key, value))
            })
            .collect::<Result<_, _>>()?;

        Ok(CursorPage::new(results, next_cursor))
    }

    /// Return entries whose key starts with `prefix` from a KV namespace, one cursor
    /// page at a time.
    ///
    /// For `key_type = str` the prefix is a plain string matched against the
    /// UTF-8 key bytes.  For `key_type = int` the prefix is a decimal integer
    /// serialised as big-endian bytes (i.e. an exact-key prefix scan). The prefix
    /// is scanned as the range `[prefix, prefix⁺)`, so only the page's keys (not the
    /// whole keyspace tail) are resolved. Pass back `next_cursor` for the next page.
    pub async fn kv_scan_prefix(
        &self,
        namespace: &str,
        prefix: &str,
        cursor: Option<Vec<u8>>,
        limit: usize,
    ) -> Result<CursorPage<(serde_json::Value, serde_json::Value)>, DocStoreError> {
        let schema = self.load_kv_schema(namespace)?;
        let prefix_bytes = schema.key_type.serialize_key_from_str(prefix)?;
        let end_bytes = prefix_upper_bound(&prefix_bytes);
        let scan_start = cursor.unwrap_or_else(|| prefix_bytes.clone());
        let ns = self.db.namespace(namespace.to_owned()).await?;
        let (pairs, next_cursor) = ns.scan(Some(scan_start), end_bytes, limit).await?;

        let results = pairs
            .into_iter()
            .map(|(k, v)| -> Result<_, DocStoreError> {
                let key = schema.key_type.deserialize_key(&k)?;
                let value = schema.value_type.deserialize_value(&v)?;
                Ok((key, value))
            })
            .collect::<Result<_, _>>()?;

        Ok(CursorPage::new(results, next_cursor))
    }

    /// Run an ANN semantic search against a KV namespace with `value_type = str`.
    ///
    /// `ranking` overrides the configured result ordering for this call (see
    /// [`DocStore::effective_ranking`]).
    ///
    /// Returns [`DocStoreError::EmbeddingFailed`] when no [`SemanticSearchContext`]
    /// is configured, when the namespace does not have `semantic_search_enabled`,
    /// or when the embedding service call fails, and
    /// [`DocStoreError::InvalidRanking`] if `ranking` yields invalid params.
    #[cfg(feature = "semantic-search")]
    pub async fn kv_search_semantic(
        &self,
        namespace: &str,
        query_text: &str,
        top_k: Option<usize>,
        ranking: &crate::semantic_search::service::RankingOverride,
        pagination: crate::doc_store::pagination::Pagination,
    ) -> Result<crate::doc_store::pagination::Page<crate::semantic_search::index::vector_index::QueryResult>, DocStoreError> {
        let ctx = self
            .semantic_ctx
            .as_ref()
            .ok_or_else(|| DocStoreError::EmbeddingFailed("semantic search not configured on this store".into()))?;
        let opts = crate::semantic_search::service::SearchOptions {
            top_k,
            ranking: Some(ctx.config.ranking.with_override(ranking)?),
        };

        let schema = self.load_kv_schema(namespace)?;
        if !schema.is_semantic_search_enabled() {
            return Err(DocStoreError::EmbeddingFailed(format!(
                "KV namespace '{namespace}' does not have semantic_search_enabled"
            )));
        }

        let (query_dense, query_sparse) = self.cached_query_embeddings(ctx, query_text).await?;

        let db_store = vector_kv::DbVectorStore::new(&self.db, namespace)
            .await
            .map_err(|e| DocStoreError::EmbeddingFailed(e.to_string()))?;
        let all = crate::semantic_search::service::search(
            &ctx.config,
            namespace,
            &ctx.cluster_index,
            &query_sparse,
            &query_dense,
            &db_store,
            None::<fn(&[u8]) -> bool>,
            opts,
        )
        .await;

        Ok(crate::doc_store::pagination::Page::from_vec(all, pagination))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc_store::store::test_support::*;

    // ── KV store helpers ────────────────────────────────────────────────────

    // ── KV CRUD ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_kv_put_get_str_key_str_value() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Str, KvValueType::Str)).await.unwrap();

        let key = serde_json::json!("hello");
        let val = serde_json::json!("world");
        store.kv_put("ns", &key, &val).await.unwrap();

        let got = store.kv_get("ns", &key).await.unwrap();
        assert_eq!(got, Some(val));
    }

    #[tokio::test]
    async fn test_kv_put_no_wal_get_round_trip() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Str, KvValueType::Str)).await.unwrap();

        let key = serde_json::json!("hello");
        let val = serde_json::json!("world");
        // The no-WAL path must be readable in-process exactly like the WAL path;
        // only crash-durability differs.
        store.kv_put_no_wal("ns", &key, &val).await.unwrap();

        let got = store.kv_get("ns", &key).await.unwrap();
        assert_eq!(got, Some(val));
    }

    #[tokio::test]
    async fn test_kv_put_get_int_key_int_value() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Int, KvValueType::Int)).await.unwrap();

        let key = serde_json::json!(99i64);
        let val = serde_json::json!(-42i64);
        store.kv_put("ns", &key, &val).await.unwrap();

        let got = store.kv_get("ns", &key).await.unwrap();
        assert_eq!(got, Some(val));
    }

    #[tokio::test]
    async fn test_kv_put_get_f32_value() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Str, KvValueType::F32)).await.unwrap();

        let key = serde_json::json!("temp");
        let val = serde_json::json!(98.5f32);
        store.kv_put("ns", &key, &val).await.unwrap();

        let got = store.kv_get("ns", &key).await.unwrap().unwrap();
        let diff = (got.as_f64().unwrap() as f32 - 98.5f32).abs();
        assert!(diff < f32::EPSILON, "f32 value mismatch: {got}");
    }

    #[tokio::test]
    async fn test_kv_put_get_vec_f32_value() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Str, KvValueType::VecF32)).await.unwrap();

        let key = serde_json::json!("embedding");
        let val = serde_json::json!([1.0f32, 0.5f32, -1.0f32]);
        store.kv_put("ns", &key, &val).await.unwrap();

        let got = store.kv_get("ns", &key).await.unwrap().unwrap();
        let arr = got.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        let expected = [1.0f32, 0.5, -1.0];
        for (v, exp) in arr.iter().zip(expected.iter()) {
            assert!((v.as_f64().unwrap() as f32 - exp).abs() < f32::EPSILON);
        }
    }

    #[tokio::test]
    async fn test_kv_put_overwrites_existing() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Str, KvValueType::Str)).await.unwrap();

        let key = serde_json::json!("k");
        store.kv_put("ns", &key, &serde_json::json!("v1")).await.unwrap();
        store.kv_put("ns", &key, &serde_json::json!("v2")).await.unwrap();

        assert_eq!(store.kv_get("ns", &key).await.unwrap(), Some(serde_json::json!("v2")));
    }

    #[tokio::test]
    async fn test_kv_get_missing_key_returns_none() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Str, KvValueType::Str)).await.unwrap();

        let got = store.kv_get("ns", &serde_json::json!("ghost")).await.unwrap();
        assert_eq!(got, None);
    }

    #[tokio::test]
    async fn test_kv_delete_removes_entry() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Str, KvValueType::Str)).await.unwrap();

        let key = serde_json::json!("gone");
        store.kv_put("ns", &key, &serde_json::json!("value")).await.unwrap();
        store.kv_delete("ns", "gone").await.unwrap();

        assert_eq!(store.kv_get("ns", &key).await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_kv_get_by_str_str_key() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Str, KvValueType::Str)).await.unwrap();

        store
            .kv_put("ns", &serde_json::json!("mykey"), &serde_json::json!("myval"))
            .await
            .unwrap();
        let got = store.kv_get_by_str("ns", "mykey").await.unwrap();
        assert_eq!(got, Some(serde_json::json!("myval")));
    }

    #[tokio::test]
    async fn test_kv_get_by_str_int_key() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Int, KvValueType::Str)).await.unwrap();

        store.kv_put("ns", &serde_json::json!(7i64), &serde_json::json!("seven")).await.unwrap();
        let got = store.kv_get_by_str("ns", "7").await.unwrap();
        assert_eq!(got, Some(serde_json::json!("seven")));
    }

    #[tokio::test]
    async fn test_kv_wrong_value_type_rejected() {
        let db_dir = TempDir::new().unwrap();
        let schema_dir = TempDir::new().unwrap();
        let store = open_fresh(db_dir.path(), schema_dir.path()).await;
        store.create_kv(make_kv_schema("ns", KvKeyType::Str, KvValueType::Int)).await.unwrap();

        // Put a string into an Int-typed namespace.
        let err = store
            .kv_put("ns", &serde_json::json!("k"), &serde_json::json!("not-int"))
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            DocStoreError::Schema(crate::doc_store::error::SchemaError::KvValueTypeMismatch { .. })
        ));
    }
}
