//! [`AsyncDb`]'s field-index methods: registering, activating and dropping
//! field indexes, their health and repair, and index queries.

use super::*;
use crate::namespace::NamespaceId;

impl AsyncDb {
    // ── Index ──────────────────────────────────────────────────────────

    /// Register an indexed field for a namespace and return its `FieldId`.
    pub fn register_index_field(&self, namespace_id: NamespaceId, field_name: &str, value_type: IndexValueType) -> Result<FieldId> {
        self.inner.inner.register_index_field(namespace_id, field_name, value_type)
    }

    /// Activate a registered field index, loading its snapshot and replaying
    /// any WAL entries since the last checkpoint.
    ///
    /// After this call every `put` / `delete` on the namespace automatically
    /// keeps the index current.
    pub async fn activate_field_index(
        &self,
        namespace_id: NamespaceId,
        field_id: FieldId,
        value_type: IndexValueType,
        extractor: ExtractorFn,
    ) -> Result<()> {
        let db = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || db.inner.activate_field_index(namespace_id, field_id, value_type, extractor))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// [`activate_field_index`](Self::activate_field_index) for a new index that
    /// a build is about to fill: it starts empty, with no WAL replay, and is safe
    /// on a store taking writes as long as the build then indexes every key.
    pub async fn activate_field_index_for_build(
        &self,
        namespace_id: NamespaceId,
        field_id: FieldId,
        value_type: IndexValueType,
        extractor: ExtractorFn,
    ) -> Result<()> {
        let db = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || db.inner.activate_field_index_for_build(namespace_id, field_id, value_type, extractor))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Remove a field index from the in-memory registry.
    ///
    /// After this call predicate queries that reference the field return an
    /// `InactiveField` error.  The on-disk checkpoint files are not touched.
    pub fn deactivate_field_index(&self, namespace_id: NamespaceId, field_id: FieldId) -> Result<()> {
        self.inner.inner.deactivate_field_index(namespace_id, field_id)
    }

    /// Permanently drop a field index: persist the drop, deregister it, and
    /// delete its on-disk directory.
    ///
    /// Use this rather than [`deactivate_field_index`](Self::deactivate_field_index)
    /// when the index is going away for good — the latter is an in-memory
    /// deregister that leaves every file in place. The field keeps its
    /// [`FieldId`] for reuse, but must be re-registered before it can be
    /// activated again. Synchronous — the work is a schema write plus a
    /// directory removal, not a scan.
    pub fn drop_field_index(&self, namespace_id: NamespaceId, field_id: FieldId) -> Result<()> {
        self.inner.inner.drop_field_index(namespace_id, field_id)
    }

    /// Free every row ID of a namespace by replacing its dense row map with an
    /// empty one; the next field builds allocate fresh dense IDs for the live
    /// keys only. Every field index of the namespace must already be dropped
    /// ([`drop_field_index`](Self::drop_field_index)) — otherwise this returns
    /// an error and changes nothing. Returns `false` for a namespace whose row
    /// IDs come from a key-derived `RowIdFn` (nothing to reset). Synchronous —
    /// a marker removal, a directory fsync and three small file truncations.
    pub fn reset_rowmap(&self, namespace_id: NamespaceId) -> Result<bool> {
        self.inner.inner.reset_rowmap(namespace_id)
    }

    /// Row IDs a namespace's dense row map has allocated — every key ever
    /// indexed, live or deleted — or `None` when the namespace uses a
    /// key-derived `RowIdFn` or no field index was ever activated.
    pub fn rowmap_ids_allocated(&self, namespace_id: NamespaceId) -> Result<Option<u64>> {
        self.inner.inner.rowmap_ids_allocated(namespace_id)
    }

    /// Register a custom row-ID function (and optionally its inverse) for a namespace.
    ///
    /// Call this before `activate_field_index` so WAL replay uses consistent row IDs.
    /// Providing `row_to_key_fn` enables O(|hits|) query resolution with zero memory overhead.
    ///
    /// See [`Database::set_row_id_fn`][crate::db::database::Database::set_row_id_fn]
    /// for full documentation.
    pub async fn set_row_id_fn(
        &self,
        namespace_id: NamespaceId,
        row_id_fn: crate::kv_store::index_set::RowIdFn,
        row_to_key_fn: Option<crate::kv_store::index_set::RowToKeyFn>,
    ) -> Result<()> {
        let db = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || db.inner.set_row_id_fn(namespace_id, row_id_fn, row_to_key_fn))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Return all indexed fields registered for a namespace, sorted by [`FieldId`].
    ///
    /// On a fresh open the list is loaded from the system catalog automatically,
    /// so callers do not need to re-call `register_index_field` after a restart.
    pub fn list_index_fields(&self, namespace_id: NamespaceId) -> Vec<crate::field_index::FieldMeta> {
        self.inner.inner.list_index_fields(namespace_id)
    }

    /// Return the number of distinct indexed values for a field, or `None` if
    /// the field is not currently active.
    pub fn field_index_distinct_count(&self, namespace_id: NamespaceId, field_id: FieldId) -> Option<usize> {
        self.inner.inner.field_index_distinct_count(namespace_id, field_id)
    }

    /// Reclaimable dead-space ratios `(bitmap_waste, keymap_waste)` for a field,
    /// or `None` if the field is not currently active.
    pub fn field_index_waste(&self, namespace_id: NamespaceId, field_id: FieldId) -> Option<(f64, f64)> {
        self.inner.inner.field_index_waste(namespace_id, field_id)
    }

    /// On-disk blob growth/waste metrics for a field — bitmap and keymap store
    /// sizes (logical vs. live bytes) and waste ratios — or `None` if the field
    /// is not currently active. Use it to monitor the append-only write
    /// amplification that low-cardinality fields suffer.
    pub fn field_index_blob_stats(&self, namespace_id: NamespaceId, field_id: FieldId) -> Option<crate::field_index::IndexBlobStats> {
        self.inner.inner.field_index_blob_stats(namespace_id, field_id)
    }

    /// Reindex a single field for a single key, re-deriving its value from the
    /// key's current stored bytes using the same logic as the put path. Touches
    /// only the named field. See [`crate::FieldReindexOutcome`].
    /// Report the health of every registered field index in a namespace.
    pub async fn index_health(&self, namespace_id: NamespaceId) -> Result<Vec<FieldIndexHealth>> {
        let db = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || db.inner.index_health(namespace_id))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Repair a degraded field index and clear its gap record.
    ///
    /// Offloaded to a blocking task: a full rebuild scans the namespace, so this
    /// must not run on the async runtime's worker threads.
    pub async fn repair_field_index(&self, namespace_id: NamespaceId, field_id: FieldId) -> Result<FieldRepairOutcome> {
        let db = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || db.inner.repair_field_index(namespace_id, field_id))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn reindex_field(&self, namespace_id: NamespaceId, field_id: FieldId, key: Vec<u8>) -> Result<FieldReindexOutcome> {
        let db = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || db.inner.reindex_field(namespace_id, field_id, &key))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// The configured field-index compaction threshold as a fraction (`0.0..1.0`).
    pub fn index_blob_waste_threshold(&self) -> f64 {
        self.inner.inner.index_blob_waste_threshold()
    }

    /// Evaluate a query string against the active field indices of a namespace
    /// and return the raw keys of all matching documents.
    pub async fn query_index(&self, namespace_id: NamespaceId, query: impl Into<String> + Send + 'static) -> Result<QueryOutcome> {
        let db = Arc::clone(&self.inner);
        let q = query.into();
        tokio::task::spawn_blocking(move || db.inner.query_keys(namespace_id, &q))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Like [`query_index`] but returns only the `[offset, offset+limit)` window
    /// of matching keys together with the full match count.
    ///
    /// [`query_index`]: AsyncDb::query_index
    pub async fn query_index_paginated(
        &self,
        namespace_id: NamespaceId,
        query: impl Into<String> + Send + 'static,
        offset: usize,
        limit: usize,
    ) -> Result<QueryOutcome> {
        let db = Arc::clone(&self.inner);
        let q = query.into();
        tokio::task::spawn_blocking(move || db.inner.query_keys_paginated(namespace_id, &q, offset, limit))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }
}
