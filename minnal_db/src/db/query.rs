//! Predicate query execution against the field indices.
//!
//! Split out of `database.rs`, which had accumulated fourteen unrelated
//! responsibilities in one file. This one needs only the namespace registry and
//! the per-namespace store, so it lifts out cleanly.
//!
//! The whole surface returns [`QueryOutcome`] rather than a bare list of keys:
//! a field index can be *degraded* -- missing updates it can no longer recover --
//! while still being queryable, and serving results that look complete over one
//! is the failure FR-001 exists to remove.

use std::sync::Arc;

use crate::db::database::Database;
use crate::db::error::Result;
use crate::db::kv_store::KVStore;
use crate::db::namespace::{FieldId, QueryOutcome};

impl Database {
    /// Evaluate a query string against the active field indices of a namespace
    /// and return the raw keys of all matching documents.
    ///
    /// # Arguments
    /// * `namespace_id` — the namespace to query
    /// * `query_str`    — query string, e.g. `"age > 30 AND status = 'active'"`
    ///
    /// # How it works
    /// 1. Parses and evaluates `query_str` against the in-memory field indices,
    ///    producing a bitmap of matching row IDs.
    /// 2. Scans all keys in the namespace's KVStore and returns those whose
    ///    row ID (from the dense row map) appears in the bitmap.
    ///
    /// # Limitations
    /// Only fields activated via `activate_field_index` are queryable.
    /// Unindexed fields in the predicate produce a [`KVError::Query`]
    /// carrying a [`crate::index::query::QueryError::InactiveField`].
    pub fn query_keys(&self, namespace_id: u32, query_str: &str) -> Result<QueryOutcome> {
        let store = self.get_store(namespace_id)?;
        let (bitmap, touched) = self.evaluate_predicate(namespace_id, &store, query_str)?;
        let degraded_fields = self.degraded_among(namespace_id, &touched);
        let total = bitmap.len();

        if bitmap.is_empty() {
            return Ok(QueryOutcome {
                keys: Vec::new(),
                total: 0,
                degraded_fields,
            });
        }

        // Fast path: when a RowToKeyFn inverse is registered, reconstruct each
        // matching key directly from its row ID — O(|hits|), zero memory overhead,
        // crash-safe (no map to rebuild on restart).
        let keys = if let Some(ref inv) = *store.row_to_key_fn.read() {
            bitmap.iter().map(|row_id| inv(row_id)).collect()
        } else if store.rowmap_active() {
            // Fast path: the dense row map resolves each hit's key directly — O(|hits|).
            bitmap.iter().filter_map(|row_id| store.rowmap_key_for(row_id)).collect()
        } else {
            // Fallback (no inverse function): scan all keys and check bitmap membership.
            // Pre-existing O(n_keys) path retained for backward compatibility.
            store
                .keys()?
                .into_iter()
                .filter(|key| store.resolve_row_id_get(key).is_some_and(|id| bitmap.contains(id)))
                .collect()
        };

        Ok(QueryOutcome {
            keys,
            total,
            degraded_fields,
        })
    }

    /// Parse and evaluate a predicate, returning the matching row bitmap **and
    /// the set of fields the predicate actually referenced**.
    ///
    /// The touched-field set comes from the evaluator itself: it calls the
    /// index lookup closure exactly once per field it needs, so recording those
    /// ids is both free and exact — no second parse, and no risk of the two
    /// disagreeing about which fields a query touched.
    fn evaluate_predicate(
        &self,
        namespace_id: u32,
        store: &Arc<KVStore>,
        query_str: &str,
    ) -> Result<(crate::index::bitmap::RoaringBitmap, Vec<FieldId>)> {
        use crate::index::query::{SchemaMap, parse_and_evaluate};

        // Build the schema map: field_name → field_id, restricted to fields
        // that have an active in-memory index. Dropped fields remain in the
        // registry for field_id reuse but must not appear as queryable fields.
        let schema_map: SchemaMap = {
            let registry = self.registry.read();
            let ns_index = store.namespace_index.read();
            registry
                .schema(namespace_id)
                .map(|s| {
                    s.list_fields()
                        .into_iter()
                        .filter(|f| ns_index.get(f.field_id).is_some())
                        .map(|f| (f.field_name, f.field_id))
                        .collect()
                })
                .unwrap_or_default()
        };

        let touched = parking_lot::Mutex::new(Vec::new());
        let get_index = |field_id: u32| {
            touched.lock().push(field_id);
            let ns_index = store.namespace_index.read();
            ns_index.get(field_id).map(|e| Arc::clone(&e.index))
        };

        let bitmap = parse_and_evaluate(query_str, &schema_map, &get_index)?;
        let mut touched = touched.into_inner();
        touched.sort_unstable();
        touched.dedup();
        Ok((bitmap, touched))
    }

    /// Of the fields a query touched, which have an outstanding gap record.
    ///
    /// One marker read per touched field — a predicate references a handful of
    /// fields, not the whole schema, so this is bounded by the query rather than
    /// by the namespace.
    fn degraded_among(&self, namespace_id: u32, touched: &[FieldId]) -> Vec<FieldId> {
        touched
            .iter()
            .copied()
            .filter(|&field_id| self.index_manager.read_gap(namespace_id, field_id).is_some())
            .collect()
    }

    /// Evaluate a query and return `(page_keys, total)` where `total` is the
    /// full match count (bitmap cardinality) and `page_keys` contains at most
    /// `limit` keys starting from `offset` in iteration order.
    ///
    /// More efficient than [`query_keys`] when only a page of results is needed:
    /// - With a registered `RowToKeyFn`: O(offset + limit) key resolutions.
    /// - Fallback (no inverse): O(n_keys) scan but no full match list allocated.
    ///
    /// [`query_keys`]: Self::query_keys
    pub fn query_keys_paginated(&self, namespace_id: u32, query_str: &str, offset: usize, limit: usize) -> Result<QueryOutcome> {
        let store = self.get_store(namespace_id)?;
        let (bitmap, touched) = self.evaluate_predicate(namespace_id, &store, query_str)?;
        let degraded_fields = self.degraded_among(namespace_id, &touched);

        let total = bitmap.len();

        if total == 0 {
            return Ok(QueryOutcome {
                keys: Vec::new(),
                total: 0,
                degraded_fields,
            });
        }

        // Both fast paths window the bitmap with `iter_page`, not
        // `iter().skip(offset).take(limit)`. `iter` deserialises and
        // materialises every container it passes over, so walking a full result
        // set page by page costs O(n²). `iter_page` is bounded at both ends: it
        // reaches the offset by skipping whole containers on their cardinality,
        // and materialises at most `limit` values from each container it opens.

        // Fast path: RowToKeyFn registered — resolve only the page window.
        if let Some(ref inv) = *store.row_to_key_fn.read() {
            let keys: Vec<Vec<u8>> = bitmap.iter_page(offset, limit).map(|row_id| inv(row_id)).collect();
            return Ok(QueryOutcome {
                keys,
                total,
                degraded_fields,
            });
        }

        // Fast path: dense row map — resolve only the page window.
        if store.rowmap_active() {
            let keys: Vec<Vec<u8>> = bitmap
                .iter_page(offset, limit)
                .filter_map(|row_id| store.rowmap_key_for(row_id))
                .collect();
            return Ok(QueryOutcome {
                keys,
                total,
                degraded_fields,
            });
        }

        // Fallback: scan all keys, filter by bitmap membership, then window.
        // The `.skip(offset)` here walks *keys*, not the bitmap, so it gets no
        // benefit from `iter_page` — and it is not the bottleneck on this
        // path anyway: `store.keys()` already materialises the whole namespace
        // regardless of the page requested. Reached only when a namespace has
        // neither a RowToKeyFn nor a loaded row map.
        let all_keys = store.keys()?;
        let keys: Vec<Vec<u8>> = all_keys
            .into_iter()
            .filter(|key| store.resolve_row_id_get(key).is_some_and(|id| bitmap.contains(id)))
            .skip(offset)
            .take(limit)
            .collect();
        Ok(QueryOutcome {
            keys,
            total,
            degraded_fields,
        })
    }
}
