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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::namespace::DEFAULT_NAMESPACE_ID;
    use crate::db::test_support::*;
    use tempfile::TempDir;

    /// Item 13: a put that adds/removes the indexed field (absent↔present) must
    /// update the index correctly via the targeted path — the row joins the new
    /// value's bucket and leaves whatever it was in (including "nothing").
    #[test]
    fn test_field_index_update_field_appears_and_disappears() {
        use crate::db::namespace_index::ExtractorFn;
        use crate::index::{IndexValue, IndexValueType};
        use std::sync::Arc;

        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = DEFAULT_NAMESPACE_ID;
        let field_id = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();
        let extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Str(v["status"].as_str()?.to_string()))
        });
        db.activate_field_index(ns, field_id, IndexValueType::Str, extractor).unwrap();

        // Field absent → not indexed.
        db.put(b"d:1", br#"{"other":1}"#).unwrap();
        assert!(db.query_keys(ns, "status = \"active\"").unwrap().keys.is_empty());

        // Field appears (None → Some): row joins the "active" bucket.
        db.put(b"d:1", br#"{"status":"active"}"#).unwrap();
        assert_eq!(db.query_keys(ns, "status = \"active\"").unwrap().keys, vec![b"d:1".to_vec()]);

        // Field disappears (Some → None): row must leave the bucket.
        db.put(b"d:1", br#"{"other":2}"#).unwrap();
        assert!(
            db.query_keys(ns, "status = \"active\"").unwrap().keys.is_empty(),
            "row must leave its bucket when the indexed field is removed"
        );

        db.shutdown().unwrap();
    }

    /// Deactivating a field index removes the in-memory bitmap so that any
    /// subsequent predicate query on that field returns an UnknownField error
    /// (the field is filtered out of the queryable schema map).
    #[test]
    fn test_deactivate_field_index_makes_field_unqueryable() {
        use crate::db::namespace_index::ExtractorFn;
        use crate::index::{IndexValue, IndexValueType};
        use std::sync::Arc;

        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = DEFAULT_NAMESPACE_ID;

        let field_id = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();
        let extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Str(v["status"].as_str()?.to_string()))
        });
        db.activate_field_index(ns, field_id, IndexValueType::Str, extractor).unwrap();

        db.put(b"doc:1", br#"{"status":"active"}"#).unwrap();
        db.put(b"doc:2", br#"{"status":"inactive"}"#).unwrap();

        // Sanity: query works before deactivation.
        let keys = db.query_keys(ns, "status = \"active\"").unwrap().keys;
        assert_eq!(keys, vec![b"doc:1".to_vec()]);

        db.deactivate_field_index(ns, field_id).unwrap();

        // Dropped fields are excluded from the queryable schema map, so the
        // query fails with "unknown field" rather than "no active index".
        let err = db.query_keys(ns, "status = \"active\"").unwrap_err();
        assert!(
            err.to_string().contains("unknown field"),
            "expected UnknownField error after deactivation, got: {err}"
        );

        db.shutdown().unwrap();
    }

    /// FR-001 step 4: a query over a degraded index still returns results, but
    /// says so — and only for the fields *this* predicate touched.
    ///
    /// Serving a complete-looking result set over an incomplete index is the
    /// original bug, so this is the assertion the whole feature turns on.
    #[test]
    fn a_query_over_a_degraded_index_reports_it() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = Database::open(temp_dir.path(), create_db_config())?;
        let ns = DEFAULT_NAMESPACE_ID;

        let status_field = activate_status_index(&db, ns);
        let other_field = activate_named_index(&db, ns, "tier");
        db.put(b"doc:1", br#"{"status":"active","tier":"gold"}"#)?;
        db.put(b"doc:2", br#"{"status":"inactive","tier":"gold"}"#)?;

        // Healthy to begin with.
        let outcome = db.query_keys(ns, "status = \"active\"")?;
        assert_eq!(outcome.keys, vec![b"doc:1".to_vec()]);
        assert!(!outcome.is_degraded(), "a healthy index must not report degradation");
        assert_eq!(outcome.total, 1);

        record_test_gap(&db, ns, status_field);

        // Same results, now flagged.
        let outcome = db.query_keys(ns, "status = \"active\"")?;
        assert_eq!(outcome.keys, vec![b"doc:1".to_vec()], "a degraded index stays queryable");
        assert_eq!(outcome.degraded_fields, vec![status_field], "the touched degraded field must be named");

        // A predicate that does not touch the degraded field is unaffected —
        // one damaged field must not taint every query in the namespace.
        let outcome = db.query_keys(ns, "tier = \"gold\"")?;
        assert_eq!(outcome.keys.len(), 2);
        assert!(!outcome.is_degraded(), "an untouched degraded field must not taint this query");

        // A predicate touching both reports only the damaged one.
        let outcome = db.query_keys(ns, "tier = \"gold\" AND status = \"active\"")?;
        assert_eq!(outcome.degraded_fields, vec![status_field]);
        assert!(!outcome.degraded_fields.contains(&other_field));

        // The paginated form carries the same signal.
        let outcome = db.query_keys_paginated(ns, "status = \"active\"", 0, 10)?;
        assert_eq!(outcome.degraded_fields, vec![status_field]);

        // An empty result over a degraded index is the dangerous case: without
        // the flag it is indistinguishable from "nothing matches".
        let outcome = db.query_keys(ns, "status = \"nonexistent\"")?;
        assert!(outcome.keys.is_empty());
        assert!(outcome.is_degraded(), "an EMPTY result over a degraded index must still report it");

        db.shutdown()?;
        Ok(())
    }
}
