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

    /// Every L0 flush marks WAL entries persisted, after which recovery skips them
    /// and WAL GC deletes them, so the value-log records those SSTable entries point
    /// at must already be on stable storage. Writes fsync the value log only every
    /// `records_per_sync`, so without the flush observer's fsync a power loss right
    /// after the flush lost those acknowledged values for good.
    #[test]
    fn test_a_flush_to_l0_leaves_no_unsynced_values_behind() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        for i in 0..50u32 {
            db.put(format!("doc:{i}").as_bytes(), b"a value that is not yet fsynced").unwrap();
        }
        let store = db.get_store(DEFAULT_NAMESPACE_ID).unwrap();
        assert!(
            store.value_log.unsynced_bytes() > 0,
            "setup: fewer writes than records_per_sync leave values unsynced"
        );
        store.flush_memtable_to_level0().unwrap();
        assert_eq!(
            store.value_log.unsynced_bytes(),
            0,
            "WAL entries were marked persisted over unsynced values"
        );
        db.shutdown().unwrap();
    }

    /// A write that loses its seq race must leave the field index alone, because
    /// the LSM drops it: reads keep showing the winner, so the index must too.
    /// Both halves used to act as if they had won. A delete older than the live
    /// value stripped that live document's row, so it vanished from every query
    /// on the field; and a put older than a delete counted the key as "absent,
    /// so I win" and added a row for a key that reads as deleted. Only TTL expiry
    /// produced these out-of-order seqs in practice (it now takes the key stripe),
    /// so they are driven here directly with explicit seqs.
    #[test]
    fn test_a_write_that_loses_its_seq_race_leaves_the_index_alone() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = DEFAULT_NAMESPACE_ID;
        activate_status_index(&db, ns);
        let store = db.get_store(ns).unwrap();

        // A losing delete: the live value (seq 100) must stay indexed.
        store.put_to_storage_seq(b"doc:A", br#"{"status":"active"}"#, 1_000_100).unwrap();
        store.delete_from_storage_seq(b"doc:A", 1_000_050).unwrap();
        // A losing put: the key stays deleted (seq 300), so it must not be indexed.
        store.put_to_storage_seq(b"doc:B", br#"{"status":"idle"}"#, 1_000_200).unwrap();
        store.delete_from_storage_seq(b"doc:B", 1_000_300).unwrap();
        store.put_to_storage_seq(b"doc:B", br#"{"status":"active"}"#, 1_000_250).unwrap();

        assert!(store.get(b"doc:A").unwrap().is_some(), "setup: doc:A is live");
        assert!(store.get(b"doc:B").unwrap().is_none(), "setup: doc:B is deleted");
        assert_eq!(
            db.query_keys(ns, "status = \"active\"").unwrap().keys,
            vec![b"doc:A".to_vec()],
            "the index must agree with reads: doc:A live, doc:B deleted"
        );
        db.shutdown().unwrap();
    }

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

#[cfg(test)]
mod differential_tests {
    use crate::db::database::Database;
    use crate::db::namespace::DEFAULT_NAMESPACE_ID;
    use crate::db::namespace_index::ExtractorFn;
    use crate::db::test_support::create_db_config;
    use crate::index::{IndexValue, IndexValueType};
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;
    use tempfile::TempDir;

    struct R(u64);
    impl R {
        fn n(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 11
        }
        fn b(&mut self, k: u64) -> u64 {
            self.n() % k
        }
    }
    #[derive(Clone, Debug, Default)]
    struct Doc {
        status: Option<String>,
        age: Option<i64>,
        flag: Option<bool>,
    }
    #[derive(Debug)]
    enum E {
        Leaf(&'static str, &'static str, V),
        And(Box<E>, Box<E>),
        Or(Box<E>, Box<E>),
        Not(Box<E>),
    }
    #[derive(Debug, Clone)]
    enum V {
        S(String),
        I(i64),
        B(bool),
        L(Vec<V>),
    }

    const AGES: [i64; 8] = [i64::MIN, -5, -1, 0, 1, 7, 20, i64::MAX];
    fn rs(r: &mut R) -> String {
        ["a", "b", "c", "d"][r.b(4) as usize].to_string()
    }
    fn ri(r: &mut R) -> i64 {
        AGES[r.b(8) as usize]
    }

    fn gen_e(r: &mut R, depth: u32) -> E {
        if depth == 0 || r.b(3) == 0 {
            let ops = ["=", "!=", "<", "<=", ">", ">=", "IN"];
            return match r.b(3) {
                0 => {
                    let op = ops[r.b(7) as usize];
                    let v = if op == "IN" {
                        V::L((0..r.b(3) + 1).map(|_| V::S(rs(r))).collect())
                    } else {
                        V::S(rs(r))
                    };
                    E::Leaf("status", op, v)
                }
                1 => {
                    let op = ops[r.b(7) as usize];
                    let v = if op == "IN" {
                        V::L((0..r.b(3) + 1).map(|_| V::I(ri(r))).collect())
                    } else {
                        V::I(ri(r))
                    };
                    E::Leaf("age", op, v)
                }
                _ => {
                    let op = ["=", "!=", "IN"][r.b(3) as usize];
                    let v = if op == "IN" { V::L(vec![V::B(r.b(2) == 0)]) } else { V::B(r.b(2) == 0) };
                    E::Leaf("flag", op, v)
                }
            };
        }
        match r.b(3) {
            0 => E::And(Box::new(gen_e(r, depth - 1)), Box::new(gen_e(r, depth - 1))),
            1 => E::Or(Box::new(gen_e(r, depth - 1)), Box::new(gen_e(r, depth - 1))),
            _ => E::Not(Box::new(gen_e(r, depth - 1))),
        }
    }
    fn vs(v: &V) -> String {
        match v {
            V::S(s) => format!("\"{s}\""),
            V::I(i) => i.to_string(),
            V::B(b) => b.to_string(),
            V::L(l) => format!("({})", l.iter().map(vs).collect::<Vec<_>>().join(", ")),
        }
    }
    fn render(e: &E) -> String {
        match e {
            E::Leaf(f, op, v) => format!("{f} {op} {}", vs(v)),
            E::And(a, b) => format!("({} AND {})", render(a), render(b)),
            E::Or(a, b) => format!("({} OR {})", render(a), render(b)),
            E::Not(a) => format!("NOT ({})", render(a)),
        }
    }
    fn fields(e: &E, out: &mut BTreeSet<&'static str>) {
        match e {
            E::Leaf(f, ..) => {
                out.insert(f);
            }
            E::And(a, b) | E::Or(a, b) => {
                fields(a, out);
                fields(b, out);
            }
            E::Not(a) => fields(a, out),
        }
    }
    fn has(d: &Doc, f: &str) -> bool {
        match f {
            "status" => d.status.is_some(),
            "age" => d.age.is_some(),
            _ => d.flag.is_some(),
        }
    }
    fn cmp<T: Ord>(x: &T, op: &str, y: &T) -> bool {
        match op {
            "=" => x == y,
            "!=" => x != y,
            "<" => x < y,
            "<=" => x <= y,
            ">" => x > y,
            _ => x >= y,
        }
    }
    fn leaf(d: &Doc, f: &str, op: &str, v: &V) -> bool {
        match (f, v) {
            ("status", V::L(l)) => d.status.as_ref().is_some_and(|s| l.iter().any(|x| matches!(x, V::S(y) if y == s))),
            ("status", V::S(y)) => d.status.as_ref().is_some_and(|s| cmp(s, op, y)),
            ("age", V::L(l)) => d.age.is_some_and(|a| l.iter().any(|x| matches!(x, V::I(y) if *y == a))),
            ("age", V::I(y)) => d.age.is_some_and(|a| cmp(&a, op, y)),
            ("flag", V::L(l)) => d.flag.is_some_and(|a| l.iter().any(|x| matches!(x, V::B(y) if *y == a))),
            ("flag", V::B(y)) => d.flag.is_some_and(|a| cmp(&a, op, y)),
            _ => unreachable!(),
        }
    }
    fn model(e: &E, docs: &BTreeMap<Vec<u8>, Doc>) -> BTreeSet<Vec<u8>> {
        match e {
            E::Leaf(f, op, v) => docs.iter().filter(|(_, d)| leaf(d, f, op, v)).map(|(k, _)| k.clone()).collect(),
            E::And(a, b) => model(a, docs).intersection(&model(b, docs)).cloned().collect(),
            E::Or(a, b) => model(a, docs).union(&model(b, docs)).cloned().collect(),
            E::Not(a) => {
                let mut fs = BTreeSet::new();
                fields(a, &mut fs);
                let universe: BTreeSet<Vec<u8>> = docs
                    .iter()
                    .filter(|(_, d)| fs.iter().any(|f| has(d, f)))
                    .map(|(k, _)| k.clone())
                    .collect();
                universe.difference(&model(a, docs)).cloned().collect()
            }
        }
    }

    /// Every query must return exactly what brute-force evaluation of the same
    /// predicate over the documents returns, with the documented semantics
    /// (a missing field matches nothing; `NOT` complements against rows that have
    /// a value for a referenced field). Random documents with fields present or
    /// absent are put, rewritten and deleted through the real write path, with
    /// index checkpoints, clean reopens and crash-style reopens (index rebuilt by
    /// WAL replay) mixed in, and every result is also paged through to check
    /// `query_keys_paginated` and `total`.
    #[test]
    fn test_queries_match_brute_force_across_reopens() {
        let dir = TempDir::new().unwrap();
        let ns = DEFAULT_NAMESPACE_ID;
        let open = || {
            let db = Database::open(dir.path(), create_db_config()).unwrap();
            let reg = |name: &'static str, t: IndexValueType, ex: ExtractorFn| {
                let id = db.register_index_field(ns, name, t).unwrap();
                db.activate_field_index(ns, id, t, ex).unwrap();
            };
            let json = |b: &[u8]| serde_json::from_slice::<serde_json::Value>(b).ok();
            reg(
                "status",
                IndexValueType::Str,
                Arc::new(move |b: &[u8]| Some(IndexValue::Str(json(b)?["status"].as_str()?.to_string()))),
            );
            reg(
                "age",
                IndexValueType::Int,
                Arc::new(move |b: &[u8]| Some(IndexValue::Int(json(b)?["age"].as_i64()?))),
            );
            reg(
                "flag",
                IndexValueType::Bool,
                Arc::new(move |b: &[u8]| Some(IndexValue::Bool(json(b)?["flag"].as_bool()?))),
            );
            db
        };
        let mut db = open();
        let (mut reopens, mut crashes, mut ckpts) = (0, 0, 0);

        let mut r = R(777);
        let mut docs: BTreeMap<Vec<u8>, Doc> = BTreeMap::new();
        let mut checked = 0;
        for round in 0..20 {
            match r.b(8) {
                0 => {
                    db.shutdown().unwrap();
                    drop(db);
                    db = open();
                    reopens += 1;
                }
                1 => {
                    drop(db);
                    db = open();
                    crashes += 1;
                }
                2 | 3 => {
                    use crate::db::index_checkpoint_worker::IndexCheckpointTarget;
                    db.run_index_checkpoint().unwrap();
                    ckpts += 1;
                }
                _ => {}
            }
            for _ in 0..40 {
                let key = format!("doc:{:03}", r.b(50)).into_bytes();
                if r.b(6) == 0 {
                    db.delete(&key).unwrap();
                    docs.remove(&key);
                } else {
                    let d = Doc {
                        status: (r.b(5) != 0).then(|| rs(&mut r)),
                        age: (r.b(5) != 0).then(|| ri(&mut r)),
                        flag: (r.b(5) != 0).then(|| r.b(2) == 0),
                    };
                    let mut j = serde_json::Map::new();
                    if let Some(s) = &d.status {
                        j.insert("status".into(), s.clone().into());
                    }
                    if let Some(a) = d.age {
                        j.insert("age".into(), a.into());
                    }
                    if let Some(f) = d.flag {
                        j.insert("flag".into(), f.into());
                    }
                    db.put(&key, serde_json::to_vec(&serde_json::Value::Object(j)).unwrap().as_slice())
                        .unwrap();
                    docs.insert(key, d);
                }
            }
            for _ in 0..25 {
                let e = gen_e(&mut r, 3);
                let q = render(&e);
                let got: BTreeSet<Vec<u8>> = db.query_keys(ns, &q).unwrap().keys.into_iter().collect();
                let want = model(&e, &docs);
                if got != want {
                    let show = |s: &BTreeSet<Vec<u8>>| s.iter().map(|k| String::from_utf8_lossy(k).into_owned()).collect::<Vec<_>>();
                    panic!(
                        "round {round} query `{q}`\n extra: {:?}\n missing: {:?}",
                        show(&got.difference(&want).cloned().collect()),
                        show(&want.difference(&got).cloned().collect())
                    );
                }
                // Pagination: pages concatenate to the full result; total is exact.
                let full = db.query_keys(ns, &q).unwrap();
                assert_eq!(full.total as usize, want.len(), "round {round} `{q}`: total");
                let mut paged = Vec::new();
                let mut off = 0;
                loop {
                    let p = db.query_keys_paginated(ns, &q, off, 7).unwrap();
                    assert_eq!(p.total as usize, want.len(), "round {round} `{q}`: paginated total");
                    if p.keys.is_empty() {
                        break;
                    }
                    off += p.keys.len();
                    paged.extend(p.keys);
                }
                assert_eq!(paged, full.keys, "round {round} `{q}`: pages differ from the full result");
                checked += 1;
            }
        }
        assert!(
            checked > 0 && reopens + crashes > 0 && ckpts > 0,
            "the run must exercise reopens and checkpoints"
        );
        db.shutdown().unwrap();
    }
}
