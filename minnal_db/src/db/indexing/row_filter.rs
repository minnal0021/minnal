//! A predicate's matches as a test on keys, for a search that checks every
//! candidate it scans: the filtered semantic search's Pass 1 skips a chunk
//! whose document the predicate does not match, before scoring it, so the
//! filter is applied *before* top-k selection (a pre-filter), never to the
//! results.
//!
//! A predicate can match most of a store, and turning every match into its key
//! cost one row-map lookup and one allocation per match, per request, before
//! the search even started. So only a predicate matching few rows is turned
//! into a set of keys; a broader one keeps its row bitmap, and each scanned
//! key is turned into its row id instead (no lookup at all for a store whose
//! row ids come from the key, such as a `u64` doc store).

use std::collections::HashSet;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::db::database::Database;
use crate::db::error::Result;
use crate::field_index::bitmap::RoaringBitmap;
use crate::field_index::{QueryOutcome, RowMap};
use crate::kv_store::index_set::RowIdFn;
use crate::namespace::NamespaceId;

/// The most matching rows turned into a set of keys: a key-set lookup is the
/// cheapest test per candidate, and this many keys cost little to resolve.
/// Two in tests, so the tests' filtered searches take both paths.
#[cfg(not(test))]
const KEY_SET_MAX_ROWS: usize = 10_000;
#[cfg(test)]
const KEY_SET_MAX_ROWS: usize = 2;

/// Whether a key's document matches a predicate ([`Database::query_filter`]).
pub(crate) struct RowFilter(Matches);

enum Matches {
    /// The matching rows' keys.
    Keys(HashSet<Vec<u8>>),
    /// The matching rows, and how a key becomes its row id.
    Rows { rows: RoaringBitmap, row_id: RowId },
}

/// How a key becomes its row id: the store's `RowIdFn`, else its row map.
enum RowId {
    Derived(RowIdFn),
    Mapped(Arc<RwLock<Option<RowMap>>>),
}

impl RowFilter {
    /// Whether the document stored under `key` matches. A key with no row id
    /// was never indexed, so it matches nothing.
    pub(crate) fn contains(&self, key: &[u8]) -> bool {
        match &self.0 {
            Matches::Keys(keys) => keys.contains(key),
            Matches::Rows { rows, row_id } => {
                let id = match row_id {
                    RowId::Derived(f) => Some(f(key)),
                    // A read lock for one in-memory probe; nothing else is done under it.
                    RowId::Mapped(map) => map.read().as_ref().and_then(|m| m.get(key)),
                };
                id.is_some_and(|id| rows.contains(id))
            }
        }
    }

    /// Whether the matches are held as keys (test-only, to check both forms are used).
    #[cfg(test)]
    pub(crate) fn holds_keys(&self) -> bool {
        matches!(self.0, Matches::Keys(_))
    }
}

impl Database {
    /// Evaluate `query_str` against `namespace_id`'s field indexes and return
    /// a test on keys for its matches, with the query's outcome (its total,
    /// degraded and touched fields; no keys).
    pub(crate) fn query_filter(&self, namespace_id: NamespaceId, query_str: &str) -> Result<(RowFilter, QueryOutcome)> {
        let store = self.get_store(namespace_id)?;
        let (rows, touched) = self.evaluate_predicate(namespace_id, &store, query_str)?;
        let outcome = QueryOutcome {
            keys: Vec::new(),
            total: rows.len(),
            degraded_fields: self.degraded_among(namespace_id, &touched),
            touched_fields: touched,
        };
        let matches = if rows.len() <= KEY_SET_MAX_ROWS {
            Matches::Keys(super::query::keys_for_rows(&store, rows.iter())?.into_iter().collect())
        } else if let Some(f) = store.row_id_fn.read().clone() {
            Matches::Rows {
                rows,
                row_id: RowId::Derived(f),
            }
        } else if store.rowmap_active() {
            Matches::Rows {
                rows,
                row_id: RowId::Mapped(Arc::clone(&store.rowmap)),
            }
        } else {
            return Err(crate::KVError::Io(std::io::Error::other(format!(
                "namespace '{}' has an active field index but neither a row map nor a row id function",
                store.name
            ))));
        };
        Ok((RowFilter(matches), outcome))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test_support::*;
    use crate::namespace::DEFAULT_NAMESPACE_ID;
    use tempfile::TempDir;

    /// Write `n` documents, every third one active, and check the filter of
    /// `status = "active"` against every key and one never written.
    fn check(db: &Database, n: u64, key: impl Fn(u64) -> Vec<u8>, keys_expected: bool) {
        let ns = DEFAULT_NAMESPACE_ID;
        for i in 0..n {
            let status = if i % 3 == 0 { "active" } else { "idle" };
            db.put(&key(i), format!(r#"{{"status":"{status}"}}"#).as_bytes()).unwrap();
        }
        let (filter, outcome) = db.query_filter(ns, r#"status = "active""#).unwrap();
        assert_eq!(outcome.total as u64, n.div_ceil(3));
        assert!(outcome.keys.is_empty(), "the outcome carries no keys");
        assert_eq!(filter.holds_keys(), keys_expected);
        for i in 0..n {
            assert_eq!(filter.contains(&key(i)), i % 3 == 0, "key {i}");
        }
        assert!(!filter.contains(&key(n + 100)), "a key never written matches nothing");
    }

    /// Few matches are held as keys; more stay a bitmap and each key is looked
    /// up in the row map. Either way the filter matches exactly the predicate's
    /// rows.
    #[test]
    fn a_filter_matches_exactly_the_predicates_rows_through_the_row_map() {
        for (n, keys_expected) in [(4u64, true), (30, false)] {
            let dir = TempDir::new().unwrap();
            let db = Database::open(dir.path(), create_db_config()).unwrap();
            activate_status_index(&db, DEFAULT_NAMESPACE_ID);
            check(&db, n, |i| format!("doc:{i}").into_bytes(), keys_expected);
            db.shutdown().unwrap();
        }
    }

    /// A store whose row ids come from the key (as a `u64` doc store's do)
    /// tests a key without any lookup.
    #[test]
    fn a_filter_derives_row_ids_from_keys_when_the_store_does() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = DEFAULT_NAMESPACE_ID;
        let row_id: RowIdFn = Arc::new(|k: &[u8]| u128::from(u64::from_be_bytes(k.try_into().unwrap())));
        let row_to_key: crate::kv_store::index_set::RowToKeyFn = Arc::new(|id: u128| (id as u64).to_be_bytes().to_vec());
        db.set_row_id_fn(ns, row_id, row_to_key).unwrap();
        activate_status_index(&db, ns);
        check(&db, 30, |i| i.to_be_bytes().to_vec(), false);
        db.shutdown().unwrap();
    }
}
