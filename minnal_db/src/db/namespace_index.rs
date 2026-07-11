//! Namespace Index Set
//!
//! Holds the active in-memory field indices for a single namespace, plus the
//! extractor closures that map raw document bytes to [`IndexValue`]s.
//!
//! Each [`IndexEntry`] is registered via `Database::activate_field_index`
//! after the database is open.  The entry is then consulted by
//! `KVStore::put_to_storage` and `KVStore::delete_from_storage` to keep
//! the in-memory index current.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::db::namespace::FieldId;
use crate::index::{DynFieldIndex, IndexValue};

// ── Extractor closure ──────────────────────────────────────────────────────

/// Caller-provided closure that extracts an indexed value from raw document
/// bytes, or returns `None` if the field is absent or the bytes cannot be
/// parsed.
///
/// The closure must be `Send + Sync` so it can be called from any thread that
/// holds a `KVStore`.
pub type ExtractorFn = Arc<dyn Fn(&[u8]) -> Option<IndexValue> + Send + Sync>;

// ── RowIdFn / RowToKeyFn ───────────────────────────────────────────────────
//
// # Row IDs, keys, and the isomorphism question
//
// Every entry in a field index is stored against a u128 "row ID".  By default
// the row ID is a dense, monotonic integer assigned per namespace by the
// `RowMap` sidecar (`crate::index::RowMap`), which keeps the RoaringBitmaps compact.
// A doc-store layer may still want to supply its own IDs for two reasons:
//
//   1. STABILITY — hashes are implementation details.  A doc store wants the
//      row ID to *be* the document ID (e.g. a UUID), so it stays stable across
//      any future key encoding changes.
//
//   2. QUERY PERFORMANCE — after bitmap evaluation, `query_keys` has to turn
//      the matching row IDs back into raw keys.  Without extra information it
//      must scan every key in the namespace (O(n)) and check membership.
//      If the caller can supply the *inverse* function — reconstructing a key
//      from its row ID — the scan becomes O(|hits|) with zero memory overhead.
//
// ## When is an inverse (isomorphism) possible?
//
// A true bijection between key bytes and u128 requires the key to encode
// *exactly* 128 bits of information.  Three common cases:
//
//   A. Pure 16-byte UUID key
//      → Perfect bijection; key and row ID are the same bits, different types.
//        row_id_fn:     |k|  u128::from_be_bytes(k[..16].try_into().unwrap())
//        row_to_key_fn: |id| id.to_be_bytes().to_vec()
//
//   B. Prefixed UUID key  (e.g. b"docs:<16-byte-uuid>")
//      → Still invertible.  The prefix is a namespace constant, not encoded in
//        the row ID, so the inverse just prepends it again.
//        row_id_fn:     |k|  u128::from_be_bytes(k[5..21].try_into().unwrap())
//        row_to_key_fn: |id| [b"docs:", id.to_be_bytes().as_slice()].concat()
//
//   C. Arbitrary-length string key  (e.g. "user/alice/settings/theme")
//      → The key has more than 128 bits of entropy; RowIdFn must hash/truncate,
//        so information is lost and inversion is impossible.  Pass `None` for
//        `row_to_key_fn` and query_keys falls back to the O(n) key scan.
//
// ## Key length is not restricted
//
// KVStore accepts keys of any length.  Only the *row ID* is fixed at u128
// (that is baked into the RoaringBitmap layer, not a choice made here).
// The doc-store ID does not have to be a UUID — any scheme where a unique
// 128-bit value can be embedded in the key (and the key reconstructed from it)
// gets the O(|hits|) fast path.  Sequential u64 IDs, for example, cast
// trivially to u128.  String IDs that exceed 128 bits of entropy use the
// hash fallback and still work correctly, just without the fast path.
//
// ## Crash safety
//
// Both closures are Rust function pointers and cannot be serialized.  The
// caller must re-register them after every restart (before calling
// `activate_field_index`).  This is not a problem: `RowToKeyFn` is purely
// computational — it holds no state and requires no warm-up.  There is no
// in-memory map to rebuild; correctness is guaranteed solely by the
// injectivity of `RowIdFn`.

/// Caller-provided closure that derives a stable 128-bit row ID from the raw
/// key bytes of a document.
///
/// Must be injective: distinct keys must produce distinct row IDs so that the
/// companion [`RowToKeyFn`] can reconstruct the original key without a map.
///
/// Register the pair via
/// [`Database::set_row_id_fn`][crate::db::database::Database::set_row_id_fn].
pub type RowIdFn = Arc<dyn Fn(&[u8]) -> u128 + Send + Sync>;

/// Inverse of [`RowIdFn`]: reconstructs the raw key bytes from a row ID.
///
/// Because [`RowIdFn`] is injective, this reconstruction is always exact —
/// no in-memory map or disk lookup is required.  Registering this alongside
/// [`RowIdFn`] enables O(|hits|) query resolution instead of an O(n_keys)
/// full-namespace scan.
///
/// See the module-level comment on `RowIdFn` / `RowToKeyFn` for a detailed
/// discussion of when an inverse is possible and when it is not.
///
/// For a pure UUID key the pair is simply:
/// ```ignore
/// let row_id_fn:     RowIdFn     = Arc::new(|k| u128::from_be_bytes(k[..16].try_into().unwrap()));
/// let row_to_key_fn: RowToKeyFn  = Arc::new(|id| id.to_be_bytes().to_vec());
/// ```
pub type RowToKeyFn = Arc<dyn Fn(u128) -> Vec<u8> + Send + Sync>;

// ── IndexEntry ─────────────────────────────────────────────────────────────

/// One registered field index within a namespace.
pub struct IndexEntry {
    /// Stable field identifier matching the one stored in `NamespaceSchema`.
    pub field_id: FieldId,
    /// Extracts the field value from raw document bytes.
    pub extractor: ExtractorFn,
    /// The live in-memory index — shared between the KVStore write path and
    /// the checkpoint worker.
    pub index: Arc<RwLock<DynFieldIndex>>,
}

// ── NamespaceIndexSet ──────────────────────────────────────────────────────

/// All active field indices for a single namespace.
///
/// Keyed by [`FieldId`] for O(1) lookup during write-path hooks.
pub struct NamespaceIndexSet {
    entries: HashMap<FieldId, IndexEntry>,
}

impl NamespaceIndexSet {
    pub fn new() -> Self {
        Self { entries: HashMap::new() }
    }

    /// Register (or replace) a field index entry.
    pub fn register(&mut self, entry: IndexEntry) {
        self.entries.insert(entry.field_id, entry);
    }

    /// Borrow the entry for `field_id`, if registered.
    pub fn get(&self, field_id: FieldId) -> Option<&IndexEntry> {
        self.entries.get(&field_id)
    }

    /// Iterate over all registered entries.
    pub fn iter(&self) -> impl Iterator<Item = &IndexEntry> {
        self.entries.values()
    }

    /// True if there are no registered field indices.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove the entry for `field_id`. Returns `true` if it was present.
    pub fn deregister(&mut self, field_id: FieldId) -> bool {
        self.entries.remove(&field_id).is_some()
    }
}

impl Default for NamespaceIndexSet {
    fn default() -> Self {
        Self::new()
    }
}

// ── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{DynFieldIndex, IndexValue, IndexValueType};

    #[test]
    fn test_register_and_lookup() {
        let mut set = NamespaceIndexSet::new();
        let entry = IndexEntry {
            field_id: 0,
            extractor: Arc::new(|_| Some(IndexValue::Int(1))),
            index: Arc::new(RwLock::new(DynFieldIndex::new(IndexValueType::Int))),
        };
        set.register(entry);
        assert!(set.get(0).is_some());
        assert!(set.get(1).is_none());
    }
}
