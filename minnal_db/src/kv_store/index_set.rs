//! Namespace Index Set
//!
//! Holds the active in-memory field indices for a single namespace, plus the
//! extractor closures that map raw document bytes to [`IndexValue`]s.
//!
//! Each [`IndexEntry`] is registered via `Database::activate_field_index`
//! after the database is open.  The entry is then consulted by
//! `KVStore::put_to_storage` and `KVStore::delete_from_storage` to keep
//! the in-memory index current.

use crate::namespace::NamespaceId;
use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::field_index::FieldId;
use crate::field_index::{DynFieldIndex, IndexValue};

/// Write-path sink for field-index updates that were **rejected**.
///
/// A rejected update leaves the row silently absent from that field forever. The
/// write path knows the namespace, field and key exactly, so it is the cheapest
/// index gap to capture, but it must not pay for durability inline, so the
/// sink only buffers and the index checkpoint persists what it collected.
///
/// That is sound because the buffer and the index have the *same* durability
/// point: if a crash loses the buffered key, it also lost the index update, and
/// WAL replay re-runs the write, re-rejects it, and re-buffers the key.
///
/// Mirrors the WAL/LSM observer pattern: the write path holds a cheap handle and
/// signals, rather than knowing how a gap is recorded.
pub trait IndexGapSink: Send + Sync {
    /// Note that `field_id` in `namespace_id` refused to index `key`.
    fn note_rejected_update(&self, namespace_id: NamespaceId, field_id: FieldId, key: &[u8]);
}

/// Write-path requests for an early index checkpoint. Both are O(1), non-blocking
/// and debounced by whoever runs the checkpoints.
pub(crate) trait CheckpointRequests: Send + Sync {
    /// Request one if a field's reclaimable `dead_bytes` has reached the
    /// configured cap.
    fn request_if_over_cap(&self, dead_bytes: u64);
    /// Request one regardless of any cap.
    fn request(&self);
}

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
// Every entry in a field index is stored against a u128 row ID. By default the
// row ID is a dense integer assigned per namespace by the row map
// (`crate::field_index::RowMap`), which keeps the RoaringBitmaps compact and
// also turns a query's matching row IDs back into keys.
//
// A caller whose keys carry their own ID of at most 128 bits (a u64 doc-store
// key, a 16-byte UUID) can register a pair instead and keep no map:
// `RowIdFn` derives the row ID from the key, and `RowToKeyFn` rebuilds the key
// from the row ID. The pair must be exact inverses on every key stored, so a key
// holding more than 128 bits of information (an arbitrary string) cannot use
// one and takes its row IDs from the row map. Two examples:
//
//   Pure 16-byte UUID key: key and row ID are the same bits.
//     row_id_fn:     |k|  u128::from_be_bytes(k[..16].try_into().unwrap())
//     row_to_key_fn: |id| id.to_be_bytes().to_vec()
//
//   Prefixed UUID key (b"docs:<16-byte-uuid>"): the prefix is a constant, so the
//   inverse prepends it again.
//     row_id_fn:     |k|  u128::from_be_bytes(k[5..21].try_into().unwrap())
//     row_to_key_fn: |id| [b"docs:", id.to_be_bytes().as_slice()].concat()
//
// The closures are not persisted: the caller registers them again after every
// restart, before `activate_field_index`. They hold no state, so there is
// nothing to rebuild.

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
/// Because [`RowIdFn`] is injective, this reconstruction is always exact, with
/// no in-memory map or disk lookup: a query turns each matching row ID into its
/// key through it.
///
/// See the comment above [`RowIdFn`] for when such a pair is possible.
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
    use crate::field_index::{DynFieldIndex, IndexValue, IndexValueType};

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
