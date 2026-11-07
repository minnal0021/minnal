//! Field-index row IDs: from the dense row map, or derived from the key.

use super::*;

impl KVStore {
    /// Load (or create) the dense row-ID map for this namespace from `dir`.
    ///
    /// Idempotent: a no-op once the map is loaded. Called when the first field
    /// index is activated, before any replay, so the write/replay paths can
    /// resolve through it.
    pub(crate) fn ensure_rowmap(&self, dir: &Path) -> Result<()> {
        let mut guard = self.rowmap.write();
        if guard.is_none() {
            *guard = Some(crate::field_index::RowMap::open(dir).map_err(KVError::Io)?);
        }
        Ok(())
    }

    /// Replace this namespace's row map with a fresh, empty one in `dir`
    /// (`RowMap::reset`). The caller guarantees no field index references a
    /// row ID (see `Database::reset_rowmap`). The old map is dropped before its
    /// files are truncated, so no live mapping outlives them.
    pub(crate) fn reset_rowmap(&self, dir: &Path) -> Result<()> {
        let mut guard = self.rowmap.write();
        *guard = None;
        *guard = Some(crate::field_index::RowMap::reset(dir).map_err(KVError::Io)?);
        Ok(())
    }

    /// Number of row IDs the dense row map has allocated, or `None` when the
    /// namespace has no loaded row map (or uses a key-derived `RowIdFn`).
    pub(crate) fn rowmap_ids_allocated(&self) -> Option<u64> {
        if self.row_id_fn.read().is_some() {
            return None;
        }
        self.rowmap.read().as_ref().map(|rm| rm.next_id())
    }

    /// Resolve the row ID for `key`, **allocating** a new dense ID if the key is
    /// unseen. Used on the put and WAL-replay paths.
    ///
    /// Precedence: a caller-supplied [`RowIdFn`] (the escape hatch for keys that
    /// embed their own ID) wins; otherwise the dense [`RowMap`](crate::field_index::RowMap).
    ///
    /// Only reached once a namespace has an active field index, and
    /// `activate_field_index` loads the row map (`ensure_rowmap`) before
    /// activating any field — so exactly one of the two sources is always
    /// present. If neither is (which should be impossible), we log an error and
    /// return `0` rather than crash the write path.
    pub(crate) fn resolve_row_id_alloc(&self, key: &[u8]) -> u128 {
        if let Some(f) = self.row_id_fn.read().as_ref() {
            return f(key);
        }
        if let Some(rm) = self.rowmap.write().as_mut() {
            return rm.get_or_alloc(key);
        }
        error!(
            "[KVStore '{}'] resolve_row_id_alloc reached with neither a row_id_fn nor a loaded \
             RowMap despite an active field index — indexing may be inconsistent for this key",
            self.name
        );
        0
    }

    /// Resolve the row ID for `key` **without allocating** — returns `None` when
    /// the dense map has never seen the key. Used on the delete and query
    /// fallback paths, where an unseen key simply has nothing indexed.
    ///
    /// Like `resolve_row_id_alloc`, only reached with an active field index, so a
    /// `row_id_fn` or a loaded `RowMap` is always present. If neither is (which
    /// should be impossible), we log an error and return `None` — the caller then
    /// treats the key as having nothing indexed rather than crashing.
    pub(crate) fn resolve_row_id_get(&self, key: &[u8]) -> Option<u128> {
        if let Some(f) = self.row_id_fn.read().as_ref() {
            return Some(f(key));
        }
        if let Some(rm) = self.rowmap.read().as_ref() {
            return rm.get(key);
        }
        error!(
            "[KVStore '{}'] resolve_row_id_get reached with neither a row_id_fn nor a loaded \
             RowMap despite an active field index — treating key as unindexed",
            self.name
        );
        None
    }

    /// Resolve a row ID back to its key via the dense map, if loaded.
    /// The key a field-index row ID stands for, from whichever source assigns
    /// row IDs here: the registered `RowToKeyFn`, else the dense row map.
    pub(crate) fn key_for_row(&self, row_id: u128) -> Option<Vec<u8>> {
        if let Some(inv) = self.row_to_key_fn.read().as_ref() {
            return Some(inv(row_id));
        }
        self.rowmap_key_for(row_id)
    }

    pub(crate) fn rowmap_key_for(&self, row_id: u128) -> Option<Vec<u8>> {
        self.rowmap.read().as_ref().and_then(|rm| rm.key_for(row_id))
    }

    /// True when the dense row map is the active ID resolver (no custom
    /// `row_id_fn`, and the map is loaded) — i.e. `rowmap_key_for` can resolve
    /// query hits.
    pub(crate) fn rowmap_active(&self) -> bool {
        self.row_id_fn.read().is_none() && self.rowmap.read().is_some()
    }

    /// True when the dense row map is loaded but has never allocated an ID
    /// (`next_id == 0`). A namespace driven by a custom `row_id_fn` never touches
    /// the row map, so an empty map alongside indexed data is a signal that the
    /// custom resolver was not installed before the index was built.
    pub(crate) fn rowmap_is_empty(&self) -> bool {
        self.rowmap.read().as_ref().is_some_and(|rm| rm.is_empty())
    }

    /// Flush the dense row map at `wal_offset` (the checkpoint WAL tail). Must be
    /// called **before** flushing any field index so the map stays at least as
    /// durable as every persisted bitmap bit.
    pub(crate) fn flush_rowmap(&self, wal_offset: u64) -> Result<()> {
        if let Some(rm) = self.rowmap.read().as_ref() {
            rm.flush(wal_offset).map_err(KVError::Io)?;
        }
        Ok(())
    }

    /// Register a custom row-ID function and its inverse for this namespace.
    ///
    /// After this call every key passed through the index write path is
    /// assigned the row ID returned by `row_id_fn` instead of the default
    /// dense row-ID map.  Providing `row_to_key_fn` (the inverse) also enables
    /// O(|hits|) query resolution in `Database::query_keys` — the inverse
    /// reconstructs each matching key directly from its row ID, requiring
    /// zero extra memory and no maintenance on writes.
    ///
    /// `row_id_fn` must be injective: distinct keys must produce distinct
    /// row IDs so that `row_to_key_fn` can invert it exactly.
    pub fn set_row_id_fn(&self, f: RowIdFn, inv: Option<RowToKeyFn>) -> Result<()> {
        *self.row_id_fn.write() = Some(f);
        *self.row_to_key_fn.write() = inv;
        Ok(())
    }
}
