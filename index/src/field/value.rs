//! Dynamic index value types
//!
//! [`IndexValue`] and [`IndexValueType`] are the bridge between the caller's
//! document model and the typed [`FieldIndex<V>`].  The caller (DocStore or
//! KVStore index hooks) extracts a field value from raw bytes and returns an
//! [`IndexValue`]; the database layer then dispatches to the correct
//! [`DynFieldIndex`] variant.
//!
//! [`DynFieldIndex`] provides a uniform mutation interface over the
//! mmap-backed [`FieldIndex`] variants.  Persistence is handled directly by
//! the underlying [`BlobStore`] mmap files; use [`open`] / [`flush`] to
//! open or flush a file-backed index.
//!
//! The value→slot-id mapping (the "keymap") is persisted in a second
//! [`BlobStore`] under a `keymap/` subdirectory.  Each entry maps
//! `slot_id (u128) → serialised value bytes`, so individual inserts and
//! removes are immediately durable without rewriting the entire keymap.
//!
//! # Crash atomicity — DynFieldIndex is NOT independently crash-atomic
//!
//! A [`DynFieldIndex`] spans **two** separate [`BlobStore`]s — the bitmap store
//! (`slot_id → RoaringBitmap`) and the keymap store (`slot_id → value bytes`) —
//! and there is **no index-level marker tying them to one logical point**.
//! [`flush`] flushes the bitmap store and then the keymap store as two distinct
//! `msync`s, so a crash *between* them leaves the two stores at different
//! versions (a "skew"): e.g. a slot's bitmap is on disk but its keymap entry is
//! not, or vice-versa. Each store on its own is still structurally valid (they
//! pass [`BlobStore::open`]'s header/bounds checks) — they simply disagree.
//!
//! This is **by design**: the field index is a *derived, reconstructable*
//! structure, so consistency is the **owner's** responsibility, not the index
//! crate's. In `minnal_db`, `run_index_checkpoint` flushes both stores and only
//! *then* records the WAL offset (`IndexManager::checkpoint_fields`) as the
//! single atomic marker. A crash mid-flush leaves that offset at the previous
//! checkpoint, so on open `minnal_db` replays every WAL entry since then on top
//! of the loaded index, re-applying the affected inserts/removes in their
//! original order and reconciling any skew. A skewed reopen never panics or
//! reads out of bounds — at worst a torn value queries empty (or leaves an
//! orphaned slot reclaimed by a later [`compact`]) until replay heals it.
//!
//! **Standalone users must provide their own reconciliation** (e.g. an external
//! log to replay, or a wrapping checkpoint marker). Do not assume that opening a
//! `DynFieldIndex` after a crash, with no replay, yields a self-consistent
//! bitmap/keymap pair. An index-level marker is intentionally **not** provided
//! here because it would duplicate the owner's WAL-offset checkpoint; add one
//! only if a genuine standalone-crash-atomic use case appears.
//!
//! [`compact`]: BlobStore::compact

use std::collections::BTreeMap;
use std::path::Path;

use crate::RoaringBitmap;
use crate::blob_store::BlobStore;
use crate::field::field_index::FieldIndex;

// ── Value types ────────────────────────────────────────────────────────────

/// A single typed field value extracted from a document.
#[derive(Debug, Clone, PartialEq)]
pub enum IndexValue {
    Bool(bool),
    Int(i64),
    Str(String),
}

/// Discriminant for the value type stored in a [`DynFieldIndex`].
///
/// Used when creating a new field index so the correct variant is allocated
/// before any values are inserted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum IndexValueType {
    Bool,
    Int,
    Str,
}

// ── DynFieldIndex ──────────────────────────────────────────────────────────

/// On-disk blob growth/waste metrics for one field index, returned by
/// [`DynFieldIndex::blob_stats`]. `*_logical_bytes` is everything ever appended
/// (live + stale); `*_live_bytes` is what survives compaction; the difference is
/// reclaimable dead space. `*_waste_ratio` is `dead / logical`. The bitmap store
/// is the one that balloons under low-cardinality write amplification; track it
/// to guardrail against runaway disk use between compactions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexBlobStats {
    /// Number of distinct values currently indexed.
    pub distinct_values: usize,
    pub bitmap_logical_bytes: u64,
    pub bitmap_live_bytes: u64,
    pub bitmap_waste_ratio: f64,
    pub keymap_logical_bytes: u64,
    pub keymap_live_bytes: u64,
    pub keymap_waste_ratio: f64,
}

/// Inner (private) typed storage — one variant per supported value type.
pub(crate) enum DynFieldIndexInner {
    Bool(FieldIndex<bool>),
    Int(FieldIndex<i64>),
    Str(FieldIndex<String>),
}

/// Type-erased field index backed by mmap [`BlobStore`] instances.
///
/// Two stores are involved:
/// - **bitmap store** (`blobs.keys` / `blobs.vals`) — holds the per-value
///   [`RoaringBitmap`] data, managed by the inner [`FieldIndex`].
/// - **keymap store** (`keymap/blobs.keys` / `keymap/blobs.vals`) — maps each
///   `slot_id` to the serialised field value so the `BTreeMap<V, u128>`
///   ordering can be rebuilt on open.
///
/// For ephemeral (test) use, create with [`new`] — no keymap store is
/// allocated.  For persistent use, open the index directory with [`open`]
/// and flush changes with [`flush`].
pub struct DynFieldIndex {
    pub(crate) inner: DynFieldIndexInner,
    /// Mmap-backed keymap store: `slot_id → serialised V`.
    /// `None` for anonymous (in-memory) indexes.
    keymap_store: Option<BlobStore>,
}

impl DynFieldIndex {
    /// Create an empty, anonymous (in-memory) index of the given type.
    pub fn new(value_type: IndexValueType) -> Self {
        let inner = match value_type {
            IndexValueType::Bool => DynFieldIndexInner::Bool(FieldIndex::new()),
            IndexValueType::Int => DynFieldIndexInner::Int(FieldIndex::new()),
            IndexValueType::Str => DynFieldIndexInner::Str(FieldIndex::new()),
        };
        Self { inner, keymap_store: None }
    }

    /// Open (or create) a file-backed index in `dir`.
    ///
    /// The keymap is persisted in a `keymap/` subdirectory as an mmap-backed
    /// [`BlobStore`].  On open the entries are iterated to rebuild the
    /// in-memory `BTreeMap` ordering.
    pub fn open(value_type: IndexValueType, dir: &Path) -> std::io::Result<Self> {
        let bitmaps = if BlobStore::exists(dir) {
            BlobStore::open(dir)?
        } else {
            BlobStore::create(dir)?
        };

        let keymap_dir = dir.join("keymap");

        let (inner, keymap_store) = if BlobStore::exists(&keymap_dir) {
            // ── open existing mmap-backed keymap ──────────────────────────
            let ks = BlobStore::open(&keymap_dir)?;
            let inner = match value_type {
                IndexValueType::Bool => {
                    let (ordering, next_slot) = rebuild_ordering_bool(&ks);
                    tracing::debug!(
                        ordering_len = ordering.len(),
                        next_slot,
                        "DynFieldIndex::open: Bool keymap rebuilt from mmap"
                    );
                    DynFieldIndexInner::Bool(FieldIndex::from_parts(ordering, bitmaps, next_slot))
                }
                IndexValueType::Int => {
                    let (ordering, next_slot) = rebuild_ordering_int(&ks);
                    tracing::debug!(
                        ordering_len = ordering.len(),
                        next_slot,
                        "DynFieldIndex::open: Int keymap rebuilt from mmap"
                    );
                    DynFieldIndexInner::Int(FieldIndex::from_parts(ordering, bitmaps, next_slot))
                }
                IndexValueType::Str => {
                    let (ordering, next_slot) = rebuild_ordering_str(&ks);
                    tracing::debug!(
                        ordering_len = ordering.len(),
                        next_slot,
                        "DynFieldIndex::open: Str keymap rebuilt from mmap"
                    );
                    DynFieldIndexInner::Str(FieldIndex::from_parts(ordering, bitmaps, next_slot))
                }
            };
            (inner, ks)
        } else {
            // ── fresh index — create empty keymap store ───────────────────
            tracing::debug!("DynFieldIndex::open: no keymap found — starting empty");
            std::fs::create_dir_all(&keymap_dir)?;
            let ks = BlobStore::create(&keymap_dir)?;
            let inner = match value_type {
                IndexValueType::Bool => DynFieldIndexInner::Bool(FieldIndex::from_parts(BTreeMap::new(), bitmaps, 0)),
                IndexValueType::Int => DynFieldIndexInner::Int(FieldIndex::from_parts(BTreeMap::new(), bitmaps, 0)),
                IndexValueType::Str => DynFieldIndexInner::Str(FieldIndex::from_parts(BTreeMap::new(), bitmaps, 0)),
            };
            (inner, ks)
        };

        Ok(Self {
            inner,
            keymap_store: Some(keymap_store),
        })
    }

    /// Flush both the bitmap and keymap mmap stores to disk.
    ///
    /// The bitmap store is flushed first, then the keymap store, as **two
    /// independent `msync`s with no marker between them** — so a crash in
    /// between leaves the two stores skewed. This call is *not* crash-atomic on
    /// its own; the owner heals skew by replaying its log from the last
    /// checkpoint offset. See the module-level *Crash atomicity* section.
    ///
    /// The `dir` parameter is retained for API compatibility but is no longer
    /// used — all state is already in the mmap stores opened at construction.
    pub fn flush(&self, dir: &Path) -> std::io::Result<()> {
        let _ = dir; // retained for backward compatibility
        // Flush bitmap mmap pages.
        match &self.inner {
            DynFieldIndexInner::Bool(fi) => fi.flush()?,
            DynFieldIndexInner::Int(fi) => fi.flush()?,
            DynFieldIndexInner::Str(fi) => fi.flush()?,
        }

        // Flush keymap mmap store.
        if let Some(ks) = &self.keymap_store {
            ks.flush()?;
        }
        Ok(())
    }

    /// Fraction (`0.0..1.0`) of the bitmap value region that is reclaimable
    /// dead space accumulated by the append-only per-insert bitmap rewrites.
    pub fn bitmap_waste_ratio(&self) -> f64 {
        match &self.inner {
            DynFieldIndexInner::Bool(fi) => fi.bitmap_waste_ratio(),
            DynFieldIndexInner::Int(fi) => fi.bitmap_waste_ratio(),
            DynFieldIndexInner::Str(fi) => fi.bitmap_waste_ratio(),
        }
    }

    /// Fraction (`0.0..1.0`) of the keymap value region that is reclaimable dead
    /// space. The keymap is written once per distinct value (not per document),
    /// so a fixed value set never bloats it — but under **distinct-value churn**
    /// (values that appear and are later fully removed, freeing their slot) the
    /// removed entries' bytes accumulate as dead space until compacted.
    pub fn keymap_waste_ratio(&self) -> f64 {
        self.keymap_store.as_ref().map_or(0.0, |ks| ks.waste_ratio())
    }

    /// Snapshot of this field's on-disk blob growth and reclaimable waste, for
    /// monitoring the append-only write amplification (worst for low-cardinality
    /// fields — see `index/CLAUDE.md`). Cheap: reads cached header fields and
    /// scans live slots, no blob deserialisation.
    pub fn blob_stats(&self) -> IndexBlobStats {
        let (bitmap_logical_bytes, bitmap_live_bytes) = match &self.inner {
            DynFieldIndexInner::Bool(fi) => fi.bitmap_blob_bytes(),
            DynFieldIndexInner::Int(fi) => fi.bitmap_blob_bytes(),
            DynFieldIndexInner::Str(fi) => fi.bitmap_blob_bytes(),
        };
        let (keymap_logical_bytes, keymap_live_bytes) = self.keymap_store.as_ref().map_or((0, 0), |ks| (ks.logical_bytes(), ks.live_bytes()));
        IndexBlobStats {
            distinct_values: self.distinct_count(),
            bitmap_logical_bytes,
            bitmap_live_bytes,
            bitmap_waste_ratio: self.bitmap_waste_ratio(),
            keymap_logical_bytes,
            keymap_live_bytes,
            keymap_waste_ratio: self.keymap_waste_ratio(),
        }
    }

    /// Compact the bitmap store and/or the keymap store whose waste ratio
    /// reaches `waste_threshold` (a fraction, `0.0..1.0`), reclaiming dead
    /// space. Returns `Ok(true)` if either store was compacted, `Ok(false)` if
    /// both were below threshold.
    ///
    /// The two stores key on the same slot IDs and tombstone the same slots
    /// together (a value's slot is freed in both when its bitmap empties), so
    /// compacting them independently keeps their live slot sets consistent.
    ///
    /// The caller is expected to [`flush`] afterwards to persist the shrink.
    ///
    /// [`flush`]: DynFieldIndex::flush
    pub fn maybe_compact(&mut self, waste_threshold: f64) -> std::io::Result<bool> {
        let mut compacted = false;
        if self.bitmap_waste_ratio() >= waste_threshold {
            match &mut self.inner {
                DynFieldIndexInner::Bool(fi) => fi.compact_bitmaps()?,
                DynFieldIndexInner::Int(fi) => fi.compact_bitmaps()?,
                DynFieldIndexInner::Str(fi) => fi.compact_bitmaps()?,
            };
            compacted = true;
        }
        if self.keymap_waste_ratio() >= waste_threshold
            && let Some(ks) = &mut self.keymap_store
        {
            ks.compact()?;
            compacted = true;
        }
        Ok(compacted)
    }

    /// Returns the type discriminant for this index.
    pub fn value_type(&self) -> IndexValueType {
        match &self.inner {
            DynFieldIndexInner::Bool(_) => IndexValueType::Bool,
            DynFieldIndexInner::Int(_) => IndexValueType::Int,
            DynFieldIndexInner::Str(_) => IndexValueType::Str,
        }
    }

    /// Returns the current number of distinct indexed values.
    pub fn distinct_count(&self) -> usize {
        match &self.inner {
            DynFieldIndexInner::Bool(fi) => fi.distinct_count(),
            DynFieldIndexInner::Int(fi) => fi.distinct_count(),
            DynFieldIndexInner::Str(fi) => fi.distinct_count(),
        }
    }

    /// Whether any bitmap blob has failed to load or store since this index was
    /// opened — i.e. a query may have silently dropped rows. The owner should
    /// rebuild this field from the WAL when this is `true`. Latches once set.
    /// See [`FieldIndex::corruption_detected`].
    pub fn corruption_detected(&self) -> bool {
        match &self.inner {
            DynFieldIndexInner::Bool(fi) => fi.corruption_detected(),
            DynFieldIndexInner::Int(fi) => fi.corruption_detected(),
            DynFieldIndexInner::Str(fi) => fi.corruption_detected(),
        }
    }

    /// Record that `row_id` has `value` for this field.
    ///
    /// If this is the first row for `value`, the new slot is also written to
    /// the keymap mmap store for immediate durability.
    ///
    /// # Errors
    /// Returns an error string on type mismatch (runtime type of `value` does
    /// not match the variant this index was created with).
    pub fn insert(&mut self, value: &IndexValue, row_id: u128) -> Result<(), String> {
        match (&mut self.inner, value) {
            (DynFieldIndexInner::Bool(idx), IndexValue::Bool(v)) => {
                let prev = idx.next_slot();
                idx.insert(*v, row_id);
                if idx.next_slot() != prev
                    && let Some(ks) = &mut self.keymap_store
                {
                    let slot_id = idx.slot_id_for(v).unwrap();
                    ks.upsert(slot_id, &serialize_bool(*v));
                }
            }
            (DynFieldIndexInner::Int(idx), IndexValue::Int(v)) => {
                let prev = idx.next_slot();
                idx.insert(*v, row_id);
                if idx.next_slot() != prev
                    && let Some(ks) = &mut self.keymap_store
                {
                    let slot_id = idx.slot_id_for(v).unwrap();
                    ks.upsert(slot_id, &serialize_int(*v));
                }
            }
            (DynFieldIndexInner::Str(idx), IndexValue::Str(v)) => {
                let prev = idx.next_slot();
                idx.insert(v.clone(), row_id);
                if idx.next_slot() != prev
                    && let Some(ks) = &mut self.keymap_store
                {
                    let slot_id = idx.slot_id_for(v).unwrap();
                    ks.upsert(slot_id, &serialize_str(v));
                }
            }
            _ => {
                return Err(format!(
                    "type mismatch: index holds {:?} values but received {:?}",
                    self.value_type(),
                    value
                ));
            }
        }
        Ok(())
    }

    /// Scalar update: make `value` the **only** value `row_id` holds for this
    /// field, clearing it from every other bucket first.
    ///
    /// This is the safe single-call API for single-valued fields: it is
    /// [`remove_all_for_row`](Self::remove_all_for_row) followed by
    /// [`insert`](Self::insert), so the row ends up under exactly one value and
    /// the caller cannot forget the clear step that a bare `insert` requires.
    /// For genuinely multi-valued fields call [`insert`](Self::insert) directly.
    ///
    /// # Errors
    /// Returns the same type-mismatch error as [`insert`](Self::insert). The row
    /// has already been cleared from all buckets when this returns `Err`, so a
    /// type mismatch leaves the field with no value for the row (matching the
    /// clear-then-insert order callers used before this method existed).
    pub fn set(&mut self, value: &IndexValue, row_id: u128) -> Result<(), String> {
        self.remove_all_for_row(row_id);
        self.insert(value, row_id)
    }

    /// Targeted scalar update when the caller knows `row_id`'s **previous** value
    /// for this field: move the row from `old`'s bucket to `new`'s.
    ///
    /// This is `O(1)` — it touches at most the two affected buckets — unlike
    /// [`set`](Self::set) / [`remove_all_for_row`](Self::remove_all_for_row),
    /// which scan **every** value bucket to find the row (`O(distinct values)`,
    /// painful for high-cardinality fields). The document layer already has the
    /// old document, so it can supply `old` and avoid the scan.
    ///
    /// Cases: `old == new` is a no-op (value unchanged); `old = Some, new = None`
    /// removes the row (field gone); `old = None, new = Some` is a plain insert
    /// (first time / fresh row); `old = None, new = None` is a no-op. Use this
    /// only when `old` is **known correct** — if it is stale the row would be
    /// left in its real old bucket; callers that can't be sure should fall back
    /// to [`set`](Self::set) / [`remove_all_for_row`](Self::remove_all_for_row).
    ///
    /// # Errors
    /// Type mismatch from the `new` insert (same as [`insert`](Self::insert));
    /// the `old` removal has already happened when this returns `Err`.
    pub fn update(&mut self, old: Option<&IndexValue>, new: Option<&IndexValue>, row_id: u128) -> Result<(), String> {
        if old == new {
            return Ok(());
        }
        if let Some(o) = old {
            self.remove(o, row_id);
        }
        match new {
            Some(n) => self.insert(n, row_id),
            None => Ok(()),
        }
    }

    /// Remove `row_id` from the bucket for `value`.
    ///
    /// If the value's bitmap becomes empty, the keymap entry is also removed
    /// from the mmap store.
    ///
    /// Returns `false` on type mismatch.
    pub fn remove(&mut self, value: &IndexValue, row_id: u128) -> bool {
        match (&mut self.inner, value) {
            (DynFieldIndexInner::Bool(idx), IndexValue::Bool(v)) => {
                let slot_id = idx.slot_id_for(v);
                idx.remove(*v, row_id);
                if let Some(sid) = slot_id
                    && !idx.contains_value(v)
                    && let Some(ks) = &mut self.keymap_store
                {
                    ks.remove_key(sid);
                }
                true
            }
            (DynFieldIndexInner::Int(idx), IndexValue::Int(v)) => {
                let slot_id = idx.slot_id_for(v);
                idx.remove(*v, row_id);
                if let Some(sid) = slot_id
                    && !idx.contains_value(v)
                    && let Some(ks) = &mut self.keymap_store
                {
                    ks.remove_key(sid);
                }
                true
            }
            (DynFieldIndexInner::Str(idx), IndexValue::Str(v)) => {
                let slot_id = idx.slot_id_for(v);
                idx.remove(v.clone(), row_id);
                if let Some(sid) = slot_id
                    && !idx.contains_value(v.as_str())
                    && let Some(ks) = &mut self.keymap_store
                {
                    ks.remove_key(sid);
                }
                true
            }
            _ => false,
        }
    }

    /// Remove `row_id` from every value bucket it appears in.
    ///
    /// Any value entries whose bitmaps become empty are also purged from the
    /// keymap mmap store.
    pub fn remove_all_for_row(&mut self, row_id: u128) {
        let removed_slots = match &mut self.inner {
            DynFieldIndexInner::Bool(idx) => idx.remove_all_for_row(row_id),
            DynFieldIndexInner::Int(idx) => idx.remove_all_for_row(row_id),
            DynFieldIndexInner::Str(idx) => idx.remove_all_for_row(row_id),
        };
        if let Some(ks) = &mut self.keymap_store {
            for slot_id in removed_slots {
                ks.remove_key(slot_id);
            }
        }
    }

    /// Call `f` for every bitmap across all value buckets.
    pub fn for_each_bitmap(&self, mut f: impl FnMut(&RoaringBitmap)) {
        match &self.inner {
            DynFieldIndexInner::Bool(fi) => {
                for (_, bm) in fi.iter() {
                    f(&bm);
                }
            }
            DynFieldIndexInner::Int(fi) => {
                for (_, bm) in fi.iter() {
                    f(&bm);
                }
            }
            DynFieldIndexInner::Str(fi) => {
                for (_, bm) in fi.iter() {
                    f(&bm);
                }
            }
        }
    }
}

// ── Value serialisation helpers ────────────────────────────────────────────
//
// Each value type has a simple byte encoding used in the keymap BlobStore.
// Format:
//   Bool → 1B  (0x00 or 0x01)
//   Int  → 8B  LE i64
//   Str  → raw UTF-8 bytes

#[inline]
fn serialize_bool(v: bool) -> Vec<u8> {
    vec![u8::from(v)]
}

#[inline]
fn serialize_int(v: i64) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

#[inline]
fn serialize_str(v: &str) -> Vec<u8> {
    v.as_bytes().to_vec()
}

// ── Rebuild BTreeMap ordering from keymap BlobStore ────────────────────────

fn rebuild_ordering_bool(store: &BlobStore) -> (BTreeMap<bool, u128>, u128) {
    let mut map = BTreeMap::new();
    let mut max_slot: Option<u128> = None;
    for (slot_id, bytes) in store.iter_entries() {
        if bytes.is_empty() {
            continue;
        }
        let v = bytes[0] != 0;
        map.insert(v, slot_id);
        max_slot = Some(max_slot.map_or(slot_id, |m: u128| m.max(slot_id)));
    }
    let next_slot = max_slot.map_or(0, |m| m + 1);
    (map, next_slot)
}

fn rebuild_ordering_int(store: &BlobStore) -> (BTreeMap<i64, u128>, u128) {
    let mut map = BTreeMap::new();
    let mut max_slot: Option<u128> = None;
    for (slot_id, bytes) in store.iter_entries() {
        if bytes.len() < 8 {
            continue;
        }
        let v = i64::from_le_bytes(bytes[..8].try_into().unwrap());
        map.insert(v, slot_id);
        max_slot = Some(max_slot.map_or(slot_id, |m: u128| m.max(slot_id)));
    }
    let next_slot = max_slot.map_or(0, |m| m + 1);
    (map, next_slot)
}

fn rebuild_ordering_str(store: &BlobStore) -> (BTreeMap<String, u128>, u128) {
    let mut map = BTreeMap::new();
    let mut max_slot: Option<u128> = None;
    for (slot_id, bytes) in store.iter_entries() {
        let Ok(v) = String::from_utf8(bytes) else { continue };
        map.insert(v, slot_id);
        max_slot = Some(max_slot.map_or(slot_id, |m: u128| m.max(slot_id)));
    }
    let next_slot = max_slot.map_or(0, |m| m + 1);
    (map, next_slot)
}

// ── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_insert_query() {
        let mut idx = DynFieldIndex::new(IndexValueType::Bool);
        idx.insert(&IndexValue::Bool(true), 1).unwrap();
        idx.insert(&IndexValue::Bool(false), 2).unwrap();
        idx.insert(&IndexValue::Bool(true), 3).unwrap();
        assert_eq!(idx.distinct_count(), 2);
    }

    #[test]
    fn test_int_insert_query() {
        let mut idx = DynFieldIndex::new(IndexValueType::Int);
        idx.insert(&IndexValue::Int(42), 10).unwrap();
        idx.insert(&IndexValue::Int(-7), 20).unwrap();
        idx.insert(&IndexValue::Int(42), 30).unwrap();
        assert_eq!(idx.distinct_count(), 2);
    }

    #[test]
    fn set_replaces_scalar_value_through_public_api() {
        // The scalar-update contract via the type-erased API: set() makes the
        // given value the row's only value for the field.
        let mut idx = DynFieldIndex::new(IndexValueType::Int);
        idx.set(&IndexValue::Int(42), 1).unwrap();
        idx.set(&IndexValue::Int(7), 1).unwrap(); // update row 1: 42 -> 7
        assert_eq!(query_int_eq(&idx, 7), vec![1], "row now matches its new value");
        assert!(query_int_eq(&idx, 42).is_empty(), "row no longer matches its old value");
        assert_eq!(idx.distinct_count(), 1, "no stale value bucket left behind");
    }

    #[test]
    fn update_moves_row_between_buckets_without_scanning() {
        let mut idx = DynFieldIndex::new(IndexValueType::Int);
        // Many distinct values so the row's value is unrelated to bucket count;
        // update must hit only the old + new buckets.
        for v in 0..100i64 {
            idx.insert(&IndexValue::Int(v), v as u128).unwrap();
        }
        // Row 5 currently has value 5; move it to a brand-new value 1000.
        idx.update(Some(&IndexValue::Int(5)), Some(&IndexValue::Int(1000)), 5).unwrap();
        assert_eq!(query_int_eq(&idx, 1000), vec![5]);
        assert!(query_int_eq(&idx, 5).is_empty(), "old value no longer matches the row");
        // Untouched rows are intact.
        assert_eq!(query_int_eq(&idx, 50), vec![50]);
    }

    #[test]
    fn update_unchanged_value_is_a_noop() {
        let mut idx = DynFieldIndex::new(IndexValueType::Int);
        idx.insert(&IndexValue::Int(7), 1).unwrap();
        let before = idx.distinct_count();
        // old == new ⇒ nothing changes, no bucket churn.
        idx.update(Some(&IndexValue::Int(7)), Some(&IndexValue::Int(7)), 1).unwrap();
        assert_eq!(query_int_eq(&idx, 7), vec![1]);
        assert_eq!(idx.distinct_count(), before);
    }

    #[test]
    fn update_none_to_some_is_a_fresh_insert() {
        let mut idx = DynFieldIndex::new(IndexValueType::Int);
        idx.update(None, Some(&IndexValue::Int(9)), 1).unwrap();
        assert_eq!(query_int_eq(&idx, 9), vec![1]);
    }

    #[test]
    fn update_some_to_none_removes_the_row() {
        let mut idx = DynFieldIndex::new(IndexValueType::Int);
        idx.insert(&IndexValue::Int(9), 1).unwrap();
        idx.update(Some(&IndexValue::Int(9)), None, 1).unwrap();
        assert!(query_int_eq(&idx, 9).is_empty());
        assert_eq!(idx.distinct_count(), 0, "emptied value bucket is dropped");
    }

    #[test]
    fn set_update_survives_reopen_with_keymap() {
        // A scalar update through set() must persist correctly: the emptied old
        // value's keymap entry is purged and the new one is durable across reopen.
        let dir = tempfile::TempDir::new().unwrap();
        {
            let mut idx = DynFieldIndex::open(IndexValueType::Int, dir.path()).unwrap();
            idx.set(&IndexValue::Int(42), 1).unwrap();
            idx.set(&IndexValue::Int(7), 1).unwrap();
            idx.flush(dir.path()).unwrap();
        }
        let idx = DynFieldIndex::open(IndexValueType::Int, dir.path()).unwrap();
        assert_eq!(query_int_eq(&idx, 7), vec![1]);
        assert!(query_int_eq(&idx, 42).is_empty());
        assert_eq!(idx.distinct_count(), 1);
    }

    #[test]
    fn test_str_insert_query() {
        let mut idx = DynFieldIndex::new(IndexValueType::Str);
        idx.insert(&IndexValue::Str("hello".into()), 1).unwrap();
        idx.insert(&IndexValue::Str("world".into()), 2).unwrap();
        idx.insert(&IndexValue::Str("hello".into()), 3).unwrap();
        assert_eq!(idx.distinct_count(), 2);
    }

    #[test]
    fn test_remove_all_for_row() {
        let mut idx = DynFieldIndex::new(IndexValueType::Int);
        idx.insert(&IndexValue::Int(1), 100).unwrap();
        idx.insert(&IndexValue::Int(2), 100).unwrap();
        idx.insert(&IndexValue::Int(1), 200).unwrap();
        idx.remove_all_for_row(100);
        assert_eq!(idx.distinct_count(), 1);
    }

    #[test]
    fn test_type_mismatch_returns_error() {
        let mut idx = DynFieldIndex::new(IndexValueType::Bool);
        assert!(idx.insert(&IndexValue::Int(5), 1).is_err());
    }

    #[test]
    fn test_insert_many_distinct_values() {
        let mut idx = DynFieldIndex::new(IndexValueType::Int);
        for i in 0..10_000i64 {
            idx.insert(&IndexValue::Int(i), i as u128).unwrap();
        }
        assert_eq!(idx.distinct_count(), 10_000);
    }

    #[test]
    fn distinct_count_counts_unique_values_not_total_rows() {
        let mut idx = DynFieldIndex::new(IndexValueType::Str);
        assert_eq!(idx.distinct_count(), 0);
        idx.insert(&IndexValue::Str("a".into()), 1).unwrap();
        idx.insert(&IndexValue::Str("a".into()), 2).unwrap();
        assert_eq!(idx.distinct_count(), 1);
        idx.insert(&IndexValue::Str("b".into()), 3).unwrap();
        assert_eq!(idx.distinct_count(), 2);
    }

    #[test]
    fn test_open_flush_roundtrip_int() {
        let dir = tempfile::TempDir::new().unwrap();
        {
            let mut idx = DynFieldIndex::open(IndexValueType::Int, dir.path()).unwrap();
            idx.insert(&IndexValue::Int(42), 10).unwrap();
            idx.insert(&IndexValue::Int(-7), 20).unwrap();
            idx.flush(dir.path()).unwrap();
        }
        let idx = DynFieldIndex::open(IndexValueType::Int, dir.path()).unwrap();
        assert_eq!(idx.distinct_count(), 2);
    }

    #[test]
    fn test_open_flush_roundtrip_str() {
        let dir = tempfile::TempDir::new().unwrap();
        {
            let mut idx = DynFieldIndex::open(IndexValueType::Str, dir.path()).unwrap();
            idx.insert(&IndexValue::Str("hello".into()), 1).unwrap();
            idx.insert(&IndexValue::Str("world".into()), 2).unwrap();
            idx.flush(dir.path()).unwrap();
        }
        let idx = DynFieldIndex::open(IndexValueType::Str, dir.path()).unwrap();
        assert_eq!(idx.distinct_count(), 2);
    }

    #[test]
    fn test_open_flush_roundtrip_bool() {
        let dir = tempfile::TempDir::new().unwrap();
        {
            let mut idx = DynFieldIndex::open(IndexValueType::Bool, dir.path()).unwrap();
            idx.insert(&IndexValue::Bool(true), 1).unwrap();
            idx.insert(&IndexValue::Bool(false), 2).unwrap();
            idx.flush(dir.path()).unwrap();
        }
        let idx = DynFieldIndex::open(IndexValueType::Bool, dir.path()).unwrap();
        assert_eq!(idx.distinct_count(), 2);
    }

    #[test]
    fn blob_stats_surfaces_low_cardinality_bloat_and_compaction() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut idx = DynFieldIndex::open(IndexValueType::Bool, dir.path()).unwrap();
        // Worst case for append-only write amplification: a single value (true)
        // over many rows. Every insert re-appends the whole (growing) bitmap, so
        // logical bytes balloon while the live footprint stays one small bitmap.
        for row in 0..2_000u128 {
            idx.insert(&IndexValue::Bool(true), row).unwrap();
        }

        let before = idx.blob_stats();
        assert_eq!(before.distinct_values, 1);
        assert!(
            before.bitmap_logical_bytes > before.bitmap_live_bytes.saturating_mul(4),
            "append-only rewrites must bloat logical >> live: {before:?}"
        );
        assert!(before.bitmap_waste_ratio > 0.5, "waste should be high under bloat: {before:?}");

        // Compaction reclaims the dead space the metric reported.
        idx.maybe_compact(0.0).unwrap();
        let after = idx.blob_stats();
        assert!(
            after.bitmap_logical_bytes <= before.bitmap_logical_bytes / 2,
            "compaction must shrink logical bytes: before={before:?} after={after:?}"
        );
        assert!(after.bitmap_waste_ratio < 0.1, "waste reads ≈0 after compaction: {after:?}");
        assert_eq!(after.distinct_values, 1, "live data is unchanged by compaction");
    }

    #[test]
    fn test_mmap_keymap_remove_cleans_up() {
        let dir = tempfile::TempDir::new().unwrap();
        {
            let mut idx = DynFieldIndex::open(IndexValueType::Int, dir.path()).unwrap();
            idx.insert(&IndexValue::Int(10), 1).unwrap();
            idx.insert(&IndexValue::Int(20), 2).unwrap();
            // Remove the only row for value 10 — keymap entry should be purged.
            idx.remove(&IndexValue::Int(10), 1);
            idx.flush(dir.path()).unwrap();
        }
        let idx = DynFieldIndex::open(IndexValueType::Int, dir.path()).unwrap();
        assert_eq!(idx.distinct_count(), 1);
    }

    #[test]
    fn test_mmap_keymap_remove_all_for_row_cleans_up() {
        let dir = tempfile::TempDir::new().unwrap();
        {
            let mut idx = DynFieldIndex::open(IndexValueType::Int, dir.path()).unwrap();
            idx.insert(&IndexValue::Int(1), 100).unwrap();
            idx.insert(&IndexValue::Int(2), 100).unwrap();
            idx.insert(&IndexValue::Int(1), 200).unwrap();
            idx.remove_all_for_row(100);
            idx.flush(dir.path()).unwrap();
        }
        let idx = DynFieldIndex::open(IndexValueType::Int, dir.path()).unwrap();
        // Value 2 had only row 100 → removed; value 1 still has row 200.
        assert_eq!(idx.distinct_count(), 1);
    }

    // ── Compaction (the checkpoint's actual entry point) ─────────────────────

    use crate::field::predicate::Predicate;

    /// Collect the rows for `value` from an `Int` index via the typed query path.
    fn int_rows(idx: &DynFieldIndex, value: i64) -> Vec<u128> {
        match &idx.inner {
            DynFieldIndexInner::Int(fi) => fi.evaluate(&Predicate::Eq(value)).iter().collect(),
            _ => unreachable!("expected Int index"),
        }
    }

    #[test]
    fn maybe_compact_respects_threshold_and_preserves_queries() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut idx = DynFieldIndex::open(IndexValueType::Int, dir.path()).unwrap();

        // Two distinct values over many rows: each insert re-serialises and
        // appends the whole bitmap, so the append-only value region accumulates
        // dead space (the bloat compaction exists to reclaim).
        for row in 0..2_000u128 {
            let v = if row % 2 == 0 { 1 } else { 2 };
            idx.insert(&IndexValue::Int(v), row).unwrap();
        }

        let waste = idx.bitmap_waste_ratio();
        assert!(waste > 0.0, "append-only rewrites must accumulate waste, got {waste}");

        // Below threshold (waste can never reach 100%) → no-op.
        assert!(!idx.maybe_compact(1.0).unwrap(), "waste below threshold must be a no-op");

        // At/above threshold → compaction runs and reclaims the dead space.
        assert!(idx.maybe_compact(0.0).unwrap(), "threshold 0 must always compact");
        assert!(idx.bitmap_waste_ratio() < waste, "compaction must reduce waste");

        // Queries still return the exact rows after compaction.
        let evens: Vec<u128> = (0..2_000u128).filter(|r| r % 2 == 0).collect();
        let odds: Vec<u128> = (0..2_000u128).filter(|r| r % 2 == 1).collect();
        assert_eq!(int_rows(&idx, 1), evens);
        assert_eq!(int_rows(&idx, 2), odds);
        assert_eq!(idx.distinct_count(), 2);
    }

    #[test]
    fn compact_then_reopen_preserves_int_field_index() {
        let dir = tempfile::TempDir::new().unwrap();
        {
            let mut idx = DynFieldIndex::open(IndexValueType::Int, dir.path()).unwrap();
            for row in 0..1_500u128 {
                idx.insert(&IndexValue::Int((row % 3) as i64), row).unwrap();
            }
            assert!(idx.bitmap_waste_ratio() > 0.0);
            assert!(idx.maybe_compact(0.0).unwrap());
            idx.flush(dir.path()).unwrap();
        }

        // Reopen from disk: the staged-swap-compacted bitmap store is loaded and
        // the `ordering` map is rebuilt from the keymap — the two must still align.
        let idx = DynFieldIndex::open(IndexValueType::Int, dir.path()).unwrap();
        assert_eq!(idx.distinct_count(), 3);
        for v in 0..3i64 {
            let expected: Vec<u128> = (0..1_500u128).filter(|r| (*r % 3) as i64 == v).collect();
            assert_eq!(int_rows(&idx, v), expected, "value {v} rows after compact + reopen");
        }
    }

    #[test]
    fn keymap_compaction_reclaims_churn_and_survives_reopen() {
        let dir = tempfile::TempDir::new().unwrap();
        {
            let mut idx = DynFieldIndex::open(IndexValueType::Int, dir.path()).unwrap();

            // Distinct-value churn: create many distinct values (each appends a
            // keymap entry), then fully remove each (emptying its bitmap frees
            // the slot in both stores, leaving dead keymap space behind).
            for v in 0..500i64 {
                idx.insert(&IndexValue::Int(v), v as u128).unwrap();
            }
            for v in 0..500i64 {
                idx.remove(&IndexValue::Int(v), v as u128);
            }
            // A handful of survivors that must remain queryable after compaction.
            for v in 1_000..1_005i64 {
                idx.insert(&IndexValue::Int(v), v as u128).unwrap();
            }

            let keymap_waste = idx.keymap_waste_ratio();
            assert!(keymap_waste > 0.0, "churn must leave reclaimable keymap dead space, got {keymap_waste}");
            assert!(idx.maybe_compact(0.0).unwrap(), "compaction must run");
            assert!(idx.keymap_waste_ratio() < keymap_waste, "compaction must reduce keymap waste");
            idx.flush(dir.path()).unwrap();
        }

        // Reopen: the `ordering` map is rebuilt from the compacted keymap.
        let idx = DynFieldIndex::open(IndexValueType::Int, dir.path()).unwrap();
        assert_eq!(idx.distinct_count(), 5);
        match &idx.inner {
            DynFieldIndexInner::Int(fi) => {
                for v in 1_000..1_005i64 {
                    assert_eq!(int_rows(&idx, v), vec![v as u128], "survivor {v} after keymap compact + reopen");
                }
                // A churned-out value must not resurrect.
                assert!(fi.evaluate(&Predicate::Eq(0)).is_empty(), "removed value must stay gone");
            }
            _ => unreachable!("expected Int index"),
        }
    }

    #[test]
    fn compact_then_reopen_preserves_str_field_index() {
        let dir = tempfile::TempDir::new().unwrap();
        let labels = ["alpha", "beta", "gamma"];
        {
            let mut idx = DynFieldIndex::open(IndexValueType::Str, dir.path()).unwrap();
            for row in 0..1_500u128 {
                idx.insert(&IndexValue::Str(labels[row as usize % labels.len()].into()), row).unwrap();
            }
            assert!(idx.bitmap_waste_ratio() > 0.0);
            assert!(idx.maybe_compact(0.0).unwrap());
            idx.flush(dir.path()).unwrap();
        }

        // Variable-length string values exercise the UTF-8 keymap rebuild path.
        let idx = DynFieldIndex::open(IndexValueType::Str, dir.path()).unwrap();
        assert_eq!(idx.distinct_count(), labels.len());
        match &idx.inner {
            DynFieldIndexInner::Str(fi) => {
                for (i, label) in labels.iter().enumerate() {
                    let rows: Vec<u128> = fi.evaluate(&Predicate::Eq((*label).to_string())).iter().collect();
                    let expected: Vec<u128> = (0..1_500u128).filter(|r| *r as usize % labels.len() == i).collect();
                    assert_eq!(rows, expected, "label {label} rows after compact + reopen");
                }
            }
            _ => unreachable!("expected Str index"),
        }
    }

    // ── Bitmap/keymap skew (crash mid-flush) ────────────────────────────────
    //
    // flush() msyncs the bitmap store then the keymap store as two separate
    // steps with no marker between them, so a crash in between leaves the two
    // stores at different logical points. These tests reproduce that skew by
    // rolling one store's files back to an earlier snapshot (each snapshot is
    // itself a valid flushed state) and assert that reopening is non-fatal and
    // that re-applying the op — as the owner's WAL replay would — reconciles it.
    // See the module-level *Crash atomicity* docs.

    fn query_int_eq(idx: &DynFieldIndex, v: i64) -> Vec<u128> {
        match &idx.inner {
            DynFieldIndexInner::Int(fi) => fi.evaluate(&Predicate::Eq(v)).iter().collect(),
            _ => unreachable!("expected Int index"),
        }
    }

    #[test]
    fn skew_keymap_lags_bitmap_is_nonfatal_and_reconcilable() {
        let dir = tempfile::TempDir::new().unwrap();
        let keymap = dir.path().join("keymap");

        // Durable state knowing only value 100 (slot 0).
        {
            let mut idx = DynFieldIndex::open(IndexValueType::Int, dir.path()).unwrap();
            idx.insert(&IndexValue::Int(100), 0).unwrap();
            idx.flush(dir.path()).unwrap();
        }
        let ks_keys = std::fs::read(keymap.join("blobs.keys")).unwrap();
        let ks_vals = std::fs::read(keymap.join("blobs.vals")).unwrap();

        // Add value 200 (slot 1) and flush both stores durably.
        {
            let mut idx = DynFieldIndex::open(IndexValueType::Int, dir.path()).unwrap();
            idx.insert(&IndexValue::Int(200), 1).unwrap();
            idx.flush(dir.path()).unwrap();
        }

        // Roll the KEYMAP back: the bitmap store still has slot 1's bitmap, but
        // the keymap no longer maps slot 1 → 200 — exactly the state a crash
        // after the bitmap flush but before the keymap flush leaves behind.
        std::fs::write(keymap.join("blobs.keys"), &ks_keys).unwrap();
        std::fs::write(keymap.join("blobs.vals"), &ks_vals).unwrap();

        // Reopen: must not panic. The intact value survives; the torn value's
        // bitmap is orphaned (no keymap entry) so it queries empty.
        let mut idx = DynFieldIndex::open(IndexValueType::Int, dir.path()).unwrap();
        assert_eq!(query_int_eq(&idx, 100), vec![0], "intact value survives the skew");
        assert_eq!(query_int_eq(&idx, 200), Vec::<u128>::new(), "torn value's keymap entry was lost");

        // Owner reconciliation: replaying the same insert restores consistency.
        idx.insert(&IndexValue::Int(200), 1).unwrap();
        idx.flush(dir.path()).unwrap();
        assert_eq!(query_int_eq(&idx, 200), vec![1], "re-insert (WAL replay) reconciles the skew");
    }

    #[test]
    fn skew_bitmap_lags_keymap_is_nonfatal_and_reconcilable() {
        let dir = tempfile::TempDir::new().unwrap();

        // Durable state knowing only value 100 (slot 0).
        {
            let mut idx = DynFieldIndex::open(IndexValueType::Int, dir.path()).unwrap();
            idx.insert(&IndexValue::Int(100), 0).unwrap();
            idx.flush(dir.path()).unwrap();
        }
        let bm_keys = std::fs::read(dir.path().join("blobs.keys")).unwrap();
        let bm_vals = std::fs::read(dir.path().join("blobs.vals")).unwrap();

        // Add value 200 (slot 1) and flush both stores durably.
        {
            let mut idx = DynFieldIndex::open(IndexValueType::Int, dir.path()).unwrap();
            idx.insert(&IndexValue::Int(200), 1).unwrap();
            idx.flush(dir.path()).unwrap();
        }

        // Roll the BITMAP store back: the keymap still maps slot 1 → 200, but the
        // bitmap store lacks slot 1's bitmap — the reverse skew (crash after the
        // keymap flush but before the bitmap flush).
        std::fs::write(dir.path().join("blobs.keys"), &bm_keys).unwrap();
        std::fs::write(dir.path().join("blobs.vals"), &bm_vals).unwrap();

        // Reopen: must not panic. Value 200 is known to the ordering but its
        // bitmap is missing, so it queries empty rather than crashing.
        let mut idx = DynFieldIndex::open(IndexValueType::Int, dir.path()).unwrap();
        assert_eq!(query_int_eq(&idx, 100), vec![0], "intact value survives the skew");
        assert_eq!(query_int_eq(&idx, 200), Vec::<u128>::new(), "torn value's bitmap is missing");

        // Re-applying the insert (WAL replay) restores the missing bitmap bit.
        idx.insert(&IndexValue::Int(200), 1).unwrap();
        idx.flush(dir.path()).unwrap();
        assert_eq!(query_int_eq(&idx, 200), vec![1], "re-insert (WAL replay) reconciles the skew");
    }
}
