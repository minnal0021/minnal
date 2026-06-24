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
}
