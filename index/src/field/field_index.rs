use std::collections::BTreeMap;
use std::ops::Bound;

use crate::RoaringBitmap;
use crate::blob_store::BlobStore;
use crate::storage;

use super::predicate::Predicate;

/// A single-field index mapping ordered field values to sets of row IDs.
///
/// Backed by two structures:
/// - `ordering`: an in-memory `BTreeMap<V, u128>` mapping each distinct field
///   value to a stable slot ID. This is small (keys only, no bitmap data) and
///   provides the sorted iteration needed for range predicates.
/// - `bitmaps`: a [`BlobStore`] mapping slot IDs to serialised [`RoaringBitmap`]
///   data. This can be backed by anonymous mmaps (for transient / test use) or
///   by persistent files under the field's on-disk directory.
///
/// Because the bitmap data lives in the `BlobStore` (off-heap for file-backed
/// stores), there is no inherent cardinality cap from a memory-pressure
/// perspective.
///
/// # This index is multi-valued; the scalar invariant is **caller-enforced**
///
/// `FieldIndex` is an inverted `value → rows` index, so a single `row_id` may
/// appear under **several** values at once — [`insert`](Self::insert) adds the
/// row to one value's bucket and never touches the others. That is correct and
/// intended for genuinely multi-valued fields (e.g. a `tags` array).
///
/// For a **scalar** field (one value per row), "a row is in at most one bucket"
/// is an invariant the **caller** must maintain — `insert` does *not* enforce
/// it. Updating a scalar field with a bare `insert` *adds* a second value
/// rather than replacing the old one, which then makes `Eq` / `Ne` / range /
/// `In` report the row under both values. Use [`set`](Self::set) for scalar
/// updates: it clears the row from every bucket and then inserts the new value
/// in one call, so callers cannot forget the clear step. (`minnal_db`'s index
/// hook does exactly this.) A reverse `row → slot` map is deliberately *not*
/// kept — see [`remove_all_for_row`](Self::remove_all_for_row).
#[derive(Debug)]
pub struct FieldIndex<V: Ord + Clone> {
    ordering: BTreeMap<V, u128>,
    bitmaps: BlobStore,
    next_slot: u128,
}

impl<V: Ord + Clone> FieldIndex<V> {
    /// Create an empty index backed by an anonymous (transient) mmap.
    pub fn new() -> Self {
        Self {
            ordering: BTreeMap::new(),
            bitmaps: BlobStore::new_anon(),
            next_slot: 0,
        }
    }

    /// Reconstruct an index from a pre-loaded ordering map and an already-open
    /// [`BlobStore`].
    ///
    /// Called by [`DynFieldIndex`] after rebuilding the ordering from the
    /// keymap mmap store.
    pub(crate) fn from_parts(ordering: BTreeMap<V, u128>, bitmaps: BlobStore, next_slot: u128) -> Self {
        Self {
            ordering,
            bitmaps,
            next_slot,
        }
    }

    /// Record that `row_id` has `value` for this field.
    ///
    /// Inserting the same `(value, row_id)` pair twice is idempotent. This adds
    /// the row to `value`'s bucket **without** removing it from any other value
    /// it may already be under — see the type-level docs. For a scalar field,
    /// prefer [`set`](Self::set), which replaces rather than accumulates.
    pub fn insert(&mut self, value: V, row_id: u128) {
        let slot_id = *self.ordering.entry(value).or_insert_with(|| {
            let id = self.next_slot;
            self.next_slot += 1;
            id
        });
        let mut bm = self.load_bitmap(slot_id);
        bm.insert(row_id);
        self.store_bitmap(slot_id, &bm);
    }

    /// Scalar update: make `value` the **only** value `row_id` holds for this
    /// field.
    ///
    /// Equivalent to [`remove_all_for_row`](Self::remove_all_for_row) followed
    /// by [`insert`](Self::insert): the row is cleared from every bucket it
    /// currently occupies and then inserted under `value`, so afterwards it
    /// appears under exactly one value (the scalar invariant). Returns the slot
    /// IDs of any buckets that became empty during the clear (so a `DynFieldIndex`
    /// can purge the matching keymap entries).
    ///
    /// This is the safe single-call API for single-valued fields; use it instead
    /// of remembering to clear before each `insert`. For genuinely multi-valued
    /// fields, call [`insert`](Self::insert) directly.
    pub fn set(&mut self, value: V, row_id: u128) -> Vec<u128> {
        let removed = self.remove_all_for_row(row_id);
        self.insert(value, row_id);
        removed
    }

    /// Remove `row_id` from the entry for `value`.
    ///
    /// Removes the map entry entirely when the bitmap becomes empty.
    pub fn remove(&mut self, value: V, row_id: u128) {
        let Some(&slot_id) = self.ordering.get(&value) else { return };
        let mut bm = self.load_bitmap(slot_id);
        bm.remove(row_id);
        if bm.is_empty() {
            self.bitmaps.remove_key(slot_id);
            self.ordering.remove(&value);
        } else {
            self.store_bitmap(slot_id, &bm);
        }
    }

    /// Return the set of row IDs whose field value satisfies `predicate`.
    pub fn evaluate(&self, predicate: &Predicate<V>) -> RoaringBitmap {
        let mut result = RoaringBitmap::new();
        match predicate {
            Predicate::Eq(v) => {
                if let Some(&slot_id) = self.ordering.get(v) {
                    return self.load_bitmap(slot_id);
                }
            }
            Predicate::Ne(v) => {
                for (k, &slot_id) in &self.ordering {
                    if k != v {
                        result.or_inplace(&self.load_bitmap(slot_id));
                    }
                }
            }
            Predicate::Lt(v) => {
                for (_, &slot_id) in self.ordering.range((Bound::Unbounded, Bound::Excluded(v.clone()))) {
                    result.or_inplace(&self.load_bitmap(slot_id));
                }
            }
            Predicate::Le(v) => {
                for (_, &slot_id) in self.ordering.range((Bound::Unbounded, Bound::Included(v.clone()))) {
                    result.or_inplace(&self.load_bitmap(slot_id));
                }
            }
            Predicate::Gt(v) => {
                for (_, &slot_id) in self.ordering.range((Bound::Excluded(v.clone()), Bound::Unbounded)) {
                    result.or_inplace(&self.load_bitmap(slot_id));
                }
            }
            Predicate::Ge(v) => {
                for (_, &slot_id) in self.ordering.range((Bound::Included(v.clone()), Bound::Unbounded)) {
                    result.or_inplace(&self.load_bitmap(slot_id));
                }
            }
            Predicate::Between { lo, hi } => {
                for (_, &slot_id) in self.ordering.range((Bound::Included(lo.clone()), Bound::Included(hi.clone()))) {
                    result.or_inplace(&self.load_bitmap(slot_id));
                }
            }
            Predicate::In(values) => {
                for v in values {
                    if let Some(&slot_id) = self.ordering.get(v) {
                        result.or_inplace(&self.load_bitmap(slot_id));
                    }
                }
            }
        }
        result
    }

    /// Remove `row_id` from every value bucket it appears in.
    ///
    /// Returns the slot IDs of value entries whose bitmaps became empty and
    /// were removed from the index.
    ///
    /// Correct for both scalar and multi-valued use: it scans **all** buckets
    /// and writes back **every** bucket that actually contained `row_id` (so a
    /// multi-valued row is removed from all of them). Buckets that did not
    /// contain it are left untouched — this is load-bearing for high-cardinality
    /// fields, since re-serialising and appending *every* bucket on each
    /// update/delete would be O(distinct) write amplification against the
    /// append-only bitmap store. For a scalar field the common case touches a
    /// single bucket, so the write-back cost is one bitmap.
    ///
    /// The scan still *loads* each bucket to test membership; making this O(1)
    /// would need a `row → slot` reverse index, deliberately not kept — it would
    /// double the per-write index mutations and its own append-only churn for a
    /// structure that is already rebuilt from the WAL on recovery.
    pub fn remove_all_for_row(&mut self, row_id: u128) -> Vec<u128> {
        // Collect slot IDs and which values to drop before any mutation to
        // satisfy the borrow checker (can't borrow ordering immutably and
        // mutably at the same time).
        let slots: Vec<(V, u128)> = self.ordering.iter().map(|(v, &id)| (v.clone(), id)).collect();

        let mut empty_values: Vec<V> = Vec::new();
        let mut removed_slots: Vec<u128> = Vec::new();
        for (value, slot_id) in slots {
            let mut bm = self.load_bitmap(slot_id);
            // Only the bucket that held the row changes — skip the write-back for
            // every other bucket (avoids appending an unchanged bitmap copy).
            if !bm.remove(row_id) {
                continue;
            }
            if bm.is_empty() {
                self.bitmaps.remove_key(slot_id);
                empty_values.push(value);
                removed_slots.push(slot_id);
            } else {
                self.store_bitmap(slot_id, &bm);
            }
        }
        for v in empty_values {
            self.ordering.remove(&v);
        }
        removed_slots
    }

    /// Returns `true` if `value` already has an entry in the index.
    pub fn contains_value<Q>(&self, value: &Q) -> bool
    where
        V: std::borrow::Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.ordering.contains_key(value)
    }

    /// Look up the slot ID assigned to `value`, or `None` if the value has not
    /// been indexed.
    pub(crate) fn slot_id_for(&self, value: &V) -> Option<u128> {
        self.ordering.get(value).copied()
    }

    /// Iterate over all `(value, bitmap)` pairs in sorted value order.
    ///
    /// Each bitmap is deserialised on demand; the iterator is lazy over the
    /// `BTreeMap` but eagerly deserialises each bitmap as it is visited.
    pub fn iter(&self) -> impl Iterator<Item = (&V, RoaringBitmap)> {
        self.ordering.iter().map(|(v, &slot_id)| (v, self.load_bitmap(slot_id)))
    }

    /// Number of distinct indexed values currently stored.
    pub fn distinct_count(&self) -> usize {
        self.ordering.len()
    }

    /// The next slot ID to be assigned.
    pub(crate) fn next_slot(&self) -> u128 {
        self.next_slot
    }

    /// Flush the underlying [`BlobStore`] mmap to disk.
    ///
    /// No-op for anonymous (transient) stores.
    pub fn flush(&self) -> std::io::Result<()> {
        self.bitmaps.flush()
    }

    /// Fraction (`0.0..1.0`) of the bitmap value region that is reclaimable
    /// dead space. See [`BlobStore::waste_ratio`].
    pub fn bitmap_waste_ratio(&self) -> f64 {
        self.bitmaps.waste_ratio()
    }

    /// Compact the bitmap value region, reclaiming dead space left by the
    /// per-insert whole-bitmap rewrites. See [`BlobStore::compact`].
    pub fn compact_bitmaps(&mut self) -> std::io::Result<u64> {
        self.bitmaps.compact()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn load_bitmap(&self, slot_id: u128) -> RoaringBitmap {
        match self.bitmaps.get(slot_id) {
            Some(bytes) => match storage::deserialize(&bytes) {
                Ok(bm) => bm,
                Err(e) => {
                    tracing::warn!(slot = slot_id, error = %e, "load_bitmap: deserialization failed, returning empty");
                    RoaringBitmap::new()
                }
            },
            None => RoaringBitmap::new(),
        }
    }

    fn store_bitmap(&mut self, slot_id: u128, bm: &RoaringBitmap) {
        let bytes = storage::serialize(bm).unwrap_or_default();
        self.bitmaps.upsert(slot_id, &bytes);
    }
}

impl<V: Ord + Clone> Default for FieldIndex<V> {
    fn default() -> Self {
        Self::new()
    }
}

// ── std::fmt::Debug for BlobStore ─────────────────────────────────────────────

impl std::fmt::Debug for BlobStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlobStore").field("count", &self.count()).finish()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_is_idempotent() {
        let mut idx = FieldIndex::<i64>::new();
        idx.insert(7, 1);
        idx.insert(7, 1); // same (value, row) again
        assert_eq!(idx.evaluate(&Predicate::Eq(7)).iter().collect::<Vec<_>>(), vec![1]);
        assert_eq!(idx.distinct_count(), 1);
    }

    #[test]
    fn remove_all_for_row_returns_only_emptied_slots() {
        let mut idx = FieldIndex::<i64>::new();
        idx.insert(1, 100);
        idx.insert(2, 100);
        idx.insert(1, 200);
        // Row 100 empties value 2 (its only row) but not value 1 (row 200 remains).
        let emptied = idx.remove_all_for_row(100);
        assert_eq!(emptied.len(), 1, "only value 2's slot becomes empty");
        assert!(!idx.contains_value(&2i64));
        assert!(idx.contains_value(&1i64));
        assert_eq!(idx.evaluate(&Predicate::Eq(1)).iter().collect::<Vec<_>>(), vec![200]);
        assert_eq!(idx.distinct_count(), 1);
    }

    #[test]
    fn insert_is_multivalued_a_row_can_be_under_several_values() {
        // Documented contract: insert() accumulates — a bare insert of a second
        // value puts the row under BOTH (correct for multi-valued fields, and the
        // footgun for scalar callers who forget to clear first).
        let mut idx = FieldIndex::<i64>::new();
        idx.insert(1, 100);
        idx.insert(2, 100); // second value for the same row, no clear
        assert_eq!(idx.evaluate(&Predicate::Eq(1)).iter().collect::<Vec<_>>(), vec![100]);
        assert_eq!(idx.evaluate(&Predicate::Eq(2)).iter().collect::<Vec<_>>(), vec![100]);
    }

    #[test]
    fn set_enforces_one_value_per_row() {
        // set() is the scalar API: it replaces rather than accumulates.
        let mut idx = FieldIndex::<i64>::new();
        idx.set(1, 100);
        assert_eq!(idx.evaluate(&Predicate::Eq(1)).iter().collect::<Vec<_>>(), vec![100]);

        // Updating row 100 to value 2 removes it from value 1.
        let emptied = idx.set(2, 100);
        assert_eq!(idx.evaluate(&Predicate::Eq(2)).iter().collect::<Vec<_>>(), vec![100]);
        assert!(idx.evaluate(&Predicate::Eq(1)).is_empty(), "old value no longer matches the row");
        assert_eq!(emptied.len(), 1, "value 1's bucket emptied and is reported");
        assert_eq!(idx.distinct_count(), 1, "row 100 is under exactly one value");
    }

    #[test]
    fn set_leaves_other_rows_under_the_same_value_intact() {
        let mut idx = FieldIndex::<i64>::new();
        idx.set(7, 1);
        idx.set(7, 2); // two rows share value 7
        // Re-pointing row 1 to value 9 must not disturb row 2's membership in 7.
        idx.set(9, 1);
        assert_eq!(idx.evaluate(&Predicate::Eq(7)).iter().collect::<Vec<_>>(), vec![2]);
        assert_eq!(idx.evaluate(&Predicate::Eq(9)).iter().collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn remove_last_row_drops_value_entry() {
        let mut idx = FieldIndex::<i64>::new();
        idx.insert(5, 1);
        assert!(idx.contains_value(&5i64));
        idx.remove(5, 1);
        assert!(!idx.contains_value(&5i64));
        assert_eq!(idx.distinct_count(), 0);
        assert!(idx.evaluate(&Predicate::Eq(5)).is_empty());
    }

    #[test]
    fn iter_yields_values_in_sorted_order() {
        let mut idx = FieldIndex::<i64>::new();
        idx.insert(30, 3);
        idx.insert(10, 1);
        idx.insert(20, 2);
        let values: Vec<i64> = idx.iter().map(|(v, _)| *v).collect();
        assert_eq!(values, vec![10, 20, 30]);
    }

    #[test]
    fn update_only_rewrites_the_affected_bucket() {
        let mut idx = FieldIndex::<i64>::new();
        // High cardinality: 200 distinct values, one row each.
        for v in 0..200i64 {
            idx.insert(v, v as u128);
        }

        // Re-assign row 0 to a fresh distinct value 100 times. With the
        // per-bucket write-back each reassignment touches only the old and new
        // buckets, so the append-only bitmap store's waste stays bounded; the
        // previous "rewrite every bucket" behaviour re-appended all ~200 bitmaps
        // on every update, driving waste toward 1.0.
        for v in 1_000..1_100i64 {
            idx.remove_all_for_row(0);
            idx.insert(v, 0);
        }

        // Correctness: row 0 ended up in value 1099 only; its prior values are gone.
        assert_eq!(idx.evaluate(&Predicate::Eq(1099)).iter().collect::<Vec<_>>(), vec![0]);
        assert!(idx.evaluate(&Predicate::Eq(0)).is_empty());
        assert!(idx.evaluate(&Predicate::Eq(1050)).is_empty());
        // The other 199 untouched values still resolve to their single row.
        assert_eq!(idx.evaluate(&Predicate::Eq(50)).iter().collect::<Vec<_>>(), vec![50]);

        // Bounded write amplification: updates must not have re-appended every
        // bucket (which would push waste toward ~0.99).
        let waste = idx.bitmap_waste_ratio();
        assert!(waste < 0.8, "update must not rewrite every bucket; waste={waste}");
    }

    #[test]
    fn load_bitmap_returns_empty_on_corrupt_blob() {
        let mut idx = FieldIndex::<i64>::new();
        idx.insert(7, 1);
        let slot = idx.slot_id_for(&7).unwrap();
        // A blob claiming one container but carrying no key bytes fails the
        // length-framing check in storage::deserialize, so load_bitmap falls
        // back to an empty bitmap instead of propagating an error.
        idx.bitmaps.upsert(slot, &1u32.to_le_bytes());
        assert!(idx.load_bitmap(slot).is_empty());
    }

    #[test]
    fn load_bitmap_returns_empty_on_corrupt_rkyv_payload() {
        // The recoverable path the checked-deserialization fix protects: a blob
        // with *valid framing* but an invalid rkyv payload. Under the old
        // `access_unchecked` this was UB/panic inside load_bitmap; with checked
        // access it surfaces an error that load_bitmap swallows into an empty
        // bitmap. Framing = [count=1][16B key][blob_len=32][32 garbage bytes].
        let mut idx = FieldIndex::<i64>::new();
        idx.insert(7, 1);
        let slot = idx.slot_id_for(&7).unwrap();

        let mut blob = Vec::new();
        blob.extend_from_slice(&1u32.to_le_bytes());
        blob.extend_from_slice(&0u128.to_le_bytes());
        blob.extend_from_slice(&32u32.to_le_bytes());
        blob.extend_from_slice(&[0xFFu8; 32]);
        idx.bitmaps.upsert(slot, &blob);

        assert!(idx.load_bitmap(slot).is_empty());
    }
}
