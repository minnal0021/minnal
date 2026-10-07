use std::collections::{BTreeMap, HashMap};
use std::ops::Bound;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::index::RoaringBitmap;
use crate::index::blob_store::BlobStore;
use crate::index::overlay_budget::IndexOverlayBudget;
use crate::index::storage;

use super::predicate::Predicate;

/// A single-field index mapping ordered field values to sets of row IDs.
///
/// Backed by two structures:
/// - `ordering`: an in-memory `BTreeMap<V, u128>` mapping each distinct field
///   value to a stable slot ID. This is small (keys only, no bitmap data) and
///   provides the sorted iteration needed for range predicates.
/// - `bitmaps`: a `BlobStore` mapping slot IDs to serialised [`RoaringBitmap`]
///   data. This can be backed by anonymous mmaps (for transient / test use) or
///   by persistent files under the field's on-disk directory.
///
/// Because the bitmap data lives in the `BlobStore` (off-heap for file-backed
/// stores), there is no inherent cardinality cap from a memory-pressure
/// perspective.
///
/// # Changes are buffered in memory until the next spill
///
/// A write never touches the blob store. The bitmaps a write changes are kept
/// in an in-memory **overlay** (`slot → bitmap`), read in preference to the
/// store, and written out once by [`spill`](Self::spill) — at the index
/// checkpoint, or sooner when the shared [`IndexOverlayBudget`] runs out. The
/// blob store is append-only, so this turns "one copy of the whole bitmap per
/// write" into "one copy per changed bitmap per spill".
///
/// A slot whose bitmap empties stays reserved until the spill: its value goes
/// from `ordering` to `emptied`, and the store blob (and the owner's keymap
/// entry) are removed only by the spill. Re-inserting the value before then
/// reuses the slot, so a value never maps to two slots on disk.
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
    /// Latches `true` the first time a bitmap blob fails to **deserialise** on
    /// read or **serialise** on write. The query/mutation methods stay infallible
    /// (a failed read serves an empty bitmap, a failed write leaves the prior
    /// blob untouched), but this flag makes the corruption observable so the
    /// owner can distinguish "no rows matched" from "index data failed to load"
    /// and rebuild the field from the WAL. Never auto-cleared; reset only by
    /// rebuilding the index. See [`corruption_detected`](Self::corruption_detected).
    corrupted: AtomicBool,
    /// Bitmaps changed since the last spill, by slot. An empty bitmap is a slot
    /// that emptied; the spill removes its blob.
    overlay: HashMap<u128, RoaringBitmap>,
    /// Values whose bitmap emptied since the last spill, with the slot they
    /// keep until then (see the type docs).
    emptied: BTreeMap<V, u128>,
    /// Heap bytes the overlay holds, as charged to `budget`.
    overlay_bytes: u64,
    /// Shared budget the overlay is charged against.
    budget: Arc<IndexOverlayBudget>,
}

/// Fixed cost charged per overlay entry beyond the bitmap's own heap bytes.
const OVERLAY_ENTRY_OVERHEAD: u64 = 64;

impl<V: Ord + Clone> FieldIndex<V> {
    /// Create an empty index backed by an anonymous (transient) mmap.
    pub fn new() -> Self {
        Self::from_parts(BTreeMap::new(), BlobStore::new_anon(), 0)
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
            corrupted: AtomicBool::new(false),
            overlay: HashMap::new(),
            emptied: BTreeMap::new(),
            overlay_bytes: 0,
            budget: Arc::new(IndexOverlayBudget::default()),
        }
    }

    /// Charge this index's overlay to `budget` instead of its own private one.
    /// Bytes already held move from the old budget to the new.
    pub fn set_overlay_budget(&mut self, budget: Arc<IndexOverlayBudget>) {
        let held = self.overlay_bytes as i64;
        self.budget.charge(-held);
        budget.charge(held);
        self.budget = budget;
    }

    /// The budget this index's overlay is charged against.
    pub fn overlay_budget(&self) -> &Arc<IndexOverlayBudget> {
        &self.budget
    }

    /// Heap bytes held by this index's overlay.
    pub fn overlay_bytes(&self) -> u64 {
        self.overlay_bytes
    }

    /// Write every bitmap in the overlay to the blob store and empty it.
    ///
    /// Changed bitmaps are serialised once each and appended; bitmaps that
    /// emptied have their blobs removed. Returns the slots that were freed —
    /// the owner removes their keymap entries. This copies into the store's
    /// memory map but does not `msync`: [`flush`](Self::flush) does that.
    pub fn spill(&mut self) -> Vec<u128> {
        for (slot_id, bm) in std::mem::take(&mut self.overlay) {
            if bm.is_empty() {
                self.bitmaps.remove_key(slot_id);
            } else {
                self.store_bitmap(slot_id, &bm);
            }
        }
        self.charge(-(self.overlay_bytes as i64));
        std::mem::take(&mut self.emptied).into_values().collect()
    }

    fn charge(&mut self, delta: i64) {
        self.overlay_bytes = (self.overlay_bytes as i64 + delta).max(0) as u64;
        self.budget.charge(delta);
    }

    /// Run `f` on the overlay copy of `slot_id`'s bitmap, loading it from the
    /// store first if it is not in the overlay yet. Keeps the budget current.
    /// Returns `f`'s result and whether the bitmap is now empty.
    fn mutate_slot<R>(&mut self, slot_id: u128, f: impl FnOnce(&mut RoaringBitmap) -> R) -> (R, bool) {
        if !self.overlay.contains_key(&slot_id) {
            let bm = self.load_from_store(slot_id);
            self.charge(bm.heap_bytes() as i64 + OVERLAY_ENTRY_OVERHEAD as i64);
            self.overlay.insert(slot_id, bm);
        }
        let bm = self.overlay.get_mut(&slot_id).expect("just inserted");
        let before = bm.heap_bytes() as i64;
        let r = f(bm);
        let (after, empty) = (bm.heap_bytes() as i64, bm.is_empty());
        self.charge(after - before);
        (r, empty)
    }

    /// The slot for `value`, allocating one (or reclaiming the slot it emptied
    /// since the last spill) if it has none.
    fn slot_for_insert(&mut self, value: V) -> u128 {
        if let Some(&id) = self.ordering.get(&value) {
            return id;
        }
        let id = match self.emptied.remove(&value) {
            Some(id) => id,
            None => {
                let id = self.next_slot;
                self.next_slot += 1;
                id
            }
        };
        self.ordering.insert(value, id);
        id
    }

    /// Whether any bitmap blob has failed to load or store since this index was
    /// opened.
    ///
    /// A `true` here means at least one query may have silently returned fewer
    /// rows than it should (a corrupt blob is served as empty) or a write was
    /// dropped — the index is **derived**, so the owner should rebuild this
    /// field from the WAL rather than trust further results. Latches once set.
    pub fn corruption_detected(&self) -> bool {
        self.corrupted.load(Ordering::Relaxed)
    }

    /// Record that `row_id` has `value` for this field.
    ///
    /// Inserting the same `(value, row_id)` pair twice is idempotent. This adds
    /// the row to `value`'s bucket **without** removing it from any other value
    /// it may already be under — see the type-level docs. For a scalar field,
    /// prefer [`set`](Self::set), which replaces rather than accumulates.
    pub fn insert(&mut self, value: V, row_id: u128) {
        self.insert_many(value, std::slice::from_ref(&row_id));
    }

    /// Record that every row in `row_ids` has `value` for this field.
    ///
    /// The point of this over a loop of [`insert`](Self::insert) is that the
    /// bitmap is loaded and **re-serialised once for the whole batch** rather
    /// than once per row. The blob store is append-only, so a per-row loop
    /// leaves one dead copy of the entire bitmap behind per row — the dominant
    /// cost when many rows share one value, which is exactly the shape of a
    /// low-cardinality field.
    ///
    /// Used by WAL replay, which knows its full key set up front and can group
    /// by value before writing. An empty `row_ids` is a no-op and does not
    /// allocate a slot.
    pub fn insert_many(&mut self, value: V, row_ids: &[u128]) {
        if row_ids.is_empty() {
            return;
        }
        let slot_id = self.slot_for_insert(value);
        self.mutate_slot(slot_id, |bm| {
            for &row_id in row_ids {
                bm.insert(row_id);
            }
        });
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
        let (_, empty) = self.mutate_slot(slot_id, |bm| bm.remove(row_id));
        if empty {
            self.ordering.remove(&value);
            self.emptied.insert(value, slot_id);
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
    /// The scan still *loads* each bucket to test membership — `O(distinct
    /// values)`. **When the caller knows the row's old value** (the document
    /// layer always does, from the prior document), prefer the `O(1)` targeted
    /// path — [`remove(old, row)`](Self::remove), or `DynFieldIndex::update` /
    /// `DynFieldIndex::remove` — which touches only the affected bucket. This
    /// full scan is the fallback for when the old value is unknown. A persistent
    /// `row → slot` reverse index is **deliberately not kept** (it would double
    /// the per-write index mutations and add its own append-only churn for a
    /// structure already rebuilt from the WAL on recovery); supplying the old
    /// value gives the same `O(1)` without that cost.
    pub fn remove_all_for_row(&mut self, row_id: u128) -> Vec<u128> {
        self.remove_all_for_rows(std::slice::from_ref(&row_id))
    }

    /// Clear every row in `row_ids` from every bucket it occupies.
    ///
    /// One load and at most one write-back **per bucket for the whole batch**,
    /// rather than per row. The single-row form loads every bucket's bitmap and
    /// rewrites the one that changed, so clearing N rows costs N loads of every
    /// bucket and N appended copies — the same append-only quadratic that makes
    /// a per-key `insert` loop expensive.
    ///
    /// Used by WAL replay, which clears its whole affected row set before
    /// re-inserting. An empty `row_ids` is a no-op.
    pub fn remove_all_for_rows(&mut self, row_ids: &[u128]) -> Vec<u128> {
        if row_ids.is_empty() {
            return Vec::new();
        }
        let slots: Vec<(V, u128)> = self.ordering.iter().map(|(v, &id)| (v.clone(), id)).collect();

        let mut removed_slots: Vec<u128> = Vec::new();
        for (value, slot_id) in slots {
            // Probe first, so a bucket none of these rows is in is neither
            // loaded whole nor added to the overlay. A stored bucket is probed
            // on its serialised form, decoding only the rows' containers.
            if !self.slot_contains_any(slot_id, row_ids) {
                continue;
            }
            let (_, empty) = self.mutate_slot(slot_id, |bm| {
                for &row_id in row_ids {
                    bm.remove(row_id);
                }
            });
            if empty {
                self.ordering.remove(&value);
                self.emptied.insert(value, slot_id);
                removed_slots.push(slot_id);
            }
        }
        removed_slots
    }

    /// Whether `slot_id`'s bitmap holds any of `row_ids`, without loading the
    /// whole bitmap when it is only in the store.
    fn slot_contains_any(&self, slot_id: u128, row_ids: &[u128]) -> bool {
        if let Some(bm) = self.overlay.get(&slot_id) {
            return row_ids.iter().any(|&r| bm.contains(r));
        }
        match self.bitmaps.get(slot_id) {
            None => false,
            Some(bytes) => storage::contains_any(&bytes, row_ids).unwrap_or_else(|e| {
                // Same policy as `load_bitmap`: treat as empty, flag corruption.
                self.corrupted.store(true, Ordering::Relaxed);
                log::error!(
                    "slot_contains_any: bitmap blob failed to decode; treating as empty and flagging the index corrupt (slot={slot_id}, error={e})"
                );
                false
            }),
        }
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

    /// Flush the underlying `BlobStore` mmap to disk.
    ///
    /// Writes only what has been [`spill`](Self::spill)ed: overlay changes are
    /// not in the store yet. No-op for anonymous (transient) stores.
    pub fn flush(&self) -> std::io::Result<()> {
        self.bitmaps.flush()
    }

    /// Fraction (`0.0..1.0`) of the bitmap value region that is reclaimable
    /// dead space. See `BlobStore::waste_ratio`.
    pub fn bitmap_waste_ratio(&self) -> f64 {
        self.bitmaps.waste_ratio()
    }

    /// `(logical, live)` bytes of the bitmap value region: total bytes ever
    /// appended vs. what survives compaction. The gap is reclaimable dead space
    /// from the append-only whole-bitmap rewrites — large for low-cardinality
    /// fields. See `BlobStore::logical_bytes` / `BlobStore::live_bytes`.
    pub fn bitmap_blob_bytes(&self) -> (u64, u64) {
        (self.bitmaps.logical_bytes(), self.bitmaps.live_bytes())
    }

    /// Reclaimable dead bytes in the bitmap value region — **O(1)** (see
    /// `BlobStore::dead_bytes`). Used to drive checkpoint backpressure without a
    /// per-write waste-ratio scan.
    pub fn bitmap_dead_bytes(&self) -> u64 {
        self.bitmaps.dead_bytes()
    }

    /// Compact the bitmap value region, reclaiming dead space left by the
    /// per-insert whole-bitmap rewrites. See `BlobStore::compact`.
    pub fn compact_bitmaps(&mut self) -> std::io::Result<u64> {
        self.bitmaps.compact()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// The current bitmap for `slot_id`: the overlay copy if the slot changed
    /// since the last spill, else the stored one.
    fn load_bitmap(&self, slot_id: u128) -> RoaringBitmap {
        match self.overlay.get(&slot_id) {
            Some(bm) => bm.clone(),
            None => self.load_from_store(slot_id),
        }
    }

    fn load_from_store(&self, slot_id: u128) -> RoaringBitmap {
        match self.bitmaps.get(slot_id) {
            // A *missing* slot is a normal empty bitmap, not corruption.
            None => RoaringBitmap::new(),
            Some(bytes) => match storage::deserialize(&bytes) {
                Ok(bm) => bm,
                Err(e) => {
                    // The blob exists but won't deserialise: serve empty (so the
                    // query/mutation stays infallible) but latch the corruption
                    // flag so the owner can rebuild instead of trusting a result
                    // that silently dropped this value's rows.
                    self.corrupted.store(true, Ordering::Relaxed);
                    log::error!(
                        "load_bitmap: bitmap blob failed to deserialize; serving empty and flagging the index corrupt (rebuild this field from the WAL) (slot={slot_id}, error={e})"
                    );
                    RoaringBitmap::new()
                }
            },
        }
    }

    fn store_bitmap(&mut self, slot_id: u128, bm: &RoaringBitmap) {
        match storage::serialize(bm) {
            Ok(bytes) => {
                self.bitmaps.upsert(slot_id, &bytes);
            }
            Err(e) => {
                // Serialising a valid bitmap is effectively infallible, so this is
                // a genuine "should never happen". Do NOT write empty bytes (the
                // old `unwrap_or_default()` behaviour) — that would silently drop
                // every row under this value. Leave the prior blob intact and flag
                // the corruption for the owner to repair from the WAL.
                self.corrupted.store(true, Ordering::Relaxed);
                log::error!(
                    "store_bitmap: bitmap failed to serialize; keeping the prior blob and flagging the index corrupt (slot={slot_id}, error={e})"
                );
            }
        }
    }
}

impl<V: Ord + Clone> Default for FieldIndex<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Ord + Clone> Drop for FieldIndex<V> {
    /// Return the overlay's bytes to the shared budget. An index dropped without
    /// a spill (a dropped field) must not leave the budget looking full.
    fn drop(&mut self) {
        self.budget.charge(-(self.overlay_bytes as i64));
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
    fn healthy_index_reports_no_corruption() {
        let mut idx = FieldIndex::<i64>::new();
        idx.insert(7, 1);
        idx.insert(8, 2);
        assert_eq!(idx.evaluate(&Predicate::Eq(7)).iter().collect::<Vec<_>>(), vec![1]);
        assert!(!idx.corruption_detected(), "normal load/store must not flag corruption");
    }

    #[test]
    fn load_bitmap_returns_empty_on_corrupt_blob() {
        let mut idx = FieldIndex::<i64>::new();
        idx.insert(7, 1);
        idx.spill(); // the corruption below is in the store, not the overlay
        let slot = idx.slot_id_for(&7).unwrap();
        // A blob claiming one container but carrying no key bytes fails the
        // length-framing check in storage::deserialize, so load_bitmap falls
        // back to an empty bitmap instead of propagating an error.
        idx.bitmaps.upsert(slot, &1u32.to_le_bytes());
        assert!(idx.load_bitmap(slot).is_empty());
        // …but the failure is now observable rather than silent.
        assert!(idx.corruption_detected(), "a failed bitmap load must latch the corruption flag");
    }

    #[test]
    fn missing_slot_is_not_treated_as_corruption() {
        // A slot with no blob at all is a legitimate empty bitmap, not corruption.
        let idx = FieldIndex::<i64>::new();
        assert!(idx.evaluate(&Predicate::Eq(123)).is_empty());
        assert!(!idx.corruption_detected());
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
        idx.spill(); // the corruption below is in the store, not the overlay
        let slot = idx.slot_id_for(&7).unwrap();

        let mut blob = Vec::new();
        blob.extend_from_slice(&1u32.to_le_bytes());
        blob.extend_from_slice(&0u128.to_le_bytes());
        blob.extend_from_slice(&32u32.to_le_bytes());
        blob.extend_from_slice(&[0xFFu8; 32]);
        idx.bitmaps.upsert(slot, &blob);

        assert!(idx.load_bitmap(slot).is_empty());
        assert!(idx.corruption_detected(), "an invalid rkyv payload must latch the corruption flag");
    }
}
