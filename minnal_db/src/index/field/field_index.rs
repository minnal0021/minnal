use std::collections::{BTreeMap, HashMap};
use std::ops::Bound;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::index::RoaringBitmap;
use crate::index::bitmap::decompose;
use crate::index::blob_store::{BlobLayout, BlobStore};
use crate::index::container::Container;
use crate::index::overlay_budget::IndexOverlayBudget;
use crate::index::storage::{self, DirEntry};

use super::predicate::Predicate;

/// A single-field index mapping ordered field values to sets of row IDs.
///
/// Backed by two structures:
/// - `ordering`: an in-memory `BTreeMap<V, u128>` mapping each distinct field
///   value to a stable slot ID. This is small (keys only, no bitmap data) and
///   provides the sorted iteration needed for range predicates.
/// - `bitmaps`: a `BlobStore` in the directory layout. Each slot points at a
///   directory listing that value's [`RoaringBitmap`] containers, each stored
///   as its own blob (see `index::storage`). This can be backed by anonymous
///   mmaps (for transient / test use) or by persistent files under the field's
///   on-disk directory.
///
/// Because the bitmap data lives in the `BlobStore` (off-heap for file-backed
/// stores), there is no inherent cardinality cap from a memory-pressure
/// perspective.
///
/// # Changes are buffered in memory until the next spill
///
/// A write never touches the blob store. The **containers** a write changes
/// (a container holds the row ids sharing their upper 112 bits, at most 8 KB)
/// are kept in an in-memory **overlay** (`slot → container key → container`),
/// read in preference to the store, and written out once by
/// [`spill`](Self::spill) — at the index checkpoint, or sooner when the shared
/// [`IndexOverlayBudget`] runs out. A spill appends each changed container and
/// a new directory for its slot; containers that did not change are not
/// rewritten. So the cost of a spill follows what changed, not the size of the
/// bitmaps.
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
    /// Latches `true` the first time a stored directory or container fails to
    /// **decode** on read or a container fails to **encode** on write. The
    /// query/mutation methods stay infallible (a failed read serves an empty
    /// container, a failed write leaves the prior blob untouched), but this
    /// flag makes the corruption observable so the owner can distinguish "no
    /// rows matched" from "index data failed to load" and rebuild the field
    /// from the WAL. Never auto-cleared; reset only by rebuilding the index.
    /// See [`corruption_detected`](Self::corruption_detected).
    corrupted: AtomicBool,
    /// Containers changed since the last spill, by slot.
    overlay: HashMap<u128, SlotChanges>,
    /// Values whose bitmap emptied since the last spill, with the slot they
    /// keep until then (see the type docs).
    emptied: BTreeMap<V, u128>,
    /// Version of each overlay entry, bumped on every change, so a spill commit
    /// can tell whether the entry changed after it was staged.
    overlay_ver: HashMap<u128, u64>,
    next_ver: u64,
    /// Heap bytes the overlay holds, as charged to `budget`.
    overlay_bytes: u64,
    /// Shared budget the overlay is charged against.
    budget: Arc<IndexOverlayBudget>,
}

/// One slot's changes since the last spill.
#[derive(Debug, Default)]
struct SlotChanges {
    /// The slot's stored directory, decoded on first touch so later writes to
    /// the slot need not decode it again. `None` until loaded, and reset when
    /// the stored directory changes under it (a commit or a compaction).
    base: Option<Vec<DirEntry>>,
    /// Changed containers: the new contents, or `None` for a container that
    /// emptied (the spill drops it from the directory).
    containers: BTreeMap<u128, Option<Container>>,
    /// The slot version at each container's last change, so a commit can drop
    /// the containers it wrote out even when others changed after the stage.
    container_ver: HashMap<u128, u64>,
    /// Whether the slot's whole bitmap is now empty (the spill removes it).
    empty: bool,
    /// Bytes this entry has charged to the budget.
    bytes: u64,
}

/// Fixed cost charged per overlay slot beyond its containers.
const OVERLAY_SLOT_OVERHEAD: u64 = 64;
/// Fixed cost charged per overlay container beyond its own heap bytes.
const OVERLAY_CONTAINER_OVERHEAD: u64 = (std::mem::size_of::<(u128, Option<Container>)>() + 16) as u64;

/// Heap bytes charged for a cached directory.
fn base_bytes(base: &[DirEntry]) -> u64 {
    std::mem::size_of_val(base) as u64
}

/// Group row ids by container key: `high → [low]`.
fn group_rows(row_ids: &[u128]) -> BTreeMap<u128, Vec<u16>> {
    let mut groups: BTreeMap<u128, Vec<u16>> = BTreeMap::new();
    for &row in row_ids {
        let (high, low) = decompose(row);
        groups.entry(high).or_default().push(low);
    }
    groups
}

/// What [`FieldIndex::stage`] appended and [`FieldIndex::commit`] applies:
/// a new directory for each changed bitmap, and `(slot, overlay version)` for
/// emptied ones.
#[derive(Debug, Default)]
pub struct SpillStage {
    sets: Vec<StagedDir>,
    removals: Vec<(u128, u64)>,
}

/// One changed bitmap's staged directory.
#[derive(Debug)]
struct StagedDir {
    slot: u128,
    /// The slot's overlay version when staged.
    ver: u64,
    offset: u64,
    len: u32,
    /// `(container key, container version)` of every container staged.
    containers: Vec<(u128, u64)>,
}

impl<V: Ord + Clone> FieldIndex<V> {
    /// Create an empty index backed by an anonymous (transient) mmap.
    pub fn new() -> Self {
        Self::from_parts(BTreeMap::new(), BlobStore::new_anon_with(BlobLayout::Directory), 0)
    }

    /// Reconstruct an index from a pre-loaded ordering map and an already-open
    /// [`BlobStore`] in the directory layout.
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
            overlay_ver: HashMap::new(),
            next_ver: 0,
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

    /// Write every change in the overlay to the blob store and empty it:
    /// [`stage`](Self::stage), sync the values, [`commit`](Self::commit).
    /// Returns the slots that were freed (the owner removes their keymap
    /// entries). Syncs the value region; the key table is synced by
    /// [`flush`](Self::flush).
    pub fn spill(&mut self) -> Vec<u128> {
        let stage = self.stage();
        if let Err(e) = self.bitmaps.sync_values() {
            log::error!("FieldIndex::spill: syncing the bitmap values failed: {e}");
        }
        self.commit(stage)
    }

    /// Phase 1 of a spill: for each changed slot, append its changed
    /// containers and a new directory to the value region, pointing no slot at
    /// them yet. The overlay keeps serving reads.
    ///
    /// Slots must not point at a blob before the blob is on disk, and the
    /// kernel writes a shared memory map back in any order, so the slots are
    /// changed only by [`commit`](Self::commit), after the caller has synced
    /// the values ([`sync_values`](Self::sync_values)).
    pub fn stage(&mut self) -> SpillStage {
        let mut stage = SpillStage::default();
        let slots: Vec<u128> = self.overlay.keys().copied().collect();
        'slots: for slot_id in slots {
            let ver = self.overlay_ver[&slot_id];
            if self.overlay[&slot_id].empty {
                stage.removals.push((slot_id, ver));
                continue;
            }
            let base = match &self.overlay[&slot_id].base {
                Some(b) => b.clone(),
                None => self.stored_dir(slot_id),
            };
            let mut entries: BTreeMap<u128, DirEntry> = base.into_iter().map(|e| (e.key, e)).collect();
            let mut blobs = Vec::new();
            let staged: Vec<(u128, u64)> = self.overlay[&slot_id].container_ver.iter().map(|(&k, &v)| (k, v)).collect();
            for (&key, c) in &self.overlay[&slot_id].containers {
                match c {
                    None => {
                        entries.remove(&key);
                    }
                    Some(c) => match storage::encode_container(c) {
                        Ok(bytes) => blobs.push((key, bytes)),
                        Err(e) => {
                            // Keep the stored copy, flag it.
                            self.corrupted.store(true, Ordering::Relaxed);
                            log::error!("FieldIndex::stage: container failed to serialize; keeping the stored copy (slot={slot_id}, error={e})");
                            continue 'slots;
                        }
                    },
                }
            }
            for (key, bytes) in blobs {
                let (offset, len) = self.bitmaps.append_value(&bytes);
                entries.insert(key, DirEntry { key, offset, len });
            }
            let dir: Vec<DirEntry> = entries.into_values().collect();
            let (offset, len) = self.bitmaps.append_value(&storage::encode_dir(&dir));
            stage.sets.push(StagedDir {
                slot: slot_id,
                ver,
                offset,
                len,
                containers: staged,
            });
        }
        stage
    }

    /// Phase 2 of a spill: point the slots at the staged directories, remove
    /// the blobs of emptied slots, and drop every overlay entry that has not
    /// changed since it was staged (one that has stays for the next spill).
    /// Returns the slots freed for good.
    pub fn commit(&mut self, stage: SpillStage) -> Vec<u128> {
        let mut freed = Vec::new();
        for set in stage.sets {
            self.bitmaps.set_slot(set.slot, set.offset, set.len);
            if !self.drop_overlay_entry_if_unchanged(set.slot, set.ver) {
                // Written to after the stage: keep only the containers that
                // changed since, and reload the stored directory (it just
                // changed) on next touch. Without this a bitmap written in
                // every checkpoint window would never shed a container, and
                // every spill would rewrite all of them.
                self.drop_staged_containers(set.slot, &set.containers);
                self.forget_base(set.slot);
            }
        }
        for (slot_id, ver) in stage.removals {
            if self.drop_overlay_entry_if_unchanged(slot_id, ver) {
                self.bitmaps.remove_key(slot_id);
                if let Some(value) = self.emptied.iter().find(|(_, s)| **s == slot_id).map(|(v, _)| v.clone()) {
                    self.emptied.remove(&value);
                    freed.push(slot_id);
                }
            }
        }
        freed
    }

    /// Remove `slot_id` from the overlay if its version is still `ver`, and
    /// return its bytes to the budget. Returns whether it was removed.
    fn drop_overlay_entry_if_unchanged(&mut self, slot_id: u128, ver: u64) -> bool {
        if self.overlay_ver.get(&slot_id) != Some(&ver) {
            return false;
        }
        self.overlay_ver.remove(&slot_id);
        if let Some(ch) = self.overlay.remove(&slot_id) {
            self.charge(-(ch.bytes as i64));
        }
        true
    }

    /// Drop from `slot_id`'s overlay entry every container in `staged` whose
    /// version is unchanged: the commit just made the staged copy the stored
    /// one.
    fn drop_staged_containers(&mut self, slot_id: u128, staged: &[(u128, u64)]) {
        let Some(ch) = self.overlay.get_mut(&slot_id) else { return };
        let mut freed = 0u64;
        for &(key, ver) in staged {
            if ch.container_ver.get(&key) == Some(&ver) {
                ch.container_ver.remove(&key);
                if let Some(c) = ch.containers.remove(&key) {
                    freed += OVERLAY_CONTAINER_OVERHEAD + c.map_or(0, |c| c.heap_bytes() as u64);
                }
            }
        }
        ch.bytes = ch.bytes.saturating_sub(freed);
        self.charge(-(freed as i64));
    }

    /// Drop the cached stored directory of `slot_id`'s overlay entry.
    fn forget_base(&mut self, slot_id: u128) {
        if let Some(ch) = self.overlay.get_mut(&slot_id)
            && let Some(base) = ch.base.take()
        {
            let b = base_bytes(&base);
            ch.bytes -= b;
            self.charge(-(b as i64));
        }
    }

    /// Sync the bitmap store's value region to disk (between
    /// [`stage`](Self::stage) and [`commit`](Self::commit)).
    pub fn sync_values(&self) -> std::io::Result<()> {
        self.bitmaps.sync_values()
    }

    /// Sync the bitmap store's key table to disk (after [`commit`](Self::commit)).
    pub fn sync_keys(&self) -> std::io::Result<()> {
        self.bitmaps.sync_keys()
    }

    /// Slots dropped as damaged when the bitmap store was opened.
    pub(crate) fn damaged_at_open(&self) -> usize {
        self.bitmaps.damaged_at_open()
    }

    /// Mutable access to the bitmap store, for the owner's open-time
    /// reconciliation with its keymap.
    pub(crate) fn bitmaps_mut(&mut self) -> &mut BlobStore {
        &mut self.bitmaps
    }

    /// Every slot a value maps to.
    pub(crate) fn slots(&self) -> impl Iterator<Item = u128> + '_ {
        self.ordering.values().copied()
    }

    /// Every `(value, slot)` pair, copied (for open-time reconciliation).
    pub(crate) fn mapped_values(&self) -> Vec<(V, u128)> {
        self.ordering.iter().map(|(v, &s)| (v.clone(), s)).collect()
    }

    /// Drop `value` from the ordering without touching any store.
    pub(crate) fn forget_value(&mut self, value: &V) {
        self.ordering.remove(value);
    }

    /// Make sure new slot ids start at `at` or above.
    pub(crate) fn raise_next_slot(&mut self, at: u128) {
        self.next_slot = self.next_slot.max(at);
    }

    fn charge(&mut self, delta: i64) {
        self.overlay_bytes = (self.overlay_bytes as i64 + delta).max(0) as u64;
        self.budget.charge(delta);
    }

    /// Charge `delta` bytes to `slot_id`'s overlay entry and the budget.
    fn charge_slot(&mut self, slot_id: u128, delta: i64) {
        if let Some(ch) = self.overlay.get_mut(&slot_id) {
            ch.bytes = (ch.bytes as i64 + delta).max(0) as u64;
        }
        self.charge(delta);
    }

    /// Make sure `slot_id` has an overlay entry with its stored directory
    /// loaded, and bump its version (the caller is about to change it).
    fn touch(&mut self, slot_id: u128) {
        if let std::collections::hash_map::Entry::Vacant(e) = self.overlay.entry(slot_id) {
            e.insert(SlotChanges::default());
            self.charge_slot(slot_id, OVERLAY_SLOT_OVERHEAD as i64);
        }
        if self.overlay[&slot_id].base.is_none() {
            let base = self.stored_dir(slot_id);
            let b = base_bytes(&base);
            self.overlay.get_mut(&slot_id).expect("just inserted").base = Some(base);
            self.charge_slot(slot_id, b as i64);
        }
        self.overlay_ver.insert(slot_id, self.next_ver);
        self.next_ver += 1;
    }

    /// Run `f` on container `key` of `slot_id` in the overlay, loading it from
    /// the store on first touch. A container the slot does not hold is created
    /// empty when `create` is set; otherwise `f` is not called and nothing
    /// changes. A container left empty is recorded as removed. Keeps the
    /// budget and the slot's `empty` flag current. Returns `f`'s result (or
    /// `false` when `f` was not called).
    fn modify_container(&mut self, slot_id: u128, key: u128, create: bool, f: impl FnOnce(&mut Container) -> bool) -> bool {
        if !create && !self.slot_has_container(slot_id, key) {
            return false;
        }
        self.touch(slot_id);
        let ver = self.overlay_ver[&slot_id];
        self.overlay.get_mut(&slot_id).expect("touched").container_ver.insert(key, ver);
        if !self.overlay[&slot_id].containers.contains_key(&key) {
            let entry = self.overlay[&slot_id].base.as_deref().and_then(|b| storage::find_entry(b, key));
            let c = entry.and_then(|e| self.stored_container(slot_id, e));
            let charge = OVERLAY_CONTAINER_OVERHEAD + c.as_ref().map_or(0, |c| c.heap_bytes() as u64);
            self.overlay.get_mut(&slot_id).expect("touched").containers.insert(key, c);
            self.charge_slot(slot_id, charge as i64);
        }
        let ch = self.overlay.get_mut(&slot_id).expect("touched");
        let slot = ch.containers.get_mut(&key).expect("just loaded");
        if slot.is_none() {
            if !create {
                return false;
            }
            *slot = Some(Container::new_array());
        }
        let c = slot.as_mut().expect("set above");
        let before = c.heap_bytes() as i64;
        let changed = f(c);
        let (after, emptied) = if c.is_empty() {
            *slot = None;
            (0, true)
        } else {
            (c.heap_bytes() as i64, false)
        };
        if emptied {
            self.recompute_empty(slot_id);
        } else {
            self.overlay.get_mut(&slot_id).expect("touched").empty = false;
        }
        self.charge_slot(slot_id, after - before);
        changed
    }

    /// Whether `slot_id`'s current bitmap (overlay over store) has a
    /// container under `key`.
    fn slot_has_container(&self, slot_id: u128, key: u128) -> bool {
        if let Some(ch) = self.overlay.get(&slot_id) {
            if let Some(c) = ch.containers.get(&key) {
                return c.is_some();
            }
            if let Some(base) = &ch.base {
                return storage::find_entry(base, key).is_some();
            }
        }
        storage::find_entry(&self.stored_dir(slot_id), key).is_some()
    }

    /// Recompute whether `slot_id`'s bitmap is empty: every stored container
    /// removed in the overlay and no overlay container left.
    fn recompute_empty(&mut self, slot_id: u128) {
        let ch = self.overlay.get_mut(&slot_id).expect("touched");
        let any_live = ch.containers.values().any(Option::is_some);
        let any_stored_left = ch.base.as_deref().unwrap_or_default().iter().any(|e| !ch.containers.contains_key(&e.key));
        ch.empty = !any_live && !any_stored_left;
    }

    /// Whether the slot is now empty (only meaningful for a slot in the overlay).
    fn slot_is_empty(&self, slot_id: u128) -> bool {
        self.overlay.get(&slot_id).is_some_and(|ch| ch.empty)
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

    /// Whether any stored directory or container has failed to load, or a
    /// container to store, since this index was opened.
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
    /// value's slot is looked up once, and each container the rows fall in is
    /// fetched (loaded from the store on first touch) **once for the whole
    /// batch** rather than once per row.
    ///
    /// Used by WAL replay, which knows its full key set up front and can group
    /// by value before writing. An empty `row_ids` is a no-op and does not
    /// allocate a slot.
    pub fn insert_many(&mut self, value: V, row_ids: &[u128]) {
        if row_ids.is_empty() {
            return;
        }
        let slot_id = self.slot_for_insert(value);
        for (high, lows) in group_rows(row_ids) {
            self.modify_container(slot_id, high, true, |c| lows.iter().fold(false, |ch, &low| c.insert(low) | ch));
        }
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
        let (high, low) = decompose(row_id);
        let changed = self.modify_container(slot_id, high, false, |c| c.remove(low));
        if changed && self.slot_is_empty(slot_id) {
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
    /// and changes **every** bucket that actually contained `row_id` (so a
    /// multi-valued row is removed from all of them). Buckets that did not
    /// contain it are left untouched — this is load-bearing for high-cardinality
    /// fields: a changed bucket joins the overlay (memory charged to the
    /// budget) and gets a new directory at the next spill, so touching *every*
    /// bucket would cost O(distinct) of both. For a scalar field the common
    /// case touches a single bucket.
    ///
    /// The scan still *probes* each bucket for membership — `O(distinct
    /// values)`, each probe one directory lookup and one container decode.
    /// **When the caller knows the row's old value** (the document layer always
    /// does, from the prior document), prefer the `O(1)` targeted path —
    /// [`remove(old, row)`](Self::remove), or `DynFieldIndex::update` /
    /// `DynFieldIndex::remove` — which touches only the affected bucket. This
    /// full scan is the fallback for when the old value is unknown. A persistent
    /// `row → slot` reverse index is **deliberately not kept** (it would double
    /// the per-write index mutations and need its own storage for a structure
    /// already rebuilt from the WAL on recovery); supplying the old
    /// value gives the same `O(1)` without that cost.
    pub fn remove_all_for_row(&mut self, row_id: u128) -> Vec<u128> {
        self.remove_all_for_rows(std::slice::from_ref(&row_id))
    }

    /// Clear every row in `row_ids` from every bucket it occupies.
    ///
    /// One membership probe and at most one change per container **per bucket
    /// for the whole batch**, rather than per row: clearing N rows one at a
    /// time probes every bucket N times.
    ///
    /// Used by WAL replay, which clears its whole affected row set before
    /// re-inserting. An empty `row_ids` is a no-op.
    pub fn remove_all_for_rows(&mut self, row_ids: &[u128]) -> Vec<u128> {
        if row_ids.is_empty() {
            return Vec::new();
        }
        let groups = group_rows(row_ids);
        let slots: Vec<(V, u128)> = self.ordering.iter().map(|(v, &id)| (v.clone(), id)).collect();

        let mut removed_slots: Vec<u128> = Vec::new();
        for (value, slot_id) in slots {
            // Probe first, so a bucket none of these rows is in is not added
            // to the overlay. A stored bucket is probed through its directory,
            // decoding only the rows' containers.
            if !self.slot_contains_any(slot_id, &groups) {
                continue;
            }
            let mut changed = false;
            for (&high, lows) in &groups {
                changed |= self.modify_container(slot_id, high, false, |c| lows.iter().fold(false, |ch, &low| c.remove(low) | ch));
            }
            if changed && self.slot_is_empty(slot_id) {
                self.ordering.remove(&value);
                self.emptied.insert(value, slot_id);
                removed_slots.push(slot_id);
            }
        }
        removed_slots
    }

    /// Whether `slot_id`'s bitmap holds any of the grouped rows, decoding only
    /// the containers they fall in.
    fn slot_contains_any(&self, slot_id: u128, groups: &BTreeMap<u128, Vec<u16>>) -> bool {
        let ch = self.overlay.get(&slot_id);
        let stored;
        let base: &[DirEntry] = match ch.and_then(|c| c.base.as_deref()) {
            Some(b) => b,
            None => {
                stored = self.stored_dir(slot_id);
                &stored
            }
        };
        groups.iter().any(|(&high, lows)| match ch.and_then(|c| c.containers.get(&high)) {
            Some(c) => c.as_ref().is_some_and(|c| lows.iter().any(|&l| c.contains(l))),
            None => storage::find_entry(base, high)
                .and_then(|e| self.stored_container(slot_id, e))
                .is_some_and(|c| lows.iter().any(|&l| c.contains(l))),
        })
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
    /// Each bitmap is assembled on demand; the iterator is lazy over the
    /// `BTreeMap` but eagerly loads each bitmap as it is visited.
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
    /// appended vs. what survives compaction. The gap is reclaimable dead space:
    /// each spill appends a fresh copy of every container that changed, plus a
    /// new directory for its slot, and leaves the old copies behind. See
    /// `BlobStore::logical_bytes` / `BlobStore::live_bytes`.
    pub fn bitmap_blob_bytes(&self) -> (u64, u64) {
        (self.bitmaps.logical_bytes(), self.bitmaps.live_bytes())
    }

    /// Reclaimable dead bytes in the bitmap value region — **O(1)** (see
    /// `BlobStore::dead_bytes`). Read on the write path by the backpressure
    /// valve, which cannot afford a waste-ratio scan per write.
    pub fn bitmap_dead_bytes(&self) -> u64 {
        self.bitmaps.dead_bytes()
    }

    /// Compact the bitmap value region, reclaiming the dead space left by
    /// spills (old copies of changed containers and directories). See
    /// `BlobStore::compact`.
    pub fn compact_bitmaps(&mut self) -> std::io::Result<u64> {
        let reclaimed = self.bitmaps.compact()?;
        // Every stored offset moved: cached directories are stale.
        let slots: Vec<u128> = self.overlay.keys().copied().collect();
        for slot_id in slots {
            self.forget_base(slot_id);
        }
        Ok(reclaimed)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// The current bitmap for `slot_id`: the stored containers, with the
    /// overlay's changed containers in place of theirs.
    fn load_bitmap(&self, slot_id: u128) -> RoaringBitmap {
        let ch = self.overlay.get(&slot_id);
        let stored;
        let base: &[DirEntry] = match ch.and_then(|c| c.base.as_deref()) {
            Some(b) => b,
            None => {
                stored = self.stored_dir(slot_id);
                &stored
            }
        };
        let mut bm = RoaringBitmap::new();
        for &e in base {
            if ch.is_some_and(|c| c.containers.contains_key(&e.key)) {
                continue;
            }
            if let Some(c) = self.stored_container(slot_id, e) {
                bm.store.insert_owned(e.key, c);
            }
        }
        if let Some(ch) = ch {
            for (&key, c) in &ch.containers {
                if let Some(c) = c {
                    bm.store.insert_owned(key, c.clone());
                }
            }
        }
        bm
    }

    /// `slot_id`'s stored directory. A *missing* slot is a normal empty
    /// bitmap; a directory that does not decode is served as empty and latches
    /// the corruption flag.
    fn stored_dir(&self, slot_id: u128) -> Vec<DirEntry> {
        match self.bitmaps.get_ref(slot_id) {
            None => Vec::new(),
            Some(bytes) => storage::decode_dir(bytes).unwrap_or_else(|e| {
                self.corrupted.store(true, Ordering::Relaxed);
                log::error!(
                    "FieldIndex: bitmap directory failed to decode; serving empty and flagging the index corrupt (rebuild this field from the WAL) (slot={slot_id}, error={e})"
                );
                Vec::new()
            }),
        }
    }

    /// Decode the stored container `entry` points at. A blob that is out of
    /// range or does not decode is served as absent and latches the corruption
    /// flag.
    fn stored_container(&self, slot_id: u128, entry: DirEntry) -> Option<Container> {
        let decoded = match self.bitmaps.value_slice(entry.offset, entry.len) {
            Some(bytes) => storage::decode_container(bytes).map_err(|e| e.to_string()),
            None => Err("container blob runs past the value region".to_string()),
        };
        decoded
            .map_err(|e| {
                self.corrupted.store(true, Ordering::Relaxed);
                log::error!(
                    "FieldIndex: bitmap container failed to load; serving it empty and flagging the index corrupt (rebuild this field from the WAL) (slot={slot_id}, key={}, error={e})",
                    entry.key
                );
            })
            .ok()
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
        // A blob that is not a directory fails its magic/checksum check, so
        // load_bitmap falls back to an empty bitmap instead of propagating an
        // error.
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
        // A valid directory pointing at a container blob whose rkyv payload is
        // garbage: checked access surfaces an error that load_bitmap swallows
        // into an empty container, and the corruption flag latches.
        let mut idx = FieldIndex::<i64>::new();
        idx.insert(7, 1);
        idx.spill(); // the corruption below is in the store, not the overlay
        let slot = idx.slot_id_for(&7).unwrap();

        let (offset, len) = idx.bitmaps.append_value(&[0xFFu8; 32]);
        let dir = storage::encode_dir(&[DirEntry { key: 0, offset, len }]);
        idx.bitmaps.upsert(slot, &dir);

        assert!(idx.load_bitmap(slot).is_empty());
        assert!(idx.corruption_detected(), "an invalid rkyv payload must latch the corruption flag");
    }

    /// The point of the directory layout: one write to a large bitmap appends
    /// one container and one directory at the next spill, not the bitmap.
    #[test]
    fn a_spill_appends_only_the_changed_container() {
        let mut idx = FieldIndex::<bool>::new();
        // 64 containers' worth of rows under `true`.
        idx.insert_many(true, &(0..64u128 * 65_536).step_by(3).collect::<Vec<_>>());
        idx.spill();
        let (before, _) = idx.bitmap_blob_bytes();
        let whole: u64 = idx.load_bitmap(idx.slot_id_for(&true).unwrap()).num_containers() as u64;
        assert_eq!(whole, 64);

        idx.insert(true, 5 * 65_536 + 1);
        idx.spill();
        let (after, live) = idx.bitmap_blob_bytes();
        let appended = after - before;
        // One bitset container (8 KiB + framing) plus a 64-entry directory.
        assert!(
            appended < 8_192 + 512 + 64 * 28 + 64,
            "appended {appended} bytes for one changed container"
        );
        assert!(live > 64 * 8_192, "the bitmap itself is ~{live} bytes");
        assert!(idx.evaluate(&Predicate::Eq(true)).contains(5 * 65_536 + 1));
    }

    /// A bitmap written between a spill's stage and its commit keeps only the
    /// containers changed since the stage; the ones the commit wrote out leave
    /// the overlay (and the budget), and later spills do not rewrite them.
    #[test]
    fn a_commit_sheds_the_containers_it_wrote_even_when_the_slot_changed_since() {
        let mut idx = FieldIndex::<bool>::new();
        // Touch 8 containers of `true`.
        idx.insert_many(true, &(0..8u128).map(|c| c * 65_536 + 1).collect::<Vec<_>>());
        let stage = idx.stage();
        idx.insert(true, 8 * 65_536 + 1); // a 9th container, after the stage
        idx.commit(stage);
        let slot = idx.slot_id_for(&true).unwrap();
        let held: Vec<u128> = idx.overlay[&slot].containers.keys().copied().collect();
        assert_eq!(held, vec![8], "only the container changed after the stage stays buffered");
        let one = idx.overlay_bytes();
        assert!(one < 1_024, "overlay holds {one} bytes for one small container");

        let (before, _) = idx.bitmap_blob_bytes();
        idx.spill();
        let (after, _) = idx.bitmap_blob_bytes();
        assert!(after - before < 512, "the next spill rewrites one container, appended {}", after - before);
        let got: Vec<u128> = idx.evaluate(&Predicate::Eq(true)).iter().collect();
        assert_eq!(got, (0..9u128).map(|c| c * 65_536 + 1).collect::<Vec<_>>());
        assert_eq!(idx.overlay_bytes(), 0);
    }

    /// Reads merge the overlay over the stored containers, and emptying every
    /// container (some stored, some only in the overlay) frees the slot.
    #[test]
    fn overlay_and_store_merge_and_empty_together() {
        let mut idx = FieldIndex::<i64>::new();
        let rows: Vec<u128> = vec![1, 70_000, 140_000, 5 << 40];
        idx.insert_many(3, &rows[..2]);
        idx.spill();
        idx.insert_many(3, &rows[2..]);
        let got: Vec<u128> = idx.evaluate(&Predicate::Eq(3)).iter().collect();
        assert_eq!(got, rows);
        for &r in &rows[..3] {
            idx.remove(3, r);
        }
        assert!(idx.contains_value(&3i64));
        let emptied = idx.remove_all_for_row(5 << 40);
        assert_eq!(emptied.len(), 1);
        assert!(!idx.contains_value(&3i64));
        assert_eq!(idx.spill().len(), 1, "the emptied slot is freed by the spill");
        assert_eq!(idx.bitmaps.count(), 0);
        assert_eq!(idx.overlay_bytes(), 0);
    }
}
