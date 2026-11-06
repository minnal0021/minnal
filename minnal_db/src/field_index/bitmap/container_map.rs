//! The sorted `high key → Container` map behind a
//! [`RoaringBitmap`](crate::field_index::RoaringBitmap).
//!
//! It lives on the heap and changes containers in place, so a bitmap's memory is
//! its real size. It keeps the bitmap's total bit count and heap size current on
//! every change, which makes both O(1) to read.

use std::collections::BTreeMap;

use crate::field_index::bitmap::container::Container;

/// Fixed per-container cost beyond the container's own buffer: the map entry
/// (key + `Container` enum) plus B-tree node overhead.
const ENTRY_OVERHEAD: usize = std::mem::size_of::<(u128, Container)>() + 16;

/// Heap bytes one stored container accounts for.
fn entry_cost(c: &Container) -> usize {
    c.heap_bytes() + ENTRY_OVERHEAD
}

/// A bitmap's containers, by high key. See the module docs.
#[derive(Default)]
pub(crate) struct ContainerMap {
    map: BTreeMap<u128, Container>,
    /// Total bit count across all containers.
    cardinality: u64,
    /// Heap bytes held (see [`entry_cost`]).
    heap_bytes: usize,
}

impl ContainerMap {
    /// An empty map.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Total bit count across all containers.
    pub(crate) fn cardinality(&self) -> usize {
        self.cardinality as usize
    }

    /// Number of containers.
    pub(crate) fn count(&self) -> usize {
        self.map.len()
    }

    /// Heap bytes held: each container's buffer plus a fixed per-entry cost.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.heap_bytes
    }

    /// Insert or replace the container for `key`, copying it.
    ///
    /// Returns `true` if this was a new key.
    pub(crate) fn upsert(&mut self, key: u128, container: &Container) -> bool {
        self.insert_owned(key, container.clone())
    }

    /// Insert or replace the container for `key`, taking ownership.
    ///
    /// Returns `true` if this was a new key.
    pub(crate) fn insert_owned(&mut self, key: u128, container: Container) -> bool {
        self.cardinality += container.cardinality() as u64;
        self.heap_bytes += entry_cost(&container);
        match self.map.insert(key, container) {
            Some(old) => {
                self.cardinality -= old.cardinality() as u64;
                self.heap_bytes -= entry_cost(&old);
                false
            }
            None => true,
        }
    }

    /// Change the container for `key` in place.
    ///
    /// `f` gets the container (a new empty array container if `key` is absent
    /// and `create` is true) and returns whether it changed anything. A changed
    /// container that ends up empty is removed. Returns `f`'s result, or
    /// `false` without calling `f` when `key` is absent and `create` is false.
    pub(crate) fn modify(&mut self, key: u128, create: bool, f: impl FnOnce(&mut Container) -> bool) -> bool {
        // (changed, old card, new card, old cost, new cost, now empty)
        let (changed, old_card, new_card, old_cost, new_cost, empty) = match self.map.get_mut(&key) {
            Some(c) => {
                let (old_card, old_cost) = (c.cardinality() as u64, entry_cost(c));
                let changed = f(c);
                (changed, old_card, c.cardinality() as u64, old_cost, entry_cost(c), c.is_empty())
            }
            None if create => {
                let mut c = Container::new_array();
                let changed = f(&mut c);
                let (card, cost, empty) = (c.cardinality() as u64, entry_cost(&c), c.is_empty());
                if changed && !empty {
                    self.map.insert(key, c);
                }
                (changed, 0, card, 0, if changed && !empty { cost } else { 0 }, empty)
            }
            None => return false,
        };
        self.heap_bytes = self.heap_bytes - old_cost + new_cost;
        if changed {
            self.cardinality = self.cardinality - old_card + new_card;
            if empty && self.map.remove(&key).is_some() {
                self.heap_bytes -= new_cost;
            }
        }
        changed
    }

    /// Remove all containers.
    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }

    /// A copy of the container for `key`, or `None`. Prefer
    /// [`with_container`](Self::with_container) for reads: it borrows.
    pub(crate) fn get(&self, key: u128) -> Option<Container> {
        self.map.get(&key).cloned()
    }

    /// Run `f` on the container for `key`. `None` if `key` is absent.
    pub(crate) fn with_container<R>(&self, key: u128, f: impl FnOnce(&Container) -> R) -> Option<R> {
        self.map.get(&key).map(f)
    }

    /// `(key, cardinality)` for every container, sorted by key.
    pub(crate) fn sorted_key_cards(&self) -> Vec<(u128, u32)> {
        self.map.iter().map(|(&k, c)| (k, c.cardinality() as u32)).collect()
    }

    /// All keys, sorted.
    pub(crate) fn sorted_keys(&self) -> Vec<u128> {
        self.map.keys().copied().collect()
    }

    /// Copies of all `(key, container)` pairs, sorted by key.
    pub(crate) fn sorted_entries(&self) -> Vec<(u128, Container)> {
        self.map.iter().map(|(&k, c)| (k, c.clone())).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_index::bitmap::container::array::ArrayContainer;

    fn make_array(vals: &[u16]) -> Container {
        Container::Array(ArrayContainer::from_sorted(vals.to_vec()))
    }

    #[test]
    fn insert_get_roundtrip() {
        let mut map = ContainerMap::new();
        assert!(map.upsert(42, &make_array(&[1, 2, 3])));
        assert_eq!(map.get(42).unwrap().to_values(), vec![1, 2, 3]);
        assert_eq!(map.count(), 1);
        assert_eq!(map.cardinality(), 3);
    }

    #[test]
    fn update_existing_adjusts_cardinality_and_memory() {
        let mut map = ContainerMap::new();
        assert!(map.upsert(10, &make_array(&[1, 2, 3])));
        assert!(!map.upsert(10, &make_array(&[1, 2, 3, 4, 5])));
        assert_eq!(map.cardinality(), 5);
        assert_eq!(map.count(), 1);
        assert_eq!(map.heap_bytes(), entry_cost(&map.get(10).unwrap()));
    }

    #[test]
    fn sorted_views_are_in_key_order() {
        let mut map = ContainerMap::new();
        map.upsert(300, &make_array(&[1]));
        map.upsert(100, &make_array(&[1, 2]));
        map.upsert(200, &make_array(&[1, 2, 3]));
        assert_eq!(map.sorted_key_cards(), vec![(100, 2), (200, 3), (300, 1)]);
        assert_eq!(map.sorted_keys(), vec![100, 200, 300]);
        let keys: Vec<u128> = map.sorted_entries().into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec![100, 200, 300]);
    }

    #[test]
    fn clear_resets_everything() {
        let mut map = ContainerMap::new();
        map.upsert(1, &make_array(&[1, 2, 3]));
        map.clear();
        assert_eq!(map.count(), 0);
        assert_eq!(map.cardinality(), 0);
        assert_eq!(map.heap_bytes(), 0);
        assert!(map.get(1).is_none());
    }

    /// Changes must be in place: a design that keeps a copy per change would
    /// hold hundreds of MiB here for an 8 KB bitmap.
    #[test]
    fn memory_is_the_bitmaps_size_not_its_history() {
        let mut map = ContainerMap::new();
        for low in 0..=u16::MAX {
            assert!(map.modify(0, true, |c| c.insert(low)));
        }
        assert_eq!(map.cardinality(), 65_536);
        assert_eq!(map.count(), 1);
        let bytes = map.heap_bytes();
        assert!(bytes <= 16 * 1024, "one full container should hold ~8 KB, holds {bytes} bytes");
    }

    #[test]
    fn modify_keeps_counts_and_removes_emptied_containers() {
        let mut map = ContainerMap::new();
        assert!(map.modify(7, true, |c| c.insert(1)));
        assert!(map.modify(7, true, |c| c.insert(2)));
        assert!(!map.modify(7, true, |c| c.insert(2)), "re-inserting is no change");
        assert_eq!(map.cardinality(), 2);
        assert!(!map.modify(8, false, |c| c.insert(1)), "absent key without create is untouched");
        assert!(map.get(8).is_none());
        assert!(map.modify(7, false, |c| c.remove(1)));
        assert!(map.modify(7, false, |c| c.remove(2)));
        assert_eq!(map.cardinality(), 0);
        assert_eq!(map.heap_bytes(), 0, "an emptied container's memory is released");
        assert!(map.get(7).is_none(), "an emptied container is removed");
        assert_eq!(map.with_container(7, |c| c.cardinality()), None);
    }

    /// A no-op `f` on a new key must leave nothing behind, and no accounting.
    #[test]
    fn modify_that_changes_nothing_on_a_new_key_stores_nothing() {
        let mut map = ContainerMap::new();
        assert!(!map.modify(9, true, |_| false));
        assert_eq!(map.count(), 0);
        assert_eq!(map.heap_bytes(), 0);
    }
}
