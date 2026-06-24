use std::cmp::Ordering;
use std::fmt;
use std::path::Path;

use crate::container::Container;
use crate::container::array::ARRAY_TO_BITSET_THRESHOLD;
use crate::container::array::ArrayContainer;
use crate::container::bitset::BitsetContainer;
use crate::container_store::ContainerStore;

/// Decompose a u128 key into a high key (upper 112 bits) and low value (lower 16 bits).
#[inline]
pub fn decompose(key: u128) -> (u128, u16) {
    let high = key >> 16;
    let low = (key & 0xFFFF) as u16;
    (high, low)
}

/// Recompose a high key and low value back into the original u128.
#[inline]
pub fn compose(high: u128, low: u16) -> u128 {
    (high << 16) | (low as u128)
}

/// A Roaring Bitmap supporting u128 keys.
///
/// Backed by a two-file memory-mapped store: a key file (open-addressing
/// hash table of `u128 → offset`) and a value file (append-only serialised
/// [`Container`] blobs). Both files can be anonymous (transient bitmaps
/// produced by set operations) or file-backed (persistent field-index bitmaps).
///
/// Anonymous stores are created by [`RoaringBitmap::new`] and all constructors
/// that do not accept a path. File-backed stores are created by
/// [`RoaringBitmap::create`] / [`RoaringBitmap::open`].
///
/// `cardinality()` is O(1) — the total bit count is maintained in the store
/// header and updated on every insert/remove/clear.
pub struct RoaringBitmap {
    pub(crate) store: ContainerStore,
}

impl RoaringBitmap {
    /// Create a new empty bitmap backed by an anonymous (transient) mmap.
    pub fn new() -> Self {
        Self {
            store: ContainerStore::new_anon(),
        }
    }

    /// Create a new empty bitmap backed by files under `dir`.
    pub fn create(dir: &Path) -> std::io::Result<Self> {
        Ok(Self {
            store: ContainerStore::create(dir)?,
        })
    }

    /// Open an existing file-backed bitmap from `dir`.
    pub fn open(dir: &Path) -> std::io::Result<Self> {
        Ok(Self {
            store: ContainerStore::open(dir)?,
        })
    }

    /// Flush the underlying mmaps to disk (no-op for anonymous bitmaps).
    pub fn flush(&self) -> std::io::Result<()> {
        self.store.flush()
    }

    // ── Mutation ─────────────────────────────────────────────────────

    /// Insert a value. Returns true if the value was not already present.
    pub fn insert(&mut self, value: u128) -> bool {
        let (high, low) = decompose(value);
        let mut container = self.store.get(high).unwrap_or_else(Container::new_array);
        let inserted = container.insert(low);
        if inserted {
            self.store.upsert(high, &container);
        }
        inserted
    }

    /// Remove a value. Returns true if the value was present.
    pub fn remove(&mut self, value: u128) -> bool {
        let (high, low) = decompose(value);
        let mut container = match self.store.get(high) {
            Some(c) => c,
            None => return false,
        };
        let removed = container.remove(low);
        if removed {
            if container.is_empty() {
                self.store.remove_key(high);
            } else {
                self.store.upsert(high, &container);
            }
        }
        removed
    }

    /// Remove all values.
    pub fn clear(&mut self) {
        self.store.clear();
    }

    // ── Query ────────────────────────────────────────────────────────

    pub fn contains(&self, value: u128) -> bool {
        let (high, low) = decompose(value);
        self.store.get(high).is_some_and(|c| c.contains(low))
    }

    /// O(1) — returns the cached count.
    pub fn cardinality(&self) -> usize {
        self.store.cardinality()
    }

    pub fn len(&self) -> usize {
        self.store.cardinality()
    }

    pub fn is_empty(&self) -> bool {
        self.store.cardinality() == 0
    }

    pub fn min(&self) -> Option<u128> {
        let keys = self.store.sorted_keys();
        keys.first()
            .and_then(|&high| self.store.get(high).and_then(|c| c.min().map(|low| compose(high, low))))
    }

    pub fn max(&self) -> Option<u128> {
        let keys = self.store.sorted_keys();
        keys.last()
            .and_then(|&high| self.store.get(high).and_then(|c| c.max().map(|low| compose(high, low))))
    }

    /// Number of containers (useful for diagnostics).
    pub fn num_containers(&self) -> usize {
        self.store.count()
    }

    // ── Set operations (return new bitmap) ──────────────────────────

    /// AND: intersection of two bitmaps.
    pub fn and(&self, other: &Self) -> Self {
        let a = self.store.sorted_entries();
        let b = other.store.sorted_entries();
        let mut result = Self::new();
        let mut ai = a.iter().peekable();
        let mut bi = b.iter().peekable();
        loop {
            let ak = ai.peek().map(|(k, _)| *k);
            let bk = bi.peek().map(|(k, _)| *k);
            match (ak, bk) {
                (Some(ak), Some(bk)) => match ak.cmp(&bk) {
                    Ordering::Less => {
                        ai.next();
                    }
                    Ordering::Greater => {
                        bi.next();
                    }
                    Ordering::Equal => {
                        let (_, av) = ai.next().unwrap();
                        let (_, bv) = bi.next().unwrap();
                        let c = av.and(bv);
                        if !c.is_empty() {
                            result.store.upsert(ak, &c);
                        }
                    }
                },
                _ => break,
            }
        }
        result
    }

    /// OR: union of two bitmaps.
    pub fn or(&self, other: &Self) -> Self {
        let a = self.store.sorted_entries();
        let b = other.store.sorted_entries();
        let mut result = Self::new();
        let mut ai = a.iter().peekable();
        let mut bi = b.iter().peekable();
        loop {
            let ak = ai.peek().map(|(k, _)| *k);
            let bk = bi.peek().map(|(k, _)| *k);
            match (ak, bk) {
                (Some(ak), Some(bk)) => match ak.cmp(&bk) {
                    Ordering::Less => {
                        let (_, c) = ai.next().unwrap();
                        result.store.upsert(ak, c);
                    }
                    Ordering::Greater => {
                        let (_, c) = bi.next().unwrap();
                        result.store.upsert(bk, c);
                    }
                    Ordering::Equal => {
                        let (_, av) = ai.next().unwrap();
                        let (_, bv) = bi.next().unwrap();
                        let c = av.or(bv);
                        result.store.upsert(ak, &c);
                    }
                },
                (Some(_), None) => {
                    for (k, c) in ai {
                        result.store.upsert(*k, c);
                    }
                    break;
                }
                (None, Some(_)) => {
                    for (k, c) in bi {
                        result.store.upsert(*k, c);
                    }
                    break;
                }
                (None, None) => break,
            }
        }
        result
    }

    /// AND NOT: elements in self but not in other.
    pub fn and_not(&self, other: &Self) -> Self {
        let a = self.store.sorted_entries();
        let b = other.store.sorted_entries();
        let mut result = Self::new();
        let mut bi = b.iter().peekable();
        for (ak, av) in &a {
            while bi.peek().is_some_and(|(bk, _)| bk < ak) {
                bi.next();
            }
            let c = match bi.peek() {
                Some((bk, _)) if bk == ak => {
                    let (_, bv) = bi.next().unwrap();
                    av.and_not(bv)
                }
                _ => av.clone(),
            };
            if !c.is_empty() {
                result.store.upsert(*ak, &c);
            }
        }
        result
    }

    // ── In-place set operations ─────────────────────────────────────

    pub fn and_inplace(&mut self, other: &Self) {
        *self = self.and(other);
    }

    pub fn or_inplace(&mut self, other: &Self) {
        *self = self.or(other);
    }

    pub fn and_not_inplace(&mut self, other: &Self) {
        *self = self.and_not(other);
    }

    // ── Rank / select ───────────────────────────────────────────────

    /// Count of elements ≤ `value` in the bitmap.
    pub fn rank(&self, value: u128) -> usize {
        let (high, low) = decompose(value);
        let key_cards = self.store.sorted_key_cards();
        let mut prefix = 0usize;
        for (k, card) in &key_cards {
            match k.cmp(&high) {
                Ordering::Less => prefix += *card as usize,
                Ordering::Equal => {
                    if let Some(c) = self.store.get(*k) {
                        prefix += c.rank(low);
                    }
                    break;
                }
                Ordering::Greater => break,
            }
        }
        prefix
    }

    /// The `rank`-th element (0-indexed) in ascending order, or `None` if out of bounds.
    pub fn select(&self, mut rank: usize) -> Option<u128> {
        let key_cards = self.store.sorted_key_cards();
        for (high, card) in key_cards {
            let card = card as usize;
            if rank < card {
                return self.store.get(high).and_then(|c| c.select(rank)).map(|low| compose(high, low));
            }
            rank -= card;
        }
        None
    }

    // ── Flip ────────────────────────────────────────────────────────

    /// Complement bits in the half-open range [`lo`, `hi`) in place.
    pub fn flip(&mut self, lo: u128, hi: u128) {
        if lo >= hi {
            return;
        }
        let (lo_high, lo_low) = decompose(lo);
        let (hi_high, hi_low) = decompose(hi - 1);

        let keys: Vec<u128> = self.store.sorted_keys().into_iter().filter(|&k| k >= lo_high && k <= hi_high).collect();

        for key in keys {
            if let Some(mut container) = self.store.get(key) {
                let range_lo = if key == lo_high { lo_low } else { 0 };
                let range_hi = if key == hi_high { hi_low } else { u16::MAX };
                container.flip_range(range_lo, range_hi);
                if container.is_empty() {
                    self.store.remove_key(key);
                } else {
                    self.store.upsert(key, &container);
                }
            }
        }
    }

    // ── Range-scoped set operations ─────────────────────────────────

    /// Intersection of `self` and `other`, restricted to [`lo`, `hi`).
    pub fn range_and(&self, other: &Self, lo: u128, hi: u128) -> Self {
        if lo >= hi {
            return Self::new();
        }
        let (lo_high, lo_low) = decompose(lo);
        let (hi_high, hi_low) = decompose(hi - 1);

        let mut result = Self::new();
        let self_keys = self.store.sorted_keys();
        for &key in self_keys.iter().filter(|&&k| k >= lo_high && k <= hi_high) {
            let other_c = match other.store.get(key) {
                Some(c) => c,
                None => continue,
            };
            let self_c = match self.store.get(key) {
                Some(c) => c,
                None => continue,
            };
            let lo_clip = if key == lo_high { lo_low } else { 0 };
            let hi_clip = if key == hi_high { hi_low } else { u16::MAX };
            let c = self_c.and(&other_c).clip_to_range(lo_clip, hi_clip);
            if !c.is_empty() {
                result.store.upsert(key, &c);
            }
        }
        result
    }

    /// Union of `self` and `other`, restricted to [`lo`, `hi`).
    pub fn range_or(&self, other: &Self, lo: u128, hi: u128) -> Self {
        if lo >= hi {
            return Self::new();
        }
        let (lo_high, lo_low) = decompose(lo);
        let (hi_high, hi_low) = decompose(hi - 1);

        let mut result = Self::new();
        let self_entries: Vec<(u128, Container)> = self
            .store
            .sorted_entries()
            .into_iter()
            .filter(|(k, _)| *k >= lo_high && *k <= hi_high)
            .collect();
        let other_entries: Vec<(u128, Container)> = other
            .store
            .sorted_entries()
            .into_iter()
            .filter(|(k, _)| *k >= lo_high && *k <= hi_high)
            .collect();

        let mut ai = self_entries.iter().peekable();
        let mut bi = other_entries.iter().peekable();

        loop {
            let ak = ai.peek().map(|(k, _)| *k);
            let bk = bi.peek().map(|(k, _)| *k);

            let (key, c) = match (ak, bk) {
                (None, None) => break,
                (Some(k), None) => (k, ai.next().unwrap().1.clone()),
                (None, Some(k)) => (k, bi.next().unwrap().1.clone()),
                (Some(ak), Some(bk)) => match ak.cmp(&bk) {
                    Ordering::Less => (ak, ai.next().unwrap().1.clone()),
                    Ordering::Greater => (bk, bi.next().unwrap().1.clone()),
                    Ordering::Equal => {
                        let (_, ac) = ai.next().unwrap();
                        let (_, bc) = bi.next().unwrap();
                        (ak, ac.or(bc))
                    }
                },
            };
            let lo_clip = if key == lo_high { lo_low } else { 0 };
            let hi_clip = if key == hi_high { hi_low } else { u16::MAX };
            let c = c.clip_to_range(lo_clip, hi_clip);
            if !c.is_empty() {
                result.store.upsert(key, &c);
            }
        }
        result
    }

    // ── Iteration ───────────────────────────────────────────────────

    pub fn iter(&self) -> impl Iterator<Item = u128> {
        self.store.sorted_entries().into_iter().flat_map(|(high, container)| {
            let lows: Vec<u16> = container.iter().collect();
            lows.into_iter().map(move |low| compose(high, low))
        })
    }

    // ── Optimization ────────────────────────────────────────────────

    /// Re-evaluate container types across the entire bitmap.
    pub fn optimize(&mut self) {
        let keys = self.store.sorted_keys();
        for key in keys {
            if let Some(mut c) = self.store.get(key) {
                c.optimize();
                self.store.upsert(key, &c);
            }
        }
    }

    // ── Bulk load ────────────────────────────────────────────────────

    /// Build a `RoaringBitmap` from a **sorted, deduplicated** iterator.
    pub fn from_sorted_iter<I: IntoIterator<Item = u128>>(iter: I) -> Self {
        let mut bm = Self::new();
        let mut current_high: Option<u128> = None;
        let mut lows: Vec<u16> = Vec::new();

        for key in iter {
            let (high, low) = decompose(key);
            if current_high == Some(high) {
                lows.push(low);
            } else {
                if let Some(h) = current_high {
                    let c = Self::build_container(std::mem::take(&mut lows));
                    bm.store.upsert(h, &c);
                }
                current_high = Some(high);
                lows.push(low);
            }
        }
        if let Some(h) = current_high {
            let c = Self::build_container(lows);
            bm.store.upsert(h, &c);
        }
        bm
    }

    /// Build a `RoaringBitmap` from an unsorted or deduplicated iterator.
    pub fn from_unsorted_iter<I: IntoIterator<Item = u128>>(iter: I) -> Self {
        let mut keys: Vec<u128> = iter.into_iter().collect();
        keys.sort_unstable();
        keys.dedup();
        Self::from_sorted_iter(keys)
    }

    fn build_container(lows: Vec<u16>) -> Container {
        if lows.len() >= ARRAY_TO_BITSET_THRESHOLD {
            Container::Bitset(BitsetContainer::from_sorted_values(&lows))
        } else {
            Container::Array(ArrayContainer::from_sorted(lows))
        }
    }
}

impl Default for RoaringBitmap {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for RoaringBitmap {
    fn clone(&self) -> Self {
        let mut bm = Self::new();
        for (key, container) in self.store.sorted_entries() {
            bm.store.upsert(key, &container);
        }
        bm
    }
}

impl fmt::Debug for RoaringBitmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RoaringBitmap")
            .field("cardinality", &self.cardinality())
            .field("num_containers", &self.num_containers())
            .finish()
    }
}

impl PartialEq for RoaringBitmap {
    fn eq(&self, other: &Self) -> bool {
        if self.cardinality() != other.cardinality() {
            return false;
        }
        let a = self.store.sorted_key_cards();
        let b = other.store.sorted_key_cards();
        if a.len() != b.len() {
            return false;
        }
        for ((ka, _), (kb, _)) in a.iter().zip(b.iter()) {
            if ka != kb {
                return false;
            }
        }
        for (key, _) in &a {
            let ca = self.store.get(*key);
            let cb = other.store.get(*key);
            if ca != cb {
                return false;
            }
        }
        true
    }
}

impl Eq for RoaringBitmap {}

impl FromIterator<u128> for RoaringBitmap {
    fn from_iter<I: IntoIterator<Item = u128>>(iter: I) -> Self {
        let mut bm = Self::new();
        for v in iter {
            bm.insert(v);
        }
        bm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_contains_remove() {
        let mut bm = RoaringBitmap::new();
        assert!(bm.insert(42));
        assert!(bm.insert(u128::MAX));
        assert!(bm.insert(0));
        assert!(!bm.insert(42)); // duplicate
        assert_eq!(bm.cardinality(), 3);
        assert!(bm.contains(42));
        assert!(bm.contains(u128::MAX));
        assert!(bm.contains(0));
        assert!(!bm.contains(1));

        assert!(bm.remove(42));
        assert!(!bm.remove(42));
        assert_eq!(bm.cardinality(), 2);
    }

    #[test]
    fn cardinality_cached_o1() {
        let mut bm = RoaringBitmap::new();
        for i in 0..500u128 {
            bm.insert(i);
        }
        assert_eq!(bm.cardinality(), 500);
        for i in 0..250u128 {
            bm.remove(i);
        }
        assert_eq!(bm.cardinality(), 250);
        bm.clear();
        assert_eq!(bm.cardinality(), 0);
    }

    #[test]
    fn set_ops_cardinality_correct() {
        let a: RoaringBitmap = (0..100u128).collect();
        let b: RoaringBitmap = (50..150u128).collect();
        assert_eq!(a.and(&b).cardinality(), 50);
        assert_eq!(a.or(&b).cardinality(), 150);
        assert_eq!(a.and_not(&b).cardinality(), 50);
    }

    #[test]
    fn min_max() {
        let mut bm = RoaringBitmap::new();
        assert_eq!(bm.min(), None);
        bm.insert(100);
        bm.insert(50);
        bm.insert(200);
        assert_eq!(bm.min(), Some(50));
        assert_eq!(bm.max(), Some(200));
    }

    #[test]
    fn key_decomposition() {
        let mut bm = RoaringBitmap::new();
        bm.insert(0x1_0000);
        bm.insert(0x1_0001);
        bm.insert(0x1_FFFF);
        assert_eq!(bm.num_containers(), 1);
        assert_eq!(bm.cardinality(), 3);
        bm.insert(0x2_0000);
        assert_eq!(bm.num_containers(), 2);
    }

    #[test]
    fn large_u128_keys() {
        let mut bm = RoaringBitmap::new();
        let base: u128 = 0xDEAD_BEEF_CAFE_BABE_1234_5678_0000_0000;
        for i in 0..1000u128 {
            bm.insert(base + i);
        }
        assert_eq!(bm.cardinality(), 1000);
        for i in 0..1000u128 {
            assert!(bm.contains(base + i));
        }
        assert!(!bm.contains(base + 1000));
    }

    #[test]
    fn out_of_order_inserts() {
        let mut bm = RoaringBitmap::new();
        for i in (0u128..10).rev() {
            bm.insert(i << 16);
        }
        assert_eq!(bm.num_containers(), 10);
        assert_eq!(bm.cardinality(), 10);
        let keys: Vec<u128> = bm.iter().collect();
        assert!(keys.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn and_operation() {
        let a: RoaringBitmap = [1u128, 3, 5, 7, 100_000].into_iter().collect();
        let b: RoaringBitmap = [2u128, 3, 5, 8, 100_000].into_iter().collect();
        let result = a.and(&b);
        let values: Vec<u128> = result.iter().collect();
        assert_eq!(values, vec![3, 5, 100_000]);
    }

    #[test]
    fn or_operation() {
        let a: RoaringBitmap = [1u128, 3, 5].into_iter().collect();
        let b: RoaringBitmap = [2u128, 3, 6].into_iter().collect();
        let result = a.or(&b);
        let values: Vec<u128> = result.iter().collect();
        assert_eq!(values, vec![1, 2, 3, 5, 6]);
    }

    #[test]
    fn and_not_operation() {
        let a: RoaringBitmap = [1u128, 3, 5, 7].into_iter().collect();
        let b: RoaringBitmap = [3u128, 7, 9].into_iter().collect();
        let result = a.and_not(&b);
        let values: Vec<u128> = result.iter().collect();
        assert_eq!(values, vec![1, 5]);
    }

    #[test]
    fn cross_container_operations() {
        let mut a = RoaringBitmap::new();
        let mut b = RoaringBitmap::new();
        a.insert(1);
        a.insert(2);
        b.insert(2);
        b.insert(3);
        a.insert(0x10005);
        b.insert(0x10005);
        b.insert(0x10006);

        let and_result: Vec<u128> = a.and(&b).iter().collect();
        assert_eq!(and_result, vec![2, 0x10005]);

        let or_result: Vec<u128> = a.or(&b).iter().collect();
        assert_eq!(or_result, vec![1, 2, 3, 0x10005, 0x10006]);

        let andnot_result: Vec<u128> = a.and_not(&b).iter().collect();
        assert_eq!(andnot_result, vec![1]);
    }

    #[test]
    fn iteration_order() {
        let mut bm = RoaringBitmap::new();
        bm.insert(100);
        bm.insert(1);
        bm.insert(50);
        bm.insert(0x2_0000);
        let values: Vec<u128> = bm.iter().collect();
        assert_eq!(values, vec![1, 50, 100, 0x2_0000]);
    }

    #[test]
    fn clear() {
        let mut bm: RoaringBitmap = (0..100u128).collect();
        assert_eq!(bm.cardinality(), 100);
        bm.clear();
        assert!(bm.is_empty());
        assert_eq!(bm.cardinality(), 0);
    }

    #[test]
    fn from_sorted_iter_matches_insert() {
        let keys: Vec<u128> = (0u128..1000).collect();
        let bulk = RoaringBitmap::from_sorted_iter(keys.iter().copied());
        let incremental: RoaringBitmap = keys.iter().copied().collect();
        assert_eq!(bulk, incremental);
        assert_eq!(bulk.cardinality(), 1000);
    }

    #[test]
    fn from_unsorted_iter_matches_insert() {
        let keys: Vec<u128> = (0u128..500).chain(0u128..500).rev().collect();
        let bulk = RoaringBitmap::from_unsorted_iter(keys.iter().copied());
        let incremental: RoaringBitmap = (0u128..500).collect();
        assert_eq!(bulk, incremental);
        assert_eq!(bulk.cardinality(), 500);
    }

    #[test]
    fn from_sorted_iter_multi_container() {
        let keys: Vec<u128> = (0u128..10).flat_map(|bucket| (0u128..100).map(move |i| (bucket << 16) | i)).collect();
        let bulk = RoaringBitmap::from_sorted_iter(keys.iter().copied());
        assert_eq!(bulk.num_containers(), 10);
        assert_eq!(bulk.cardinality(), 1000);
        for &k in &keys {
            assert!(bulk.contains(k));
        }
    }

    #[test]
    fn rank_single_container() {
        let bm: RoaringBitmap = [10u128, 20, 30, 40].into_iter().collect();
        assert_eq!(bm.rank(5), 0);
        assert_eq!(bm.rank(10), 1);
        assert_eq!(bm.rank(15), 1);
        assert_eq!(bm.rank(40), 4);
        assert_eq!(bm.rank(99), 4);
    }

    #[test]
    fn rank_cross_containers() {
        let bm: RoaringBitmap = [1u128, 2, 0x1_0000, 0x1_0001].into_iter().collect();
        assert_eq!(bm.rank(0), 0);
        assert_eq!(bm.rank(1), 1);
        assert_eq!(bm.rank(2), 2);
        assert_eq!(bm.rank(0x1_0000), 3);
        assert_eq!(bm.rank(0x1_0001), 4);
        assert_eq!(bm.rank(0x2_0000), 4);
    }

    #[test]
    fn select_single_container() {
        let bm: RoaringBitmap = [5u128, 10, 15].into_iter().collect();
        assert_eq!(bm.select(0), Some(5));
        assert_eq!(bm.select(1), Some(10));
        assert_eq!(bm.select(2), Some(15));
        assert_eq!(bm.select(3), None);
    }

    #[test]
    fn select_cross_containers() {
        let bm: RoaringBitmap = [1u128, 2, 0x1_0000, 0x1_0001].into_iter().collect();
        assert_eq!(bm.select(0), Some(1));
        assert_eq!(bm.select(1), Some(2));
        assert_eq!(bm.select(2), Some(0x1_0000));
        assert_eq!(bm.select(3), Some(0x1_0001));
        assert_eq!(bm.select(4), None);
    }

    #[test]
    fn rank_select_round_trip() {
        let vals: Vec<u128> = (0u128..500).map(|i| i * 7).collect();
        let bm = RoaringBitmap::from_sorted_iter(vals.iter().copied());
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(bm.select(i), Some(v), "select({i})");
            assert_eq!(bm.rank(v), i + 1, "rank({v})");
        }
    }

    #[test]
    fn flip_within_existing_container() {
        let mut bm: RoaringBitmap = [1u128, 3, 5].into_iter().collect();
        bm.flip(0, 6);
        let vals: Vec<u128> = bm.iter().collect();
        assert_eq!(vals, vec![0, 2, 4]);
    }

    #[test]
    fn flip_no_op_on_empty_range() {
        let mut bm: RoaringBitmap = [1u128, 2, 3].into_iter().collect();
        bm.flip(5, 5);
        assert_eq!(bm.cardinality(), 3);
    }

    #[test]
    fn flip_twice_is_identity() {
        let bm: RoaringBitmap = (0u128..100).collect();
        let mut bm2 = bm.clone();
        bm2.flip(10, 50);
        bm2.flip(10, 50);
        assert_eq!(bm, bm2);
    }

    #[test]
    fn range_and_subset_of_full_and() {
        let a: RoaringBitmap = (0u128..100).collect();
        let b: RoaringBitmap = (50u128..150).collect();
        let full = a.and(&b);
        let ranged = a.range_and(&b, 50, 100);
        assert_eq!(full, ranged);
    }

    #[test]
    fn range_and_narrows_result() {
        let a: RoaringBitmap = (0u128..100).collect();
        let b: RoaringBitmap = (0u128..100).collect();
        let ranged = a.range_and(&b, 20, 40);
        let expected: RoaringBitmap = (20u128..40).collect();
        assert_eq!(ranged, expected);
    }

    #[test]
    fn range_or_narrows_result() {
        let a: RoaringBitmap = (0u128..50).collect();
        let b: RoaringBitmap = (50u128..100).collect();
        let ranged = a.range_or(&b, 25, 75);
        let expected: RoaringBitmap = (25u128..75).collect();
        assert_eq!(ranged, expected);
    }

    #[test]
    fn range_and_cross_containers() {
        let a: RoaringBitmap = [1u128, 0x1_0001, 0x1_0002].into_iter().collect();
        let b: RoaringBitmap = [1u128, 0x1_0001, 0x1_0003].into_iter().collect();
        let result = a.range_and(&b, 0x1_0000, 0x2_0000);
        let vals: Vec<u128> = result.iter().collect();
        assert_eq!(vals, vec![0x1_0001]);
    }

    #[test]
    fn from_sorted_iter_promotes_to_bitset() {
        let keys: Vec<u128> = (0u128..4096).collect();
        let bulk = RoaringBitmap::from_sorted_iter(keys.iter().copied());
        assert_eq!(bulk.cardinality(), 4096);
        assert_eq!(bulk.num_containers(), 1);
        for &k in &keys {
            assert!(bulk.contains(k));
        }
    }

    #[test]
    fn from_sorted_iter_large_u128_keys() {
        let base: u128 = 0xDEAD_BEEF_CAFE_0000_0000_0000_0000_0000;
        let keys: Vec<u128> = (0u128..2000).map(|i| base + i).collect();
        let bulk = RoaringBitmap::from_sorted_iter(keys.iter().copied());
        assert_eq!(bulk.cardinality(), 2000);
        for &k in &keys {
            assert!(bulk.contains(k));
        }
    }

    #[test]
    fn dense_ids_pack_into_far_fewer_containers_than_sparse() {
        // The rationale for RowMap's dense IDs: values sharing a high key (the
        // upper 112 bits) collapse into one container, whereas IDs scattered
        // across distinct high keys (what random hash IDs produce) cost one
        // container each.
        let dense: RoaringBitmap = (0u128..1000).collect();
        let sparse: RoaringBitmap = (0u128..1000).map(|i| i << 16).collect();

        assert_eq!(dense.num_containers(), 1, "1000 dense ids share one high key");
        assert_eq!(sparse.num_containers(), 1000, "each sparse id lands in its own high-key bucket");
        assert_eq!(dense.cardinality(), sparse.cardinality());
    }
}
