use crate::simd_support::array_merge;
use rkyv::Archive;

/// Threshold above which an ArrayContainer should convert to a BitsetContainer.
pub const ARRAY_TO_BITSET_THRESHOLD: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ArrayContainer {
    /// Sorted, deduplicated array of u16 values.
    values: Vec<u16>,
}

impl ArrayContainer {
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            values: Vec::with_capacity(cap),
        }
    }

    pub fn from_sorted(values: Vec<u16>) -> Self {
        debug_assert!(values.windows(2).all(|w| w[0] < w[1]));
        Self { values }
    }

    pub fn insert(&mut self, value: u16) -> bool {
        match self.values.binary_search(&value) {
            Ok(_) => false, // already present
            Err(pos) => {
                self.values.insert(pos, value);
                true
            }
        }
    }

    pub fn remove(&mut self, value: u16) -> bool {
        match self.values.binary_search(&value) {
            Ok(pos) => {
                self.values.remove(pos);
                true
            }
            Err(_) => false,
        }
    }

    pub fn contains(&self, value: u16) -> bool {
        self.values.binary_search(&value).is_ok()
    }

    pub fn cardinality(&self) -> usize {
        self.values.len()
    }

    /// Alias for `cardinality()`. Array cardinality is O(1) — no SIMD needed.
    pub fn popcount(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn min(&self) -> Option<u16> {
        self.values.first().copied()
    }

    pub fn max(&self) -> Option<u16> {
        self.values.last().copied()
    }

    pub fn values(&self) -> &[u16] {
        &self.values
    }

    pub fn into_values(self) -> Vec<u16> {
        self.values
    }

    pub fn iter(&self) -> impl Iterator<Item = u16> + '_ {
        self.values.iter().copied()
    }

    /// Count of elements ≤ `value` in this container.
    ///
    /// Uses binary search — O(log n).
    pub fn rank(&self, value: u16) -> usize {
        // partition_point returns the index of the first element > value,
        // which equals the count of elements ≤ value.
        self.values.partition_point(|&v| v <= value)
    }

    /// The `rank`-th element (0-indexed) in sorted order, or `None` if out of bounds.
    pub fn select(&self, rank: usize) -> Option<u16> {
        self.values.get(rank).copied()
    }

    /// Whether this container should be converted to a bitset.
    pub fn should_promote(&self) -> bool {
        self.values.len() >= ARRAY_TO_BITSET_THRESHOLD
    }

    // ── Set operations ──────────────────────────────────────────────

    /// Intersection of two sorted arrays (AND).
    pub fn and(&self, other: &ArrayContainer) -> ArrayContainer {
        ArrayContainer {
            values: array_merge::and_sorted_u16(&self.values, &other.values),
        }
    }

    /// Union of two sorted arrays (OR).
    pub fn or(&self, other: &ArrayContainer) -> ArrayContainer {
        ArrayContainer {
            values: array_merge::or_sorted_u16(&self.values, &other.values),
        }
    }

    /// Difference: elements in self but not in other (AND NOT).
    pub fn and_not(&self, other: &ArrayContainer) -> ArrayContainer {
        ArrayContainer {
            values: array_merge::and_not_sorted_u16(&self.values, &other.values),
        }
    }
}

impl Default for ArrayContainer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_contains() {
        let mut c = ArrayContainer::new();
        assert!(c.insert(10));
        assert!(c.insert(5));
        assert!(c.insert(20));
        assert!(!c.insert(10)); // duplicate
        assert_eq!(c.cardinality(), 3);
        assert!(c.contains(5));
        assert!(c.contains(10));
        assert!(c.contains(20));
        assert!(!c.contains(15));
        // sorted order
        assert_eq!(c.values(), &[5, 10, 20]);
    }

    #[test]
    fn remove() {
        let mut c = ArrayContainer::from_sorted(vec![1, 3, 5, 7]);
        assert!(c.remove(3));
        assert!(!c.remove(3));
        assert_eq!(c.values(), &[1, 5, 7]);
    }

    #[test]
    fn min_max() {
        let c = ArrayContainer::from_sorted(vec![10, 20, 30]);
        assert_eq!(c.min(), Some(10));
        assert_eq!(c.max(), Some(30));
        assert_eq!(ArrayContainer::new().min(), None);
    }

    #[test]
    fn and_operation() {
        let a = ArrayContainer::from_sorted(vec![1, 3, 5, 7, 9]);
        let b = ArrayContainer::from_sorted(vec![2, 3, 5, 8, 9]);
        let result = a.and(&b);
        assert_eq!(result.values(), &[3, 5, 9]);
    }

    #[test]
    fn or_operation() {
        let a = ArrayContainer::from_sorted(vec![1, 3, 5]);
        let b = ArrayContainer::from_sorted(vec![2, 3, 6]);
        let result = a.or(&b);
        assert_eq!(result.values(), &[1, 2, 3, 5, 6]);
    }

    #[test]
    fn and_not_operation() {
        let a = ArrayContainer::from_sorted(vec![1, 3, 5, 7]);
        let b = ArrayContainer::from_sorted(vec![3, 7, 9]);
        let result = a.and_not(&b);
        assert_eq!(result.values(), &[1, 5]);
    }

    #[test]
    fn rank_basic() {
        let c = ArrayContainer::from_sorted(vec![10, 20, 30, 40]);
        assert_eq!(c.rank(5), 0); // below all values
        assert_eq!(c.rank(10), 1); // exactly at first value
        assert_eq!(c.rank(15), 1); // between 10 and 20
        assert_eq!(c.rank(20), 2); // exactly at second value
        assert_eq!(c.rank(40), 4); // at last value
        assert_eq!(c.rank(99), 4); // beyond all values
    }

    #[test]
    fn rank_empty() {
        let c = ArrayContainer::new();
        assert_eq!(c.rank(0), 0);
        assert_eq!(c.rank(u16::MAX), 0);
    }

    #[test]
    fn select_basic() {
        let c = ArrayContainer::from_sorted(vec![10, 20, 30]);
        assert_eq!(c.select(0), Some(10));
        assert_eq!(c.select(1), Some(20));
        assert_eq!(c.select(2), Some(30));
        assert_eq!(c.select(3), None);
    }

    #[test]
    fn rank_select_round_trip() {
        let vals: Vec<u16> = (0..100).map(|i| i * 3).collect();
        let c = ArrayContainer::from_sorted(vals.clone());
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(c.select(i), Some(v));
            assert_eq!(c.rank(v), i + 1);
        }
    }
}
