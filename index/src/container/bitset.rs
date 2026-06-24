use crate::simd_support::bitwise;
use crate::simd_support::extract::extract_bitset_to_values;
use crate::simd_support::popcount::popcount_u64_slice;
use rkyv::Archive;

/// Number of u64 words in the bitset (65536 bits / 64 = 1024).
pub const BITSET_WORDS: usize = 1024;

/// Below this cardinality, a BitsetContainer should demote to ArrayContainer.
pub const BITSET_TO_ARRAY_THRESHOLD: usize = 4096;

/// 8 KB bit array aligned to a 64-byte cache line.
///
/// The alignment ensures that AVX-512 512-bit (64-byte) loads and stores
/// issued by the SIMD kernels in `simd_support::bitwise` and
/// `simd_support::extract` never straddle a cache-line boundary, avoiding
/// the split-line penalty on Intel Sapphire Rapids / Ice Lake processors.
#[derive(Clone, Archive, rkyv::Serialize, rkyv::Deserialize)]
#[repr(align(64))]
pub struct AlignedBits(pub [u64; BITSET_WORDS]);

#[derive(Clone, Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BitsetContainer {
    bits: Box<AlignedBits>,
    cardinality: usize,
}

impl BitsetContainer {
    pub fn new() -> Self {
        Self {
            bits: Box::new(AlignedBits([0u64; BITSET_WORDS])),
            cardinality: 0,
        }
    }

    /// Create from a sorted slice of u16 values.
    pub fn from_values(values: &[u16]) -> Self {
        let mut c = Self::new();
        for &v in values {
            c.set(v);
        }
        c
    }

    /// Create from a sorted, deduplicated slice without per-element exists checks.
    /// Uses `set_bits_from_array` for batch bit-setting, then recomputes cardinality once.
    /// Faster than `from_values` for bulk loads.
    pub fn from_sorted_values(values: &[u16]) -> Self {
        let mut bits = Box::new(AlignedBits([0u64; BITSET_WORDS]));
        crate::simd_support::run_bitset::set_bits_from_array(&mut bits.0, values);
        let cardinality = crate::simd_support::popcount::popcount_u64_slice(&bits.0);
        Self { bits, cardinality }
    }

    /// Construct directly from a pre-computed aligned bits array and known cardinality.
    /// Used by cross-container SIMD ops in `container::ops`.
    pub fn from_raw_bits(bits: AlignedBits, cardinality: usize) -> Self {
        Self {
            bits: Box::new(bits),
            cardinality,
        }
    }

    fn word_and_bit(value: u16) -> (usize, u64) {
        let word = (value >> 6) as usize; // value / 64
        let bit = 1u64 << (value & 63); // value % 64
        (word, bit)
    }

    fn set(&mut self, value: u16) {
        let (word, bit) = Self::word_and_bit(value);
        if self.bits.0[word] & bit == 0 {
            self.bits.0[word] |= bit;
            self.cardinality += 1;
        }
    }

    pub fn insert(&mut self, value: u16) -> bool {
        let (word, bit) = Self::word_and_bit(value);
        if self.bits.0[word] & bit == 0 {
            self.bits.0[word] |= bit;
            self.cardinality += 1;
            true
        } else {
            false
        }
    }

    pub fn remove(&mut self, value: u16) -> bool {
        let (word, bit) = Self::word_and_bit(value);
        if self.bits.0[word] & bit != 0 {
            self.bits.0[word] &= !bit;
            self.cardinality -= 1;
            true
        } else {
            false
        }
    }

    pub fn contains(&self, value: u16) -> bool {
        let (word, bit) = Self::word_and_bit(value);
        self.bits.0[word] & bit != 0
    }

    pub fn cardinality(&self) -> usize {
        self.cardinality
    }

    /// Alias for `cardinality()`.
    pub fn popcount(&self) -> usize {
        self.cardinality
    }

    pub fn is_empty(&self) -> bool {
        self.cardinality == 0
    }

    /// Recompute cardinality from the raw bits using SIMD (AVX-512 VPOPCNTDQ when available,
    /// scalar fallback otherwise). Called after bulk AND/OR/ANDNOT operations.
    pub fn recompute_cardinality(&mut self) {
        self.cardinality = popcount_u64_slice(&self.bits.0);
    }

    pub fn bits(&self) -> &[u64; BITSET_WORDS] {
        &self.bits.0
    }

    pub fn bits_mut(&mut self) -> &mut [u64; BITSET_WORDS] {
        &mut self.bits.0
    }

    pub fn min(&self) -> Option<u16> {
        for (i, &word) in self.bits.0.iter().enumerate() {
            if word != 0 {
                return Some((i as u16) * 64 + word.trailing_zeros() as u16);
            }
        }
        None
    }

    pub fn max(&self) -> Option<u16> {
        for (i, &word) in self.bits.0.iter().enumerate().rev() {
            if word != 0 {
                return Some((i as u16) * 64 + 63 - word.leading_zeros() as u16);
            }
        }
        None
    }

    /// Whether this container should demote to an ArrayContainer.
    pub fn should_demote(&self) -> bool {
        self.cardinality < BITSET_TO_ARRAY_THRESHOLD
    }

    /// Extract all set bits as a sorted Vec<u16>.
    ///
    /// Dispatches to the AVX-512 + byte lookup table path for dense bitsets
    /// (cardinality >= [`DENSE_EXTRACT_THRESHOLD`]); falls back to scalar otherwise.
    pub fn to_values(&self) -> Vec<u16> {
        extract_bitset_to_values(&self.bits.0, self.cardinality)
    }

    /// Count of elements ≤ `value` in this container.
    ///
    /// Counts set bits in all full words before `value`'s word using SIMD popcount,
    /// then counts the partial last word with a bit mask — O(n_words / SIMD_width).
    pub fn rank(&self, value: u16) -> usize {
        let word_idx = (value >> 6) as usize;
        let bit_pos = value & 63;
        let prefix = popcount_u64_slice(&self.bits.0[..word_idx]);
        // Mask: bits 0..=bit_pos inclusive
        let mask = if bit_pos == 63 { u64::MAX } else { (1u64 << (bit_pos + 1)) - 1 };
        let partial = (self.bits.0[word_idx] & mask).count_ones() as usize;
        prefix + partial
    }

    /// The `rank`-th element (0-indexed) in sorted order, or `None` if out of bounds.
    ///
    /// Walks words until the cumulative popcount exceeds `rank`, then locates
    /// the exact bit with `trailing_zeros` after clearing lower set bits.
    pub fn select(&self, mut rank: usize) -> Option<u16> {
        for (word_idx, &word) in self.bits.0.iter().enumerate() {
            let cnt = word.count_ones() as usize;
            if rank < cnt {
                // Clear the lowest rank set bits, leaving the target bit as the LSB.
                let mut w = word;
                for _ in 0..rank {
                    w &= w - 1;
                }
                return Some((word_idx as u16) * 64 + w.trailing_zeros() as u16);
            }
            rank -= cnt;
        }
        None
    }

    /// Complement (XOR) all bits in the inclusive range [`lo`, `hi_inclusive`].
    ///
    /// Updates the cached cardinality via `recompute_cardinality` after the flip.
    pub fn flip_range(&mut self, lo: u16, hi_inclusive: u16) {
        if lo > hi_inclusive {
            return;
        }
        let lo_word = (lo >> 6) as usize;
        let hi_word = (hi_inclusive >> 6) as usize;
        let lo_bit = lo & 63;
        let hi_bit = hi_inclusive & 63;

        if lo_word == hi_word {
            // Single word: set bits in [lo_bit, hi_bit]
            let lo_mask = u64::MAX << lo_bit;
            let hi_mask = if hi_bit == 63 { u64::MAX } else { (1u64 << (hi_bit + 1)) - 1 };
            self.bits.0[lo_word] ^= lo_mask & hi_mask;
        } else {
            // First partial word: bits lo_bit..=63
            self.bits.0[lo_word] ^= u64::MAX << lo_bit;
            // Middle full words
            for w in (lo_word + 1)..hi_word {
                self.bits.0[w] ^= u64::MAX;
            }
            // Last partial word: bits 0..=hi_bit
            let hi_mask = if hi_bit == 63 { u64::MAX } else { (1u64 << (hi_bit + 1)) - 1 };
            self.bits.0[hi_word] ^= hi_mask;
        }
        self.recompute_cardinality();
    }

    pub fn iter(&self) -> BitsetIter<'_> {
        BitsetIter {
            bits: &self.bits.0,
            word_idx: 0,
            current_word: 0,
            started: false,
        }
    }

    // ── Set operations ──────────────────────────────────────────────
    //
    // All six operations delegate to `simd_support::bitwise`, which dispatches:
    //   AVX-512F + VPOPCNTDQ → bitwise op + fused popcount in one ZMM pass
    //   AVX-512F only         → bitwise op in ZMM, then separate popcount scan
    //   Scalar fallback       → word-at-a-time loop with count_ones()

    /// AND: intersection of two bitsets.
    pub fn and(&self, other: &BitsetContainer) -> BitsetContainer {
        let r = bitwise::and(&self.bits.0, &other.bits.0);
        BitsetContainer {
            bits: Box::new(AlignedBits(*r.bits)),
            cardinality: r.cardinality,
        }
    }

    /// OR: union of two bitsets.
    pub fn or(&self, other: &BitsetContainer) -> BitsetContainer {
        let r = bitwise::or(&self.bits.0, &other.bits.0);
        BitsetContainer {
            bits: Box::new(AlignedBits(*r.bits)),
            cardinality: r.cardinality,
        }
    }

    /// AND NOT: difference (self minus other).
    pub fn and_not(&self, other: &BitsetContainer) -> BitsetContainer {
        let r = bitwise::and_not(&self.bits.0, &other.bits.0);
        BitsetContainer {
            bits: Box::new(AlignedBits(*r.bits)),
            cardinality: r.cardinality,
        }
    }

    /// In-place AND.
    pub fn and_inplace(&mut self, other: &BitsetContainer) {
        self.cardinality = bitwise::and_inplace(&mut self.bits.0, &other.bits.0);
    }

    /// In-place OR.
    pub fn or_inplace(&mut self, other: &BitsetContainer) {
        self.cardinality = bitwise::or_inplace(&mut self.bits.0, &other.bits.0);
    }

    /// In-place AND NOT.
    pub fn and_not_inplace(&mut self, other: &BitsetContainer) {
        self.cardinality = bitwise::and_not_inplace(&mut self.bits.0, &other.bits.0);
    }
}

impl Default for BitsetContainer {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for BitsetContainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitsetContainer").field("cardinality", &self.cardinality).finish()
    }
}

impl PartialEq for BitsetContainer {
    fn eq(&self, other: &Self) -> bool {
        self.cardinality == other.cardinality && self.bits.0[..] == other.bits.0[..]
    }
}

impl Eq for BitsetContainer {}

pub struct BitsetIter<'a> {
    bits: &'a [u64; BITSET_WORDS],
    word_idx: usize,
    current_word: u64,
    started: bool,
}

impl Iterator for BitsetIter<'_> {
    type Item = u16;

    fn next(&mut self) -> Option<u16> {
        loop {
            if !self.started {
                if self.word_idx >= BITSET_WORDS {
                    return None;
                }
                self.current_word = self.bits[self.word_idx];
                self.started = true;
            }

            if self.current_word != 0 {
                let bit_pos = self.current_word.trailing_zeros() as u16;
                self.current_word &= self.current_word - 1;
                return Some((self.word_idx as u16) * 64 + bit_pos);
            }

            self.word_idx += 1;
            if self.word_idx >= BITSET_WORDS {
                return None;
            }
            self.current_word = self.bits[self.word_idx];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_contains() {
        let mut c = BitsetContainer::new();
        assert!(c.insert(0));
        assert!(c.insert(63));
        assert!(c.insert(64));
        assert!(c.insert(65535));
        assert!(!c.insert(63)); // duplicate
        assert_eq!(c.cardinality(), 4);
        assert!(c.contains(0));
        assert!(c.contains(63));
        assert!(c.contains(64));
        assert!(c.contains(65535));
        assert!(!c.contains(1));
    }

    #[test]
    fn remove() {
        let mut c = BitsetContainer::from_values(&[10, 20, 30]);
        assert!(c.remove(20));
        assert!(!c.remove(20));
        assert_eq!(c.cardinality(), 2);
        assert!(!c.contains(20));
    }

    #[test]
    fn min_max() {
        let c = BitsetContainer::from_values(&[100, 200, 300]);
        assert_eq!(c.min(), Some(100));
        assert_eq!(c.max(), Some(300));
        assert_eq!(BitsetContainer::new().min(), None);
    }

    #[test]
    fn and_operation() {
        let a = BitsetContainer::from_values(&[1, 3, 5, 7, 9]);
        let b = BitsetContainer::from_values(&[2, 3, 5, 8, 9]);
        let result = a.and(&b);
        assert_eq!(result.to_values(), vec![3, 5, 9]);
    }

    #[test]
    fn or_operation() {
        let a = BitsetContainer::from_values(&[1, 3, 5]);
        let b = BitsetContainer::from_values(&[2, 3, 6]);
        let result = a.or(&b);
        assert_eq!(result.to_values(), vec![1, 2, 3, 5, 6]);
    }

    #[test]
    fn and_not_operation() {
        let a = BitsetContainer::from_values(&[1, 3, 5, 7]);
        let b = BitsetContainer::from_values(&[3, 7, 9]);
        let result = a.and_not(&b);
        assert_eq!(result.to_values(), vec![1, 5]);
    }

    #[test]
    fn iter() {
        let c = BitsetContainer::from_values(&[0, 1, 63, 64, 128, 65535]);
        let collected: Vec<u16> = c.iter().collect();
        assert_eq!(collected, vec![0, 1, 63, 64, 128, 65535]);
    }

    #[test]
    fn rank_basic() {
        let c = BitsetContainer::from_values(&[10, 20, 30, 40]);
        assert_eq!(c.rank(5), 0);
        assert_eq!(c.rank(10), 1);
        assert_eq!(c.rank(15), 1);
        assert_eq!(c.rank(20), 2);
        assert_eq!(c.rank(40), 4);
        assert_eq!(c.rank(99), 4);
    }

    #[test]
    fn rank_word_boundaries() {
        // Values straddling u64 word boundaries (every 64 values)
        let c = BitsetContainer::from_values(&[63, 64, 127, 128]);
        assert_eq!(c.rank(63), 1);
        assert_eq!(c.rank(64), 2);
        assert_eq!(c.rank(127), 3);
        assert_eq!(c.rank(128), 4);
    }

    #[test]
    fn select_basic() {
        let c = BitsetContainer::from_values(&[10, 20, 30]);
        assert_eq!(c.select(0), Some(10));
        assert_eq!(c.select(1), Some(20));
        assert_eq!(c.select(2), Some(30));
        assert_eq!(c.select(3), None);
    }

    #[test]
    fn rank_select_round_trip() {
        let vals: Vec<u16> = (0..200u16).map(|i| i * 3).collect();
        let c = BitsetContainer::from_sorted_values(&vals);
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(c.select(i), Some(v), "select({i})");
            assert_eq!(c.rank(v), i + 1, "rank({v})");
        }
    }

    #[test]
    fn flip_range_single_word() {
        let mut c = BitsetContainer::new();
        c.flip_range(0, 3); // flip bits 0,1,2,3 → set them
        assert_eq!(c.to_values(), vec![0, 1, 2, 3]);
        c.flip_range(1, 2); // unset bits 1,2
        assert_eq!(c.to_values(), vec![0, 3]);
    }

    #[test]
    fn flip_range_cross_word() {
        let mut c = BitsetContainer::new();
        c.flip_range(60, 67); // straddles first word boundary (63/64)
        let vals = c.to_values();
        assert_eq!(vals, vec![60, 61, 62, 63, 64, 65, 66, 67]);
    }

    #[test]
    fn flip_range_then_flip_back() {
        let mut c = BitsetContainer::from_values(&[1, 2, 3]);
        let orig_card = c.cardinality();
        c.flip_range(0, 65535);
        c.flip_range(0, 65535);
        assert_eq!(c.cardinality(), orig_card);
        assert_eq!(c.to_values(), vec![1, 2, 3]);
    }

    #[test]
    fn flip_range_empty_range_is_noop() {
        let mut c = BitsetContainer::from_values(&[5, 10]);
        c.flip_range(7, 4); // lo > hi_inclusive, should be a no-op
        assert_eq!(c.to_values(), vec![5, 10]);
    }
}
