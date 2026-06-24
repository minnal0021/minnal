//! Per-SSTable Bloom filter — a probabilistic "is this key absent?" test used
//! to skip the linear SSTable scan on a miss (see
//! [`LSMTree::lookup_in_sstable_file`](super::lsm_tree)).
//!
//! [`contains`](BloomFilter::contains) returning `false` is **exact**: the key
//! is definitely not in the file, so the caller can return `Missing` without
//! scanning. A `true` answer may be a false positive (default rate ~1%), so the
//! caller still scans.
//!
//! The filter is a **derived in-memory structure**: it is rebuilt from the
//! SSTable on open (in the same pass that computes min/max) and from the merged
//! entries after a compaction — never persisted. So it is always consistent
//! with the file it guards, and a crash can never leave a stale filter that
//! would falsely skip a live key.

use mm3h::Murmur3Hasher;
use std::hash::Hasher;

pub(crate) struct BloomFilter {
    bits: Vec<u64>,
    /// Number of addressable bits (`bits.len() * 64`); always ≥ 64.
    num_bits: u64,
    num_hashes: u32,
}

impl BloomFilter {
    /// Bits per key — sized for a ~1% false-positive rate.
    const BITS_PER_KEY: usize = 10;
    /// Hash functions: k = ln2 · (m/n) ≈ 7 at 10 bits/key.
    const NUM_HASHES: u32 = 7;

    /// Build a filter sized for `expected` keys and insert every key in `keys`.
    pub(crate) fn build<'a>(keys: impl IntoIterator<Item = &'a [u8]>, expected: usize) -> Self {
        let words = ((expected.max(1) * Self::BITS_PER_KEY) as u64).div_ceil(64).max(1);
        let mut filter = Self {
            bits: vec![0u64; words as usize],
            num_bits: words * 64,
            num_hashes: Self::NUM_HASHES,
        };
        for key in keys {
            filter.insert(key);
        }
        filter
    }

    fn insert(&mut self, key: &[u8]) {
        let (h1, h2) = Self::base_hashes(key);
        for i in 0..self.num_hashes as u64 {
            let bit = h1.wrapping_add(i.wrapping_mul(h2)) % self.num_bits;
            self.bits[(bit / 64) as usize] |= 1u64 << (bit % 64);
        }
    }

    /// Returns `false` only when `key` is **definitely** absent; `true` means
    /// "possibly present" (scan to confirm).
    pub(crate) fn contains(&self, key: &[u8]) -> bool {
        let (h1, h2) = Self::base_hashes(key);
        for i in 0..self.num_hashes as u64 {
            let bit = h1.wrapping_add(i.wrapping_mul(h2)) % self.num_bits;
            if self.bits[(bit / 64) as usize] & (1u64 << (bit % 64)) == 0 {
                return false;
            }
        }
        true
    }

    /// Two independent 64-bit hashes, combined by double hashing
    /// (`g_i = h1 + i·h2`) to derive the `num_hashes` bit positions.
    fn base_hashes(key: &[u8]) -> (u64, u64) {
        let mut h1 = Murmur3Hasher::new_with_seed(0x9E37_79B1);
        h1.write(key);
        let mut h2 = Murmur3Hasher::new_with_seed(0x85EB_CA77);
        h2.write(key);
        (h1.finish(), h2.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_false_negatives_and_low_fpr() {
        let present: Vec<Vec<u8>> = (0..10_000u32).map(|i| format!("key:{i:08}").into_bytes()).collect();
        let bloom = BloomFilter::build(present.iter().map(|k| k.as_slice()), present.len());

        // Every inserted key must test positive (Bloom filters have no false negatives).
        for k in &present {
            assert!(bloom.contains(k), "false negative for {:?}", String::from_utf8_lossy(k));
        }

        // Absent keys should mostly test negative; verify the FP rate is sane (<5%).
        let mut false_positives = 0;
        let trials = 10_000u32;
        for i in 10_000..10_000 + trials {
            if bloom.contains(format!("key:{i:08}").as_bytes()) {
                false_positives += 1;
            }
        }
        let fpr = false_positives as f64 / trials as f64;
        assert!(fpr < 0.05, "false-positive rate too high: {fpr}");
    }

    #[test]
    fn empty_build_is_safe() {
        let bloom = BloomFilter::build(std::iter::empty::<&[u8]>(), 0);
        assert!(!bloom.contains(b"anything"));
    }
}
