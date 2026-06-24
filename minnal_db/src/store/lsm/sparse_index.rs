//! Sparse in-memory index over an L1 SSTable's sorted keys.
//!
//! It records the byte offset of every `SAMPLE_INTERVAL`-th entry's frame, so a
//! point lookup can binary-search to a nearby start offset and scan at most
//! `SAMPLE_INTERVAL` entries instead of scanning the whole file from offset 0.
//! This turns a present-key lookup from O(N) into O(log(samples) + interval).
//!
//! Like the [bloom filter](super::bloom), it is **derived** and rebuilt from the
//! file (open / compaction), never persisted. Because [`block_start`] returns a
//! *file offset*, the caller must validate that offset against the file it is
//! about to read (a concurrent compaction can swap the L1 file) and fall back to
//! a full scan if it no longer lines up — so a stale index can never produce a
//! wrong result. See `LSMTree::lookup_in_sstable_file_from`.
//!
//! [`block_start`]: SparseIndex::block_start

pub(crate) struct SparseIndex {
    /// `(first key of the sampled block, byte offset of that frame)`, ascending
    /// by key and offset (SSTable entries are written in sorted key order).
    samples: Vec<(Vec<u8>, u64)>,
}

impl SparseIndex {
    /// Sample one entry per this many. Smaller ⇒ faster lookups, more memory.
    pub(crate) const SAMPLE_INTERVAL: u64 = 16;

    pub(crate) fn new() -> Self {
        Self { samples: Vec::new() }
    }

    /// Record a sampled entry. Callers push in ascending key/offset order, once
    /// per [`SAMPLE_INTERVAL`](Self::SAMPLE_INTERVAL) entries (including entry 0).
    pub(crate) fn push(&mut self, key: Vec<u8>, offset: u64) {
        self.samples.push((key, offset));
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Byte offset at which to begin scanning for `key`: the offset of the latest
    /// sample whose key is `<= key`, or 0 when `key` precedes the first sample.
    pub(crate) fn block_start(&self, key: &[u8]) -> u64 {
        match self.samples.binary_search_by(|(k, _)| k.as_slice().cmp(key)) {
            Ok(i) => self.samples[i].1,
            Err(0) => 0,
            Err(i) => self.samples[i - 1].1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_start_picks_latest_sample_at_or_below_key() {
        let mut idx = SparseIndex::new();
        // samples at keys b, d, f with offsets 100, 200, 300
        idx.push(b"b".to_vec(), 100);
        idx.push(b"d".to_vec(), 200);
        idx.push(b"f".to_vec(), 300);

        assert_eq!(idx.block_start(b"a"), 0); // before first sample
        assert_eq!(idx.block_start(b"b"), 100); // exact
        assert_eq!(idx.block_start(b"c"), 100); // between b and d
        assert_eq!(idx.block_start(b"d"), 200); // exact
        assert_eq!(idx.block_start(b"e"), 200); // between d and f
        assert_eq!(idx.block_start(b"f"), 300); // exact
        assert_eq!(idx.block_start(b"z"), 300); // after last sample
    }

    #[test]
    fn empty_index_starts_at_zero() {
        let idx = SparseIndex::new();
        assert_eq!(idx.block_start(b"anything"), 0);
    }
}
