//! The value-log pointers the LSM stops referencing, reported so the value log
//! can count their records as garbage.
//!
//! A pointer stops being referenced in exactly two places: a memtable write that
//! replaces a live value in place (reported by [`Applied`](super::Applied) to the
//! writer), and the L0→L1 merge, which keeps one entry per key and drops the
//! others. This module covers the merge. Counting there instead of on the write
//! path is what lets a write skip looking up the key's previous pointer.
//!
//! The counters are a hint for choosing which segments GC rewrites; GC decides
//! what is live from the LSM itself. So "referenced" means held by *any* entry in
//! any layer, the newest or not: a record shadowed by a newer write stays live in
//! the counters until the merge drops its entry, and
//! [`referenced_pointers`](super::LSMTree::referenced_pointers) counts it the same
//! way, so a rebuild agrees with the running total.

use std::collections::HashSet;

/// A record whose last LSM entry a merge dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Displaced {
    pub key_len: usize,
    pub pointer: u128,
}

/// Receives the records an L0→L1 merge stopped referencing, once the merged L1
/// file is installed. `KVStore` counts them as value-log garbage.
pub(crate) trait DisplacedSink: Send + Sync {
    fn on_displaced(&self, records: &[Displaced]);
}

/// Collects one merge's dropped pointers, each once.
///
/// A pointer names one write of one key, so the same pointer can appear twice
/// only as a duplicate of the same entry (an L0 file a crash left behind after
/// its merge, say). It is reported only if it is not also the entry the merge
/// keeps, and only once however many copies lose.
#[derive(Default)]
pub(super) struct DisplacedCollector {
    seen: HashSet<u128>,
    out: Vec<Displaced>,
}

impl DisplacedCollector {
    /// `loser` lost to `winner` for a key of `key_len` bytes. Each is
    /// `Some(pointer)` for a live entry and `None` for a tombstone.
    pub(super) fn lost(&mut self, key_len: usize, loser: Option<u128>, winner: Option<u128>) {
        let Some(pointer) = loser else {
            return;
        };
        if winner == Some(pointer) || !self.seen.insert(pointer) {
            return;
        }
        self.out.push(Displaced { key_len, pointer });
    }

    pub(super) fn into_records(self) -> Vec<Displaced> {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pointer_is_reported_once_and_never_when_it_is_kept() {
        let mut c = DisplacedCollector::default();
        c.lost(3, Some(7), Some(9)); // an older write
        c.lost(3, Some(7), Some(9)); // its duplicate
        c.lost(3, Some(9), Some(9)); // a duplicate of the kept entry
        c.lost(3, None, Some(9)); // a tombstone holds no record
        c.lost(4, Some(11), None); // dropped under a tombstone
        assert_eq!(
            c.into_records(),
            vec![Displaced { key_len: 3, pointer: 7 }, Displaced { key_len: 4, pointer: 11 }]
        );
    }
}
