//! The memtable's sorted map: a skip list from byte keys to `u128` value-log
//! pointers, where every entry carries the write sequence that produced it and
//! a tombstone flag. Conflicting writes to one key resolve to the higher
//! sequence, so the winner in memory is the one WAL replay would pick.

use crate::storage::lsm::version::Version;
use crate::support::key_prefix_of;
use crate::support::simd_support::simd_support::compare_bytes_simd;
use core::cmp::Ordering;
#[cfg(test)]
use std::borrow::Cow;

const MAX_LEVEL: usize = 32;
const NONE_VALUE: u32 = u32::MAX;

/// A materialized entry snapshot for external use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyValueRecord {
    pub key: Vec<u8>,
    pub value: u128,
    pub tombstone: bool,
    pub version: Version,
}

/// What a seq-aware write did to the skip list. The value-log garbage
/// accounting depends on it: the pointer a write replaced here is referenced
/// by nothing any more, and a stale write's own pointer never was.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Applied {
    /// The write is now the key's entry in this skip list. `displaced` is the
    /// live value it replaced in place; `None` when the key was absent here
    /// or held a tombstone.
    Won { displaced: Option<u128> },
    /// The key already held a newer write, so this one was dropped.
    Stale,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum InsertError {
    #[error("capacity exceeded (nodes/links/keys)")]
    CapacityExceeded,
}

/// One entry. A node owns no allocation: its key bytes live in the list's
/// shared key arena (`SkipList::keys`) and its forward links in the shared
/// link arena (`SkipList::links`), so an insert appends to three vectors
/// instead of allocating per entry.
#[derive(Clone, Debug)]
struct Node {
    /// Where the key starts in the key arena: the key is
    /// `keys[key_index_offset..key_index_offset + key_length]`.
    key_index_offset: u32,
    key_length: u32,
    /// The key's first 8 bytes as an integer, so most comparisons are
    /// settled without reading the arena.
    key_prefix: u64,
    value: u128,
    tombstone: bool,
    version: Version,
    /// How many levels the node is linked at (its number of forward links).
    height: u8,
    /// Where the node's forward links start in the link arena: the link at
    /// level `l` (for `l < height`) is `links[links_index_offset + l]`.
    links_index_offset: u32,
}

impl Node {
    #[allow(clippy::too_many_arguments)]
    fn new(
        key_index_offset: u32,
        key_length: u32,
        key_prefix: u64,
        value: u128,
        tombstone: bool,
        version: Version,
        height: u8,
        links_index_offset: u32,
    ) -> Self {
        Self {
            key_index_offset,
            key_length,
            key_prefix,
            value,
            tombstone,
            version,
            height,
            links_index_offset,
        }
    }
}

/// A CPU-cache-friendly skip list for byte-array keys and `u128` values.
/// \- Keys are stored in sorted (lexicographic) order.
/// \- Expected O(log n) `get/insert/remove`.
/// \- Max level is 32.
/// \- Cache-friendliness comes from storing nodes, forward links, and key bytes in contiguous arenas
///   (`Vec<Node>` + `Vec<u32>` + `Vec<u8>`), and linking via `u32` indices.
#[derive(Clone, Debug)]
pub struct SkipList {
    nodes: Vec<Node>,
    links: Vec<u32>,
    keys: Vec<u8>,
    head: u32,
    level: u8,
    no_live_nodes: usize,      // live (non-tombstoned) count
    no_tombstone_nodes: usize, // tombstoned count
    rng: XorShift64,
    max_capacity: usize,
}

impl Default for SkipList {
    fn default() -> Self {
        Self::new()
    }
}

impl SkipList {
    pub fn new() -> Self {
        Self::with_capacity(crate::defaults::memtable::MAX_CAPACITY)
    }

    pub fn with_capacity(max_capacity: usize) -> Self {
        let mut nodes = Vec::new();
        let mut links = Vec::new();
        let keys = Vec::new();

        links.resize(MAX_LEVEL, NONE_VALUE);

        // Sentinel head node at index 0.
        nodes.push(Node::new(0, 0, 0u64, 0, true, Version::default(), MAX_LEVEL as u8, 0));

        Self {
            nodes,
            links,
            keys,
            head: 0,
            level: 1,
            no_live_nodes: 0,
            no_tombstone_nodes: 0,
            rng: XorShift64::seeded(0x9E37_79B9_7F4A_7C15),
            max_capacity,
        }
    }

    /// Returns true when `new` is the same as, or newer than, `existing`.
    ///
    /// Callers feed the global monotonically-increasing write sequence
    /// (`Database::next_seq`) so that conflicting writes to one key resolve
    /// to the higher sequence.
    ///
    /// The sequence is carried at its full `u64` width, so this is a plain
    /// comparison. **Do not narrow it** to a smaller, wrapping type: an L1
    /// entry keeps its sequence for the life of the database, so the gap to a
    /// later write of the same key is unbounded, and a wrapping comparison
    /// would rank that newer write as older. See
    /// `tree::test_write_after_huge_seq_gap_beats_cold_l1_entry`.
    #[inline]
    fn seq_is_newer_or_equal(new: Version, existing: Version) -> bool {
        new >= existing
    }

    /// Get the maximum capacity (total nodes that can be stored)
    pub fn max_capacity(&self) -> usize {
        self.max_capacity
    }

    /// Live (non-tombstoned) entry count.
    pub fn number_of_live_nodes(&self) -> usize {
        self.no_live_nodes
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.no_live_nodes == 0
    }

    /// Tombstoned entry count.
    pub fn number_of_tombstone_nodes(&self) -> usize {
        self.no_tombstone_nodes
    }

    /// Returns true if any internal arena (nodes, links, keys) has reached
    /// at least half of its u32 address-space limit. This is a conservative
    /// early-warning so callers can flush before inserts start failing.
    pub fn arenas_half_full(&self) -> bool {
        let half = (u32::MAX as usize) / 2;
        self.nodes.len() >= half || self.links.len() >= half || self.keys.len() >= half
    }

    /// Returns the raw entry for a key as `(value, version, tombstone)`, or
    /// `None` if the key is absent from this skip list. Crucially this
    /// distinguishes a **tombstone** (present but deleted) from **absent**,
    /// which a caller searching layered storage needs so that a tombstone in
    /// a newer layer shadows a live value in an older one.
    pub fn entry(&self, key: &[u8]) -> Option<(u128, Version, bool)> {
        let (found, _) = self.find_path(key);
        let idx = found?;
        let n = &self.nodes[idx as usize];
        Some((n.value, n.version, n.tombstone))
    }

    #[cfg(test)]
    pub fn get_value(&self, key: &[u8]) -> Option<u128> {
        match self.entry(key) {
            Some((v, _, false)) => Some(v),
            _ => None,
        }
    }

    /// Insert/update using an explicit, caller-supplied sequence (the global
    /// write sequence), with **highest-sequence-wins** conflict resolution.
    ///
    /// If the key already exists with a *newer* sequence, the write is
    /// dropped ([`Applied::Stale`]). So the in-memory winner of two racing
    /// writes to one key is the one recovery picks (it replays in sequence
    /// order), and a value seen live survives a crash.
    pub fn try_insert_with_seq(&mut self, key: &[u8], value: u128, seq: impl Into<Version>) -> Result<Applied, InsertError> {
        let seq = seq.into();
        let (found, update) = self.find_path(key);

        if let Some(idx) = found {
            let node = &mut self.nodes[idx as usize];
            if !Self::seq_is_newer_or_equal(seq, node.version) {
                // A newer write already won for this key; drop this stale one.
                return Ok(Applied::Stale);
            }
            let old = node.value;
            let was_tombstone = node.tombstone;
            node.value = value;
            node.tombstone = false;
            node.version = seq;
            if was_tombstone {
                self.no_tombstone_nodes = self.no_tombstone_nodes.saturating_sub(1);
                self.no_live_nodes += 1;
                return Ok(Applied::Won { displaced: None });
            }
            return Ok(Applied::Won { displaced: Some(old) });
        }

        self.splice_new_node(key, value, false, seq, update)?;
        self.no_live_nodes += 1;
        Ok(Applied::Won { displaced: None })
    }

    /// Insert a tombstone for `key` using an explicit sequence, with
    /// **highest-sequence-wins** resolution. This is the production delete
    /// path: unlike a value insert followed by a remove, the node is created
    /// *directly* as a tombstone — no transient live state (and no
    /// placeholder value) ever exists, even within the caller's lock.
    ///
    /// - Key absent → a new node is allocated already tombstoned (its value
    ///   is `0`, never readable: tombstone value bits are meaningless).
    /// - Key present with an older/equal sequence → flipped to a tombstone
    ///   in place; its stored value is left untouched.
    /// - Key present with a newer sequence → this delete is stale (a newer
    ///   write superseded it) and is dropped.
    ///
    /// [`Applied::Won`] carries the live value a flip displaced.
    pub fn insert_tombstone_with_seq(&mut self, key: &[u8], seq: impl Into<Version>) -> Result<Applied, InsertError> {
        let seq = seq.into();
        let (found, update) = self.find_path(key);

        if let Some(idx) = found {
            let node = &mut self.nodes[idx as usize];
            if !Self::seq_is_newer_or_equal(seq, node.version) {
                // A newer write already won for this key; drop the stale delete.
                return Ok(Applied::Stale);
            }
            node.version = seq;
            if !node.tombstone {
                node.tombstone = true;
                self.no_live_nodes -= 1;
                self.no_tombstone_nodes += 1;
                return Ok(Applied::Won { displaced: Some(node.value) });
            }
            return Ok(Applied::Won { displaced: None });
        }

        self.splice_new_node(key, 0, true, seq, update)?;
        self.no_tombstone_nodes += 1;
        Ok(Applied::Won { displaced: None })
    }

    /// Allocate a node with a freshly drawn height and splice it in at every
    /// level it owns. `update` is the predecessor array from `find_path` for
    /// the same key; the caller is responsible for the live/tombstone counters.
    fn splice_new_node(&mut self, key: &[u8], value: u128, tombstone: bool, seq: Version, mut update: [u32; MAX_LEVEL]) -> Result<(), InsertError> {
        let node_level = self.random_level();
        if node_level > self.level {
            #[allow(clippy::needless_range_loop)]
            for lvl in self.level as usize..node_level as usize {
                update[lvl] = self.head;
            }
        }

        let new_idx = self
            .allocate_node(key, value, tombstone, seq, node_level)
            .ok_or(InsertError::CapacityExceeded)?;

        #[allow(clippy::needless_range_loop)]
        for lvl in 0..node_level as usize {
            let prev = update[lvl];
            let next = self.next(prev, lvl);
            self.set_next(new_idx, lvl, next);
            self.set_next(prev, lvl, new_idx);
        }
        if node_level > self.level {
            self.level = node_level;
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            list: self,
            current: self.next(self.head, 0),
        }
    }

    /// Returns an iterator over all live entries whose keys start with `prefix`.
    ///
    /// Efficient: seeks to the first key `>= prefix` then walks level-0 until
    /// the prefix no longer matches. Accepts both borrowed and owned prefixes.
    #[cfg(test)]
    pub fn scan_prefix<'a>(&'a self, prefix: impl Into<Cow<'a, [u8]>>) -> PrefixIter<'a> {
        let prefix = prefix.into();
        let start = self.lower_bound(prefix.as_ref());
        PrefixIter {
            list: self,
            current: start,
            prefix,
        }
    }

    /// Returns the first node index whose key is `>= target` (or NONE if no such node).
    #[inline]
    fn lower_bound(&self, target: &[u8]) -> u32 {
        let (_, update) = self.find_path(target);
        let cand = self.next(update[0], 0);
        if cand == NONE_VALUE {
            return NONE_VALUE;
        }
        // `update[0]` is the rightmost node < target; hence cand is >= target.
        cand
    }

    #[inline]
    fn key_slice(&self, node: u32) -> &[u8] {
        let n = &self.nodes[node as usize];
        let start = n.key_index_offset as usize;
        let end = start + n.key_length as usize;
        &self.keys[start..end]
    }

    fn allocate_node(&mut self, key: &[u8], value: u128, tombstone: bool, seq: Version, height: u8) -> Option<u32> {
        let idx = self.nodes.len();

        let links_base = self.links.len();
        let h = height as usize;
        if links_base.saturating_add(h) >= (u32::MAX as usize) {
            return None;
        }
        self.links.resize(links_base + h, NONE_VALUE);

        let key_base = self.keys.len();
        let key_len = key.len();
        if key_base.saturating_add(key_len) >= (u32::MAX as usize) {
            return None;
        }
        self.keys.extend_from_slice(key);

        let prefix = key_prefix_of(key);

        self.nodes.push(Node::new(
            key_base as u32,
            key_len as u32,
            prefix,
            value,
            tombstone,
            seq,
            height,
            links_base as u32,
        ));

        Some(idx as u32)
    }

    #[inline]
    fn next(&self, node: u32, lvl: usize) -> u32 {
        let n = &self.nodes[node as usize];
        if lvl >= n.height as usize {
            return NONE_VALUE;
        }
        let i = n.links_index_offset as usize + lvl;
        self.links[i]
    }

    #[inline]
    fn set_next(&mut self, node: u32, lvl: usize, val: u32) {
        let height = self.nodes[node as usize].height as usize;
        debug_assert!(lvl < height, "level out of bounds for node height");
        let base = self.nodes[node as usize].links_index_offset as usize;
        self.links[base + lvl] = val;
    }

    #[inline]
    fn key_cmp_node_target(&self, node: u32, target: &[u8], target_prefix: u64) -> Ordering {
        let n = &self.nodes[node as usize];
        match n.key_prefix.cmp(&target_prefix) {
            Ordering::Equal => compare_bytes_simd(self.key_slice(node), target),
            other => other,
        }
    }

    /// Returns (found_index, update_predecessors).
    /// `update[l]` is the index of the rightmost node at level `l` whose key is < target.
    fn find_path(&self, key: &[u8]) -> (Option<u32>, [u32; MAX_LEVEL]) {
        let mut update = [self.head; MAX_LEVEL];
        let mut x = self.head;

        let key_prefix = key_prefix_of(key);

        // Start from top level and walk down.
        let mut lvl = self.level as i32 - 1;
        while lvl >= 0 {
            let l = lvl as usize;
            loop {
                let next = self.next(x, l);
                if next == NONE_VALUE {
                    break;
                }

                let ord = { self.key_cmp_node_target(next, key, key_prefix) };

                match ord {
                    Ordering::Less => x = next,
                    Ordering::Equal | Ordering::Greater => break,
                }
            }
            update[l] = x;
            lvl -= 1;
        }

        let candidate = self.next(x, 0);
        if candidate != NONE_VALUE && self.key_slice(candidate) == key {
            return (Some(candidate), update);
        }
        (None, update)
    }

    fn random_level(&mut self) -> u8 {
        // Geometric distribution with p=0.5.
        let mut lvl: u8 = 1;
        while lvl < (MAX_LEVEL as u8) {
            if (self.rng.next_u64() & 1) == 0 {
                break;
            }
            lvl += 1;
        }
        lvl
    }

    #[cfg(test)]
    fn validate_insert(&self) {
        let mut prev_key: Option<Vec<u8>> = None;
        let mut count = 0usize;
        let mut cur = self.next(self.head, 0);
        while cur != NONE_VALUE {
            let k = self.key_slice(cur).to_vec();
            if let Some(pk) = prev_key.as_ref() {
                assert!(pk.as_slice() < k.as_slice(), "keys not strictly increasing");
            }
            prev_key = Some(k);
            count += 1;
            assert!(count <= self.no_live_nodes + self.no_tombstone_nodes + 1, "cycle detected");
            cur = self.next(cur, 0);
        }
        assert_eq!(count, self.no_live_nodes + self.no_tombstone_nodes);

        for l in 1..self.level as usize {
            let mut low = self.next(self.head, 0);
            let mut high = self.next(self.head, l);
            while high != NONE_VALUE {
                while low != NONE_VALUE && self.key_slice(low) != self.key_slice(high) {
                    low = self.next(low, 0);
                }
                assert!(low != NONE_VALUE, "level {l} contains a node missing in level 0");
                high = self.next(high, l);
            }
        }
    }

    /// Iterates over all entries including tombstones, yielding `(key, value, seq, tombstone)`.
    ///
    /// Flushes and scans use it: a tombstone must reach the older layers it shadows.
    pub fn iter_raw(&self) -> IterRaw<'_> {
        IterRaw {
            list: self,
            current: self.next(self.head, 0),
        }
    }

    /// Like `iter_raw` but starts at the first entry whose key is `>= start`.
    pub fn iter_raw_from(&self, start: &[u8]) -> IterRaw<'_> {
        IterRaw {
            list: self,
            current: self.lower_bound(start),
        }
    }

    /// Collects all entries (including tombstones) into a sorted vector.
    ///
    /// - Order: ascending lexicographic key order.
    /// - Includes tombstones.
    /// - Excludes the sentinel/head entry (empty key; key_len == 0).
    pub fn collect_key_value_records(&self) -> Vec<KeyValueRecord> {
        let mut out = Vec::with_capacity(self.no_live_nodes + self.no_tombstone_nodes);

        let mut cur = self.next(self.head, 0);
        while cur != NONE_VALUE {
            let n = &self.nodes[cur as usize];
            // Skip sentinel / special empty key.

            if n.key_length == 0 {
                cur = self.next(cur, 0);
                continue;
            }

            out.push(KeyValueRecord {
                key: self.key_slice(cur).to_vec(),
                value: n.value,
                tombstone: n.tombstone,
                version: n.version,
            });

            cur = self.next(cur, 0);
        }
        out
    }

    /// Returns `true` if allocating a new node with the given key length and height would fit
    /// within the skip list's internal u32-indexed arenas.
    ///
    /// This is a fast, side-effect-free capacity guard you can use before attempting an insert.
    ///
    /// Note: this only checks for *representable indices* (u32 limits), not allocator OOM.
    pub fn can_alloc_node_for(&self, key_len: usize, height: u8) -> bool {
        // Node index must fit in u32.
        let idx = self.nodes.len();
        if idx >= (u32::MAX as usize) {
            return false;
        }

        // Links arena must fit in u32 and have room for `height` forward pointers.
        let links_base = self.links.len();
        let h = height as usize;
        if links_base.saturating_add(h) >= (u32::MAX as usize) {
            return false;
        }

        // Keys arena must fit in u32 and have room for the key bytes.
        let key_base = self.keys.len();
        if key_base.saturating_add(key_len) >= (u32::MAX as usize) {
            return false;
        }

        true
    }

    /// Convenience guard for typical inserts when you don't know the height yet.
    ///
    /// This makes a conservative check using `MAX_LEVEL` (worst-case node height). If this
    /// returns `false`, an insert may fail with `CapacityExceeded`.
    pub fn can_insert_key(&self, key: &[u8]) -> bool {
        self.can_alloc_node_for(key.len(), MAX_LEVEL as u8)
    }
}

/// Iterates over all nodes (including tombstones) yielding `(key, value, seq, tombstone)`.
pub struct IterRaw<'a> {
    list: &'a SkipList,
    current: u32,
}

impl<'a> Iterator for IterRaw<'a> {
    type Item = (&'a [u8], u128, Version, bool);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current == NONE_VALUE {
            return None;
        }
        let idx = self.current;
        self.current = self.list.next(idx, 0);
        let n = &self.list.nodes[idx as usize];
        Some((self.list.key_slice(idx), n.value, n.version, n.tombstone))
    }
}

#[derive(Clone, Debug)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn seeded(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xDEAD_BEEF_CAFE_BABE } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

/// Iterates over live entries (tombstones skipped) in key order.
#[cfg(test)]
pub struct Iter<'a> {
    list: &'a SkipList,
    current: u32,
}

#[cfg(test)]
impl<'a> Iterator for Iter<'a> {
    type Item = (&'a [u8], u128);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current != NONE_VALUE {
            let idx = self.current;
            self.current = self.list.next(idx, 0);
            let n = &self.list.nodes[idx as usize];
            if n.tombstone {
                continue;
            }
            return Some((self.list.key_slice(idx), n.value));
        }
        None
    }
}

/// Iterates over live entries whose keys start with `prefix`, skipping tombstones.
///
/// Holds the prefix as `Cow<[u8]>` so it works with both borrowed slices and
/// owned `Vec<u8>` without duplicating the struct.
#[cfg(test)]
pub struct PrefixIter<'a> {
    list: &'a SkipList,
    current: u32,
    prefix: Cow<'a, [u8]>,
}

#[cfg(test)]
impl<'a> Iterator for PrefixIter<'a> {
    type Item = (&'a [u8], u128);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current != NONE_VALUE {
            let idx = self.current;
            let key = self.list.key_slice(idx);
            if !key.starts_with(self.prefix.as_ref()) {
                self.current = NONE_VALUE;
                return None;
            }
            self.current = self.list.next(idx, 0);
            let n = &self.list.nodes[idx as usize];
            if n.tombstone {
                continue;
            }
            return Some((key, n.value));
        }
        None
    }
}

/// Test-only convenience wrappers. They allocate a monotonically increasing
/// sequence and delegate to the production `*_with_seq` API, so the unit
/// tests exercise the real (highest-sequence-wins) conflict-resolution path
/// rather than a parallel code path. A single process-wide counter gives
/// every call a strictly higher sequence than any prior one, reproducing
/// last-write-wins for repeated writes to one key.
#[cfg(test)]
impl SkipList {
    fn next_test_seq() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(1);
        SEQ.fetch_add(1, Ordering::Relaxed)
    }

    pub fn insert(&mut self, key: &[u8], value: u128) -> bool {
        self.try_insert_with_seq(key, value, Self::next_test_seq()).is_ok()
    }

    pub fn try_insert(&mut self, key: &[u8], value: u128) -> Result<Option<u128>, InsertError> {
        Ok(match self.try_insert_with_seq(key, value, Self::next_test_seq())? {
            Applied::Won { displaced } => displaced,
            Applied::Stale => None,
        })
    }

    pub fn remove(&mut self, key: &[u8]) -> Option<u128> {
        // Report the displaced live value (model semantics), then delete via
        // the production tombstone path.
        let old = match self.entry(key) {
            Some((v, _, false)) => Some(v),
            _ => None,
        };
        self.insert_tombstone_with_seq(key, Self::next_test_seq()).unwrap();
        old
    }
}

#[cfg(test)]
mod tests;
