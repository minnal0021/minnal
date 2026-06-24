#[allow(clippy::module_inception)]
pub mod skip_list {
    use crate::support::key_prefix_of;
    use crate::support::simd_support::simd_support::compare_bytes_simd;
    use core::cmp::Ordering;
    use std::borrow::Cow;

    const DEFAULT_MAX_CAPACITY: usize = 100_000; // Default max number of nodes (including tombstones)
    const MAX_LEVEL: usize = 32;
    const NONE_VALUE: u32 = u32::MAX;

    /// A materialized entry snapshot for external use.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct KeyValueRecord {
        pub key: Vec<u8>,
        pub value: u128,
        pub tombstone: bool,
        pub seq: u32,
    }

    #[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
    pub enum InsertError {
        #[error("capacity exceeded (nodes/links/keys)")]
        CapacityExceeded,
    }

    // key_index_offset is the offset (start index) into the skip list’s keys: Vec<u8> arena where this node’s key bytes are stored.
    // In this implementation, keys aren’t stored as Vec<u8> per node (which would allocate per entry).
    // Instead, all key bytes are appended into one contiguous buffer (SkipList.keys), and each node stores a slice descriptor:
    // key_index_offset: u32 → start position in self.keys
    // key_length: u32 → length of the key
    // Then the real key for a node is obtained via:
    // key_slice(node):
    // start = n.key_index_offset as usize
    // end = start + n.key_length as usize
    // returns &self.keys[start..end]

    ///links_index_offset is the start offset (index) into the skip list’s links: Vec<u32> arena where this node’s forward pointers (“tower”) are stored.
    // The implementation stores all forward-links for all nodes in one contiguous Vec<u32> (an arena), instead of Vec<u32> per node. Each node then needs two things to locate its links:
    // links_index_offset: u32 → the first index in self.links for this node
    // height: u8 → how many levels this node actually has (how many forward pointers it owns)
    // So for a given node:
    // the forward pointer at level lvl lives at:
    // self.links[node.links_index_offset as usize + lvl]
    // but only if lvl < node.height (otherwise that node has no link at that level)
    #[derive(Clone, Debug)]
    struct Node {
        key_index_offset: u32,
        key_length: u32,
        key_prefix: u64,
        value: u128,
        tombstone: bool,
        sequence: u32,
        height: u8,
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
            sequence: u32,
            height: u8,
            links_index_offset: u32,
        ) -> Self {
            Self {
                key_index_offset,
                key_length,
                key_prefix,
                value,
                tombstone,
                sequence,
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
            Self::with_capacity(DEFAULT_MAX_CAPACITY)
        }

        pub fn with_capacity(max_capacity: usize) -> Self {
            let mut nodes = Vec::new();
            let mut links = Vec::new();
            let keys = Vec::new();

            links.resize(MAX_LEVEL, NONE_VALUE);

            // Sentinel head node at index 0.
            nodes.push(Node::new(0, 0, 0u64, 0, true, 0, MAX_LEVEL as u8, 0));

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

        /// Serial-number ("RFC 1982") comparison: returns true when `new` is the
        /// same as, or newer than, `existing` in a wrapping `u32` sequence space.
        ///
        /// Callers feed a global monotonically-increasing write sequence
        /// (truncated to `u32`) so that conflicting writes to one key resolve to
        /// the higher sequence. Wrapping comparison keeps that correct even when
        /// the truncated counter rolls over, as long as the two values being
        /// compared are within 2^31 of each other — always true here, because a
        /// node only carries a *recent* sequence (the memtable flushes long
        /// before 2 billion writes accumulate).
        #[inline]
        fn seq_is_newer_or_equal(new: u32, existing: u32) -> bool {
            new.wrapping_sub(existing) < 0x8000_0000
        }

        /// Get the maximum capacity (total nodes that can be stored)
        pub fn max_capacity(&self) -> usize {
            self.max_capacity
        }

        /// Live (non-tombstoned) entry count.
        pub fn number_of_live_nodes(&self) -> usize {
            self.no_live_nodes
        }

        #[allow(dead_code)]
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

        /// Returns the raw entry for a key as `(value, sequence, tombstone)`, or
        /// `None` if the key is absent from this skip list. Crucially this
        /// distinguishes a **tombstone** (present but deleted) from **absent**,
        /// which a caller searching layered storage needs so that a tombstone in
        /// a newer layer shadows a live value in an older one.
        pub fn entry(&self, key: &[u8]) -> Option<(u128, u32, bool)> {
            let (found, _) = self.find_path(key);
            let idx = found?;
            let n = &self.nodes[idx as usize];
            Some((n.value, n.sequence, n.tombstone))
        }

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
        /// dropped (returns `Ok(None)`) — this is what makes the in-memory winner
        /// for two racing writes to one key identical to the winner recovery
        /// would pick (it replays in sequence order), closing the live-vs-recovery
        /// divergence for concurrent same-key writes.
        pub fn try_insert_with_seq(&mut self, key: &[u8], value: u128, seq: u32) -> Result<Option<u128>, InsertError> {
            let (found, update) = self.find_path(key);

            if let Some(idx) = found {
                let node = &mut self.nodes[idx as usize];
                if !Self::seq_is_newer_or_equal(seq, node.sequence) {
                    // A newer write already won for this key; drop this stale one.
                    return Ok(None);
                }
                let old = node.value;
                let was_tombstone = node.tombstone;
                node.value = value;
                node.tombstone = false;
                node.sequence = seq;
                if was_tombstone {
                    self.no_tombstone_nodes = self.no_tombstone_nodes.saturating_sub(1);
                    self.no_live_nodes += 1;
                    return Ok(None);
                }
                return Ok(Some(old));
            }

            let node_level = self.random_level();
            let mut update = update;
            if node_level > self.level {
                #[allow(clippy::needless_range_loop)]
                for lvl in self.level as usize..node_level as usize {
                    update[lvl] = self.head;
                }
            }

            let new_idx = self
                .allocate_node(key, value, false, seq, node_level)
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
            self.no_live_nodes += 1;
            Ok(None)
        }

        /// Tombstone using an explicit sequence, with highest-sequence-wins
        /// resolution: a tombstone older than the node's current sequence is
        /// dropped (a newer write has superseded the delete). Returns
        /// `Some(old_value)` only if a live node was actually tombstoned.
        pub fn remove_with_seq(&mut self, key: &[u8], seq: u32) -> Option<u128> {
            let (found, _) = self.find_path(key);
            let idx = found?;
            let node = &mut self.nodes[idx as usize];
            if node.tombstone {
                return None;
            }
            if !Self::seq_is_newer_or_equal(seq, node.sequence) {
                return None;
            }
            let old = node.value;
            node.tombstone = true;
            node.sequence = seq;
            self.no_live_nodes -= 1;
            self.no_tombstone_nodes += 1;
            Some(old)
        }

        /// Physically purges a tombstoned key by unlinking it from the list.
        /// Returns `true` if a tombstoned key was found and purged.
        #[allow(dead_code)]
        pub fn purge_tombstone(&mut self, key: &[u8]) -> bool {
            let (found, update) = self.find_path(key);
            let idx = match found {
                Some(i) => i,
                None => return false,
            };

            if !self.nodes[idx as usize].tombstone {
                return false;
            }

            #[allow(clippy::needless_range_loop)]
            for lvl in 0..self.level as usize {
                let prev = update[lvl];
                if self.next(prev, lvl) != idx {
                    break;
                }
                let next = self.next(idx, lvl);
                self.set_next(prev, lvl, next);
            }

            // Reduce current max level if upper levels are empty.
            while self.level > 1 {
                let top = (self.level - 1) as usize;
                if self.next(self.head, top) != NONE_VALUE {
                    break;
                }
                self.level -= 1;
            }

            self.no_tombstone_nodes = self.no_tombstone_nodes.saturating_sub(1);
            true
        }

        #[allow(dead_code)]
        pub fn iter(&self) -> Iter<'_> {
            Iter {
                list: self,
                current: self.next(self.head, 0),
            }
        }

        /// Returns an iterator over live entries whose keys are `>= start`, in key order.
        ///
        /// Performs an O(log N) seek via `lower_bound` then walks level-0 links forward,
        /// skipping tombstones.  Much faster than `iter().skip_while(|(k, _)| k < start)`
        /// for large skip lists when the cursor is deep into the key space.
        #[allow(dead_code)]
        pub fn iter_from(&self, start: &[u8]) -> Iter<'_> {
            Iter {
                list: self,
                current: self.lower_bound(start),
            }
        }

        /// Returns an iterator over all live entries whose keys start with `prefix`.
        ///
        /// Efficient: seeks to the first key `>= prefix` then walks level-0 until
        /// the prefix no longer matches. Accepts both borrowed and owned prefixes.
        #[allow(dead_code)]
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

        fn allocate_node(&mut self, key: &[u8], value: u128, tombstone: bool, seq: u32, height: u8) -> Option<u32> {
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
        /// Intended for merge/compaction logic. Use `iter()` for normal live-entry access.
        #[allow(dead_code)]
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

        /// Returns `true` if the key exists in the skip list.
        ///
        /// Note: this returns `true` even if the key is tombstoned.
        #[allow(dead_code)]
        pub fn contains_key(&self, key: &[u8]) -> bool {
            let (found, _) = self.find_path(key);
            found.is_some()
        }

        /// Like [`contains_key`], but only returns `true` if the key exists and is live (not tombstoned).
        #[allow(dead_code)]
        pub fn contains_live_key(&self, key: &[u8]) -> bool {
            self.get_value(key).is_some()
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
                    seq: n.sequence,
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
    ///
    /// Used by merge/compaction logic. For normal use, prefer `iter()`.
    #[allow(dead_code)]
    pub struct IterRaw<'a> {
        list: &'a SkipList,
        current: u32,
    }

    impl<'a> Iterator for IterRaw<'a> {
        type Item = (&'a [u8], u128, u32, bool);

        fn next(&mut self) -> Option<Self::Item> {
            if self.current == NONE_VALUE {
                return None;
            }
            let idx = self.current;
            self.current = self.list.next(idx, 0);
            let n = &self.list.nodes[idx as usize];
            Some((self.list.key_slice(idx), n.value, n.sequence, n.tombstone))
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
    pub struct Iter<'a> {
        list: &'a SkipList,
        current: u32,
    }

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
    #[allow(dead_code)]
    pub struct PrefixIter<'a> {
        list: &'a SkipList,
        current: u32,
        prefix: Cow<'a, [u8]>,
    }

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
        fn next_test_seq() -> u32 {
            use std::sync::atomic::{AtomicU32, Ordering};
            static SEQ: AtomicU32 = AtomicU32::new(1);
            SEQ.fetch_add(1, Ordering::Relaxed)
        }

        pub fn insert(&mut self, key: &[u8], value: u128) -> bool {
            self.try_insert_with_seq(key, value, Self::next_test_seq()).is_ok()
        }

        pub fn try_insert(&mut self, key: &[u8], value: u128) -> Result<Option<u128>, InsertError> {
            self.try_insert_with_seq(key, value, Self::next_test_seq())
        }

        pub fn remove(&mut self, key: &[u8]) -> Option<u128> {
            self.remove_with_seq(key, Self::next_test_seq())
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::store::lsm::skip_list::skip_list::DEFAULT_MAX_CAPACITY;

        use super::{KeyValueRecord, SkipList};
        use std::collections::BTreeMap;

        #[test]
        fn try_insert_with_seq_keeps_highest_sequence() {
            // Two writes to one key: the higher sequence must win regardless of
            // the order they are applied in. This is what makes the in-memory
            // winner for racing same-key writes match recovery (which replays in
            // sequence order).
            let mut sl = SkipList::new();
            sl.try_insert_with_seq(b"k", 100, 5).unwrap();
            // A lower-sequence write applied *later* must be dropped.
            sl.try_insert_with_seq(b"k", 200, 3).unwrap();
            assert_eq!(sl.get_value(b"k"), Some(100), "older-sequence write must not overwrite");
            // A higher-sequence write wins.
            sl.try_insert_with_seq(b"k", 300, 9).unwrap();
            assert_eq!(sl.get_value(b"k"), Some(300));
        }

        #[test]
        fn remove_with_seq_respects_sequence() {
            let mut sl = SkipList::new();
            sl.try_insert_with_seq(b"k", 100, 5).unwrap();
            // A delete older than the live write is dropped (the write supersedes).
            sl.remove_with_seq(b"k", 3);
            assert_eq!(sl.get_value(b"k"), Some(100), "older-sequence delete must not tombstone");
            // A newer delete tombstones.
            sl.remove_with_seq(b"k", 7);
            assert_eq!(sl.get_value(b"k"), None);
            // A write newer than the tombstone resurrects (un-deletes) the key.
            sl.try_insert_with_seq(b"k", 400, 9).unwrap();
            assert_eq!(sl.get_value(b"k"), Some(400));
        }

        #[test]
        fn seq_comparison_handles_u32_wraparound() {
            // Serial-number comparison must stay correct when the truncated u32
            // sequence wraps, as long as the two values are within 2^31.
            let mut sl = SkipList::new();
            let big = u32::MAX - 2;
            sl.try_insert_with_seq(b"k", 1, big).unwrap();
            // `big + 3` wraps past 0 but is still the newer write.
            sl.try_insert_with_seq(b"k", 2, big.wrapping_add(3)).unwrap();
            assert_eq!(sl.get_value(b"k"), Some(2), "newer write across a wrap boundary must win");
        }

        #[test]
        fn entry_distinguishes_tombstone_from_absent() {
            let mut sl = SkipList::new();
            assert_eq!(sl.entry(b"k"), None, "absent key");
            sl.try_insert_with_seq(b"k", 7, 1).unwrap();
            assert_eq!(sl.entry(b"k"), Some((7, 1, false)));
            sl.remove_with_seq(b"k", 2);
            assert_eq!(sl.entry(b"k"), Some((7, 2, true)), "tombstone must be reported, not None");
        }

        #[test]
        fn basic_insert_get_remove() {
            let mut sl = SkipList::new();
            assert!(sl.is_empty());

            assert_eq!(sl.try_insert(b"a", 1).unwrap(), None);
            assert_eq!(sl.try_insert(b"b", 2).unwrap(), None);
            assert_eq!(sl.try_insert(b"a", 3).unwrap(), Some(1));

            assert_eq!(sl.get_value(b"a"), Some(3));
            assert_eq!(sl.get_value(b"b"), Some(2));
            assert_eq!(sl.get_value(b"c"), None);

            sl.validate_insert();

            assert_eq!(sl.remove(b"b"), Some(2));
            assert_eq!(sl.remove(b"b"), None);
            assert_eq!(sl.get_value(b"b"), None);

            sl.validate_insert();
            assert_eq!(sl.number_of_live_nodes(), 1);
        }

        #[test]
        fn iteration_is_sorted_and_skips_tombstones() {
            let mut sl = SkipList::new();
            assert!(sl.insert(b"b", 2));
            assert!(sl.insert(b"a", 1));
            assert!(sl.insert(b"c", 3));
            sl.remove(b"b");

            let items: Vec<(Vec<u8>, u128)> = sl.iter().map(|(k, v)| (k.to_vec(), v)).collect();
            assert_eq!(items, vec![(b"a".to_vec(), 1), (b"c".to_vec(), 3)]);
        }

        #[test]
        fn randomized_against_btreemap() {
            let max_capacity = DEFAULT_MAX_CAPACITY;
            let mut sl = SkipList::new();
            let mut model = BTreeMap::<Vec<u8>, u128>::new();

            // deterministic LCG for test-data generation
            let mut s: u64 = 0x1234_5678_9ABC_DEF0;
            let mut next = || {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                s
            };

            for i in 0..max_capacity {
                let r = next();
                let mut key = [0u8; 16];
                key[..8].copy_from_slice(&r.to_le_bytes());
                key[8..].copy_from_slice(&next().to_le_bytes());

                match r % 3 {
                    0 => {
                        let val = r as u128;
                        let a = sl.try_insert(&key, val).unwrap();
                        let b = model.insert(key.to_vec(), val);
                        assert_eq!(a, b);
                    }
                    1 => {
                        let a = sl.remove(&key);
                        let b = model.remove(key.as_slice());
                        assert_eq!(a, b);
                    }
                    _ => {
                        let a = sl.get_value(&key);
                        let b = model.get(key.as_slice()).copied();
                        assert_eq!(a, b);
                    }
                }

                if i % 257 == 0 {
                    sl.validate_insert();
                    let sl_items: Vec<(Vec<u8>, u128)> = sl.iter().map(|(k, v)| (k.to_vec(), v)).collect();
                    let model_items: Vec<(Vec<u8>, u128)> = model.iter().map(|(k, v)| (k.clone(), *v)).collect();
                    assert_eq!(sl_items, model_items);
                }
            }

            sl.validate_insert();
        }

        #[test]
        fn zero_value_is_valid() {
            let mut sl = SkipList::new();

            // value=0 is now a valid data value
            assert!(sl.insert(b"a", 0));
            assert_eq!(sl.get_value(b"a"), Some(0));

            // u128::MAX is also allowed; tombstones are explicit.
            assert!(sl.insert(b"b", u128::MAX));
            assert_eq!(sl.get_value(b"b"), Some(u128::MAX));

            assert!(sl.try_insert(b"c", 0).is_ok());
        }

        #[test]
        fn iter_raw_yields_all_entries_including_tombstones() {
            let mut sl = SkipList::new();
            assert!(sl.insert(b"a", 1));
            assert!(sl.insert(b"b", 2));
            assert!(sl.insert(b"c", 3));
            sl.remove(b"b");

            let raw: Vec<_> = sl.iter_raw().map(|(k, v, _s, t)| (k.to_vec(), v, t)).collect();
            assert_eq!(raw.len(), 3);

            // "a" — live
            assert_eq!(raw[0].0, b"a");
            assert_eq!(raw[0].1, 1);
            assert!(!raw[0].2);

            // "b" — tombstoned
            assert_eq!(raw[1].0, b"b");
            assert!(raw[1].2);

            // "c" — live
            assert_eq!(raw[2].0, b"c");
            assert_eq!(raw[2].1, 3);
            assert!(!raw[2].2);
        }

        #[test]
        fn prefix_iter_borrowed_prefix() {
            let mut sl = SkipList::new();
            assert!(sl.insert(b"foo:1", 1));
            assert!(sl.insert(b"foo:2", 2));
            assert!(sl.insert(b"bar:1", 3));
            // tombstone one matching key
            sl.remove(b"foo:2");

            let items: Vec<_> = sl.scan_prefix(b"foo:" as &[u8]).map(|(k, v)| (k.to_vec(), v)).collect();
            assert_eq!(items, vec![(b"foo:1".to_vec(), 1)]);
        }

        #[test]
        fn prefix_iter_owned_prefix() {
            let mut sl = SkipList::new();
            assert!(sl.insert(b"ns:a", 10));
            assert!(sl.insert(b"ns:b", 20));
            assert!(sl.insert(b"other", 99));

            let owned_prefix: Vec<u8> = b"ns:".to_vec();
            let items: Vec<_> = sl.scan_prefix(owned_prefix).map(|(k, v)| (k.to_vec(), v)).collect();
            assert_eq!(items, vec![(b"ns:a".to_vec(), 10), (b"ns:b".to_vec(), 20)]);
        }

        #[test]
        fn prefix_iter_empty_prefix_yields_all_live() {
            let mut sl = SkipList::new();
            assert!(sl.insert(b"a", 1));
            assert!(sl.insert(b"b", 2));
            sl.remove(b"b");

            let items: Vec<_> = sl.scan_prefix(b"" as &[u8]).map(|(k, _)| k.to_vec()).collect();
            assert_eq!(items, vec![b"a".to_vec()]);
        }

        #[test]
        fn prefix_iter_no_match_returns_empty() {
            let mut sl = SkipList::new();
            assert!(sl.insert(b"abc", 1));

            let items: Vec<_> = sl.scan_prefix(b"xyz" as &[u8]).collect();
            assert!(items.is_empty());
        }

        #[test]
        fn collect_entries_sorted_is_sorted_and_excludes_sentinel() {
            let mut sl = SkipList::new();
            assert!(sl.insert(b"b", 2));
            assert!(sl.insert(b"a", 1));
            sl.remove(b"b");

            let v = sl.collect_key_value_records();
            assert_eq!(v.len(), 2);

            // Sorted ascending by key.
            assert_eq!(v[0].key.as_slice(), b"a");
            assert!(!v[0].tombstone);
            assert_eq!(v[0].value, 1);

            assert_eq!(v[1].key.as_slice(), b"b");
            assert!(v[1].tombstone);

            // Ensure no empty key entry leaked.
            assert!(v.iter().all(|e| !e.key.is_empty()));

            // Touch KeyValueRecord so the import isn't unused if cfg differs.
            let _ = KeyValueRecord {
                key: vec![],
                value: 0,
                tombstone: true,
                seq: 0,
            };
        }
    }
}
