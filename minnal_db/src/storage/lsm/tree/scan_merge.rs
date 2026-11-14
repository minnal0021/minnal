//! How a scan merges its layers: Level-1 runs packed into one buffer and merged,
//! newer layers folded over them by highest seq.

use super::*;

/// Resolve a prefix scan's layers into live `(key, pointer)` pairs in ascending key order.
///
/// `l1` is the oldest layer, in the order its entries were read: bucket by bucket,
/// each bucket's run key-sorted, each key a range of its bucket's buffer in
/// `l1_keys`. `newer` holds every newer layer (L0, read-only and active memtables)
/// already folded by [`fold_newest`]. The result is what folding `l1` in order and
/// then `newer` into one map by [`fold_newest`] would yield, minus tombstones —
/// without paying a map insert, or an allocation, per L1 entry:
///
/// * [`merge_l1_runs`] merges the runs into key order, keeping the newest copy of
///   a key found in more than one run (the later run on a tie), as the fold would;
/// * [`resolve_scan_layers`] then resolves a key present in both by seq, `newer`
///   winning a tie.
///
/// `scan_prefixes` merges its runs on its workers and calls
/// [`resolve_scan_layers`] itself; this whole form is kept for the test that pins
/// the pair against the fold.
#[cfg(test)]
pub(super) fn merge_scan_layers(l1_keys: &[Vec<u8>], l1: Vec<PackedScanEntry>, newer: ScanMap) -> PrefixScanKeys {
    let l1 = merge_l1_runs(&l1, |e: &PackedScanEntry| &l1_keys[e.0][e.1..e.2]);
    resolve_scan_layers(l1_keys, l1, newer)
}

/// The second half of [`merge_scan_layers`], for `l1` already in key order with one
/// entry per key (as [`merge_l1_runs`] leaves it): resolve it against `newer` and
/// drop tombstones.
pub(super) fn resolve_scan_layers(l1_keys: &[Vec<u8>], l1: Vec<PackedScanEntry>, newer: ScanMap) -> PrefixScanKeys {
    let key = |e: &PackedScanEntry| &l1_keys[e.0][e.1..e.2];
    let key_bytes = l1.iter().map(|e| e.2 - e.1).sum::<usize>() + newer.keys().map(Vec::len).sum::<usize>();
    let mut out = PrefixScanKeys::with_capacity(l1.len() + newer.len(), key_bytes);
    let mut older = l1.into_iter().peekable();
    for (newer_key, (val, seq)) in newer {
        while let Some(e) = older.next_if(|e| key(e) < newer_key.as_slice()) {
            if let Some(v) = e.3 {
                out.push(key(&e), v);
            }
        }
        // Same key in both: the higher seq wins, `newer` on a tie.
        let winner = match older.next_if(|e| key(e) == newer_key.as_slice()) {
            Some(e) if seq_newer(e.4, seq) => e.3,
            _ => val,
        };
        if let Some(v) = winner {
            out.push(&newer_key, v);
        }
    }
    for e in older {
        if let Some(v) = e.3 {
            out.push(key(&e), v);
        }
    }
    out
}

/// Merge a scan's L1 entries into key order, one entry per key (the newest, as
/// `seq_newer_or_eq` decides; the later entry on a tie).
///
/// A scan's L1 entries arrive as one run per bucket, each in key order (an
/// SSTable is sorted). Runs are found where the order steps down, and merged
/// directly, taking the smallest head at each step. Sorting the whole set
/// instead was most of a large scan's time: 6–7 ms of a 160k-entry scan, moving
/// every entry through several merge passes. Heads are the keys' first 16 bytes
/// as integers (`key_head`), so the bytes are read only when two heads tie. Input
/// in more than `MAX_MERGED_RUNS` runs is stable-sorted instead, which bounds the
/// cost of an input far from sorted.
pub(super) fn merge_l1_runs<'a>(l1: &[PackedScanEntry], key: impl Fn(&PackedScanEntry) -> &'a [u8]) -> Vec<PackedScanEntry> {
    let heads: Vec<u128> = l1.iter().map(|e| key_head(key(e))).collect();
    let order = |i: usize, j: usize| heads[i].cmp(&heads[j]).then_with(|| key(&l1[i]).cmp(key(&l1[j])));
    let mut runs: Vec<std::ops::Range<usize>> = Vec::new();
    let mut start = 0;
    for i in 1..=l1.len() {
        if i == l1.len() || order(i - 1, i).is_gt() {
            runs.push(start..i);
            start = i;
        }
    }
    if runs.len() <= MAX_MERGED_RUNS {
        return merge_runs(l1, &runs, &order);
    }
    let mut sorted: Vec<usize> = (0..l1.len()).collect();
    sorted.sort_by(|&i, &j| order(i, j));
    merge_runs_in(l1, &sorted, &order)
}

/// Keep the newest of each key's entries, visiting `l1` in `sorted` order (key
/// order, input order among equal keys).
pub(super) fn merge_runs_in(l1: &[PackedScanEntry], sorted: &[usize], order: &impl Fn(usize, usize) -> std::cmp::Ordering) -> Vec<PackedScanEntry> {
    let mut out: Vec<PackedScanEntry> = Vec::with_capacity(sorted.len());
    let mut last: Option<usize> = None;
    for &i in sorted {
        if let Some(l) = last
            && order(l, i).is_eq()
        {
            let kept = out.last_mut().expect("`last` names the entry just pushed");
            if seq_newer_or_eq(l1[i].4, kept.4) {
                kept.3 = l1[i].3;
                kept.4 = l1[i].4;
            }
            continue;
        }
        out.push(l1[i]);
        last = Some(i);
    }
    out
}

/// Merge `runs` of `l1` (each in key order by `order`) into one, keeping the
/// newest copy of each key (the later run on a tie).
pub(super) fn merge_runs(
    l1: &[PackedScanEntry],
    runs: &[std::ops::Range<usize>],
    order: &impl Fn(usize, usize) -> std::cmp::Ordering,
) -> Vec<PackedScanEntry> {
    let mut next: Vec<usize> = runs.iter().map(|r| r.start).collect();
    let mut out: Vec<PackedScanEntry> = Vec::with_capacity(runs.iter().map(|r| r.len()).sum());
    let mut last: Option<usize> = None;
    loop {
        let mut best: Option<usize> = None;
        for (r, run) in runs.iter().enumerate() {
            if next[r] < run.end && best.is_none_or(|b| order(next[r], next[b]).is_lt()) {
                best = Some(r);
            }
        }
        let Some(b) = best else { break };
        let i = next[b];
        next[b] += 1;
        // A key in more than one run: keep whichever copy is newer.
        if let Some(l) = last
            && order(l, i).is_eq()
        {
            let kept = out.last_mut().expect("`last` names the entry just pushed");
            if seq_newer_or_eq(l1[i].4, kept.4) {
                kept.3 = l1[i].3;
                kept.4 = l1[i].4;
            }
            continue;
        }
        out.push(l1[i]);
        last = Some(i);
    }
    out
}

/// A key's first 16 bytes as a big-endian integer, zero-padded: comparing heads,
/// then the keys on a tie, is the keys' byte order.
pub(super) fn key_head(key: &[u8]) -> u128 {
    let mut head = [0u8; 16];
    let n = key.len().min(16);
    head[..n].copy_from_slice(&key[..n]);
    u128::from_be_bytes(head)
}

impl PrefixScanKeys {
    pub(crate) fn with_capacity(entries: usize, key_bytes: usize) -> Self {
        Self {
            keys: Vec::with_capacity(key_bytes),
            ends: Vec::with_capacity(entries),
            pointers: Vec::with_capacity(entries),
        }
    }

    pub(crate) fn push(&mut self, key: &[u8], pointer: u128) {
        self.keys.extend_from_slice(key);
        self.ends.push(self.keys.len());
        self.pointers.push(pointer);
    }

    /// The number of keys.
    pub(crate) fn len(&self) -> usize {
        self.ends.len()
    }

    /// Whether no key matched.
    pub(crate) fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// Key `i`.
    pub(crate) fn key(&self, i: usize) -> &[u8] {
        let start = if i == 0 { 0 } else { self.ends[i - 1] };
        &self.keys[start..self.ends[i]]
    }

    /// Key `i`'s value pointer.
    pub(crate) fn pointer(&self, i: usize) -> u128 {
        self.pointers[i]
    }

    /// The `(key, pointer)` pairs as owned values, for comparing in tests.
    #[cfg(test)]
    pub(crate) fn to_pairs(&self) -> Vec<(Vec<u8>, u128)> {
        (0..self.len()).map(|i| (self.key(i).to_vec(), self.pointer(i))).collect()
    }
}

/// Fold one layer's copy of `key` into a scan's running view, keeping the
/// highest `seq`. Callers fold layers **oldest first**, so `>=` lets an exact tie
/// go to the newer layer — the same rule `get_with_seq` applies.
///
/// Every multi-key read resolves through this, never by "the newer layer
/// overwrites". The writers keep newer versions in newer layers
/// (`docs/Layer-Order-Design.md`), but the scans do not rely on it; were a writer
/// to break it, overwriting by layer would hide a live key or surface a deleted
/// one (`test_scans_agree_with_brute_force_resolution`).
pub(super) fn fold_newest(map: &mut ScanMap, key: Vec<u8>, value: Option<u128>, seq: u64) {
    match map.entry(key) {
        std::collections::btree_map::Entry::Occupied(mut e) => {
            if seq_newer_or_eq(seq, e.get().1) {
                e.insert((value, seq));
            }
        }
        std::collections::btree_map::Entry::Vacant(e) => {
            e.insert((value, seq));
        }
    }
}
