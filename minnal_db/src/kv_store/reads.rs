//! Reads: point gets, key iteration, batch reads and scans.

use super::*;
use std::ops::ControlFlow;

/// Keys per page of a whole-namespace pass ([`KVStore::for_each_key_page`]):
/// large enough that a page's fixed cost is small beside its reads, small enough
/// that a page's keys take little memory.
pub(crate) const KEY_PAGE: usize = 1024;

/// What a prefix-batch scan (`scan_prefixes_batch`) returns: every live entry
/// under the requested 4-byte prefixes, in key order, so grouped by prefix.
///
/// The keys sit in one buffer and the values in one buffer per value-log bucket,
/// not in two allocations per entry: a semantic search's first pass reads tens of
/// thousands of entries at once and frees them all when it is done.
#[derive(Debug, Default)]
pub struct PrefixScan {
    keys: PrefixScanKeys,
    /// One buffer per value-log bucket read, then one per value re-read after its
    /// segment was reclaimed mid-scan.
    values: Vec<Vec<u8>>,
    /// Per entry, in key order: its index in `keys`, its value's buffer in
    /// `values` and its range there.
    entries: Vec<(usize, usize, std::ops::Range<usize>)>,
}

impl PrefixScan {
    /// Pack owned `(key, value)` pairs, each key at least 4 bytes (its prefix id),
    /// into a scan in key order. For stores that hold their entries in memory,
    /// such as test doubles.
    pub fn from_pairs(pairs: impl IntoIterator<Item = KeyValue>) -> Self {
        let mut pairs: Vec<KeyValue> = pairs.into_iter().filter(|(k, _)| k.len() >= 4).collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        let mut keys = PrefixScanKeys::with_capacity(pairs.len(), pairs.iter().map(|(k, _)| k.len()).sum());
        let mut values = Vec::with_capacity(pairs.iter().map(|(_, v)| v.len()).sum());
        let mut entries = Vec::with_capacity(pairs.len());
        for (i, (key, value)) in pairs.iter().enumerate() {
            keys.push(key, 0);
            entries.push((i, 0, values.len()..values.len() + value.len()));
            values.extend_from_slice(value);
        }
        Self {
            keys,
            values: vec![values],
            entries,
        }
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no entry matched.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `(prefix_id, key, value)` for every entry, in key order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (u32, &[u8], &[u8])> + '_ {
        (0..self.entries.len()).map(|i| self.get(i))
    }

    /// Entry `i` in key order as `(prefix_id, key, value)`, for callers that
    /// visit entries by position (in parallel, say). Panics if `i >= len()`.
    pub fn get(&self, i: usize) -> (u32, &[u8], &[u8]) {
        let (key, buffer, value) = &self.entries[i];
        let key = self.keys.key(*key);
        (
            u32::from_be_bytes(key[..4].try_into().unwrap()),
            key,
            &self.values[*buffer][value.clone()],
        )
    }
}

/// A batch read's values (one per key) and the positions to re-resolve through
/// the single-key path because their segment vanished mid-batch.
type BatchReadWithRetries = (Vec<Option<Vec<u8>>>, Vec<usize>);

impl KVStore {
    /// Read attempts before falling back to a lock-serialised read. A read only loops
    /// when GC unlinks a segment in the exact window between resolving the pointer and
    /// reading it, so in practice it succeeds first time; the cap only bounds the
    /// pathological case of GC churning one bucket continuously.
    const MAX_READ_ATTEMPTS: usize = 8;

    /// Get a value by key.
    ///
    /// Two steps: ask the LSM where the value is, then read it there. GC can relocate
    /// a value between those steps — but it cannot make this read *wrong*. Segments
    /// are immutable and their ids are never reused, so a pointer GC has superseded
    /// still resolves to **this key's own bytes** (GC's copy holds the same value).
    /// The only thing
    /// that can go wrong is arriving after GC unlinked the segment, which fails
    /// loudly as [`crate::ValueLogError::SegmentMissing`] — never silently as another key's
    /// record. So the read simply re-resolves through the LSM (which by then holds the
    /// new pointer) and retries. No lock or sequence check is needed on the read.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let result = self.get_inner(key);
        if let Some(m) = self.metrics() {
            // Count the read; a hit is a successful lookup that found a live value.
            m.record_read(matches!(result, Ok(Some(_))));
        }
        result
    }

    fn get_inner(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let mut last_err: Option<KVError> = None;

        for _ in 0..Self::MAX_READ_ATTEMPTS {
            // A transient LSM error (an SSTable being swapped by compaction) is
            // retryable, not fatal.
            let offset_u128 = match self.lsm.get(key) {
                Ok(Some(o)) => o,
                Ok(None) => return Ok(None),
                Err(e) => {
                    last_err = Some(e.into());
                    continue;
                }
            };
            let Some(pointer) = decode_sharded_pointer(offset_u128) else {
                return Ok(None);
            };

            match self.value_log.read_value(pointer) {
                Ok(value) => return Ok(Some(value)),
                // GC reclaimed the segment after relocating this record; the LSM now
                // holds the new pointer. Re-resolve and read again. This is the ONLY
                // retryable read error.
                Err(e) if e.is_segment_missing() => {
                    last_err = Some(e.into());
                    continue;
                }
                // Corruption or an IO fault is real: surface it immediately instead of
                // spinning the retry ladder (which is for reclaimed segments) or letting
                // a caller mistake it for a missing key.
                Err(e) => return Err(e.into()),
            }
        }

        // Pathological GC churn. Holding the bucket write lock excludes GC from this
        // bucket entirely, so the pointer we resolve cannot be invalidated under us —
        // this guarantees forward progress and surfaces genuine corruption.
        let bucket = self.value_log.bucket_for_key(key);
        let _bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;
        match self.lsm.get(key)? {
            Some(offset_u128) => match decode_sharded_pointer(offset_u128) {
                Some(pointer) => Ok(Some(self.value_log.read_value(pointer)?)),
                None => Ok(None),
            },
            None => Ok(None),
        }
        .map_err(|e: KVError| last_err.unwrap_or(e))
    }

    pub fn keys(&self) -> Result<Vec<Vec<u8>>> {
        self.lsm.keys().map_err(KVError::from)
    }

    /// Calls `visit` with successive pages of at most [`KEY_PAGE`] live
    /// `(key, pointer, seq)` entries in `[start, end)`, in key order, until
    /// `visit` breaks or the range ends. Each page is read on its own, so a pass
    /// over a whole namespace holds one page of keys, not all of them.
    ///
    /// The pages are not one snapshot: a key written behind the cursor during
    /// the pass is not visited, and one written ahead of it is. The pointers are
    /// as current as the page; resolve them as for
    /// [`scan_prefix_batch`](Self::scan_prefix_batch).
    pub(crate) fn for_each_key_page(
        &self,
        start: &[u8],
        end: Option<&[u8]>,
        mut visit: impl FnMut(Vec<KeyPointer>) -> Result<ControlFlow<()>>,
    ) -> Result<()> {
        let mut from = start.to_vec();
        loop {
            let page = self.lsm.range_pointers_bounded(&from, end, KEY_PAGE)?;
            let last_page = page.len() < KEY_PAGE;
            // The next page starts just after this page's last key: the key
            // followed by a zero byte is the smallest key greater than it.
            if let Some((last, ..)) = page.last() {
                from.clone_from(last);
                from.push(0);
            }
            if visit(page)?.is_break() || last_page {
                return Ok(());
            }
        }
    }

    /// The number of live keys, counted a page at a time (`for_each_key_page`):
    /// every key is read, but only one page is held.
    pub fn count_keys(&self) -> Result<usize> {
        let mut count = 0;
        self.for_each_key_page(&[], None, |page| {
            count += page.len();
            Ok(ControlFlow::Continue(()))
        })?;
        Ok(count)
    }

    /// At most `limit` live keys in `[cursor, end)`, in key order, and the key the
    /// next page starts at (`None` after the last page). No value is read.
    pub fn key_page(&self, cursor: Option<&[u8]>, end: Option<&[u8]>, limit: usize) -> Result<KeyPage> {
        // `limit` 0 would return the cursor as its own next cursor, forever.
        let limit = limit.max(1);
        let mut keys: Vec<Vec<u8>> = self
            .lsm
            .range_pointers_bounded(cursor.unwrap_or(&[]), end, limit + 1)?
            .into_iter()
            .map(|(key, ..)| key)
            .collect();
        let next_cursor = if keys.len() > limit { keys.pop() } else { None };
        Ok((keys, next_cursor))
    }

    pub fn resolve_entries_from_pointers(&self, key_pointers: Vec<KeyPointer>) -> Result<Vec<ResolvedEntry>> {
        let mut entries = Vec::with_capacity(key_pointers.len());
        for (key, offset_u128, _seq) in key_pointers {
            if let Some(pointer) = decode_sharded_pointer(offset_u128) {
                entries.push((key, pointer));
            }
        }
        Ok(entries)
    }

    /// Fetch multiple keys in parallel by grouping I/O across value-log buckets.
    ///
    /// Uses `lsm.get_multiple` to resolve all key→pointer mappings in one forward
    /// pass per SSTable bucket, reading near the keys, then reads values from the
    /// value log with one thread per bucket.
    ///
    /// Returns one `Option<Vec<u8>>` per input key in the same order; `None` means
    /// the key is absent or deleted. A storage error (an LSM read failure,
    /// corruption, an IO fault) is an `Err` for the whole call, never `None`: a
    /// caller must be able to tell a failed read from a missing key.
    pub fn get_multiple(&self, keys: &[Vec<u8>]) -> Result<Vec<Option<Vec<u8>>>> {
        let (mut results, retry) = self.get_multiple_inner(keys)?;

        // Keys that resolved to a pointer but whose value could not be read: usually GC
        // reclaimed the segment between the LSM lookup and the read, and the LSM now
        // holds the new pointer, so re-resolving through the single-key `get` succeeds.
        // A hard error from `get` (corruption, IO fault) is the caller's to see.
        for idx in retry {
            results[idx] = self.get(&keys[idx])?;
        }
        Ok(results)
    }

    /// Returns `(values, retry_indices)` where `retry_indices` are positions that
    /// `lsm.get_multiple` resolved to a pointer but whose value read returned `None`
    /// (a transient GC race the caller should re-resolve via [`get`](Self::get)).
    fn get_multiple_inner(&self, keys: &[Vec<u8>]) -> Result<BatchReadWithRetries> {
        let mut results: Vec<Option<Vec<u8>>> = vec![None; keys.len()];

        // ── Step 1: single-pass LSM lookup for all keys ───────────────────────
        // One forward pass per bucket's SSTables, reading near the keys only.
        let pointers = self.lsm.get_multiple(keys)?;

        // ── Step 2: group the pointers by value-log bucket ────────────────────
        let num_buckets = self.value_log.num_buckets();
        let mut bucket_work: Vec<Vec<(usize, ValueLocation)>> = (0..num_buckets).map(|_| Vec::new()).collect();
        let mut resolved: Vec<usize> = Vec::new();
        for (orig_idx, pointer_opt) in pointers.into_iter().enumerate() {
            if let Some((offset_u128, _seq)) = pointer_opt
                && let Some(pointer) = decode_sharded_pointer(offset_u128)
                && (pointer.bucket as usize) < num_buckets
            {
                bucket_work[pointer.bucket as usize].push((orig_idx, pointer.location));
                resolved.push(orig_idx);
            }
        }
        if resolved.is_empty() {
            return Ok((results, Vec::new()));
        }

        // ── Step 3: read each bucket's values in parallel ─────────────────────
        // No sequence validation: a segment id is never reused, so a pointer cannot
        // resolve to another key's record. A read can only fail outright — its
        // segment was reclaimed by GC — and those keys are re-resolved below.
        std::thread::scope(|s| {
            let handles: Vec<_> = bucket_work
                .iter()
                .enumerate()
                .filter(|(_, work)| !work.is_empty())
                .map(|(bucket, work)| s.spawn(move || self.value_log.read_values_batch(bucket as u32, work)))
                .collect();
            for handle in handles {
                if let Ok(batch) = handle.join() {
                    for (idx, val) in batch {
                        results[idx] = val;
                    }
                }
            }
        });

        // Keys we resolved to a pointer but could not read: GC reclaimed the segment
        // between the LSM lookup and the read. The LSM now holds the new pointer, so
        // the caller re-resolves them through the single-key `get`.
        let retry: Vec<usize> = resolved.into_iter().filter(|&idx| results[idx].is_none()).collect();

        Ok((results, retry))
    }

    /// Scan multiple 4-byte BE cluster prefixes and resolve their values.
    ///
    /// Collects every live `(key, pointer)` for the whole prefix set in one
    /// `LSMTree::scan_prefixes` call (one pass per layer, not one per prefix), then
    /// resolves the pointers against the value log with one scoped thread per bucket,
    /// each reading its values into one buffer. Each value is its own `pread` (in
    /// key order, not file order); the reads are not coalesced, so the syscall count
    /// equals the number of entries. Keys whose segment GC reclaimed mid-scan are
    /// re-resolved through [`get`](Self::get).
    ///
    /// Keys shorter than 4 bytes are skipped.
    pub fn scan_prefixes_batch(&self, prefix_ids: &[u32]) -> Result<PrefixScan> {
        let prefix_id_set: std::collections::HashSet<u32> = prefix_ids.iter().copied().collect();
        let keys = self.lsm.scan_prefixes(&prefix_id_set)?;
        if keys.is_empty() {
            return Ok(PrefixScan::default());
        }

        let num_buckets = self.value_log.num_buckets();
        let mut bucket_work: Vec<Vec<(usize, ValueLocation)>> = (0..num_buckets).map(|_| Vec::new()).collect();
        for i in 0..keys.len() {
            if keys.key(i).len() < 4 {
                continue;
            }
            if let Some(pointer) = decode_sharded_pointer(keys.pointer(i)) {
                bucket_work[pointer.bucket as usize].push((i, pointer.location));
            }
        }

        let mut values: Vec<Vec<u8>> = Vec::new();
        // Each key's value as (buffer, range), placed by key index, so the result
        // comes out in key order without sorting.
        let mut slots: Vec<Option<(usize, std::ops::Range<usize>)>> = vec![None; keys.len()];
        let mut retry: Vec<usize> = Vec::new();
        std::thread::scope(|s| {
            let handles: Vec<_> = bucket_work
                .iter()
                .enumerate()
                .filter(|(_, work)| !work.is_empty())
                .map(|(bucket, work)| {
                    let handle = s.spawn(move || {
                        let mut buffer = Vec::new();
                        let ranges = self.value_log.read_values_batch_into(bucket as u32, work, &mut buffer);
                        (buffer, ranges)
                    });
                    (work, handle)
                })
                .collect();
            for (work, handle) in handles {
                match handle.join() {
                    Ok((buffer, ranges)) => {
                        for (idx, range) in ranges {
                            match range {
                                Some(range) => slots[idx] = Some((values.len(), range)),
                                None => retry.push(idx),
                            }
                        }
                        values.push(buffer);
                    }
                    Err(_) => retry.extend(work.iter().map(|(idx, _)| *idx)),
                }
            }
        });

        for idx in retry {
            if let Some(value) = self.get(keys.key(idx))? {
                slots[idx] = Some((values.len(), 0..value.len()));
                values.push(value);
            }
        }
        let entries = slots
            .into_iter()
            .enumerate()
            .filter_map(|(idx, slot)| slot.map(|(buffer, range)| (idx, buffer, range)))
            .collect();
        Ok(PrefixScan { keys, values, entries })
    }

    /// Parallel value-log reads for a pre-built entry list.
    ///
    /// Groups entries by value-log bucket and reads each bucket in a scoped thread.
    /// Returns `(pairs, retry_keys)`: `retry_keys` are keys whose pointer resolved but
    /// whose segment GC reclaimed before the read — the caller re-resolves them via
    /// [`get`](Self::get).
    fn batch_read_from_entries(&self, entries: Vec<ResolvedEntry>) -> Result<(Vec<KeyValue>, Vec<Vec<u8>>)> {
        if entries.is_empty() {
            return Ok((vec![], vec![]));
        }
        let n = entries.len();
        let num_buckets = self.value_log.num_buckets();
        let mut bucket_work: Vec<Vec<(usize, ValueLocation)>> = (0..num_buckets).map(|_| Vec::new()).collect();
        for (idx, (_, pointer)) in entries.iter().enumerate() {
            bucket_work[pointer.bucket as usize].push((idx, pointer.location));
        }

        let mut values: Vec<Option<Vec<u8>>> = vec![None; n];
        std::thread::scope(|s| {
            let handles: Vec<_> = bucket_work
                .iter()
                .enumerate()
                .filter(|(_, work)| !work.is_empty())
                .map(|(bucket, work)| s.spawn(move || self.value_log.read_values_batch(bucket as u32, work)))
                .collect();
            for handle in handles {
                if let Ok(results) = handle.join() {
                    for (idx, val) in results {
                        values[idx] = val;
                    }
                }
            }
        });

        let mut pairs = Vec::with_capacity(n);
        let mut retry = Vec::new();
        for ((key, _), val) in entries.into_iter().zip(values) {
            match val {
                Some(v) => pairs.push((key, v)),
                None => retry.push(key),
            }
        }
        Ok((pairs, retry))
    }

    /// Re-resolve keys whose segment GC reclaimed between the LSM scan and the
    /// value read, through the single-key [`get`](Self::get) (which retries resolve
    /// and read together, and falls back to a read under the bucket lock). A key
    /// that now reads as absent was deleted meanwhile and is left out.
    ///
    /// `pairs` is in key order and stays so: the re-read keys are put back in
    /// their places, not appended.
    pub(super) fn refetch_dropped(&self, pairs: &mut Vec<KeyValue>, retry: Vec<Vec<u8>>) -> Result<()> {
        if retry.is_empty() {
            return Ok(());
        }
        for key in retry {
            if let Some(v) = self.get(&key)? {
                pairs.push((key, v));
            }
        }
        pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        Ok(())
    }

    /// Scan keys with prefix and return `(key, value)` pairs using batched I/O.
    ///
    /// One LSM pass returns `(key, pointer)` pairs directly (no per-key
    /// `lsm.get()`), then values are read in parallel per bucket. The whole result
    /// is held in memory.
    ///
    /// A pointer can go stale between the LSM scan and the value read (GC relocates
    /// the value and unlinks its segment); such keys are re-read through
    /// [`get`](Self::get), so the result is complete.
    pub fn scan_prefix_batch(&self, prefix: &[u8]) -> Result<Vec<KeyValue>> {
        let key_pointers = self.lsm.scan_prefix(prefix)?;
        let entries = self.resolve_entries_from_pointers(key_pointers)?;
        let (mut pairs, retry) = self.batch_read_from_entries(entries)?;
        self.refetch_dropped(&mut pairs, retry)?;
        if let Some(m) = self.metrics() {
            m.record_scan(pairs.len() as u64);
        }
        Ok(pairs)
    }

    /// Scan keys in `[start, end)` and return `(key, value)` pairs using batched I/O.
    ///
    /// One LSM pass (`range_pointers_bounded`) gives each key's pointer, so no key
    /// is looked up twice. The whole result is held in memory.
    ///
    /// A pointer can go stale between the LSM scan and the value read (GC relocates
    /// the value and unlinks its segment); such keys are re-read through
    /// [`get`](Self::get), so the result is complete.
    pub fn scan_range_batch(&self, start: &[u8], end: Option<&[u8]>) -> Result<Vec<KeyValue>> {
        let key_pointers = self.lsm.range_pointers_bounded(start, end, usize::MAX)?;
        let entries = self.resolve_entries_from_pointers(key_pointers)?;
        let (mut pairs, retry) = self.batch_read_from_entries(entries)?;
        self.refetch_dropped(&mut pairs, retry)?;
        if let Some(m) = self.metrics() {
            m.record_scan(pairs.len() as u64);
        }
        Ok(pairs)
    }

    /// Paginated cursor scan over `[cursor, end)` returning at most `limit` pairs
    /// and an optional next cursor.
    ///
    /// Fetches `limit + 1` key-pointer pairs from the LSM in one pass, uses the
    /// extra entry solely to compute the next cursor, then batch-reads values for
    /// the page. `end` (exclusive) bounds the scan — pass `None` to scan to the
    /// last key; a prefix scan supplies the prefix's upper bound here so only the
    /// page's keys (not the whole tail of the keyspace) are resolved.
    ///
    /// A pointer can go stale between the LSM scan and the value read (GC relocates
    /// the value and unlinks its segment); such keys are re-read through
    /// [`get`](Self::get), so the result is complete.
    pub fn scan_page_batch(&self, cursor: Option<&[u8]>, end: Option<&[u8]>, limit: usize) -> Result<ScanPage> {
        // An empty page cannot carry a cursor that advances: with `limit = 0` the
        // next cursor would be the cursor itself, and a caller following
        // `next_cursor` would loop forever. Treat 0 as 1.
        let limit = limit.max(1);
        let start = cursor.unwrap_or(&[]);
        let key_pointers = self.lsm.range_pointers_bounded(start, end, limit + 1)?;
        let has_more = key_pointers.len() > limit;
        let next_cursor = if has_more { Some(key_pointers[limit].0.clone()) } else { None };
        let page_kps: Vec<_> = key_pointers.into_iter().take(limit).collect();
        let entries = self.resolve_entries_from_pointers(page_kps)?;
        let (mut pairs, retry) = self.batch_read_from_entries(entries)?;
        self.refetch_dropped(&mut pairs, retry)?;
        if let Some(m) = self.metrics() {
            m.record_scan(pairs.len() as u64);
        }
        Ok((pairs, next_cursor))
    }
}
