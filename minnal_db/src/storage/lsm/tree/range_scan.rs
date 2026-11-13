//! The bounded range scan: a page of live `(key, pointer, seq)` entries from a
//! start key, resolved across every layer of the tree.
//!
//! Every layer is read in key order from `start`: the active memtable, each
//! read-only memtable, each Level-0 file (seeking through its sparse index) and
//! each bucket's Level-1 file. A page reads at most a *window* of entries from
//! each layer, not every entry after `start`. Keys up to the smallest last key of
//! any layer the window cut short are present in full in every layer, so they
//! resolve exactly ([`fold_newest`]). If those keys hold fewer than `limit` live
//! keys (newer layers' tombstones deleted the rest), the page is read again with
//! a window four times larger. A page's work and memory are therefore about its
//! size times the number of layers, whatever lies after the cursor.
//!
//! An example, with a page of 2 keys (so a window of 3) starting from `b`:
//!
//! ```text
//! memtable  b  d(tombstone)  e  h        read: b d e   cut, last key e
//! L1        a  c  d  f  g                read: c d f   cut, last key f
//! ```
//!
//! The smallest last key of a cut layer is `e`, so every key up to `e` has been
//! read from both layers. Up to `e` they resolve to `b`, `c`, `d` (deleted: the
//! memtable's tombstone is newer than L1's copy) and `e`; the live keys are
//! `b c e`, and the page is `b c`. `f` is not used yet: the memtable may hold a
//! tombstone for `f` beyond `h`, which this read did not reach.
//!
//! A tombstone in a newer layer deletes a key in an older one, so a page cannot
//! simply take the first keys of each layer and merge them; a key is resolved
//! only once every layer has been read past it.

use super::*;

/// Windows above this read each bucket's Level-0 and Level-1 files on a thread
/// of their own; smaller ones read them in turn, since a thread per bucket costs
/// more than a small read.
const PARALLEL_WINDOW: usize = 10_000;

/// One layer's entries from `start`, in key order: at most a window of them.
struct LayerRun {
    entries: Vec<ScanEntry>,
    /// The window ended the read: this layer may hold keys after its last entry.
    cut: bool,
}

impl LayerRun {
    const EMPTY: LayerRun = LayerRun {
        entries: Vec::new(),
        cut: false,
    };

    /// The first `window` entries of a layer already in key order. A layer with
    /// exactly `window` entries counts as cut: the read cannot tell.
    fn take(entries: impl Iterator<Item = ScanEntry>, window: usize) -> Self {
        let entries: Vec<ScanEntry> = entries.take(window).collect();
        let cut = entries.len() == window;
        Self { entries, cut }
    }

    /// The last key read, when the window cut this layer short.
    fn cut_at(&self) -> Option<&[u8]> {
        if self.cut { self.entries.last().map(|e| e.0.as_slice()) } else { None }
    }
}

#[cfg(test)]
thread_local! {
    /// Entries the range scans on this thread have read from every layer, so a
    /// test can check a page reads about its own size.
    pub(super) static ENTRIES_READ: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl LSMTree {
    /// The first `limit` live `(key, pointer, seq)` entries in `[start, end)`, in
    /// key order. Each key resolves to its highest-seq copy across the layers,
    /// so a tombstone in a newer layer hides an older live copy.
    ///
    /// Its pointers can go stale as [`scan_prefix`](Self::scan_prefix)'s can.
    pub(crate) fn range_pointers_bounded(&self, start: &[u8], end: Option<&[u8]>, limit: usize) -> Result<Vec<(Vec<u8>, u128, u64)>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut window = limit.saturating_add(1);
        loop {
            let layers = self.read_range_window(start, end, window)?;
            // Keys at or below this are read in full from every layer.
            let complete_to: Option<Vec<u8>> = layers.iter().filter_map(LayerRun::cut_at).min().map(<[u8]>::to_vec);
            let mut resolved = ScanMap::new();
            for run in layers {
                for (key, value, seq) in run.entries {
                    if complete_to.as_deref().is_some_and(|c| key.as_slice() > c) {
                        break;
                    }
                    fold_newest(&mut resolved, key, value, seq);
                }
            }
            let page: Vec<(Vec<u8>, u128, u64)> = resolved
                .into_iter()
                .filter_map(|(key, (value, seq))| value.map(|v| (key, v, seq)))
                .take(limit)
                .collect();
            if page.len() == limit || complete_to.is_none() {
                return Ok(page);
            }
            window = window.saturating_mul(4);
        }
    }

    /// Up to `window` entries in `[start, end)` from every layer, oldest layer
    /// first, so that folding them in order lets an exact seq tie go to the
    /// newer layer: each bucket's Level 1, each bucket's Level-0 files oldest
    /// first, the read-only memtables oldest first, the active memtable.
    ///
    /// The memtables are captured first, under one guard, so a flush cannot
    /// move a key from the active memtable to the read-only list unseen; then
    /// the SSTables, newest first (L0 before L1), so a concurrent flush or merge
    /// cannot drop a live key between them. The L0 guards keep their files
    /// readable until the reads are done.
    fn read_range_window(&self, start: &[u8], end: Option<&[u8]>, window: usize) -> Result<Vec<LayerRun>> {
        let in_range = |key: &[u8]| end.is_none_or(|e| key < e);
        let (active, read_only) = {
            let memtable = self.memtable.read();
            let active = LayerRun::take(
                memtable
                    .skip_list
                    .iter_raw_from(start)
                    .take_while(|(key, ..)| in_range(key))
                    .map(|(key, value, seq, tombstone)| (key.to_vec(), (!tombstone).then_some(value), seq)),
                window,
            );
            (active, self.read_only_memtables.read().clone())
        };
        let (l0_guards, l1_files) = self.capture_sstable_layers();

        let read_bucket = |bucket: usize| -> Result<(LayerRun, Vec<LayerRun>)> {
            let l1 = self.read_l1_window(bucket, &l1_files[bucket], start, end, window)?;
            let l0 = l0_guards[bucket]
                .iter()
                .map(|guard| read_l0_window(&guard.entry, start, end, window))
                .collect::<Result<Vec<_>>>()?;
            Ok((l1, l0))
        };
        let buckets: Vec<(LayerRun, Vec<LayerRun>)> = if window > PARALLEL_WINDOW {
            std::thread::scope(|s| {
                let handles: Vec<_> = (0..self.config.num_buckets).map(|b| s.spawn(move || read_bucket(b))).collect();
                handles
                    .into_iter()
                    .map(|h| {
                        h.join()
                            .unwrap_or_else(|_| Err(LSMError::Io(std::io::Error::other("range scan thread panicked"))))
                    })
                    .collect::<Result<Vec<_>>>()
            })?
        } else {
            (0..self.config.num_buckets).map(read_bucket).collect::<Result<Vec<_>>>()?
        };

        let mut layers = Vec::with_capacity(buckets.len() * 2 + read_only.len() + 1);
        let mut level0 = Vec::new();
        for (l1, l0) in buckets {
            layers.push(l1);
            level0.extend(l0);
        }
        layers.extend(level0);
        for ro in read_only.iter() {
            let records = ro.records();
            let from = records.partition_point(|r| r.key.as_slice() < start);
            layers.push(LayerRun::take(
                records[from..]
                    .iter()
                    .take_while(|r| in_range(&r.key))
                    .map(|r| (r.key.clone(), (!r.tombstone).then_some(r.value), r.seq)),
                window,
            ));
        }
        layers.push(active);

        #[cfg(test)]
        ENTRIES_READ.with(|n| n.set(n.get() + layers.iter().map(|l| l.entries.len()).sum::<usize>()));
        Ok(layers)
    }

    /// Up to `window` entries of `bucket`'s Level-1 file in `[start, end)`,
    /// seeking near `start` through its sparse index.
    fn read_l1_window(&self, bucket: usize, file: &File, start: &[u8], end: Option<&[u8]>, window: usize) -> Result<LayerRun> {
        if !self.l1_overlaps(bucket, start, end) {
            return Ok(LayerRun::EMPTY);
        }
        let mut entries = Vec::new();
        let mut cut = false;
        // `max_live` stops after `window` live entries; tombstones (which Level 1
        // does not keep) would not count towards it.
        self.for_each_l1_entry_in_range(bucket, file, start, end, Some(window), |archived| {
            let value = (!archived.tombstone).then(|| archived.value.to_native());
            entries.push((archived.key.as_slice().to_vec(), value, archived.seq.to_native()));
        })?;
        if entries.iter().filter(|e| e.1.is_some()).count() >= window {
            cut = true;
        }
        Ok(LayerRun { entries, cut })
    }
}

/// Up to `window` entries of one Level-0 file in `[start, end)`. A file whose
/// key range misses the window is not opened; one with a sparse index is read
/// from the block holding `start`.
fn read_l0_window(entry: &L0FileEntry, start: &[u8], end: Option<&[u8]>, window: usize) -> Result<LayerRun> {
    if let Some(summary) = &entry.summary
        && (summary.max_key.as_slice() < start || end.is_some_and(|e| summary.min_key.as_slice() >= e))
    {
        return Ok(LayerRun::EMPTY);
    }
    let file = open_registered_l0(&entry.path)?;
    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
    let hint = entry.scan_start(start);
    let offset = if hint != 0 && LSMTree::valid_scan_start(&file, hint, file_len, start) {
        hint
    } else {
        SSTABLE_DATA_START
    };
    let mut frames = FrameReader::new(&file, offset);
    let mut entries = Vec::new();
    while let Some(body) = frames.next_frame(file_len)? {
        // SAFETY: `verify_sstable_payload` checked the frame's CRC, and a
        // registered L0 file's header was validated when it entered the registry.
        let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(body)?) };
        let key = archived.key.as_slice();
        if end.is_some_and(|e| key >= e) {
            break;
        }
        if key < start {
            continue;
        }
        entries.push((
            key.to_vec(),
            (!archived.tombstone).then(|| archived.value.to_native()),
            archived.seq.to_native(),
        ));
        if entries.len() == window {
            return Ok(LayerRun { entries, cut: true });
        }
    }
    Ok(LayerRun { entries, cut: false })
}
