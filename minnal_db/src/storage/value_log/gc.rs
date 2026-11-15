//! What GC needs from a bucket's log: which segments to collect, sealing the
//! tail, walking a segment's records, unlinking it; and the bucket's stats.

use super::*;

/// Default garbage share at which GC seals the active tail so it can be collected,
/// used when a caller does not supply one (tests, manual single-threshold GC). The
/// production path derives the bar from config instead
/// ([`ThresholdConfig::effective_tail_gc_min_garbage_pct`](crate::db::config::ThresholdConfig::effective_tail_gc_min_garbage_pct)),
/// which by default tracks the waste trigger so garbage never strands between the
/// two.
///
/// The tail is normally off-limits (it is still being appended to), which is fine while
/// it keeps filling: it seals on its own and becomes collectable. But a small or idle
/// namespace may never fill a segment, so garbage can pile up in the tail and be
/// unreclaimable *forever* — while the bucket-level waste trigger keeps firing, so GC
/// wakes every interval, finds no candidate, and reclaims nothing.
///
/// Sealing the tail early costs a rewrite of whatever is still live in it: at `G%`
/// garbage it rewrites `(100-G)%` to free `G%`, so `50` is the breakeven (rewrite ==
/// freed). A lower bar reclaims more of the tail sooner at the cost of more rewriting.
pub const TAIL_GC_MIN_GARBAGE_PCT: f64 = 50.0;

impl ValueLog {
    /// Is there anything for GC to actually do in this bucket?
    ///
    /// Cheap (in-memory counters) and exact enough to keep the GC worker from waking on a
    /// bucket whose garbage it cannot collect; without it a small, fully deleted
    /// namespace would log "starting GC ... reclaimed 0 bytes" on every tick.
    pub fn has_gc_work(&self, segment_threshold_pct: f64, tail_threshold_pct: f64) -> bool {
        !self.gc_candidates(segment_threshold_pct).is_empty() || self.tail_needs_sealing(tail_threshold_pct)
    }

    /// The bucket's overall waste ratio `garbage / (live + garbage)`, as a percentage —
    /// the per-bucket analogue of the namespace trigger, so a hot bucket can be spotted
    /// even when the namespace average is low.
    pub fn waste_ratio_pct(&self) -> f64 {
        let inner = self.inner.read();
        let garbage: u64 = inner.segments.values().map(|s| s.garbage_bytes).sum();
        let total: u64 = inner.segments.values().map(|s| s.total_bytes).sum();
        if total == 0 { 0.0 } else { (garbage as f64 / total as f64) * 100.0 }
    }

    /// True when the active tail holds at least `threshold_pct` garbage and should be
    /// sealed so GC can collect it (see [`TAIL_GC_MIN_GARBAGE_PCT`]).
    pub(super) fn tail_needs_sealing(&self, threshold_pct: f64) -> bool {
        let inner = self.inner.read();
        inner
            .segments
            .get(&inner.active_id)
            .is_some_and(|s| s.garbage_bytes > 0 && s.garbage_ratio_pct() >= threshold_pct)
    }

    /// Seal the active tail if it is mostly garbage, opening a fresh one in its place, and
    /// return the id it just sealed so GC can collect it in this same pass.
    ///
    /// The caller must hold the bucket write lock: this rolls the segment writers append
    /// to. A record appended just before the roll is simply live data in the sealed
    /// segment, and GC will relocate it like any other survivor.
    pub fn seal_tail_for_gc(&self, threshold_pct: f64) -> Result<Option<u32>> {
        if !self.tail_needs_sealing(threshold_pct) {
            return Ok(None);
        }
        let mut inner = self.inner.write();
        let sealed = inner.active_id;
        self.roll_active_segment(&mut inner)?;
        Ok(Some(sealed))
    }

    /// Sealed segments at or above `threshold_pct` garbage, worst first. The active tail is
    /// not a candidate here — it must be sealed first (see [`seal_tail_for_gc`](Self::seal_tail_for_gc)).
    pub fn gc_candidates(&self, threshold_pct: f64) -> Vec<SegmentStats> {
        let inner = self.inner.read();
        let mut candidates: Vec<SegmentStats> = inner
            .segments
            .values()
            .filter(|s| s.id != inner.active_id && s.sealed && s.total_bytes > 0)
            .filter(|s| s.garbage_ratio_pct() >= threshold_pct)
            .copied()
            .collect();
        candidates.sort_by(|a, b| b.garbage_ratio_pct().total_cmp(&a.garbage_ratio_pct()));
        candidates
    }

    /// Walk every record of a sealed segment in write order, handing the callback
    /// `(key, value, meta, location)`.
    ///
    /// This is how GC learns what a segment holds — the key is stored in the record
    /// precisely so a segment can be collected without inverting the entire LSM.
    pub fn for_each_record(&self, segment_id: u32, mut f: impl FnMut(&[u8], Vec<u8>, ValueRecordMeta, ValueLocation)) -> Result<()> {
        let file = self.segment_file(segment_id)?;
        let len = file.metadata()?.len();
        let mut offset = SEGMENT_HEADER_SIZE;

        while offset + ValueRecordHeader::SIZE as u64 <= len {
            let mut hdr_buf = [0u8; ValueRecordHeader::SIZE];
            match file.read_exact_at(&mut hdr_buf, offset) {
                Ok(()) => {}
                // A short read is a truncated tail — same "stop cleanly" case as a torn
                // header below. Nothing references a record past here.
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(ValueLogError::Io(e)),
            }
            let Some(header) = ValueRecordHeader::from_bytes(&hdr_buf) else {
                // A torn tail from a crash mid-append. Nothing references it — the LSM
                // only ever learned about records whose append returned — so stopping
                // here is complete.
                break;
            };
            if offset + header.total_len as u64 > len {
                break;
            }
            let body_len = header.total_len as usize - ValueRecordHeader::SIZE;
            let mut body = vec![0u8; body_len];
            match file.read_exact_at(&mut body, offset + ValueRecordHeader::SIZE as u64) {
                Ok(()) => {}
                // The header was whole but its body is truncated: still a torn tail, stop.
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(ValueLogError::Io(e)),
            }
            let (value, key) = body.split_at(header.value_len as usize);

            f(
                key,
                value.to_vec(),
                header.meta(),
                ValueLocation {
                    segment_id,
                    rec_offset: offset as u32,
                    value_len: header.value_len,
                },
            );
            offset += header.total_len as u64;
        }
        Ok(())
    }

    /// Unlink a segment and drop it from the inventory. Returns its size.
    ///
    /// **Only call this once the LSM re-point of its survivors is durable** (flushed
    /// to L0). Until then the durable LSM still points into this segment and the WAL
    /// entries that could replay those writes are long gone — unlinking early is data
    /// loss. Readers holding an open handle keep reading safely (the fd outlives the
    /// unlink); readers that open it afterwards get `SegmentMissing` and re-resolve.
    pub fn unlink_segment(&self, segment_id: u32) -> Result<u64> {
        let mut inner = self.inner.write();
        if segment_id == inner.active_id {
            return Ok(0);
        }
        let reclaimed = inner.segments.remove(&segment_id).map(|s| s.total_bytes).unwrap_or(0);
        drop(inner);

        self.fd_cache.lock().remove(segment_id);
        match std::fs::remove_file(Self::segment_path(&self.dir, self.bucket, segment_id)) {
            Ok(()) => {}
            // Already gone (a previous run unlinked it but crashed before persisting
            // metadata) — the desired end state either way.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(ValueLogError::Io(e)),
        }
        fsync_dir(&self.dir)?;
        Ok(reclaimed)
    }

    pub fn record_gc_run(&self, bytes_reclaimed: u64) {
        let mut inner = self.inner.write();
        inner.total_gc_runs += 1;
        inner.total_bytes_reclaimed += bytes_reclaimed;
    }

    // ── Stats ─────────────────────────────────────────────────────────────

    pub fn segment_stats(&self) -> Vec<SegmentStats> {
        self.inner.read().segments.values().copied().collect()
    }

    pub fn metadata_snapshot(&self) -> ValueLogMetadata {
        let inner = self.inner.read();
        ValueLogMetadata {
            next_segment_id: self.next_segment_id.load(Ordering::Acquire),
            active_segment_id: inner.active_id,
            segments: inner.segments.values().copied().collect(),
            total_gc_runs: inner.total_gc_runs,
            total_bytes_reclaimed: inner.total_bytes_reclaimed,
        }
    }

    /// Replace live/garbage accounting with an exact recomputation from the LSM's
    /// pointers: `live[segment] = bytes still referenced`. Everything else in a
    /// segment is garbage by definition. Used when the metadata file was lost.
    pub fn rebuild_stats(&self, live: &HashMap<u32, u64>) {
        {
            let mut inner = self.inner.write();
            for stats in inner.segments.values_mut() {
                let live_bytes = live.get(&stats.id).copied().unwrap_or(0).min(stats.total_bytes);
                stats.live_bytes = live_bytes;
                stats.garbage_bytes = stats.total_bytes - live_bytes;
            }
        }
        self.stats_need_rebuild.store(false, Ordering::Release);
    }

    pub fn flush_metadata(&self) -> Result<()> {
        let bytes = self.metadata_snapshot().to_file_bytes()?;
        crate::support::write_atomic_durable(&Self::metadata_path(&self.dir, self.bucket), &bytes)?;
        Ok(())
    }

    /// Physical bytes this bucket occupies on disk: the sum of its segment files.
    /// Segments are dense, appended files, so this is exact.
    pub fn disk_bytes(&self) -> u64 {
        self.inner
            .read()
            .segments
            .keys()
            .filter_map(|&id| std::fs::metadata(Self::segment_path(&self.dir, self.bucket, id)).ok())
            .map(|m| m.len())
            .sum()
    }
}
