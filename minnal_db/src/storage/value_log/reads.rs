//! Reading records: one value, its metadata, or a batch of values with
//! nearby records merged into one `pread`.

use super::*;

/// `(original_index, Some(value))` for a readable slot, or `(index, None)` when the
/// pointer could not be resolved (segment reclaimed, or the record failed
/// validation) — the caller re-resolves those through the LSM.
pub(crate) type BatchValue = (usize, Option<Vec<u8>>);

/// `read_exact_at`, mapping a short read (EOF before the buffer is filled) to
/// [`ValueLogError::CorruptedLog`]. `FileExt::read_at` can return fewer bytes than
/// requested and leave the tail of the buffer untouched; every caller here needs the
/// *exact* bytes of a record the pointer says is present, so a short read means the
/// record is truncated — corruption, not a benign torn tail (that case is handled
/// inline by the record scanners, which stop rather than error).
pub(super) fn read_exact_or_corrupt(file: &File, buf: &mut [u8], offset: u64) -> Result<()> {
    file.read_exact_at(buf, offset).map_err(|e| match e.kind() {
        std::io::ErrorKind::UnexpectedEof => ValueLogError::CorruptedLog,
        _ => ValueLogError::Io(e),
    })
}

/// One segment's share of a batch read: its records in offset order, and the
/// runs read with one `pread` each, as `(first record, past the last, end byte)`.
pub(super) type PlannedSegment = (u32, Vec<(usize, ValueLocation)>, Vec<(usize, usize, u64)>);

/// Largest gap a batch read spans to read two records with one `pread`. The
/// gap holds the earlier record's key (about 12 bytes for a vector entry) and
/// whatever records lie between them.
pub(super) const MERGED_READ_MAX_GAP: u64 = 512;

/// Where the bytes a read of the record at `loc` needs end: its header and value
/// (the key that follows is not read).
pub(super) fn record_end(loc: ValueLocation) -> u64 {
    loc.rec_offset as u64 + ValueRecordHeader::SIZE as u64 + loc.value_len as u64
}

impl ValueLog {
    pub(super) fn parse_record(&self, buf: &[u8], location: ValueLocation) -> Result<(Vec<u8>, ValueRecordMeta)> {
        let (value, meta) = self.check_record(buf, location)?;
        Ok((buf[value].to_vec(), meta))
    }

    /// Check a record read into `buf` (header first) against the pointer that named
    /// it, and return where its value lies in `buf`.
    pub(super) fn check_record(&self, buf: &[u8], location: ValueLocation) -> Result<(std::ops::Range<usize>, ValueRecordMeta)> {
        let header = ValueRecordHeader::from_bytes(buf).ok_or(ValueLogError::CorruptedLog)?;
        // The pointer's value_len decides how much we read; a record that disagrees
        // is not the record this pointer names.
        if header.value_len != location.value_len {
            return Err(ValueLogError::InvalidLocation);
        }
        let start = ValueRecordHeader::SIZE;
        let end = start + header.value_len as usize;
        if buf.len() < end {
            return Err(ValueLogError::CorruptedLog);
        }
        if self.verify_checksums_on_read() && value_checksum(&buf[start..end]) != header.checksum {
            return Err(ValueLogError::CorruptedLog);
        }
        Ok((start..end, header.meta()))
    }

    /// Read a value: **one `pread`** of exactly `header + value_len` bytes.
    pub fn read_value(&self, location: ValueLocation) -> Result<Vec<u8>> {
        Ok(self.read_value_and_meta(location)?.0)
    }

    pub fn read_value_and_meta(&self, location: ValueLocation) -> Result<(Vec<u8>, ValueRecordMeta)> {
        if location.segment_id == 0 {
            return Err(ValueLogError::InvalidLocation);
        }
        let file = self.segment_file(location.segment_id)?;
        self.read_value_from_file(&file, location)
    }

    pub fn read_value_from_file(&self, file: &File, location: ValueLocation) -> Result<(Vec<u8>, ValueRecordMeta)> {
        if location.segment_id == 0 {
            return Err(ValueLogError::InvalidLocation);
        }
        let mut buf = vec![0u8; ValueRecordHeader::SIZE + location.value_len as usize];
        read_exact_or_corrupt(file, &mut buf, location.rec_offset as u64)?;
        self.parse_record(&buf, location)
    }

    /// The record's metadata only — a 32-byte read. TTL expiry uses it for `epoch`.
    pub fn read_record_meta(&self, location: ValueLocation) -> Result<ValueRecordMeta> {
        if location.segment_id == 0 {
            return Err(ValueLogError::InvalidLocation);
        }
        let file = self.segment_file(location.segment_id)?;
        let mut buf = [0u8; ValueRecordHeader::SIZE];
        read_exact_or_corrupt(&file, &mut buf, location.rec_offset as u64)?;
        let header = ValueRecordHeader::from_bytes(&buf).ok_or(ValueLogError::CorruptedLog)?;
        Ok(header.meta())
    }

    /// Read many of this bucket's values. Grouped by segment so each file is opened
    /// once; one `pread` per value.
    pub fn read_values_batch(&self, entries: &[(usize, ValueLocation)]) -> Vec<BatchValue> {
        let mut out = Vec::with_capacity(entries.len());
        let mut by_segment: HashMap<u32, Vec<(usize, ValueLocation)>> = HashMap::new();
        for &(idx, loc) in entries {
            if loc.segment_id == 0 {
                out.push((idx, None));
                continue;
            }
            by_segment.entry(loc.segment_id).or_default().push((idx, loc));
        }
        for (segment_id, group) in by_segment {
            match self.segment_file(segment_id) {
                Ok(file) => {
                    for (idx, loc) in group {
                        match self.read_value_from_file(&file, loc) {
                            Ok((value, _)) => out.push((idx, Some(value))),
                            Err(_) => out.push((idx, None)),
                        }
                    }
                }
                // Reclaimed mid-batch: the caller re-resolves these keys.
                Err(_) => out.extend(group.into_iter().map(|(idx, _)| (idx, None))),
            }
        }
        out
    }

    /// [`read_values_batch`](Self::read_values_batch), reading every value into one
    /// buffer, `out`, instead of allocating each: returns each entry's value as a
    /// range of `out` (`None` where the read failed). Each record's header stays in
    /// `out` in front of its value. A large prefix scan reads tens of thousands of
    /// small values, so this saves an allocation, a copy and a free per value.
    pub fn read_values_batch_into(&self, entries: &[(usize, ValueLocation)], out: &mut Vec<u8>) -> Vec<(usize, Option<std::ops::Range<usize>>)> {
        let mut ranges = Vec::with_capacity(entries.len());
        let mut by_segment: HashMap<u32, Vec<(usize, ValueLocation)>> = HashMap::new();
        for &(idx, loc) in entries {
            if loc.segment_id == 0 {
                ranges.push((idx, None));
                continue;
            }
            by_segment.entry(loc.segment_id).or_default().push((idx, loc));
        }
        // In file order, records that sit close together are read with one
        // `pread` per run instead of one each: a posting's values are usually
        // adjacent, since a split or rebuild copies them together. The runs are
        // planned before anything is read so `out` can be sized once: a run
        // also reads the gaps between its records, so the records' own sizes
        // undercount it, and growing a buffer of tens of megabytes copies all of
        // it (measured at 12% of a 4-bit first-pass semantic search).
        let mut planned: Vec<PlannedSegment> = Vec::with_capacity(by_segment.len());
        let mut total = 0usize;
        for (segment_id, mut group) in by_segment {
            group.sort_unstable_by_key(|(_, loc)| loc.rec_offset);
            let mut runs = Vec::new();
            let mut run_start = 0;
            while run_start < group.len() {
                let mut run_end = run_start + 1;
                let mut end = record_end(group[run_start].1);
                while run_end < group.len() {
                    let next = group[run_end].1.rec_offset as u64;
                    if next < end || next - end > MERGED_READ_MAX_GAP {
                        break;
                    }
                    end = record_end(group[run_end].1);
                    run_end += 1;
                }
                total += (end - group[run_start].1.rec_offset as u64) as usize;
                runs.push((run_start, run_end, end));
                run_start = run_end;
            }
            planned.push((segment_id, group, runs));
        }
        out.reserve(total);
        for (segment_id, group, runs) in planned {
            let file = match self.segment_file(segment_id) {
                Ok(file) => file,
                // Reclaimed mid-batch: the caller re-resolves these keys.
                Err(_) => {
                    ranges.extend(group.into_iter().map(|(idx, _)| (idx, None)));
                    continue;
                }
            };
            for (run_start, run_end, end) in runs {
                let run = &group[run_start..run_end];
                if run.len() == 1 || !self.read_run_into(&file, run, end, out, &mut ranges) {
                    for &(idx, loc) in run {
                        self.read_record_into(&file, idx, loc, out, &mut ranges);
                    }
                }
            }
        }
        ranges
    }

    /// Read `run` (records of one segment in offset order, ending at byte `end`)
    /// with one `pread` into `out` and push each record's value range. Returns
    /// `false`, with `out` and `ranges` unchanged, if the read fails, so the
    /// caller can fall back to reading each record alone.
    pub(super) fn read_run_into(
        &self,
        file: &File,
        run: &[(usize, ValueLocation)],
        end: u64,
        out: &mut Vec<u8>,
        ranges: &mut Vec<(usize, Option<std::ops::Range<usize>>)>,
    ) -> bool {
        let first = run[0].1.rec_offset as u64;
        let base = out.len();
        out.resize(base + (end - first) as usize, 0);
        if read_exact_or_corrupt(file, &mut out[base..], first).is_err() {
            out.truncate(base);
            return false;
        }
        for &(idx, loc) in run {
            let start = base + (loc.rec_offset as u64 - first) as usize;
            let len = ValueRecordHeader::SIZE + loc.value_len as usize;
            match self.check_record(&out[start..start + len], loc) {
                Ok((value, _)) => ranges.push((idx, Some(start + value.start..start + value.end))),
                Err(_) => ranges.push((idx, None)),
            }
        }
        true
    }

    /// Read one record into `out` and push its value range (`None` on failure).
    pub(super) fn read_record_into(
        &self,
        file: &File,
        idx: usize,
        loc: ValueLocation,
        out: &mut Vec<u8>,
        ranges: &mut Vec<(usize, Option<std::ops::Range<usize>>)>,
    ) {
        let start = out.len();
        out.resize(start + ValueRecordHeader::SIZE + loc.value_len as usize, 0);
        let read = read_exact_or_corrupt(file, &mut out[start..], loc.rec_offset as u64).and_then(|()| self.check_record(&out[start..], loc));
        match read {
            Ok((value, _)) => ranges.push((idx, Some(start + value.start..start + value.end))),
            Err(_) => {
                out.truncate(start);
                ranges.push((idx, None));
            }
        }
    }
}
