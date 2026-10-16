//! The vector index WAL: a log of the vector worker's payload writes.
//!
//! The vector worker writes chunk codes (`{ns}_sparse_vector`) and whole-document
//! codes (`{ns}_dense_vector`) without the main WAL, and must make them durable
//! before it completes a queue entry. This log is how: each such write is
//! appended here as it is applied, and one [`sync`](VectorIndexWal::sync) per
//! completion batch makes the batch durable — one fsync of one sequential file
//! instead of flushing three memtables. After a crash, the records not yet
//! retired are replayed with their original seqs. Design:
//! `semantic_search/report/Vector-Index-WAL-Design.md`.
//!
//! This module is the log only: segments, record format, append, `sync` and
//! reading. Retirement and GC, recovery and the worker's use of it are built on
//! top (steps 2–4 of the design's plan).
//!
//! # Format
//!
//! One log per database, in `{db_path}/vector_index_wal/`, as segment files
//! `{segment_id:010}.log` with contiguous ids. Each segment is:
//!
//! ```text
//! [header 16B: magic "MVIW" | format_version u32 | segment_id u64]
//! [frame] [frame] ... [end marker: u32 0, sealed segments only]
//! frame: [len u32][crc32(body) u32][body]
//! body:  [ns_id u32][seq u64][key_len u32][key][value_len u32][value]
//! ```
//!
//! All integers are little-endian. Every segment but the last is **sealed**: it
//! ends with the end marker, and space for the marker is reserved, so a record
//! never fills a segment's last four bytes.
//!
//! # Positions
//!
//! A position is `(segment_id << 32) | byte_in_segment`. Segments are at most
//! 4 GiB, so the segment size is not encoded in positions and can differ from
//! one open to the next. Positions grow with log order, so cuts compare as
//! plain integers.
//!
//! # Torn tail and corruption
//!
//! Only the last segment can hold bytes that were never synced: leaving a
//! segment syncs it first. So a frame that runs past the end, or whose CRC
//! fails, in the **last** segment is a torn tail: [`open`](VectorIndexWal::open)
//! truncates the file there, so stale bytes can never be read as records once
//! new appends land behind them. The same in an **earlier** segment, a gap in
//! the segment ids, or a bad header fails the open.

#![cfg_attr(not(test), expect(dead_code, reason = "wired into Database in steps 2-4 of the vector index WAL design"))]

use crate::db::metrics::Metrics;
use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

/// Default segment size (64 MiB, as the main WAL).
pub(crate) const DEFAULT_SEGMENT_SIZE: u32 = 64 * 1024 * 1024;

/// Smallest segment size accepted: room for a header, one empty record and the
/// end marker, rounded up.
pub(crate) const MIN_SEGMENT_SIZE: u32 = 64;

const SEGMENT_MAGIC: [u8; 4] = *b"MVIW";
const FORMAT_VERSION: u32 = 1;
/// Bytes of the segment header; the first record starts here.
pub(crate) const SEGMENT_HEADER_LEN: u32 = 16;
/// `len` and `crc32` before each body.
const FRAME_HEADER_LEN: u32 = 8;
/// The end marker of a sealed segment: a zero `len`.
const END_MARKER_LEN: u32 = 4;
/// The body's fixed fields: `ns_id`, `seq`, `key_len`, `value_len`.
const BODY_FIXED_LEN: usize = 4 + 8 + 4 + 4;

/// Errors from the vector index WAL.
#[derive(Error, Debug)]
pub enum VectorIndexWalError {
    /// An I/O error reading or writing the log.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// The log is damaged somewhere other than its torn tail.
    #[error("vector index WAL corrupt: {0}")]
    Corruption(String),
    /// A record too large for one segment. Nothing was written.
    #[error("vector index WAL record of {len} bytes does not fit a segment (at most {max})")]
    RecordTooLarge {
        /// The record's framed length.
        len: u64,
        /// The largest framed record a segment holds.
        max: u64,
    },
}

/// Result type of this module.
pub type Result<T> = std::result::Result<T, VectorIndexWalError>;

/// The position of byte `byte` of segment `segment_id`.
pub(crate) fn position(segment_id: u64, byte: u32) -> u64 {
    (segment_id << 32) | u64::from(byte)
}

/// A position's segment id and byte within it.
fn split(pos: u64) -> (u64, u32) {
    (pos >> 32, pos as u32)
}

/// One record read from the log. Borrows the reader's buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RecordRef<'a> {
    pub ns_id: u32,
    pub seq: u64,
    pub key: &'a [u8],
    pub value: &'a [u8],
}

/// Where an append landed: `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Appended {
    pub start: u64,
    pub end: u64,
}

/// What [`VectorIndexWal::open`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OpenReport {
    /// Position of the next append.
    pub tail: u64,
    /// Highest seq of any record in the log (0 if none).
    pub last_seq: u64,
    /// Bytes of a torn tail truncated from the last segment.
    pub torn_bytes_dropped: u64,
}

/// The vector index WAL. See the module docs.
pub(crate) struct VectorIndexWal {
    dir: PathBuf,
    segment_size: u32,
    /// The append lock and the active segment.
    active: Mutex<Active>,
    /// Every byte below this position is on stable storage.
    durable_tail: AtomicU64,
    /// The database's engine-wide counters.
    metrics: Arc<Metrics>,
}

struct Active {
    segment_id: u64,
    file: Arc<File>,
    /// Next append's byte within the segment.
    pos: u32,
    last_seq: u64,
}

/// How a walk over one segment ended.
#[derive(Debug, PartialEq, Eq)]
enum WalkEnd {
    /// Reached the limit on a frame boundary.
    Clean { pos: u32 },
    /// Found the end marker at `pos`.
    Sealed { pos: u32 },
    /// Found a frame that is incomplete or fails its CRC at `pos`.
    Torn { pos: u32, reason: String },
}

fn segment_path(dir: &Path, segment_id: u64) -> PathBuf {
    dir.join(format!("{segment_id:010}.log"))
}

/// Segment ids in `dir`, ascending. Files that are not segments are ignored.
fn list_segments(dir: &Path) -> Result<Vec<u64>> {
    let mut ids = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let name = entry?.file_name();
        let Some(stem) = name.to_str().and_then(|n| n.strip_suffix(".log")) else {
            continue;
        };
        if stem.len() == 10
            && stem.bytes().all(|b| b.is_ascii_digit())
            && let Ok(id) = stem.parse::<u64>()
        {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    Ok(ids)
}

fn sync_dir(dir: &Path) -> Result<()> {
    File::open(dir)?.sync_all()?;
    Ok(())
}

fn header_bytes(segment_id: u64) -> [u8; SEGMENT_HEADER_LEN as usize] {
    let mut h = [0u8; SEGMENT_HEADER_LEN as usize];
    h[0..4].copy_from_slice(&SEGMENT_MAGIC);
    h[4..8].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    h[8..16].copy_from_slice(&segment_id.to_le_bytes());
    h
}

/// Checks a segment's header. The file must be at least a header long.
fn check_header(file: &File, segment_id: u64) -> Result<()> {
    let mut h = [0u8; SEGMENT_HEADER_LEN as usize];
    file.read_exact_at(&mut h, 0)?;
    if h[0..4] != SEGMENT_MAGIC {
        return Err(VectorIndexWalError::Corruption(format!(
            "segment {segment_id} has no vector index WAL header"
        )));
    }
    let version = u32::from_le_bytes(h[4..8].try_into().unwrap());
    if version != FORMAT_VERSION {
        return Err(VectorIndexWalError::Corruption(format!(
            "segment {segment_id} has format version {version}, expected {FORMAT_VERSION}"
        )));
    }
    let id = u64::from_le_bytes(h[8..16].try_into().unwrap());
    if id != segment_id {
        return Err(VectorIndexWalError::Corruption(format!("segment file {segment_id} holds segment {id}")));
    }
    Ok(())
}

/// Creates segment `segment_id` with its header, durable (contents and name)
/// before it returns, so a record in it always has a header behind it.
fn create_segment(dir: &Path, segment_id: u64) -> Result<File> {
    if segment_id > u64::from(u32::MAX) {
        return Err(VectorIndexWalError::Corruption("segment ids exhausted".to_owned()));
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(segment_path(dir, segment_id))?;
    file.write_all_at(&header_bytes(segment_id), 0)?;
    file.sync_all()?;
    sync_dir(dir)?;
    Ok(file)
}

fn encode_frame(ns_id: u32, seq: u64, key: &[u8], value: &[u8]) -> Vec<u8> {
    let body_len = BODY_FIXED_LEN + key.len() + value.len();
    let mut buf = Vec::with_capacity(FRAME_HEADER_LEN as usize + body_len);
    buf.extend_from_slice(&(body_len as u32).to_le_bytes());
    buf.extend_from_slice(&[0u8; 4]); // crc, filled below
    buf.extend_from_slice(&ns_id.to_le_bytes());
    buf.extend_from_slice(&seq.to_le_bytes());
    buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
    buf.extend_from_slice(value);
    let crc = crc32fast::hash(&buf[FRAME_HEADER_LEN as usize..]);
    buf[4..8].copy_from_slice(&crc.to_le_bytes());
    buf
}

/// Decodes a body whose CRC has passed. A layout that does not add up is
/// corruption, not a torn write: the CRC says these are the bytes written.
fn decode_body(body: &[u8], at: u64) -> Result<RecordRef<'_>> {
    let bad = || VectorIndexWalError::Corruption(format!("record at {at:#x} passes its CRC but does not decode"));
    if body.len() < BODY_FIXED_LEN {
        return Err(bad());
    }
    let ns_id = u32::from_le_bytes(body[0..4].try_into().unwrap());
    let seq = u64::from_le_bytes(body[4..12].try_into().unwrap());
    let key_len = u32::from_le_bytes(body[12..16].try_into().unwrap()) as usize;
    let key_end = 16usize.checked_add(key_len).filter(|&e| e + 4 <= body.len()).ok_or_else(bad)?;
    let value_len = u32::from_le_bytes(body[key_end..key_end + 4].try_into().unwrap()) as usize;
    if key_end + 4 + value_len != body.len() {
        return Err(bad());
    }
    Ok(RecordRef {
        ns_id,
        seq,
        key: &body[16..key_end],
        value: &body[key_end + 4..],
    })
}

/// Walks the frames of one segment from byte `start` up to byte `limit`,
/// handing each record and its position to `f`. The one frame-parsing loop:
/// `open` and `for_each_from` both use it, so the torn-tail and end-marker
/// rules exist once.
///
/// Every length is checked against the bytes left before anything is
/// allocated for it (a corrupt length could otherwise ask for 4 GiB).
fn walk_segment<E: From<VectorIndexWalError>>(
    file: &File,
    segment_id: u64,
    start: u32,
    limit: u32,
    buf: &mut Vec<u8>,
    mut f: impl FnMut(u64, RecordRef<'_>) -> std::result::Result<(), E>,
) -> std::result::Result<WalkEnd, E> {
    let io = |e: std::io::Error| E::from(VectorIndexWalError::Io(e));
    let mut pos = start;
    loop {
        if pos >= limit {
            return Ok(WalkEnd::Clean { pos });
        }
        let left = limit - pos;
        if left < END_MARKER_LEN {
            return Ok(WalkEnd::Torn {
                pos,
                reason: format!("{left} bytes where a frame length belongs"),
            });
        }
        let mut len_buf = [0u8; 4];
        file.read_exact_at(&mut len_buf, u64::from(pos)).map_err(io)?;
        let len = u32::from_le_bytes(len_buf);
        if len == 0 {
            return Ok(WalkEnd::Sealed { pos });
        }
        if left < FRAME_HEADER_LEN || len > left - FRAME_HEADER_LEN {
            return Ok(WalkEnd::Torn {
                pos,
                reason: format!("frame length {len} runs past the end ({left} bytes left)"),
            });
        }
        let mut crc_buf = [0u8; 4];
        file.read_exact_at(&mut crc_buf, u64::from(pos) + 4).map_err(io)?;
        buf.resize(len as usize, 0);
        file.read_exact_at(buf, u64::from(pos + FRAME_HEADER_LEN)).map_err(io)?;
        if crc32fast::hash(buf) != u32::from_le_bytes(crc_buf) {
            return Ok(WalkEnd::Torn {
                pos,
                reason: "CRC mismatch".to_owned(),
            });
        }
        let at = position(segment_id, pos);
        let record = decode_body(buf, at)?;
        f(at, record)?;
        pos += FRAME_HEADER_LEN + len;
    }
}

fn file_len(file: &File) -> Result<u32> {
    let len = file.metadata()?.len();
    u32::try_from(len).map_err(|_| VectorIndexWalError::Corruption(format!("segment of {len} bytes")))
}

impl VectorIndexWal {
    /// Opens the log in `dir`, creating it if absent, and finds its tail.
    ///
    /// Validates every segment and reads every record (for the highest seq). A
    /// torn tail in the last segment is truncated away; any other damage fails
    /// the open. A last segment that was sealed but has no successor (a crash
    /// mid-rotation) gets one. The last segment is fsynced before returning, so
    /// everything below the tail is durable.
    pub(crate) fn open(dir: &Path, segment_size: u32, metrics: Arc<Metrics>) -> Result<(Self, OpenReport)> {
        let segment_size = segment_size.max(MIN_SEGMENT_SIZE);
        if !dir.exists() {
            std::fs::create_dir_all(dir)?;
            if let Some(parent) = dir.parent().filter(|p| !p.as_os_str().is_empty()) {
                sync_dir(parent)?;
            }
        }

        let ids = list_segments(dir)?;
        if let Some(w) = ids.windows(2).find(|w| w[1] != w[0] + 1) {
            return Err(VectorIndexWalError::Corruption(format!(
                "segment {} is missing (segments {} and {} exist)",
                w[0] + 1,
                w[0],
                w[1]
            )));
        }

        let mut last_seq = 0u64;
        let mut buf = Vec::new();
        let mut note_seq = |_: u64, r: RecordRef<'_>| -> Result<()> {
            last_seq = last_seq.max(r.seq);
            Ok(())
        };

        // Sealed segments: complete, ending in the end marker.
        let (sealed, last) = match ids.split_last() {
            Some((&last, sealed)) => (sealed, Some(last)),
            None => (&[][..], None),
        };
        for &id in sealed {
            let file = File::open(segment_path(dir, id))?;
            let len = file_len(&file)?;
            if len < SEGMENT_HEADER_LEN {
                return Err(VectorIndexWalError::Corruption(format!("segment {id} is shorter than its header")));
            }
            check_header(&file, id)?;
            match walk_segment(&file, id, SEGMENT_HEADER_LEN, len, &mut buf, &mut note_seq)? {
                WalkEnd::Sealed { .. } => {}
                WalkEnd::Clean { .. } => {
                    return Err(VectorIndexWalError::Corruption(format!("segment {id} is not sealed but is not the last")));
                }
                WalkEnd::Torn { pos, reason } => {
                    return Err(VectorIndexWalError::Corruption(format!(
                        "segment {id} (not the last) is damaged at byte {pos}: {reason}"
                    )));
                }
            }
        }

        // The last segment: the active one, possibly torn.
        let mut torn_bytes_dropped = 0u64;
        let (segment_id, file, pos) = match last {
            None => (0, create_segment(dir, 0)?, SEGMENT_HEADER_LEN),
            Some(id) => {
                let file = OpenOptions::new().read(true).write(true).open(segment_path(dir, id))?;
                let len = file_len(&file)?;
                if len < SEGMENT_HEADER_LEN || (len == SEGMENT_HEADER_LEN && check_header(&file, id).is_err()) {
                    // A crash while creating the segment: its header was not yet
                    // durable, so no record can be in it.
                    torn_bytes_dropped = u64::from(len);
                    drop(file);
                    (id, create_segment(dir, id)?, SEGMENT_HEADER_LEN)
                } else {
                    check_header(&file, id)?;
                    match walk_segment(&file, id, SEGMENT_HEADER_LEN, len, &mut buf, &mut note_seq)? {
                        WalkEnd::Clean { pos } => (id, file, pos),
                        WalkEnd::Torn { pos, reason } => {
                            log::warn!(
                                "[vector index WAL] truncating a torn tail: segment {id}, byte {pos}, {} bytes dropped ({reason})",
                                len - pos
                            );
                            file.set_len(u64::from(pos))?;
                            torn_bytes_dropped = u64::from(len - pos);
                            (id, file, pos)
                        }
                        WalkEnd::Sealed { .. } => {
                            // A crash after sealing, before the next segment existed.
                            drop(file);
                            (id + 1, create_segment(dir, id + 1)?, SEGMENT_HEADER_LEN)
                        }
                    }
                }
            }
        };
        file.sync_all()?;

        let tail = position(segment_id, pos);
        let wal = Self {
            dir: dir.to_path_buf(),
            segment_size,
            active: Mutex::new(Active {
                segment_id,
                file: Arc::new(file),
                pos,
                last_seq,
            }),
            durable_tail: AtomicU64::new(tail),
            metrics,
        };
        Ok((
            wal,
            OpenReport {
                tail,
                last_seq,
                torn_bytes_dropped,
            },
        ))
    }

    /// The largest framed record a segment holds.
    fn max_frame_len(&self) -> u64 {
        u64::from(self.segment_size - SEGMENT_HEADER_LEN - END_MARKER_LEN)
    }

    /// Appends one record. Not durable until a later [`sync`](Self::sync)
    /// returns.
    ///
    /// Records are in append order, which is not necessarily seq order: seqs
    /// are allocated before the append lock is taken.
    pub(crate) fn append(&self, ns_id: u32, key: &[u8], value: &[u8], seq: u64) -> Result<Appended> {
        let frame_len = FRAME_HEADER_LEN as u64 + (BODY_FIXED_LEN + key.len() + value.len()) as u64;
        if frame_len > self.max_frame_len() {
            return Err(VectorIndexWalError::RecordTooLarge {
                len: frame_len,
                max: self.max_frame_len(),
            });
        }
        let frame = encode_frame(ns_id, seq, key, value);
        let frame_len = frame.len() as u32;

        let mut a = self.active.lock();
        if u64::from(a.pos) + u64::from(frame_len) + u64::from(END_MARKER_LEN) > u64::from(self.segment_size) {
            self.rotate(&mut a)?;
        }
        // On a failed write `pos` stays put, so the next append overwrites the
        // partial bytes.
        a.file.write_all_at(&frame, u64::from(a.pos))?;
        let start = position(a.segment_id, a.pos);
        a.pos += frame_len;
        a.last_seq = a.last_seq.max(seq);
        let end = position(a.segment_id, a.pos);
        drop(a);

        Metrics::bump(&self.metrics.vector_wal_appends);
        Metrics::add(&self.metrics.vector_wal_bytes_appended, u64::from(frame_len));
        Ok(Appended { start, end })
    }

    /// Seals the active segment and starts the next. The sealed segment is
    /// fsynced first, so only the last segment can ever hold unsynced bytes.
    fn rotate(&self, a: &mut Active) -> Result<()> {
        a.file.write_all_at(&0u32.to_le_bytes(), u64::from(a.pos))?;
        a.file.sync_data()?;
        self.durable_tail.fetch_max(position(a.segment_id, a.pos), Ordering::AcqRel);
        let next = a.segment_id + 1;
        a.file = Arc::new(create_segment(&self.dir, next)?);
        a.segment_id = next;
        a.pos = SEGMENT_HEADER_LEN;
        Ok(())
    }

    /// Makes every append that returned before this call durable.
    ///
    /// The fsync runs outside the append lock, so appends carry on meanwhile.
    /// Does no I/O when nothing was appended since the last sync.
    pub(crate) fn sync(&self) -> Result<()> {
        let (file, tail) = {
            let a = self.active.lock();
            (Arc::clone(&a.file), position(a.segment_id, a.pos))
        };
        if tail <= self.durable_tail.load(Ordering::Acquire) {
            return Ok(());
        }
        // Earlier segments were fsynced when they were sealed; the one at
        // `tail` holds every append that returned before the snapshot.
        file.sync_data()?;
        self.durable_tail.fetch_max(tail, Ordering::AcqRel);
        Metrics::bump(&self.metrics.vector_wal_fsyncs);
        Ok(())
    }

    /// Reads every record from `start` (a record's position, or a segment's
    /// start) up to the tail as it is when the call begins, in log order.
    ///
    /// Streams with one reused buffer, so a long replay holds one record at a
    /// time. A missing segment or a damaged record in the range is corruption.
    pub(crate) fn for_each_from<E: From<VectorIndexWalError>>(
        &self,
        start: u64,
        mut f: impl FnMut(u64, RecordRef<'_>) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        let (end_id, end_pos) = {
            let a = self.active.lock();
            (a.segment_id, a.pos)
        };
        let (first_id, first_pos) = split(start);
        let mut buf = Vec::new();
        for id in first_id..=end_id {
            let file = File::open(segment_path(&self.dir, id)).map_err(|e| {
                E::from(if e.kind() == std::io::ErrorKind::NotFound {
                    VectorIndexWalError::Corruption(format!("segment {id} is missing"))
                } else {
                    VectorIndexWalError::Io(e)
                })
            })?;
            check_header(&file, id)?;
            let from = if id == first_id {
                first_pos.max(SEGMENT_HEADER_LEN)
            } else {
                SEGMENT_HEADER_LEN
            };
            let limit = if id == end_id { end_pos } else { file_len(&file)? };
            let end = walk_segment(&file, id, from, limit, &mut buf, &mut f)?;
            match end {
                WalkEnd::Clean { .. } if id == end_id => {}
                WalkEnd::Sealed { .. } if id != end_id => {}
                WalkEnd::Torn { pos, reason } => {
                    return Err(VectorIndexWalError::Corruption(format!("segment {id} is damaged at byte {pos}: {reason}")).into());
                }
                other => {
                    return Err(VectorIndexWalError::Corruption(format!("segment {id} ends unexpectedly: {other:?}")).into());
                }
            }
        }
        Ok(())
    }

    /// Position of the next append.
    pub(crate) fn tail(&self) -> u64 {
        let a = self.active.lock();
        position(a.segment_id, a.pos)
    }

    /// Every byte below this position is on stable storage.
    pub(crate) fn durable_tail(&self) -> u64 {
        self.durable_tail.load(Ordering::Acquire)
    }

    /// Highest seq of any record appended or found at open (0 if none).
    pub(crate) fn last_seq(&self) -> u64 {
        self.active.lock().last_seq
    }

    /// Every segment on disk, `(segment_id, bytes)`, oldest first.
    pub(crate) fn segments(&self) -> Result<Vec<(u64, u64)>> {
        list_segments(&self.dir)?
            .into_iter()
            .map(|id| Ok((id, std::fs::metadata(segment_path(&self.dir, id))?.len())))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering::Relaxed;
    use tempfile::TempDir;

    type Row = (u64, u32, u64, Vec<u8>, Vec<u8>);

    fn open_at(dir: &Path, segment_size: u32) -> (VectorIndexWal, OpenReport) {
        VectorIndexWal::open(dir, segment_size, Arc::new(Metrics::default())).unwrap()
    }

    fn rows_from(wal: &VectorIndexWal, start: u64) -> Vec<Row> {
        let mut rows = Vec::new();
        wal.for_each_from(start, |at, r| -> Result<()> {
            rows.push((at, r.ns_id, r.seq, r.key.to_vec(), r.value.to_vec()));
            Ok(())
        })
        .unwrap();
        rows
    }

    fn key(i: u64) -> Vec<u8> {
        format!("key-{i:04}").into_bytes()
    }

    fn value(i: u64) -> Vec<u8> {
        vec![i as u8; 40]
    }

    /// Appends records 0..n (ns `i % 3`, seq `i + 1`) and returns where each landed.
    fn append_n(wal: &VectorIndexWal, n: u64) -> Vec<Appended> {
        (0..n).map(|i| wal.append((i % 3) as u32, &key(i), &value(i), i + 1).unwrap()).collect()
    }

    fn expected(appended: &[Appended], range: std::ops::Range<u64>) -> Vec<Row> {
        range
            .map(|i| (appended[i as usize].start, (i % 3) as u32, i + 1, key(i), value(i)))
            .collect()
    }

    /// Flips one byte of a segment file.
    fn flip(dir: &Path, segment_id: u64, byte: u64) {
        let f = OpenOptions::new().read(true).write(true).open(segment_path(dir, segment_id)).unwrap();
        let mut b = [0u8; 1];
        f.read_exact_at(&mut b, byte).unwrap();
        f.write_all_at(&[b[0] ^ 0xff], byte).unwrap();
    }

    fn set_len(dir: &Path, segment_id: u64, len: u64) {
        OpenOptions::new()
            .write(true)
            .open(segment_path(dir, segment_id))
            .unwrap()
            .set_len(len)
            .unwrap();
    }

    fn corruption(r: Result<(VectorIndexWal, OpenReport)>) -> String {
        match r {
            Err(VectorIndexWalError::Corruption(m)) => m,
            Err(e) => panic!("expected corruption, got {e}"),
            Ok(_) => panic!("expected corruption, the open succeeded"),
        }
    }

    fn reopen(dir: &Path, segment_size: u32) -> Result<(VectorIndexWal, OpenReport)> {
        VectorIndexWal::open(dir, segment_size, Arc::new(Metrics::default()))
    }

    /// Frame length of the records `append_n` writes.
    const FRAME: u32 = FRAME_HEADER_LEN + BODY_FIXED_LEN as u32 + 8 + 40;

    #[test]
    fn appended_records_read_back_in_order() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        let appended = append_n(&wal, 50);
        assert!(appended.windows(2).all(|w| w[0].end == w[1].start && w[0].start < w[1].start));
        assert_eq!(rows_from(&wal, 0), expected(&appended, 0..50));
        assert_eq!(wal.tail(), appended[49].end);
    }

    #[test]
    fn reopen_finds_the_tail_and_the_highest_seq() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        wal.append(1, b"a", b"x", 10).unwrap();
        let last = wal.append(1, b"b", b"y", 5).unwrap(); // seqs are not in log order
        wal.sync().unwrap();
        drop(wal);

        let (wal, report) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        assert_eq!(
            report,
            OpenReport {
                tail: last.end,
                last_seq: 10,
                torn_bytes_dropped: 0
            }
        );
        assert_eq!(wal.last_seq(), 10);
        let more = wal.append(2, b"c", b"z", 11).unwrap();
        assert_eq!(more.start, last.end);
        let rows = rows_from(&wal, 0);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[2], (more.start, 2, 11, b"c".to_vec(), b"z".to_vec()));
    }

    #[test]
    fn records_span_segments_and_every_sealed_segment_ends_in_a_marker() {
        let dir = TempDir::new().unwrap();
        // Three records fit a segment: 16 + 3 * 76 + 4 = 248 <= 256.
        let (wal, _) = open_at(dir.path(), 256);
        let appended = append_n(&wal, 20);
        let segs = wal.segments().unwrap();
        assert_eq!(segs.len(), 7);
        for &(id, len) in &segs[..segs.len() - 1] {
            assert_eq!(len, u64::from(SEGMENT_HEADER_LEN + 3 * FRAME + END_MARKER_LEN), "segment {id}");
            let f = File::open(segment_path(dir.path(), id)).unwrap();
            check_header(&f, id).unwrap();
            let mut m = [0xffu8; 4];
            f.read_exact_at(&mut m, len - 4).unwrap();
            assert_eq!(m, [0; 4], "segment {id} ends in the end marker");
        }
        assert_eq!(split(appended[3].start), (1, SEGMENT_HEADER_LEN));
        assert_eq!(rows_from(&wal, 0), expected(&appended, 0..20));
        drop(wal);
        let (wal, report) = open_at(dir.path(), 256);
        assert_eq!(report.tail, appended[19].end);
        assert_eq!(rows_from(&wal, 0), expected(&appended, 0..20));
    }

    #[test]
    fn reading_starts_at_a_record_or_a_segment_start() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), 256);
        let appended = append_n(&wal, 20);
        assert_eq!(rows_from(&wal, appended[7].start), expected(&appended, 7..20));
        assert_eq!(rows_from(&wal, position(2, 0)), expected(&appended, 6..20));
        assert_eq!(rows_from(&wal, wal.tail()), vec![]);
    }

    #[test]
    fn a_torn_tail_is_truncated_and_later_appends_read_back() {
        for cut in [2u64, u64::from(FRAME_HEADER_LEN) + 5, u64::from(FRAME) - 1] {
            let dir = TempDir::new().unwrap();
            let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
            let appended = append_n(&wal, 5);
            drop(wal);
            let (_, last_byte) = split(appended[4].start);
            set_len(dir.path(), 0, u64::from(last_byte) + cut);

            let (wal, report) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
            assert_eq!(report.tail, appended[4].start, "cut {cut}");
            assert_eq!(report.torn_bytes_dropped, cut);
            assert_eq!(report.last_seq, 4);
            let fresh = wal.append(9, b"new", b"v", 99).unwrap();
            assert_eq!(fresh.start, appended[4].start);
            let mut want = expected(&appended, 0..4);
            want.push((fresh.start, 9, 99, b"new".to_vec(), b"v".to_vec()));
            assert_eq!(rows_from(&wal, 0), want);
            drop(wal);
            let (wal, report) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
            assert_eq!(report.torn_bytes_dropped, 0);
            assert_eq!(rows_from(&wal, 0), want, "nothing stale after the new record");
        }
    }

    #[test]
    fn a_crc_failure_is_a_torn_tail_only_in_the_last_segment() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), 256);
        let appended = append_n(&wal, 8); // segments 0, 1, 2
        drop(wal);

        // Last record of the last segment.
        let (seg, byte) = split(appended[7].start);
        assert_eq!(seg, 2);
        flip(dir.path(), seg, u64::from(byte) + 20);
        let (wal, report) = reopen(dir.path(), 256).unwrap();
        assert_eq!(report.tail, appended[7].start);
        assert_eq!(rows_from(&wal, 0), expected(&appended, 0..7));
        drop(wal);

        // A record of an earlier segment.
        let (seg, byte) = split(appended[1].start);
        flip(dir.path(), seg, u64::from(byte) + 20);
        let msg = corruption(reopen(dir.path(), 256));
        assert!(msg.contains("segment 0") && msg.contains("CRC"), "{msg}");
    }

    #[test]
    fn a_corrupt_length_is_rejected_before_allocating() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), 256);
        let appended = append_n(&wal, 8);
        drop(wal);
        let write_len = |at: u64| {
            let (seg, byte) = split(at);
            let f = OpenOptions::new().write(true).open(segment_path(dir.path(), seg)).unwrap();
            f.write_all_at(&u32::MAX.to_le_bytes(), u64::from(byte)).unwrap();
        };

        write_len(appended[7].start);
        let (_, report) = reopen(dir.path(), 256).unwrap();
        assert_eq!(report.tail, appended[7].start);
        assert_eq!(report.torn_bytes_dropped, u64::from(FRAME));

        write_len(appended[4].start);
        let msg = corruption(reopen(dir.path(), 256));
        assert!(msg.contains("segment 1") && msg.contains("runs past the end"), "{msg}");
    }

    #[test]
    fn a_bad_header_or_a_missing_segment_fails_the_open() {
        let fresh = || {
            let dir = TempDir::new().unwrap();
            let (wal, _) = open_at(dir.path(), 256);
            append_n(&wal, 8); // segments 0, 1, 2
            dir
        };

        let dir = fresh();
        flip(dir.path(), 2, 0);
        assert!(corruption(reopen(dir.path(), 256)).contains("no vector index WAL header"));

        let dir = fresh();
        flip(dir.path(), 0, 4);
        assert!(corruption(reopen(dir.path(), 256)).contains("format version"));

        let dir = fresh();
        let f = OpenOptions::new().write(true).open(segment_path(dir.path(), 1)).unwrap();
        f.write_all_at(&7u64.to_le_bytes(), 8).unwrap();
        assert!(corruption(reopen(dir.path(), 256)).contains("segment file 1 holds segment 7"));

        let dir = fresh();
        std::fs::remove_file(segment_path(dir.path(), 1)).unwrap();
        assert!(corruption(reopen(dir.path(), 256)).contains("segment 1 is missing"));
    }

    #[test]
    fn a_crash_mid_rotation_is_finished_at_open() {
        // Sealed, but the next segment was never created.
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), 256);
        let appended = append_n(&wal, 4); // record 3 starts segment 1
        drop(wal);
        std::fs::remove_file(segment_path(dir.path(), 1)).unwrap();
        let (wal, report) = reopen(dir.path(), 256).unwrap();
        assert_eq!(report.tail, position(1, SEGMENT_HEADER_LEN));
        assert_eq!(rows_from(&wal, 0), expected(&appended, 0..3));
        drop(wal);

        // The next segment exists but its header never landed.
        set_len(dir.path(), 1, 5);
        let (wal, report) = reopen(dir.path(), 256).unwrap();
        assert_eq!(report.tail, position(1, SEGMENT_HEADER_LEN));
        assert_eq!(report.torn_bytes_dropped, 5);
        let more = wal.append(0, b"k", b"v", 50).unwrap();
        assert_eq!(more.start, position(1, SEGMENT_HEADER_LEN));
        assert_eq!(rows_from(&wal, 0).len(), 4);
    }

    #[test]
    fn a_record_too_large_for_a_segment_writes_nothing() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), 256);
        append_n(&wal, 2);
        let (tail, segs) = (wal.tail(), wal.segments().unwrap());
        let max = u64::from(256 - SEGMENT_HEADER_LEN - END_MARKER_LEN);
        let fits = vec![0u8; (max - u64::from(FRAME_HEADER_LEN) - BODY_FIXED_LEN as u64) as usize];
        let too_big = vec![0u8; fits.len() + 1];
        match wal.append(0, b"", &too_big, 9) {
            Err(VectorIndexWalError::RecordTooLarge { len, max: m }) => assert_eq!((len, m), (max + 1, max)),
            other => panic!("expected RecordTooLarge, got {other:?}"),
        }
        assert_eq!((wal.tail(), wal.segments().unwrap()), (tail, segs));
        // The largest record that fits is accepted, alone in its segment.
        let a = wal.append(0, b"", &fits, 10).unwrap();
        assert_eq!(split(a.start), (1, SEGMENT_HEADER_LEN));
        assert_eq!(rows_from(&wal, a.start).len(), 1);
    }

    #[test]
    fn sync_covers_earlier_appends_and_skips_when_idle() {
        let dir = TempDir::new().unwrap();
        let metrics = Arc::new(Metrics::default());
        let (wal, report) = VectorIndexWal::open(dir.path(), 256, Arc::clone(&metrics)).unwrap();
        assert_eq!(wal.durable_tail(), report.tail);
        wal.sync().unwrap();
        assert_eq!(metrics.vector_wal_fsyncs.load(Relaxed), 0, "nothing to sync");

        let appended = append_n(&wal, 2);
        assert!(wal.durable_tail() < wal.tail());
        wal.sync().unwrap();
        wal.sync().unwrap();
        assert_eq!(wal.durable_tail(), appended[1].end);
        assert_eq!(metrics.vector_wal_fsyncs.load(Relaxed), 1);

        // Sealing a segment makes it durable without a sync.
        let appended = append_n(&wal, 2);
        assert_eq!(wal.durable_tail(), position(0, split(appended[0].end).1));
    }

    #[test]
    fn concurrent_appends_and_syncs_lose_nothing() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), 4096);
        let wal = Arc::new(wal);
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let syncer = {
            let (wal, done) = (Arc::clone(&wal), Arc::clone(&done));
            std::thread::spawn(move || {
                while !done.load(Relaxed) {
                    wal.sync().unwrap();
                    assert!(wal.durable_tail() <= wal.tail());
                }
            })
        };
        let writers: Vec<_> = (0..8u64)
            .map(|t| {
                let wal = Arc::clone(&wal);
                std::thread::spawn(move || {
                    for i in 0..200u64 {
                        let seq = t * 1000 + i;
                        wal.append(t as u32, &seq.to_le_bytes(), &[t as u8; 30], seq).unwrap();
                    }
                })
            })
            .collect();
        writers.into_iter().for_each(|w| w.join().unwrap());
        done.store(true, Relaxed);
        syncer.join().unwrap();
        wal.sync().unwrap();
        assert_eq!(wal.durable_tail(), wal.tail());

        let mut seqs: Vec<u64> = rows_from(&wal, 0)
            .into_iter()
            .map(|(_, ns, seq, k, v)| {
                assert_eq!(k, seq.to_le_bytes());
                assert_eq!(v, vec![ns as u8; 30]);
                seq
            })
            .collect();
        seqs.sort_unstable();
        let want: Vec<u64> = (0..8u64).flat_map(|t| (0..200).map(move |i| t * 1000 + i)).collect();
        assert_eq!(seqs, want);
    }

    #[test]
    fn counters_count_appends_bytes_and_fsyncs() {
        let dir = TempDir::new().unwrap();
        let metrics = Arc::new(Metrics::default());
        let (wal, _) = VectorIndexWal::open(dir.path(), DEFAULT_SEGMENT_SIZE, Arc::clone(&metrics)).unwrap();
        append_n(&wal, 7);
        wal.sync().unwrap();
        append_n(&wal, 3);
        wal.sync().unwrap();
        let s = metrics.snapshot();
        assert_eq!(s.vector_wal_appends, 10);
        assert_eq!(s.vector_wal_bytes_appended, 10 * u64::from(FRAME));
        assert_eq!(s.vector_wal_fsyncs, 2);
    }

    #[test]
    fn an_empty_log_opens_and_reopens_the_same() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vector_index_wal");
        let (wal, report) = open_at(&path, DEFAULT_SEGMENT_SIZE);
        let empty = OpenReport {
            tail: position(0, SEGMENT_HEADER_LEN),
            last_seq: 0,
            torn_bytes_dropped: 0,
        };
        assert_eq!(report, empty);
        assert_eq!(wal.segments().unwrap(), vec![(0, u64::from(SEGMENT_HEADER_LEN))]);
        assert_eq!(rows_from(&wal, 0), vec![]);
        drop(wal);
        std::fs::write(path.join("retired_cut"), b"not a segment").unwrap();
        let (wal, report) = open_at(&path, DEFAULT_SEGMENT_SIZE);
        assert_eq!(report, empty);
        assert_eq!(wal.segments().unwrap().len(), 1);
    }
}
