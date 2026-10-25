//! A segmented, checksummed, append-only log: the format and mechanics every
//! write-ahead log in the engine shares.
//!
//! Both logs are built on it — the main WAL (`db/wal.rs`) and the vector index
//! WAL (`db/vector_index_wal.rs`) — and a new one needs only a [`LogFormat`]
//! (its magic and body version) and a body encoding. Everything about how
//! bytes reach the disk and come back after a crash lives here, once:
//! segments, frames, positions, preallocation, commit rounds, the torn-tail and
//! corruption rules at open, reads, and deleting retired segments. What a body
//! means, which records are retired and when, is each log's own business.
//!
//! # Format
//!
//! A log is a directory of segment files `{segment_id:010}.log` with
//! contiguous ids. Each segment is:
//!
//! ```text
//! [header 16B: magic (4, per log) | format_version u32 | segment_id u64]
//! [frame] [frame] ... [end marker: u32 0xFFFF_FFFF, sealed segments only]
//! [zeros to the end of the file: unwritten space]
//! frame: [len u32][crc32(body) u32][body]
//! ```
//!
//! All integers are little-endian. Segments are created at their full size,
//! and from a log's second segment on they are zero-filled ahead
//! (`db/segment_prep.rs`), so a sync commits no new file size or block
//! allocation. A length of 0 is therefore **unwritten space** — the segment's
//! records end there — and the end marker is a length no frame can have. Every
//! segment but the last is **sealed**: it ends with the end marker, and space
//! for the marker is reserved, so a frame never fills a segment's last four
//! bytes.
//!
//! # Positions
//!
//! A position is `(segment_id << 32) | byte_in_segment`. Segments are at most
//! 4 GiB, so the segment size is not encoded in positions and can change from
//! one open to the next. Positions grow with log order, so cuts compare as
//! plain integers. A segment's first record is at
//! [`segment_start`]`(id)`, past its header.
//!
//! # Durability
//!
//! [`commit`](SegmentedLog::commit) and [`sync`](SegmentedLog::sync) run as
//! commit rounds (`db/commit_rounds.rs`): concurrent callers share one fsync,
//! run outside the append lock. A segment is fsynced as it is sealed, before
//! the next one is created, so only the last segment can ever hold bytes no
//! fsync covered. A failed fsync, or a failed rotation, **poisons** the log:
//! appends and commits fail until the database is reopened (after a failed
//! fsync the kernel may have dropped the dirty pages, and a later fsync that
//! succeeds says nothing about them).
//!
//! # Torn tail and corruption
//!
//! Since only the last segment can hold unsynced bytes, in the **last**
//! segment unwritten space, or a frame that runs past the end or fails its
//! CRC (a torn tail), is the tail: [`open`](SegmentedLog::open) zeroes every
//! byte from there to the end of the file before any append. A power loss can
//! keep some pages past the tail and lose others, so a whole frame can sit
//! beyond the gap; new frames laid over the gap could end exactly where it
//! starts, and a later open would read it as part of the log. In an
//! **earlier** segment at or above where reading starts, a bad frame or
//! unwritten space before the end marker, a gap in the segment ids, or a bad
//! header fails the open — except unwritten space right where reading starts,
//! in a segment whose end marker lies at or below that start (a saved
//! position past the durable tail, see below). Segments wholly below where
//! reading starts are retired and not read, so damage there does not fail the
//! open.
//!
//! A new segment's header and directory entry are fsynced before any record
//! goes in, so a short, headerless or all-zero-header *last* segment holds
//! nothing and is recreated.
//!
//! **Saved progress past the durable tail.** A log's progress (its retired
//! cut, per-namespace offsets) can name positions a power loss took from the
//! log: a flush can save progress covering appends whose sync failed, or that
//! were never synced. New appends must land above every position it names, or
//! replay would skip them, so `open` seals the last segment at the tail and
//! starts the next one when the furthest saved position is past the tail.

use crate::db::commit_rounds::CommitRounds;
use crate::db::segment_prep::{self, SegmentPrep};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use thiserror::Error;

/// Bytes of the segment header; the first record starts here.
pub(crate) const SEGMENT_HEADER_LEN: u32 = 16;
/// `len` and `crc32` before each body.
pub(crate) const FRAME_HEADER_LEN: u32 = 8;
/// The end marker of a sealed segment: this `len`, which no frame can have (a
/// frame must fit a segment). A `len` of 0 is unwritten space.
pub(crate) const END_MARKER: u32 = u32::MAX;
/// Bytes of the end marker.
pub(crate) const END_MARKER_LEN: u32 = 4;
/// Smallest segment size accepted: room for a header, one empty frame and the
/// end marker, rounded up.
pub(crate) const MIN_SEGMENT_SIZE: u32 = 64;

/// What tells one log's segments from another's: checked on every segment
/// read, so a file from another log, or written in another body layout, fails
/// loudly instead of being parsed.
#[derive(Debug)]
pub(crate) struct LogFormat {
    /// The log's name, in messages.
    pub name: &'static str,
    /// The first four bytes of each of its segments.
    pub magic: [u8; 4],
    /// The body layout's version; bump it whenever the layout changes.
    pub version: u32,
}

/// Errors from a [`SegmentedLog`], and from a log built on it: each log wraps
/// this in one variant of its own error type (naming the log), rather than
/// mirroring its variants.
#[derive(Error, Debug)]
pub enum LogError {
    /// An I/O error reading or writing the log.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// The log is damaged somewhere other than its torn tail (including a
    /// body that passes its CRC but does not decode, or a damaged progress
    /// file).
    #[error("corrupt: {0}")]
    Corruption(String),
    /// A record too large for one segment. Nothing was written.
    #[error("a record of {len} bytes does not fit a segment (at most {max})")]
    RecordTooLarge {
        /// The record's framed length.
        len: u64,
        /// The largest framed record a segment holds.
        max: u64,
    },
    /// An fsync or a rotation failed; the log refuses appends and commits
    /// until the database is reopened.
    #[error("stopped after a failed fsync ({0}); reopen the database to recover")]
    Poisoned(String),
}

/// Result type of this module.
pub(crate) type Result<T> = std::result::Result<T, LogError>;

/// A log position as the admin API shows it, for every log: the segment file
/// and the byte within it (the raw position is `(segment_id << 32) | byte`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct LogPosition {
    /// The segment file.
    pub segment_id: u64,
    /// The byte within it.
    pub byte: u32,
}

impl LogPosition {
    /// The position `pos` names.
    pub(crate) fn of(pos: u64) -> Self {
        let (segment_id, byte) = split(pos);
        Self { segment_id, byte }
    }
}

/// The position of byte `byte` of segment `segment_id`.
pub(crate) fn position(segment_id: u64, byte: u32) -> u64 {
    (segment_id << 32) | u64::from(byte)
}

/// A position's segment id and byte within it.
pub(crate) fn split(pos: u64) -> (u64, u32) {
    (pos >> 32, pos as u32)
}

/// The segment a position lies in.
pub(crate) fn segment_of(pos: u64) -> u64 {
    pos >> 32
}

/// The position of segment `segment_id`'s first record, past its header.
pub(crate) fn segment_start(segment_id: u64) -> u64 {
    position(segment_id, SEGMENT_HEADER_LEN)
}

/// Where reading starts at [`SegmentedLog::open`], from the log's saved progress.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OpenFrom {
    /// The retired cut: records below it are in SSTables and never read
    /// again. `None` when no progress was ever saved; then the log must start
    /// at segment 0, and it is read whole.
    pub cut: Option<u64>,
    /// The furthest position the saved progress names (the cut, or a
    /// namespace's offset above it). New appends land above it.
    pub furthest: u64,
}

/// A log's saved retirement progress: everything below `cut` is in SSTables,
/// and each namespace listed in `safe` has everything of its own below its
/// offset in SSTables too (offsets above the cut only, sorted by namespace).
/// Replay applies a record only at or above its namespace's
/// [`replay_from`](Self::replay_from).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Progress {
    /// Records below this are retired; GC deletes segments wholly below it.
    pub cut: u64,
    /// Namespaces whose safe offset is above the cut, sorted by namespace.
    pub safe: Vec<(u32, u64)>,
}

impl Progress {
    /// Where replay starts for namespace `ns`.
    pub(crate) fn replay_from(&self, ns: u32) -> u64 {
        self.safe
            .binary_search_by_key(&ns, |&(n, _)| n)
            .map_or(self.cut, |i| self.safe[i].1.max(self.cut))
    }

    /// The furthest position the progress names.
    pub(crate) fn furthest(&self) -> u64 {
        self.safe.iter().map(|&(_, s)| s).chain([self.cut]).max().unwrap_or(0)
    }

    /// Where [`SegmentedLog::open`] reads from, given the saved progress
    /// (`None`: none was ever saved).
    pub(crate) fn open_from(saved: Option<&Progress>) -> OpenFrom {
        OpenFrom {
            cut: saved.map(|p| p.cut),
            furthest: saved.map_or(0, Progress::furthest),
        }
    }
}

/// A log's progress file: [`Progress`] plus the log's own extra bytes (the
/// main WAL's counters; nothing for the vector index WAL), written atomically
/// and durably, checked on read. One format for every log, told apart by its
/// magic.
///
/// ```text
/// [magic 4][version u32][cut u64][count u32][count × (ns u32, safe u64)]
/// [extra_len u32][extra][crc32 of everything before it]
/// ```
pub(crate) struct ProgressFile {
    /// The file's first four bytes, per log.
    pub magic: [u8; 4],
}

/// The progress file layout's version.
const PROGRESS_VERSION: u32 = 3;
const PROGRESS_FIXED_LEN: usize = 4 + 4 + 8 + 4;
const PROGRESS_ENTRY_LEN: usize = 4 + 8;

impl ProgressFile {
    /// The file's bytes for `p` and `extra`.
    pub(crate) fn encode(&self, p: &Progress, extra: &[u8]) -> Vec<u8> {
        let mut b = Vec::with_capacity(PROGRESS_FIXED_LEN + p.safe.len() * PROGRESS_ENTRY_LEN + 4 + extra.len() + 4);
        b.extend_from_slice(&self.magic);
        b.extend_from_slice(&PROGRESS_VERSION.to_le_bytes());
        b.extend_from_slice(&p.cut.to_le_bytes());
        b.extend_from_slice(&(p.safe.len() as u32).to_le_bytes());
        for (ns, safe) in &p.safe {
            b.extend_from_slice(&ns.to_le_bytes());
            b.extend_from_slice(&safe.to_le_bytes());
        }
        b.extend_from_slice(&(extra.len() as u32).to_le_bytes());
        b.extend_from_slice(extra);
        let crc = crc32fast::hash(&b);
        b.extend_from_slice(&crc.to_le_bytes());
        b
    }

    /// Decode and check a progress file: its magic, version, checksum and
    /// length, that the namespaces are sorted and unique, and that no safe
    /// offset is below the cut. Any failure is corruption.
    pub(crate) fn decode(&self, bytes: &[u8]) -> Result<(Progress, Vec<u8>)> {
        let bad = |why: &str| LogError::Corruption(format!("the progress file is damaged: {why}"));
        if bytes.len() < PROGRESS_FIXED_LEN + 8 || bytes[0..4] != self.magic {
            return Err(bad("no header"));
        }
        let (body, crc) = bytes.split_at(bytes.len() - 4);
        if crc32fast::hash(body) != u32::from_le_bytes(crc.try_into().unwrap()) {
            return Err(bad("CRC mismatch"));
        }
        let version = u32::from_le_bytes(body[4..8].try_into().unwrap());
        if version != PROGRESS_VERSION {
            return Err(bad(&format!("version {version}, expected {PROGRESS_VERSION}")));
        }
        let cut = u64::from_le_bytes(body[8..16].try_into().unwrap());
        let count = u32::from_le_bytes(body[16..20].try_into().unwrap()) as usize;
        let entries_end = PROGRESS_FIXED_LEN
            .checked_add(count.checked_mul(PROGRESS_ENTRY_LEN).ok_or_else(|| bad("entry count overflows"))?)
            .filter(|&e| e + 4 <= body.len())
            .ok_or_else(|| bad("length does not match its entry count"))?;
        let safe: Vec<(u32, u64)> = body[PROGRESS_FIXED_LEN..entries_end]
            .as_chunks::<PROGRESS_ENTRY_LEN>()
            .0
            .iter()
            .map(|e| {
                (
                    u32::from_le_bytes(e[0..4].try_into().unwrap()),
                    u64::from_le_bytes(e[4..12].try_into().unwrap()),
                )
            })
            .collect();
        let extra_len = u32::from_le_bytes(body[entries_end..entries_end + 4].try_into().unwrap()) as usize;
        if body.len() != entries_end + 4 + extra_len {
            return Err(bad("length does not match its extra bytes"));
        }
        if safe.windows(2).any(|w| w[0].0 >= w[1].0) {
            return Err(bad("namespaces are not sorted"));
        }
        if let Some((ns, offset)) = safe.iter().find(|&&(_, s)| s < cut) {
            return Err(bad(&format!(
                "namespace {ns}'s safe offset {:?} is below the cut {:?}",
                LogPosition::of(*offset),
                LogPosition::of(cut)
            )));
        }
        Ok((Progress { cut, safe }, body[entries_end + 4..].to_vec()))
    }

    /// Read and decode the file at `path`; `None` if it does not exist.
    pub(crate) fn read(&self, path: &Path) -> Result<Option<(Progress, Vec<u8>)>> {
        match std::fs::read(path) {
            Ok(bytes) => self.decode(&bytes).map(Some),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Write `p` and `extra` to `path`, atomically and durably (temp file,
    /// fsync, rename, directory fsync).
    pub(crate) fn write(&self, path: &Path, p: &Progress, extra: &[u8]) -> std::io::Result<()> {
        crate::support::write_atomic_durable(path, &self.encode(p, extra))
    }
}

/// What [`SegmentedLog::open`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OpenReport {
    /// Position of the next append.
    pub tail: u64,
    /// The torn frame's length (capped at the end of its segment) when open
    /// found a torn tail, or the bytes of a last segment's unfinished header;
    /// 0 otherwise.
    pub torn_bytes_dropped: u64,
}

/// How a walk over one segment ended.
#[derive(Debug, PartialEq, Eq)]
enum WalkEnd {
    /// Reached unwritten space at `pos`: a zero length, or the limit, on a
    /// frame boundary.
    Unwritten { pos: u32 },
    /// Found the end marker at `pos`.
    Sealed { pos: u32 },
    /// Found a frame that is incomplete or fails its CRC at `pos`; it claims
    /// `bytes` (at most the bytes left).
    Torn { pos: u32, bytes: u32, reason: String },
}

struct Active {
    segment_id: u64,
    file: Arc<File>,
    /// Next append's byte within the segment.
    pos: u32,
}

/// See the module docs.
pub(crate) struct SegmentedLog {
    dir: PathBuf,
    format: &'static LogFormat,
    segment_size: u32,
    /// The append lock and the active segment.
    active: Mutex<Active>,
    /// Position of the next append. Stored under `active`, after the append's
    /// `under_lock` step, so a reader that sees a tail including a record also
    /// sees whatever that step registered. Lets readers take the tail without
    /// waiting on `active`, which a rotation holds across fsyncs.
    tail: AtomicU64,
    /// Syncs as commit rounds: the durable position, coalescing, poisoning.
    pub(crate) rounds: CommitRounds,
    /// The next segment, zero-filled ahead.
    pub(crate) prep: SegmentPrep,
    /// Where each sealed segment at or above where reading started ends (its
    /// end marker): a segment's file is its full size whatever it holds.
    sealed_ends: Mutex<BTreeMap<u64, u32>>,
    /// Test hook for the power-loss model: per segment, how far frames and end
    /// markers have been written, and how much of that an fsync issued by this
    /// process covered; and the first segment this process could write (every
    /// earlier one survived a reopen, so it is durable).
    #[cfg(test)]
    written: Mutex<BTreeMap<u64, u32>>,
    #[cfg(test)]
    synced: Mutex<BTreeMap<u64, u32>>,
    #[cfg(test)]
    first_open_segment: u64,
    /// Test-only fault: the next fsync fails.
    #[cfg(test)]
    fail_next_sync: std::sync::atomic::AtomicBool,
    /// Test-only: every fsync takes at least this long (µs), as on a real
    /// disk, so tests of fsync sharing do not depend on the machine.
    #[cfg(test)]
    fsync_delay_us: AtomicU64,
}

/// A segment file's path in `dir`.
pub(crate) fn segment_path(dir: &Path, segment_id: u64) -> PathBuf {
    dir.join(format!("{segment_id:010}.log"))
}

/// Where segment `segment_id` is zero-filled before it is needed.
pub(crate) fn prep_path(dir: &Path, segment_id: u64) -> PathBuf {
    dir.join(format!("{segment_id:010}.log.prep"))
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

/// Whether `dir` holds any segment file (a log that may have records to
/// replay). A missing directory holds none.
pub(crate) fn has_segments(dir: &Path) -> std::io::Result<bool> {
    match list_segments(dir) {
        Ok(ids) => Ok(!ids.is_empty()),
        Err(LogError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(LogError::Io(e)) => Err(e),
        Err(e) => Err(std::io::Error::other(e.to_string())),
    }
}

fn sync_dir(dir: &Path) -> Result<()> {
    File::open(dir)?.sync_all()?;
    Ok(())
}

fn file_len(file: &File) -> Result<u32> {
    let len = file.metadata()?.len();
    u32::try_from(len).map_err(|_| LogError::Corruption(format!("segment of {len} bytes")))
}

/// The header of segment `segment_id` of a log in `format`.
pub(crate) fn header_bytes(format: &LogFormat, segment_id: u64) -> [u8; SEGMENT_HEADER_LEN as usize] {
    let mut h = [0u8; SEGMENT_HEADER_LEN as usize];
    h[0..4].copy_from_slice(&format.magic);
    h[4..8].copy_from_slice(&format.version.to_le_bytes());
    h[8..16].copy_from_slice(&segment_id.to_le_bytes());
    h
}

/// Checks a segment's header. The file must be at least a header long.
pub(crate) fn check_header(format: &LogFormat, file: &File, segment_id: u64) -> Result<()> {
    let mut h = [0u8; SEGMENT_HEADER_LEN as usize];
    file.read_exact_at(&mut h, 0)?;
    if h[0..4] != format.magic {
        return Err(LogError::Corruption(format!("segment {segment_id} has no {} header", format.name)));
    }
    let version = u32::from_le_bytes(h[4..8].try_into().unwrap());
    if version != format.version {
        return Err(LogError::Corruption(format!(
            "{} segment {segment_id} has format version {version}, expected {}",
            format.name, format.version
        )));
    }
    let id = u64::from_le_bytes(h[8..16].try_into().unwrap());
    if id != segment_id {
        return Err(LogError::Corruption(format!(
            "{} segment file {segment_id} holds segment {id}",
            format.name
        )));
    }
    Ok(())
}

/// Whether a segment's header is all zeros: a segment whose creation a power
/// loss cut short (its header is fsynced before any record goes in).
fn header_is_blank(file: &File) -> Result<bool> {
    let mut h = [0u8; SEGMENT_HEADER_LEN as usize];
    file.read_exact_at(&mut h, 0)?;
    Ok(h.iter().all(|&b| b == 0))
}

fn check_segment_id(segment_id: u64) -> Result<()> {
    if segment_id > u64::from(u32::MAX) {
        return Err(LogError::Corruption("segment ids exhausted".to_owned()));
    }
    Ok(())
}

/// Creates segment `segment_id` with its header, at its full `size` (sparse),
/// durable (contents, size and name) before it returns, so a record in it
/// always has a header behind it and a sync never commits a new size.
fn create_segment(format: &LogFormat, dir: &Path, segment_id: u64, size: u32) -> Result<File> {
    check_segment_id(segment_id)?;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(segment_path(dir, segment_id))?;
    file.write_all_at(&header_bytes(format, segment_id), 0)?;
    file.set_len(u64::from(size))?;
    file.sync_all()?;
    sync_dir(dir)?;
    Ok(file)
}

/// Zero every byte of the last segment from the tail `pos` to the end of its
/// file (`len` bytes): shrink the file to the tail, then size it back, to at
/// least `segment_size`. So nothing past the tail can be read as a record once
/// new records land behind it. The caller fsyncs.
fn zero_past(file: &File, pos: u32, len: u32, segment_size: u32) -> Result<()> {
    if pos < len {
        file.set_len(u64::from(pos))?;
    }
    file.set_len(u64::from(len.max(segment_size)))?;
    Ok(())
}

/// Bytes a walk reads at a time: frames are parsed out of this window, so a
/// run of small records costs one read per window rather than one or more per
/// record.
const READ_AHEAD: u32 = 64 * 1024;

/// Make `buf` hold bytes `[pos, pos + need)` of `file`. `held` is the range it
/// holds now; a miss reads `max(need, READ_AHEAD)` bytes from `pos`, never past
/// `limit` (the caller has checked `need <= limit - pos`). Returns where `pos`
/// is in `buf`.
fn window(file: &File, buf: &mut Vec<u8>, held: &mut (u32, u32), pos: u32, need: u32, limit: u32) -> std::io::Result<usize> {
    if pos < held.0 || pos + need > held.1 {
        let len = need.max(READ_AHEAD).min(limit - pos);
        buf.resize(len as usize, 0);
        file.read_exact_at(&mut buf[..len as usize], u64::from(pos))?;
        *held = (pos, pos + len);
    }
    Ok((pos - held.0) as usize)
}

/// Walks the frames of one segment from byte `start` up to byte `limit`,
/// handing each body and its position to `f`. The one frame-parsing loop:
/// opening and reading both use it, so the torn-tail and end-marker rules
/// exist once. Reads ahead in [`READ_AHEAD`] windows (never past `limit`, so a
/// reader bounded by the tail never reads a frame still being written).
///
/// Every length is checked against the bytes left before anything is
/// allocated for it (a corrupt length could otherwise ask for 4 GiB).
fn walk_segment<E>(
    file: &File,
    segment_id: u64,
    start: u32,
    limit: u32,
    buf: &mut Vec<u8>,
    lift: impl Fn(LogError) -> E,
    mut f: impl FnMut(u64, &[u8]) -> std::result::Result<(), E>,
) -> std::result::Result<WalkEnd, E> {
    let io = |e: std::io::Error| lift(LogError::Io(e));
    // The file bytes `buf` holds; empty to start (the buffer is reused).
    let mut held = (0u32, 0u32);
    let mut pos = start;
    loop {
        if pos >= limit {
            return Ok(WalkEnd::Unwritten { pos });
        }
        let left = limit - pos;
        if left < END_MARKER_LEN {
            return Ok(WalkEnd::Torn {
                pos,
                bytes: left,
                reason: format!("{left} bytes where a frame length belongs"),
            });
        }
        let at = window(file, buf, &mut held, pos, left.min(FRAME_HEADER_LEN), limit).map_err(io)?;
        let len = u32::from_le_bytes(buf[at..at + 4].try_into().unwrap());
        if len == 0 {
            return Ok(WalkEnd::Unwritten { pos });
        }
        if len == END_MARKER {
            return Ok(WalkEnd::Sealed { pos });
        }
        if left < FRAME_HEADER_LEN || len > left - FRAME_HEADER_LEN {
            return Ok(WalkEnd::Torn {
                pos,
                bytes: left,
                reason: format!("frame length {len} runs past the end ({left} bytes left)"),
            });
        }
        let at = window(file, buf, &mut held, pos, FRAME_HEADER_LEN + len, limit).map_err(io)?;
        let crc = u32::from_le_bytes(buf[at + 4..at + 8].try_into().unwrap());
        let body = &buf[at + FRAME_HEADER_LEN as usize..at + (FRAME_HEADER_LEN + len) as usize];
        if crc32fast::hash(body) != crc {
            return Ok(WalkEnd::Torn {
                pos,
                bytes: FRAME_HEADER_LEN + len,
                reason: "CRC mismatch".to_owned(),
            });
        }
        f(position(segment_id, pos), body)?;
        pos += FRAME_HEADER_LEN + len;
    }
}

/// The end of a sealed segment's records, its end marker, when that lies at or
/// below `from`; `None` otherwise. For a walk of a sealed segment that found
/// unwritten space right where it started: fine when reading starts past the
/// segment's end marker (see the module docs).
fn sealed_at_or_below(file: &File, segment_id: u64, from: u32, len: u32, buf: &mut Vec<u8>) -> Result<Option<u32>> {
    let limit = from.saturating_add(END_MARKER_LEN).min(len);
    match walk_segment(file, segment_id, SEGMENT_HEADER_LEN, limit, buf, |e| e, |_, _| Ok::<(), LogError>(()))? {
        WalkEnd::Sealed { pos } if pos <= from => Ok(Some(pos)),
        _ => Ok(None),
    }
}

impl SegmentedLog {
    /// Opens the log in `dir`, creating it if absent, and finds its tail.
    ///
    /// Validates the segments at or above `from.cut` and hands every record
    /// there to `visit`, in log order (records of the last segment below the
    /// cut are read to find the tail, but not visited); segments wholly below
    /// the cut are retired and not read. A torn tail in the last segment is
    /// dropped and everything past the tail zeroed; any other damage fails the
    /// open (see the module docs). A last segment that was sealed but has no
    /// successor (a crash mid-rotation) gets one, as does a log whose saved
    /// progress names positions past the tail. The last segment is fsynced
    /// before returning, so everything below the tail is durable.
    pub(crate) fn open<E: From<LogError>>(
        dir: &Path,
        format: &'static LogFormat,
        segment_size: u32,
        from: OpenFrom,
        mut visit: impl FnMut(u64, &[u8]) -> std::result::Result<(), E>,
    ) -> std::result::Result<(Self, OpenReport), E> {
        let corrupt = |why: String| E::from(LogError::Corruption(why));
        let segment_size = segment_size.max(MIN_SEGMENT_SIZE);
        if !dir.exists() {
            std::fs::create_dir_all(dir).map_err(LogError::from)?;
            if let Some(parent) = dir.parent().filter(|p| !p.as_os_str().is_empty()) {
                sync_dir(parent)?;
            }
        }

        // Segments a previous run left half-prepared are not part of the log.
        segment_prep::remove_leftovers(dir, "");
        let ids = list_segments(dir)?;
        if let Some(w) = ids.windows(2).find(|w| w[1] != w[0] + 1) {
            return Err(corrupt(format!("segment {} is missing (segments {} and {} exist)", w[0] + 1, w[0], w[1])));
        }
        let oldest = ids.first().copied();
        match (from.cut, oldest) {
            (Some(cut), Some(oldest)) if segment_of(cut) < oldest => {
                return Err(corrupt(format!("the retired cut {cut:#x} is below the oldest segment {oldest}")));
            }
            (None, Some(oldest)) if oldest > 0 => {
                return Err(corrupt(format!("no progress was saved but segments start at {oldest}")));
            }
            _ => {}
        }
        let cut = from.cut.unwrap_or(0);
        let (cut_id, cut_byte) = split(cut);

        let mut buf = Vec::new();
        let mut visit_from_cut = |at: u64, body: &[u8]| if at < cut { Ok(()) } else { visit(at, body) };

        // Sealed segments: complete, ending in the end marker.
        let (sealed, last) = match ids.split_last() {
            Some((&last, sealed)) => (sealed, Some(last)),
            None => (&[][..], None),
        };
        let mut sealed_ends = BTreeMap::new();
        for &id in sealed {
            let file = File::open(segment_path(dir, id)).map_err(LogError::from)?;
            let len = file_len(&file)?;
            if len < SEGMENT_HEADER_LEN {
                return Err(corrupt(format!("segment {id} is shorter than its header")));
            }
            check_header(format, &file, id)?;
            let start = match id.cmp(&cut_id) {
                std::cmp::Ordering::Less => continue, // wholly retired
                std::cmp::Ordering::Equal => cut_byte.max(SEGMENT_HEADER_LEN),
                std::cmp::Ordering::Greater => SEGMENT_HEADER_LEN,
            };
            match walk_segment(&file, id, start, len, &mut buf, E::from, &mut visit_from_cut)? {
                WalkEnd::Sealed { pos } => {
                    sealed_ends.insert(id, pos);
                }
                WalkEnd::Unwritten { pos } => {
                    // Unwritten space where reading starts, past the end
                    // marker: the cut is past this segment's end (progress
                    // naming records a power loss took, after which an earlier
                    // open started a new segment). Anywhere else, the segment
                    // was never sealed.
                    match sealed_at_or_below(&file, id, start, len, &mut buf)? {
                        Some(end) if pos == start && start > SEGMENT_HEADER_LEN => {
                            sealed_ends.insert(id, end);
                        }
                        _ => return Err(corrupt(format!("segment {id} is not sealed but is not the last"))),
                    }
                }
                WalkEnd::Torn { pos, reason, .. } => {
                    return Err(corrupt(format!("segment {id} (not the last) is damaged at byte {pos}: {reason}")));
                }
            }
        }

        // The last segment: the active one, possibly torn. Read in full, to
        // find the tail.
        let mut torn_bytes_dropped = 0u64;
        let (mut segment_id, mut file, mut pos) = match last {
            None => (0, create_segment(format, dir, 0, segment_size)?, SEGMENT_HEADER_LEN),
            Some(id) => {
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(segment_path(dir, id))
                    .map_err(LogError::from)?;
                let len = file_len(&file)?;
                if len < SEGMENT_HEADER_LEN || header_is_blank(&file)? {
                    // A crash while creating the segment: its header was not
                    // yet durable, so no record can be in it.
                    torn_bytes_dropped = u64::from(len.min(SEGMENT_HEADER_LEN));
                    drop(file);
                    (id, create_segment(format, dir, id, segment_size)?, SEGMENT_HEADER_LEN)
                } else {
                    check_header(format, &file, id)?;
                    match walk_segment(&file, id, SEGMENT_HEADER_LEN, len, &mut buf, E::from, &mut visit_from_cut)? {
                        WalkEnd::Unwritten { pos } => {
                            zero_past(&file, pos, len, segment_size)?;
                            (id, file, pos)
                        }
                        WalkEnd::Torn { pos, bytes, reason } => {
                            log::warn!(
                                "[{}] dropping a torn tail: segment {id}, byte {pos}, a {bytes}-byte frame ({reason})",
                                format.name
                            );
                            zero_past(&file, pos, len, segment_size)?;
                            torn_bytes_dropped = u64::from(bytes);
                            (id, file, pos)
                        }
                        WalkEnd::Sealed { pos } => {
                            // A crash after sealing, before the next segment existed.
                            sealed_ends.insert(id, pos);
                            drop(file);
                            (id + 1, create_segment(format, dir, id + 1, segment_size)?, SEGMENT_HEADER_LEN)
                        }
                    }
                }
            }
        };
        file.sync_all().map_err(LogError::from)?;

        if segment_of(from.furthest) > segment_id {
            return Err(corrupt(format!(
                "saved progress names {:#x}, past the last segment {segment_id}",
                from.furthest
            )));
        }
        if from.furthest > position(segment_id, pos) {
            // A power loss took records whose progress was saved: their data
            // is in SSTables, but new appends must land above every position
            // the progress names, or replay would skip them.
            log::warn!(
                "[{}] saved progress names {:#x}, past the durable tail {:#x}; starting segment {}",
                format.name,
                from.furthest,
                position(segment_id, pos),
                segment_id + 1
            );
            file.write_all_at(&END_MARKER.to_le_bytes(), u64::from(pos)).map_err(LogError::from)?;
            file.sync_all().map_err(LogError::from)?;
            sealed_ends.insert(segment_id, pos);
            segment_id += 1;
            file = create_segment(format, dir, segment_id, segment_size)?;
            pos = SEGMENT_HEADER_LEN;
        }
        let tail = position(segment_id, pos);
        let log = Self {
            dir: dir.to_path_buf(),
            format,
            segment_size,
            active: Mutex::new(Active {
                segment_id,
                file: Arc::new(file),
                pos,
            }),
            tail: AtomicU64::new(tail),
            rounds: CommitRounds::new(tail),
            prep: SegmentPrep::new(format.name),
            sealed_ends: Mutex::new(sealed_ends),
            #[cfg(test)]
            written: Mutex::new(BTreeMap::from([(segment_id, pos)])),
            // `open` fsynced the active segment up to the tail.
            #[cfg(test)]
            synced: Mutex::new(BTreeMap::from([(segment_id, pos)])),
            #[cfg(test)]
            first_open_segment: segment_id,
            #[cfg(test)]
            fail_next_sync: std::sync::atomic::AtomicBool::new(false),
            #[cfg(test)]
            fsync_delay_us: AtomicU64::new(0),
        };
        Ok((log, OpenReport { tail, torn_bytes_dropped }))
    }

    /// The directory holding the segments.
    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }

    /// The size new segments are created at.
    pub(crate) fn segment_size(&self) -> u32 {
        self.segment_size
    }

    /// The largest framed record a segment holds.
    pub(crate) fn max_frame_len(&self) -> u64 {
        u64::from(self.segment_size - SEGMENT_HEADER_LEN - END_MARKER_LEN)
    }

    /// Appends one record whose body is the concatenation of `body`. Not
    /// durable until a later [`commit`](Self::commit) or [`sync`](Self::sync)
    /// covering it returns. Returns where the frame landed, `[start, end)`.
    ///
    /// `under_lock(start, end)` runs under the append lock after the frame is
    /// written and before the new tail is published, for a log that must
    /// register the record (as not yet applied) before anyone can see a tail
    /// that includes it.
    ///
    /// On a failed write the tail stays put, so the next append overwrites the
    /// partial bytes. A failed rotation poisons the log.
    pub(crate) fn append(&self, body: &[&[u8]], under_lock: impl FnOnce(u64, u64)) -> Result<(u64, u64)> {
        let body_len: usize = body.iter().map(|p| p.len()).sum();
        let frame_len = u64::from(FRAME_HEADER_LEN) + body_len as u64;
        if frame_len > self.max_frame_len() {
            return Err(LogError::RecordTooLarge {
                len: frame_len,
                max: self.max_frame_len(),
            });
        }
        let mut frame = Vec::with_capacity(frame_len as usize);
        let mut crc = crc32fast::Hasher::new();
        for part in body {
            crc.update(part);
        }
        frame.extend_from_slice(&(body_len as u32).to_le_bytes());
        frame.extend_from_slice(&crc.finalize().to_le_bytes());
        for part in body {
            frame.extend_from_slice(part);
        }
        let frame_len = frame_len as u32;

        let mut a = self.active.lock();
        self.check_poisoned()?;
        if u64::from(a.pos) + u64::from(frame_len) + u64::from(END_MARKER_LEN) > u64::from(self.segment_size) {
            self.rotate(&mut a)?;
        }
        a.file.write_all_at(&frame, u64::from(a.pos))?;
        let start = position(a.segment_id, a.pos);
        a.pos += frame_len;
        #[cfg(test)]
        self.note_written(a.segment_id, a.pos);
        let next = a.segment_id + 1;
        if u64::from(a.pos) * 2 >= u64::from(self.segment_size) && !self.prep.started(next) {
            self.prep.start(
                next,
                prep_path(&self.dir, next),
                u64::from(self.segment_size),
                header_bytes(self.format, next).to_vec(),
            );
        }
        let end = position(a.segment_id, a.pos);
        under_lock(start, end);
        self.tail.store(end, Ordering::Release);
        Ok((start, end))
    }

    /// Seals the active segment and starts the next. The sealed segment is
    /// fsynced first, so only the last segment can ever hold unsynced bytes.
    ///
    /// Any failure poisons the log: past the end marker's write the segment is
    /// sealed on disk, and a later append laid over the marker would leave a
    /// sealed segment that is not sealed, with a successor.
    fn rotate(&self, a: &mut Active) -> Result<()> {
        self.try_rotate(a).map_err(|e| match e {
            LogError::Poisoned(why) => LogError::Poisoned(why),
            other => self.poison(format!("rotating to segment {}: {other}", a.segment_id + 1)),
        })
    }

    fn try_rotate(&self, a: &mut Active) -> Result<()> {
        a.file.write_all_at(&END_MARKER.to_le_bytes(), u64::from(a.pos))?;
        #[cfg(test)]
        self.note_written(a.segment_id, a.pos + END_MARKER_LEN);
        self.sync_file(&a.file, a.segment_id)?;
        self.rounds.advance(position(a.segment_id, a.pos));
        self.sealed_ends.lock().insert(a.segment_id, a.pos);
        let next = a.segment_id + 1;
        check_segment_id(next)?;
        let path = segment_path(&self.dir, next);
        let file = if self.prep.install(next, &path)? {
            OpenOptions::new().read(true).write(true).open(&path)?
        } else {
            create_segment(self.format, &self.dir, next, self.segment_size)?
        };
        // Its header is durable either way.
        #[cfg(test)]
        {
            self.written.lock().insert(next, SEGMENT_HEADER_LEN);
            self.synced.lock().insert(next, SEGMENT_HEADER_LEN);
        }
        a.file = Arc::new(file);
        a.segment_id = next;
        a.pos = SEGMENT_HEADER_LEN;
        self.tail.store(segment_start(next), Ordering::Release);
        Ok(())
    }

    /// Position of the next append.
    pub(crate) fn tail(&self) -> u64 {
        self.tail.load(Ordering::Acquire)
    }

    /// Every byte below this position is on stable storage.
    pub(crate) fn durable(&self) -> u64 {
        self.rounds.durable()
    }

    /// The active segment: the one appends go to.
    pub(crate) fn active_segment(&self) -> u64 {
        segment_of(self.tail())
    }

    /// Waits until every byte below `end` is durable, as a commit round's
    /// leader or follower (`db/commit_rounds.rs`); `delay` is the round's
    /// optional wait for more callers. Returns `true` if this caller issued
    /// the round's fsync. A failed fsync poisons the log.
    pub(crate) fn commit(&self, end: u64, delay: Duration) -> Result<bool> {
        let led = self
            .rounds
            .commit(end, delay, || {
                // Earlier segments were fsynced when they were sealed; the
                // active one holds every append made before this snapshot.
                let (file, segment_id, tail) = {
                    let a = self.active.lock();
                    (Arc::clone(&a.file), a.segment_id, position(a.segment_id, a.pos))
                };
                self.sync_file(&file, segment_id).map_err(|e| {
                    let why = format!("an fsync of segment {segment_id} failed: {e}");
                    log::error!(
                        "[{}] stopping the log; appends are refused until the database is reopened: {why}",
                        self.format.name
                    );
                    why
                })?;
                Ok(tail)
            })
            .map_err(LogError::Poisoned)?;
        #[cfg(test)]
        assert!(
            self.covered(end),
            "[{}] a commit returned before an fsync covered {end:#x}",
            self.format.name
        );
        Ok(led)
    }

    /// Makes every append that returned before this call durable. Does no I/O
    /// when nothing was appended since the last fsync. Returns `true` if this
    /// call issued an fsync.
    pub(crate) fn sync(&self) -> Result<bool> {
        self.commit(self.tail(), Duration::ZERO)
    }

    /// fsync segment `segment_id`'s `file`; the caller poisons the log on an
    /// error. `segment_id` feeds only the test-only power-loss model.
    #[cfg_attr(not(test), allow(unused_variables))]
    fn sync_file(&self, file: &File, segment_id: u64) -> Result<()> {
        #[cfg(test)]
        if self.fail_next_sync.swap(false, Ordering::SeqCst) {
            return Err(std::io::Error::other("injected fsync failure").into());
        }
        #[cfg(test)]
        {
            let delay = self.fsync_delay_us.load(Ordering::Relaxed);
            if delay > 0 {
                std::thread::sleep(Duration::from_micros(delay));
            }
        }
        // What this fsync covers, for the power-loss model: what was written
        // before it starts (bytes written later may or may not be covered).
        #[cfg(test)]
        let covered = self.written.lock().get(&segment_id).copied().unwrap_or(0);
        file.sync_data()?;
        #[cfg(test)]
        {
            let mut synced = self.synced.lock();
            let entry = synced.entry(segment_id).or_insert(0);
            *entry = (*entry).max(covered);
        }
        Ok(())
    }

    /// `Err(Poisoned)` once the log has been stopped.
    pub(crate) fn check_poisoned(&self) -> Result<()> {
        self.rounds.check_poisoned().map_err(LogError::Poisoned)
    }

    /// The reason the log is stopped, if it is.
    pub(crate) fn poisoned(&self) -> Option<String> {
        self.rounds.poisoned()
    }

    /// Stop the log: every waiting and later append and commit fails until
    /// reopen. Returns the error to report (the first reason recorded).
    pub(crate) fn poison(&self, why: String) -> LogError {
        log::error!(
            "[{}] stopping the log; appends are refused until the database is reopened: {why}",
            self.format.name
        );
        LogError::Poisoned(self.rounds.poison(why))
    }

    /// [`for_each_with`](Self::for_each_with) for a caller whose error type
    /// converts from [`LogError`].
    pub(crate) fn for_each<E: From<LogError>>(
        &self,
        start: u64,
        end: u64,
        f: impl FnMut(u64, &[u8]) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        self.for_each_with(start, end, E::from, f)
    }

    /// Hands every record in `[start, end)` to `f`, in log order. `start` is a
    /// record's position or a segment's start; `end` is at most the tail.
    /// `lift` turns the log's own errors into the caller's error type, so a
    /// log whose caller's error does not convert from [`LogError`] directly
    /// needs no wrapper type.
    /// Segments wholly below the oldest one on disk are retired and skipped
    /// (a reader racing GC skips a segment GC deletes under it the same way),
    /// so a caller that must know about them asks
    /// [`missing_segments`](Self::missing_segments). A missing segment above
    /// the oldest, or a damaged record, is corruption.
    ///
    /// Streams with one reused buffer, so a long read holds one record at a
    /// time.
    pub(crate) fn for_each_with<E>(
        &self,
        start: u64,
        end: u64,
        lift: impl Fn(LogError) -> E,
        mut f: impl FnMut(u64, &[u8]) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        let corrupt = |why: String| lift(LogError::Corruption(why));
        if end <= start {
            return Ok(());
        }
        let (end_id, end_pos) = split(end);
        let mut start = start;
        let mut buf = Vec::new();
        let mut id = segment_of(start);
        while id <= end_id {
            let file = match File::open(segment_path(&self.dir, id)) {
                Ok(file) => file,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    // Retired and deleted (now, or before the call): skip to
                    // the oldest segment still on disk.
                    match list_segments(&self.dir).map_err(&lift)?.first() {
                        Some(&oldest) if oldest > id => {
                            id = oldest;
                            start = segment_start(oldest);
                            continue;
                        }
                        _ => return Err(corrupt(format!("segment {id} is missing"))),
                    }
                }
                Err(e) => return Err(lift(LogError::Io(e))),
            };
            check_header(self.format, &file, id).map_err(&lift)?;
            let from = if id == segment_of(start) {
                split(start).1.max(SEGMENT_HEADER_LEN)
            } else {
                SEGMENT_HEADER_LEN
            };
            let len = file_len(&file).map_err(&lift)?;
            let limit = if id == end_id { end_pos.min(len) } else { len };
            match walk_segment(&file, id, from, limit, &mut buf, &lift, &mut f)? {
                WalkEnd::Unwritten { .. } if id == end_id => {}
                WalkEnd::Sealed { .. } => {}
                // A start past a sealed segment's end marker (see `open`).
                WalkEnd::Unwritten { pos }
                    if pos == from && from > SEGMENT_HEADER_LEN && sealed_at_or_below(&file, id, from, len, &mut buf).map_err(&lift)?.is_some() => {}
                WalkEnd::Torn { pos, reason, .. } => {
                    return Err(corrupt(format!("segment {id} is damaged at byte {pos}: {reason}")));
                }
                other => return Err(corrupt(format!("segment {id} ends unexpectedly: {other:?}"))),
            }
            id += 1;
        }
        Ok(())
    }

    /// Where segment `segment_id`'s records end: its end marker if it is
    /// sealed, the tail if it is the active segment, `None` if unknown (a
    /// segment below where this open started reading).
    pub(crate) fn segment_end(&self, segment_id: u64) -> Option<u64> {
        let tail = self.tail();
        if segment_id == segment_of(tail) {
            return Some(tail);
        }
        self.sealed_ends.lock().get(&segment_id).map(|&pos| position(segment_id, pos))
    }

    /// Bytes of records in `[from, to)`: segment headers and end markers
    /// excluded. Segments whose end is unknown (below where this open started
    /// reading) count nothing.
    pub(crate) fn bytes_between(&self, from: u64, to: u64) -> u64 {
        if to <= from {
            return 0;
        }
        let (to_id, to_byte) = split(to);
        let sealed_ends = self.sealed_ends.lock();
        (segment_of(from)..=to_id)
            .map(|id| {
                let end = if id == to_id { Some(to_byte) } else { sealed_ends.get(&id).copied() };
                let start = if id == segment_of(from) {
                    split(from).1.max(SEGMENT_HEADER_LEN)
                } else {
                    SEGMENT_HEADER_LEN
                };
                end.map_or(0, |end| u64::from(end.saturating_sub(start)))
            })
            .sum()
    }

    /// Segment ids in `[segment_of(start), segment_of(end)]` whose files are
    /// absent. Checked against the filesystem, so a file lost any way is
    /// reported, not only one GC deleted.
    pub(crate) fn missing_segments(&self, start: u64, end: u64) -> Vec<u64> {
        if end <= start {
            return Vec::new();
        }
        (segment_of(start)..=segment_of(end))
            .filter(|&id| !segment_path(&self.dir, id).exists())
            .collect()
    }

    /// Every segment on disk, `(segment_id, bytes)`, oldest first. A segment
    /// GC deletes while this runs is left out.
    pub(crate) fn segments(&self) -> Result<Vec<(u64, u64)>> {
        let mut out = Vec::new();
        for id in list_segments(&self.dir)? {
            match std::fs::metadata(segment_path(&self.dir, id)) {
                Ok(m) => out.push((id, m.len())),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
        Ok(out)
    }

    /// The oldest segment on disk.
    pub(crate) fn oldest_segment(&self) -> Result<Option<u64>> {
        Ok(list_segments(&self.dir)?.first().copied())
    }

    /// Deletes every segment below `segment_id` (never the active one), oldest
    /// first, then fsyncs the directory. Stops at the first failure, so what
    /// is left is always one contiguous run. Returns `(segments, bytes)`
    /// deleted. The caller decides what is retired.
    pub(crate) fn delete_segments_below(&self, segment_id: u64) -> Result<(u64, u64)> {
        let below = segment_id.min(self.active_segment());
        let (mut segments, mut bytes) = (0u64, 0u64);
        let mut failed = None;
        for id in list_segments(&self.dir)?.into_iter().filter(|&id| id < below) {
            let path = segment_path(&self.dir, id);
            let len = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            match std::fs::remove_file(&path) {
                Ok(()) => {
                    segments += 1;
                    bytes += len;
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
        }
        if segments > 0 {
            sync_dir(&self.dir)?;
        }
        self.sealed_ends.lock().retain(|&id, _| id >= below);
        match failed {
            Some(e) if segments == 0 => Err(e.into()),
            Some(e) => {
                // What was deleted is still one run from the oldest up; the
                // rest is retried by the next pass.
                log::warn!(
                    "[{}] deleted {segments} retired segment(s), then could not delete the next: {e}",
                    self.format.name
                );
                Ok((segments, bytes))
            }
            None => Ok((segments, bytes)),
        }
    }

    #[cfg(test)]
    fn note_written(&self, segment_id: u64, end: u32) {
        let mut written = self.written.lock();
        let entry = written.entry(segment_id).or_insert(0);
        *entry = (*entry).max(end);
    }

    /// Test hook: whether fsyncs issued by this process (or the reopen) cover
    /// every byte below `end`: each earlier segment up to what was written in
    /// it (its end marker), and `end`'s own segment up to `end`.
    #[cfg(test)]
    pub(crate) fn covered(&self, end: u64) -> bool {
        let (seg, byte) = split(end);
        if seg < self.first_open_segment {
            return true;
        }
        let written = self.written.lock();
        let synced = self.synced.lock();
        let synced_to = |s: u64| synced.get(&s).copied().unwrap_or(0);
        (self.first_open_segment..seg).all(|s| synced_to(s) >= written.get(&s).copied().unwrap_or(0)) && synced_to(seg) >= byte
    }

    /// Test hook for the power-loss model: for every segment this process
    /// could have written, its file and how many of its bytes an fsync
    /// actually covered. Taken from the fsyncs issued, not from the durable
    /// position, so a missing fsync shows up as lost bytes.
    #[cfg(test)]
    pub(crate) fn power_loss_extents(&self) -> Vec<(PathBuf, u64)> {
        let last = self.active.lock().segment_id;
        let synced = self.synced.lock();
        (self.first_open_segment..=last)
            .map(|id| (segment_path(&self.dir, id), u64::from(synced.get(&id).copied().unwrap_or(0))))
            .collect()
    }

    /// Test-only: zero every byte no fsync covered, in each segment this
    /// process wrote, keeping the files' sizes, as a power loss would.
    #[cfg(test)]
    pub(crate) fn lose_unsynced(&self) {
        for (path, durable) in self.power_loss_extents() {
            let file = OpenOptions::new().write(true).open(&path).expect("open a segment");
            let len = file.metadata().expect("segment metadata").len();
            if len > durable {
                file.set_len(durable).expect("cut a segment");
                file.set_len(len).expect("size a segment back");
            }
        }
    }

    /// Test-only: make the next fsync fail (and so poison the log).
    #[cfg(test)]
    pub(crate) fn fail_next_sync(&self) {
        self.fail_next_sync.store(true, Ordering::SeqCst);
    }

    /// Test-only: make every fsync take at least `delay`.
    #[cfg(test)]
    pub(crate) fn set_fsync_delay(&self, delay: Duration) {
        self.fsync_delay_us.store(delay.as_micros() as u64, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    static FORMAT: LogFormat = LogFormat {
        name: "test log",
        magic: *b"TLOG",
        version: 1,
    };
    static OTHER: LogFormat = LogFormat {
        name: "other log",
        magic: *b"OLOG",
        version: 1,
    };
    const SEGMENT: u32 = 256;
    const FRESH: OpenFrom = OpenFrom { cut: None, furthest: 0 };

    type Records = Vec<(u64, Vec<u8>)>;

    fn open(dir: &Path, from: OpenFrom) -> Result<(SegmentedLog, OpenReport, Records)> {
        let mut seen = Vec::new();
        let (log, report) = SegmentedLog::open(dir, &FORMAT, SEGMENT, from, |at, body| -> Result<()> {
            seen.push((at, body.to_vec()));
            Ok(())
        })?;
        Ok((log, report, seen))
    }

    fn body(i: u32) -> Vec<u8> {
        format!("record-{i:03}-{}", "x".repeat((i % 7) as usize)).into_bytes()
    }

    fn append_n(log: &SegmentedLog, range: std::ops::Range<u32>) -> Records {
        range
            .map(|i| {
                let b = body(i);
                let (start, _) = log.append(&[&b], |_, _| {}).unwrap();
                (start, b)
            })
            .collect()
    }

    fn read_all(log: &SegmentedLog, start: u64) -> Records {
        let mut out = Vec::new();
        log.for_each(start, log.tail(), |at, b| -> Result<()> {
            out.push((at, b.to_vec()));
            Ok(())
        })
        .unwrap();
        out
    }

    fn corruption<T>(r: Result<T>) -> String {
        match r {
            Err(LogError::Corruption(why)) => why,
            Err(e) => panic!("expected corruption, got {e}"),
            Ok(_) => panic!("expected corruption, got Ok"),
        }
    }

    /// Records span segments, read back in order, and every segment but the
    /// last ends in the end marker; a reopen visits them all and finds the tail.
    #[test]
    fn records_span_segments_and_read_back_after_a_reopen() {
        let dir = TempDir::new().unwrap();
        let (log, _, _) = open(dir.path(), FRESH).unwrap();
        let appended = append_n(&log, 0..60);
        assert!(log.active_segment() >= 3, "setup: the records should span segments");
        log.sync().unwrap();
        assert_eq!(read_all(&log, segment_start(0)), appended);
        let tail = log.tail();
        for id in 0..log.active_segment() {
            let end = log.segment_end(id).unwrap();
            let mut len = [0u8; 4];
            File::open(segment_path(dir.path(), id))
                .unwrap()
                .read_exact_at(&mut len, u64::from(split(end).1))
                .unwrap();
            assert_eq!(u32::from_le_bytes(len), END_MARKER, "segment {id} is not sealed");
        }
        drop(log);
        let (log, report, seen) = open(dir.path(), OpenFrom { cut: Some(0), furthest: 0 }).unwrap();
        assert_eq!(report.tail, tail);
        assert_eq!(seen, appended);
        assert_eq!(read_all(&log, segment_start(0)), appended);
    }

    /// Positions do not encode the segment size: a log reopened with another
    /// size reads back, and new segments take the new size.
    #[test]
    fn the_segment_size_can_change_between_opens() {
        let dir = TempDir::new().unwrap();
        let (log, _, _) = open(dir.path(), FRESH).unwrap();
        let mut appended = append_n(&log, 0..30);
        log.sync().unwrap();
        drop(log);
        let (log, _) = SegmentedLog::open(dir.path(), &FORMAT, 1024, OpenFrom { cut: Some(0), furthest: 0 }, |_, _| {
            Ok::<(), LogError>(())
        })
        .unwrap();
        appended.extend(append_n(&log, 30..90));
        log.sync().unwrap();
        assert_eq!(read_all(&log, segment_start(0)), appended);
        let last = log.active_segment();
        assert_eq!(std::fs::metadata(segment_path(dir.path(), last)).unwrap().len(), 1024);
    }

    /// Only records at or above the cut are visited at open; segments wholly
    /// below it are not read, so damage there does not fail the open.
    #[test]
    fn open_reads_from_the_cut_and_ignores_retired_segments() {
        let dir = TempDir::new().unwrap();
        let (log, _, _) = open(dir.path(), FRESH).unwrap();
        let appended = append_n(&log, 0..60);
        log.sync().unwrap();
        let cut = appended.iter().find(|(at, _)| segment_of(*at) == 2).unwrap().0;
        drop(log);
        // Damage segment 0, wholly below the cut.
        let f = OpenOptions::new().write(true).open(segment_path(dir.path(), 0)).unwrap();
        f.write_all_at(&[0xFF; 8], u64::from(SEGMENT_HEADER_LEN) + 2).unwrap();
        let (_, _, seen) = open(
            dir.path(),
            OpenFrom {
                cut: Some(cut),
                furthest: cut,
            },
        )
        .unwrap();
        let expected: Records = appended.into_iter().filter(|(at, _)| *at >= cut).collect();
        assert_eq!(seen, expected);
    }

    /// A torn frame in the last segment is the tail: dropped, the rest of the
    /// segment zeroed, and later appends read back (no stale frame rejoins).
    #[test]
    fn a_torn_tail_is_dropped_and_zeroed() {
        let dir = TempDir::new().unwrap();
        let (log, _, _) = open(dir.path(), FRESH).unwrap();
        let appended = append_n(&log, 0..4);
        log.sync().unwrap();
        let tail = log.tail();
        drop(log);
        // A frame that claims more than was written, then a stale valid frame
        // after a gap, as a power loss can leave.
        let f = OpenOptions::new().write(true).open(segment_path(dir.path(), 0)).unwrap();
        let byte = u64::from(split(tail).1);
        f.write_all_at(&40u32.to_le_bytes(), byte).unwrap();
        f.write_all_at(&[0xAB; 6], byte + 4).unwrap();
        let (log, report, seen) = open(dir.path(), FRESH).unwrap();
        assert_eq!(report.tail, tail);
        assert!(report.torn_bytes_dropped > 0);
        assert_eq!(seen, appended);
        let mut rest = vec![0u8; (SEGMENT - split(tail).1) as usize];
        File::open(segment_path(dir.path(), 0)).unwrap().read_exact_at(&mut rest, byte).unwrap();
        assert!(rest.iter().all(|&b| b == 0), "the bytes past the tail must be zeroed");
        let more = append_n(&log, 4..6);
        log.sync().unwrap();
        assert_eq!(read_all(&log, segment_start(0)), [appended, more].concat());
    }

    /// A bad frame in a segment that has a successor is damage to synced
    /// data, not a torn tail: open fails rather than cutting the log there.
    #[test]
    fn a_bad_frame_before_the_last_segment_fails_the_open() {
        let dir = TempDir::new().unwrap();
        let (log, _, _) = open(dir.path(), FRESH).unwrap();
        let appended = append_n(&log, 0..40);
        log.sync().unwrap();
        drop(log);
        let f = OpenOptions::new().write(true).open(segment_path(dir.path(), 1)).unwrap();
        let first_in_1 = appended.iter().find(|(at, _)| segment_of(*at) == 1).unwrap().0;
        f.write_all_at(&[0x5A], u64::from(split(first_in_1).1 + FRAME_HEADER_LEN)).unwrap();
        let why = corruption(open(dir.path(), FRESH));
        assert!(why.contains("segment 1") && why.contains("CRC"), "{why}");
        // Unwritten space before the end marker of a sealed segment, too.
        f.write_all_at(&[0u8; 4], u64::from(split(first_in_1).1)).unwrap();
        let why = corruption(open(dir.path(), FRESH));
        assert!(why.contains("not sealed"), "{why}");
    }

    /// A gap in the segment ids, a segment from another log, a wrong version
    /// and a header naming another segment all fail the open.
    #[test]
    fn foreign_or_missing_segments_fail_the_open() {
        let dir = TempDir::new().unwrap();
        let (log, _, _) = open(dir.path(), FRESH).unwrap();
        append_n(&log, 0..60);
        log.sync().unwrap();
        drop(log);
        let seg1 = segment_path(dir.path(), 1);
        let saved = std::fs::read(&seg1).unwrap();

        std::fs::remove_file(&seg1).unwrap();
        assert!(corruption(open(dir.path(), FRESH)).contains("segment 1 is missing"));

        let mut foreign = saved.clone();
        foreign[..16].copy_from_slice(&header_bytes(&OTHER, 1));
        std::fs::write(&seg1, &foreign).unwrap();
        assert!(corruption(open(dir.path(), FRESH)).contains("no test log header"));

        let mut version = saved.clone();
        version[4..8].copy_from_slice(&2u32.to_le_bytes());
        std::fs::write(&seg1, &version).unwrap();
        assert!(corruption(open(dir.path(), FRESH)).contains("format version 2"));

        let mut id = saved.clone();
        id[8..16].copy_from_slice(&7u64.to_le_bytes());
        std::fs::write(&seg1, &id).unwrap();
        assert!(corruption(open(dir.path(), FRESH)).contains("holds segment 7"));

        std::fs::write(&seg1, &saved).unwrap();
        open(dir.path(), FRESH).unwrap();
    }

    /// No saved progress with segments that start above 0 means the progress
    /// was lost: reading from the oldest segment could replay retired records,
    /// so the open fails. A cut below the oldest segment fails too.
    #[test]
    fn progress_must_match_the_segments_on_disk() {
        let dir = TempDir::new().unwrap();
        let (log, _, _) = open(dir.path(), FRESH).unwrap();
        append_n(&log, 0..60);
        log.sync().unwrap();
        let cut = segment_start(2);
        log.delete_segments_below(2).unwrap();
        drop(log);
        assert!(corruption(open(dir.path(), FRESH)).contains("no progress was saved"));
        assert!(
            corruption(open(
                dir.path(),
                OpenFrom {
                    cut: Some(segment_start(1)),
                    furthest: 0
                }
            ))
            .contains("below the oldest")
        );
        open(
            dir.path(),
            OpenFrom {
                cut: Some(cut),
                furthest: cut,
            },
        )
        .unwrap();
    }

    /// A segment created but whose header never became durable holds nothing:
    /// as the last segment it is recreated.
    #[test]
    fn a_last_segment_without_a_durable_header_is_recreated() {
        let dir = TempDir::new().unwrap();
        let (log, _, _) = open(dir.path(), FRESH).unwrap();
        append_n(&log, 0..2);
        log.sync().unwrap();
        drop(log);
        // A blank segment after one that was never sealed is not a crash
        // mid-rotation (sealing comes first): segment 0 is damaged.
        let next = segment_path(dir.path(), 1);
        std::fs::write(&next, vec![0u8; SEGMENT as usize]).unwrap();
        let why = corruption(open(dir.path(), FRESH));
        assert!(why.contains("segment 0 is not sealed"), "{why}");
        std::fs::remove_file(&next).unwrap();
        // A crash mid-rotation: segment 0 sealed, segment 1 blank.
        let (log, _, _) = open(dir.path(), FRESH).unwrap();
        append_n(&log, 2..40);
        log.sync().unwrap();
        let last = log.active_segment();
        drop(log);
        std::fs::write(segment_path(dir.path(), last), vec![0u8; SEGMENT as usize]).unwrap();
        let (log, report, _) = open(dir.path(), FRESH).unwrap();
        assert_eq!(report.tail, segment_start(last));
        assert_eq!(report.torn_bytes_dropped, u64::from(SEGMENT_HEADER_LEN));
        check_header(&FORMAT, &File::open(segment_path(dir.path(), last)).unwrap(), last).unwrap();
        append_n(&log, 40..42);
        log.sync().unwrap();
    }

    /// Saved progress past the durable tail (records a power loss took after
    /// a flush saved progress over them): open starts a new segment, so new
    /// records land above every saved position and a second open still reads
    /// them from the cut.
    #[test]
    fn progress_past_the_durable_tail_starts_a_new_segment() {
        let dir = TempDir::new().unwrap();
        let (log, _, _) = open(dir.path(), FRESH).unwrap();
        append_n(&log, 0..3);
        log.sync().unwrap();
        append_n(&log, 3..6);
        // Progress over the unsynced records, then a power loss.
        let furthest = log.tail();
        log.lose_unsynced();
        drop(log);
        let from = OpenFrom {
            cut: Some(furthest),
            furthest,
        };
        let (log, report, _) = open(dir.path(), from).unwrap();
        assert!(report.tail > furthest);
        assert_eq!(segment_of(report.tail), segment_of(furthest) + 1);
        let after = append_n(&log, 6..8);
        log.sync().unwrap();
        drop(log);
        // Twice: a cut past segment 0's end marker is legal.
        for _ in 0..2 {
            let (log, _, seen) = open(dir.path(), from).unwrap();
            assert_eq!(seen, after);
            assert_eq!(read_all(&log, furthest), after);
        }
        assert!(
            corruption(open(
                dir.path(),
                OpenFrom {
                    cut: Some(0),
                    furthest: segment_start(9)
                }
            ))
            .contains("past the last segment")
        );
    }

    /// Concurrent commits share fsyncs, and each returns covered.
    #[test]
    fn concurrent_commits_share_fsyncs() {
        let dir = TempDir::new().unwrap();
        let (log, _, _) = open(dir.path(), FRESH).unwrap();
        let led = AtomicU64::new(0);
        std::thread::scope(|s| {
            for t in 0..8u32 {
                let (log, led) = (&log, &led);
                s.spawn(move || {
                    for i in 0..50u32 {
                        let b = body(t * 100 + i);
                        let (_, end) = log.append(&[&b], |_, _| {}).unwrap();
                        if log.commit(end, Duration::ZERO).unwrap() {
                            led.fetch_add(1, Ordering::SeqCst);
                        }
                        assert!(log.durable() >= end);
                    }
                });
            }
        });
        assert!(led.load(Ordering::SeqCst) < 400);
        assert_eq!(read_all(&log, segment_start(0)).len(), 400);
    }

    /// A failed fsync stops the log: the commit fails, and so does every later
    /// append and commit.
    #[test]
    fn a_failed_fsync_poisons_the_log() {
        let dir = TempDir::new().unwrap();
        let (log, _, _) = open(dir.path(), FRESH).unwrap();
        let b = body(0);
        let (_, end) = log.append(&[&b], |_, _| {}).unwrap();
        log.fail_next_sync();
        assert!(matches!(log.commit(end, Duration::ZERO), Err(LogError::Poisoned(_))));
        assert!(matches!(log.append(&[&b], |_, _| {}), Err(LogError::Poisoned(_))));
        assert!(matches!(log.sync(), Err(LogError::Poisoned(_))));
        assert!(log.poisoned().is_some());
    }

    /// A failed fsync while sealing a segment poisons the log too, so no
    /// append can land over the end marker of a segment left half-rotated.
    #[test]
    fn a_failed_rotation_poisons_the_log() {
        let dir = TempDir::new().unwrap();
        let (log, _, _) = open(dir.path(), FRESH).unwrap();
        let first = append_n(&log, 0..5);
        log.sync().unwrap();
        log.fail_next_sync();
        let big = vec![b'z'; 200];
        assert!(matches!(log.append(&[&big], |_, _| {}), Err(LogError::Poisoned(_))));
        assert!(matches!(log.append(&[b"small"], |_, _| {}), Err(LogError::Poisoned(_))));
        drop(log);
        let (_, _, seen) = open(dir.path(), FRESH).unwrap();
        assert_eq!(seen, first);
    }

    /// A record too large for a segment is refused before anything is written.
    #[test]
    fn a_record_too_large_for_a_segment_is_refused() {
        let dir = TempDir::new().unwrap();
        let (log, _, _) = open(dir.path(), FRESH).unwrap();
        let max = log.max_frame_len() as usize - FRAME_HEADER_LEN as usize;
        assert!(matches!(
            log.append(&[&vec![0u8; max + 1]], |_, _| {}),
            Err(LogError::RecordTooLarge { .. })
        ));
        assert_eq!(log.tail(), segment_start(0));
        log.append(&[&vec![1u8; max]], |_, _| {}).unwrap();
    }

    /// A body given in parts is one record whose CRC covers them all.
    #[test]
    fn a_body_in_parts_is_one_record() {
        let dir = TempDir::new().unwrap();
        let (log, _, _) = open(dir.path(), FRESH).unwrap();
        let (start, end) = log.append(&[b"ab", b"", b"cde"], |_, _| {}).unwrap();
        assert_eq!(end - start, u64::from(FRAME_HEADER_LEN) + 5);
        log.sync().unwrap();
        assert_eq!(read_all(&log, segment_start(0)), vec![(start, b"abcde".to_vec())]);
    }

    /// `under_lock` runs before the tail including the record is published.
    #[test]
    fn under_lock_runs_before_the_tail_is_published() {
        let dir = TempDir::new().unwrap();
        let (log, _, _) = open(dir.path(), FRESH).unwrap();
        let before = log.tail();
        let (start, end) = log
            .append(&[b"x"], |start, _| {
                assert_eq!(log.tail(), before);
                assert_eq!(start, before);
            })
            .unwrap();
        assert_eq!((start, log.tail()), (before, end));
    }

    /// Deleting retired segments removes whole segments from the oldest up,
    /// never the active one; a read from below the oldest skips to it, and
    /// `missing_segments` reports what is gone.
    #[test]
    fn retired_segments_are_deleted_and_skipped() {
        let dir = TempDir::new().unwrap();
        let (log, _, _) = open(dir.path(), FRESH).unwrap();
        let appended = append_n(&log, 0..60);
        log.sync().unwrap();
        let active = log.active_segment();
        let (deleted, bytes) = log.delete_segments_below(2).unwrap();
        assert_eq!((deleted, bytes), (2, 2 * u64::from(SEGMENT)));
        assert_eq!(log.oldest_segment().unwrap(), Some(2));
        assert_eq!(log.missing_segments(segment_start(0), log.tail()), vec![0, 1]);
        let kept: Records = appended.into_iter().filter(|(at, _)| segment_of(*at) >= 2).collect();
        assert_eq!(read_all(&log, segment_start(0)), kept);
        log.delete_segments_below(u64::MAX).unwrap();
        assert_eq!(log.oldest_segment().unwrap(), Some(active), "the active segment is never deleted");
    }

    /// `bytes_between` counts record bytes only, across segments.
    #[test]
    fn bytes_between_counts_record_bytes() {
        let dir = TempDir::new().unwrap();
        let (log, _, _) = open(dir.path(), FRESH).unwrap();
        let appended = append_n(&log, 0..30);
        let total: u64 = appended.iter().map(|(_, b)| u64::from(FRAME_HEADER_LEN) + b.len() as u64).sum();
        assert_eq!(log.bytes_between(segment_start(0), log.tail()), total);
        let from = appended[10].0;
        let after: u64 = appended[10..].iter().map(|(_, b)| u64::from(FRAME_HEADER_LEN) + b.len() as u64).sum();
        assert_eq!(log.bytes_between(from, log.tail()), after);
        assert_eq!(log.bytes_between(log.tail(), from), 0);
    }

    /// Past half of a segment the next one is zero-filled ahead under its
    /// temporary name, with its header, and installed when the log gets there.
    #[test]
    fn the_next_segment_is_prepared_ahead_and_installed() {
        let dir = TempDir::new().unwrap();
        let (log, _, _) = open(dir.path(), FRESH).unwrap();
        while split(log.tail()).1 * 2 < SEGMENT {
            log.append(&[b"0123456789"], |_, _| {}).unwrap();
        }
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !log.prep.pending().is_some_and(|(_, done)| done) {
            assert!(std::time::Instant::now() < deadline, "the next segment was never prepared");
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(prep_path(dir.path(), 1).exists());
        assert!(!segment_path(dir.path(), 1).exists(), "open must not see the next segment early");
        while log.active_segment() == 0 {
            log.append(&[b"0123456789"], |_, _| {}).unwrap();
        }
        assert!(!prep_path(dir.path(), 1).exists());
        check_header(&FORMAT, &File::open(segment_path(dir.path(), 1)).unwrap(), 1).unwrap();
        // A leftover preparation is removed at open.
        std::fs::write(prep_path(dir.path(), 9), b"x").unwrap();
        log.sync().unwrap();
        drop(log);
        open(dir.path(), FRESH).unwrap();
        assert!(!prep_path(dir.path(), 9).exists());
    }

    /// Records of every size — tiny, straddling a read-ahead window, larger
    /// than a window — read back exactly, from the start and from a record in
    /// the middle, and a reopen finds them all; a CRC failure inside a large
    /// record of a sealed segment is still caught.
    #[test]
    fn records_read_back_across_read_ahead_windows() {
        const BIG_SEGMENT: u32 = 1 << 20;
        let open_big = |dir: &Path| {
            let mut seen = Vec::new();
            let (log, _) = SegmentedLog::open(dir, &FORMAT, BIG_SEGMENT, FRESH, |at, body| -> Result<()> {
                seen.push((at, body.to_vec()));
                Ok(())
            })
            .unwrap();
            (log, seen)
        };
        let dir = TempDir::new().unwrap();
        let (log, _) = open_big(dir.path());
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let mut appended: Records = Vec::new();
        for i in 0..300u32 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let len = match i % 10 {
                0 => 3 * READ_AHEAD as usize + (rng % 1000) as usize,
                1..=3 => (READ_AHEAD as usize / 2) + (rng % 4096) as usize,
                _ => 1 + (rng % 300) as usize,
            };
            let body: Vec<u8> = (0..len).map(|j| (j as u32 ^ i) as u8).collect();
            let (start, _) = log.append(&[&body], |_, _| {}).unwrap();
            appended.push((start, body));
        }
        log.sync().unwrap();
        assert!(log.active_segment() >= 3, "setup: the records should span segments");
        assert_eq!(read_all(&log, segment_start(0)), appended);
        let mid = appended.len() / 2;
        assert_eq!(read_all(&log, appended[mid].0), appended[mid..].to_vec());
        drop(log);
        let (_, seen) = open_big(dir.path());
        assert_eq!(seen, appended);

        // Damage a large record in the first (sealed) segment, past its first
        // read-ahead window.
        let (at, body) = appended
            .iter()
            .find(|(at, b)| segment_of(*at) == 0 && b.len() > READ_AHEAD as usize)
            .unwrap();
        let f = OpenOptions::new().write(true).open(segment_path(dir.path(), 0)).unwrap();
        let byte = u64::from(split(*at).1 + FRAME_HEADER_LEN) + body.len() as u64 - 10;
        f.write_all_at(&[0xEE], byte).unwrap();
        let why = corruption(SegmentedLog::open(dir.path(), &FORMAT, BIG_SEGMENT, FRESH, |_, _| Ok::<(), LogError>(())));
        assert!(why.contains("segment 0") && why.contains("CRC"), "{why}");
    }

    /// A progress file round-trips with its extra bytes, and every kind of
    /// damage is corruption: another log's magic, a changed byte, a truncated
    /// file, unsorted namespaces, an offset below the cut.
    #[test]
    fn a_progress_file_round_trips_and_rejects_damage() {
        let file = ProgressFile { magic: *b"TPRG" };
        let p = Progress {
            cut: segment_start(2),
            safe: vec![(1, position(2, 900)), (7, position(3, 40))],
        };
        let bytes = file.encode(&p, b"extra");
        assert_eq!(file.decode(&bytes).unwrap(), (p.clone(), b"extra".to_vec()));
        assert_eq!(p.replay_from(7), position(3, 40));
        assert_eq!(p.replay_from(4), p.cut);
        assert_eq!(p.furthest(), position(3, 40));

        let damaged = |b: Vec<u8>| match file.decode(&b) {
            Err(LogError::Corruption(why)) => why,
            other => panic!("expected corruption, got {other:?}"),
        };
        assert!(damaged(ProgressFile { magic: *b"OTHR" }.encode(&p, &[])).contains("no header"));
        let mut flipped = bytes.clone();
        flipped[10] ^= 1;
        assert!(damaged(flipped).contains("CRC"));
        assert!(damaged(bytes[..bytes.len() - 3].to_vec()).contains("CRC"));
        let unsorted = Progress {
            cut: 0,
            safe: vec![(7, 5), (1, 5)],
        };
        assert!(damaged(file.encode(&unsorted, &[])).contains("not sorted"));
        let below = Progress {
            cut: 100,
            safe: vec![(1, 50)],
        };
        assert!(damaged(file.encode(&below, &[])).contains("below the cut"));

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("progress");
        assert_eq!(file.read(&path).unwrap(), None);
        file.write(&path, &p, &[]).unwrap();
        assert_eq!(file.read(&path).unwrap(), Some((p, Vec::new())));
    }

    /// The power-loss model: what no fsync covered is lost, and what one did
    /// survives (a sync, or the fsync that sealed a segment), for many random
    /// cut points.
    #[test]
    fn every_synced_record_survives_random_power_losses() {
        let dir = TempDir::new().unwrap();
        let mut survivors: Records = Vec::new();
        let mut next = 0u32;
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut rand = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for _ in 0..40 {
            let (log, _, seen) = open(dir.path(), FRESH).unwrap();
            assert_eq!(seen, survivors, "a durable record was lost, or a lost one came back");
            let mut appended = Vec::new();
            for _ in 0..(rand() % 4) {
                let n = (rand() % 12) as u32;
                for i in next..next + n {
                    let b = body(i);
                    let (start, end) = log.append(&[&b], |_, _| {}).unwrap();
                    appended.push((start, end, b));
                }
                next += n;
                if rand() % 2 == 0 {
                    log.sync().unwrap();
                }
            }
            let durable = log.durable();
            survivors.extend(appended.into_iter().filter(|&(_, end, _)| end <= durable).map(|(start, _, b)| (start, b)));
            log.lose_unsynced();
        }
    }
}
