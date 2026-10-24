//! The vector index WAL: a log of the vector worker's payload writes.
//!
//! The vector worker writes chunk codes (`{ns}_sparse_vector`) and whole-document
//! codes (`{ns}_dense_vector`) without the main WAL, and must make them durable
//! before it completes a queue entry. This log is how: each such write is
//! appended here as it is applied, and one [`sync`](VectorIndexWal::sync) per
//! completion batch makes the batch durable — one fsync of one sequential file
//! instead of flushing three memtables. After a crash, the records not yet
//! retired are replayed with their original seqs. Partition maintenance logs
//! its per-document meta writes and key deletes here too. Design:
//! `semantic_search/report/Vector-Index-WAL-Design.md`.
//!
//! # What survives a crash is a prefix of the log
//!
//! A sync makes a prefix durable, and a torn tail drops only a suffix. A
//! namespace's memtable can also reach disk on its own, as L0 files; so before
//! a flush writes any, the log is synced if the namespace has records past the
//! durable tail ([`before_memtable_written`](VectorIndexWal::before_memtable_written)).
//! An L0 file therefore never holds a logged write the log could still lose,
//! and the logged writes that survive any crash are exactly a prefix of the
//! log, across namespaces. Maintenance relies on it: its meta changes and key
//! deletes live in different namespaces, and the prefix keeps them in program
//! order.
//!
//! Segments, record format, append, `sync` and reading, plus **retirement**:
//! a record is retired once the memtable holding its write is flushed to L0,
//! the retired cut is kept durable in a marker, and GC deletes segments wholly
//! below it. Recovery and the worker's use of the log are built on top (steps
//! 3–4 of the design's plan).
//!
//! # Retirement
//!
//! Every namespace's flush observer reports seals and flushes here. A
//! namespace is tracked from its first append. On a seal of its memtable
//! version `v`, the log records the namespace's **ceiling**: the tail lowered
//! to the oldest of *that namespace's* appends not yet applied to a memtable
//! (`LoggedWrite` registers an append while the append lock is held and
//! clears it when dropped, after the apply). The seal callback runs under the
//! memtable's write lock, so every one of the namespace's records below the
//! ceiling is in `v` or an older memtable. When `v`'s L0 files are written —
//! in version order, and **before they are registered**
//! (`LsmFlushObserver::on_ro_memtable_written`) — the namespace's
//! `safe_offset` advances to that ceiling, and the change takes effect only
//! once the marker recording it is on disk: if the marker cannot be written,
//! the event returns an error, the LSM undoes the flush (nothing registered,
//! the memtable kept for a retry) and nothing changes here. The **retired
//! cut** is the lowest `safe_offset` over namespaces with records above
//! theirs, capped by the global ceiling (the oldest append of any namespace
//! not yet applied). It only moves forward; GC deletes segments wholly below
//! it.
//!
//! # Replay is per namespace, and progress is durable before files are mergeable
//!
//! Replay applies a record only if it lies at or above **its own namespace's**
//! `safe_offset`, not merely the global cut. The vector writes' deletes live
//! in the main WAL, so a record of a namespace that already flushed past it
//! can have a later delete that reached L1, where the tombstone is dropped:
//! replaying it from the global cut would bring the key back with nothing
//! newer to beat it (`replay_does_not_bring_back_a_key_whose_delete_reached_l1`).
//! So the marker `retired_cut` holds the cut and every tracked namespace's
//! `safe_offset` above it, and it is written in the written event, before the
//! memtable's L0 files are registered: by the time any merge could drop a
//! tombstone in them, the namespace's progress past the records they cover is
//! on disk.
//!
//! **After a restart** the in-memory state is gone, so `open` rebuilds it from
//! the log: every namespace with a record at or above the durable cut starts
//! out *holding* the cut (`safe_offset` = the cut, `last_write` = its last
//! record's end). Without that, the first flush of any other namespace would
//! find nobody holding the cut and move it to the tail, retiring records that
//! exist only in replayed memtables. Records of a namespace dropped since are
//! released by [`forget_namespace`](VectorIndexWal::forget_namespace), and a
//! forgotten namespace is never tracked again (ids are not reused), so a late
//! append for it cannot hold the cut.
//!
//! Replay applies the records with no observer attached, so no seal or flush
//! reports them. Recovery therefore flushes the namespaces it replayed into,
//! fsyncs their value logs, and records them as safe up to the tail
//! ([`retire_replayed`](VectorIndexWal::retire_replayed)) **before** main-WAL
//! recovery compacts anything. Otherwise that compaction could drop a delete's
//! tombstone while the marker still lists the deleted key's record, and a
//! second crash would replay it.
//!
//! The ceiling is per namespace, not global as in the main WAL, so that a
//! namespace sealed while *another* namespace's write is in flight still
//! retires everything it flushed. A namespace whose memtable is empty but
//! that still holds the cut (its last record was never applied, say) is
//! released by the WAL GC tick, which seals even an empty memtable.
//!
//! # Locks
//!
//! `retirement` serialises changes to the namespaces' progress and is held
//! across the marker write. Neither the write path nor a seal takes it: a
//! seal takes `tracking` (and `in_flight` inside it) and does no I/O, so a
//! memtable's write lock — held during the seal callback, for every namespace
//! of the database — never waits on this log's fsyncs. Order: `retirement` →
//! `tracking` → `in_flight`, and `active` → `in_flight`; `active` is never
//! held together with `tracking` or `retirement`, and none is held while a
//! memtable is applied to.
//!
//! # A failed fsync stops the log
//!
//! After an `fsync` fails, the kernel may have dropped the dirty pages it
//! could not write, and a later `fsync` that succeeds says nothing about
//! them. Advancing the durable tail past them would let later batches
//! complete over a hole, which `open` then reads as a torn tail, dropping
//! those batches' records. So the first failure poisons the log: appends and
//! syncs fail until the database is reopened, and the vector worker keeps its
//! entries queued meanwhile.
//!
//! # Format
//!
//! One log per database, in `{db_path}/vector_index_wal/`. Segments, frames,
//! positions, the torn-tail and corruption rules and durability are the shared
//! segmented log's (`db/segmented_log.rs`), with magic `MVIW`. A record's body
//! is:
//!
//! ```text
//! body: [ns_id u32][seq u64][key_len u32][key][value_len u32][value]
//! ```
//!
//! A delete is a body whose `value_len` is `u32::MAX` and that has no value
//! bytes (no put can be that long: a record must fit a segment). A body that
//! passes its CRC but does not decode is corruption: the CRC says these are
//! the bytes written.

use crate::db::metrics::Metrics;
use crate::db::segmented_log::{self, LogError, LogFormat, OpenFrom, SegmentedLog, segment_of};
use parking_lot::Mutex;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

/// Default segment size (64 MiB, as the main WAL).
pub(crate) const DEFAULT_SEGMENT_SIZE: u32 = 64 * 1024 * 1024;

/// This log's segments: magic `MVIW`, version 2 (bump it whenever the body
/// layout changes).
pub(crate) static FORMAT: LogFormat = LogFormat {
    name: "vector index WAL",
    magic: *b"MVIW",
    version: 2,
};
/// The body's fixed fields: `ns_id`, `seq`, `key_len`, `value_len`.
const BODY_FIXED_LEN: usize = 4 + 8 + 4 + 4;
/// The `value_len` of a delete record, which has no value bytes.
const DELETE_VALUE_LEN: u32 = u32::MAX;

/// The progress marker file, in the log's directory.
const RETIRED_CUT_FILE: &str = "retired_cut";
/// Marker layout: magic, version, cut (u64), count (u32), then `count` ×
/// (namespace u32, safe offset u64), then crc32 of everything before it.
const RETIRED_CUT_MAGIC: [u8; 4] = *b"MVIC";
const RETIRED_CUT_VERSION: u32 = 2;
const RETIRED_CUT_HEADER_LEN: usize = 4 + 4 + 8 + 4;
const RETIRED_CUT_ENTRY_LEN: usize = 4 + 8;

/// What the progress marker records.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Progress {
    /// Records below this are retired; GC deletes segments wholly below it.
    cut: u64,
    /// Namespaces whose `safe_offset` is above the cut, sorted by namespace.
    safe: Vec<(u32, u64)>,
}

impl Progress {
    /// Where replay starts for `ns`.
    fn replay_from(&self, ns: u32) -> u64 {
        self.safe.binary_search_by_key(&ns, |(n, _)| *n).map_or(self.cut, |i| self.safe[i].1)
    }
}

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
    /// An earlier fsync failed; the log refuses appends and syncs until the
    /// database is reopened (see the module docs).
    #[error("vector index WAL stopped after a failed fsync ({0}); reopen the database to recover")]
    Poisoned(String),
}

impl From<LogError> for VectorIndexWalError {
    fn from(e: LogError) -> Self {
        match e {
            LogError::Io(e) => Self::Io(e),
            LogError::Corruption(why) => Self::Corruption(why),
            LogError::RecordTooLarge { len, max } => Self::RecordTooLarge { len, max },
            LogError::Poisoned(why) => Self::Poisoned(why),
        }
    }
}

/// Result type of this module.
pub type Result<T> = std::result::Result<T, VectorIndexWalError>;

/// One record read from the log. Borrows the reader's buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RecordRef<'a> {
    pub ns_id: u32,
    pub seq: u64,
    pub key: &'a [u8],
    /// The value of a put; `None` for a delete.
    pub value: Option<&'a [u8]>,
}

/// The framed length of a record of `key` and `value` (`None`: a delete).
fn frame_len_of(key: &[u8], value: Option<&[u8]>) -> u64 {
    u64::from(segmented_log::FRAME_HEADER_LEN) + (BODY_FIXED_LEN + key.len() + value.map_or(0, <[u8]>::len)) as u64
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
    let value_len = u32::from_le_bytes(body[key_end..key_end + 4].try_into().unwrap());
    let rest = &body[key_end + 4..];
    let value = match value_len {
        DELETE_VALUE_LEN if rest.is_empty() => None,
        n if n != DELETE_VALUE_LEN && n as usize == rest.len() => Some(rest),
        _ => return Err(bad()),
    };
    Ok(RecordRef {
        ns_id,
        seq,
        key: &body[16..key_end],
        value,
    })
}

/// Where an append landed: `[start, end)` (tests keep it past the registration).
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Appended {
    pub start: u64,
    pub end: u64,
}

/// An appended record that is not yet applied to its memtable.
///
/// Holds the record's namespace ceiling (and the global one) at or below its
/// start until dropped. Keep it alive until the memtable apply is done. RAII
/// because a leaked registration would pin the retired cut, and with it GC,
/// for good — the failure the main WAL's `InFlightWrite` guards against.
pub(crate) struct LoggedWrite<'a> {
    wal: &'a VectorIndexWal,
    /// Where the record landed.
    pub start: u64,
    /// Just past the record.
    pub end: u64,
}

impl LoggedWrite<'_> {
    /// Where the record landed, without the registration.
    #[cfg(test)]
    pub(crate) fn appended(&self) -> Appended {
        Appended {
            start: self.start,
            end: self.end,
        }
    }
}

impl std::fmt::Debug for LoggedWrite<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoggedWrite").field("start", &self.start).field("end", &self.end).finish()
    }
}

impl Drop for LoggedWrite<'_> {
    fn drop(&mut self) {
        self.wal.in_flight.lock().remove(&self.start);
    }
}

/// A position in the vector index WAL, as the admin API shows it (the shared
/// [`LogPosition`](segmented_log::LogPosition)).
pub type VectorIndexWalPosition = segmented_log::LogPosition;

/// One segment file of the vector index WAL.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VectorIndexWalSegment {
    /// Segment id (its file is `{segment_id:010}.log`).
    pub segment_id: u64,
    /// Bytes on disk.
    pub bytes: u64,
}

/// A namespace holding the retired cut back: it has records above its
/// `safe_offset` that no flush has covered yet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VectorIndexWalHolder {
    /// The namespace's name.
    pub namespace: String,
    /// Everything of this namespace below here is in an SSTable.
    pub safe_offset: VectorIndexWalPosition,
}

/// The vector index WAL's state, read from the log and its files.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VectorIndexWalStats {
    /// Every segment file, oldest first.
    pub segments: Vec<VectorIndexWalSegment>,
    /// Number of segment files.
    pub segment_count: usize,
    /// Total bytes of the segment files.
    pub disk_bytes: u64,
    /// Where the next record goes.
    pub tail: VectorIndexWalPosition,
    /// Everything below here is fsynced.
    pub durable_tail: VectorIndexWalPosition,
    /// Records below here are retired: in SSTables, skipped by replay, and
    /// their segments are deleted by GC once wholly below it.
    pub retired_cut: VectorIndexWalPosition,
    /// Bytes of records at or above the retired cut (segment headers and end
    /// markers excluded): what a restart would replay.
    pub unretired_bytes: u64,
    /// Highest seq of any record in the log.
    pub last_sequence: u64,
    /// Namespaces holding the retired cut back.
    pub holding_cut: Vec<VectorIndexWalHolder>,
    /// Set after a failed fsync, with its error: the log refuses appends and
    /// syncs until the database is reopened. `None` while it works.
    pub stopped: Option<String>,
}

/// What [`VectorIndexWal::open`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OpenReport {
    /// Position of the next append.
    pub tail: u64,
    /// Highest seq of any record in the log (0 if none).
    pub last_seq: u64,
    /// The torn frame's length (capped at the end of its segment) when open
    /// found a torn tail, or the bytes of a last segment's unfinished header;
    /// 0 otherwise.
    pub torn_bytes_dropped: u64,
}

/// The vector index WAL. See the module docs.
pub(crate) struct VectorIndexWal {
    /// The segments, frames and syncs (`db/segmented_log.rs`).
    pub(crate) log: SegmentedLog,
    /// Highest seq of any record appended since open, or found at open at or
    /// above the retired cut.
    last_seq: AtomicU64,
    /// Appends not yet applied to a memtable: start → namespace. Registered
    /// under the append lock, before the log publishes the tail.
    in_flight: Mutex<BTreeMap<u64, u32>>,
    /// Which namespaces are tracked, their latest records and their seals.
    /// Held briefly, never across I/O.
    tracking: Mutex<Tracking>,
    /// Each namespace's durable progress and the marker on disk. Held across
    /// the marker write.
    retirement: Mutex<Retirement>,
    /// Records below this position are retired. Durable in `retired_cut`.
    retired_cut: AtomicU64,
    /// Where each namespace's replay starts, as of this open (the durable
    /// marker read at open). Used by [`replay`](VectorIndexWal::replay).
    opened_progress: Progress,
    /// The database's engine-wide counters.
    metrics: Arc<Metrics>,
    /// Test-only fault: the next marker write fails.
    #[cfg(test)]
    fail_next_marker_write: std::sync::atomic::AtomicBool,
}

#[derive(Default)]
struct Tracking {
    /// The end of each tracked namespace's latest record. A namespace is
    /// tracked from its first append (at open: from a record at or above where
    /// its replay starts).
    last_write: HashMap<u32, u64>,
    /// `(namespace, memtable version)` → the namespace's ceiling at the seal.
    seals: BTreeMap<(u32, u64), PendingFlush>,
    /// Dropped namespaces. Ids are never reused, so these are never tracked
    /// again: a late append for one would otherwise hold the cut for good.
    forgotten: HashSet<u32>,
}

#[derive(Default)]
struct Retirement {
    /// Every record of the namespace below here is in an SSTable (absent: 0).
    safe: HashMap<u32, u64>,
    /// What the marker on disk says now.
    written: Progress,
}

struct PendingFlush {
    ceiling: u64,
    flushed: bool,
}

fn read_progress(dir: &Path) -> Result<Option<Progress>> {
    let bytes = match std::fs::read(dir.join(RETIRED_CUT_FILE)) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let bad = |why: &str| VectorIndexWalError::Corruption(format!("the retired-cut marker is damaged: {why}"));
    if bytes.len() < RETIRED_CUT_HEADER_LEN + 4 || bytes[0..4] != RETIRED_CUT_MAGIC {
        return Err(bad("no header"));
    }
    let (body, crc) = bytes.split_at(bytes.len() - 4);
    if crc32fast::hash(body) != u32::from_le_bytes(crc.try_into().unwrap()) {
        return Err(bad("CRC mismatch"));
    }
    let version = u32::from_le_bytes(body[4..8].try_into().unwrap());
    if version != RETIRED_CUT_VERSION {
        return Err(bad(&format!("version {version}, expected {RETIRED_CUT_VERSION}")));
    }
    let cut = u64::from_le_bytes(body[8..16].try_into().unwrap());
    let count = u32::from_le_bytes(body[16..20].try_into().unwrap()) as usize;
    if body.len() != RETIRED_CUT_HEADER_LEN + count * RETIRED_CUT_ENTRY_LEN {
        return Err(bad("length does not match its entry count"));
    }
    let safe = body[RETIRED_CUT_HEADER_LEN..]
        .as_chunks::<RETIRED_CUT_ENTRY_LEN>()
        .0
        .iter()
        .map(|e| {
            (
                u32::from_le_bytes(e[0..4].try_into().unwrap()),
                u64::from_le_bytes(e[4..12].try_into().unwrap()),
            )
        })
        .collect();
    Ok(Some(Progress { cut, safe }))
}

fn progress_bytes(p: &Progress) -> Vec<u8> {
    let mut b = Vec::with_capacity(RETIRED_CUT_HEADER_LEN + p.safe.len() * RETIRED_CUT_ENTRY_LEN + 4);
    b.extend_from_slice(&RETIRED_CUT_MAGIC);
    b.extend_from_slice(&RETIRED_CUT_VERSION.to_le_bytes());
    b.extend_from_slice(&p.cut.to_le_bytes());
    b.extend_from_slice(&(p.safe.len() as u32).to_le_bytes());
    for (ns, safe) in &p.safe {
        b.extend_from_slice(&ns.to_le_bytes());
        b.extend_from_slice(&safe.to_le_bytes());
    }
    let crc = crc32fast::hash(&b);
    b.extend_from_slice(&crc.to_le_bytes());
    b
}

impl VectorIndexWal {
    /// Opens the log in `dir`, creating it if absent, and finds its tail.
    ///
    /// Reads the progress marker first (it says where reading starts), then
    /// opens the segmented log from its cut (`SegmentedLog::open`: validation,
    /// torn tail, a new segment when the marker names positions past the
    /// tail), decoding every record at or above the cut for the highest seq
    /// and for who holds the cut.
    pub(crate) fn open(dir: &Path, segment_size: u32, metrics: Arc<Metrics>) -> Result<(Self, OpenReport)> {
        let marker = if dir.exists() { read_progress(dir)? } else { None };
        if let Some(p) = &marker
            && let Some((ns, safe)) = p.safe.iter().find(|(_, s)| *s < p.cut)
        {
            return Err(VectorIndexWalError::Corruption(format!(
                "namespace {ns}'s safe offset {:?} is below the cut",
                VectorIndexWalPosition::of(*safe)
            )));
        }
        let cut = marker.as_ref().map_or(0, |p| p.cut);
        let furthest = marker
            .as_ref()
            .map_or(0, |p| p.safe.iter().map(|(_, s)| *s).chain([p.cut]).max().unwrap_or(0));

        let mut last_seq = 0u64;
        // Each namespace's last record end, to rebuild who holds the cut.
        let mut last_write: HashMap<u32, u64> = HashMap::new();
        let (log, report) = SegmentedLog::open(
            dir,
            &FORMAT,
            segment_size,
            OpenFrom {
                cut: marker.as_ref().map(|p| p.cut),
                furthest,
            },
            |at, body| -> Result<()> {
                let r = decode_body(body, at)?;
                last_seq = last_seq.max(r.seq);
                let end = at + frame_len_of(r.key, r.value);
                let e = last_write.entry(r.ns_id).or_insert(end);
                *e = (*e).max(end);
                Ok(())
            },
        )?;

        // Each namespace's records below its safe offset (the cut, if the
        // marker does not list it) are in SSTables; one with a record above it
        // holds the cut until a flush covers that record. Namespaces the marker
        // lists keep their safe offsets, so they are written again.
        let opened_progress = marker.unwrap_or_default();
        let mut safe: HashMap<u32, u64> = opened_progress.safe.iter().copied().collect();
        let mut tracked: HashMap<u32, u64> = HashMap::new();
        for (ns, end) in last_write {
            let from = opened_progress.replay_from(ns);
            if end > from {
                safe.entry(ns).or_insert(from);
                tracked.insert(ns, end);
            }
        }
        let wal = Self {
            log,
            last_seq: AtomicU64::new(last_seq),
            in_flight: Mutex::new(BTreeMap::new()),
            tracking: Mutex::new(Tracking {
                last_write: tracked,
                ..Tracking::default()
            }),
            retirement: Mutex::new(Retirement {
                safe,
                written: opened_progress.clone(),
            }),
            retired_cut: AtomicU64::new(cut),
            opened_progress,
            metrics,
            #[cfg(test)]
            fail_next_marker_write: std::sync::atomic::AtomicBool::new(false),
        };
        Ok((
            wal,
            OpenReport {
                tail: report.tail,
                last_seq,
                torn_bytes_dropped: report.torn_bytes_dropped,
            },
        ))
    }

    /// The log's directory.
    fn dir(&self) -> &Path {
        self.log.dir()
    }

    /// Appends one record. Not durable until a later [`sync`](Self::sync)
    /// returns. Keep the returned [`LoggedWrite`] until the record's memtable
    /// apply is done.
    ///
    /// Records are in append order, which is not necessarily seq order: seqs
    /// are allocated before the append lock is taken.
    pub(crate) fn append(&self, ns_id: u32, key: &[u8], value: &[u8], seq: u64) -> Result<LoggedWrite<'_>> {
        self.append_record(ns_id, key, Some(value), seq)
    }

    /// Appends a delete of `key`, like [`append`](Self::append).
    pub(crate) fn append_delete(&self, ns_id: u32, key: &[u8], seq: u64) -> Result<LoggedWrite<'_>> {
        self.append_record(ns_id, key, None, seq)
    }

    fn append_record(&self, ns_id: u32, key: &[u8], value: Option<&[u8]>, seq: u64) -> Result<LoggedWrite<'_>> {
        let ns_bytes = ns_id.to_le_bytes();
        let seq_bytes = seq.to_le_bytes();
        let key_len = (key.len() as u32).to_le_bytes();
        let value_len = value.map_or(DELETE_VALUE_LEN, |v| v.len() as u32).to_le_bytes();
        let body: [&[u8]; 6] = [&ns_bytes, &seq_bytes, &key_len, key, &value_len, value.unwrap_or(&[])];
        // Registered under the append lock, and before the tail is published:
        // no ceiling can see a tail that includes this record without also
        // seeing it in flight.
        let (start, end) = self.log.append(&body, |start, _| {
            self.in_flight.lock().insert(start, ns_id);
        })?;
        let write = LoggedWrite { wal: self, start, end };
        self.last_seq.fetch_max(seq, Ordering::AcqRel);

        // Before the caller can apply (and drop `write`), so a seal never
        // misses a namespace with a record in its memtable.
        {
            let mut t = self.tracking.lock();
            if !t.forgotten.contains(&ns_id) {
                let e = t.last_write.entry(ns_id).or_insert(end);
                *e = (*e).max(end);
            }
        }

        Metrics::bump(&self.metrics.vector_wal_appends);
        Metrics::add(&self.metrics.vector_wal_bytes_appended, end - start);
        Ok(write)
    }

    /// Makes every append that returned before this call durable.
    ///
    /// A commit round (`db/commit_rounds.rs`): the fsync runs outside the
    /// append lock, so appends carry on meanwhile, and concurrent syncs share
    /// one fsync. Does no I/O when nothing was appended since the last one. A
    /// failure poisons the log (see the module docs).
    pub(crate) fn sync(&self) -> Result<()> {
        if self.log.sync()? {
            Metrics::bump(&self.metrics.vector_wal_fsyncs);
        }
        Ok(())
    }

    /// `ns` is about to write a memtable to L0. If the namespace has records
    /// past the durable tail, syncs the log first, so the L0 files never hold a
    /// record the log could still lose (see *What survives a crash is a prefix
    /// of the log*). Every record in the memtable was appended before it was
    /// applied, so before this call, and is covered by the namespace's latest
    /// record. No I/O for a namespace that never appended, or whose records are
    /// all durable.
    pub(crate) fn before_memtable_written(&self, ns: u32) -> Result<()> {
        let last = self.tracking.lock().last_write.get(&ns).copied();
        match last {
            Some(end) if end > self.log.durable() => {
                Metrics::bump(&self.metrics.vector_wal_flush_syncs);
                self.sync()
            }
            _ => Ok(()),
        }
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
        self.log
            .for_each(start, self.log.tail(), |at, body| -> std::result::Result<(), Wrapped<E>> {
                let record = decode_body(body, at).map_err(|e| Wrapped(E::from(e)))?;
                f(at, record).map_err(Wrapped)
            })
            .map_err(|Wrapped(e)| e)
    }

    /// Position of the next append.
    #[cfg(test)]
    pub(crate) fn tail(&self) -> u64 {
        self.log.tail()
    }

    /// Every byte below this position is on stable storage.
    pub(crate) fn durable_tail(&self) -> u64 {
        self.log.durable()
    }

    /// Highest seq of any record appended since open, or found at open at or
    /// above the retired cut (0 if none).
    pub(crate) fn last_seq(&self) -> u64 {
        self.last_seq.load(Ordering::Acquire)
    }

    /// Every segment on disk, `(segment_id, bytes)`, oldest first. A segment
    /// GC deletes while this runs is left out.
    pub(crate) fn segments(&self) -> Result<Vec<(u64, u64)>> {
        Ok(self.log.segments()?)
    }

    // ── Retirement ──────────────────────────────────────────────────────

    /// The tail lowered to the oldest append not yet applied — of `ns` only,
    /// or of any namespace when `ns` is `None`. Reads the published tail
    /// first, then `in_flight`: an append registers there before publishing
    /// its tail, so a record below the tail read is either still found in
    /// flight or already applied.
    fn ceiling(&self, ns: Option<u32>) -> u64 {
        let tail = self.log.tail();
        let in_flight = self.in_flight.lock();
        let oldest = match ns {
            None => in_flight.keys().next().copied(),
            Some(ns) => in_flight.iter().find(|(_, n)| **n == ns).map(|(s, _)| *s),
        };
        oldest.map_or(tail, |s| tail.min(s))
    }

    /// Records below this position are retired (in SSTables); replay starts
    /// here.
    pub(crate) fn retired_cut(&self) -> u64 {
        self.retired_cut.load(Ordering::Acquire)
    }

    /// `ns` sealed memtable `version`. Called under that memtable's write
    /// lock, so no record can be applied into the next memtable meanwhile.
    /// Takes only `tracking` and `in_flight`, never a lock held across I/O.
    pub(crate) fn on_memtable_sealed(&self, ns: u32, version: u64) {
        let mut t = self.tracking.lock();
        if !t.last_write.contains_key(&ns) {
            return;
        }
        let ceiling = self.ceiling(Some(ns));
        t.seals.entry((ns, version)).or_insert(PendingFlush { ceiling, flushed: false });
    }

    /// `ns`'s memtable `version` is written to L0 and fsynced, value log
    /// fsynced, and its files are **not yet registered** (so no merge can have
    /// dropped a tombstone in them). Advances the namespace's `safe_offset`
    /// over its written versions in order, then writes the marker.
    ///
    /// The change takes effect only once the marker is on disk. If it cannot
    /// be written, nothing changes and the error vetoes the flush: the LSM
    /// removes the files and flushes the memtable again later, under the same
    /// version.
    ///
    /// A version with no seal record (the seal came before the namespace's
    /// first append, or before this log observed it) retires nothing but
    /// lets later versions fold past it; a later flush covers its records.
    pub(crate) fn on_memtable_written(&self, ns: u32, version: u64) -> Result<()> {
        let mut r = self.retirement.lock();
        // Under `retirement`, which serialises written events: nothing else
        // changes this namespace's seals meanwhile, except new seals of later
        // versions.
        let (folded, ceiling, recorded) = {
            let mut t = self.tracking.lock();
            if !t.last_write.contains_key(&ns) {
                t.seals.remove(&(ns, version));
                return Ok(());
            }
            let recorded = t.seals.contains_key(&(ns, version));
            t.seals
                .entry((ns, version))
                .or_insert(PendingFlush { ceiling: 0, flushed: false })
                .flushed = true;
            let mut folded = Vec::new();
            let mut ceiling = 0u64;
            for (&key, p) in t.seals.range((ns, 0)..=(ns, u64::MAX)) {
                if !p.flushed {
                    break;
                }
                ceiling = ceiling.max(p.ceiling);
                folded.push(key);
            }
            (folded, ceiling, recorded)
        };
        let before = r.safe.get(&ns).copied();
        r.safe.insert(ns, before.unwrap_or(0).max(ceiling));
        match self.persist_progress(&mut r) {
            Ok(()) => {
                let mut t = self.tracking.lock();
                for key in folded {
                    t.seals.remove(&key);
                }
                Ok(())
            }
            Err(e) => {
                // Undo, so no later marker carries this flush.
                match before {
                    Some(s) => r.safe.insert(ns, s),
                    None => r.safe.remove(&ns),
                };
                let mut t = self.tracking.lock();
                if recorded {
                    if let Some(p) = t.seals.get_mut(&(ns, version)) {
                        p.flushed = false;
                    }
                } else {
                    t.seals.remove(&(ns, version));
                }
                Err(e)
            }
        }
    }

    /// Records that every record of `namespaces` so far is in an SSTable:
    /// recovery replayed them with no observer attached, then flushed those
    /// namespaces to L0 and fsynced their value logs. Writes the marker before
    /// returning; run at open, before anything can merge.
    pub(crate) fn retire_replayed(&self, namespaces: &[u32]) -> Result<()> {
        let mut r = self.retirement.lock();
        // Nothing is in flight at open, so this is the tail.
        let tail = self.ceiling(None);
        for &ns in namespaces {
            let s = r.safe.entry(ns).or_insert(0);
            *s = (*s).max(tail);
        }
        self.persist_progress(&mut r)
    }

    /// Stop tracking a dropped namespace: it can never flush again, and replay
    /// skips its records. It is never tracked again.
    pub(crate) fn forget_namespace(&self, ns: u32) {
        let mut r = self.retirement.lock();
        {
            let mut t = self.tracking.lock();
            t.forgotten.insert(ns);
            t.last_write.remove(&ns);
            t.seals.retain(|(n, _), _| *n != ns);
        }
        r.safe.remove(&ns);
        // A failed write keeps the older marker, which only replays more (the
        // namespace's records are skipped at replay anyway).
        if let Err(e) = self.persist_progress(&mut r) {
            log::error!("[vector index WAL] could not write the progress marker after dropping ns={ns}: {e}");
        }
    }

    /// Namespaces the log tracks (every one that has a record above where its
    /// replay starts or has appended since open), plus those the marker lists.
    pub(crate) fn tracked_namespaces(&self) -> Vec<u32> {
        let r = self.retirement.lock();
        let t = self.tracking.lock();
        let mut ns: Vec<u32> = t.last_write.keys().chain(r.safe.keys()).copied().collect();
        ns.sort_unstable();
        ns.dedup();
        ns
    }

    /// Namespaces with records no flush has covered yet, with their safe
    /// offsets.
    fn holders(r: &Retirement, t: &Tracking) -> Vec<(u32, u64)> {
        let mut h: Vec<(u32, u64)> = t
            .last_write
            .iter()
            .filter_map(|(&ns, &last)| {
                let safe = r.safe.get(&ns).copied().unwrap_or(0);
                (last > safe).then_some((ns, safe))
            })
            .collect();
        h.sort_unstable();
        h
    }

    /// Namespaces with records no flush has covered yet.
    pub(crate) fn namespaces_holding_cut(&self) -> Vec<u32> {
        let r = self.retirement.lock();
        let t = self.tracking.lock();
        Self::holders(&r, &t).into_iter().map(|(ns, _)| ns).collect()
    }

    /// Writes the marker when the cut or any namespace's safe offset above it
    /// changed, then publishes the cut. The cut only moves forward. On an
    /// error the marker on disk and the published cut are unchanged.
    fn persist_progress(&self, r: &mut Retirement) -> Result<()> {
        // The ceiling before the latest writes: a record not in flight any
        // more was added to `last_write` before it left `in_flight`.
        let ceiling = self.ceiling(None);
        let slowest = {
            let t = self.tracking.lock();
            Self::holders(r, &t).into_iter().map(|(_, safe)| safe).min()
        };
        let cut = slowest.map_or(ceiling, |s| s.min(ceiling)).max(self.retired_cut());
        let mut safe: Vec<(u32, u64)> = r.safe.iter().filter(|(_, s)| **s > cut).map(|(n, s)| (*n, *s)).collect();
        safe.sort_unstable();
        let next = Progress { cut, safe };
        if next == r.written {
            return Ok(());
        }
        #[cfg(test)]
        if self.fail_next_marker_write.swap(false, Ordering::SeqCst) {
            return Err(std::io::Error::other("injected marker write failure").into());
        }
        crate::support::write_atomic_durable(&self.dir().join(RETIRED_CUT_FILE), &progress_bytes(&next)).map_err(|e| {
            log::error!("[vector index WAL] could not write the progress marker: {e}");
            VectorIndexWalError::Io(e)
        })?;
        self.retired_cut.store(next.cut, Ordering::Release);
        r.written = next;
        Ok(())
    }

    /// Hands every record that replay must apply to `f`, in log order: those
    /// at or above their namespace's durable safe offset (the cut, for a
    /// namespace the marker does not list), as of this open. Run once at
    /// open, before any write.
    pub(crate) fn replay<E: From<VectorIndexWalError>>(
        &self,
        mut f: impl FnMut(RecordRef<'_>) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        let progress = &self.opened_progress;
        self.for_each_from(progress.cut, |at, r| if at >= progress.replay_from(r.ns_id) { f(r) } else { Ok(()) })
    }

    /// Test hook: whether fsyncs issued by this process (or the reopen) cover
    /// every byte below `end`.
    #[cfg(test)]
    pub(crate) fn covered(&self, end: u64) -> bool {
        self.log.covered(end)
    }

    /// Test hook for the power-loss model (`SegmentedLog::power_loss_extents`).
    #[cfg(test)]
    pub(crate) fn power_loss_extents(&self) -> Vec<(PathBuf, u64)> {
        self.log.power_loss_extents()
    }

    /// Test-only: make the next `sync` fail (and so poison the log).
    #[cfg(test)]
    pub(crate) fn fail_next_sync(&self) {
        self.log.fail_next_sync();
    }

    /// Test-only: make the next marker write fail.
    #[cfg(test)]
    pub(crate) fn fail_next_marker_write(&self) {
        self.fail_next_marker_write.store(true, Ordering::SeqCst);
    }

    /// Test-only: zero every byte no fsync covered, in each segment this
    /// process wrote, keeping the files' sizes, as a power loss would.
    /// `AsyncDb::crash` calls it, so a crash test can see a missing `sync`.
    #[cfg(test)]
    pub(crate) fn lose_unsynced(&self) {
        self.log.lose_unsynced();
    }

    /// Deletes every segment wholly below the retired cut, oldest first, then
    /// fsyncs the directory. Returns `(segments, bytes)` deleted.
    pub(crate) fn garbage_collect(&self) -> Result<(u64, u64)> {
        Metrics::bump(&self.metrics.vector_wal_gc_runs);
        let (segments, bytes) = self.log.delete_segments_below(segment_of(self.retired_cut()))?;
        if segments > 0 {
            Metrics::add(&self.metrics.vector_wal_segments_deleted, segments);
        }
        Ok((segments, bytes))
    }

    /// Whether any segment lies wholly below the retired cut.
    pub(crate) fn has_deletable_segments(&self) -> Result<bool> {
        Ok(self.log.oldest_segment()?.is_some_and(|id| id < segment_of(self.retired_cut())))
    }

    /// The log's state, naming namespaces with `name_of`.
    pub(crate) fn stats(&self, name_of: impl Fn(u32) -> String) -> Result<VectorIndexWalStats> {
        let holders = {
            let r = self.retirement.lock();
            let t = self.tracking.lock();
            Self::holders(&r, &t)
        };
        let tail = self.log.tail();
        let cut = self.retired_cut();
        let segments: Vec<VectorIndexWalSegment> = self
            .segments()?
            .into_iter()
            .map(|(segment_id, bytes)| VectorIndexWalSegment { segment_id, bytes })
            .collect();
        Ok(VectorIndexWalStats {
            segment_count: segments.len(),
            disk_bytes: segments.iter().map(|s| s.bytes).sum(),
            segments,
            tail: VectorIndexWalPosition::of(tail),
            durable_tail: VectorIndexWalPosition::of(self.durable_tail()),
            retired_cut: VectorIndexWalPosition::of(cut),
            // Counted from where each segment's records end, not its file
            // size: a segment is created at its full size.
            unretired_bytes: self.log.bytes_between(cut, tail),
            last_sequence: self.last_seq(),
            holding_cut: holders
                .into_iter()
                .map(|(ns, safe)| VectorIndexWalHolder {
                    namespace: name_of(ns),
                    safe_offset: VectorIndexWalPosition::of(safe),
                })
                .collect(),
            stopped: self.log.poisoned(),
        })
    }
}

/// An error of a [`VectorIndexWal::for_each_from`] caller, carried through
/// the segmented log's reader.
struct Wrapped<E>(E);

impl<E: From<VectorIndexWalError>> From<LogError> for Wrapped<E> {
    fn from(e: LogError) -> Self {
        Wrapped(E::from(VectorIndexWalError::from(e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::segmented_log::{END_MARKER, END_MARKER_LEN, FRAME_HEADER_LEN, SEGMENT_HEADER_LEN, position, prep_path, segment_path, split};
    use std::fs::{File, OpenOptions};
    use std::os::unix::fs::FileExt;
    use std::sync::atomic::Ordering::Relaxed;
    use tempfile::TempDir;

    fn check_header(file: &File, segment_id: u64) -> Result<()> {
        Ok(segmented_log::check_header(&FORMAT, file, segment_id)?)
    }

    fn header_bytes(segment_id: u64) -> [u8; SEGMENT_HEADER_LEN as usize] {
        segmented_log::header_bytes(&FORMAT, segment_id)
    }

    type Row = (u64, u32, u64, Vec<u8>, Option<Vec<u8>>);

    fn open_at(dir: &Path, segment_size: u32) -> (VectorIndexWal, OpenReport) {
        VectorIndexWal::open(dir, segment_size, Arc::new(Metrics::default())).unwrap()
    }

    fn rows_from(wal: &VectorIndexWal, start: u64) -> Vec<Row> {
        let mut rows = Vec::new();
        wal.for_each_from(start, |at, r| -> Result<()> {
            rows.push((at, r.ns_id, r.seq, r.key.to_vec(), r.value.map(<[u8]>::to_vec)));
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
        (0..n)
            .map(|i| wal.append((i % 3) as u32, &key(i), &value(i), i + 1).unwrap().appended())
            .collect()
    }

    fn expected(appended: &[Appended], range: std::ops::Range<u64>) -> Vec<Row> {
        range
            .map(|i| (appended[i as usize].start, (i % 3) as u32, i + 1, key(i), Some(value(i))))
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
    fn a_delete_reads_back_as_a_delete_and_a_torn_one_is_dropped() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        let put = wal.append(1, b"k", b"v", 1).unwrap().appended();
        let del = wal.append_delete(1, b"k", 2).unwrap().appended();
        let empty = wal.append(1, b"e", b"", 3).unwrap().appended(); // an empty value is still a put
        wal.sync().unwrap();
        assert_eq!(del.end - del.start, u64::from(FRAME_HEADER_LEN) + BODY_FIXED_LEN as u64 + 1);
        let want = vec![
            (put.start, 1, 1, b"k".to_vec(), Some(b"v".to_vec())),
            (del.start, 1, 2, b"k".to_vec(), None),
            (empty.start, 1, 3, b"e".to_vec(), Some(vec![])),
        ];
        assert_eq!(rows_from(&wal, 0), want);
        drop(wal);
        let (wal, report) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        assert_eq!(report.last_seq, 3);
        assert_eq!(rows_from(&wal, 0), want);
        drop(wal);

        // Torn inside the delete: it goes, with everything after it.
        set_len(dir.path(), 0, u64::from(split(del.end).1) - 1);
        let (wal, report) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        assert_eq!(report.tail, del.start);
        assert_eq!(rows_from(&wal, 0), want[..1].to_vec());
    }

    #[test]
    fn reopen_finds_the_tail_and_the_highest_seq() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        wal.append(1, b"a", b"x", 10).unwrap();
        let last = wal.append(1, b"b", b"y", 5).unwrap().appended(); // seqs are not in log order
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
        assert_eq!(rows[2], (more.start, 2, 11, b"c".to_vec(), Some(b"z".to_vec())));
    }

    #[test]
    fn records_span_segments_and_every_sealed_segment_ends_in_a_marker() {
        let dir = TempDir::new().unwrap();
        // Three records fit a segment: 16 + 3 * 76 + 4 = 248 <= 256.
        let (wal, _) = open_at(dir.path(), 256);
        let appended = append_n(&wal, 20);
        let segs = wal.segments().unwrap();
        assert_eq!(segs.len(), 7);
        for &(id, len) in &segs {
            assert_eq!(len, 256, "segment {id} is created at its full size");
        }
        for &(id, _) in &segs[..segs.len() - 1] {
            let f = File::open(segment_path(dir.path(), id)).unwrap();
            check_header(&f, id).unwrap();
            let mut m = [0u8; 4];
            f.read_exact_at(&mut m, u64::from(SEGMENT_HEADER_LEN + 3 * FRAME)).unwrap();
            assert_eq!(m, END_MARKER.to_le_bytes(), "segment {id} ends in the end marker");
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
    fn a_torn_tail_is_dropped_and_later_appends_read_back() {
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
            let fresh = wal.append(9, b"new", b"v", 99).unwrap().appended();
            assert_eq!(fresh.start, appended[4].start);
            let mut want = expected(&appended, 0..4);
            want.push((fresh.start, 9, 99, b"new".to_vec(), Some(b"v".to_vec())));
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
            // Not `u32::MAX`: that is the end marker.
            f.write_all_at(&(u32::MAX - 1).to_le_bytes(), u64::from(byte)).unwrap();
        };

        write_len(appended[7].start);
        let (_, report) = reopen(dir.path(), 256).unwrap();
        assert_eq!(report.tail, appended[7].start);
        // The frame claims more than the segment holds: capped at its end.
        assert_eq!(report.torn_bytes_dropped, 256 - u64::from(split(appended[7].start).1));

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
                assert_eq!(v, Some(vec![ns as u8; 30]));
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
        assert_eq!(wal.segments().unwrap(), vec![(0, u64::from(DEFAULT_SEGMENT_SIZE))]);
        assert_eq!(rows_from(&wal, 0), vec![]);
        drop(wal);
        std::fs::write(path.join("notes.txt"), b"not a segment").unwrap();
        let (wal, report) = open_at(&path, DEFAULT_SEGMENT_SIZE);
        assert_eq!(report, empty);
        assert_eq!(wal.segments().unwrap().len(), 1);
    }

    // ── Retirement and GC ───────────────────────────────────────────────

    /// Appends `n` records for `ns` and lets them count as applied.
    fn applied(wal: &VectorIndexWal, ns: u32, n: u64) -> Vec<Appended> {
        (0..n).map(|i| wal.append(ns, &key(i), &value(i), i + 1).unwrap().appended()).collect()
    }

    fn seal_and_flush(wal: &VectorIndexWal, ns: u32, version: u64) {
        wal.on_memtable_sealed(ns, version);
        wal.on_memtable_written(ns, version).unwrap();
    }

    #[test]
    fn a_flush_retires_its_memtables_records_and_the_cut_survives_reopen() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        applied(&wal, 1, 3);
        assert_eq!(wal.retired_cut(), 0);
        assert_eq!(wal.namespaces_holding_cut(), vec![1]);
        seal_and_flush(&wal, 1, 0);
        assert_eq!(wal.retired_cut(), wal.tail());
        assert!(wal.namespaces_holding_cut().is_empty());
        let cut = wal.retired_cut();
        drop(wal);
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        assert_eq!(wal.retired_cut(), cut);
        assert_eq!(rows_from(&wal, wal.retired_cut()), vec![]);
    }

    #[test]
    fn memtables_retire_in_version_order_only() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        applied(&wal, 1, 1);
        wal.on_memtable_sealed(1, 0);
        applied(&wal, 1, 1);
        wal.on_memtable_sealed(1, 1);
        wal.on_memtable_written(1, 1).unwrap();
        assert_eq!(wal.retired_cut(), 0, "version 0 is not flushed yet");
        wal.on_memtable_written(1, 0).unwrap();
        assert_eq!(wal.retired_cut(), wal.tail());
    }

    #[test]
    fn a_record_not_yet_applied_holds_the_cut() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        applied(&wal, 1, 2);
        let in_flight = wal.append(1, b"late", b"v", 9).unwrap();
        // Its apply will land in the next memtable, so this seal must not claim it.
        seal_and_flush(&wal, 1, 0);
        assert_eq!(wal.retired_cut(), in_flight.start);
        assert_eq!(wal.namespaces_holding_cut(), vec![1]);
        drop(in_flight);
        seal_and_flush(&wal, 1, 1);
        assert_eq!(wal.retired_cut(), wal.tail());
    }

    #[test]
    fn another_namespaces_record_in_flight_does_not_hold_a_flushed_namespace() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        applied(&wal, 1, 2);
        let other = wal.append(2, b"k", b"v", 9).unwrap();
        applied(&wal, 1, 1);
        seal_and_flush(&wal, 1, 0);
        // Namespace 1 flushed everything it wrote; only 2 holds the cut.
        assert_eq!(wal.namespaces_holding_cut(), vec![2]);
        assert_eq!(wal.retired_cut(), 0);
        drop(other);
        seal_and_flush(&wal, 2, 0);
        assert_eq!(wal.retired_cut(), wal.tail());
        assert!(wal.namespaces_holding_cut().is_empty());
    }

    #[test]
    fn the_cut_is_the_slowest_unflushed_namespace() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        applied(&wal, 2, 1);
        seal_and_flush(&wal, 2, 0);
        let after_two = wal.retired_cut();
        assert_eq!(after_two, wal.tail());
        applied(&wal, 1, 2);
        applied(&wal, 2, 1);
        seal_and_flush(&wal, 1, 0);
        assert_eq!(wal.namespaces_holding_cut(), vec![2]);
        assert_eq!(wal.retired_cut(), after_two, "namespace 2's safe offset");
        seal_and_flush(&wal, 2, 1);
        assert_eq!(wal.retired_cut(), wal.tail());
    }

    #[test]
    fn a_flush_without_a_seal_record_retires_nothing() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        applied(&wal, 1, 2);
        wal.on_memtable_written(1, 0).unwrap();
        assert_eq!(wal.retired_cut(), 0);
        assert_eq!(wal.namespaces_holding_cut(), vec![1]);
        seal_and_flush(&wal, 1, 1);
        assert_eq!(wal.retired_cut(), wal.tail());
    }

    #[test]
    fn events_for_a_namespace_that_never_appended_are_ignored() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        seal_and_flush(&wal, 7, 0);
        applied(&wal, 1, 1);
        seal_and_flush(&wal, 7, 1);
        assert_eq!(wal.retired_cut(), 0);
        assert_eq!(wal.namespaces_holding_cut(), vec![1]);
    }

    #[test]
    fn forgetting_a_namespace_releases_the_cut() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        applied(&wal, 1, 1);
        applied(&wal, 2, 1);
        seal_and_flush(&wal, 1, 0);
        assert_eq!(wal.retired_cut(), 0);
        wal.forget_namespace(2);
        assert_eq!(wal.retired_cut(), wal.tail());
        assert!(wal.namespaces_holding_cut().is_empty());
    }

    #[test]
    fn gc_deletes_only_segments_wholly_below_the_cut() {
        let dir = TempDir::new().unwrap();
        let metrics = Arc::new(Metrics::default());
        let (wal, _) = VectorIndexWal::open(dir.path(), 256, Arc::clone(&metrics)).unwrap();
        applied(&wal, 1, 8); // segments 0, 1, 2
        assert!(!wal.has_deletable_segments().unwrap());
        assert_eq!(wal.garbage_collect().unwrap(), (0, 0));
        seal_and_flush(&wal, 1, 0);
        assert_eq!(split(wal.retired_cut()).0, 2);
        assert!(wal.has_deletable_segments().unwrap());
        // Segment files are their full size.
        assert_eq!(wal.garbage_collect().unwrap(), (2, 2 * 256));
        assert_eq!(wal.segments().unwrap().iter().map(|s| s.0).collect::<Vec<_>>(), vec![2]);
        assert!(!wal.has_deletable_segments().unwrap());
        let s = metrics.snapshot();
        assert_eq!((s.vector_wal_gc_runs, s.vector_wal_segments_deleted), (2, 2));

        let more = applied(&wal, 1, 2);
        let cut = wal.retired_cut();
        drop(wal);
        let (wal, _) = open_at(dir.path(), 256);
        assert_eq!(wal.retired_cut(), cut);
        let rows = rows_from(&wal, cut);
        assert_eq!(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
            more.iter().map(|a| a.start).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_bad_retired_cut_marker_fails_the_open() {
        let fresh = || {
            let dir = TempDir::new().unwrap();
            let (wal, _) = open_at(dir.path(), 256);
            applied(&wal, 1, 8);
            seal_and_flush(&wal, 1, 0);
            wal.garbage_collect().unwrap();
            dir
        };
        let marker = |dir: &TempDir| dir.path().join(RETIRED_CUT_FILE);

        let dir = fresh();
        std::fs::write(
            marker(&dir),
            progress_bytes(&Progress {
                cut: position(9, 16),
                safe: vec![],
            }),
        )
        .unwrap();
        assert!(corruption(reopen(dir.path(), 256)).contains("past the last segment"));

        let dir = fresh();
        let cut = read_progress(dir.path()).unwrap().unwrap().cut;
        std::fs::write(
            marker(&dir),
            progress_bytes(&Progress {
                cut,
                safe: vec![(1, cut - 1)],
            }),
        )
        .unwrap();
        assert!(corruption(reopen(dir.path(), 256)).contains("below the cut"));

        let dir = fresh();
        std::fs::write(
            marker(&dir),
            progress_bytes(&Progress {
                cut: position(0, 16),
                safe: vec![],
            }),
        )
        .unwrap();
        assert!(corruption(reopen(dir.path(), 256)).contains("below the oldest segment"));

        let dir = fresh();
        std::fs::remove_file(marker(&dir)).unwrap();
        assert!(corruption(reopen(dir.path(), 256)).contains("no progress was saved"));

        let dir = fresh();
        let mut bytes = std::fs::read(marker(&dir)).unwrap();
        bytes[5] ^= 0xff;
        std::fs::write(marker(&dir), bytes).unwrap();
        assert!(corruption(reopen(dir.path(), 256)).contains("damaged"));
    }

    #[test]
    fn the_cut_never_moves_back() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        applied(&wal, 1, 2);
        seal_and_flush(&wal, 1, 0);
        let cut = wal.retired_cut();
        // A namespace that starts writing now has a safe offset of 0 until it
        // flushes; that must not drag the cut back.
        applied(&wal, 2, 1);
        wal.forget_namespace(3);
        assert_eq!(wal.retired_cut(), cut);
        drop(wal);
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        assert_eq!(wal.retired_cut(), cut);
    }

    #[test]
    fn stats_report_positions_unretired_bytes_and_holders() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), 256);
        applied(&wal, 1, 4); // segment 0 sealed with 3, segment 1 holds 1
        applied(&wal, 2, 1);
        let name = |ns: u32| format!("ns{ns}");
        let s = wal.stats(name).unwrap();
        assert_eq!(s.segment_count, 2);
        assert_eq!(
            s.tail,
            VectorIndexWalPosition {
                segment_id: 1,
                byte: SEGMENT_HEADER_LEN + 2 * FRAME
            }
        );
        assert_eq!(s.retired_cut, VectorIndexWalPosition { segment_id: 0, byte: 0 });
        assert_eq!(s.unretired_bytes, u64::from(5 * FRAME), "record bytes only: no headers, no end marker");
        assert_eq!(s.last_sequence, 4);
        assert_eq!(s.holding_cut.iter().map(|h| h.namespace.as_str()).collect::<Vec<_>>(), vec!["ns1", "ns2"]);

        seal_and_flush(&wal, 1, 0);
        seal_and_flush(&wal, 2, 0);
        let s = wal.stats(name).unwrap();
        assert_eq!(s.retired_cut, s.tail);
        assert_eq!(s.unretired_bytes, 0);
        assert!(s.holding_cut.is_empty());
        // Sealing segment 0 fsynced it; nothing has synced segment 1 yet.
        assert_eq!(
            s.durable_tail,
            VectorIndexWalPosition {
                segment_id: 0,
                byte: SEGMENT_HEADER_LEN + 3 * FRAME
            }
        );
    }

    #[test]
    fn a_reopened_log_holds_the_cut_for_records_nobody_flushed() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        applied(&wal, 3, 1);
        seal_and_flush(&wal, 3, 0);
        let cut = wal.retired_cut();
        let unflushed = applied(&wal, 1, 2);
        drop(wal); // a crash: the in-memory state is gone, the records are not

        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        // Namespace 3 has nothing above the cut, so only 1 holds it.
        assert_eq!(wal.namespaces_holding_cut(), vec![1]);
        assert_eq!(wal.tracked_namespaces(), vec![1]);
        // Another namespace's flush must not carry the cut past 1's records.
        applied(&wal, 2, 1);
        seal_and_flush(&wal, 2, 0);
        assert_eq!(wal.retired_cut(), cut);
        assert!(wal.retired_cut() <= unflushed[0].start);
        // Only 1's own flush releases it.
        seal_and_flush(&wal, 1, 0);
        assert_eq!(wal.retired_cut(), wal.tail());
    }

    #[test]
    fn the_marker_round_trips_with_per_namespace_offsets() {
        let p = Progress {
            cut: position(1, 40),
            safe: vec![(2, position(1, 90)), (7, position(2, 16))],
        };
        let dir = TempDir::new().unwrap();
        crate::support::write_atomic_durable(&dir.path().join(RETIRED_CUT_FILE), &progress_bytes(&p)).unwrap();
        assert_eq!(read_progress(dir.path()).unwrap(), Some(p.clone()));
        assert_eq!(p.replay_from(2), position(1, 90));
        assert_eq!(p.replay_from(7), position(2, 16));
        assert_eq!(p.replay_from(3), position(1, 40), "an unlisted namespace replays from the cut");
        let mut bytes = progress_bytes(&p);
        bytes.truncate(bytes.len() - 5);
        std::fs::write(dir.path().join(RETIRED_CUT_FILE), &bytes).unwrap();
        assert!(read_progress(dir.path()).is_err());
    }

    #[test]
    fn replay_skips_each_namespaces_records_below_its_own_safe_offset() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        let flushed = applied(&wal, 1, 2);
        let held = applied(&wal, 2, 1);
        let after = applied(&wal, 1, 1);
        wal.on_memtable_sealed(1, 0); // covers all of namespace 1's records so far
        wal.on_memtable_written(1, 0).unwrap();
        assert_eq!(wal.retired_cut(), 0, "namespace 2 holds the global cut");
        drop(wal);

        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        let mut replayed = Vec::new();
        wal.replay(|r| -> Result<()> {
            replayed.push((r.ns_id, r.seq));
            Ok(())
        })
        .unwrap();
        // Only namespace 2's record: namespace 1's are below its own offset,
        // although they are above the global cut.
        assert_eq!(replayed, vec![(2, 1)]);
        let _ = (flushed, held, after);
    }

    #[test]
    fn a_marker_past_the_durable_tail_starts_a_new_segment() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        applied(&wal, 1, 3); // never synced
        seal_and_flush(&wal, 1, 0); // but written to L0: the marker names the tail
        let marked = wal.retired_cut();
        wal.lose_unsynced(); // power loss
        drop(wal);

        let (wal, report) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        assert_eq!(report.tail, position(1, SEGMENT_HEADER_LEN));
        assert!(wal.tail() > marked, "new appends land above every position the marker names");
        let fresh = wal.append(1, b"new", b"v", 9).unwrap().appended();
        drop(wal);
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        let mut replayed = Vec::new();
        wal.replay(|r| -> Result<()> {
            replayed.push(r.key.to_vec());
            Ok(())
        })
        .unwrap();
        assert_eq!(replayed, vec![b"new".to_vec()], "the new record at {:?} is replayed", fresh);
    }

    /// A marker that cannot be written changes nothing: the event fails (so
    /// the LSM undoes the flush), the cut and the marker stay, no later marker
    /// carries the failed flush, and the retry retires the records.
    #[test]
    fn a_failed_marker_write_changes_nothing_and_the_retry_retires() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        let ns1 = applied(&wal, 1, 3);
        wal.on_memtable_sealed(1, 0);
        let marker_before = read_progress(dir.path()).unwrap();
        wal.fail_next_marker_write();
        assert!(wal.on_memtable_written(1, 0).is_err());
        assert_eq!(wal.retired_cut(), 0);
        assert_eq!(wal.namespaces_holding_cut(), vec![1]);
        assert_eq!(read_progress(dir.path()).unwrap(), marker_before);

        // Another namespace's flush writes a marker: it must not carry 1's.
        applied(&wal, 2, 1);
        seal_and_flush(&wal, 2, 0);
        assert!(wal.retired_cut() <= ns1[0].start, "1's records are not retired");
        let marker = read_progress(dir.path()).unwrap().unwrap();
        assert!(marker.replay_from(1) <= ns1[0].start);

        // The LSM retries the same memtable version.
        wal.on_memtable_written(1, 0).unwrap();
        assert!(wal.namespaces_holding_cut().is_empty());
        assert_eq!(wal.retired_cut(), wal.tail());
    }

    /// A dropped namespace is never tracked again: a write that raced the
    /// drop would otherwise hold the cut until the next restart.
    #[test]
    fn an_append_after_a_namespace_is_forgotten_does_not_hold_the_cut() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        applied(&wal, 1, 1);
        wal.forget_namespace(1);
        applied(&wal, 1, 1); // a late write that raced the drop
        assert!(wal.namespaces_holding_cut().is_empty());
        wal.on_memtable_sealed(1, 0);
        wal.on_memtable_written(1, 0).unwrap();
        applied(&wal, 2, 1);
        seal_and_flush(&wal, 2, 0);
        assert_eq!(wal.retired_cut(), wal.tail(), "nothing holds the cut");
    }

    /// After an fsync fails, the log refuses appends and syncs until it is
    /// reopened, and the stats say why.
    #[test]
    fn a_failed_fsync_stops_the_log_until_it_is_reopened() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        applied(&wal, 1, 1);
        wal.fail_next_sync();
        assert!(matches!(wal.sync(), Err(VectorIndexWalError::Poisoned(_))));
        assert!(matches!(wal.append(1, b"k", b"v", 9), Err(VectorIndexWalError::Poisoned(_))));
        assert!(
            matches!(wal.sync(), Err(VectorIndexWalError::Poisoned(_))),
            "a later sync must not succeed"
        );
        assert!(wal.stats(|n| n.to_string()).unwrap().stopped.is_some());
        drop(wal);
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        wal.append(1, b"k", b"v", 9).unwrap();
        wal.sync().unwrap();
        assert!(wal.stats(|n| n.to_string()).unwrap().stopped.is_none());
    }

    /// Concurrent syncs share fsyncs (commit rounds, as in the main WAL), and
    /// each still returns only once every append made before it is durable.
    #[test]
    fn concurrent_syncs_share_fsyncs() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        let before = wal.metrics.vector_wal_fsyncs.load(Ordering::SeqCst);
        let seq = AtomicU64::new(1);
        std::thread::scope(|s| {
            for t in 0..16u32 {
                let (wal, seq) = (&wal, &seq);
                s.spawn(move || {
                    for i in 0..50u32 {
                        let write = wal
                            .append(t, format!("k{i}").as_bytes(), b"v", seq.fetch_add(1, Ordering::SeqCst))
                            .unwrap();
                        let end = write.end;
                        drop(write);
                        wal.sync().unwrap();
                        assert!(wal.durable_tail() >= end, "a sync returned before its append was durable");
                    }
                });
            }
        });
        let fsyncs = wal.metrics.vector_wal_fsyncs.load(Ordering::SeqCst) - before;
        assert!(
            fsyncs < 800,
            "16 threads syncing concurrently never shared an fsync ({fsyncs} fsyncs for 800 syncs)"
        );
    }

    /// Syncs that arrive while others are pending share one fsync: held at the
    /// commit point together, then released, eight syncs cost one fsync, not
    /// eight. (Without commit rounds each would fsync on its own, since none
    /// sees another's fsync finish first.)
    #[test]
    fn syncs_waiting_together_share_one_fsync() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        let before = wal.metrics.vector_wal_fsyncs.load(Ordering::SeqCst);
        wal.log.rounds.freeze(true);
        std::thread::scope(|s| {
            let syncs: Vec<_> = (0..8u32)
                .map(|t| {
                    let wal = &wal;
                    s.spawn(move || {
                        drop(wal.append(t, b"k", b"v", u64::from(t) + 1).unwrap());
                        wal.sync()
                    })
                })
                .collect();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while wal.log.rounds.frozen_callers() < 8 {
                assert!(std::time::Instant::now() < deadline, "the syncs never reached the commit point");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            wal.log.rounds.freeze(false);
            for sync in syncs {
                sync.join().unwrap().unwrap();
            }
        });
        assert_eq!(
            wal.metrics.vector_wal_fsyncs.load(Ordering::SeqCst) - before,
            1,
            "eight waiting syncs should share one fsync"
        );
        assert_eq!(wal.durable_tail(), wal.tail());
    }

    /// A failed fsync fails every sync waiting on that round, and stops the log.
    #[test]
    fn a_failed_round_fails_every_waiting_sync() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        wal.log.rounds.freeze(true);
        std::thread::scope(|s| {
            let syncs: Vec<_> = (0..4u32)
                .map(|t| {
                    let wal = &wal;
                    s.spawn(move || {
                        drop(wal.append(t, b"k", b"v", u64::from(t) + 1).unwrap());
                        wal.sync()
                    })
                })
                .collect();
            while wal.log.rounds.frozen_callers() < 4 {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            wal.fail_next_sync();
            wal.log.rounds.freeze(false);
            for sync in syncs {
                assert!(
                    matches!(sync.join().unwrap(), Err(VectorIndexWalError::Poisoned(_))),
                    "a waiting sync succeeded"
                );
            }
        });
        assert!(wal.stats(|n| n.to_string()).unwrap().stopped.is_some());
    }

    /// Segments wholly below the cut are retired: open does not read them, so
    /// damage there (they only wait for GC) does not stop the database.
    #[test]
    fn open_does_not_read_segments_below_the_cut() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), 256);
        let records = applied(&wal, 1, 8); // segments 0, 1, 2
        seal_and_flush(&wal, 1, 0);
        assert_eq!(split(wal.retired_cut()).0, 2);
        let last_seq = wal.last_seq();
        drop(wal);
        flip(dir.path(), 0, u64::from(SEGMENT_HEADER_LEN + FRAME_HEADER_LEN + 2));
        let (wal, report) = open_at(dir.path(), 256);
        assert_eq!(report.tail, records.last().unwrap().end);
        assert!(wal.namespaces_holding_cut().is_empty());
        // Nothing at or above the cut, so no seq to report from the log.
        assert!(report.last_seq < last_seq);
    }

    /// A seal runs under a memtable's write lock, for every namespace of the
    /// database. It must never wait for `retirement`, which is held across
    /// the marker's fsyncs.
    #[test]
    fn a_seal_never_waits_for_a_marker_write() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), DEFAULT_SEGMENT_SIZE);
        applied(&wal, 1, 1);
        let held = wal.retirement.lock(); // as during a marker write
        std::thread::scope(|s| {
            let (tx, rx) = std::sync::mpsc::channel();
            let wal = &wal;
            s.spawn(move || {
                wal.on_memtable_sealed(1, 0); // tracked
                wal.on_memtable_sealed(7, 0); // untracked
                tx.send(()).unwrap();
            });
            let done = rx.recv_timeout(std::time::Duration::from_secs(5));
            drop(held);
            assert!(done.is_ok(), "a seal waited for the retirement lock");
        });
    }

    // ── Wired into Database ─────────────────────────────────────────────

    use crate::db::database::Database;
    use crate::db::test_support::create_db_config;

    /// Logs and applies one write to `ns`, as step 4's worker will.
    fn log_and_apply(db: &Database, ns: u32, key: &[u8], value: &[u8]) {
        let seq = db.next_seq.fetch_add(1, Ordering::SeqCst);
        let logged = db.vector_index_wal.append(ns, key, value, seq).unwrap();
        // Applied with the logged seq, and synced before the caller goes on,
        // as step 4's worker will (it completes an entry only after a sync).
        db.get_store(ns).unwrap().replay_upsert(key, value, seq).unwrap();
        drop(logged);
        db.vector_index_wal.sync().unwrap();
    }

    #[test]
    fn a_namespace_flush_retires_its_logged_writes() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = db.create_namespace("v_sparse_vector").unwrap();
        for i in 0..5u64 {
            log_and_apply(&db, ns, &key(i), &value(i));
        }
        assert_eq!(db.vector_index_wal.retired_cut(), 0);
        db.get_store(ns).unwrap().flush_memtable_to_level0().unwrap();
        assert_eq!(db.vector_index_wal.retired_cut(), db.vector_index_wal.tail());
        let stats = db.vector_index_wal_stats().unwrap();
        assert_eq!(stats.unretired_bytes, 0);
        assert!(stats.holding_cut.is_empty());
    }

    #[test]
    fn the_gc_pass_flushes_an_idle_namespace_holding_the_cut() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let busy = db.create_namespace("busy").unwrap();
        let idle = db.create_namespace("idle").unwrap();
        log_and_apply(&db, idle, b"k", b"v");
        log_and_apply(&db, busy, b"k", b"v");
        db.get_store(busy).unwrap().flush_memtable_to_level0().unwrap();
        let stats = db.vector_index_wal_stats().unwrap();
        assert_eq!(stats.holding_cut.iter().map(|h| h.namespace.as_str()).collect::<Vec<_>>(), vec!["idle"]);
        assert_eq!(db.vector_index_wal.retired_cut(), 0);

        assert_eq!(db.flush_namespaces_pinning_vector_wal(), 1);
        assert_eq!(db.vector_index_wal.retired_cut(), db.vector_index_wal.tail());
        assert_eq!(db.flush_namespaces_pinning_vector_wal(), 0, "nothing holds the cut now");
        assert_eq!(db.metrics_snapshot().vector_wal_pin_flushes, 1);
    }

    #[test]
    fn dropping_a_namespace_releases_the_cut() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let kept = db.create_namespace("kept").unwrap();
        let dropped = db.create_namespace("dropped").unwrap();
        log_and_apply(&db, dropped, b"k", b"v");
        log_and_apply(&db, kept, b"k", b"v");
        db.get_store(kept).unwrap().flush_memtable_to_level0().unwrap();
        assert_eq!(db.vector_index_wal.retired_cut(), 0);
        db.remove_namespace("dropped").unwrap();
        assert_eq!(db.vector_index_wal.retired_cut(), db.vector_index_wal.tail());
    }

    #[test]
    fn a_database_reopens_with_its_retired_cut() {
        let dir = TempDir::new().unwrap();
        let cut = {
            let db = Database::open(dir.path(), create_db_config()).unwrap();
            let ns = db.create_namespace("v").unwrap();
            log_and_apply(&db, ns, b"k", b"v");
            db.get_store(ns).unwrap().flush_memtable_to_level0().unwrap();
            let cut = db.vector_index_wal.retired_cut();
            db.shutdown().unwrap();
            cut
        };
        assert!(cut > 0);
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        assert_eq!(db.vector_index_wal.retired_cut(), cut);
    }

    #[test]
    fn a_restart_keeps_records_nobody_flushed() {
        let dir = TempDir::new().unwrap();
        let (v, first) = {
            let db = Database::open(dir.path(), create_db_config()).unwrap();
            let v = db.create_namespace("v").unwrap();
            db.create_namespace("u").unwrap();
            let first = db.vector_index_wal.tail();
            log_and_apply(&db, v, b"k1", b"x");
            log_and_apply(&db, v, b"k2", b"y");
            // A namespace that no longer exists after the restart.
            db.vector_index_wal.append(999, b"gone", b"z", 1).unwrap();
            db.vector_index_wal.sync().unwrap();
            db.mark_crashed();
            (v, first)
        };

        let db = Database::open(dir.path(), create_db_config()).unwrap();
        // 999 was released at open; v's records were replayed, flushed and
        // retired by recovery.
        assert!(db.vector_index_wal.namespaces_holding_cut().is_empty());
        assert!(db.vector_index_wal.retired_cut() > first);
        assert_eq!(db.metrics_snapshot().vector_wal_records_skipped, 1, "999's record is not replayed");
        assert_eq!(db.metrics_snapshot().vector_wal_records_replayed, 2);
        // Another namespace's flush moves the cut to the tail; v's records must
        // be on disk by then, which a second crash shows.
        let u = db.registry.read().get_id("u").unwrap();
        log_and_apply(&db, u, b"k", b"v");
        db.get_store(u).unwrap().flush_memtable_to_level0().unwrap();
        assert_eq!(db.vector_index_wal.retired_cut(), db.vector_index_wal.tail());
        db.mark_crashed();
        drop(db);
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        assert_eq!(db.get_ns(v, b"k1").unwrap(), Some(b"x".to_vec()));
        assert_eq!(db.get_ns(v, b"k2").unwrap(), Some(b"y".to_vec()));
    }

    /// The timeline from the step 3 review: a chunk key whose delete reached
    /// L1 (tombstone dropped) while another vector namespace held the global
    /// cut below the key's record. Replaying from the global cut brings the
    /// key back, with nothing newer left to beat it.
    #[test]
    fn replay_does_not_bring_back_a_key_whose_delete_reached_l1() {
        let dir = TempDir::new().unwrap();
        let (sparse, k) = {
            let db = Database::open(dir.path(), create_db_config()).unwrap();
            let sparse = db.create_namespace("docs_sparse_vector").unwrap();
            let dense = db.create_namespace("docs_dense_vector").unwrap();
            let k: &[u8] = b"posting-7|doc-42";

            // 1-2. A chunk key in the sparse namespace, a dense code in the other.
            log_and_apply(&db, sparse, k, b"chunk code");
            let r_start = db.vector_index_wal.tail();
            log_and_apply(&db, dense, b"doc-42", b"dense code");
            // 3. Sparse flushes; dense still holds the global cut below R.
            db.get_store(sparse).unwrap().flush_memtable_to_level0().unwrap();
            assert_eq!(db.vector_index_wal.namespaces_holding_cut(), vec![dense]);
            assert!(db.vector_index_wal.retired_cut() < r_start, "the cut stays below R");
            // 4-6. The chunk moves away: K is deleted (main WAL), flushed and
            // merged into L1, which drops the tombstone.
            db.delete_ns(sparse, k).unwrap();
            let store = db.get_store(sparse).unwrap();
            store.flush_memtable_to_level0().unwrap();
            store.compact_lsm().unwrap();
            assert_eq!(db.get_ns(sparse, k).unwrap(), None);
            // The namespace keeps working: its next compaction pass also removes
            // the L0 files the merge made obsolete, tombstone included.
            log_and_apply(&db, sparse, b"posting-9|doc-42", b"chunk code");
            store.flush_memtable_to_level0().unwrap();
            store.compact_lsm().unwrap();
            let m = db.wal_metadata();
            assert_eq!(m.cut, m.tail, "main-WAL recovery has nothing to replay");
            // 7. Crash.
            db.mark_crashed();
            (sparse, k.to_vec())
        };

        // 8. Restart: open replays the vector index WAL. R lies below the sparse
        // namespace's own durable safe offset, so it is not replayed; the dense
        // namespace's record is.
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        assert_eq!(
            db.get_ns(sparse, &k).unwrap(),
            None,
            "a deleted chunk key must stay deleted after recovery"
        );
        let dense = db.registry.read().get_id("docs_dense_vector").unwrap();
        assert_eq!(db.get_ns(dense, b"doc-42").unwrap(), Some(b"dense code".to_vec()));
        assert_eq!(db.metrics_snapshot().vector_wal_records_replayed, 1);
    }

    fn reopen_db(dir: &TempDir) -> Database {
        Database::open(dir.path(), create_db_config()).unwrap()
    }

    #[test]
    fn records_are_replayed_after_a_crash_and_retired_by_recovery() {
        let dir = TempDir::new().unwrap();
        let ns = {
            let db = reopen_db(&dir);
            let ns = db.create_namespace("v_sparse_vector").unwrap();
            for i in 0..20u64 {
                log_and_apply(&db, ns, &key(i), &value(i));
            }
            db.mark_crashed(); // nothing flushed
            ns
        };
        let db = reopen_db(&dir);
        for i in 0..20u64 {
            assert_eq!(db.get_ns(ns, &key(i)).unwrap(), Some(value(i)), "key {i} replayed");
        }
        assert_eq!(db.metrics_snapshot().vector_wal_records_replayed, 20);
        // Recovery flushed what it replayed and retired it: nothing holds the cut.
        assert!(db.vector_index_wal.namespaces_holding_cut().is_empty());
        assert_eq!(db.vector_index_wal.retired_cut(), db.vector_index_wal.tail());
        db.mark_crashed();
        drop(db);

        let db = reopen_db(&dir);
        assert_eq!(db.metrics_snapshot().vector_wal_records_replayed, 0, "everything was retired");
        for i in 0..20u64 {
            assert_eq!(db.get_ns(ns, &key(i)).unwrap(), Some(value(i)), "key {i} survives a second crash");
        }
    }

    #[test]
    fn a_write_after_restart_beats_a_replayed_record() {
        let dir = TempDir::new().unwrap();
        let (ns, seq) = {
            let db = reopen_db(&dir);
            let ns = db.create_namespace("v").unwrap();
            db.put_ns(ns, b"main-wal", b"x").unwrap(); // the main WAL's last seq is below the record's
            log_and_apply(&db, ns, b"k", b"old");
            let seq = db.vector_index_wal.last_seq();
            db.mark_crashed();
            (ns, seq)
        };
        let db = reopen_db(&dir);
        assert!(db.next_seq.load(Ordering::SeqCst) > seq, "the counter is raised above the log");
        log_and_apply(&db, ns, b"k", b"new");
        assert_eq!(db.get_ns(ns, b"k").unwrap(), Some(b"new".to_vec()));
    }

    #[test]
    fn a_crash_loses_appends_that_were_never_synced() {
        let dir = TempDir::new().unwrap();
        let ns = {
            let db = reopen_db(&dir);
            let ns = db.create_namespace("v").unwrap();
            log_and_apply(&db, ns, b"synced", b"1"); // syncs
            let seq = db.next_seq.fetch_add(1, Ordering::SeqCst);
            let logged = db.vector_index_wal.append(ns, b"unsynced", b"2", seq).unwrap();
            db.get_store(ns).unwrap().replay_upsert(b"unsynced", b"2", seq).unwrap();
            drop(logged);
            db.mark_crashed();
            ns
        };
        let db = reopen_db(&dir);
        assert_eq!(db.get_ns(ns, b"synced").unwrap(), Some(b"1".to_vec()));
        assert_eq!(db.get_ns(ns, b"unsynced").unwrap(), None, "a power loss takes what was never synced");
    }

    #[test]
    fn a_logged_delete_replays_with_its_seq() {
        let dir = TempDir::new().unwrap();
        let ns = {
            let db = reopen_db(&dir);
            let ns = db.create_namespace("v").unwrap();
            db.put_ns(ns, b"gone", b"1").unwrap();
            db.put_ns(ns, b"back", b"1").unwrap();
            db.delete_ns_logged(ns, b"gone").unwrap();
            db.delete_ns_logged(ns, b"back").unwrap();
            db.put_ns(ns, b"back", b"2").unwrap(); // newer than its logged delete
            db.delete_ns_logged(ns, b"unsynced").unwrap();
            db.sync_vector_index_wal().unwrap();
            assert_eq!(db.get_ns(ns, b"gone").unwrap(), None);
            db.mark_crashed();
            ns
        };
        let db = reopen_db(&dir);
        assert_eq!(db.metrics_snapshot().vector_wal_records_replayed, 3);
        assert_eq!(db.get_ns(ns, b"gone").unwrap(), None, "the delete replays");
        assert_eq!(db.get_ns(ns, b"back").unwrap(), Some(b"2".to_vec()), "a newer put beats it");
    }

    /// The prefix rule: a flush syncs the log before it writes L0 files. Here
    /// a meta write logged to one namespace is followed by a key delete logged
    /// to another, and only the second namespace is flushed before the crash.
    /// Without the sync its L0 file would keep the delete while the log lost
    /// the meta write before it.
    #[test]
    fn a_flush_syncs_the_log_so_a_crash_keeps_a_prefix() {
        let dir = TempDir::new().unwrap();
        let (meta, sparse) = {
            let db = reopen_db(&dir);
            let meta = db.create_namespace("v_sparse_vector_meta").unwrap();
            let sparse = db.create_namespace("v_sparse_vector").unwrap();
            db.put_ns(sparse, b"7|doc", b"codes").unwrap();
            db.put_ns(meta, b"doc", b"lists 7").unwrap();
            db.put_ns_logged(meta, b"doc", b"lists nothing").unwrap();
            db.delete_ns_logged(sparse, b"7|doc").unwrap();
            let before = db.metrics_snapshot();
            db.get_store(sparse).unwrap().flush_memtable_to_level0().unwrap();
            let after = db.metrics_snapshot();
            assert_eq!(after.vector_wal_flush_syncs - before.vector_wal_flush_syncs, 1);
            assert_eq!(after.vector_wal_fsyncs - before.vector_wal_fsyncs, 1);
            // Nothing unsynced now: the next flush does not sync.
            db.put_ns(sparse, b"8|doc", b"codes").unwrap();
            db.get_store(sparse).unwrap().flush_memtable_to_level0().unwrap();
            assert_eq!(db.metrics_snapshot().vector_wal_flush_syncs, after.vector_wal_flush_syncs);
            db.mark_crashed();
            (meta, sparse)
        };
        let db = reopen_db(&dir);
        assert_eq!(db.get_ns(sparse, b"7|doc").unwrap(), None);
        assert_eq!(
            db.get_ns(meta, b"doc").unwrap(),
            Some(b"lists nothing".to_vec()),
            "the meta write logged before the delete survives with it"
        );
    }

    #[test]
    fn a_flush_of_a_namespace_that_never_logged_does_not_sync() {
        let dir = TempDir::new().unwrap();
        let db = reopen_db(&dir);
        let logged = db.create_namespace("v").unwrap();
        let plain = db.create_namespace("plain").unwrap();
        db.put_ns_logged(logged, b"k", b"v").unwrap(); // unsynced
        db.put_ns(plain, b"k", b"v").unwrap();
        db.get_store(plain).unwrap().flush_memtable_to_level0().unwrap();
        assert_eq!(db.metrics_snapshot().vector_wal_flush_syncs, 0);
        assert!(db.vector_index_wal.durable_tail() < db.vector_index_wal.tail());
    }

    #[test]
    fn a_crash_right_after_recovery_loses_nothing() {
        let dir = TempDir::new().unwrap();
        let ns = {
            let db = reopen_db(&dir);
            let ns = db.create_namespace("v").unwrap();
            for i in 0..5u64 {
                log_and_apply(&db, ns, &key(i), &value(i));
            }
            db.mark_crashed();
            ns
        };
        // Replays (recovery makes the records durable in L0), then crashes
        // before anything else runs.
        let db = reopen_db(&dir);
        assert_eq!(db.metrics_snapshot().vector_wal_records_replayed, 5);
        db.mark_crashed();
        drop(db);
        let db = reopen_db(&dir);
        assert_eq!(db.metrics_snapshot().vector_wal_records_replayed, 0, "retired by the first recovery");
        for i in 0..5u64 {
            assert_eq!(db.get_ns(ns, &key(i)).unwrap(), Some(value(i)));
        }
    }

    /// Two crashes in a row. Recovery replays the chunk key and, from the main
    /// WAL, its delete; main-WAL recovery then compacts into L1 (tombstone
    /// dropped) and marks the delete persisted, and a later compaction pass
    /// removes the obsolete L0 file. Had recovery not recorded what it
    /// replayed, the second crash would replay the key with nothing left to
    /// delete it.
    #[test]
    fn a_second_crash_does_not_bring_back_a_deleted_key() {
        let dir = TempDir::new().unwrap();
        let k: &[u8] = b"posting-7|doc-42";
        let sparse = {
            let db = reopen_db(&dir);
            let sparse = db.create_namespace("docs_sparse_vector").unwrap();
            log_and_apply(&db, sparse, k, b"chunk code"); // vector index WAL, synced
            db.delete_ns(sparse, k).unwrap(); // main WAL, later seq
            db.mark_crashed(); // neither in an SSTable
            sparse
        };
        {
            let db = reopen_db(&dir);
            assert_eq!(db.get_ns(sparse, k).unwrap(), None);
            // The compaction worker's pass: nothing to seal (the memtable is
            // empty), but it removes the L0 files recovery's merge made obsolete.
            db.get_store(sparse).unwrap().compact_lsm().unwrap();
            db.mark_crashed();
        }
        let db = reopen_db(&dir);
        assert_eq!(db.get_ns(sparse, k).unwrap(), None, "a deleted chunk key must stay deleted");
    }

    /// A namespace can hold the cut with an empty memtable (here: a logged
    /// record whose apply failed). The GC tick's flush must still release it,
    /// which takes a seal of the empty memtable.
    #[test]
    fn a_pin_flush_releases_a_holder_whose_memtable_is_empty() {
        let dir = TempDir::new().unwrap();
        let db = reopen_db(&dir);
        let ns = db.create_namespace("v").unwrap();
        let seq = db.next_seq.fetch_add(1, Ordering::SeqCst);
        drop(db.vector_index_wal.append(ns, b"k", b"v", seq).unwrap()); // never applied
        db.vector_index_wal.sync().unwrap();
        assert_eq!(db.vector_index_wal.namespaces_holding_cut(), vec![ns]);
        assert_eq!(db.flush_namespaces_pinning_vector_wal(), 1);
        assert!(db.vector_index_wal.namespaces_holding_cut().is_empty());
        assert_eq!(db.vector_index_wal.retired_cut(), db.vector_index_wal.tail());
    }

    /// A flush whose marker cannot be written registers no L0 file, keeps the
    /// data readable, and is completed by the next flush; the namespace's
    /// progress is not wedged.
    #[test]
    fn a_flush_whose_marker_fails_registers_nothing_and_is_retried() {
        let dir = TempDir::new().unwrap();
        let db = reopen_db(&dir);
        let ns = db.create_namespace("v").unwrap();
        log_and_apply(&db, ns, b"k1", b"x");
        let store = db.get_store(ns).unwrap();
        let l0_files = || -> usize {
            let m = store.lsm.build_manifest_snapshot().unwrap();
            m.levels
                .iter()
                .filter(|l| l.level == 0)
                .flat_map(|l| &l.buckets)
                .map(|b| b.files.len())
                .sum()
        };
        db.vector_index_wal.fail_next_marker_write();
        assert!(store.flush_memtable_to_level0().is_err());
        assert_eq!(l0_files(), 0, "nothing registered");
        assert_eq!(db.get_ns(ns, b"k1").unwrap(), Some(b"x".to_vec()));
        assert_eq!(db.vector_index_wal.namespaces_holding_cut(), vec![ns]);

        store.flush_memtable_to_level0().unwrap();
        assert!(l0_files() > 0);
        assert!(db.vector_index_wal.namespaces_holding_cut().is_empty());
        // Later flushes keep retiring.
        log_and_apply(&db, ns, b"k2", b"y");
        store.flush_memtable_to_level0().unwrap();
        assert!(db.vector_index_wal.namespaces_holding_cut().is_empty());
        assert_eq!(db.vector_index_wal.retired_cut(), db.vector_index_wal.tail());
    }

    /// A sealed memtable is not on disk until its L0 files are written; its
    /// records must still replay. Retiring at the seal would skip them.
    #[test]
    fn records_of_a_sealed_but_unwritten_memtable_replay_after_a_crash() {
        let dir = TempDir::new().unwrap();
        let config = || {
            let mut c = create_db_config();
            c.skip_list_capacity = 8; // seals every few writes; no worker writes them to L0
            c
        };
        let ns = {
            let db = Database::open(dir.path(), config()).unwrap();
            let ns = db.create_namespace("v").unwrap();
            for i in 0..30u64 {
                log_and_apply(&db, ns, &key(i), &value(i));
            }
            assert!(db.metrics_snapshot().memtable_flushes > 0, "the memtable sealed");
            db.mark_crashed();
            ns
        };
        let db = Database::open(dir.path(), config()).unwrap();
        for i in 0..30u64 {
            assert_eq!(db.get_ns(ns, &key(i)).unwrap(), Some(value(i)), "key {i}");
        }
    }

    /// A clean shutdown compacts every store, which drops tombstones in L1.
    /// The vector index WAL must have retired what those stores flushed by
    /// then, or the next open replays records whose keys were deleted.
    #[test]
    fn a_clean_shutdown_does_not_bring_back_deleted_keys() {
        let dir = TempDir::new().unwrap();
        let ns = {
            let db = Database::open(dir.path(), create_db_config()).unwrap();
            let ns = db.create_namespace("v_sparse_vector").unwrap();
            log_and_apply(&db, ns, b"moved-away", b"chunk code");
            log_and_apply(&db, ns, b"kept", b"chunk code");
            db.delete_ns(ns, b"moved-away").unwrap();
            db.shutdown().unwrap(); // flushes, then compacts into L1
            ns
        };
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        assert_eq!(
            db.get_ns(ns, b"moved-away").unwrap(),
            None,
            "a deleted key came back after a clean shutdown"
        );
        assert_eq!(db.get_ns(ns, b"kept").unwrap(), Some(b"chunk code".to_vec()));
        assert_eq!(
            db.metrics_snapshot().vector_wal_records_replayed,
            0,
            "a clean shutdown retires everything"
        );
    }

    // ── Preallocated segments (design note §18) ──────────────────────────

    /// The bytes of segment `segment_id` from `from` to `to`.
    fn bytes_of(dir: &Path, segment_id: u64, from: u64, to: u64) -> Vec<u8> {
        let f = File::open(segment_path(dir, segment_id)).unwrap();
        let mut b = vec![0u8; (to - from) as usize];
        f.read_exact_at(&mut b, from).unwrap();
        b
    }

    fn zero(dir: &Path, segment_id: u64, from: u64, to: u64) {
        let f = OpenOptions::new().write(true).open(segment_path(dir, segment_id)).unwrap();
        f.write_all_at(&vec![0u8; (to - from) as usize], from).unwrap();
    }

    /// The last segment's records end at unwritten space; a reopen finds the
    /// same tail and the file keeps its full size.
    #[test]
    fn a_preallocated_segment_reopens_at_the_same_tail() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), 4096);
        let appended = append_n(&wal, 5);
        wal.sync().unwrap();
        let tail = wal.tail();
        drop(wal);
        let (wal, report) = reopen(dir.path(), 4096).unwrap();
        assert_eq!((report.tail, report.torn_bytes_dropped), (tail, 0));
        assert_eq!(rows_from(&wal, 0), expected(&appended, 0..5));
        assert_eq!(wal.segments().unwrap(), vec![(0, 4096)]);
    }

    /// An earlier segment whose end marker is gone reads as unwritten space
    /// before its end: it was never sealed, which only the last segment may be.
    #[test]
    fn unwritten_space_in_an_earlier_segment_is_corruption() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), 256);
        append_n(&wal, 4); // segment 0 sealed with 3
        drop(wal);
        let marker = u64::from(SEGMENT_HEADER_LEN + 3 * FRAME);
        zero(dir.path(), 0, marker, marker + 4);
        let msg = corruption(reopen(dir.path(), 256));
        assert!(msg.contains("segment 0 is not sealed"), "{msg}");
    }

    /// A last segment whose header is all zeros was being created when the
    /// power went (its header is fsynced before any record goes in): it is
    /// created again. Other bad headers stay corruption.
    #[test]
    fn a_last_segment_with_a_blank_header_is_created_again() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), 256);
        let appended = append_n(&wal, 3);
        wal.append(0, b"k", b"v", 99).unwrap(); // starts segment 1
        drop(wal);
        zero(dir.path(), 1, 0, 256);
        let (wal, report) = reopen(dir.path(), 256).unwrap();
        assert_eq!(report.tail, position(1, SEGMENT_HEADER_LEN));
        assert_eq!(rows_from(&wal, 0), expected(&appended, 0..3));
        let f = File::open(segment_path(dir.path(), 1)).unwrap();
        check_header(&f, 1).unwrap();
        drop(wal);

        flip(dir.path(), 1, 2);
        let msg = corruption(reopen(dir.path(), 256));
        assert!(msg.contains("no vector index WAL header"), "{msg}");
    }

    /// A segment of the earlier format (end marker 0) fails the open: no
    /// conversion.
    #[test]
    fn a_version_1_segment_fails_the_open() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), 256);
        append_n(&wal, 2);
        drop(wal);
        let f = OpenOptions::new().write(true).open(segment_path(dir.path(), 0)).unwrap();
        f.write_all_at(&1u32.to_le_bytes(), 4).unwrap();
        let msg = corruption(reopen(dir.path(), 256));
        assert!(msg.contains("format version 1, expected 2"), "{msg}");
    }

    /// Past half of the active segment the next one is zero-filled ahead,
    /// with its header, under a `.prep` name; moving to it installs it with
    /// its blocks written, and the log reads back whole.
    #[test]
    fn the_next_segment_is_prepared_ahead_and_installed() {
        use std::os::unix::fs::MetadataExt;
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), 64 * 1024);
        let per_segment = u64::from((64 * 1024 - SEGMENT_HEADER_LEN - END_MARKER_LEN) / FRAME);
        let mut appended = append_n(&wal, per_segment / 2 + 1);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while !wal.log.prep.pending().is_some_and(|(id, done)| id == 1 && done) {
            assert!(std::time::Instant::now() < deadline, "segment 1 was never prepared");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let prep = prep_path(dir.path(), 1);
        assert!(prep.exists() && !segment_path(dir.path(), 1).exists());
        let before = appended.len() as u64;
        appended.extend((before..per_segment + 2).map(|i| wal.append((i % 3) as u32, &key(i), &value(i), i + 1).unwrap().appended()));
        assert!(!prep.exists(), "the prepared file was moved into place");
        let m = std::fs::metadata(segment_path(dir.path(), 1)).unwrap();
        assert_eq!(m.len(), 64 * 1024);
        assert!(m.blocks() * 512 >= 64 * 1024, "the installed segment's blocks are written");
        wal.sync().unwrap();
        drop(wal);
        let (wal, _) = reopen(dir.path(), 64 * 1024).unwrap();
        assert_eq!(rows_from(&wal, 0), expected(&appended, 0..per_segment + 2));
    }

    /// A segment a previous run left half-prepared is removed at open and
    /// never read as part of the log.
    #[test]
    fn a_leftover_prepared_segment_is_removed_at_open() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), 256);
        let appended = append_n(&wal, 2);
        drop(wal);
        let prep = prep_path(dir.path(), 1);
        std::fs::write(&prep, header_bytes(1)).unwrap();
        let (wal, report) = reopen(dir.path(), 256).unwrap();
        assert!(!prep.exists());
        assert_eq!(split(report.tail).0, 0);
        assert_eq!(rows_from(&wal, 0), expected(&appended, 0..2));
    }

    /// A power loss kept records past one it lost. Open zeroes everything
    /// past the tail, so a record later laid over the gap — ending exactly
    /// where an old record starts, as same-size records do — is followed by
    /// unwritten space, not by the old records.
    #[test]
    fn records_past_a_lost_one_never_rejoin_the_log() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), 64 * 1024);
        let appended = append_n(&wal, 40);
        wal.sync().unwrap();
        drop(wal);
        let (lost_seg, lost) = split(appended[10].start);
        zero(dir.path(), lost_seg, u64::from(lost), u64::from(split(appended[10].end).1));

        let (wal, report) = reopen(dir.path(), 64 * 1024).unwrap();
        assert_eq!(report.tail, appended[10].start);
        let past = bytes_of(dir.path(), 0, u64::from(lost), 64 * 1024);
        assert!(past.iter().all(|&b| b == 0), "open zeroes every byte past the tail");
        let new = wal.append(0, &key(9_000), &value(9_000), 9_001).unwrap().appended();
        assert_eq!(new.end, appended[11].start, "a same-size record ends where an old one started");
        wal.sync().unwrap();
        drop(wal);

        let (wal, _) = reopen(dir.path(), 64 * 1024).unwrap();
        let mut want = expected(&appended, 0..10);
        want.push((new.start, 0, 9_001, key(9_000), Some(value(9_000))));
        assert_eq!(rows_from(&wal, 0), want);
    }

    /// `lose_unsynced` and the power-loss image cut each segment at what an
    /// fsync covered, not at the durable tail: an append that no sync covered
    /// is gone, a synced one stays, and the file keeps its size.
    #[test]
    fn a_power_loss_keeps_exactly_what_an_fsync_covered() {
        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), 256);
        let appended = append_n(&wal, 4); // segment 0 sealed (fsynced) with 3
        wal.sync().unwrap();
        let lost = wal.append(1, &key(50), &value(50), 51).unwrap().appended();
        assert!(!wal.covered(lost.end));
        wal.lose_unsynced();
        drop(wal);
        let (wal, report) = reopen(dir.path(), 256).unwrap();
        assert_eq!(report.tail, lost.start);
        assert_eq!(rows_from(&wal, 0), expected(&appended, 0..4));
        assert!(wal.segments().unwrap().iter().all(|&(_, len)| len == 256));
    }

    /// Writers append same-size records and sync from several threads, on
    /// small segments; at a random moment the power fails, keeping or losing
    /// each 4 KiB page that no fsync covered at random. Every synced record
    /// survives and every record read is one that was appended there. Then
    /// more records are appended and synced over the gap, the log is reopened
    /// with nothing unsynced, and it must read as exactly the surviving
    /// records and then the new ones: nothing from before the power loss comes
    /// back after them.
    fn vector_wal_power_loss_round(seed: u64) {
        use std::sync::atomic::AtomicBool;
        const WRITERS: usize = 6;
        const SEGMENT: u32 = 16 * 1024;
        const PAGE: u64 = 4096;
        struct Rng(u64);
        impl Rng {
            fn below(&mut self, n: u64) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0 % n
            }
        }
        // Same-size records: 9-byte keys, 300-byte values.
        let rec_key = |s: u64| format!("k{s:08}").into_bytes();
        let rec_value = |s: u64| vec![(s % 251) as u8; 300];

        let dir = TempDir::new().unwrap();
        let (wal, _) = open_at(dir.path(), SEGMENT);
        let appended: Mutex<BTreeMap<u64, (u32, u64)>> = Mutex::new(BTreeMap::new());
        let synced: Mutex<Vec<u64>> = Mutex::new(Vec::new());
        let synced_at_loss: Mutex<Vec<u64>> = Mutex::new(Vec::new());
        let next_seq = AtomicU64::new(1);
        let stop = AtomicBool::new(false);
        let image = TempDir::new().unwrap();
        let mut rng = Rng(seed);

        std::thread::scope(|s| {
            for w in 0..WRITERS {
                let (wal, appended, synced, next_seq, stop) = (&wal, &appended, &synced, &next_seq, &stop);
                s.spawn(move || {
                    let mut rng = Rng(seed ^ (w as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    while !stop.load(Ordering::SeqCst) {
                        let mut mine = Vec::new();
                        for _ in 0..1 + rng.below(12) {
                            let seq = next_seq.fetch_add(1, Ordering::SeqCst);
                            let at = wal.append(w as u32, &rec_key(seq), &rec_value(seq), seq).unwrap();
                            appended.lock().insert(at.start, (w as u32, seq));
                            mine.push(at.start);
                        }
                        wal.sync().unwrap();
                        synced.lock().extend(mine);
                    }
                });
            }

            // If a check fails, release the writers so the scope can end.
            struct Release<'a>(&'a AtomicBool, &'a VectorIndexWal);
            impl Drop for Release<'_> {
                fn drop(&mut self) {
                    self.0.store(true, Ordering::SeqCst);
                    self.1.log.rounds.freeze(false);
                }
            }
            let _release = Release(&stop, &wal);

            std::thread::sleep(std::time::Duration::from_millis(2 + rng.below(30)));
            // Stop the world: every writer waits inside a sync, its records
            // written but not yet covered.
            wal.log.rounds.freeze(true);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            while wal.log.rounds.frozen_callers() < WRITERS {
                assert!(std::time::Instant::now() < deadline, "seed {seed}: the writers never stopped");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            // What was synced when the power failed: writers stopped inside a
            // sync finish it after the image is taken.
            *synced_at_loss.lock() = synced.lock().clone();
            for entry in std::fs::read_dir(dir.path()).unwrap() {
                let entry = entry.unwrap();
                std::fs::copy(entry.path(), image.path().join(entry.file_name())).unwrap();
            }
            for (path, durable) in wal.power_loss_extents() {
                let copy = image.path().join(path.file_name().unwrap());
                let f = OpenOptions::new().write(true).open(&copy).unwrap();
                let len = f.metadata().unwrap().len();
                let mut at = durable;
                while at < len {
                    let page_end = ((at / PAGE) + 1) * PAGE;
                    let end = page_end.min(len);
                    if rng.below(2) == 0 {
                        f.write_all_at(&vec![0u8; (end - at) as usize], at).unwrap();
                    }
                    at = end;
                }
            }
        });
        drop(wal);

        let appended = appended.into_inner();
        let (wal, _) = reopen(image.path(), SEGMENT).unwrap();
        let rows = rows_from(&wal, 0);
        for (at, ns, seq, key, value) in &rows {
            assert_eq!(
                appended.get(at),
                Some(&(*ns, *seq)),
                "seed {seed}: a record at {at:#x} that was never appended there"
            );
            assert_eq!(
                (key, value),
                (&rec_key(*seq), &Some(rec_value(*seq))),
                "seed {seed}: record {seq} damaged"
            );
        }
        let read: HashSet<u64> = rows.iter().map(|r| r.0).collect();
        for at in synced_at_loss.into_inner() {
            assert!(read.contains(&at), "seed {seed}: the synced record at {at:#x} was lost");
        }

        // Over the gap: more records, synced; then a reopen with nothing
        // unsynced reads the survivors, then exactly these.
        let first = appended.values().map(|v| v.1).max().unwrap_or(0) + 1;
        let mut want = rows;
        for seq in first..first + 1 + rng.below(200) {
            let at = wal.append(7, &rec_key(seq), &rec_value(seq), seq).unwrap().appended();
            want.push((at.start, 7, seq, rec_key(seq), Some(rec_value(seq))));
        }
        wal.sync().unwrap();
        drop(wal);
        let (wal, _) = reopen(image.path(), SEGMENT).unwrap();
        assert_eq!(rows_from(&wal, 0), want, "seed {seed}: the log after the power loss and new records");
    }

    #[test]
    fn every_synced_record_survives_random_power_losses() {
        for seed in 1..=12u64 {
            vector_wal_power_loss_round(0xF00D_0000 + seed);
        }
    }

    /// The long variant: `MINNAL_POWER_LOSS_SEEDS` rounds (default 300).
    #[test]
    #[ignore = "long; run explicitly with --ignored"]
    fn every_synced_record_survives_many_random_power_losses() {
        let rounds: u64 = std::env::var("MINNAL_POWER_LOSS_SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(300);
        for seed in 1..=rounds {
            vector_wal_power_loss_round(0xBEEF_0000 + seed);
        }
    }

    /// Measurement, not a check: one caller appending a batch of 8 records of
    /// 1 KiB, then syncing, on the real disk (the repository's `target/`).
    /// Past the first segment, so the log is on prepared segments.
    #[test]
    #[ignore = "measurement; run explicitly with --release --ignored --nocapture"]
    fn measure_batch_sync_latency() {
        const SEGMENT: u32 = 4 << 20;
        let dir = tempfile::tempdir_in(concat!(env!("CARGO_MANIFEST_DIR"), "/../target")).unwrap();
        let (wal, _) = open_at(dir.path(), SEGMENT);
        let value = vec![7u8; 1024];
        let mut seq = 0u64;
        let mut batch = |wal: &VectorIndexWal| {
            for _ in 0..8 {
                seq += 1;
                wal.append(1, format!("k{seq:09}").as_bytes(), &value, seq).unwrap();
            }
            wal.sync().unwrap();
        };
        while split(wal.tail()).0 < 2 {
            batch(&wal);
        }
        let n = 1500u32;
        let started = std::time::Instant::now();
        for _ in 0..n {
            batch(&wal);
        }
        println!("vector WAL batch of 8 x 1 KiB + sync: {:?} each", started.elapsed() / n);
    }
}
