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
//! `semantic_search/docs/Vector-Index-WAL-Design.md`.
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
//! segmented log's (`db/wal/segmented_log.rs`), with magic `MVIW`. A record's body
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

use crate::db::wal::segmented_log::{self, LogError, LogFormat, Progress, ProgressFile, SegmentedLog, segment_of};
use crate::support::metrics::Metrics;
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

/// The progress marker file, in the log's directory: the shared progress file
/// format (`segmented_log::ProgressFile`) with magic `MVIC` and no extra bytes.
const RETIRED_CUT_FILE: &str = "retired_cut";
const MARKER: ProgressFile = ProgressFile { magic: *b"MVIC" };

/// Errors from the vector index WAL: the shared log's errors, naming this log.
/// Its own damage (a body or the progress marker that does not decode) is
/// `LogError::Corruption` too.
#[derive(Error, Debug)]
pub enum VectorIndexWalError {
    /// From the segmented log or this log's own files.
    #[error("vector index WAL: {0}")]
    Log(#[from] LogError),
}

impl From<std::io::Error> for VectorIndexWalError {
    fn from(e: std::io::Error) -> Self {
        Self::Log(LogError::Io(e))
    }
}

impl VectorIndexWalError {
    /// Damage to the log or its marker.
    fn corruption(why: String) -> Self {
        Self::Log(LogError::Corruption(why))
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
    let bad = || VectorIndexWalError::corruption(format!("record at {at:#x} passes its CRC but does not decode"));
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
    /// The segments, frames and syncs (`db/wal/segmented_log.rs`).
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
    Ok(MARKER.read(&dir.join(RETIRED_CUT_FILE))?.map(|(p, _)| p))
}

#[cfg(test)]
fn progress_bytes(p: &Progress) -> Vec<u8> {
    MARKER.encode(p, &[])
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
        // Checked on read: a safe offset below the cut is corruption.
        let marker = if dir.exists() { read_progress(dir)? } else { None };
        let cut = marker.as_ref().map_or(0, |p| p.cut);

        let mut last_seq = 0u64;
        // Each namespace's last record end, to rebuild who holds the cut.
        let mut last_write: HashMap<u32, u64> = HashMap::new();
        let (log, report) = SegmentedLog::open(
            dir,
            &FORMAT,
            segment_size,
            Progress::open_from(marker.as_ref()),
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
    /// A commit round (`db/wal/commit_rounds.rs`): the fsync runs outside the
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
        let lift = |e: LogError| E::from(VectorIndexWalError::from(e));
        self.log.for_each_with(start, self.log.tail(), lift, |at, body| {
            let record = decode_body(body, at).map_err(E::from)?;
            f(at, record)
        })
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
        MARKER.write(&self.dir().join(RETIRED_CUT_FILE), &next, &[]).map_err(|e| {
            log::error!("[vector index WAL] could not write the progress marker: {e}");
            VectorIndexWalError::from(e)
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

#[cfg(test)]
mod tests;
