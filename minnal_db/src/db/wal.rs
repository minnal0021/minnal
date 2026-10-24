//! The main write-ahead log: every WAL-backed `put`, `delete` and `merge`.
//!
//! The log itself is the shared segmented log (`db/segmented_log.rs`), with
//! magic `MWAL`: segments, checksummed frames, positions
//! `(segment_id << 32) | byte`, preallocation, group commit, the torn-tail and
//! corruption rules, and deleting retired segments all live there, the same as
//! for the vector index WAL. This module adds what is the main WAL's own:
//!
//! - the body, an rkyv [`WalEntry`] (operation, namespace, sequence, key,
//!   value), whose validation runs on every read;
//! - [`WalMetadata`], the saved retirement progress (the cut and each
//!   namespace's safe offset) that recovery and WAL GC read, kept in the log's
//!   directory as `metadata`;
//! - the scans recovery, field-index replay and the backstop harvest use.
//!
//! What is retired, and when, is decided by `WalPersistObserver` in
//! `database.rs`.

use crate::db::segmented_log::{self, LogError, LogFormat, LogPosition, OpenFrom, SegmentedLog};
use crc32fast::Hasher;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;
use std::time::Duration;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum WalError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    /// Damage open or a read cannot treat as a torn tail.
    #[error("WAL corrupted: {0}")]
    Corruption(String),
    /// An entry too large for one segment. Nothing was written.
    #[error("WAL entry of {len} bytes does not fit a segment (at most {max})")]
    RecordTooLarge {
        /// The entry's framed length.
        len: u64,
        /// The largest framed entry a segment holds.
        max: u64,
    },
    /// An fsync of the log failed, so the log refuses writes until the database
    /// is reopened (see [`Wal::commit`]).
    #[error("WAL stopped after a failed fsync: {0}")]
    Poisoned(String),
}

impl From<LogError> for WalError {
    fn from(e: LogError) -> Self {
        match e {
            LogError::Io(e) => Self::Io(e),
            LogError::Corruption(why) => Self::Corruption(why),
            LogError::RecordTooLarge { len, max } => Self::RecordTooLarge { len, max },
            LogError::Poisoned(why) => Self::Poisoned(why),
        }
    }
}

pub type Result<T> = std::result::Result<T, WalError>;

/// This log's segments: magic `MWAL`, version 1 (bump it whenever
/// [`WalEntry`]'s archived layout changes).
pub(crate) static FORMAT: LogFormat = LogFormat {
    name: "main WAL",
    magic: *b"MWAL",
    version: 1,
};

/// Default WAL segment size (64 MiB).
pub const DEFAULT_SEGMENT_SIZE: u32 = 64 * 1024 * 1024;

/// The metadata file, in the log's directory.
pub(crate) const METADATA_FILE: &str = "metadata";

const WAL_METADATA_MAGIC: [u8; 4] = *b"WALM";
const WAL_METADATA_VERSION: u32 = 2;

/// Operation type for WAL entries
#[derive(Debug, Clone, Copy, Archive, RkyvSerialize, RkyvDeserialize)]
pub enum WalOperationType {
    Upsert,
    Delete,
}

/// A single WAL entry containing the operation and its data.
///
/// The `namespace_id` identifies which KV store namespace this entry belongs to.
/// The default namespace always has ID `0`.
///
/// `sequence` is a global monotonic counter assigned at write time. Recovery
/// replays entries in sequence order so the last writer to a key wins, exactly
/// as it did live.
///
/// `op_name` is a caller-supplied label (e.g. `"document_write"`) that appears
/// in the fail log when an entry cannot be applied after recovery retry.
/// Empty string for unnamed writes.
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct WalEntry {
    pub operation: WalOperationType,
    pub namespace_id: u32,
    /// Global monotonic sequence number assigned at write time.
    pub sequence: u64,
    /// Human-readable label for this operation, used in fail-log output.
    pub op_name: String,
    pub key: Vec<u8>,
    pub value: Option<Vec<u8>>, // None for DELETE, Some for UPSERT
}

impl WalEntry {
    pub fn new_upsert(key: Vec<u8>, value: Vec<u8>) -> Self {
        Self::new_upsert_ns(0, key, value)
    }

    pub fn new_delete(key: Vec<u8>) -> Self {
        Self::new_delete_ns(0, key)
    }

    pub fn new_upsert_ns(namespace_id: u32, key: Vec<u8>, value: Vec<u8>) -> Self {
        Self {
            operation: WalOperationType::Upsert,
            namespace_id,
            sequence: 0,
            op_name: String::new(),
            key,
            value: Some(value),
        }
    }

    pub fn new_delete_ns(namespace_id: u32, key: Vec<u8>) -> Self {
        Self {
            operation: WalOperationType::Delete,
            namespace_id,
            sequence: 0,
            op_name: String::new(),
            key,
            value: None,
        }
    }

    /// Set the global sequence number on this entry.
    pub fn with_sequence(mut self, sequence: u64) -> Self {
        self.sequence = sequence;
        self
    }

    /// Set the operation name on this entry (used in fail-log output).
    pub fn with_op_name(mut self, name: impl Into<String>) -> Self {
        self.op_name = name.into();
        self
    }

    /// Serialize the entry to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.archive().map(|buf| buf.to_vec())
    }

    /// The serialized entry in rkyv's own buffer, for the append path, which
    /// copies it straight into the frame (one copy fewer than
    /// [`to_bytes`](Self::to_bytes), under the WAL lock).
    fn archive(&self) -> Result<impl std::ops::Deref<Target = [u8]>> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self).map_err(|e| WalError::Serialization(format!("Failed to serialize WAL entry: {}", e)))
    }

    /// Read only the namespace id and key, copying the key but never the value.
    ///
    /// Used when harvesting the keys a WAL segment holds before it is reclaimed
    /// (see [`Wal::scan_segment_keys`]): a segment can hold tens of thousands of
    /// documents, and full deserialisation would copy every one of their values
    /// into memory just to throw them away.
    pub fn peek_namespace_and_key(bytes: &[u8]) -> Result<(u32, Vec<u8>)> {
        let archived = rkyv::access::<ArchivedWalEntry, rkyv::rancor::Error>(bytes)
            .map_err(|e| WalError::Serialization(format!("WAL entry validation failed: {}", e)))?;
        Ok((archived.namespace_id.into(), archived.key.to_vec()))
    }

    /// Read only the sequence, copying nothing (`Wal::recover_sequence`).
    fn peek_sequence(bytes: &[u8]) -> Result<u64> {
        let archived = rkyv::access::<ArchivedWalEntry, rkyv::rancor::Error>(bytes)
            .map_err(|e| WalError::Serialization(format!("WAL entry validation failed: {}", e)))?;
        Ok(archived.sequence.to_native())
    }

    /// Deserialize entry from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let archived = rkyv::access::<ArchivedWalEntry, rkyv::rancor::Error>(bytes)
            .map_err(|e| WalError::Serialization(format!("WAL entry validation failed: {}", e)))?;

        let operation = match archived.operation {
            ArchivedWalOperationType::Upsert => WalOperationType::Upsert,
            ArchivedWalOperationType::Delete => WalOperationType::Delete,
        };

        Ok(WalEntry {
            operation,
            namespace_id: archived.namespace_id.into(),
            sequence: archived.sequence.to_native(),
            op_name: archived.op_name.to_string(),
            key: archived.key.to_vec(),
            value: archived.value.as_ref().map(|v| v.to_vec()),
        })
    }
}

/// Pointer to a WAL entry: where its frame starts and its body length.
#[derive(Debug, Clone, Copy, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct WalPointer {
    pub offset: u64,
    pub size: u32,
}

impl WalPointer {
    pub fn new(offset: u64, size: u32) -> Self {
        Self { offset, size }
    }

    /// Serialize the pointer to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .map_err(|e| WalError::Serialization(format!("Failed to serialize WAL pointer: {}", e)))
            .map(|buf| buf.to_vec())
    }

    /// Deserialize pointer from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let archived = rkyv::access::<ArchivedWalPointer, rkyv::rancor::Error>(bytes)
            .map_err(|e| WalError::Serialization(format!("WAL pointer validation failed: {}", e)))?;

        Ok(WalPointer {
            offset: archived.offset.to_native(),
            size: archived.size.to_native(),
        })
    }
}

/// Metadata for tracking WAL state.
///
/// Besides the log's bounds it holds the **retirement progress**: how far each
/// namespace's writes are in L0 files. Recovery replays an entry only if it
/// lies at or above its namespace's [`replay_from`](Self::replay_from), and WAL
/// GC deletes segments wholly below [`cut`](Self::cut). It is saved by the
/// flush that moves it, before that flush's L0 files are registered (see
/// `WalPersistObserver::prepare_written` in `database.rs`).
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, serde::Serialize)]
pub struct WalMetadata {
    /// The first segment still on disk starts here (WAL GC moves it).
    pub head: u64,
    /// Where new writes go, as of the save (capped at what was fsynced).
    pub tail: u64,
    /// Every entry below this is in an SSTable. The lowest safe offset among
    /// namespaces with unflushed writes, capped at the oldest in-flight write.
    pub cut: u64,
    /// Namespaces whose safe offset is above `cut`, sorted by namespace: each
    /// one's writes below its offset are in L0 files too.
    pub safe: Vec<(u32, u64)>,
    pub total_gc_runs: u64,         // Number of GC operations
    pub total_bytes_reclaimed: u64, // Total bytes freed by GC
    /// High-water mark for the global sequence number, updated on every save.
    /// Open reads the log only from the cut, so this is what covers the
    /// sequences of entries below it (their SSTables cover them too).
    pub last_sequence: u64,
}

impl Default for WalMetadata {
    fn default() -> Self {
        Self::new()
    }
}

impl WalMetadata {
    /// A log with nothing written and nothing retired: the head, the tail and
    /// the cut all at the first record's position.
    pub fn new() -> Self {
        let first = segmented_log::segment_start(0);
        Self {
            head: first,
            tail: first,
            cut: first,
            safe: Vec::new(),
            total_gc_runs: 0,
            total_bytes_reclaimed: 0,
            last_sequence: 0,
        }
    }

    /// Where replay starts for namespace `ns`: its saved safe offset, or the
    /// cut for a namespace not listed.
    pub fn replay_from(&self, ns: u32) -> u64 {
        match self.safe.binary_search_by_key(&ns, |&(n, _)| n) {
            Ok(i) => self.safe[i].1.max(self.cut),
            Err(_) => self.cut,
        }
    }

    /// Serialize metadata to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .map_err(|e| WalError::Serialization(format!("Failed to serialize WAL metadata: {}", e)))
            .map(|buf| buf.to_vec())
    }

    /// Deserialize metadata from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let min_size = std::mem::size_of::<ArchivedWalMetadata>();
        if bytes.len() < min_size {
            return Err(WalError::Serialization(format!("WAL metadata too small: {} < {}", bytes.len(), min_size)));
        }
        let archived = rkyv::access::<ArchivedWalMetadata, rkyv::rancor::Error>(bytes)
            .map_err(|e| WalError::Serialization(format!("WAL metadata validation failed: {}", e)))?;
        Ok(WalMetadata {
            head: archived.head.to_native(),
            tail: archived.tail.to_native(),
            cut: archived.cut.to_native(),
            safe: archived.safe.iter().map(|t| (t.0.to_native(), t.1.to_native())).collect(),
            total_gc_runs: archived.total_gc_runs.to_native(),
            total_bytes_reclaimed: archived.total_bytes_reclaimed.to_native(),
            last_sequence: archived.last_sequence.to_native(),
        })
    }

    pub fn to_file_bytes(&self) -> Result<Vec<u8>> {
        let payload = self.to_bytes()?;
        let mut hasher = Hasher::new();
        hasher.update(&payload);
        let checksum = hasher.finalize();

        let mut out = Vec::with_capacity(16 + payload.len());
        out.extend_from_slice(&WAL_METADATA_MAGIC);
        out.extend_from_slice(&WAL_METADATA_VERSION.to_le_bytes());
        out.extend_from_slice(&checksum.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&payload);
        Ok(out)
    }

    pub fn from_file_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() >= 16 && bytes.get(0..4) == Some(&WAL_METADATA_MAGIC) {
            let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
            if version != WAL_METADATA_VERSION {
                return Err(WalError::Serialization(format!("Unsupported WAL metadata version: {}", version)));
            }
            let checksum = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
            let payload_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
            let payload_end = 16usize.saturating_add(payload_len);
            if bytes.len() < payload_end {
                return Err(WalError::Serialization(format!(
                    "WAL metadata payload truncated: {} < {}",
                    bytes.len(),
                    payload_end
                )));
            }
            let payload = &bytes[16..payload_end];
            let mut hasher = Hasher::new();
            hasher.update(payload);
            let actual = hasher.finalize();
            if actual != checksum {
                return Err(WalError::Serialization("WAL metadata checksum mismatch".to_string()));
            }
            return Self::from_bytes(payload);
        }

        Self::from_bytes(bytes)
    }
}

/// The main WAL's state for operators, from [`Db::wal_stats`](crate::Db::wal_stats).
#[derive(Debug, Clone, serde::Serialize)]
pub struct WalStats {
    /// The first record still on disk; segments below it were reclaimed.
    pub head: LogPosition,
    /// Where the next record goes.
    pub tail: LogPosition,
    /// Everything below it is fsynced.
    pub durable_tail: LogPosition,
    /// Every entry below this is in an SSTable; WAL GC reclaims segments
    /// wholly below it.
    pub cut: LogPosition,
    /// Bytes of entries recovery would replay: from the cut to the tail.
    pub pending_bytes: u64,
    pub segment_size: u64,
    /// Namespaces whose writes are in L0 further than the cut: recovery skips
    /// their entries below `safe_offset`.
    pub namespaces_ahead_of_cut: Vec<WalNamespaceProgress>,
    /// Namespaces with writes not yet in an SSTable; the slowest of them
    /// holds the cut.
    pub namespaces_holding_cut: Vec<String>,
    pub total_gc_runs: u64,
    pub total_bytes_reclaimed: u64,
    /// Highest write sequence handed out as of the last metadata save.
    pub last_sequence: u64,
    /// Every segment file on disk, oldest first.
    pub segments: Vec<WalSegmentState>,
    /// Set after a failed fsync, with its error: the WAL refuses writes until
    /// the database is reopened. `None` while it works.
    pub stopped: Option<String>,
}

/// One namespace's saved progress, in [`WalStats`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct WalNamespaceProgress {
    pub namespace: String,
    pub safe_offset: LogPosition,
}

/// One WAL segment, in [`WalStats`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct WalSegmentState {
    pub segment_id: u64,
    /// Bytes of the segment file (segments are preallocated, so this is the
    /// segment size it was created at, whatever it holds).
    pub bytes: u64,
    /// Wholly below the cut: nothing in it is replayed, and WAL GC may
    /// delete it (unless a field index still needs it).
    pub retired: bool,
    /// Bytes of this segment recovery would replay.
    pub pending_bytes: u64,
}

/// What [`Wal::open`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WalOpenReport {
    /// Position of the next append.
    pub tail: u64,
    /// Highest sequence of any entry at or above the cut (0 if none).
    pub max_sequence: u64,
    /// See `segmented_log::OpenReport::torn_bytes_dropped`.
    pub torn_bytes_dropped: u64,
}

/// The main WAL: the shared segmented log, holding [`WalEntry`] bodies.
pub struct Wal {
    log: SegmentedLog,
}

/// A caller's error from a scan, carried through the segmented log's reader.
struct Scan(WalError);

impl From<LogError> for Scan {
    fn from(e: LogError) -> Self {
        Scan(e.into())
    }
}

impl Wal {
    /// Opens the log in `dir` from its saved progress (`SegmentedLog::open`:
    /// validation, the torn tail, a new segment when the progress names
    /// positions past the tail), reading each entry at or above the cut for
    /// the highest sequence. An entry that passes its CRC but does not decode
    /// is corruption.
    pub(crate) fn open(dir: &Path, segment_size: u32, from: OpenFrom) -> Result<(Self, WalOpenReport)> {
        let mut max_sequence = 0u64;
        let (log, report) = SegmentedLog::open(dir, &FORMAT, segment_size, from, |at, body| -> std::result::Result<(), Scan> {
            max_sequence = max_sequence.max(WalEntry::peek_sequence(body).map_err(|e| Scan(undecodable(at, e)))?);
            Ok(())
        })
        .map_err(|Scan(e)| e)?;
        Ok((
            Self { log },
            WalOpenReport {
                tail: report.tail,
                max_sequence,
                torn_bytes_dropped: report.torn_bytes_dropped,
            },
        ))
    }

    /// Open (or create) a standalone log in `dir` with no saved progress: it
    /// is read whole. For benchmarks and tools; a database opens its WAL from
    /// its metadata. Returns the log and its tail.
    pub fn open_standalone(dir: &Path, segment_size: u32) -> Result<(Self, u64)> {
        let (wal, report) = Self::open(dir, segment_size, OpenFrom { cut: None, furthest: 0 })?;
        Ok((wal, report.tail))
    }

    /// Append `entry` and move `tail` past it. Not durable until a
    /// [`commit`](Self::commit) covering it returns. The caller holds the WAL
    /// metadata write lock, which orders sequence allocation with appends.
    pub fn append_entry(&self, entry: &WalEntry, tail: &mut u64) -> Result<WalPointer> {
        let body = entry.archive()?;
        let (start, end) = self.log.append(&[&body], |_, _| {})?;
        *tail = end;
        Ok(WalPointer::new(start, body.len() as u32))
    }

    /// Wait until every byte below `end` is durable: **group commit**
    /// (`SegmentedLog::commit`). `delay` (C2): a leader with others already
    /// waiting first sleeps this long, so more writes join its round; a lone
    /// writer never waits. Returns `true` if this caller led a round (issued
    /// an fsync). A failed fsync poisons the log: every writer in that round,
    /// and every later one, gets [`WalError::Poisoned`].
    pub fn commit(&self, end: u64, delay: Duration) -> Result<bool> {
        Ok(self.log.commit(end, delay)?)
    }

    /// Make everything appended so far durable (shutdown, test helpers).
    pub fn sync(&self) -> Result<()> {
        self.log.sync()?;
        Ok(())
    }

    /// Every byte below this position is fsynced.
    pub fn durable_tail(&self) -> u64 {
        self.log.durable()
    }

    /// Position of the next append.
    #[cfg(test)]
    pub(crate) fn tail(&self) -> u64 {
        self.log.tail()
    }

    /// Why the log is stopped, if it is (a failed fsync).
    pub(crate) fn poisoned(&self) -> Option<String> {
        self.log.poisoned()
    }

    /// `Err(Poisoned)` once the log has been stopped.
    pub fn check_poisoned(&self) -> Result<()> {
        Ok(self.log.check_poisoned()?)
    }

    /// Stop the log: every waiting and later commit fails until reopen.
    pub(crate) fn poison(&self, why: String) -> WalError {
        self.log.poison(why).into()
    }

    /// The segment a position lies in.
    pub fn segment_of(&self, position: u64) -> u64 {
        segmented_log::segment_of(position)
    }

    /// The size new segments are created at.
    pub fn segment_size(&self) -> u64 {
        u64::from(self.log.segment_size())
    }

    /// The segment appends go to; never deleted.
    pub(crate) fn active_segment(&self) -> u64 {
        self.log.active_segment()
    }

    /// Bytes of entries in `[from, to)` (frames only).
    pub(crate) fn bytes_between(&self, from: u64, to: u64) -> u64 {
        self.log.bytes_between(from, to)
    }

    /// Every segment on disk, `(segment_id, bytes)`, oldest first.
    pub(crate) fn segments(&self) -> Result<Vec<(u64, u64)>> {
        Ok(self.log.segments()?)
    }

    /// Segment ids covering `[start, end)` whose files are **absent**.
    ///
    /// A scan below the oldest segment skips the reclaimed ones (they are
    /// retired, below the cut), which is right for recovery but means any
    /// *other* consumer replaying that range — a field index replaying from its
    /// checkpoint — is handed an incomplete result. This reports the holes so
    /// such a consumer can say so.
    pub fn missing_segments(&self, start: u64, end: u64) -> Vec<u64> {
        self.log.missing_segments(start, end)
    }

    /// Delete every segment below `segment_id` (never the active one), oldest
    /// first. Returns `(segments, bytes)` deleted.
    pub(crate) fn delete_segments_below(&self, segment_id: u64) -> Result<(u64, u64)> {
        Ok(self.log.delete_segments_below(segment_id)?)
    }

    /// Walk the entries in `[start, end)`, handing each one's position and
    /// body to `extract`. The one loop every scan uses.
    fn walk<T>(&self, start: u64, end: u64, mut extract: impl FnMut(u64, &[u8]) -> Result<T>) -> Result<Vec<T>> {
        let mut out = Vec::new();
        self.log
            .for_each(start, end, |at, body| -> std::result::Result<(), Scan> {
                out.push(extract(at, body).map_err(Scan)?);
                Ok(())
            })
            .map_err(|Scan(e)| e)?;
        Ok(out)
    }

    /// Every entry in `[start, end)`, in log order.
    pub fn scan_entries(&self, start: u64, end: u64) -> Result<Vec<(WalPointer, WalEntry)>> {
        self.walk(start, end, |at, body| {
            let entry = WalEntry::from_bytes(body).map_err(|e| undecodable(at, e))?;
            Ok((WalPointer::new(at, body.len() as u32), entry))
        })
    }

    /// `(namespace_id, key)` of every entry in `[start, end)`, **without
    /// materialising any values**, in log order and with duplicates: a key
    /// written five times appears five times. Callers wanting a work list want
    /// [`scan_segment_keys`](Self::scan_segment_keys), which deduplicates.
    pub fn scan_keys_in_entries(&self, start: u64, end: u64) -> Result<Vec<(u32, Vec<u8>)>> {
        self.walk(start, end, |at, body| {
            WalEntry::peek_namespace_and_key(body).map_err(|e| undecodable(at, e))
        })
    }

    /// The distinct `(namespace_id, key)` pairs one segment holds, **without
    /// materialising any values**, in first-seen order.
    ///
    /// This is the harvest that makes row-scoped index repair possible. When the
    /// backstop forces WAL GC to reclaim a segment an active field index still
    /// needs (see `ThresholdConfig::max_pinned_wal_segments`), the keys of the
    /// affected documents exist *only* in that segment — a gap record holding a
    /// WAL range and segment ids cannot name them, and once the file is unlinked
    /// they are unrecoverable. So they are harvested here, immediately before the
    /// unlink, and recorded as a repair worklist. Both upserts and deletes are
    /// included — a lost *delete* leaves a stale row in the index, which is just
    /// as wrong as a missing one.
    ///
    /// A segment already reclaimed yields an empty result rather than an error:
    /// failing here would abort the whole GC pass.
    pub fn scan_segment_keys(&self, segment_id: u64) -> Result<Vec<(u32, Vec<u8>)>> {
        if !self
            .missing_segments(segmented_log::segment_start(segment_id), segmented_log::segment_start(segment_id) + 1)
            .is_empty()
        {
            return Ok(Vec::new());
        }
        let end = self
            .log
            .segment_end(segment_id)
            .unwrap_or_else(|| segmented_log::segment_start(segment_id + 1));
        let pairs = self.scan_keys_in_entries(segmented_log::segment_start(segment_id), end)?;
        let mut seen: std::collections::HashSet<(u32, Vec<u8>)> = std::collections::HashSet::with_capacity(pairs.len());
        let mut keys = Vec::with_capacity(pairs.len());
        for pair in pairs {
            if seen.insert(pair.clone()) {
                keys.push(pair);
            }
        }
        keys.shrink_to_fit();
        Ok(keys)
    }

    /// Test hook: while `on`, every writer reaching [`commit`](Self::commit)
    /// waits there, its entry appended but not yet fsynced — the state a power
    /// loss would catch it in. Turning it off releases them.
    #[cfg(test)]
    pub(crate) fn freeze_commits(&self, on: bool) {
        self.log.rounds.freeze(on);
    }

    /// Test hook: how many writers are waiting in a frozen commit.
    #[cfg(test)]
    pub(crate) fn frozen_writers(&self) -> usize {
        self.log.rounds.frozen_callers()
    }

    /// Test hook for the power-loss model (`SegmentedLog::power_loss_extents`).
    #[cfg(test)]
    pub(crate) fn power_loss_extents(&self) -> Vec<(PathBuf, u64)> {
        self.log.power_loss_extents()
    }

    /// Test hook: make the next fsync of the log fail.
    #[cfg(test)]
    pub(crate) fn fail_next_sync(&self) {
        self.log.fail_next_sync();
    }
}

/// An entry at `at` that passed its CRC but does not decode: the CRC says
/// these are the bytes written, so this is corruption (or a layout change
/// without a version bump), never a torn tail.
fn undecodable(at: u64, e: WalError) -> WalError {
    WalError::Corruption(format!("the entry at {at:#x} passes its CRC but does not decode: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::segmented_log::{SEGMENT_HEADER_LEN, segment_start};
    use tempfile::TempDir;

    const FRESH: OpenFrom = OpenFrom { cut: None, furthest: 0 };

    fn open(dir: &Path, segment_size: u32) -> Result<(Wal, WalOpenReport)> {
        Wal::open(dir, segment_size, FRESH)
    }

    fn append(wal: &Wal, entries: impl IntoIterator<Item = WalEntry>) -> Result<(Vec<WalPointer>, u64)> {
        let mut tail = wal.tail();
        let mut pointers = Vec::new();
        for e in entries {
            pointers.push(wal.append_entry(&e, &mut tail)?);
        }
        wal.commit(tail, Duration::ZERO)?;
        Ok((pointers, tail))
    }

    fn upserts(n: u32) -> impl Iterator<Item = WalEntry> {
        (0..n).map(|i| WalEntry::new_upsert_ns(i % 3, format!("k{i}").into_bytes(), vec![b'x'; 40]).with_sequence(u64::from(i) + 1))
    }

    #[test]
    fn test_wal_entry_serialization() -> Result<()> {
        let entry = WalEntry::new_upsert(b"key1".to_vec(), b"value1".to_vec());

        let bytes = entry.to_bytes()?;
        let restored = WalEntry::from_bytes(&bytes)?;

        assert_eq!(entry.key, restored.key);
        assert_eq!(entry.value, restored.value);
        Ok(())
    }

    #[test]
    fn test_wal_entry_delete_serialization() -> Result<()> {
        let entry = WalEntry::new_delete(b"key1".to_vec());

        let bytes = entry.to_bytes()?;
        let restored = WalEntry::from_bytes(&bytes)?;

        assert_eq!(entry.key, restored.key);
        assert!(restored.value.is_none());
        Ok(())
    }

    #[test]
    fn test_wal_metadata_file_encoding() -> Result<()> {
        let metadata = WalMetadata {
            head: 16,
            tail: 1024,
            cut: 300,
            safe: vec![(2, 500), (7, 900)],
            total_gc_runs: 2,
            total_bytes_reclaimed: 256,
            last_sequence: 41,
        };

        let file_bytes = metadata.to_file_bytes()?;
        let restored = WalMetadata::from_file_bytes(&file_bytes)?;

        assert_eq!(metadata.head, restored.head);
        assert_eq!(metadata.tail, restored.tail);
        assert_eq!(metadata.cut, restored.cut);
        assert_eq!(metadata.safe, restored.safe);
        assert_eq!(metadata.last_sequence, restored.last_sequence);

        let mut damaged = file_bytes.clone();
        *damaged.last_mut().unwrap() ^= 1;
        assert!(WalMetadata::from_file_bytes(&damaged).is_err(), "a checksum mismatch must be rejected");
        Ok(())
    }

    #[test]
    fn test_with_sequence_sets_field() {
        let entry = WalEntry::new_upsert(b"k".to_vec(), b"v".to_vec()).with_sequence(42);
        assert_eq!(entry.sequence, 42);
    }

    #[test]
    fn test_sequence_survives_roundtrip() -> Result<()> {
        let entry = WalEntry::new_upsert(b"k".to_vec(), b"v".to_vec()).with_sequence(999);
        let restored = WalEntry::from_bytes(&entry.to_bytes()?)?;
        assert_eq!(restored.sequence, 999);
        assert_eq!(WalEntry::peek_sequence(&entry.to_bytes()?)?, 999);
        Ok(())
    }

    #[test]
    fn test_default_constructors_have_zero_sequence() {
        assert_eq!(WalEntry::new_upsert(b"k".to_vec(), b"v".to_vec()).sequence, 0);
        assert_eq!(WalEntry::new_delete(b"k".to_vec()).sequence, 0);
        assert_eq!(WalEntry::new_upsert_ns(1, b"k".to_vec(), b"v".to_vec()).sequence, 0);
        assert_eq!(WalEntry::new_delete_ns(1, b"k".to_vec()).sequence, 0);
    }

    /// Entries written across segments read back in order and in full, and a
    /// reopen finds the tail and the highest sequence above the cut.
    #[test]
    fn entries_read_back_across_segments_and_after_a_reopen() -> Result<()> {
        let dir = TempDir::new()?;
        let (wal, report) = open(dir.path(), 512)?;
        assert_eq!(report.tail, segment_start(0));
        let (pointers, tail) = append(&wal, upserts(60))?;
        assert!(wal.segment_of(tail) >= 3, "setup: the entries should span segments");
        let scanned = wal.scan_entries(segment_start(0), tail)?;
        assert_eq!(scanned.len(), 60);
        for (i, ((pointer, entry), expected)) in scanned.iter().zip(&pointers).enumerate() {
            assert_eq!(pointer.offset, expected.offset);
            assert_eq!(entry.key, format!("k{i}").into_bytes());
            assert_eq!(entry.namespace_id, i as u32 % 3);
        }
        drop(wal);
        let cut = pointers[40].offset;
        let (wal, report) = Wal::open(
            dir.path(),
            512,
            OpenFrom {
                cut: Some(cut),
                furthest: cut,
            },
        )?;
        assert_eq!(report.tail, tail);
        assert_eq!(report.max_sequence, 60);
        assert_eq!(wal.scan_entries(cut, tail)?.len(), 20);
        Ok(())
    }

    /// A changed byte inside a value still decodes (rkyv checks structure, not
    /// contents), so without a checksum replay would write the wrong value.
    /// The frame's CRC catches it: in the last segment it is the torn tail,
    /// earlier it fails the open.
    #[test]
    fn a_changed_byte_in_a_value_is_caught_by_the_crc() -> Result<()> {
        use std::os::unix::fs::FileExt;
        let dir = TempDir::new()?;
        let (wal, _) = open(dir.path(), 512)?;
        let (pointers, _) = append(&wal, upserts(3))?;
        drop(wal);
        let seg0 = segmented_log::segment_path(dir.path(), 0);
        let f = std::fs::OpenOptions::new().read(true).write(true).open(&seg0)?;
        let second = pointers[1];
        let at = u64::from(segmented_log::split(second.offset).1 + segmented_log::FRAME_HEADER_LEN);
        let mut body = vec![0u8; second.size as usize];
        f.read_exact_at(&mut body, at)?;
        let i = body.windows(2).position(|w| w == b"xx").expect("value bytes in the body");
        body[i] = b'X';
        assert!(WalEntry::from_bytes(&body).is_ok(), "setup: the changed body must still decode");
        f.write_all_at(&body, at)?;
        let (wal, report) = open(dir.path(), 512)?;
        assert_eq!(report.tail, second.offset, "the tail is cut at the damaged entry");
        assert_eq!(wal.scan_entries(segment_start(0), wal.tail())?.len(), 1);
        Ok(())
    }

    /// A body that passes its CRC but does not decode as an entry is
    /// corruption, wherever it is.
    #[test]
    fn an_undecodable_entry_is_corruption() -> Result<()> {
        let dir = TempDir::new()?;
        let (wal, _) = open(dir.path(), 512)?;
        let mut tail = wal.tail();
        wal.append_entry(&WalEntry::new_upsert(b"k".to_vec(), b"v".to_vec()), &mut tail)?;
        // Not an entry, but a well-formed frame.
        let (_, end) = wal.log.append(&[b"not an rkyv entry"], |_, _| {})?;
        wal.commit(end, Duration::ZERO)?;
        assert!(matches!(wal.scan_entries(segment_start(0), end), Err(WalError::Corruption(_))));
        drop(wal);
        assert!(matches!(open(dir.path(), 512), Err(WalError::Corruption(_))));
        Ok(())
    }

    /// The key-only scan and the full scan agree, entry for entry, and both
    /// skip segments GC reclaimed below the oldest one.
    #[test]
    fn key_only_scan_agrees_with_the_full_scan() -> Result<()> {
        let dir = TempDir::new()?;
        let (wal, _) = open(dir.path(), 512)?;
        let (_, tail) = append(&wal, upserts(60))?;
        let expect =
            |wal: &Wal| -> Result<Vec<(u32, Vec<u8>)>> { Ok(wal.scan_entries(0, tail)?.into_iter().map(|(_, e)| (e.namespace_id, e.key)).collect()) };
        assert_eq!(wal.scan_keys_in_entries(0, tail)?, expect(&wal)?);
        wal.delete_segments_below(2)?;
        let keys = wal.scan_keys_in_entries(0, tail)?;
        assert_eq!(keys, expect(&wal)?, "both scans must skip reclaimed segments identically");
        assert!(keys.len() < 60, "the reclaimed segments' entries are gone");
        assert_eq!(wal.missing_segments(0, tail), vec![0, 1]);
        Ok(())
    }

    /// `scan_segment_keys` covers exactly one segment and deduplicates; a
    /// reclaimed segment yields nothing rather than an error.
    #[test]
    fn segment_key_scan_is_scoped_and_deduplicated() -> Result<()> {
        let dir = TempDir::new()?;
        let (wal, _) = open(dir.path(), 512)?;
        let hot = (0..8).map(|_| WalEntry::new_upsert(b"hot".to_vec(), vec![b'x'; 40]));
        let (_, _) = append(&wal, hot.chain([WalEntry::new_upsert(b"cold".to_vec(), vec![b'x'; 40])]))?;
        let (_, tail) = append(&wal, upserts(20))?;
        assert!(wal.segment_of(tail) >= 1);
        let seg0 = wal.scan_segment_keys(0)?;
        assert_eq!(
            seg0.iter().filter(|(_, k)| k == b"hot").count(),
            1,
            "a key written eight times appears once"
        );
        let in_segment_0 = wal.scan_keys_in_entries(segment_start(0), segment_start(1))?;
        for pair in &seg0 {
            assert!(in_segment_0.contains(pair), "{pair:?} came from outside segment 0");
        }
        wal.delete_segments_below(1)?;
        assert!(wal.scan_segment_keys(0)?.is_empty());
        Ok(())
    }

    /// The segment header names the log: a vector index WAL segment in the
    /// main WAL's directory fails the open instead of being parsed.
    #[test]
    fn a_segment_from_another_log_fails_the_open() -> Result<()> {
        let dir = TempDir::new()?;
        let (wal, _) = open(dir.path(), 512)?;
        append(&wal, upserts(3))?;
        drop(wal);
        let seg0 = segmented_log::segment_path(dir.path(), 0);
        let mut bytes = std::fs::read(&seg0)?;
        bytes[..SEGMENT_HEADER_LEN as usize].copy_from_slice(&segmented_log::header_bytes(&crate::db::vector_index_wal::FORMAT, 0));
        std::fs::write(&seg0, bytes)?;
        match open(dir.path(), 512) {
            Err(WalError::Corruption(why)) => assert!(why.contains("main WAL"), "{why}"),
            other => panic!("expected corruption, got {:?}", other.map(|_| ())),
        }
        Ok(())
    }

    /// An entry too large for a segment is refused before anything is written.
    #[test]
    fn an_entry_too_large_for_a_segment_is_refused() -> Result<()> {
        let dir = TempDir::new()?;
        let (wal, _) = open(dir.path(), 512)?;
        let mut tail = wal.tail();
        let big = WalEntry::new_upsert(b"k".to_vec(), vec![0u8; 1024]);
        assert!(matches!(wal.append_entry(&big, &mut tail), Err(WalError::RecordTooLarge { .. })));
        assert_eq!(tail, segment_start(0));
        Ok(())
    }
}
