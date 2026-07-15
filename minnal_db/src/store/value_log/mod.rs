//! Value Log — immutable segment files.
//!
//! Each bucket's values live in a series of **segment files**
//! (`value_log_{bucket}.seg000123`). A segment is appended to while it is the
//! bucket's *active tail*; once it fills it is **sealed and never modified
//! again**. Segment ids are **monotone and never reused**.
//!
//! That single property is what the rest of the engine leans on:
//!
//! - A value's address is `(segment_id, rec_offset, value_len)`, and because ids
//!   are never reused, an address means **one record forever**. A pointer the LSM
//!   has since re-pointed still resolves to *that key's own bytes* (GC preserves a
//!   key's sequence when it relocates the record), and a pointer into a segment GC
//!   has already deleted fails loudly with [`ValueLogError::SegmentMissing`] —
//!   never silently to a *different* key's record. Readers therefore need no
//!   seqlock and no stale-slot detection: on `SegmentMissing` they re-resolve
//!   through the LSM and retry.
//! - GC's unit is a segment, not the whole bucket. It rewrites one sealed
//!   segment's survivors and unlinks the file; it never copies untouched data and
//!   never rewrites a file in place.
//!
//! # Layout
//!
//! ```text
//! segment file:  [file header 16B] [record] [record] ...
//!
//! record:        [header 36B][value bytes][key bytes]
//!                 ^ rec_offset
//! ```
//!
//! The value sits **before** the key deliberately: a reader knows `value_len` from
//! the pointer, so a `get` is **one `pread` of exactly `36 + value_len` bytes**.
//! Only GC needs the key, and it reads whole records sequentially anyway.
//!
//! # Liveness and garbage
//!
//! Records carry **no tombstone/updated flags** — the LSM is the sole authority on
//! what is live. When a write displaces an older record the writer just adds that
//! record's size to its segment's garbage counter: it already holds the old
//! pointer, and the pointer carries `value_len`, so the accounting costs **zero
//! I/O** (the old design did a read-modify-write of a record header at a random,
//! possibly-cold offset on *every* overwrite and delete). The counters are a *hint*
//! for GC selection; the exact live set always comes from the LSM.

pub mod sharded;

use crc32fast::Hasher;
use parking_lot::{Mutex, RwLock};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ValueLogError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Value log corrupted")]
    CorruptedLog,
    #[error("Invalid value location")]
    InvalidLocation,
    /// The pointer names a segment whose file is gone — GC reclaimed it after
    /// relocating its records. Re-resolve the key through the LSM (which holds the
    /// new pointer) and retry. An expected, benign race; not corruption.
    #[error("value-log segment {0} no longer exists (reclaimed by GC); re-resolve the pointer")]
    SegmentMissing(u32),
    #[error(
        "invalid segment size {0} bytes: must be a multiple of {SEGMENT_SIZE_ALIGNMENT}, at least {MIN_SEGMENT_SIZE_BYTES}, and at most {MAX_SEGMENT_SIZE_BYTES}"
    )]
    InvalidSegmentSize(u64),
    /// A single record cannot exceed one segment.
    #[error("value too large: a record of {record_len} bytes does not fit in a {segment_size}-byte segment")]
    ValueTooLarge { record_len: u64, segment_size: u64 },
}

pub type Result<T> = std::result::Result<T, ValueLogError>;

/// Default size at which a segment is sealed and a new one opened.
pub const DEFAULT_SEGMENT_SIZE_BYTES: u64 = 256 * 1024 * 1024;

const SEGMENT_SIZE_ALIGNMENT: u64 = 4096;
const MIN_SEGMENT_SIZE_BYTES: u64 = 64 * 1024;
/// `rec_offset` in the value pointer is a `u32`, so every record must be
/// addressable within 4 GiB of its segment.
const MAX_SEGMENT_SIZE_BYTES: u64 = u32::MAX as u64;

/// Validate a configured segment size.
///
/// Unlike the page size it replaces, this is **not** fixed at creation: a
/// segment's size is not encoded in any pointer (only its id and a byte offset
/// are), so existing segments keep the size they were written at and new ones use
/// whatever is configured now.
pub fn validate_segment_size(segment_size: u64) -> Result<()> {
    if !(MIN_SEGMENT_SIZE_BYTES..=MAX_SEGMENT_SIZE_BYTES).contains(&segment_size) || !segment_size.is_multiple_of(SEGMENT_SIZE_ALIGNMENT) {
        return Err(ValueLogError::InvalidSegmentSize(segment_size));
    }
    Ok(())
}

// ── On-disk formats ────────────────────────────────────────────────────────

const SEGMENT_MAGIC: [u8; 4] = *b"VSG1";
const SEGMENT_FORMAT_VERSION: u32 = 1;
/// magic(4) + version(4) + segment_id(4) + reserved(4)
const SEGMENT_HEADER_SIZE: u64 = 16;

const VALUE_LOG_METADATA_MAGIC: [u8; 4] = *b"VLOG";
const VALUE_LOG_METADATA_VERSION: u32 = 3;

/// Garbage share at which GC will **seal the active tail** so it can be collected.
///
/// The tail is normally off-limits (it is still being appended to), which is fine while
/// it keeps filling: it seals on its own and becomes collectable. But a small or idle
/// namespace may never fill a segment, so garbage can pile up in the tail and be
/// unreclaimable *forever* — while the bucket-level waste trigger keeps firing, so GC
/// wakes every interval, finds no candidate, and reclaims nothing.
///
/// Sealing the tail early costs a rewrite of whatever is still live in it, so it is only
/// worth doing when garbage **dominates**. At 50% the pass rewrites at most as much as it
/// frees; the common case that motivates this — a store whose keys were all deleted or
/// overwritten — is at or near 100%, where there is nothing to rewrite at all.
pub const TAIL_GC_MIN_GARBAGE_PCT: f64 = 50.0;

/// How many segment files a bucket keeps open at once.
///
/// Segments are immutable, so an evicted handle can always be reopened. Eviction
/// only drops the cache's `Arc<File>`, so an in-flight reader's clone keeps the fd
/// alive — and an already-open fd keeps reading correctly even after the file is
/// unlinked.
const MAX_OPEN_SEGMENTS_PER_BUCKET: usize = 32;

/// `(original_index, Some(value))` for a readable slot, or `(index, None)` when the
/// pointer could not be resolved (segment reclaimed, or the record failed
/// validation) — the caller re-resolves those through the LSM.
pub(crate) type BatchValue = (usize, Option<Vec<u8>>);

/// Where a value lives: which segment, where in it, and how long it is.
/// `segment_id == 0` is the reserved "no value" sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueLocation {
    pub segment_id: u32,
    pub rec_offset: u32,
    pub value_len: u32,
}

/// Per-record metadata that is not the value itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueRecordMeta {
    /// Per-key version counter, incremented on each overwrite.
    pub version: u32,
    /// Creation time in Unix millis. TTL expiry reads this straight from the record.
    pub epoch: u64,
    /// The global write sequence that produced this record. GC preserves it when it
    /// relocates a record. Reads no longer verify it: a segment id is never reused,
    /// so a pointer can never resolve to a different write.
    pub seq: u64,
}

fn value_checksum(value: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(value);
    hasher.finalize()
}

fn fsync_dir(dir: &Path) -> std::io::Result<()> {
    File::open(dir)?.sync_all()
}

/// 36 bytes, followed by the value bytes and then the key bytes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ValueRecordHeader {
    total_len: u32,
    version: u32,
    epoch: u64,
    seq: u64,
    key_len: u32,
    value_len: u32,
    /// CRC32 over the value payload.
    checksum: u32,
}

impl ValueRecordHeader {
    pub(crate) const SIZE: usize = 36;

    pub(crate) fn record_len(key_len: usize, value_len: usize) -> u64 {
        Self::SIZE as u64 + key_len as u64 + value_len as u64
    }

    fn to_bytes(self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        out[0..4].copy_from_slice(&self.total_len.to_le_bytes());
        out[4..8].copy_from_slice(&self.version.to_le_bytes());
        out[8..16].copy_from_slice(&self.epoch.to_le_bytes());
        out[16..24].copy_from_slice(&self.seq.to_le_bytes());
        out[24..28].copy_from_slice(&self.key_len.to_le_bytes());
        out[28..32].copy_from_slice(&self.value_len.to_le_bytes());
        out[32..36].copy_from_slice(&self.checksum.to_le_bytes());
        out
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::SIZE {
            return None;
        }
        let total_len = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
        let version = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
        let epoch = u64::from_le_bytes(bytes[8..16].try_into().ok()?);
        let seq = u64::from_le_bytes(bytes[16..24].try_into().ok()?);
        let key_len = u32::from_le_bytes(bytes[24..28].try_into().ok()?);
        let value_len = u32::from_le_bytes(bytes[28..32].try_into().ok()?);
        let checksum = u32::from_le_bytes(bytes[32..36].try_into().ok()?);
        // The length must agree with its parts, or these bytes are not a record
        // header — a torn tail, or a pointer into the middle of a record.
        if total_len as u64 != Self::record_len(key_len as usize, value_len as usize) {
            return None;
        }
        Some(Self {
            total_len,
            version,
            epoch,
            seq,
            key_len,
            value_len,
            checksum,
        })
    }

    fn meta(&self) -> ValueRecordMeta {
        ValueRecordMeta {
            version: self.version,
            epoch: self.epoch,
            seq: self.seq,
        }
    }
}

/// Live/garbage accounting for one segment. `live + garbage == total`.
#[derive(Debug, Clone, Copy, PartialEq, Archive, RkyvSerialize, RkyvDeserialize, serde::Serialize)]
pub struct SegmentStats {
    pub id: u32,
    /// Record bytes written into this segment (excludes the 16-byte file header).
    pub total_bytes: u64,
    /// Bytes still referenced by the LSM, as tracked by writers.
    pub live_bytes: u64,
    /// Bytes displaced by an overwrite or a delete.
    pub garbage_bytes: u64,
    /// Sealed segments are immutable. Exactly one segment per bucket is unsealed —
    /// the active tail — and GC never selects it.
    pub sealed: bool,
}

impl SegmentStats {
    /// The segment file's size on disk: its 16-byte header plus its records. Segments
    /// are dense append-only files, so this is exact — no `stat` needed, and there are
    /// no sparse holes to reason about.
    pub fn file_bytes(&self) -> u64 {
        SEGMENT_HEADER_SIZE + self.total_bytes
    }

    pub fn garbage_ratio_pct(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        (self.garbage_bytes as f64 / self.total_bytes as f64) * 100.0
    }
}

/// Durable per-bucket state: the segment inventory and the id high-water mark.
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, serde::Serialize)]
pub struct ValueLogMetadata {
    /// High-water mark of segment ids handed out, so ids stay unique for the life of
    /// the database across restarts too — see [`ValueLog::open`].
    pub next_segment_id: u64,
    pub active_segment_id: u32,
    pub segments: Vec<SegmentStats>,
    pub total_gc_runs: u64,
    pub total_bytes_reclaimed: u64,
}

impl Default for ValueLogMetadata {
    fn default() -> Self {
        Self::new()
    }
}

impl ValueLogMetadata {
    pub fn new() -> Self {
        Self {
            next_segment_id: 1,
            active_segment_id: 0,
            segments: Vec::new(),
            total_gc_runs: 0,
            total_bytes_reclaimed: 0,
        }
    }

    pub fn live_bytes(&self) -> u64 {
        self.segments.iter().map(|s| s.live_bytes).sum()
    }

    pub fn garbage_bytes(&self) -> u64 {
        self.segments.iter().map(|s| s.garbage_bytes).sum()
    }

    pub fn total_bytes(&self) -> u64 {
        self.segments.iter().map(|s| s.total_bytes).sum()
    }

    fn to_file_bytes(&self) -> Result<Vec<u8>> {
        let payload =
            rkyv::to_bytes::<rkyv::rancor::Error>(self).map_err(|e| ValueLogError::Serialization(format!("Failed to serialize metadata: {}", e)))?;
        let mut hasher = Hasher::new();
        hasher.update(&payload);
        let checksum = hasher.finalize();

        let mut out = Vec::with_capacity(16 + payload.len());
        out.extend_from_slice(&VALUE_LOG_METADATA_MAGIC);
        out.extend_from_slice(&VALUE_LOG_METADATA_VERSION.to_le_bytes());
        out.extend_from_slice(&checksum.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Decode metadata. An older version is **rejected, never upgraded**: the
    /// segmented value log is a hard format break (segment files, keys in records, a
    /// re-meaning of the value pointer), so an old file cannot be reinterpreted —
    /// only misread. The caller rebuilds from the segment files instead.
    fn from_file_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 16 || bytes.get(0..4) != Some(&VALUE_LOG_METADATA_MAGIC) {
            return Err(ValueLogError::Serialization("value log metadata: bad magic".into()));
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != VALUE_LOG_METADATA_VERSION {
            return Err(ValueLogError::Serialization(format!(
                "value log metadata version {version} unsupported (expected {VALUE_LOG_METADATA_VERSION})"
            )));
        }
        let checksum = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let payload_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let end = 16usize.saturating_add(payload_len);
        if bytes.len() < end {
            return Err(ValueLogError::Serialization("value log metadata: truncated payload".into()));
        }
        let payload = &bytes[16..end];
        let mut hasher = Hasher::new();
        hasher.update(payload);
        if hasher.finalize() != checksum {
            return Err(ValueLogError::Serialization("value log metadata: checksum mismatch".into()));
        }
        rkyv::from_bytes::<Self, rkyv::rancor::Error>(payload)
            .map_err(|e| ValueLogError::Serialization(format!("Failed to deserialize metadata: {}", e)))
    }
}

// ── The bucket's value log ─────────────────────────────────────────────────

/// Bounded cache of open segment handles.
struct FdCache {
    open: HashMap<u32, Arc<File>>,
    order: Vec<u32>,
    capacity: usize,
}

impl FdCache {
    fn new(capacity: usize) -> Self {
        Self {
            open: HashMap::new(),
            order: Vec::new(),
            capacity,
        }
    }

    fn get(&self, id: u32) -> Option<Arc<File>> {
        self.open.get(&id).cloned()
    }

    fn insert(&mut self, id: u32, file: Arc<File>) {
        if let std::collections::hash_map::Entry::Occupied(mut e) = self.open.entry(id) {
            e.insert(file);
            return;
        }
        while self.open.len() >= self.capacity && !self.order.is_empty() {
            let victim = self.order.remove(0);
            self.open.remove(&victim);
        }
        self.open.insert(id, file);
        self.order.push(id);
    }

    fn remove(&mut self, id: u32) {
        self.open.remove(&id);
        self.order.retain(|&i| i != id);
    }
}

/// The mutable half of a bucket's value log: the segment inventory and the tail it
/// is appending to.
struct Inner {
    segments: BTreeMap<u32, SegmentStats>,
    active_id: u32,
    active_file: Arc<File>,
    /// Offset in the active segment where the next record goes.
    active_offset: u64,
    total_gc_runs: u64,
    total_bytes_reclaimed: u64,
}

/// One bucket's value log: a series of immutable segment files plus an active tail.
pub struct ValueLog {
    dir: PathBuf,
    bucket: u32,
    segment_size: u64,
    /// Monotone id allocator. **Never derived from the filesystem at runtime**: if
    /// GC unlinked the highest segment, `max(files) + 1` would hand its id out again
    /// and a reader holding the old pointer would silently read a *different*
    /// record. Seeded at open from `max(persisted high-water mark, max id + 1)`.
    next_segment_id: AtomicU64,
    inner: RwLock<Inner>,
    fd_cache: Mutex<FdCache>,
    verify_checksums_on_read: AtomicBool,
    /// Set when the metadata file was missing or unreadable, so live/garbage
    /// accounting could not be loaded. `KVStore::open` then recomputes it exactly
    /// from the LSM's pointers; until it does, GC would under-trigger.
    stats_need_rebuild: AtomicBool,
}

impl ValueLog {
    fn segment_path(dir: &Path, bucket: u32, id: u32) -> PathBuf {
        dir.join(format!("value_log_{bucket}.seg{id:06}"))
    }

    fn metadata_path(dir: &Path, bucket: u32) -> PathBuf {
        dir.join(format!("value_log_{bucket}.metadata"))
    }

    fn parse_segment_id(bucket: u32, name: &str) -> Option<u32> {
        name.strip_prefix(&format!("value_log_{bucket}.seg"))?.parse::<u32>().ok()
    }

    fn write_segment_header(file: &File, id: u32) -> Result<()> {
        let mut header = [0u8; SEGMENT_HEADER_SIZE as usize];
        header[0..4].copy_from_slice(&SEGMENT_MAGIC);
        header[4..8].copy_from_slice(&SEGMENT_FORMAT_VERSION.to_le_bytes());
        header[8..12].copy_from_slice(&id.to_le_bytes());
        file.write_all_at(&header, 0)?;
        file.sync_all()?;
        Ok(())
    }

    /// The offset at which this segment's last complete record ends — i.e. the length
    /// the file would have with any torn tail from a crash mid-append removed.
    ///
    /// Only ever called on a not-full segment about to be **reused as the active
    /// tail** on open. Resuming appends *past* a torn record is data loss waiting to
    /// happen: the records appended after it would sit beyond a gap that GC's
    /// sequential [`for_each_record`](Self::for_each_record) scan halts at, so GC
    /// would relocate only the records before the gap and then unlink the whole
    /// segment — dropping any live key after it. Reusing the segment therefore
    /// truncates to this offset first. (Reads only headers, so it is cheap.)
    fn valid_record_end(file: &File, len: u64) -> Result<u64> {
        let mut offset = SEGMENT_HEADER_SIZE;
        while offset + ValueRecordHeader::SIZE as u64 <= len {
            let mut hdr_buf = [0u8; ValueRecordHeader::SIZE];
            file.read_at(&mut hdr_buf, offset).map_err(ValueLogError::Io)?;
            let Some(header) = ValueRecordHeader::from_bytes(&hdr_buf) else {
                break;
            };
            if offset + header.total_len as u64 > len {
                break;
            }
            offset += header.total_len as u64;
        }
        Ok(offset)
    }

    /// Open (or create) the value log for `bucket`.
    ///
    /// The **files on disk** are the source of truth for what exists; the metadata
    /// file supplies live/garbage accounting and the id high-water mark. The id
    /// allocator starts at the **max** of both, so neither a lost metadata file nor a
    /// GC-deleted top segment can hand out an id that was already used.
    pub fn open(dir: &Path, bucket: u32, segment_size: u64) -> Result<Self> {
        validate_segment_size(segment_size)?;
        std::fs::create_dir_all(dir)?;

        // 1. What exists on disk.
        let mut on_disk: BTreeMap<u32, u64> = BTreeMap::new(); // id -> record bytes
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(id) = Self::parse_segment_id(bucket, name) else {
                continue;
            };
            let len = entry.metadata()?.len();
            on_disk.insert(id, len.saturating_sub(SEGMENT_HEADER_SIZE));
        }

        // 2. What we last recorded about it.
        let meta_path = Self::metadata_path(dir, bucket);
        let (persisted, stats_loaded) = match std::fs::read(&meta_path) {
            Ok(bytes) => match ValueLogMetadata::from_file_bytes(&bytes) {
                Ok(m) => (m, true),
                Err(_) => {
                    let _ = std::fs::rename(&meta_path, meta_path.with_extension("corrupt"));
                    (ValueLogMetadata::new(), false)
                }
            },
            // No metadata and no segments = a brand-new bucket, nothing to rebuild.
            Err(_) => (ValueLogMetadata::new(), on_disk.is_empty()),
        };

        // 3. Reconcile. Accounting survives only for segments whose file is still
        //    there; a file we have no accounting for is assumed fully live
        //    (conservative — GC under-triggers rather than dropping live data) and
        //    flagged for an exact rebuild from the LSM.
        let mut need_rebuild = !stats_loaded;
        let mut segments: BTreeMap<u32, SegmentStats> = BTreeMap::new();
        for stats in &persisted.segments {
            if on_disk.contains_key(&stats.id) {
                segments.insert(stats.id, *stats);
            }
        }
        for (&id, &total) in &on_disk {
            segments.entry(id).or_insert_with(|| {
                need_rebuild = true;
                SegmentStats {
                    id,
                    total_bytes: total,
                    live_bytes: total,
                    garbage_bytes: 0,
                    sealed: true,
                }
            });
        }

        // 4. The id floor: never below a segment that exists, never below what we
        //    already promised was handed out.
        let max_existing = on_disk.keys().next_back().copied().unwrap_or(0) as u64;
        let mut next_id = persisted.next_segment_id.max(max_existing + 1).max(1);

        // 5. The active tail: reuse the newest segment if it still has room (so a
        //    restart doesn't strand a half-empty file), else start a fresh one.
        let reusable = on_disk
            .iter()
            .next_back()
            .filter(|(_, bytes)| SEGMENT_HEADER_SIZE + **bytes < segment_size)
            .map(|(&id, &bytes)| (id, bytes));

        let (active_id, active_file, active_offset) = match reusable {
            Some((id, bytes)) => {
                let file = OpenOptions::new().read(true).write(true).open(Self::segment_path(dir, bucket, id))?;
                let physical_len = SEGMENT_HEADER_SIZE + bytes;
                // A crash can leave a torn (partial) record at the tail. We are about to
                // append *past* it, so truncate it away first — otherwise the records we
                // append would sit beyond a gap that GC's sequential scan halts at, and
                // GC could unlink the whole segment after relocating only the records
                // before the gap, losing every live key after it.
                let valid_end = Self::valid_record_end(&file, physical_len)?;
                if valid_end < physical_len {
                    file.set_len(valid_end)?;
                    file.sync_all()?;
                }
                if let Some(s) = segments.get_mut(&id) {
                    s.sealed = false;
                    let new_total = valid_end - SEGMENT_HEADER_SIZE;
                    if s.total_bytes != new_total {
                        // The persisted byte count disagrees with the segment's true tail
                        // (a torn record, or records appended after the last metadata
                        // flush), so its live/garbage split can't be trusted. Fix the
                        // total and let the exact rebuild from the LSM recompute the split.
                        s.total_bytes = new_total;
                        s.live_bytes = new_total;
                        s.garbage_bytes = 0;
                        need_rebuild = true;
                    }
                }
                (id, Arc::new(file), valid_end)
            }
            None => {
                let id = next_id as u32;
                next_id += 1;
                let path = Self::segment_path(dir, bucket, id);
                let file = OpenOptions::new().create(true).truncate(true).read(true).write(true).open(&path)?;
                Self::write_segment_header(&file, id)?;
                fsync_dir(dir)?;
                segments.insert(
                    id,
                    SegmentStats {
                        id,
                        total_bytes: 0,
                        live_bytes: 0,
                        garbage_bytes: 0,
                        sealed: false,
                    },
                );
                (id, Arc::new(file), SEGMENT_HEADER_SIZE)
            }
        };

        let mut fd_cache = FdCache::new(MAX_OPEN_SEGMENTS_PER_BUCKET);
        fd_cache.insert(active_id, Arc::clone(&active_file));

        let log = Self {
            dir: dir.to_path_buf(),
            bucket,
            segment_size,
            next_segment_id: AtomicU64::new(next_id),
            inner: RwLock::new(Inner {
                segments,
                active_id,
                active_file,
                active_offset,
                total_gc_runs: persisted.total_gc_runs,
                total_bytes_reclaimed: persisted.total_bytes_reclaimed,
            }),
            fd_cache: Mutex::new(fd_cache),
            verify_checksums_on_read: AtomicBool::new(false),
            stats_need_rebuild: AtomicBool::new(need_rebuild),
        };
        log.flush_metadata()?;
        Ok(log)
    }

    /// True when live/garbage accounting could not be loaded and must be recomputed
    /// from the LSM's pointers.
    pub fn stats_need_rebuild(&self) -> bool {
        self.stats_need_rebuild.load(Ordering::Acquire)
    }

    #[allow(dead_code)] // surfaced through ShardedValueLog
    pub fn segment_size(&self) -> u64 {
        self.segment_size
    }

    pub fn set_verify_checksums_on_read(&self, verify: bool) {
        self.verify_checksums_on_read.store(verify, Ordering::Relaxed);
    }

    #[inline]
    fn verify_checksums_on_read(&self) -> bool {
        self.verify_checksums_on_read.load(Ordering::Relaxed)
    }

    /// Allocate the next segment id. Monotone in-process; the persisted high-water
    /// mark keeps it monotone across restarts.
    fn alloc_segment_id(&self) -> u32 {
        self.next_segment_id.fetch_add(1, Ordering::AcqRel) as u32
    }

    /// Seal the current tail and open a fresh segment as the new one.
    fn roll_active_segment(&self, inner: &mut Inner) -> Result<()> {
        if let Some(s) = inner.segments.get_mut(&inner.active_id) {
            s.sealed = true;
        }

        let id = self.alloc_segment_id();
        let path = Self::segment_path(&self.dir, self.bucket, id);
        let file = OpenOptions::new().create(true).truncate(true).read(true).write(true).open(&path)?;
        Self::write_segment_header(&file, id)?;
        fsync_dir(&self.dir)?;

        let file = Arc::new(file);
        self.fd_cache.lock().insert(id, Arc::clone(&file));
        inner.segments.insert(
            id,
            SegmentStats {
                id,
                total_bytes: 0,
                live_bytes: 0,
                garbage_bytes: 0,
                sealed: false,
            },
        );
        inner.active_id = id;
        inner.active_file = file;
        inner.active_offset = SEGMENT_HEADER_SIZE;
        Ok(())
    }

    /// A handle to a segment's file, opening and caching it on demand.
    ///
    /// A missing file is [`ValueLogError::SegmentMissing`], not an IO error: GC
    /// reclaimed it after relocating its records, so the caller should re-resolve the
    /// key through the LSM and retry.
    fn segment_file(&self, id: u32) -> Result<Arc<File>> {
        if let Some(f) = self.fd_cache.lock().get(id) {
            return Ok(f);
        }
        let path = Self::segment_path(&self.dir, self.bucket, id);
        match OpenOptions::new().read(true).open(&path) {
            Ok(f) => {
                let f = Arc::new(f);
                self.fd_cache.lock().insert(id, Arc::clone(&f));
                Ok(f)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(ValueLogError::SegmentMissing(id)),
            Err(e) => Err(ValueLogError::Io(e)),
        }
    }

    // ── Writes ────────────────────────────────────────────────────────────

    /// Append a record, rolling to a new segment if this one is full. Callers hold
    /// the bucket write lock.
    pub fn append(&self, key: &[u8], value: &[u8], meta: ValueRecordMeta, sync: bool) -> Result<ValueLocation> {
        let record_len = ValueRecordHeader::record_len(key.len(), value.len());
        if SEGMENT_HEADER_SIZE + record_len > self.segment_size {
            return Err(ValueLogError::ValueTooLarge {
                record_len,
                segment_size: self.segment_size,
            });
        }

        let mut inner = self.inner.write();
        if inner.active_offset + record_len > self.segment_size {
            self.roll_active_segment(&mut inner)?;
        }

        let segment_id = inner.active_id;
        let rec_offset = inner.active_offset;
        let file = Arc::clone(&inner.active_file);

        let header = ValueRecordHeader {
            total_len: record_len as u32,
            version: meta.version,
            epoch: meta.epoch,
            seq: meta.seq,
            key_len: key.len() as u32,
            value_len: value.len() as u32,
            checksum: value_checksum(value),
        };

        // [header][value][key] — one write, and a later read of just the value is a
        // single pread of `header + value_len`.
        let mut buf = Vec::with_capacity(record_len as usize);
        buf.extend_from_slice(&header.to_bytes());
        buf.extend_from_slice(value);
        buf.extend_from_slice(key);
        file.write_all_at(&buf, rec_offset)?;
        if sync {
            file.sync_data()?;
        }

        inner.active_offset += record_len;
        if let Some(s) = inner.segments.get_mut(&segment_id) {
            s.total_bytes += record_len;
            s.live_bytes += record_len;
        }

        Ok(ValueLocation {
            segment_id,
            rec_offset: rec_offset as u32,
            value_len: value.len() as u32,
        })
    }

    /// Account for a record a newer write (or a delete) has just displaced.
    /// Costs **no I/O** — the caller holds the old pointer, which carries `value_len`.
    pub fn note_displaced(&self, location: ValueLocation, key_len: usize) {
        if location.segment_id == 0 {
            return;
        }
        let bytes = ValueRecordHeader::record_len(key_len, location.value_len as usize);
        let mut inner = self.inner.write();
        if let Some(s) = inner.segments.get_mut(&location.segment_id) {
            s.live_bytes = s.live_bytes.saturating_sub(bytes);
            s.garbage_bytes = s.garbage_bytes.saturating_add(bytes);
        }
    }

    pub fn sync(&self) -> Result<()> {
        let file = Arc::clone(&self.inner.read().active_file);
        file.sync_data()?;
        Ok(())
    }

    // ── Reads ─────────────────────────────────────────────────────────────

    fn parse_record(&self, buf: &[u8], location: ValueLocation) -> Result<(Vec<u8>, ValueRecordMeta)> {
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
        let value = buf[start..end].to_vec();
        if self.verify_checksums_on_read() && value_checksum(&value) != header.checksum {
            return Err(ValueLogError::CorruptedLog);
        }
        Ok((value, header.meta()))
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
        file.read_at(&mut buf, location.rec_offset as u64).map_err(ValueLogError::Io)?;
        self.parse_record(&buf, location)
    }

    /// The record's metadata only — a 36-byte read. TTL expiry uses it for `epoch`.
    pub fn read_record_meta(&self, location: ValueLocation) -> Result<ValueRecordMeta> {
        if location.segment_id == 0 {
            return Err(ValueLogError::InvalidLocation);
        }
        let file = self.segment_file(location.segment_id)?;
        let mut buf = [0u8; ValueRecordHeader::SIZE];
        file.read_at(&mut buf, location.rec_offset as u64).map_err(ValueLogError::Io)?;
        let header = ValueRecordHeader::from_bytes(&buf).ok_or(ValueLogError::CorruptedLog)?;
        Ok(header.meta())
    }

    /// Read many of this bucket's values. Grouped by segment so each file is opened
    /// once; one `pread` per value (there is no shared page header or slot directory
    /// left to amortise — there isn't one).
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

    // ── GC support ────────────────────────────────────────────────────────

    /// Is there anything for GC to actually do in this bucket?
    ///
    /// Cheap (in-memory counters) and exact enough to keep the GC worker from waking on a
    /// bucket whose garbage it cannot collect — which is what made a small, fully-deleted
    /// namespace log "starting GC ... reclaimed 0 bytes" on every tick.
    pub fn has_gc_work(&self, threshold_pct: f64) -> bool {
        !self.gc_candidates(threshold_pct).is_empty() || self.tail_needs_sealing()
    }

    /// True when the active tail is mostly garbage and should be sealed so GC can collect
    /// it (see [`TAIL_GC_MIN_GARBAGE_PCT`]).
    fn tail_needs_sealing(&self) -> bool {
        let inner = self.inner.read();
        inner
            .segments
            .get(&inner.active_id)
            .is_some_and(|s| s.garbage_bytes > 0 && s.garbage_ratio_pct() >= TAIL_GC_MIN_GARBAGE_PCT)
    }

    /// Seal the active tail if it is mostly garbage, opening a fresh one in its place, and
    /// return the id it just sealed so GC can collect it in this same pass.
    ///
    /// The caller must hold the bucket write lock: this rolls the segment writers append
    /// to. A record appended just before the roll is simply live data in the sealed
    /// segment, and GC will relocate it like any other survivor.
    pub fn seal_tail_for_gc(&self) -> Result<Option<u32>> {
        if !self.tail_needs_sealing() {
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
            file.read_at(&mut hdr_buf, offset).map_err(ValueLogError::Io)?;
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
            file.read_at(&mut body, offset + ValueRecordHeader::SIZE as u64)
                .map_err(ValueLogError::Io)?;
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
    /// (Segments are dense, appended files — there are no sparse holes to reason
    /// about any more.)
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const SEG: u64 = 64 * 1024; // the minimum legal segment size

    fn meta(seq: u64) -> ValueRecordMeta {
        ValueRecordMeta {
            version: 1,
            epoch: 1_700_000_000_000,
            seq,
        }
    }

    #[test]
    fn record_round_trips_key_and_value() -> Result<()> {
        let dir = TempDir::new()?;
        let log = ValueLog::open(dir.path(), 0, SEG)?;

        let loc = log.append(b"the-key", b"the-value", meta(7), true)?;
        // The pointer carries the value length, which is what makes a read one pread.
        assert_eq!(loc.value_len, b"the-value".len() as u32);

        let (value, m) = log.read_value_and_meta(loc)?;
        assert_eq!(value, b"the-value");
        assert_eq!(m.seq, 7);

        // The key is in the record too — this is what lets GC decide liveness without
        // inverting the whole LSM.
        let mut seen = Vec::new();
        log.for_each_record(loc.segment_id, |key, value, _, _| {
            seen.push((key.to_vec(), value));
        })?;
        assert_eq!(seen, vec![(b"the-key".to_vec(), b"the-value".to_vec())]);
        Ok(())
    }

    #[test]
    fn appends_roll_into_new_segments_when_full() -> Result<()> {
        let dir = TempDir::new()?;
        let log = ValueLog::open(dir.path(), 0, SEG)?;

        let value = vec![0xAB; 8 * 1024];
        let mut locations = Vec::new();
        for i in 0..20 {
            locations.push(log.append(format!("k{i}").as_bytes(), &value, meta(i as u64), false)?);
        }

        let segments: Vec<u32> = locations.iter().map(|l| l.segment_id).collect();
        assert!(
            segments.windows(2).any(|w| w[0] != w[1]),
            "20 × 8 KiB records cannot fit one 64 KiB segment — expected a rollover, got {segments:?}"
        );
        // Ids only ever go up.
        assert!(segments.windows(2).all(|w| w[0] <= w[1]), "segment ids must be monotone");

        for (i, loc) in locations.iter().enumerate() {
            assert_eq!(log.read_value(*loc)?, value, "record {i} unreadable after rollover");
        }
        Ok(())
    }

    #[test]
    fn segment_ids_are_never_reused_even_after_the_newest_is_unlinked() -> Result<()> {
        // The landmine: if the next id were `max(existing files) + 1`, unlinking the
        // newest segment would hand its id out again — and a reader holding the old
        // pointer would silently read a different record.
        let dir = TempDir::new()?;
        let log = ValueLog::open(dir.path(), 0, SEG)?;

        let value = vec![0xCD; 8 * 1024];
        let mut ids = Vec::new();
        for i in 0..12 {
            ids.push(log.append(format!("k{i}").as_bytes(), &value, meta(i as u64), false)?.segment_id);
        }
        let max_ever = *ids.iter().max().unwrap();

        // Unlink the newest SEALED segment: on disk, `max(files)` now skips its id.
        let highest_sealed = *ids.iter().filter(|&&id| id != log.inner.read().active_id).max().unwrap();
        log.unlink_segment(highest_sealed)?;

        // Keep appending so fresh segments get created, and check every NEWLY created
        // id is above everything ever handed out — including the one we just freed.
        for i in 20..60 {
            let id = log.append(format!("n{i}").as_bytes(), &value, meta(i as u64), false)?.segment_id;
            if !ids.contains(&id) {
                assert!(
                    id > max_ever,
                    "segment id {id} was reused (max ever handed out was {max_ever}); \
                     ids must come from the monotone counter, never max(existing files) + 1"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn segment_id_high_water_mark_survives_restart() -> Result<()> {
        let dir = TempDir::new()?;
        let value = vec![0xEF; 8 * 1024];

        let last_id = {
            let log = ValueLog::open(dir.path(), 0, SEG)?;
            let mut id = 0;
            for i in 0..12 {
                id = log.append(format!("k{i}").as_bytes(), &value, meta(i as u64), false)?.segment_id;
            }
            // Drop every sealed segment, so the filesystem no longer remembers those ids.
            let sealed: Vec<u32> = log.segment_stats().iter().filter(|s| s.sealed).map(|s| s.id).collect();
            for id in sealed {
                log.unlink_segment(id)?;
            }
            log.flush_metadata()?;
            id
        };

        // Reopen: the id allocator must not regress, even though the files that used
        // those ids are gone. The persisted high-water mark is what guarantees it.
        let log = ValueLog::open(dir.path(), 0, SEG)?;
        let next = log.append(b"after-restart", &value, meta(99), false)?.segment_id;
        assert!(
            next >= last_id,
            "segment id regressed across restart: was {last_id}, handed out {next} — the high-water mark was lost"
        );
        Ok(())
    }

    #[test]
    fn displaced_records_become_garbage_without_touching_the_record() -> Result<()> {
        let dir = TempDir::new()?;
        let log = ValueLog::open(dir.path(), 0, SEG)?;

        let old = log.append(b"k", &vec![1u8; 1000], meta(1), false)?;
        let stats = log.segment_stats();
        let live_before = stats[0].live_bytes;
        assert_eq!(stats[0].garbage_bytes, 0);

        log.note_displaced(old, b"k".len());

        let stats = log.segment_stats();
        assert_eq!(stats[0].garbage_bytes, live_before, "the displaced record's bytes become garbage");
        assert_eq!(stats[0].live_bytes, 0);

        // The record itself is untouched — its segment may be sealed and immutable, and
        // liveness is the LSM's business, not the record's.
        assert_eq!(log.read_value(old)?, vec![1u8; 1000]);
        Ok(())
    }

    #[test]
    fn gc_candidates_are_sealed_segments_over_the_threshold_worst_first() -> Result<()> {
        let dir = TempDir::new()?;
        let log = ValueLog::open(dir.path(), 0, SEG)?;

        let value = vec![0x11; 8 * 1024];
        let mut locs = Vec::new();
        for i in 0..20 {
            locs.push(log.append(format!("k{i}").as_bytes(), &value, meta(i as u64), false)?);
        }
        // Garbage the records in the earliest segment only.
        let first = locs[0].segment_id;
        for loc in locs.iter().filter(|l| l.segment_id == first) {
            log.note_displaced(*loc, 2);
        }

        let candidates = log.gc_candidates(30.0);
        assert!(candidates.iter().any(|c| c.id == first), "the garbage-heavy segment must be a candidate");
        assert!(
            candidates.iter().all(|c| c.sealed),
            "the active tail must never be collected — it is still being appended to"
        );
        assert!(
            candidates.windows(2).all(|w| w[0].garbage_ratio_pct() >= w[1].garbage_ratio_pct()),
            "candidates must be worst-first"
        );
        Ok(())
    }

    #[test]
    fn a_reader_holding_an_open_handle_survives_the_unlink() -> Result<()> {
        // POSIX: an open fd outlives unlink. This is what lets GC delete a segment
        // without coordinating with in-flight readers.
        let dir = TempDir::new()?;
        let log = ValueLog::open(dir.path(), 0, SEG)?;

        let value = vec![0x77; 8 * 1024];
        let mut locs = Vec::new();
        for i in 0..12 {
            locs.push(log.append(format!("k{i}").as_bytes(), &value, meta(i as u64), false)?);
        }
        let sealed = log.segment_stats().iter().find(|s| s.sealed).map(|s| s.id).expect("a sealed segment");
        let victim = *locs.iter().find(|l| l.segment_id == sealed).unwrap();

        let handle = log.segment_file(sealed)?; // a reader's in-flight handle
        log.unlink_segment(sealed)?;

        // The held handle still reads correctly...
        assert_eq!(log.read_value_from_file(&handle, victim)?.0, value);
        // ...while a fresh resolution reports the segment is gone, so the caller
        // re-resolves through the LSM. Loud, local, retryable — never a wrong value.
        assert!(
            matches!(log.read_value(victim), Err(ValueLogError::SegmentMissing(id)) if id == sealed),
            "a pointer into a reclaimed segment must fail loudly, not silently resolve"
        );
        Ok(())
    }

    #[test]
    fn invalid_segment_sizes_are_rejected() {
        let dir = TempDir::new().unwrap();
        for bad in [0, 4096, SEG + 1, u32::MAX as u64 + 4096] {
            assert!(
                matches!(ValueLog::open(dir.path(), 0, bad), Err(ValueLogError::InvalidSegmentSize(got)) if got == bad),
                "segment size {bad} must be rejected"
            );
        }
    }

    #[test]
    fn a_value_too_large_for_a_segment_is_rejected() -> Result<()> {
        let dir = TempDir::new()?;
        let log = ValueLog::open(dir.path(), 0, SEG)?;
        let huge = vec![0u8; SEG as usize];
        assert!(
            matches!(log.append(b"k", &huge, meta(1), false), Err(ValueLogError::ValueTooLarge { .. })),
            "a record that cannot fit a segment must be refused, not silently truncated"
        );
        Ok(())
    }

    #[test]
    fn stats_rebuild_from_the_lsm_when_metadata_is_lost() -> Result<()> {
        let dir = TempDir::new()?;
        let value = vec![0x33; 4 * 1024];
        let (live_loc, dead_loc) = {
            let log = ValueLog::open(dir.path(), 0, SEG)?;
            let live = log.append(b"live", &value, meta(1), false)?;
            let dead = log.append(b"dead", &value, meta(2), false)?;
            log.flush_metadata()?;
            (live, dead)
        };

        // Lose the metadata file, as a crash before the flush would.
        std::fs::remove_file(dir.path().join("value_log_0.metadata"))?;

        let log = ValueLog::open(dir.path(), 0, SEG)?;
        assert!(log.stats_need_rebuild(), "a missing metadata file must be flagged for rebuild");

        // The caller recomputes liveness from the LSM's pointers: only `live` is still
        // referenced, so `dead`'s bytes are garbage.
        let live_bytes = ValueRecordHeader::record_len(b"live".len(), value.len());
        let mut live_map = HashMap::new();
        live_map.insert(live_loc.segment_id, live_bytes);
        log.rebuild_stats(&live_map);

        assert!(!log.stats_need_rebuild());
        let stats = log.segment_stats();
        let seg = stats.iter().find(|s| s.id == live_loc.segment_id).unwrap();
        assert_eq!(seg.live_bytes, live_bytes);
        assert!(seg.garbage_bytes > 0, "the unreferenced record must be counted as garbage");
        assert_eq!(log.read_value(dead_loc)?, value, "the bytes are still there — just unreferenced");
        Ok(())
    }

    #[test]
    fn a_corrupted_value_is_detected_when_checksums_are_verified() -> Result<()> {
        // Regression: values carry a CRC32 of their payload, and `verify_checksums_on_read`
        // makes reads re-check it. Without this the structural checks (length, offset)
        // would happily serve bit-rotted bytes.
        let dir = TempDir::new()?;
        let log = ValueLog::open(dir.path(), 0, SEG)?;
        let loc = log.append(b"k", b"the-original-value", meta(1), true)?;

        // Flip a byte inside the value payload, in place, behind the log's back.
        let path = ValueLog::segment_path(dir.path(), 0, loc.segment_id);
        let mut bytes = std::fs::read(&path)?;
        let value_at = loc.rec_offset as usize + ValueRecordHeader::SIZE;
        bytes[value_at] ^= 0xFF;
        std::fs::write(&path, &bytes)?;

        // A fresh log (so nothing is cached) with verification off still returns the
        // corrupted bytes — that is the documented latency-first default.
        let log = ValueLog::open(dir.path(), 0, SEG)?;
        assert_ne!(log.read_value(loc)?, b"the-original-value", "sanity: the bytes really were corrupted");

        // With verification on, the read fails instead of serving them.
        log.set_verify_checksums_on_read(true);
        assert!(
            matches!(log.read_value(loc), Err(ValueLogError::CorruptedLog)),
            "a corrupted value must be rejected, not served"
        );
        Ok(())
    }

    #[test]
    fn a_torn_tail_from_a_crash_mid_append_is_ignored() -> Result<()> {
        // A crash can leave a partial record at the end of the active segment. Nothing
        // references it — the LSM only ever learned about records whose append
        // returned — so the scan must stop cleanly there and keep everything before it.
        let dir = TempDir::new()?;
        let log = ValueLog::open(dir.path(), 0, SEG)?;
        let a = log.append(b"a", b"value-a", meta(1), true)?;
        let b = log.append(b"b", b"value-b", meta(2), true)?;

        // Simulate the torn write: garbage bytes appended past the last good record.
        let path = ValueLog::segment_path(dir.path(), 0, a.segment_id);
        let mut bytes = std::fs::read(&path)?;
        bytes.extend_from_slice(&[0xAB; 20]); // shorter than a header, and not one anyway
        std::fs::write(&path, &bytes)?;

        let log = ValueLog::open(dir.path(), 0, SEG)?;
        let mut seen = Vec::new();
        log.for_each_record(a.segment_id, |key, _, _, _| seen.push(key.to_vec()))?;
        assert_eq!(
            seen,
            vec![b"a".to_vec(), b"b".to_vec()],
            "the complete records must survive the torn tail"
        );

        // ...and both are still readable by pointer.
        assert_eq!(log.read_value(a)?, b"value-a");
        assert_eq!(log.read_value(b)?, b"value-b");
        Ok(())
    }

    #[test]
    fn records_appended_after_a_torn_tail_stay_visible_to_gc() -> Result<()> {
        // The dangerous sequence: a crash leaves a torn record at the tail, then on
        // reopen we resume appending into that same not-full segment. If the torn bytes
        // were left in place, a record appended after them would sit beyond a gap that
        // GC's sequential scan stops at — GC would relocate only the records before the
        // gap and then unlink the segment, silently losing the later live record.
        let dir = TempDir::new()?;
        let (a, b) = {
            let log = ValueLog::open(dir.path(), 0, SEG)?;
            let a = log.append(b"a", b"value-a", meta(1), true)?;
            let b = log.append(b"b", b"value-b", meta(2), true)?;
            (a, b)
        };

        // Simulate the torn write: partial bytes appended past the last good record.
        let path = ValueLog::segment_path(dir.path(), 0, a.segment_id);
        let mut bytes = std::fs::read(&path)?;
        bytes.extend_from_slice(&[0xAB; 20]); // shorter than a header, and not one anyway
        std::fs::write(&path, &bytes)?;

        // Reopen (recovery) and resume appending into the reused tail.
        let log = ValueLog::open(dir.path(), 0, SEG)?;
        let c = log.append(b"c", b"value-c", meta(3), true)?;
        assert_eq!(c.segment_id, a.segment_id, "c must land in the reused tail, exercising the torn-tail path");

        // GC learns a segment's contents by scanning it. All three records — including
        // the one appended after the torn tail — must be visited in order, or GC would
        // drop c while unlinking the segment.
        let mut seen = Vec::new();
        log.for_each_record(a.segment_id, |key, _, _, _| seen.push(key.to_vec()))?;
        assert_eq!(
            seen,
            vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()],
            "the record appended after the torn tail must be visible to GC"
        );

        // The torn bytes were truncated, not skipped: c sits immediately after b, and
        // every pointer still resolves to its own value.
        assert_eq!(log.read_value(a)?, b"value-a");
        assert_eq!(log.read_value(b)?, b"value-b");
        assert_eq!(log.read_value(c)?, b"value-c");
        Ok(())
    }

    #[test]
    fn reads_stay_correct_when_the_fd_cache_evicts() -> Result<()> {
        // The fd cache is bounded (MAX_OPEN_SEGMENTS_PER_BUCKET). A bucket with more
        // segments than that must still read correctly from the evicted ones — they are
        // immutable, so an evicted handle is simply reopened on demand.
        let dir = TempDir::new()?;
        let log = ValueLog::open(dir.path(), 0, SEG)?;

        // ~2 records per 64 KiB segment → comfortably more segments than the cache holds.
        let value = vec![0x5A; 30 * 1024];
        let mut locs = Vec::new();
        for i in 0..(MAX_OPEN_SEGMENTS_PER_BUCKET * 3) {
            locs.push(log.append(format!("k{i:04}").as_bytes(), &value, meta(i as u64), false)?);
        }
        let distinct: std::collections::HashSet<u32> = locs.iter().map(|l| l.segment_id).collect();
        assert!(
            distinct.len() > MAX_OPEN_SEGMENTS_PER_BUCKET,
            "this test needs more segments ({}) than the fd cache holds ({MAX_OPEN_SEGMENTS_PER_BUCKET})",
            distinct.len()
        );

        // Read the OLDEST records last — they are the ones evicted from the cache.
        for (i, loc) in locs.iter().enumerate() {
            assert_eq!(log.read_value(*loc)?, value, "record {i} unreadable after its segment was evicted");
        }
        Ok(())
    }

    #[test]
    fn live_and_garbage_always_sum_to_total() -> Result<()> {
        let dir = TempDir::new()?;
        let log = ValueLog::open(dir.path(), 0, SEG)?;

        let value = vec![0x9C; 4 * 1024];
        let mut locs = Vec::new();
        for i in 0..12 {
            locs.push(log.append(format!("k{i}").as_bytes(), &value, meta(i as u64), false)?);
        }
        for (i, loc) in locs.iter().enumerate() {
            if i % 3 == 0 {
                log.note_displaced(*loc, format!("k{i}").len());
            }
        }

        for s in log.segment_stats() {
            assert_eq!(
                s.live_bytes + s.garbage_bytes,
                s.total_bytes,
                "segment {} broke the live+garbage==total invariant that GC selection relies on",
                s.id
            );
        }
        Ok(())
    }
}
