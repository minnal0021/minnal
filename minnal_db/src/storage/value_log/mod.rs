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
//!   never silently to a *different* key's record. So a reader needs no check
//!   that its pointer is still current: on `SegmentMissing` it re-resolves the
//!   key through the LSM and retries.
//! - GC's unit is a segment, not the whole bucket. It rewrites one sealed
//!   segment's survivors and unlinks the file; it never copies untouched data and
//!   never rewrites a file in place.
//!
//! # Layout
//!
//! ```text
//! segment file:  [file header 16B] [record] [record] ...
//!
//! record:        [header 32B][value bytes][key bytes]
//!                 ^ rec_offset
//! ```
//!
//! The value sits **before** the key deliberately: a reader knows `value_len` from
//! the pointer, so a `get` is **one `pread` of exactly `32 + value_len` bytes**.
//! Only GC needs the key, and it reads whole records sequentially anyway.
//!
//! # Liveness and garbage
//!
//! Records carry **no tombstone/updated flags** — the LSM is the sole authority on
//! what is live. When the last LSM entry pointing at a record goes (a memtable
//! write replaces it, or the L0→L1 merge drops it), that record's size is added to
//! its segment's garbage counter ([`ValueLog::note_displaced`]): the pointer
//! carries `value_len`, so the accounting costs **zero I/O** and no record on disk
//! is ever touched again. The counters are a *hint* for GC selection; the exact
//! live set always comes from the LSM.

mod format;
mod gc;
mod reads;
pub mod sharded;

pub(crate) use format::ValueRecordHeader;
use format::*;
pub use format::{SegmentStats, ValueLocation, ValueLogMetadata, ValueRecordMeta};
pub use gc::TAIL_GC_MIN_GARBAGE_PCT;
pub(crate) use reads::BatchValue;
use reads::read_exact_or_corrupt;

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
    /// A segment file on disk was written in another format (or is not a segment).
    #[error("value-log segment {path} has format {found}, expected {expected}")]
    UnsupportedSegmentFormat { path: String, found: String, expected: u32 },
}

pub type Result<T> = std::result::Result<T, ValueLogError>;

/// The largest `key.len() + value.len()` a segment of `segment_size` bytes can
/// hold, once the segment header and the per-record header are accounted for.
///
/// [`ValueLog::append`] rejects anything above this with
/// [`ValueLogError::ValueTooLarge`] — but by then the write has already been
/// appended to the WAL and fsynced, so the caller has been told it succeeded.
/// The write boundary (`Database::check_write_size`) applies the same bound
/// *before* the WAL append, which is the only place a rejection can still mean
/// "nothing happened". Keep the two in agreement.
pub(crate) fn max_record_payload(segment_size: u64) -> u64 {
    segment_size.saturating_sub(SEGMENT_HEADER_SIZE + ValueRecordHeader::SIZE as u64)
}

const SEGMENT_SIZE_ALIGNMENT: u64 = 4096;
const MIN_SEGMENT_SIZE_BYTES: u64 = 64 * 1024;
/// `rec_offset` in the value pointer is a `u32`, so every record must be
/// addressable within 4 GiB of its segment.
const MAX_SEGMENT_SIZE_BYTES: u64 = u32::MAX as u64;

/// Validate a configured segment size.
///
/// The segment size is **not** fixed at creation: no pointer encodes a segment's
/// size (only its id and a byte offset), so existing segments keep the size they
/// were written at and new ones use whatever is configured now.
pub fn validate_segment_size(segment_size: u64) -> Result<()> {
    if !(MIN_SEGMENT_SIZE_BYTES..=MAX_SEGMENT_SIZE_BYTES).contains(&segment_size) || !segment_size.is_multiple_of(SEGMENT_SIZE_ALIGNMENT) {
        return Err(ValueLogError::InvalidSegmentSize(segment_size));
    }
    Ok(())
}

/// How many segment files a bucket keeps open at once.
///
/// Segments are immutable, so an evicted handle can always be reopened. Eviction
/// only drops the cache's `Arc<File>`, so an in-flight reader's clone keeps the fd
/// alive — and an already-open fd keeps reading correctly even after the file is
/// unlinked.
const MAX_OPEN_SEGMENTS_PER_BUCKET: usize = 32;

fn fsync_dir(dir: &Path) -> std::io::Result<()> {
    File::open(dir)?.sync_all()
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
    /// How far into the active segment is known to be on stable storage. Every
    /// sealed segment is fsynced as it is sealed, so `active_offset -
    /// synced_offset` is everything this bucket holds that a power loss could
    /// still take away.
    synced_offset: u64,
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

    /// Refuse a segment written in another format. A file shorter than the
    /// segment header holds no records (a crash between creating it and writing
    /// the header), so there is nothing to misread and it is let through.
    fn check_segment_format(path: &Path, len: u64) -> Result<()> {
        if len < SEGMENT_HEADER_SIZE {
            return Ok(());
        }
        let mut header = [0u8; 8];
        read_exact_or_corrupt(&File::open(path)?, &mut header, 0)?;
        let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
        if header[0..4] != SEGMENT_MAGIC || version != SEGMENT_FORMAT_VERSION {
            let found = if header[0..4] == SEGMENT_MAGIC {
                version.to_string()
            } else {
                "unknown (bad magic)".to_string()
            };
            return Err(ValueLogError::UnsupportedSegmentFormat {
                path: path.display().to_string(),
                found,
                expected: SEGMENT_FORMAT_VERSION,
            });
        }
        Ok(())
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
            match file.read_exact_at(&mut hdr_buf, offset) {
                Ok(()) => {}
                // A short read here is a truncated tail — the clean end is what we have.
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(ValueLogError::Io(e)),
            }
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
            Self::check_segment_format(&entry.path(), len)?;
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
            let Some(&physical) = on_disk.get(&stats.id) else {
                // File is gone (GC unlinked it) — drop its accounting with it.
                continue;
            };
            if stats.total_bytes == physical {
                segments.insert(stats.id, *stats);
            } else {
                // Metadata is not flushed on every append, so a crash can leave a
                // segment file longer than its persisted byte count (records appended
                // after the last flush) — or, for the active tail, shorter once its torn
                // tail is trimmed below. Trust the file, not the stale count: take the
                // on-disk length and let the exact rebuild from the LSM recompute the
                // live/garbage split.
                need_rebuild = true;
                segments.insert(
                    stats.id,
                    SegmentStats {
                        id: stats.id,
                        total_bytes: physical,
                        live_bytes: physical,
                        garbage_bytes: 0,
                        sealed: stats.sealed,
                    },
                );
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

        // After a restart exactly one segment — the active tail — is being appended to;
        // every other segment is immutable. Persisted `sealed` flags can disagree (a
        // segment that was active when metadata was last flushed, then rolled before a
        // crash, is recorded unsealed), and a non-active segment left unsealed would
        // never be a GC candidate. Reconstruct the flag from reality instead of trusting
        // the stale metadata.
        for (&id, s) in segments.iter_mut() {
            s.sealed = id != active_id;
        }

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
                // What a previous process wrote reached at least the page cache,
                // which survives that process; nothing here is newly written.
                synced_offset: active_offset,
                total_gc_runs: persisted.total_gc_runs,
                total_bytes_reclaimed: persisted.total_bytes_reclaimed,
            }),
            fd_cache: Mutex::new(fd_cache),
            verify_checksums_on_read: AtomicBool::new(false),
            stats_need_rebuild: AtomicBool::new(need_rebuild),
        };
        // Don't persist this view while a rebuild is still pending: the live/garbage
        // split is a conservative placeholder (everything live), and flushing it now
        // would let a crash-before-rebuild make the placeholder look authoritative on the
        // next open (metadata present, byte counts already matching → nothing re-flagged).
        // `KVStore::open` runs the exact rebuild and flushes the corrected metadata; if we
        // crash before it does, the still-stale on-disk metadata re-triggers the rebuild.
        if !need_rebuild {
            log.flush_metadata()?;
        }
        Ok(log)
    }

    /// True when live/garbage accounting could not be loaded and must be recomputed
    /// from the LSM's pointers.
    pub fn stats_need_rebuild(&self) -> bool {
        self.stats_need_rebuild.load(Ordering::Acquire)
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
    ///
    /// The tail is fsynced before it is sealed. [`sync`](Self::sync) only ever
    /// fsyncs the *active* file, so a segment sealed with unsynced records would
    /// never be made durable by anything — and once the LSM entries pointing into
    /// it reach L0 their WAL entries are marked persisted (and later reclaimed),
    /// so a power loss would lose those values for good.
    fn roll_active_segment(&self, inner: &mut Inner) -> Result<()> {
        if inner.synced_offset < inner.active_offset {
            inner.active_file.sync_data()?;
        }
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
        inner.synced_offset = SEGMENT_HEADER_SIZE;
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
        inner.active_offset += record_len;
        if sync {
            file.sync_data()?;
            inner.synced_offset = inner.active_offset;
        }

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

    /// Account for a record no LSM entry points at any more (see
    /// `storage/lsm/tree/displaced.rs`). Costs **no I/O**: the pointer carries
    /// `value_len`. A segment GC has already deleted is not charged.
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

    /// Make every record appended so far durable. Sealed segments were fsynced
    /// when sealed, so only the active tail needs it.
    pub fn sync(&self) -> Result<()> {
        let (file, id, target) = {
            let inner = self.inner.read();
            (Arc::clone(&inner.active_file), inner.active_id, inner.active_offset)
        };
        file.sync_data()?;
        let mut inner = self.inner.write();
        // A roll in between already synced (and replaced) that tail.
        if inner.active_id == id && inner.synced_offset < target {
            inner.synced_offset = target;
        }
        Ok(())
    }

    /// Test hook for the power-loss model: the active segment's file and how
    /// many of its bytes are fsynced. Every sealed segment was fsynced as it
    /// was sealed.
    #[cfg(test)]
    pub(crate) fn power_loss_extent(&self) -> (PathBuf, u64) {
        let inner = self.inner.read();
        (Self::segment_path(&self.dir, self.bucket, inner.active_id), inner.synced_offset)
    }

    /// Bytes appended to this bucket that are not yet known to be on stable
    /// storage. Zero after [`sync`](Self::sync) with no appends since.
    #[cfg(test)]
    pub fn unsynced_bytes(&self) -> u64 {
        let inner = self.inner.read();
        inner.active_offset - inner.synced_offset
    }
}

#[cfg(test)]
mod tests;
