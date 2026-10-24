//! Write-Ahead Log (WAL) Module
//!
//! This module implements a write-ahead log for the MinnalDB.
//! It provides:
//! - Operation logging (UPSERT, DELETE), append-only: a frame never changes
//!   once written
//! - Sequential record writing with head/tail tracking
//! - WAL metadata management, including the retirement progress (the cut and
//!   each namespace's safe offset) that recovery and WAL GC read
//! - Atomic file swapping during GC

use crate::db::commit_rounds::CommitRounds;
use crc32fast::Hasher;
use crossbeam_epoch::{self as epoch, Atomic, Guard, Owned};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
#[cfg(test)]
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::ErrorKind;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum WalError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("WAL corrupted")]
    CorruptedLog,
    /// An fsync of the log failed, so the log refuses writes until the database
    /// is reopened (see [`Wal::commit`]).
    #[error("WAL stopped after a failed fsync: {0}")]
    Poisoned(String),
}

pub type Result<T> = std::result::Result<T, WalError>;

/// Default WAL segment size (64 MiB). Also the size every pre-marker ("legacy")
/// WAL was created at, so it is the fallback when no creation-time size is
/// recorded — see `Database::resolve_wal_segment_size`.
pub(crate) const DEFAULT_SEGMENT_SIZE: u64 = 64 * 1024 * 1024;

/// Floor for a WAL segment size. Matches the guard in
/// [`Wal::open_with_options_and_segment_size`].
pub(crate) const MIN_SEGMENT_SIZE: u64 = 256;
const WAL_METADATA_MAGIC: [u8; 4] = *b"WALM";

/// Bytes before each frame's body: its length (u32) and the CRC32 of the body
/// (u32), little-endian. A zero length marks the end of a segment.
pub(crate) const FRAME_HEADER_LEN: u64 = 8;

/// What [`Wal::read_frame`] found at a position.
enum Frame {
    /// No room for a header, or a zero length: the segment's records end here.
    SegmentEnd,
    /// A header whose body is short, fails its CRC or does not decode: a torn
    /// or corrupt record. Nothing at or after it is part of the log.
    Bad,
    /// A valid frame whose body (in the caller's buffer) is this long.
    Valid(u32),
}

/// Size `file` to `size` bytes up front (a sparse file of zeros past its data)
/// and make that durable. An `fdatasync` on a file that grows with each write
/// must also commit its new size; on a file already at its full size it does
/// not, which measured about 3× cheaper (design note §15). Past the last frame
/// the file reads as zeros, and a zero length already ends a segment.
fn preallocate(file: &File, size: u64) -> std::io::Result<()> {
    if file.metadata()?.len() < size {
        file.set_len(size)?;
        file.sync_all()?;
    }
    Ok(())
}

/// The frame bytes for `body`: header, then the body.
fn encode_frame(body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(FRAME_HEADER_LEN as usize + body.len());
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&crc32fast::hash(body).to_le_bytes());
    out.extend_from_slice(body);
    out
}
const WAL_METADATA_VERSION: u32 = 1;

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
/// as it did live (see [`Wal::recover_sequence`]).
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
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .map_err(|e| WalError::Serialization(format!("Failed to serialize WAL entry: {}", e)))
            .map(|buf| buf.to_vec())
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
/// `WalPersistObserver::on_memtable_written_ns` in `database.rs`).
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, serde::Serialize)]
pub struct WalMetadata {
    pub head: u64, // Offset where GC starts
    pub tail: u64, // Offset where new writes go
    /// Every entry below this is in an SSTable. The lowest safe offset among
    /// namespaces with unflushed writes, capped at the oldest in-flight write.
    pub cut: u64,
    /// Namespaces whose safe offset is above `cut`, sorted by namespace: each
    /// one's writes below its offset are in L0 files too.
    pub safe: Vec<(u32, u64)>,
    pub total_gc_runs: u64,         // Number of GC operations
    pub total_bytes_reclaimed: u64, // Total bytes freed by GC
    /// High-water mark for the global sequence number. Updated before any WAL
    /// segment is deleted during GC so that recovery never misses the max
    /// sequence even if the segment is gone.
    pub last_sequence: u64,
}

impl Default for WalMetadata {
    fn default() -> Self {
        Self::new()
    }
}

impl WalMetadata {
    pub fn new() -> Self {
        Self {
            head: 0,
            tail: 0,
            cut: 0,
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

    /// Bytes recovery would replay: from the cut to the tail.
    pub fn pending_bytes(&self) -> u64 {
        self.tail.saturating_sub(self.cut)
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
    /// Offset of the oldest byte still on disk; segments below it were reclaimed.
    pub head: u64,
    /// Where the next record goes.
    pub tail: u64,
    /// Every entry below this is in an SSTable; WAL GC reclaims segments
    /// wholly below it.
    pub cut: u64,
    /// Bytes recovery would replay: `tail - cut`.
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
    /// Highest write sequence the WAL has observed (GC high-water mark).
    pub last_sequence: u64,
    /// Every segment from the head to the tail, oldest first.
    pub segments: Vec<WalSegmentState>,
}

/// One namespace's saved progress, in [`WalStats`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct WalNamespaceProgress {
    pub namespace: String,
    pub safe_offset: u64,
}

/// One WAL segment, in [`WalStats`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct WalSegmentState {
    pub segment_id: u64,
    /// Byte range `[start, end)` the segment covers in the log.
    pub start: u64,
    pub end: u64,
    /// Wholly below the cut: nothing in it is replayed, and WAL GC may
    /// delete it (unless a field index still needs it).
    pub retired: bool,
    /// Bytes of this segment recovery would replay.
    pub pending_bytes: u64,
}

#[derive(Debug)]
struct WalHandle {
    file: Arc<File>,
}

impl WalHandle {
    fn new(file: Arc<File>) -> Self {
        Self { file }
    }
}

/// Write-Ahead Log Manager
///
/// Handles WAL file operations with head/tail tracking
pub struct Wal {
    // WAL file handle stored behind an epoch/RCU atomic for lock-free swaps
    handle: Atomic<WalHandle>,

    // Path to the WAL file
    path: PathBuf,

    segment_size: u64,
    current_segment_id: AtomicU64,

    /// Group commit: the durable tail, commit rounds and poisoning, shared in
    /// shape with the vector index WAL (`db/commit_rounds.rs`).
    rounds: CommitRounds,
    #[cfg(test)]
    fail_next_sync: std::sync::atomic::AtomicBool,
    /// Test hook for the power-loss model: per segment, the most bytes an fsync
    /// issued by this process covered, and the first segment this process
    /// could write (every earlier one survived a reopen, so it is durable).
    #[cfg(test)]
    synced: parking_lot::Mutex<BTreeMap<u64, u64>>,
    /// Test hook for the power-loss model: per segment, how far frames have
    /// been written (segments are preallocated, so the file length says
    /// nothing about it).
    #[cfg(test)]
    written: parking_lot::Mutex<BTreeMap<u64, u64>>,
    #[cfg(test)]
    first_open_segment: AtomicU64,
}

/// Reclaims the current [`WalHandle`], and with it the open file descriptor.
///
/// **A crossbeam `Atomic` does not drop its pointee.** Reclamation there is
/// always explicit — `rotate_to_segment` and `swap_file` hand each *replaced*
/// handle to `Guard::defer_destroy`, but the handle still installed when the
/// `Wal` itself goes away has no such owner, so without this impl every dropped
/// `Wal` leaked its boxed handle and leaked the file open with it. Measured
/// before this existed: a benchmark opening one `Wal` per iteration climbed past
/// 450,000 open descriptors in ten seconds and died on `EMFILE`.
///
/// The reclamation here is **immediate**, not deferred, and that distinction is
/// the point. `defer_destroy` only runs once the global epoch advances, so a
/// workload that opens and drops WALs in a tight loop would pile the same
/// descriptors up as *pending* garbage and exhaust the fd limit exactly as
/// before. Dropping in place closes the file when the `Wal` dies.
///
/// This is sound because `Drop::drop` takes `&mut self`: no other thread can
/// hold a `Shared` or `&WalHandle` borrowed from this `Wal` (both borrow
/// `&self`), and a `Wal` behind an `Arc` reaches drop only after the last
/// reference is gone. This mirrors the `Drop` pattern in `Atomic`'s own docs.
impl Drop for Wal {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, Atomic::null());
        // SAFETY: we hold `&mut self`, so this pointer is unreachable from any
        // other thread and is dropped exactly once. The pointer can be null if
        // a previous drop already took it, hence `try_into_owned`.
        unsafe {
            if let Some(owned) = handle.try_into_owned() {
                drop(owned);
            }
        }
    }
}

impl Wal {
    fn segment_id_for(&self, offset: u64) -> u64 {
        offset / self.segment_size
    }

    fn segment_offset_for(&self, offset: u64) -> u64 {
        offset % self.segment_size
    }

    fn segment_path_for(&self, segment_id: u64) -> PathBuf {
        if segment_id == 0 {
            return self.path.clone();
        }
        let base = self.path.to_string_lossy();
        PathBuf::from(format!("{}.seg{:06}", base, segment_id))
    }

    fn open_segment_file(&self, segment_id: u64, create: bool) -> Result<File> {
        let path = self.segment_path_for(segment_id);
        let is_new = create && !path.exists();
        let mut options = OpenOptions::new();
        options.read(true).write(true);
        if create {
            options.create(true);
        }
        let file = options.open(&path)?;
        if is_new {
            preallocate(&file, self.segment_size)?;
        }
        // A new segment's *name* must be durable before any write into it is
        // acknowledged. Every append fsyncs the file's contents, but a file's
        // directory entry is only durable once the directory is fsynced — until
        // then a power loss can drop the whole segment, and every write
        // acknowledged into it since the rotation with it. One fsync per segment.
        if is_new && let Some(dir) = path.parent() {
            File::open(dir)?.sync_all()?;
        }
        Ok(file)
    }

    fn rotate_to_segment(&self, segment_id: u64) -> Result<()> {
        let file = self.open_segment_file(segment_id, true)?;
        let new_handle = WalHandle::new(Arc::new(file));
        let guard = epoch::pin();
        let old = self.handle.swap(Owned::new(new_handle), Ordering::AcqRel, &guard);
        self.current_segment_id.store(segment_id, Ordering::Release);
        unsafe {
            guard.defer_destroy(old);
        }
        Ok(())
    }

    fn write_all_at(&self, file: &File, mut buf: &[u8], mut offset: u64) -> std::io::Result<()> {
        while !buf.is_empty() {
            let written = file.write_at(buf, offset)?;
            if written == 0 {
                return Err(std::io::Error::new(ErrorKind::WriteZero, "failed to write WAL"));
            }
            offset += written as u64;
            buf = &buf[written..];
        }
        Ok(())
    }

    fn read_exact_at(&self, file: &File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
        while !buf.is_empty() {
            let read = file.read_at(buf, offset)?;
            if read == 0 {
                return Err(std::io::Error::new(ErrorKind::UnexpectedEof, "failed to fill whole buffer"));
            }
            offset += read as u64;
            let tmp = buf;
            buf = &mut tmp[read..];
        }
        Ok(())
    }

    fn load_handle<'g>(&'g self, guard: &'g Guard) -> Result<&'g WalHandle> {
        let shared = self.handle.load(Ordering::Acquire, guard);
        unsafe { shared.as_ref().ok_or(WalError::CorruptedLog) }
    }

    /// Open or create a WAL with optional memory-mapping
    ///
    /// # Arguments
    /// * `path` - Path to the WAL file
    /// * `use_mmap` - If true, use memory-mapped I/O on Unix systems
    pub fn open_with_options<P: AsRef<Path>>(path: P, use_mmap: bool) -> Result<Self> {
        Self::open_with_options_and_segment_size(path, use_mmap, DEFAULT_SEGMENT_SIZE)
    }

    pub fn open_with_options_and_segment_size<P: AsRef<Path>>(path: P, _use_mmap: bool, segment_size: u64) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let is_new = !path.exists();
        let file = OpenOptions::new().create(true).truncate(false).read(true).write(true).open(&path)?;
        if is_new {
            preallocate(&file, segment_size.max(MIN_SEGMENT_SIZE))?;
        }
        // As in `open_segment_file`: a newly created WAL file's name must be
        // durable before the first write into it is acknowledged.
        if is_new && let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            File::open(dir)?.sync_all()?;
        }

        let handle = WalHandle::new(Arc::new(file));

        Ok(Self {
            handle: Atomic::new(handle),
            path,
            segment_size: segment_size.max(MIN_SEGMENT_SIZE),
            current_segment_id: AtomicU64::new(0),
            rounds: CommitRounds::new(0),
            #[cfg(test)]
            fail_next_sync: std::sync::atomic::AtomicBool::new(false),
            #[cfg(test)]
            synced: parking_lot::Mutex::new(BTreeMap::new()),
            #[cfg(test)]
            written: parking_lot::Mutex::new(BTreeMap::new()),
            #[cfg(test)]
            first_open_segment: AtomicU64::new(0),
        })
    }

    /// Open or create a WAL with default settings
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_options(path, false)
    }

    /// Append a WAL entry at the current tail position
    /// Returns the WalPointer for the written entry
    ///
    /// `sync`: when `true`, calls `fdatasync` before returning so the entry is
    /// durable on crash.  Pass `false` for test/batch scenarios where the caller
    /// manages durability separately.
    pub fn append_entry(&self, entry: &WalEntry, tail: &mut u64, sync: bool) -> Result<WalPointer> {
        let mut segment_id = self.segment_id_for(*tail);
        let mut segment_offset = self.segment_offset_for(*tail);
        let mut file_arc = self.current_file_arc()?;
        let current_segment = self.current_segment_id.load(Ordering::Acquire);
        if current_segment != segment_id {
            // The tail is already past the active segment: the last frame filled
            // it exactly. Leaving a segment makes it durable, whichever way it is
            // left (commit rounds fsync only the active segment); found by the
            // power-loss stress, which lost writes acknowledged into a segment
            // left this way. At open the active handle is a segment nothing has
            // written in this process, so it is skipped.
            if current_segment + 1 == segment_id && self.segment_offset_for(*tail) == 0 {
                self.sync_or_poison(&file_arc, current_segment, "sealing a full WAL segment")?;
                self.set_durable_tail(*tail);
            }
            self.rotate_to_segment(segment_id)?;
            file_arc = self.current_file_arc()?;
        }

        // Serialize the entry
        let entry_bytes = entry.to_bytes()?;
        let size = entry_bytes.len() as u32;
        let entry_len = FRAME_HEADER_LEN + size as u64;
        let remaining = self.segment_size.saturating_sub(segment_offset);
        if entry_len > remaining {
            if remaining >= 4 {
                self.write_all_at(&file_arc, &0u32.to_le_bytes(), segment_offset)?;
                #[cfg(test)]
                self.note_written(segment_id, segment_offset + 4);
            }
            // Commit rounds fsync only the active segment, so a segment is made
            // durable as it is left: then everything below the active segment
            // is durable, and the durable tail is one position.
            self.sync_or_poison(&file_arc, segment_id, "sealing a WAL segment")?;
            segment_id += 1;
            *tail = segment_id * self.segment_size;
            // Every segment below this one is now durable.
            self.set_durable_tail(*tail);
            segment_offset = 0;
            self.rotate_to_segment(segment_id)?;
            file_arc = self.current_file_arc()?;
        }

        // One write: header (length + CRC) and body.
        self.write_all_at(&file_arc, &encode_frame(&entry_bytes), segment_offset)?;
        #[cfg(test)]
        self.note_written(segment_id, segment_offset + entry_len);
        if sync {
            self.sync_or_poison(&file_arc, segment_id, "syncing a WAL append")?;
        }

        let pointer = WalPointer::new(*tail, size);
        *tail += entry_len;

        Ok(pointer)
    }

    /// The active segment's id and file: the one a commit round fsyncs.
    pub(crate) fn active_file(&self) -> Result<(u64, Arc<File>)> {
        let guard = epoch::pin();
        let handle = self.load_handle(&guard)?;
        Ok((self.current_segment_id.load(Ordering::Acquire), Arc::clone(&handle.file)))
    }

    /// Every byte below this offset is fsynced.
    pub fn durable_tail(&self) -> u64 {
        self.rounds.durable()
    }

    /// Record that everything below `tail` is durable: at open, after
    /// recovery has found and truncated the log's end, and as a segment is
    /// sealed.
    pub(crate) fn set_durable_tail(&self, tail: u64) {
        self.rounds.advance(tail);
    }

    /// At open: what is on disk now survived, so it is durable (for the
    /// power-loss model too).
    pub(crate) fn opened_at(&self, tail: u64) {
        self.set_durable_tail(tail);
        #[cfg(test)]
        {
            let segment_id = self.segment_id_for(tail);
            self.first_open_segment.store(segment_id, Ordering::SeqCst);
            self.synced.lock().insert(segment_id, self.segment_offset_for(tail));
            self.written.lock().insert(segment_id, self.segment_offset_for(tail));
        }
    }

    /// `Err(Poisoned)` once an fsync of the log has failed.
    pub fn check_poisoned(&self) -> Result<()> {
        self.rounds.check_poisoned().map_err(WalError::Poisoned)
    }

    /// Stop the log: every waiting and later commit fails until reopen.
    pub(crate) fn poison(&self, why: String) -> WalError {
        log::error!("[WAL] stopping the main WAL; writes are refused until the database is reopened: {why}");
        WalError::Poisoned(self.rounds.poison(why))
    }

    /// Test hook: while `on`, every writer reaching [`commit`](Self::commit)
    /// waits there, its entry appended but not yet fsynced — the state a power
    /// loss would catch it in. Turning it off releases them.
    #[cfg(test)]
    pub(crate) fn freeze_commits(&self, on: bool) {
        self.rounds.freeze(on);
    }

    /// Test hook: how many writers are waiting in a frozen commit.
    #[cfg(test)]
    pub(crate) fn frozen_writers(&self) -> usize {
        self.rounds.frozen_callers()
    }

    /// Test hook for the power-loss model: for every segment this process
    /// could have written, its file and how many of its bytes an fsync actually
    /// covered (0 if none did). Taken from the fsyncs issued, not from the
    /// log's own bookkeeping, so a missing fsync shows up as lost bytes.
    #[cfg(test)]
    pub(crate) fn power_loss_extents(&self) -> Vec<(PathBuf, u64)> {
        let first = self.first_open_segment.load(Ordering::SeqCst);
        let last = self.current_segment_id.load(Ordering::Acquire).max(first);
        let synced = self.synced.lock();
        (first..=last)
            .map(|id| (self.segment_path_for(id), synced.get(&id).copied().unwrap_or(0)))
            .collect()
    }

    #[cfg(test)]
    fn note_written(&self, segment_id: u64, end: u64) {
        let mut written = self.written.lock();
        let entry = written.entry(segment_id).or_insert(0);
        *entry = (*entry).max(end);
    }

    /// Test hook: whether an fsync issued by this process (or the reopen)
    /// covered every byte below `end`.
    #[cfg(test)]
    pub(crate) fn covered(&self, end: u64) -> bool {
        if end == 0 {
            return true;
        }
        let seg = self.segment_id_for(end - 1);
        if seg < self.first_open_segment.load(Ordering::SeqCst) {
            return true;
        }
        let need = end - seg * self.segment_size;
        self.synced.lock().get(&seg).is_some_and(|&c| c >= need)
    }

    /// Test hook: make the next fsync of the log fail.
    #[cfg(test)]
    pub(crate) fn fail_next_sync(&self) {
        self.fail_next_sync.store(true, Ordering::SeqCst);
    }

    /// fsync `file`; on failure, poison the log and return the error. After a
    /// failed fsync the kernel may have dropped the dirty pages, and a later
    /// fsync that succeeds says nothing about them, so nothing appended before
    /// it may be acknowledged: PostgreSQL's rule, and the vector index WAL's.
    fn sync_or_poison(&self, file: &File, segment_id: u64, what: &str) -> Result<()> {
        self.sync_file(file, segment_id, what).map_err(|why| self.poison(why))
    }

    /// fsync `file` (segment `segment_id`); the error says what failed. The
    /// caller poisons the log (directly, or through a commit round).
    fn sync_file(&self, file: &File, segment_id: u64, what: &str) -> std::result::Result<(), String> {
        #[cfg(test)]
        if self.fail_next_sync.swap(false, Ordering::SeqCst) {
            return Err(format!("{what}: injected fsync failure"));
        }
        // What this fsync covers, for the power-loss model: how far frames were
        // written before it starts (bytes written later may or may not be
        // covered).
        #[cfg(test)]
        let covered = self.written.lock().get(&segment_id).copied().unwrap_or(0);
        file.sync_data().map_err(|e| format!("{what}: {e}"))?;
        #[cfg(test)]
        {
            let mut synced = self.synced.lock();
            let entry = synced.entry(segment_id).or_insert(0);
            *entry = (*entry).max(covered);
        }
        let _ = segment_id;
        Ok(())
    }

    /// Wait until every byte below `end` is durable: **group commit**.
    ///
    /// The caller has appended its frame (ending at `end`) under the WAL lock
    /// and released it. If no commit round is running, it becomes the round's
    /// leader: it takes `snapshot()` — the written tail and the active segment's
    /// file, read together under the WAL lock so the tail never covers a
    /// half-written frame — fsyncs that file outside every lock, publishes the
    /// tail as durable and wakes everyone. Otherwise it waits for the running
    /// round; a writer whose frame came after that round started waits for, or
    /// leads, the next one. So one fsync covers every write appended while the
    /// previous one ran, and no write returns before an fsync that began after
    /// its bytes were written.
    ///
    /// `delay` (C2): a leader with others already waiting first sleeps this
    /// long, so more writes join its round. A lone writer never waits.
    ///
    /// Returns `true` if this caller led a round (issued an fsync). A failed
    /// fsync poisons the log: every writer in that round, and every later one,
    /// gets [`WalError::Poisoned`].
    pub fn commit(&self, end: u64, delay: std::time::Duration, snapshot: impl Fn() -> Result<(u64, u64, Arc<File>)>) -> Result<bool> {
        self.rounds
            .commit(end, delay, || {
                let (target, segment_id, file) = snapshot().map_err(|e| format!("a commit round's snapshot: {e}"))?;
                self.sync_file(&file, segment_id, "a commit round").inspect_err(|why| {
                    log::error!("[WAL] stopping the main WAL; writes are refused until the database is reopened: {why}");
                })?;
                Ok(target)
            })
            .map_err(WalError::Poisoned)
    }

    /// Read a WAL entry from the given pointer
    #[allow(dead_code)]
    pub fn read_entry(&self, pointer: WalPointer) -> Result<WalEntry> {
        let guard = epoch::pin();
        let handle = self.load_handle(&guard)?;

        let segment_id = self.segment_id_for(pointer.offset);
        let segment_offset = self.segment_offset_for(pointer.offset);
        if segment_id == self.current_segment_id.load(Ordering::Acquire) {
            return self.read_entry_from_file(&handle.file, segment_offset, pointer.size);
        }
        let file = self.open_segment_file(segment_id, false)?;
        self.read_entry_from_file(&file, segment_offset, pointer.size)
    }

    /// Segment ids covering `[head, tail)` whose files are **absent**.
    ///
    /// A caller that scans this range gets no error for a missing segment —
    /// [`scan_entries`](Self::scan_entries) deliberately treats one as a hole and
    /// skips it, because aborting there would crash recovery. That is correct
    /// for the LSM recovery path (those entries are in SSTables by definition,
    /// which is why GC reclaimed them) but it means any
    /// *other* consumer replaying this range is silently handed an incomplete
    /// result. This reports the holes so such a consumer can say so.
    ///
    /// Presence is checked against the filesystem rather than inferred from
    /// `WalMetadata::head`, so a file lost any other way is reported too.
    pub fn missing_segments(&self, head: u64, tail: u64) -> Vec<u64> {
        if tail <= head {
            return Vec::new();
        }
        let first = self.segment_id_for(head);
        let last = self.segment_id_for(tail - 1);
        (first..=last).filter(|&id| !self.segment_path_for(id).exists()).collect()
    }

    /// Walk the WAL frames in `[head, tail)`, handing each frame's **global
    /// offset** and body bytes to `extract`, and collecting what it returns.
    ///
    /// The single frame-parsing loop every WAL scan goes through. Keeping it in
    /// one place matters because the loop encodes three distinct recovery
    /// judgements that must not drift apart between callers:
    ///
    /// - a **missing segment** is a hole to skip, not an error. WAL GC only
    ///   reclaims segments below the cut, which nothing replays, and aborting
    ///   on one would crash recovery;
    /// - **no room for a header, or a zero length**, is the end of a segment's
    ///   records: advance to the next segment and keep going;
    /// - a frame whose body is **short, fails its CRC, or will not decode** ends
    ///   the scan (`extract` returns `None` for the last).
    ///
    /// The body buffer is reused across frames, so a scan allocates once rather
    /// than per entry.
    fn walk_frames<T>(&self, head: u64, tail: u64, mut extract: impl FnMut(u64, &[u8]) -> Option<T>) -> Result<Vec<T>> {
        let mut collected = Vec::new();
        let mut offset = head;
        let mut entry_bytes = Vec::new();

        // Open segments lazily as the scan crosses into them, so a missing one
        // can be skipped rather than aborting the whole scan. `u64::MAX` forces
        // an open on the first iteration.
        let mut current_file: Option<File> = None;
        let mut current_segment_id = u64::MAX;

        while offset < tail {
            let segment_id = self.segment_id_for(offset);
            if segment_id != current_segment_id {
                current_segment_id = segment_id;
                current_file = match self.open_segment_file(segment_id, false) {
                    Ok(file) => Some(file),
                    // A missing segment is a "hole": WAL GC deleted a retired
                    // segment (see `garbage_collect_wal`). Skip it and keep
                    // scanning; aborting here would crash recovery.
                    Err(WalError::Io(ref e)) if e.kind() == std::io::ErrorKind::NotFound => None,
                    Err(e) => return Err(e),
                };
            }

            let Some(current_file) = current_file.as_ref() else {
                offset = (segment_id + 1) * self.segment_size;
                continue;
            };

            let size = match self.read_frame(current_file, self.segment_offset_for(offset), &mut entry_bytes) {
                Frame::SegmentEnd => {
                    offset = (segment_id + 1) * self.segment_size;
                    continue;
                }
                Frame::Bad => break,
                Frame::Valid(size) => size,
            };

            match extract(offset, &entry_bytes[..size as usize]) {
                Some(value) => collected.push(value),
                None => break,
            }

            offset += FRAME_HEADER_LEN + size as u64;
        }

        Ok(collected)
    }

    /// Scan all entries from head to tail.
    pub fn scan_entries(&self, head: u64, tail: u64) -> Result<Vec<(WalPointer, WalEntry)>> {
        self.walk_frames(head, tail, |offset, body| {
            let entry = WalEntry::from_bytes(body).ok()?;
            Some((WalPointer::new(offset, body.len() as u32), entry))
        })
    }

    /// Scan `[head, tail)` for `(namespace_id, key)` pairs, **without
    /// materialising any values**.
    ///
    /// The key-only counterpart of [`scan_entries`](Self::scan_entries), for
    /// callers that need to know *which documents* a WAL range covers rather
    /// than what was written to them. `scan_entries` returns whole `WalEntry`
    /// values, so using it for that copies every document body in the range into
    /// memory only to discard it — a range spanning a few segments can hold tens
    /// of thousands of documents.
    ///
    /// Pairs are returned in WAL order, **with duplicates**: a key written five
    /// times appears five times. Callers wanting a work list want
    /// [`scan_segment_keys`](Self::scan_segment_keys), which deduplicates.
    pub fn scan_keys_in_entries(&self, head: u64, tail: u64) -> Result<Vec<(u32, Vec<u8>)>> {
        self.walk_frames(head, tail, |_, body| WalEntry::peek_namespace_and_key(body).ok())
    }

    /// Collect the distinct `(namespace_id, key)` pairs one WAL segment holds,
    /// **without materialising any values**.
    ///
    /// This is the harvest that makes row-scoped index repair possible. When the
    /// backstop forces WAL GC to reclaim a segment an active field index still
    /// needs (see `ThresholdConfig::max_pinned_wal_segments`), the keys of the
    /// affected documents exist *only* in that segment — a gap record holding a
    /// WAL range and segment ids cannot name them, and once the file is unlinked
    /// they are unrecoverable. So they are harvested here, immediately before the
    /// unlink, and recorded as a repair worklist.
    ///
    /// Built on [`scan_keys_in_entries`](Self::scan_keys_in_entries), so it
    /// shares the one frame-parsing loop and never materialises a value.
    ///
    /// Results are deduplicated (a key written five times in the segment needs
    /// repairing once) and returned in first-seen order. Both upserts and deletes
    /// are included — a lost *delete* leaves a stale row in the index, which is
    /// just as wrong as a missing one.
    ///
    /// A missing segment file yields an empty result rather than an error: GC may
    /// have already reclaimed it, and failing here would abort the whole pass.
    pub fn scan_segment_keys(&self, segment_id: u64) -> Result<Vec<(u32, Vec<u8>)>> {
        let start = segment_id * self.segment_size;
        let pairs = self.scan_keys_in_entries(start, start + self.segment_size)?;

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

    /// Get the current file reference
    #[allow(dead_code)]
    pub fn get_file(&self) -> Arc<File> {
        let guard = epoch::pin();
        let handle = self.load_handle(&guard).expect("WAL handle missing");
        Arc::clone(&handle.file)
    }

    /// Atomically swap the file (used during GC)
    #[allow(dead_code)]
    pub fn swap_file(&self, new_file: Arc<File>) -> Result<Arc<File>> {
        let new_handle = WalHandle::new(new_file);
        let guard = epoch::pin();
        let old = self.handle.swap(Owned::new(new_handle), Ordering::AcqRel, &guard);
        let old_handle = unsafe { old.as_ref().ok_or(WalError::CorruptedLog)? };
        let old_file = Arc::clone(&old_handle.file);
        // SAFETY: old is a valid Shared pointer from the same atomic and guard.
        unsafe {
            guard.defer_destroy(old);
        }
        Ok(old_file)
    }

    /// Flush and sync the WAL to disk
    pub fn sync(&self) -> Result<()> {
        let guard = epoch::pin();
        let handle = self.load_handle(&guard)?;
        let mut file = handle.file.try_clone()?;
        file.flush()?;
        file.sync_all()?;

        Ok(())
    }

    /// Get the path to the WAL file
    #[allow(dead_code)]
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn segment_size(&self) -> u64 {
        self.segment_size
    }

    /// Recover the next sequence number to use after a restart.
    ///
    /// Scans all WAL entries from `head` to `tail` to find the highest
    /// `sequence` value, then returns `max(hint, max_in_wal) + 1`.
    ///
    /// `hint` should be `WalMetadata::last_sequence`. It acts as a floor that
    /// protects against the case where GC has already deleted the segments that
    /// held the highest-sequenced entries (GC updates `last_sequence` in
    /// metadata **before** deleting any segment, so the hint is always ≥ the
    /// maximum sequence in surviving segments).
    pub fn recover_sequence(&self, head: u64, tail: u64, hint: u64) -> u64 {
        let max_in_wal = if tail > head {
            self.scan_entries(head, tail)
                .unwrap_or_default()
                .into_iter()
                .map(|(_, e)| e.sequence)
                .max()
                .unwrap_or(0)
        } else {
            0
        };
        hint.max(max_in_wal).saturating_add(1)
    }

    /// Reconstruct the true tail offset by scanning forward from `start` (a
    /// known entry-aligned offset, typically the persisted `WalMetadata::tail`)
    /// following the self-describing entry framing until the first position that
    /// is not a complete, valid entry: EOF, a torn final append, a corrupt
    /// entry, or segment padding with no successor segment.
    ///
    /// WAL entries are fsynced on every write, but `WalMetadata` (which records
    /// the tail) is only flushed periodically — so after a crash the persisted
    /// tail can lag the durable end of the log. This recovers the real end so
    /// recovery does not silently ignore fsynced entries. Passing `start = 0`
    /// reconstructs the tail wholesale when the metadata was lost or corrupt.
    pub fn recover_tail(&self, start: u64) -> u64 {
        let mut offset = start;
        loop {
            let segment_id = self.segment_id_for(offset);
            let file = match self.open_segment_file(segment_id, false) {
                Ok(f) => f,
                Err(_) => break, // no such segment file → end of log
            };
            let segment_offset = self.segment_offset_for(offset);

            // No room for a header, or a zero length, ends this segment's
            // records: an explicit padding marker, or a remainder too small for a
            // frame left when an entry was rotated to the next segment. The log
            // continues in the next segment *iff* that segment file exists —
            // otherwise we are at the true end. A frame whose body is short,
            // fails its CRC or does not decode is a torn or corrupt tail.
            let mut body = Vec::new();
            let size = match self.read_frame(&file, segment_offset, &mut body) {
                Frame::SegmentEnd => {
                    if self.open_segment_file(segment_id + 1, false).is_ok() {
                        offset = (segment_id + 1) * self.segment_size;
                        continue;
                    }
                    break; // no successor segment → true end of log
                }
                Frame::Bad => break,
                Frame::Valid(size) => size,
            };
            if WalEntry::from_bytes(&body).is_err() {
                break; // body present but not a valid entry → corrupt tail
            }
            offset += FRAME_HEADER_LEN + size as u64;
        }
        offset
    }

    pub fn segment_id_for_offset(&self, offset: u64) -> u64 {
        self.segment_id_for(offset)
    }

    pub fn delete_segment_file(&self, segment_id: u64) -> Result<()> {
        let path = self.segment_path_for(segment_id);
        match std::fs::remove_file(&path) {
            Ok(_) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(WalError::Io(e)),
        }
    }

    fn read_entry_from_file(&self, file: &File, segment_offset: u64, size: u32) -> Result<WalEntry> {
        let mut body = Vec::new();
        match self.read_frame(file, segment_offset, &mut body) {
            Frame::Valid(stored) if stored == size => WalEntry::from_bytes(&body),
            _ => Err(WalError::CorruptedLog),
        }
    }

    /// Read the frame at `segment_offset` of `file` into `body`. A frame is
    /// valid only if its whole body is there, fits in the segment, and matches
    /// its CRC; decoding the body is the caller's step.
    fn read_frame(&self, file: &File, segment_offset: u64, body: &mut Vec<u8>) -> Frame {
        let mut header = [0u8; FRAME_HEADER_LEN as usize];
        if self.read_exact_at(file, &mut header, segment_offset).is_err() {
            return Frame::SegmentEnd;
        }
        let size = u32::from_le_bytes(header[0..4].try_into().unwrap());
        if size == 0 {
            return Frame::SegmentEnd;
        }
        if segment_offset + FRAME_HEADER_LEN + size as u64 > self.segment_size {
            return Frame::Bad;
        }
        let crc = u32::from_le_bytes(header[4..8].try_into().unwrap());
        body.resize(size as usize, 0);
        if self.read_exact_at(file, body, segment_offset + FRAME_HEADER_LEN).is_err() || crc32fast::hash(body) != crc {
            return Frame::Bad;
        }
        Frame::Valid(size)
    }

    /// Cut the log off at `tail`, the end of its valid frames found by
    /// [`recover_tail`](Self::recover_tail): zero the segment holding it from
    /// there on (keeping it at its full, preallocated size) and remove any
    /// later segment files, then make that durable. Called at open,
    /// before anything is appended.
    ///
    /// Without it, a later, shorter record written over a torn region leaves
    /// older valid-looking frames behind it, and a second recovery would replay
    /// them; a CRC does not catch those, since they were valid when written.
    pub fn truncate_after(&self, tail: u64) -> Result<()> {
        let segment_id = self.segment_id_for(tail);
        let keep = self.segment_offset_for(tail);
        let path = self.segment_path_for(segment_id);
        let mut changed = false;
        if let Ok(file) = OpenOptions::new().write(true).open(&path) {
            // Zero everything after the tail (cut, then size back up), so the
            // active segment stays preallocated.
            file.set_len(keep)?;
            file.set_len(self.segment_size)?;
            file.sync_all()?;
            changed = true;
        }
        let mut next = segment_id + 1;
        while self.segment_path_for(next).exists() {
            self.delete_segment_file(next)?;
            next += 1;
            changed = true;
        }
        if changed && let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            File::open(dir)?.sync_all()?;
        }
        Ok(())
    }

    fn current_file_arc(&self) -> Result<Arc<File>> {
        let guard = epoch::pin();
        let handle = self.load_handle(&guard)?;
        Ok(Arc::clone(&handle.file))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom};
    use tempfile::TempDir;

    /// Open descriptors this process holds on files under `dir`.
    ///
    /// Deliberately scoped to one directory rather than counting all of
    /// `/proc/self/fd`: cargo runs the tests in this module concurrently in a
    /// single process, so a process-wide count picks up whatever the other
    /// tests happen to have open and is useless as an assertion. Linux-only
    /// because it resolves each descriptor to a path through `/proc/self/fd`;
    /// macOS has no readlink-able equivalent (it needs `fcntl(F_GETPATH)`), and
    /// a leak this coarse does not need per-platform coverage to be caught.
    #[cfg(target_os = "linux")]
    fn open_fds_under(dir: &std::path::Path) -> usize {
        std::fs::read_dir("/proc/self/fd")
            .expect("/proc/self/fd should be readable")
            .filter_map(|e| std::fs::read_link(e.ok()?.path()).ok())
            .filter(|target| target.starts_with(dir))
            .count()
    }

    /// A dropped `Wal` must close its file.
    ///
    /// Regression test for a descriptor leak: `Wal::handle` is a
    /// `crossbeam_epoch::Atomic`, which does **not** drop its pointee, so before
    /// `impl Drop for Wal` existed every dropped `Wal` leaked the open file.
    /// Counting descriptors rather than waiting for `EMFILE` keeps this
    /// independent of the host's `ulimit -n` — on a machine with a 524k limit
    /// the leak needs half a million iterations to surface as a failure.
    #[cfg(target_os = "linux")]
    #[test]
    fn dropping_a_wal_closes_its_file_descriptor() -> Result<()> {
        let temp = TempDir::new()?;
        let dir = temp.path().canonicalize()?;

        const CYCLES: usize = 64;
        for i in 0..CYCLES {
            let wal = Wal::open(dir.join(format!("wal_{i}.log")))?;
            wal.append_entry(&WalEntry::new_upsert(b"k".to_vec(), b"v".to_vec()), &mut 0, false)?;
            drop(wal);
        }

        let still_open = open_fds_under(&dir);
        assert_eq!(
            still_open, 0,
            "dropping a Wal leaked its file descriptor: {still_open} of {CYCLES} WAL files are still open after being dropped"
        );
        Ok(())
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
    fn test_wal_metadata_serialization() -> Result<()> {
        let metadata = WalMetadata {
            head: 0,
            tail: 1024,
            cut: 300,
            safe: vec![(2, 500), (7, 900)],
            total_gc_runs: 2,
            total_bytes_reclaimed: 256,
            last_sequence: 0,
        };

        let bytes = metadata.to_bytes()?;
        let restored = WalMetadata::from_bytes(&bytes)?;

        assert_eq!(metadata.head, restored.head);
        assert_eq!(metadata.tail, restored.tail);
        assert_eq!(metadata.cut, restored.cut);
        assert_eq!(metadata.safe, restored.safe);
        Ok(())
    }

    #[test]
    fn test_wal_metadata_file_encoding() -> Result<()> {
        let metadata = WalMetadata {
            head: 0,
            tail: 1024,
            cut: 300,
            safe: vec![(2, 500), (7, 900)],
            total_gc_runs: 2,
            total_bytes_reclaimed: 256,
            last_sequence: 0,
        };

        let file_bytes = metadata.to_file_bytes()?;
        let restored = WalMetadata::from_file_bytes(&file_bytes)?;

        assert_eq!(metadata.head, restored.head);
        assert_eq!(metadata.tail, restored.tail);
        assert_eq!(metadata.cut, restored.cut);
        assert_eq!(metadata.safe, restored.safe);
        Ok(())
    }

    #[test]
    fn test_wal_append_and_read() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let wal = Wal::open(temp_dir.path().join("test.wal"))?;

        let mut tail = 0u64;
        let entry = WalEntry::new_upsert(b"key1".to_vec(), b"value1".to_vec());

        let pointer = wal.append_entry(&entry, &mut tail, false)?;
        let read_entry = wal.read_entry(pointer)?;

        assert_eq!(entry.key, read_entry.key);
        assert_eq!(entry.value, read_entry.value);
        Ok(())
    }

    #[test]
    fn test_wal_multiple_entries() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let wal = Wal::open(temp_dir.path().join("test.wal"))?;

        let mut tail = 0u64;
        let entries = vec![
            WalEntry::new_upsert(b"key1".to_vec(), b"value1".to_vec()),
            WalEntry::new_upsert(b"key2".to_vec(), b"value2".to_vec()),
            WalEntry::new_delete(b"key1".to_vec()),
        ];

        let mut pointers = Vec::new();
        for entry in &entries {
            let pointer = wal.append_entry(entry, &mut tail, false)?;
            pointers.push(pointer);
        }

        // Verify all entries
        for (i, pointer) in pointers.iter().enumerate() {
            let read_entry = wal.read_entry(*pointer)?;
            assert_eq!(entries[i].key, read_entry.key);
            assert_eq!(entries[i].value, read_entry.value);
        }

        Ok(())
    }

    #[test]
    fn test_wal_scan_entries() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let wal = Wal::open(temp_dir.path().join("test.wal"))?;

        let mut tail = 0u64;
        let entries = vec![
            WalEntry::new_upsert(b"key1".to_vec(), b"value1".to_vec()),
            WalEntry::new_upsert(b"key2".to_vec(), b"value2".to_vec()),
            WalEntry::new_delete(b"key1".to_vec()),
        ];

        for entry in &entries {
            wal.append_entry(entry, &mut tail, false)?;
        }

        // Scan all entries
        let scanned = wal.scan_entries(0, tail)?;
        assert_eq!(scanned.len(), 3);

        for (i, (_pointer, entry)) in scanned.iter().enumerate() {
            assert_eq!(entries[i].key, entry.key);
            assert_eq!(entries[i].value, entry.value);
        }

        Ok(())
    }

    #[test]
    fn test_wal_swap_file_concurrent_reads() -> Result<()> {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::thread;
        use std::time::Duration;

        fn write_entries(file: &File, entries: &[WalEntry]) -> Result<Vec<WalPointer>> {
            let mut pointers = Vec::with_capacity(entries.len());
            let mut offset = 0u64;
            let mut file = file.try_clone()?;
            file.set_len(0)?;
            file.sync_all()?;

            for entry in entries {
                let entry_bytes = entry.to_bytes()?;
                let size = entry_bytes.len() as u32;
                file.seek(SeekFrom::Start(offset))?;
                file.write_all(&encode_frame(&entry_bytes))?;
                pointers.push(WalPointer::new(offset, size));
                offset += FRAME_HEADER_LEN + size as u64;
            }

            file.sync_all()?;
            Ok(pointers)
        }

        let temp_dir = TempDir::new()?;
        let wal_path_a = temp_dir.path().join("wal_a.log");
        let wal_path_b = temp_dir.path().join("wal_b.log");

        let mut entries = Vec::new();
        for i in 0..100u32 {
            let key = format!("key_{}", i).into_bytes();
            let value = format!("value_{}", i).into_bytes();
            entries.push(WalEntry::new_upsert(key, value));
        }

        let file_a = OpenOptions::new().create(true).truncate(true).read(true).write(true).open(&wal_path_a)?;
        let file_b = OpenOptions::new().create(true).truncate(true).read(true).write(true).open(&wal_path_b)?;
        let pointers = write_entries(&file_a, &entries)?;
        write_entries(&file_b, &entries)?;

        let wal = Arc::new(Wal::open_with_options(&wal_path_a, false)?);
        let stop = Arc::new(AtomicBool::new(false));
        let reader_failed = Arc::new(AtomicBool::new(false));

        let reader_wal = wal.clone();
        let reader_stop = stop.clone();
        let reader_failed_flag = reader_failed.clone();
        let reader_pointers = pointers.clone();
        let reader = thread::spawn(move || {
            while !reader_stop.load(Ordering::Relaxed) {
                for pointer in &reader_pointers {
                    if reader_wal.read_entry(*pointer).is_err() {
                        reader_failed_flag.store(true, Ordering::Relaxed);
                        return;
                    }
                }
            }
        });

        for _ in 0..200 {
            let new_file = OpenOptions::new().read(true).write(true).open(&wal_path_b)?;
            wal.swap_file(Arc::new(new_file))?;
            let new_file = OpenOptions::new().read(true).write(true).open(&wal_path_a)?;
            wal.swap_file(Arc::new(new_file))?;
        }

        thread::sleep(Duration::from_millis(50));
        stop.store(true, Ordering::Relaxed);
        let _ = reader.join();

        assert!(!reader_failed.load(Ordering::Relaxed));
        Ok(())
    }

    #[test]
    fn test_wal_segment_rotation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let wal = Wal::open_with_options_and_segment_size(temp_dir.path().join("test_rotate.wal"), false, 256)?;

        let mut tail = 0u64;
        for i in 0..50u32 {
            let key = format!("k{}", i).into_bytes();
            let value = vec![b'x'; 40];
            wal.append_entry(&WalEntry::new_upsert(key, value), &mut tail, false)?;
        }

        let scanned = wal.scan_entries(0, tail)?;
        assert_eq!(scanned.len(), 50);
        Ok(())
    }

    // Regression: WAL GC can delete a fully-persisted segment out of order
    // (leaving a "hole"), so `scan_entries` must skip a missing middle segment
    // instead of aborting. Aborting crashed recovery and stalled the persist
    // watermark (unbounded WAL growth). See `scan_entries`.
    #[test]
    fn test_scan_entries_skips_deleted_middle_segment() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let path = temp_dir.path().join("hole.wal");
        let wal = Wal::open_with_options_and_segment_size(&path, false, 256)?;

        let mut tail = 0u64;
        for i in 0..60u32 {
            wal.append_entry(&WalEntry::new_upsert(format!("k{i}").into_bytes(), vec![b'x'; 40]), &mut tail, false)?;
        }
        let full = wal.scan_entries(0, tail)?;
        assert_eq!(full.len(), 60, "baseline: full scan sees every entry");

        // Punch a hole: delete a middle segment (segment 0 stays live, so head
        // is not advanced past the hole).
        wal.delete_segment_file(1)?;

        // The scan must succeed, skipping the hole and returning the entries
        // from the surviving segments rather than erroring.
        let scanned = wal.scan_entries(0, tail)?;
        assert!(!scanned.is_empty(), "scan over a hole must still return surviving entries");
        assert!(scanned.len() < 60, "the deleted segment's entries are gone");
        // Every surviving offset must be outside the deleted segment.
        for (ptr, _) in &scanned {
            assert!(
                !(256..512).contains(&ptr.offset),
                "entry from deleted segment leaked: offset {}",
                ptr.offset
            );
        }
        Ok(())
    }

    /// `scan_keys_in_entries` and `scan_entries` share one frame-parsing loop, so
    /// they must always agree on which entries a range contains — including over
    /// a hole, where the recovery judgement "a missing segment is skipped, not an
    /// error" lives. This is what stops the two drifting apart.
    #[test]
    fn key_only_scan_agrees_with_the_full_scan() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let wal = Wal::open_with_options_and_segment_size(temp_dir.path().join("keys.wal"), false, 256)?;

        let mut tail = 0u64;
        for i in 0..60u32 {
            wal.append_entry(
                &WalEntry::new_upsert_ns(i % 3, format!("k{i}").into_bytes(), vec![b'x'; 40]),
                &mut tail,
                false,
            )?;
        }

        let expect = |wal: &Wal, tail: u64| -> Result<Vec<(u32, Vec<u8>)>> {
            Ok(wal.scan_entries(0, tail)?.into_iter().map(|(_, e)| (e.namespace_id, e.key)).collect())
        };

        assert_eq!(
            wal.scan_keys_in_entries(0, tail)?,
            expect(&wal, tail)?,
            "baseline: same entries, same order"
        );

        // ...and over a hole, so both paths make the same skip-don't-abort call.
        wal.delete_segment_file(1)?;
        let keys = wal.scan_keys_in_entries(0, tail)?;
        assert_eq!(keys, expect(&wal, tail)?, "both scans must skip the hole identically");
        assert!(keys.len() < 60, "the deleted segment's entries are gone");
        Ok(())
    }

    /// `scan_segment_keys` covers exactly one segment and deduplicates, since a
    /// key written repeatedly needs repairing once. `scan_keys_in_entries` does
    /// not deduplicate — the raw WAL order is what a caller reading a range wants.
    #[test]
    fn segment_key_scan_is_scoped_and_deduplicated() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let wal = Wal::open_with_options_and_segment_size(temp_dir.path().join("dedupe.wal"), false, 256)?;

        let mut tail = 0u64;
        // The same key rewritten many times, plus a distinct one.
        for _ in 0..8u32 {
            wal.append_entry(&WalEntry::new_upsert(b"hot".to_vec(), vec![b'x'; 40]), &mut tail, false)?;
        }
        wal.append_entry(&WalEntry::new_upsert(b"cold".to_vec(), vec![b'x'; 40]), &mut tail, false)?;

        let raw = wal.scan_keys_in_entries(0, tail)?;
        assert_eq!(raw.len(), 9, "the range scan keeps every occurrence");

        // Segment 0 only, deduplicated.
        let seg0 = wal.scan_segment_keys(0)?;
        let distinct: Vec<Vec<u8>> = seg0.iter().map(|(_, k)| k.clone()).collect();
        assert!(distinct.len() < 9, "repeated keys collapse to one entry each, got {distinct:?}");
        assert_eq!(
            distinct.iter().filter(|k| k.as_slice() == b"hot").count(),
            1,
            "a key written eight times must appear once"
        );
        // Every key came from segment 0 alone.
        let seg0_end = 256u64;
        let in_segment_0: Vec<(u32, Vec<u8>)> = wal.scan_keys_in_entries(0, seg0_end)?;
        for pair in &seg0 {
            assert!(
                in_segment_0.contains(pair),
                "scan_segment_keys returned {pair:?} from outside its segment"
            );
        }

        // A reclaimed segment yields nothing rather than an error — GC may have
        // already unlinked it, and failing would abort the whole pass.
        wal.delete_segment_file(0)?;
        assert!(wal.scan_segment_keys(0)?.is_empty());
        Ok(())
    }

    #[test]
    fn test_wal_delete_segment_file() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let wal = Wal::open_with_options_and_segment_size(temp_dir.path().join("test_delete_segment.wal"), false, 256)?;

        let mut tail = 0u64;
        for i in 0..50u32 {
            let key = format!("k{}", i).into_bytes();
            let value = vec![b'x'; 40];
            wal.append_entry(&WalEntry::new_upsert(key, value), &mut tail, false)?;
        }

        let segment_path = temp_dir.path().join("test_delete_segment.wal.seg000001");
        assert!(segment_path.exists());
        wal.delete_segment_file(1)?;
        assert!(!segment_path.exists());
        Ok(())
    }

    #[test]
    fn missing_segments_reports_a_hole_in_the_middle_of_the_range() -> Result<()> {
        // WAL GC reclaims fully-persisted segments out of *segment* order, so a
        // hole can sit in the middle of a replay window while `head` is still
        // below it. `scan_entries` skips such a hole silently, which is why a
        // consumer replaying the range needs this to know its result is partial.
        let temp_dir = TempDir::new()?;
        let path = temp_dir.path().join("holes.wal");
        let wal = Wal::open_with_options_and_segment_size(&path, false, 256)?;

        let mut tail = 0u64;
        for i in 0..50u32 {
            let key = format!("k{}", i).into_bytes();
            wal.append_entry(&WalEntry::new_upsert(key, vec![b'x'; 40]), &mut tail, false)?;
        }
        assert!(wal.segment_id_for_offset(tail - 1) >= 3, "test needs several segments, got {tail} bytes");

        assert_eq!(wal.missing_segments(0, tail), Vec::<u64>::new(), "nothing deleted yet");

        // Reclaim an interior segment, leaving 0 and everything above it in place.
        wal.delete_segment_file(1)?;
        assert_eq!(wal.missing_segments(0, tail), vec![1]);

        // A window that does not cover the hole is unaffected.
        let seg2_start = 2 * wal.segment_size();
        assert_eq!(wal.missing_segments(seg2_start, tail), Vec::<u64>::new());
        Ok(())
    }

    #[test]
    fn missing_segments_is_empty_for_an_empty_range() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let wal = Wal::open_with_options_and_segment_size(temp_dir.path().join("empty.wal"), false, 256)?;
        assert_eq!(wal.missing_segments(0, 0), Vec::<u64>::new());
        assert_eq!(wal.missing_segments(500, 100), Vec::<u64>::new(), "tail <= head");
        Ok(())
    }

    // ── sequence number tests ────────────────────────────────────────────

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
        Ok(())
    }

    #[test]
    fn test_default_constructors_have_zero_sequence() {
        assert_eq!(WalEntry::new_upsert(b"k".to_vec(), b"v".to_vec()).sequence, 0);
        assert_eq!(WalEntry::new_delete(b"k".to_vec()).sequence, 0);
        assert_eq!(WalEntry::new_upsert_ns(1, b"k".to_vec(), b"v".to_vec()).sequence, 0);
        assert_eq!(WalEntry::new_delete_ns(1, b"k".to_vec()).sequence, 0);
    }

    #[test]
    fn test_recover_sequence_empty_wal() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let wal = Wal::open(temp_dir.path().join("test.wal"))?;
        // No entries: result must be hint + 1 when hint > 0, else 1.
        assert_eq!(wal.recover_sequence(0, 0, 0), 1);
        assert_eq!(wal.recover_sequence(0, 0, 50), 51);
        Ok(())
    }

    #[test]
    fn test_recover_sequence_from_wal_entries() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let wal = Wal::open(temp_dir.path().join("test.wal"))?;

        let mut tail = 0u64;
        for seq in [3u64, 7, 1, 5] {
            let entry = WalEntry::new_upsert(b"k".to_vec(), b"v".to_vec()).with_sequence(seq);
            wal.append_entry(&entry, &mut tail, false)?;
        }

        // max sequence written is 7, so next should be 8
        assert_eq!(wal.recover_sequence(0, tail, 0), 8);
        Ok(())
    }

    #[test]
    fn test_recover_sequence_hint_wins_when_higher() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let wal = Wal::open(temp_dir.path().join("test.wal"))?;

        let mut tail = 0u64;
        let entry = WalEntry::new_upsert(b"k".to_vec(), b"v".to_vec()).with_sequence(10);
        wal.append_entry(&entry, &mut tail, false)?;

        // hint (20) > max in WAL (10): result should be 21
        assert_eq!(wal.recover_sequence(0, tail, 20), 21);
        Ok(())
    }

    #[test]
    fn test_recover_sequence_wal_wins_when_higher() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let wal = Wal::open(temp_dir.path().join("test.wal"))?;

        let mut tail = 0u64;
        let entry = WalEntry::new_upsert(b"k".to_vec(), b"v".to_vec()).with_sequence(100);
        wal.append_entry(&entry, &mut tail, false)?;

        // hint (5) < max in WAL (100): result should be 101
        assert_eq!(wal.recover_sequence(0, tail, 5), 101);
        Ok(())
    }

    #[test]
    fn test_recover_sequence_after_segment_rotation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        // Small segment size to force rotation
        let wal = Wal::open_with_options_and_segment_size(temp_dir.path().join("test_seq_rotate.wal"), false, 256)?;

        let mut tail = 0u64;
        let mut max_seq = 0u64;
        for i in 0..30u64 {
            let seq = i * 3 + 1;
            max_seq = seq;
            let entry = WalEntry::new_upsert(format!("key{}", i).into_bytes(), vec![b'x'; 40]).with_sequence(seq);
            wal.append_entry(&entry, &mut tail, false)?;
        }

        assert_eq!(wal.recover_sequence(0, tail, 0), max_seq + 1);
        Ok(())
    }

    // ── tail reconstruction tests ─────────────────────────────────────────

    #[test]
    fn test_recover_tail_empty_wal() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let wal = Wal::open(temp_dir.path().join("rt.wal"))?;
        assert_eq!(wal.recover_tail(0), 0);
        Ok(())
    }

    #[test]
    fn test_recover_tail_extends_past_stale_offset() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let wal = Wal::open(temp_dir.path().join("rt.wal"))?;

        let mut tail = 0u64;
        for i in 0..2u32 {
            wal.append_entry(&WalEntry::new_upsert(format!("k{i}").into_bytes(), b"v".to_vec()), &mut tail, false)?;
        }
        let stale = tail; // pretend metadata was flushed here
        for i in 2..6u32 {
            wal.append_entry(&WalEntry::new_upsert(format!("k{i}").into_bytes(), b"v".to_vec()), &mut tail, false)?;
        }

        // From a stale (earlier) offset, reconstruction reaches the true end.
        assert_eq!(wal.recover_tail(stale), tail);
        // From the start, also reaches the true end.
        assert_eq!(wal.recover_tail(0), tail);
        // From the true end, no extension.
        assert_eq!(wal.recover_tail(tail), tail);
        // The recovered region is fully scannable.
        assert_eq!(wal.scan_entries(stale, tail)?.len(), 4);
        Ok(())
    }

    #[test]
    fn test_recover_tail_stops_at_torn_entry() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let path = temp_dir.path().join("rt.wal");
        let wal = Wal::open(&path)?;

        let mut tail = 0u64;
        for i in 0..3u32 {
            wal.append_entry(&WalEntry::new_upsert(format!("k{i}").into_bytes(), b"v".to_vec()), &mut tail, false)?;
        }
        let good_tail = tail;

        // Simulate a crash mid-append: a size header claiming 100 bytes with only
        // 10 bytes of body following it.
        let f = OpenOptions::new().write(true).open(&path)?;
        f.write_at(&100u32.to_le_bytes(), good_tail)?;
        f.write_at(&[0xABu8; 10], good_tail + 4)?;
        f.sync_all()?;

        // Reconstruction must stop at the last complete entry, ignoring the torn one.
        assert_eq!(wal.recover_tail(0), good_tail);
        Ok(())
    }

    #[test]
    fn test_recover_tail_across_segment_rotation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        // Small segments to force rotation.
        let wal = Wal::open_with_options_and_segment_size(temp_dir.path().join("rt.wal"), false, 256)?;

        let mut tail = 0u64;
        for i in 0..40u32 {
            wal.append_entry(&WalEntry::new_upsert(format!("k{i}").into_bytes(), vec![b'x'; 40]), &mut tail, false)?;
        }
        // Reconstruct from scratch across the rotated segments.
        assert_eq!(wal.recover_tail(0), tail);
        Ok(())
    }

    /// A changed byte inside a value still decodes (rkyv checks structure, not
    /// contents), so without a checksum replay would write the wrong value. The
    /// frame's CRC rejects it, and recovery stops there.
    #[test]
    fn a_changed_byte_in_a_value_is_rejected_and_recovery_stops_there() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let path = temp_dir.path().join("crc.wal");
        let wal = Wal::open(&path)?;
        let mut tail = 0u64;
        let first = wal.append_entry(&WalEntry::new_upsert(b"k1".to_vec(), b"value-one".to_vec()), &mut tail, false)?;
        let end_of_first = tail;
        let second = wal.append_entry(&WalEntry::new_upsert(b"k2".to_vec(), b"value-two".to_vec()), &mut tail, false)?;
        wal.append_entry(&WalEntry::new_upsert(b"k3".to_vec(), b"value-three".to_vec()), &mut tail, false)?;

        // Flip one byte of the second entry's value ("value-two" → "valuE-two").
        let f = OpenOptions::new().read(true).write(true).open(&path)?;
        let mut body = vec![0u8; second.size as usize];
        f.read_exact_at(&mut body, second.offset + FRAME_HEADER_LEN)?;
        let at = body.windows(9).position(|w| w == b"value-two").expect("value bytes in the body");
        body[at + 4] = b'E';
        assert!(WalEntry::from_bytes(&body).is_ok(), "setup: the changed body must still decode");
        f.write_at(&body, second.offset + FRAME_HEADER_LEN)?;

        assert_eq!(wal.recover_tail(0), end_of_first, "recovery must stop at the corrupt frame");
        let scanned = wal.scan_entries(0, tail)?;
        assert_eq!(scanned.len(), 1, "the scan must stop at the corrupt frame");
        assert_eq!(scanned[0].0.offset, first.offset);
        assert!(wal.read_entry(second).is_err(), "reading the corrupt frame must fail");
        Ok(())
    }
}
