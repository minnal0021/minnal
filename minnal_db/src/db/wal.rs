//! Write-Ahead Log (WAL) Module
//!
//! This module implements a write-ahead log for the MinnalDB.
//! It provides:
//! - Operation logging (UPSERT, DELETE) with status tracking
//! - Sequential record writing with head/tail tracking
//! - WAL metadata management
//! - Garbage collection of persisted entries based on status
//! - Atomic file swapping during GC

use crc32fast::Hasher;
use crossbeam_epoch::{self as epoch, Atomic, Guard, Owned};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
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
}

pub type Result<T> = std::result::Result<T, WalError>;

const DEFAULT_SEGMENT_SIZE: u64 = 64 * 1024 * 1024;
const WAL_METADATA_MAGIC: [u8; 4] = *b"WALM";
const WAL_METADATA_VERSION: u32 = 1;

/// Status of a WAL entry
#[derive(Debug, Clone, Copy, Archive, RkyvSerialize, RkyvDeserialize, PartialEq, Eq)]
pub enum WalEntryStatus {
    /// Entry just written to WAL, not yet persisted to LSM or value store
    Inserted,
    /// Entry has been persisted to LSM and value store, safe to garbage collect
    Persisted,
}

/// Operation type for WAL entries
#[derive(Debug, Clone, Copy, Archive, RkyvSerialize, RkyvDeserialize)]
pub enum WalOperationType {
    Upsert,
    Delete,
}

/// A single WAL entry containing the operation, data, and status.
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
    pub status: WalEntryStatus,
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
            status: WalEntryStatus::Inserted,
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
            status: WalEntryStatus::Inserted,
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

    pub fn with_status(mut self, status: WalEntryStatus) -> Self {
        self.status = status;
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

    /// Read only the status field without copying key/value bytes.
    ///
    /// Used to filter entries before deciding whether full deserialization is needed.
    pub fn peek_status(bytes: &[u8]) -> Result<WalEntryStatus> {
        let archived = rkyv::access::<ArchivedWalEntry, rkyv::rancor::Error>(bytes)
            .map_err(|e| WalError::Serialization(format!("WAL entry validation failed: {}", e)))?;
        Ok(match archived.status {
            ArchivedWalEntryStatus::Inserted => WalEntryStatus::Inserted,
            ArchivedWalEntryStatus::Persisted => WalEntryStatus::Persisted,
        })
    }

    /// Deserialize entry from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let archived = rkyv::access::<ArchivedWalEntry, rkyv::rancor::Error>(bytes)
            .map_err(|e| WalError::Serialization(format!("WAL entry validation failed: {}", e)))?;

        let status = match archived.status {
            ArchivedWalEntryStatus::Inserted => WalEntryStatus::Inserted,
            ArchivedWalEntryStatus::Persisted => WalEntryStatus::Persisted,
        };

        let operation = match archived.operation {
            ArchivedWalOperationType::Upsert => WalOperationType::Upsert,
            ArchivedWalOperationType::Delete => WalOperationType::Delete,
        };

        Ok(WalEntry {
            status,
            operation,
            namespace_id: archived.namespace_id.into(),
            sequence: archived.sequence.to_native(),
            op_name: archived.op_name.to_string(),
            key: archived.key.to_vec(),
            value: archived.value.as_ref().map(|v| v.to_vec()),
        })
    }
}

/// Pointer to a WAL entry with status information
#[derive(Debug, Clone, Copy, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct WalPointer {
    pub offset: u64,
    pub size: u32,
    pub status: WalEntryStatus,
}

impl WalPointer {
    pub fn new(offset: u64, size: u32, status: WalEntryStatus) -> Self {
        Self { offset, size, status }
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

        let status = match archived.status {
            ArchivedWalEntryStatus::Inserted => WalEntryStatus::Inserted,
            ArchivedWalEntryStatus::Persisted => WalEntryStatus::Persisted,
        };

        Ok(WalPointer {
            offset: archived.offset.to_native(),
            size: archived.size.to_native(),
            status,
        })
    }
}

/// Metadata for tracking WAL state
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, serde::Serialize)]
pub struct WalMetadata {
    pub head: u64,              // Offset where GC starts
    pub tail: u64,              // Offset where new writes go
    pub total_entries: u64,     // Total entries written
    pub persisted_entries: u64, // Entries persisted to LSM
    /// Absolute segment id that `segment_*_entries[0]` refers to. Counters for
    /// segments below this have been reclaimed and trimmed, so the per-segment
    /// vecs stay proportional to the live segment window rather than growing
    /// with every segment ever created. Maintained equal to `head`'s segment.
    pub base_segment_id: u64,
    pub segment_total_entries: Vec<u64>,
    pub segment_persisted_entries: Vec<u64>,
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
            total_entries: 0,
            persisted_entries: 0,
            base_segment_id: 0,
            segment_total_entries: Vec::new(),
            segment_persisted_entries: Vec::new(),
            total_gc_runs: 0,
            total_bytes_reclaimed: 0,
            last_sequence: 0,
        }
    }

    // ── Per-segment counters (addressed by absolute segment id) ──────────
    //
    // The dense `segment_*_entries` vecs start at `base_segment_id`, so all of
    // these translate an absolute segment id to the relative slot. Segments
    // below the base were reclaimed and trimmed; reads there return 0 and writes
    // are ignored (you never write to an already-reclaimed segment).

    #[inline]
    fn segment_index(&self, segment_id: u64) -> Option<usize> {
        segment_id.checked_sub(self.base_segment_id).map(|i| i as usize)
    }

    /// Entries recorded for `segment_id` (0 if trimmed/untracked).
    pub fn segment_total(&self, segment_id: u64) -> u64 {
        self.segment_index(segment_id)
            .and_then(|i| self.segment_total_entries.get(i).copied())
            .unwrap_or(0)
    }

    /// Persisted entries recorded for `segment_id` (0 if trimmed/untracked).
    pub fn segment_persisted(&self, segment_id: u64) -> u64 {
        self.segment_index(segment_id)
            .and_then(|i| self.segment_persisted_entries.get(i).copied())
            .unwrap_or(0)
    }

    /// Absolute segment ids that currently have counters (`base .. base+len`).
    pub fn tracked_segments(&self) -> std::ops::Range<u64> {
        self.base_segment_id..self.base_segment_id + self.segment_total_entries.len() as u64
    }

    /// Grow the dense vecs so `segment_id` is addressable; returns its relative
    /// index, or `None` if it is below the trimmed base.
    fn grow_to(&mut self, segment_id: u64) -> Option<usize> {
        let idx = self.segment_index(segment_id)?;
        while self.segment_total_entries.len() <= idx {
            self.segment_total_entries.push(0);
            self.segment_persisted_entries.push(0);
        }
        Some(idx)
    }

    pub fn add_segment_total(&mut self, segment_id: u64, n: u64) {
        if let Some(idx) = self.grow_to(segment_id) {
            self.segment_total_entries[idx] = self.segment_total_entries[idx].saturating_add(n);
        }
    }

    pub fn add_segment_persisted(&mut self, segment_id: u64, n: u64) {
        if let Some(idx) = self.grow_to(segment_id) {
            self.segment_persisted_entries[idx] = self.segment_persisted_entries[idx].saturating_add(n);
        }
    }

    pub fn set_segment_persisted(&mut self, segment_id: u64, value: u64) {
        if let Some(idx) = self.grow_to(segment_id) {
            self.segment_persisted_entries[idx] = value;
        }
    }

    /// Zero a reclaimed segment's counters (kept as a dense hole until trimmed).
    pub fn clear_segment(&mut self, segment_id: u64) {
        if let Some(idx) = self.segment_index(segment_id)
            && idx < self.segment_total_entries.len()
        {
            self.segment_total_entries[idx] = 0;
            self.segment_persisted_entries[idx] = 0;
        }
    }

    /// Make both counter vecs the same length (defensive reconciliation).
    pub fn reconcile_segment_lengths(&mut self) {
        let n = self.segment_total_entries.len().max(self.segment_persisted_entries.len());
        self.segment_total_entries.resize(n, 0);
        self.segment_persisted_entries.resize(n, 0);
    }

    /// Reset all persisted counters to zero (rebuild paths recompute them).
    pub fn reset_segment_persisted(&mut self) {
        for p in &mut self.segment_persisted_entries {
            *p = 0;
        }
    }

    /// Drop counters for every segment below `segment_id` (reclaimed), bumping
    /// the base so the dense vecs track only the live segment window instead of
    /// growing with every segment ever created.
    pub fn trim_segments_before(&mut self, segment_id: u64) {
        let Some(drop) = self.segment_index(segment_id) else {
            return;
        };
        let drop = drop.min(self.segment_total_entries.len());
        if drop == 0 {
            return;
        }
        self.segment_total_entries.drain(0..drop);
        self.segment_persisted_entries.drain(0..drop);
        self.base_segment_id += drop as u64;
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
        let segment_total_entries: Vec<u64> = archived.segment_total_entries.iter().map(|v| v.to_native()).collect();
        let segment_persisted_entries: Vec<u64> = archived.segment_persisted_entries.iter().map(|v| v.to_native()).collect();
        Ok(WalMetadata {
            head: archived.head.to_native(),
            tail: archived.tail.to_native(),
            total_entries: archived.total_entries.to_native(),
            persisted_entries: archived.persisted_entries.to_native(),
            base_segment_id: archived.base_segment_id.to_native(),
            segment_total_entries,
            segment_persisted_entries,
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
        let mut options = OpenOptions::new();
        options.read(true).write(true);
        if create {
            options.create(true);
        }
        Ok(options.open(path)?)
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
        let file = OpenOptions::new().create(true).truncate(false).read(true).write(true).open(&path)?;

        let handle = WalHandle::new(Arc::new(file));

        Ok(Self {
            handle: Atomic::new(handle),
            path,
            segment_size: segment_size.max(256),
            current_segment_id: AtomicU64::new(0),
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
            self.rotate_to_segment(segment_id)?;
            file_arc = self.current_file_arc()?;
        }

        // Serialize the entry
        let entry_bytes = entry.to_bytes()?;
        let size = entry_bytes.len() as u32;
        let entry_len = 4u64 + size as u64;
        let remaining = self.segment_size.saturating_sub(segment_offset);
        if entry_len > remaining {
            if remaining >= 4 {
                self.write_all_at(&file_arc, &0u32.to_le_bytes(), segment_offset)?;
            }
            segment_id += 1;
            *tail = segment_id * self.segment_size;
            segment_offset = 0;
            self.rotate_to_segment(segment_id)?;
            file_arc = self.current_file_arc()?;
        }

        // Write size header (4 bytes) + serialized entry
        self.write_all_at(&file_arc, &size.to_le_bytes(), segment_offset)?;
        self.write_all_at(&file_arc, &entry_bytes, segment_offset + 4)?;
        if sync {
            file_arc.sync_data()?;
        }

        let pointer = WalPointer::new(*tail, size, entry.status);
        *tail += 4 + size as u64;

        Ok(pointer)
    }

    /// Read a WAL entry from the given pointer
    #[allow(dead_code)]
    pub fn read_entry(&self, pointer: WalPointer) -> Result<WalEntry> {
        let guard = epoch::pin();
        let handle = self.load_handle(&guard)?;

        let segment_id = self.segment_id_for(pointer.offset);
        let segment_offset = self.segment_offset_for(pointer.offset);
        let file = if segment_id == self.current_segment_id.load(Ordering::Acquire) {
            &*handle.file
        } else {
            let file = self.open_segment_file(segment_id, false)?;
            return self.read_entry_from_file(&file, segment_offset, pointer.size);
        };

        // Read size header
        let mut size_buf = [0u8; 4];
        self.read_exact_at(file, &mut size_buf, segment_offset)?;
        let size = u32::from_le_bytes(size_buf);

        // Verify size matches pointer
        if size != pointer.size {
            return Err(WalError::CorruptedLog);
        }

        // Read entry bytes
        let mut entry_bytes = vec![0u8; size as usize];
        self.read_exact_at(file, &mut entry_bytes, segment_offset + 4)?;

        WalEntry::from_bytes(&entry_bytes)
    }

    /// Update entry status in-place (for marking as persisted)
    /// This is efficient with memory mapping
    pub fn update_entry_status(&self, offset: u64, new_status: WalEntryStatus) -> Result<()> {
        // Read the entry
        let guard = epoch::pin();
        let handle = self.load_handle(&guard)?;
        let segment_id = self.segment_id_for(offset);
        let segment_offset = self.segment_offset_for(offset);
        let file = if segment_id == self.current_segment_id.load(Ordering::Acquire) {
            &*handle.file
        } else {
            let file = self.open_segment_file(segment_id, false)?;
            return self.update_entry_status_in_file(&file, segment_offset, new_status);
        };

        let mut size_buf = [0u8; 4];
        self.read_exact_at(file, &mut size_buf, segment_offset)?;
        let size = u32::from_le_bytes(size_buf);

        let mut entry_bytes = vec![0u8; size as usize];
        self.read_exact_at(file, &mut entry_bytes, segment_offset + 4)?;

        // Deserialize, update status, and re-serialize
        let mut entry = WalEntry::from_bytes(&entry_bytes)?;
        entry.status = new_status;
        let updated_bytes = entry.to_bytes()?;

        if updated_bytes.len() != size as usize {
            return Err(WalError::CorruptedLog); // Status change caused size change
        }

        // Write back
        self.write_all_at(file, &updated_bytes, segment_offset + 4)?; // Skip size header
        file.sync_data()?;

        Ok(())
    }

    /// Scan all entries from head to tail, filtering by status
    pub fn scan_entries(&self, head: u64, tail: u64) -> Result<Vec<(WalPointer, WalEntry)>> {
        let mut entries = Vec::new();
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
                    // A missing segment is a "hole": WAL GC deleted a
                    // fully-persisted segment out of segment order (e.g. a
                    // dropped namespace persisted a trailing segment while an
                    // earlier one stayed live — see `garbage_collect_wal`).
                    // Skip the hole and keep scanning; aborting here would crash
                    // recovery and stall `mark_persisted_range` (so WAL GC could
                    // never reclaim past the hole — unbounded WAL growth).
                    Err(WalError::Io(ref e)) if e.kind() == std::io::ErrorKind::NotFound => None,
                    Err(e) => return Err(e),
                };
            }

            let Some(current_file) = current_file.as_ref() else {
                offset = (segment_id + 1) * self.segment_size;
                continue;
            };

            let segment_offset = self.segment_offset_for(offset);
            let mut size_buf = [0u8; 4];
            if self.read_exact_at(current_file, &mut size_buf, segment_offset).is_err() {
                let next_segment = (segment_id + 1) * self.segment_size;
                if next_segment > offset {
                    offset = next_segment;
                    continue;
                }
                break;
            }
            let size = u32::from_le_bytes(size_buf);
            if size == 0 {
                offset = (segment_id + 1) * self.segment_size;
                continue;
            }

            entry_bytes.resize(size as usize, 0);
            if self
                .read_exact_at(current_file, &mut entry_bytes[..size as usize], segment_offset + 4)
                .is_err()
            {
                let next_segment = (segment_id + 1) * self.segment_size;
                if next_segment > offset {
                    offset = next_segment;
                    continue;
                }
                break;
            }

            match WalEntry::from_bytes(&entry_bytes[..size as usize]) {
                Ok(entry) => {
                    let pointer = WalPointer::new(offset, size, entry.status);
                    entries.push((pointer, entry));
                }
                Err(_) => break,
            }

            offset += 4 + size as u64;
        }

        Ok(entries)
    }

    /// Scan only INSERTED entries from head onwards (for GC)
    #[allow(dead_code)]
    pub fn scan_inserted_entries(&self, head: u64, tail: u64) -> Result<Vec<(WalPointer, WalEntry)>> {
        let all_entries = self.scan_entries(head, tail)?;
        Ok(all_entries
            .into_iter()
            .filter(|(_, entry)| entry.status == WalEntryStatus::Inserted)
            .collect())
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

            // A readable size header that is non-zero begins an entry. A zero or
            // unreadable header means end-of-segment: either an explicit padding
            // marker, or a sub-4-byte remainder left when an entry was rotated to
            // the next segment without room for padding. In both cases the log
            // continues in the next segment *iff* that segment file exists —
            // otherwise we are at the true end.
            let mut size_buf = [0u8; 4];
            let header_ok = self.read_exact_at(&file, &mut size_buf, segment_offset).is_ok();
            let size = if header_ok { u32::from_le_bytes(size_buf) } else { 0 };

            if !header_ok || size == 0 {
                if self.open_segment_file(segment_id + 1, false).is_ok() {
                    offset = (segment_id + 1) * self.segment_size;
                    continue;
                }
                break; // no successor segment → true end of log
            }

            let mut entry_bytes = vec![0u8; size as usize];
            if self.read_exact_at(&file, &mut entry_bytes, segment_offset + 4).is_err() {
                break; // header present but body truncated → torn final append
            }
            if WalEntry::from_bytes(&entry_bytes).is_err() {
                break; // body present but not a valid entry → corrupt tail
            }
            offset += 4 + size as u64;
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

    #[allow(dead_code)]
    fn read_entry_from_file(&self, file: &File, segment_offset: u64, size: u32) -> Result<WalEntry> {
        let mut size_buf = [0u8; 4];
        self.read_exact_at(file, &mut size_buf, segment_offset)?;
        let stored_size = u32::from_le_bytes(size_buf);
        if stored_size != size {
            return Err(WalError::CorruptedLog);
        }
        let mut entry_bytes = vec![0u8; size as usize];
        self.read_exact_at(file, &mut entry_bytes, segment_offset + 4)?;
        WalEntry::from_bytes(&entry_bytes)
    }

    fn update_entry_status_in_file(&self, file: &File, segment_offset: u64, new_status: WalEntryStatus) -> Result<()> {
        let mut size_buf = [0u8; 4];
        self.read_exact_at(file, &mut size_buf, segment_offset)?;
        let size = u32::from_le_bytes(size_buf);
        let mut entry_bytes = vec![0u8; size as usize];
        self.read_exact_at(file, &mut entry_bytes, segment_offset + 4)?;
        let mut entry = WalEntry::from_bytes(&entry_bytes)?;
        entry.status = new_status;
        let updated_bytes = entry.to_bytes()?;
        if updated_bytes.len() != size as usize {
            return Err(WalError::CorruptedLog);
        }
        self.write_all_at(file, &updated_bytes, segment_offset + 4)?;
        file.sync_data()?;
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

    #[test]
    fn test_wal_entry_serialization() -> Result<()> {
        let entry = WalEntry::new_upsert(b"key1".to_vec(), b"value1".to_vec());
        assert_eq!(entry.status, WalEntryStatus::Inserted);

        let bytes = entry.to_bytes()?;
        let restored = WalEntry::from_bytes(&bytes)?;

        assert_eq!(entry.key, restored.key);
        assert_eq!(entry.value, restored.value);
        assert_eq!(entry.status, restored.status);
        Ok(())
    }

    #[test]
    fn test_wal_entry_delete_serialization() -> Result<()> {
        let entry = WalEntry::new_delete(b"key1".to_vec());
        assert_eq!(entry.status, WalEntryStatus::Inserted);

        let bytes = entry.to_bytes()?;
        let restored = WalEntry::from_bytes(&bytes)?;

        assert_eq!(entry.key, restored.key);
        assert!(restored.value.is_none());
        assert_eq!(entry.status, restored.status);
        Ok(())
    }

    #[test]
    fn test_wal_entry_status_update() -> Result<()> {
        let entry = WalEntry::new_upsert(b"key1".to_vec(), b"value1".to_vec());
        let original_status = entry.status;
        let updated = entry.clone().with_status(WalEntryStatus::Persisted);

        assert_eq!(original_status, WalEntryStatus::Inserted);
        assert_eq!(updated.status, WalEntryStatus::Persisted);
        Ok(())
    }

    #[test]
    fn test_wal_metadata_serialization() -> Result<()> {
        let metadata = WalMetadata {
            head: 0,
            tail: 1024,
            total_entries: 10,
            persisted_entries: 8,
            base_segment_id: 0,
            segment_total_entries: vec![5, 5],
            segment_persisted_entries: vec![4, 4],
            total_gc_runs: 2,
            total_bytes_reclaimed: 256,
            last_sequence: 0,
        };

        let bytes = metadata.to_bytes()?;
        let restored = WalMetadata::from_bytes(&bytes)?;

        assert_eq!(metadata.head, restored.head);
        assert_eq!(metadata.tail, restored.tail);
        assert_eq!(metadata.total_entries, restored.total_entries);
        assert_eq!(metadata.persisted_entries, restored.persisted_entries);
        assert_eq!(metadata.segment_total_entries, restored.segment_total_entries);
        assert_eq!(metadata.segment_persisted_entries, restored.segment_persisted_entries);
        Ok(())
    }

    #[test]
    fn test_wal_metadata_file_encoding() -> Result<()> {
        let metadata = WalMetadata {
            head: 0,
            tail: 1024,
            total_entries: 10,
            persisted_entries: 8,
            base_segment_id: 0,
            segment_total_entries: vec![5, 5],
            segment_persisted_entries: vec![4, 4],
            total_gc_runs: 2,
            total_bytes_reclaimed: 256,
            last_sequence: 0,
        };

        let file_bytes = metadata.to_file_bytes()?;
        let restored = WalMetadata::from_file_bytes(&file_bytes)?;

        assert_eq!(metadata.head, restored.head);
        assert_eq!(metadata.tail, restored.tail);
        assert_eq!(metadata.total_entries, restored.total_entries);
        assert_eq!(metadata.persisted_entries, restored.persisted_entries);
        assert_eq!(metadata.segment_total_entries, restored.segment_total_entries);
        assert_eq!(metadata.segment_persisted_entries, restored.segment_persisted_entries);
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
        assert_eq!(entry.status, read_entry.status);
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
            assert_eq!(entries[i].status, read_entry.status);
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
            assert_eq!(entries[i].status, entry.status);
        }

        Ok(())
    }

    #[test]
    fn test_wal_scan_inserted_entries() -> Result<()> {
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

        // Scan only INSERTED entries
        let inserted = wal.scan_inserted_entries(0, tail)?;
        assert_eq!(inserted.len(), 3); // All should be INSERTED initially

        // Now mark some as persisted and scan again
        if let Some((pointer, _)) = inserted.first() {
            wal.update_entry_status(pointer.offset, WalEntryStatus::Persisted)?;
        }

        let inserted_after = wal.scan_inserted_entries(0, tail)?;
        assert_eq!(inserted_after.len(), 2); // One is persisted now

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
                file.write_all(&size.to_le_bytes())?;
                file.write_all(&entry_bytes)?;
                pointers.push(WalPointer::new(offset, size, entry.status));
                offset += 4 + size as u64;
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
}
