//! Value Log Module
//!
//! This module handles storage and management of actual values in a file.
//! It provides:
//! - Value pointer serialization/deserialization
//! - Value log file operations (read/write)
//! - Metadata management (head/tail tracking)
//! - Value log compaction during GC

pub mod sharded;

use crc32fast::Hasher;
use parking_lot::RwLock;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
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
}

pub type Result<T> = std::result::Result<T, ValueLogError>;

pub const PAGE_SIZE_BYTES: u64 = 64 * 1024 * 1024;
const PAGE_HEADER_MAGIC: [u8; 4] = *b"VPG1";
const PAGE_HEADER_VERSION: u32 = 1;
const PAGE_HEADER_SIZE: u32 = 32;
const TABLE_ENTRY_SIZE: u32 = 8;

const FLAG_TOMBSTONE: u8 = 0x01;
const FLAG_UPDATED: u8 = 0x02;

/// A batched value read result: `(original_index, Some((value, record_seq)))`
/// for a readable slot, or `(index, None)`. `record_seq` lets the caller verify
/// the slot still holds the write its pointer referred to.
pub(crate) type BatchValue = (usize, Option<(Vec<u8>, u64)>);

/// Location of a value record inside a bucket file (page offset + segment id).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueLocation {
    pub page_offset: u64,
    pub segment_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueRecordMeta {
    pub version: u32,
    pub tombstone: bool,
    pub updated: bool,
    pub epoch: u64,
    /// Full global write sequence (u64) of the write that produced this record.
    /// Mirrors the LSM's per-key seq so a reader can verify that a resolved
    /// pointer still refers to this exact write (a recycled slot carries a
    /// different, globally-unique seq → stale-pointer detection). Stored at full
    /// u64 width for debugging precision; the read-time check compares the low
    /// 32 bits against the LSM's u32 seq.
    pub seq: u64,
}

/// CRC32 of a value record's payload bytes. Stored in the record header and
/// re-verified on every value read to catch silent on-disk corruption (bit
/// rot, torn writes) that the structural checks (segment-id / length bounds)
/// would otherwise miss.
fn value_checksum(value: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(value);
    hasher.finalize()
}

#[derive(Debug, Clone, Copy)]
struct ValueRecordHeader {
    total_len: u32,
    version: u32,
    flags: u8,
    epoch: u64,
    value_len: u32,
    /// CRC32 over the value payload (`value_checksum`).
    checksum: u32,
    /// Full global write sequence (see [`ValueRecordMeta::seq`]).
    seq: u64,
}

impl ValueRecordHeader {
    const SIZE: usize = 36;

    fn to_bytes(self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        out[0..4].copy_from_slice(&self.total_len.to_le_bytes());
        out[4..8].copy_from_slice(&self.version.to_le_bytes());
        out[8] = self.flags;
        out[9..12].copy_from_slice(&[0u8; 3]);
        out[12..20].copy_from_slice(&self.epoch.to_le_bytes());
        out[20..24].copy_from_slice(&self.value_len.to_le_bytes());
        out[24..28].copy_from_slice(&self.checksum.to_le_bytes());
        out[28..36].copy_from_slice(&self.seq.to_le_bytes());
        out
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::SIZE {
            return None;
        }
        let total_len = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
        let version = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
        let flags = bytes[8];
        let epoch = u64::from_le_bytes(bytes[12..20].try_into().ok()?);
        let value_len = u32::from_le_bytes(bytes[20..24].try_into().ok()?);
        let checksum = u32::from_le_bytes(bytes[24..28].try_into().ok()?);
        let seq = u64::from_le_bytes(bytes[28..36].try_into().ok()?);
        Some(Self {
            total_len,
            version,
            flags,
            epoch,
            value_len,
            checksum,
            seq,
        })
    }

    fn meta(&self) -> ValueRecordMeta {
        ValueRecordMeta {
            version: self.version,
            tombstone: (self.flags & FLAG_TOMBSTONE) != 0,
            updated: (self.flags & FLAG_UPDATED) != 0,
            epoch: self.epoch,
            seq: self.seq,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PageHeader {
    magic: [u8; 4],
    version: u32,
    free_offset: u32,
    table_offset: u32,
    next_segment_id: u32,
    page_size: u32,
    reserved0: u32,
    reserved1: u32,
}

impl PageHeader {
    fn new() -> Self {
        Self {
            magic: PAGE_HEADER_MAGIC,
            version: PAGE_HEADER_VERSION,
            free_offset: PAGE_HEADER_SIZE,
            table_offset: PAGE_SIZE_BYTES as u32,
            next_segment_id: 1,
            page_size: PAGE_SIZE_BYTES as u32,
            reserved0: 0,
            reserved1: 0,
        }
    }

    fn to_bytes(self) -> [u8; PAGE_HEADER_SIZE as usize] {
        let mut out = [0u8; PAGE_HEADER_SIZE as usize];
        out[0..4].copy_from_slice(&self.magic);
        out[4..8].copy_from_slice(&self.version.to_le_bytes());
        out[8..12].copy_from_slice(&self.free_offset.to_le_bytes());
        out[12..16].copy_from_slice(&self.table_offset.to_le_bytes());
        out[16..20].copy_from_slice(&self.next_segment_id.to_le_bytes());
        out[20..24].copy_from_slice(&self.page_size.to_le_bytes());
        out[24..28].copy_from_slice(&self.reserved0.to_le_bytes());
        out[28..32].copy_from_slice(&self.reserved1.to_le_bytes());
        out
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < PAGE_HEADER_SIZE as usize {
            return None;
        }
        let magic: [u8; 4] = bytes[0..4].try_into().ok()?;
        if magic != PAGE_HEADER_MAGIC {
            return None;
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
        if version != PAGE_HEADER_VERSION {
            return None;
        }
        Some(Self {
            magic,
            version,
            free_offset: u32::from_le_bytes(bytes[8..12].try_into().ok()?),
            table_offset: u32::from_le_bytes(bytes[12..16].try_into().ok()?),
            next_segment_id: u32::from_le_bytes(bytes[16..20].try_into().ok()?),
            page_size: u32::from_le_bytes(bytes[20..24].try_into().ok()?),
            reserved0: u32::from_le_bytes(bytes[24..28].try_into().ok()?),
            reserved1: u32::from_le_bytes(bytes[28..32].try_into().ok()?),
        })
    }
}

/// Per-page garbage statistics for selective compaction.
#[derive(Debug, Clone)]
pub struct PageGarbageStats {
    pub page_offset: u64,
    pub live_bytes: u64,
    pub garbage_bytes: u64,
    pub total_records: u32,
    pub garbage_records: u32,
}

impl PageGarbageStats {
    /// Returns the garbage ratio as a percentage (0.0 - 100.0).
    pub fn garbage_ratio_pct(&self) -> f64 {
        let total = self.live_bytes + self.garbage_bytes;
        if total == 0 {
            return 0.0;
        }
        (self.garbage_bytes as f64 / total as f64) * 100.0
    }
}

/// Metadata for tracking value log state
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, serde::Serialize)]
pub struct ValueLogMetadata {
    pub head: u64,
    pub tail: u64,
    pub current_page_offset: u64,
    pub current_page_free_offset: u32,
    pub current_page_table_offset: u32,
    pub current_page_next_segment_id: u32,
    pub total_gc_runs: u64,
    pub total_bytes_reclaimed: u64,
    pub live_bytes: u64,
    pub garbage_bytes: u64,
}

const VALUE_LOG_METADATA_MAGIC: [u8; 4] = *b"VLOG";
const VALUE_LOG_METADATA_VERSION: u32 = 2;

#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
struct ValueLogMetadataV1 {
    pub head: u64,
    pub tail: u64,
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
            head: 0,
            tail: PAGE_SIZE_BYTES,
            current_page_offset: 0,
            current_page_free_offset: PAGE_HEADER_SIZE,
            current_page_table_offset: PAGE_SIZE_BYTES as u32,
            current_page_next_segment_id: 1,
            total_gc_runs: 0,
            total_bytes_reclaimed: 0,
            live_bytes: 0,
            garbage_bytes: 0,
        }
    }

    /// Serialize metadata to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .map_err(|e| ValueLogError::Serialization(format!("Failed to serialize metadata: {}", e)))
            .map(|buf| buf.to_vec())
    }

    pub fn to_file_bytes(&self) -> Result<Vec<u8>> {
        let payload = self.to_bytes()?;
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

    /// Deserialize metadata from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let min_size = std::mem::size_of::<ArchivedValueLogMetadata>();
        if bytes.len() < min_size {
            return Err(ValueLogError::Serialization(format!(
                "Value log metadata too small: {} < {}",
                bytes.len(),
                min_size
            )));
        }
        let archived = rkyv::access::<ArchivedValueLogMetadata, rkyv::rancor::Error>(bytes)
            .map_err(|e| ValueLogError::Serialization(format!("Value log metadata validation failed: {}", e)))?;
        Ok(ValueLogMetadata {
            head: archived.head.to_native(),
            tail: archived.tail.to_native(),
            current_page_offset: archived.current_page_offset.to_native(),
            current_page_free_offset: archived.current_page_free_offset.to_native(),
            current_page_table_offset: archived.current_page_table_offset.to_native(),
            current_page_next_segment_id: archived.current_page_next_segment_id.to_native(),
            total_gc_runs: archived.total_gc_runs.to_native(),
            total_bytes_reclaimed: archived.total_bytes_reclaimed.to_native(),
            live_bytes: archived.live_bytes.to_native(),
            garbage_bytes: archived.garbage_bytes.to_native(),
        })
    }

    pub fn from_file_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() >= 16 && bytes.get(0..4) == Some(&VALUE_LOG_METADATA_MAGIC) {
            let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
            let checksum = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
            let payload_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
            let payload_end = 16usize.saturating_add(payload_len);
            if bytes.len() < payload_end {
                return Err(ValueLogError::Serialization(format!(
                    "Value log metadata payload truncated: {} < {}",
                    bytes.len(),
                    payload_end
                )));
            }
            let payload = &bytes[16..payload_end];
            let mut hasher = Hasher::new();
            hasher.update(payload);
            let actual = hasher.finalize();
            if actual != checksum {
                return Err(ValueLogError::Serialization("Value log metadata checksum mismatch".to_string()));
            }
            if version == VALUE_LOG_METADATA_VERSION {
                return Self::from_bytes(payload);
            }
            if version == 1 {
                let archived = rkyv::access::<ArchivedValueLogMetadataV1, rkyv::rancor::Error>(payload)
                    .map_err(|e| ValueLogError::Serialization(format!("Value log metadata v1 validation failed: {}", e)))?;
                return Ok(ValueLogMetadata {
                    head: archived.head.to_native(),
                    tail: PAGE_SIZE_BYTES,
                    current_page_offset: 0,
                    current_page_free_offset: PAGE_HEADER_SIZE,
                    current_page_table_offset: PAGE_SIZE_BYTES as u32,
                    current_page_next_segment_id: 1,
                    total_gc_runs: archived.total_gc_runs.to_native(),
                    total_bytes_reclaimed: archived.total_bytes_reclaimed.to_native(),
                    live_bytes: 0,
                    garbage_bytes: 0,
                });
            }
            return Err(ValueLogError::Serialization(format!(
                "Unsupported value log metadata version: {}",
                version
            )));
        }

        Self::from_bytes(bytes)
    }
}

/// Value Log Manager
///
/// Handles all file I/O operations for the value log
pub struct ValueLog {
    // Value log file
    // Arc<RwLock<Arc<File>>> enables atomic pointer swaps during GC
    // This allows readers to hold their own Arc reference while new readers switch to new file
    file: Arc<RwLock<Arc<File>>>,

    // Seqlock-style generation counter for GC swaps. A GC compaction of this
    // bucket brackets BOTH the file swap and the subsequent LSM pointer re-point
    // with [`begin_swap`](Self::begin_swap) (→ odd) and the returned guard's drop
    // (→ even). An *even* value means "consistent"; an *odd* value means "a swap
    // is in progress, the LSM and the value-log file may disagree right now".
    //
    // Readers sample it before and after a pointer+value read and trust the read
    // only if the sample was even and unchanged. This is stronger than tracking
    // the file swap alone: the file swap and the LSM re-point are two separate
    // steps, so a counter bumped only at the file swap leaves a window where the
    // file is new but the LSM still holds the old pointer — a reader there pairs
    // a stale pointer with the new file and reads garbage. Bracketing both steps
    // closes that window. The counter never resets, so a completed swap (+2) can
    // never alias back to a prior even value.
    generation: AtomicU64,

    // When true, every value read re-verifies the record's CRC32 against the
    // value bytes. Off by default (latency first) — see `DbConfig::verify_checksums_on_read`.
    verify_checksums_on_read: AtomicBool,

    // Path to the value log file
    #[allow(dead_code)]
    path: PathBuf,
}

/// RAII guard for a GC swap epoch (see [`ValueLog::begin_swap`]). While alive,
/// the bucket's generation is odd ("swap in progress"); on drop it is bumped
/// back to even ("consistent"), so the epoch is closed even on an early return
/// or panic mid-swap.
pub struct SwapEpochGuard<'a> {
    value_log: &'a ValueLog,
}

impl Drop for SwapEpochGuard<'_> {
    fn drop(&mut self) {
        let prev = self.value_log.generation.fetch_add(1, Ordering::AcqRel);
        debug_assert!(
            !prev.is_multiple_of(2),
            "SwapEpochGuard drop on a bucket not mid-swap (generation {prev} was even)"
        );
    }
}

impl ValueLog {
    /// Open or create a value log
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new().create(true).truncate(false).read(true).write(true).open(&path)?;

        Ok(Self {
            file: Arc::new(RwLock::new(Arc::new(file))),
            generation: AtomicU64::new(0),
            verify_checksums_on_read: AtomicBool::new(false),
            path,
        })
    }

    /// Enable or disable re-verifying each value's CRC32 on read (default off).
    pub fn set_verify_checksums_on_read(&self, verify: bool) {
        self.verify_checksums_on_read.store(verify, Ordering::Relaxed);
    }

    #[inline]
    fn verify_checksums_on_read(&self) -> bool {
        self.verify_checksums_on_read.load(Ordering::Relaxed)
    }

    /// Current swap generation for this bucket (see [`generation`](Self::generation) field).
    ///
    /// Sampled by readers around a pointer+value read. The read is trustworthy
    /// only if this was **even** (no swap in progress) and **unchanged** across
    /// the read; an odd value, or any change, means a GC swap raced the read and
    /// it must be retried.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Begin a GC swap epoch for this bucket: bump the generation to odd ("swap
    /// in progress") and return a guard whose drop bumps it to even again. The
    /// caller must hold this guard across BOTH the value-log file swap and the
    /// LSM pointer re-point, so readers observe the two as a single atomic step.
    pub fn begin_swap(&self) -> SwapEpochGuard<'_> {
        let prev = self.generation.fetch_add(1, Ordering::AcqRel);
        debug_assert!(
            prev.is_multiple_of(2),
            "begin_swap on a bucket already mid-swap (generation {prev} was odd)"
        );
        SwapEpochGuard { value_log: self }
    }

    pub fn ensure_current_page(&self, metadata: &mut ValueLogMetadata) -> Result<()> {
        let file = self.get_file();
        let desired_len = metadata.tail.max(PAGE_SIZE_BYTES);
        if file.metadata()?.len() < desired_len {
            file.set_len(desired_len)?;
        }

        let header = self.read_page_header(&file, metadata.current_page_offset)?;
        if header.is_none() {
            self.write_page_header(&file, metadata.current_page_offset, PageHeader::new())?;
        } else if let Some(header) = header {
            metadata.current_page_free_offset = header.free_offset;
            metadata.current_page_table_offset = header.table_offset;
            metadata.current_page_next_segment_id = header.next_segment_id.max(1);
        }
        Ok(())
    }

    fn record_storage_bytes(total_len: u32) -> u64 {
        total_len as u64 + TABLE_ENTRY_SIZE as u64
    }

    /// Write a value record into the current page or a new page if needed.
    pub fn write_record(&self, value: &[u8], meta: ValueRecordMeta, metadata: &mut ValueLogMetadata, sync: bool) -> Result<ValueLocation> {
        self.ensure_current_page(metadata)?;

        let record_len = ValueRecordHeader::SIZE as u32 + value.len() as u32;
        let required = record_len.saturating_add(TABLE_ENTRY_SIZE);

        if metadata.current_page_free_offset.saturating_add(required) > metadata.current_page_table_offset {
            metadata.current_page_offset = metadata.tail;
            metadata.tail = metadata.tail.saturating_add(PAGE_SIZE_BYTES);
            metadata.current_page_free_offset = PAGE_HEADER_SIZE;
            metadata.current_page_table_offset = PAGE_SIZE_BYTES as u32;
            metadata.current_page_next_segment_id = 1;
            self.ensure_current_page(metadata)?;
        }

        if metadata.current_page_next_segment_id == 0 {
            metadata.current_page_next_segment_id = 1;
        }
        let segment_id = metadata.current_page_next_segment_id;
        metadata.current_page_next_segment_id = metadata.current_page_next_segment_id.saturating_add(1);

        let record_offset = metadata.current_page_free_offset;
        metadata.current_page_free_offset = metadata.current_page_free_offset.saturating_add(record_len);
        metadata.current_page_table_offset = metadata.current_page_table_offset.saturating_sub(TABLE_ENTRY_SIZE);

        let flags = (if meta.tombstone { FLAG_TOMBSTONE } else { 0 }) | (if meta.updated { FLAG_UPDATED } else { 0 });

        let header = ValueRecordHeader {
            total_len: record_len,
            version: meta.version,
            flags,
            epoch: meta.epoch,
            value_len: value.len() as u32,
            checksum: value_checksum(value),
            seq: meta.seq,
        };

        let file = self.get_file();
        self.ensure_page_len(&file, metadata.current_page_offset)?;

        let page_base = metadata.current_page_offset;

        // Write record header
        let abs_record = page_base + record_offset as u64;
        Self::write_all_at(&file, &header.to_bytes(), abs_record)?;

        // Write value
        let abs_value = abs_record + ValueRecordHeader::SIZE as u64;
        Self::write_all_at(&file, value, abs_value)?;

        // Write table entry (segment_id + record_offset, 8 bytes)
        let abs_table_entry = page_base + metadata.current_page_table_offset as u64;
        let mut table_entry = [0u8; TABLE_ENTRY_SIZE as usize];
        table_entry[0..4].copy_from_slice(&segment_id.to_le_bytes());
        table_entry[4..8].copy_from_slice(&record_offset.to_le_bytes());
        Self::write_all_at(&file, &table_entry, abs_table_entry)?;

        // Write updated page header
        let page_header = PageHeader {
            magic: PAGE_HEADER_MAGIC,
            version: PAGE_HEADER_VERSION,
            free_offset: metadata.current_page_free_offset,
            table_offset: metadata.current_page_table_offset,
            next_segment_id: metadata.current_page_next_segment_id,
            page_size: PAGE_SIZE_BYTES as u32,
            reserved0: 0,
            reserved1: 0,
        };
        Self::write_all_at(&file, &page_header.to_bytes(), page_base)?;

        if sync {
            file.sync_data()?;
        }

        let record_bytes = Self::record_storage_bytes(record_len);
        if meta.tombstone || meta.updated {
            metadata.garbage_bytes = metadata.garbage_bytes.saturating_add(record_bytes);
        } else {
            metadata.live_bytes = metadata.live_bytes.saturating_add(record_bytes);
        }

        Ok(ValueLocation {
            page_offset: metadata.current_page_offset,
            segment_id,
        })
    }

    /// Read a value from a specific file handle (used for snapshot reads)
    pub fn read_value_from_file(&self, file: &File, location: ValueLocation) -> Result<Vec<u8>> {
        Ok(self.read_value_and_seq_from_file(file, location)?.0)
    }

    /// Like [`read_value_from_file`](Self::read_value_from_file) but also returns
    /// the record's stored write `seq`, so a reader can verify the resolved
    /// pointer still refers to the write the LSM associated with it (a recycled
    /// slot carries a different, globally-unique seq).
    pub fn read_value_and_seq_from_file(&self, file: &File, location: ValueLocation) -> Result<(Vec<u8>, u64)> {
        if location.segment_id == 0 {
            return Err(ValueLogError::InvalidLocation);
        }
        // 1. Read page header for table_offset validation.
        let mut hdr_buf = [0u8; PAGE_HEADER_SIZE as usize];
        file.read_at(&mut hdr_buf, location.page_offset).map_err(ValueLogError::Io)?;
        let page_header = PageHeader::from_bytes(&hdr_buf).ok_or(ValueLogError::CorruptedLog)?;
        // 2. Compute and validate the table entry position.
        let entry_offset = (PAGE_SIZE_BYTES as usize)
            .checked_sub(location.segment_id as usize * TABLE_ENTRY_SIZE as usize)
            .ok_or(ValueLogError::InvalidLocation)?;
        if entry_offset < page_header.table_offset as usize {
            return Err(ValueLogError::InvalidLocation);
        }
        let abs_entry_offset = location.page_offset + entry_offset as u64;
        let mut entry_buf = [0u8; TABLE_ENTRY_SIZE as usize];
        file.read_at(&mut entry_buf, abs_entry_offset).map_err(ValueLogError::Io)?;
        let stored_segment = u32::from_le_bytes(entry_buf[0..4].try_into().unwrap());
        if stored_segment != location.segment_id {
            return Err(ValueLogError::InvalidLocation);
        }
        let record_offset = u32::from_le_bytes(entry_buf[4..8].try_into().unwrap());
        // 3. Read record header.
        let abs_record_offset = location.page_offset + record_offset as u64;
        let mut rec_hdr_buf = [0u8; ValueRecordHeader::SIZE];
        file.read_at(&mut rec_hdr_buf, abs_record_offset).map_err(ValueLogError::Io)?;
        let rec_header = ValueRecordHeader::from_bytes(&rec_hdr_buf).ok_or(ValueLogError::CorruptedLog)?;
        // 4. Read value bytes.
        let value_abs_offset = abs_record_offset + ValueRecordHeader::SIZE as u64;
        let mut value = vec![0u8; rec_header.value_len as usize];
        file.read_at(&mut value, value_abs_offset).map_err(ValueLogError::Io)?;
        if self.verify_checksums_on_read() && value_checksum(&value) != rec_header.checksum {
            return Err(ValueLogError::CorruptedLog);
        }
        Ok((value, rec_header.seq))
    }

    /// Read a value from the value log
    #[allow(dead_code)]
    pub fn read_value(&self, location: ValueLocation) -> Result<Vec<u8>> {
        let file = self.get_file();
        self.read_value_from_file(&file, location)
    }

    /// Read multiple values from the same file handle efficiently.
    ///
    /// Groups entries by page and issues:
    /// - 1 pread for the page header (shared by all entries on the same page)
    /// - 1 pread for all slot-directory entries in the segment_id range
    /// - 1 speculative pread per entry that reads record-header + value together
    ///
    /// This reduces pread() calls from 4×N (individual) to ≈ 2 + N per page group.
    /// Values larger than SPECULATIVE_READ_SIZE fall back to a second direct read.
    /// Returns `(orig_idx, Some((value, record_seq)))` for each readable entry, or
    /// `(orig_idx, None)` when the slot is empty/unreadable. `record_seq` lets the
    /// caller verify the slot still holds the write the pointer referred to.
    pub fn read_values_batch_from_file(&self, file: &File, entries: &[(usize, ValueLocation)]) -> Vec<BatchValue> {
        // Buffer size for speculative record+value reads.
        // Dense vectors (768-dim × 8 bits ≈ 800 bytes + rkyv overhead) always fit.
        const SPECULATIVE_READ_SIZE: usize = 1280;

        let mut results: Vec<BatchValue> = Vec::with_capacity(entries.len());
        if entries.is_empty() {
            return results;
        }
        let verify = self.verify_checksums_on_read();

        // Group by page_offset — almost always a single page per bucket batch.
        let mut by_page: std::collections::HashMap<u64, Vec<(usize, u32)>> = std::collections::HashMap::new();
        for &(orig_idx, location) in entries {
            if location.segment_id == 0 {
                results.push((orig_idx, None));
                continue;
            }
            by_page.entry(location.page_offset).or_default().push((orig_idx, location.segment_id));
        }

        for (page_offset, page_entries) in by_page {
            // ── Read 1: page header once per page ────────────────────────────
            let mut hdr_buf = [0u8; PAGE_HEADER_SIZE as usize];
            if file.read_at(&mut hdr_buf, page_offset).is_err() {
                results.extend(page_entries.iter().map(|(idx, _)| (*idx, None)));
                continue;
            }
            let page_header = match PageHeader::from_bytes(&hdr_buf) {
                Some(h) => h,
                None => {
                    results.extend(page_entries.iter().map(|(idx, _)| (*idx, None)));
                    continue;
                }
            };

            // ── Read 2: bulk slot-directory read ─────────────────────────────
            // Table entries grow from the end of the page downward.
            // segment_id K → file offset page_offset + PAGE_SIZE - K * TABLE_ENTRY_SIZE.
            // max_seg has the lowest file offset; min_seg has the highest.
            let max_seg = page_entries.iter().map(|(_, s)| *s).max().unwrap();
            let min_seg = page_entries.iter().map(|(_, s)| *s).min().unwrap();
            let entry_offset_for_max = (PAGE_SIZE_BYTES as usize).saturating_sub(max_seg as usize * TABLE_ENTRY_SIZE as usize);
            if entry_offset_for_max < page_header.table_offset as usize {
                results.extend(page_entries.iter().map(|(idx, _)| (*idx, None)));
                continue;
            }
            let slot_count = (max_seg - min_seg + 1) as usize;
            let slot_buf_size = slot_count * TABLE_ENTRY_SIZE as usize;
            let slot_buf_file_offset = page_offset + entry_offset_for_max as u64;
            let mut slot_buf = vec![0u8; slot_buf_size];
            if file.read_at(&mut slot_buf, slot_buf_file_offset).is_err() {
                results.extend(page_entries.iter().map(|(idx, _)| (*idx, None)));
                continue;
            }

            // ── Read 3: speculative record+value read per entry ───────────────
            for (orig_idx, segment_id) in page_entries {
                let slot_idx = (max_seg - segment_id) as usize;
                let slot_start = slot_idx * TABLE_ENTRY_SIZE as usize;
                if slot_start + TABLE_ENTRY_SIZE as usize > slot_buf.len() {
                    results.push((orig_idx, None));
                    continue;
                }
                let stored_segment = u32::from_le_bytes(slot_buf[slot_start..slot_start + 4].try_into().unwrap());
                if stored_segment != segment_id {
                    results.push((orig_idx, None));
                    continue;
                }
                let record_offset = u32::from_le_bytes(slot_buf[slot_start + 4..slot_start + 8].try_into().unwrap());
                let abs_record_offset = page_offset + record_offset as u64;

                let mut spec_buf = [0u8; ValueRecordHeader::SIZE + SPECULATIVE_READ_SIZE];
                let n_read = file.read_at(&mut spec_buf, abs_record_offset).unwrap_or(0);
                if n_read < ValueRecordHeader::SIZE {
                    results.push((orig_idx, None));
                    continue;
                }
                let rec_header = match ValueRecordHeader::from_bytes(&spec_buf[..ValueRecordHeader::SIZE]) {
                    Some(h) => h,
                    None => {
                        results.push((orig_idx, None));
                        continue;
                    }
                };
                let value_len = rec_header.value_len as usize;
                let value_start = ValueRecordHeader::SIZE;
                let value_end = value_start + value_len;
                let value = if value_end <= n_read {
                    spec_buf[value_start..value_end].to_vec()
                } else {
                    // Slow path: value larger than speculative buffer
                    let abs_value_offset = abs_record_offset + ValueRecordHeader::SIZE as u64;
                    let mut buf = vec![0u8; value_len];
                    match file.read_at(&mut buf, abs_value_offset) {
                        Ok(_) => buf,
                        Err(_) => {
                            results.push((orig_idx, None));
                            continue;
                        }
                    }
                };
                if verify && value_checksum(&value) != rec_header.checksum {
                    results.push((orig_idx, None));
                    continue;
                }
                results.push((orig_idx, Some((value, rec_header.seq))));
            }
        }

        results
    }

    pub fn read_record_meta_from_file(&self, file: &File, location: ValueLocation) -> Result<ValueRecordMeta> {
        if location.segment_id == 0 {
            return Err(ValueLogError::InvalidLocation);
        }
        let mut hdr_buf = [0u8; PAGE_HEADER_SIZE as usize];
        file.read_at(&mut hdr_buf, location.page_offset).map_err(ValueLogError::Io)?;
        let page_header = PageHeader::from_bytes(&hdr_buf).ok_or(ValueLogError::CorruptedLog)?;
        let entry_offset = (PAGE_SIZE_BYTES as usize)
            .checked_sub(location.segment_id as usize * TABLE_ENTRY_SIZE as usize)
            .ok_or(ValueLogError::InvalidLocation)?;
        if entry_offset < page_header.table_offset as usize {
            return Err(ValueLogError::InvalidLocation);
        }
        let abs_entry_offset = location.page_offset + entry_offset as u64;
        let mut entry_buf = [0u8; TABLE_ENTRY_SIZE as usize];
        file.read_at(&mut entry_buf, abs_entry_offset).map_err(ValueLogError::Io)?;
        let stored_segment = u32::from_le_bytes(entry_buf[0..4].try_into().unwrap());
        if stored_segment != location.segment_id {
            return Err(ValueLogError::InvalidLocation);
        }
        let record_offset = u32::from_le_bytes(entry_buf[4..8].try_into().unwrap());
        let abs_record_offset = location.page_offset + record_offset as u64;
        let mut rec_hdr_buf = [0u8; ValueRecordHeader::SIZE];
        file.read_at(&mut rec_hdr_buf, abs_record_offset).map_err(ValueLogError::Io)?;
        let rec_header = ValueRecordHeader::from_bytes(&rec_hdr_buf).ok_or(ValueLogError::CorruptedLog)?;
        Ok(rec_header.meta())
    }

    pub fn update_record_meta(
        &self,
        location: ValueLocation,
        tombstone: Option<bool>,
        updated: Option<bool>,
        epoch: Option<u64>,
    ) -> Result<(ValueRecordMeta, ValueRecordMeta, u64)> {
        let file = self.get_file();

        if location.segment_id == 0 {
            return Err(ValueLogError::InvalidLocation);
        }
        // Read page header for table_offset
        let mut hdr_buf = [0u8; PAGE_HEADER_SIZE as usize];
        file.read_at(&mut hdr_buf, location.page_offset).map_err(ValueLogError::Io)?;
        let page_header = PageHeader::from_bytes(&hdr_buf).ok_or(ValueLogError::CorruptedLog)?;

        // Look up record offset from translation table
        let entry_offset = (PAGE_SIZE_BYTES as usize)
            .checked_sub(location.segment_id as usize * TABLE_ENTRY_SIZE as usize)
            .ok_or(ValueLogError::InvalidLocation)?;
        if entry_offset < page_header.table_offset as usize {
            return Err(ValueLogError::InvalidLocation);
        }
        let abs_entry_offset = location.page_offset + entry_offset as u64;
        let mut entry_buf = [0u8; TABLE_ENTRY_SIZE as usize];
        file.read_at(&mut entry_buf, abs_entry_offset).map_err(ValueLogError::Io)?;
        let stored_segment = u32::from_le_bytes(entry_buf[0..4].try_into().unwrap());
        if stored_segment != location.segment_id {
            return Err(ValueLogError::InvalidLocation);
        }
        let record_offset = u32::from_le_bytes(entry_buf[4..8].try_into().unwrap());

        // Read existing record header
        let abs_record_offset = location.page_offset + record_offset as u64;
        let mut rec_hdr_buf = [0u8; ValueRecordHeader::SIZE];
        file.read_at(&mut rec_hdr_buf, abs_record_offset).map_err(ValueLogError::Io)?;
        let header = ValueRecordHeader::from_bytes(&rec_hdr_buf).ok_or(ValueLogError::CorruptedLog)?;

        let previous_meta = header.meta();
        let mut flags = header.flags;
        if let Some(t) = tombstone {
            if t {
                flags |= FLAG_TOMBSTONE;
            } else {
                flags &= !FLAG_TOMBSTONE;
            }
        }
        if let Some(u) = updated {
            if u {
                flags |= FLAG_UPDATED;
            } else {
                flags &= !FLAG_UPDATED;
            }
        }
        let new_epoch = epoch.unwrap_or(header.epoch);

        let updated_header = ValueRecordHeader {
            total_len: header.total_len,
            version: header.version,
            flags,
            epoch: new_epoch,
            value_len: header.value_len,
            checksum: header.checksum,
            seq: header.seq,
        };

        // Write modified header back in-place
        Self::write_all_at(&file, &updated_header.to_bytes(), abs_record_offset)?;

        let record_bytes = Self::record_storage_bytes(header.total_len);
        Ok((previous_meta, updated_header.meta(), record_bytes))
    }

    /// Get the current file reference
    pub fn get_file(&self) -> Arc<File> {
        Arc::clone(&*self.file.read())
    }

    /// Swap in a freshly-compacted file (used during GC).
    ///
    /// The generation is **not** bumped here — the swap epoch is owned by the
    /// [`begin_swap`](Self::begin_swap) guard the caller holds across both this
    /// swap and the subsequent LSM re-point, so readers treat the pair as one
    /// atomic step. This must only be called while that guard is held.
    pub fn swap_file(&self, new_file: Arc<File>) -> Arc<File> {
        let mut guard = self.file.write();
        std::mem::replace(&mut *guard, new_file)
    }

    /// Flush and sync the value log to disk
    pub fn sync(&self) -> Result<()> {
        let file = self.file.read();
        file.sync_all()?;
        Ok(())
    }

    /// Get the path to the value log
    #[allow(dead_code)]
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn ensure_page_len(&self, file: &File, page_offset: u64) -> Result<()> {
        let expected = page_offset.saturating_add(PAGE_SIZE_BYTES);
        if file.metadata()?.len() < expected {
            file.set_len(expected)?;
        }
        Ok(())
    }

    fn write_all_at(file: &File, mut buf: &[u8], mut offset: u64) -> Result<()> {
        while !buf.is_empty() {
            let n = file.write_at(buf, offset).map_err(ValueLogError::Io)?;
            if n == 0 {
                return Err(ValueLogError::Io(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "write_at wrote 0 bytes",
                )));
            }
            offset += n as u64;
            buf = &buf[n..];
        }
        Ok(())
    }

    fn read_page_header(&self, file: &File, page_offset: u64) -> Result<Option<PageHeader>> {
        if file.metadata()?.len() < page_offset.saturating_add(PAGE_HEADER_SIZE as u64) {
            return Ok(None);
        }
        let mut buf = [0u8; PAGE_HEADER_SIZE as usize];
        file.read_at(&mut buf, page_offset).map_err(ValueLogError::Io)?;
        Ok(PageHeader::from_bytes(&buf))
    }

    fn write_page_header(&self, file: &File, page_offset: u64, header: PageHeader) -> Result<()> {
        self.ensure_page_len(file, page_offset)?;
        Self::write_all_at(file, &header.to_bytes(), page_offset)
    }

    /// Scan a single page and compute garbage statistics by walking the segment table.
    pub fn scan_page_stats(&self, file: &File, page_offset: u64) -> Result<PageGarbageStats> {
        let page_header = self.read_page_header(file, page_offset)?.ok_or(ValueLogError::CorruptedLog)?;

        let mut stats = PageGarbageStats {
            page_offset,
            live_bytes: 0,
            garbage_bytes: 0,
            total_records: 0,
            garbage_records: 0,
        };

        // Walk the segment table from table_offset to PAGE_SIZE_BYTES in TABLE_ENTRY_SIZE steps.
        // Each entry is 8 bytes: [segment_id: u32, record_offset: u32].
        let mut table_pos = page_header.table_offset as u64;
        while table_pos + TABLE_ENTRY_SIZE as u64 <= PAGE_SIZE_BYTES {
            let abs_entry = page_offset + table_pos;
            let mut entry_buf = [0u8; TABLE_ENTRY_SIZE as usize];
            file.read_at(&mut entry_buf, abs_entry).map_err(ValueLogError::Io)?;

            let _segment_id = u32::from_le_bytes(entry_buf[0..4].try_into().unwrap());
            let record_offset = u32::from_le_bytes(entry_buf[4..8].try_into().unwrap());

            // Read record header at the record offset within this page
            let abs_record = page_offset + record_offset as u64;
            let mut rec_hdr_buf = [0u8; ValueRecordHeader::SIZE];
            file.read_at(&mut rec_hdr_buf, abs_record).map_err(ValueLogError::Io)?;

            if let Some(header) = ValueRecordHeader::from_bytes(&rec_hdr_buf) {
                let record_bytes = Self::record_storage_bytes(header.total_len);
                let is_garbage = (header.flags & FLAG_TOMBSTONE) != 0 || (header.flags & FLAG_UPDATED) != 0;

                stats.total_records += 1;
                if is_garbage {
                    stats.garbage_bytes += record_bytes;
                    stats.garbage_records += 1;
                } else {
                    stats.live_bytes += record_bytes;
                }
            }

            table_pos += TABLE_ENTRY_SIZE as u64;
        }

        Ok(stats)
    }

    /// Scan all pages in this value log and return per-page garbage stats.
    pub fn scan_all_page_stats(&self, file: &File, metadata: &ValueLogMetadata) -> Result<Vec<PageGarbageStats>> {
        let mut all_stats = Vec::new();
        let mut page_offset = 0u64;

        // Iterate through all pages up to and including the current page.
        //
        // A page without a valid header is a hole — a prior GC rewrote a dirty
        // page's live records to fresh pages further on and left this slot empty.
        // SKIP such pages and keep scanning; stopping at the first hole would hide
        // every page beyond it, so GC would stop finding garbage and never reclaim
        // again (unbounded value-log growth).
        while page_offset <= metadata.current_page_offset {
            if let Ok(stats) = self.scan_page_stats(file, page_offset) {
                all_stats.push(stats);
            }
            page_offset += PAGE_SIZE_BYTES;
        }

        Ok(all_stats)
    }

    /// Rebuild this bucket's metadata by scanning its value-log pages.
    ///
    /// Used at open when the persisted metadata is missing or corrupt. Resetting
    /// to [`ValueLogMetadata::new`] there would be unsafe in two ways: it zeroes
    /// the live/garbage byte counters (so GC's waste ratio reads 0 and never
    /// triggers — garbage grows unbounded), and it resets `current_page_offset`
    /// to 0, so once the current page fills a *new* page would be allocated at
    /// `tail` (= one page in) **over existing data**. Scanning rebuilds both the
    /// byte counts and the page cursors from the on-disk page headers.
    ///
    /// Pages are contiguous from offset 0, so the scan stops at the first slot
    /// without a valid header (the sparse tail left by `set_len`). Cost is
    /// O(pages × records), but this runs only on the rare metadata-loss path.
    ///
    /// `head`, `total_gc_runs` and `total_bytes_reclaimed` are observability-only
    /// cumulative counters that cannot be derived from the pages and reset to 0.
    pub fn reconstruct_metadata(&self) -> Result<ValueLogMetadata> {
        let file = self.get_file();
        let file_len = file.metadata()?.len();

        let mut meta = ValueLogMetadata::new();
        let mut live_bytes = 0u64;
        let mut garbage_bytes = 0u64;
        let mut last_valid_page: Option<u64> = None;

        let mut page_offset = 0u64;
        while page_offset.saturating_add(PAGE_HEADER_SIZE as u64) <= file_len {
            if self.read_page_header(&file, page_offset)?.is_none() {
                break;
            }
            let stats = self.scan_page_stats(&file, page_offset)?;
            live_bytes = live_bytes.saturating_add(stats.live_bytes);
            garbage_bytes = garbage_bytes.saturating_add(stats.garbage_bytes);
            last_valid_page = Some(page_offset);
            page_offset = page_offset.saturating_add(PAGE_SIZE_BYTES);
        }

        if let Some(last) = last_valid_page {
            let header = self.read_page_header(&file, last)?.ok_or(ValueLogError::CorruptedLog)?;
            meta.current_page_offset = last;
            meta.tail = last.saturating_add(PAGE_SIZE_BYTES);
            meta.current_page_free_offset = header.free_offset;
            meta.current_page_table_offset = header.table_offset;
            meta.current_page_next_segment_id = header.next_segment_id.max(1);
            meta.live_bytes = live_bytes;
            meta.garbage_bytes = garbage_bytes;
        }
        // An empty file leaves the `ValueLogMetadata::new` defaults, which are
        // correct for a fresh bucket.

        Ok(meta)
    }

    /// Copy a raw 64MB page from one file to another at the same offset.
    pub fn copy_page_raw(src_file: &File, dst_file: &File, page_offset: u64) -> Result<()> {
        // Copy in 64KB chunks to avoid huge stack/heap allocations
        const CHUNK_SIZE: usize = 64 * 1024;
        let mut buf = vec![0u8; CHUNK_SIZE];
        let mut offset = page_offset;
        let end = page_offset + PAGE_SIZE_BYTES;

        // Ensure destination file is large enough
        let expected_len = end;
        if dst_file.metadata()?.len() < expected_len {
            dst_file.set_len(expected_len)?;
        }

        while offset < end {
            let to_read = CHUNK_SIZE.min((end - offset) as usize);
            let buf_slice = &mut buf[..to_read];
            src_file.read_at(buf_slice, offset).map_err(ValueLogError::Io)?;
            Self::write_all_at(dst_file, buf_slice, offset)?;
            offset += to_read as u64;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn generation_is_a_seqlock_around_each_swap() {
        // The swap generation is the signal readers use to detect a GC swap that
        // raced their pointer+value read (see `KVStore::get`). It is a seqlock:
        // `begin_swap` makes it odd ("swap in progress, don't trust a read"), and
        // the guard's drop makes it even ("consistent") — so each full swap adds
        // two and never resets, and an odd value is observable mid-swap.
        let dir = TempDir::new().unwrap();
        let vl = ValueLog::open(dir.path().join("vl.log")).unwrap();
        assert_eq!(vl.generation(), 0, "fresh value log must start at generation 0 (even = consistent)");

        {
            let _epoch = vl.begin_swap();
            assert_eq!(vl.generation(), 1, "begin_swap must make the generation odd (swap in progress)");
            let f = vl.get_file();
            vl.swap_file(f); // swap no longer touches the generation
            assert_eq!(vl.generation(), 1, "swap_file must not change the generation");
        }
        assert_eq!(vl.generation(), 2, "closing the epoch must make the generation even again");

        {
            let _epoch = vl.begin_swap();
            assert_eq!(vl.generation(), 3, "second swap must go odd at 3");
        }
        assert_eq!(vl.generation(), 4, "second full swap must leave the generation even at 4");
    }

    #[test]
    fn test_value_pointer_serialization() -> Result<()> {
        let location = ValueLocation {
            page_offset: 0,
            segment_id: 10,
        };
        assert_eq!(location.page_offset, 0);
        assert_eq!(location.segment_id, 10);
        Ok(())
    }

    #[test]
    fn test_value_log_metadata_serialization() -> Result<()> {
        let metadata = ValueLogMetadata {
            head: 0,
            tail: 1024,
            current_page_offset: 0,
            current_page_free_offset: PAGE_HEADER_SIZE,
            current_page_table_offset: PAGE_SIZE_BYTES as u32,
            current_page_next_segment_id: 3,
            total_gc_runs: 5,
            total_bytes_reclaimed: 512,
            live_bytes: 256,
            garbage_bytes: 128,
        };

        let bytes = metadata.to_bytes()?;
        let restored = ValueLogMetadata::from_bytes(&bytes)?;

        assert_eq!(metadata.head, restored.head);
        assert_eq!(metadata.tail, restored.tail);
        assert_eq!(metadata.current_page_offset, restored.current_page_offset);
        assert_eq!(metadata.current_page_free_offset, restored.current_page_free_offset);
        assert_eq!(metadata.current_page_table_offset, restored.current_page_table_offset);
        assert_eq!(metadata.current_page_next_segment_id, restored.current_page_next_segment_id);
        assert_eq!(metadata.total_gc_runs, restored.total_gc_runs);
        assert_eq!(metadata.total_bytes_reclaimed, restored.total_bytes_reclaimed);
        assert_eq!(metadata.live_bytes, restored.live_bytes);
        assert_eq!(metadata.garbage_bytes, restored.garbage_bytes);
        Ok(())
    }

    #[test]
    fn test_reconstruct_metadata_matches_tracked_counters() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let log = ValueLog::open(temp_dir.path().join("recon.log"))?;

        // Write a mix of live and garbage (tombstoned) records, letting the
        // in-memory metadata track the counters as the source of truth.
        let mut tracked = ValueLogMetadata::new();
        log.ensure_current_page(&mut tracked)?;
        let live = ValueRecordMeta {
            version: 1,
            tombstone: false,
            updated: false,
            epoch: 0,
            seq: 0,
        };
        let garbage = ValueRecordMeta {
            version: 1,
            tombstone: true,
            updated: false,
            epoch: 0,
            seq: 0,
        };
        for i in 0u8..12 {
            let meta = if i % 3 == 0 { garbage } else { live };
            log.write_record(&[i; 80], meta, &mut tracked, false)?;
        }
        assert!(tracked.live_bytes > 0 && tracked.garbage_bytes > 0);

        // Rebuilding from the pages alone must reproduce the byte counts and cursors.
        let rebuilt = log.reconstruct_metadata()?;
        assert_eq!(rebuilt.live_bytes, tracked.live_bytes, "live_bytes");
        assert_eq!(rebuilt.garbage_bytes, tracked.garbage_bytes, "garbage_bytes");
        assert_eq!(rebuilt.current_page_offset, tracked.current_page_offset);
        assert_eq!(rebuilt.current_page_free_offset, tracked.current_page_free_offset);
        assert_eq!(rebuilt.current_page_table_offset, tracked.current_page_table_offset);
        assert_eq!(rebuilt.current_page_next_segment_id, tracked.current_page_next_segment_id);
        assert_eq!(rebuilt.tail, tracked.tail);
        Ok(())
    }

    #[test]
    fn test_reconstruct_metadata_empty_file_is_fresh() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let log = ValueLog::open(temp_dir.path().join("empty.log"))?;
        let rebuilt = log.reconstruct_metadata()?;
        let fresh = ValueLogMetadata::new();
        assert_eq!(rebuilt.current_page_offset, fresh.current_page_offset);
        assert_eq!(rebuilt.tail, fresh.tail);
        assert_eq!(rebuilt.live_bytes, 0);
        assert_eq!(rebuilt.garbage_bytes, 0);
        Ok(())
    }

    #[test]
    fn test_value_log_write_read() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let log = ValueLog::open(temp_dir.path().join("test.log"))?;

        let mut metadata = ValueLogMetadata::new();
        log.ensure_current_page(&mut metadata)?;
        let value = b"test_value";

        let location = log.write_record(
            &value[..],
            ValueRecordMeta {
                version: 1,
                tombstone: false,
                updated: false,
                epoch: 0,
                seq: 0,
            },
            &mut metadata,
            false,
        )?;
        let read_value = log.read_value(location)?;

        assert_eq!(value, &read_value[..]);
        Ok(())
    }

    #[test]
    fn test_value_log_detects_corrupted_value() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let log_path = temp_dir.path().join("test.log");
        let location = {
            let log = ValueLog::open(&log_path)?;
            let mut metadata = ValueLogMetadata::new();
            log.ensure_current_page(&mut metadata)?;
            log.write_record(
                b"a_value_long_enough_to_corrupt",
                ValueRecordMeta {
                    version: 1,
                    tombstone: false,
                    updated: false,
                    epoch: 0,
                    seq: 0,
                },
                &mut metadata,
                true,
            )?
        };

        // Flip a byte in the value payload directly on disk. The first record
        // sits at page header + record header on the first page.
        let value_offset = location.page_offset + PAGE_HEADER_SIZE as u64 + ValueRecordHeader::SIZE as u64;
        let f = OpenOptions::new().read(true).write(true).open(&log_path)?;
        let mut byte = [0u8; 1];
        f.read_at(&mut byte, value_offset)?;
        byte[0] ^= 0xFF;
        f.write_at(&byte, value_offset)?;
        f.sync_all()?;
        drop(f);

        let log = ValueLog::open(&log_path)?;
        log.set_verify_checksums_on_read(true);
        match log.read_value(location) {
            Err(ValueLogError::CorruptedLog) => {}
            other => panic!("expected CorruptedLog on a flipped value byte, got {:?}", other),
        }

        // With verification off (the default), the corrupt value is returned as-is.
        log.set_verify_checksums_on_read(false);
        assert!(log.read_value(location).is_ok(), "verification off should not error on corruption");
        Ok(())
    }

    #[test]
    fn test_value_log_multiple_writes() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let log = ValueLog::open(temp_dir.path().join("test.log"))?;

        let mut metadata = ValueLogMetadata::new();
        log.ensure_current_page(&mut metadata)?;
        let values: Vec<&[u8]> = vec![b"value1", b"value2", b"value3"];
        let mut locations = vec![];

        for value in &values {
            let location = log.write_record(
                value,
                ValueRecordMeta {
                    version: 1,
                    tombstone: false,
                    updated: false,
                    epoch: 0,
                    seq: 0,
                },
                &mut metadata,
                false,
            )?;
            locations.push(location);
        }

        for (i, value) in values.iter().enumerate() {
            let read_value = log.read_value(locations[i])?;
            assert_eq!(value.to_vec(), read_value);
        }

        Ok(())
    }

    #[test]
    fn test_value_log_metadata_file_encoding() -> Result<()> {
        let metadata = ValueLogMetadata {
            head: 0,
            tail: 1024,
            current_page_offset: 0,
            current_page_free_offset: PAGE_HEADER_SIZE,
            current_page_table_offset: PAGE_SIZE_BYTES as u32,
            current_page_next_segment_id: 1,
            total_gc_runs: 5,
            total_bytes_reclaimed: 512,
            live_bytes: 256,
            garbage_bytes: 128,
        };

        let bytes = metadata.to_file_bytes()?;
        let restored = ValueLogMetadata::from_file_bytes(&bytes)?;

        assert_eq!(metadata.head, restored.head);
        assert_eq!(metadata.tail, restored.tail);
        assert_eq!(metadata.current_page_offset, restored.current_page_offset);
        assert_eq!(metadata.current_page_free_offset, restored.current_page_free_offset);
        assert_eq!(metadata.current_page_table_offset, restored.current_page_table_offset);
        assert_eq!(metadata.current_page_next_segment_id, restored.current_page_next_segment_id);
        assert_eq!(metadata.total_gc_runs, restored.total_gc_runs);
        assert_eq!(metadata.total_bytes_reclaimed, restored.total_bytes_reclaimed);
        assert_eq!(metadata.live_bytes, restored.live_bytes);
        assert_eq!(metadata.garbage_bytes, restored.garbage_bytes);
        Ok(())
    }

    fn write_value_helper(log: &ValueLog, meta: &mut ValueLogMetadata, value: &[u8]) -> Result<ValueLocation> {
        log.write_record(
            value,
            ValueRecordMeta {
                version: 1,
                tombstone: false,
                updated: false,
                epoch: 0,
                seq: 0,
            },
            meta,
            false,
        )
    }

    #[test]
    fn test_batch_read_matches_individual() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let log = ValueLog::open(temp_dir.path().join("test.log"))?;
        let mut meta = ValueLogMetadata::new();
        log.ensure_current_page(&mut meta)?;

        let values: Vec<Vec<u8>> = (0u8..10).map(|i| vec![i; (i as usize + 1) * 8]).collect();
        let mut locations = Vec::new();
        for v in &values {
            locations.push(write_value_helper(&log, &mut meta, v)?);
        }

        let file = log.get_file();
        let entries: Vec<(usize, ValueLocation)> = locations.iter().enumerate().map(|(i, &loc)| (i, loc)).collect();
        let mut batch = log.read_values_batch_from_file(&file, &entries);
        batch.sort_by_key(|(idx, _)| *idx);

        assert_eq!(batch.len(), values.len());
        for (idx, val_opt) in batch {
            assert_eq!(val_opt.map(|(v, _)| v).as_deref(), Some(values[idx].as_slice()));
        }
        Ok(())
    }

    #[test]
    fn test_batch_read_empty_input() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let log = ValueLog::open(temp_dir.path().join("test.log"))?;
        let mut meta = ValueLogMetadata::new();
        log.ensure_current_page(&mut meta)?;

        let file = log.get_file();
        let result = log.read_values_batch_from_file(&file, &[]);
        assert!(result.is_empty());
        Ok(())
    }

    #[test]
    fn test_batch_read_segment_id_zero_returns_none() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let log = ValueLog::open(temp_dir.path().join("test.log"))?;
        let mut meta = ValueLogMetadata::new();
        log.ensure_current_page(&mut meta)?;

        // Write one real entry and mix in an invalid location (segment_id=0)
        let loc_real = write_value_helper(&log, &mut meta, b"hello")?;
        let loc_bad = ValueLocation {
            page_offset: loc_real.page_offset,
            segment_id: 0,
        };

        let file = log.get_file();
        let entries = vec![(0usize, loc_bad), (1usize, loc_real)];
        let mut batch = log.read_values_batch_from_file(&file, &entries);
        batch.sort_by_key(|(idx, _)| *idx);

        assert_eq!(batch[0], (0, None));
        assert_eq!(batch[1].0, 1);
        assert_eq!(batch[1].1.as_ref().map(|(v, _)| v.as_slice()), Some(b"hello".as_slice()));
        Ok(())
    }

    #[test]
    fn test_batch_read_large_value_fallback() -> Result<()> {
        // Values > 1280 bytes exercise the slow-path fallback read in read_values_batch_from_file.
        let temp_dir = TempDir::new()?;
        let log = ValueLog::open(temp_dir.path().join("test.log"))?;
        let mut meta = ValueLogMetadata::new();
        log.ensure_current_page(&mut meta)?;

        let small = b"tiny".to_vec();
        let large = vec![0xABu8; 2048]; // exceeds SPECULATIVE_READ_SIZE=1280
        let loc_small = write_value_helper(&log, &mut meta, &small)?;
        let loc_large = write_value_helper(&log, &mut meta, &large)?;

        let file = log.get_file();
        let entries = vec![(0usize, loc_small), (1usize, loc_large)];
        let mut batch = log.read_values_batch_from_file(&file, &entries);
        batch.sort_by_key(|(idx, _)| *idx);

        assert_eq!(batch[0].1.as_ref().map(|(v, _)| v.as_slice()), Some(small.as_slice()));
        assert_eq!(batch[1].1.as_ref().map(|(v, _)| v.as_slice()), Some(large.as_slice()));
        Ok(())
    }

    #[test]
    fn test_batch_read_non_contiguous_segment_ids() -> Result<()> {
        // Write 20 values, read every other one — segment_id range has gaps, exercising
        // the bulk slot-directory read that covers the full max_seg..min_seg span.
        let temp_dir = TempDir::new()?;
        let log = ValueLog::open(temp_dir.path().join("test.log"))?;
        let mut meta = ValueLogMetadata::new();
        log.ensure_current_page(&mut meta)?;

        let mut all_locs = Vec::new();
        let mut all_vals: Vec<Vec<u8>> = Vec::new();
        for i in 0u8..20 {
            let v = vec![i; 32];
            all_locs.push(write_value_helper(&log, &mut meta, &v)?);
            all_vals.push(v);
        }

        // Select every other entry (non-contiguous segment IDs)
        let entries: Vec<(usize, ValueLocation)> = (0..20usize).step_by(2).map(|i| (i / 2, all_locs[i])).collect();

        let file = log.get_file();
        let mut batch = log.read_values_batch_from_file(&file, &entries);
        batch.sort_by_key(|(idx, _)| *idx);

        assert_eq!(batch.len(), 10);
        for (result_idx, (_, val_opt)) in batch.iter().enumerate() {
            let original_idx = result_idx * 2;
            assert_eq!(val_opt.as_ref().map(|(v, _)| v.as_slice()), Some(all_vals[original_idx].as_slice()));
        }
        Ok(())
    }
}
