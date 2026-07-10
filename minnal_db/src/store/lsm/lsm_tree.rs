//! LSM Tree implementation using skip_list as in-memory store with Minnal-style architecture
//!
//! Architecture:
//! - In-memory: SkipList storing (key -> value_log_offset)
//! - Read-only layer: Vector of KeyValueRecords when skiplist approaches capacity
//! - Persistent layer: 16 sharded SSTable files (bucketed by key hash)
//!
//! Operations:
//! - Write: Always goes to in-memory skip list (auto-triggers compaction when full)
//! - Read: Skip list -> read-only records -> SSTable files
//! - Compaction: Background thread merges read-only records into SSTable files

use super::bloom::BloomFilter;
use super::lsm_manifest::{LsmManifest, ManifestBucket, ManifestFile, ManifestLevel};
use super::skip_list::skip_list::{KeyValueRecord, SkipList};
use super::sparse_index::SparseIndex;
use crate::support::{get_bucket_for_key, key_prefix_of};
use log::{info, warn};
use parking_lot::RwLock;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use thiserror::Error;

use crate::db::metrics::Metrics;

/// Observer interface for memtable flush lifecycle events.
pub(crate) trait LsmFlushObserver: Send + Sync {
    fn on_memtable_sealed(&self, version: u64);
    fn on_ro_memtable_flushed_to_level0(&self, version: u64);
}

#[derive(Error, Debug)]
pub enum LSMError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Capacity exceeded")]
    CapacityExceeded,
    #[error("SSTable corruption: {0}")]
    Corruption(String),
}

pub(crate) type Result<T> = std::result::Result<T, LSMError>;

/// Configuration for LSM tree compaction behavior
#[derive(Clone, Debug)]
pub struct LSMConfig {
    /// Trigger compaction when skip list reaches this percentage capacity (0-100)
    pub(crate) compaction_threshold_percent: usize,
    /// Base directory for SSTable files
    pub(crate) data_dir: PathBuf,
    /// Number of sharding buckets
    pub(crate) num_buckets: usize,
    /// Max entries in the skip list memtable
    pub(crate) skip_list_capacity: usize,
}

impl Default for LSMConfig {
    fn default() -> Self {
        Self {
            compaction_threshold_percent: 95,
            data_dir: PathBuf::from("lsm_data"),
            num_buckets: crate::support::DEFAULT_NUM_BUCKETS,
            skip_list_capacity: 100_000,
        }
    }
}

impl LSMConfig {
    #[allow(dead_code)]
    pub fn new(compaction_threshold_percent: usize, data_dir: PathBuf) -> Self {
        Self {
            compaction_threshold_percent,
            data_dir,
            num_buckets: crate::support::DEFAULT_NUM_BUCKETS,
            skip_list_capacity: 100_000,
        }
    }
}

/// Metadata for an SSTable file
#[derive(Clone, Debug, Archive, RkyvSerialize, RkyvDeserialize)]
struct SStableMetadata {
    /// Bucket index (0-15)
    bucket: u32,
    /// Minimum key in this SSTable
    min_key: Vec<u8>,
    /// Maximum key in this SSTable
    max_key: Vec<u8>,
    /// Minimum key prefix (first 8 bytes as u64)
    min_key_prefix: u64,
    /// Maximum key prefix (first 8 bytes as u64)
    max_key_prefix: u64,
    /// Number of entries in this SSTable
    entry_count: u64,
    /// Offset where entries data starts
    data_offset: u64,
    /// Total size of entries data
    data_size: u64,
}

/// On disk, each SSTable entry is framed as:
///
/// ```text
/// [u32 payload_len][u32 crc32(body)][body = rkyv-serialized SStableEntry]
/// ```
///
/// `payload_len` counts the CRC word plus the rkyv `body`. The CRC guards
/// against silent on-disk corruption (bit rot, torn writes): it is verified
/// before the body is handed to the *unchecked* rkyv accessor, which assumes
/// well-formed input and would otherwise interpret garbage as a valid entry.
fn encode_sstable_entry(entry: &SStableEntry) -> Result<Vec<u8>> {
    let body = rkyv::to_bytes::<rkyv::rancor::Error>(entry).map_err(|e| LSMError::Serialization(format!("Failed to serialize entry: {}", e)))?;
    let crc = crc32fast::hash(&body);
    let mut payload = Vec::with_capacity(4 + body.len());
    payload.extend_from_slice(&crc.to_le_bytes());
    payload.extend_from_slice(&body);
    Ok(payload)
}

/// Verify a framed SSTable entry payload and return its rkyv `body` slice.
///
/// `framed` is the `payload_len` bytes that follow the length prefix:
/// `[u32 crc32(body)][body]`. Returns [`LSMError::Corruption`] if the payload
/// is truncated or the CRC does not match, so a corrupt entry never reaches
/// `rkyv::access_unchecked`.
fn verify_sstable_payload(framed: &[u8]) -> Result<&[u8]> {
    if framed.len() < 4 {
        return Err(LSMError::Corruption(format!("SSTable entry payload too small: {} bytes", framed.len())));
    }
    let stored = u32::from_le_bytes(framed[0..4].try_into().unwrap());
    let body = &framed[4..];
    let actual = crc32fast::hash(body);
    if actual != stored {
        return Err(LSMError::Corruption(format!(
            "SSTable entry CRC mismatch: stored {:#010x}, computed {:#010x}",
            stored, actual
        )));
    }
    Ok(body)
}

/// Result of writing a merged SSTable: `(min_key, max_key, entry_count,
/// data_size, sparse_index)`.
type MergedSstableInfo = (Vec<u8>, Vec<u8>, u64, u64, SparseIndex);

/// Result of scanning an existing SSTable file at open time:
/// `(metadata, bloom, sparse_index, max_seq)`. The bloom/index/max_seq are
/// `None` when the file has no entries.
type LoadedSstable = (SStableMetadata, Option<BloomFilter>, Option<SparseIndex>, Option<u32>);

/// A prefix/range scan layer result: `(key, Some((pointer, seq)))` for a live
/// entry or `(key, None)` for a tombstone — `seq` is the LSM write sequence used
/// for the read-time value validity check.
type ScanEntry = (Vec<u8>, Option<(u128, u32)>);

/// A batched point-lookup result slot: `(original_index, Some((pointer, seq)))`.
type IdxEntry = (usize, Option<(u128, u32)>);

/// A liveness-scan entry from `key_pointer_pairs`' L0 read: `(key, value-or-None
/// for a tombstone, write seq)`, used for the seq-aware GC liveness merge.
type KpScanEntry = (Vec<u8>, Option<u128>, u32);

/// A single entry in the SSTable
#[derive(Clone, Debug, Archive, RkyvSerialize, RkyvDeserialize)]
struct SStableEntry {
    key: Vec<u8>,
    key_prefix: u64, // First 8 bytes of key as u64 for fast prefix matching
    value: u128,
    tombstone: bool,
    seq: u32,
}

/// In-memory component of LSM tree
struct MemTable {
    skip_list: SkipList,
    #[allow(dead_code)]
    creation_time: std::time::Instant,
}

impl MemTable {
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            skip_list: SkipList::new(),
            creation_time: std::time::Instant::now(),
        }
    }

    fn with_capacity(skip_list_capacity: usize) -> Self {
        Self {
            skip_list: SkipList::with_capacity(skip_list_capacity),
            creation_time: std::time::Instant::now(),
        }
    }

    /// Check if memtable should be flushed (approaching capacity)
    /// Flushes if ANY condition is true:
    /// 1. Capacity usage percentage >= threshold_percent, OR
    /// 2. Cannot insert a new key into in memory memtable, OR
    /// 3. Any internal arena (nodes, links, keys) has reached half of its u32 limit.
    fn should_flush(&self, threshold_percent: usize) -> bool {
        // Get max capacity from SkipList
        let max_capacity = self.skip_list.max_capacity();

        // Calculate current usage (live nodes + tombstones)
        let live_nodes = self.skip_list.number_of_live_nodes();
        let tombstone_nodes = self.skip_list.number_of_tombstone_nodes();
        let total_nodes = live_nodes + tombstone_nodes;

        // Calculate usage percentage
        let usage_percent = (total_nodes * 100) / max_capacity;

        // Check if threshold percentage is exceeded
        let threshold_exceeded = usage_percent >= threshold_percent;

        // Check if we can still insert a reasonable sized key
        let cannot_insert = !self.skip_list.can_insert_key(&[0u8; 256]);

        // Check if any arena is approaching its u32 address-space limit
        let arenas_pressure = self.skip_list.arenas_half_full();

        // Flush if ANY condition is true
        threshold_exceeded || cannot_insert || arenas_pressure
    }

    /// Collect all entries for compaction
    fn collect_records(&self) -> Vec<KeyValueRecord> {
        self.skip_list.collect_key_value_records()
    }
}

/// Read-only snapshot of memtable entries pending compaction
#[derive(Clone)]
struct ReadOnlyMemTable {
    records: Arc<Vec<KeyValueRecord>>,
    #[allow(dead_code)]
    creation_time: std::time::Instant,
    /// Reference count of active readers
    reader_count: Arc<AtomicU64>,
    /// Version number for tracking updates
    version: u64,
    /// Whether this memtable has been flushed to Level 0 files
    flushed_to_level0: Arc<AtomicBool>,
}

impl ReadOnlyMemTable {
    fn new(records: Vec<KeyValueRecord>, version: u64) -> Self {
        Self {
            records: Arc::new(records),
            creation_time: std::time::Instant::now(),
            reader_count: Arc::new(AtomicU64::new(0)),
            version,
            flushed_to_level0: Arc::new(AtomicBool::new(false)),
        }
    }

    fn records(&self) -> Arc<Vec<KeyValueRecord>> {
        Arc::clone(&self.records)
    }

    fn increment_reader_count(&self) -> u64 {
        self.reader_count.fetch_add(1, Ordering::SeqCst)
    }

    fn decrement_reader_count(&self) {
        self.reader_count.fetch_sub(1, Ordering::SeqCst);
    }

    fn reader_count(&self) -> u64 {
        self.reader_count.load(Ordering::SeqCst)
    }

    fn version(&self) -> u64 {
        self.version
    }

    fn is_flushed_to_level0(&self) -> bool {
        self.flushed_to_level0.load(Ordering::SeqCst)
    }

    fn mark_flushed_to_level0(&self) {
        self.flushed_to_level0.store(true, Ordering::SeqCst);
    }
}

impl std::fmt::Debug for ReadOnlyMemTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadOnlyMemTable")
            .field("version", &self.version)
            .field("reader_count", &self.reader_count())
            .field("creation_time", &self.creation_time)
            .finish()
    }
}

/// Level 0 file tracking struct
struct L0FileEntry {
    path: PathBuf,
    created_at_ms: u128,
    reader_count: AtomicU64,
    obsolete: AtomicBool,
}

impl L0FileEntry {
    fn new(path: PathBuf, created_at_ms: u128) -> Self {
        Self {
            path,
            created_at_ms,
            reader_count: AtomicU64::new(0),
            obsolete: AtomicBool::new(false),
        }
    }

    fn is_obsolete(&self) -> bool {
        self.obsolete.load(Ordering::SeqCst)
    }

    fn mark_obsolete(&self) {
        self.obsolete.store(true, Ordering::SeqCst);
    }

    fn increment_readers(&self) {
        self.reader_count.fetch_add(1, Ordering::SeqCst);
    }

    fn decrement_readers(&self) {
        self.reader_count.fetch_sub(1, Ordering::SeqCst);
    }

    fn readers(&self) -> u64 {
        self.reader_count.load(Ordering::SeqCst)
    }
}

/// Keeps a Level-0 file readable while a scan holds it. Owns an `Arc<L0FileEntry>`
/// (rather than borrowing) so a guard captured under the `level0_files` lock can be
/// held *across* the subsequent lock-free I/O: `cleanup_obsolete_level0_files` only
/// deletes a file once `readers() == 0`, so an outstanding guard pins the file even
/// after a concurrent merge marks it obsolete. This is what lets scans capture L0
/// before L1 and read it safely.
struct L0ReadGuard {
    entry: Arc<L0FileEntry>,
}

impl L0ReadGuard {
    fn new(entry: Arc<L0FileEntry>) -> Self {
        entry.increment_readers();
        Self { entry }
    }
}

impl Drop for L0ReadGuard {
    fn drop(&mut self) {
        self.entry.decrement_readers();
    }
}

/// The smallest key strictly greater than every key that has `prefix` as a
/// prefix, so a prefix scan over `P` equals a range scan over `[P, upper)`.
///
/// Increments the last byte that is not `0xFF` and drops the all-`0xFF` tail.
/// Returns `None` when `prefix` is empty or all `0xFF` — there is no finite upper
/// bound, so the caller scans to the end of the keyspace.
fn prefix_upper_bound(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut upper = prefix.to_vec();
    while let Some(last) = upper.last_mut() {
        if *last != 0xFF {
            *last += 1;
            return Some(upper);
        }
        upper.pop();
    }
    None
}

/// Whether `a` is a strictly-newer write sequence than `b`, using the same
/// wraparound-aware serial-number comparison as the memtable
/// (`SkipList::seq_is_newer_or_equal`). Resolution across layers picks the entry
/// with the newest sequence, so order of inspection only breaks exact ties.
#[inline]
fn seq_newer(a: u32, b: u32) -> bool {
    a != b && a.wrapping_sub(b) < 0x8000_0000
}

/// Whether `a` is newer than *or equal to* `b` (same serial-number comparison).
#[inline]
fn seq_newer_or_eq(a: u32, b: u32) -> bool {
    a.wrapping_sub(b) < 0x8000_0000
}

/// Outcome of looking a key up in a single SSTable layer, carrying the matched
/// entry's write sequence so resolution can pick the newest version across all
/// layers (a GC re-point can place a low-sequence value in a *newer* layer, so
/// layer order alone is not a safe proxy for recency).
#[derive(Debug, Clone, Copy)]
enum SsLookup {
    /// Live value pointer found in this layer, with its stored sequence.
    Found(u128, u32),
    /// An explicit tombstone for the key was found, with its stored sequence.
    Deleted(u32),
    /// The key is simply not present in this layer.
    Missing,
}

impl SsLookup {
    /// The matched entry's sequence, or `None` for [`SsLookup::Missing`].
    fn seq(&self) -> Option<u32> {
        match self {
            SsLookup::Found(_, seq) | SsLookup::Deleted(seq) => Some(*seq),
            SsLookup::Missing => None,
        }
    }
}

/// Main LSM tree structure
pub(crate) struct LSMTree {
    /// Current active memtable
    memtable: Arc<RwLock<MemTable>>,

    /// Read-only memtables waiting for compaction
    /// Now uses Arc<Vec<>> to support reference counting
    read_only_memtables: Arc<RwLock<Vec<ReadOnlyMemTable>>>,

    /// SSTable files for each bucket wrapped in Arc for atomic swaps
    /// Arc<RwLock<Arc<File>>> allows readers to keep old file while new file is installed
    sstable_files: Vec<Arc<RwLock<Arc<File>>>>,
    sstable_metadata: Vec<Arc<RwLock<Vec<SStableMetadata>>>>,
    /// Per-bucket Bloom filter over the L1 file's keys, used to skip the linear
    /// scan on a miss. `None` until the L1 file is built/loaded. Derived in
    /// memory (see [`BloomFilter`]); replaced atomically with the L1 file +
    /// metadata under the bucket write lock during compaction.
    sstable_blooms: Vec<Arc<RwLock<Option<Arc<BloomFilter>>>>>,
    /// Per-bucket sparse index over the L1 file's keys, used to binary-search to
    /// a near-by scan start (O(log + interval) instead of O(N)). `None` until the
    /// L1 file is built/loaded. Derived in memory (see [`SparseIndex`]); since it
    /// yields a *file offset*, lookups validate the offset against the file and
    /// fall back to a full scan if a compaction has swapped the file underneath.
    sstable_indexes: Vec<Arc<RwLock<Option<Arc<SparseIndex>>>>>,

    /// Upper bound on the write sequence of any entry in a non-active layer
    /// (read-only memtables + L0 + L1). Read resolution is highest-sequence-wins
    /// across all layers, but an active-memtable hit whose sequence is `>=` this
    /// bound dominates everything below it and can be returned without scanning
    /// the lower layers — restoring the early-return fast path for normal writes,
    /// which always carry the newest sequence. Only a GC re-point (a low sequence
    /// re-inserted into the active memtable) fails the bound and falls through to
    /// the full seq-aware scan. Folded (serial-max) on every memtable flush and
    /// from the L1 files loaded at open.
    max_lower_seq: AtomicU32,

    /// Engine-wide operational counters, shared from `Database` via
    /// [`set_metrics`](Self::set_metrics). `None` until set (e.g. a standalone
    /// tree in tests), in which case metric updates are no-ops.
    metrics: OnceLock<Arc<Metrics>>,

    config: LSMConfig,

    /// Per-bucket compaction flags - each bucket can compact independently
    compaction_in_progress: Vec<Arc<AtomicBool>>,

    /// Global version for read-only memtables
    ro_memtable_version: Arc<AtomicU64>,

    memtable_sequence: Arc<AtomicUsize>,

    /// Pending old read-only memtables to be deleted on next compaction
    /// Similar to pending_old_logs in ValueLog GC
    pending_old_memtables: Arc<RwLock<Vec<ReadOnlyMemTable>>>,

    /// Level 0 files tracked per bucket for reader-safe compaction/deletion.
    level0_files: Vec<Arc<RwLock<Vec<Arc<L0FileEntry>>>>>,

    /// Optional observer for flush lifecycle events.
    flush_observer: Arc<RwLock<Option<Arc<dyn LsmFlushObserver>>>>,

    base_path: PathBuf,
}

impl LSMTree {
    fn read_exact_at(file: &File, buf: &mut [u8], offset: &mut u64) -> Result<bool> {
        let mut read_total = 0usize;
        while read_total < buf.len() {
            match file.read_at(&mut buf[read_total..], *offset + read_total as u64) {
                Ok(0) => {
                    if read_total == 0 {
                        return Ok(false);
                    }
                    return Err(LSMError::Io(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "Unexpected EOF")));
                }
                Ok(n) => {
                    read_total += n;
                }
                Err(err) => return Err(LSMError::Io(err)),
            }
        }
        *offset += buf.len() as u64;
        Ok(true)
    }

    /// Tri-state result of looking a key up in one SSTable layer.
    ///
    /// The distinction between [`SsLookup::Deleted`] and [`SsLookup::Missing`]
    /// is load-bearing: a tombstone found in a newer layer (e.g. an L0 file)
    /// must *shadow* a live value in an older layer (L1). Collapsing both into
    /// `None` (as the old `search_in_sstable_file` did) let `get` fall through
    /// an L0 tombstone to a stale L1 value, resurrecting deleted keys.
    fn lookup_in_sstable_file(&self, file: &File, key: &[u8]) -> Result<SsLookup> {
        self.lookup_in_sstable_file_from(file, key, 0)
    }

    /// Scan an SSTable file for `key`, starting at `start_offset` (a sparse-index
    /// hint, or 0 for the whole file).
    ///
    /// The hint is validated against the file first: if it does not point at a
    /// complete, CRC-valid frame whose key is `<= key` — e.g. a concurrent
    /// compaction swapped the file under the in-memory index, leaving the offset
    /// stale — the scan restarts from 0. Entries are key-sorted, so beginning at
    /// *any* valid frame boundary with key `<= key` still finds `key` if present;
    /// this makes a stale hint a performance issue, never a correctness one.
    fn lookup_in_sstable_file_from(&self, file: &File, key: &[u8], start_offset: u64) -> Result<SsLookup> {
        let mut offset = if start_offset != 0 {
            let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
            if Self::valid_scan_start(file, start_offset, file_len, key) {
                start_offset
            } else {
                0
            }
        } else {
            0
        };

        loop {
            let mut size_buf = [0u8; 4];
            if !Self::read_exact_at(file, &mut size_buf, &mut offset)? {
                break;
            }

            let size = u32::from_le_bytes(size_buf);
            let mut entry_bytes = vec![0u8; size as usize];
            Self::read_exact_at(file, &mut entry_bytes, &mut offset)?;

            let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes)?) };

            if archived.key.as_slice() == key {
                if archived.tombstone {
                    return Ok(SsLookup::Deleted(archived.seq.to_native()));
                }
                return Ok(SsLookup::Found(archived.value.to_native(), archived.seq.to_native()));
            }

            if archived.key.as_slice() > key {
                break;
            }
        }

        Ok(SsLookup::Missing)
    }

    /// True if `offset` begins a complete, CRC-valid frame whose key is `<= key`
    /// — a safe place to start a forward scan. The size is bounds-checked against
    /// `file_len` so a stale offset cannot trigger a huge allocation, and the CRC
    /// catches an offset that lands mid-frame or in a different (swapped) file.
    fn valid_scan_start(file: &File, offset: u64, file_len: u64, key: &[u8]) -> bool {
        // Saturating arithmetic: a stale hint may be an arbitrarily large offset,
        // and `offset + 4` must not overflow (it would panic in debug).
        if offset.saturating_add(4) > file_len {
            return false;
        }
        let mut size_buf = [0u8; 4];
        if file.read_at(&mut size_buf, offset).is_err() {
            return false;
        }
        let size = u32::from_le_bytes(size_buf) as u64;
        if size == 0 || offset.saturating_add(4).saturating_add(size) > file_len {
            return false;
        }
        let mut buf = vec![0u8; size as usize];
        if file.read_at(&mut buf, offset + 4).is_err() {
            return false;
        }
        match verify_sstable_payload(&buf) {
            Ok(body) => {
                let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(body) };
                archived.key.as_slice() <= key
            }
            Err(_) => false,
        }
    }

    fn read_sstable_entries_from_file(&self, file: &File) -> Result<Vec<SStableEntry>> {
        let mut entries = Vec::new();
        let mut offset = 0u64;

        loop {
            let mut size_buf = [0u8; 4];
            if !Self::read_exact_at(file, &mut size_buf, &mut offset)? {
                break;
            }

            let size = u32::from_le_bytes(size_buf);
            let mut entry_bytes = vec![0u8; size as usize];
            Self::read_exact_at(file, &mut entry_bytes, &mut offset)?;

            let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes)?) };
            let key = archived.key.to_vec();
            let entry = SStableEntry {
                key_prefix: key_prefix_of(&key),
                key,
                value: archived.value.to_native(),
                tombstone: archived.tombstone,
                seq: archived.seq.to_native(),
            };

            entries.push(entry);
        }

        Ok(entries)
    }
    /// Create or open an LSM tree
    pub(crate) fn open<P: AsRef<Path>>(base_path: P, config: LSMConfig) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        let level1_dir = Self::level1_dir_from(&base_path);
        let level0_dir = Self::level0_dir_from(&base_path);
        std::fs::create_dir_all(&level1_dir)?;
        std::fs::create_dir_all(&level0_dir)?;

        let num_buckets = config.num_buckets;

        // Initialize SSTable files and metadata with Arc<File> for atomic swaps
        let mut sstable_files_vec: Vec<Arc<RwLock<Arc<File>>>> = Vec::with_capacity(num_buckets);
        let mut sstable_metadata_vec: Vec<Arc<RwLock<Vec<SStableMetadata>>>> = Vec::with_capacity(num_buckets);
        let mut sstable_blooms_vec: Vec<Arc<RwLock<Option<Arc<BloomFilter>>>>> = Vec::with_capacity(num_buckets);
        let mut sstable_indexes_vec: Vec<Arc<RwLock<Option<Arc<SparseIndex>>>>> = Vec::with_capacity(num_buckets);
        // Newest sequence across all loaded L1 files — the initial lower-layer
        // bound for the read fast path.
        let mut max_lower_seq_init: Option<u32> = None;

        // Open or create Level 1 SSTable files per bucket (similar to sharded value log)
        for bucket in 0u32..num_buckets as u32 {
            let level1_path = Self::level1_path_from(&base_path, bucket);
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&level1_path)?;
            sstable_files_vec.push(Arc::new(RwLock::new(Arc::new(file))));

            let mut metadata = Vec::new();
            let mut bloom = None;
            let mut index = None;
            let file_len = std::fs::metadata(&level1_path)?.len();
            if file_len > 0
                && let Ok((meta, built_bloom, built_index, file_max_seq)) = Self::load_metadata_from_file(&level1_path, bucket)
            {
                metadata.push(meta);
                bloom = built_bloom.map(Arc::new);
                index = built_index.map(Arc::new);
                if let Some(seq) = file_max_seq
                    && max_lower_seq_init.is_none_or(|m| seq_newer(seq, m))
                {
                    max_lower_seq_init = Some(seq);
                }
            }
            sstable_metadata_vec.push(Arc::new(RwLock::new(metadata)));
            sstable_blooms_vec.push(Arc::new(RwLock::new(bloom)));
            sstable_indexes_vec.push(Arc::new(RwLock::new(index)));
        }

        let sstable_files: Vec<Arc<RwLock<Arc<File>>>> = sstable_files_vec;
        let sstable_metadata: Vec<Arc<RwLock<Vec<SStableMetadata>>>> = sstable_metadata_vec;
        let sstable_blooms: Vec<Arc<RwLock<Option<Arc<BloomFilter>>>>> = sstable_blooms_vec;
        let sstable_indexes: Vec<Arc<RwLock<Option<Arc<SparseIndex>>>>> = sstable_indexes_vec;

        let level0_files: Vec<Arc<RwLock<Vec<Arc<L0FileEntry>>>>> = (0..num_buckets).map(|_| Arc::new(RwLock::new(Vec::new()))).collect();

        // L0 is newer than L1, so its sequences must be folded into the read
        // fast-path bound too — otherwise the bound (L1-only above) is too low
        // after a restart and a low-seq active entry could wrongly short-circuit.
        if let Some(seq) = Self::load_existing_level0_files(&base_path, &level0_files)?
            && max_lower_seq_init.is_none_or(|m| seq_newer(seq, m))
        {
            max_lower_seq_init = Some(seq);
        }

        // Per-bucket compaction flags
        let compaction_in_progress: Vec<Arc<AtomicBool>> = (0..num_buckets).map(|_| Arc::new(AtomicBool::new(false))).collect();

        Ok(Self {
            memtable: Arc::new(RwLock::new(MemTable::with_capacity(config.skip_list_capacity))),
            read_only_memtables: Arc::new(RwLock::new(Vec::new())),
            sstable_files,
            sstable_metadata,
            sstable_blooms,
            sstable_indexes,
            max_lower_seq: AtomicU32::new(max_lower_seq_init.unwrap_or(0)),
            metrics: OnceLock::new(),
            config,
            compaction_in_progress,
            ro_memtable_version: Arc::new(AtomicU64::new(0)),
            memtable_sequence: Arc::new(AtomicUsize::new(0)),
            pending_old_memtables: Arc::new(RwLock::new(Vec::new())),
            level0_files,
            flush_observer: Arc::new(RwLock::new(None)),
            base_path,
        })
    }

    /// Register or clear a flush observer.
    pub(crate) fn set_flush_observer(&self, observer: Option<Arc<dyn LsmFlushObserver>>) {
        *self.flush_observer.write() = observer;
    }

    fn notify_memtable_sealed(&self, version: u64) {
        if let Some(observer) = self.flush_observer.read().as_ref() {
            observer.on_memtable_sealed(version);
        }
    }

    fn notify_ro_memtable_flushed_to_level0(&self, version: u64) {
        if let Some(observer) = self.flush_observer.read().as_ref() {
            observer.on_ro_memtable_flushed_to_level0(version);
        }
    }

    /// Cleanup any old SSTable files left from previous runs
    /// Called on startup to clean up .old files
    pub(crate) fn cleanup_old_files_on_startup(&self) -> Result<()> {
        for bucket in 0u32..self.config.num_buckets as u32 {
            let old_path = Self::level1_dir_from(&self.base_path).join(format!("level1_{}.dat.old", bucket));
            if old_path.exists() {
                match std::fs::remove_file(&old_path) {
                    Ok(_) => {
                        info!("[LSM] Cleaned up old SSTable file on startup: level1_{}.dat.old", bucket);
                    }
                    Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                        warn!("[LSM] Failed to cleanup old SSTable file: {:?}", e);
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Create a null/empty file for initialization
    fn create_null_file() -> Result<File> {
        // Return a temporary file that will be replaced
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open("/dev/null")
            .or_else(|_| OpenOptions::new().read(true).write(true).create(true).truncate(false).open("nul"))
            .map_err(LSMError::Io)
    }

    /// Level 1 directory path under base path
    fn level1_dir_from(base_path: &Path) -> PathBuf {
        base_path.join("level1")
    }

    /// Level 1 SSTable path for a bucket
    ///
    /// Naming note:
    /// - Level 1 uses a fixed file per bucket (level1_<bucket>.dat).
    /// - Level 0 uses timestamp-based filenames per flush (see create_level0_file_for_records).
    ///   This is why you can see gaps in L0 and the same bucket id in both levels.
    fn level1_path_from(base_path: &Path, bucket: u32) -> PathBuf {
        Self::level1_dir_from(base_path).join(format!("level1_{}.dat", bucket))
    }

    /// Level 0 directory path under base path
    fn level0_dir_from(base_path: &Path) -> PathBuf {
        base_path.join("level0")
    }

    /// Level 0 directory for a specific bucket
    fn level0_bucket_dir_from(base_path: &Path, bucket: u32) -> PathBuf {
        Self::level0_dir_from(base_path).join(format!("level0_{}", bucket))
    }

    fn level0_created_at_from_path(path: &Path) -> u128 {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("0");
        let first = stem.split('_').next().unwrap_or("0");
        if let Ok(ts) = first.parse::<u128>() {
            return ts;
        }
        path.metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis())
            .unwrap_or(0)
    }

    /// Scan a Level-0 file and return the newest write sequence it contains
    /// (`None` if empty/unreadable). Used at open to fold L0 into `max_lower_seq`.
    fn level0_file_max_seq(path: &Path) -> Result<Option<u32>> {
        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return Ok(None),
        };
        let mut max_seq: Option<u32> = None;
        loop {
            let mut size_buf = [0u8; 4];
            match file.read_exact(&mut size_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(LSMError::Io(e)),
            }
            let size = u32::from_le_bytes(size_buf);
            let mut entry_bytes = vec![0u8; size as usize];
            file.read_exact(&mut entry_bytes)?;
            let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes)?) };
            let seq = archived.seq.to_native();
            if max_seq.is_none_or(|m| seq_newer(seq, m)) {
                max_seq = Some(seq);
            }
        }
        Ok(max_seq)
    }

    /// Load the on-disk Level-0 files into the in-memory registry and return the
    /// newest write sequence across all of them. The returned seq is folded into
    /// `max_lower_seq` so the read fast path's bound covers L0 too — L0 is newer
    /// than L1, so omitting it would leave the bound too low after a restart (a
    /// low-seq active entry could then wrongly short-circuit above a higher-seq
    /// L0 tombstone).
    fn load_existing_level0_files(base_path: &Path, level0_files: &[Arc<RwLock<Vec<Arc<L0FileEntry>>>>]) -> Result<Option<u32>> {
        let mut max_seq: Option<u32> = None;
        for bucket in 0u32..level0_files.len() as u32 {
            let dir = Self::level0_bucket_dir_from(base_path, bucket);
            if !dir.exists() {
                continue;
            }
            let mut guard = level0_files[bucket as usize].write();
            for entry in std::fs::read_dir(&dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map(|ext| ext == "dat").unwrap_or(false) {
                    if let Some(seq) = Self::level0_file_max_seq(&path)?
                        && max_seq.is_none_or(|m| seq_newer(seq, m))
                    {
                        max_seq = Some(seq);
                    }
                    let created_at = Self::level0_created_at_from_path(&path);
                    guard.push(Arc::new(L0FileEntry::new(path, created_at)));
                }
            }
        }
        Ok(max_seq)
    }

    fn register_level0_file(&self, bucket: u32, path: PathBuf, created_at_ms: u128) {
        let mut guard = self.level0_files[bucket as usize].write();
        if guard.iter().any(|entry| entry.path == path) {
            return;
        }
        guard.push(Arc::new(L0FileEntry::new(path, created_at_ms)));
    }

    /// Capture frozen SSTable read handles for a scan, **newest layer first**:
    /// per-bucket Level-0 read guards (skipping already-obsolete files, ordered
    /// oldest→newest within a bucket) captured *before* the per-bucket Level-1
    /// `Arc<File>`s.
    ///
    /// Capturing in the same direction that compaction relocates keys (L0→L1) is
    /// what makes a scan complete under a concurrent merge: a key the merge moves
    /// into L1 after we captured L0 is still found when we capture L1 immediately
    /// after, so it is never absent from both. The returned guards pin their files
    /// (`cleanup_obsolete_level0_files` only deletes when `readers() == 0`), so a
    /// merge that obsoletes them mid-scan cannot delete them before the caller has
    /// read them. The caller still merges oldest→newest (L1 then L0) so precedence
    /// is unchanged.
    fn capture_sstable_layers(&self) -> (Vec<Vec<L0ReadGuard>>, Vec<Arc<File>>) {
        let num_buckets = self.config.num_buckets;
        let l0_guards: Vec<Vec<L0ReadGuard>> = (0..num_buckets)
            .map(|bucket| {
                let entries = self.level0_files[bucket].read();
                let mut guards: Vec<L0ReadGuard> = entries
                    .iter()
                    .filter(|e| !e.is_obsolete())
                    .map(|e| L0ReadGuard::new(Arc::clone(e)))
                    .collect();
                guards.sort_by(|a, b| a.entry.created_at_ms.cmp(&b.entry.created_at_ms));
                guards
            })
            .collect();
        let l1_files: Vec<Arc<File>> = (0..num_buckets).map(|bucket| self.sstable_files[bucket].read().clone()).collect();
        (l0_guards, l1_files)
    }

    fn level0_entries_snapshot(&self, bucket: u32) -> Vec<Arc<L0FileEntry>> {
        self.level0_files[bucket as usize].read().clone()
    }

    fn cleanup_obsolete_level0_files(&self) {
        for bucket in 0u32..self.config.num_buckets as u32 {
            let mut guard = self.level0_files[bucket as usize].write();
            guard.retain(|entry| {
                if entry.is_obsolete() && entry.readers() == 0 {
                    let _ = std::fs::remove_file(&entry.path);
                    false
                } else {
                    true
                }
            });
        }
    }

    fn flush_ro_memtable_to_level0(&self, ro_memtable: &ReadOnlyMemTable) -> Result<()> {
        if ro_memtable.is_flushed_to_level0() {
            return Ok(());
        }

        let records = ro_memtable.records();
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let timestamp = now_ms.saturating_mul(1_000_000).saturating_add(ro_memtable.version() as u128);

        for bucket in 0u32..self.config.num_buckets as u32 {
            self.create_level0_file_for_records(bucket, &records, timestamp)?;
        }

        ro_memtable.mark_flushed_to_level0();
        self.notify_ro_memtable_flushed_to_level0(ro_memtable.version());
        Ok(())
    }

    fn create_level0_file_for_records(&self, bucket: u32, records: &[KeyValueRecord], timestamp_ms: u128) -> Result<()> {
        let mut bucket_records = Vec::new();
        for record in records {
            let record_bucket = get_bucket_for_key(&record.key, self.config.num_buckets);
            if record_bucket == bucket {
                bucket_records.push(record.clone());
            }
        }

        if bucket_records.is_empty() {
            return Ok(());
        }

        bucket_records.sort_by(|a, b| a.key.cmp(&b.key));

        let level0_dir = Self::level0_bucket_dir_from(&self.base_path, bucket);
        std::fs::create_dir_all(&level0_dir)?;

        // Level 0 files are named by timestamp/version, not by a global sequence.
        let level0_path = level0_dir.join(format!("{}.dat", timestamp_ms));

        let mut file = std::fs::File::create(&level0_path)?;
        for entry in bucket_records {
            let key_prefix = key_prefix_of(&entry.key);
            let entry_obj = SStableEntry {
                key: entry.key,
                key_prefix,
                value: entry.value,
                tombstone: entry.tombstone,
                seq: entry.seq,
            };
            let payload = encode_sstable_entry(&entry_obj)?;
            file.write_all(&(payload.len() as u32).to_le_bytes())?;
            file.write_all(&payload)?;
        }
        file.sync_all()?;

        self.register_level0_file(bucket, level0_path, timestamp_ms);
        Ok(())
    }

    /// Read all entries from an existing SSTable file
    fn read_sstable_entries(&self, bucket: u32) -> Result<Vec<SStableEntry>> {
        let file = self.sstable_files[bucket as usize].read().clone();
        self.read_sstable_entries_from_file(&file)
    }

    /// Read SSTable entries from a specific file path
    fn read_sstable_entries_from_path(&self, path: &Path) -> Result<Vec<SStableEntry>> {
        if !path.exists() {
            return Ok(Vec::new());
        }

        let mut entries = Vec::new();
        let mut file = File::open(path)?;

        loop {
            let mut size_buf = [0u8; 4];
            match file.read_exact(&mut size_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(LSMError::Io(e)),
            }

            let size = u32::from_le_bytes(size_buf);
            let mut entry_bytes = vec![0u8; size as usize];
            file.read_exact(&mut entry_bytes)?;

            let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes)?) };
            let key = archived.key.to_vec();
            let entry = SStableEntry {
                key_prefix: key_prefix_of(&key),
                key,
                value: archived.value.to_native(),
                tombstone: archived.tombstone,
                seq: archived.seq.to_native(),
            };

            entries.push(entry);
        }

        Ok(entries)
    }

    /// Perform 2-way merge of existing entries and new records
    fn two_way_merge(&self, existing: Vec<SStableEntry>, new_records: Vec<KeyValueRecord>) -> Vec<SStableEntry> {
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < existing.len() && j < new_records.len() {
            let existing_entry = &existing[i];
            let new_record = &new_records[j];

            match existing_entry.key.as_slice().cmp(new_record.key.as_slice()) {
                std::cmp::Ordering::Less => {
                    if !existing_entry.tombstone {
                        result.push(existing_entry.clone());
                    }
                    i += 1;
                }
                std::cmp::Ordering::Equal => {
                    // Resolve the same-key conflict by highest write `seq` (globally
                    // ordered), NOT by layer — an out-of-order ro→L0 flush or a GC
                    // re-point can leave a lower-seq value in a "newer" file above a
                    // higher-seq tombstone, so file recency is not a safe proxy for
                    // recency here. On an exact tie prefer the L0 record (newer
                    // layer). A winning tombstone drops the key (L1 is the bottom
                    // level, so there is nothing below for it to shadow).
                    if seq_newer_or_eq(new_record.seq, existing_entry.seq) {
                        if !new_record.tombstone {
                            result.push(SStableEntry {
                                key: new_record.key.clone(),
                                key_prefix: key_prefix_of(&new_record.key),
                                value: new_record.value,
                                tombstone: new_record.tombstone,
                                seq: new_record.seq,
                            });
                        }
                    } else if !existing_entry.tombstone {
                        result.push(existing_entry.clone());
                    }
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Greater => {
                    if !new_record.tombstone {
                        result.push(SStableEntry {
                            key: new_record.key.clone(),
                            key_prefix: key_prefix_of(&new_record.key),
                            value: new_record.value,
                            tombstone: new_record.tombstone,
                            seq: new_record.seq,
                        });
                    }
                    j += 1;
                }
            }
        }

        while i < existing.len() {
            if !existing[i].tombstone {
                result.push(existing[i].clone());
            }
            i += 1;
        }

        while j < new_records.len() {
            if !new_records[j].tombstone {
                result.push(SStableEntry {
                    key: new_records[j].key.clone(),
                    key_prefix: key_prefix_of(&new_records[j].key),
                    value: new_records[j].value,
                    tombstone: new_records[j].tombstone,
                    seq: new_records[j].seq,
                });
            }
            j += 1;
        }

        result
    }

    /// Write merged entries to a temporary SSTable file, building the sparse
    /// index from the byte offsets as it writes (no extra pass).
    fn write_merged_sstable(&self, path: &Path, entries: &[SStableEntry]) -> Result<MergedSstableInfo> {
        if entries.is_empty() {
            return Err(LSMError::Serialization("Cannot write empty SSTable".to_string()));
        }

        let mut file = OpenOptions::new().create(true).write(true).truncate(true).open(path)?;

        let mut total_size = 0u64;
        let min_key = entries[0].key.clone();
        let max_key = entries[entries.len() - 1].key.clone();
        let mut index = SparseIndex::new();

        for (i, entry) in entries.iter().enumerate() {
            if (i as u64).is_multiple_of(SparseIndex::SAMPLE_INTERVAL) {
                index.push(entry.key.clone(), total_size);
            }
            let payload = encode_sstable_entry(entry)?;
            let size = payload.len() as u32;
            file.write_all(&size.to_le_bytes())?;
            file.write_all(&payload)?;
            total_size += 4 + size as u64;
        }

        file.flush()?;
        file.sync_all()?;

        Ok((min_key, max_key, entries.len() as u64, total_size, index))
    }

    /// Get a value from the LSM tree
    /// Attach the engine-wide operational counters (called once by `KVStore`).
    pub(crate) fn set_metrics(&self, metrics: Arc<Metrics>) {
        let _ = self.metrics.set(metrics);
    }

    /// The operational counters, if attached. `None` for a standalone tree.
    #[inline]
    fn metrics(&self) -> Option<&Metrics> {
        self.metrics.get().map(|m| m.as_ref())
    }

    pub(crate) fn get(&self, key: &[u8]) -> Result<Option<u128>> {
        Ok(self.get_with_seq(key)?.map(|(ptr, _)| ptr))
    }

    /// Fold a candidate `(value-or-tombstone, seq)` into the running best,
    /// keeping the newest sequence. Callers feed candidates newest-layer-first so
    /// an exact sequence tie stays with the newer copy.
    fn merge_candidate(best: &mut Option<(Option<u128>, u32)>, value: Option<u128>, seq: u32) {
        if best.is_none_or(|(_, b)| seq_newer(seq, b)) {
            *best = Some((value, seq));
        }
    }

    /// Like [`get`](Self::get) but also returns the stored sequence of the live
    /// value. Used by GC / journal replay to relocate a value's pointer while
    /// preserving its sequence, so the relocation neither loses to nor blocks a
    /// real write under highest-sequence-wins resolution.
    pub(crate) fn get_with_seq(&self, key: &[u8]) -> Result<Option<(u128, u32)>> {
        // Resolve by highest write sequence across ALL layers. Layer order is no
        // longer assumed to imply recency — a GC re-point can re-insert a value at
        // its old (low) sequence into a newer layer, above a deleting tombstone in
        // an older layer — so we cannot early-return at the first layer holding
        // the key. We inspect every layer and keep the newest-sequence entry; a
        // tombstone winning means the key is deleted. Layers are visited
        // newest-first so an exact sequence tie resolves to the newer copy.
        let mut best: Option<(Option<u128>, u32)> = None;

        if let Some(m) = self.metrics() {
            Metrics::bump(&m.lookups);
        }

        {
            let memtable = self.memtable.read();
            if let Some((value, seq, tombstone)) = memtable.skip_list.entry(key) {
                // Fast path: if this active-memtable entry is at least as new as
                // everything in the lower layers, it is authoritative — no buried
                // tombstone can be newer — so return without scanning below. Only
                // a GC re-point's low sequence fails this and falls through.
                if seq_newer_or_eq(seq, self.max_lower_seq.load(Ordering::Relaxed)) {
                    if let Some(m) = self.metrics() {
                        Metrics::bump(&m.fast_path_hits);
                    }
                    return Ok((!tombstone).then_some((value, seq)));
                }
                Self::merge_candidate(&mut best, (!tombstone).then_some(value), seq);
            }
        }

        {
            let ro_memtables = self.read_only_memtables.read();
            for ro in ro_memtables.iter().rev() {
                ro.increment_reader_count();
                let records = ro.records();
                let hit = records
                    .iter()
                    .find(|record| record.key.as_slice() == key)
                    .map(|r| (r.value, r.seq, r.tombstone));
                ro.decrement_reader_count();
                if let Some((value, seq, tombstone)) = hit {
                    Self::merge_candidate(&mut best, (!tombstone).then_some(value), seq);
                }
            }
        }

        let bucket = get_bucket_for_key(key, self.config.num_buckets);
        match self.lookup_sstable(bucket, key)? {
            SsLookup::Found(ptr, seq) => Self::merge_candidate(&mut best, Some(ptr), seq),
            SsLookup::Deleted(seq) => Self::merge_candidate(&mut best, None, seq),
            SsLookup::Missing => {}
        }

        Ok(best.and_then(|(value, seq)| value.map(|ptr| (ptr, seq))))
    }

    /// Insert/update a value carrying an explicit global write sequence, with
    /// highest-sequence-wins conflict resolution (see
    /// [`SkipList::try_insert_with_seq`]). This is the production write path; it
    /// makes the in-memory winner for racing same-key writes match the winner
    /// recovery would pick (recovery replays in sequence order).
    pub(crate) fn insert_with_seq(&self, key: &[u8], value: u128, seq: u32) -> Result<()> {
        {
            let memtable = self.memtable.read();
            if memtable.should_flush(self.config.compaction_threshold_percent) {
                drop(memtable);
                self.flush_memtable()?;
            }
        }

        let mut memtable = self.memtable.write();
        match memtable.skip_list.try_insert_with_seq(key, value, seq) {
            Ok(_) => Ok(()),
            Err(super::skip_list::skip_list::InsertError::CapacityExceeded) => Err(LSMError::CapacityExceeded),
        }
    }

    /// Tombstone a key carrying an explicit global write sequence, with
    /// highest-sequence-wins resolution: a delete older than a concurrent write
    /// to the same key is dropped, matching recovery's sequence-ordered replay.
    pub(crate) fn delete_with_seq(&self, key: &[u8], seq: u32) -> Result<()> {
        {
            let memtable = self.memtable.read();
            if memtable.should_flush(self.config.compaction_threshold_percent) {
                drop(memtable);
                self.flush_memtable()?;
            }
        }

        let mut memtable = self.memtable.write();
        // Ensure the node exists so a delete of a key not currently in the
        // memtable still records a tombstone, then tombstone it — both honour
        // the sequence guard, so a newer write to this key is left untouched.
        let _ = memtable.skip_list.try_insert_with_seq(key, 1, seq);
        memtable.skip_list.remove_with_seq(key, seq);

        Ok(())
    }

    /// Fold a sequence into the lower-layer bound (serial-max), used whenever
    /// entries move from the active memtable into a lower layer.
    fn note_lower_seq(&self, seq: u32) {
        let mut cur = self.max_lower_seq.load(Ordering::Relaxed);
        while seq_newer(seq, cur) {
            match self.max_lower_seq.compare_exchange_weak(cur, seq, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(observed) => cur = observed,
            }
        }
    }

    /// Flush current memtable to read-only layer
    fn flush_memtable(&self) -> Result<()> {
        if let Some(m) = self.metrics() {
            Metrics::bump(&m.memtable_flushes);
        }
        let mut memtable = self.memtable.write();
        let records = memtable.collect_records();
        // These records are leaving the active memtable for a lower layer, so
        // advance the lower-layer sequence bound to cover them.
        if let Some(max) = records.iter().map(|r| r.seq).reduce(|a, b| if seq_newer(b, a) { b } else { a }) {
            self.note_lower_seq(max);
        }
        let version = self.ro_memtable_version.fetch_add(1, Ordering::SeqCst);
        let ro_memtable = ReadOnlyMemTable::new(records, version);

        {
            let mut ro_memtables = self.read_only_memtables.write();
            ro_memtables.push(ro_memtable);
        }

        *memtable = MemTable::with_capacity(self.config.skip_list_capacity);
        self.memtable_sequence.fetch_add(1, Ordering::SeqCst);
        self.notify_memtable_sealed(version);
        Ok(())
    }

    /// Flush the current memtable (if non-empty) and all read-only memtables to
    /// level-0 SSTable files on disk.  Unlike `flush_and_compact_all` this does
    /// not run a full compaction, so it is cheaper and suitable for use in the
    /// GC path where we only need to guarantee that in-memory LSM pointer updates
    /// survive a crash before the GCJournal is deleted.
    pub(crate) fn flush_memtable_to_level0(&self) -> Result<()> {
        {
            let memtable = self.memtable.read();
            let has_data = memtable.skip_list.number_of_live_nodes() > 0 || memtable.skip_list.number_of_tombstone_nodes() > 0;
            drop(memtable);
            if has_data {
                self.flush_memtable()?;
            }
        }
        let ro_memtables = self.read_only_memtables.read();
        for ro in ro_memtables.iter() {
            self.flush_ro_memtable_to_level0(ro)?;
        }
        Ok(())
    }

    /// Force flush all data to persistent SSTables (for database close)
    pub(crate) fn flush_and_compact_all(&self) -> Result<()> {
        {
            let memtable = self.memtable.read();
            if memtable.skip_list.number_of_live_nodes() > 0 || memtable.skip_list.number_of_tombstone_nodes() > 0 {
                drop(memtable);
                self.flush_memtable()?;
            }
        }

        self.compact_all()?;
        Ok(())
    }

    /// Tri-state lookup of a key in the SSTable hierarchy (Level 0 then Level 1).
    ///
    /// L0 is newer than L1: a `Found` or a tombstone (`Deleted`) in L0 is
    /// authoritative. Only when the key is absent from all L0 files do we
    /// consult L1. Treating an L0 tombstone as authoritative is what stops a
    /// deleted key from being resurrected by a stale L1 entry.
    fn lookup_sstable(&self, bucket: u32, key: &[u8]) -> Result<SsLookup> {
        // Resolve by newest sequence across both SSTable layers. L0 is newer than
        // L1, so an exact sequence tie resolves to L0 (take L1 only if strictly
        // newer). Layer order is no longer assumed to imply recency — a GC
        // re-point can leave a lower-sequence value above a deleting tombstone.
        let l0 = self.search_level0_files(bucket, key)?;
        let l1 = self.lookup_level1(bucket, key)?;
        Ok(match (l0.seq(), l1.seq()) {
            (Some(s0), Some(s1)) if seq_newer(s1, s0) => l1,
            (Some(_), _) => l0,
            (None, _) => l1,
        })
    }

    /// Look the key up in this bucket's Level-1 file, using the min/max,
    /// bloom-filter and sparse-index fast-rejects before any linear scan.
    fn lookup_level1(&self, bucket: u32, key: &[u8]) -> Result<SsLookup> {
        // L1 key-range early-out: if the key falls outside the L1 file's
        // [min_key, max_key] it cannot be present. The metadata is cleared/updated
        // atomically with the file under the bucket write lock during compaction,
        // so a present entry is always within the recorded range — this can never
        // turn a live key into a false `Missing`. Empty metadata (no L1 file yet,
        // or it failed to load) simply skips the optimisation.
        {
            let metadata = self.sstable_metadata[bucket as usize].read();
            if let Some(meta) = metadata.first()
                && (key < meta.min_key.as_slice() || key > meta.max_key.as_slice())
            {
                return Ok(SsLookup::Missing);
            }
        }
        // Bloom early-out: a negative is exact, so skip the linear scan when the
        // filter says the key is definitely absent. A positive may be a false
        // positive, so fall through and scan.
        {
            let bloom = self.sstable_blooms[bucket as usize].read();
            if let Some(bloom) = bloom.as_ref()
                && !bloom.contains(key)
            {
                if let Some(m) = self.metrics() {
                    Metrics::bump(&m.bloom_rejects);
                }
                return Ok(SsLookup::Missing);
            }
        }
        // Sparse-index hint: binary-search to a near-by start offset so the scan
        // covers ~SAMPLE_INTERVAL entries instead of the whole file. The hint is
        // validated against the file inside `lookup_in_sstable_file_from`.
        let start_offset = {
            let index = self.sstable_indexes[bucket as usize].read();
            index.as_ref().map(|i| i.block_start(key)).unwrap_or(0)
        };
        let file = self.sstable_files[bucket as usize].read().clone();
        if let Some(m) = self.metrics() {
            Metrics::bump(&m.l1_probes);
        }
        self.lookup_in_sstable_file_from(&file, key, start_offset)
    }

    /// Tri-state lookup of a key in a specific SSTable file (by path).
    fn lookup_in_sstable_file_path(&self, path: std::path::PathBuf, key: &[u8]) -> Result<SsLookup> {
        let file = File::open(&path)?;
        self.lookup_in_sstable_file(&file, key)
    }

    /// Load metadata from an existing SSTable file by scanning it, and build the
    /// L1 Bloom filter from the same pass (no extra I/O). Returns `None` for the
    /// bloom when the file has no entries.
    fn load_metadata_from_file(path: &Path, bucket: u32) -> Result<LoadedSstable> {
        let mut file = File::open(path)?;

        let mut min_key: Option<Vec<u8>> = None;
        let mut max_key: Option<Vec<u8>> = None;
        let mut entry_count = 0u64;
        let mut data_size = 0u64;
        let mut keys: Vec<Vec<u8>> = Vec::new();
        let mut index = SparseIndex::new();
        let mut max_seq: Option<u32> = None;

        loop {
            // `data_size` is the cumulative byte length read so far == the offset
            // of the frame we are about to read.
            let frame_offset = data_size;

            let mut size_buf = [0u8; 4];
            match file.read_exact(&mut size_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(LSMError::Io(e)),
            }

            let size = u32::from_le_bytes(size_buf);
            let mut entry_bytes = vec![0u8; size as usize];
            file.read_exact(&mut entry_bytes)?;

            let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes)?) };
            let key = archived.key.to_vec();

            let seq = archived.seq.to_native();
            if max_seq.is_none_or(|m| seq_newer(seq, m)) {
                max_seq = Some(seq);
            }

            if min_key.is_none() {
                min_key = Some(key.clone());
            }
            max_key = Some(key.clone());
            if entry_count.is_multiple_of(SparseIndex::SAMPLE_INTERVAL) {
                index.push(key.clone(), frame_offset);
            }
            keys.push(key);

            entry_count += 1;
            data_size += 4 + size as u64;
        }

        let bloom = (!keys.is_empty()).then(|| BloomFilter::build(keys.iter().map(|k| k.as_slice()), keys.len()));
        let index = (!index.is_empty()).then_some(index);

        let min_key = min_key.unwrap_or_default();
        let max_key = max_key.unwrap_or_default();

        Ok((
            SStableMetadata {
                bucket,
                min_key_prefix: key_prefix_of(&min_key),
                max_key_prefix: key_prefix_of(&max_key),
                min_key,
                max_key,
                entry_count,
                data_offset: 0,
                data_size,
            },
            bloom,
            index,
            max_seq,
        ))
    }

    /// Search all Level 0 files for a key and return the hit with the newest
    /// sequence (a value or a tombstone), or [`SsLookup::Missing`] if no L0 file
    /// mentions it.
    ///
    /// Resolution is by sequence, not file recency: a GC re-point can land a
    /// low-sequence value in a newer L0 file while the deleting tombstone sits in
    /// an older one, so we must inspect every file and keep the newest-sequence
    /// entry. Files are visited newest-first only so an exact sequence tie
    /// resolves to the newer file (the relocation, whose old copy is stale).
    fn search_level0_files(&self, bucket: u32, key: &[u8]) -> Result<SsLookup> {
        let mut entries = self.level0_entries_snapshot(bucket);
        if entries.is_empty() {
            return Ok(SsLookup::Missing);
        }
        if let Some(m) = self.metrics() {
            Metrics::bump(&m.l0_probes);
        }

        entries.sort_by(|a, b| b.created_at_ms.cmp(&a.created_at_ms));

        let mut best = SsLookup::Missing;
        for entry in entries {
            if entry.is_obsolete() {
                continue;
            }
            let _guard = L0ReadGuard::new(Arc::clone(&entry));
            let hit = self.lookup_in_sstable_file_path(entry.path.clone(), key)?;
            if let Some(seq) = hit.seq()
                && best.seq().is_none_or(|b| seq_newer(seq, b))
            {
                best = hit;
            }
        }

        Ok(best)
    }

    /// Merge all Level 0 files into Level 1 file for a bucket
    fn merge_level0_to_level1(&self, bucket: u32) -> Result<()> {
        let mut entries = self.level0_entries_snapshot(bucket);
        if entries.is_empty() {
            return Ok(());
        }
        let compaction_started = std::time::Instant::now();
        let mut bytes_merged = 0u64;

        // Collapse same-key conflicts across L0 files by highest write `seq`
        // (globally ordered, one counter for WAL and internal/test/bulk writes
        // alike), NOT by file recency. File `created_at_ms` is assigned at ro→L0
        // *flush* time, which can run out of seal/seq order (a higher-version
        // memtable can flush to L0 before a lower one, inverting recency relative
        // to seq); a GC re-point likewise preserves a low seq into a newer file.
        // Resolving by recency then let a lower-seq value beat a higher-seq
        // tombstone and resurrected deleted keys. Point reads (`get_with_seq`)
        // already resolve by seq, so seq-aware resolution here keeps the merge
        // consistent with them. The oldest-first sort only breaks exact seq ties
        // (prefer the newer file).
        entries.sort_by(|a, b| a.created_at_ms.cmp(&b.created_at_ms));

        let mut by_key: std::collections::BTreeMap<Vec<u8>, SStableEntry> = std::collections::BTreeMap::new();
        let mut any_read = false;
        for entry in &entries {
            if entry.is_obsolete() {
                continue;
            }
            any_read = true;
            let _guard = L0ReadGuard::new(Arc::clone(entry));
            // Within one file each key appears once; across files the highest-seq
            // record wins (exact ties resolve to the later/newer file).
            for record in self.read_sstable_entries_from_path(&entry.path)? {
                match by_key.entry(record.key.clone()) {
                    std::collections::btree_map::Entry::Occupied(mut e) => {
                        if seq_newer_or_eq(record.seq, e.get().seq) {
                            e.insert(record);
                        }
                    }
                    std::collections::btree_map::Entry::Vacant(e) => {
                        e.insert(record);
                    }
                }
            }
        }

        if !any_read || by_key.is_empty() {
            return Ok(());
        }

        let level0_records: Vec<KeyValueRecord> = by_key
            .into_values()
            .map(|e| KeyValueRecord {
                key: e.key,
                value: e.value,
                tombstone: e.tombstone,
                seq: e.seq,
            })
            .collect();

        let level1_path = Self::level1_path_from(&self.base_path, bucket);
        let level1_entries = self.read_sstable_entries(bucket)?;

        let merged_entries = self.two_way_merge(level1_entries, level0_records);

        if merged_entries.is_empty() {
            let bucket_idx = bucket as usize;
            let null_file = Arc::new(Self::create_null_file()?);
            {
                let mut file_guard = self.sstable_files[bucket_idx].write();
                *file_guard = null_file;
            }
            // Keep an empty file so the per-bucket Level-1 path is always present.
            if let Ok(f) = OpenOptions::new().write(true).create(true).truncate(true).open(&level1_path) {
                let _ = f.sync_all();
            }
            let mut metadata_guard = self.sstable_metadata[bucket_idx].write();
            metadata_guard.clear();
            *self.sstable_blooms[bucket_idx].write() = None;
            *self.sstable_indexes[bucket_idx].write() = None;
        } else {
            let temp_path = Self::level1_dir_from(&self.base_path).join(format!("level1_{}.dat.tmp", bucket));
            let (min_key, max_key, entry_count, data_size, index) = self.write_merged_sstable(&temp_path, &merged_entries)?;
            bytes_merged = data_size;

            let bucket_idx = bucket as usize;
            {
                let mut file_guard = self.sstable_files[bucket_idx].write();

                if level1_path.exists() {
                    let old_path = Self::level1_dir_from(&self.base_path).join(format!("level1_{}.dat.old", bucket));
                    std::fs::rename(&level1_path, &old_path)?;
                }

                std::fs::rename(&temp_path, &level1_path)?;

                let new_file = OpenOptions::new().read(true).write(true).open(&level1_path)?;
                *file_guard = Arc::new(new_file);
            }

            let mut metadata_guard = self.sstable_metadata[bucket_idx].write();
            metadata_guard.clear();
            metadata_guard.push(SStableMetadata {
                bucket,
                min_key_prefix: key_prefix_of(&min_key),
                max_key_prefix: key_prefix_of(&max_key),
                min_key,
                max_key,
                entry_count,
                data_offset: 0,
                data_size,
            });

            // Rebuild the L1 bloom from the merged entries (in memory, no extra
            // I/O). Replaced after the file swap so it always matches the file
            // now installed.
            let bloom = BloomFilter::build(merged_entries.iter().map(|e| e.key.as_slice()), merged_entries.len());
            *self.sstable_blooms[bucket_idx].write() = Some(Arc::new(bloom));
            // The sparse index was built from the byte offsets as the file was
            // written, so it matches the file just installed.
            *self.sstable_indexes[bucket_idx].write() = (!index.is_empty()).then(|| Arc::new(index));

            let old_path = Self::level1_dir_from(&self.base_path).join(format!("level1_{}.dat.old", bucket));
            if old_path.exists() {
                let _ = std::fs::remove_file(old_path);
            }
        }

        for entry in entries {
            entry.mark_obsolete();
        }

        if let Some(m) = self.metrics() {
            Metrics::bump(&m.l0_l1_compactions);
            Metrics::add(&m.compaction_bytes_merged, bytes_merged);
            Metrics::add(&m.compaction_duration_ms, compaction_started.elapsed().as_millis() as u64);
        }

        Ok(())
    }

    /// Compact a specific bucket's SSTable asynchronously
    pub(crate) fn compact_bucket(&self, bucket: u32) -> Result<()> {
        let bucket_idx = bucket as usize;

        if self.compaction_in_progress[bucket_idx].swap(true, Ordering::SeqCst) {
            return Ok(());
        }

        // Clear the in-progress flag on every exit path — including the `?` error
        // return below and any panic. Resetting only on the success path would
        // let a failed merge wedge the flag at `true`, making every later
        // compact_bucket for this bucket a silent no-op until restart.
        struct InProgressGuard<'a>(&'a AtomicBool);
        impl Drop for InProgressGuard<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        let _guard = InProgressGuard(&self.compaction_in_progress[bucket_idx]);

        self.merge_level0_to_level1(bucket)?;
        Ok(())
    }

    /// Compact all buckets in parallel
    pub(crate) fn compact_all(&self) -> Result<()> {
        self.cleanup_pending_memtables();
        self.cleanup_obsolete_level0_files();

        let ro_snapshot = self.read_only_memtables.read().clone();
        if ro_snapshot.is_empty() && !self.has_level0_files() {
            return Ok(());
        }
        for ro in ro_snapshot.iter() {
            self.flush_ro_memtable_to_level0(ro)?;
        }

        for bucket in 0u32..self.config.num_buckets as u32 {
            if let Err(err) = self.compact_bucket(bucket) {
                if let LSMError::Io(io_err) = &err
                    && io_err.kind() == std::io::ErrorKind::NotFound
                {
                    continue;
                }
                return Err(err);
            }
        }

        self.defer_old_memtables_cleanup()?;
        self.write_manifest_snapshot()?;
        Ok(())
    }

    pub(crate) fn has_level0_files(&self) -> bool {
        for bucket in 0u32..self.config.num_buckets as u32 {
            if !self.level0_files[bucket as usize].read().is_empty() {
                return true;
            }
        }
        false
    }

    pub(crate) fn has_compaction_work(&self) -> bool {
        !self.read_only_memtables.read().is_empty() || self.has_level0_files()
    }

    fn manifest_path(&self) -> PathBuf {
        self.base_path.join("manifest")
    }

    pub fn build_manifest_snapshot(&self) -> Result<LsmManifest> {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut levels = Vec::new();

        let mut l0_buckets = Vec::with_capacity(self.config.num_buckets);
        for bucket in 0u32..self.config.num_buckets as u32 {
            let files = self.level0_files[bucket as usize]
                .read()
                .iter()
                .map(|entry| ManifestFile {
                    path: entry.path.to_string_lossy().into_owned(),
                    created_at_ms: entry.created_at_ms,
                    entry_count: 0,
                })
                .collect();
            l0_buckets.push(ManifestBucket { bucket, files });
        }
        levels.push(ManifestLevel {
            level: 0,
            buckets: l0_buckets,
        });

        let mut l1_buckets = Vec::with_capacity(self.config.num_buckets);
        for bucket in 0u32..self.config.num_buckets as u32 {
            let mut files = Vec::new();
            let metadata_guard = self.sstable_metadata[bucket as usize].read();
            if let Some(meta) = metadata_guard.first() {
                let path = Self::level1_path_from(&self.base_path, bucket);
                files.push(ManifestFile {
                    path: path.to_string_lossy().into_owned(),
                    created_at_ms: 0,
                    entry_count: meta.entry_count,
                });
            }
            l1_buckets.push(ManifestBucket { bucket, files });
        }
        levels.push(ManifestLevel {
            level: 1,
            buckets: l1_buckets,
        });

        Ok(LsmManifest::new(levels, now_ms))
    }

    pub(crate) fn write_manifest_snapshot(&self) -> Result<()> {
        let manifest = self.build_manifest_snapshot()?;
        let path = self.manifest_path();
        std::fs::create_dir_all(path.parent().unwrap_or(&self.base_path))?;
        manifest.write_to_path(&path).map_err(LSMError::Io)
    }

    #[allow(dead_code)]
    pub fn load_manifest(&self) -> Result<Option<LsmManifest>> {
        let path = self.manifest_path();
        if !path.exists() {
            return Ok(None);
        }
        let manifest = LsmManifest::read_from_path(&path).map_err(LSMError::Io)?;
        Ok(Some(manifest))
    }

    /// Get statistics about the LSM tree
    pub(crate) fn stats(&self) -> LSMStats {
        let memtable = self.memtable.read();
        let read_only = self.read_only_memtables.read();

        let compaction_in_progress = self.compaction_in_progress.iter().any(|flag| flag.load(Ordering::SeqCst));

        LSMStats {
            memtable_entries: memtable.skip_list.number_of_live_nodes(),
            read_only_entries: read_only.iter().map(|ro| ro.records().len()).sum(),
            read_only_count: read_only.len(),
            compaction_in_progress,
        }
    }

    /// Efficient prefix scan across all layers (memtable, read-only, SSTables)
    /// Returns vector of (key, value_log_offset) pairs matching the prefix
    ///
    /// POINTER-RESOLUTION INVARIANT: the returned pointers guarantee LSM-side
    /// completeness only. They are safe to resolve against the value log *solely*
    /// inside the caller's value-log generation bracket
    /// ([`KVStore::read_generation_stable`](crate::db::kv_store::KVStore)) or under
    /// the value-log bucket write lock (GC compaction). This snapshot does NOT
    /// extend value-log GC protection — resolving a returned pointer outside that
    /// bracket reopens the wrong-file window closed in commit 420ac8e.
    pub(crate) fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, u128, u32)>> {
        // Merge stays oldest→newest into a BTreeMap<key, Option<(pointer, seq)>> with
        // overwrite, so active-memtable tombstones still shadow stale SSTable
        // entries (the precedence the tombstone fixes established).
        //
        // NOTE: this resolves by recency, whereas `get_with_seq` resolves by
        // `seq`; they agree only under the recency==seq invariant the GC re-point
        // guard maintains (see `key_pointer_pairs` / `merge_level0_to_level1`).
        //
        // What changes for concurrency is the *capture* order, which is the
        // reverse — newest layer first. Flush/compaction relocate a key
        // active→RO→L0→L1; capturing in that same direction guarantees the key is
        // seen in the newer layer before it can move into an older layer we have
        // not captured yet, so it is never absent from every captured layer at
        // once. We capture cheap, frozen handles (active+RO snapshot, L0 read
        // guards, L1 `Arc<File>`) up front and do the I/O afterwards, then merge.
        let mut all_entries: std::collections::BTreeMap<Vec<u8>, Option<(u128, u32)>> = std::collections::BTreeMap::new();

        // Capture the memtable layers (newest) first: snapshot the active
        // memtable's matching records and clone the read-only list under a single
        // `memtable.read()` guard. Holding that guard excludes `flush_memtable`
        // (which needs the write lock), so a key cannot slip from active into RO
        // between the two captures. The read-only records are `Arc`s, frozen even
        // if the list is later mutated.
        let (active_snapshot, ro_snapshot) = {
            let memtable = self.memtable.read();
            let mut active = Vec::new();
            for (key, value, seq, tombstone) in memtable.skip_list.iter_raw_from(prefix) {
                if !key.starts_with(prefix) {
                    break;
                }
                active.push((key.to_vec(), if tombstone { None } else { Some((value, seq)) }));
            }
            let ro = self.read_only_memtables.read().clone();
            (active, ro)
        };

        // SSTables (oldest layers), buckets scanned in parallel. Each bucket
        // captures L0 before L1 internally and returns entries ordered L1 (oldest)
        // → L0 (oldest→newest) with tombstones carried as `None`.
        //
        // Only buckets that could hold a matching key spawn a worker: one whose L1
        // does not overlap `[prefix, upper)` *and* has no non-obsolete L0 file has
        // nothing to contribute. A key that races into such a bucket after this
        // check (memtable→L0 flush, or L0→L1 merge) was already in the memtable
        // snapshot captured above, so skipping the spawn cannot drop it.
        let upper_owned = prefix_upper_bound(prefix);
        let upper = upper_owned.as_deref();
        {
            let mut first_err: Option<LSMError> = None;
            std::thread::scope(|s| {
                let handles: Vec<_> = (0u32..self.config.num_buckets as u32)
                    .filter(|&bucket| self.bucket_has_nonobsolete_l0(bucket as usize) || self.l1_overlaps(bucket as usize, prefix, upper))
                    .map(|bucket| s.spawn(move || self.scan_prefix_in_bucket(bucket, prefix, upper)))
                    .collect();
                for handle in handles {
                    match handle
                        .join()
                        .unwrap_or_else(|_| Err(LSMError::Io(std::io::Error::other("thread panicked"))))
                    {
                        Ok(entries) => {
                            for (key, value) in entries {
                                all_entries.insert(key, value);
                            }
                        }
                        Err(e) => {
                            if first_err.is_none() {
                                first_err = Some(e);
                            }
                        }
                    }
                }
            });
            if let Some(e) = first_err {
                return Err(e);
            }
        }

        // Read-only memtables oldest-first — overwrite SSTable data.
        for ro_memtable in ro_snapshot.iter() {
            let records = ro_memtable.records();
            for record in records.iter() {
                if record.key.starts_with(prefix) {
                    all_entries.insert(record.key.clone(), if record.tombstone { None } else { Some((record.value, record.seq)) });
                }
            }
        }

        // Active memtable (newest layer) — overwrites everything below.
        for (key, value) in active_snapshot {
            all_entries.insert(key, value);
        }

        Ok(all_entries.into_iter().filter_map(|(key, val)| val.map(|(v, s)| (key, v, s))).collect())
    }

    /// Whether this bucket's L1 file holds any key in `[start, end)`, decided from
    /// the in-memory min/max metadata alone (no file I/O). A `false` provably means
    /// the L1 file has no matching key, so a scan can skip reading it entirely; an
    /// empty/absent L1 (no metadata, or `entry_count == 0`) is also `false`.
    ///
    /// Safe under concurrent compaction *provided the caller has already captured
    /// this bucket's L0 read guards*: a merge marks the old L0 files obsolete only
    /// after installing the new L1 file and updating this metadata (see
    /// `merge_level0_to_level1`). So whatever version of the metadata a scan reads,
    /// any key it would cause the scan to skip is either already reflected here or
    /// still present in a captured (not-yet-obsolete) L0 file — never invisible.
    fn l1_overlaps(&self, bucket: usize, start: &[u8], end: Option<&[u8]>) -> bool {
        let metadata = self.sstable_metadata[bucket].read();
        match metadata.first() {
            Some(meta) if meta.entry_count > 0 => meta.max_key.as_slice() >= start && end.is_none_or(|e| meta.min_key.as_slice() < e),
            _ => false,
        }
    }

    /// Whether this bucket has any non-obsolete Level-0 file. Used only to decide
    /// whether a scan needs to spawn a worker for the bucket at all.
    fn bucket_has_nonobsolete_l0(&self, bucket: usize) -> bool {
        self.level0_files[bucket].read().iter().any(|e| !e.is_obsolete())
    }

    /// Read this bucket's L1 file for keys in `[start, end)`, appending each as a
    /// [`ScanEntry`] (tombstone carried as `None`). Entries are key-sorted, so the
    /// scan seeks near `start` via the sparse index and stops as soon as it reaches
    /// `end`.
    ///
    /// The sparse-index offset is validated against the file with `valid_scan_start`
    /// (a concurrent compaction can swap the file under the in-memory index): a
    /// stale hint just falls back to a full scan from 0, so it is a performance
    /// hint, never a correctness input.
    fn scan_l1_range_into(
        &self,
        bucket: usize,
        file: &File,
        start: &[u8],
        end: Option<&[u8]>,
        max_live: Option<usize>,
        out: &mut Vec<ScanEntry>,
    ) -> Result<()> {
        self.for_each_l1_entry_in_range(bucket, file, start, end, max_live, |archived| {
            let val = if archived.tombstone {
                None
            } else {
                Some((archived.value.to_native(), archived.seq.to_native()))
            };
            out.push((archived.key.as_slice().to_vec(), val));
        })
    }

    /// Visit every L1 entry whose key is in `[start, end)`, in ascending key order,
    /// seeking near `start` via the sparse index and stopping at `end`.
    ///
    /// The shared L1 range primitive behind `scan_l1_range_into`, `scan_prefixes`
    /// (once per requested prefix), and `range_keys_bounded`. `visit` receives each
    /// archived entry (borrowed only for the call) so callers extract whatever value
    /// shape they need — pointer+seq, pointer only, or liveness.
    ///
    /// The sparse-index offset is validated against the file (`valid_scan_start`): a
    /// stale hint after a concurrent compaction falls back to a full scan, so it is a
    /// performance hint, never a correctness input.
    /// `max_live`, when set, stops the scan after that many **non-tombstone** entries
    /// have been visited (tombstones are still visited but do not count). This is the
    /// per-bucket early-stop that bounds a paginated scan: since keys are hash-sharded
    /// (each key lives in exactly one bucket) and this file is key-sorted, a bucket's
    /// smallest `k` live keys are enough for the caller to assemble the global smallest
    /// `k`. Callers that need the whole range (unbounded scans) pass `None`.
    fn for_each_l1_entry_in_range(
        &self,
        bucket: usize,
        file: &File,
        start: &[u8],
        end: Option<&[u8]>,
        max_live: Option<usize>,
        mut visit: impl FnMut(&ArchivedSStableEntry),
    ) -> Result<()> {
        let start_offset = {
            let index = self.sstable_indexes[bucket].read();
            index.as_ref().map(|i| i.block_start(start)).unwrap_or(0)
        };
        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
        let mut offset = if start_offset != 0 && Self::valid_scan_start(file, start_offset, file_len, start) {
            start_offset
        } else {
            0
        };

        let mut live_seen = 0usize;
        let mut entry_bytes = Vec::new();
        loop {
            let mut size_buf = [0u8; 4];
            if !Self::read_exact_at(file, &mut size_buf, &mut offset)? {
                break;
            }
            let size = u32::from_le_bytes(size_buf) as usize;
            entry_bytes.resize(size, 0);
            Self::read_exact_at(file, &mut entry_bytes[..size], &mut offset)?;
            let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes[..size])?) };
            let key = archived.key.as_slice();
            // Sorted order: once we reach `end`, no later key can match.
            if end.is_some_and(|e| key >= e) {
                break;
            }
            // The seeked block may begin below `start`; skip those pre-window keys.
            if key >= start {
                visit(archived);
                if !archived.tombstone {
                    live_seen += 1;
                    if max_live.is_some_and(|m| live_seen >= m) {
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    /// Scan prefix within a specific SSTable bucket using key_prefix for efficiency.
    ///
    /// Returns matching entries as `(key, Option<pointer>)` ordered **L1 (oldest)
    /// first, then L0 oldest-to-newest** — the same oldest→newest layer order the
    /// other scans use. A tombstone is carried as `None` (not dropped) so the
    /// caller's merge map can let a newer L0 tombstone shadow a live L1 value,
    /// and a newer L0 value shadow a stale L1 value. Dropping tombstones here (as
    /// the old implementation did) resurrected deleted keys and could surface a
    /// stale L1 value over an updated L0 one, because L0 holds entries — including
    /// tombstones — not yet compacted into L1.
    ///
    /// `upper` is the prefix's upper bound (`prefix_upper_bound(prefix)`): the L1
    /// read skips a non-overlapping file via min/max and seeks via the sparse index
    /// over `[prefix, upper)`. L0 files carry no min/max or index, so they are still
    /// scanned in full with a prefix match.
    fn scan_prefix_in_bucket(&self, bucket: u32, prefix: &[u8], upper: Option<&[u8]>) -> Result<Vec<ScanEntry>> {
        let mut results = Vec::new();
        let mut entry_bytes = Vec::new();

        let matches_prefix = |archived: &ArchivedSStableEntry, key_slice: &[u8]| -> bool {
            if prefix.len() <= 8 {
                let prefix_len = prefix.len();
                let entry_prefix_bytes = archived.key_prefix.to_native().to_be_bytes();
                &entry_prefix_bytes[..prefix_len] == prefix
            } else {
                key_slice.starts_with(prefix)
            }
        };

        // Capture the SSTable handles newest-first — L0 *before* L1 — so a
        // concurrent L0→L1 merge cannot drop a key in the gap between the two
        // reads. The read guards pin every captured L0 file, so a merge that
        // marks them obsolete (and a cleanup that would delete them) cannot run
        // until this scan releases them; any key the merge moves into L1 after we
        // captured L0 is still found in L1, which we capture afterwards. The merge
        // order itself is unchanged: L1 (oldest) then L0 oldest→newest, so the
        // caller's overwrite lets newer L0 entries/tombstones shadow older ones.
        let mut l0_guards: Vec<L0ReadGuard> = {
            let entries = self.level0_files[bucket as usize].read();
            entries
                .iter()
                .filter(|e| !e.is_obsolete())
                .map(|e| L0ReadGuard::new(Arc::clone(e)))
                .collect()
        };
        l0_guards.sort_by(|a, b| a.entry.created_at_ms.cmp(&b.entry.created_at_ms));

        // Search the Level 1 file (oldest layer) first — but only if its min/max
        // says it overlaps `[prefix, upper)`, and then seeking near `prefix` via the
        // sparse index rather than reading from offset 0. The L0 guards are captured
        // above, before this metadata read, so the skip is race-safe (see
        // `l1_overlaps`). `[prefix, upper)` is exactly the keys starting with `prefix`.
        if self.l1_overlaps(bucket as usize, prefix, upper) {
            let level1_file = self.sstable_files[bucket as usize].read().clone();
            self.scan_l1_range_into(bucket as usize, &level1_file, prefix, upper, None, &mut results)?;
        }

        // Then the captured Level 0 files oldest-first.
        for guard in &l0_guards {
            let mut file = File::open(&guard.entry.path)?;
            loop {
                let mut size_buf = [0u8; 4];
                match file.read_exact(&mut size_buf) {
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(LSMError::Io(e)),
                }

                let size = u32::from_le_bytes(size_buf) as usize;
                entry_bytes.resize(size, 0);
                file.read_exact(&mut entry_bytes[..size])?;

                let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes[..size])?) };

                let key_slice = archived.key.as_slice();
                if matches_prefix(archived, key_slice) {
                    let val = if archived.tombstone {
                        None
                    } else {
                        Some((archived.value.to_native(), archived.seq.to_native()))
                    };
                    results.push((key_slice.to_vec(), val));
                }
            }
        }

        Ok(results)
    }

    /// Scan multiple 4-byte BE u32 prefixes in a **single pass per bucket**.
    ///
    /// Where `scan_prefix` × N prefixes would do N × num_buckets full linear
    /// scans of the level1 SSTables (each entry read via two `pread` syscalls),
    /// this method reads each bucket's level1 file **once** into an in-memory
    /// buffer and checks every entry against the full `prefix_ids` set — cutting
    /// `pread` syscall count from O(N_prefixes × entries_per_bucket) to O(1)
    /// per bucket.
    ///
    /// Returns a flat `Vec<(key, pointer)>`; the caller maps entries back to
    /// their originating prefix via the first 4 bytes of each key.
    ///
    /// Same pointer-resolution invariant as [`scan_prefix`](Self::scan_prefix):
    /// returned pointers are LSM-complete but must be resolved against the value log
    /// only inside the value-log generation bracket / GC bucket lock.
    pub(crate) fn scan_prefixes(&self, prefix_ids: &std::collections::HashSet<u32>) -> Result<Vec<(Vec<u8>, u128)>> {
        if prefix_ids.is_empty() {
            return Ok(vec![]);
        }

        // Merge stays oldest→newest into a BTreeMap<key, Option<pointer>> with
        // overwrite (L1 → L0 → ro_memtables → active), so tombstone precedence is
        // unchanged. Only the *capture* order is reversed to newest-first so a
        // concurrent flush/compaction cannot drop a live key — see `scan_prefix`.
        let mut all_entries: std::collections::BTreeMap<Vec<u8>, Option<u128>> = std::collections::BTreeMap::new();

        // Memtable layers (newest) captured first, under one `memtable.read()`
        // guard so a flush cannot split a key between active and RO.
        let (active_snapshot, ro_snapshot) = {
            let memtable = self.memtable.read();
            let mut active = Vec::new();
            for prefix_id in prefix_ids {
                let prefix = prefix_id.to_be_bytes();
                for (key, value, _seq, tombstone) in memtable.skip_list.iter_raw_from(&prefix) {
                    if key.len() < 4 || u32::from_be_bytes(key[..4].try_into().unwrap()) != *prefix_id {
                        break;
                    }
                    active.push((key.to_vec(), if tombstone { None } else { Some(value) }));
                }
            }
            let ro = self.read_only_memtables.read().clone();
            (active, ro)
        };

        // SSTable handles captured newest-first (L0 before L1); read in merge
        // order (L1 then L0). Guards in `l0_guards` pin the files until we finish.
        let (l0_guards, l1_files) = self.capture_sstable_layers();

        // Each requested prefix is a 4-byte big-endian cluster id; a prefix scan
        // over id `P` is a range over `[P, P+1)`. Sorted so per-bucket seeks are
        // monotonic. The span `[min_id, max_id+1)` gives a cheap per-bucket overlap
        // test (keys are hash-sharded, so a bucket's L1 spans many ids), and each
        // bucket then seeks per requested id via the sparse index rather than
        // reading the whole file — the win for probing a few clusters out of a
        // large L1.
        let mut sorted_ids: Vec<u32> = prefix_ids.iter().copied().collect();
        sorted_ids.sort_unstable();
        let sorted_ids = &sorted_ids;
        let span_start = sorted_ids[0].to_be_bytes();
        let span_end: Option<[u8; 4]> = sorted_ids[sorted_ids.len() - 1].checked_add(1).map(u32::to_be_bytes);
        let span_end = span_end.as_ref().map(|e| e.as_slice());

        // ── 1. L1 SSTables (oldest layer) — overlapping buckets read in parallel ─
        {
            let mut first_err: Option<LSMError> = None;
            std::thread::scope(|s| {
                let handles: Vec<_> = l1_files
                    .iter()
                    .enumerate()
                    .filter(|(bucket, _)| self.l1_overlaps(*bucket, &span_start, span_end))
                    .map(|(bucket, file)| {
                        s.spawn(move || -> Result<Vec<(Vec<u8>, Option<u128>)>> {
                            let mut entries = Vec::new();
                            for &prefix_id in sorted_ids {
                                let start = prefix_id.to_be_bytes();
                                let end = prefix_id.checked_add(1).map(u32::to_be_bytes);
                                self.for_each_l1_entry_in_range(bucket, file, &start, end.as_ref().map(|e| e.as_slice()), None, |archived| {
                                    let val = if archived.tombstone { None } else { Some(archived.value.to_native()) };
                                    entries.push((archived.key.as_slice().to_vec(), val));
                                })?;
                            }
                            Ok(entries)
                        })
                    })
                    .collect();
                for handle in handles {
                    match handle
                        .join()
                        .unwrap_or_else(|_| Err(LSMError::Io(std::io::Error::other("thread panicked"))))
                    {
                        Ok(entries) => {
                            for (k, v) in entries {
                                all_entries.insert(k, v);
                            }
                        }
                        Err(e) => {
                            if first_err.is_none() {
                                first_err = Some(e);
                            }
                        }
                    }
                }
            });
            if let Some(e) = first_err {
                return Err(e);
            }
        }

        // ── 2. L0 SSTables (newer than L1) — captured guards, oldest-first per
        //    bucket so newer L0 entries overwrite older ones. Only buckets that hold
        //    a captured L0 file spawn a worker (L0 has no min/max or sparse index).
        {
            let mut first_err: Option<LSMError> = None;
            std::thread::scope(|s| {
                let handles: Vec<_> = l0_guards
                    .iter()
                    .filter(|guards| !guards.is_empty())
                    .map(|guards| {
                        s.spawn(move || -> Result<Vec<(Vec<u8>, Option<u128>)>> {
                            let mut bucket_entries = Vec::new();
                            for guard in guards {
                                let mut file = File::open(&guard.entry.path)?;
                                let mut entry_bytes: Vec<u8> = Vec::new();
                                loop {
                                    let mut size_buf = [0u8; 4];
                                    match file.read_exact(&mut size_buf) {
                                        Ok(_) => {}
                                        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                                        Err(e) => return Err(LSMError::Io(e)),
                                    }
                                    let size = u32::from_le_bytes(size_buf) as usize;
                                    entry_bytes.resize(size, 0);
                                    file.read_exact(&mut entry_bytes[..size])?;
                                    let archived =
                                        unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes[..size])?) };
                                    if archived.key.len() >= 4 {
                                        let prefix_be = archived.key_prefix.to_native().to_be_bytes();
                                        let prefix_id = u32::from_be_bytes(prefix_be[..4].try_into().unwrap());
                                        if prefix_ids.contains(&prefix_id) {
                                            let val = if archived.tombstone { None } else { Some(archived.value.to_native()) };
                                            bucket_entries.push((archived.key.as_slice().to_vec(), val));
                                        }
                                    }
                                }
                            }
                            Ok(bucket_entries)
                        })
                    })
                    .collect();
                for handle in handles {
                    match handle
                        .join()
                        .unwrap_or_else(|_| Err(LSMError::Io(std::io::Error::other("thread panicked"))))
                    {
                        Ok(entries) => {
                            for (k, v) in entries {
                                all_entries.insert(k, v);
                            }
                        }
                        Err(e) => {
                            if first_err.is_none() {
                                first_err = Some(e);
                            }
                        }
                    }
                }
            });
            if let Some(e) = first_err {
                return Err(e);
            }
        }

        // ── 3. Read-only memtables oldest-first — overwrite SSTable data ───────
        for ro_memtable in ro_snapshot.iter() {
            for record in ro_memtable.records().iter() {
                if record.key.len() >= 4 {
                    let prefix_id = u32::from_be_bytes(record.key[..4].try_into().unwrap());
                    if prefix_ids.contains(&prefix_id) {
                        all_entries.insert(record.key.clone(), if record.tombstone { None } else { Some(record.value) });
                    }
                }
            }
        }

        // ── 4. Active memtable (newest layer) — overwrites everything below ────
        for (key, value) in active_snapshot {
            all_entries.insert(key, value);
        }

        Ok(all_entries.into_iter().filter_map(|(key, val)| val.map(|v| (key, v))).collect())
    }

    /// Fetch multiple keys in a **single pass per bucket**.
    ///
    /// Where `get` × N keys calls `search_in_sstable_file` N times (each a linear
    /// scan with per-entry `pread` syscalls), this method groups keys by their hash
    /// bucket, reads each bucket's level1 file **once** into memory, and resolves all
    /// keys for that bucket in a single in-memory pass.
    ///
    /// Returns one `Option<u128>` per input key in the same order.
    /// `None` means the key does not exist or was deleted (tombstone).
    pub(crate) fn get_multiple(&self, keys: &[Vec<u8>]) -> Result<Vec<Option<(u128, u32)>>> {
        let n = keys.len();
        // Each resolved key carries its write `seq` (low u32) so the caller can
        // verify the value record still belongs to this write (stale-pointer /
        // recycled-slot detection — see `KVStore::get_multiple_inner`).
        let mut results: Vec<Option<(u128, u32)>> = vec![None; n];
        let mut resolved: Vec<bool> = vec![false; n];

        // ── 1. Active memtable ────────────────────────────────────────────────
        {
            let memtable = self.memtable.read();
            for i in 0..n {
                // `entry` returns the value, seq and tombstone flag, so we can
                // distinguish "not present" from "deleted" and avoid falling
                // through to older layers for tombstoned keys.
                if let Some((value, seq, tombstone)) = memtable.skip_list.entry(&keys[i]) {
                    results[i] = (!tombstone).then_some((value, seq));
                    resolved[i] = true;
                }
            }
        }

        // ── 2. Read-only memtables ────────────────────────────────────────────
        {
            let ro_memtables = self.read_only_memtables.read();
            for ro in ro_memtables.iter().rev() {
                let records = ro.records();
                for i in 0..n {
                    if resolved[i] {
                        continue;
                    }
                    if let Some(record) = records.iter().find(|r| r.key.as_slice() == keys[i].as_slice()) {
                        if !record.tombstone {
                            results[i] = Some((record.value, record.seq));
                        }
                        resolved[i] = true;
                    }
                }
            }
        }

        // ── 3. Group unresolved keys by SSTable bucket ────────────────────────
        let mut pending_by_bucket: Vec<Vec<usize>> = (0..self.config.num_buckets).map(|_| Vec::new()).collect();
        for i in 0..n {
            if !resolved[i] {
                let bucket = get_bucket_for_key(&keys[i], self.config.num_buckets) as usize;
                pending_by_bucket[bucket].push(i);
            }
        }

        // ── 4. One L0+L1 scan per bucket in parallel — each bucket's key-set is
        //    disjoint (hash-sharded), so threads write to non-overlapping result slots.
        {
            let mut per_bucket: Vec<Vec<IdxEntry>> = (0..self.config.num_buckets).map(|_| Vec::new()).collect();
            let mut first_err: Option<LSMError> = None;
            std::thread::scope(|s| {
                let handles: Vec<_> = (0..self.config.num_buckets)
                    .map(|bucket| {
                        let pending = &pending_by_bucket[bucket];
                        s.spawn(move || -> Result<Vec<IdxEntry>> {
                            if pending.is_empty() {
                                return Ok(Vec::new());
                            }
                            // Owned keys so the map is self-contained within the thread.
                            let mut lookup: std::collections::HashMap<Vec<u8>, usize> = pending.iter().map(|&idx| (keys[idx].clone(), idx)).collect();
                            let mut updates: Vec<IdxEntry> = Vec::new();

                            // Level 0: captured newest-first. The guards are held for the whole
                            // bucket's work (through the L1 pre-filter below), so a key a stale
                            // L1 skip would drop is still in a captured, already-scanned L0 file.
                            let mut l0_guards: Vec<L0ReadGuard> = self
                                .level0_entries_snapshot(bucket as u32)
                                .iter()
                                .filter(|e| !e.is_obsolete())
                                .map(|e| L0ReadGuard::new(Arc::clone(e)))
                                .collect();
                            l0_guards.sort_by(|a, b| b.entry.created_at_ms.cmp(&a.entry.created_at_ms));
                            for guard in &l0_guards {
                                if lookup.is_empty() {
                                    break;
                                }
                                let mut file = File::open(&guard.entry.path)?;
                                let mut entry_bytes: Vec<u8> = Vec::new();
                                loop {
                                    let mut size_buf = [0u8; 4];
                                    match file.read_exact(&mut size_buf) {
                                        Ok(_) => {}
                                        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                                        Err(e) => return Err(LSMError::Io(e)),
                                    }
                                    let size = u32::from_le_bytes(size_buf) as usize;
                                    entry_bytes.resize(size, 0);
                                    file.read_exact(&mut entry_bytes[..size])?;
                                    let archived =
                                        unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes[..size])?) };
                                    let key_vec = archived.key.as_slice().to_vec();
                                    if let Some(orig_idx) = lookup.remove(&key_vec) {
                                        let val = if archived.tombstone {
                                            None
                                        } else {
                                            Some((archived.value.to_native(), archived.seq.to_native()))
                                        };
                                        updates.push((orig_idx, val));
                                    }
                                }
                            }

                            if lookup.is_empty() {
                                return Ok(updates);
                            }

                            // L1 pre-filter: drop keys that provably cannot be in this bucket's L1
                            // file, so a bucket whose every remaining key is out of range or
                            // bloom-negative skips the whole-file read entirely. Same exact
                            // fast-rejects the point-lookup path uses (`lookup_level1`): min/max is
                            // exact, a bloom negative is exact.
                            //
                            // Race-safety mirrors the scan paths: the L0 guards captured above are
                            // still held, and `merge_level0_to_level1` installs the new L1 file +
                            // metadata/bloom BEFORE marking the old L0 files obsolete. So any key
                            // left in `lookup` that stale L1 metadata would let us skip must still
                            // be in a captured L0 file — which we have already scanned and removed
                            // from `lookup`. A key still in `lookup` and rejected here is therefore
                            // absent from every layer below the memtable, so leaving it `None` is
                            // correct.
                            //
                            // Conservative like `lookup_level1`: filters are applied only when the
                            // metadata/bloom are actually present. Absent metadata (e.g. a load
                            // failure on a non-empty file) applies no filter, so we fall through and
                            // read rather than risk skipping a live key — the L1 read stays correct.
                            {
                                let min_max = {
                                    let meta = self.sstable_metadata[bucket].read();
                                    meta.first().filter(|m| m.entry_count > 0).map(|m| (m.min_key.clone(), m.max_key.clone()))
                                };
                                if let Some((min_key, max_key)) = min_max {
                                    lookup.retain(|k, _| k.as_slice() >= min_key.as_slice() && k.as_slice() <= max_key.as_slice());
                                }
                                if !lookup.is_empty() {
                                    let bloom = self.sstable_blooms[bucket].read();
                                    if let Some(bloom) = bloom.as_ref() {
                                        lookup.retain(|k, _| bloom.contains(k));
                                    }
                                }
                                if lookup.is_empty() {
                                    // Every remaining key is provably absent from L1 — skip the read.
                                    return Ok(updates);
                                }
                            }

                            // Level 1: single large read, then in-memory key lookup.
                            let level1_file = self.sstable_files[bucket].read().clone();
                            let file_size = level1_file.metadata().map(|m| m.len() as usize).unwrap_or(0);
                            if file_size == 0 {
                                return Ok(updates);
                            }
                            let mut file_buf = vec![0u8; file_size];
                            let mut total_read = 0usize;
                            while total_read < file_size {
                                match level1_file.read_at(&mut file_buf[total_read..], total_read as u64) {
                                    Ok(0) => break,
                                    Ok(nr) => total_read += nr,
                                    Err(e) => return Err(LSMError::Io(e)),
                                }
                            }
                            let file_content = &file_buf[..total_read];
                            let mut pos = 0usize;
                            while pos + 4 <= file_content.len() && !lookup.is_empty() {
                                let size = u32::from_le_bytes(file_content[pos..pos + 4].try_into().unwrap()) as usize;
                                pos += 4;
                                if pos + size > file_content.len() {
                                    break;
                                }
                                let entry_slice = &file_content[pos..pos + size];
                                pos += size;
                                let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(entry_slice)?) };
                                let key_vec = archived.key.as_slice().to_vec();
                                if let Some(orig_idx) = lookup.remove(&key_vec) {
                                    let val = if archived.tombstone {
                                        None
                                    } else {
                                        Some((archived.value.to_native(), archived.seq.to_native()))
                                    };
                                    updates.push((orig_idx, val));
                                }
                            }
                            Ok(updates)
                        })
                    })
                    .collect();
                for (i, handle) in handles.into_iter().enumerate() {
                    match handle
                        .join()
                        .unwrap_or_else(|_| Err(LSMError::Io(std::io::Error::other("thread panicked"))))
                    {
                        Ok(updates) => per_bucket[i] = updates,
                        Err(e) => {
                            if first_err.is_none() {
                                first_err = Some(e);
                            }
                        }
                    }
                }
            });
            if let Some(e) = first_err {
                return Err(e);
            }
            for updates in per_bucket {
                for (idx, val) in updates {
                    results[idx] = val;
                }
            }
        }

        Ok(results)
    }

    /// Get all keys in the LSM tree (for compatibility with existing API)
    pub(crate) fn keys(&self) -> Result<Vec<Vec<u8>>> {
        // Delegate to key_pointer_pairs which merges layers in the correct order
        // (SSTable → read-only memtables → active memtable) with proper tombstone
        // suppression.  The old inline implementation added SSTable keys last,
        // so active-memtable tombstones never suppressed SSTable entries — causing
        // deleted keys to re-appear in iterators after a flush (e.g. WAL recovery).
        Ok(self.key_pointer_pairs(None)?.into_iter().map(|(k, _)| k).collect())
    }

    /// Get all key-value pairs from the LSM tree, optionally filtered by qualifying buckets.
    ///
    /// Eliminates the double-read pattern where `keys()` + per-key `get()` was used.
    /// When `qualifying_buckets` is provided, only keys that hash to a qualifying bucket
    /// are returned, and SSTable I/O is skipped entirely for non-qualifying buckets.
    ///
    /// Same pointer-resolution invariant as [`scan_prefix`](Self::scan_prefix):
    /// returned pointers are LSM-complete but must be resolved against the value log
    /// only inside the value-log generation bracket / GC bucket lock. (Value-log GC's
    /// `compact_single_bucket` satisfies this by holding the bucket write lock across
    /// this scan, the value reads, and the reinsert.)
    pub(crate) fn key_pointer_pairs(&self, qualifying_buckets: Option<&[bool]>) -> Result<Vec<(Vec<u8>, u128)>> {
        // Merge stays oldest→newest into a BTreeMap so last insert wins (L1 → L0 →
        // read-only → active), preserving tombstone precedence. The *capture* order
        // is reversed to newest-first so a concurrent flush/compaction cannot drop a
        // live key — critical here because this scan also backs value-log GC's
        // compaction pass, where a dropped live key means lost data. See `scan_prefix`.
        //
        // Liveness is resolved by highest write `seq` (globally ordered), matching
        // the seq-resolved point reads (`get_with_seq`) and the seq-aware L0→L1
        // merge. Layer/file recency is NOT used to pick the winner: an out-of-order
        // ro→L0 flush (file `created_at_ms` is set at flush time, which can run out
        // of seal/seq order) or a GC re-point can leave a lower-seq value in a
        // "newer" file above a higher-seq tombstone. Resolving by recency was the
        // *data-loss* direction — value-log GC would treat a higher-seq live value
        // as dead (a newer-file lower-seq tombstone shadowing it) and drop it. The
        // map carries each key's best `(value-or-tombstone, seq)`; a later insert
        // wins only if its seq is newer-or-equal. The *capture* order stays
        // newest-first (memtables before SSTables) so a concurrent flush/compaction
        // cannot drop a live key; only the conflict *resolution* is seq-based.
        let mut all_entries = std::collections::BTreeMap::<Vec<u8>, (Option<u128>, u32)>::new();
        // Seq-aware merge: keep the newest-seq view of each key (ties keep the
        // later insert, i.e. the newer capture layer).
        fn merge_kp(map: &mut std::collections::BTreeMap<Vec<u8>, (Option<u128>, u32)>, key: Vec<u8>, value: Option<u128>, seq: u32) {
            match map.entry(key) {
                std::collections::btree_map::Entry::Occupied(mut e) => {
                    if seq_newer_or_eq(seq, e.get().1) {
                        e.insert((value, seq));
                    }
                }
                std::collections::btree_map::Entry::Vacant(e) => {
                    e.insert((value, seq));
                }
            }
        }

        let buckets_to_read: Vec<u32> = (0u32..self.config.num_buckets as u32)
            .filter(|&b| qualifying_buckets.is_none_or(|qb| qb[b as usize]))
            .collect();

        // Memtable layers (newest) captured first, under one `memtable.read()` guard
        // so a flush cannot split a key between active and RO.
        let (active_snapshot, ro_snapshot) = {
            let memtable = self.memtable.read();
            let mut active = Vec::new();
            for (key, value, seq, tombstone) in memtable.skip_list.iter_raw() {
                if let Some(qb) = qualifying_buckets
                    && !qb[get_bucket_for_key(key, self.config.num_buckets) as usize]
                {
                    continue;
                }
                active.push((key.to_vec(), if tombstone { None } else { Some(value) }, seq));
            }
            let ro = self.read_only_memtables.read().clone();
            (active, ro)
        };

        // SSTable handles captured newest-first (L0 before L1); read in merge order.
        // We capture all buckets but only read the qualifying ones.
        let (l0_guards, l1_files) = self.capture_sstable_layers();

        // 1. Level-1 SSTables (oldest layer) — qualifying buckets read in parallel.
        //    scan_bucket_key_pointers returns only live entries (no tombstones at L1),
        //    so there is no cross-bucket tombstone ordering concern.
        {
            let l1_ref = &l1_files;
            let mut first_err: Option<LSMError> = None;
            std::thread::scope(|s| {
                let handles: Vec<_> = buckets_to_read
                    .iter()
                    .map(|&bucket| s.spawn(move || self.scan_bucket_key_pointers(&l1_ref[bucket as usize])))
                    .collect();
                for handle in handles {
                    match handle
                        .join()
                        .unwrap_or_else(|_| Err(LSMError::Io(std::io::Error::other("thread panicked"))))
                    {
                        Ok(kvs) => {
                            for (key, value, seq) in kvs {
                                merge_kp(&mut all_entries, key, Some(value), seq);
                            }
                        }
                        Err(e) => {
                            if first_err.is_none() {
                                first_err = Some(e);
                            }
                        }
                    }
                }
            });
            if let Some(e) = first_err {
                return Err(e);
            }
        }

        // 2. Level-0 SSTable files (newer than L1, older than the memtables). This
        //    step is essential for correctness: L0 holds data flushed off the
        //    memtable/read-only layer but not yet compacted into L1 — including
        //    tombstones. Omitting L0 makes a flushed tombstone invisible to GC, so
        //    GC resurrects a deleted key still live in L1, and can drop a live key
        //    that exists only in L0. Captured guards are read oldest-file-first per
        //    bucket so newer L0 entries (and tombstones) shadow older ones.
        {
            let l0_ref = &l0_guards;
            let mut first_err: Option<LSMError> = None;
            std::thread::scope(|s| {
                let handles: Vec<_> = buckets_to_read
                    .iter()
                    .map(|&bucket| {
                        s.spawn(move || -> Result<Vec<KpScanEntry>> {
                            let mut out = Vec::new();
                            for guard in &l0_ref[bucket as usize] {
                                let mut file = File::open(&guard.entry.path)?;
                                let mut entry_bytes: Vec<u8> = Vec::new();
                                loop {
                                    let mut size_buf = [0u8; 4];
                                    match file.read_exact(&mut size_buf) {
                                        Ok(_) => {}
                                        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                                        Err(e) => return Err(LSMError::Io(e)),
                                    }
                                    let size = u32::from_le_bytes(size_buf) as usize;
                                    entry_bytes.resize(size, 0);
                                    file.read_exact(&mut entry_bytes[..size])?;
                                    let archived =
                                        unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes[..size])?) };
                                    let val = if archived.tombstone { None } else { Some(archived.value.to_native()) };
                                    out.push((archived.key.as_slice().to_vec(), val, archived.seq.to_native()));
                                }
                            }
                            Ok(out)
                        })
                    })
                    .collect();
                for handle in handles {
                    match handle
                        .join()
                        .unwrap_or_else(|_| Err(LSMError::Io(std::io::Error::other("thread panicked"))))
                    {
                        Ok(kvs) => {
                            for (k, v, seq) in kvs {
                                merge_kp(&mut all_entries, k, v, seq);
                            }
                        }
                        Err(e) => {
                            if first_err.is_none() {
                                first_err = Some(e);
                            }
                        }
                    }
                }
            });
            if let Some(e) = first_err {
                return Err(e);
            }
        }

        // 3. Read-only memtables — resolved by seq (oldest-first only tie-breaks).
        for ro_memtable in ro_snapshot.iter() {
            for record in ro_memtable.records().iter() {
                if let Some(qb) = qualifying_buckets
                    && !qb[get_bucket_for_key(&record.key, self.config.num_buckets) as usize]
                {
                    continue;
                }
                merge_kp(
                    &mut all_entries,
                    record.key.clone(),
                    if record.tombstone { None } else { Some(record.value) },
                    record.seq,
                );
            }
        }

        // 4. Active memtable (captured last, but resolved by seq like the rest).
        for (key, value, seq) in active_snapshot {
            merge_kp(&mut all_entries, key, value, seq);
        }

        // Filter out tombstoned entries and collect results
        Ok(all_entries
            .into_iter()
            .filter_map(|(key, (value, _seq))| value.map(|v| (key, v)))
            .collect())
    }

    /// Scan all live key-value pairs (with their write `seq`) from a bucket's
    /// (pre-captured) Level-1 file. Tombstones are skipped: a seq-aware merge
    /// (`merge_level0_to_level1`) never writes a tombstone into L1, so an L1 entry
    /// is always live; a newer layer carrying the key shadows it by seq.
    fn scan_bucket_key_pointers(&self, level1_file: &File) -> Result<Vec<(Vec<u8>, u128, u32)>> {
        let mut kvs = Vec::new();
        let mut offset = 0u64;
        let mut entry_bytes = Vec::new();

        loop {
            let mut size_buf = [0u8; 4];
            if !Self::read_exact_at(level1_file, &mut size_buf, &mut offset)? {
                break;
            }

            let size = u32::from_le_bytes(size_buf) as usize;
            entry_bytes.resize(size, 0);
            Self::read_exact_at(level1_file, &mut entry_bytes[..size], &mut offset)?;

            let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes[..size])?) };

            if !archived.tombstone {
                kvs.push((archived.key.to_vec(), archived.value.to_native(), archived.seq.to_native()));
            }
        }

        Ok(kvs)
    }

    /// Returns the first `limit` live keys that are `>= start`, in sorted order.
    ///
    /// Unlike `keys()` which loads the entire key space, this method:
    /// - Uses an O(log N) seek in the memtable skip list (`iter_from`)
    /// - Filters read-only memtable records by `>= start`
    /// - Scans SSTable entries but skips those before `start`
    /// - Stops after collecting `limit` keys
    ///
    /// Used by cursor-based `scan()` to avoid O(N_total) work per page fetch.
    pub(crate) fn range_keys_bounded(&self, start: &[u8], limit: usize) -> Result<Vec<Vec<u8>>> {
        // Merge stays oldest→newest with overwrite (L1 → L0 → RO → active) so
        // tombstone precedence is unchanged; only the *capture* order is reversed to
        // newest-first so a concurrent flush/compaction cannot drop a live key. See
        // `scan_prefix`. The `take(limit)` runs on the fully merged, sorted map, so
        // limiting is unaffected by the capture order.
        let mut all_entries: std::collections::BTreeMap<Vec<u8>, bool> = std::collections::BTreeMap::new();

        // Memtable layers (newest) captured first, under one `memtable.read()` guard
        // so a flush cannot split a key between active and RO.
        let (active_snapshot, ro_snapshot) = {
            let memtable = self.memtable.read();
            let active: Vec<(Vec<u8>, bool)> = memtable
                .skip_list
                .iter_raw_from(start)
                .map(|(key, _value, _seq, tombstone)| (key.to_vec(), !tombstone))
                .collect();
            let ro = self.read_only_memtables.read().clone();
            (active, ro)
        };

        // SSTable handles captured newest-first (L0 before L1); read in merge order.
        let (l0_guards, l1_files) = self.capture_sstable_layers();

        // Pages ≤ this bound use the limit-bounded serial L1 path; larger scans keep the
        // parallel read-everything path.
        const BOUNDED_SCAN_LIMIT: usize = 10_000;

        // Collect the layers NEWER than L1 (L0 oldest-first per bucket → RO → active) into
        // one list in oldest→newest order — used to bound the L1 scan (over-read by the
        // newer in-range count, a safe upper bound on same-bucket deletes) and replayed over
        // L1 afterwards. See `range_pointers_bounded` for the full rationale.
        let mut newer: Vec<(Vec<u8>, bool)> = Vec::new();
        let bounded = limit <= BOUNDED_SCAN_LIMIT;

        // Reads one bucket's L0 files (oldest-first so newer L0 entries shadow older ones),
        // range-filtered (`>= start`). L0 has no min/max or sparse index, so files are scanned
        // in full.
        let read_l0_bucket = |bucket: usize| -> Result<Vec<(Vec<u8>, bool)>> {
            let mut bucket_entries = Vec::new();
            for guard in &l0_guards[bucket] {
                let mut file = File::open(&guard.entry.path)?;
                let mut entry_bytes: Vec<u8> = Vec::new();
                loop {
                    let mut size_buf = [0u8; 4];
                    match file.read_exact(&mut size_buf) {
                        Ok(_) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                        Err(e) => return Err(LSMError::Io(e)),
                    }
                    let size = u32::from_le_bytes(size_buf) as usize;
                    entry_bytes.resize(size, 0);
                    file.read_exact(&mut entry_bytes[..size])?;
                    let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes[..size])?) };
                    if archived.key.as_slice() >= start {
                        bucket_entries.push((archived.key.as_slice().to_vec(), !archived.tombstone));
                    }
                }
            }
            Ok(bucket_entries)
        };

        // L0 (newer than L1). Bounded page → read busy buckets SERIALLY (L0 is small in the
        // steady state, so a thread per bucket costs more than it saves); full/large scan →
        // parallel read-everything. See `range_pointers_bounded`.
        let busy_l0: Vec<usize> = (0..self.config.num_buckets).filter(|b| !l0_guards[*b].is_empty()).collect();
        if bounded {
            for bucket in busy_l0 {
                newer.extend(read_l0_bucket(bucket)?);
            }
        } else {
            let mut first_err: Option<LSMError> = None;
            std::thread::scope(|s| {
                let handles: Vec<_> = busy_l0.into_iter().map(|bucket| s.spawn(move || read_l0_bucket(bucket))).collect();
                for handle in handles {
                    match handle
                        .join()
                        .unwrap_or_else(|_| Err(LSMError::Io(std::io::Error::other("thread panicked"))))
                    {
                        Ok(entries) => newer.extend(entries),
                        Err(e) => {
                            if first_err.is_none() {
                                first_err = Some(e);
                            }
                        }
                    }
                }
            });
            if let Some(e) = first_err {
                return Err(e);
            }
        }
        for ro_memtable in ro_snapshot.iter() {
            for record in ro_memtable.records().iter() {
                if record.key.as_slice() >= start {
                    newer.push((record.key.clone(), !record.tombstone));
                }
            }
        }
        newer.extend(active_snapshot);

        // L1 SSTables (oldest layer) — only buckets whose min/max reaches `start` (range is
        // unbounded above). Bounded page → serial, read ≤ `limit + 1 + |newer in THIS bucket|`
        // LIVE keys per bucket (slack sharded per bucket since only a same-bucket newer entry
        // can shadow a bucket's L1 key); full/large scan → parallel, read-everything. See
        // `range_pointers_bounded`.
        match bounded {
            true => {
                let mut newer_per_bucket = vec![0usize; self.config.num_buckets];
                for (key, _) in newer.iter() {
                    newer_per_bucket[get_bucket_for_key(key, self.config.num_buckets) as usize] += 1;
                }
                for (bucket, file) in l1_files.iter().enumerate() {
                    if !self.l1_overlaps(bucket, start, None) {
                        continue;
                    }
                    let cap = limit.saturating_add(1).saturating_add(newer_per_bucket[bucket]);
                    self.for_each_l1_entry_in_range(bucket, file, start, None, Some(cap), |archived| {
                        all_entries.insert(archived.key.as_slice().to_vec(), !archived.tombstone);
                    })?;
                }
            }
            false => {
                let mut first_err: Option<LSMError> = None;
                std::thread::scope(|s| {
                    let handles: Vec<_> = l1_files
                        .iter()
                        .enumerate()
                        .filter(|(bucket, _)| self.l1_overlaps(*bucket, start, None))
                        .map(|(bucket, file)| {
                            s.spawn(move || -> Result<Vec<(Vec<u8>, bool)>> {
                                let mut entries = Vec::new();
                                self.for_each_l1_entry_in_range(bucket, file, start, None, None, |archived| {
                                    entries.push((archived.key.as_slice().to_vec(), !archived.tombstone));
                                })?;
                                Ok(entries)
                            })
                        })
                        .collect();
                    for handle in handles {
                        match handle
                            .join()
                            .unwrap_or_else(|_| Err(LSMError::Io(std::io::Error::other("thread panicked"))))
                        {
                            Ok(entries) => {
                                for (k, v) in entries {
                                    all_entries.insert(k, v);
                                }
                            }
                            Err(e) => {
                                if first_err.is_none() {
                                    first_err = Some(e);
                                }
                            }
                        }
                    }
                });
                if let Some(e) = first_err {
                    return Err(e);
                }
            }
        }

        // Replay the newer layers over L1 (oldest→newest).
        for (key, live) in newer {
            all_entries.insert(key, live);
        }

        Ok(all_entries
            .into_iter()
            .filter_map(|(key, live)| if live { Some(key) } else { None })
            .take(limit)
            .collect())
    }

    /// Like `range_keys_bounded` but returns `(key, pointer)` pairs instead of bare keys.
    ///
    /// Avoids a redundant per-key `lsm.get()` call in callers that immediately need to
    /// decode the pointer anyway (e.g. `resolve_entries_from_pointers`).
    ///
    /// Uses the same oldest-first, last-writer-wins `BTreeMap<key, Option<pointer>>`
    /// pattern as `key_pointer_pairs` so active-memtable tombstones always shadow stale
    /// SSTable entries — the same guarantee `lsm.keys()` provides.
    ///
    /// Same pointer-resolution invariant as [`scan_prefix`](Self::scan_prefix):
    /// returned pointers are LSM-complete but must be resolved against the value log
    /// only inside the value-log generation bracket / GC bucket lock.
    pub(crate) fn range_pointers_bounded(&self, start: &[u8], end: Option<&[u8]>, limit: usize) -> Result<Vec<(Vec<u8>, u128, u32)>> {
        // Merge stays oldest→newest with overwrite (L1 → L0 → RO → active) so
        // tombstone precedence is unchanged; only the *capture* order is reversed to
        // newest-first so a concurrent flush/compaction cannot drop a live key. See
        // `scan_prefix`. `take(limit)` runs on the fully merged, sorted map.
        //
        // `end` (exclusive) bounds the scan to `[start, end)`; entries `>= end` are
        // never inserted so the merge map and the resulting page stay within the
        // requested key window.
        let mut all_entries: std::collections::BTreeMap<Vec<u8>, Option<(u128, u32)>> = std::collections::BTreeMap::new();

        // Memtable layers (newest) captured first, under one `memtable.read()` guard
        // so a flush cannot split a key between active and RO.
        let (active_snapshot, ro_snapshot) = {
            let memtable = self.memtable.read();
            let active: Vec<ScanEntry> = memtable
                .skip_list
                .iter_raw_from(start)
                .take_while(|(key, _value, _seq, _tombstone)| end.is_none_or(|e| *key < e))
                .map(|(key, value, seq, tombstone)| (key.to_vec(), if tombstone { None } else { Some((value, seq)) }))
                .collect();
            let ro = self.read_only_memtables.read().clone();
            (active, ro)
        };

        // SSTable handles captured newest-first (L0 before L1); read in merge order.
        let (l0_guards, l1_files) = self.capture_sstable_layers();

        // Pages ≤ this bound use the limit-bounded serial L1 path; larger scans keep the
        // parallel read-everything path (see the L1 step below).
        const BOUNDED_SCAN_LIMIT: usize = 10_000;

        // Collect the layers NEWER than L1 — L0 (oldest-first per bucket), then read-only
        // memtables (oldest-first), then the active memtable (newest) — into one list in
        // that oldest→newest order. Two uses: (a) bound the L1 scan (a page needs only each
        // bucket's smallest `limit` live keys, but a newer-layer tombstone can delete one,
        // so we over-read by the count of newer in-range entries — a safe upper bound on
        // possible deletes, since keys are bucket-partitioned so only a same-bucket newer
        // entry can shadow a bucket's L1 key); (b) replay over L1 for identical
        // tombstone-suppression semantics.
        let mut newer: Vec<ScanEntry> = Vec::new();
        let bounded = limit <= BOUNDED_SCAN_LIMIT;

        // Reads one bucket's L0 files (oldest-first so newer L0 entries shadow older ones),
        // range-filtered. L0 has no min/max or sparse index, so files are scanned in full.
        let read_l0_bucket = |bucket: usize| -> Result<Vec<ScanEntry>> {
            let mut bucket_entries = Vec::new();
            for guard in &l0_guards[bucket] {
                let mut file = File::open(&guard.entry.path)?;
                let mut entry_bytes: Vec<u8> = Vec::new();
                loop {
                    let mut size_buf = [0u8; 4];
                    match file.read_exact(&mut size_buf) {
                        Ok(_) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                        Err(e) => return Err(LSMError::Io(e)),
                    }
                    let size = u32::from_le_bytes(size_buf) as usize;
                    entry_bytes.resize(size, 0);
                    file.read_exact(&mut entry_bytes[..size])?;
                    let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes[..size])?) };
                    if archived.key.as_slice() >= start && end.is_none_or(|e| archived.key.as_slice() < e) {
                        let val = if archived.tombstone {
                            None
                        } else {
                            Some((archived.value.to_native(), archived.seq.to_native()))
                        };
                        bucket_entries.push((archived.key.as_slice().to_vec(), val));
                    }
                }
            }
            Ok(bucket_entries)
        };

        // L0 (newer than L1). Bounded page → read busy buckets SERIALLY: L0 is small in the
        // steady state, so spawning a thread per bucket costs more than it saves (same
        // reasoning as the bounded L1 path below). Full/large scan → parallel read-everything.
        let busy_l0: Vec<usize> = (0..self.config.num_buckets).filter(|b| !l0_guards[*b].is_empty()).collect();
        if bounded {
            for bucket in busy_l0 {
                newer.extend(read_l0_bucket(bucket)?);
            }
        } else {
            let mut first_err: Option<LSMError> = None;
            std::thread::scope(|s| {
                let handles: Vec<_> = busy_l0.into_iter().map(|bucket| s.spawn(move || read_l0_bucket(bucket))).collect();
                for handle in handles {
                    match handle
                        .join()
                        .unwrap_or_else(|_| Err(LSMError::Io(std::io::Error::other("thread panicked"))))
                    {
                        Ok(entries) => newer.extend(entries),
                        Err(e) => {
                            if first_err.is_none() {
                                first_err = Some(e);
                            }
                        }
                    }
                }
            });
            if let Some(e) = first_err {
                return Err(e);
            }
        }
        for ro_memtable in ro_snapshot.iter() {
            for record in ro_memtable.records().iter() {
                if record.key.as_slice() >= start && end.is_none_or(|e| record.key.as_slice() < e) {
                    newer.push((record.key.clone(), if record.tombstone { None } else { Some((record.value, record.seq)) }));
                }
            }
        }
        newer.extend(active_snapshot);

        // L1 SSTables (oldest layer) — only buckets whose min/max overlaps `[start, end)`;
        // each seeks near `start` via the sparse index and stops at `end`. The L0 guards
        // captured above make the metadata read race-safe (see `l1_overlaps`). Keys are
        // hash-sharded so bucket key-sets are disjoint.
        //
        // Bounded page: read at most `limit + 1 + |newer in range in THIS bucket|` LIVE keys
        // per bucket and scan buckets SERIALLY — per-bucket work is tiny, so a thread per
        // bucket costs more than it saves. Full/large scan: keep the parallel, read-everything
        // path. Only a same-bucket newer entry can shadow a bucket's L1 key (keys are
        // hash-sharded), so the slack is sharded per bucket — a global `|newer|` slack would
        // over-read every bucket by the total newer count. See `get_bucket_for_key`.
        match bounded {
            true => {
                let mut newer_per_bucket = vec![0usize; self.config.num_buckets];
                for (key, _) in newer.iter() {
                    newer_per_bucket[get_bucket_for_key(key, self.config.num_buckets) as usize] += 1;
                }
                for (bucket, file) in l1_files.iter().enumerate() {
                    if !self.l1_overlaps(bucket, start, end) {
                        continue;
                    }
                    let cap = limit.saturating_add(1).saturating_add(newer_per_bucket[bucket]);
                    let mut entries = Vec::new();
                    self.scan_l1_range_into(bucket, file, start, end, Some(cap), &mut entries)?;
                    for (k, v) in entries {
                        all_entries.insert(k, v);
                    }
                }
            }
            false => {
                let mut first_err: Option<LSMError> = None;
                std::thread::scope(|s| {
                    let handles: Vec<_> = l1_files
                        .iter()
                        .enumerate()
                        .filter(|(bucket, _)| self.l1_overlaps(*bucket, start, end))
                        .map(|(bucket, file)| {
                            s.spawn(move || -> Result<Vec<ScanEntry>> {
                                let mut entries = Vec::new();
                                self.scan_l1_range_into(bucket, file, start, end, None, &mut entries)?;
                                Ok(entries)
                            })
                        })
                        .collect();
                    for handle in handles {
                        match handle
                            .join()
                            .unwrap_or_else(|_| Err(LSMError::Io(std::io::Error::other("thread panicked"))))
                        {
                            Ok(entries) => {
                                for (k, v) in entries {
                                    all_entries.insert(k, v);
                                }
                            }
                            Err(e) => {
                                if first_err.is_none() {
                                    first_err = Some(e);
                                }
                            }
                        }
                    }
                });
                if let Some(e) = first_err {
                    return Err(e);
                }
            }
        }

        // Replay the newer layers over L1 (oldest→newest) so tombstones and updates shadow
        // the L1 values exactly as before.
        for (key, value) in newer {
            all_entries.insert(key, value);
        }

        Ok(all_entries
            .into_iter()
            .filter_map(|(key, val)| val.map(|(v, s)| (key, v, s)))
            .take(limit)
            .collect())
    }

    /// Cleanup old read-only memtables that were deferred from previous compaction
    fn cleanup_pending_memtables(&self) {
        let mut pending = self.pending_old_memtables.write();

        let still_in_use: Vec<ReadOnlyMemTable> = pending.drain(..).filter(|ro| ro.reader_count() > 0).collect();

        pending.extend(still_in_use);
    }

    /// Defer cleanup of old read-only memtables to next compaction
    fn defer_old_memtables_cleanup(&self) -> Result<()> {
        let version = self.ro_memtable_version.load(Ordering::SeqCst);

        let mut ro_memtables = self.read_only_memtables.write();
        let mut pending = self.pending_old_memtables.write();

        let (old, new): (Vec<_>, Vec<_>) = ro_memtables.drain(..).partition(|ro| ro.version() < version);

        for ro in old {
            if !ro.is_flushed_to_level0() {
                // Not yet flushed to L0 — keep it in the active list
                ro_memtables.push(ro);
            } else if ro.reader_count() == 0 {
                // drop
            } else {
                pending.push(ro);
            }
        }

        ro_memtables.extend(new);
        Ok(())
    }

    /// Cleanup all pending memtables on database close
    /// Called during database shutdown to ensure cleanup happens
    pub(crate) fn cleanup_pending_memtables_on_close(&self) {
        let mut pending = self.pending_old_memtables.write();
        let count = pending.len();

        if count > 0 {
            info!("[LSM] Cleaning up {} pending read-only memtables on close", count);
            pending.clear();
        }
    }

    /// Drop all in-memory read-only memtables without compacting them to L1.
    ///
    /// Only used in tests to simulate the state where data has been flushed to
    /// L0 SSTable files but the corresponding in-memory ro_memtable has been
    /// evicted, so the only copy of the data is in the L0 files on disk.
    #[cfg(test)]
    pub(crate) fn purge_ro_memtables_for_test(&self) {
        self.read_only_memtables.write().clear();
    }
}

/// Test-only convenience wrappers that allocate a monotonically increasing
/// sequence and delegate to the production `*_with_seq` write path, so unit
/// tests exercise the real conflict-resolution logic. A single process-wide
/// counter gives every call a strictly higher sequence than any prior one.
#[cfg(test)]
impl LSMTree {
    fn next_test_seq() -> u32 {
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEQ: AtomicU32 = AtomicU32::new(1);
        SEQ.fetch_add(1, Ordering::Relaxed)
    }

    pub(crate) fn insert(&self, key: &[u8], value: u128) -> Result<()> {
        self.insert_with_seq(key, value, Self::next_test_seq())
    }

    pub(crate) fn delete(&self, key: &[u8]) -> Result<()> {
        self.delete_with_seq(key, Self::next_test_seq())
    }
}

/// In-memory (non-SSTable) statistics about an LSM tree: the live state that
/// the on-disk manifest does not capture.
#[derive(Clone, Debug, serde::Serialize)]
pub struct LSMStats {
    /// Live entries in the active memtable.
    pub memtable_entries: usize,
    /// Total entries across all sealed (read-only) memtables awaiting flush.
    pub read_only_entries: usize,
    /// Number of sealed read-only memtables awaiting flush.
    pub read_only_count: usize,
    /// True if any bucket is currently compacting.
    pub compaction_in_progress: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::TempDir;

    /// LSM config for tests: a small bucket count keeps the eager per-tree fd
    /// footprint low (one L1 file per bucket) so the suite survives high
    /// `cargo test` parallelism. See [`crate::support::TEST_NUM_BUCKETS`]. None
    /// of these tests depend on the specific bucket count, only on correctness.
    fn test_lsm_config() -> LSMConfig {
        LSMConfig {
            num_buckets: crate::support::TEST_NUM_BUCKETS,
            ..LSMConfig::default()
        }
    }

    #[test]
    fn test_sstable_entry_codec_roundtrip_and_corruption() {
        let entry = SStableEntry {
            key: b"some_key".to_vec(),
            key_prefix: key_prefix_of(b"some_key"),
            value: 0xdead_beef,
            tombstone: false,
            seq: 42,
        };

        let mut payload = encode_sstable_entry(&entry).expect("encode");

        // Round-trip: verify passes and the body decodes to the same entry.
        let body = verify_sstable_payload(&payload).expect("verify clean payload");
        let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(body) };
        assert_eq!(archived.key.as_slice(), b"some_key");
        assert_eq!(archived.value.to_native(), 0xdead_beef);
        assert_eq!(archived.seq.to_native(), 42);

        // Flip a byte in the rkyv body (past the 4-byte CRC prefix) → mismatch.
        let last = payload.len() - 1;
        payload[last] ^= 0xFF;
        assert!(matches!(verify_sstable_payload(&payload), Err(LSMError::Corruption(_))));

        // A truncated payload (shorter than the CRC word) is also rejected.
        assert!(matches!(verify_sstable_payload(&[0u8; 2]), Err(LSMError::Corruption(_))));
    }

    #[test]
    fn test_sparse_index_lookup_after_compaction() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        // Enough keys that some bucket's L1 file spans many SAMPLE_INTERVAL
        // blocks, so `block_start` returns non-zero offsets and the validated
        // scan-from-hint path is exercised (not just scan-from-0).
        let n: u128 = 3000;
        for i in 0..n {
            lsm.insert(format!("k:{i:08}").as_bytes(), i)?;
        }
        lsm.flush_and_compact_all()?; // push everything to L1; memtable now empty

        // Every present key must still be found via the sparse-index hint.
        for i in 0..n {
            assert_eq!(lsm.get(format!("k:{i:08}").as_bytes())?, Some(i), "missing key {i}");
        }

        // Misses: an in-range gap and keys below min / above max.
        assert_eq!(lsm.get(b"k:00000000x")?, None); // between existing keys
        assert_eq!(lsm.get(b"j:00000000")?, None); // below min ("j" < "k")
        assert_eq!(lsm.get(b"zzzzzzzz")?, None); // above max
        Ok(())
    }

    // The sparse index hint must actually be USED (non-zero, valid) — a buggy
    // index that always produced unusable offsets would still pass the lookup
    // tests (they'd silently fall back to a full scan), so assert the hints are
    // real frame boundaries with key <= target.
    #[test]
    fn test_sparse_index_hint_is_valid_and_engaged() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
        let n: u128 = 3000;
        for i in 0..n {
            lsm.insert(format!("k:{i:08}").as_bytes(), i)?;
        }
        lsm.flush_and_compact_all()?;

        let mut saw_nonzero_hint = false;
        for i in 0..n {
            let key = format!("k:{i:08}").into_bytes();
            let bucket = get_bucket_for_key(&key, lsm.config.num_buckets) as usize;
            let file = lsm.sstable_files[bucket].read().clone();
            let file_len = file.metadata()?.len();
            let hint = lsm.sstable_indexes[bucket].read().as_ref().map(|x| x.block_start(&key)).unwrap_or(0);
            if hint != 0 {
                saw_nonzero_hint = true;
                assert!(LSMTree::valid_scan_start(&file, hint, file_len, &key), "hint {hint} invalid for key {i}");
            }
        }
        assert!(saw_nonzero_hint, "expected some non-zero index hints across 3000 keys");
        Ok(())
    }

    // THE correctness guarantee: a bogus/stale start offset (as if a concurrent
    // compaction swapped the L1 file under the in-memory index) must never change
    // the result. The validated scan falls back to a full scan from 0.
    #[test]
    fn test_sparse_index_stale_hint_falls_back_correctly() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
        let n: u128 = 2000;
        for i in 0..n {
            lsm.insert(format!("k:{i:08}").as_bytes(), i + 1)?; // values are 1-based (non-zero)
        }
        lsm.flush_and_compact_all()?;

        let present = format!("k:{:08}", 1234).into_bytes();
        let bucket = get_bucket_for_key(&present, lsm.config.num_buckets) as usize;
        let file = lsm.sstable_files[bucket].read().clone();
        let file_len = file.metadata()?.len();

        // Ground truth from a full scan.
        let truth = lsm.lookup_in_sstable_file_from(&file, &present, 0)?;
        let truth_ptr = match truth {
            SsLookup::Found(p, _) => p,
            other => panic!("expected present key to be Found, got {other:?}"),
        };

        // Every bogus offset must yield the same Found pointer: past EOF (overflow
        // guard), mid-frame, near/at EOF, and a hint whose frame key > target.
        let big_key_hint = lsm.sstable_indexes[bucket]
            .read()
            .as_ref()
            .map(|x| x.block_start(b"k:99999999"))
            .unwrap_or(0);
        for bad in [u64::MAX, 1, 3, 7, file_len.saturating_sub(1), file_len, file_len / 2 + 1, big_key_hint] {
            match lsm.lookup_in_sstable_file_from(&file, &present, bad)? {
                SsLookup::Found(p, _) => assert_eq!(p, truth_ptr, "bogus hint {bad} changed the value"),
                other => panic!("bogus hint {bad} lost a present key: {other:?}"),
            }
        }

        // An in-range absent key must stay Missing regardless of the hint.
        let absent = format!("k:{:08}x", 1234).into_bytes();
        let abucket = get_bucket_for_key(&absent, lsm.config.num_buckets) as usize;
        let afile = lsm.sstable_files[abucket].read().clone();
        for bad in [0u64, u64::MAX, 5, afile.metadata()?.len()] {
            assert!(
                matches!(lsm.lookup_in_sstable_file_from(&afile, &absent, bad)?, SsLookup::Missing),
                "bogus hint {bad} fabricated a key"
            );
        }
        Ok(())
    }

    // The index built during the open-time scan (`load_metadata_from_file`) is a
    // distinct code path from the one built during compaction; exercise it.
    #[test]
    fn test_sparse_index_rebuilt_on_reopen() -> Result<()> {
        let temp_dir = TempDir::new()?;
        {
            let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
            for i in 0..2000u128 {
                lsm.insert(format!("k:{i:08}").as_bytes(), i + 1)?;
            }
            lsm.flush_and_compact_all()?;
        }
        // Reopen: indexes are rebuilt from the L1 files during the open scan.
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
        for i in 0..2000u128 {
            assert_eq!(lsm.get(format!("k:{i:08}").as_bytes())?, Some(i + 1), "key {i} after reopen");
        }
        assert_eq!(lsm.get(b"k:00000005x")?, None);
        Ok(())
    }

    // Deletes must resolve correctly through the sparse-index hint.
    #[test]
    fn test_sparse_index_lookup_with_deletes() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
        let n: u128 = 1000;
        for i in 0..n {
            lsm.insert(format!("k:{i:08}").as_bytes(), i + 1)?;
        }
        for i in (0..n).step_by(2) {
            lsm.delete(format!("k:{i:08}").as_bytes())?; // delete even keys
        }
        lsm.flush_and_compact_all()?;

        for i in 0..n {
            let got = lsm.get(format!("k:{i:08}").as_bytes())?;
            if i % 2 == 0 {
                assert_eq!(got, None, "deleted key {i} should be absent");
            } else {
                assert_eq!(got, Some(i + 1), "live key {i} has wrong value");
            }
        }
        Ok(())
    }

    #[test]
    fn test_lsm_basic_operations() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        // Insert a value
        lsm.insert(b"key1", 12345)?;

        // Retrieve the value
        assert_eq!(lsm.get(b"key1")?, Some(12345));

        // Insert another value
        lsm.insert(b"key2", 67890)?;
        assert_eq!(lsm.get(b"key2")?, Some(67890));

        // Update existing value
        lsm.insert(b"key1", 54321)?;
        assert_eq!(lsm.get(b"key1")?, Some(54321));

        Ok(())
    }

    // Regression test for the GC resurrection bug: a relocated value re-inserted
    // at the key's OLD sequence into the newest layer must NOT shadow a
    // higher-sequence tombstone that has flushed to an older layer. Before the
    // seq-aware read resolution (highest-seq-wins across all layers), `get`
    // early-returned at the first layer holding the key, so this inverted state
    // resolved to the value and resurrected a deleted key. This constructs the
    // inverted state directly; production prevents it at the GC re-point guard
    // (seq-aware `get_with_seq` under the bucket lock), so this guards the read
    // path as defense-in-depth.
    #[test]
    fn test_repoint_at_old_seq_does_not_resurrect_across_layers() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert_with_seq(b"k", 0xAAAA, 1)?; // value @ seq 1
        lsm.delete_with_seq(b"k", 5)?; // tombstone @ seq 5 (newer)
        assert_eq!(lsm.get(b"k")?, None, "deleted before flush");

        lsm.flush_memtable_to_level0()?; // tombstone @ seq 5 → L0
        assert_eq!(lsm.get(b"k")?, None, "deleted after flush");

        // GC re-point: re-insert the relocated value at the OLD seq into the
        // now-empty active memtable. The key is still logically deleted.
        lsm.insert_with_seq(b"k", 0xBBBB, 1)?;
        assert_eq!(lsm.get(b"k")?, None, "GC re-point at old seq resurrected a deleted key");
        Ok(())
    }

    // Regression: the read fast-path bound `max_lower_seq` must be initialized
    // from L0 files (not just L1) at open. L0 is newer than L1, so an L0-only
    // tombstone's seq would otherwise be missing from the bound after a restart,
    // letting a low-seq active entry wrongly short-circuit above it. This is the
    // restart variant of test_repoint_at_old_seq_does_not_resurrect_across_layers.
    #[test]
    fn test_max_lower_seq_folds_l0_at_open_so_fast_path_is_safe() -> Result<()> {
        let temp_dir = TempDir::new()?;
        {
            let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
            lsm.insert_with_seq(b"k", 0xAAAA, 1)?; // value @ seq 1
            lsm.delete_with_seq(b"k", 5)?; // tombstone @ seq 5 (newer)
            lsm.flush_memtable_to_level0()?; // tombstone @ seq 5 → an L0 file (no L1)
            assert_eq!(lsm.get(b"k")?, None, "deleted before restart");
        }

        // Reopen: max_lower_seq must be folded from the L0 file (seq 5), not left
        // at 0 (which only L1 would have contributed).
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
        assert_eq!(lsm.get(b"k")?, None, "still deleted after restart");

        // Re-point at the old seq into the fresh (empty) active memtable. With the
        // bound correctly at 5, the fast path must NOT short-circuit on seq 1 — it
        // falls through to the full scan that sees the higher-seq L0 tombstone.
        lsm.insert_with_seq(b"k", 0xBBBB, 1)?;
        assert_eq!(
            lsm.get(b"k")?,
            None,
            "stale max_lower_seq let the fast path resurrect a deleted key after restart"
        );
        Ok(())
    }

    #[test]
    fn test_lsm_delete() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"key1", 12345)?;
        assert_eq!(lsm.get(b"key1")?, Some(12345));

        // Delete the key
        lsm.delete(b"key1")?;

        // Key should not be found
        assert_eq!(lsm.get(b"key1")?, None);

        Ok(())
    }

    // Regression test: a delete tombstone in the active memtable must suppress
    // the corresponding entry in the SSTable.  The old keys() implementation
    // added SSTable keys last (without tombstone suppression), so after a flush
    // deleted keys re-appeared in iterators — causing the vec-index queue worker
    // to loop forever after restart (WAL recovery flushed queue entries to SSTable,
    // then cancel_pending_embed tombstones were ignored by keys()).
    #[test]
    fn test_keys_excludes_sstable_entry_tombstoned_in_memtable() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"keep", 1)?;
        lsm.insert(b"delete_me", 2)?;

        // Flush to SSTable — simulates WAL recovery flushing the queue entries.
        lsm.flush_memtable()?;
        lsm.compact_all()?;

        // Delete one key — tombstone lands in the fresh active memtable.
        lsm.delete(b"delete_me")?;

        // keys() must not return the deleted key even though it is still in the SSTable.
        let keys = lsm.keys()?;
        assert!(!keys.contains(&b"delete_me".to_vec()), "deleted SSTable key must not appear in keys()");
        assert!(keys.contains(&b"keep".to_vec()));
        assert_eq!(keys.len(), 1);

        Ok(())
    }

    #[test]
    fn test_lsm_stats() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"key1", 12345)?;
        lsm.insert(b"key2", 67890)?;

        let stats = lsm.stats();
        assert_eq!(stats.memtable_entries, 2);

        Ok(())
    }

    #[test]
    fn test_lsm_flush_and_readonly() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        // Insert data
        lsm.insert(b"key1", 100)?;
        lsm.insert(b"key2", 200)?;

        // Manually flush to create read-only memtable
        lsm.flush_memtable()?;

        // Check stats
        let stats = lsm.stats();
        assert_eq!(stats.memtable_entries, 0); // New memtable is empty
        assert_eq!(stats.read_only_count, 1); // One read-only snapshot

        // Should still be able to read from read-only layer
        assert_eq!(lsm.get(b"key1")?, Some(100));
        assert_eq!(lsm.get(b"key2")?, Some(200));

        Ok(())
    }

    #[test]
    fn test_lsm_compaction() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        // Insert and flush to create read-only memtable
        lsm.insert(b"apple", 1)?;
        lsm.insert(b"banana", 2)?;
        lsm.insert(b"cherry", 3)?;
        lsm.flush_memtable()?;

        // Compact all buckets
        lsm.compact_all()?;

        // After compaction, read-only memtables should be cleared
        let stats = lsm.stats();
        assert_eq!(stats.read_only_count, 0);

        // Data should still be accessible from SSTable
        assert_eq!(lsm.get(b"apple")?, Some(1));
        assert_eq!(lsm.get(b"banana")?, Some(2));
        assert_eq!(lsm.get(b"cherry")?, Some(3));

        Ok(())
    }

    /// Regression: a failed `merge_level0_to_level1` must not leave a bucket's
    /// `compaction_in_progress` flag stuck at `true` (which would make every
    /// future compaction of that bucket a silent no-op until restart). We force
    /// the merge to fail by deleting the on-disk L0 file while its in-memory L0
    /// entry still references it, then assert the flag is cleared and a retry
    /// actually re-enters the merge.
    #[test]
    fn test_compact_bucket_resets_flag_on_merge_error() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"key1", 12345)?;
        lsm.flush_memtable_to_level0()?; // writes L0 SSTable file(s) without compacting

        // Corrupt every L0 file: a size header claiming more bytes than actually
        // follow, so the merge's read_sstable_entries_from_path fails on
        // read_exact. (A *missing* file is treated as empty, not an error, so the
        // file must exist but be malformed.)
        let mut corrupted = 0;
        for bucket in 0..lsm.config.num_buckets as u32 {
            let bucket_dir = LSMTree::level0_bucket_dir_from(&lsm.base_path, bucket);
            let Ok(rd) = std::fs::read_dir(&bucket_dir) else { continue };
            for entry in rd {
                let path = entry?.path();
                if path.is_file() {
                    let mut bad = 100u32.to_le_bytes().to_vec(); // claims a 100-byte entry…
                    bad.extend_from_slice(&[0xABu8; 10]); // …but only 10 bytes follow
                    std::fs::write(&path, &bad)?;
                    corrupted += 1;
                }
            }
        }
        assert!(corrupted > 0, "expected at least one L0 file to corrupt");

        // Compact the bucket holding the orphaned entry — the merge must error.
        let errored_bucket = (0..lsm.config.num_buckets as u32)
            .find(|&b| lsm.compact_bucket(b).is_err())
            .expect("compacting the bucket with the missing L0 file must error");

        // The flag must be reset (not wedged at true) after the failed merge…
        assert!(
            !lsm.compaction_in_progress[errored_bucket as usize].load(Ordering::SeqCst),
            "compaction_in_progress must be cleared after a failed merge"
        );
        // …and a retry must actually re-enter the merge (error again), not
        // silently no-op via a stuck flag.
        assert!(
            lsm.compact_bucket(errored_bucket).is_err(),
            "retry must re-attempt the merge, not short-circuit on a stuck flag"
        );

        Ok(())
    }

    #[test]
    fn test_lsm_two_way_merge() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        // First batch: insert initial data
        lsm.insert(b"key1", 100)?;
        lsm.insert(b"key2", 200)?;
        lsm.insert(b"key3", 300)?;
        lsm.flush_memtable()?;

        // Compact to SSTable
        lsm.compact_all()?;

        // Second batch: update and add new keys
        lsm.insert(b"key1", 111)?; // Update
        lsm.insert(b"key4", 400)?; // New
        lsm.flush_memtable()?;

        // Compact again - should merge
        lsm.compact_all()?;

        // Verify merged results
        assert_eq!(lsm.get(b"key1")?, Some(111)); // Updated value
        assert_eq!(lsm.get(b"key2")?, Some(200)); // Original value
        assert_eq!(lsm.get(b"key3")?, Some(300)); // Original value
        assert_eq!(lsm.get(b"key4")?, Some(400)); // New value

        Ok(())
    }

    #[test]
    fn test_lsm_delete_with_compaction() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        // Insert data
        lsm.insert(b"key1", 100)?;
        lsm.insert(b"key2", 200)?;
        lsm.insert(b"key3", 300)?;
        lsm.flush_memtable()?;

        // Compact to SSTable
        lsm.compact_all()?;

        // Delete a key
        lsm.delete(b"key2")?;
        lsm.flush_memtable()?;

        // Compact again - tombstone should remove the entry
        lsm.compact_all()?;

        // Verify deletion
        assert_eq!(lsm.get(b"key1")?, Some(100));
        assert_eq!(lsm.get(b"key2")?, None); // Deleted
        assert_eq!(lsm.get(b"key3")?, Some(300));

        Ok(())
    }

    #[test]
    fn test_lsm_persistence() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let path = temp_dir.path().to_path_buf();

        // First session: insert and compact
        {
            let lsm = LSMTree::open(&path, test_lsm_config())?;
            lsm.insert(b"persistent_key1", 1000)?;
            lsm.insert(b"persistent_key2", 2000)?;
            lsm.flush_memtable()?;

            lsm.compact_all()?;
        }

        // Second session: reopen and verify
        {
            let lsm = LSMTree::open(&path, test_lsm_config())?;
            assert_eq!(lsm.get(b"persistent_key1")?, Some(1000));
            assert_eq!(lsm.get(b"persistent_key2")?, Some(2000));
        }

        Ok(())
    }

    #[test]
    fn test_lsm_prefix_scan() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        // Insert keys with various prefixes
        lsm.insert(b"user:1:name", 100)?;
        lsm.insert(b"user:1:email", 101)?;
        lsm.insert(b"user:2:name", 200)?;
        lsm.insert(b"user:2:email", 201)?;
        lsm.insert(b"product:1:name", 300)?;
        lsm.insert(b"product:1:price", 301)?;

        // Test prefix scan for "user:1"
        let results = lsm.scan_prefix(b"user:1")?;
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|(k, v, _)| k == b"user:1:name" && *v == 100));
        assert!(results.iter().any(|(k, v, _)| k == b"user:1:email" && *v == 101));

        // Test prefix scan for "user:"
        let results = lsm.scan_prefix(b"user:")?;
        assert_eq!(results.len(), 4);

        // Test prefix scan for "product:"
        let results = lsm.scan_prefix(b"product:")?;
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|(k, v, _)| k == b"product:1:name" && *v == 300));
        assert!(results.iter().any(|(k, v, _)| k == b"product:1:price" && *v == 301));

        // Test prefix scan with no matches
        let results = lsm.scan_prefix(b"order:")?;
        assert_eq!(results.len(), 0);

        Ok(())
    }

    #[test]
    fn test_lsm_prefix_scan_with_flush() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        // Insert and flush some data
        lsm.insert(b"app:config:version", 100)?;
        lsm.insert(b"app:config:debug", 101)?;
        lsm.flush_memtable()?;

        // Insert more data in new memtable
        lsm.insert(b"app:config:logging", 102)?;
        lsm.insert(b"app:state:active", 200)?;

        // Test prefix scan should find data from both memtable and read-only layer
        let results = lsm.scan_prefix(b"app:config:")?;
        assert_eq!(results.len(), 3);
        assert!(results.iter().any(|(k, v, _)| k == b"app:config:version" && *v == 100));
        assert!(results.iter().any(|(k, v, _)| k == b"app:config:debug" && *v == 101));
        assert!(results.iter().any(|(k, v, _)| k == b"app:config:logging" && *v == 102));

        Ok(())
    }

    #[test]
    fn test_lsm_prefix_scan_with_compaction() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        // Insert data and compact to SSTable
        lsm.insert(b"session:123:user", 1000)?;
        lsm.insert(b"session:123:timestamp", 1001)?;
        lsm.insert(b"session:456:user", 2000)?;
        lsm.flush_memtable()?;
        lsm.compact_all()?;

        // Insert more data in new memtable
        lsm.insert(b"session:123:active", 1002)?;

        // Test prefix scan should find data from both SSTable and memtable
        let results = lsm.scan_prefix(b"session:123:")?;
        assert_eq!(results.len(), 3);
        assert!(results.iter().any(|(k, v, _)| k == b"session:123:user" && *v == 1000));
        assert!(results.iter().any(|(k, v, _)| k == b"session:123:timestamp" && *v == 1001));
        assert!(results.iter().any(|(k, v, _)| k == b"session:123:active" && *v == 1002));

        Ok(())
    }

    #[test]
    fn test_lsm_compaction_stress() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = Arc::new(LSMTree::open(temp_dir.path(), test_lsm_config())?);

        let writer_threads = 4usize;
        let writes_per_thread = 200usize;
        let stop_compaction = Arc::new(AtomicBool::new(false));

        let compaction_lsm = lsm.clone();
        let compaction_stop = stop_compaction.clone();
        let compaction_handle = std::thread::spawn(move || -> Result<()> {
            while !compaction_stop.load(Ordering::SeqCst) {
                if let Err(err) = compaction_lsm.compact_all() {
                    if let LSMError::Io(io_err) = &err
                        && io_err.kind() == std::io::ErrorKind::NotFound
                    {
                        std::thread::yield_now();
                        continue;
                    }
                    return Err(err);
                }
                std::thread::yield_now();
            }
            Ok(())
        });

        let mut handles = Vec::with_capacity(writer_threads);
        for thread_id in 0..writer_threads {
            let writer_lsm = lsm.clone();
            let handle = std::thread::spawn(move || -> Result<()> {
                for i in 0..writes_per_thread {
                    let key = format!("stress:key:{}:{}", thread_id, i).into_bytes();
                    let value = (((thread_id as u128) << 32) | i as u128).saturating_add(1);
                    let mut attempts = 0u8;
                    loop {
                        match writer_lsm.insert(&key, value) {
                            Ok(()) => break,
                            Err(LSMError::CapacityExceeded) => {
                                attempts += 1;
                                writer_lsm.flush_memtable()?;
                                std::thread::sleep(Duration::from_millis(1));
                                if attempts >= 10 {
                                    return Err(LSMError::CapacityExceeded);
                                }
                            }
                            Err(err) => return Err(err),
                        }
                    }
                    if i % 25 == 0 {
                        writer_lsm.delete(&key)?;
                    }
                }
                Ok(())
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap()?;
        }

        stop_compaction.store(true, Ordering::SeqCst);
        compaction_handle.join().unwrap()?;

        lsm.flush_and_compact_all()?;

        for thread_id in 0..writer_threads {
            let key = format!("stress:key:{}:{}", thread_id, writes_per_thread - 1).into_bytes();
            let value = lsm.get(&key)?;
            assert!(value.is_some());
        }

        Ok(())
    }

    // ── get_multiple ──────────────────────────────────────────────────────────────

    #[test]
    fn test_get_multiple_matches_individual_get() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        for i in 0u128..20 {
            lsm.insert(format!("k{i}").as_bytes(), i * 10)?;
        }

        let keys: Vec<Vec<u8>> = (0u128..20).map(|i| format!("k{i}").into_bytes()).collect();
        let batch: Vec<Option<u128>> = lsm.get_multiple(&keys)?.into_iter().map(|o| o.map(|(p, _)| p)).collect();

        assert_eq!(batch.len(), keys.len());
        for (i, result) in batch.iter().enumerate() {
            assert_eq!(*result, lsm.get(&keys[i])?, "mismatch at index {i}");
        }
        Ok(())
    }

    #[test]
    fn test_get_multiple_missing_keys_return_none() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"present", 42)?;

        let keys = vec![b"present".to_vec(), b"absent".to_vec()];
        let batch: Vec<Option<u128>> = lsm.get_multiple(&keys)?.into_iter().map(|o| o.map(|(p, _)| p)).collect();

        assert_eq!(batch[0], Some(42));
        assert_eq!(batch[1], None);
        Ok(())
    }

    #[test]
    fn test_get_multiple_after_flush_to_sstable() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        // Write to memtable, flush to SSTable, write more to a fresh memtable
        for i in 0u128..10 {
            lsm.insert(format!("flushed:{i}").as_bytes(), i)?;
        }
        lsm.flush_memtable()?;
        for i in 10u128..15 {
            lsm.insert(format!("flushed:{i}").as_bytes(), i)?;
        }

        let keys: Vec<Vec<u8>> = (0u128..15).map(|i| format!("flushed:{i}").into_bytes()).collect();
        let batch: Vec<Option<u128>> = lsm.get_multiple(&keys)?.into_iter().map(|o| o.map(|(p, _)| p)).collect();

        for (i, result) in batch.iter().enumerate() {
            assert_eq!(*result, Some(i as u128), "missing key at index {i}");
        }
        Ok(())
    }

    #[test]
    fn test_get_multiple_after_compaction() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        for i in 0u128..20 {
            lsm.insert(format!("compact:{i}").as_bytes(), i + 100)?;
        }
        lsm.flush_memtable()?;
        lsm.compact_all()?;

        let keys: Vec<Vec<u8>> = (0u128..20).map(|i| format!("compact:{i}").into_bytes()).collect();
        let batch: Vec<Option<u128>> = lsm.get_multiple(&keys)?.into_iter().map(|o| o.map(|(p, _)| p)).collect();

        for (i, result) in batch.iter().enumerate() {
            assert_eq!(*result, Some(i as u128 + 100), "missing key at index {i}");
        }
        Ok(())
    }

    #[test]
    fn test_get_multiple_l1_prefilter_skips_absent_keys() -> Result<()> {
        // Single bucket so every key shares one L1 file with a known [min, max]; the
        // batch mixes present keys, keys below/above the L1 range (min/max reject), and
        // an in-range key never inserted (bloom reject / L1 miss). The L1 pre-filter
        // must never drop a live key and must return None for every absent one.
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(
            temp_dir.path(),
            LSMConfig {
                num_buckets: 1,
                ..test_lsm_config()
            },
        )?;

        for i in 30u128..50 {
            lsm.insert(format!("m{i}").as_bytes(), i)?;
        }
        lsm.flush_memtable()?;
        lsm.compact_all()?; // land everything in L1 (min "m30", max "m49") with a bloom + min/max

        let keys = vec![
            b"m30".to_vec(),  // present, == min
            b"m49".to_vec(),  // present, == max
            b"m40".to_vec(),  // present, interior
            b"a00".to_vec(),  // absent, below min       -> min/max reject
            b"z99".to_vec(),  // absent, above max       -> min/max reject
            b"m300".to_vec(), // absent, inside [min,max] -> bloom reject / L1 miss
        ];
        let batch: Vec<Option<u128>> = lsm.get_multiple(&keys)?.into_iter().map(|o| o.map(|(p, _)| p)).collect();
        assert_eq!(batch[0], Some(30));
        assert_eq!(batch[1], Some(49));
        assert_eq!(batch[2], Some(40));
        assert_eq!(batch[3], None);
        assert_eq!(batch[4], None);
        assert_eq!(batch[5], None);

        // A batch of only out-of-range keys exercises the full skip (no L1 read): all None.
        let absent_only = vec![b"a00".to_vec(), b"b11".to_vec(), b"z99".to_vec()];
        let batch2: Vec<Option<u128>> = lsm.get_multiple(&absent_only)?.into_iter().map(|o| o.map(|(p, _)| p)).collect();
        assert!(batch2.iter().all(|r| r.is_none()));

        // The pre-filtered batch path must agree with the single-key point-lookup path.
        for k in &keys {
            let single = lsm.get(k)?;
            let multi = lsm.get_multiple(std::slice::from_ref(k))?[0].map(|(p, _)| p);
            assert_eq!(single, multi, "get vs get_multiple mismatch for {:?}", String::from_utf8_lossy(k));
        }
        Ok(())
    }

    #[test]
    fn test_get_multiple_deleted_keys_return_none() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"alive", 1)?;
        lsm.insert(b"dead", 2)?;
        lsm.delete(b"dead")?;

        let keys = vec![b"alive".to_vec(), b"dead".to_vec()];
        let batch: Vec<Option<u128>> = lsm.get_multiple(&keys)?.into_iter().map(|o| o.map(|(p, _)| p)).collect();

        assert_eq!(batch[0], Some(1));
        assert_eq!(batch[1], None);
        Ok(())
    }

    #[test]
    fn test_get_multiple_deleted_after_flush() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"kept", 10)?;
        lsm.insert(b"gone", 20)?;
        lsm.flush_memtable()?;
        lsm.delete(b"gone")?;

        let keys = vec![b"kept".to_vec(), b"gone".to_vec()];
        let batch: Vec<Option<u128>> = lsm.get_multiple(&keys)?.into_iter().map(|o| o.map(|(p, _)| p)).collect();

        assert_eq!(batch[0], Some(10));
        assert_eq!(batch[1], None);
        Ok(())
    }

    // ── scan_prefix tombstone regression ──────────────────────────────────────

    #[test]
    fn test_scan_prefix_deleted_after_flush_returns_none() -> Result<()> {
        // Regression: before the fix, scan_prefix used the active memtable's live
        // iterator (which skips tombstones), so a key deleted after a flush still
        // appeared in prefix-scan results because the SSTable entry was not shadowed.
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"doc:1", 10)?;
        lsm.insert(b"doc:2", 20)?;
        lsm.insert(b"other:1", 99)?;
        lsm.flush_memtable()?; // both doc: keys now in SSTable

        lsm.delete(b"doc:2")?; // tombstone in active memtable

        let results = lsm.scan_prefix(b"doc:")?;

        assert_eq!(results.len(), 1, "deleted key must not appear");
        assert_eq!(results[0].0, b"doc:1");
        assert_eq!(results[0].1, 10);
        Ok(())
    }

    #[test]
    fn test_scan_prefix_deleted_key_then_reinserted() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"k:1", 1)?;
        lsm.flush_memtable()?;
        lsm.delete(b"k:1")?;
        lsm.flush_memtable()?; // tombstone now in read-only memtable
        lsm.insert(b"k:1", 2)?; // re-insert in fresh active memtable

        let results = lsm.scan_prefix(b"k:")?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 2, "re-inserted value must win");
        Ok(())
    }

    // ── scan_prefix L0-vs-L1 regressions ──────────────────────────────────────
    // scan_prefix_in_bucket used to scan L0 then L1 into one flat live-only Vec,
    // dropping tombstones and letting the older L1 entry overwrite the newer L0
    // one in the caller's merge. These cover the L0/L1 interactions that the
    // memtable/read-only-only tests above never exercised.

    #[test]
    fn test_scan_prefix_sees_l0_only_data() -> Result<()> {
        // Data lives only in L0 (memtable + read-only memtables empty).
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"p:1", 100)?;
        lsm.insert(b"p:2", 200)?;
        lsm.flush_memtable_to_level0()?; // data only in L0
        lsm.purge_ro_memtables_for_test();

        let map: std::collections::HashMap<Vec<u8>, u128> = lsm.scan_prefix(b"p:")?.into_iter().map(|(k, v, _)| (k, v)).collect();
        assert_eq!(map.get(b"p:1".as_slice()), Some(&100), "L0-only key p:1 must appear");
        assert_eq!(map.get(b"p:2".as_slice()), Some(&200), "L0-only key p:2 must appear");
        Ok(())
    }

    #[test]
    fn test_scan_prefix_l0_value_shadows_stale_l1() -> Result<()> {
        // Updated value: new in L0, old in L1 — the newer L0 value must win.
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"p:a", 111)?;
        lsm.flush_and_compact_all()?; // p:a=111 -> L1
        lsm.insert(b"p:a", 222)?;
        lsm.flush_memtable_to_level0()?; // p:a=222 -> L0
        lsm.purge_ro_memtables_for_test(); // only L0 value + stale L1 value remain

        let map: std::collections::HashMap<Vec<u8>, u128> = lsm.scan_prefix(b"p:")?.into_iter().map(|(k, v, _)| (k, v)).collect();
        assert_eq!(map.get(b"p:a".as_slice()), Some(&222), "newer L0 value must shadow stale L1 value");
        Ok(())
    }

    #[test]
    fn test_scan_prefix_l0_tombstone_suppresses_l1() -> Result<()> {
        // Deleted key: tombstone in L0, live value in L1 — the tombstone must win.
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"p:1", 100)?;
        lsm.flush_and_compact_all()?; // p:1 -> L1
        lsm.insert(b"p:2", 200)?;
        lsm.delete(b"p:1")?; // tombstone in active memtable
        lsm.flush_memtable_to_level0()?; // tombstone + p:2 -> L0
        lsm.purge_ro_memtables_for_test(); // only L0 tombstone + L1 value remain

        let map: std::collections::HashMap<Vec<u8>, u128> = lsm.scan_prefix(b"p:")?.into_iter().map(|(k, v, _)| (k, v)).collect();
        assert_eq!(map.get(b"p:1".as_slice()), None, "L0 tombstone must suppress live L1 value");
        assert_eq!(map.get(b"p:2".as_slice()), Some(&200), "live key p:2 must appear");
        Ok(())
    }

    #[test]
    fn test_scan_prefix_l0_reinsert_after_l1_delete() -> Result<()> {
        // Re-insert into L0 after the key was deleted (tombstone) in L1.
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"p:1", 100)?;
        lsm.flush_and_compact_all()?;
        lsm.delete(b"p:1")?;
        lsm.compact_all()?; // tombstone in L1
        lsm.insert(b"p:1", 999)?; // re-insert into memtable
        lsm.flush_memtable_to_level0()?; // new value to L0
        lsm.purge_ro_memtables_for_test();

        let map: std::collections::HashMap<Vec<u8>, u128> = lsm.scan_prefix(b"p:")?.into_iter().map(|(k, v, _)| (k, v)).collect();
        assert_eq!(map.get(b"p:1".as_slice()), Some(&999), "re-inserted L0 value must win over L1 tombstone");
        Ok(())
    }

    // ── min/max skip + sparse-index seek over real L1 SSTables ──────────────

    #[test]
    fn prefix_upper_bound_cases() {
        assert_eq!(prefix_upper_bound(b"abc"), Some(b"abd".to_vec()));
        assert_eq!(prefix_upper_bound(b"a"), Some(b"b".to_vec()));
        // A trailing 0xFF is dropped and the previous byte incremented.
        assert_eq!(prefix_upper_bound(b"ab\xff"), Some(b"ac".to_vec()));
        assert_eq!(prefix_upper_bound(b"ab\xff\xff"), Some(b"ac".to_vec()));
        // No finite upper bound.
        assert_eq!(prefix_upper_bound(b"\xff\xff"), None);
        assert_eq!(prefix_upper_bound(b""), None);
    }

    /// Sort scan output to `(key, pointer)` for order-insensitive comparison.
    fn kv_sorted(v: Vec<(Vec<u8>, u128, u32)>) -> Vec<(Vec<u8>, u128)> {
        let mut r: Vec<(Vec<u8>, u128)> = v.into_iter().map(|(k, val, _)| (k, val)).collect();
        r.sort();
        r
    }

    #[test]
    fn scan_over_l1_matches_reference_with_seek_and_skip() -> Result<()> {
        // 1000 keys, all compacted into L1 (well past SAMPLE_INTERVAL per bucket, so
        // the sparse index has multiple samples and block_start is exercised).
        let temp = TempDir::new()?;
        let lsm = LSMTree::open(temp.path(), test_lsm_config())?;
        let n = 1000u64;
        for seq in 0..n {
            lsm.insert(format!("key:{seq:06}").as_bytes(), seq as u128)?;
        }
        lsm.flush_and_compact_all()?;
        lsm.purge_ro_memtables_for_test(); // data lives only in L1 now

        // Full prefix → every key, values intact.
        let all = kv_sorted(lsm.scan_prefix(b"key:")?);
        assert_eq!(all.len(), n as usize);
        for (seq, (k, v)) in all.iter().enumerate() {
            assert_eq!((k.clone(), *v), (format!("key:{seq:06}").into_bytes(), seq as u128));
        }

        // Middle range → sparse-index seek near start + early stop at end.
        let mid = kv_sorted(lsm.range_pointers_bounded(b"key:000100", Some(b"key:000200"), usize::MAX)?);
        assert_eq!(mid.len(), 100);
        assert_eq!(mid.first().unwrap().1, 100);
        assert_eq!(mid.last().unwrap().1, 199);

        // Sub-prefix selecting a middle slice (key:000100..000199).
        assert_eq!(kv_sorted(lsm.scan_prefix(b"key:0001")?).len(), 100);

        // Ranges/prefixes that fall entirely outside [min,max] → min/max skip → empty.
        assert!(lsm.range_pointers_bounded(b"aaa", Some(b"aab"), usize::MAX)?.is_empty());
        assert!(lsm.range_pointers_bounded(b"zzz", None, usize::MAX)?.is_empty());
        assert!(lsm.scan_prefix(b"nomatch:")?.is_empty());

        // A bounded page still honours `limit` after the seek.
        assert_eq!(lsm.range_pointers_bounded(b"key:", Some(b"key;"), 10)?.len(), 10);
        Ok(())
    }

    #[test]
    fn scan_over_l1_suppresses_tombstone() -> Result<()> {
        let temp = TempDir::new()?;
        let lsm = LSMTree::open(temp.path(), test_lsm_config())?;
        for seq in 0..200u64 {
            lsm.insert(format!("k:{seq:04}").as_bytes(), seq as u128)?;
        }
        lsm.flush_and_compact_all()?;
        lsm.delete(b"k:0100")?;
        lsm.flush_and_compact_all()?; // tombstone merged into L1
        lsm.purge_ro_memtables_for_test();

        let keys: Vec<Vec<u8>> = lsm
            .range_pointers_bounded(b"k:0050", Some(b"k:0150"), usize::MAX)?
            .into_iter()
            .map(|(k, _, _)| k)
            .collect();
        assert!(!keys.contains(&b"k:0100".to_vec()), "deleted key must be excluded after min/max seek");
        assert_eq!(keys.len(), 99, "50..149 minus the one deleted key");
        Ok(())
    }

    #[test]
    fn scan_across_l0_and_l1_after_partial_flush() -> Result<()> {
        // Some keys in L1, a newer overlapping batch in L0 (no min/max there): the
        // scan must still merge both. Exercises the "L1 skipped/seeked, L0 always
        // scanned" split.
        let temp = TempDir::new()?;
        let lsm = LSMTree::open(temp.path(), test_lsm_config())?;
        for seq in 0..100u64 {
            lsm.insert(format!("m:{seq:04}").as_bytes(), seq as u128)?;
        }
        lsm.flush_and_compact_all()?; // → L1
        // Overwrite a slice with new values, land them in L0 only.
        for seq in 40..60u64 {
            lsm.insert(format!("m:{seq:04}").as_bytes(), 1000 + seq as u128)?;
        }
        lsm.flush_memtable_to_level0()?; // → L0
        lsm.purge_ro_memtables_for_test();

        let map: std::collections::HashMap<Vec<u8>, u128> = lsm
            .range_pointers_bounded(b"m:", Some(b"m;"), usize::MAX)?
            .into_iter()
            .map(|(k, v, _)| (k, v))
            .collect();
        assert_eq!(map.len(), 100);
        assert_eq!(map[b"m:0005".as_slice()], 5, "untouched L1 value");
        assert_eq!(map[b"m:0050".as_slice()], 1050, "L0 overwrite must win over L1");
        Ok(())
    }

    // ── scan_prefixes ──────────────────────────────────────────────

    fn u32_prefixed_key(prefix: u32, suffix: u32) -> Vec<u8> {
        let mut k = prefix.to_be_bytes().to_vec();
        k.extend_from_slice(&suffix.to_be_bytes());
        k
    }

    #[test]
    fn scan_prefixes_over_l1_seeks_per_prefix() -> Result<()> {
        // Many cluster prefixes flushed to L1; probing a few must return exactly
        // those clusters' keys (per-prefix sparse seek + bucket min/max skip).
        let temp = TempDir::new()?;
        let lsm = LSMTree::open(temp.path(), test_lsm_config())?;
        let n_clusters = 200u32;
        let per = 30u32;
        for cid in 0..n_clusters {
            for did in 0..per {
                lsm.insert(&u32_prefixed_key(cid, did), (cid as u128) * 1000 + did as u128)?;
            }
        }
        lsm.flush_and_compact_all()?;
        lsm.purge_ro_memtables_for_test();

        // Probe three scattered clusters out of 200.
        let ids: std::collections::HashSet<u32> = [3u32, 100, 199].into_iter().collect();
        let results = lsm.scan_prefixes(&ids)?;
        assert_eq!(results.len(), (ids.len() as u32 * per) as usize);
        for (k, v) in &results {
            let cid = u32::from_be_bytes(k[..4].try_into().unwrap());
            let did = u32::from_be_bytes(k[4..8].try_into().unwrap());
            assert!(ids.contains(&cid), "returned a non-probed cluster {cid}");
            assert_eq!(*v, (cid as u128) * 1000 + did as u128);
        }
        // A cluster id past every key's range → min/max skip → empty.
        let none: std::collections::HashSet<u32> = [u32::MAX].into_iter().collect();
        assert!(lsm.scan_prefixes(&none)?.is_empty());
        Ok(())
    }

    #[test]
    fn range_keys_bounded_over_l1_seeks_and_skips() -> Result<()> {
        let temp = TempDir::new()?;
        let lsm = LSMTree::open(temp.path(), test_lsm_config())?;
        for seq in 0..1000u64 {
            lsm.insert(format!("r:{seq:06}").as_bytes(), seq as u128)?;
        }
        lsm.flush_and_compact_all()?;
        lsm.purge_ro_memtables_for_test();

        // Unbounded from a mid start, limited page — seeks to start, min/max keeps
        // only buckets reaching it.
        let keys = lsm.range_keys_bounded(b"r:000900", 50)?;
        assert_eq!(keys.len(), 50);
        assert_eq!(keys[0], b"r:000900".to_vec());
        assert!(keys.iter().all(|k| k.as_slice() >= b"r:000900".as_slice()));

        // Start past the max key → every bucket skipped → empty.
        assert!(lsm.range_keys_bounded(b"zzz", usize::MAX)?.is_empty());
        // Full range from the front returns everything.
        assert_eq!(lsm.range_keys_bounded(b"r:", usize::MAX)?.len(), 1000);
        Ok(())
    }

    #[test]
    fn test_scan_prefixes_basic() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        // cluster 1: docs 10, 11, 12; cluster 2: doc 20; cluster 3: doc 30
        let entries = [(1u32, 10u32, 100u128), (1, 11, 101), (1, 12, 102), (2, 20, 200), (3, 30, 300)];
        for (cid, did, val) in entries {
            lsm.insert(&u32_prefixed_key(cid, did), val)?;
        }

        let ids: std::collections::HashSet<u32> = [1, 2].iter().copied().collect();
        let results = lsm.scan_prefixes(&ids)?;

        // Should return all keys for clusters 1 and 2, not cluster 3
        assert_eq!(results.len(), 4);
        let other_key = u32_prefixed_key(3, 30);
        assert!(!results.iter().any(|(k, _)| k == &other_key));
        assert!(results.iter().any(|(k, v)| k == &u32_prefixed_key(1, 10) && *v == 100));
        assert!(results.iter().any(|(k, v)| k == &u32_prefixed_key(2, 20) && *v == 200));
        Ok(())
    }

    #[test]
    fn test_scan_prefixes_after_flush_and_compaction() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        // Write to memtable and flush
        lsm.insert(&u32_prefixed_key(5, 1), 501)?;
        lsm.insert(&u32_prefixed_key(5, 2), 502)?;
        lsm.flush_memtable()?;
        lsm.compact_all()?;

        // Add more in fresh memtable
        lsm.insert(&u32_prefixed_key(5, 3), 503)?;
        lsm.insert(&u32_prefixed_key(6, 1), 601)?;

        let ids: std::collections::HashSet<u32> = [5].iter().copied().collect();
        let results = lsm.scan_prefixes(&ids)?;

        assert_eq!(results.len(), 3);
        assert!(results.iter().any(|(k, v)| k == &u32_prefixed_key(5, 1) && *v == 501));
        assert!(results.iter().any(|(k, v)| k == &u32_prefixed_key(5, 2) && *v == 502));
        assert!(results.iter().any(|(k, v)| k == &u32_prefixed_key(5, 3) && *v == 503));
        Ok(())
    }

    #[test]
    fn test_scan_prefixes_empty_result() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(&u32_prefixed_key(1, 1), 10)?;

        let ids: std::collections::HashSet<u32> = [99].iter().copied().collect();
        let results = lsm.scan_prefixes(&ids)?;
        assert!(results.is_empty());
        Ok(())
    }

    // ── Scan latency benchmark (memtable / L1 / L0+L1) ──────────────────────
    //
    // A kept, deterministic micro-benchmark for the scan paths across LSM layer
    // states. `#[ignore]`d so it never runs in normal `cargo test`; run it on demand:
    //
    //   cargo test -p minnal_db --lib scan_latency_benchmark -- --ignored --nocapture
    //
    // It lives here (not in `benches/`) on purpose: building an exact L0+L1 state is
    // only possible through the crate-internal `LSMTree` flush methods — the public
    // `Db` facade only flushes to SSTables via the background worker, so a criterion
    // bench cannot construct these states deterministically. The full scans show
    // throughput per layer; the *selective* scans show the L1 min/max skip +
    // sparse-index seek (a bounded slice out of a large L1 should not read the whole
    // file), which the memtable-resident timings never exercised.

    /// Insert `k:{seq:06}` for `seq` in `0..total`, all left in the active memtable.
    fn bench_state_memtable(total: u64) -> (TempDir, LSMTree) {
        let temp = TempDir::new().unwrap();
        let lsm = LSMTree::open(temp.path(), test_lsm_config()).unwrap();
        for seq in 0..total {
            lsm.insert(format!("k:{seq:06}").as_bytes(), seq as u128).unwrap();
        }
        (temp, lsm)
    }

    /// Same keys, all compacted into L1 (memtable/RO drained).
    fn bench_state_l1(total: u64) -> (TempDir, LSMTree) {
        let temp = TempDir::new().unwrap();
        let lsm = LSMTree::open(temp.path(), test_lsm_config()).unwrap();
        for seq in 0..total {
            lsm.insert(format!("k:{seq:06}").as_bytes(), seq as u128).unwrap();
        }
        lsm.flush_and_compact_all().unwrap();
        lsm.purge_ro_memtables_for_test();
        (temp, lsm)
    }

    /// Same keys spread across all three layers: ~80% in L1, ~15% in an uncompacted
    /// L0 file, ~5% left in the memtable — the realistic "recently written over a
    /// compacted base" shape a scan must merge.
    fn bench_state_l0_l1(total: u64) -> (TempDir, LSMTree) {
        let temp = TempDir::new().unwrap();
        let lsm = LSMTree::open(temp.path(), test_lsm_config()).unwrap();
        let l1_cut = total * 80 / 100;
        let l0_cut = total * 95 / 100;
        for seq in 0..l1_cut {
            lsm.insert(format!("k:{seq:06}").as_bytes(), seq as u128).unwrap();
        }
        lsm.flush_and_compact_all().unwrap(); // → L1
        for seq in l1_cut..l0_cut {
            lsm.insert(format!("k:{seq:06}").as_bytes(), seq as u128).unwrap();
        }
        lsm.flush_memtable_to_level0().unwrap(); // → L0 (uncompacted)
        lsm.purge_ro_memtables_for_test();
        for seq in l0_cut..total {
            lsm.insert(format!("k:{seq:06}").as_bytes(), seq as u128).unwrap(); // → memtable
        }
        (temp, lsm)
    }

    #[test]
    #[ignore]
    fn scan_latency_benchmark() {
        use std::time::Instant;
        let total = 5_000u64;

        // Warm up, run `iters`, report median-ish average µs/call. Asserts the result
        // count so a broken scan can't silently post a fast time.
        fn time_it(label: &str, iters: u32, mut f: impl FnMut() -> usize, expect: usize) {
            for _ in 0..5 {
                assert_eq!(f(), expect);
            }
            let t = Instant::now();
            for _ in 0..iters {
                assert_eq!(f(), expect);
            }
            let per = t.elapsed().as_nanos() as f64 / iters as f64 / 1e3;
            eprintln!("    {label:34} {expect:>5} results  {per:>9.2} µs/call");
        }

        // A 50-key slice in the middle of the keyspace, for the selective scans.
        let mid = total / 2;
        let sel_start = format!("k:{mid:06}").into_bytes();
        let sel_end = format!("k:{:06}", mid + 50).into_bytes();
        let sel_prefix = format!("k:{:04}", mid / 100).into_bytes(); // matches k:{mid/100}XX → 100 keys

        eprintln!("\n=== scan latency: {total} keys, 128-bit values, per LSM layer state ===");
        for (state, (_temp, lsm)) in [
            ("memtable", bench_state_memtable(total)),
            ("L1-only", bench_state_l1(total)),
            ("L0+L1+mem", bench_state_l0_l1(total)),
        ] {
            eprintln!("  [{state}]");
            let l = &lsm;
            time_it("prefix(k:) full", 60, || l.scan_prefix(b"k:").unwrap().len(), total as usize);
            time_it(
                "range(k:..) full",
                60,
                || l.range_pointers_bounded(b"k:", Some(b"k;"), usize::MAX).unwrap().len(),
                total as usize,
            );
            let (s, e) = (&sel_start, &sel_end);
            time_it(
                "range selective [50]",
                500,
                || l.range_pointers_bounded(s, Some(e), usize::MAX).unwrap().len(),
                50,
            );
            let p = &sel_prefix;
            time_it("prefix selective [100]", 500, || l.scan_prefix(p).unwrap().len(), 100);
        }
    }

    // ── scan_prefixes tombstone regressions ───────────────────────

    #[test]
    fn test_scan_prefixes_deleted_after_flush() -> Result<()> {
        // Regression: before the fix, scan_prefixes used the active
        // memtable's live-only iterator, so a key deleted after a flush still
        // appeared because the SSTable entry was never shadowed by the tombstone.
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(&u32_prefixed_key(1, 10), 100)?;
        lsm.insert(&u32_prefixed_key(1, 11), 101)?;
        lsm.flush_memtable()?; // both keys now in SSTable

        lsm.delete(&u32_prefixed_key(1, 11))?; // tombstone in active memtable

        let ids: std::collections::HashSet<u32> = [1].iter().copied().collect();
        let results = lsm.scan_prefixes(&ids)?;

        assert_eq!(results.len(), 1, "deleted key must not appear");
        assert!(results.iter().any(|(k, v)| k == &u32_prefixed_key(1, 10) && *v == 100));
        Ok(())
    }

    #[test]
    fn test_scan_prefixes_tombstone_in_ro_memtable() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(&u32_prefixed_key(2, 1), 201)?;
        lsm.flush_memtable()?; // key in SSTable
        lsm.delete(&u32_prefixed_key(2, 1))?;
        lsm.flush_memtable()?; // tombstone now in ro_memtable

        let ids: std::collections::HashSet<u32> = [2].iter().copied().collect();
        let results = lsm.scan_prefixes(&ids)?;
        assert!(results.is_empty(), "key deleted in ro_memtable must not appear");
        Ok(())
    }

    #[test]
    fn test_scan_prefixes_delete_then_reinsert() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(&u32_prefixed_key(3, 5), 300)?;
        lsm.flush_memtable()?;
        lsm.delete(&u32_prefixed_key(3, 5))?;
        lsm.flush_memtable()?; // tombstone in ro_memtable
        lsm.insert(&u32_prefixed_key(3, 5), 999)?; // re-insert in active memtable

        let ids: std::collections::HashSet<u32> = [3].iter().copied().collect();
        let results = lsm.scan_prefixes(&ids)?;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 999, "re-inserted value must win");
        Ok(())
    }

    // ── range_keys_bounded tombstone regressions ──────────────────────────────

    #[test]
    fn test_range_keys_bounded_deleted_after_flush() -> Result<()> {
        // Regression: before the fix, range_keys_bounded scanned SSTables last and
        // used blind set-insert, so a key deleted after a flush still appeared.
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"r:1", 1)?;
        lsm.insert(b"r:2", 2)?;
        lsm.insert(b"r:3", 3)?;
        lsm.flush_memtable()?;

        lsm.delete(b"r:2")?; // tombstone in active memtable

        let keys = lsm.range_keys_bounded(b"r:", 100)?;
        assert!(!keys.contains(&b"r:2".to_vec()), "deleted key must not appear");
        assert!(keys.contains(&b"r:1".to_vec()));
        assert!(keys.contains(&b"r:3".to_vec()));
        assert_eq!(keys.len(), 2);
        Ok(())
    }

    #[test]
    fn test_range_keys_bounded_tombstone_in_ro_memtable() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"m:1", 10)?;
        lsm.insert(b"m:2", 20)?;
        lsm.flush_memtable()?; // both in SSTable
        lsm.delete(b"m:1")?;
        lsm.flush_memtable()?; // tombstone in ro_memtable

        let keys = lsm.range_keys_bounded(b"m:", 100)?;
        assert!(!keys.contains(&b"m:1".to_vec()), "key tombstoned in ro_memtable must not appear");
        assert!(keys.contains(&b"m:2".to_vec()));
        assert_eq!(keys.len(), 1);
        Ok(())
    }

    #[test]
    fn test_range_keys_bounded_delete_then_reinsert() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"q:1", 1)?;
        lsm.flush_memtable()?;
        lsm.delete(b"q:1")?;
        lsm.flush_memtable()?; // tombstone in ro_memtable
        lsm.insert(b"q:1", 2)?; // re-insert in active memtable

        let keys = lsm.range_keys_bounded(b"q:", 100)?;
        assert_eq!(keys, vec![b"q:1".to_vec()], "re-inserted key must appear exactly once");
        Ok(())
    }

    // ── range_pointers_bounded tombstone regressions ─────────────────────

    #[test]
    fn test_range_pointers_bounded_deleted_after_flush() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"p:1", 10)?;
        lsm.insert(b"p:2", 20)?;
        lsm.flush_memtable()?;

        lsm.delete(b"p:2")?; // tombstone in active memtable

        let pairs = lsm.range_pointers_bounded(b"p:", None, 100)?;
        assert_eq!(pairs.len(), 1, "deleted key must not appear");
        assert_eq!(pairs[0].0, b"p:1");
        assert_eq!(pairs[0].1, 10);
        Ok(())
    }

    #[test]
    fn test_range_pointers_bounded_tombstone_in_ro_memtable() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"w:1", 100)?;
        lsm.insert(b"w:2", 200)?;
        lsm.flush_memtable()?;
        lsm.delete(b"w:1")?;
        lsm.flush_memtable()?; // tombstone in ro_memtable

        let pairs = lsm.range_pointers_bounded(b"w:", None, 100)?;
        assert!(!pairs.iter().any(|(k, _, _)| k == b"w:1"), "key deleted in ro_memtable must not appear");
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, b"w:2");
        Ok(())
    }

    // ── L0 visibility tests (buffered L1 + L0 scan correctness) ───────────────

    // Flush data to L0 SSTable files and then purge the in-memory ro_memtable so
    // the only copy of the data is on disk in L0.  Both range functions must still
    // return the data by reading L0 files directly.

    #[test]
    fn test_range_pointers_bounded_sees_l0_only_data() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"l0:1", 11)?;
        lsm.insert(b"l0:2", 22)?;
        lsm.insert(b"l0:3", 33)?;
        // Flush active memtable → ro_memtable, then flush ro_memtable → L0 files.
        lsm.flush_memtable_to_level0()?;
        // Evict the ro_memtable: data is now only in L0 SSTable files.
        lsm.purge_ro_memtables_for_test();

        let pairs = lsm.range_pointers_bounded(b"l0:", None, 100)?;
        assert_eq!(pairs.len(), 3, "all three keys must be visible from L0");
        let keys: Vec<&[u8]> = pairs.iter().map(|(k, _, _)| k.as_slice()).collect();
        assert!(keys.contains(&b"l0:1".as_ref()));
        assert!(keys.contains(&b"l0:2".as_ref()));
        assert!(keys.contains(&b"l0:3".as_ref()));
        assert_eq!(pairs.iter().find(|(k, _, _)| k == b"l0:1").unwrap().1, 11);
        assert_eq!(pairs.iter().find(|(k, _, _)| k == b"l0:2").unwrap().1, 22);
        assert_eq!(pairs.iter().find(|(k, _, _)| k == b"l0:3").unwrap().1, 33);
        Ok(())
    }

    #[test]
    fn test_range_keys_bounded_sees_l0_only_data() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"lk:a", 1)?;
        lsm.insert(b"lk:b", 2)?;
        lsm.flush_memtable_to_level0()?;
        lsm.purge_ro_memtables_for_test();

        let keys = lsm.range_keys_bounded(b"lk:", 100)?;
        assert_eq!(keys.len(), 2, "both keys must be visible from L0");
        assert!(keys.contains(&b"lk:a".to_vec()));
        assert!(keys.contains(&b"lk:b".to_vec()));
        Ok(())
    }

    #[test]
    fn test_range_pointers_bounded_l0_tombstone_suppresses_l1() -> Result<()> {
        // Write a key → compact to L1 → delete it (write tombstone to L0) → verify
        // the range scan returns nothing.
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"lt:1", 99)?;
        lsm.flush_memtable()?;
        lsm.compact_all()?; // data is now in L1, ro_memtable dropped

        lsm.delete(b"lt:1")?;
        lsm.flush_memtable_to_level0()?; // tombstone written to L0 file
        lsm.purge_ro_memtables_for_test(); // only L0 tombstone + L1 data remain

        let pairs = lsm.range_pointers_bounded(b"lt:", None, 100)?;
        assert!(pairs.is_empty(), "L0 tombstone must suppress L1 data");
        Ok(())
    }

    #[test]
    fn test_range_keys_bounded_l0_tombstone_suppresses_l1() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"lkt:x", 5)?;
        lsm.flush_memtable()?;
        lsm.compact_all()?;

        lsm.delete(b"lkt:x")?;
        lsm.flush_memtable_to_level0()?;
        lsm.purge_ro_memtables_for_test();

        let keys = lsm.range_keys_bounded(b"lkt:", 100)?;
        assert!(keys.is_empty(), "L0 tombstone must suppress L1 data");
        Ok(())
    }

    #[test]
    fn test_range_pointers_bounded_l0_reinsert_after_l1_delete() -> Result<()> {
        // Delete a key (tombstone in L1 after compact), then re-insert it in L0.
        // The re-insert in L0 must win.
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"lr:1", 10)?;
        lsm.flush_memtable()?;
        lsm.delete(b"lr:1")?;
        lsm.flush_memtable()?;
        lsm.compact_all()?; // tombstone in L1, ro_memtable dropped

        lsm.insert(b"lr:1", 20)?; // re-insert
        lsm.flush_memtable_to_level0()?;
        lsm.purge_ro_memtables_for_test(); // re-insert only in L0

        let pairs = lsm.range_pointers_bounded(b"lr:", None, 100)?;
        assert_eq!(pairs.len(), 1, "re-inserted key must be visible");
        assert_eq!(pairs[0].0, b"lr:1");
        assert_eq!(pairs[0].1, 20);
        Ok(())
    }

    #[test]
    fn test_range_pointers_bounded_after_compact_all() -> Result<()> {
        // Verify that after a full compact_all cycle, data is visible via L1.
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        lsm.insert(b"ca:1", 1)?;
        lsm.insert(b"ca:2", 2)?;
        lsm.flush_memtable()?;
        lsm.compact_all()?; // data in L1, ro_memtable dropped, L0 obsolete

        let pairs = lsm.range_pointers_bounded(b"ca:", None, 100)?;
        assert_eq!(pairs.len(), 2);
        assert!(pairs.iter().any(|(k, v, _)| k == b"ca:1" && *v == 1));
        assert!(pairs.iter().any(|(k, v, _)| k == b"ca:2" && *v == 2));
        Ok(())
    }

    #[test]
    fn test_range_pointers_bounded_limit_respected_with_l0() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        for i in 0u128..10 {
            lsm.insert(format!("lim:{:02}", i).as_bytes(), i)?;
        }
        lsm.flush_memtable_to_level0()?;
        lsm.purge_ro_memtables_for_test();

        let pairs = lsm.range_pointers_bounded(b"lim:", None, 3)?;
        assert_eq!(pairs.len(), 3, "limit must be respected");
        // BTreeMap ordering guarantees the first 3 lexicographic keys.
        assert_eq!(pairs[0].0, b"lim:00");
        assert_eq!(pairs[1].0, b"lim:01");
        assert_eq!(pairs[2].0, b"lim:02");
        Ok(())
    }

    #[test]
    fn test_range_keys_bounded_limit_respected_with_l0() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        for i in 0u128..8 {
            lsm.insert(format!("klim:{:02}", i).as_bytes(), i)?;
        }
        lsm.flush_memtable_to_level0()?;
        lsm.purge_ro_memtables_for_test();

        let keys = lsm.range_keys_bounded(b"klim:", 4)?;
        assert_eq!(keys.len(), 4, "limit must be respected");
        assert_eq!(keys[0], b"klim:00");
        assert_eq!(keys[3], b"klim:03");
        Ok(())
    }

    // ── bounded-page L1 scan (limit-bounded merge) ─────────────────────────────

    /// A newer-layer (memtable) tombstone shadowing an L1 key *inside* the first `limit`
    /// keys must not short the page. The bounded L1 scan over-reads each bucket by the
    /// count of newer in-range entries, so `limit` live keys still come back. num_buckets=1
    /// exercises the per-bucket cap tightly. **This would FAIL if L1 were capped at exactly
    /// `limit`+1** (the page would come back with only ~11 keys).
    #[test]
    fn test_range_pointers_bounded_page_not_shorted_by_newer_tombstones() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(
            temp_dir.path(),
            LSMConfig {
                num_buckets: 1,
                ..test_lsm_config()
            },
        )?;
        for i in 0u128..50 {
            lsm.insert(format!("k{:02}", i).as_bytes(), i)?;
        }
        lsm.flush_memtable()?;
        lsm.compact_all()?; // k00..k49 all live in L1
        // Delete the first 10 keys — tombstones live in the active memtable (newer than L1).
        for i in 0u128..10 {
            lsm.delete(format!("k{:02}", i).as_bytes())?;
        }
        let pairs = lsm.range_pointers_bounded(b"k", None, 20)?;
        let keys: Vec<String> = pairs.iter().map(|(k, _, _)| String::from_utf8_lossy(k).into_owned()).collect();
        assert_eq!(
            keys.len(),
            20,
            "page must return a full `limit` of live keys despite newer tombstones; got {keys:?}"
        );
        assert_eq!(keys[0], "k10", "deleted head keys must be skipped");
        assert_eq!(keys[19], "k29");
        Ok(())
    }

    /// Keys-only twin of the tombstone-slack test.
    #[test]
    fn test_range_keys_bounded_page_not_shorted_by_newer_tombstones() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(
            temp_dir.path(),
            LSMConfig {
                num_buckets: 1,
                ..test_lsm_config()
            },
        )?;
        for i in 0u128..50 {
            lsm.insert(format!("k{:02}", i).as_bytes(), i)?;
        }
        lsm.flush_memtable()?;
        lsm.compact_all()?;
        for i in 0u128..10 {
            lsm.delete(format!("k{:02}", i).as_bytes())?;
        }
        let keys = lsm.range_keys_bounded(b"k", 20)?;
        assert_eq!(keys.len(), 20, "keys page must be full despite newer tombstones");
        assert_eq!(keys[0], b"k10");
        assert_eq!(keys[19], b"k29");
        Ok(())
    }

    /// Per-bucket slack teeth: the L1 cap is sized by each bucket's OWN newer-entry count,
    /// not a global one spread across buckets. Here 20 tombstones all hash to a SINGLE bucket
    /// and sit at the page head; that bucket's cap must be `limit + 1 + 20` for the page to
    /// come back full. A bound that undercounts the clustered bucket (e.g. `|newer| /
    /// num_buckets ≈ 2`) would read only tombstones and short the page to empty.
    #[test]
    fn test_range_pointers_bounded_per_bucket_slack_clustered_tombstones() -> Result<()> {
        let num_buckets = 8;
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(
            temp_dir.path(),
            LSMConfig {
                num_buckets,
                ..test_lsm_config()
            },
        )?;

        // 40 keys that all hash to bucket 0 (prefix "a", so they sort ahead of everything else).
        let mut bucket0: Vec<Vec<u8>> = Vec::new();
        let mut i = 0u64;
        while bucket0.len() < 40 {
            let k = format!("a{i:06}").into_bytes();
            if get_bucket_for_key(&k, num_buckets) == 0 {
                bucket0.push(k);
            }
            i += 1;
        }
        // Filler spread across the OTHER buckets, at higher keys (prefix "b").
        let mut others: Vec<Vec<u8>> = Vec::new();
        let mut j = 0u64;
        while others.len() < 40 {
            let k = format!("b{j:06}").into_bytes();
            if get_bucket_for_key(&k, num_buckets) != 0 {
                others.push(k);
            }
            j += 1;
        }
        for (v, k) in bucket0.iter().chain(others.iter()).enumerate() {
            lsm.insert(k, v as u128)?;
        }
        lsm.flush_memtable()?;
        lsm.compact_all()?; // all live in L1

        // Delete the first 20 bucket-0 keys — all in ONE bucket, all at the page head.
        for k in &bucket0[..20] {
            lsm.delete(k)?;
        }

        // Scan the "a" range: only bucket 0 has in-range keys, so its per-bucket cap is what
        // gates the page. limit=15 needs the cap to reach past all 20 head tombstones.
        let pairs = lsm.range_pointers_bounded(b"a", Some(b"b"), 15)?;
        let keys: Vec<Vec<u8>> = pairs.iter().map(|(k, _, _)| k.clone()).collect();
        assert_eq!(
            keys.len(),
            15,
            "page must be full despite 20 clustered tombstones in one bucket; got {}",
            keys.len()
        );
        assert_eq!(
            keys.as_slice(),
            &bucket0[20..35],
            "page must be the first live keys after the deleted head"
        );
        Ok(())
    }

    /// Keys-only twin of the clustered-tombstone per-bucket-slack test.
    #[test]
    fn test_range_keys_bounded_per_bucket_slack_clustered_tombstones() -> Result<()> {
        let num_buckets = 8;
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(
            temp_dir.path(),
            LSMConfig {
                num_buckets,
                ..test_lsm_config()
            },
        )?;

        let mut bucket0: Vec<Vec<u8>> = Vec::new();
        let mut i = 0u64;
        while bucket0.len() < 40 {
            let k = format!("a{i:06}").into_bytes();
            if get_bucket_for_key(&k, num_buckets) == 0 {
                bucket0.push(k);
            }
            i += 1;
        }
        let mut others: Vec<Vec<u8>> = Vec::new();
        let mut j = 0u64;
        while others.len() < 40 {
            let k = format!("b{j:06}").into_bytes();
            if get_bucket_for_key(&k, num_buckets) != 0 {
                others.push(k);
            }
            j += 1;
        }
        for (v, k) in bucket0.iter().chain(others.iter()).enumerate() {
            lsm.insert(k, v as u128)?;
        }
        lsm.flush_memtable()?;
        lsm.compact_all()?;

        for k in &bucket0[..20] {
            lsm.delete(k)?;
        }

        let keys = lsm.range_keys_bounded(b"a", 15)?;
        assert_eq!(keys.len(), 15, "keys page must be full despite clustered tombstones; got {}", keys.len());
        assert_eq!(keys.as_slice(), &bucket0[20..35]);
        Ok(())
    }

    /// A run of tombstones **in L1 itself** at the page head must be skipped without
    /// shorting the page (the L1 early-stop counts LIVE entries, not raw entries). The
    /// tombstones reach L1 by compacting a delete-over-an-L0-value into level 1.
    #[test]
    fn test_range_pointers_bounded_l1_tombstones_at_head_skipped() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(
            temp_dir.path(),
            LSMConfig {
                num_buckets: 1,
                ..test_lsm_config()
            },
        )?;
        for i in 0u128..40 {
            lsm.insert(format!("k{:02}", i).as_bytes(), i)?;
        }
        lsm.flush_memtable_to_level0()?; // values land in L0
        // Delete the first 8, flush the tombstones to L0, then compact everything to L1 so
        // the L1 file physically carries tombstones for k00..k07 ahead of the live keys.
        for i in 0u128..8 {
            lsm.delete(format!("k{:02}", i).as_bytes())?;
        }
        lsm.flush_and_compact_all()?;
        lsm.purge_ro_memtables_for_test();
        let pairs = lsm.range_pointers_bounded(b"k", None, 20)?;
        let keys: Vec<String> = pairs.iter().map(|(k, _, _)| String::from_utf8_lossy(k).into_owned()).collect();
        assert_eq!(keys.len(), 20, "page must be full past the L1-tombstone head; got {keys:?}");
        assert_eq!(keys[0], "k08");
        assert_eq!(keys[19], "k27");
        Ok(())
    }

    /// A full cursor walk with small pages, over multiple buckets with scattered deletes,
    /// must return exactly the same ordered set as one unbounded scan — the bounded
    /// per-bucket L1 cap must never drop or duplicate a live key across page boundaries.
    #[test]
    fn test_bounded_page_walk_matches_unbounded_reference() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?; // default multi-bucket
        for i in 0u128..500 {
            lsm.insert(format!("row:{:04}", i).as_bytes(), i)?;
        }
        lsm.flush_memtable()?;
        lsm.compact_all()?;
        // Delete every 7th key (scattered across buckets); tombstones live in the memtable.
        for i in (0u128..500).filter(|i| i % 7 == 0) {
            lsm.delete(format!("row:{:04}", i).as_bytes())?;
        }
        // Reference: one unbounded scan (uses the parallel read-everything path).
        let reference: Vec<Vec<u8>> = lsm
            .range_pointers_bounded(b"row:", None, usize::MAX)?
            .into_iter()
            .map(|(k, _, _)| k)
            .collect();
        assert!(reference.len() >= 400, "sanity: most keys still live");
        assert!(!reference.iter().any(|k| k == b"row:0000"), "row:0000 was deleted");

        // Paginated walk with a small page (limit 20; +1 sentinel for has_more).
        let mut walked: Vec<Vec<u8>> = Vec::new();
        let mut cursor: Vec<u8> = b"row:".to_vec();
        loop {
            let page = lsm.range_pointers_bounded(&cursor, None, 21)?;
            let has_more = page.len() > 20;
            for (k, _, _) in page.iter().take(20) {
                walked.push(k.clone());
            }
            if !has_more {
                break;
            }
            cursor = page[20].0.clone();
        }
        assert_eq!(walked, reference, "bounded paginated walk must equal the unbounded reference");
        Ok(())
    }
}
