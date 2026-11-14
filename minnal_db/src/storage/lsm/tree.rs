//! One namespace's LSM tree: the index from each key to its value's place in
//! the value log (a `u128` pointer), with the write sequence that produced it.
//!
//! Layers, newest first:
//! - **Active memtable**: an in-memory skip list that takes every write.
//! - **Read-only memtables**: the active memtable, sealed once it nears capacity
//!   and kept as a sorted record list until it is flushed.
//! - **Level 0**: a flush writes each read-only memtable as one file per bucket.
//!   A key's bucket is a hash of its first 8 bytes (`num_buckets`, default 8).
//! - **Level 1**: one sorted file per bucket. A bucket's Level-0 files are merged
//!   into it once they are numerous or large enough ([`LSMTree::compaction_due`]).
//!
//! A key can have copies in several layers; the copy with the highest write
//! sequence wins, and if that copy is a tombstone the key is deleted. Every read
//! resolves this way, never by which layer a copy is in.

use super::bloom::BloomFilter;
use super::manifest::{LsmManifest, ManifestBucket, ManifestFile, ManifestLevel};
use super::skip_list::skip_list::{KeyValueRecord, SkipList};
use super::sparse_index::SparseIndex;
pub(crate) use super::version::Version;
use crate::support::{get_bucket_for_key, key_prefix_of, prefix_upper_bound};
use log::warn;
use parking_lot::RwLock;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use thiserror::Error;

use crate::support::metrics::Metrics;

mod batch_read;
mod compaction;
mod displaced;
mod flush;
mod key_scan;
#[cfg(test)]
mod layer_order;
mod level0;
mod memtable;
mod open;
mod point_read;
mod prefix_scan;
mod prefix_sets;
mod range_scan;
mod scan_merge;
mod sstable;
#[cfg(test)]
mod tests;

use level0::*;
use scan_merge::*;
use sstable::*;

pub(crate) use super::skip_list::skip_list::Applied;
use displaced::DisplacedCollector;
pub(crate) use displaced::{Displaced, DisplacedSink};

/// An L0 file written and fsynced by a flush, not yet registered.
struct WrittenLevel0File {
    path: PathBuf,
    created_at_ms: u128,
    summary: Option<L0Summary>,
}

/// Observer interface for memtable flush lifecycle events.
///
/// For one read-only memtable the events arrive in this order, and a flusher
/// handles memtables in version order (it waits on each one's `flush_lock`):
///
/// 1. `on_memtable_sealed` — under the active memtable's write lock, so no
///    write can be applied into the next memtable meanwhile;
/// 2. `on_ro_memtable_flushing` — before the first L0 file is written. At open
///    L0 files are found by listing each bucket's directory, so a file on disk
///    survives a crash whether or not it was registered: whatever must be
///    durable before the memtable's records are (the vector index WAL's
///    records, so that a crash keeps a prefix of that log) is made durable
///    here. An error **vetoes the flush** with nothing written;
/// 3. `on_ro_memtable_written` — every bucket's L0 file is written and fsynced,
///    and **none is registered yet**: nothing can read or merge them, so no
///    merge can have dropped a tombstone they hold. The vector index WAL makes
///    its progress durable here (`db/wal/vector_index_wal/mod.rs`). An error **vetoes
///    the flush**: the files are removed, the memtable stays read-only and
///    unflushed, and a later flush retries it under the same version;
/// 4. `on_ro_memtable_flushed_to_level0` — the files are registered and the
///    memtable is marked flushed. The main WAL marks entries persisted here.
pub(crate) trait LsmFlushObserver: Send + Sync {
    fn on_memtable_sealed(&self, version: u64);
    /// The memtable is about to be written to L0; nothing is on disk yet. An
    /// error vetoes the flush. No default: a wrapping observer that forgot to
    /// forward it would silently skip the vector index WAL's sync.
    fn on_ro_memtable_flushing(&self, version: u64) -> std::io::Result<()>;
    /// The memtable's L0 files are durable but not yet registered. An error
    /// keeps them from being registered. Default: no-op.
    fn on_ro_memtable_written(&self, _version: u64) -> std::io::Result<()> {
        Ok(())
    }
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
    /// A read-only memtable was flushed while an older one was still in
    /// memory. Level 0 must receive memtables in the order they were frozen,
    /// so that a newer layer always holds newer versions; nothing was written.
    #[error("read-only memtable {version} flushed before the older memtable {older}")]
    FlushOutOfOrder { version: u64, older: u64 },
}

pub(crate) type Result<T> = std::result::Result<T, LSMError>;

/// A bucket's Level-0 files are merged into its Level 1 once this many have
/// built up, however small. Every L0 file a point read cannot rule out by its
/// key range or Bloom filter is read, so this bounds the read cost
/// that waiting adds. See [`LSMTree::compaction_due`].
pub(crate) const L0_COMPACTION_MAX_FILES: usize = 8;

/// A bucket's Level-0 files are merged into its Level 1 once they hold at least
/// this percentage of Level 1's entries. A merge rewrites the whole Level-1
/// file, so this bounds what compaction rewrites to about 100 / this times what
/// was flushed. See [`LSMTree::compaction_due`].
pub(crate) const L0_COMPACTION_RATIO_PERCENT: u64 = 10;

/// Configuration for LSM tree compaction behavior
#[derive(Clone, Debug, PartialEq)]
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
            compaction_threshold_percent: crate::defaults::lsm::COMPACTION_THRESHOLD_PERCENT,
            data_dir: PathBuf::from("lsm_data"),
            num_buckets: crate::defaults::sharding::NUM_BUCKETS,
            skip_list_capacity: crate::defaults::memtable::MAX_CAPACITY,
        }
    }
}

impl LSMConfig {
    pub fn new(compaction_threshold_percent: usize, data_dir: PathBuf) -> Self {
        Self {
            compaction_threshold_percent,
            data_dir,
            ..Self::default()
        }
    }
}

/// In-memory summary of a bucket's Level-1 SSTable, rebuilt by scanning the file
/// at open — it is never serialised, so it has no on-disk format of its own.
///
/// Only what readers use: `min_key`/`max_key` bound the range check that lets a
/// lookup skip the file entirely, and `entry_count` tells a populated bucket from
/// an empty one. **Do not add a field without a reader**, and do not derive a
/// trait (such as `rkyv`'s) that nothing uses: the `dead_code` lint is what keeps
/// an unused field from going stale here.
#[derive(Clone, Debug)]
struct SStableMetadata {
    /// Minimum key in this SSTable
    min_key: Vec<u8>,
    /// Maximum key in this SSTable
    max_key: Vec<u8>,
    /// Number of entries in this SSTable
    entry_count: u64,
}

/// Magic identifying an SSTable file, written at offset 0 of every **non-empty**
/// SSTable (L0 and L1 alike). A freshly created, never-written L1 file is 0 bytes
/// and carries no header — `file_len == 0` means "no entries", which stays valid.
const SSTABLE_MAGIC: [u8; 4] = *b"MSST";

/// On-disk format version for the entry encoding that follows the header.
///
/// Bump this whenever the `SStableEntry` layout changes. A file with another
/// version, or without the magic, is rejected by [`validate_sstable_header`]
/// rather than misread: the entry CRC covers only the bytes, so it cannot catch a
/// change in how they are interpreted.
const SSTABLE_FORMAT_VERSION: u32 = 2;

/// Byte offset at which entry frames begin: `magic(4) | version(4) | reserved(8)`.
///
/// The reserved tail keeps the header 16-byte aligned and leaves room to add
/// fields without moving the data start again. Sparse-index offsets are absolute
/// file positions and are seeded with this, so seek-based readers need no
/// adjustment; only scans that start at the head of the file begin here.
const SSTABLE_DATA_START: u64 = 16;

/// Suffix of a Level-0 file being written (`{stamp}.dat.tmp`), renamed to
/// `{stamp}.dat` once complete; see `write_level0_file_for_records`.
const LEVEL0_TMP_SUFFIX: &str = ".dat.tmp";

/// Buffered, positional reader of consecutive SSTable frames (`u32 LE len ‖ body`).
///
/// One `pread` per buffer refill, not two per entry (length, then body):
/// sparse-vector entries are ~60 bytes, so reading each alone would make
/// syscalls, not parsing, the main cost of a cluster range walk.
///
/// The refill size starts at [`MIN_READ`](Self::MIN_READ) and doubles up to
/// [`MAX_READ`](Self::MAX_READ): most range walks are short (a median IVF cluster
/// is a few dozen entries per bucket), so a large fixed read-ahead would copy far
/// more page cache than the walk consumes, while long walks quickly reach the
/// large reads. A frame larger than the buffer grows it to fit.
///
/// EOF semantics match [`LSMTree::read_exact_at`]: a clean EOF exactly at a frame
/// boundary ends the walk (`Ok(None)`); EOF inside a length prefix or body is
/// `UnexpectedEof`. Frame lengths are checked with [`check_frame_len`] before the
/// buffer is sized for them.
struct FrameReader<'a> {
    file: &'a File,
    buf: Vec<u8>,
    /// First unconsumed byte in `buf`.
    pos: usize,
    /// End of the valid bytes in `buf`.
    len: usize,
    /// File offset of `buf[len]` — where the next refill reads from.
    next_offset: u64,
    /// Size of the next refill read.
    read_size: usize,
    /// Bytes read from the file so far (tests use it to check that a batch
    /// lookup reads near its keys, not the whole file).
    bytes_read: u64,
}

/// Most runs [`merge_l1_runs`] merges directly; a scan has one per bucket.
const MAX_MERGED_RUNS: usize = 64;

/// Result of writing a merged SSTable: `(min_key, max_key, entry_count,
/// data_size, sparse_index)`. `data_size` is entry bytes only — it excludes the
/// header, so it is a size for metrics, never a file position.
type MergedSstableInfo = (Vec<u8>, Vec<u8>, u64, u64, SparseIndex);

/// Result of scanning an existing SSTable file at open time:
/// `(metadata, bloom, sparse_index, max_seq)`. The bloom/index/max_seq are
/// `None` when the file has no entries.
type LoadedSstable = (SStableMetadata, Option<BloomFilter>, Option<SparseIndex>, Option<u64>);

/// One layer's copy of a key during a scan: `(key, Some(pointer) | None for a
/// tombstone, seq)`. Tombstones keep their seq so they can be resolved against
/// other layers' copies by [`fold_newest`], never by layer order.
type ScanEntry = (Vec<u8>, Option<u128>, Version);

/// One L1 entry of a prefix scan, its key a range of its bucket's key buffer:
/// `(buffer, start, end, Some(pointer) | None for a tombstone, seq)`. A large scan
/// reads tens of thousands of entries; packing their keys saves an allocation
/// (and a free) each.
type PackedScanEntry = (usize, usize, usize, Option<u128>, Version);

/// What [`LSMTree::scan_prefixes`] returns: every live key under the requested
/// prefixes in ascending order, each with its value pointer. The keys are packed
/// into one buffer rather than one allocation each.
#[derive(Debug, Default)]
pub(crate) struct PrefixScanKeys {
    keys: Vec<u8>,
    /// Where each key ends in `keys`; key `i` starts where key `i - 1` ends.
    ends: Vec<usize>,
    pointers: Vec<u128>,
}

/// A scan's running view of each key: the copy in the newest layer folded so
/// far, `(Some(pointer) | None for a tombstone, version)`.
type ScanMap = std::collections::BTreeMap<Vec<u8>, (Option<u128>, Version)>;

/// A key's current copy: `(Some(pointer) | None for a tombstone, version)`, or
/// `None` if no layer holds the key.
type Candidate = Option<(Option<u128>, Version)>;

/// One bucket's share of a `get_multiple` batch: each distinct pending key and
/// its running [`Candidate`].
type BatchSlots<'k> = std::collections::HashMap<&'k [u8], Candidate>;

/// Per input key, its current `(pointer, version)` or `None` (absent or deleted).
type BatchLookup = Vec<Option<(u128, Version)>>;

/// A single entry in the SSTable
#[derive(Clone, Debug, Archive, RkyvSerialize, RkyvDeserialize)]
struct SStableEntry {
    key: Vec<u8>,
    key_prefix: u64, // First 8 bytes of key as u64 for fast prefix matching
    value: u128,
    tombstone: bool,
    /// The version's user seq ([`Version::user`]).
    seq: u64,
    /// The version's system seq ([`Version::system`]).
    system: u32,
}

impl SStableEntry {
    fn version(&self) -> Version {
        Version {
            user: self.seq,
            system: self.system,
        }
    }

    /// An archived entry's version, without deserialising it.
    fn archived_version(archived: &ArchivedSStableEntry) -> Version {
        Version {
            user: archived.seq.to_native(),
            system: archived.system.to_native(),
        }
    }

    /// A memtable record as an SSTable entry.
    fn of_record(record: &KeyValueRecord) -> Self {
        Self {
            key: record.key.clone(),
            key_prefix: key_prefix_of(&record.key),
            value: record.value,
            tombstone: record.tombstone,
            seq: record.version.user,
            system: record.version.system,
        }
    }

    /// The value-log pointer this entry holds, or `None` for a tombstone.
    fn live_pointer(&self) -> Option<u128> {
        (!self.tombstone).then_some(self.value)
    }
}

/// In-memory component of LSM tree
struct MemTable {
    skip_list: SkipList,
}

/// Read-only snapshot of memtable entries pending compaction
#[derive(Clone)]
struct ReadOnlyMemTable {
    /// Shared with every scan that cloned the read-only list, so a memtable
    /// taken off the list stays readable until the last such scan drops it.
    records: Arc<Vec<KeyValueRecord>>,
    /// Version number for tracking updates
    version: u64,
    /// Whether this memtable has been flushed to Level 0 files
    flushed_to_level0: Arc<AtomicBool>,
    /// Held across a flush of this memtable to Level 0, so two flushers (the
    /// compaction worker, a GC pass, the no-WAL unpin) cannot both write it.
    /// See `flush_ro_memtable_to_level0`.
    flush_lock: Arc<parking_lot::Mutex<()>>,
}

/// What a point lookup needs to know about one Level-0 file without opening it:
/// its key range, a Bloom filter over its keys, and its newest write sequence.
///
/// Computed once, when the file enters the registry, and never updated — which is
/// sound because an L0 file is write-once (never rewritten in place, unlinked only
/// once `readers() == 0`), so nothing can make it stale. Contrast L1, whose
/// metadata is swapped with the file on every merge.
struct L0Summary {
    min_key: Vec<u8>,
    max_key: Vec<u8>,
    /// Highest user seq in the file: folded at open into the bound the write
    /// sequence must start above ([`LSMTree::max_stored_seq`]).
    max_seq: u64,
    /// Records in the file; what [`LSMTree::compaction_due`] weighs against Level 1.
    entries: u64,
    bloom: BloomFilter,
    /// Offsets of every `SAMPLE_INTERVAL`-th frame, so a hit scans a few entries
    /// instead of the file. Each hint is still validated against the file
    /// (`valid_scan_start`), so a bad one costs a full scan, never a wrong answer.
    index: Option<SparseIndex>,
}

/// Level 0 file tracking struct
struct L0FileEntry {
    path: PathBuf,
    created_at_ms: u128,
    /// `None` when the file is empty or its summary could not be built; a lookup
    /// then reads the file rather than skip it.
    summary: Option<L0Summary>,
    reader_count: AtomicU64,
    obsolete: AtomicBool,
    /// Frames [`read_l0_range`] has read from this file, so a test can check a
    /// scan seeks instead of reading the whole file.
    #[cfg(test)]
    frames_read: AtomicU64,
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

/// Whether `a` is a strictly-newer user seq than `b`.
///
/// Sequences are the global `u64` counter (`Database::next_seq`) at full width,
/// so this is a plain comparison. **Do not narrow it** to a smaller, wrapping
/// type: an L1 entry keeps its sequence for the life of the database, so the gap
/// to a later write of the same key is unbounded, and a wrapping comparison
/// inverts once two sequences are more than half its range apart (2^31 writes,
/// about 2.5 days at 10k writes/s, for 32 bits). The newer write would then read
/// as older, and `merge_level0_to_level1` would keep the stale copy in L1 for
/// good. See `test_write_after_huge_seq_gap_beats_cold_l1_entry`.
#[inline]
fn seq_newer(a: u64, b: u64) -> bool {
    a > b
}

/// Outcome of looking a key up in a single SSTable: a value, a tombstone (which
/// must shadow older layers, so "deleted" and "absent" stay distinct), or
/// nothing.
#[derive(Debug, Clone, Copy)]
enum SsLookup {
    /// Live value pointer found in this layer, with its version.
    Found(u128, Version),
    /// An explicit tombstone for the key was found, with its version.
    Deleted(Version),
    /// The key is simply not present in this layer.
    Missing,
}

impl SsLookup {
    /// The copy this lookup found, as a [`Candidate`].
    fn candidate(self) -> Candidate {
        match self {
            SsLookup::Found(ptr, version) => Some((Some(ptr), version)),
            SsLookup::Deleted(version) => Some((None, version)),
            SsLookup::Missing => None,
        }
    }
}

/// Main LSM tree structure
pub(crate) struct LSMTree {
    /// Current active memtable
    memtable: Arc<RwLock<MemTable>>,

    /// Sealed memtables, oldest first, waiting to be flushed to Level 0 (and,
    /// once flushed, to be released).
    read_only_memtables: Arc<RwLock<Vec<ReadOnlyMemTable>>>,

    /// Each bucket's Level-1 file. A merge swaps in the new file's `Arc<File>`;
    /// a reader holding the previous one keeps reading it until it is done.
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

    /// Upper bound on the user seq of any entry below the active memtable,
    /// raised by every memtable sealed and from every L0 and L1 file at open. At
    /// open the memtable is empty, so it bounds everything stored, and the write
    /// sequence must start above it ([`max_stored_seq`](Self::max_stored_seq)).
    max_stored_seq: AtomicU64,

    /// Engine-wide operational counters, shared from `Database` via
    /// [`set_metrics`](Self::set_metrics). `None` until set (e.g. a standalone
    /// tree in tests), in which case metric updates are no-ops.
    metrics: OnceLock<Arc<Metrics>>,
    /// Told which records each L0→L1 merge stopped referencing. Unset for a
    /// standalone tree.
    displaced_sink: OnceLock<Arc<dyn DisplacedSink>>,

    config: LSMConfig,

    /// Per-bucket compaction flags - each bucket can compact independently
    compaction_in_progress: Vec<Arc<AtomicBool>>,

    /// Global version for read-only memtables
    ro_memtable_version: Arc<AtomicU64>,

    /// Level 0 files tracked per bucket for reader-safe compaction/deletion.
    level0_files: Vec<Arc<RwLock<Vec<Arc<L0FileEntry>>>>>,

    /// The newest L0 stamp (`created_at_ms`, also the file name) issued or found
    /// on disk. Every new flush's stamp is clamped above it by
    /// [`next_l0_stamp`](Self::next_l0_stamp), so file order is flush order even
    /// when the wall clock steps backwards.
    last_l0_stamp: parking_lot::Mutex<u128>,
    /// Held across a deletion of merged Level-0 files, unlinks and directory
    /// fsyncs included, so two cleanups cannot interleave their unlinks. See
    /// `cleanup_obsolete_level0_files`.
    level0_cleanup_lock: parking_lot::Mutex<()>,

    /// Optional observer for flush lifecycle events.
    flush_observer: Arc<RwLock<Option<Arc<dyn LsmFlushObserver>>>>,

    base_path: PathBuf,

    /// Test-only fault: how many of the next `insert_with_seq` calls fail
    /// before inserting.
    #[cfg(test)]
    fail_next_inserts: std::sync::atomic::AtomicU32,
}

impl LSMTree {
    /// An upper bound on every sequence this tree holds below its active
    /// memtable (folded from every L0 and L1 file at open, and from each flush).
    /// At open the memtable is empty, so it bounds everything stored: the write
    /// sequence must start above it (see `KVStore::set_seq_counter`).
    pub(crate) fn max_stored_seq(&self) -> u64 {
        self.max_stored_seq.load(Ordering::Relaxed)
    }

    /// Attach the engine-wide operational counters (called once by `KVStore`).
    pub(crate) fn set_metrics(&self, metrics: Arc<Metrics>) {
        let _ = self.metrics.set(metrics);
    }

    /// Attach the receiver of the records each L0→L1 merge stops referencing
    /// (called once by `KVStore`, before any merge can run).
    pub(crate) fn set_displaced_sink(&self, sink: Arc<dyn DisplacedSink>) {
        let _ = self.displaced_sink.set(sink);
    }

    /// The operational counters, if attached. `None` for a standalone tree.
    #[inline]
    fn metrics(&self) -> Option<&Metrics> {
        self.metrics.get().map(|m| m.as_ref())
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

    /// Test-only: make the next `insert_with_seq` fail without inserting, as a
    /// memtable arena past `u32::MAX` would.
    #[cfg(test)]
    pub(crate) fn fail_next_insert(&self) {
        self.fail_next_inserts(1);
    }

    /// Test-only: make the next `count` calls of `insert_with_seq` fail, as
    /// [`fail_next_insert`](Self::fail_next_insert) does for one.
    #[cfg(test)]
    pub(crate) fn fail_next_inserts(&self, count: u32) {
        self.fail_next_inserts.store(count, Ordering::SeqCst);
    }

    /// Test-only: hold the active memtable, so a flush of this tree waits.
    #[cfg(test)]
    pub(crate) fn hold_memtable(&self) -> impl Sized + '_ {
        self.memtable.write()
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
    fn next_test_seq() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(1);
        SEQ.fetch_add(1, Ordering::Relaxed)
    }

    pub(crate) fn insert(&self, key: &[u8], value: u128) -> Result<()> {
        self.insert_with_seq(key, value, Self::next_test_seq()).map(|_| ())
    }

    pub(crate) fn delete(&self, key: &[u8]) -> Result<()> {
        self.delete_with_seq(key, Self::next_test_seq()).map(|_| ())
    }
}

/// In-memory (non-SSTable) statistics about an LSM tree: the live state that
/// the manifest snapshot (`LSMTree::build_manifest_snapshot`) does not capture.
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
