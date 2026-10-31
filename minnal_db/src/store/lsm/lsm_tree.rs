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
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use thiserror::Error;

use crate::db::metrics::Metrics;

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
///    its progress durable here (`db/vector_index_wal.rs`). An error **vetoes
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
}

pub(crate) type Result<T> = std::result::Result<T, LSMError>;

/// A bucket's Level-0 files are merged into its Level 1 once this many have
/// built up, however small. Every L0 file a point read cannot rule out by its
/// key range, Bloom filter or `max_seq` is read, so this bounds the read cost
/// that waiting adds. See [`LSMTree::compaction_due`].
pub(crate) const L0_COMPACTION_MAX_FILES: usize = 8;

/// A bucket's Level-0 files are merged into its Level 1 once they hold at least
/// this percentage of Level 1's entries. A merge rewrites the whole Level-1
/// file, so this bounds what compaction rewrites to about 100 / this times what
/// was flushed. See [`LSMTree::compaction_due`].
pub(crate) const L0_COMPACTION_RATIO_PERCENT: u64 = 10;

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

/// In-memory summary of a bucket's Level-1 SSTable, rebuilt by scanning the file
/// at open — it is never serialised, so it has no on-disk format of its own.
///
/// Deliberately only what readers consume: `min_key`/`max_key` bound the range
/// check that lets a lookup skip the file entirely, and `entry_count` distinguishes
/// a populated bucket from an empty one. It previously also carried `bucket`,
/// `min_key_prefix`, `max_key_prefix`, `data_offset` and `data_size`, none of which
/// were ever read — `bucket` duplicates the index into `sstable_metadata`, and
/// `data_offset`/`data_size` were an accident waiting to happen: `data_offset` was
/// hard-coded to `0` while its doc claimed "offset where entries data starts"
/// (which is [`SSTABLE_DATA_START`] since the header landed), and `data_size`
/// counts entry bytes only, so using it as a file position lands 16 bytes short.
/// The dead fields were masked from the `dead_code` lint by `rkyv` derives that
/// nothing exercised; those are gone too. **Do not add a field here without a
/// reader** — the lint is the only thing keeping this honest.
#[derive(Clone, Debug)]
struct SStableMetadata {
    /// Minimum key in this SSTable
    min_key: Vec<u8>,
    /// Maximum key in this SSTable
    max_key: Vec<u8>,
    /// Number of entries in this SSTable
    entry_count: u64,
    /// Newest write sequence in this SSTable (`None` when it has no entries). An
    /// upper bound used to skip the L1 probe once a newer layer already holds a
    /// copy at least this new. It lives here, not beside the file, so it is
    /// published under the same lock as min/max and inherits their race argument
    /// (see `merge_level0_to_level1`). Too high is safe; too low resurrects.
    max_seq: Option<u64>,
}

/// Magic identifying an SSTable file, written at offset 0 of every **non-empty**
/// SSTable (L0 and L1 alike). A freshly created, never-written L1 file is 0 bytes
/// and carries no header — `file_len == 0` means "no entries", which stays valid.
const SSTABLE_MAGIC: [u8; 4] = *b"MSST";

/// On-disk format version for the entry encoding that follows the header.
///
/// Bump this whenever the `SStableEntry` layout changes in a way that makes old
/// files unreadable. Version 1 is the first versioned format; it carries a `u64`
/// write sequence. Files written before headers existed had a `u32` sequence and
/// no magic, so they are rejected by [`validate_sstable_header`] rather than
/// silently misread — the entry CRC covers only the body and cannot catch a
/// layout change, since the bytes are unchanged and only their interpretation
/// differs.
const SSTABLE_FORMAT_VERSION: u32 = 1;

/// Byte offset at which entry frames begin: `magic(4) | version(4) | reserved(8)`.
///
/// The reserved tail keeps the header 16-byte aligned and leaves room to add
/// fields without moving the data start again. Sparse-index offsets are absolute
/// file positions and are seeded with this, so seek-based readers need no
/// adjustment; only scans that start at the head of the file begin here.
const SSTABLE_DATA_START: u64 = 16;

/// The header bytes written at offset 0 of a non-empty SSTable file.
fn sstable_header_bytes() -> [u8; SSTABLE_DATA_START as usize] {
    let mut header = [0u8; SSTABLE_DATA_START as usize];
    header[0..4].copy_from_slice(&SSTABLE_MAGIC);
    header[4..8].copy_from_slice(&SSTABLE_FORMAT_VERSION.to_le_bytes());
    header
}

/// Suffix of a Level-0 file being written (`{stamp}.dat.tmp`), renamed to
/// `{stamp}.dat` once complete; see `write_level0_file_for_records`.
const LEVEL0_TMP_SUFFIX: &str = ".dat.tmp";

/// Verify that `file` begins with a header this build can read.
///
/// Returns [`LSMError::Corruption`] for a truncated header, a bad magic (which is
/// what a pre-header file looks like), or a format version this build does not
/// understand.
fn validate_sstable_header(file: &File, path: &Path) -> Result<()> {
    let mut header = [0u8; SSTABLE_DATA_START as usize];
    file.read_exact_at(&mut header, 0)
        .map_err(|e| LSMError::Corruption(format!("SSTable {} has no readable header: {e}", path.display())))?;
    if header[0..4] != SSTABLE_MAGIC {
        return Err(LSMError::Corruption(format!(
            "SSTable {} is not a recognised SSTable (bad magic {:02x?}). A database written before SSTable \
             headers existed used a narrower write sequence and cannot be read by this build; recreate it.",
            path.display(),
            &header[0..4]
        )));
    }
    let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
    if version != SSTABLE_FORMAT_VERSION {
        return Err(LSMError::Corruption(format!(
            "SSTable {} has format version {version}, but this build reads version {SSTABLE_FORMAT_VERSION}",
            path.display()
        )));
    }
    Ok(())
}

/// Open an SSTable for a head-of-file scan, **validating its header**, and leave
/// the read cursor at [`SSTABLE_DATA_START`] so sequential readers can keep using
/// `read_exact` unchanged.
///
/// This is the *first-touch* opener: use it where the file has not been vetted
/// yet — the L1 files and on-disk L0 files inspected during [`LSMTree::open`].
/// Files already in the Level-0 registry are opened with
/// [`open_registered_l0_for_scan`] instead; see it for why re-validating is
/// redundant there.
fn open_sstable_for_scan(path: &Path) -> Result<File> {
    let mut file = File::open(path)?;
    validate_sstable_header(&file, path)?;
    file.seek(SeekFrom::Start(SSTABLE_DATA_START))?;
    Ok(file)
}

/// Reject a frame whose declared payload length cannot fit in the file holding it,
/// **before** that length is used to allocate.
///
/// The 4-byte length prefix sits *outside* the entry CRC — which covers only the
/// body — so a flipped bit there is undetectable until after the payload has been
/// read. Allocating first meant a corrupt length could request up to 4 GiB from a
/// file of a few hundred bytes. A frame can never be longer than its file, so that
/// is the bound; `valid_scan_start` already applies the same reasoning to a sparse
/// hint, and this brings the sequential readers in line.
///
/// Zero is rejected too: a body is `crc32(4) + rkyv payload`, so it is never empty.
#[inline]
fn check_frame_len(size: u64, file_len: u64) -> Result<()> {
    if size == 0 || size > file_len {
        return Err(LSMError::Corruption(format!(
            "SSTable frame declares a {size}-byte payload in a {file_len}-byte file"
        )));
    }
    Ok(())
}

/// Buffered, positional reader of consecutive SSTable frames (`u32 LE len ‖ body`).
///
/// Replaces two `pread` syscalls per entry (length, then body) with one `pread`
/// per buffer refill. Sparse-vector entries are ~60 bytes, so a per-entry read
/// made syscalls — not parsing — the dominant cost of a cluster range walk.
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

impl<'a> FrameReader<'a> {
    const MIN_READ: usize = 8 * 1024;
    const MAX_READ: usize = 256 * 1024;

    fn new(file: &'a File, offset: u64) -> Self {
        Self {
            file,
            buf: Vec::new(),
            pos: 0,
            len: 0,
            next_offset: offset,
            read_size: Self::MIN_READ,
            bytes_read: 0,
        }
    }

    /// File offset of the next frame.
    fn offset(&self) -> u64 {
        self.next_offset - (self.len - self.pos) as u64
    }

    /// Continue reading at `offset`, a frame boundary at or past
    /// [`offset`](Self::offset). Skips within the buffer when the target is
    /// already in it; otherwise drops the buffer and restarts small read-ahead.
    fn seek(&mut self, offset: u64) {
        let here = self.offset();
        debug_assert!(offset >= here, "FrameReader only seeks forward");
        if offset >= here && offset < self.next_offset {
            self.pos += (offset - here) as usize;
            return;
        }
        self.pos = 0;
        self.len = 0;
        self.next_offset = offset;
        self.read_size = Self::MIN_READ;
    }

    /// Buffer at least `n` unconsumed bytes, reading more only if needed. Returns
    /// the number of unconsumed bytes, which is below `n` only at end of file.
    fn fill(&mut self, n: usize) -> Result<usize> {
        if self.len - self.pos >= n {
            return Ok(self.len - self.pos);
        }
        self.buf.copy_within(self.pos..self.len, 0);
        self.len -= self.pos;
        self.pos = 0;
        let want = self.len + n.max(self.read_size);
        if self.buf.len() < want {
            self.buf.resize(want, 0);
        }
        self.read_size = (self.read_size * 2).min(Self::MAX_READ);
        while self.len < n {
            match self.file.read_at(&mut self.buf[self.len..], self.next_offset) {
                Ok(0) => break,
                Ok(read) => {
                    self.len += read;
                    self.next_offset += read as u64;
                    self.bytes_read += read as u64;
                }
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {}
                Err(err) => return Err(LSMError::Io(err)),
            }
        }
        Ok(self.len - self.pos)
    }

    /// The next frame's body (`crc32 ‖ payload`), or `None` at a clean EOF.
    fn next_frame(&mut self, file_len: u64) -> Result<Option<&[u8]>> {
        let unexpected_eof = || LSMError::Io(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "Unexpected EOF"));
        match self.fill(4)? {
            0 => return Ok(None),
            1..=3 => return Err(unexpected_eof()),
            _ => {}
        }
        let size = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap()) as usize;
        check_frame_len(size as u64, file_len)?;
        self.pos += 4;
        if self.fill(size)? < size {
            return Err(unexpected_eof());
        }
        let body = &self.buf[self.pos..self.pos + size];
        self.pos += size;
        Ok(Some(body))
    }
}

/// Resolve a prefix scan's layers into live `(key, pointer)` pairs in ascending key order.
///
/// `l1` is the oldest layer, in the order its entries were read: bucket by bucket,
/// each bucket's run key-sorted, each key a range of its bucket's buffer in
/// `l1_keys`. `newer` holds every newer layer (L0, read-only and active memtables)
/// already folded by [`fold_newest`]. The result is what folding `l1` in order and
/// then `newer` into one map by [`fold_newest`] would yield, minus tombstones —
/// without paying a map insert, or an allocation, per L1 entry:
///
/// * [`merge_l1_runs`] merges the runs into key order, keeping the newest copy of
///   a key found in more than one run (the later run on a tie), as the fold would;
/// * [`resolve_scan_layers`] then resolves a key present in both by seq, `newer`
///   winning a tie.
///
/// `scan_prefixes` merges its runs on its workers and calls
/// [`resolve_scan_layers`] itself; this whole form is kept for the test that pins
/// the pair against the fold.
#[cfg(test)]
fn merge_scan_layers(l1_keys: &[Vec<u8>], l1: Vec<PackedScanEntry>, newer: ScanMap) -> PrefixScanKeys {
    let l1 = merge_l1_runs(&l1, |e: &PackedScanEntry| &l1_keys[e.0][e.1..e.2]);
    resolve_scan_layers(l1_keys, l1, newer)
}

/// The second half of [`merge_scan_layers`], for `l1` already in key order with one
/// entry per key (as [`merge_l1_runs`] leaves it): resolve it against `newer` and
/// drop tombstones.
fn resolve_scan_layers(l1_keys: &[Vec<u8>], l1: Vec<PackedScanEntry>, newer: ScanMap) -> PrefixScanKeys {
    let key = |e: &PackedScanEntry| &l1_keys[e.0][e.1..e.2];
    let key_bytes = l1.iter().map(|e| e.2 - e.1).sum::<usize>() + newer.keys().map(Vec::len).sum::<usize>();
    let mut out = PrefixScanKeys::with_capacity(l1.len() + newer.len(), key_bytes);
    let mut older = l1.into_iter().peekable();
    for (newer_key, (val, seq)) in newer {
        while let Some(e) = older.next_if(|e| key(e) < newer_key.as_slice()) {
            if let Some(v) = e.3 {
                out.push(key(&e), v);
            }
        }
        // Same key in both: the higher seq wins, `newer` on a tie.
        let winner = match older.next_if(|e| key(e) == newer_key.as_slice()) {
            Some(e) if seq_newer(e.4, seq) => e.3,
            _ => val,
        };
        if let Some(v) = winner {
            out.push(&newer_key, v);
        }
    }
    for e in older {
        if let Some(v) = e.3 {
            out.push(key(&e), v);
        }
    }
    out
}

/// Merge a scan's L1 entries into key order, one entry per key (the newest, as
/// `seq_newer_or_eq` decides; the later entry on a tie).
///
/// A scan's L1 entries arrive as one run per bucket, each in key order (an
/// SSTable is sorted). Runs are found where the order steps down, and merged
/// directly, taking the smallest head at each step. Sorting the whole set
/// instead was most of a large scan's time: 6–7 ms of a 160k-entry scan, moving
/// every entry through several merge passes. Heads are the keys' first 16 bytes
/// as integers (`key_head`), so the bytes are read only when two heads tie. Input
/// in more than `MAX_MERGED_RUNS` runs is stable-sorted instead, which bounds the
/// cost of an input far from sorted.
fn merge_l1_runs<'a>(l1: &[PackedScanEntry], key: impl Fn(&PackedScanEntry) -> &'a [u8]) -> Vec<PackedScanEntry> {
    let heads: Vec<u128> = l1.iter().map(|e| key_head(key(e))).collect();
    let order = |i: usize, j: usize| heads[i].cmp(&heads[j]).then_with(|| key(&l1[i]).cmp(key(&l1[j])));
    let mut runs: Vec<std::ops::Range<usize>> = Vec::new();
    let mut start = 0;
    for i in 1..=l1.len() {
        if i == l1.len() || order(i - 1, i).is_gt() {
            runs.push(start..i);
            start = i;
        }
    }
    if runs.len() <= MAX_MERGED_RUNS {
        return merge_runs(l1, &runs, &order);
    }
    let mut sorted: Vec<usize> = (0..l1.len()).collect();
    sorted.sort_by(|&i, &j| order(i, j));
    merge_runs_in(l1, &sorted, &order)
}

/// Most runs [`merge_l1_runs`] merges directly; a scan has one per bucket.
const MAX_MERGED_RUNS: usize = 64;

/// Keep the newest of each key's entries, visiting `l1` in `sorted` order (key
/// order, input order among equal keys).
fn merge_runs_in(l1: &[PackedScanEntry], sorted: &[usize], order: &impl Fn(usize, usize) -> std::cmp::Ordering) -> Vec<PackedScanEntry> {
    let mut out: Vec<PackedScanEntry> = Vec::with_capacity(sorted.len());
    let mut last: Option<usize> = None;
    for &i in sorted {
        if let Some(l) = last
            && order(l, i).is_eq()
        {
            let kept = out.last_mut().expect("`last` names the entry just pushed");
            if seq_newer_or_eq(l1[i].4, kept.4) {
                kept.3 = l1[i].3;
                kept.4 = l1[i].4;
            }
            continue;
        }
        out.push(l1[i]);
        last = Some(i);
    }
    out
}

/// Merge `runs` of `l1` (each in key order by `order`) into one, keeping the
/// newest copy of each key (the later run on a tie).
fn merge_runs(l1: &[PackedScanEntry], runs: &[std::ops::Range<usize>], order: &impl Fn(usize, usize) -> std::cmp::Ordering) -> Vec<PackedScanEntry> {
    let mut next: Vec<usize> = runs.iter().map(|r| r.start).collect();
    let mut out: Vec<PackedScanEntry> = Vec::with_capacity(runs.iter().map(|r| r.len()).sum());
    let mut last: Option<usize> = None;
    loop {
        let mut best: Option<usize> = None;
        for (r, run) in runs.iter().enumerate() {
            if next[r] < run.end && best.is_none_or(|b| order(next[r], next[b]).is_lt()) {
                best = Some(r);
            }
        }
        let Some(b) = best else { break };
        let i = next[b];
        next[b] += 1;
        // A key in more than one run: keep whichever copy is newer.
        if let Some(l) = last
            && order(l, i).is_eq()
        {
            let kept = out.last_mut().expect("`last` names the entry just pushed");
            if seq_newer_or_eq(l1[i].4, kept.4) {
                kept.3 = l1[i].3;
                kept.4 = l1[i].4;
            }
            continue;
        }
        out.push(l1[i]);
        last = Some(i);
    }
    out
}

/// A key's first 16 bytes as a big-endian integer, zero-padded: comparing heads,
/// then the keys on a tie, is the keys' byte order.
fn key_head(key: &[u8]) -> u128 {
    let mut head = [0u8; 16];
    let n = key.len().min(16);
    head[..n].copy_from_slice(&key[..n]);
    u128::from_be_bytes(head)
}

/// Open a **registered** Level-0 file — one reachable from `level0_files` — for a
/// positional reader, skipping header validation.
///
/// # Why skipping is sound
///
/// A file's format version is fixed when it is written and cannot change
/// afterwards: L0 files are write-once (never mutated in place) and are unlinked
/// only once `readers() == 0`. So the header is a property to check when a file
/// *enters* the registry, not on every read. Both entry points establish it:
/// [`LSMTree::open`] validates every on-disk L0 file it loads
/// (`load_existing_level0_files` → `level0_file_summary`), and
/// [`register_level0_file`](LSMTree::register_level0_file) only ever receives a
/// file this process just wrote with a header. Re-validating on each read bought
/// nothing and cost a `pread` per L0 file on every point get —
/// `search_level0_files` opens every L0 file in the bucket its summary cannot
/// rule out.
///
/// **Precondition:** `path` comes from an `L0FileEntry`. Opening an unvetted file
/// with this would parse a foreign layout as entries — use
/// [`open_sstable_for_scan`] there.
fn open_registered_l0(path: &Path) -> Result<File> {
    Ok(File::open(path)?)
}

/// [`open_registered_l0`] for a sequential reader: leaves the cursor at
/// [`SSTABLE_DATA_START`]. Same precondition — the caller's path must come from
/// an `L0FileEntry`.
fn open_registered_l0_for_scan(path: &Path) -> Result<File> {
    let mut file = open_registered_l0(path)?;
    file.seek(SeekFrom::Start(SSTABLE_DATA_START))?;
    Ok(file)
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
type ScanEntry = (Vec<u8>, Option<u128>, u64);

/// One L1 entry of a prefix scan, its key a range of its bucket's key buffer:
/// `(buffer, start, end, Some(pointer) | None for a tombstone, seq)`. A large scan
/// reads tens of thousands of entries; packing their keys saves an allocation
/// (and a free) each.
type PackedScanEntry = (usize, usize, usize, Option<u128>, u64);

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

impl PrefixScanKeys {
    pub(crate) fn with_capacity(entries: usize, key_bytes: usize) -> Self {
        Self {
            keys: Vec::with_capacity(key_bytes),
            ends: Vec::with_capacity(entries),
            pointers: Vec::with_capacity(entries),
        }
    }

    pub(crate) fn push(&mut self, key: &[u8], pointer: u128) {
        self.keys.extend_from_slice(key);
        self.ends.push(self.keys.len());
        self.pointers.push(pointer);
    }

    /// The number of keys.
    pub(crate) fn len(&self) -> usize {
        self.ends.len()
    }

    /// Whether no key matched.
    pub(crate) fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// Key `i`.
    pub(crate) fn key(&self, i: usize) -> &[u8] {
        let start = if i == 0 { 0 } else { self.ends[i - 1] };
        &self.keys[start..self.ends[i]]
    }

    /// Key `i`'s value pointer.
    pub(crate) fn pointer(&self, i: usize) -> u128 {
        self.pointers[i]
    }

    /// The `(key, pointer)` pairs as owned values, for comparing in tests.
    #[cfg(test)]
    pub(crate) fn to_pairs(&self) -> Vec<(Vec<u8>, u128)> {
        (0..self.len()).map(|i| (self.key(i).to_vec(), self.pointer(i))).collect()
    }
}

/// A scan's running view of each key: the newest copy seen so far,
/// `(Some(pointer) | None for a tombstone, seq)`.
type ScanMap = std::collections::BTreeMap<Vec<u8>, (Option<u128>, u64)>;

/// Fold one layer's copy of `key` into a scan's running view, keeping the
/// highest `seq`. Callers fold layers **oldest first**, so `>=` lets an exact tie
/// go to the newer layer — the same rule `get_with_seq` applies.
///
/// Every multi-key read resolves through this, never by "the newer layer
/// overwrites": a lower-seq copy can sit in a newer layer (a TTL delete that
/// allocated its seq before a put it lost to, or L0 files ordered by a wall clock
/// that stepped backwards), and overwriting by layer then hides a live key or
/// surfaces a deleted one (`test_scans_agree_with_brute_force_resolution`).
fn fold_newest(map: &mut ScanMap, key: Vec<u8>, value: Option<u128>, seq: u64) {
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

/// A key's running winner during cross-layer resolution: the newest-`seq` copy
/// seen so far, `(Some(pointer) | None for a tombstone, seq)`, or `None` if no
/// layer visited yet holds the key.
type Candidate = Option<(Option<u128>, u64)>;

/// One bucket's share of a `get_multiple` batch: each distinct pending key and
/// its running [`Candidate`].
type BatchSlots<'k> = std::collections::HashMap<&'k [u8], Candidate>;

/// Per input key, its winning `(pointer, seq)` or `None` (absent or deleted).
type BatchLookup = Vec<Option<(u128, u64)>>;

/// A single entry in the SSTable
#[derive(Clone, Debug, Archive, RkyvSerialize, RkyvDeserialize)]
struct SStableEntry {
    key: Vec<u8>,
    key_prefix: u64, // First 8 bytes of key as u64 for fast prefix matching
    value: u128,
    tombstone: bool,
    seq: u64,
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
    /// Held across a flush of this memtable to Level 0, so two flushers (the
    /// compaction worker, a GC pass, the no-WAL unpin) cannot both write it.
    /// See `flush_ro_memtable_to_level0`.
    flush_lock: Arc<parking_lot::Mutex<()>>,
}

impl ReadOnlyMemTable {
    fn new(records: Vec<KeyValueRecord>, version: u64) -> Self {
        // `find` binary-searches; an unsorted or duplicated dump would make
        // lookups miss silently rather than fail.
        debug_assert!(
            records.windows(2).all(|w| w[0].key < w[1].key),
            "read-only memtable records must be strictly ascending by key"
        );
        Self {
            records: Arc::new(records),
            creation_time: std::time::Instant::now(),
            reader_count: Arc::new(AtomicU64::new(0)),
            version,
            flushed_to_level0: Arc::new(AtomicBool::new(false)),
            flush_lock: Arc::new(parking_lot::Mutex::new(())),
        }
    }

    fn records(&self) -> Arc<Vec<KeyValueRecord>> {
        Arc::clone(&self.records)
    }

    /// The record for `key`, if this memtable holds one. Binary search: the
    /// records are the skip list's dump, which is in ascending byte order (its
    /// zero-padded big-endian prefix compare agrees with `[u8]`'s `Ord`) with one
    /// record per key — `new` checks both in debug builds.
    fn find(&self, key: &[u8]) -> Option<&KeyValueRecord> {
        self.records
            .binary_search_by(|record| record.key.as_slice().cmp(key))
            .ok()
            .map(|i| &self.records[i])
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
    max_seq: u64,
    /// Records in the file; what [`LSMTree::compaction_due`] weighs against Level 1.
    entries: u64,
    bloom: BloomFilter,
    /// Offsets of every `SAMPLE_INTERVAL`-th frame, so a hit scans a few entries
    /// instead of the file. Each hint is still validated against the file
    /// (`valid_scan_start`), so a bad one costs a full scan, never a wrong answer.
    index: Option<SparseIndex>,
}

impl L0Summary {
    /// Summarise a file's records, or `None` for an empty file. `records` must be
    /// key-sorted, as every L0 file is.
    fn of_sorted<'a>(records: impl ExactSizeIterator<Item = (&'a [u8], u64)> + Clone) -> Option<Self> {
        let len = records.len();
        let mut iter = records.clone();
        let (min_key, first_seq) = iter.next()?;
        let (max_key, max_seq) = iter.fold((min_key, first_seq), |(_, max_seq), (key, seq)| {
            (key, if seq_newer(seq, max_seq) { seq } else { max_seq })
        });
        Some(Self {
            min_key: min_key.to_vec(),
            max_key: max_key.to_vec(),
            max_seq,
            entries: len as u64,
            bloom: BloomFilter::build(records.map(|(key, _)| key), len),
            index: None,
        })
    }
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
}

impl L0FileEntry {
    fn new(path: PathBuf, created_at_ms: u128, summary: Option<L0Summary>) -> Self {
        Self {
            path,
            created_at_ms,
            summary,
            reader_count: AtomicU64::new(0),
            obsolete: AtomicBool::new(false),
        }
    }

    /// Whether `key` could be in this file. `false` is exact (outside the key
    /// range, or a Bloom negative); `true` means read the file to find out.
    fn may_contain(&self, key: &[u8]) -> bool {
        self.summary
            .as_ref()
            .is_none_or(|s| key >= s.min_key.as_slice() && key <= s.max_key.as_slice() && s.bloom.contains(key))
    }

    /// Whether every entry in this file is at most `seq` — so, visiting layers
    /// newest first with a candidate at `seq` already in hand, nothing here can
    /// win (an exact tie stays with the newer layer already visited).
    fn cannot_beat(&self, seq: u64) -> bool {
        self.summary.as_ref().is_some_and(|s| seq_newer_or_eq(seq, s.max_seq))
    }

    /// Where to start scanning this file for `key`: a sparse-index hint, or the
    /// head of the file when there is no index.
    fn scan_start(&self, key: &[u8]) -> u64 {
        self.summary
            .as_ref()
            .and_then(|s| s.index.as_ref())
            .map_or(0, |index| index.block_start(key))
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

/// Whether `a` is a strictly-newer write sequence than `b`, matching the
/// memtable's comparison (`SkipList::seq_is_newer_or_equal`). Resolution across
/// layers picks the entry with the newest sequence, so order of inspection only
/// breaks exact ties.
///
/// Sequences are the global `u64` counter (`Database::next_seq`) at full width,
/// so this is a plain comparison. It was previously a `u32` truncation compared
/// with wrapping serial-number arithmetic, which silently inverts once two
/// sequences differ by more than 2^31 — reachable here because an L1 entry keeps
/// its original sequence for the lifetime of the database, so the gap to a later
/// write of the same key is unbounded (~2.1B writes, ~2.5 days at 10k writes/s).
/// The effect was a stale read that `merge_level0_to_level1` then baked into L1
/// permanently. **Do not narrow the sequence again** — see
/// `test_write_after_huge_seq_gap_beats_cold_l1_entry`.
#[inline]
fn seq_newer(a: u64, b: u64) -> bool {
    a > b
}

/// Whether `a` is newer than *or equal to* `b` (see [`seq_newer`]).
#[inline]
fn seq_newer_or_eq(a: u64, b: u64) -> bool {
    a >= b
}

/// Outcome of looking a key up in a single SSTable layer, carrying the matched
/// entry's write sequence so resolution can pick the newest version across all
/// layers (a GC re-point can place a low-sequence value in a *newer* layer, so
/// layer order alone is not a safe proxy for recency).
#[derive(Debug, Clone, Copy)]
enum SsLookup {
    /// Live value pointer found in this layer, with its stored sequence.
    Found(u128, u64),
    /// An explicit tombstone for the key was found, with its stored sequence.
    Deleted(u64),
    /// The key is simply not present in this layer.
    Missing,
}

impl SsLookup {
    /// The matched entry's sequence, or `None` for [`SsLookup::Missing`].
    fn seq(&self) -> Option<u64> {
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
    max_lower_seq: AtomicU64,

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

    /// The newest L0 stamp (`created_at_ms`, also the file name) issued or found
    /// on disk. Every new flush's stamp is clamped above it by
    /// [`next_l0_stamp`](Self::next_l0_stamp), so file order is flush order even
    /// when the wall clock steps backwards.
    last_l0_stamp: parking_lot::Mutex<u128>,

    /// Optional observer for flush lifecycle events.
    flush_observer: Arc<RwLock<Option<Arc<dyn LsmFlushObserver>>>>,

    base_path: PathBuf,
}

impl LSMTree {
    /// An upper bound on every sequence this tree holds below its active
    /// memtable (folded from every L0 and L1 file at open, and from each flush).
    /// At open the memtable is empty, so it bounds everything stored: the write
    /// sequence must start above it (see `KVStore::set_seq_counter`).
    pub(crate) fn max_stored_seq(&self) -> u64 {
        self.max_lower_seq.load(Ordering::Relaxed)
    }

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

    /// Scan an SSTable file for `key`, starting at `start_offset` (a sparse-index
    /// hint, or 0 for the whole file).
    ///
    /// The result is tri-state, and the distinction between
    /// [`SsLookup::Deleted`] and [`SsLookup::Missing`] is load-bearing: a
    /// tombstone found in a newer layer (e.g. an L0 file) must *shadow* a live
    /// value in an older layer (L1). Collapsing both into `None` let `get` fall
    /// through an L0 tombstone to a stale L1 value, resurrecting deleted keys.
    ///
    /// The hint is validated against the file first: if it does not point at a
    /// complete, CRC-valid frame whose key is `<= key` — e.g. a concurrent
    /// compaction swapped the file under the in-memory index, leaving the offset
    /// stale — the scan restarts from 0. Entries are key-sorted, so beginning at
    /// *any* valid frame boundary with key `<= key` still finds `key` if present;
    /// this makes a stale hint a performance issue, never a correctness one.
    fn lookup_in_sstable_file_from(&self, file: &File, key: &[u8], start_offset: u64) -> Result<SsLookup> {
        Ok(self.lookup_in_sstable_file_hinted(file, key, start_offset)?.0)
    }

    /// [`lookup_in_sstable_file_from`](Self::lookup_in_sstable_file_from), also
    /// reporting whether a non-zero `start_offset` was rejected and the scan fell
    /// back to the head of the file.
    ///
    /// A rejection is never a wrong answer, only a slow one, which is exactly why
    /// it is counted (`sparse_hint_rejects`): it is otherwise invisible. For L1 it
    /// happens legitimately while a compaction swaps the file under the index; for
    /// an immutable L0 file it means the index's offsets are wrong, and the L0
    /// caller asserts on it in debug builds.
    fn lookup_in_sstable_file_hinted(&self, file: &File, key: &[u8], start_offset: u64) -> Result<(SsLookup, bool)> {
        // Also the bound for `check_frame_len` below: a frame can never be longer
        // than its file.
        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
        let hint_rejected = start_offset != 0 && !Self::valid_scan_start(file, start_offset, file_len, key);
        if hint_rejected && let Some(m) = self.metrics() {
            Metrics::bump(&m.sparse_hint_rejects);
        }
        let offset = if start_offset != 0 && !hint_rejected {
            start_offset
        } else {
            SSTABLE_DATA_START
        };
        Ok((Self::scan_sstable_file_for(file, key, offset, file_len)?, hint_rejected))
    }

    /// The forward scan behind [`lookup_in_sstable_file_hinted`](Self::lookup_in_sstable_file_hinted),
    /// from `offset` (a validated frame boundary) until `key` is found or passed.
    fn scan_sstable_file_for(file: &File, key: &[u8], mut offset: u64, file_len: u64) -> Result<SsLookup> {
        loop {
            let mut size_buf = [0u8; 4];
            if !Self::read_exact_at(file, &mut size_buf, &mut offset)? {
                break;
            }

            let size = u32::from_le_bytes(size_buf);
            check_frame_len(u64::from(size), file_len)?;
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
        // Bound for `check_frame_len`: a frame can never be longer than its file.
        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
        let mut offset = SSTABLE_DATA_START;

        loop {
            let mut size_buf = [0u8; 4];
            if !Self::read_exact_at(file, &mut size_buf, &mut offset)? {
                break;
            }

            let size = u32::from_le_bytes(size_buf);
            check_frame_len(u64::from(size), file_len)?;
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
        let mut max_lower_seq_init: Option<u64> = None;

        // Open or create Level 1 SSTable files per bucket (similar to sharded value log)
        for bucket in 0u32..num_buckets as u32 {
            let level1_path = Self::level1_path_from(&base_path, bucket);
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&level1_path)?;

            let file_len = std::fs::metadata(&level1_path)?.len();
            // Reject an unreadable on-disk format loudly, before anything tries to
            // interpret the file. The metadata load below is deliberately lenient
            // (a partially-written tail leaves the bucket without a sparse index
            // rather than failing open), but that leniency must not extend to a
            // file this build cannot parse at all — scanning it would yield
            // garbage entries instead of an error.
            if file_len > 0 {
                validate_sstable_header(&file, &level1_path)?;
            }
            sstable_files_vec.push(Arc::new(RwLock::new(Arc::new(file))));

            let mut metadata = Vec::new();
            let mut bloom = None;
            let mut index = None;
            if file_len > 0
                && let Ok((meta, built_bloom, built_index, file_max_seq)) = Self::load_metadata_from_file(&level1_path)
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

        // Seed the L0 stamp floor from the newest file on disk, so the first flush
        // after a restart sorts after every file already there even if the clock
        // is now behind the one that named them.
        let last_l0_stamp = level0_files
            .iter()
            .flat_map(|bucket| bucket.read().iter().map(|e| e.created_at_ms).collect::<Vec<_>>())
            .max()
            .unwrap_or(0);

        // Per-bucket compaction flags
        let compaction_in_progress: Vec<Arc<AtomicBool>> = (0..num_buckets).map(|_| Arc::new(AtomicBool::new(false))).collect();

        Ok(Self {
            memtable: Arc::new(RwLock::new(MemTable::with_capacity(config.skip_list_capacity))),
            read_only_memtables: Arc::new(RwLock::new(Vec::new())),
            sstable_files,
            sstable_metadata,
            sstable_blooms,
            sstable_indexes,
            max_lower_seq: AtomicU64::new(max_lower_seq_init.unwrap_or(0)),
            metrics: OnceLock::new(),
            config,
            compaction_in_progress,
            ro_memtable_version: Arc::new(AtomicU64::new(0)),
            memtable_sequence: Arc::new(AtomicUsize::new(0)),
            pending_old_memtables: Arc::new(RwLock::new(Vec::new())),
            level0_files,
            last_l0_stamp: parking_lot::Mutex::new(last_l0_stamp),
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

    fn notify_ro_memtable_flushing(&self, version: u64) -> std::io::Result<()> {
        match self.flush_observer.read().as_ref() {
            Some(observer) => observer.on_ro_memtable_flushing(version),
            None => Ok(()),
        }
    }

    fn notify_ro_memtable_written(&self, version: u64) -> std::io::Result<()> {
        match self.flush_observer.read().as_ref() {
            Some(observer) => observer.on_ro_memtable_written(version),
            None => Ok(()),
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

    /// Whether the tree at `base_path` holds any L1 data, without opening it:
    /// a non-empty `level1_{bucket}.dat`. L1 is where compaction drops
    /// tombstones, so it is what a full WAL replay could contradict.
    pub(crate) fn has_level1_data(base_path: &Path) -> std::io::Result<bool> {
        let dir = Self::level1_dir_from(base_path);
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(e),
        };
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("level1_") && name.ends_with(".dat") && entry.metadata()?.len() > 0 {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Level 1 directory path under base path
    fn level1_dir_from(base_path: &Path) -> PathBuf {
        base_path.join("level1")
    }

    /// Level 1 SSTable path for a bucket
    ///
    /// Naming note:
    /// - Level 1 uses a fixed file per bucket (level1_<bucket>.dat).
    /// - Level 0 uses timestamp-based filenames per flush (see write_level0_file_for_records).
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

    /// Scan an on-disk Level-0 file at open (validating its header) and summarise
    /// it for the read path. `Ok(None)` when the file is empty or cannot be opened;
    /// a frame that fails its length or CRC check is an error, as in every reader.
    fn level0_file_summary(path: &Path) -> Result<Option<L0Summary>> {
        let (meta, bloom, index, max_seq) = match Self::load_metadata_from_file(path) {
            Ok(loaded) => loaded,
            Err(LSMError::Io(_)) => return Ok(None),
            Err(e) => return Err(e),
        };
        Ok(bloom.zip(max_seq).map(|(bloom, max_seq)| L0Summary {
            min_key: meta.min_key,
            max_key: meta.max_key,
            max_seq,
            entries: meta.entry_count,
            bloom,
            index,
        }))
    }

    /// Load the on-disk Level-0 files into the in-memory registry and return the
    /// newest write sequence across all of them. The returned seq is folded into
    /// `max_lower_seq` so the read fast path's bound covers L0 too — L0 is newer
    /// than L1, so omitting it would leave the bound too low after a restart (a
    /// low-seq active entry could then wrongly short-circuit above a higher-seq
    /// L0 tombstone).
    fn load_existing_level0_files(base_path: &Path, level0_files: &[Arc<RwLock<Vec<Arc<L0FileEntry>>>>]) -> Result<Option<u64>> {
        let mut max_seq: Option<u64> = None;
        for bucket in 0u32..level0_files.len() as u32 {
            let dir = Self::level0_bucket_dir_from(base_path, bucket);
            if !dir.exists() {
                continue;
            }
            let mut guard = level0_files[bucket as usize].write();
            for entry in std::fs::read_dir(&dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.file_name().is_some_and(|n| n.to_string_lossy().ends_with(LEVEL0_TMP_SUFFIX)) {
                    // A flush killed before its rename. It never reported itself
                    // written, so its WAL entries were never retired: replay
                    // restores its records.
                    warn!(
                        "removing unfinished Level-0 file {} (a flush interrupted before its rename)",
                        path.display()
                    );
                    std::fs::remove_file(&path)?;
                    continue;
                }
                if path.extension().map(|ext| ext == "dat").unwrap_or(false) {
                    // L0 files are now renamed into place once complete (above), so a
                    // crash no longer leaves a `.dat` too short to hold a header; one
                    // can still come from a build that wrote L0 in place. Such a file
                    // is crash debris, not a format mismatch: skip it rather than
                    // register it, or every later scan would fail header validation on
                    // it. Its records are still in the WAL — the flush marks them
                    // persisted only after the file is durable — so replay restores
                    // them. A file long enough to have a header still goes through
                    // validation, so a pre-header database is rejected rather than
                    // quietly skipped, and a `.dat` cut inside a frame fails the open.
                    if std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) < SSTABLE_DATA_START {
                        warn!("skipping incomplete Level-0 file {} (shorter than an SSTable header)", path.display());
                        continue;
                    }
                    let summary = Self::level0_file_summary(&path)?;
                    if let Some(seq) = summary.as_ref().map(|s| s.max_seq)
                        && max_seq.is_none_or(|m| seq_newer(seq, m))
                    {
                        max_seq = Some(seq);
                    }
                    let created_at = Self::level0_created_at_from_path(&path);
                    guard.push(Arc::new(L0FileEntry::new(path, created_at, summary)));
                }
            }
        }
        Ok(max_seq)
    }

    /// The stamp for the next L0 flush: `candidate` (wall-clock based, so file names
    /// stay readable), but never at or below any stamp already issued or on disk.
    ///
    /// L0 file order breaks **exact seq ties**, and GC's relocation makes them: it
    /// re-inserts a key at its existing seq with a new pointer, so the old and the
    /// relocated copy can sit in two L0 files with the same seq, and only file
    /// order says the relocation is newer. Ordering by raw wall time let a clock
    /// step backwards (NTP, a VM or WSL resume) between the two flushes put the
    /// relocation *first*: every read returned the pre-relocation pointer, the
    /// L0→L1 merge baked it into L1, and once GC unlinked its segment the key read
    /// `SegmentMissing` for good (and the next GC pass judged the real record
    /// dead). Stamps are taken after the per-memtable flush lock, and memtables
    /// flush in version order, so a later flush always gets a larger stamp.
    fn next_l0_stamp(&self, candidate: u128) -> u128 {
        let mut last = self.last_l0_stamp.lock();
        let stamp = candidate.max(last.saturating_add(1));
        *last = stamp;
        stamp
    }

    /// Add a Level-0 file to the in-memory registry.
    ///
    /// One of the two points that establish "a registered L0 file has a valid
    /// header" — the invariant [`open_registered_l0`] relies on to skip
    /// re-validation on every read. Here it holds by construction: the only caller
    /// is `flush_ro_memtable_to_level0`, whose `write_level0_file_for_records` just wrote the header and
    /// fsynced. The other point is `load_existing_level0_files` at open, which
    /// validates what it finds on disk. **Do not register a file from anywhere
    /// else without validating its header first.**
    fn register_level0_file(&self, bucket: u32, path: PathBuf, created_at_ms: u128, summary: Option<L0Summary>) {
        let mut guard = self.level0_files[bucket as usize].write();
        if guard.iter().any(|entry| entry.path == path) {
            return;
        }
        guard.push(Arc::new(L0FileEntry::new(path, created_at_ms, summary)));
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
                guards.sort_by_key(|a| a.entry.created_at_ms);
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
        // The flushed check and the flush must be one step. Several paths flush
        // (the compaction worker, a GC pass, the no-WAL unpin) and nothing else
        // serialises them, so an unlocked check let two write the same memtable:
        // a duplicate L0 file, or — in the same millisecond — the same file name,
        // with the second `File::create` truncating the first mid-write. A second
        // flusher WAITS rather than skipping, because callers rely on the flush
        // being durable when this returns: GC unlinks value-log segments right
        // after, and returning early would let it unlink before the fsync.
        let _flushing = ro_memtable.flush_lock.lock();
        if ro_memtable.is_flushed_to_level0() {
            return Ok(());
        }

        // Nothing is on disk yet, so a veto leaves nothing to undo: the
        // memtable stays read-only and unflushed, and a later flush retries it.
        self.notify_ro_memtable_flushing(ro_memtable.version()).map_err(LSMError::Io)?;

        let records = ro_memtable.records();
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let timestamp = self.next_l0_stamp(now_ms.saturating_mul(1_000_000).saturating_add(ro_memtable.version() as u128));

        // Write and fsync every bucket's file first, register them after. Between
        // the two, observers learn the files are durable while nothing can yet
        // read or merge them (`on_ro_memtable_written`): the vector index WAL
        // relies on that to make its progress durable before a merge could drop
        // a tombstone these files hold. Reads stay correct meanwhile, since the
        // read-only memtable is searched until it is marked flushed below.
        let mut written: Vec<(u32, WrittenLevel0File)> = Vec::new();
        for bucket in 0u32..self.config.num_buckets as u32 {
            match self.write_level0_file_for_records(bucket, &records, timestamp) {
                Ok(Some(file)) => written.push((bucket, file)),
                Ok(None) => {}
                Err(e) => {
                    // Unregistered files are not live; remove them so a retry
                    // (with a new stamp) does not leave duplicates on disk.
                    for (_, file) in &written {
                        let _ = std::fs::remove_file(&file.path);
                    }
                    return Err(e);
                }
            }
        }

        if let Err(e) = self.notify_ro_memtable_written(ro_memtable.version()) {
            // An observer could not record the flush (the vector index WAL's
            // progress marker, or the value-log fsync before it). Registering
            // anyway would make the files mergeable while that progress is not
            // durable, so the flush is undone and retried later.
            for (_, file) in &written {
                let _ = std::fs::remove_file(&file.path);
            }
            return Err(LSMError::Io(e));
        }
        for (bucket, file) in written {
            self.register_level0_file(bucket, file.path, file.created_at_ms, file.summary);
        }

        ro_memtable.mark_flushed_to_level0();
        self.notify_ro_memtable_flushed_to_level0(ro_memtable.version());
        Ok(())
    }

    /// Writes and fsyncs (file and directory) one bucket's L0 file for
    /// `records`, without registering it. `None` when the bucket has no records.
    fn write_level0_file_for_records(&self, bucket: u32, records: &[KeyValueRecord], timestamp_ms: u128) -> Result<Option<WrittenLevel0File>> {
        let mut bucket_records = Vec::new();
        for record in records {
            let record_bucket = get_bucket_for_key(&record.key, self.config.num_buckets);
            if record_bucket == bucket {
                bucket_records.push(record.clone());
            }
        }

        if bucket_records.is_empty() {
            return Ok(None);
        }

        bucket_records.sort_by(|a, b| a.key.cmp(&b.key));
        let mut summary = L0Summary::of_sorted(bucket_records.iter().map(|r| (r.key.as_slice(), r.seq)));
        // Absolute frame offsets, sampled exactly as `write_merged_sstable` and
        // `load_metadata_from_file` do, so all three agree.
        let mut index = SparseIndex::new();
        let mut offset = SSTABLE_DATA_START;

        let level0_dir = Self::level0_bucket_dir_from(&self.base_path, bucket);
        let dir_is_new = !level0_dir.exists();
        std::fs::create_dir_all(&level0_dir)?;

        // Level 0 files are named by timestamp/version, not by a global sequence.
        let level0_path = level0_dir.join(format!("{}.dat", timestamp_ms));
        // Written under a temporary name and renamed once complete and fsynced, as
        // L1 is, so a `.dat` file is always whole. Written in place, a process
        // killed mid-flush left a `.dat` cut anywhere, even inside a frame, and the
        // next open failed on it (`UnexpectedEof`, or a frame longer than its file):
        // the database would not open again until the file was removed by hand.
        // Open deletes leftover temporaries (`load_existing_level0_files`).
        let tmp_path = level0_dir.join(format!("{timestamp_ms}{LEVEL0_TMP_SUFFIX}"));

        let write = || -> Result<()> {
            let file = std::fs::File::create(&tmp_path)?;
            let mut out = std::io::BufWriter::new(&file);
            out.write_all(&sstable_header_bytes())?;
            for (i, entry) in bucket_records.into_iter().enumerate() {
                if (i as u64).is_multiple_of(SparseIndex::SAMPLE_INTERVAL) {
                    index.push(entry.key.clone(), offset);
                }
                let key_prefix = key_prefix_of(&entry.key);
                let entry_obj = SStableEntry {
                    key: entry.key,
                    key_prefix,
                    value: entry.value,
                    tombstone: entry.tombstone,
                    seq: entry.seq,
                };
                let payload = encode_sstable_entry(&entry_obj)?;
                out.write_all(&(payload.len() as u32).to_le_bytes())?;
                out.write_all(&payload)?;
                offset += 4 + payload.len() as u64;
            }
            out.flush()?;
            drop(out);
            file.sync_all()?;
            std::fs::rename(&tmp_path, &level0_path)?;
            Ok(())
        };
        let written = write();
        if written.is_err() {
            let _ = std::fs::remove_file(&tmp_path);
        }
        written?;
        // Make the new file's directory entry durable too (the rename's, here). `sync_all` on the file
        // covers its contents, not its name: until the directory is fsynced a power
        // loss can drop the file entirely — after the flush observer has already
        // let its WAL entries be marked persisted, leaving nothing to replay. A
        // directory this call created needs its own entry made durable as well.
        File::open(&level0_dir)?.sync_all()?;
        if dir_is_new {
            let level0_root = Self::level0_dir_from(&self.base_path);
            File::open(&level0_root)?.sync_all()?;
            File::open(&self.base_path)?.sync_all()?;
        }
        if let Some(summary) = summary.as_mut() {
            summary.index = (!index.is_empty()).then_some(index);
        }

        Ok(Some(WrittenLevel0File {
            path: level0_path,
            created_at_ms: timestamp_ms,
            summary,
        }))
    }

    /// Read all entries from an existing SSTable file
    fn read_sstable_entries(&self, bucket: u32) -> Result<Vec<SStableEntry>> {
        let file = self.sstable_files[bucket as usize].read().clone();
        self.read_sstable_entries_from_file(&file)
    }

    /// Read SSTable entries from a specific file path.
    ///
    /// Called by the L0→L1 merge with a registered `L0FileEntry`'s path, so it uses
    /// the registered-file opener (see [`open_registered_l0`]).
    fn read_sstable_entries_from_path(&self, path: &Path) -> Result<Vec<SStableEntry>> {
        if !path.exists() {
            return Ok(Vec::new());
        }

        let mut entries = Vec::new();
        let mut file = open_registered_l0_for_scan(path)?;
        // Bound for `check_frame_len`: a frame can never be longer than its file.
        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);

        loop {
            let mut size_buf = [0u8; 4];
            match file.read_exact(&mut size_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(LSMError::Io(e)),
            }

            let size = u32::from_le_bytes(size_buf);
            check_frame_len(u64::from(size), file_len)?;
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
        file.write_all(&sstable_header_bytes())?;

        // Counts entry bytes only — it is returned as the merge's `bytes_merged`
        // metric, not as a file position. The sparse index adds `SSTABLE_DATA_START`
        // so its offsets stay absolute.
        let mut total_size = 0u64;
        let min_key = entries[0].key.clone();
        let max_key = entries[entries.len() - 1].key.clone();
        let mut index = SparseIndex::new();

        for (i, entry) in entries.iter().enumerate() {
            if (i as u64).is_multiple_of(SparseIndex::SAMPLE_INTERVAL) {
                index.push(entry.key.clone(), SSTABLE_DATA_START + total_size);
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
    fn merge_candidate(best: &mut Candidate, value: Option<u128>, seq: u64) {
        if best.is_none_or(|(_, b)| seq_newer(seq, b)) {
            *best = Some((value, seq));
        }
    }

    /// Like [`get`](Self::get) but also returns the stored sequence of the live
    /// value. Used by GC / journal replay to relocate a value's pointer while
    /// preserving its sequence, so the relocation neither loses to nor blocks a
    /// real write under highest-sequence-wins resolution.
    pub(crate) fn get_with_seq(&self, key: &[u8]) -> Result<Option<(u128, u64)>> {
        Ok(self.newest_copy(key)?.and_then(|(value, seq)| value.map(|ptr| (ptr, seq))))
    }

    /// The winning copy of `key` across every layer — `Some((Some(pointer), seq))`
    /// for a live value, `Some((None, seq))` when a **tombstone** wins, `None`
    /// when no layer holds the key at all. [`get_with_seq`](Self::get_with_seq)
    /// is this with tombstones read as absent; the write path needs the
    /// tombstone's seq to tell whether a write is newer than the delete.
    fn newest_copy(&self, key: &[u8]) -> Result<Candidate> {
        // Resolve by highest write sequence across ALL layers. Layer order is no
        // longer assumed to imply recency — a GC re-point can re-insert a value at
        // its old (low) sequence into a newer layer, above a deleting tombstone in
        // an older layer — so we cannot early-return at the first layer holding
        // the key. We inspect every layer and keep the newest-sequence entry; a
        // tombstone winning means the key is deleted. Layers are visited
        // newest-first so an exact sequence tie resolves to the newer copy.
        let mut best: Candidate = None;

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
                    return Ok(Some(((!tombstone).then_some(value), seq)));
                }
                Self::merge_candidate(&mut best, (!tombstone).then_some(value), seq);
            }
        }

        {
            let ro_memtables = self.read_only_memtables.read();
            for ro in ro_memtables.iter().rev() {
                ro.increment_reader_count();
                let hit = ro.find(key).map(|r| (r.value, r.seq, r.tombstone));
                ro.decrement_reader_count();
                if let Some((value, seq, tombstone)) = hit {
                    Self::merge_candidate(&mut best, (!tombstone).then_some(value), seq);
                }
            }
        }

        let bucket = get_bucket_for_key(key, self.config.num_buckets);
        match self.lookup_sstable(bucket, key, best.map(|(_, seq)| seq))? {
            SsLookup::Found(ptr, seq) => Self::merge_candidate(&mut best, Some(ptr), seq),
            SsLookup::Deleted(seq) => Self::merge_candidate(&mut best, None, seq),
            SsLookup::Missing => {}
        }

        Ok(best)
    }

    /// The pointer currently referenced for `key` (highest sequence across all
    /// layers), and whether a write carrying `new_seq` would become that key's
    /// winner under highest-sequence-wins resolution.
    ///
    /// The write path needs both, under the bucket write lock, to account value-log
    /// garbage correctly when same-key writes race: the winner displaces the existing
    /// record, but a **loser** (a lower-sequence write that arrived at the lock after a
    /// higher-sequence one) does not — its own freshly-appended record is the garbage
    /// instead. Deciding this from a pre-lock read (as the old code did) let two writers
    /// see the same old pointer and both mark it displaced, double-counting it while
    /// leaking the loser's record.
    ///
    /// A winning **tombstone** counts: the pointer is `None` (nothing to displace),
    /// but a write older than the delete still loses to it. Reading that case as
    /// "absent, so any write wins" made a lost put add a field-index row for a
    /// key that reads as deleted. `true` only when no layer holds the key at all,
    /// or the write is at least as new as whatever does.
    pub(crate) fn current_pointer_and_wins(&self, key: &[u8], new_seq: u64) -> Result<(Option<u128>, bool)> {
        match self.newest_copy(key)? {
            Some((ptr, existing_seq)) => Ok((ptr, seq_newer_or_eq(new_seq, existing_seq))),
            None => Ok((None, true)),
        }
    }

    /// Insert/update a value carrying an explicit global write sequence, with
    /// highest-sequence-wins conflict resolution (see
    /// [`SkipList::try_insert_with_seq`]). This is the production write path; it
    /// makes the in-memory winner for racing same-key writes match the winner
    /// recovery would pick (recovery replays in sequence order).
    pub(crate) fn insert_with_seq(&self, key: &[u8], value: u128, seq: u64) -> Result<()> {
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
    pub(crate) fn delete_with_seq(&self, key: &[u8], seq: u64) -> Result<()> {
        {
            let memtable = self.memtable.read();
            if memtable.should_flush(self.config.compaction_threshold_percent) {
                drop(memtable);
                self.flush_memtable()?;
            }
        }

        let mut memtable = self.memtable.write();
        // Record the tombstone directly — creating the node if the key is not
        // currently in the memtable, so the delete still shadows lower layers.
        // Honours the sequence guard: a newer write to this key is left untouched.
        match memtable.skip_list.insert_tombstone_with_seq(key, seq) {
            Ok(()) => Ok(()),
            Err(super::skip_list::skip_list::InsertError::CapacityExceeded) => Err(LSMError::CapacityExceeded),
        }
    }

    /// Fold a sequence into the lower-layer bound (serial-max), used whenever
    /// entries move from the active memtable into a lower layer.
    fn note_lower_seq(&self, seq: u64) {
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
        self.flush_to_level0(false)
    }

    /// As [`Self::flush_memtable_to_level0`], but seals the active memtable
    /// even when it is empty, so the observers see a seal and a flush. The
    /// vector index WAL needs that to retire a namespace whose records are all
    /// on disk already (or were never applied) while its memtable is empty.
    pub(crate) fn seal_and_flush_to_level0(&self) -> Result<()> {
        self.flush_to_level0(true)
    }

    fn flush_to_level0(&self, seal_empty: bool) -> Result<()> {
        {
            let memtable = self.memtable.read();
            let has_data = memtable.skip_list.number_of_live_nodes() > 0 || memtable.skip_list.number_of_tombstone_nodes() > 0;
            drop(memtable);
            if has_data || seal_empty {
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

    /// Tri-state lookup of a key in the SSTable hierarchy (Level 0 then Level 1),
    /// resolved by highest sequence.
    ///
    /// `floor` is the sequence of the best candidate the caller already holds from
    /// a newer layer (the memtables), if any. A file whose newest entry is `<=` the
    /// best seen so far cannot win — an exact tie stays with the newer layer — so
    /// it is skipped unread. The result is still only this layer's best hit; the
    /// caller merges it with its own candidate.
    fn lookup_sstable(&self, bucket: u32, key: &[u8], floor: Option<u64>) -> Result<SsLookup> {
        // L0 is newer than L1, so an exact sequence tie resolves to L0 (take L1
        // only if strictly newer). Layer order is not assumed to imply recency — a
        // GC re-point or an out-of-order flush can leave a lower-sequence value
        // above a deleting tombstone.
        let l0 = self.search_level0_files(bucket, key, floor)?;
        let floor = match (floor, l0.seq()) {
            (Some(f), Some(s)) => Some(if seq_newer(s, f) { s } else { f }),
            (f, s) => f.or(s),
        };
        let l1 = self.lookup_level1(bucket, key, floor)?;
        Ok(match (l0.seq(), l1.seq()) {
            (Some(s0), Some(s1)) if seq_newer(s1, s0) => l1,
            (Some(_), _) => l0,
            (None, _) => l1,
        })
    }

    /// Look the key up in this bucket's Level-1 file, using the max-seq, min/max,
    /// bloom-filter and sparse-index fast-rejects before any linear scan.
    fn lookup_level1(&self, bucket: u32, key: &[u8], floor: Option<u64>) -> Result<SsLookup> {
        // L1 early-outs from its metadata: the key falls outside the file's
        // [min_key, max_key], or a newer layer already holds a copy at least as new
        // as anything in the file. The metadata is updated after the file swap and
        // before the merged L0 files are marked obsolete, so a reader that sees stale
        // metadata still sees those L0 files (and has folded their seqs into
        // `floor`) — this can never turn a live key into a false `Missing`. Empty
        // metadata (no L1 file yet, or it failed to load) skips the optimisation.
        {
            let metadata = self.sstable_metadata[bucket as usize].read();
            if let Some(meta) = metadata.first() {
                if key < meta.min_key.as_slice() || key > meta.max_key.as_slice() {
                    return Ok(SsLookup::Missing);
                }
                if let (Some(floor), Some(max_seq)) = (floor, meta.max_seq)
                    && seq_newer_or_eq(floor, max_seq)
                {
                    if let Some(m) = self.metrics() {
                        Metrics::bump(&m.seq_prunes);
                    }
                    return Ok(SsLookup::Missing);
                }
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
    fn lookup_in_sstable_file_path(&self, path: std::path::PathBuf, key: &[u8], start_offset: u64) -> Result<SsLookup> {
        // The hot path: `search_level0_files` calls this for every L0 file its
        // summary cannot rule out. Reads below are positional, so no cursor seek,
        // and the file is registered, so no header re-validation.
        let file = open_registered_l0(&path)?;
        let (hit, hint_rejected) = self.lookup_in_sstable_file_hinted(&file, key, start_offset)?;
        // An L0 file is never rewritten, so its sparse-index hint has no
        // legitimate way to go stale: a rejection means the index's offsets
        // disagree with the file. Release builds survive it with a full scan (and
        // count it in `sparse_hint_rejects`); debug builds fail, so every test
        // that reaches an L0 lookup guards the offsets, not just the dedicated one.
        debug_assert!(
            !hint_rejected,
            "L0 sparse-index hint {start_offset} is not a valid scan start in {}",
            path.display()
        );
        Ok(hit)
    }

    /// Load metadata from an existing SSTable file by scanning it, and build the
    /// L1 Bloom filter from the same pass (no extra I/O). Returns `None` for the
    /// bloom when the file has no entries.
    fn load_metadata_from_file(path: &Path) -> Result<LoadedSstable> {
        let mut file = open_sstable_for_scan(path)?;
        // Bound for `check_frame_len`: a frame can never be longer than its file.
        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);

        let mut min_key: Option<Vec<u8>> = None;
        let mut max_key: Option<Vec<u8>> = None;
        let mut entry_count = 0u64;
        let mut data_size = 0u64;
        let mut keys: Vec<Vec<u8>> = Vec::new();
        let mut index = SparseIndex::new();
        let mut max_seq: Option<u64> = None;

        loop {
            // `data_size` counts entry bytes only; the frame's absolute file
            // offset is that plus the header the scan started after.
            let frame_offset = SSTABLE_DATA_START + data_size;

            let mut size_buf = [0u8; 4];
            match file.read_exact(&mut size_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(LSMError::Io(e)),
            }

            let size = u32::from_le_bytes(size_buf);
            check_frame_len(u64::from(size), file_len)?;
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
                min_key,
                max_key,
                entry_count,
                max_seq,
            },
            bloom,
            index,
            max_seq,
        ))
    }

    /// Search the Level 0 files for a key and return the hit with the newest
    /// sequence (a value or a tombstone), or [`SsLookup::Missing`] if no L0 file
    /// mentions it.
    ///
    /// Resolution is by sequence, not file recency: a GC re-point can land a
    /// low-sequence value in a newer L0 file while the deleting tombstone sits in
    /// an older one, so no hit settles the key on its own. Files are visited
    /// newest-first only so an exact sequence tie resolves to the newer file (the
    /// relocation, whose old copy is stale).
    ///
    /// A file is skipped unread when its summary proves it cannot matter: the key
    /// is outside its range or bloom-negative, or its newest entry is `<=` the best
    /// sequence seen so far (`floor` from the memtables, or an earlier file here).
    fn search_level0_files(&self, bucket: u32, key: &[u8], floor: Option<u64>) -> Result<SsLookup> {
        let mut entries = self.level0_entries_snapshot(bucket);
        if entries.is_empty() {
            return Ok(SsLookup::Missing);
        }

        entries.sort_by_key(|b| std::cmp::Reverse(b.created_at_ms));

        let mut best = SsLookup::Missing;
        let mut probed = false;
        for entry in entries {
            if entry.is_obsolete() {
                continue;
            }
            if !entry.may_contain(key) {
                if let Some(m) = self.metrics() {
                    Metrics::bump(&m.l0_bloom_rejects);
                }
                continue;
            }
            let best_seq = match (floor, best.seq()) {
                (Some(f), Some(b)) => Some(if seq_newer(b, f) { b } else { f }),
                (f, b) => f.or(b),
            };
            if best_seq.is_some_and(|s| entry.cannot_beat(s)) {
                if let Some(m) = self.metrics() {
                    Metrics::bump(&m.seq_prunes);
                }
                continue;
            }
            if !probed {
                probed = true;
                if let Some(m) = self.metrics() {
                    Metrics::bump(&m.l0_probes);
                }
            }
            let _guard = L0ReadGuard::new(Arc::clone(&entry));
            let hit = self.lookup_in_sstable_file_path(entry.path.clone(), key, entry.scan_start(key))?;
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
        entries.sort_by_key(|a| a.created_at_ms);

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

                // Publish with a SINGLE atomic rename. `rename(2)` replaces the
                // destination atomically, so the bucket's L1 path always names a
                // complete SSTable — the old one before this call, the new one after,
                // never nothing.
                //
                // This deliberately does NOT move the old file aside first. Doing so
                // (`rename(level1 -> .old)` then `rename(.tmp -> level1)`) opened a
                // crash window in which the L1 path did not exist at all: `LSMTree::open`
                // would then `create(true)` a 0-byte file, `file_len == 0` would
                // short-circuit the metadata load into "empty bucket", and
                // `cleanup_old_files_on_startup` would delete the `.old` holding the only
                // surviving copy. Every key that lived only in L1 was lost — silently,
                // and with both a complete `.old` and a complete `.tmp` sitting on disk
                // unread. Regression test: `test_crash_between_l1_publish_steps_keeps_l1`.
                //
                // Readers holding the previous `Arc<File>` keep reading the old inode
                // until they drop it, which is unchanged from before.
                std::fs::rename(&temp_path, &level1_path)?;

                // Make the rename itself durable: without an fsync of the containing
                // directory the swap can be lost on power failure, silently reverting
                // to the pre-merge L1. That is consistent (the L0 inputs are still
                // there, so it only costs the merge work) but it is not what the code
                // above just reported as done.
                if let Ok(dir) = File::open(Self::level1_dir_from(&self.base_path)) {
                    let _ = dir.sync_all();
                }

                let new_file = OpenOptions::new().read(true).write(true).open(&level1_path)?;
                *file_guard = Arc::new(new_file);
            }

            let mut metadata_guard = self.sstable_metadata[bucket_idx].write();
            metadata_guard.clear();
            metadata_guard.push(SStableMetadata {
                min_key,
                max_key,
                entry_count,
                max_seq: merged_entries.iter().map(|e| e.seq).reduce(|a, b| if seq_newer(b, a) { b } else { a }),
            });

            // Rebuild the L1 bloom from the merged entries (in memory, no extra
            // I/O). Replaced after the file swap so it always matches the file
            // now installed.
            let bloom = BloomFilter::build(merged_entries.iter().map(|e| e.key.as_slice()), merged_entries.len());
            *self.sstable_blooms[bucket_idx].write() = Some(Arc::new(bloom));
            // The sparse index was built from the byte offsets as the file was
            // written, so it matches the file just installed.
            *self.sstable_indexes[bucket_idx].write() = (!index.is_empty()).then(|| Arc::new(index));

            // No `.old` to clean up — the publish above is a single atomic rename and
            // never creates one. `cleanup_old_files_on_startup` still sweeps any left
            // by a database written before that change.
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

    /// Whether `bucket`'s Level-0 files should be merged into its Level 1 now.
    ///
    /// A merge rewrites the bucket's whole Level-1 file, so merging whatever L0
    /// holds at every opportunity costs the size of Level 1 per flush. Callers
    /// that flush small memtables often (the vector worker's durability barrier,
    /// one flush per batch) then rewrote megabytes to add a dozen entries: at 50
    /// requests/s the M5 benchmark measured 10 flushes and 16 merges a second,
    /// about 2 MB rewritten per document and 93% of all disk writes.
    ///
    /// So a bucket waits until its L0 files hold [`L0_COMPACTION_RATIO_PERCENT`]
    /// of Level 1's entries, which bounds the rewrite to about 100 / that times
    /// what was flushed, or until there are [`L0_COMPACTION_MAX_FILES`] of them,
    /// which bounds what a point read may have to open. An L0 file without a
    /// summary counts as due: its size is unknown.
    pub(crate) fn compaction_due(&self, bucket: u32) -> bool {
        let files = self.level0_files[bucket as usize].read();
        let live: Vec<&Arc<L0FileEntry>> = files.iter().filter(|f| !f.is_obsolete()).collect();
        if live.is_empty() {
            return false;
        }
        if live.len() >= L0_COMPACTION_MAX_FILES {
            return true;
        }
        let mut l0_entries = 0u64;
        for f in &live {
            match &f.summary {
                Some(s) => l0_entries += s.entries,
                None => return true,
            }
        }
        let l1_entries: u64 = self.sstable_metadata[bucket as usize].read().iter().map(|m| m.entry_count).sum();
        l0_entries * 100 >= l1_entries * L0_COMPACTION_RATIO_PERCENT
    }

    /// Whether [`compact_due`](Self::compact_due) has anything to do: a read-only
    /// memtable to flush or release (a flushed one is held until a tick drops
    /// it), or a bucket whose merge is due.
    pub(crate) fn has_due_compaction_work(&self) -> bool {
        !self.read_only_memtables.read().is_empty() || (0..self.config.num_buckets as u32).any(|b| self.compaction_due(b))
    }

    /// The background worker's compaction: flush every read-only memtable to
    /// Level 0 (they pin the WAL), then merge only the buckets whose merge is
    /// [due](Self::compaction_due). [`compact_all`](Self::compact_all) merges
    /// every bucket with any L0 file, for shutdown and explicit requests.
    pub(crate) fn compact_due(&self) -> Result<()> {
        self.cleanup_pending_memtables();
        self.cleanup_obsolete_level0_files();

        let ro_snapshot = self.read_only_memtables.read().clone();
        let mut changed = false;
        for ro in ro_snapshot.iter() {
            changed |= !ro.is_flushed_to_level0();
            self.flush_ro_memtable_to_level0(ro)?;
        }

        for bucket in 0u32..self.config.num_buckets as u32 {
            if !self.compaction_due(bucket) {
                continue;
            }
            changed = true;
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
        if changed {
            self.write_manifest_snapshot()?;
        }
        Ok(())
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
    pub(crate) fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, u128, u64)>> {
        // Layers are folded oldest→newest by `fold_newest`: each key resolves to its
        // highest-seq copy, exactly as `get_with_seq` resolves it, so a tombstone
        // shadows only copies older than itself.
        //
        // What changes for concurrency is the *capture* order, which is the
        // reverse — newest layer first. Flush/compaction relocate a key
        // active→RO→L0→L1; capturing in that same direction guarantees the key is
        // seen in the newer layer before it can move into an older layer we have
        // not captured yet, so it is never absent from every captured layer at
        // once. We capture cheap, frozen handles (active+RO snapshot, L0 read
        // guards, L1 `Arc<File>`) up front and do the I/O afterwards, then merge.
        let mut all_entries = ScanMap::new();

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
                active.push((key.to_vec(), (!tombstone).then_some(value), seq));
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
                            for (key, value, seq) in entries {
                                fold_newest(&mut all_entries, key, value, seq);
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

        // Read-only memtables oldest-first.
        for ro_memtable in ro_snapshot.iter() {
            let records = ro_memtable.records();
            for record in records.iter() {
                if record.key.starts_with(prefix) {
                    fold_newest(
                        &mut all_entries,
                        record.key.clone(),
                        (!record.tombstone).then_some(record.value),
                        record.seq,
                    );
                }
            }
        }

        // Active memtable (newest layer).
        for (key, value, seq) in active_snapshot {
            fold_newest(&mut all_entries, key, value, seq);
        }

        Ok(all_entries
            .into_iter()
            .filter_map(|(key, (val, seq))| val.map(|v| (key, v, seq)))
            .collect())
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
            let val = (!archived.tombstone).then(|| archived.value.to_native());
            out.push((archived.key.as_slice().to_vec(), val, archived.seq.to_native()));
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
        let offset = if start_offset != 0 && Self::valid_scan_start(file, start_offset, file_len, start) {
            start_offset
        } else {
            SSTABLE_DATA_START
        };

        let mut live_seen = 0usize;
        let mut frames = FrameReader::new(file, offset);
        while let Some(body) = frames.next_frame(file_len)? {
            let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(body)?) };
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

    /// Visit every entry of `bucket`'s L1 file whose key starts with one of
    /// `sorted_ids` (4-byte big-endian prefixes, ascending), in key order, with one
    /// forward cursor over the file.
    ///
    /// Each prefix's seek point comes from the sparse index, all read under one
    /// lock before any I/O, and the cursor jumps only when that point lies past it.
    /// When the prefixes are close together, as when a search probes most
    /// postings, the jumps vanish and this is one sequential read. Calling
    /// [`for_each_l1_entry_in_range`](Self::for_each_l1_entry_in_range) per prefix
    /// started a fresh reader (and an `fstat` and a seek check) for every prefix,
    /// which dominated a search probing hundreds of postings.
    ///
    /// Keys shorter than 4 bytes have no prefix id and are skipped, as the L0 and
    /// memtable passes of [`scan_prefixes`](Self::scan_prefixes) skip them.
    fn for_each_l1_entry_in_prefixes(&self, bucket: usize, file: &File, sorted_ids: &[u32], visit: impl FnMut(&ArchivedSStableEntry)) -> Result<()> {
        let starts: Vec<u64> = {
            let index = self.sstable_indexes[bucket].read();
            sorted_ids
                .iter()
                .map(|id| index.as_ref().map(|i| i.block_start(&id.to_be_bytes())).unwrap_or(0))
                .collect()
        };
        Self::for_each_entry_in_prefixes(file, &starts, sorted_ids, visit).map(|_| ())
    }

    /// The cursor behind [`for_each_l1_entry_in_prefixes`](Self::for_each_l1_entry_in_prefixes),
    /// for any key-sorted SSTable: `starts[i]` is a sparse-index hint for
    /// `sorted_ids[i]` (0 for none), used only after
    /// [`valid_scan_start`](Self::valid_scan_start) accepts it. A file without an
    /// index is read from its head to its last requested prefix. Returns the bytes
    /// read (tests use it to check that the cursor seeks).
    fn for_each_entry_in_prefixes(file: &File, starts: &[u64], sorted_ids: &[u32], mut visit: impl FnMut(&ArchivedSStableEntry)) -> Result<u64> {
        debug_assert!(sorted_ids.windows(2).all(|w| w[0] < w[1]), "prefixes must be sorted and distinct");
        debug_assert_eq!(starts.len(), sorted_ids.len());
        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
        let mut frames = FrameReader::new(file, SSTABLE_DATA_START);
        let mut next = 0usize;
        'prefixes: while next < sorted_ids.len() {
            let wanted = sorted_ids[next];
            if starts[next] > frames.offset() && Self::valid_scan_start(file, starts[next], file_len, &wanted.to_be_bytes()) {
                frames.seek(starts[next]);
            }
            while let Some(body) = frames.next_frame(file_len)? {
                let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(body)?) };
                let Some(prefix) = archived.key.as_slice().get(..4) else {
                    continue;
                };
                let id = u32::from_be_bytes(prefix.try_into().unwrap());
                if id < wanted {
                    continue;
                }
                while next < sorted_ids.len() && sorted_ids[next] < id {
                    next += 1;
                }
                if next == sorted_ids.len() {
                    break 'prefixes;
                }
                if sorted_ids[next] == id {
                    visit(archived);
                    continue;
                }
                // `id` lies between two requested prefixes: jump to the next one if
                // its block starts past the cursor, else keep reading.
                continue 'prefixes;
            }
            break;
        }
        Ok(frames.bytes_read)
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

        // For a prefix of at most 8 bytes, compare the entry's stored 8-byte
        // `key_prefix` instead of the key. That field is the key ZERO-PADDED to 8
        // bytes, so a key shorter than the prefix must be rejected first: key "ab"
        // pads to "ab\0…", which would otherwise match prefix "ab\0".
        let matches_prefix = |archived: &ArchivedSStableEntry, key_slice: &[u8]| -> bool {
            if key_slice.len() < prefix.len() {
                false
            } else if prefix.len() <= 8 {
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
        l0_guards.sort_by_key(|a| a.entry.created_at_ms);

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
            let mut file = open_registered_l0_for_scan(&guard.entry.path)?;
            // Bound for `check_frame_len`: a frame can never be longer than its file.
            let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
            loop {
                let mut size_buf = [0u8; 4];
                match file.read_exact(&mut size_buf) {
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(LSMError::Io(e)),
                }

                let size = u32::from_le_bytes(size_buf) as usize;
                check_frame_len(size as u64, file_len)?;
                entry_bytes.resize(size, 0);
                file.read_exact(&mut entry_bytes[..size])?;

                let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes[..size])?) };

                let key_slice = archived.key.as_slice();
                if matches_prefix(archived, key_slice) {
                    let val = (!archived.tombstone).then(|| archived.value.to_native());
                    results.push((key_slice.to_vec(), val, archived.seq.to_native()));
                }
            }
        }

        Ok(results)
    }

    /// Scan multiple 4-byte BE u32 prefixes in one call, returning live
    /// `(key, pointer)` pairs in ascending key order.
    ///
    /// Each layer is read once for the whole prefix set rather than once per prefix:
    ///
    /// * **L1** — the sorted ids are split into contiguous groups, one scoped thread
    ///   per group (as many as the buckets whose key span overlaps them). Each
    ///   thread seeks to every id of its group through each bucket's sparse index
    ///   ([`for_each_l1_entry_in_prefixes`](Self::for_each_l1_entry_in_prefixes))
    ///   and merges the buckets' key-sorted runs itself, so the groups' outputs
    ///   follow one another in key order.
    /// * **L0** — one scoped thread per bucket holding L0 files; a file whose key
    ///   range misses every id is skipped, the rest are seeked per id through
    ///   their sparse index and read only within the requested ranges.
    /// * **Memtables** — the active one is range-iterated per prefix; read-only ones
    ///   are filtered.
    ///
    /// The layers are resolved by [`merge_scan_layers`]: L1 entries are sorted and
    /// deduplicated in bulk, then a sorted merge lets the newer layers shadow them.
    /// The newer layers go through a `BTreeMap` first, which is cheap because they are
    /// usually small; L1 is not, since a per-entry map insert was most of the cost of
    /// a large scan.
    ///
    /// The caller maps entries back to their originating prefix via the first 4
    /// bytes of each key.
    ///
    /// Same pointer-resolution invariant as [`scan_prefix`](Self::scan_prefix):
    /// returned pointers are LSM-complete but must be resolved against the value log
    /// only inside the value-log generation bracket / GC bucket lock.
    pub(crate) fn scan_prefixes(&self, prefix_ids: &std::collections::HashSet<u32>) -> Result<PrefixScanKeys> {
        if prefix_ids.is_empty() {
            return Ok(PrefixScanKeys::default());
        }

        // Layers resolve by highest seq (`fold_newest`, oldest→newest: L1 → L0 →
        // ro_memtables → active). The *capture* order is newest-first so a
        // concurrent flush/compaction cannot drop a live key — see `scan_prefix`.
        //
        // L1 — the bulk of a large scan — is gathered into a Vec and resolved by
        // `resolve_scan_layers`, not inserted into the map: a per-entry BTreeMap
        // insert of an owned key was ~8 ms of a ~10 ms scan over 87k entries, while
        // the parallel L1 reads were ~1 ms. The newer layers (usually small) still
        // merge through `newer` by `fold_newest`. Each worker packs its L1 keys into
        // one buffer (`l1_keys`) instead of allocating one per entry.
        let mut l1_keys: Vec<Vec<u8>> = Vec::new();
        let mut l1_entries: Vec<PackedScanEntry> = Vec::new();
        let mut newer = ScanMap::new();

        // Memtable layers (newest) captured first, under one `memtable.read()`
        // guard so a flush cannot split a key between active and RO.
        let (active_snapshot, ro_snapshot) = {
            let memtable = self.memtable.read();
            let mut active = Vec::new();
            for prefix_id in prefix_ids {
                let prefix = prefix_id.to_be_bytes();
                for (key, value, seq, tombstone) in memtable.skip_list.iter_raw_from(&prefix) {
                    if key.len() < 4 || u32::from_be_bytes(key[..4].try_into().unwrap()) != *prefix_id {
                        break;
                    }
                    active.push((key.to_vec(), (!tombstone).then_some(value), seq));
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

        // ── 1. L1 SSTables (oldest layer) — postings split into groups, read in
        //    parallel. Each worker reads its group's postings from every overlapping
        //    bucket and merges those runs itself; the groups are ascending, disjoint
        //    ranges of prefix ids, so their merged outputs simply follow one another
        //    in key order. One worker per bucket merging the buckets' runs afterwards
        //    on one thread was about half a 1,024-posting scan's time.
        let l1_buckets: Vec<(usize, &Arc<File>)> = l1_files
            .iter()
            .enumerate()
            .filter(|(bucket, _)| self.l1_overlaps(*bucket, &span_start, span_end))
            .collect();
        let l1_buckets = &l1_buckets;
        if !l1_buckets.is_empty() {
            let group_len = sorted_ids.len().div_ceil(l1_buckets.len());
            let mut first_err: Option<LSMError> = None;
            std::thread::scope(|s| {
                let handles: Vec<_> = sorted_ids
                    .chunks(group_len)
                    .map(|group| {
                        s.spawn(move || -> Result<(Vec<u8>, Vec<PackedScanEntry>)> {
                            let mut keys = Vec::new();
                            let mut entries = Vec::new();
                            for &(bucket, file) in l1_buckets {
                                self.for_each_l1_entry_in_prefixes(bucket, file, group, |archived| {
                                    let val = (!archived.tombstone).then(|| archived.value.to_native());
                                    let key_start = keys.len();
                                    keys.extend_from_slice(archived.key.as_slice());
                                    entries.push((0, key_start, keys.len(), val, archived.seq.to_native()));
                                })?;
                            }
                            let merged = merge_l1_runs(&entries, |e: &PackedScanEntry| &keys[e.1..e.2]);
                            Ok((keys, merged))
                        })
                    })
                    .collect();
                for handle in handles {
                    match handle
                        .join()
                        .unwrap_or_else(|_| Err(LSMError::Io(std::io::Error::other("thread panicked"))))
                    {
                        Ok((keys, entries)) => {
                            let buffer = l1_keys.len();
                            l1_keys.push(keys);
                            l1_entries.extend(entries.into_iter().map(|(_, start, end, val, seq)| (buffer, start, end, val, seq)));
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
        //    bucket so an exact seq tie goes to the newer file. Only buckets that hold
        //    a captured L0 file spawn a worker. A file whose key range misses every
        //    probed prefix is skipped, and the rest are read with one cursor that
        //    seeks per prefix through the file's sparse index, as L1 is: L0 can hold
        //    a tenth of L1's entries before it is merged, and reading every file
        //    whole cost every search that share of L1 again.
        {
            let mut first_err: Option<LSMError> = None;
            std::thread::scope(|s| {
                let handles: Vec<_> = l0_guards
                    .iter()
                    .filter(|guards| !guards.is_empty())
                    .map(|guards| {
                        s.spawn(move || -> Result<Vec<ScanEntry>> {
                            let mut bucket_entries = Vec::new();
                            for guard in guards {
                                let summary = guard.entry.summary.as_ref();
                                if summary.is_some_and(|s| {
                                    s.max_key.as_slice() < span_start.as_slice() || span_end.is_some_and(|e| s.min_key.as_slice() >= e)
                                }) {
                                    continue;
                                }
                                let file = open_registered_l0(&guard.entry.path)?;
                                let starts: Vec<u64> = sorted_ids.iter().map(|id| guard.entry.scan_start(&id.to_be_bytes())).collect();
                                Self::for_each_entry_in_prefixes(&file, &starts, sorted_ids, |archived| {
                                    let val = (!archived.tombstone).then(|| archived.value.to_native());
                                    bucket_entries.push((archived.key.as_slice().to_vec(), val, archived.seq.to_native()));
                                })?;
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
                            for (k, v, seq) in entries {
                                fold_newest(&mut newer, k, v, seq);
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

        // ── 3. Read-only memtables oldest-first ───────────────────────────────
        for ro_memtable in ro_snapshot.iter() {
            for record in ro_memtable.records().iter() {
                if record.key.len() >= 4 {
                    let prefix_id = u32::from_be_bytes(record.key[..4].try_into().unwrap());
                    if prefix_ids.contains(&prefix_id) {
                        fold_newest(&mut newer, record.key.clone(), (!record.tombstone).then_some(record.value), record.seq);
                    }
                }
            }
        }

        // ── 4. Active memtable (newest layer) ─────────────────────────────────
        for (key, value, seq) in active_snapshot {
            fold_newest(&mut newer, key, value, seq);
        }

        Ok(resolve_scan_layers(&l1_keys, l1_entries, newer))
    }

    /// Fetch multiple keys in a **single pass per bucket**.
    ///
    /// Where `get` × N keys would scan every Level-0 file N times, this groups the
    /// keys by hash bucket and makes one forward pass over each of the bucket's
    /// L0 files and its L1 file, resolving all of that bucket's keys together.
    /// The pass reads only near the keys: it jumps between them with the
    /// sparse indexes (`scan_sorted_candidates`), so a query page of a few keys
    /// reads a few KiB per file and a large batch one sequential read. It used to
    /// read the whole L1 file into memory for any batch, which made every
    /// document-store query page cost 1-14 ms on a 1M-document store (0.3 ms
    /// after the change).
    ///
    /// Resolution is identical to [`get_with_seq`](Self::get_with_seq): the
    /// highest-`seq` copy across **every** layer wins (a winning tombstone ⇒
    /// `None`), with the same active-memtable fast path. It must never become
    /// first-hit-by-layer: value-log GC decides liveness with this method, so a stale
    /// lower-seq tombstone in a newer layer answering "deleted" for a live key would
    /// make GC drop that key's only record (see
    /// `test_get_multiple_resolves_by_seq_not_layer_order`).
    ///
    /// Returns one `Option<(pointer, seq)>` per input key, in input order; a key
    /// given more than once resolves at every position. `None` means the key does
    /// not exist or was deleted.
    pub(crate) fn get_multiple(&self, keys: &[Vec<u8>]) -> Result<BatchLookup> {
        Ok(self.get_multiple_counted(keys)?.0)
    }

    /// [`get_multiple`](Self::get_multiple), also returning the SSTable bytes
    /// read (for tests).
    fn get_multiple_counted(&self, keys: &[Vec<u8>]) -> Result<(BatchLookup, u64)> {
        let n = keys.len();
        let mut sstable_bytes = 0u64;
        // Running winner per input position, folded newest layer first through
        // `merge_candidate` so an exact sequence tie stays with the newer layer.
        let mut best: Vec<Candidate> = vec![None; n];
        // Positions the active-memtable fast path settled: nothing below can beat them.
        let mut settled: Vec<bool> = vec![false; n];

        // ── 1. Active memtable ────────────────────────────────────────────────
        {
            let memtable = self.memtable.read();
            // Loaded under the memtable lock, as in `get_with_seq`: a flush raises
            // the bound while holding the write lock, so a bound read before taking
            // the lock could be too low for the memtable we then read.
            let max_lower_seq = self.max_lower_seq.load(Ordering::Relaxed);
            for i in 0..n {
                if let Some((value, seq, tombstone)) = memtable.skip_list.entry(&keys[i]) {
                    Self::merge_candidate(&mut best[i], (!tombstone).then_some(value), seq);
                    settled[i] = seq_newer_or_eq(seq, max_lower_seq);
                }
            }
        }

        // ── 2. Read-only memtables, newest first ─────────────────────────────
        {
            let ro_memtables = self.read_only_memtables.read();
            for ro in ro_memtables.iter().rev() {
                for i in 0..n {
                    if settled[i] {
                        continue;
                    }
                    if let Some(record) = ro.find(&keys[i]) {
                        Self::merge_candidate(&mut best[i], (!record.tombstone).then_some(record.value), record.seq);
                    }
                }
            }
        }

        // ── 3. Group unsettled keys by SSTable bucket, one slot per distinct key ─
        let mut pending_by_bucket: Vec<BatchSlots> = (0..self.config.num_buckets).map(|_| std::collections::HashMap::new()).collect();
        for i in 0..n {
            if !settled[i] {
                let bucket = get_bucket_for_key(&keys[i], self.config.num_buckets) as usize;
                // Every position of a duplicated key carries the same memtable winner.
                pending_by_bucket[bucket].entry(keys[i].as_slice()).or_insert(best[i]);
            }
        }

        // ── 4. One L0+L1 pass per bucket in parallel. Buckets are hash-disjoint,
        //    so each thread owns its bucket's slots exclusively.
        {
            let mut first_err: Option<LSMError> = None;
            std::thread::scope(|s| {
                let handles: Vec<_> = pending_by_bucket
                    .iter_mut()
                    .enumerate()
                    .filter(|(_, pending)| !pending.is_empty())
                    .map(|(bucket, pending)| s.spawn(move || self.resolve_bucket_batch(bucket, pending)))
                    .collect();
                for handle in handles {
                    let outcome = handle
                        .join()
                        .unwrap_or_else(|_| Err(LSMError::Io(std::io::Error::other("thread panicked"))));
                    match outcome {
                        Ok(read) => sstable_bytes += read,
                        Err(e) if first_err.is_none() => first_err = Some(e),
                        Err(_) => {}
                    }
                }
            });
            if let Some(e) = first_err {
                return Err(e);
            }
        }

        let results = (0..n)
            .map(|i| {
                let winner = if settled[i] {
                    best[i]
                } else {
                    let bucket = get_bucket_for_key(&keys[i], self.config.num_buckets) as usize;
                    pending_by_bucket[bucket].get(keys[i].as_slice()).copied().flatten()
                };
                winner.and_then(|(value, seq)| value.map(|ptr| (ptr, seq)))
            })
            .collect();
        Ok((results, sstable_bytes))
    }

    /// The SSTable half of [`get_multiple`](Self::get_multiple) for one bucket: fold
    /// every L0 file's and then the L1 file's copy of each pending key into its slot,
    /// highest `seq` winning. Each slot arrives holding the memtable winner (if any).
    fn resolve_bucket_batch(&self, bucket: usize, pending: &mut BatchSlots<'_>) -> Result<u64> {
        let mut bytes_read = 0u64;
        let fold = |pending: &mut BatchSlots<'_>, archived: &ArchivedSStableEntry| {
            if let Some(slot) = pending.get_mut(archived.key.as_slice()) {
                let value = (!archived.tombstone).then(|| archived.value.to_native());
                Self::merge_candidate(slot, value, archived.seq.to_native());
            }
        };

        // Level 0: EVERY captured file, newest first. A hit in one file does not
        // settle the key — an older file (or L1) can hold a higher seq, because
        // flush order is not seq order. The guards are held through the L1 read
        // below, so a key a concurrent merge moves into L1 is still in a captured,
        // already-scanned L0 file.
        let mut l0_guards: Vec<L0ReadGuard> = self
            .level0_entries_snapshot(bucket as u32)
            .iter()
            .filter(|e| !e.is_obsolete())
            .map(|e| L0ReadGuard::new(Arc::clone(e)))
            .collect();
        l0_guards.sort_by_key(|b| std::cmp::Reverse(b.entry.created_at_ms));
        for guard in &l0_guards {
            // Read the file only if some pending key may be in it AND could still
            // be beaten by it — the same exact rejects as `search_level0_files`.
            let entry = &guard.entry;
            let mut wanted: Vec<&[u8]> = pending
                .iter()
                .filter(|(key, slot)| entry.may_contain(key) && !slot.is_some_and(|(_, seq)| entry.cannot_beat(seq)))
                .map(|(key, _)| *key)
                .collect();
            if wanted.is_empty() {
                continue;
            }
            wanted.sort_unstable();
            let wanted: Vec<(&[u8], u64)> = wanted.into_iter().map(|k| (k, entry.scan_start(k))).collect();
            let file = open_registered_l0_for_scan(&entry.path)?;
            let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
            let (read, rejected) = self.scan_sorted_candidates(&file, file_len, &wanted, |a| fold(pending, a))?;
            // An L0 file is never rewritten, so its hints cannot go stale.
            debug_assert!(
                !rejected,
                "an L0 sparse-index hint was rejected: the index offsets disagree with the file"
            );
            bytes_read += read;
        }

        // L1 pre-filter: skip the file when no pending key can be in it.
        // Same exact fast-rejects as `lookup_level1` (min/max is exact, a bloom
        // negative is exact), and only applied when the metadata/bloom are present —
        // absent metadata reads the file rather than risk skipping a live key.
        //
        // Race-safety mirrors the scan paths: the L0 guards above are still held, and
        // `merge_level0_to_level1` installs the new L1 file + metadata/bloom BEFORE
        // marking the old L0 files obsolete, so a key stale L1 metadata would reject
        // is still in a captured L0 file that we have already folded in.
        //
        // A key whose slot already holds a copy at least as new as the file's
        // newest entry is dropped too (it cannot be beaten there), by the same
        // argument: stale metadata implies the merged L0 files are still captured
        // and their seqs already folded into the slots.
        let may_be_in_l1: Vec<&[u8]> = {
            let (min_max, max_seq) = {
                let meta = self.sstable_metadata[bucket].read();
                let meta = meta.first().filter(|m| m.entry_count > 0);
                (meta.map(|m| (m.min_key.clone(), m.max_key.clone())), meta.and_then(|m| m.max_seq))
            };
            let bloom = self.sstable_blooms[bucket].read();
            pending
                .iter()
                .filter(|(_, slot)| !max_seq.is_some_and(|max| slot.is_some_and(|(_, seq)| seq_newer_or_eq(seq, max))))
                .map(|(k, _)| *k)
                .filter(|k| min_max.as_ref().is_none_or(|(min, max)| *k >= min.as_slice() && *k <= max.as_slice()))
                .filter(|k| bloom.as_ref().is_none_or(|b| b.contains(k)))
                .collect()
        };
        let mut may_be_in_l1 = may_be_in_l1;
        if may_be_in_l1.is_empty() {
            return Ok(bytes_read);
        }
        may_be_in_l1.sort_unstable();

        // Level 1: hop between the candidates with the sparse index, reading only
        // near them. The index may belong to a newer file than the handle (a
        // compaction swapping it), so each hint is validated before use.
        let level1_file = self.sstable_files[bucket].read().clone();
        let file_size = level1_file.metadata().map(|m| m.len()).unwrap_or(0);
        if file_size == 0 {
            return Ok(bytes_read);
        }
        let wanted: Vec<(&[u8], u64)> = {
            let index = self.sstable_indexes[bucket].read();
            may_be_in_l1
                .iter()
                .map(|&k| (k, index.as_ref().map_or(0, |i| i.block_start(k))))
                .collect()
        };
        let (read, rejected) = self.scan_sorted_candidates(&level1_file, file_size, &wanted, |a| fold(pending, a))?;
        if rejected && let Some(m) = self.metrics() {
            Metrics::bump(&m.sparse_hint_rejects);
        }
        Ok(bytes_read + read)
    }

    /// Visit the entries of one sorted SSTable file near each of `candidates`
    /// (`(key, sparse-index hint)`, sorted by key), passing every entry read to
    /// `visit`.
    ///
    /// One forward cursor: for each candidate, jump to its hint when that lies
    /// past the cursor, then read forward until the
    /// cursor passes the candidate. A batch of a few keys therefore reads about
    /// `SAMPLE_INTERVAL` entries per key, while a dense batch turns into one
    /// sequential read, never more than the file. Entries are key-sorted, so any
    /// valid frame boundary at or below a candidate finds it: a hint is used
    /// only after [`valid_scan_start`](Self::valid_scan_start) accepts it, and a
    /// rejected one just keeps scanning from the cursor, which is always below
    /// the current candidate.
    ///
    /// Returns the bytes read and whether any hint was rejected.
    fn scan_sorted_candidates(
        &self,
        file: &File,
        file_len: u64,
        candidates: &[(&[u8], u64)],
        mut visit: impl FnMut(&ArchivedSStableEntry),
    ) -> Result<(u64, bool)> {
        debug_assert!(candidates.windows(2).all(|w| w[0].0 <= w[1].0), "candidates must be sorted");
        let mut frames = FrameReader::new(file, SSTABLE_DATA_START);
        let mut rejected = false;
        let mut next = 0usize;
        'candidates: while next < candidates.len() {
            let (wanted, start) = candidates[next];
            if start > frames.offset() {
                if Self::valid_scan_start(file, start, file_len, wanted) {
                    frames.seek(start);
                } else {
                    rejected = true;
                }
            }
            while let Some(body) = frames.next_frame(file_len)? {
                let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(body)?) };
                visit(archived);
                let key = archived.key.as_slice();
                if key >= wanted {
                    while next < candidates.len() && candidates[next].0 <= key {
                        next += 1;
                    }
                    continue 'candidates;
                }
            }
            break;
        }
        Ok((frames.bytes_read, rejected))
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
        let mut all_entries = ScanMap::new();

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
                                fold_newest(&mut all_entries, key, Some(value), seq);
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
                        s.spawn(move || -> Result<Vec<ScanEntry>> {
                            let mut out = Vec::new();
                            for guard in &l0_ref[bucket as usize] {
                                let mut file = open_registered_l0_for_scan(&guard.entry.path)?;
                                // Bound for `check_frame_len`: a frame can never be longer than its file.
                                let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
                                let mut entry_bytes: Vec<u8> = Vec::new();
                                loop {
                                    let mut size_buf = [0u8; 4];
                                    match file.read_exact(&mut size_buf) {
                                        Ok(_) => {}
                                        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                                        Err(e) => return Err(LSMError::Io(e)),
                                    }
                                    let size = u32::from_le_bytes(size_buf) as usize;
                                    check_frame_len(size as u64, file_len)?;
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
                                fold_newest(&mut all_entries, k, v, seq);
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
                fold_newest(
                    &mut all_entries,
                    record.key.clone(),
                    (!record.tombstone).then_some(record.value),
                    record.seq,
                );
            }
        }

        // 4. Active memtable (captured last, but resolved by seq like the rest).
        for (key, value, seq) in active_snapshot {
            fold_newest(&mut all_entries, key, value, seq);
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
    fn scan_bucket_key_pointers(&self, level1_file: &File) -> Result<Vec<(Vec<u8>, u128, u64)>> {
        let mut kvs = Vec::new();
        // Bound for `check_frame_len`: a frame can never be longer than its file.
        let file_len = level1_file.metadata().map(|m| m.len()).unwrap_or(0);
        let mut offset = SSTABLE_DATA_START;
        let mut entry_bytes = Vec::new();

        loop {
            let mut size_buf = [0u8; 4];
            if !Self::read_exact_at(level1_file, &mut size_buf, &mut offset)? {
                break;
            }

            let size = u32::from_le_bytes(size_buf) as usize;
            check_frame_len(size as u64, file_len)?;
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
        // Layers resolve by highest seq (`fold_newest`, oldest→newest: L1 → L0 → RO →
        // active); only the *capture* order is newest-first so a concurrent
        // flush/compaction cannot drop a live key. See `scan_prefix`. The
        // `take(limit)` runs on the fully merged, sorted map, so limiting is
        // unaffected by the capture order.
        let mut all_entries = ScanMap::new();

        // Memtable layers (newest) captured first, under one `memtable.read()` guard
        // so a flush cannot split a key between active and RO.
        let (active_snapshot, ro_snapshot) = {
            let memtable = self.memtable.read();
            let active: Vec<ScanEntry> = memtable
                .skip_list
                .iter_raw_from(start)
                .map(|(key, value, seq, tombstone)| (key.to_vec(), (!tombstone).then_some(value), seq))
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
        let mut newer: Vec<ScanEntry> = Vec::new();
        let bounded = limit <= BOUNDED_SCAN_LIMIT;

        // Reads one bucket's L0 files (oldest-first, so an exact seq tie goes to the
        // newer file), range-filtered (`>= start`). L0 has no min/max or sparse index, so files are scanned
        // in full.
        let read_l0_bucket = |bucket: usize| -> Result<Vec<ScanEntry>> {
            let mut bucket_entries = Vec::new();
            for guard in &l0_guards[bucket] {
                let mut file = open_registered_l0_for_scan(&guard.entry.path)?;
                // Bound for `check_frame_len`: a frame can never be longer than its file.
                let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
                let mut entry_bytes: Vec<u8> = Vec::new();
                loop {
                    let mut size_buf = [0u8; 4];
                    match file.read_exact(&mut size_buf) {
                        Ok(_) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                        Err(e) => return Err(LSMError::Io(e)),
                    }
                    let size = u32::from_le_bytes(size_buf) as usize;
                    check_frame_len(size as u64, file_len)?;
                    entry_bytes.resize(size, 0);
                    file.read_exact(&mut entry_bytes[..size])?;
                    let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes[..size])?) };
                    if archived.key.as_slice() >= start {
                        let val = (!archived.tombstone).then(|| archived.value.to_native());
                        bucket_entries.push((archived.key.as_slice().to_vec(), val, archived.seq.to_native()));
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
                    newer.push((record.key.clone(), (!record.tombstone).then_some(record.value), record.seq));
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
                for (key, _, _) in newer.iter() {
                    newer_per_bucket[get_bucket_for_key(key, self.config.num_buckets) as usize] += 1;
                }
                for (bucket, file) in l1_files.iter().enumerate() {
                    if !self.l1_overlaps(bucket, start, None) {
                        continue;
                    }
                    let cap = limit.saturating_add(1).saturating_add(newer_per_bucket[bucket]);
                    let mut entries = Vec::new();
                    self.scan_l1_range_into(bucket, file, start, None, Some(cap), &mut entries)?;
                    for (k, v, seq) in entries {
                        fold_newest(&mut all_entries, k, v, seq);
                    }
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
                            s.spawn(move || -> Result<Vec<ScanEntry>> {
                                let mut entries = Vec::new();
                                self.scan_l1_range_into(bucket, file, start, None, None, &mut entries)?;
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
                                for (k, v, seq) in entries {
                                    fold_newest(&mut all_entries, k, v, seq);
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

        // Fold the newer layers over L1 (oldest→newest), resolving by seq.
        for (key, value, seq) in newer {
            fold_newest(&mut all_entries, key, value, seq);
        }

        Ok(all_entries
            .into_iter()
            .filter_map(|(key, (value, _))| value.map(|_| key))
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
    pub(crate) fn range_pointers_bounded(&self, start: &[u8], end: Option<&[u8]>, limit: usize) -> Result<Vec<(Vec<u8>, u128, u64)>> {
        // Layers resolve by highest seq (`fold_newest`, oldest→newest: L1 → L0 → RO →
        // active); only the *capture* order is newest-first so a concurrent
        // flush/compaction cannot drop a live key. See `scan_prefix`. `take(limit)`
        // runs on the fully merged, sorted map.
        //
        // `end` (exclusive) bounds the scan to `[start, end)`; entries `>= end` are
        // never inserted so the merge map and the resulting page stay within the
        // requested key window.
        let mut all_entries = ScanMap::new();

        // Memtable layers (newest) captured first, under one `memtable.read()` guard
        // so a flush cannot split a key between active and RO.
        let (active_snapshot, ro_snapshot) = {
            let memtable = self.memtable.read();
            let active: Vec<ScanEntry> = memtable
                .skip_list
                .iter_raw_from(start)
                .take_while(|(key, _value, _seq, _tombstone)| end.is_none_or(|e| *key < e))
                .map(|(key, value, seq, tombstone)| (key.to_vec(), (!tombstone).then_some(value), seq))
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

        // Reads one bucket's L0 files (oldest-first, so an exact seq tie goes to the
        // newer file), range-filtered. L0 has no min/max or sparse index, so files are
        // scanned in full.
        let read_l0_bucket = |bucket: usize| -> Result<Vec<ScanEntry>> {
            let mut bucket_entries = Vec::new();
            for guard in &l0_guards[bucket] {
                let mut file = open_registered_l0_for_scan(&guard.entry.path)?;
                // Bound for `check_frame_len`: a frame can never be longer than its file.
                let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
                let mut entry_bytes: Vec<u8> = Vec::new();
                loop {
                    let mut size_buf = [0u8; 4];
                    match file.read_exact(&mut size_buf) {
                        Ok(_) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                        Err(e) => return Err(LSMError::Io(e)),
                    }
                    let size = u32::from_le_bytes(size_buf) as usize;
                    check_frame_len(size as u64, file_len)?;
                    entry_bytes.resize(size, 0);
                    file.read_exact(&mut entry_bytes[..size])?;
                    let archived = unsafe { rkyv::access_unchecked::<ArchivedSStableEntry>(verify_sstable_payload(&entry_bytes[..size])?) };
                    if archived.key.as_slice() >= start && end.is_none_or(|e| archived.key.as_slice() < e) {
                        let val = (!archived.tombstone).then(|| archived.value.to_native());
                        bucket_entries.push((archived.key.as_slice().to_vec(), val, archived.seq.to_native()));
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
                    newer.push((record.key.clone(), (!record.tombstone).then_some(record.value), record.seq));
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
                for (key, _, _) in newer.iter() {
                    newer_per_bucket[get_bucket_for_key(key, self.config.num_buckets) as usize] += 1;
                }
                for (bucket, file) in l1_files.iter().enumerate() {
                    if !self.l1_overlaps(bucket, start, end) {
                        continue;
                    }
                    let cap = limit.saturating_add(1).saturating_add(newer_per_bucket[bucket]);
                    let mut entries = Vec::new();
                    self.scan_l1_range_into(bucket, file, start, end, Some(cap), &mut entries)?;
                    for (k, v, seq) in entries {
                        fold_newest(&mut all_entries, k, v, seq);
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
                                for (k, v, seq) in entries {
                                    fold_newest(&mut all_entries, k, v, seq);
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

        // Fold the newer layers over L1 (oldest→newest), resolving by seq.
        for (key, value, seq) in newer {
            fold_newest(&mut all_entries, key, value, seq);
        }

        Ok(all_entries
            .into_iter()
            .filter_map(|(key, (value, seq))| value.map(|v| (key, v, seq)))
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
    fn next_test_seq() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(1);
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

    // The entry CRC covers only the body, so it cannot detect a *layout* change:
    // the bytes are identical and only their interpretation differs. That is why
    // SSTable files carry a magic + format version — without one, a database
    // written before the `seq` widening would pass CRC and then be reinterpreted
    // under the new layout, yielding garbage pointers instead of an error.
    // These pin the guard that makes such a file fail loudly.
    #[test]
    fn test_pre_header_sstable_is_rejected_not_misread() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let path = temp_dir.path().join("legacy.dat");

        // A file in the old headerless format: entry frames straight from offset 0.
        {
            let entry = SStableEntry {
                key: b"k".to_vec(),
                key_prefix: key_prefix_of(b"k"),
                value: 42,
                tombstone: false,
                seq: 7,
            };
            let payload = encode_sstable_entry(&entry)?;
            let mut f = File::create(&path)?;
            f.write_all(&(payload.len() as u32).to_le_bytes())?;
            f.write_all(&payload)?;
            f.sync_all()?;
        }

        let err = open_sstable_for_scan(&path).expect_err("a pre-header SSTable must be rejected, not parsed");
        assert!(
            matches!(err, LSMError::Corruption(ref m) if m.contains("magic")),
            "expected a bad-magic corruption error, got: {err:?}"
        );
        Ok(())
    }

    #[test]
    fn test_sstable_with_unknown_format_version_is_rejected() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let path = temp_dir.path().join("future.dat");

        // Correct magic, but a version this build does not know how to read.
        let mut header = sstable_header_bytes();
        header[4..8].copy_from_slice(&(SSTABLE_FORMAT_VERSION + 1).to_le_bytes());
        let mut f = File::create(&path)?;
        f.write_all(&header)?;
        f.sync_all()?;
        drop(f);

        let err = open_sstable_for_scan(&path).expect_err("an unknown format version must be rejected");
        assert!(
            matches!(err, LSMError::Corruption(ref m) if m.contains("format version")),
            "expected a format-version corruption error, got: {err:?}"
        );
        Ok(())
    }

    // The guard is only useful if it fires on the path a real database takes at
    // startup — `LSMTree::open` deliberately tolerates a partial metadata load, so
    // this checks that leniency does not swallow an unreadable format.
    #[test]
    fn test_open_rejects_l1_file_written_in_an_older_format() -> Result<()> {
        let temp_dir = TempDir::new()?;
        {
            let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
            lsm.insert_with_seq(b"k", 0xAAAA, 1)?;
            lsm.flush_memtable_to_level0()?;
            lsm.compact_all()?; // → a real, header-bearing L1 file
        }

        // Strip the header to simulate a file written before headers existed.
        let bucket = get_bucket_for_key(b"k", crate::support::TEST_NUM_BUCKETS);
        let l1 = LSMTree::level1_path_from(temp_dir.path(), bucket);
        let body = std::fs::read(&l1)?;
        assert!(body.len() as u64 > SSTABLE_DATA_START, "setup: L1 file should hold entries");
        std::fs::write(&l1, &body[SSTABLE_DATA_START as usize..])?;

        // `LSMTree` is not `Debug`, so match on the result rather than `expect_err`.
        match LSMTree::open(temp_dir.path(), test_lsm_config()) {
            Ok(_) => panic!("open must reject an unreadable L1 format, but it succeeded"),
            Err(LSMError::Corruption(m)) => assert!(m.contains("magic"), "expected a bad-magic corruption error, got: {m}"),
            Err(e) => panic!("expected a corruption error, got: {e:?}"),
        }
        Ok(())
    }

    // A frame's 4-byte length prefix sits *outside* the entry CRC (which covers only
    // the body), so a flipped bit there is undetectable until after the payload has
    // been read — and the readers used to allocate from it first. A corrupt length
    // could therefore request up to 4 GiB from a file of a few dozen bytes.
    //
    // Both the old and new code end in an error here, so the assertion has to be on
    // *which* error: the guard reports the impossible length itself, whereas an
    // unguarded reader allocates, tries to fill the buffer, and surfaces an unrelated
    // `Io(UnexpectedEof)` — having already committed the allocation.
    #[test]
    fn test_corrupt_frame_length_is_rejected_before_allocating() -> Result<()> {
        let temp_dir = TempDir::new()?;

        for bogus in [u32::MAX, 1 << 30, 0] {
            let path = temp_dir.path().join(format!("frame_{bogus}.dat"));
            let mut f = File::create(&path)?;
            f.write_all(&sstable_header_bytes())?;
            f.write_all(&bogus.to_le_bytes())?; // claims `bogus` bytes…
            f.write_all(&[0xABu8; 8])?; // …but the file holds 8
            f.sync_all()?;
            drop(f);

            let file_len = std::fs::metadata(&path)?.len();
            assert!(file_len < 64, "setup: the file must be far smaller than the claimed frame");

            let err = LSMTree::level0_file_summary(&path)
                .map(|_| ())
                .expect_err("a frame longer than its file must be rejected");
            match err {
                LSMError::Corruption(m) => assert!(
                    m.contains("declares"),
                    "expected the frame-length guard to fire, got a different corruption: {m}"
                ),
                other => panic!("expected the frame-length guard to fire before allocating, got: {other:?}"),
            }
        }
        Ok(())
    }

    /// Write `bodies` as consecutive `u32 LE len ‖ body` frames after `lead` junk bytes.
    fn write_frames(path: &Path, lead: usize, bodies: &[Vec<u8>]) -> Result<()> {
        let mut f = File::create(path)?;
        f.write_all(&vec![0xEEu8; lead])?;
        for body in bodies {
            f.write_all(&(body.len() as u32).to_le_bytes())?;
            f.write_all(body)?;
        }
        f.sync_all()?;
        Ok(())
    }

    // The buffered frame reader must return exactly the frames the per-entry
    // `read_exact_at` loop did — including frames that straddle a refill boundary
    // and a frame larger than the maximum read size — starting from a mid-file
    // offset (the sparse-index seek), and end cleanly at EOF.
    #[test]
    fn test_frame_reader_returns_every_frame_across_refills() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let path = temp_dir.path().join("frames.dat");
        let mut bodies: Vec<Vec<u8>> = (0..5000u32).map(|i| vec![(i % 251) as u8; 1 + (i as usize * 7) % 97]).collect();
        bodies.insert(2500, vec![0x5Au8; FrameReader::MAX_READ + 12_345]); // bigger than any refill
        let lead = 16;
        write_frames(&path, lead, &bodies)?;

        let file = File::open(&path)?;
        let file_len = file.metadata()?.len();
        let mut reader = FrameReader::new(&file, lead as u64);
        let mut read = Vec::new();
        while let Some(body) = reader.next_frame(file_len)? {
            read.push(body.to_vec());
        }
        assert_eq!(read.len(), bodies.len());
        assert!(read == bodies, "frames must round-trip byte for byte");
        Ok(())
    }

    // EOF inside a frame is corruption, not a clean end: a truncated length prefix
    // or a truncated body must surface `UnexpectedEof`, exactly as `read_exact_at`
    // did, rather than silently ending the scan early.
    #[test]
    fn test_frame_reader_truncated_frame_is_an_error() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let path = temp_dir.path().join("frames.dat");
        write_frames(&path, 0, &[vec![1u8; 10], vec![2u8; 10]])?;
        let full = std::fs::read(&path)?;

        // Cut 1: inside the second frame's body. Cut 2: inside its length prefix.
        for cut in [full.len() - 3, 14 + 2] {
            std::fs::write(&path, &full[..cut])?;
            let file = File::open(&path)?;
            let file_len = file.metadata()?.len();
            let mut reader = FrameReader::new(&file, 0);
            assert_eq!(reader.next_frame(file_len)?.map(<[u8]>::to_vec), Some(vec![1u8; 10]));
            match reader.next_frame(file_len) {
                Err(LSMError::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::UnexpectedEof, "cut at {cut}"),
                other => panic!("cut at {cut}: expected UnexpectedEof, got {:?}", other.map(|o| o.map(<[u8]>::to_vec))),
            }
        }
        Ok(())
    }

    // `merge_scan_layers` avoids a map insert per L1 entry, but must give exactly
    // what folding L1 and then every newer layer into one map by `fold_newest`
    // gives: highest seq wins, a tie goes to the later (newer) copy. Checked over
    // many generated layer sets with a tiny key and seq space, so duplicate keys
    // within L1, tombstones on both sides, and exact seq ties are all common.
    #[test]
    fn test_merge_scan_layers_matches_seq_fold() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64; // deterministic LCG, no RNG dependency
        let mut next = |bound: u64| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % bound
        };
        for case in 0..2_000 {
            // L1 as the scan reads it: several buckets, each run key-sorted.
            let mut l1: Vec<ScanEntry> = Vec::new();
            for _bucket in 0..next(4) {
                let mut run: Vec<ScanEntry> = (0..next(12))
                    .map(|_| {
                        (
                            vec![next(3) as u8, next(6) as u8],
                            (next(4) != 0).then(|| u128::from(next(1000))),
                            next(8),
                        )
                    })
                    .collect();
                run.sort_by(|a, b| a.0.cmp(&b.0));
                l1.extend(run);
            }
            let mut newer = ScanMap::new();
            for _ in 0..next(6) {
                fold_newest(
                    &mut newer,
                    vec![next(3) as u8, next(6) as u8],
                    (next(3) != 0).then(|| u128::from(next(1000))),
                    next(8),
                );
            }

            let mut reference = ScanMap::new();
            for (k, v, seq) in l1.iter().cloned() {
                fold_newest(&mut reference, k, v, seq);
            }
            for (k, (v, seq)) in newer.clone() {
                fold_newest(&mut reference, k, v, seq);
            }
            let expected: Vec<(Vec<u8>, u128)> = reference.into_iter().filter_map(|(k, (v, _))| v.map(|v| (k, v))).collect();

            let l1_keys = vec![l1.iter().flat_map(|(k, _, _)| k.iter().copied()).collect::<Vec<u8>>()];
            let mut at = 0;
            let packed: Vec<PackedScanEntry> = l1
                .iter()
                .map(|(k, v, seq)| {
                    at += k.len();
                    (0, at - k.len(), at, *v, *seq)
                })
                .collect();
            assert_eq!(
                merge_scan_layers(&l1_keys, packed, newer.clone()).to_pairs(),
                expected,
                "case {case}: l1={l1:?} newer={newer:?}"
            );
        }
    }

    // The L1 publish must never leave the bucket's L1 path missing. It used to move the
    // live file aside (`rename(level1 -> .old)`) before renaming the new one in, and a
    // crash in between was silent, permanent loss of every key that lived only in L1:
    // `LSMTree::open` created a 0-byte L1, `file_len == 0` short-circuited the metadata
    // load into "empty bucket", and `cleanup_old_files_on_startup` then deleted the
    // `.old` that still held the data.
    //
    // The fix is structural — one atomic `rename(2)`, which replaces the destination —
    // so the property to pin is that no reachable intermediate state has the L1 path
    // absent or empty while data lives only there.
    //
    // Note the setup must call `cleanup_obsolete_level0_files()`: without it the merged
    // L0 file is still on disk and answers the read, masking the loss entirely.
    #[test]
    fn test_crash_between_l1_publish_steps_keeps_l1() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let bucket = get_bucket_for_key(b"k", crate::support::TEST_NUM_BUCKETS);
        let l1 = LSMTree::level1_path_from(temp_dir.path(), bucket);
        {
            let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
            lsm.insert_with_seq(b"k", 0xAAAA, 1)?;
            lsm.flush_memtable_to_level0()?;
            lsm.compact_all()?;
            lsm.cleanup_obsolete_level0_files();
            assert_eq!(lsm.get(b"k")?, Some(0xAAAA), "setup: key should be readable from L1");
        }

        // The key must now live *only* in L1, or this test cannot observe the bug.
        let l0_dir = LSMTree::level0_bucket_dir_from(temp_dir.path(), bucket);
        let l0_files = std::fs::read_dir(&l0_dir).map(|r| r.count()).unwrap_or(0);
        assert_eq!(l0_files, 0, "setup: no L0 file may remain, or it would mask the loss");
        assert!(std::fs::metadata(&l1)?.len() > SSTABLE_DATA_START, "setup: L1 must hold the entry");

        // The window is only visible *during* a publish — a completed merge removed its
        // `.old`, so inspecting the final state proves nothing (an earlier version of
        // this test asserted exactly that and passed against the bug). Watch the L1 path
        // from another thread while compactions run, and assert it is never absent: that
        // is precisely the state a crash would freeze, and the state `LSMTree::open`
        // turns into an empty bucket.
        let lsm = Arc::new(LSMTree::open(temp_dir.path(), test_lsm_config())?);
        let stop = Arc::new(AtomicBool::new(false));
        let vanished = Arc::new(AtomicUsize::new(0));

        let observer = {
            let (l1, stop, vanished) = (l1.clone(), Arc::clone(&stop), Arc::clone(&vanished));
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    // "Absent" and "present but 0 bytes" are the same failure: both make
                    // `LSMTree::open` short-circuit into an empty bucket.
                    match std::fs::metadata(&l1) {
                        Err(_) => {
                            vanished.fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(m) if m.len() == 0 => {
                            vanished.fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(_) => {}
                    }
                    std::hint::spin_loop();
                }
            })
        };

        // seq 1 belongs to the key seeded above, so these start at 2.
        for round in 0..200u64 {
            lsm.insert_with_seq(format!("k{round:04}").as_bytes(), 0xCCCC + round as u128, 2 + round)?;
            lsm.flush_memtable_to_level0()?;
            lsm.compact_bucket(bucket)?;
        }

        stop.store(true, Ordering::Relaxed);
        observer.join().expect("observer thread panicked");

        assert_eq!(
            vanished.load(Ordering::Relaxed),
            0,
            "the L1 path was absent/empty mid-publish — a crash in that window makes \
             LSMTree::open create a 0-byte L1 and lose every key that lived only in L1"
        );
        assert_eq!(lsm.get(b"k")?, Some(0xAAAA), "the original L1-only key must survive the merges");
        Ok(())
    }

    // Reads of a *registered* L0 file skip header validation (`open_registered_l0`),
    // so `LSMTree::open` is the only thing standing between a pre-header database and
    // a garbage reinterpretation of its L0 files. This pins that single guard: without
    // it, the stripped file would be loaded and its entries parsed under the wrong
    // layout, with nothing downstream to notice.
    #[test]
    fn test_open_rejects_l0_file_written_in_an_older_format() -> Result<()> {
        let temp_dir = TempDir::new()?;
        {
            let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
            lsm.insert_with_seq(b"k", 0xAAAA, 1)?;
            lsm.flush_memtable_to_level0()?; // → a real, header-bearing L0 file
        }

        let bucket = get_bucket_for_key(b"k", crate::support::TEST_NUM_BUCKETS);
        let dir = LSMTree::level0_bucket_dir_from(temp_dir.path(), bucket);
        let l0 = std::fs::read_dir(&dir)?.next().expect("setup: an L0 file should exist")?.path();
        let body = std::fs::read(&l0)?;
        // Strip the header, leaving enough bytes that this is a *format* mismatch and
        // not the sub-header "incomplete debris" case, which is skipped by design.
        let stripped = &body[SSTABLE_DATA_START as usize..];
        assert!(
            stripped.len() as u64 > SSTABLE_DATA_START,
            "setup: stripped L0 body must stay longer than a header"
        );
        std::fs::write(&l0, stripped)?;

        match LSMTree::open(temp_dir.path(), test_lsm_config()) {
            Ok(_) => panic!("open must reject an unreadable L0 format, but it succeeded"),
            Err(LSMError::Corruption(m)) => assert!(m.contains("magic"), "expected a bad-magic corruption error, got: {m}"),
            Err(e) => panic!("expected a corruption error, got: {e:?}"),
        }
        Ok(())
    }

    // The L0 counterpart of the check above, and the case where the two levels must
    // behave *differently*. L1 is written to a temp file and renamed, so an L1 file at
    // its final path is always complete; L0 is written in place, so a crash mid-flush
    // can leave one shorter than a header. That is incomplete debris, not a stale
    // format, and the records are still in the WAL — so open must skip it and carry on
    // rather than refusing to start the database.
    #[test]
    fn test_open_skips_a_crash_truncated_level0_file() -> Result<()> {
        let temp_dir = TempDir::new()?;
        {
            let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
            lsm.insert_with_seq(b"survivor", 0xAAAA, 1)?;
            lsm.insert_with_seq(b"k", 0xBBBB, 2)?;
            lsm.flush_memtable_to_level0()?;
        }

        let bucket = get_bucket_for_key(b"k", crate::support::TEST_NUM_BUCKETS);
        let dir = LSMTree::level0_bucket_dir_from(temp_dir.path(), bucket);
        let torn = std::fs::read_dir(&dir)?.next().expect("setup: an L0 file should exist")?.path();
        // What a build that wrote L0 in place left after a crash between
        // `File::create` and the header write: 0 bytes.
        std::fs::write(&torn, b"")?;

        let lsm = match LSMTree::open(temp_dir.path(), test_lsm_config()) {
            Ok(lsm) => lsm,
            Err(e) => panic!("open must skip an incomplete L0 file, not fail the whole database: {e:?}"),
        };
        // Reads must still work — the torn file must not be registered, or every later
        // scan would fail header validation on it.
        assert_eq!(
            lsm.get(b"k")?,
            None,
            "the torn file's records are gone from L0 (WAL replay restores them)"
        );
        lsm.scan_prefix(b"")?;
        Ok(())
    }

    /// A process killed while a flush writes an L0 file leaves it cut anywhere,
    /// including inside a frame's body (a kill can land mid-`write`). The file
    /// is written under a temporary name and renamed only once complete, so what
    /// a kill leaves is a `.dat.tmp`: open removes it, and since its flush never
    /// retired its WAL entries, replay restores its records. Written in place,
    /// it was a torn `.dat` that failed every later open (found by
    /// `acknowledged_writes_survive_repeated_kills`, ~1 kill in 800).
    #[test]
    fn test_open_removes_a_level0_file_a_kill_left_unfinished() -> Result<()> {
        let temp_dir = TempDir::new()?;
        {
            let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
            lsm.insert_with_seq(b"k", 0xBBBB, 2)?;
            lsm.flush_memtable_to_level0()?;
        }
        let bucket = get_bucket_for_key(b"k", crate::support::TEST_NUM_BUCKETS);
        let dir = LSMTree::level0_bucket_dir_from(temp_dir.path(), bucket);
        let done = std::fs::read_dir(&dir)?.next().expect("setup: an L0 file should exist")?.path();
        assert_eq!(done.extension().unwrap(), "dat", "a finished flush leaves only `.dat` files");
        // The state a kill mid-flush leaves: the temporary, cut inside a frame body.
        let torn = dir.join(format!("{}{LEVEL0_TMP_SUFFIX}", done.file_stem().unwrap().to_string_lossy()));
        std::fs::rename(&done, &torn)?;
        let cut = SSTABLE_DATA_START + 4 + 3;
        assert!(std::fs::metadata(&torn)?.len() > cut, "setup: the frame must be longer than the cut");
        std::fs::OpenOptions::new().write(true).open(&torn)?.set_len(cut)?;

        let lsm = match LSMTree::open(temp_dir.path(), test_lsm_config()) {
            Ok(lsm) => lsm,
            Err(e) => panic!("open must remove an unfinished L0 file, not fail: {e:?}"),
        };
        assert!(!torn.exists(), "the unfinished file was not removed");
        assert_eq!(lsm.get(b"k")?, None, "its records are not in L0 (WAL replay restores them)");
        lsm.scan_prefix(b"")?;
        lsm.compact_all()?;
        Ok(())
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
        let metrics = Arc::new(Metrics::default());
        lsm.set_metrics(Arc::clone(&metrics));
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
        assert_eq!(metrics.snapshot().sparse_hint_rejects, 0, "a zero hint is no hint, not a rejection");

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
        // A fallback is correct but slow, so it must be counted: that counter is
        // the only production signal that hints are going wrong. (At least 7 of 8:
        // the mid-file offset could in principle land on a real frame boundary.)
        assert!(
            metrics.snapshot().sparse_hint_rejects >= 7,
            "rejected hints were not counted: {}",
            metrics.snapshot().sparse_hint_rejects
        );

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

    // Regression: a cold key's L1 entry keeps its original seq forever, so the
    // gap between it and a much-later write to the same key is unbounded. The
    // sequence width must therefore cover the database's whole lifetime — a
    // 32-bit seq with RFC-1982 serial comparison silently inverts once the gap
    // exceeds 2^31 (~2.1B writes, ~2.5 days at 10k writes/s), making the *newer*
    // write compare as older. The skip list's own justification for a narrow seq
    // ("the memtable flushes long before 2 billion writes accumulate") holds only
    // for the memtable and must not be extended to the SSTable levels.
    //
    // Symptoms this pins: `get` returns the stale L1 value, and
    // `merge_level0_to_level1` then bakes it into L1 permanently.
    #[test]
    fn test_write_after_huge_seq_gap_beats_cold_l1_entry() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

        // A cold key written early in the database's life, compacted down to L1
        // where it keeps seq 1000 indefinitely.
        const OLD_SEQ: u64 = 1_000;
        lsm.insert_with_seq(b"cold", 0xAAAA, OLD_SEQ)?;
        // `compact_all` only drains read-only memtables, so seal the active one
        // first — otherwise the entry never leaves the memtable and this exercises
        // the skip list instead of the L1 path it is meant to cover.
        lsm.flush_memtable_to_level0()?;
        lsm.compact_all()?;
        let bucket = get_bucket_for_key(b"cold", crate::support::TEST_NUM_BUCKETS);
        let l1_len = std::fs::metadata(LSMTree::level1_path_from(temp_dir.path(), bucket))?.len();
        assert!(
            l1_len > SSTABLE_DATA_START,
            "setup: the cold entry must actually be in L1, not still in memory"
        );
        assert_eq!(lsm.get(b"cold")?, Some(0xAAAA), "setup: cold value should read back from L1");

        // ~3B writes later the same key is updated. The gap exceeds 2^31, which
        // is exactly what a 32-bit serial comparison cannot represent.
        const NEW_SEQ: u64 = OLD_SEQ + 3_000_000_000;
        assert!(NEW_SEQ - OLD_SEQ > u64::from(u32::MAX) / 2, "test premise: gap must exceed 2^31");
        lsm.insert_with_seq(b"cold", 0xBBBB, NEW_SEQ)?;

        assert_eq!(
            lsm.get(b"cold")?,
            Some(0xBBBB),
            "a newer write lost to a cold L1 entry — sequence comparison inverted across the gap"
        );

        // The inversion is permanent once the merge resolves the conflict, so
        // check the post-compaction state too.
        lsm.flush_memtable_to_level0()?;
        lsm.compact_all()?;
        assert_eq!(
            lsm.get(b"cold")?,
            Some(0xBBBB),
            "the stale value was baked into L1 — the merge resolved the conflict backwards"
        );
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
        //
        // The corruption must keep a **valid SSTable header** — these files stay in
        // the L0 registry, and registered files are opened without re-validating
        // (see `open_registered_l0`). A headerless body would just read as EOF past
        // `SSTABLE_DATA_START`, i.e. an empty file, and the merge would succeed.
        let mut corrupted = 0;
        for bucket in 0..lsm.config.num_buckets as u32 {
            let bucket_dir = LSMTree::level0_bucket_dir_from(&lsm.base_path, bucket);
            let Ok(rd) = std::fs::read_dir(&bucket_dir) else { continue };
            for entry in rd {
                let path = entry?.path();
                if path.is_file() {
                    let mut bad = sstable_header_bytes().to_vec();
                    bad.extend_from_slice(&100u32.to_le_bytes()); // claims a 100-byte entry…
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

    // Regression: `get_multiple` must resolve a key by highest `seq` across all
    // layers, exactly like `get_with_seq` — never by layer order. Value-log GC
    // decides liveness with `get_multiple`, so a layer-order answer of "deleted"
    // for a live key makes GC drop its record and unlink the segment: permanent
    // loss. The inversion (a lower-seq copy in a newer layer) is reachable in
    // production: a TTL delete allocates its seq before taking the bucket lock, so
    // it can apply after a higher-seq put has already been sealed below it.
    //
    // Each case places `high` (seq 20) in an older layer and `low` (seq 10) in a
    // newer one, at every layer boundary the batch path walks.
    fn assert_batch_matches_point_gets(lsm: &LSMTree, keys: &[&[u8]], label: &str) -> Result<()> {
        let owned: Vec<Vec<u8>> = keys.iter().map(|k| k.to_vec()).collect();
        let batch = lsm.get_multiple(&owned)?;
        for (key, got) in keys.iter().zip(batch) {
            assert_eq!(
                got,
                lsm.get_with_seq(key)?,
                "{label}: get_multiple disagrees with get_with_seq for {:?}",
                String::from_utf8_lossy(key)
            );
        }
        Ok(())
    }

    #[test]
    fn test_get_multiple_resolves_by_seq_not_layer_order() -> Result<()> {
        // Stage: which layer the newer-but-lower-seq copy ends up in.
        #[derive(Debug, Clone, Copy)]
        enum Newer {
            ActiveMemtable,
            ReadOnlyMemtable,
            Level0,
        }
        #[derive(Debug, Clone, Copy)]
        enum Older {
            ReadOnlyMemtable,
            Level0,
            Level1,
        }
        let cases = [
            (Older::ReadOnlyMemtable, Newer::ActiveMemtable),
            (Older::Level0, Newer::ActiveMemtable),
            (Older::Level1, Newer::ActiveMemtable),
            (Older::ReadOnlyMemtable, Newer::ReadOnlyMemtable),
            (Older::Level0, Newer::ReadOnlyMemtable),
            (Older::Level1, Newer::ReadOnlyMemtable),
            (Older::Level0, Newer::Level0),
            (Older::Level1, Newer::Level0),
        ];
        for (older, newer) in cases {
            let temp_dir = TempDir::new()?;
            let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;

            // Older layer, higher seq: a live value and a tombstone.
            lsm.insert_with_seq(b"live", 0xAAAA, 20)?;
            lsm.delete_with_seq(b"dead", 20)?;
            match older {
                Older::ReadOnlyMemtable => lsm.flush_memtable()?,
                Older::Level0 => lsm.flush_memtable_to_level0()?,
                Older::Level1 => {
                    lsm.flush_memtable_to_level0()?;
                    lsm.compact_all()?;
                }
            }

            // Newer layer, lower seq: a stale tombstone over `live`, a stale value
            // over `dead`.
            lsm.delete_with_seq(b"live", 10)?;
            lsm.insert_with_seq(b"dead", 0xBBBB, 10)?;
            match newer {
                Newer::ActiveMemtable => {}
                Newer::ReadOnlyMemtable => lsm.flush_memtable()?,
                Newer::Level0 => lsm.flush_memtable_to_level0()?,
            }

            assert_eq!(lsm.get_with_seq(b"live")?, Some((0xAAAA, 20)), "{older:?}/{newer:?}: point get");
            // L1 is the bottom level, so compaction drops the tombstone: nothing is
            // left to shadow the seq-10 value, and both paths must agree it is live.
            let dead_expected = match older {
                Older::Level1 => Some((0xBBBB, 10)),
                _ => None,
            };
            assert_eq!(lsm.get_with_seq(b"dead")?, dead_expected, "{older:?}/{newer:?}: point get");
            // `live` twice: a duplicated input key must resolve at every position.
            assert_batch_matches_point_gets(&lsm, &[b"live", b"dead", b"never", b"live"], &format!("{older:?}/{newer:?}"))?;
        }
        Ok(())
    }

    /// Brute-force point lookup for the differential test below: reads every
    /// layer in full — no summaries, no seq pruning, no fast path — and keeps the
    /// highest seq, newest layer first so an exact tie stays with the newer copy.
    fn reference_get(lsm: &LSMTree, key: &[u8]) -> Result<Option<(u128, u64)>> {
        let mut best: Candidate = None;
        if let Some((value, seq, tombstone)) = lsm.memtable.read().skip_list.entry(key) {
            LSMTree::merge_candidate(&mut best, (!tombstone).then_some(value), seq);
        }
        for ro in lsm.read_only_memtables.read().iter().rev() {
            if let Some(r) = ro.records().iter().find(|r| r.key == key) {
                LSMTree::merge_candidate(&mut best, (!r.tombstone).then_some(r.value), r.seq);
            }
        }
        let bucket = get_bucket_for_key(key, lsm.config.num_buckets);
        let mut files = lsm.level0_entries_snapshot(bucket);
        files.retain(|f| !f.is_obsolete());
        files.sort_by_key(|f| std::cmp::Reverse(f.created_at_ms));
        for file in files {
            for e in lsm.read_sstable_entries_from_path(&file.path)? {
                if e.key == key {
                    LSMTree::merge_candidate(&mut best, (!e.tombstone).then_some(e.value), e.seq);
                }
            }
        }
        for e in lsm.read_sstable_entries(bucket)? {
            if e.key == key {
                LSMTree::merge_candidate(&mut best, (!e.tombstone).then_some(e.value), e.seq);
            }
        }
        Ok(best.and_then(|(value, seq)| value.map(|ptr| (ptr, seq))))
    }

    // Differential guard for the read path's shortcuts — the active-memtable fast
    // path, RO binary search, L0 summaries (range/bloom/max-seq) and L1 max-seq
    // pruning — in both `get_with_seq` and `get_multiple`. Random writes carry
    // OUT-OF-ORDER sequences (so lower-seq copies land in newer layers, the case
    // every shortcut must respect) plus GC-style re-points at an existing seq, with
    // flushes, compactions and reopens mixed in. After every step both lookup paths
    // must agree with `reference_get` for every key, including never-written ones.
    #[test]
    fn test_read_shortcuts_agree_with_brute_force_resolution() -> Result<()> {
        // splitmix64: tiny, seedable, reproducible across runs.
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            }
            fn below(&mut self, n: u64) -> u64 {
                self.next() % n
            }
        }

        // Keys share 8-byte prefixes in groups so several land in each bucket and
        // L0 files hold overlapping key ranges; "absent_*" keys are never written.
        let keys: Vec<Vec<u8>> = (0..48).map(|i| format!("key{:02}_{}", i % 12, i).into_bytes()).collect();
        let probes: Vec<Vec<u8>> = keys
            .iter()
            .cloned()
            .chain((0..6).map(|i| format!("absent_{i}").into_bytes()))
            .chain(std::iter::once(keys[0].clone())) // a duplicated input key
            .collect();

        // Shared across every tree so the end-of-test check can prove each
        // shortcut actually fired — a guard that never exercises them proves nothing.
        let metrics = Arc::new(Metrics::default());
        for seed in 0..16u64 {
            let temp_dir = TempDir::new()?;
            let mut lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
            lsm.set_metrics(Arc::clone(&metrics));
            let mut rng = Rng(seed);
            let mut clock: u64 = 1_000;

            for step in 0..300 {
                clock += 8;
                let key = &keys[rng.below(keys.len() as u64) as usize];
                // Jitter the sequence backwards by up to 5 writes' worth, so later
                // writes often carry lower seqs than earlier ones.
                let seq = clock - rng.below(40);
                match rng.below(100) {
                    0..=54 => lsm.insert_with_seq(key, u128::from(rng.next()), seq)?,
                    55..=79 => lsm.delete_with_seq(key, seq)?,
                    80..=86 => {
                        // GC re-point: same seq as the current winner, new pointer.
                        if let Some((_, cur)) = lsm.get_with_seq(key)? {
                            lsm.insert_with_seq(key, u128::from(rng.next()), cur)?;
                        }
                    }
                    87..=91 => lsm.flush_memtable()?,
                    92..=95 => lsm.flush_memtable_to_level0()?,
                    96..=98 => lsm.compact_all()?,
                    _ => {
                        // Reopen. A bare LSM has no WAL, so make everything durable first.
                        lsm.flush_memtable_to_level0()?;
                        drop(lsm);
                        lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
                        lsm.set_metrics(Arc::clone(&metrics));
                    }
                }

                let batch = lsm.get_multiple(&probes)?;
                for (probe, batched) in probes.iter().zip(batch) {
                    let expected = reference_get(&lsm, probe)?;
                    let label = format!("seed {seed} step {step} key {:?}", String::from_utf8_lossy(probe));
                    assert_eq!(lsm.get_with_seq(probe)?, expected, "{label}: get_with_seq");
                    assert_eq!(batched, expected, "{label}: get_multiple");
                }
            }
        }
        let m = metrics.snapshot();
        assert!(m.fast_path_hits > 0, "the active-memtable fast path never fired");
        assert!(m.l0_bloom_rejects > 0, "no L0 file was ever skipped by range/bloom");
        assert!(m.seq_prunes > 0, "no SSTable was ever skipped by max-seq");
        assert!(m.l0_probes > 0 && m.l1_probes > 0, "the slow path never read L0 and L1");
        Ok(())
    }

    /// A batch read must read near its keys, not whole SSTables: a one-key batch
    /// on a large L1 file reads a few KiB, and every batch size agrees with the
    /// single-key `get` across L1, several overlapping L0 files and the memtable
    /// (overwrites, tombstones, absent and duplicated keys).
    #[test]
    fn test_get_multiple_reads_near_its_keys_and_agrees_with_get() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = LSMConfig {
            num_buckets: 2,
            ..LSMConfig::default()
        };
        let lsm = LSMTree::open(temp_dir.path(), config)?;
        let key = |i: u64| format!("doc:{i:08}").into_bytes();
        let mut seq = 1u64;
        for i in 0..20_000u64 {
            lsm.insert_with_seq(&key(i), u128::from(i) + 1, seq)?;
            seq += 1;
        }
        lsm.flush_and_compact_all()?;
        // Two L0 files of overwrites and deletes, then unflushed memtable writes.
        for round in 0..2u64 {
            for i in (round..20_000).step_by(7) {
                if i % 3 == 0 {
                    lsm.delete_with_seq(&key(i), seq)?;
                } else {
                    lsm.insert_with_seq(&key(i), u128::from(i + 1_000_000 * (round + 1)), seq)?;
                }
                seq += 1;
            }
            lsm.flush_memtable_to_level0()?;
        }
        for i in (0..20_000u64).step_by(101) {
            lsm.insert_with_seq(&key(i), 7, seq)?;
            seq += 1;
        }
        let l1_bytes: u64 = (0..2).map(|b| lsm.sstable_files[b].read().metadata().map(|m| m.len()).unwrap_or(0)).sum();
        assert!(l1_bytes > 500_000, "setup: L1 should be large, is {l1_bytes} bytes");

        let mut probes: Vec<Vec<u8>> = (0..20_000u64).step_by(13).map(key).collect();
        probes.extend((0..50u64).map(|i| format!("doc:{i:08}x").into_bytes())); // absent, between keys
        probes.push(key(4_000)); // duplicated
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut pick = |n: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % n as u64) as usize
        };
        for size in [1usize, 2, 5, 20, 200, probes.len()] {
            for _ in 0..10 {
                let batch: Vec<Vec<u8>> = (0..size).map(|_| probes[pick(probes.len())].clone()).collect();
                let (got, read) = lsm.get_multiple_counted(&batch)?;
                for (k, g) in batch.iter().zip(&got) {
                    assert_eq!(*g, lsm.get_with_seq(k)?, "batch of {size}: key {:?}", String::from_utf8_lossy(k));
                }
                if size == 1 {
                    assert!(read < 64 * 1024, "a one-key batch read {read} bytes (L1 is {l1_bytes})");
                }
                assert!(read <= 2 * l1_bytes, "a batch of {size} read {read} bytes, more than the files");
            }
        }
        Ok(())
    }

    /// A stale or bogus sparse-index hint (an L1 file swapped by a compaction
    /// under the index) must never lose a candidate in a batch read: it is
    /// rejected and the cursor keeps scanning from where it is.
    #[test]
    fn test_batch_scan_survives_bogus_hints() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = LSMConfig {
            num_buckets: 1,
            ..LSMConfig::default()
        };
        let lsm = LSMTree::open(temp_dir.path(), config)?;
        for i in 0..3_000u64 {
            lsm.insert_with_seq(format!("k:{i:08}").as_bytes(), u128::from(i) + 1, i + 1)?;
        }
        lsm.flush_and_compact_all()?;
        let file = lsm.sstable_files[0].read().clone();
        let file_len = file.metadata()?.len();
        let keys: Vec<Vec<u8>> = [5u64, 900, 901, 2_999].iter().map(|i| format!("k:{i:08}").into_bytes()).collect();
        let late = lsm.sstable_indexes[0].read().as_ref().map(|x| x.block_start(b"k:99999999")).unwrap_or(0);
        for bad in [u64::MAX, 1, 3, file_len / 2 + 1, file_len, late] {
            let candidates: Vec<(&[u8], u64)> = keys.iter().map(|k| (k.as_slice(), bad)).collect();
            let mut seen = Vec::new();
            let (_, rejected) = lsm.scan_sorted_candidates(&file, file_len, &candidates, |a| {
                if keys.iter().any(|k| k.as_slice() == a.key.as_slice()) {
                    seen.push(a.key.to_vec());
                }
            })?;
            assert_eq!(seen, keys, "bogus hint {bad} lost a candidate");
            // A hint at or below the cursor's start is never needed, so never checked.
            assert_eq!(rejected, bad > SSTABLE_DATA_START, "bogus hint {bad}: rejection not reported as expected");
        }
        Ok(())
    }

    // The L0 sparse index is built twice: from offsets tracked while the flush
    // writes the file, and by rescanning the file at open. A wrong offset is
    // invisible to correctness tests — `valid_scan_start` rejects the hint and the
    // lookup silently falls back to a full scan — so pin it directly: both builds
    // must agree, and every hint must be a valid place to start scanning.
    #[test]
    fn test_l0_sparse_index_hints_are_valid_and_survive_reopen() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = LSMConfig {
            num_buckets: 1,
            ..LSMConfig::default()
        };
        let keys: Vec<Vec<u8>> = (0..500u32).map(|i| format!("key{:05}", i * 2).into_bytes()).collect();
        // Probe present keys and the gaps between them (odd numbers are absent).
        let probes: Vec<Vec<u8>> = (0..1000u32).map(|i| format!("key{i:05}").into_bytes()).collect();

        let hints_at_flush: Vec<u64> = {
            let lsm = LSMTree::open(temp_dir.path(), config.clone())?;
            for (i, key) in keys.iter().enumerate() {
                // Values of varying width so frames differ in length.
                lsm.insert_with_seq(key, u128::from(i as u32) << (i % 96), i as u64 + 1)?;
            }
            lsm.flush_memtable_to_level0()?;
            let entries = lsm.level0_entries_snapshot(0);
            assert_eq!(entries.len(), 1, "setup: one L0 file");
            let entry = &entries[0];
            assert!(entry.summary.as_ref().is_some_and(|s| s.index.is_some()), "the flush must build an index");
            probes.iter().map(|k| entry.scan_start(k)).collect()
        };

        let lsm = LSMTree::open(temp_dir.path(), config)?;
        let entry = lsm.level0_entries_snapshot(0).remove(0);
        let hints_at_open: Vec<u64> = probes.iter().map(|k| entry.scan_start(k)).collect();
        if let Some(i) = (0..probes.len()).find(|&i| hints_at_flush[i] != hints_at_open[i]) {
            panic!(
                "flush-time and open-time indexes disagree for {:?}: {} at flush, {} at open",
                String::from_utf8_lossy(&probes[i]),
                hints_at_flush[i],
                hints_at_open[i]
            );
        }

        let file = File::open(&entry.path)?;
        let file_len = file.metadata()?.len();
        let mut hinted = 0;
        for (key, &hint) in probes.iter().zip(&hints_at_open) {
            if hint != 0 {
                hinted += 1;
                assert!(
                    LSMTree::valid_scan_start(&file, hint, file_len, key),
                    "hint {hint} for {:?} is not a valid scan start",
                    String::from_utf8_lossy(key)
                );
            }
        }
        assert!(hinted > 900, "almost every probe should get a hint, got {hinted}");
        for (i, key) in keys.iter().enumerate() {
            assert_eq!(lsm.get(key)?, Some(u128::from(i as u32) << (i % 96)), "key {i} after reopen");
        }
        Ok(())
    }

    // Every multi-key scan must resolve each key exactly as a point read does —
    // highest seq across all layers — never by "the newer layer overwrites". Random
    // writes carry OUT-OF-ORDER seqs, so a lower-seq copy often sits in a newer
    // layer (the shape a TTL delete racing a put, or a wall-clock step between two
    // L0 flushes, produces); flushes, compactions and reopens are mixed in. After
    // every step each scan must equal the brute-force view built from
    // `reference_get`, including bounded pages small enough to exercise the L1 cap.
    #[test]
    fn test_scans_agree_with_brute_force_resolution() -> Result<()> {
        let mut state = 0xD1B5_4A32_D192_ED03u64;
        let mut next = |bound: u64| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % bound
        };
        // Two 4-byte prefixes ("key0", "key1") so `scan_prefixes` has ids to probe.
        let keys: Vec<Vec<u8>> = (0..40).map(|i| format!("key{}_{:02}", i % 2, i).into_bytes()).collect();

        for seed in 0..8u64 {
            let temp_dir = TempDir::new()?;
            let mut lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
            let mut clock: u64 = 1_000 + seed;

            for step in 0..250 {
                clock += 8;
                let key = &keys[next(keys.len() as u64) as usize];
                let seq = clock - next(40);
                match next(100) {
                    0..=54 => lsm.insert_with_seq(key, u128::from(next(1 << 30)) + 1, seq)?,
                    55..=81 => lsm.delete_with_seq(key, seq)?,
                    82..=88 => lsm.flush_memtable()?,
                    89..=94 => lsm.flush_memtable_to_level0()?,
                    95..=98 => lsm.compact_all()?,
                    _ => {
                        lsm.flush_memtable_to_level0()?;
                        drop(lsm);
                        lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
                    }
                }

                // Truth: every key's point-read answer, in key order.
                let mut truth: Vec<(Vec<u8>, u128, u64)> = Vec::new();
                for k in &keys {
                    if let Some((ptr, s)) = reference_get(&lsm, k)? {
                        truth.push((k.clone(), ptr, s));
                    }
                }
                truth.sort();
                truth.dedup();
                let with_prefix = |p: &[u8]| truth.iter().filter(|(k, _, _)| k.starts_with(p)).cloned().collect::<Vec<_>>();
                let label = format!("seed {seed} step {step}");

                assert_eq!(lsm.scan_prefix(b"key1")?, with_prefix(b"key1"), "{label}: scan_prefix");
                let ids: std::collections::HashSet<u32> = [u32::from_be_bytes(*b"key0"), u32::from_be_bytes(*b"key1")].into_iter().collect();
                let pairs = |v: Vec<(Vec<u8>, u128, u64)>| v.into_iter().map(|(k, p, _)| (k, p)).collect::<Vec<_>>();
                assert_eq!(lsm.scan_prefixes(&ids)?.to_pairs(), pairs(truth.clone()), "{label}: scan_prefixes");
                assert_eq!(lsm.key_pointer_pairs(None)?, pairs(truth.clone()), "{label}: key_pointer_pairs");
                assert_eq!(
                    lsm.range_pointers_bounded(b"", None, usize::MAX)?,
                    truth,
                    "{label}: range_pointers_bounded, unbounded"
                );
                // A small page from the middle: exercises the per-bucket L1 read cap.
                let start = b"key0_10".as_slice();
                let page: Vec<_> = truth.iter().filter(|(k, _, _)| k.as_slice() >= start).take(3).cloned().collect();
                assert_eq!(
                    lsm.range_pointers_bounded(start, None, 3)?,
                    page,
                    "{label}: range_pointers_bounded, limit 3"
                );
                let page_keys: Vec<Vec<u8>> = page.iter().map(|(k, _, _)| k.clone()).collect();
                assert_eq!(lsm.range_keys_bounded(start, 3)?, page_keys, "{label}: range_keys_bounded, limit 3");
            }
        }
        Ok(())
    }

    // The L0 prefix filter compares a key's zero-padded 8-byte `key_prefix`, so a
    // key shorter than a prefix that ends in 0x00 bytes used to match it — but only
    // while the key sat in L0; the memtable and L1 paths compare real key bytes.
    // All three layers must agree that "ab" does not start with "ab\0".
    #[test]
    fn test_scan_prefix_rejects_a_key_shorter_than_the_prefix_in_every_layer() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(temp_dir.path(), test_lsm_config())?;
        lsm.insert_with_seq(b"ab", 1, 1)?;
        lsm.insert_with_seq(b"ab\0x", 2, 2)?; // a real match, to show the filter still admits one
        let expected = vec![(b"ab\0x".to_vec(), 2u128, 2u64)];

        assert_eq!(lsm.scan_prefix(b"ab\0")?, expected, "memtable");
        lsm.flush_memtable_to_level0()?;
        assert_eq!(lsm.scan_prefix(b"ab\0")?, expected, "L0");
        lsm.compact_all()?;
        assert_eq!(lsm.scan_prefix(b"ab\0")?, expected, "L1");
        Ok(())
    }

    // Two threads flushing the same read-only memtable must produce ONE L0 file,
    // and both must return only once it is durable. Without the per-memtable flush
    // lock both wrote it (a duplicate file; in the same millisecond, one file name
    // with the second `File::create` truncating the first).
    /// Records each flush event with the number of L0 files registered at that
    /// moment, reading the tree through a weak handle set after it is built.
    struct EventLog {
        tree: std::sync::OnceLock<std::sync::Weak<LSMTree>>,
        events: parking_lot::Mutex<Vec<(&'static str, u64, usize)>>,
    }

    impl EventLog {
        fn registered(&self) -> usize {
            let tree = self.tree.get().and_then(|t| t.upgrade()).expect("tree alive");
            (0..tree.config.num_buckets as u32).map(|b| tree.level0_entries_snapshot(b).len()).sum()
        }
    }

    impl LsmFlushObserver for EventLog {
        fn on_memtable_sealed(&self, _version: u64) {}
        fn on_ro_memtable_flushing(&self, version: u64) -> std::io::Result<()> {
            let n = self.registered();
            self.events.lock().push(("flushing", version, n));
            Ok(())
        }
        fn on_ro_memtable_written(&self, version: u64) -> std::io::Result<()> {
            let n = self.registered();
            self.events.lock().push(("written", version, n));
            Ok(())
        }
        fn on_ro_memtable_flushed_to_level0(&self, version: u64) {
            let n = self.registered();
            self.events.lock().push(("flushed", version, n));
        }
    }

    // The vector index WAL syncs in `on_ro_memtable_flushing`, makes its
    // progress durable in `on_ro_memtable_written`, and relies on: the events
    // come in that order, none of the memtable's L0 files is registered
    // (mergeable) at either, and memtables arrive in version order even when
    // several threads flush at once. All are pinned here.
    #[test]
    fn the_written_event_comes_before_registration_and_in_version_order() -> Result<()> {
        for round in 0..50 {
            let temp_dir = TempDir::new()?;
            let lsm = Arc::new(LSMTree::open(
                temp_dir.path(),
                LSMConfig {
                    num_buckets: 2,
                    ..LSMConfig::default()
                },
            )?);
            let log = Arc::new(EventLog {
                tree: std::sync::OnceLock::new(),
                events: parking_lot::Mutex::new(Vec::new()),
            });
            log.tree.set(Arc::downgrade(&lsm)).ok();
            lsm.set_flush_observer(Some(Arc::clone(&log) as Arc<dyn LsmFlushObserver>));
            let mut seq = 0u64;
            for _ in 0..3 {
                for i in 0..50u64 {
                    seq += 1;
                    lsm.insert_with_seq(format!("k{i:03}").as_bytes(), u128::from(seq), seq)?;
                }
                lsm.flush_memtable()?; // three read-only memtables, none on disk
            }
            let barrier = Arc::new(std::sync::Barrier::new(2));
            let flushers: Vec<_> = (0..2)
                .map(|_| {
                    let (lsm, barrier) = (Arc::clone(&lsm), Arc::clone(&barrier));
                    std::thread::spawn(move || {
                        barrier.wait();
                        lsm.flush_memtable_to_level0()
                    })
                })
                .collect();
            for flusher in flushers {
                flusher.join().expect("flusher panicked")?;
            }
            // 50 keys over 2 buckets: each memtable registers one file per bucket.
            let want: Vec<(&str, u64, usize)> = (0..3u64)
                .flat_map(|v| {
                    [
                        ("flushing", v, 2 * v as usize),
                        ("written", v, 2 * v as usize),
                        ("flushed", v, 2 * (v as usize + 1)),
                    ]
                })
                .collect();
            assert_eq!(*log.events.lock(), want, "round {round}");
        }
        Ok(())
    }

    /// Refuses the first `fail` flushing or written events (`at_flushing`
    /// picks which), then accepts.
    struct Veto {
        fail: std::sync::atomic::AtomicUsize,
        at_flushing: bool,
    }

    impl Veto {
        fn refuse(&self) -> std::io::Result<()> {
            let left = self.fail.load(Ordering::SeqCst);
            if left == 0 {
                return Ok(());
            }
            self.fail.store(left - 1, Ordering::SeqCst);
            Err(std::io::Error::other("cannot record the flush"))
        }
    }

    impl LsmFlushObserver for Veto {
        fn on_memtable_sealed(&self, _version: u64) {}
        fn on_ro_memtable_flushing(&self, _version: u64) -> std::io::Result<()> {
            if self.at_flushing { self.refuse() } else { Ok(()) }
        }
        fn on_ro_memtable_written(&self, _version: u64) -> std::io::Result<()> {
            if self.at_flushing { Ok(()) } else { self.refuse() }
        }
        fn on_ro_memtable_flushed_to_level0(&self, _version: u64) {}
    }

    // A flushing or written event that fails must leave nothing registered and
    // no file behind, and the memtable must stay readable and be flushed by a
    // retry.
    #[test]
    fn a_failed_flush_event_undoes_the_flush_and_a_retry_completes_it() -> Result<()> {
        for at_flushing in [true, false] {
            a_failed_flush_event_undoes_the_flush(at_flushing)?;
        }
        Ok(())
    }

    fn a_failed_flush_event_undoes_the_flush(at_flushing: bool) -> Result<()> {
        let temp_dir = TempDir::new()?;
        let lsm = LSMTree::open(
            temp_dir.path(),
            LSMConfig {
                num_buckets: 2,
                ..LSMConfig::default()
            },
        )?;
        lsm.set_flush_observer(Some(Arc::new(Veto {
            fail: std::sync::atomic::AtomicUsize::new(1),
            at_flushing,
        })));
        for i in 0..20u64 {
            lsm.insert_with_seq(format!("k{i:02}").as_bytes(), u128::from(i), i + 1)?;
        }
        let registered = |lsm: &LSMTree| (0..2).map(|b| lsm.level0_entries_snapshot(b).len()).sum::<usize>();
        let l0_files = || {
            std::fs::read_dir(temp_dir.path().join("level0"))
                .map(|d| d.flatten().filter(|e| e.path().is_file()).count())
                .unwrap_or(0)
        };

        assert!(lsm.flush_memtable_to_level0().is_err(), "the vetoed flush reports its error");
        assert_eq!(registered(&lsm), 0, "nothing registered");
        assert_eq!(l0_files(), 0, "no L0 file left behind");
        assert_eq!(lsm.get(b"k07")?, Some(7), "the read-only memtable still serves reads");

        lsm.flush_memtable_to_level0()?;
        assert_eq!(registered(&lsm), 2, "the retry registers one file per bucket");
        assert_eq!(lsm.get(b"k07")?, Some(7));
        Ok(())
    }

    #[test]
    fn test_concurrent_flushes_of_one_memtable_write_it_once() -> Result<()> {
        for round in 0..100 {
            let temp_dir = TempDir::new()?;
            let lsm = Arc::new(LSMTree::open(
                temp_dir.path(),
                LSMConfig {
                    num_buckets: 1,
                    ..LSMConfig::default()
                },
            )?);
            for i in 0..500u64 {
                lsm.insert_with_seq(format!("k{i:05}").as_bytes(), u128::from(i) + 1, i + 1)?;
            }
            lsm.flush_memtable()?; // one read-only memtable, not yet on disk
            let barrier = Arc::new(std::sync::Barrier::new(2));
            let flushers: Vec<_> = (0..2)
                .map(|_| {
                    let (lsm, barrier) = (Arc::clone(&lsm), Arc::clone(&barrier));
                    std::thread::spawn(move || {
                        barrier.wait();
                        lsm.flush_memtable_to_level0()
                    })
                })
                .collect();
            for flusher in flushers {
                flusher.join().expect("flusher panicked")?;
            }
            assert_eq!(lsm.level0_entries_snapshot(0).len(), 1, "round {round}: the memtable was flushed twice");
            lsm.purge_ro_memtables_for_test(); // read from L0 alone
            for i in 0..500u64 {
                assert_eq!(lsm.get(format!("k{i:05}").as_bytes())?, Some(u128::from(i) + 1), "round {round}: key {i}");
            }
        }
        Ok(())
    }

    // GC relocation re-inserts a key at its existing seq with a new pointer, so the
    // old and relocated copies tie on seq and only L0 file order says which is
    // newer. File order came from the raw wall clock, so a clock step backwards
    // between the two flushes inverted it: every read (and the L0→L1 merge) chose
    // the stale pointer, whose segment GC then unlinks. Simulated here by moving
    // the first file's stamp into the future — the state a clock step leaves
    // behind — and reopening, which also checks the stamp floor is seeded from
    // disk.
    #[test]
    fn test_l0_order_survives_the_wall_clock_stepping_back() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = LSMConfig {
            num_buckets: 1,
            ..LSMConfig::default()
        };
        let (original, relocated) = (0xA_u128, 0xB_u128);
        {
            let lsm = LSMTree::open(temp_dir.path(), config.clone())?;
            lsm.insert_with_seq(b"k", original, 5)?;
            lsm.flush_memtable_to_level0()?;
        }
        // The clock that named that file was an hour ahead of the one we have now.
        let l0_dir = LSMTree::level0_bucket_dir_from(temp_dir.path(), 0);
        let first = std::fs::read_dir(&l0_dir)?.next().expect("one L0 file")?.path();
        let stamp = LSMTree::level0_created_at_from_path(&first);
        let hour = 3_600_000u128 * 1_000_000;
        std::fs::rename(&first, l0_dir.join(format!("{}.dat", stamp + hour)))?;

        let lsm = LSMTree::open(temp_dir.path(), config.clone())?;
        lsm.insert_with_seq(b"k", relocated, 5)?; // GC relocation: same seq, new pointer
        lsm.flush_memtable_to_level0()?;
        // Drop the read-only memtable (still holding the relocation) so reads must
        // settle the tie by L0 file order alone.
        lsm.purge_ro_memtables_for_test();

        assert_eq!(lsm.get(b"k")?, Some(relocated), "point read chose the pre-relocation copy");
        assert_eq!(lsm.get_multiple(&[b"k".to_vec()])?[0].map(|(p, _)| p), Some(relocated), "get_multiple");
        lsm.compact_all()?;
        assert_eq!(lsm.get(b"k")?, Some(relocated), "the L0→L1 merge baked in the stale copy");

        // And in-process: a candidate stamp from a clock that went backwards is
        // still issued above everything before it.
        let before = *lsm.last_l0_stamp.lock();
        assert!(lsm.next_l0_stamp(before - hour) > before, "a stamp went backwards");
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
    fn kv_sorted(v: Vec<(Vec<u8>, u128, u64)>) -> Vec<(Vec<u8>, u128)> {
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
        let results = lsm.scan_prefixes(&ids)?.to_pairs();
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

    /// `merge_scan_layers` sorts by `key_head` and then the key; that must be the
    /// keys' byte order, including keys that are prefixes of others, zero bytes
    /// where padding sits, and keys on both sides of 16 bytes.
    #[test]
    fn key_head_then_key_is_byte_order() {
        let mut s = 0x1d87_2b41_ee3a_9f05_u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut keys: Vec<Vec<u8>> = (0..5_000).map(|_| (0..next() % 24).map(|_| (next() % 3) as u8).collect()).collect();
        let mut by_head = keys.clone();
        by_head.sort_by(|a, b| key_head(a).cmp(&key_head(b)).then_with(|| a.cmp(b)));
        keys.sort();
        assert_eq!(by_head, keys);
    }

    /// The run merge must equal sorting every entry and keeping the newest copy
    /// of each key: keys of 0–23 bytes (many sharing a 16-byte head) in 8 runs,
    /// with the same key in several runs at different seqs, and the same entries
    /// shuffled.
    #[test]
    fn merging_l1_runs_equals_sorting_and_keeping_the_newest() {
        let mut s = 0x5851_f42d_4c95_7f2d_u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for total in [1_000usize, 60_000] {
            let mut buffers: Vec<Vec<u8>> = Vec::new();
            let mut entries: Vec<PackedScanEntry> = Vec::new();
            for run in 0..8usize {
                let mut keys: Vec<Vec<u8>> = (0..total / 8)
                    .map(|_| {
                        let len = (next() % 24) as usize;
                        let mut k: Vec<u8> = (0..len).map(|_| (next() % 4) as u8).collect();
                        if next() % 2 == 0 && len > 16 {
                            k[..16].fill(1);
                        }
                        k
                    })
                    .collect();
                keys.sort();
                keys.dedup();
                let mut buffer = Vec::new();
                for k in keys {
                    let start = buffer.len();
                    buffer.extend_from_slice(&k);
                    let seq = next() % 1_000;
                    entries.push((run, start, buffer.len(), Some(seq as u128), seq));
                }
                buffers.push(buffer);
            }
            let key = |e: &PackedScanEntry| &buffers[e.0][e.1..e.2];
            let merged: Vec<(Vec<u8>, u64)> = merge_l1_runs(&entries, key).iter().map(|e| (key(e).to_vec(), e.4)).collect();

            let mut reference: std::collections::BTreeMap<Vec<u8>, u64> = std::collections::BTreeMap::new();
            for e in &entries {
                let seq = reference.entry(key(e).to_vec()).or_insert(e.4);
                if e.4 > *seq {
                    *seq = e.4;
                }
            }
            let reference: Vec<(Vec<u8>, u64)> = reference.into_iter().collect();
            assert_eq!(merged, reference, "{total} entries");

            // In random order the input breaks into far more than `MAX_MERGED_RUNS`
            // runs, which takes the sorting path; it must give the same result.
            let mut shuffled = entries.clone();
            for i in (1..shuffled.len()).rev() {
                shuffled.swap(i, (next() % (i as u64 + 1)) as usize);
            }
            let merged: Vec<(Vec<u8>, u64)> = merge_l1_runs(&shuffled, key).iter().map(|e| (key(e).to_vec(), e.4)).collect();
            assert_eq!(merged, reference, "{total} entries, shuffled");
        }
    }

    /// The per-prefix cursors must return exactly each probed prefix's keys for
    /// any probe set: sparse or dense, adjacent or far apart, prefixes with no
    /// keys, postings that span several sparse-index blocks, and keys too short to
    /// have a prefix (which no layer returns). Eight buckets, so the scan splits
    /// the prefixes over several workers that each read every bucket. Checked with
    /// everything in L1, then again with rewrites, deletes and new keys in several
    /// L0 files above it, which the L0 cursor must seek through the same way.
    #[test]
    fn scan_prefixes_over_l1_and_l0_matches_a_reference_for_any_probe_set() -> Result<()> {
        let temp = TempDir::new()?;
        let lsm = LSMTree::open(
            temp.path(),
            LSMConfig {
                num_buckets: 8,
                ..LSMConfig::default()
            },
        )?;
        let mut s = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let n_clusters = 300u32;
        let mut truth: std::collections::BTreeMap<Vec<u8>, u128> = std::collections::BTreeMap::new();
        for cid in 0..n_clusters {
            // 0–80 keys per cluster, so a cluster can span up to five
            // `SAMPLE_INTERVAL`-entry blocks per bucket or have no keys at all.
            for did in 0..(next() % 81) as u32 {
                let key = u32_prefixed_key(cid, did);
                let value = (cid as u128) << 32 | did as u128;
                lsm.insert(&key, value)?;
                truth.insert(key, value);
            }
        }
        // Three bytes: no prefix id, never returned. Four bytes: prefix 5 alone.
        lsm.insert(&[0u8, 0, 1], 1)?;
        lsm.insert(&[0u8, 0, 0, 5], 2)?;
        truth.insert(vec![0, 0, 0, 5], 2);
        lsm.flush_and_compact_all()?;
        lsm.purge_ro_memtables_for_test();

        let mut check = |lsm: &LSMTree, truth: &std::collections::BTreeMap<Vec<u8>, u128>, layers: &str| -> Result<()> {
            for density in [1u64, 5, 50, 90, 100] {
                for _ in 0..5 {
                    let ids: std::collections::HashSet<u32> = (0..n_clusters + 5).filter(|_| next() % 100 < density).collect();
                    if ids.is_empty() {
                        continue;
                    }
                    let expected: Vec<(Vec<u8>, u128)> = truth
                        .iter()
                        .filter(|(k, _)| ids.contains(&u32::from_be_bytes(k[..4].try_into().unwrap())))
                        .map(|(k, v)| (k.clone(), *v))
                        .collect();
                    assert_eq!(lsm.scan_prefixes(&ids)?.to_pairs(), expected, "{layers}, density {density}%");
                }
            }
            Ok(())
        };
        check(&lsm, &truth, "L1 only")?;

        // Three L0 files, each over every other cluster from a different start, so
        // a file holds some probed prefixes and not others and spans many
        // sparse-index blocks. A quarter of the writes delete (an L0 tombstone
        // over an L1 key, or over nothing).
        let mut wave_rng = 0x9e37_79b9_7f4a_7c15_u64;
        let mut wave_next = move || {
            wave_rng ^= wave_rng << 13;
            wave_rng ^= wave_rng >> 7;
            wave_rng ^= wave_rng << 17;
            wave_rng
        };
        for wave in 0..3u32 {
            for cid in (wave..n_clusters).step_by(2) {
                for did in 0..(wave_next() % 120) as u32 {
                    let key = u32_prefixed_key(cid, did);
                    if wave_next() % 4 == 0 {
                        lsm.delete(&key)?;
                        truth.remove(&key);
                    } else {
                        let value = ((cid as u128) << 32 | did as u128) + (u128::from(wave) + 1) * 1_000_000;
                        lsm.insert(&key, value)?;
                        truth.insert(key, value);
                    }
                }
            }
            lsm.flush_memtable_to_level0()?;
        }
        // Flushed read-only memtables still answer scans until released; drop them
        // so the second wave is read from L0 alone.
        lsm.purge_ro_memtables_for_test();
        let l0_files: usize = lsm
            .level0_files
            .iter()
            .map(|b| b.read().iter().filter(|e| !e.is_obsolete()).count())
            .sum();
        assert!(l0_files >= 3, "the second wave must stay in L0 ({l0_files} files)");
        check(&lsm, &truth, "L1 and L0")?;
        Ok(())
    }

    /// A scan probing one posting of an L0 file must seek to it through the
    /// file's sparse index, not read the file from its head: the result alone
    /// cannot show this, because a rejected or missing hint falls back to reading
    /// in order and still finds every key.
    #[test]
    fn scan_prefixes_seeks_through_level0_files() -> Result<()> {
        let temp = TempDir::new()?;
        let lsm = LSMTree::open(
            temp.path(),
            LSMConfig {
                num_buckets: 1,
                ..LSMConfig::default()
            },
        )?;
        let (n_clusters, per) = (300u32, 40u32);
        for cid in 0..n_clusters {
            for did in 0..per {
                lsm.insert(&u32_prefixed_key(cid, did), u128::from(cid) << 32 | u128::from(did))?;
            }
        }
        lsm.flush_memtable_to_level0()?;
        lsm.purge_ro_memtables_for_test();

        let probed = [n_clusters / 2];
        let (mut found, mut read, mut file_bytes) = (0u32, 0u64, 0u64);
        for entry in lsm.level0_files[0].read().iter() {
            let file = open_registered_l0(&entry.path)?;
            let starts: Vec<u64> = probed.iter().map(|id| entry.scan_start(&id.to_be_bytes())).collect();
            read += LSMTree::for_each_entry_in_prefixes(&file, &starts, &probed, |_| found += 1)?;
            file_bytes += file.metadata()?.len();
        }
        assert_eq!(found, per);
        assert!(
            read * 10 < file_bytes,
            "read {read} of {file_bytes} bytes for one posting of {n_clusters}"
        );

        let ids: std::collections::HashSet<u32> = probed.into_iter().collect();
        assert_eq!(lsm.scan_prefixes(&ids)?.len(), per as usize);
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
        let results = lsm.scan_prefixes(&ids)?.to_pairs();

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
        let results = lsm.scan_prefixes(&ids)?.to_pairs();

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
        let results = lsm.scan_prefixes(&ids)?.to_pairs();
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
        let results = lsm.scan_prefixes(&ids)?.to_pairs();

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
        let results = lsm.scan_prefixes(&ids)?.to_pairs();
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
        let results = lsm.scan_prefixes(&ids)?.to_pairs();

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

    /// Live (not yet merged) Level-0 files in `bucket`.
    fn live_l0(lsm: &LSMTree, bucket: usize) -> usize {
        lsm.level0_files[bucket].read().iter().filter(|f| !f.is_obsolete()).count()
    }

    /// The worker merges a bucket's L0 into L1 only when the L0 files hold
    /// `L0_COMPACTION_RATIO_PERCENT` of L1's entries, not after every flush: a
    /// merge rewrites the whole L1 file. Into an empty L1 any L0 is due.
    #[test]
    fn small_flushes_wait_until_level0_is_a_share_of_level1() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = LSMConfig {
            num_buckets: 1,
            ..LSMConfig::default()
        };
        let lsm = LSMTree::open(temp_dir.path(), config)?;
        let key = |i: u64| format!("doc:{i:08}").into_bytes();
        let mut seq = 1u64;
        lsm.insert_with_seq(&key(0), 1, seq)?;
        seq += 1;
        lsm.flush_memtable_to_level0()?;
        assert!(lsm.compaction_due(0), "an empty L1 takes any L0");
        lsm.compact_due()?;
        assert_eq!(live_l0(&lsm, 0), 0);

        for i in 1..1_000u64 {
            lsm.insert_with_seq(&key(i), 1, seq)?;
            seq += 1;
        }
        lsm.flush_and_compact_all()?;
        // 1,000 entries in L1: flushes of 30 wait until L0 holds 100 (10%).
        let mut flushed = 0u64;
        for round in 0..3u64 {
            for i in 0..30u64 {
                lsm.insert_with_seq(&key(2_000 + round * 30 + i), 1, seq)?;
                seq += 1;
            }
            lsm.flush_memtable_to_level0()?;
            flushed += 30;
            assert!(!lsm.compaction_due(0), "{flushed} L0 entries against 1,000 in L1 is not due");
            lsm.compact_due()?;
            assert_eq!(live_l0(&lsm, 0), round as usize + 1, "nothing merged yet");
            assert!(!lsm.has_due_compaction_work(), "the tick released the flushed memtable");
        }
        for i in 0..10u64 {
            lsm.insert_with_seq(&key(3_000 + i), 1, seq)?;
            seq += 1;
        }
        lsm.flush_memtable_to_level0()?;
        assert!(lsm.compaction_due(0), "100 L0 entries is 10% of L1");
        lsm.compact_due()?;
        assert_eq!(live_l0(&lsm, 0), 0);
        for i in (0..1_000u64).chain(2_000..2_090).chain(3_000..3_010) {
            assert!(lsm.get_with_seq(&key(i))?.is_some(), "key {i} survives the merge");
        }
        Ok(())
    }

    /// However small, `L0_COMPACTION_MAX_FILES` L0 files make a merge due: each
    /// one a point read cannot rule out is a file it reads.
    #[test]
    fn many_tiny_flushes_are_merged_at_the_file_cap() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = LSMConfig {
            num_buckets: 1,
            ..LSMConfig::default()
        };
        let lsm = LSMTree::open(temp_dir.path(), config)?;
        let key = |i: u64| format!("doc:{i:08}").into_bytes();
        let mut seq = 1u64;
        for i in 0..10_000u64 {
            lsm.insert_with_seq(&key(i), 1, seq)?;
            seq += 1;
        }
        lsm.flush_and_compact_all()?;
        for n in 1..=L0_COMPACTION_MAX_FILES {
            lsm.insert_with_seq(&key(n as u64), 2, seq)?;
            seq += 1;
            lsm.flush_memtable_to_level0()?;
            assert_eq!(lsm.compaction_due(0), n == L0_COMPACTION_MAX_FILES, "{n} one-entry files");
        }
        lsm.compact_due()?;
        assert_eq!(live_l0(&lsm, 0), 0);
        Ok(())
    }

    /// The explicit and shutdown path, `compact_all`, still merges every bucket
    /// that has any L0 file; only the worker's `compact_due` waits.
    #[test]
    fn compact_all_merges_what_compact_due_leaves() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = LSMConfig {
            num_buckets: 1,
            ..LSMConfig::default()
        };
        let lsm = LSMTree::open(temp_dir.path(), config)?;
        let key = |i: u64| format!("doc:{i:08}").into_bytes();
        let mut seq = 1u64;
        for i in 0..1_000u64 {
            lsm.insert_with_seq(&key(i), 1, seq)?;
            seq += 1;
        }
        lsm.flush_and_compact_all()?;
        lsm.delete_with_seq(&key(5), seq)?;
        lsm.flush_memtable_to_level0()?;
        lsm.compact_due()?;
        assert_eq!(live_l0(&lsm, 0), 1, "one tombstone is not due");
        assert!(lsm.get_with_seq(&key(5))?.is_none(), "the L0 tombstone shadows L1");
        lsm.compact_all()?;
        assert_eq!(live_l0(&lsm, 0), 0);
        assert!(lsm.get_with_seq(&key(5))?.is_none(), "the merge drops the tombstone and the key");
        assert!(lsm.get_with_seq(&key(6))?.is_some());
        Ok(())
    }
}
