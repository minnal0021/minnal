//! KVStore - Per-namespace key-value storage
//!
//! Each KVStore owns its own LSM tree and sharded value log,
//! representing an independent namespace within the database.
//! The WAL is shared across all namespaces and managed by the
//! Database coordinator.

use crate::db::error::{KVError, Result};
use crate::db::stats::{GCStats, Stats};
use crate::store::gc_journal::{GCJournal, fsync_dir};
use crate::store::lsm::lsm_tree::{LSMConfig, LSMTree, LsmFlushObserver};
use crate::store::lsm_worker::LsmCompactionCommand;
use crate::store::value_log::sharded::{ShardedValueLog, ShardedValuePointer};
use crate::store::value_log::{PAGE_SIZE_BYTES, ValueLog, ValueLogMetadata, ValueRecordMeta};

use log::{debug, error, info, warn};
use parking_lot::RwLock;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::db::metrics::Metrics;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::db::config::SyncConfig;
use crate::db::namespace_index::{NamespaceIndexSet, RowIdFn, RowToKeyFn};
use std::time::Duration;

/// A resolved key/value pair: both the key and the value bytes read from the value log.
pub(crate) type KeyValue = (Vec<u8>, Vec<u8>);

/// A key paired with its encoded value-log pointer and the LSM write `seq` (low
/// u32) of that pointer — the seq lets the value read verify the pointer's slot
/// was not recycled by GC for a different write (stale-pointer detection).
pub(crate) type KeyPointer = (Vec<u8>, u128, u32);

/// One page of a cursor scan: the resolved pairs plus the next cursor
/// (`None` when the scan is exhausted).
pub(crate) type ScanPage = (Vec<KeyValue>, Option<Vec<u8>>);

/// A scan entry resolved to everything needed to read its value: the key, its
/// value-log pointer, a pinned handle to the bucket file it lives in, and the
/// LSM write `seq` (low u32) used to validate the value record on read.
pub(crate) type ResolvedEntry = (Vec<u8>, ShardedValuePointer, Arc<File>, u32);

/// A key tagged with the prefix ID it was found under — used to carry keys that
/// need a single-key retry back to their originating prefix bucket.
pub(crate) type PrefixIdKey = (u32, Vec<u8>);

/// Map from prefix ID to the [`KeyValue`] pairs that share it.
///
/// Returned by [`KVStore::scan_prefixes_batch`].
pub(crate) type PrefixBatchResult = std::collections::HashMap<u32, Vec<KeyValue>>;

fn current_epoch_millis() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

fn decode_sharded_pointer(offset_u128: u128) -> Option<ShardedValuePointer> {
    ShardedValuePointer::from_u128(offset_u128).ok()
}

/// Per-namespace key-value store with its own LSM tree and value log.
pub struct KVStore {
    /// Namespace identifier (0 = "default")
    #[allow(dead_code)]
    pub(crate) namespace_id: u32,
    pub(crate) name: String,

    // LSM tree for keys
    pub(crate) lsm: Arc<LSMTree>,
    #[allow(dead_code)]
    pub(crate) lsm_path: PathBuf,

    // Sharded value log (16 buckets)
    pub(crate) value_log: ShardedValueLog,
    pub(crate) value_log_path: PathBuf,
    pub(crate) metadata_path: PathBuf,

    // Value log metadata
    pub(crate) metadata: Arc<RwLock<ValueLogMetadata>>,

    // Sync configuration
    pub(crate) sync_config: SyncConfig,

    // Counts writes for periodic sync
    pub(crate) write_count: Arc<AtomicU64>,

    // Flag to prevent concurrent value log GC operations
    pub(crate) value_log_gc_in_progress: Arc<AtomicBool>,

    // Track old log files pending deletion
    pub(crate) value_pending_old_logs: Arc<RwLock<Vec<PathBuf>>>,

    // LSM compaction trigger channel
    pub(crate) lsm_compaction_trigger: Arc<RwLock<Option<tokio::sync::mpsc::UnboundedSender<LsmCompactionCommand>>>>,

    // Optional TTL for automatic record expiry
    pub(crate) ttl: Option<Duration>,

    // Active field indices for this namespace (populated via activate_field_index)
    pub(crate) namespace_index: Arc<RwLock<NamespaceIndexSet>>,

    // Optional caller-supplied row-ID function.  When set, every key passed
    // to the index write path is resolved via this closure instead of the
    // default dense row-ID map (`RowMap`).
    pub(crate) row_id_fn: Arc<RwLock<Option<RowIdFn>>>,

    // Optional inverse of `row_id_fn`.  When set alongside `row_id_fn`,
    // query_keys can reconstruct the original key from a row ID in O(1)
    // with zero memory overhead — no map required.
    pub(crate) row_to_key_fn: Arc<RwLock<Option<RowToKeyFn>>>,

    // Per-namespace dense row-ID map (the default ID source when no custom
    // `row_id_fn` is registered).  Loaded lazily when the first field index is
    // activated; `None` until then.  See `index::RowMap`.
    pub(crate) rowmap: Arc<RwLock<Option<index::RowMap>>>,

    // Global write-sequence counter used for highest-sequence-wins conflict
    // resolution in the memtable. In production this is shared with the WAL
    // (`Database::next_seq`) via `set_seq_counter`, so a value's sequence is the
    // same whether it was assigned at WAL-append time (`put_to_storage_seq`) or
    // allocated here for a non-WAL write (TTL/bulk). A standalone KVStore (tests)
    // gets its own fresh counter, which is internally consistent.
    pub(crate) seq_counter: RwLock<Arc<AtomicU64>>,

    /// Engine-wide operational counters, shared from `Database` via
    /// [`set_metrics`](Self::set_metrics). `None` for a standalone store (tests).
    pub(crate) metrics: OnceLock<Arc<crate::db::metrics::Metrics>>,
}

/// Result of compacting a single value log bucket.
struct BucketGCResult {
    bytes_reclaimed: u64,
    bytes_live: u64,
    old_path: Option<PathBuf>,
    new_metadata: ValueLogMetadata,
    old_metadata: ValueLogMetadata,
    bucket: u32,
    /// True if at least one page in this bucket crossed the page-GC threshold
    /// and was selected for rewriting.  False means the bucket was a no-op.
    had_dirty_pages: bool,
    /// True if a GCJournal was written for this bucket and has not yet been deleted.
    /// The journal is deleted by the caller after the memtable is flushed to disk.
    had_journal: bool,
}

impl KVStore {
    /// Enable or disable re-verifying each value's CRC32 on read (default off).
    /// See `DbConfig::verify_checksums_on_read`.
    pub fn set_verify_checksums_on_read(&self, verify: bool) {
        self.value_log.set_verify_checksums_on_read(verify);
    }

    /// Open a KVStore for the given namespace at the specified base directory.
    /// The directory will contain `lsm/` and `value_logs/` subdirectories.
    pub fn open(namespace_id: u32, name: &str, base_path: &Path, lsm_config: LSMConfig, sync_config: SyncConfig) -> Result<Self> {
        Self::open_with_ttl(namespace_id, name, base_path, lsm_config, sync_config, None)
    }

    /// Open a KVStore with an optional TTL for automatic record expiry.
    pub fn open_with_ttl(
        namespace_id: u32,
        name: &str,
        base_path: &Path,
        lsm_config: LSMConfig,
        sync_config: SyncConfig,
        ttl: Option<Duration>,
    ) -> Result<Self> {
        std::fs::create_dir_all(base_path)?;

        let lsm_path = base_path.join("lsm");
        let value_log_path = base_path.join("value_logs");
        let metadata_path = base_path.join("metadata");

        // Open LSM tree
        let mut lsm_cfg = lsm_config;
        lsm_cfg.data_dir = lsm_path.clone();
        let num_buckets = lsm_cfg.num_buckets;
        let lsm = Arc::new(LSMTree::open(&lsm_path, lsm_cfg)?);

        // Recover any interrupted GC value-log swap at the FILE level BEFORE the
        // value log opens the bucket files — a revert (restoring the old file)
        // must happen before the fd is bound to the new inode. Completes a
        // committed swap forward, or reverts one whose journal is unreadable.
        Self::recover_gc_swaps(&value_log_path, num_buckets)?;

        // Open sharded value log
        let value_log = ShardedValueLog::open(&value_log_path, num_buckets).map_err(|e| {
            KVError::Io(std::io::Error::other(format!(
                "Failed to open sharded value log for namespace '{}': {:?}",
                name, e
            )))
        })?;

        // Replay any GC journals left from a prior crash (before anything else reads the LSM).
        let gc_replayed = Self::recover_gc_journals(&lsm, &value_log_path, num_buckets)?;
        if gc_replayed > 0 {
            info!("[KVStore '{}'] Recovered {} stale LSM pointers from GC journal", name, gc_replayed);
        }

        // Load or initialize value log metadata
        let metadata = if metadata_path.exists() {
            let data = std::fs::read(&metadata_path)?;
            match ValueLogMetadata::from_file_bytes(&data) {
                Ok(m) => m,
                Err(_) => {
                    let backup = metadata_path.with_extension("corrupt");
                    let _ = std::fs::rename(&metadata_path, &backup);
                    ValueLogMetadata::new()
                }
            }
        } else {
            ValueLogMetadata::new()
        };

        let lsm_compaction_trigger = Arc::new(RwLock::new(None));

        Ok(Self {
            namespace_id,
            name: name.to_string(),
            lsm,
            lsm_path,
            value_log,
            value_log_path,
            metadata_path,
            metadata: Arc::new(RwLock::new(metadata)),
            sync_config,
            write_count: Arc::new(AtomicU64::new(0)),
            value_log_gc_in_progress: Arc::new(AtomicBool::new(false)),
            value_pending_old_logs: Arc::new(RwLock::new(Vec::new())),
            lsm_compaction_trigger,
            ttl,
            namespace_index: Arc::new(RwLock::new(NamespaceIndexSet::new())),
            row_id_fn: Arc::new(RwLock::new(None)),
            row_to_key_fn: Arc::new(RwLock::new(None)),
            rowmap: Arc::new(RwLock::new(None)),
            seq_counter: RwLock::new(Arc::new(AtomicU64::new(1))),
            metrics: OnceLock::new(),
        })
    }

    /// Attach the engine-wide operational counters, propagating them to the LSM
    /// tree as well (called once by `Database` after open).
    pub(crate) fn set_metrics(&self, metrics: Arc<crate::db::metrics::Metrics>) {
        self.lsm.set_metrics(metrics.clone());
        let _ = self.metrics.set(metrics);
    }

    /// The operational counters, if attached. `None` for a standalone store.
    #[inline]
    pub(crate) fn metrics(&self) -> Option<&crate::db::metrics::Metrics> {
        self.metrics.get().map(|m| m.as_ref())
    }

    /// Share a global write-sequence counter with this store (called by the
    /// `Database` coordinator with its WAL sequence counter, so all writes —
    /// WAL-backed and non-WAL — draw from one monotonic sequence space).
    pub(crate) fn set_seq_counter(&self, counter: Arc<AtomicU64>) {
        *self.seq_counter.write() = counter;
    }

    /// Allocate the next write sequence (truncated to the skip list's `u32`
    /// sequence width; serial-number comparison tolerates the truncation).
    pub(crate) fn alloc_seq(&self) -> u32 {
        self.seq_counter.read().fetch_add(1, Ordering::Relaxed) as u32
    }

    // ── Row-ID resolution ──────────────────────────────────────────────

    /// Load (or create) the dense row-ID map for this namespace from `dir`.
    ///
    /// Idempotent: a no-op once the map is loaded. Called when the first field
    /// index is activated, before any replay, so the write/replay paths can
    /// resolve through it.
    pub(crate) fn ensure_rowmap(&self, dir: &Path) -> Result<()> {
        let mut guard = self.rowmap.write();
        if guard.is_none() {
            *guard = Some(index::RowMap::open(dir).map_err(KVError::Io)?);
        }
        Ok(())
    }

    /// Resolve the row ID for `key`, **allocating** a new dense ID if the key is
    /// unseen. Used on the put and WAL-replay paths.
    ///
    /// Precedence: a caller-supplied [`RowIdFn`] (the escape hatch for keys that
    /// embed their own ID) wins; otherwise the dense [`RowMap`](index::RowMap).
    ///
    /// Only reached once a namespace has an active field index, and
    /// `activate_field_index` loads the row map (`ensure_rowmap`) before
    /// activating any field — so exactly one of the two sources is always
    /// present. If neither is (which should be impossible), we log an error and
    /// return `0` rather than crash the write path.
    pub(crate) fn resolve_row_id_alloc(&self, key: &[u8]) -> u128 {
        if let Some(f) = self.row_id_fn.read().as_ref() {
            return f(key);
        }
        if let Some(rm) = self.rowmap.write().as_mut() {
            return rm.get_or_alloc(key);
        }
        error!(
            "[KVStore '{}'] resolve_row_id_alloc reached with neither a row_id_fn nor a loaded \
             RowMap despite an active field index — indexing may be inconsistent for this key",
            self.name
        );
        0
    }

    /// Resolve the row ID for `key` **without allocating** — returns `None` when
    /// the dense map has never seen the key. Used on the delete and query
    /// fallback paths, where an unseen key simply has nothing indexed.
    ///
    /// Like `resolve_row_id_alloc`, only reached with an active field index, so a
    /// `row_id_fn` or a loaded `RowMap` is always present. If neither is (which
    /// should be impossible), we log an error and return `None` — the caller then
    /// treats the key as having nothing indexed rather than crashing.
    pub(crate) fn resolve_row_id_get(&self, key: &[u8]) -> Option<u128> {
        if let Some(f) = self.row_id_fn.read().as_ref() {
            return Some(f(key));
        }
        if let Some(rm) = self.rowmap.read().as_ref() {
            return rm.get(key);
        }
        error!(
            "[KVStore '{}'] resolve_row_id_get reached with neither a row_id_fn nor a loaded \
             RowMap despite an active field index — treating key as unindexed",
            self.name
        );
        None
    }

    /// Resolve a row ID back to its key via the dense map, if loaded.
    pub(crate) fn rowmap_key_for(&self, row_id: u128) -> Option<Vec<u8>> {
        self.rowmap.read().as_ref().and_then(|rm| rm.key_for(row_id))
    }

    /// True when the dense row map is the active ID resolver (no custom
    /// `row_id_fn`, and the map is loaded) — i.e. `rowmap_key_for` can resolve
    /// query hits.
    pub(crate) fn rowmap_active(&self) -> bool {
        self.row_id_fn.read().is_none() && self.rowmap.read().is_some()
    }

    /// True when the dense row map is loaded but has never allocated an ID
    /// (`next_id == 0`). A namespace driven by a custom `row_id_fn` never touches
    /// the row map, so an empty map alongside indexed data is a signal that the
    /// custom resolver was not installed before the index was built.
    pub(crate) fn rowmap_is_empty(&self) -> bool {
        self.rowmap.read().as_ref().is_some_and(|rm| rm.is_empty())
    }

    /// Flush the dense row map at `wal_offset` (the checkpoint WAL tail). Must be
    /// called **before** flushing any field index so the map stays at least as
    /// durable as every persisted bitmap bit.
    pub(crate) fn flush_rowmap(&self, wal_offset: u64) -> Result<()> {
        if let Some(rm) = self.rowmap.read().as_ref() {
            rm.flush(wal_offset).map_err(KVError::Io)?;
        }
        Ok(())
    }

    /// Register a custom row-ID function and its inverse for this namespace.
    ///
    /// After this call every key passed through the index write path is
    /// assigned the row ID returned by `row_id_fn` instead of the default
    /// dense row-ID map.  Providing `row_to_key_fn` (the inverse) also enables
    /// O(|hits|) query resolution in `Database::query_keys` — the inverse
    /// reconstructs each matching key directly from its row ID, requiring
    /// zero extra memory and no maintenance on writes.
    ///
    /// `row_id_fn` must be injective: distinct keys must produce distinct
    /// row IDs so that `row_to_key_fn` can invert it exactly.
    pub fn set_row_id_fn(&self, f: RowIdFn, inv: Option<RowToKeyFn>) -> Result<()> {
        *self.row_id_fn.write() = Some(f);
        *self.row_to_key_fn.write() = inv;
        Ok(())
    }

    /// Pre-value-log-open recovery of an interrupted GC swap (file level only).
    ///
    /// MUST run before [`ShardedValueLog::open`] opens the bucket files: once a
    /// file's fd is bound to its inode, a later rename of `bucket_path` would not
    /// be observed, so any *revert* has to happen here, before the open.
    ///
    /// Drives the swap to a consistent on-disk state using the commit marker as
    /// the durable "swap committed" signal (see [`GCJournal::commit_marker_path`]):
    /// - **marker present + journal readable** → the swap committed; complete the
    ///   file rename *forward* idempotently so `bucket_path` is the new file. The
    ///   journal + marker are left for [`recover_gc_journals`] to apply the LSM
    ///   pointer updates and clean up.
    /// - **marker present + journal unreadable** → the swap committed but its LSM
    ///   updates cannot be reconstructed; **revert** to the preserved old file so
    ///   the (still-stale) LSM pointers match it again. The GC run is undone and a
    ///   future GC retries.
    /// - **marker absent** → the swap never committed (crash before the marker) or
    ///   fully finalized (marker dropped after the LSM flush). `bucket_path` is
    ///   authoritative either way; discard staged leftovers and any journal — a
    ///   journal without a marker must **not** be replayed (its pointers do not
    ///   match `bucket_path`).
    ///
    /// Invariant after this pass: `bucket_path` is fully old or fully new, and a
    /// journal survives to [`recover_gc_journals`] only when its swap committed
    /// and `bucket_path` is the new file — so replay is always safe.
    fn recover_gc_swaps(value_log_path: &Path, num_buckets: usize) -> Result<()> {
        if !value_log_path.exists() {
            return Ok(());
        }

        for bucket in 0..num_buckets as u32 {
            let journal_path = GCJournal::journal_path(value_log_path, bucket);
            let marker_present = GCJournal::commit_marker_exists(value_log_path, bucket);
            let has_journal = journal_path.exists();
            if !marker_present && !has_journal {
                continue;
            }

            let bucket_path = value_log_path.join(format!("value_log_{}.log", bucket));
            let new_path = value_log_path.join(format!("value_log_{}.log.new", bucket));
            let old_path = value_log_path.join(format!("value_log_{}.log.old", bucket));

            if marker_present {
                let journal_ok = has_journal && GCJournal::read(&journal_path).is_some();
                if journal_ok {
                    // Complete the swap forward so `bucket_path` is the new file.
                    // Idempotent: if the renames already finished, `new_path` is
                    // gone and this is a no-op.
                    if new_path.exists() {
                        if bucket_path.exists() && !old_path.exists() {
                            std::fs::rename(&bucket_path, &old_path)?;
                        }
                        std::fs::rename(&new_path, &bucket_path)?;
                    }
                    // Journal + marker kept for recover_gc_journals.
                } else {
                    // Revert to the preserved old file; the stale LSM pointers
                    // match it. Restores `bucket_path` whether the renames were
                    // done (old_path holds the old file) or not (bucket_path is
                    // already old; old_path absent → nothing to restore).
                    if old_path.exists() {
                        std::fs::rename(&old_path, &bucket_path)?;
                    }
                    let _ = std::fs::remove_file(&new_path);
                    let _ = std::fs::remove_file(&journal_path);
                    GCJournal::delete_commit_marker(value_log_path, bucket)?;
                }
            } else {
                // No commit marker — discard staged leftovers and any journal.
                let _ = std::fs::remove_file(&new_path);
                let _ = std::fs::remove_file(&journal_path);
            }

            fsync_dir(value_log_path)?;
        }

        Ok(())
    }

    /// Replay GC journals left by a crash, applying their LSM pointer updates.
    ///
    /// Runs *after* [`recover_gc_swaps`] (and after the value log is open), so by
    /// here every surviving journal belongs to a committed swap whose `bucket_path`
    /// is already the new file — replaying its new pointers is therefore safe. We
    /// fix the LSM pointer for each key, then delete the journal and its commit
    /// marker. This is O(journal entries), not O(all keys).
    fn recover_gc_journals(lsm: &LSMTree, value_log_path: &Path, num_buckets: usize) -> Result<usize> {
        let journals = GCJournal::find_journals(value_log_path);
        if journals.is_empty() {
            return Ok(0);
        }

        let mut total_replayed = 0usize;

        for journal_path in &journals {
            let (bucket, entries) = match GCJournal::read(journal_path) {
                Some(parsed) => parsed,
                None => {
                    // Should be unreachable: recover_gc_swaps already reverted or
                    // discarded every unreadable journal. Defensively drop it.
                    let _ = std::fs::remove_file(journal_path);
                    continue;
                }
            };

            // Replay each entry: fix the LSM pointer for this key to its new
            // (post-swap) location — but ONLY if the key currently still has a
            // live pointer. If the key was deleted before the crash (its
            // tombstone is now the live state), re-inserting the journalled
            // pointer would resurrect it. The journal exists to repair *stale*
            // pointers into the swapped-away file, not to revive deletions, so a
            // key that reads as absent is left deleted.
            for entry in &entries {
                // Skip keys that are now absent/deleted (don't resurrect), and
                // re-point survivors under their *existing* sequence so the
                // relocation preserves the key's version.
                let Some((_, seq)) = lsm.get_with_seq(&entry.key)? else {
                    continue;
                };
                let new_ptr = ShardedValuePointer::new(entry.new_bucket, entry.new_page_offset, entry.new_segment_id, num_buckets);
                if let Ok(ptr) = new_ptr {
                    lsm.insert_with_seq(&entry.key, ptr.to_u128(), seq)?;
                }
            }

            total_replayed += entries.len();

            // Journal replayed — drop it and its commit marker.
            let _ = std::fs::remove_file(journal_path);
            let _ = GCJournal::delete_commit_marker(value_log_path, bucket);
        }

        if total_replayed > 0 {
            // Flush LSM to ensure replayed pointers are persisted.
            lsm.flush_and_compact_all()?;
        }

        Ok(total_replayed)
    }

    /// Set the LSM flush observer (used to wire up WAL persistence callbacks)
    pub(crate) fn set_flush_observer(&self, observer: Option<Arc<dyn LsmFlushObserver>>) {
        self.lsm.set_flush_observer(observer);
    }

    /// Set the LSM compaction trigger channel
    pub fn set_compaction_trigger(&self, sender: tokio::sync::mpsc::UnboundedSender<LsmCompactionCommand>) {
        *self.lsm_compaction_trigger.write() = Some(sender);
    }

    // ── Core data operations ───────────────────────────────────────────

    /// Put a key-value pair into storage (LSM + value log).
    /// WAL writing is handled by the caller (Database coordinator).
    ///
    /// Allocates the write sequence from the store's counter. Used by non-WAL
    /// writes (TTL expiry, bulk load) and standalone tests. The WAL-backed path
    /// uses [`put_to_storage_seq`](Self::put_to_storage_seq) so the conflict-
    /// resolution sequence equals the WAL append order (live == recovery).
    pub fn put_to_storage(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.put_to_storage_inner(key, value, self.alloc_seq() as u64)
    }

    /// Put a key-value pair carrying an explicit global write sequence, so
    /// concurrent writes to the same key resolve highest-sequence-wins
    /// (live == recovery). `seq` is the full global u64 sequence; the LSM stores
    /// its low 32 bits, while the value record stores the full width.
    pub fn put_to_storage_seq(&self, key: &[u8], value: &[u8], seq: u64) -> Result<()> {
        self.put_to_storage_inner(key, value, seq)
    }

    fn put_to_storage_inner(&self, key: &[u8], value: &[u8], seq: u64) -> Result<()> {
        let epoch = current_epoch_millis();
        let mut next_version = 1u32;
        // Capture the prior value's bytes so field-index updates can be O(1)
        // (targeted at the old value's bucket) instead of scanning every bucket.
        // Only worth a value-log read when the namespace actually has indexes.
        let want_old_for_index = !self.namespace_index.read().is_empty();
        let mut old_value: Option<Vec<u8>> = None;
        let mut key_existed = false;
        if let Some(existing) = self.lsm.get(key)?
            && let Some(existing_ptr) = decode_sharded_pointer(existing)
        {
            key_existed = true;
            if let Ok(meta) = self.value_log.read_record_meta(existing_ptr) {
                next_version = meta.version.saturating_add(1);
            }
            if want_old_for_index {
                old_value = self.value_log.read_value(existing_ptr).ok();
            }
            let _ = self.value_log.update_record_meta(existing_ptr, None, Some(true), Some(epoch));
        }

        let record_meta = ValueRecordMeta {
            version: next_version,
            tombstone: false,
            updated: false,
            epoch,
            seq,
        };

        // Hold the bucket lock across the value-log write AND the LSM insert so that
        // GC cannot swap the file between the two operations and produce a stale pointer.
        let bucket = self.value_log.bucket_for_key(key);
        let _bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;
        let sharded_pointer = self.value_log.write_record_to_locked_bucket(bucket, value, record_meta, false)?;
        let offset_u128 = sharded_pointer.to_u128();
        self.lsm.insert_with_seq(key, offset_u128, seq as u32)?;
        drop(_bucket_guard);

        // Update in-memory field indices
        self.update_indices_on_put(key, value, old_value.as_deref(), key_existed);

        Ok(())
    }

    /// Update field indices for a put.
    ///
    /// `old_value` is the document's *prior* stored bytes when this put replaced
    /// an existing key (`None` for a fresh insert, or when the prior bytes could
    /// not be read). Supplying it lets each field do an `O(1)` targeted update
    /// (move the row from its old value's bucket to the new one) instead of the
    /// `O(distinct values)` scan `remove_all_for_row` performs — a big win for
    /// high-cardinality fields. When the prior bytes are unavailable on a replace
    /// we fall back to the scan so the row never lingers in a stale bucket.
    fn update_indices_on_put(&self, key: &[u8], value: &[u8], old_value: Option<&[u8]>, key_existed: bool) {
        let ns_index = self.namespace_index.read();
        if ns_index.is_empty() {
            return;
        }
        // Resolve only after confirming there are indexed fields — an unindexed
        // namespace has no row map loaded and nothing to resolve for.
        let row_id = self.resolve_row_id_alloc(key);
        for entry in ns_index.iter() {
            let new_val = (entry.extractor)(value);
            let mut idx = entry.index.write();
            let result = match (key_existed, old_value) {
                // Replace with the prior bytes in hand → O(1) targeted update.
                (true, Some(ov)) => {
                    let old_val = (entry.extractor)(ov);
                    idx.update(old_val.as_ref(), new_val.as_ref(), row_id)
                }
                // Fresh insert: the row is in no bucket yet, so skip the scan.
                (false, _) => match &new_val {
                    Some(v) => idx.insert(v, row_id),
                    None => Ok(()),
                },
                // Replace but prior bytes unreadable: scan to clear the old bucket,
                // then insert the new value, preserving one-value-per-row.
                (true, None) => match &new_val {
                    Some(v) => idx.set(v, row_id),
                    None => {
                        idx.remove_all_for_row(row_id);
                        Ok(())
                    }
                },
            };
            if let Err(e) = result {
                warn!("[KVStore '{}'] Index update rejected for field {}: {}", self.name, entry.field_id, e);
            }
        }
    }

    /// Reindex a single field for a single key: re-derive the field value from
    /// the key's *current* stored bytes and rewrite its entry in that field's
    /// index, using the same extractor + `DynFieldIndex` ops as the put path
    /// (clear the row's old buckets, then insert the freshly-extracted value).
    ///
    /// This touches only the named field — it does not rewrite the stored value,
    /// re-run other fields' extractors, or trigger any vector re-embedding. Use
    /// it to repair a single document's entry in one field index.
    ///
    /// Returns [`FieldReindexOutcome`]: `Reindexed` on success, `KeyNotFound`
    /// when the key has no value, `FieldNotActive` when `field_id` has no live
    /// index in this namespace.
    pub fn reindex_field(&self, field_id: crate::db::namespace::FieldId, key: &[u8]) -> Result<crate::db::namespace::FieldReindexOutcome> {
        use crate::db::namespace::FieldReindexOutcome;

        let value = match self.get(key)? {
            Some(v) => v,
            None => return Ok(FieldReindexOutcome::KeyNotFound),
        };

        let ns_index = self.namespace_index.read();
        let Some(entry) = ns_index.get(field_id) else {
            return Ok(FieldReindexOutcome::FieldNotActive);
        };

        // Same row-ID resolution and per-field ops as `update_indices_on_put`'s
        // replace-without-prior-bytes branch: `set` clears the row's existing
        // buckets (preserving one-value-per-row) and inserts the new value.
        let row_id = self.resolve_row_id_alloc(key);
        let new_val = (entry.extractor)(&value);
        let mut idx = entry.index.write();
        match &new_val {
            Some(v) => idx.set(v, row_id).map_err(|e| KVError::Io(std::io::Error::other(e)))?,
            None => idx.remove_all_for_row(row_id),
        }
        Ok(FieldReindexOutcome::Reindexed)
    }

    /// Maximum number of generation-stable read attempts before falling back to
    /// a lock-serialised read. A read only loops when a GC file swap lands in
    /// the exact window between sampling the LSM pointer and the value, so in
    /// practice it succeeds on the first attempt; the cap only bounds the
    /// pathological case of GC churning a single bucket continuously.
    const MAX_GENERATION_READ_ATTEMPTS: usize = 8;

    /// Get a value by key.
    ///
    /// Reads are lock-free on the common path. Concurrent GC mutates two
    /// structures the read depends on — the value-log file (swapped under the
    /// bucket lock, bumping the bucket `generation`)
    /// and the LSM's SSTable files (swapped during L0→L1 compaction). Either can
    /// momentarily make a freshly-read pointer inconsistent with the file it is
    /// read from, surfacing as a transient error or a stale value.
    ///
    /// The fast path is therefore *optimistic*: we sample the bucket generation
    /// before and after the pointer+value read and only trust a clean success
    /// when the generation is unchanged. Any error, or a generation change, is
    /// treated as a transient race and retried. After
    /// `MAX_GENERATION_READ_ATTEMPTS` we fall back to a read that holds the
    /// bucket lock (excluding value-log GC) — the authoritative path that
    /// guarantees forward progress and surfaces genuine corruption.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let result = self.get_inner(key);
        if let Some(m) = self.metrics() {
            // Count the read; a hit is a successful lookup that found a live value.
            m.record_read(matches!(result, Ok(Some(_))));
        }
        result
    }

    fn get_inner(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let bucket = self.value_log.bucket_for_key(key);

        for _ in 0..Self::MAX_GENERATION_READ_ATTEMPTS {
            let gen_before = self.value_log.bucket_generation(bucket);
            // Odd generation = a GC swap is in progress on this bucket, so the
            // LSM pointer and the value-log file may disagree right now. Retry.
            if !gen_before.is_multiple_of(2) {
                continue;
            }

            // A transient I/O error from the LSM (e.g. an SSTable file being
            // swapped by compaction) is retryable, not fatal. Resolve the pointer
            // together with its write `seq` so we can verify the value record
            // still belongs to this write.
            let (offset_u128, lsm_seq) = match self.lsm.get_with_seq(key) {
                Ok(Some((o, s))) => (o, s),
                Ok(None) => return Ok(None),
                Err(_) => continue,
            };
            let pointer = match decode_sharded_pointer(offset_u128) {
                Some(p) => p,
                None => return Ok(None),
            };

            match self.value_log.read_value_with_seq(pointer) {
                // Authoritative when: the generation was even and unchanged across
                // the read (no GC swap raced it) AND the value record's seq matches
                // the LSM's seq (the slot was not recycled by GC for a different
                // write). A seq mismatch means a stale/recycled pointer → retry.
                Ok((value, rec_seq)) if self.value_log.bucket_generation(bucket) == gen_before && (rec_seq as u32) == lsm_seq => {
                    return Ok(Some(value));
                }
                _ => continue,
            }
        }

        // Pathological churn: serialise against value-log GC for a consistent
        // read. Holding the bucket lock means no value-log swap can invalidate
        // the pointer we read. The LSM's own SSTable compaction is not covered by
        // this lock (it opens L0 files by path, which a concurrent compaction may
        // remove), so we still tolerate a transient LSM error here by retrying;
        // a value-log read error under the lock, by contrast, is genuine.
        let _bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;
        let mut last_err: Option<KVError> = None;
        for _ in 0..Self::MAX_GENERATION_READ_ATTEMPTS {
            match self.lsm.get(key) {
                Ok(Some(offset_u128)) => {
                    return match decode_sharded_pointer(offset_u128) {
                        Some(pointer) => Ok(Some(self.value_log.read_value(pointer)?)),
                        None => Ok(None),
                    };
                }
                Ok(None) => return Ok(None),
                Err(e) => last_err = Some(e.into()),
            }
        }
        Err(last_err.unwrap_or(KVError::CorruptedLog))
    }

    /// Delete a key from storage (creates tombstone in LSM).
    /// WAL writing is handled by the caller. Allocates the sequence from the
    /// store's counter (used by TTL expiry / standalone tests); the WAL-backed
    /// path uses [`delete_from_storage_seq`](Self::delete_from_storage_seq).
    pub fn delete_from_storage(&self, key: &[u8]) -> Result<()> {
        self.delete_from_storage_inner(key, self.alloc_seq() as u64)
    }

    /// Delete carrying an explicit global write sequence, so a delete racing a
    /// write to the same key resolves highest-sequence-wins (live == recovery).
    pub fn delete_from_storage_seq(&self, key: &[u8], seq: u64) -> Result<()> {
        self.delete_from_storage_inner(key, seq)
    }

    fn delete_from_storage_inner(&self, key: &[u8], seq: u64) -> Result<()> {
        // Hold the bucket lock across the value-log tombstone AND the LSM delete,
        // for the same reason `put_to_storage` does (see its comment): GC scans
        // the LSM for live keys under this lock and then re-inserts the pointers
        // of the records it copied. Without the lock, a delete landing in GC's
        // scan→reinsert window would be undone — GC would copy the about-to-be
        // deleted record and re-insert its pointer *after* our `lsm.delete`,
        // resurrecting the key. Taking the lock serialises us with GC so the
        // delete is either fully visible to GC's scan or applied after GC ends.
        let bucket = self.value_log.bucket_for_key(key);
        let _bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;

        // Read the prior value (under the bucket lock, so GC can't relocate it)
        // so index removal can target the old value's bucket instead of scanning
        // every bucket. Only when the namespace has indexes to update.
        let want_old_for_index = !self.namespace_index.read().is_empty();
        let mut old_value: Option<Vec<u8>> = None;
        if let Some(existing) = self.lsm.get(key)?
            && let Some(existing_ptr) = decode_sharded_pointer(existing)
        {
            if want_old_for_index {
                old_value = self.value_log.read_value(existing_ptr).ok();
            }
            let _ = self
                .value_log
                .update_record_meta(existing_ptr, Some(true), None, Some(current_epoch_millis()));
        }
        self.lsm.delete_with_seq(key, seq as u32)?;
        drop(_bucket_guard);

        // Remove row from all field indices
        self.update_indices_on_delete(key, old_value.as_deref());

        Ok(())
    }

    /// Remove a deleted key's row from the field indices.
    ///
    /// With the deleted document's prior bytes (`old_value`) in hand, each field
    /// removes the row from just its old value's bucket — `O(1)` instead of the
    /// `O(distinct values)` scan. Falls back to the scan only when those bytes
    /// could not be read, so a row never lingers in a stale bucket.
    fn update_indices_on_delete(&self, key: &[u8], old_value: Option<&[u8]>) {
        let ns_index = self.namespace_index.read();
        if ns_index.is_empty() {
            return;
        }
        // No allocation: a key the dense map has never seen has nothing indexed.
        let Some(row_id) = self.resolve_row_id_get(key) else {
            return;
        };
        for entry in ns_index.iter() {
            let mut idx = entry.index.write();
            match old_value {
                // Targeted: the row is under the old value's bucket (if any).
                Some(ov) => {
                    if let Some(old_val) = (entry.extractor)(ov) {
                        idx.remove(&old_val, row_id);
                    }
                    // Field absent in the old doc → row was never indexed for it.
                }
                // Prior bytes unavailable → scan to be safe.
                None => idx.remove_all_for_row(row_id),
            }
        }
    }

    // ── Batch sync ─────────────────────────────────────────────────────

    /// Increment write count and return true if a sync should be triggered
    pub fn should_sync(&self) -> bool {
        let n = self.sync_config.records_per_sync;
        if n == 0 {
            return false;
        }
        let count = self.write_count.fetch_add(1, Ordering::Relaxed) + 1;
        count.is_multiple_of(n as u64)
    }

    pub fn sync_value_log(&self) -> Result<()> {
        self.value_log.sync().map_err(|e| {
            warn!("[KVStore:{}] Value log sync failed: {:?}", self.name, e);
            KVError::from(e)
        })
    }

    // ── Iterator support ───────────────────────────────────────────────

    /// Snapshot the GC swap generation of every value-log bucket.
    fn snapshot_bucket_generations(&self) -> Vec<u64> {
        (0..self.value_log.num_buckets() as u32)
            .map(|b| self.value_log.bucket_generation(b))
            .collect()
    }

    /// True unless every bucket was demonstrably swap-free for the whole window
    /// the `snapshot` brackets: each sampled generation must have been **even**
    /// (no swap in progress when sampled) and **unchanged** now (no swap since).
    /// An odd sample or any change means a GC swap raced the operation.
    fn bucket_generations_unstable(&self, snapshot: &[u64]) -> bool {
        snapshot
            .iter()
            .enumerate()
            .any(|(b, &g)| !g.is_multiple_of(2) || self.value_log.bucket_generation(b as u32) != g)
    }

    /// Run a batch read closure so its result is consistent with a single
    /// value-log generation across every bucket it touched.
    ///
    /// The batch paths sample `(key, pointer)` from the LSM and *then* capture
    /// each bucket's current file handle, so a GC swap landing in between could
    /// pair a pre-swap pointer with the post-swap file and read the wrong record.
    /// We bracket the whole closure with the per-bucket seqlock generation:
    /// snapshot every bucket before it runs, re-check after, and trust the result
    /// only if every sample was **even** (no swap in progress) and **unchanged**
    /// (no swap completed since) — see [`bucket_generations_unstable`](Self::bucket_generations_unstable).
    /// On instability we retry; after [`MAX_GENERATION_READ_ATTEMPTS`](Self::MAX_GENERATION_READ_ATTEMPTS)
    /// we run once more holding every bucket write lock, which excludes GC and so
    /// is guaranteed swap-free.
    ///
    /// This closes the *silent-wrong-value* window (a stale pointer that happens
    /// to decode to a valid record). The complementary *failed-read* window (a
    /// stale pointer that fails to decode) is handled by the callers, which
    /// re-resolve any resolved-but-unreadable key through the single-key
    /// [`get`](Self::get) *outside* this bracket (calling `get` inside would
    /// deadlock against the lock-all fallback).
    fn read_generation_stable<T>(&self, mut op: impl FnMut() -> Result<T>) -> Result<T> {
        for _ in 0..Self::MAX_GENERATION_READ_ATTEMPTS {
            let gens = self.snapshot_bucket_generations();
            let result = op()?;
            if !self.bucket_generations_unstable(&gens) {
                return Ok(result);
            }
        }
        // Pathological GC churn: serialise against GC by holding every bucket
        // write lock (ascending order — GC only ever holds one bucket lock at a
        // time, so this cannot deadlock) so no file swap can race the final run.
        let mut _guards = Vec::with_capacity(self.value_log.num_buckets());
        for b in 0..self.value_log.num_buckets() as u32 {
            _guards.push(self.value_log.lock_bucket_for_write(b)?);
        }
        op()
    }

    pub fn keys(&self) -> Result<Vec<Vec<u8>>> {
        self.lsm.keys().map_err(KVError::from)
    }

    /// Return keys in `[start, end)`. Pass `None` for `end` to scan to the last key.
    pub fn range_keys(&self, start: &[u8], end: Option<&[u8]>) -> Result<Vec<Vec<u8>>> {
        let all = self.lsm.range_keys_bounded(start, usize::MAX)?;
        match end {
            Some(end_key) => Ok(all.into_iter().filter(|k| k.as_slice() < end_key).collect()),
            None => Ok(all),
        }
    }

    /// Return keys that start with `prefix`.
    pub fn scan_prefix_keys(&self, prefix: &[u8]) -> Result<Vec<Vec<u8>>> {
        Ok(self.lsm.scan_prefix(prefix)?.into_iter().map(|(k, _, _)| k).collect())
    }

    pub fn resolve_entries(&self, keys: Vec<Vec<u8>>) -> Result<Vec<ResolvedEntry>> {
        let mut entries = Vec::with_capacity(keys.len());
        for key in keys {
            if let Some((offset_u128, seq)) = self.lsm.get_with_seq(&key)?
                && let Some(pointer) = decode_sharded_pointer(offset_u128)
            {
                let log = self.value_log.get_bucket_log(pointer.bucket)?;
                let file = log.get_file();
                entries.push((key, pointer, file, seq));
            }
        }
        Ok(entries)
    }

    pub fn resolve_entries_from_pointers(&self, key_pointers: Vec<KeyPointer>) -> Result<Vec<ResolvedEntry>> {
        let mut entries = Vec::with_capacity(key_pointers.len());
        for (key, offset_u128, seq) in key_pointers {
            if let Some(pointer) = decode_sharded_pointer(offset_u128) {
                let log = self.value_log.get_bucket_log(pointer.bucket)?;
                let file = log.get_file();
                entries.push((key, pointer, file, seq));
            }
        }
        Ok(entries)
    }

    /// Tombstone records whose creation epoch is older than `ttl`.
    ///
    /// Scans live keys, reads each record's `epoch` straight from the value log,
    /// and deletes those that have aged past `ttl` (already-tombstoned records are
    /// skipped). At most `max_deletes_per_run` records are removed per call so a
    /// single pass can't stall on a huge backlog; the value-log GC reclaims the
    /// physical space afterwards. Returns the number of records tombstoned.
    ///
    /// This is the work the global TTL worker schedules for each TTL-enabled
    /// namespace; the expiry policy lives here on the store that owns the data.
    pub(crate) fn expire_records(&self, ttl: Duration, max_deletes_per_run: usize) -> Result<usize> {
        let now_millis = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
        let ttl_millis = ttl.as_millis() as u64;

        let keys = self.keys()?;
        let entries = self.resolve_entries(keys)?;

        let mut deleted = 0usize;

        for (key, pointer, file, _seq) in entries {
            if deleted >= max_deletes_per_run {
                break;
            }

            let bucket_log = match self.value_log.get_bucket_log(pointer.bucket) {
                Ok(log) => log,
                Err(_) => continue,
            };

            let meta = match bucket_log.read_record_meta_from_file(&file, pointer.location) {
                Ok(m) => m,
                Err(_) => continue,
            };

            // Skip already-tombstoned records.
            if meta.tombstone {
                continue;
            }

            if now_millis.saturating_sub(meta.epoch) >= ttl_millis {
                self.delete_from_storage(&key)?;
                deleted += 1;
            }
        }

        Ok(deleted)
    }

    /// Fetch multiple keys in parallel by grouping I/O across value-log buckets.
    ///
    /// Uses `lsm.get_multiple` to resolve all key→pointer mappings in a single pass
    /// per SSTable bucket (reads each level1 file once into memory), then reads
    /// values from the value-log with `num_buckets` parallel threads.
    ///
    /// Returns one `Option<Vec<u8>>` per input key in the same order.
    /// Missing keys and I/O errors both produce `None`.
    pub fn get_multiple(&self, keys: &[Vec<u8>]) -> Vec<Option<Vec<u8>>> {
        // Bracket the resolve-then-read against GC file swaps (see
        // `read_generation_stable`); the inner read
        // is infallible, so the wrapper only ever errors from the lock fallback,
        // which we treat as "all missing" to preserve the infallible signature.
        let (mut results, retry) = self
            .read_generation_stable(|| Ok(self.get_multiple_inner(keys)))
            .unwrap_or_else(|_| (vec![None; keys.len()], Vec::new()));

        // A key that `lsm.get_multiple` resolved to a pointer but whose value read
        // came back `None` hit the brief GC window where the LSM pointer and the
        // value-log file momentarily disagree. Re-resolve those — and only those —
        // through the single-key `get`, which retries the resolve+read together
        // and falls back to a lock-held read, so it sees a consistent pair. This
        // runs OUTSIDE the generation bracket: `get` may take a bucket lock, which
        // would deadlock against the bracket's lock-all fallback.
        for idx in retry {
            results[idx] = self.get(&keys[idx]).ok().flatten();
        }
        results
    }

    /// Returns `(values, retry_indices)` where `retry_indices` are positions that
    /// `lsm.get_multiple` resolved to a pointer but whose value read returned `None`
    /// (a transient GC race the caller should re-resolve via [`get`](Self::get)).
    fn get_multiple_inner(&self, keys: &[Vec<u8>]) -> (Vec<Option<Vec<u8>>>, Vec<usize>) {
        let mut results: Vec<Option<Vec<u8>>> = vec![None; keys.len()];

        // ── Step 1: single-pass LSM lookup for all keys ───────────────────────
        // Reads each bucket's level1 file ONCE instead of once per key.
        let Ok(pointers) = self.lsm.get_multiple(keys) else {
            return (results, Vec::new());
        };

        struct Entry {
            orig_idx: usize,
            pointer: ShardedValuePointer,
            file: Arc<File>,
            /// LSM's seq for this key — the value record must carry the same low
            /// u32, or the slot was recycled (stale pointer) and must be refetched.
            lsm_seq: u32,
        }
        let mut found: Vec<Entry> = Vec::new();
        for (orig_idx, pointer_opt) in pointers.into_iter().enumerate() {
            if let Some((offset_u128, lsm_seq)) = pointer_opt
                && let Some(pointer) = decode_sharded_pointer(offset_u128)
                && let Ok(log) = self.value_log.get_bucket_log(pointer.bucket)
            {
                found.push(Entry {
                    orig_idx,
                    pointer,
                    file: log.get_file(),
                    lsm_seq,
                });
            }
        }

        if found.is_empty() {
            return (results, Vec::new());
        }
        // Expected seq per original index, to validate each value read against.
        let mut expected_seq: Vec<Option<u32>> = vec![None; keys.len()];
        for e in &found {
            expected_seq[e.orig_idx] = Some(e.lsm_seq);
        }

        // ── Step 2: group by value-log bucket ────────────────────────────────
        let num_buckets = self.value_log.num_buckets();
        let mut bucket_work: Vec<Vec<(usize, ShardedValuePointer, Arc<File>)>> = (0..num_buckets).map(|_| Vec::new()).collect();
        for entry in &found {
            bucket_work[entry.pointer.bucket as usize].push((entry.orig_idx, entry.pointer, entry.file.clone()));
        }

        // ── Step 3: parallel reads per bucket ────────────────────────────────
        // A read counts only if the record's seq matches the LSM's seq for that
        // key; a mismatch means GC recycled the slot for a different write, so we
        // leave the slot `None` and let the caller refetch it via single `get`.
        std::thread::scope(|s| {
            let handles: Vec<_> = bucket_work
                .iter()
                .filter(|b| !b.is_empty())
                .map(|bucket| s.spawn(|| self.value_log.read_values_batch(bucket)))
                .collect();
            for handle in handles {
                if let Ok(batch) = handle.join() {
                    for (idx, val) in batch {
                        if let Some((bytes, rec_seq)) = val
                            && expected_seq[idx] == Some(rec_seq as u32)
                        {
                            results[idx] = Some(bytes);
                        }
                    }
                }
            }
        });

        // Entries we resolved to a pointer but couldn't read, or whose value
        // record's seq did not match (recycled slot) → re-resolve via the
        // single-key `get`, which retries the resolve+read together (and itself
        // validates the seq) under the generation/lock fallback.
        let retry: Vec<usize> = found.iter().map(|e| e.orig_idx).filter(|&idx| results[idx].is_none()).collect();

        (results, retry)
    }

    /// Scan multiple 4-byte BE cluster prefixes in a single pass.
    ///
    /// Uses `lsm.scan_prefixes` to read each bucket's level1
    /// SSTable **once** into memory and check all cluster prefixes in a single
    /// in-memory pass — replacing N_clusters × num_buckets full linear scans
    /// (each with per-entry pread() syscalls) with exactly num_buckets large
    /// reads followed by CPU-only work.
    ///
    /// Returns a map from `prefix_id` to `(key_bytes, value_bytes)` pairs.
    /// Keys shorter than 4 bytes or with no matching pointer are silently skipped.
    ///
    /// Same bracket invariant as [`scan_prefix_batch`](Self::scan_prefix_batch): the
    /// LSM scan + value reads (here in `scan_prefixes_batch_inner`) must stay inside
    /// `read_generation_stable`.
    pub fn scan_prefixes_batch(&self, prefix_ids: &[u32]) -> Result<PrefixBatchResult> {
        let (mut result, retry) = self.read_generation_stable(|| self.scan_prefixes_batch_inner(prefix_ids))?;
        // Re-resolve keys dropped to a transient GC race via the single-key path
        // (outside the bracket — see [`refetch_dropped`](Self::refetch_dropped)).
        for (prefix_id, key) in retry {
            if let Some(v) = self.get(&key)? {
                result.entry(prefix_id).or_default().push((key, v));
            }
        }
        Ok(result)
    }

    fn scan_prefixes_batch_inner(&self, prefix_ids: &[u32]) -> Result<(PrefixBatchResult, Vec<PrefixIdKey>)> {
        // ── Step 1: single-pass LSM scan across all prefixes ──────────────────
        // Reads each bucket's level1 file ONCE instead of once per prefix.
        let prefix_id_set: std::collections::HashSet<u32> = prefix_ids.iter().copied().collect();
        struct Entry {
            prefix_id: u32,
            key: Vec<u8>,
            pointer: ShardedValuePointer,
            file: Arc<File>,
        }
        let mut all_entries: Vec<Entry> = Vec::new();
        let key_pointers = self.lsm.scan_prefixes(&prefix_id_set)?;
        for (key, offset_u128) in key_pointers {
            if key.len() < 4 {
                continue;
            }
            let prefix_id = u32::from_be_bytes(key[..4].try_into().unwrap());
            if let Some(pointer) = decode_sharded_pointer(offset_u128) {
                let log = self.value_log.get_bucket_log(pointer.bucket)?;
                let file = log.get_file();
                all_entries.push(Entry {
                    prefix_id,
                    key,
                    pointer,
                    file,
                });
            }
        }

        if all_entries.is_empty() {
            return Ok((std::collections::HashMap::new(), Vec::new()));
        }

        // ── Step 2: group ALL entries across all prefixes by value-log bucket ──
        let num_buckets = self.value_log.num_buckets();
        let mut bucket_work: Vec<Vec<(usize, ShardedValuePointer, Arc<File>)>> = (0..num_buckets).map(|_| Vec::new()).collect();
        for (idx, entry) in all_entries.iter().enumerate() {
            bucket_work[entry.pointer.bucket as usize].push((idx, entry.pointer, entry.file.clone()));
        }

        // ── Step 3: exactly num_buckets threads read all values ───────────────
        let mut values: Vec<Option<Vec<u8>>> = vec![None; all_entries.len()];
        std::thread::scope(|s| {
            let handles: Vec<_> = bucket_work
                .iter()
                .filter(|b| !b.is_empty())
                .map(|bucket| s.spawn(|| self.value_log.read_values_batch(bucket)))
                .collect();
            for handle in handles {
                if let Ok(results) = handle.join() {
                    for (idx, val) in results {
                        // TODO(scan seq-check): validate val's seq against the LSM seq
                        // once scan resolution carries it (INC2 step 5). For now take
                        // the value; correctness for scans still relies on the
                        // generation bracket until then.
                        values[idx] = val.map(|(bytes, _seq)| bytes);
                    }
                }
            }
        });

        // ── Step 4: group by prefix_id; entries that read None → retry ────────
        let mut result: PrefixBatchResult = std::collections::HashMap::new();
        let mut retry: Vec<(u32, Vec<u8>)> = Vec::new();
        for (entry, val_opt) in all_entries.into_iter().zip(values) {
            match val_opt {
                Some(val) => result.entry(entry.prefix_id).or_default().push((entry.key, val)),
                None => retry.push((entry.prefix_id, entry.key)),
            }
        }
        Ok((result, retry))
    }

    /// Parallel value-log reads for a pre-built entry list.
    ///
    /// Groups entries by value-log bucket and reads each bucket in a dedicated
    /// scoped thread, then zips keys back with their values. Returns
    /// `(pairs, retry_keys)`: `retry_keys` are keys that had a pointer but read
    /// `None` — a transient GC race the caller re-resolves via [`get`](Self::get).
    fn batch_read_from_entries(&self, entries: Vec<ResolvedEntry>) -> Result<(Vec<KeyValue>, Vec<Vec<u8>>)> {
        if entries.is_empty() {
            return Ok((vec![], vec![]));
        }
        let n = entries.len();
        let num_buckets = self.value_log.num_buckets();
        let mut bucket_work: Vec<Vec<(usize, ShardedValuePointer, Arc<File>)>> = (0..num_buckets).map(|_| Vec::new()).collect();
        // Expected seq per entry index, to validate the value record against.
        let mut expected_seq: Vec<u32> = vec![0; n];
        for (idx, (_, pointer, file, seq)) in entries.iter().enumerate() {
            bucket_work[pointer.bucket as usize].push((idx, *pointer, file.clone()));
            expected_seq[idx] = *seq;
        }
        let mut values: Vec<Option<Vec<u8>>> = vec![None; n];
        std::thread::scope(|s| {
            let handles: Vec<_> = bucket_work
                .iter()
                .filter(|b| !b.is_empty())
                .map(|bucket| s.spawn(|| self.value_log.read_values_batch(bucket)))
                .collect();
            for handle in handles {
                if let Ok(results) = handle.join() {
                    for (idx, val) in results {
                        // A value counts only if its record seq matches the LSM
                        // seq; a mismatch means GC recycled the slot for a different
                        // write → leave None so `refetch_dropped` re-resolves it.
                        if let Some((bytes, rec_seq)) = val
                            && rec_seq as u32 == expected_seq[idx]
                        {
                            values[idx] = Some(bytes);
                        }
                    }
                }
            }
        });
        let mut pairs = Vec::with_capacity(n);
        let mut retry = Vec::new();
        for ((key, _, _, _), val) in entries.into_iter().zip(values) {
            match val {
                Some(v) => pairs.push((key, v)),
                None => retry.push(key),
            }
        }
        Ok((pairs, retry))
    }

    /// Re-resolve keys a batch read dropped due to a transient GC race, via the
    /// single-key [`get`](Self::get) (which retries resolve+read together and
    /// falls back to a lock-held read). Must be called OUTSIDE the generation
    /// bracket — `get` may take a bucket lock that would deadlock against the
    /// bracket's lock-all fallback. A key that `get` now reports absent was
    /// concurrently deleted and is correctly left out.
    fn refetch_dropped(&self, pairs: &mut Vec<KeyValue>, retry: Vec<Vec<u8>>) -> Result<()> {
        for key in retry {
            if let Some(v) = self.get(&key)? {
                pairs.push((key, v));
            }
        }
        Ok(())
    }

    /// Scan keys with prefix and return `(key, value)` pairs using batched I/O.
    ///
    /// One LSM pass returns `(key, pointer)` pairs directly — no per-key
    /// `lsm.get()` re-lookup — then values are read in parallel per bucket.
    ///
    /// INVARIANT: the LSM scan, the file-handle capture, and the value reads MUST all
    /// stay inside the `read_generation_stable`
    /// closure. Resolving a pointer outside the bracket reopens the wrong-file window
    /// closed in 420ac8e — the LSM scan's own snapshot does not protect against
    /// value-log GC.
    pub fn scan_prefix_batch(&self, prefix: &[u8]) -> Result<Vec<KeyValue>> {
        let (mut pairs, retry) = self.read_generation_stable(|| {
            let key_pointers = self.lsm.scan_prefix(prefix)?;
            let entries = self.resolve_entries_from_pointers(key_pointers)?;
            self.batch_read_from_entries(entries)
        })?;
        self.refetch_dropped(&mut pairs, retry)?;
        if let Some(m) = self.metrics() {
            m.record_scan(pairs.len() as u64);
        }
        Ok(pairs)
    }

    /// Scan keys in `[start, end)` and return `(key, value)` pairs using batched I/O.
    ///
    /// One LSM pass via `range_pointers_bounded` avoids the redundant per-key
    /// `lsm.get()` that the old `range_keys` + `collect_kv_pairs` path performed.
    ///
    /// Same bracket invariant as [`scan_prefix_batch`](Self::scan_prefix_batch): the
    /// LSM scan + value reads must stay inside `read_generation_stable`.
    pub fn scan_range_batch(&self, start: &[u8], end: Option<&[u8]>) -> Result<Vec<KeyValue>> {
        let (mut pairs, retry) = self.read_generation_stable(|| {
            let key_pointers = self.lsm.range_pointers_bounded(start, end, usize::MAX)?;
            let entries = self.resolve_entries_from_pointers(key_pointers)?;
            self.batch_read_from_entries(entries)
        })?;
        self.refetch_dropped(&mut pairs, retry)?;
        if let Some(m) = self.metrics() {
            m.record_scan(pairs.len() as u64);
        }
        Ok(pairs)
    }

    /// Paginated cursor scan over `[cursor, end)` returning at most `limit` pairs
    /// and an optional next cursor.
    ///
    /// Fetches `limit + 1` key-pointer pairs from the LSM in one pass, uses the
    /// extra entry solely to compute the next cursor, then batch-reads values for
    /// the page. `end` (exclusive) bounds the scan — pass `None` to scan to the
    /// last key; a prefix scan supplies the prefix's upper bound here so only the
    /// page's keys (not the whole tail of the keyspace) are resolved.
    ///
    /// Same bracket invariant as [`scan_prefix_batch`](Self::scan_prefix_batch): the
    /// LSM scan + value reads must stay inside `read_generation_stable`.
    pub fn scan_page_batch(&self, cursor: Option<&[u8]>, end: Option<&[u8]>, limit: usize) -> Result<ScanPage> {
        let start = cursor.unwrap_or(&[]);
        let ((mut pairs, retry), next_cursor) = self.read_generation_stable(|| {
            let key_pointers = self.lsm.range_pointers_bounded(start, end, limit + 1)?;
            let has_more = key_pointers.len() > limit;
            let next_cursor = if has_more { Some(key_pointers[limit].0.clone()) } else { None };
            let page_kps: Vec<_> = key_pointers.into_iter().take(limit).collect();
            let entries = self.resolve_entries_from_pointers(page_kps)?;
            let pairs_retry = self.batch_read_from_entries(entries)?;
            Ok((pairs_retry, next_cursor))
        })?;
        self.refetch_dropped(&mut pairs, retry)?;
        if let Some(m) = self.metrics() {
            m.record_scan(pairs.len() as u64);
        }
        Ok((pairs, next_cursor))
    }

    // ── Startup / cleanup ──────────────────────────────────────────────

    /// Cleanup old files from previous runs
    pub fn cleanup_old_files_on_startup(&self) -> Result<()> {
        // Cleanup LSM old SSTable files
        self.lsm
            .cleanup_old_files_on_startup()
            .map_err(|e| KVError::Io(std::io::Error::other(format!("Failed to cleanup LSM old files: {:?}", e))))?;

        // Cleanup ValueLog old files (16 buckets)
        for bucket in 0u32..self.value_log.num_buckets() as u32 {
            let active_path = self.value_log_path.join(format!("value_log_{}.log", bucket));
            let old_value_log = self.value_log_path.join(format!("value_log_{}.log.old", bucket));
            if !old_value_log.exists() {
                continue;
            }

            let probe_ok = self.probe_bucket_readable(bucket);
            if probe_ok {
                match std::fs::remove_file(&old_value_log) {
                    Ok(_) => info!("[STARTUP:{}] Cleaned up old value log: value_log_{}.log.old", self.name, bucket),
                    Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                        warn!("[STARTUP:{}] Failed to cleanup old value log: {:?}", self.name, e);
                    }
                    _ => {}
                }
            } else {
                info!("[STARTUP:{}] GC was interrupted for bucket {}; rolling back", self.name, bucket);
                if let Err(e) = std::fs::remove_file(&active_path)
                    && e.kind() != std::io::ErrorKind::NotFound
                {
                    warn!("[STARTUP:{}] Failed to remove partial GC file: {:?}", self.name, e);
                    continue;
                }
                if let Err(e) = std::fs::rename(&old_value_log, &active_path) {
                    warn!("[STARTUP:{}] Failed to restore old value log: {:?}", self.name, e);
                }
            }
        }

        Ok(())
    }

    fn probe_bucket_readable(&self, bucket: u32) -> bool {
        let keys = match self.lsm.keys() {
            Ok(k) => k,
            Err(_) => return true,
        };
        for key in keys {
            let u128_val = match self.lsm.get(&key) {
                Ok(Some(v)) => v,
                _ => continue,
            };
            let ptr = match decode_sharded_pointer(u128_val) {
                Some(p) => p,
                None => continue,
            };
            if ptr.bucket != bucket {
                continue;
            }
            return self.value_log.read_value(ptr).is_ok();
        }
        true
    }

    // ── Recovery (replay WAL entries into this namespace) ──────────────

    /// Replay a single WAL entry into this namespace's storage, stamping the
    /// memtable with the entry's original WAL sequence so recovery reproduces the
    /// same per-key winner that was live before the crash (highest-sequence-wins).
    pub fn replay_upsert(&self, key: &[u8], value: &[u8], seq: u64) -> Result<()> {
        let epoch = current_epoch_millis();
        let mut next_version = 1u32;
        if let Ok(Some(existing)) = self.lsm.get(key)
            && let Some(existing_ptr) = decode_sharded_pointer(existing)
        {
            if let Ok(meta) = self.value_log.read_record_meta(existing_ptr) {
                next_version = meta.version.saturating_add(1);
            }
            let _ = self.value_log.update_record_meta(existing_ptr, None, Some(true), Some(epoch));
        }

        let record_meta = ValueRecordMeta {
            version: next_version,
            tombstone: false,
            updated: false,
            epoch,
            seq,
        };

        let sharded_pointer = self
            .value_log
            .write_record(key, value, record_meta, false)
            .map_err(|e| KVError::Io(std::io::Error::other(format!("Failed to write to value log during recovery: {:?}", e))))?;
        let offset_u128 = sharded_pointer.to_u128();
        self.lsm.insert_with_seq(key, offset_u128, seq as u32)?;
        Ok(())
    }

    /// Replay a delete WAL entry into this namespace's storage, stamped with the
    /// entry's original WAL sequence (see [`replay_upsert`](Self::replay_upsert)).
    pub fn replay_delete(&self, key: &[u8], seq: u64) -> Result<()> {
        if let Ok(Some(existing)) = self.lsm.get(key)
            && let Some(existing_ptr) = decode_sharded_pointer(existing)
        {
            let _ = self
                .value_log
                .update_record_meta(existing_ptr, Some(true), None, Some(current_epoch_millis()));
        }
        self.lsm.delete_with_seq(key, seq as u32)?;
        Ok(())
    }

    /// Flush all recovered entries to SSTables
    pub fn flush_and_compact_all(&self) -> Result<()> {
        self.lsm.flush_and_compact_all().map_err(KVError::from)
    }

    // ── Garbage collection ─────────────────────────────────────────────

    /// Compact a single bucket with selective page compaction (Optimization 4).
    ///
    /// Scans per-page garbage stats. Clean pages (garbage < 50%) are block-copied
    /// to the new file at the same offset — their pointers remain valid, so no LSM
    /// updates are needed. Dirty pages are rewritten with only live records into
    /// fresh pages appended at the end, requiring LSM pointer updates.
    fn compact_single_bucket(&self, bucket: u32, reclaim_cutoff_epoch: u64, page_gc_threshold_pct: f64) -> Result<BucketGCResult> {
        let old_meta = self.value_log.get_bucket_metadata(bucket)?;

        let bucket_path = self.value_log.base_path().join(format!("value_log_{}.log", bucket));
        let new_path = self.value_log.base_path().join(format!("value_log_{}.log.new", bucket));
        let old_path = bucket_path.with_extension("log.old");

        if old_path.exists() {
            let _ = std::fs::remove_file(&old_path);
        }

        // Acquire the bucket lock before scanning LSM so that no concurrent write
        // can write to the value log and update the LSM between the two steps.
        // Any write that already holds this lock will complete (both value-log write
        // and LSM insert) before we proceed, giving us a consistent snapshot.
        let _bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;

        // Scan LSM for entries belonging to this bucket under the lock. We keep
        // the *original* encoded pointer (`offset_u128`) alongside the decoded
        // form: the reinsert step below uses it as a compare-and-set witness so
        // a key that was deleted or overwritten since this snapshot is not
        // resurrected by GC's relocation.
        let kv_pairs = self.lsm.key_pointer_pairs(None)?;
        let entries: Vec<(Vec<u8>, u128, ShardedValuePointer)> = kv_pairs
            .into_iter()
            .filter_map(|(key, offset_u128)| {
                decode_sharded_pointer(offset_u128)
                    .filter(|p| p.bucket == bucket)
                    .map(|p| (key, offset_u128, p))
            })
            .collect();
        let log = self.value_log.get_bucket_log(bucket)?;
        let old_file = log.get_file();

        let page_stats = log.scan_all_page_stats(&old_file, &old_meta)?;

        // Classify pages as dirty (needs rewrite) or clean (copy as-is)
        let mut dirty_pages: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for stats in &page_stats {
            if stats.garbage_ratio_pct() >= page_gc_threshold_pct {
                dirty_pages.insert(stats.page_offset);
            }
        }

        // If no pages are dirty, skip this bucket entirely
        if dirty_pages.is_empty() {
            drop(_bucket_guard);
            return Ok(BucketGCResult {
                bytes_reclaimed: 0,
                bytes_live: old_meta.live_bytes,
                old_path: None,
                new_metadata: old_meta.clone(),
                old_metadata: old_meta,
                bucket,
                had_dirty_pages: false,
                had_journal: false,
            });
        }

        let new_log = ValueLog::open(&new_path)?;
        let new_file = new_log.get_file();

        // Track live/garbage bytes for the new file
        let mut new_live_bytes = 0u64;
        let mut new_garbage_bytes = 0u64;

        // Step 1: Copy clean pages as-is at their original offsets.
        // Pointers (page_offset + segment_id) remain valid → no LSM updates needed.
        for stats in &page_stats {
            if !dirty_pages.contains(&stats.page_offset) {
                ValueLog::copy_page_raw(&old_file, &new_file, stats.page_offset)?;
                new_live_bytes += stats.live_bytes;
                new_garbage_bytes += stats.garbage_bytes;
            }
        }

        // Step 2: Rewrite dirty page records into fresh pages.
        //
        // Start the rewrite just past the highest CLEAN page we copied in step 1
        // (or at offset 0 when every page is dirty). This is safe — every clean
        // copy sits at or below that offset, so the survivors can never collide
        // with one — and it keeps the file compact: the survivors reuse the freed
        // pages above the clean region each cycle instead of being appended past
        // `tail` forever. Appending at `tail` (the old behaviour) made an
        // overwrite/delete-heavy value log grow without bound — every GC
        // abandoned the vacated pages as holes and pushed `tail` up by the churn.
        let fresh_page_start = page_stats
            .iter()
            .filter(|s| !dirty_pages.contains(&s.page_offset))
            .map(|s| s.page_offset)
            .max()
            .map(|highest_clean| highest_clean.saturating_add(PAGE_SIZE_BYTES))
            .unwrap_or(0);
        let mut new_meta = ValueLogMetadata {
            head: 0,
            // tail must always be current_page_offset + PAGE_SIZE_BYTES.
            // Initialising it equal to current_page_offset causes write_record to
            // overwrite the current page when the page boundary is first crossed.
            tail: fresh_page_start.saturating_add(PAGE_SIZE_BYTES),
            current_page_offset: fresh_page_start,
            current_page_free_offset: 0, // will be set by ensure_current_page
            current_page_table_offset: PAGE_SIZE_BYTES as u32,
            current_page_next_segment_id: 1,
            total_gc_runs: 0,
            total_bytes_reclaimed: 0,
            live_bytes: 0,
            garbage_bytes: 0,
        };
        // Initialize the first fresh page
        new_log.ensure_current_page(&mut new_meta)?;

        // Each entry: (key, old encoded pointer, new pointer). The old pointer is
        // the compare-and-set witness used at reinsert time.
        let mut lsm_updates: Vec<(Vec<u8>, u128, ShardedValuePointer)> = Vec::new();

        for (key, old_u128, pointer) in entries.iter() {
            // Only process entries that point to dirty pages
            if !dirty_pages.contains(&pointer.location.page_offset) {
                continue;
            }

            let value = match self.value_log.read_value_with_file(*pointer, &old_file) {
                Ok(v) => v,
                Err(_) => {
                    let latest = self.lsm.get(key)?.ok_or(KVError::KeyNotFound)?;
                    let resolved = ShardedValuePointer::from_u128(latest).map_err(|e| {
                        KVError::Io(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("Invalid sharded pointer: {:?}", e),
                        ))
                    })?;
                    match self.value_log.read_value(resolved) {
                        Ok(v) => v,
                        Err(_) => continue,
                    }
                }
            };

            let record_meta = self.value_log.read_record_meta(*pointer).unwrap_or(ValueRecordMeta {
                version: 1,
                tombstone: false,
                updated: false,
                epoch: current_epoch_millis(),
                seq: 0,
            });

            // Drop records that are obsolete (overwritten or tombstoned) as of this
            // GC run. The cutoff is the GC run's wall-clock start, so a record whose
            // epoch is somehow newer than that (backward clock skew) is conservatively
            // retained and reclaimed on a later run instead.
            if (record_meta.tombstone || record_meta.updated) && record_meta.epoch <= reclaim_cutoff_epoch {
                continue;
            }

            // Write live record to fresh pages in the new file
            let new_location = new_log.write_record(&value, record_meta, &mut new_meta, false)?;
            let new_sharded = ShardedValuePointer::new(bucket, new_location.page_offset, new_location.segment_id, self.value_log.num_buckets())?;
            lsm_updates.push((key.clone(), *old_u128, new_sharded));
        }

        // Combine byte counts: clean pages (copied) + fresh pages (rewritten live records)
        new_meta.live_bytes += new_live_bytes;
        new_meta.garbage_bytes += new_garbage_bytes;

        new_log.sync()?;

        // Write GC journal BEFORE the file swap. If we crash after the swap
        // but before LSM updates, the journal is replayed at startup.
        let journal_entries: Vec<(Vec<u8>, u32, u64, u32)> = lsm_updates
            .iter()
            .map(|(key, _old, ptr)| (key.clone(), ptr.bucket, ptr.location.page_offset, ptr.location.segment_id))
            .collect();
        if !journal_entries.is_empty() {
            GCJournal::write(self.value_log.base_path(), bucket, &journal_entries)?;
            // Commit point: the marker makes the swap recoverable. Present at
            // startup ⇒ the swap committed but its LSM pointer updates are not
            // yet durable, so recovery completes the swap forward (replaying the
            // journal) or, if the journal is unreadable, reverts to the preserved
            // old file. It is deleted only after the LSM updates are flushed.
            // Written BEFORE the rename so a crash mid-rename is still recoverable.
            GCJournal::write_commit_marker(self.value_log.base_path(), bucket)?;
        }

        // Open a swap epoch: the bucket generation goes odd here and back to even
        // when `_swap_epoch` drops at the end of this function — AFTER the LSM
        // re-point below. Readers observe the file swap and the re-point as one
        // atomic step (odd generation = "don't trust a read of this bucket"),
        // closing the window where the file is new but the LSM pointer is stale.
        // The guard's Drop runs even on an early `?` return, so the generation
        // can never get stuck odd.
        let _swap_epoch = log.begin_swap();

        // File swap — the dangerous operation. After this point, the old file
        // is gone and the new file is live. If we crash here, the journal
        // will replay the LSM updates at startup.
        if bucket_path.exists() {
            std::fs::rename(&bucket_path, &old_path)?;
        }
        std::fs::rename(&new_path, &bucket_path)?;
        log.swap_file(new_log.get_file());

        // Apply LSM pointer updates immediately after file swap, as a
        // compare-and-set: only re-point a key if it STILL maps to the exact
        // pointer we copied. GC's scan (`key_pointer_pairs`) is a non-atomic
        // multi-layer read, so it can momentarily miss a tombstone that a
        // concurrent delete flushed between layers — without this guard GC would
        // re-insert that key's old value and resurrect the deletion. Validating
        // against the authoritative point-query `get_with_seq` (which honours
        // tombstones across all layers, newest-first) closes that gap: a key that
        // was deleted (None) or overwritten (a different pointer) since the scan
        // is left untouched. We re-point under the key's *existing* sequence so
        // the relocation preserves its version (highest-sequence-wins).
        for (key, old_u128, new_sharded) in &lsm_updates {
            if let Some((cur, seq)) = self.lsm.get_with_seq(key)?
                && cur == *old_u128
            {
                self.lsm.insert_with_seq(key, new_sharded.to_u128(), seq)?;
            }
        }

        // Close the swap epoch (generation → even) now that the file swap AND the
        // LSM re-point are both done — while still under the bucket lock, so the
        // "consistent" publish is serialised against the next swap. Explicit so it
        // happens before the lock is released below rather than at function exit.
        drop(_swap_epoch);

        // The GCJournal is intentionally NOT deleted here.  Deleting it now would
        // be premature: the lsm.insert calls above only updated the in-memory
        // memtable, and a crash before the next memtable flush would lose those
        // updates, leaving stale value-log pointers.
        //
        // The caller (garbage_collect_with_threshold) flushes the memtable to level-0
        // SSTables for all buckets in one shot, and then deletes all journals.

        drop(_bucket_guard);

        let result_old_path = if old_path.exists() { Some(old_path) } else { None };

        Ok(BucketGCResult {
            // Garbage bytes dropped = old total garbage - garbage remaining in clean pages.
            // (fresh pages contain only live records so they contribute 0 garbage.)
            bytes_reclaimed: old_meta.garbage_bytes.saturating_sub(new_meta.garbage_bytes),
            bytes_live: new_meta.live_bytes,
            old_path: result_old_path,
            new_metadata: new_meta,
            old_metadata: old_meta,
            bucket,
            had_dirty_pages: true,
            had_journal: !journal_entries.is_empty(),
        })
    }

    /// Value log garbage collection.
    ///
    /// Obsolete (overwritten/tombstoned) records are reclaimed up to a cutoff
    /// sampled once at the start of the run — the run's wall-clock time — so the
    /// whole pass shares a single, consistent reclaim horizon.
    pub fn garbage_collect_with_threshold(&self, page_gc_threshold_pct: f64) -> Result<GCStats> {
        let reclaim_cutoff_epoch = current_epoch_millis();

        if self
            .value_log_gc_in_progress
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            let (total_gc_runs, total_bytes_reclaimed, bytes_live) = self.aggregate_value_log_stats();
            return Ok(GCStats {
                bytes_reclaimed: 0,
                bytes_live,
                gc_run_count: total_gc_runs,
                total_bytes_reclaimed,
                gc_duration_ms: 0,
            });
        }

        struct GcGuard<'a>(&'a AtomicBool);
        impl<'a> Drop for GcGuard<'a> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        let _guard = GcGuard(&self.value_log_gc_in_progress);
        self.cleanup_pending_logs();

        let start_time = std::time::Instant::now();
        let bucket_count = self.value_log.num_buckets();

        // ── Optimization 2: Pre-compute qualifying buckets before LSM scan ──
        const GC_MIN_GARBAGE_RATIO_PCT: f64 = 10.0;
        let mut qualifying_buckets = vec![false; bucket_count];
        let mut any_qualifying = false;

        for bucket in 0u32..bucket_count as u32 {
            let meta = self.value_log.get_bucket_metadata(bucket)?;
            let total_bytes = meta.live_bytes.saturating_add(meta.garbage_bytes);
            if total_bytes > 0 {
                let garbage_pct = (meta.garbage_bytes as f64 / total_bytes as f64) * 100.0;
                if garbage_pct >= GC_MIN_GARBAGE_RATIO_PCT {
                    qualifying_buckets[bucket as usize] = true;
                    any_qualifying = true;
                }
            }
        }

        // Early return if no buckets need GC
        if !any_qualifying {
            if let Some(m) = self.metrics() {
                Metrics::bump(&m.vlog_gc_runs);
                Metrics::add(&m.vlog_gc_duration_ms, start_time.elapsed().as_millis() as u64);
            }
            let (total_gc_runs, total_bytes_reclaimed, bytes_live) = self.aggregate_value_log_stats();
            return Ok(GCStats {
                bytes_reclaimed: 0,
                bytes_live,
                gc_run_count: total_gc_runs,
                total_bytes_reclaimed,
                gc_duration_ms: start_time.elapsed().as_millis(),
            });
        }

        // ── Parallel bucket compaction using std::thread::scope ──
        // Each thread acquires its own bucket lock and scans the LSM inside it, so
        // there is no pre-scan race between enumeration and compaction.
        let qualifying_indices: Vec<u32> = (0u32..bucket_count as u32).filter(|b| qualifying_buckets[*b as usize]).collect();

        let results: Vec<Result<BucketGCResult>> = std::thread::scope(|s| {
            let handles: Vec<_> = qualifying_indices
                .iter()
                .map(|&bucket| s.spawn(move || self.compact_single_bucket(bucket, reclaim_cutoff_epoch, page_gc_threshold_pct)))
                .collect();

            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        // ── Aggregate results and apply LSM updates sequentially ──
        let mut bytes_reclaimed = 0u64;
        let mut bytes_live = 0u64;
        let mut total_gc_runs = 0u64;
        let mut total_bytes_reclaimed = 0u64;
        let mut old_paths_to_delete: Vec<PathBuf> = Vec::new();
        let mut journal_buckets: Vec<u32> = Vec::new();
        let mut buckets_compacted = 0usize;
        let mut buckets_below_page_threshold = 0usize;

        for result in results {
            let gc_result = result?;

            if gc_result.had_dirty_pages {
                buckets_compacted += 1;
            } else {
                buckets_below_page_threshold += 1;
            }

            if gc_result.had_journal {
                journal_buckets.push(gc_result.bucket);
            }

            // LSM updates were already applied inside compact_single_bucket
            // (right after file swap, under bucket lock) for crash safety.

            // Update bucket metadata
            let updated_meta = ValueLogMetadata {
                head: 0,
                tail: gc_result.new_metadata.tail,
                current_page_offset: gc_result.new_metadata.current_page_offset,
                current_page_free_offset: gc_result.new_metadata.current_page_free_offset,
                current_page_table_offset: gc_result.new_metadata.current_page_table_offset,
                current_page_next_segment_id: gc_result.new_metadata.current_page_next_segment_id,
                total_gc_runs: gc_result.old_metadata.total_gc_runs.saturating_add(1),
                total_bytes_reclaimed: gc_result.old_metadata.total_bytes_reclaimed.saturating_add(gc_result.bytes_reclaimed),
                live_bytes: gc_result.new_metadata.live_bytes,
                garbage_bytes: gc_result.new_metadata.garbage_bytes,
            };
            self.value_log.update_bucket_metadata(gc_result.bucket, updated_meta)?;

            bytes_reclaimed = bytes_reclaimed.saturating_add(gc_result.bytes_reclaimed);
            bytes_live = bytes_live.saturating_add(gc_result.bytes_live);
            total_gc_runs = total_gc_runs.saturating_add(1);
            total_bytes_reclaimed = total_bytes_reclaimed.saturating_add(gc_result.bytes_reclaimed);

            if let Some(old_path) = gc_result.old_path {
                old_paths_to_delete.push(old_path);
            }
        }

        // Flush in-memory LSM pointer updates to level-0 SSTables before
        // deleting the GCJournals.  This ensures a crash after journal deletion
        // but before the next memtable flush cannot produce stale value-log
        // pointers (InvalidLocation on the subsequent restart).
        if !journal_buckets.is_empty() {
            self.lsm
                .flush_memtable_to_level0()
                .map_err(|e| KVError::Io(std::io::Error::other(format!("GC: LSM flush failed: {:?}", e))))?;
            // The LSM pointer updates are now durable, so the swap is complete.
            // Drop the commit marker FIRST (its absence is what tells recovery the
            // updates are durable), then the journal.
            for bucket in &journal_buckets {
                let _ = GCJournal::delete_commit_marker(self.value_log.base_path(), *bucket);
                let _ = GCJournal::delete(self.value_log.base_path(), *bucket);
            }
        }

        if !old_paths_to_delete.is_empty() {
            self.lsm
                .flush_and_compact_all()
                .map_err(|e| KVError::Io(std::io::Error::other(format!("GC: LSM flush failed, old value log files kept: {:?}", e))))?;
        }

        for old_path in old_paths_to_delete {
            if let Err(e) = std::fs::remove_file(&old_path)
                && e.kind() != std::io::ErrorKind::NotFound
            {
                self.value_pending_old_logs.write().push(old_path);
            }
        }

        // Value-log metadata is flushed last (and is NOT rebuilt during GC-swap
        // recovery), so a crash between a swap and this flush leaves the new file
        // paired with the old file's metadata. That is safe, not merely
        // slack-free: on open `ValueLog::ensure_current_page` re-reads the live
        // page header for the within-page cursors, and a compacted file only
        // shrinks, so the stale `current_page_offset` is >= the new file's last
        // page — a subsequent write can never land on a live record. Stale byte
        // accounting only nudges GC trigger timing and is recomputed by the next
        // GC scan; reads ignore these cursors and resolve pointers directly.
        self.value_log.flush_all_metadata()?;

        if buckets_below_page_threshold > 0 && buckets_compacted == 0 {
            info!(
                "[GCWorker] ns='{}' — {} bucket(s) qualified on namespace waste (garbage / total written bytes) \
                 but no single page reached the {:.0}% per-page rewrite threshold (a page is rewritten only when \
                 its own garbage / page bytes crosses it); the dead bytes are spread across pages, none dense \
                 enough to rewrite, so 0 bytes reclaimed this run",
                self.name, buckets_below_page_threshold, page_gc_threshold_pct,
            );
        } else if buckets_below_page_threshold > 0 {
            debug!(
                "[GCWorker] ns='{}' — {} bucket(s) compacted, {} bucket(s) skipped (page threshold not met)",
                self.name, buckets_compacted, buckets_below_page_threshold,
            );
        }

        if let Some(m) = self.metrics() {
            Metrics::bump(&m.vlog_gc_runs);
            Metrics::add(&m.vlog_gc_duration_ms, start_time.elapsed().as_millis() as u64);
        }

        Ok(GCStats {
            bytes_reclaimed,
            bytes_live,
            gc_run_count: total_gc_runs,
            total_bytes_reclaimed,
            gc_duration_ms: start_time.elapsed().as_millis(),
        })
    }

    #[allow(dead_code)]
    pub fn garbage_collect(&self) -> Result<GCStats> {
        self.garbage_collect_with_threshold(crate::db::config::ThresholdConfig::default().value_log_waste_threshold)
    }

    pub fn get_waste_ratio(&self) -> f64 {
        self.value_log.get_total_garbage_ratio()
    }

    /// Aggregate `(garbage_bytes, written_bytes)` across all value-log buckets,
    /// where `written = live + garbage`. The waste ratio reported by
    /// [`get_waste_ratio`](Self::get_waste_ratio) is `garbage / written` — these
    /// are the raw counts behind that percentage, useful for logging so a high
    /// ratio over a tiny absolute volume is obvious.
    pub(crate) fn waste_bytes(&self) -> (u64, u64) {
        let mut garbage = 0u64;
        let mut written = 0u64;
        for (_bucket, metadata) in self.value_log.get_all_bucket_stats() {
            garbage = garbage.saturating_add(metadata.garbage_bytes);
            written = written.saturating_add(metadata.live_bytes).saturating_add(metadata.garbage_bytes);
        }
        (garbage, written)
    }

    #[allow(dead_code)]
    pub fn get_garbage_ratio(&self) -> f64 {
        self.value_log.get_total_garbage_ratio()
    }

    #[allow(dead_code)]
    pub fn get_free_space_ratio(&self) -> f64 {
        self.value_log.get_total_free_space_ratio()
    }

    pub(crate) fn aggregate_value_log_stats(&self) -> (u64, u64, u64) {
        let mut total_gc_runs = 0u64;
        let mut total_bytes_reclaimed = 0u64;
        let mut bytes_live = 0u64;

        for (_bucket, metadata) in self.value_log.get_all_bucket_stats() {
            total_gc_runs = total_gc_runs.saturating_add(metadata.total_gc_runs);
            total_bytes_reclaimed = total_bytes_reclaimed.saturating_add(metadata.total_bytes_reclaimed);
            bytes_live = bytes_live.saturating_add(metadata.live_bytes);
        }

        (total_gc_runs, total_bytes_reclaimed, bytes_live)
    }

    #[allow(dead_code)]
    pub fn stats(&self) -> Stats {
        let mut head = 0u64;
        let mut tail = 0u64;
        let mut total_gc_runs = 0u64;
        let mut total_bytes_reclaimed = 0u64;
        let mut total_live_bytes = 0u64;
        let mut total_garbage_bytes = 0u64;

        for (_bucket, metadata) in self.value_log.get_all_bucket_stats() {
            head = head.saturating_add(metadata.head);
            tail = tail.saturating_add(metadata.tail);
            total_gc_runs = total_gc_runs.saturating_add(metadata.total_gc_runs);
            total_bytes_reclaimed = total_bytes_reclaimed.saturating_add(metadata.total_bytes_reclaimed);
            total_live_bytes = total_live_bytes.saturating_add(metadata.live_bytes);
            total_garbage_bytes = total_garbage_bytes.saturating_add(metadata.garbage_bytes);
        }

        let total_written = total_live_bytes.saturating_add(total_garbage_bytes);
        let garbage_ratio = if total_written > 0 {
            (total_garbage_bytes as f64 / total_written as f64) * 100.0
        } else {
            0.0
        };
        let free_space_ratio = self.value_log.get_total_free_space_ratio();

        Stats {
            head,
            tail,
            garbage_size: total_garbage_bytes,
            waste_ratio: garbage_ratio,
            free_space_ratio,
            total_gc_runs,
            total_bytes_reclaimed,
            live_bytes: total_live_bytes,
        }
    }

    // ── LSM compaction ─────────────────────────────────────────────────

    pub fn compact_lsm(&self) -> Result<()> {
        if !self.lsm.has_compaction_work() {
            return Ok(());
        }
        self.lsm.compact_all().map_err(KVError::from)
    }

    pub fn has_lsm_compaction_work(&self) -> bool {
        self.lsm.has_compaction_work()
    }

    // ── Metadata flush ─────────────────────────────────────────────────

    pub fn flush_metadata(&self) -> Result<()> {
        let metadata = self.metadata.read();
        let bytes = metadata.to_file_bytes()?;
        crate::support::write_atomic_durable(&self.metadata_path, &bytes)?;
        Ok(())
    }

    // ── Shutdown ──────────────────────────────────────────────────────────

    /// Flush and sync all data for this namespace
    pub fn shutdown(&self) -> Result<()> {
        self.cleanup_pending_logs();
        self.lsm.cleanup_pending_memtables_on_close();
        self.lsm.flush_and_compact_all()?;
        self.flush_metadata()?;
        self.value_log.sync()?;
        // Persist per-bucket live/garbage byte counters so they survive restart.
        // Without this, no-WAL bulk loads report 0 bytes after reopen because
        // there is no WAL replay to restore the in-memory counters.
        self.value_log.flush_all_metadata()?;
        Ok(())
    }

    fn cleanup_pending_logs(&self) {
        let mut pending = self.value_pending_old_logs.write();
        pending.retain(|path| std::fs::remove_file(path).is_err());
    }
}

impl Drop for KVStore {
    fn drop(&mut self) {
        let _ = self.lsm.flush_and_compact_all();
        let _ = self.flush_metadata();
        let _ = self.value_log.sync();
        let _ = self.value_log.flush_all_metadata();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn default_lsm_config() -> LSMConfig {
        LSMConfig {
            num_buckets: crate::support::TEST_NUM_BUCKETS,
            ..LSMConfig::default()
        }
    }

    // ── GC swap-commit recovery (recover_gc_swaps) ──────────────────────────
    //
    // These exercise the FILE-level pass directly with distinct "OLD"/"NEW"
    // contents in the .log files, asserting which file lands at `bucket_path`
    // and how the journal/marker are disposed of. They do not need a real
    // value-log format — only which inode becomes live and whether a journal
    // survives to be replayed.
    mod gc_swap_recovery {
        use super::super::*;

        fn lpath(dir: &Path, b: u32) -> PathBuf {
            dir.join(format!("value_log_{}.log", b))
        }
        fn npath(dir: &Path, b: u32) -> PathBuf {
            dir.join(format!("value_log_{}.log.new", b))
        }
        fn opath(dir: &Path, b: u32) -> PathBuf {
            dir.join(format!("value_log_{}.log.old", b))
        }
        fn valid_journal(dir: &Path, b: u32) {
            GCJournal::write(dir, b, &[(b"k".to_vec(), b, 0u64, 1u32)]).unwrap();
        }
        fn corrupt_journal(dir: &Path, b: u32) {
            std::fs::write(GCJournal::journal_path(dir, b), b"not a valid GC journal").unwrap();
        }
        fn read(p: &Path) -> Option<Vec<u8>> {
            std::fs::read(p).ok()
        }

        #[test]
        fn forward_completes_swap_when_renames_unfinished() {
            let dir = tempfile::tempdir().unwrap();
            let d = dir.path();
            std::fs::write(lpath(d, 0), b"OLD").unwrap(); // swap not started: live = old
            std::fs::write(npath(d, 0), b"NEW").unwrap();
            valid_journal(d, 0);
            GCJournal::write_commit_marker(d, 0).unwrap();

            KVStore::recover_gc_swaps(d, 1).unwrap();

            assert_eq!(read(&lpath(d, 0)).as_deref(), Some(b"NEW".as_ref()), "bucket must be the new file");
            assert_eq!(read(&opath(d, 0)).as_deref(), Some(b"OLD".as_ref()), "old preserved at .old");
            assert!(!npath(d, 0).exists(), ".new consumed");
            assert!(GCJournal::journal_path(d, 0).exists(), "journal kept for replay");
            assert!(GCJournal::commit_marker_exists(d, 0), "marker kept for replay");
        }

        #[test]
        fn forward_is_noop_when_renames_already_done() {
            let dir = tempfile::tempdir().unwrap();
            let d = dir.path();
            std::fs::write(lpath(d, 0), b"NEW").unwrap(); // renames already finished
            std::fs::write(opath(d, 0), b"OLD").unwrap();
            valid_journal(d, 0);
            GCJournal::write_commit_marker(d, 0).unwrap();

            KVStore::recover_gc_swaps(d, 1).unwrap();

            assert_eq!(read(&lpath(d, 0)).as_deref(), Some(b"NEW".as_ref()));
            assert!(GCJournal::journal_path(d, 0).exists(), "journal kept for replay");
            assert!(GCJournal::commit_marker_exists(d, 0));
        }

        #[test]
        fn reverts_on_corrupt_journal_after_renames() {
            let dir = tempfile::tempdir().unwrap();
            let d = dir.path();
            std::fs::write(lpath(d, 0), b"NEW").unwrap(); // swap done, live = new
            std::fs::write(opath(d, 0), b"OLD").unwrap();
            corrupt_journal(d, 0);
            GCJournal::write_commit_marker(d, 0).unwrap();

            KVStore::recover_gc_swaps(d, 1).unwrap();

            assert_eq!(read(&lpath(d, 0)).as_deref(), Some(b"OLD".as_ref()), "reverted to old file");
            assert!(!opath(d, 0).exists(), ".old consumed by revert");
            assert!(!GCJournal::journal_path(d, 0).exists(), "corrupt journal dropped");
            assert!(!GCJournal::commit_marker_exists(d, 0), "marker dropped on revert");
        }

        #[test]
        fn reverts_on_corrupt_journal_before_renames() {
            let dir = tempfile::tempdir().unwrap();
            let d = dir.path();
            std::fs::write(lpath(d, 0), b"OLD").unwrap(); // swap not started, live still old
            std::fs::write(npath(d, 0), b"NEW").unwrap();
            corrupt_journal(d, 0);
            GCJournal::write_commit_marker(d, 0).unwrap();

            KVStore::recover_gc_swaps(d, 1).unwrap();

            assert_eq!(read(&lpath(d, 0)).as_deref(), Some(b"OLD".as_ref()), "stays old");
            assert!(!npath(d, 0).exists(), ".new discarded");
            assert!(!GCJournal::journal_path(d, 0).exists());
            assert!(!GCJournal::commit_marker_exists(d, 0));
        }

        #[test]
        fn no_marker_discards_staged_and_does_not_replay_journal() {
            // Finding 2: a valid journal with no commit marker (swap never
            // committed) must NOT survive to replay — its pointers would be
            // applied against the un-swapped old file.
            let dir = tempfile::tempdir().unwrap();
            let d = dir.path();
            std::fs::write(lpath(d, 0), b"OLD").unwrap();
            std::fs::write(npath(d, 0), b"NEW").unwrap();
            valid_journal(d, 0); // valid, but uncommitted

            KVStore::recover_gc_swaps(d, 1).unwrap();

            assert_eq!(read(&lpath(d, 0)).as_deref(), Some(b"OLD".as_ref()), "old stays live");
            assert!(!npath(d, 0).exists(), "uncommitted .new discarded");
            assert!(!GCJournal::journal_path(d, 0).exists(), "uncommitted journal NOT replayed");
        }

        #[test]
        fn no_marker_finalized_drops_stray_journal() {
            let dir = tempfile::tempdir().unwrap();
            let d = dir.path();
            std::fs::write(lpath(d, 0), b"NEW").unwrap(); // finalized: live = new, no marker
            valid_journal(d, 0); // marker already deleted, journal not yet

            KVStore::recover_gc_swaps(d, 1).unwrap();

            assert_eq!(read(&lpath(d, 0)).as_deref(), Some(b"NEW".as_ref()), "new stays live");
            assert!(!GCJournal::journal_path(d, 0).exists(), "stray journal dropped");
        }

        // A single recovery pass must handle several buckets in different crash
        // states independently (the real GC swaps up to num_buckets per run).
        #[test]
        fn multi_bucket_mixed_states_recover_independently() {
            let dir = tempfile::tempdir().unwrap();
            let d = dir.path();

            // bucket 0: forward-complete (renames unfinished)
            std::fs::write(lpath(d, 0), b"OLD0").unwrap();
            std::fs::write(npath(d, 0), b"NEW0").unwrap();
            valid_journal(d, 0);
            GCJournal::write_commit_marker(d, 0).unwrap();

            // bucket 1: revert (renames done, corrupt journal)
            std::fs::write(lpath(d, 1), b"NEW1").unwrap();
            std::fs::write(opath(d, 1), b"OLD1").unwrap();
            corrupt_journal(d, 1);
            GCJournal::write_commit_marker(d, 1).unwrap();

            // bucket 2: no marker, uncommitted (valid journal must NOT be replayed)
            std::fs::write(lpath(d, 2), b"OLD2").unwrap();
            std::fs::write(npath(d, 2), b"NEW2").unwrap();
            valid_journal(d, 2);

            // bucket 3: untouched — no journal or marker at all
            std::fs::write(lpath(d, 3), b"LIVE3").unwrap();

            KVStore::recover_gc_swaps(d, 4).unwrap();

            // bucket 0 → forward to new, journal+marker kept
            assert_eq!(read(&lpath(d, 0)).as_deref(), Some(b"NEW0".as_ref()));
            assert_eq!(read(&opath(d, 0)).as_deref(), Some(b"OLD0".as_ref()));
            assert!(!npath(d, 0).exists());
            assert!(GCJournal::journal_path(d, 0).exists());
            assert!(GCJournal::commit_marker_exists(d, 0));

            // bucket 1 → reverted to old, journal+marker dropped
            assert_eq!(read(&lpath(d, 1)).as_deref(), Some(b"OLD1".as_ref()));
            assert!(!GCJournal::journal_path(d, 1).exists());
            assert!(!GCJournal::commit_marker_exists(d, 1));

            // bucket 2 → stays old, staged + journal discarded (not replayed)
            assert_eq!(read(&lpath(d, 2)).as_deref(), Some(b"OLD2".as_ref()));
            assert!(!npath(d, 2).exists());
            assert!(!GCJournal::journal_path(d, 2).exists());

            // bucket 3 → completely untouched
            assert_eq!(read(&lpath(d, 3)).as_deref(), Some(b"LIVE3".as_ref()));
        }
    }

    #[test]
    fn test_kvstore_basic_operations() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();

        store.put_to_storage(b"key1", b"value1").unwrap();
        store.put_to_storage(b"key2", b"value2").unwrap();

        assert_eq!(store.get(b"key1").unwrap(), Some(b"value1".to_vec()));
        assert_eq!(store.get(b"key2").unwrap(), Some(b"value2".to_vec()));

        store.delete_from_storage(b"key2").unwrap();
        assert_eq!(store.get(b"key2").unwrap(), None);
    }

    #[test]
    fn test_kvstore_update() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();

        store.put_to_storage(b"key1", b"v1").unwrap();
        assert_eq!(store.get(b"key1").unwrap(), Some(b"v1".to_vec()));

        store.put_to_storage(b"key1", b"v2").unwrap();
        assert_eq!(store.get(b"key1").unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn test_kvstore_recovery_replay() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();

        store.replay_upsert(b"rkey1", b"rval1", 1).unwrap();
        store.replay_upsert(b"rkey2", b"rval2", 2).unwrap();
        store.replay_delete(b"rkey1", 3).unwrap();

        assert_eq!(store.get(b"rkey1").unwrap(), None);
        assert_eq!(store.get(b"rkey2").unwrap(), Some(b"rval2".to_vec()));
    }

    #[test]
    fn test_kvstore_gc() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();

        // Write and overwrite to create garbage
        for i in 0..5 {
            let key = format!("key{}", i).into_bytes();
            store.put_to_storage(&key, &[0u8; 50]).unwrap();
        }
        for round in 0..3 {
            for i in 0..5 {
                let key = format!("key{}", i).into_bytes();
                store.put_to_storage(&key, &[(round + 1) as u8; 50]).unwrap();
            }
        }

        let _gc_stats = store.garbage_collect().unwrap();

        // Verify data still readable
        for i in 0..5 {
            let key = format!("key{}", i).into_bytes();
            assert!(store.get(&key).unwrap().is_some());
        }
    }

    #[test]
    fn test_gc_leaves_no_stray_journal_or_marker_and_reopens() {
        let dir = TempDir::new().unwrap();
        let num_buckets = default_lsm_config().num_buckets as u32;
        {
            let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();
            for i in 0..5 {
                store.put_to_storage(&format!("key{i}").into_bytes(), &[0u8; 50]).unwrap();
            }
            for round in 0..3u8 {
                for i in 0..5 {
                    store.put_to_storage(&format!("key{i}").into_bytes(), &[round + 1; 50]).unwrap();
                }
            }
            store.garbage_collect().unwrap();

            // A normal (finalized) GC run must leave no journal and no commit
            // marker — both are deleted after the LSM pointer updates are flushed.
            let vlp = dir.path().join("value_logs");
            assert!(GCJournal::find_journals(&vlp).is_empty(), "stray journal after a finalized GC");
            for b in 0..num_buckets {
                assert!(!GCJournal::commit_marker_exists(&vlp, b), "stray commit marker for bucket {b}");
            }
        }

        // Reopen: recover_gc_swaps + recover_gc_journals run over the finalized
        // state and must be a clean no-op (no panic / error, data intact).
        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();
        for i in 0..5 {
            assert!(
                store.get(&format!("key{i}").into_bytes()).unwrap().is_some(),
                "key{i} lost after GC+reopen"
            );
        }
    }

    // End-to-end: a committed swap whose journal then bit-rots must REVERT to the
    // preserved old file so the key reads its correct (old) value. Exercises the
    // real value-log + LSM, not just file disposition.
    #[test]
    fn test_revert_restores_correct_value_end_to_end() {
        let dir = TempDir::new().unwrap();
        let cfg = LSMConfig {
            num_buckets: 1,
            ..LSMConfig::default()
        }; // force key into bucket 0
        {
            let store = KVStore::open(0, "default", dir.path(), cfg.clone(), SyncConfig::default()).unwrap();
            store.put_to_storage(b"k", b"value_v1").unwrap();
            store.flush_and_compact_all().unwrap(); // persist the LSM pointer (KVStore has no WAL)
            assert_eq!(store.get(b"k").unwrap(), Some(b"value_v1".to_vec()));
        }

        // Reproduce the committed-swap-with-corrupt-journal state: the real file
        // becomes .old (the swap's first rename), bucket_path holds an unrelated
        // "new" file, plus a corrupt journal and a commit marker.
        let vlp = dir.path().join("value_logs");
        let bucket_path = vlp.join("value_log_0.log");
        let old_path = vlp.join("value_log_0.log.old");
        std::fs::rename(&bucket_path, &old_path).unwrap();
        std::fs::write(&bucket_path, b"bogus new compacted file").unwrap();
        std::fs::write(GCJournal::journal_path(&vlp, 0), b"not a valid GC journal").unwrap();
        GCJournal::write_commit_marker(&vlp, 0).unwrap();

        // Reopen: recover_gc_swaps reverts (renames .old back over bucket_path),
        // so the LSM pointer for "k" resolves against the restored file.
        let store = KVStore::open(0, "default", dir.path(), cfg, SyncConfig::default()).unwrap();
        assert_eq!(
            store.get(b"k").unwrap(),
            Some(b"value_v1".to_vec()),
            "revert must restore the correct value"
        );
        assert!(!GCJournal::journal_path(&vlp, 0).exists());
        assert!(!GCJournal::commit_marker_exists(&vlp, 0));
    }

    // End-to-end: a committed swap whose LSM pointer update was lost (not flushed)
    // must be repaired by replaying the journal, so the key resolves to its NEW
    // location in the new file. The new file holds a padding record before the
    // value, so a missed replay (still pointing at the old location) would read
    // the wrong record — distinguishing replay from no-op.
    #[test]
    fn test_forward_replay_repoints_to_new_location_end_to_end() {
        use crate::store::value_log::{ValueLog, ValueLogMetadata, ValueRecordMeta};
        let dir = TempDir::new().unwrap();
        let cfg = LSMConfig {
            num_buckets: 1,
            ..LSMConfig::default()
        };
        {
            let store = KVStore::open(0, "default", dir.path(), cfg.clone(), SyncConfig::default()).unwrap();
            store.put_to_storage(b"k", b"value_v1").unwrap(); // → old file, segment 1
            store.flush_and_compact_all().unwrap(); // persist old pointer
        }

        let vlp = dir.path().join("value_logs");
        let bucket_path = vlp.join("value_log_0.log");
        let old_path = vlp.join("value_log_0.log.old");
        let build_path = vlp.join("value_log_0.log.build");

        // Build the "compacted" new file: a pad record (segment 1) then k's value
        // (segment 2), so the new location differs from the old one.
        let meta_rec = ValueRecordMeta {
            version: 1,
            tombstone: false,
            updated: false,
            epoch: 0,
            seq: 0,
        };
        let new_loc = {
            let vl = ValueLog::open(&build_path).unwrap();
            let mut m = ValueLogMetadata::new();
            vl.ensure_current_page(&mut m).unwrap();
            vl.write_record(b"pad_record", meta_rec, &mut m, true).unwrap(); // segment 1
            vl.write_record(b"value_v1", meta_rec, &mut m, true).unwrap() // segment 2 (the live one)
        };

        // Swap the files into the committed state; LSM still holds the OLD pointer.
        std::fs::rename(&bucket_path, &old_path).unwrap();
        std::fs::rename(&build_path, &bucket_path).unwrap();
        // Valid journal repointing k to its new location, plus the commit marker.
        GCJournal::write(&vlp, 0, &[(b"k".to_vec(), 0u32, new_loc.page_offset, new_loc.segment_id)]).unwrap();
        GCJournal::write_commit_marker(&vlp, 0).unwrap();

        // Reopen: recover_gc_journals replays, repointing k to the new location.
        let store = KVStore::open(0, "default", dir.path(), cfg, SyncConfig::default()).unwrap();
        assert_eq!(
            store.get(b"k").unwrap(),
            Some(b"value_v1".to_vec()),
            "replay must repoint k to the new file location"
        );
        assert!(!GCJournal::journal_path(&vlp, 0).exists(), "journal dropped after replay");
        assert!(!GCJournal::commit_marker_exists(&vlp, 0), "marker dropped after replay");
    }

    // ── get_multiple ─────────────────────────────────────────────────

    #[test]
    fn test_get_multiple_matches_individual_get() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "ns", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();

        let pairs: Vec<(Vec<u8>, Vec<u8>)> = (0u8..20).map(|i| (format!("doc:{i:03}").into_bytes(), vec![i; 64])).collect();
        for (k, v) in &pairs {
            store.put_to_storage(k, v).unwrap();
        }

        let keys: Vec<Vec<u8>> = pairs.iter().map(|(k, _)| k.clone()).collect();
        let batch = store.get_multiple(&keys);

        assert_eq!(batch.len(), keys.len());
        for (i, result) in batch.iter().enumerate() {
            let expected = store.get(&keys[i]).unwrap();
            assert_eq!(result, &expected, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_get_multiple_missing_keys_are_none() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "ns", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();

        store.put_to_storage(b"exists", b"val").unwrap();

        let keys = vec![b"exists".to_vec(), b"no_such_key".to_vec()];
        let batch = store.get_multiple(&keys);

        assert_eq!(batch[0], Some(b"val".to_vec()));
        assert_eq!(batch[1], None);
    }

    #[test]
    fn test_get_multiple_after_flush_to_sstable() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "ns", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();

        for i in 0u8..10 {
            store.put_to_storage(&[i], &[i; 32]).unwrap();
        }
        // Force a flush so keys live in the SSTable, not the active memtable.
        store.flush_and_compact_all().unwrap();

        let keys: Vec<Vec<u8>> = (0u8..10).map(|i| vec![i]).collect();
        let batch = store.get_multiple(&keys);

        for (i, result) in batch.iter().enumerate() {
            assert_eq!(result.as_deref(), Some(vec![i as u8; 32].as_slice()), "missing key {i}");
        }
    }

    // ── scan_prefix_batch ─────────────────────────────────────────────────────

    fn u32_prefixed_key(prefix: u32, suffix: u32) -> Vec<u8> {
        let mut k = prefix.to_be_bytes().to_vec();
        k.extend_from_slice(&suffix.to_be_bytes());
        k
    }

    #[test]
    fn test_scan_prefixes_batch_basic() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "test_ns", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();

        let entries = [
            (7u32, 1u32, b"val_7_1".as_slice()),
            (7, 2, b"val_7_2"),
            (7, 3, b"val_7_3"),
            (8, 1, b"val_8_1"),
        ];
        for (pid, sid, val) in entries {
            store.put_to_storage(&u32_prefixed_key(pid, sid), val).unwrap();
        }

        let result = store.scan_prefixes_batch(&[7, 8]).unwrap();

        assert_eq!(result[&7].len(), 3);
        assert_eq!(result[&8].len(), 1);

        let vals_7: Vec<&[u8]> = result[&7].iter().map(|(_, v)| v.as_slice()).collect();
        assert!(vals_7.contains(&b"val_7_1".as_slice()));
        assert!(vals_7.contains(&b"val_7_2".as_slice()));
        assert!(vals_7.contains(&b"val_7_3".as_slice()));
    }

    #[test]
    fn test_scan_page_batch_end_bound_excludes_upper() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "test_ns", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();
        for i in 0u32..10 {
            store.put_to_storage(format!("k{i:02}").as_bytes(), format!("v{i}").as_bytes()).unwrap();
        }

        // [None, "k05") must yield k00..k04 and stop *before* the exclusive bound.
        let (pairs, next) = store.scan_page_batch(None, Some(b"k05"), 100).unwrap();
        let keys: Vec<Vec<u8>> = pairs.into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys, (0u32..5).map(|i| format!("k{i:02}").into_bytes()).collect::<Vec<_>>());
        assert!(next.is_none(), "the whole [start, end) window fit in one page");
    }

    #[test]
    fn test_scan_page_batch_cursor_walk_is_complete_and_ordered() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "test_ns", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();
        for i in 0u32..10 {
            store.put_to_storage(format!("k{i:02}").as_bytes(), format!("v{i}").as_bytes()).unwrap();
        }

        // Walk the whole keyspace in pages of 3 and confirm no key is dropped or
        // duplicated and global order is preserved across page boundaries.
        let mut cursor: Option<Vec<u8>> = None;
        let mut seen: Vec<Vec<u8>> = Vec::new();
        loop {
            let (pairs, next) = store.scan_page_batch(cursor.as_deref(), None, 3).unwrap();
            assert!(pairs.len() <= 3);
            seen.extend(pairs.into_iter().map(|(k, _)| k));
            match next {
                Some(c) => cursor = Some(c),
                None => break,
            }
        }
        assert_eq!(seen, (0u32..10).map(|i| format!("k{i:02}").into_bytes()).collect::<Vec<_>>());
    }

    #[test]
    fn test_scan_prefixes_batch_missing_prefix_absent() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "test_ns", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();

        store.put_to_storage(&u32_prefixed_key(1, 1), b"v").unwrap();

        let result = store.scan_prefixes_batch(&[99]).unwrap();
        assert!(!result.contains_key(&99));
    }

    #[test]
    fn test_scan_prefixes_batch_after_flush() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "test_ns", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();

        for sid in 0u32..5 {
            store.put_to_storage(&u32_prefixed_key(2, sid), &[sid as u8; 16]).unwrap();
        }
        store.flush_and_compact_all().unwrap();

        // Add more entries in a fresh memtable to test the multi-source merge
        for sid in 5u32..8 {
            store.put_to_storage(&u32_prefixed_key(2, sid), &[sid as u8; 16]).unwrap();
        }

        let result = store.scan_prefixes_batch(&[2]).unwrap();
        assert_eq!(result[&2].len(), 8);
    }

    // ── GC race regression tests ──────────────────────────────────────────────
    //
    // These tests guard against the bug where GC scanned the LSM *before*
    // acquiring the bucket lock.  Any write that completed its value-log write
    // but had not yet updated the LSM in that window was permanently lost:
    // GC deleted the old file without copying the record, leaving a stale
    // LSM pointer into a zero-filled page (→ CorruptedLog on the next read).
    //
    // The fix: writes hold the bucket lock across the value-log write AND
    // the LSM insert; GC scans the LSM inside the same lock, guaranteeing a
    // consistent snapshot.

    #[test]
    fn test_gc_write_before_gc_survives() {
        // A write that completes immediately before GC starts must not be lost.
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();

        // Create enough garbage to trigger GC (75% garbage after 4 overwrites).
        for round in 0..4u8 {
            for i in 0u32..50 {
                store.put_to_storage(&format!("base:{i:04}").into_bytes(), &[round; 128]).unwrap();
            }
        }

        // Write a sentinel key, then immediately run GC.
        // GC must scan LSM inside the bucket lock and include this key.
        store.put_to_storage(b"sentinel", b"sentinel_value").unwrap();
        store.garbage_collect_with_threshold(0.0).unwrap();

        assert_eq!(
            store.get(b"sentinel").unwrap(),
            Some(b"sentinel_value".to_vec()),
            "sentinel key lost after GC"
        );
    }

    #[test]
    fn test_gc_concurrent_writes_no_data_loss() {
        // Regression: concurrent writes and GC must not lose any data.
        // Writer thread writes 200 unique keys while GC runs 5 times in parallel.
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();

        // Create garbage so GC actually fires.
        for round in 0..4u8 {
            for i in 0u32..50 {
                store.put_to_storage(&format!("base:{i:04}").into_bytes(), &[round; 128]).unwrap();
            }
        }

        const NUM_NEW: usize = 200;

        std::thread::scope(|s| {
            s.spawn(|| {
                for i in 0..NUM_NEW {
                    let key = format!("new:{i:04}").into_bytes();
                    store.put_to_storage(&key, &[(i % 256) as u8; 64]).unwrap();
                }
            });
            s.spawn(|| {
                for _ in 0..5 {
                    store.garbage_collect_with_threshold(0.0).unwrap();
                }
            });
        });

        for i in 0..NUM_NEW {
            let key = format!("new:{i:04}").into_bytes();
            let expected = vec![(i % 256) as u8; 64];
            assert_eq!(
                store.get(&key).unwrap(),
                Some(expected),
                "key new:{i:04} was lost or corrupted after concurrent GC"
            );
        }

        // Pre-existing keys must also be intact.
        for i in 0u32..50 {
            let key = format!("base:{i:04}").into_bytes();
            assert!(store.get(&key).unwrap().is_some(), "base key {i} lost after GC");
        }
    }

    #[test]
    fn test_gc_writes_after_swap_go_to_new_file() {
        // After GC swaps the value-log file, subsequent writes must go to the
        // new file and be readable.
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();

        for round in 0..4u8 {
            for i in 0u32..30 {
                store.put_to_storage(&format!("k{i}").into_bytes(), &[round; 128]).unwrap();
            }
        }

        store.garbage_collect_with_threshold(0.0).unwrap();

        for i in 0u32..30 {
            let key = format!("post_gc:{i}").into_bytes();
            store.put_to_storage(&key, &[0xFFu8; 64]).unwrap();
        }

        for i in 0u32..30 {
            let key = format!("post_gc:{i}").into_bytes();
            assert_eq!(store.get(&key).unwrap(), Some(vec![0xFFu8; 64]));
        }
    }

    #[test]
    fn test_gc_multi_round_all_values_survive() {
        // Multiple sequential GC runs must not lose any values.
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();

        for round in 0..6u8 {
            for i in 0u32..40 {
                store.put_to_storage(&format!("key:{i:04}").into_bytes(), &[round; 64]).unwrap();
            }
            store.garbage_collect_with_threshold(0.0).unwrap();
        }

        for i in 0u32..40 {
            let key = format!("key:{i:04}").into_bytes();
            // Last written value was round=5
            assert_eq!(store.get(&key).unwrap(), Some(vec![5u8; 64]), "key {i} has wrong value or is missing");
        }
    }

    #[test]
    fn test_gc_concurrent_reads_never_error_or_return_stale() {
        // Item 6 regression: a reader must get a *generation-stable* snapshot.
        // GC swaps the value-log file and the LSM pointers as a pair under the
        // bucket lock; a lock-free reader could otherwise pair a stale pointer
        // with the freshly swapped file, which previously surfaced as a transient
        // CorruptedLog (and was only papered over by a single retry). With the
        // per-bucket generation guard, reads must always succeed and return the
        // last-written value while GC churns the same buckets continuously.
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();

        const N: u32 = 120;

        // Overwrite each key several times to pile up garbage on its pages, then
        // settle on the canonical value so GC has dirty pages to compact.
        for round in 0..5u8 {
            for i in 0..N {
                store.put_to_storage(&format!("k:{i:04}").into_bytes(), &[round; 48]).unwrap();
            }
        }
        for i in 0..N {
            store.put_to_storage(&format!("k:{i:04}").into_bytes(), b"final_value").unwrap();
        }

        std::thread::scope(|s| {
            for _ in 0..3 {
                s.spawn(|| {
                    for _ in 0..150 {
                        for i in 0..N {
                            let key = format!("k:{i:04}").into_bytes();
                            let got = store.get(&key).expect("get must never error while GC runs");
                            assert_eq!(
                                got,
                                Some(b"final_value".to_vec()),
                                "read returned a stale/missing value for k:{i:04} during GC"
                            );
                        }
                    }
                });
            }
            s.spawn(|| {
                for _ in 0..10 {
                    store.garbage_collect_with_threshold(0.0).unwrap();
                }
            });
        });

        // Final state is still correct after all the churn.
        for i in 0..N {
            let key = format!("k:{i:04}").into_bytes();
            assert_eq!(store.get(&key).unwrap(), Some(b"final_value".to_vec()));
        }
    }

    #[test]
    fn test_active_memtable_tombstone_shadows_sstable() {
        // A key flushed to an SSTable, then deleted (tombstone lands in the
        // active memtable), must read as absent. The active-memtable layer must
        // treat a tombstone as authoritative and not fall through to the live
        // SSTable value — otherwise a delete is silently resurrected on read
        // until the next compaction.
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();

        store.put_to_storage(b"k", b"v").unwrap();
        store.lsm.flush_and_compact_all().unwrap(); // k now lives only in L1
        assert_eq!(store.get(b"k").unwrap(), Some(b"v".to_vec()));

        store.delete_from_storage(b"k").unwrap(); // tombstone in the fresh active memtable
        assert_eq!(store.get(b"k").unwrap(), None, "active-memtable tombstone must shadow the L1 value");
    }

    #[test]
    fn test_put_to_storage_seq_keeps_highest_sequence() {
        // The live winner for two same-key writes is the higher sequence, even
        // when the lower-sequence write is applied last — matching recovery's
        // sequence-ordered replay (live == recovery for racing same-key writes).
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();

        store.put_to_storage_seq(b"k", b"newer", 10).unwrap();
        store.put_to_storage_seq(b"k", b"older", 5).unwrap(); // applied later, lower seq → must lose
        assert_eq!(store.get(b"k").unwrap(), Some(b"newer".to_vec()));
    }

    #[test]
    fn test_delete_from_storage_seq_respects_sequence() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();

        store.put_to_storage_seq(b"k", b"v", 10).unwrap();
        store.delete_from_storage_seq(b"k", 5).unwrap(); // older delete → no-op
        assert_eq!(store.get(b"k").unwrap(), Some(b"v".to_vec()));
        store.delete_from_storage_seq(b"k", 15).unwrap(); // newer delete → removed
        assert_eq!(store.get(b"k").unwrap(), None);
    }

    #[test]
    fn test_gc_after_delete_sequential_stays_deleted() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();
        const N: u32 = 120;
        for round in 0..4u8 {
            for i in 0..N {
                store.put_to_storage(&format!("del:{i:04}").into_bytes(), &[round; 96]).unwrap();
            }
        }
        for i in 0..N {
            store.delete_from_storage(&format!("del:{i:04}").into_bytes()).unwrap();
        }
        store.garbage_collect_with_threshold(0.0).unwrap();
        let mut survivors = 0;
        for i in 0..N {
            if store.get(&format!("del:{i:04}").into_bytes()).unwrap().is_some() {
                survivors += 1;
            }
        }
        assert_eq!(survivors, 0, "{survivors} keys resurrected by sequential GC-after-delete");
    }

    #[test]
    fn test_gc_concurrent_deletes_are_not_resurrected() {
        // Item 5 regression: a delete that lands in GC's scan→reinsert window
        // must not be undone. GC scans live LSM keys under the bucket lock and
        // then re-inserts the pointers of the records it copied; without the
        // delete path taking the same bucket lock, GC could copy an
        // about-to-be-deleted record and re-insert its pointer *after* the
        // delete removed it, resurrecting the key.
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default()).unwrap();

        const N: u32 = 120;

        // Seed keys and pile up garbage so GC fires while deletes run.
        for round in 0..4u8 {
            for i in 0..N {
                store.put_to_storage(&format!("del:{i:04}").into_bytes(), &[round; 96]).unwrap();
            }
        }

        std::thread::scope(|s| {
            s.spawn(|| {
                for i in 0..N {
                    store.delete_from_storage(&format!("del:{i:04}").into_bytes()).unwrap();
                }
            });
            s.spawn(|| {
                for _ in 0..10 {
                    store.garbage_collect_with_threshold(0.0).unwrap();
                }
            });
        });

        // One more GC after everything settles, then every key must stay deleted.
        store.garbage_collect_with_threshold(0.0).unwrap();
        let survivors: Vec<u32> = (0..N)
            .filter(|i| store.get(&format!("del:{i:04}").into_bytes()).unwrap().is_some())
            .collect();
        assert!(
            survivors.is_empty(),
            "{} keys resurrected by concurrent GC: {:?}",
            survivors.len(),
            survivors
        );
    }

    #[test]
    fn test_scan_during_gc_never_returns_wrong_or_missing_value() {
        // Regression for the value-log GC race: a batch read resolves
        // (key, pointer) from the LSM and then reads the value from a separately
        // captured value-log file handle. A GC swap landing between those steps
        // pairs a pre-swap pointer with the post-swap file, so a relocated key
        // reads another key's record (wrong value) or fails to decode (a resolved
        // key comes back empty). The fix is the seqlock generation bracket plus a
        // single-key `get` re-resolution of any resolved-but-unreadable key.
        //
        // All keys share a single bucket so every GC swap races every read, and
        // each live key carries a distinct, *stable* value — overwrites rewrite
        // the same bytes purely to keep generating garbage, so the only way a read
        // can observe a wrong/missing value is the swap race itself.
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicBool, Ordering};

        let dir = TempDir::new().unwrap();
        let mut cfg = default_lsm_config();
        cfg.num_buckets = 1; // concentrate every key in one bucket
        let store = KVStore::open(0, "default", dir.path(), cfg, SyncConfig::default()).unwrap();

        const N: u32 = 200;
        let live_key = |i: u32| format!("live:{i:06}").into_bytes();
        let expected = |i: u32| {
            // Distinct prefix per key (catches cross-key misreads), padded so
            // records are large enough to span pages and force relocation.
            let mut v = format!("val:{i:06}:").into_bytes();
            v.resize(100, b'.');
            v
        };

        // Seed live keys plus junk; deleting the junk leaves garbage interleaved
        // with the survivors, so GC must rewrite dirty pages and relocate them.
        for i in 0..N {
            store.put_to_storage(&live_key(i), &expected(i)).unwrap();
            store.put_to_storage(&format!("junk:{i:06}").into_bytes(), &[0xAB; 100]).unwrap();
        }
        for i in 0..N {
            store.delete_from_storage(&format!("junk:{i:06}").into_bytes()).unwrap();
        }

        let all_live: Vec<Vec<u8>> = (0..N).map(live_key).collect();
        let stop = AtomicBool::new(false);
        let failure: Mutex<Option<String>> = Mutex::new(None);

        std::thread::scope(|s| {
            // Writer: keep overwriting live keys with their *same* value so the
            // bucket keeps accumulating garbage and GC keeps relocating records.
            s.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    for i in 0..N {
                        let _ = store.put_to_storage(&live_key(i), &expected(i));
                    }
                }
            });
            // GC: hammer compaction on the single bucket.
            s.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    let _ = store.garbage_collect_with_threshold(0.0);
                }
            });
            // Scanner. The value-log invariant under test: a committed value is
            // never read back wrong, and a live key resolved to a pointer is never
            // unreadable. `get_multiple` resolves explicit keys, so it is
            // the strong oracle — every live key must come back present and exact.
            //
            // `scan_prefix_batch` can momentarily return a *partial* key set under
            // GC churn — but NOT because of the LSM scan: `lsm.scan_prefix` is
            // complete across concurrent flush/compaction (asserted by
            // `test_scan_completeness_during_concurrent_flush_compaction_and_gc`).
            // The transient drop is in the value-resolution path layered on top, a
            // separate concern from the silent-wrong-value race under test. So here
            // we assert only that every pair it DOES return carries the correct
            // value, not that the set is complete (`get_multiple` above is
            // the completeness oracle).
            s.spawn(|| {
                let record = |msg: String| {
                    *failure.lock().unwrap() = Some(msg);
                    stop.store(true, Ordering::Relaxed);
                };
                for _ in 0..400 {
                    // scan_prefix_batch → resolve_entries_from_pointers + batch_read_from_entries
                    for (k, v) in store.scan_prefix_batch(b"live:").unwrap() {
                        let i: u32 = std::str::from_utf8(&k[5..]).unwrap().parse().unwrap();
                        if v != expected(i) {
                            record(format!("scan_prefix: live:{i:06} WRONG value (len {})", v.len()));
                            return;
                        }
                    }

                    // get_multiple has its own read loop — strong oracle.
                    let got = store.get_multiple(&all_live);
                    for (i, v) in got.iter().enumerate() {
                        if v.as_deref() != Some(expected(i as u32).as_slice()) {
                            record(format!(
                                "get_multiple: live:{i:06} wrong/missing (got {:?} len)",
                                v.as_ref().map(|x| x.len())
                            ));
                            return;
                        }
                    }

                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                }
                stop.store(true, Ordering::Relaxed);
            });
        });

        if let Some(msg) = failure.lock().unwrap().take() {
            panic!("scan raced GC and observed inconsistent state: {msg}");
        }
    }

    #[test]
    fn test_scan_completeness_during_concurrent_flush_compaction_and_gc() {
        // Integration regression proving the LSM scan-completeness fix and the
        // value-log GC seqlock fix (commit 420ac8e) COMPOSE. Under concurrent
        // memtable flush, L0→L1 compaction, and value-log GC, a scan must:
        //   * return EVERY never-deleted key — the LSM scan-completeness guarantee,
        //     across the active→RO→L0→L1 relocation a flush/compaction performs; and
        //   * pair each key with its EXACT value — the value-log seqlock guarantee,
        //     across the file swap + LSM re-point a GC compaction performs.
        // Either fix alone is insufficient: without the LSM fix a scan drops a live
        // key crossing a layer boundary; without the seqlock fix a relocated pointer
        // reads the wrong/empty value. One bucket so every flush, merge, and GC swap
        // races every scan. Live keys are never deleted and are overwritten with
        // their SAME value, so any wrong value is a race bug; active-churn
        // transient misses from APIs that document `None` for I/O races are
        // checked strictly only after the churn threads stop.
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Barrier, Mutex};

        let dir = TempDir::new().unwrap();
        let mut cfg = default_lsm_config();
        cfg.num_buckets = 1; // concentrate every key in one bucket
        let store = KVStore::open(0, "default", dir.path(), cfg, SyncConfig::default()).unwrap();

        const N: u32 = 200;
        let live_key = |i: u32| format!("live:{i:06}").into_bytes();
        let expected = |i: u32| {
            let mut v = format!("val:{i:06}:").into_bytes();
            v.resize(100, b'.'); // large enough to span pages and force relocation
            v
        };

        // Live keys plus junk; deleting the junk leaves garbage interleaved with the
        // survivors so GC must rewrite dirty pages and relocate live records.
        for i in 0..N {
            store.put_to_storage(&live_key(i), &expected(i)).unwrap();
            store.put_to_storage(&format!("junk:{i:06}").into_bytes(), &[0xAB; 100]).unwrap();
        }
        for i in 0..N {
            store.delete_from_storage(&format!("junk:{i:06}").into_bytes()).unwrap();
        }

        let all_live: Vec<Vec<u8>> = (0..N).map(live_key).collect();
        let stop = AtomicBool::new(false);
        let failure: Mutex<Option<String>> = Mutex::new(None);
        let start = Barrier::new(4);

        std::thread::scope(|s| {
            // Writer: overwrite live keys with their SAME value to keep producing
            // value-log garbage without changing the committed value.
            s.spawn(|| {
                start.wait();
                while !stop.load(Ordering::Acquire) {
                    for i in 0..N {
                        let _ = store.put_to_storage(&live_key(i), &expected(i));
                    }
                }
            });
            // LSM churn: drive the full active→RO→L0→L1 relocation chain the scans
            // must stay complete across.
            s.spawn(|| {
                start.wait();
                while !stop.load(Ordering::Acquire) {
                    let _ = store.lsm.flush_memtable_to_level0();
                    let _ = store.lsm.compact_all();
                }
            });
            // Value-log GC: relocate pointers under the scans.
            s.spawn(|| {
                start.wait();
                while !stop.load(Ordering::Acquire) {
                    let _ = store.garbage_collect_with_threshold(0.0);
                }
            });
            // Scanner.
            //
            // (1) LSM-layer COMPLETENESS — the scan-completeness fix. Each of the
            //     three pointer-returning scans must return EVERY never-deleted key
            //     while flush/compaction relocate them active→RO→L0→L1. Asserted on
            //     the key set straight off the LSM, independent of value-log GC.
            // (2) Value-log batch paths are exercised under churn, but their strict
            //     completeness/exactness is asserted only after all churn has
            //     stopped. Keeping active-churn assertions at the LSM key layer
            //     avoids turning this regression into a timing-sensitive
            //     linearizability test for batch value resolution while flush,
            //     compaction, GC, and writes are all mutating the same keys.
            s.spawn(|| {
                let record = |msg: String| {
                    *failure.lock().unwrap() = Some(msg);
                    stop.store(true, Ordering::Release);
                };
                start.wait();
                // Every live key must appear in the LSM scan's key set.
                let verify_complete = |label: &str, keys: Vec<Vec<u8>>| -> Option<String> {
                    let mut seen = vec![false; N as usize];
                    for k in keys {
                        if !k.starts_with(b"live:") {
                            continue;
                        }
                        let i: u32 = std::str::from_utf8(&k[5..]).unwrap().parse().unwrap();
                        seen[i as usize] = true;
                    }
                    seen.iter()
                        .position(|&done| !done)
                        .map(|m| format!("{label}: live:{m:06} MISSING from LSM scan"))
                };
                for _ in 0..300 {
                    if stop.load(Ordering::Acquire) {
                        return;
                    }
                    // (1) LSM-layer completeness across all three scan shapes.
                    let prefix_keys = store.lsm.scan_prefix(b"live:").unwrap().into_iter().map(|(k, _, _)| k).collect();
                    if let Some(msg) = verify_complete("lsm.scan_prefix", prefix_keys) {
                        record(msg);
                        return;
                    }
                    let range_keys = store
                        .lsm
                        .range_pointers_bounded(b"live:", None, usize::MAX)
                        .unwrap()
                        .into_iter()
                        .map(|(k, _, _)| k)
                        .collect();
                    if let Some(msg) = verify_complete("lsm.range_pointers_bounded", range_keys) {
                        record(msg);
                        return;
                    }
                    let kv_keys = store.lsm.key_pointer_pairs(None).unwrap().into_iter().map(|(k, _)| k).collect();
                    if let Some(msg) = verify_complete("lsm.key_pointer_pairs", kv_keys) {
                        record(msg);
                        return;
                    }

                    // Exercise the batch value paths under churn, but do not assert
                    // a non-linearizable active-churn snapshot here. The quiescent
                    // assertions below are the completeness/exactness oracle.
                    let _ = store.get_multiple(&all_live);
                    let _ = store.scan_prefix_batch(b"live:").unwrap();
                }
                stop.store(true, Ordering::Release);
            });
        });

        if let Some(msg) = failure.lock().unwrap().take() {
            panic!("scan raced concurrent flush/compaction/GC and observed inconsistent state: {msg}");
        }

        // Once the churn threads have joined, the batch get path must be complete:
        // all never-deleted live keys should resolve to their exact value.
        let got = store.get_multiple(&all_live);
        for (i, v) in got.iter().enumerate() {
            assert_eq!(
                v.as_deref(),
                Some(expected(i as u32).as_slice()),
                "quiescent get_multiple must return live:{i:06}"
            );
        }

        let mut scanned = store.scan_prefix_batch(b"live:").unwrap();
        scanned.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(scanned.len(), N as usize, "quiescent scan_prefix_batch must return every live key");
        for (i, (k, v)) in scanned.iter().enumerate() {
            assert_eq!(k, &live_key(i as u32), "quiescent scan_prefix_batch returned unexpected key at index {i}");
            assert_eq!(
                v.as_slice(),
                expected(i as u32).as_slice(),
                "quiescent scan_prefix_batch returned wrong value for live:{i:06}"
            );
        }
    }
}
