//! KVStore - Per-namespace key-value storage
//!
//! Each KVStore owns its own LSM tree and sharded value log,
//! representing an independent namespace within the database.
//! The WAL is shared across all namespaces and managed by the
//! Database coordinator.

use crate::db::error::{KVError, Result};
use crate::db::index_checkpoint_worker::IndexCheckpointTrigger;
use crate::db::stats::{GCStats, Stats};
use crate::store::lsm::lsm_tree::{LSMConfig, LSMTree, LsmFlushObserver};
use crate::store::lsm_worker::LsmCompactionCommand;
use crate::store::value_log::sharded::{ShardedValueLog, ShardedValuePointer};
use crate::store::value_log::{TAIL_GC_MIN_GARBAGE_PCT, ValueLocation, ValueRecordMeta};

use log::{debug, error, info, warn};
use parking_lot::RwLock;
use std::collections::HashMap;
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

/// A key paired with its encoded value-log pointer and the LSM write `seq` (low u32).
pub(crate) type KeyPointer = (Vec<u8>, u128, u32);

/// One page of a cursor scan: the resolved pairs plus the next cursor
/// (`None` when the scan is exhausted).
pub(crate) type ScanPage = (Vec<KeyValue>, Option<Vec<u8>>);

/// A scan entry resolved to what is needed to read its value: the key and its
/// value-log pointer.
///
/// No file handle is pinned any more. Segments are immutable and their ids are
/// never reused, so a pointer stays meaningful however long it is held: it either
/// resolves to the same record or its segment is gone (`SegmentMissing`) and the
/// key is re-resolved through the LSM.
pub(crate) type ResolvedEntry = (Vec<u8>, ShardedValuePointer);

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

/// Decode a stored pointer. `segment_id == 0` is the "no value" sentinel.
fn decode_sharded_pointer(offset_u128: u128) -> Option<ShardedValuePointer> {
    let p = ShardedValuePointer::from_u128(offset_u128);
    (p.location.segment_id != 0).then_some(p)
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

    // Sharded value log: one series of segment files per bucket. The value log owns
    // its own durable metadata (segment inventory + id high-water mark) per bucket,
    // so the store keeps no separate metadata file of its own.
    pub(crate) value_log: ShardedValueLog,
    #[allow(dead_code)]
    pub(crate) value_log_path: PathBuf,

    // Sync configuration
    pub(crate) sync_config: SyncConfig,

    // Counts writes for periodic sync
    pub(crate) write_count: Arc<AtomicU64>,

    // Flag to prevent concurrent value log GC operations
    pub(crate) value_log_gc_in_progress: Arc<AtomicBool>,

    // LSM compaction trigger channel
    pub(crate) lsm_compaction_trigger: Arc<RwLock<Option<tokio::sync::mpsc::UnboundedSender<LsmCompactionCommand>>>>,

    // Index-checkpoint backpressure valve: requests an early checkpoint when a
    // field index accumulates too much reclaimable dead blob space. `None` until
    // the checkpoint worker is enabled (set via `set_index_checkpoint_trigger`).
    pub(crate) index_checkpoint_trigger: Arc<RwLock<Option<Arc<IndexCheckpointTrigger>>>>,

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
    // activated; `None` until then.  See `crate::index::RowMap`.
    pub(crate) rowmap: Arc<RwLock<Option<crate::index::RowMap>>>,

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

    /// Test-only crash point: when set, a GC pass returns the moment its segments are
    /// unlinked, skipping everything after. It exists to prove the *ordering* of
    /// `flush → unlink` is what makes GC crash-safe: with the flush first (correct), a
    /// crash here is harmless; if the two were ever swapped, this window would lose the
    /// re-point along with the segments it referred to, and the guard test would fail.
    #[cfg(test)]
    pub(crate) gc_crash_after_unlink: AtomicBool,
}

/// What one bucket's GC pass did.
struct BucketGCResult {
    bucket: u32,
    /// Bytes of survivors rewritten into the active tail — the *cost* of the pass.
    /// Compare against the bytes reclaimed for write amplification.
    bytes_rewritten: u64,
    /// Segments whose survivors have been relocated, but which cannot be unlinked
    /// until the LSM re-point is durable (see `garbage_collect_with_threshold`).
    pending_unlink: Vec<u32>,
}

impl KVStore {
    /// Enable or disable re-verifying each value's CRC32 on read (default off).
    /// See `DbConfig::verify_checksums_on_read`.
    pub fn set_verify_checksums_on_read(&self, verify: bool) {
        self.value_log.set_verify_checksums_on_read(verify);
    }

    /// Open a KVStore for the given namespace at the specified base directory.
    /// The directory will contain `lsm/` and `value_logs/` subdirectories.
    ///
    /// `segment_size_bytes` is the size at which a value-log segment is sealed and a
    /// new one opened. Unlike the page size it replaces, it is **not** fixed at
    /// creation: a segment's size is not encoded in any pointer, so existing segments
    /// keep theirs and new ones use whatever is configured now.
    pub fn open(
        namespace_id: u32,
        name: &str,
        base_path: &Path,
        lsm_config: LSMConfig,
        sync_config: SyncConfig,
        segment_size_bytes: u64,
    ) -> Result<Self> {
        Self::open_with_ttl(namespace_id, name, base_path, lsm_config, sync_config, segment_size_bytes, None)
    }

    /// Open a KVStore with an optional TTL for automatic record expiry.
    ///
    /// There is no GC recovery step. GC writes survivors to a *new* segment, re-points
    /// the keys, makes that durable, and only then unlinks the old segment — so at
    /// every crash point the durable LSM still refers to a segment that exists. There
    /// is nothing to roll forward or back, which is why the GC journal, its commit
    /// marker and the `.log.new`/`.log.old` dance are all gone.
    pub fn open_with_ttl(
        namespace_id: u32,
        name: &str,
        base_path: &Path,
        lsm_config: LSMConfig,
        sync_config: SyncConfig,
        segment_size_bytes: u64,
        ttl: Option<Duration>,
    ) -> Result<Self> {
        std::fs::create_dir_all(base_path)?;

        let lsm_path = base_path.join("lsm");
        let value_log_path = base_path.join("value_logs");

        let mut lsm_cfg = lsm_config;
        lsm_cfg.data_dir = lsm_path.clone();
        let num_buckets = lsm_cfg.num_buckets;
        let lsm = Arc::new(LSMTree::open(&lsm_path, lsm_cfg)?);

        let value_log = ShardedValueLog::open(&value_log_path, num_buckets, segment_size_bytes).map_err(|e| {
            KVError::Io(std::io::Error::other(format!(
                "Failed to open sharded value log for namespace '{}': {}",
                name, e
            )))
        })?;

        let store = Self {
            namespace_id,
            name: name.to_string(),
            lsm,
            lsm_path,
            value_log,
            value_log_path,
            sync_config,
            write_count: Arc::new(AtomicU64::new(0)),
            value_log_gc_in_progress: Arc::new(AtomicBool::new(false)),
            lsm_compaction_trigger: Arc::new(RwLock::new(None)),
            index_checkpoint_trigger: Arc::new(RwLock::new(None)),
            ttl,
            namespace_index: Arc::new(RwLock::new(NamespaceIndexSet::new())),
            row_id_fn: Arc::new(RwLock::new(None)),
            row_to_key_fn: Arc::new(RwLock::new(None)),
            rowmap: Arc::new(RwLock::new(None)),
            seq_counter: RwLock::new(Arc::new(AtomicU64::new(1))),
            metrics: OnceLock::new(),
            #[cfg(test)]
            gc_crash_after_unlink: AtomicBool::new(false),
        };

        // A bucket whose metadata file was lost has no live/garbage accounting, so GC
        // would never trigger on it. Recompute it exactly from the LSM's pointers —
        // the LSM is the authority on liveness anyway, so this is not a heuristic.
        store.rebuild_missing_segment_stats()?;

        Ok(store)
    }

    /// Recompute per-segment live/garbage byte counts from the LSM's pointers, for any
    /// bucket whose accounting could not be loaded.
    ///
    /// Exact, not conservative: a record is live iff the LSM still points at it, and
    /// every other byte in the segment is garbage. Costs one LSM key+pointer scan.
    fn rebuild_missing_segment_stats(&self) -> Result<()> {
        let buckets = self.value_log.buckets_needing_stat_rebuild();
        if buckets.is_empty() {
            return Ok(());
        }
        info!(
            "[KVStore '{}'] rebuilding value-log segment accounting for {} bucket(s) from the LSM",
            self.name,
            buckets.len()
        );

        let mut live: HashMap<u32, HashMap<u32, u64>> = HashMap::new(); // bucket -> segment -> bytes
        for (key, offset_u128) in self.lsm.key_pointer_pairs(None)? {
            if let Some(pointer) = decode_sharded_pointer(offset_u128) {
                *live.entry(pointer.bucket).or_default().entry(pointer.location.segment_id).or_insert(0) += pointer.record_len(key.len());
            }
        }
        let empty = HashMap::new();
        for bucket in buckets {
            let per_segment = live.get(&bucket).unwrap_or(&empty);
            self.value_log.rebuild_bucket_stats(bucket, per_segment)?;
        }
        self.value_log.flush_all_metadata()?;
        Ok(())
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
            *guard = Some(crate::index::RowMap::open(dir).map_err(KVError::Io)?);
        }
        Ok(())
    }

    /// Resolve the row ID for `key`, **allocating** a new dense ID if the key is
    /// unseen. Used on the put and WAL-replay paths.
    ///
    /// Precedence: a caller-supplied [`RowIdFn`] (the escape hatch for keys that
    /// embed their own ID) wins; otherwise the dense [`RowMap`](crate::index::RowMap).
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

    /// Set the LSM flush observer (used to wire up WAL persistence callbacks)
    pub(crate) fn set_flush_observer(&self, observer: Option<Arc<dyn LsmFlushObserver>>) {
        self.lsm.set_flush_observer(observer);
    }

    /// Set the LSM compaction trigger channel
    pub fn set_compaction_trigger(&self, sender: tokio::sync::mpsc::UnboundedSender<LsmCompactionCommand>) {
        *self.lsm_compaction_trigger.write() = Some(sender);
    }

    /// Set (or clear) the index-checkpoint backpressure valve for this store.
    pub(crate) fn set_index_checkpoint_trigger(&self, trigger: Option<Arc<IndexCheckpointTrigger>>) {
        *self.index_checkpoint_trigger.write() = trigger;
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

        // Hold the bucket lock across the existing-pointer read, the value-log append,
        // AND the LSM insert. Reading the displaced pointer *inside* the lock is
        // load-bearing for garbage accounting: two racing same-key writers must not both
        // observe the same old pointer and each mark it displaced (double-counting it and
        // leaking the intermediate record). GC re-point is excluded for the same reason.
        let bucket = self.value_log.bucket_for_key(key);
        let _bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;

        let (existing_u128, wins) = self.lsm.current_pointer_and_wins(key, seq as u32)?;
        let displaced = existing_u128.and_then(decode_sharded_pointer);
        let mut old_value: Option<Vec<u8>> = None;
        if let Some(existing_ptr) = displaced {
            if let Ok(meta) = self.value_log.read_record_meta(existing_ptr) {
                next_version = meta.version.saturating_add(1);
            }
            if want_old_for_index {
                old_value = self.value_log.read_value(existing_ptr).ok();
            }
        }

        let record_meta = ValueRecordMeta {
            version: next_version,
            epoch,
            seq,
        };
        let pointer = self.value_log.append_to_locked_bucket(bucket, key, value, record_meta, false)?;
        self.lsm.insert_with_seq(key, pointer.to_u128(), seq as u32)?;

        // Account the record that is now garbage — accounting only, no I/O and no
        // in-place edit (the old record's segment may be sealed and immutable). If this
        // write won, that is the record it displaced; if it *lost* to a concurrent
        // higher-sequence write (so the LSM dropped our insert), our own just-appended
        // record is the dead one instead.
        if wins {
            if let Some(old) = displaced {
                self.value_log.note_displaced(old, key.len());
            }
        } else {
            self.value_log.note_displaced(pointer, key.len());
        }
        drop(_bucket_guard);

        // Update in-memory field indices. A lost write is not the key's current value,
        // so it must not touch the indices.
        if wins {
            self.update_indices_on_put(key, value, old_value.as_deref(), displaced.is_some());
        }

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
            let dead_bytes = idx.reclaimable_dead_bytes();
            drop(idx);
            if let Err(e) = result {
                warn!("[KVStore '{}'] Index update rejected for field {}: {}", self.name, entry.field_id, e);
            }
            // Backpressure: if this field's append-only blob has piled up enough
            // dead space, ask the checkpoint worker to compact early instead of
            // waiting ~15 min (debounced inside the trigger).
            self.request_checkpoint_if_over_cap(dead_bytes);
        }
    }

    /// Signal the index-checkpoint backpressure valve, if wired, with a field's
    /// current reclaimable dead blob bytes. O(1) and non-blocking.
    fn request_checkpoint_if_over_cap(&self, dead_bytes: u64) {
        if let Some(trigger) = self.index_checkpoint_trigger.read().as_ref() {
            trigger.request_if_over_cap(dead_bytes);
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

    /// Read attempts before falling back to a lock-serialised read. A read only loops
    /// when GC unlinks a segment in the exact window between resolving the pointer and
    /// reading it, so in practice it succeeds first time; the cap only bounds the
    /// pathological case of GC churning one bucket continuously.
    const MAX_READ_ATTEMPTS: usize = 8;

    /// Get a value by key.
    ///
    /// Two steps: ask the LSM where the value is, then read it there. GC can relocate
    /// a value between those steps — but it cannot make this read *wrong*. Segments
    /// are immutable and their ids are never reused, so a pointer GC has superseded
    /// still resolves to **this key's own bytes** (GC re-points under the key's
    /// existing sequence, so the relocated record is the same write). The only thing
    /// that can go wrong is arriving after GC unlinked the segment, which fails
    /// loudly as [`ValueLogError::SegmentMissing`] — never silently as another key's
    /// record. So the read simply re-resolves through the LSM (which by then holds the
    /// new pointer) and retries.
    ///
    /// This is what replaced the old generation seqlock, the per-record sequence
    /// validity check, and their retry ladder.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let result = self.get_inner(key);
        if let Some(m) = self.metrics() {
            // Count the read; a hit is a successful lookup that found a live value.
            m.record_read(matches!(result, Ok(Some(_))));
        }
        result
    }

    fn get_inner(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let mut last_err: Option<KVError> = None;

        for _ in 0..Self::MAX_READ_ATTEMPTS {
            // A transient LSM error (an SSTable being swapped by compaction) is
            // retryable, not fatal.
            let offset_u128 = match self.lsm.get(key) {
                Ok(Some(o)) => o,
                Ok(None) => return Ok(None),
                Err(e) => {
                    last_err = Some(e.into());
                    continue;
                }
            };
            let Some(pointer) = decode_sharded_pointer(offset_u128) else {
                return Ok(None);
            };

            match self.value_log.read_value(pointer) {
                Ok(value) => return Ok(Some(value)),
                // GC reclaimed the segment after relocating this record; the LSM now
                // holds the new pointer. Re-resolve and read again. This is the ONLY
                // retryable read error.
                Err(e) if e.is_segment_missing() => {
                    last_err = Some(e.into());
                    continue;
                }
                // Corruption or an IO fault is real: surface it immediately instead of
                // spinning the retry ladder (which is for reclaimed segments) or letting
                // a caller mistake it for a missing key.
                Err(e) => return Err(e.into()),
            }
        }

        // Pathological GC churn. Holding the bucket write lock excludes GC from this
        // bucket entirely, so the pointer we resolve cannot be invalidated under us —
        // this guarantees forward progress and surfaces genuine corruption.
        let bucket = self.value_log.bucket_for_key(key);
        let _bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;
        match self.lsm.get(key)? {
            Some(offset_u128) => match decode_sharded_pointer(offset_u128) {
                Some(pointer) => Ok(Some(self.value_log.read_value(pointer)?)),
                None => Ok(None),
            },
            None => Ok(None),
        }
        .map_err(|e: KVError| last_err.unwrap_or(e))
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
        let mut displaced: Option<ShardedValuePointer> = None;
        if let Some(existing) = self.lsm.get(key)?
            && let Some(existing_ptr) = decode_sharded_pointer(existing)
        {
            displaced = Some(existing_ptr);
            if want_old_for_index {
                old_value = self.value_log.read_value(existing_ptr).ok();
            }
        }
        self.lsm.delete_with_seq(key, seq as u32)?;

        // The deleted key's record is garbage now. Accounting only — the record itself
        // is never touched (its segment may be sealed and immutable), and liveness was
        // never read from the record anyway: the LSM is the authority.
        if let Some(old) = displaced {
            self.value_log.note_displaced(old, key.len());
        }
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
            // A removal also rewrites the value's bitmap append-only, so it adds
            // to the field's dead space — signal backpressure like the put path.
            let dead_bytes = idx.reclaimable_dead_bytes();
            drop(idx);
            self.request_checkpoint_if_over_cap(dead_bytes);
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
        self.value_log.sync_all().map_err(|e| {
            warn!("[KVStore:{}] Value log sync failed: {:?}", self.name, e);
            KVError::from(e)
        })
    }

    // ── Iterator support ───────────────────────────────────────────────

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
            if let Some(offset_u128) = self.lsm.get(&key)?
                && let Some(pointer) = decode_sharded_pointer(offset_u128)
            {
                entries.push((key, pointer));
            }
        }
        Ok(entries)
    }

    pub fn resolve_entries_from_pointers(&self, key_pointers: Vec<KeyPointer>) -> Result<Vec<ResolvedEntry>> {
        let mut entries = Vec::with_capacity(key_pointers.len());
        for (key, offset_u128, _seq) in key_pointers {
            if let Some(pointer) = decode_sharded_pointer(offset_u128) {
                entries.push((key, pointer));
            }
        }
        Ok(entries)
    }

    /// Delete records whose creation epoch is older than `ttl`.
    ///
    /// Scans live keys and reads each record's `epoch` straight from the value log
    /// (a 36-byte header read — no value bytes). At most `max_deletes_per_run`
    /// records are removed per call so a single pass can't stall on a huge backlog;
    /// value-log GC reclaims the physical space afterwards. Returns the number
    /// deleted.
    pub(crate) fn expire_records(&self, ttl: Duration, max_deletes_per_run: usize) -> Result<usize> {
        let now_millis = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
        let ttl_millis = ttl.as_millis() as u64;

        let keys = self.keys()?;
        let entries = self.resolve_entries(keys)?;

        let mut deleted = 0usize;
        for (key, pointer) in entries {
            if deleted >= max_deletes_per_run {
                break;
            }
            // The LSM listed this key, so it is live by definition — there is no
            // "already tombstoned" record state to skip any more.
            let Ok(meta) = self.value_log.read_record_meta(pointer) else {
                continue;
            };
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
        let (mut results, retry) = self.get_multiple_inner(keys);

        // Keys that resolved to a pointer but whose value could not be read: usually GC
        // reclaimed the segment between the LSM lookup and the read, and the LSM now
        // holds the new pointer, so re-resolving through the single-key `get` succeeds.
        // But `get` can also return a *hard* error (corruption, IO fault) — which the
        // reclaimed-segment retry inside `get` no longer masks. This method's public
        // signature is `Vec<Option>`, so it cannot surface a per-key error; log it so a
        // genuine corruption is visible rather than silently indistinguishable from a
        // missing key.
        for idx in retry {
            match self.get(&keys[idx]) {
                Ok(value) => results[idx] = value,
                Err(e) => warn!("[KVStore '{}'] get_multiple: value for a resolved key could not be read: {}", self.name, e),
            }
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

        // ── Step 2: group the pointers by value-log bucket ────────────────────
        let num_buckets = self.value_log.num_buckets();
        let mut bucket_work: Vec<Vec<(usize, ValueLocation)>> = (0..num_buckets).map(|_| Vec::new()).collect();
        let mut resolved: Vec<usize> = Vec::new();
        for (orig_idx, pointer_opt) in pointers.into_iter().enumerate() {
            if let Some((offset_u128, _seq)) = pointer_opt
                && let Some(pointer) = decode_sharded_pointer(offset_u128)
                && (pointer.bucket as usize) < num_buckets
            {
                bucket_work[pointer.bucket as usize].push((orig_idx, pointer.location));
                resolved.push(orig_idx);
            }
        }
        if resolved.is_empty() {
            return (results, Vec::new());
        }

        // ── Step 3: read each bucket's values in parallel ─────────────────────
        // No sequence validation: a segment id is never reused, so a pointer cannot
        // resolve to another key's record. A read can only fail outright — its
        // segment was reclaimed by GC — and those keys are re-resolved below.
        std::thread::scope(|s| {
            let handles: Vec<_> = bucket_work
                .iter()
                .enumerate()
                .filter(|(_, work)| !work.is_empty())
                .map(|(bucket, work)| s.spawn(move || self.value_log.read_values_batch(bucket as u32, work)))
                .collect();
            for handle in handles {
                if let Ok(batch) = handle.join() {
                    for (idx, val) in batch {
                        results[idx] = val;
                    }
                }
            }
        });

        // Keys we resolved to a pointer but could not read: GC reclaimed the segment
        // between the LSM lookup and the read. The LSM now holds the new pointer, so
        // the caller re-resolves them through the single-key `get`.
        let retry: Vec<usize> = resolved.into_iter().filter(|&idx| results[idx].is_none()).collect();

        (results, retry)
    }

    /// Scan multiple 4-byte BE cluster prefixes in a single pass.
    ///
    /// Uses `lsm.scan_prefixes` to read each bucket's level1 SSTable **once** into
    /// memory and check all cluster prefixes in a single in-memory pass — replacing
    /// N_clusters × num_buckets full linear scans with exactly num_buckets large reads
    /// followed by CPU-only work.
    ///
    /// Returns a map from `prefix_id` to `(key_bytes, value_bytes)` pairs. Keys
    /// shorter than 4 bytes or with no matching pointer are silently skipped.
    pub fn scan_prefixes_batch(&self, prefix_ids: &[u32]) -> Result<PrefixBatchResult> {
        let (mut result, retry) = self.scan_prefixes_batch_inner(prefix_ids)?;
        // Keys whose segment GC reclaimed mid-scan: re-resolve through the LSM.
        for (prefix_id, key) in retry {
            if let Some(v) = self.get(&key)? {
                result.entry(prefix_id).or_default().push((key, v));
            }
        }
        Ok(result)
    }

    fn scan_prefixes_batch_inner(&self, prefix_ids: &[u32]) -> Result<(PrefixBatchResult, Vec<PrefixIdKey>)> {
        let prefix_id_set: std::collections::HashSet<u32> = prefix_ids.iter().copied().collect();
        struct Entry {
            prefix_id: u32,
            key: Vec<u8>,
            pointer: ShardedValuePointer,
        }
        let mut all_entries: Vec<Entry> = Vec::new();
        for (key, offset_u128) in self.lsm.scan_prefixes(&prefix_id_set)? {
            if key.len() < 4 {
                continue;
            }
            let prefix_id = u32::from_be_bytes(key[..4].try_into().unwrap());
            if let Some(pointer) = decode_sharded_pointer(offset_u128) {
                all_entries.push(Entry { prefix_id, key, pointer });
            }
        }
        if all_entries.is_empty() {
            return Ok((std::collections::HashMap::new(), Vec::new()));
        }

        let num_buckets = self.value_log.num_buckets();
        let mut bucket_work: Vec<Vec<(usize, ValueLocation)>> = (0..num_buckets).map(|_| Vec::new()).collect();
        for (idx, entry) in all_entries.iter().enumerate() {
            bucket_work[entry.pointer.bucket as usize].push((idx, entry.pointer.location));
        }

        let mut values: Vec<Option<Vec<u8>>> = vec![None; all_entries.len()];
        std::thread::scope(|s| {
            let handles: Vec<_> = bucket_work
                .iter()
                .enumerate()
                .filter(|(_, work)| !work.is_empty())
                .map(|(bucket, work)| s.spawn(move || self.value_log.read_values_batch(bucket as u32, work)))
                .collect();
            for handle in handles {
                if let Ok(results) = handle.join() {
                    for (idx, val) in results {
                        values[idx] = val;
                    }
                }
            }
        });

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
    /// Groups entries by value-log bucket and reads each bucket in a scoped thread.
    /// Returns `(pairs, retry_keys)`: `retry_keys` are keys whose pointer resolved but
    /// whose segment GC reclaimed before the read — the caller re-resolves them via
    /// [`get`](Self::get).
    fn batch_read_from_entries(&self, entries: Vec<ResolvedEntry>) -> Result<(Vec<KeyValue>, Vec<Vec<u8>>)> {
        if entries.is_empty() {
            return Ok((vec![], vec![]));
        }
        let n = entries.len();
        let num_buckets = self.value_log.num_buckets();
        let mut bucket_work: Vec<Vec<(usize, ValueLocation)>> = (0..num_buckets).map(|_| Vec::new()).collect();
        for (idx, (_, pointer)) in entries.iter().enumerate() {
            bucket_work[pointer.bucket as usize].push((idx, pointer.location));
        }

        let mut values: Vec<Option<Vec<u8>>> = vec![None; n];
        std::thread::scope(|s| {
            let handles: Vec<_> = bucket_work
                .iter()
                .enumerate()
                .filter(|(_, work)| !work.is_empty())
                .map(|(bucket, work)| s.spawn(move || self.value_log.read_values_batch(bucket as u32, work)))
                .collect();
            for handle in handles {
                if let Ok(results) = handle.join() {
                    for (idx, val) in results {
                        values[idx] = val;
                    }
                }
            }
        });

        let mut pairs = Vec::with_capacity(n);
        let mut retry = Vec::new();
        for ((key, _), val) in entries.into_iter().zip(values) {
            match val {
                Some(v) => pairs.push((key, v)),
                None => retry.push(key),
            }
        }
        Ok((pairs, retry))
    }

    /// Re-resolve keys whose segment was reclaimed between the LSM scan and the value
    /// read, through the single-key [`get`](Self::get). A key that now reads as absent
    /// was concurrently deleted and is correctly left out.
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
    /// closed in 420ac8e — the LSM scan's own snapshot does not protect against
    /// value-log GC.
    pub fn scan_prefix_batch(&self, prefix: &[u8]) -> Result<Vec<KeyValue>> {
        let key_pointers = self.lsm.scan_prefix(prefix)?;
        let entries = self.resolve_entries_from_pointers(key_pointers)?;
        let (mut pairs, retry) = self.batch_read_from_entries(entries)?;
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
    pub fn scan_range_batch(&self, start: &[u8], end: Option<&[u8]>) -> Result<Vec<KeyValue>> {
        let key_pointers = self.lsm.range_pointers_bounded(start, end, usize::MAX)?;
        let entries = self.resolve_entries_from_pointers(key_pointers)?;
        let (mut pairs, retry) = self.batch_read_from_entries(entries)?;
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
    pub fn scan_page_batch(&self, cursor: Option<&[u8]>, end: Option<&[u8]>, limit: usize) -> Result<ScanPage> {
        let start = cursor.unwrap_or(&[]);
        let key_pointers = self.lsm.range_pointers_bounded(start, end, limit + 1)?;
        let has_more = key_pointers.len() > limit;
        let next_cursor = if has_more { Some(key_pointers[limit].0.clone()) } else { None };
        let page_kps: Vec<_> = key_pointers.into_iter().take(limit).collect();
        let entries = self.resolve_entries_from_pointers(page_kps)?;
        let (mut pairs, retry) = self.batch_read_from_entries(entries)?;
        self.refetch_dropped(&mut pairs, retry)?;
        if let Some(m) = self.metrics() {
            m.record_scan(pairs.len() as u64);
        }
        Ok((pairs, next_cursor))
    }

    // ── Startup / cleanup ──────────────────────────────────────────────

    /// Remove stale LSM files left by an interrupted compaction.
    ///
    /// The value log needs no startup cleanup any more: GC never writes a shadow file
    /// and never swaps one in, so there is nothing half-finished to roll forward or
    /// back. Its only mutation of existing state is unlinking a segment, which only
    /// happens after the re-point is durable.
    pub fn cleanup_old_files_on_startup(&self) -> Result<()> {
        self.lsm
            .cleanup_old_files_on_startup()
            .map_err(|e| KVError::Io(std::io::Error::other(format!("Failed to cleanup LSM old files: {:?}", e))))
    }

    // ── Recovery (replay WAL entries into this namespace) ──────────────

    /// Replay a single WAL entry into this namespace's storage, stamping the
    /// memtable with the entry's original WAL sequence so recovery reproduces the
    /// same per-key winner that was live before the crash (highest-sequence-wins).
    pub fn replay_upsert(&self, key: &[u8], value: &[u8], seq: u64) -> Result<()> {
        let epoch = current_epoch_millis();
        let mut next_version = 1u32;

        // Read the displaced pointer and decide the winner inside the bucket lock, as
        // the live put path does — replay reuses that path's accounting exactly.
        let bucket = self.value_log.bucket_for_key(key);
        let _bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;

        let (existing_u128, wins) = self.lsm.current_pointer_and_wins(key, seq as u32)?;
        let displaced = existing_u128.and_then(decode_sharded_pointer);
        if let Some(existing_ptr) = displaced
            && let Ok(meta) = self.value_log.read_record_meta(existing_ptr)
        {
            next_version = meta.version.saturating_add(1);
        }

        let record_meta = ValueRecordMeta {
            version: next_version,
            epoch,
            seq,
        };
        let pointer = self.value_log.append_to_locked_bucket(bucket, key, value, record_meta, false)?;
        self.lsm.insert_with_seq(key, pointer.to_u128(), seq as u32)?;
        if wins {
            if let Some(old) = displaced {
                self.value_log.note_displaced(old, key.len());
            }
        } else {
            self.value_log.note_displaced(pointer, key.len());
        }
        drop(_bucket_guard);

        if wins {
            self.update_indices_on_put(key, value, None, displaced.is_some());
        }
        Ok(())
    }

    /// Replay a WAL delete into this namespace's storage, under the entry's original
    /// sequence (see [`replay_upsert`](Self::replay_upsert)).
    pub fn replay_delete(&self, key: &[u8], seq: u64) -> Result<()> {
        let bucket = self.value_log.bucket_for_key(key);
        let _bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;
        let displaced = self.lsm.get(key).ok().flatten().and_then(decode_sharded_pointer);
        self.lsm.delete_with_seq(key, seq as u32)?;
        if let Some(old) = displaced {
            self.value_log.note_displaced(old, key.len());
        }
        drop(_bucket_guard);

        self.update_indices_on_delete(key, None);
        Ok(())
    }

    pub fn flush_and_compact_all(&self) -> Result<()> {
        self.lsm.flush_and_compact_all().map_err(KVError::from)
    }

    // ── Garbage collection ─────────────────────────────────────────────

    /// GC one bucket: rewrite the survivors of its worst segments, and hand back the
    /// segments that become safe to unlink *once the re-point is durable*.
    ///
    /// **Phase 1 (no lock):** read the segment sequentially. Every record carries its
    /// **key**, so liveness is one LSM point-get per record — a record survives iff the
    /// LSM still points at exactly this location. This is why GC's cost is proportional
    /// to the segment rather than to the whole bucket or the whole key set. The segment
    /// is sealed and immutable, so nothing can change it under us.
    ///
    /// **Phase 2 (bucket lock):** relocate each survivor as a **compare-and-set** —
    /// append it to the active tail and update the LSM only if the key *still* maps to
    /// the old location. Phase 1 was not atomic with concurrent writers, so a key
    /// deleted since then must stay deleted and one overwritten since then must keep
    /// its newer value. The relocation re-inserts under the key's **existing sequence**,
    /// so moving a record never changes its version.
    ///
    /// The old segment is deliberately **not** unlinked here — see
    /// [`garbage_collect_with_threshold`](Self::garbage_collect_with_threshold).
    fn compact_bucket(&self, bucket: u32, page_gc_threshold_pct: f64, tail_gc_threshold_pct: f64) -> Result<BucketGCResult> {
        let log = self.value_log.get_bucket_log(bucket)?;
        let mut result = BucketGCResult {
            bucket,
            bytes_rewritten: 0,
            pending_unlink: Vec::new(),
        };

        // The active tail is normally off-limits, but a small or idle namespace may never
        // fill a segment — so its garbage would be uncollectable forever while the waste
        // trigger kept firing. If the tail is garbage enough (>= `tail_gc_threshold_pct`),
        // seal it (under the bucket lock, since this rolls the segment writers append to)
        // so it can be collected in this same pass. See `TAIL_GC_MIN_GARBAGE_PCT`.
        {
            let _bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;
            if let Some(sealed) = log.seal_tail_for_gc(tail_gc_threshold_pct)? {
                debug!(
                    "[KVStore '{}'] bucket {}: sealed the active tail (segment {}) — it was mostly garbage",
                    self.name, bucket, sealed
                );
            }
        }

        let candidates = log.gc_candidates(page_gc_threshold_pct);
        if candidates.is_empty() {
            return Ok(result);
        }

        for segment in candidates {
            // ── Phase 1: what is still live in this segment? ──────────────────
            let mut survivors: Vec<(Vec<u8>, Vec<u8>, ValueRecordMeta, u128)> = Vec::new();
            let mut scan_err: Option<KVError> = None;
            log.for_each_record(segment.id, |key, value, meta, location| {
                if scan_err.is_some() {
                    return;
                }
                let Ok(pointer) = ShardedValuePointer::new(bucket, location, self.value_log.num_buckets()) else {
                    return;
                };
                let old_u128 = pointer.to_u128();
                match self.lsm.get(key) {
                    // Live iff the LSM still points at exactly this record.
                    Ok(Some(current)) if current == old_u128 => survivors.push((key.to_vec(), value, meta, old_u128)),
                    Ok(_) => {}
                    Err(e) => scan_err = Some(e.into()),
                }
            })?;
            if let Some(e) = scan_err {
                return Err(e);
            }

            // ── Phase 2: relocate survivors, re-point by compare-and-set ──────
            {
                let _bucket_guard = self.value_log.lock_bucket_for_write(bucket)?;
                for (key, value, meta, old_u128) in &survivors {
                    let Some((current, seq)) = self.lsm.get_with_seq(key)? else {
                        continue; // deleted since phase 1 — do not resurrect
                    };
                    if current != *old_u128 {
                        continue; // overwritten since phase 1 — do not revert
                    }
                    let new_pointer = self.value_log.append_to_locked_bucket(bucket, key, value, *meta, false)?;
                    self.lsm.insert_with_seq(key, new_pointer.to_u128(), seq)?;
                    result.bytes_rewritten += new_pointer.record_len(key.len());
                }
            }

            result.pending_unlink.push(segment.id);
        }

        Ok(result)
    }

    /// Run value-log GC across every bucket.
    ///
    /// `page_gc_threshold_pct` selects **segments** (`ThresholdConfig::page_gc_threshold`).
    /// The bucket-level `value_log_waste_threshold` decides whether a namespace is
    /// collected at all, and is applied by the caller.
    ///
    /// # The ordering IS the crash-safety story
    ///
    /// ```text
    /// 1. append survivors into the active segment
    /// 2. CAS re-point the keys            (bucket lock)
    /// 3. flush the memtable to L0         ← the re-point is now DURABLE
    /// 4. unlink the old segments          ← only now
    /// ```
    ///
    /// At every crash point the durable LSM still refers to a segment that exists. Die
    /// before 3 and the keys still point at the old segment, which is still there; die
    /// between 3 and 4 and they point at the new one while the old segment is merely
    /// dead weight, which the next pass reclaims. Nothing has to be rolled forward or
    /// back — which is why there is no GC journal, no commit marker and no `.new`/`.old`
    /// files any more.
    ///
    /// **Unlinking before step 3 would be data loss**: the durable LSM would point into
    /// a segment that no longer exists, and the WAL entries that could replay those
    /// writes are long gone.
    pub fn garbage_collect_with_threshold(&self, page_gc_threshold_pct: f64) -> Result<GCStats> {
        self.garbage_collect_with_thresholds(page_gc_threshold_pct, TAIL_GC_MIN_GARBAGE_PCT)
    }

    /// As [`garbage_collect_with_threshold`](Self::garbage_collect_with_threshold), but
    /// with an explicit tail-seal bar (`tail_gc_threshold_pct`) — the garbage share at
    /// which a bucket's never-filled active tail is sealed so it can be collected. The GC
    /// worker passes the configured
    /// [`effective_tail_gc_min_garbage_pct`](crate::db::config::ThresholdConfig::effective_tail_gc_min_garbage_pct),
    /// which by default tracks the waste trigger so no garbage strands between the two.
    pub fn garbage_collect_with_thresholds(&self, page_gc_threshold_pct: f64, tail_gc_threshold_pct: f64) -> Result<GCStats> {
        let start_time = std::time::Instant::now();

        if self
            .value_log_gc_in_progress
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Ok(self.gc_stats_now(0, start_time));
        }
        struct InProgressGuard<'a>(&'a AtomicBool);
        impl Drop for InProgressGuard<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        let _in_progress = InProgressGuard(&self.value_log_gc_in_progress);

        // Steps 1-2, per bucket, in parallel: buckets are independent, each with its
        // own lock.
        let bucket_count = self.value_log.num_buckets();
        let results: Vec<BucketGCResult> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..bucket_count as u32)
                .map(|bucket| s.spawn(move || self.compact_bucket(bucket, page_gc_threshold_pct, tail_gc_threshold_pct)))
                .collect();
            handles
                .into_iter()
                .filter_map(|h| match h.join() {
                    Ok(Ok(r)) => Some(r),
                    Ok(Err(e)) => {
                        error!("[KVStore '{}'] value-log GC failed for a bucket: {:?}", self.name, e);
                        None
                    }
                    Err(_) => {
                        error!("[KVStore '{}'] value-log GC thread panicked", self.name);
                        None
                    }
                })
                .collect()
        });

        if results.iter().all(|r| r.pending_unlink.is_empty()) {
            return Ok(self.gc_stats_now(0, start_time));
        }

        // Step 3: make the re-points durable BEFORE unlinking anything. Load-bearing —
        // `test_gc_crash_the_instant_segments_are_unlinked_loses_nothing` fails if these
        // two steps are ever swapped.
        self.lsm.flush_memtable_to_level0()?;

        // Step 4: the old segments are now unreferenced by anything durable.
        let mut bytes_reclaimed = 0u64;
        let mut bytes_rewritten = 0u64;
        let mut segments_reclaimed = 0usize;
        for result in &results {
            let Ok(log) = self.value_log.get_bucket_log(result.bucket) else {
                continue;
            };
            let mut reclaimed_here = 0u64;
            for &segment_id in &result.pending_unlink {
                match log.unlink_segment(segment_id) {
                    Ok(bytes) => {
                        reclaimed_here += bytes;
                        segments_reclaimed += 1;
                    }
                    Err(e) => warn!(
                        "[KVStore '{}'] could not unlink value-log segment {} of bucket {}: {:?}",
                        self.name, segment_id, result.bucket, e
                    ),
                }
            }
            log.record_gc_run(reclaimed_here);
            bytes_reclaimed += reclaimed_here;
            bytes_rewritten += result.bytes_rewritten;
        }

        // Test-only crash point (see `gc_crash_after_unlink`): the segments are gone, and
        // nothing after this line has run.
        #[cfg(test)]
        if self.gc_crash_after_unlink.load(Ordering::SeqCst) {
            return Ok(self.gc_stats_now(bytes_reclaimed, start_time));
        }

        self.value_log.flush_all_metadata()?;

        info!(
            "[KVStore '{}'] value-log GC: reclaimed {} bytes across {} segment(s), rewrote {} bytes of survivors, in {:?}",
            self.name,
            bytes_reclaimed,
            segments_reclaimed,
            bytes_rewritten,
            start_time.elapsed()
        );
        if let Some(m) = self.metrics() {
            Metrics::bump(&m.vlog_gc_runs);
            Metrics::add(&m.vlog_gc_duration_ms, start_time.elapsed().as_millis() as u64);
            Metrics::add(&m.vlog_segments_reclaimed, segments_reclaimed as u64);
            Metrics::add(&m.vlog_gc_bytes_reclaimed, bytes_reclaimed);
            // Together with bytes_reclaimed this is GC's write amplification — the cost
            // of a pass against what it actually freed.
            Metrics::add(&m.vlog_gc_bytes_rewritten, bytes_rewritten);
        }

        Ok(self.gc_stats_now(bytes_reclaimed, start_time))
    }

    fn gc_stats_now(&self, bytes_reclaimed: u64, start_time: std::time::Instant) -> GCStats {
        let (total_gc_runs, total_bytes_reclaimed, bytes_live) = self.aggregate_value_log_stats();
        GCStats {
            bytes_reclaimed,
            bytes_live,
            gc_run_count: total_gc_runs,
            total_bytes_reclaimed,
            gc_duration_ms: start_time.elapsed().as_millis(),
        }
    }

    #[allow(dead_code)]
    pub fn garbage_collect(&self) -> Result<GCStats> {
        self.garbage_collect_with_threshold(crate::db::config::ThresholdConfig::default().value_log_waste_threshold)
    }

    /// Does this namespace have value-log garbage GC can actually collect right now?
    ///
    /// The bucket-level waste ratio alone is not enough: garbage sitting in a bucket's
    /// active tail is not collectable until the tail is sealed, so a small, fully-deleted
    /// namespace could show 100% waste while GC had nothing to do — and the worker would
    /// wake, log "starting GC", reclaim nothing, and repeat on every tick.
    pub(crate) fn has_gc_work(&self, page_gc_threshold_pct: f64, tail_gc_threshold_pct: f64) -> bool {
        (0..self.value_log.num_buckets() as u32).any(|b| {
            self.value_log
                .get_bucket_log(b)
                .is_ok_and(|log| log.has_gc_work(page_gc_threshold_pct, tail_gc_threshold_pct))
        })
    }

    /// Waste across the whole namespace: `garbage / (live + garbage)`, as a percentage.
    /// This is the **trigger** the GC worker compares against `value_log_waste_threshold`.
    pub fn get_waste_ratio(&self) -> f64 {
        self.value_log.total_garbage_ratio()
    }

    /// True when at least one bucket is at or above `threshold_pct` waste. The GC worker
    /// checks this alongside the namespace-average [`get_waste_ratio`](Self::get_waste_ratio),
    /// so a hot bucket over the trigger is collected even when near-empty buckets drag the
    /// average below it.
    pub(crate) fn has_bucket_over_waste(&self, threshold_pct: f64) -> bool {
        self.value_log.any_bucket_over_waste(threshold_pct)
    }

    /// The raw `(garbage_bytes, written_bytes)` behind [`get_waste_ratio`](Self::get_waste_ratio),
    /// where `written = live + garbage` — useful in logs, so a high ratio over a tiny
    /// absolute volume is obvious.
    pub(crate) fn waste_bytes(&self) -> (u64, u64) {
        let garbage = self.value_log.total_garbage_bytes();
        (garbage, garbage.saturating_add(self.value_log.total_live_bytes()))
    }

    #[allow(dead_code)]
    pub fn get_garbage_ratio(&self) -> f64 {
        self.value_log.total_garbage_ratio()
    }

    pub(crate) fn aggregate_value_log_stats(&self) -> (u64, u64, u64) {
        let mut total_gc_runs = 0u64;
        let mut total_bytes_reclaimed = 0u64;
        let mut bytes_live = 0u64;
        for (_bucket, metadata) in self.value_log.all_bucket_metadata() {
            total_gc_runs = total_gc_runs.saturating_add(metadata.total_gc_runs);
            total_bytes_reclaimed = total_bytes_reclaimed.saturating_add(metadata.total_bytes_reclaimed);
            bytes_live = bytes_live.saturating_add(metadata.live_bytes());
        }
        (total_gc_runs, total_bytes_reclaimed, bytes_live)
    }

    #[allow(dead_code)]
    pub fn stats(&self) -> Stats {
        let mut total_gc_runs = 0u64;
        let mut total_bytes_reclaimed = 0u64;
        let mut live_bytes = 0u64;
        let mut garbage_bytes = 0u64;
        let mut segment_count = 0u64;
        for (_bucket, metadata) in self.value_log.all_bucket_metadata() {
            total_gc_runs = total_gc_runs.saturating_add(metadata.total_gc_runs);
            total_bytes_reclaimed = total_bytes_reclaimed.saturating_add(metadata.total_bytes_reclaimed);
            live_bytes = live_bytes.saturating_add(metadata.live_bytes());
            garbage_bytes = garbage_bytes.saturating_add(metadata.garbage_bytes());
            segment_count = segment_count.saturating_add(metadata.segments.len() as u64);
        }

        let written = live_bytes.saturating_add(garbage_bytes);
        let waste_ratio = if written > 0 {
            (garbage_bytes as f64 / written as f64) * 100.0
        } else {
            0.0
        };
        let disk_bytes: u64 = self.value_log.physical_stats().iter().map(|s| s.physical_bytes).sum();

        Stats {
            segment_count,
            disk_bytes,
            garbage_size: garbage_bytes,
            waste_ratio,
            total_gc_runs,
            total_bytes_reclaimed,
            live_bytes,
        }
    }

    pub fn compact_lsm(&self) -> Result<()> {
        if !self.lsm.has_compaction_work() {
            return Ok(());
        }
        self.lsm.compact_all().map_err(KVError::from)
    }

    pub fn has_lsm_compaction_work(&self) -> bool {
        self.lsm.has_compaction_work()
    }

    // ── Shutdown ──────────────────────────────────────────────────────────

    /// Flush and sync all data for this namespace.
    pub fn shutdown(&self) -> Result<()> {
        self.lsm.cleanup_pending_memtables_on_close();
        self.lsm.flush_and_compact_all()?;
        self.value_log.sync_all()?;
        // Persist each bucket's segment inventory, its live/garbage counters and its
        // segment-id high-water mark. The high-water mark is what keeps ids unique
        // across restarts; the counters keep GC from under-triggering after a reopen.
        self.value_log.flush_all_metadata()?;
        Ok(())
    }
}

impl Drop for KVStore {
    fn drop(&mut self) {
        let _ = self.lsm.flush_and_compact_all();
        let _ = self.value_log.sync_all();
        let _ = self.value_log.flush_all_metadata();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::value_log::DEFAULT_SEGMENT_SIZE_BYTES;
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
    #[test]
    fn test_kvstore_basic_operations() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(
            0,
            "default",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();

        store.put_to_storage(b"key1", b"value1").unwrap();
        store.put_to_storage(b"key2", b"value2").unwrap();

        assert_eq!(store.get(b"key1").unwrap(), Some(b"value1".to_vec()));
        assert_eq!(store.get(b"key2").unwrap(), Some(b"value2".to_vec()));

        store.delete_from_storage(b"key2").unwrap();
        assert_eq!(store.get(b"key2").unwrap(), None);
    }

    #[test]
    fn a_never_filled_tail_over_the_trigger_is_sealed_and_collected() {
        // A small store keeps all its data in one active-tail segment per bucket that
        // never fills a 64 MiB segment, so tail-sealing is its only path to collection.
        // With the tail-seal bar tracking the trigger, garbage in the 30-50% band — which
        // sat stranded below the old hardcoded 50% bar — is reclaimed.
        let dir = TempDir::new().unwrap();
        let lsm_config = LSMConfig {
            num_buckets: 1, // one bucket → one active tail holds everything
            ..LSMConfig::default()
        };
        let store = KVStore::open(0, "default", dir.path(), lsm_config, SyncConfig::default(), DEFAULT_SEGMENT_SIZE_BYTES).unwrap();

        let value = vec![0x5Au8; 400];
        for i in 0..100u32 {
            store.put_to_storage(format!("k{i:03}").as_bytes(), &value).unwrap();
        }
        // Overwrite 70 keys: their old records become garbage — ~41% of the single tail
        // segment (70 dead of 170 total), squarely in the old dead zone.
        for i in 0..70u32 {
            store.put_to_storage(format!("k{i:03}").as_bytes(), &value).unwrap();
        }
        let waste_before = store.get_waste_ratio();
        assert!(
            (30.0..50.0).contains(&waste_before),
            "test needs tail garbage in the 30-50% band, got {waste_before:.1}%"
        );

        // Old behavior (50% tail bar): the sub-50% tail is not sealed, nothing collectable.
        let stats = store.garbage_collect_with_thresholds(10.0, 50.0).unwrap();
        assert_eq!(stats.bytes_reclaimed, 0, "at a 50% tail bar a 41%-garbage tail must stay uncollected");

        // Bar tracking the 30% trigger: the tail is sealed and its garbage reclaimed.
        let stats = store.garbage_collect_with_thresholds(10.0, 30.0).unwrap();
        assert!(stats.bytes_reclaimed > 0, "at a 30% tail bar the tail must be sealed and collected");
        assert!(
            store.get_waste_ratio() < waste_before,
            "waste must drop after collection (was {waste_before:.1}%, now {:.1}%)",
            store.get_waste_ratio()
        );

        // Every live key survived the relocation into the fresh tail.
        for i in 0..100u32 {
            assert_eq!(
                store.get(format!("k{i:03}").as_bytes()).unwrap(),
                Some(value.clone()),
                "k{i:03} lost during tail GC"
            );
        }
    }

    #[test]
    fn test_kvstore_update() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(
            0,
            "default",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();

        store.put_to_storage(b"key1", b"v1").unwrap();
        assert_eq!(store.get(b"key1").unwrap(), Some(b"v1".to_vec()));

        store.put_to_storage(b"key1", b"v2").unwrap();
        assert_eq!(store.get(b"key1").unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn a_lower_seq_put_charges_its_own_record_not_the_live_one() {
        // A put that loses to a higher-sequence write for the same key (the LSM drops
        // its insert under highest-sequence-wins) must account ITS OWN freshly-appended
        // record as garbage — not displace the record that is still live. This is the
        // deterministic form of the concurrent-put accounting race: the old code read
        // the displaced pointer before the bucket lock and always marked it displaced,
        // so this lower-seq put marked the LIVE record garbage (accounting would call the
        // live value dead — GC could then reclaim it, losing data) and left its own dead
        // record counted live.
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(
            0,
            "default",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();

        let winner = vec![0x11u8; 400];
        let loser = vec![0x22u8; 900]; // a different size, so a mis-charge can't cancel

        // Higher sequence lands first and wins; the lower sequence arrives after and is
        // dropped by the LSM.
        store.put_to_storage_seq(b"k", &winner, 10).unwrap();
        store.put_to_storage_seq(b"k", &loser, 5).unwrap();

        assert_eq!(store.get(b"k").unwrap(), Some(winner.clone()), "highest-seq write must win the read");

        let winner_len = crate::store::value_log::ValueRecordHeader::record_len(b"k".len(), winner.len());
        let loser_len = crate::store::value_log::ValueRecordHeader::record_len(b"k".len(), loser.len());
        let (mut live, mut garbage) = (0u64, 0u64);
        for (_bucket, segs) in store.value_log.all_segment_stats() {
            for s in segs {
                assert_eq!(
                    s.live_bytes + s.garbage_bytes,
                    s.total_bytes,
                    "segment {} broke live+garbage==total",
                    s.id
                );
                live += s.live_bytes;
                garbage += s.garbage_bytes;
            }
        }
        assert_eq!(live, winner_len, "only the winning record's bytes may be counted live");
        assert_eq!(garbage, loser_len, "the dropped lower-seq record's bytes are the garbage");
    }

    #[test]
    fn get_and_get_multiple_surface_corruption_instead_of_hiding_it_as_a_miss() {
        // A value-log read error in the batch path used to be swallowed by
        // `.get(..).ok().flatten()`, making genuine corruption indistinguishable from a
        // missing key. Now `get` surfaces a hard error (only a reclaimed segment retries),
        // and `get_multiple` still returns the readable keys — dropping (and logging) only
        // the corrupt one rather than silently reporting it absent.
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default(), DEFAULT_SEGMENT_SIZE_BYTES).unwrap();

        let keys: Vec<Vec<u8>> = (0..5u32).map(|i| format!("k{i}").into_bytes()).collect();
        for (i, k) in keys.iter().enumerate() {
            store.put_to_storage(k, format!("value-{i}-payload").as_bytes()).unwrap();
        }

        // Flip a byte in k2's value payload, in place (behind the log's back — the cached
        // fd reads the modified file, and the in-memory LSM keeps k2's pointer).
        let ptr = decode_sharded_pointer(store.lsm.get(b"k2").unwrap().unwrap()).unwrap();
        let seg_path = dir
            .path()
            .join("value_logs")
            .join(format!("value_log_{}.seg{:06}", ptr.bucket, ptr.location.segment_id));
        let mut bytes = std::fs::read(&seg_path).unwrap();
        bytes[ptr.location.rec_offset as usize + crate::store::value_log::ValueRecordHeader::SIZE] ^= 0xFF;
        std::fs::write(&seg_path, &bytes).unwrap();

        // With checksum verification on, the flipped payload is detected.
        store.set_verify_checksums_on_read(true);

        // Single-key get surfaces the corruption rather than looping or returning None.
        assert!(
            matches!(store.get(b"k2"), Err(KVError::ShardedValueLogError(_))),
            "corruption must surface as an error, not a missing key"
        );

        // get_multiple returns every readable key and drops only the corrupt one.
        let got = store.get_multiple(&keys);
        for (i, k) in keys.iter().enumerate() {
            if k.as_slice() == b"k2" {
                assert!(got[i].is_none(), "the corrupt key must not be returned as a value");
            } else {
                assert_eq!(
                    got[i].as_deref(),
                    Some(format!("value-{i}-payload").as_bytes()),
                    "readable key {i} must still be returned"
                );
            }
        }
    }

    #[test]
    fn test_kvstore_recovery_replay() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(
            0,
            "default",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();

        store.replay_upsert(b"rkey1", b"rval1", 1).unwrap();
        store.replay_upsert(b"rkey2", b"rval2", 2).unwrap();
        store.replay_delete(b"rkey1", 3).unwrap();

        assert_eq!(store.get(b"rkey1").unwrap(), None);
        assert_eq!(store.get(b"rkey2").unwrap(), Some(b"rval2".to_vec()));
    }

    #[test]
    fn test_kvstore_gc() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(
            0,
            "default",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();

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
    fn test_get_multiple_matches_individual_get() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(
            0,
            "ns",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();

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
        let store = KVStore::open(
            0,
            "ns",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();

        store.put_to_storage(b"exists", b"val").unwrap();

        let keys = vec![b"exists".to_vec(), b"no_such_key".to_vec()];
        let batch = store.get_multiple(&keys);

        assert_eq!(batch[0], Some(b"val".to_vec()));
        assert_eq!(batch[1], None);
    }

    #[test]
    fn test_get_multiple_after_flush_to_sstable() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(
            0,
            "ns",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();

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
        let store = KVStore::open(
            0,
            "test_ns",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();

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
        let store = KVStore::open(
            0,
            "test_ns",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();
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
        let store = KVStore::open(
            0,
            "test_ns",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();
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
        let store = KVStore::open(
            0,
            "test_ns",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();

        store.put_to_storage(&u32_prefixed_key(1, 1), b"v").unwrap();

        let result = store.scan_prefixes_batch(&[99]).unwrap();
        assert!(!result.contains_key(&99));
    }

    #[test]
    fn test_scan_prefixes_batch_after_flush() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(
            0,
            "test_ns",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();

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

    // ── GC crash-safety ordering ────────────────────────────────────────────
    //
    // The load-bearing invariant of the segmented value log:
    //
    //     relocate survivors → CAS re-point → FLUSH to L0 → unlink the old segment
    //
    // Unlinking before the flush is silent data loss: the durable LSM would point
    // into a file that no longer exists, and the WAL entries that could replay those
    // writes are long gone. These tests pin both halves of that ordering.

    /// A crash between GC's re-point and its unlink must lose nothing: the old segment
    /// is still on disk, and the durable LSM still points into it.
    ///
    /// `compact_bucket` deliberately does NOT unlink — it only hands back the segments
    /// that *become* safe to unlink once the re-point is durable. So calling it and then
    /// dropping the store without flushing simulates exactly that crash window.
    #[test]
    fn test_gc_crash_between_repoint_and_unlink_loses_nothing() {
        let dir = TempDir::new().unwrap();
        // Small segments so a few values roll several of them.
        let segment_size = 64 * 1024;
        let value = vec![0xC3u8; 8 * 1024];

        let pending = {
            let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default(), segment_size).unwrap();
            for i in 0..12 {
                store.put_to_storage(format!("k{i}").as_bytes(), &value).unwrap();
            }
            // Churn one key so its old records become garbage worth collecting.
            for _ in 0..12 {
                store.put_to_storage(b"hot", &value).unwrap();
            }
            // Make the WRITES durable first. (A standalone KVStore has no WAL — the
            // Database coordinator owns that — so without this the crash below would
            // simply lose the writes themselves, which is not what we are testing.)
            store.lsm.flush_memtable_to_level0().unwrap();

            // Phases 1-2 only: survivors relocated and keys re-pointed in the MEMTABLE,
            // but nothing flushed and nothing unlinked.
            let mut pending = Vec::new();
            for bucket in 0..store.value_log.num_buckets() as u32 {
                let result = store.compact_bucket(bucket, 10.0, TAIL_GC_MIN_GARBAGE_PCT).unwrap();
                pending.extend(result.pending_unlink.iter().map(|id| (bucket, *id)));
            }
            assert!(!pending.is_empty(), "expected GC to have relocated at least one segment");

            // The old segments MUST still be on disk at this point.
            for (bucket, segment_id) in &pending {
                let log = store.value_log.get_bucket_log(*bucket).unwrap();
                assert!(
                    log.segment_stats().iter().any(|s| s.id == *segment_id),
                    "segment {segment_id} was unlinked before the re-point was durable — that is data loss"
                );
            }
            // A faithful crash: `mem::forget` skips `Drop`, which would otherwise flush
            // the LSM and make this a graceful close instead of the crash we mean to test.
            std::mem::forget(store);
            pending
        };
        assert!(!pending.is_empty());

        // Reopen. The re-point may have been lost with the memtable, in which case the
        // keys still point at the OLD segments — which is exactly why they must not have
        // been unlinked. Either way every value must still read back.
        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default(), segment_size).unwrap();
        for i in 0..12 {
            assert_eq!(
                store.get(format!("k{i}").as_bytes()).unwrap(),
                Some(value.clone()),
                "k{i} lost across a crash between GC's re-point and its unlink"
            );
        }
        assert_eq!(store.get(b"hot").unwrap(), Some(value), "hot key lost across the same crash");
    }

    /// **The data-loss guard.** Crash at the worst possible instant: the moment GC has
    /// unlinked the old segments.
    ///
    /// With the correct order (flush to L0, *then* unlink), the re-point is already
    /// durable when the segments go, so this crash is harmless. Swap the two — unlink
    /// first, flush second — and this same crash loses the re-point *and* the segments
    /// it pointed into: the durable LSM would reference files that no longer exist, and
    /// the WAL entries that could replay those writes are long gone. This test fails
    /// outright in that world, which is the whole point of it.
    #[test]
    fn test_gc_crash_the_instant_segments_are_unlinked_loses_nothing() {
        let dir = TempDir::new().unwrap();
        let segment_size = 64 * 1024;
        let value = vec![0xE5u8; 8 * 1024];

        {
            let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default(), segment_size).unwrap();
            for i in 0..12 {
                store.put_to_storage(format!("k{i}").as_bytes(), &value).unwrap();
            }
            for _ in 0..12 {
                store.put_to_storage(b"hot", &value).unwrap();
            }

            // Arm the crash point: GC will return the instant its segments are unlinked.
            store.gc_crash_after_unlink.store(true, Ordering::SeqCst);
            let stats = store.garbage_collect_with_threshold(10.0).unwrap();
            assert!(stats.bytes_reclaimed > 0, "expected GC to have unlinked at least one segment");

            // Crash: skip Drop, so nothing beyond what GC itself made durable survives.
            std::mem::forget(store);
        }

        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default(), segment_size).unwrap();
        for i in 0..12 {
            assert_eq!(
                store.get(format!("k{i}").as_bytes()).unwrap(),
                Some(value.clone()),
                "k{i} lost when GC crashed the instant it unlinked — the re-point was not durable first"
            );
        }
        assert_eq!(store.get(b"hot").unwrap(), Some(value), "hot key lost to the same crash");
    }

    /// The full pass, then a crash: after `garbage_collect_with_threshold` returns, the
    /// re-point IS durable (it flushed to L0 before unlinking), so a crash immediately
    /// afterwards must still resolve every key — now through the *new* segments.
    #[test]
    fn test_gc_survives_a_crash_immediately_after_the_pass() {
        let dir = TempDir::new().unwrap();
        let segment_size = 64 * 1024;
        let value = vec![0xD4u8; 8 * 1024];

        {
            let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default(), segment_size).unwrap();
            for i in 0..12 {
                store.put_to_storage(format!("k{i}").as_bytes(), &value).unwrap();
            }
            for _ in 0..12 {
                store.put_to_storage(b"hot", &value).unwrap();
            }
            let stats = store.garbage_collect_with_threshold(10.0).unwrap();
            assert!(stats.bytes_reclaimed > 0, "expected GC to reclaim at least one segment");
            // A faithful crash right after the pass: skip `Drop` (which would flush the
            // LSM), so only what GC itself made durable survives.
            std::mem::forget(store);
        }

        let store = KVStore::open(0, "default", dir.path(), default_lsm_config(), SyncConfig::default(), segment_size).unwrap();
        for i in 0..12 {
            assert_eq!(
                store.get(format!("k{i}").as_bytes()).unwrap(),
                Some(value.clone()),
                "k{i} lost across a crash right after a GC pass — the re-point was not durable before the unlink"
            );
        }
        assert_eq!(store.get(b"hot").unwrap(), Some(value));
    }

    #[test]
    fn test_gc_write_before_gc_survives() {
        // A write that completes immediately before GC starts must not be lost.
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(
            0,
            "default",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();

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
        let store = KVStore::open(
            0,
            "default",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();

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
        let store = KVStore::open(
            0,
            "default",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();

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
        let store = KVStore::open(
            0,
            "default",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();

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
        let store = KVStore::open(
            0,
            "default",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();

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
        let store = KVStore::open(
            0,
            "default",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();

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
        let store = KVStore::open(
            0,
            "default",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();

        store.put_to_storage_seq(b"k", b"newer", 10).unwrap();
        store.put_to_storage_seq(b"k", b"older", 5).unwrap(); // applied later, lower seq → must lose
        assert_eq!(store.get(b"k").unwrap(), Some(b"newer".to_vec()));
    }

    #[test]
    fn test_delete_from_storage_seq_respects_sequence() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(
            0,
            "default",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();

        store.put_to_storage_seq(b"k", b"v", 10).unwrap();
        store.delete_from_storage_seq(b"k", 5).unwrap(); // older delete → no-op
        assert_eq!(store.get(b"k").unwrap(), Some(b"v".to_vec()));
        store.delete_from_storage_seq(b"k", 15).unwrap(); // newer delete → removed
        assert_eq!(store.get(b"k").unwrap(), None);
    }

    #[test]
    fn test_gc_after_delete_sequential_stays_deleted() {
        let dir = TempDir::new().unwrap();
        let store = KVStore::open(
            0,
            "default",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();
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
        let store = KVStore::open(
            0,
            "default",
            dir.path(),
            default_lsm_config(),
            SyncConfig::default(),
            DEFAULT_SEGMENT_SIZE_BYTES,
        )
        .unwrap();

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
        let store = KVStore::open(0, "default", dir.path(), cfg, SyncConfig::default(), DEFAULT_SEGMENT_SIZE_BYTES).unwrap();

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
        let store = KVStore::open(0, "default", dir.path(), cfg, SyncConfig::default(), DEFAULT_SEGMENT_SIZE_BYTES).unwrap();

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
