//! KVStore - Per-namespace key-value storage
//!
//! Each KVStore owns its own LSM tree and sharded value log,
//! representing an independent namespace within the database.
//! The WAL is shared across all namespaces and managed by the
//! Database coordinator.

mod gc;
pub mod index_set;
mod maintenance;
mod reads;
mod row_ids;
mod writes;

pub use reads::PrefixScan;

use crate::db::stats::{GCStats, Stats};
use crate::db::workers::index_checkpoint::IndexCheckpointTrigger;
use crate::db::workers::lsm_compaction::LsmCompactionCommand;
use crate::error::{KVError, Result};
use crate::storage::lsm::tree::{LSMConfig, LSMTree, LsmFlushObserver, PrefixScanKeys};
use crate::storage::value_log::sharded::{ShardedValueLog, ShardedValuePointer};
use crate::storage::value_log::{TAIL_GC_MIN_GARBAGE_PCT, ValueLocation, ValueRecordMeta};

use log::{debug, error, info, warn};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::support::metrics::Metrics;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::db::config::SyncConfig;
use crate::db::kv_store::index_set::{NamespaceIndexSet, RowIdFn, RowToKeyFn};
use std::time::Duration;

/// A resolved key/value pair: both the key and the value bytes read from the value log.
pub(crate) type KeyValue = (Vec<u8>, Vec<u8>);

/// A key paired with its encoded value-log pointer and the full-width LSM write `seq`.
pub(crate) type KeyPointer = (Vec<u8>, u128, u64);

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

pub(crate) fn current_epoch_millis() -> u64 {
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

    // Set by every no-WAL write, cleared when this namespace's memtable is
    // flushed. No-WAL writes have no durable copy anywhere until that flush, so
    // this is what tells the background flusher which namespaces are holding
    // data a crash would destroy outright rather than merely delay.
    pub(crate) unflushed_no_wal_writes: Arc<AtomicBool>,

    // Test-only: set by a simulated crash (`AsyncDb::crash`) so that dropping
    // the store skips the flush below, as a process kill would. Without it the
    // flush made every unflushed memtable durable, no-WAL writes included, and
    // no crash test could see a write that a real crash loses.
    #[cfg(test)]
    crashed: AtomicBool,

    // Flag to prevent concurrent value log GC operations
    pub(crate) value_log_gc_in_progress: Arc<AtomicBool>,

    // LSM compaction trigger channel
    pub(crate) lsm_compaction_trigger: Arc<RwLock<Option<tokio::sync::mpsc::UnboundedSender<LsmCompactionCommand>>>>,

    // Index-checkpoint trigger: requests an early checkpoint when a field index
    // accumulates too much reclaimable dead blob space, or when field-index write
    // buffers pass the overlay budget's soft limit. `None` until
    // the checkpoint worker is enabled (set via `set_index_checkpoint_trigger`).
    pub(crate) index_checkpoint_trigger: Arc<RwLock<Option<Arc<IndexCheckpointTrigger>>>>,
    // Where rejected field-index updates are reported so they become durable gap
    // records instead of a log line (set via `set_index_gap_sink`).
    pub(crate) index_gap_sink: Arc<RwLock<Option<Arc<dyn crate::db::indexing::manager::IndexGapSink>>>>,

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
    // activated; `None` until then.  See `crate::field_index::RowMap`.
    pub(crate) rowmap: Arc<RwLock<Option<crate::field_index::RowMap>>>,

    // Global write-sequence counter used for highest-sequence-wins conflict
    // resolution in the memtable. In production this is shared with the WAL
    // (`Database::next_seq`) via `set_seq_counter`, so a value's sequence is the
    // same whether it was assigned at WAL-append time (`put_to_storage_seq`) or
    // allocated here for a non-WAL write (TTL/bulk). A standalone KVStore (tests)
    // gets its own fresh counter, which is internally consistent.
    pub(crate) seq_counter: RwLock<Arc<AtomicU64>>,

    /// Engine-wide operational counters, shared from `Database` via
    /// [`set_metrics`](Self::set_metrics). `None` for a standalone store (tests).
    pub(crate) metrics: OnceLock<Arc<crate::support::metrics::Metrics>>,

    /// Test-only crash point: when set, a GC pass returns the moment its segments are
    /// unlinked, skipping everything after. It exists to prove the *ordering* of
    /// `flush → unlink` is what makes GC crash-safe: with the flush first (correct), a
    /// crash here is harmless; if the two were ever swapped, this window would lose the
    /// re-point along with the segments it referred to, and the guard test would fail.
    #[cfg(test)]
    pub(crate) gc_crash_after_unlink: AtomicBool,
}

impl KVStore {
    /// Whether the namespace whose data directory is `base_path` holds any L1
    /// data, without opening it (see `LSMTree::has_level1_data`).
    pub(crate) fn has_level1_data(base_path: &Path) -> std::io::Result<bool> {
        crate::storage::lsm::tree::LSMTree::has_level1_data(&base_path.join("lsm"))
    }

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
            unflushed_no_wal_writes: Arc::new(AtomicBool::new(false)),
            #[cfg(test)]
            crashed: AtomicBool::new(false),
            value_log_gc_in_progress: Arc::new(AtomicBool::new(false)),
            lsm_compaction_trigger: Arc::new(RwLock::new(None)),
            index_checkpoint_trigger: Arc::new(RwLock::new(None)),
            index_gap_sink: Arc::new(RwLock::new(None)),
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
    pub(crate) fn set_metrics(&self, metrics: Arc<crate::support::metrics::Metrics>) {
        self.lsm.set_metrics(metrics.clone());
        let _ = self.metrics.set(metrics);
    }

    /// The operational counters, if attached. `None` for a standalone store.
    #[inline]
    pub(crate) fn metrics(&self) -> Option<&crate::support::metrics::Metrics> {
        self.metrics.get().map(|m| m.as_ref())
    }

    /// Share a global write-sequence counter with this store (called by the
    /// `Database` coordinator with its WAL sequence counter, so all writes —
    /// WAL-backed and non-WAL — draw from one monotonic sequence space).
    /// Share the database's write-sequence counter, first raising it above
    /// every sequence this store already holds.
    ///
    /// **Load-bearing for no-WAL writes.** The counter is recovered from the WAL
    /// at open, but a no-WAL write takes a sequence the WAL never sees. After a
    /// crash the counter could restart below a no-WAL value already flushed to an
    /// SSTable, and every later write to that key — WAL-backed or not — would
    /// lose to the stale value under highest-sequence-wins: a delete silently
    /// undone, an update silently reverted. Found by the M3a split crash audit
    /// while trying no-WAL meta writes (a meta rewritten after a restart never
    /// became visible). Regression
    /// test: `a_write_after_a_crash_beats_a_flushed_no_wal_value`.
    pub(crate) fn set_seq_counter(&self, counter: Arc<AtomicU64>) {
        counter.fetch_max(self.lsm.max_stored_seq().saturating_add(1), Ordering::SeqCst);
        *self.seq_counter.write() = counter;
    }

    /// Allocate the next write sequence at full `u64` width.
    ///
    /// Must not be narrowed: the LSM compares a fresh write's sequence against
    /// entries that can be arbitrarily old (an L1 entry keeps its sequence for
    /// the lifetime of the database), so the width has to span the whole
    /// sequence space, not just a recent window. See [`crate::storage::lsm::tree`]'s
    /// `seq_newer`.
    pub(crate) fn alloc_seq(&self) -> u64 {
        self.seq_counter.read().fetch_add(1, Ordering::Relaxed)
    }

    /// Set the LSM flush observer (used to wire up WAL persistence callbacks)
    pub(crate) fn set_flush_observer(&self, observer: Option<Arc<dyn LsmFlushObserver>>) {
        let observer = observer.map(|inner| {
            Arc::new(SyncValueLogBeforePersist {
                inner,
                logs: self.value_log.bucket_logs(),
            }) as Arc<dyn LsmFlushObserver>
        });
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

    /// Wire where rejected field-index updates are reported. See
    /// [`IndexGapSink`](crate::db::indexing::manager::IndexGapSink).
    pub(crate) fn set_index_gap_sink(&self, sink: Option<Arc<dyn crate::db::indexing::manager::IndexGapSink>>) {
        *self.index_gap_sink.write() = sink;
    }

    /// Flush and sync all data for this namespace.
    pub fn shutdown(&self) -> Result<()> {
        self.lsm.cleanup_pending_memtables_on_close();
        // Values before the LSM entries that point at them (see
        // `SyncValueLogBeforePersist`).
        self.value_log.sync_all()?;
        self.lsm.flush_and_compact_all()?;
        // Persist each bucket's segment inventory, its live/garbage counters and its
        // segment-id high-water mark. The high-water mark is what keeps ids unique
        // across restarts; the counters keep GC from under-triggering after a reopen.
        self.value_log.flush_all_metadata()?;
        Ok(())
    }
}

/// Makes a store's value log durable before any WAL entry is retired.
///
/// Writes fsync the WAL but not the value log (that is `records_per_sync`'s
/// cadence), which is safe only while the WAL still holds the value. When a
/// memtable's L0 files are written, `on_ro_memtable_written` lets both WALs
/// retire its entries — recovery then skips them and WAL GC deletes them — so
/// the value-log records those SSTable entries point at must be on stable
/// storage first. Without this, a power loss after the flush lost up to
/// `records_per_sync` acknowledged values (and any segment sealed since its
/// last fsync) with nothing left to replay them from. If the fsync fails the
/// flush is undone: the entries stay replayable and the memtable is flushed
/// again later.
struct SyncValueLogBeforePersist {
    inner: Arc<dyn LsmFlushObserver>,
    logs: Vec<Arc<crate::storage::value_log::ValueLog>>,
}

impl LsmFlushObserver for SyncValueLogBeforePersist {
    fn on_memtable_sealed(&self, version: u64) {
        self.inner.on_memtable_sealed(version);
    }

    fn on_ro_memtable_flushing(&self, version: u64) -> std::io::Result<()> {
        self.inner.on_ro_memtable_flushing(version)
    }

    // Both WALs retire records here, so the values they point at must be on
    // stable storage first. A failed fsync vetoes the flush: the files are not
    // registered and the memtable is flushed again later.
    fn on_ro_memtable_written(&self, version: u64) -> std::io::Result<()> {
        for log in &self.logs {
            if let Err(e) = log.sync() {
                error!("value-log fsync failed before memtable {version}'s L0 files were registered; the flush is undone: {e:?}");
                return Err(std::io::Error::other(format!("value-log fsync failed: {e}")));
            }
        }
        self.inner.on_ro_memtable_written(version)
    }

    fn on_ro_memtable_flushed_to_level0(&self, version: u64) {
        self.inner.on_ro_memtable_flushed_to_level0(version);
    }
}

impl KVStore {
    /// Test-only: make dropping this store skip its flush, as a crash would.
    #[cfg(test)]
    pub(crate) fn mark_crashed(&self) {
        self.crashed.store(true, Ordering::SeqCst);
    }

    /// Test-only: simulate a crash. Drops the store without its flush, and closes
    /// every file. Use this, not `std::mem::forget`, which leaks them.
    #[cfg(test)]
    pub(crate) fn crash(self) {
        self.mark_crashed();
    }
}

impl Drop for KVStore {
    fn drop(&mut self) {
        #[cfg(test)]
        if self.crashed.load(Ordering::SeqCst) {
            return;
        }
        let _ = self.value_log.sync_all();
        let _ = self.lsm.flush_and_compact_all();
        let _ = self.value_log.flush_all_metadata();
    }
}

#[cfg(test)]
mod tests;
