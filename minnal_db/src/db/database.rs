//! Database - Multi-namespace coordinator
//!
//! Manages a shared WAL, namespace registry, and multiple KVStore instances.
//! Each namespace has independent LSM + value log storage.

use crate::db::config::DbConfig;
use crate::db::error::{KVError, Result};
use crate::db::index_checkpoint_worker::{DEFAULT_CHECKPOINT_INTERVAL, IndexCheckpointTarget, IndexCheckpointWorker};
use crate::db::index_manager::IndexManager;
use crate::db::kv_store::KVStore;
use crate::db::namespace::{DEFAULT_NAMESPACE_ID, FieldId, FieldMeta, NamespaceRegistry};
use crate::db::namespace_index::{ExtractorFn, IndexEntry};
use crate::db::stats::{GCStats, Stats};
use crate::db::ttl_worker::{TtlTarget, TtlWorker};
use crate::db::wal::{Wal, WalEntry, WalEntryStatus, WalError, WalMetadata, WalOperationType};
use crate::db::wal_worker::{WalGcTarget, WalGcWorker};
use crate::store::gc_value_log_worker::{GCWorker, ValueLogGcTarget};
use crate::store::lsm::lsm_tree::LsmFlushObserver;
use crate::store::lsm_worker::{LsmCompactionCommand, LsmCompactionTarget, LsmCompactionWorker};
use crate::store::value_log::ValueLogMetadata;
use index::{DynFieldIndex, IndexValueType};

use log::{debug, error, info, warn};
use parking_lot::RwLock;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

// ── WAL persistence observer ───────────────────────────────────────────

struct WalFlushState {
    tail: u64,
    flushed: bool,
}

/// Observes LSM flush events and marks WAL entries as persisted.
/// Shared across all namespaces since the WAL is global.
pub(crate) struct WalPersistObserver {
    wal: Arc<Wal>,
    wal_metadata: Arc<RwLock<WalMetadata>>,
    wal_metadata_path: PathBuf,
    pending: Arc<RwLock<BTreeMap<u64, WalFlushState>>>,
    last_persisted_offset: Arc<RwLock<u64>>,
}

impl WalPersistObserver {
    fn new(
        wal: Arc<Wal>,
        wal_metadata: Arc<RwLock<WalMetadata>>,
        wal_metadata_path: PathBuf,
        pending: Arc<RwLock<BTreeMap<u64, WalFlushState>>>,
        last_persisted_offset: Arc<RwLock<u64>>,
    ) -> Self {
        Self {
            wal,
            wal_metadata,
            wal_metadata_path,
            pending,
            last_persisted_offset,
        }
    }

    pub(crate) fn mark_persisted_range(&self, start: u64, end: u64) {
        if end <= start {
            return;
        }

        let entries = match self.wal.scan_entries(start, end) {
            Ok(entries) => entries,
            Err(WalError::Io(ref e)) if e.kind() == std::io::ErrorKind::NotFound => return,
            Err(e) => {
                warn!("[WAL] Failed to scan entries for persist update: {:?}", e);
                return;
            }
        };

        let mut updated = 0u64;
        let mut per_segment: BTreeMap<u64, u64> = BTreeMap::new();
        let mut had_error = false;

        for (pointer, entry) in entries {
            if entry.status == WalEntryStatus::Persisted {
                continue;
            }
            if let Err(e) = self.wal.update_entry_status(pointer.offset, WalEntryStatus::Persisted) {
                warn!("[WAL] Failed to update entry status at offset {}: {:?}", pointer.offset, e);
                had_error = true;
                continue;
            }
            updated += 1;
            let segment_id = self.wal.segment_id_for_offset(pointer.offset);
            *per_segment.entry(segment_id).or_insert(0) += 1;
        }

        if updated > 0 {
            let mut wal_metadata = self.wal_metadata.write();
            for (segment_id, count) in per_segment {
                wal_metadata.add_segment_persisted(segment_id, count);
            }
            wal_metadata.persisted_entries = wal_metadata.persisted_entries.saturating_add(updated);
            let bytes = match wal_metadata.to_file_bytes() {
                Ok(bytes) => bytes,
                Err(e) => {
                    warn!("[WAL] Failed to serialize WAL metadata: {:?}", e);
                    had_error = true;
                    Vec::new()
                }
            };
            if !bytes.is_empty()
                && let Err(e) = crate::support::write_atomic_durable(&self.wal_metadata_path, &bytes)
            {
                warn!("[WAL] Failed to write WAL metadata: {:?}", e);
                had_error = true;
            }
        }

        if !had_error {
            *self.last_persisted_offset.write() = end;
        }
    }

    /// Mark WAL entries belonging to `namespace_id` in `[start, end)` as persisted.
    ///
    /// Unlike [`mark_persisted_range`], this skips entries from other namespaces and
    /// therefore does **not** advance `last_persisted_offset`. It is called when a
    /// namespace is dropped: its KVStore has been flushed and shut down, so those
    /// entries are durable and no longer need WAL protection, but other namespaces
    /// may still have un-persisted entries in the same offset window.
    pub(crate) fn mark_namespace_persisted(&self, namespace_id: u32, start: u64, end: u64) {
        if end <= start {
            return;
        }

        let entries = match self.wal.scan_entries(start, end) {
            Ok(entries) => entries,
            Err(WalError::Io(ref e)) if e.kind() == std::io::ErrorKind::NotFound => return,
            Err(e) => {
                warn!("[WAL] Failed to scan entries for namespace {} persist update: {:?}", namespace_id, e);
                return;
            }
        };

        let mut updated = 0u64;
        let mut per_segment: BTreeMap<u64, u64> = BTreeMap::new();

        for (pointer, entry) in entries {
            if entry.namespace_id != namespace_id {
                continue;
            }
            if entry.status == WalEntryStatus::Persisted {
                continue;
            }
            if let Err(e) = self.wal.update_entry_status(pointer.offset, WalEntryStatus::Persisted) {
                warn!("[WAL] Failed to update entry status at offset {}: {:?}", pointer.offset, e);
                continue;
            }
            updated += 1;
            let segment_id = self.wal.segment_id_for_offset(pointer.offset);
            *per_segment.entry(segment_id).or_insert(0) += 1;
        }

        if updated > 0 {
            let mut wal_metadata = self.wal_metadata.write();
            for (segment_id, count) in per_segment {
                wal_metadata.add_segment_persisted(segment_id, count);
            }
            wal_metadata.persisted_entries = wal_metadata.persisted_entries.saturating_add(updated);
            match wal_metadata.to_file_bytes() {
                Ok(bytes) if !bytes.is_empty() => {
                    if let Err(e) = crate::support::write_atomic_durable(&self.wal_metadata_path, &bytes) {
                        warn!("[WAL] Failed to write WAL metadata after namespace drop: {:?}", e);
                    }
                }
                Err(e) => warn!("[WAL] Failed to serialize WAL metadata after namespace drop: {:?}", e),
                _ => {}
            }
        }
    }

    fn try_advance_persisted(&self) {
        loop {
            let next = {
                let pending = self.pending.read();
                let Some((&version, state)) = pending.iter().next() else {
                    return;
                };
                if !state.flushed {
                    return;
                }
                (version, state.tail)
            };

            {
                let mut pending = self.pending.write();
                let Some(state) = pending.get(&next.0) else {
                    continue;
                };
                if !state.flushed {
                    return;
                }
                pending.remove(&next.0);
            }

            let start = *self.last_persisted_offset.read();
            self.mark_persisted_range(start, next.1);
        }
    }
}

impl LsmFlushObserver for WalPersistObserver {
    fn on_memtable_sealed(&self, version: u64) {
        let tail = self.wal_metadata.read().tail;
        let mut pending = self.pending.write();
        pending.entry(version).or_insert(WalFlushState { tail, flushed: false });
    }

    fn on_ro_memtable_flushed_to_level0(&self, version: u64) {
        {
            let mut pending = self.pending.write();
            let entry = pending.entry(version).or_insert(WalFlushState { tail: 0, flushed: false });
            if entry.tail == 0 {
                entry.tail = self.wal_metadata.read().tail;
            }
            entry.flushed = true;
        }
        self.try_advance_persisted();
    }
}

/// Hub that fans out LSM flush events to the WAL observer and compaction trigger.
struct LsmFlushObserverHub {
    wal_observer: Arc<WalPersistObserver>,
    compaction_trigger: Arc<RwLock<Option<tokio::sync::mpsc::UnboundedSender<LsmCompactionCommand>>>>,
}

impl LsmFlushObserverHub {
    fn new(
        wal_observer: Arc<WalPersistObserver>,
        compaction_trigger: Arc<RwLock<Option<tokio::sync::mpsc::UnboundedSender<LsmCompactionCommand>>>>,
    ) -> Self {
        Self {
            wal_observer,
            compaction_trigger,
        }
    }

    fn trigger_compaction(&self) {
        if let Some(sender) = self.compaction_trigger.read().as_ref() {
            let _ = sender.send(LsmCompactionCommand::Trigger);
        }
    }
}

impl LsmFlushObserver for LsmFlushObserverHub {
    fn on_memtable_sealed(&self, version: u64) {
        self.wal_observer.on_memtable_sealed(version);
        self.trigger_compaction();
    }

    fn on_ro_memtable_flushed_to_level0(&self, version: u64) {
        self.wal_observer.on_ro_memtable_flushed_to_level0(version);
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

/// Format a key for log output: UTF-8 string if valid, otherwise `hex:<hex>`.
fn display_key(key: &[u8]) -> String {
    match std::str::from_utf8(key) {
        Ok(s) => s.to_owned(),
        Err(_) => format!("hex:{}", key.iter().map(|b| format!("{:02x}", b)).collect::<String>()),
    }
}

// ── Database coordinator ───────────────────────────────────────────────

/// The main database coordinator that manages:
/// - A shared WAL across all namespaces
/// - A namespace registry (name → id mapping)
/// - A collection of KVStore instances (one per namespace)
pub struct Database {
    db_path: PathBuf,
    pub(crate) config: DbConfig,

    // Shared WAL
    wal: Arc<Wal>,
    #[allow(dead_code)]
    wal_path: PathBuf,
    wal_metadata_path: PathBuf,
    wal_metadata: Arc<RwLock<WalMetadata>>,
    wal_flush_observer: Arc<WalPersistObserver>,
    #[allow(dead_code)]
    pending_wal_flushes: Arc<RwLock<BTreeMap<u64, WalFlushState>>>,
    last_persisted_wal_offset: Arc<RwLock<u64>>,
    wal_gc_in_progress: Arc<AtomicBool>,

    // Namespace registry
    pub(crate) registry: RwLock<NamespaceRegistry>,

    // Per-namespace stores: namespace_id -> KVStore
    pub(crate) stores: RwLock<HashMap<u32, Arc<KVStore>>>,

    // Database-level flags
    closed: Arc<AtomicBool>,

    // Background workers (database-level)
    pub(crate) wal_gc_worker: Arc<tokio::sync::RwLock<Option<Arc<WalGcWorker>>>>,
    pub(crate) lsm_compaction_worker: Arc<tokio::sync::RwLock<Option<Arc<LsmCompactionWorker>>>>,
    pub(crate) value_log_gc_worker: Arc<tokio::sync::RwLock<Option<Arc<GCWorker>>>>,

    // Shared LSM compaction sender — stored so newly-opened namespaces can be wired up.
    pub(crate) lsm_compaction_sender: Arc<parking_lot::RwLock<Option<tokio::sync::mpsc::UnboundedSender<LsmCompactionCommand>>>>,

    // Single global TTL worker — one task that scans every TTL-enabled namespace
    // on each tick (mirrors `value_log_gc_worker`). `None` until the first TTL
    // namespace is registered. The per-namespace (ttl, max_deletes) config it
    // expires is the durable `NamespaceRegistry::ttl_configs`, not a field here.
    pub(crate) ttl_worker: Arc<tokio::sync::RwLock<Option<Arc<TtlWorker>>>>,

    // Index manager: tracks field definitions and manages index directories
    pub(crate) index_manager: Arc<IndexManager>,

    // Background index checkpoint worker
    pub(crate) index_checkpoint_worker: Arc<tokio::sync::RwLock<Option<Arc<IndexCheckpointWorker>>>>,

    // Global monotonic sequence counter. Seeded from the WAL on open.
    next_seq: Arc<AtomicU64>,

    // Directory for recovery fail-log files.
    fail_log_dir: PathBuf,

    // Engine-wide operational counters, shared into every KVStore/LSMTree.
    metrics: Arc<crate::db::metrics::Metrics>,
}

impl Database {
    /// Open a multi-namespace database at the given path
    pub fn open(db_path: &Path, mut config: DbConfig) -> Result<Self> {
        std::fs::create_dir_all(db_path)?;

        // Detect existing bucket count from the default namespace's value logs
        let default_vlog_dir = db_path.join("ns_default").join("value_logs");
        if default_vlog_dir.exists() {
            let existing_count = Self::detect_bucket_count(&default_vlog_dir);
            if existing_count > 0 && existing_count != config.num_buckets {
                warn!(
                    "[WARNING] Configured num_buckets={} but existing database at '{}' uses {} buckets. \
                     The configured value will NOT be applied. To change bucket count, \
                     database migration must be done externally.",
                    config.num_buckets,
                    db_path.display(),
                    existing_count,
                );
                config.num_buckets = existing_count;
            }
        }

        // Propagate top-level config values into lsm_config
        config.lsm_config.num_buckets = config.num_buckets;
        config.lsm_config.skip_list_capacity = config.skip_list_capacity;

        let wal_path = db_path.join("wal.log");
        let wal_metadata_path = db_path.join("wal_metadata");

        // Open WAL
        let wal = Arc::new(Wal::open(&wal_path)?);

        // Load or initialize WAL metadata
        let mut wal_metadata = if wal_metadata_path.exists() {
            let data = std::fs::read(&wal_metadata_path)?;
            match WalMetadata::from_file_bytes(&data) {
                Ok(m) => m,
                Err(_) => {
                    let backup = wal_metadata_path.with_extension("corrupt");
                    let _ = std::fs::rename(&wal_metadata_path, &backup);
                    WalMetadata::new()
                }
            }
        } else {
            WalMetadata::new()
        };
        wal_metadata.reconcile_segment_lengths();

        // WAL entries are fsynced on every write, but `wal_metadata` (which holds
        // the tail) is only flushed periodically — so after a crash the persisted
        // tail can lag the durable end of the log, and recovery scanning only up
        // to the stale tail would silently drop fsynced entries. Reconstruct the
        // true tail from the self-describing WAL and fold the durable-but-
        // unaccounted entries into the counters so recovery actually replays them
        // (and is not short-circuited by the stale total/persisted counts). With
        // a lost/corrupt metadata file (tail = 0) this rebuilds the tail wholesale.
        {
            let persisted_tail = wal_metadata.tail;
            let true_tail = wal.recover_tail(persisted_tail);
            if true_tail > persisted_tail {
                let extra = wal.scan_entries(persisted_tail, true_tail).unwrap_or_default();
                warn!(
                    "[RECOVERY] WAL metadata tail ({}) lagged the durable log end ({}); \
                     recovering {} entry(ies) appended since the last metadata flush",
                    persisted_tail,
                    true_tail,
                    extra.len()
                );
                for (pointer, _) in &extra {
                    let segment_id = wal.segment_id_for_offset(pointer.offset);
                    wal_metadata.add_segment_total(segment_id, 1);
                }
                wal_metadata.total_entries = wal_metadata.total_entries.saturating_add(extra.len() as u64);
                wal_metadata.tail = true_tail;
            }
        }

        let next_seq_start = wal.recover_sequence(wal_metadata.head, wal_metadata.tail, wal_metadata.last_sequence);

        let wal_metadata = Arc::new(RwLock::new(wal_metadata));
        let pending_wal_flushes = Arc::new(RwLock::new(BTreeMap::new()));
        let last_persisted_wal_offset = Arc::new(RwLock::new(0u64));
        let wal_flush_observer = Arc::new(WalPersistObserver::new(
            Arc::clone(&wal),
            Arc::clone(&wal_metadata),
            wal_metadata_path.clone(),
            Arc::clone(&pending_wal_flushes),
            Arc::clone(&last_persisted_wal_offset),
        ));

        // Open namespace registry
        let registry = NamespaceRegistry::open(db_path)?;

        // Open index manager (creates {db_path}/index/ if needed)
        let index_manager = IndexManager::open(db_path)?;

        // Open KVStores for all registered namespaces, restoring each one's
        // persisted TTL (if any) so `store.ttl` reflects the durable config.
        let mut stores = HashMap::new();
        for (name, ns_id) in registry.list() {
            let ns_path = db_path.join(format!("ns_{}", name));
            let ttl = registry.ttl_config(ns_id).map(|(ttl, _)| ttl);
            let kv_store = KVStore::open_with_ttl(ns_id, name, &ns_path, config.lsm_config.clone(), config.sync_config, ttl)?;
            kv_store.set_verify_checksums_on_read(config.verify_checksums_on_read);
            kv_store.cleanup_old_files_on_startup()?;
            stores.insert(ns_id, Arc::new(kv_store));
        }

        // Cleanup old WAL file
        let old_wal = db_path.join("wal.log.old");
        if old_wal.exists() {
            match std::fs::remove_file(&old_wal) {
                Ok(_) => info!("[STARTUP] Cleaned up old WAL file"),
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                    warn!("[STARTUP] Failed to cleanup old WAL file: {:?}", e);
                }
                _ => {}
            }
        }

        let fail_log_dir = config.fail_log_dir.clone().unwrap_or_else(|| db_path.join("fail_logs"));

        let db = Self {
            db_path: db_path.to_path_buf(),
            config,
            wal,
            wal_path,
            wal_metadata_path,
            wal_metadata,
            wal_flush_observer,
            pending_wal_flushes,
            last_persisted_wal_offset,
            wal_gc_in_progress: Arc::new(AtomicBool::new(false)),
            registry: RwLock::new(registry),
            stores: RwLock::new(stores),
            closed: Arc::new(AtomicBool::new(false)),
            wal_gc_worker: Arc::new(tokio::sync::RwLock::new(None)),
            lsm_compaction_worker: Arc::new(tokio::sync::RwLock::new(None)),
            value_log_gc_worker: Arc::new(tokio::sync::RwLock::new(None)),
            lsm_compaction_sender: Arc::new(parking_lot::RwLock::new(None)),
            ttl_worker: Arc::new(tokio::sync::RwLock::new(None)),
            index_manager,
            index_checkpoint_worker: Arc::new(tokio::sync::RwLock::new(None)),
            next_seq: Arc::new(AtomicU64::new(next_seq_start)),
            fail_log_dir,
            metrics: Arc::new(crate::db::metrics::Metrics::default()),
        };

        // Share the global WAL sequence counter with every store so that all
        // writes — WAL-backed, recovery replay, TTL expiry and bulk — draw from
        // one monotonic sequence space (required for highest-sequence-wins
        // conflict resolution in the memtable). Must happen before recovery.
        // Share the operational counters at the same time.
        {
            let stores = db.stores.read();
            for kv_store in stores.values() {
                kv_store.set_seq_counter(db.next_seq.clone());
                kv_store.set_metrics(db.metrics.clone());
            }
        }

        // Recover from WAL
        db.recover_from_wal()?;

        // Rebuild WAL persisted state and wire up observers
        let last_persisted = db.rebuild_wal_persisted_state()?;
        *db.last_persisted_wal_offset.write() = last_persisted;
        db.wire_up_flush_observers();

        Ok(db)
    }

    /// Detect existing bucket count by counting value_log_*.log files.
    fn detect_bucket_count(vlog_dir: &Path) -> usize {
        let Ok(entries) = std::fs::read_dir(vlog_dir) else {
            return 0;
        };
        entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                name.starts_with("value_log_") && name.ends_with(".log") && !name.ends_with(".old")
            })
            .count()
    }

    /// Wire up LSM flush observers for all KVStores
    fn wire_up_flush_observers(&self) {
        let stores = self.stores.read();
        for (_, kv_store) in stores.iter() {
            let observer: Arc<dyn LsmFlushObserver> = Arc::new(LsmFlushObserverHub::new(
                Arc::clone(&self.wal_flush_observer),
                Arc::clone(&kv_store.lsm_compaction_trigger),
            ));
            kv_store.set_flush_observer(Some(observer));
        }
    }

    // ── Core data operations (default namespace shortcuts) ─────────────

    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.put_ns(DEFAULT_NAMESPACE_ID, key, value)
    }

    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.get_ns(DEFAULT_NAMESPACE_ID, key)
    }

    pub fn delete(&self, key: &[u8]) -> Result<()> {
        self.delete_ns(DEFAULT_NAMESPACE_ID, key)
    }

    // ── Namespace-aware data operations ────────────────────────────────

    /// Number of attempts to apply a WAL-durable write to the in-memory store
    /// before giving up. Once the WAL fsync has succeeded the write is durable
    /// and *will* take effect (via the in-memory apply now, or via WAL replay on
    /// the next open), so this only bounds the retry of transient apply errors.
    const APPLY_RETRY_ATTEMPTS: usize = 3;

    /// Apply a WAL-durable mutation to the in-memory store, retrying transient
    /// failures up to [`APPLY_RETRY_ATTEMPTS`] times.
    ///
    /// The caller has already durably committed the corresponding WAL entry, so
    /// a persistent failure here is **not** fatal: it is logged at ERROR and the
    /// entry is replayed from the WAL on the next open. We therefore never
    /// surface the apply error to the caller — the write is durable regardless.
    /// Apply the in-memory write with bounded retry. Returns `true` if it
    /// succeeded, `false` if it failed after all attempts (the write is still
    /// durable in the WAL and will be replayed on the next open).
    fn apply_with_retry<F>(op_name: &str, namespace_id: u32, key: &[u8], seq: u64, mut apply: F) -> bool
    where
        F: FnMut() -> Result<()>,
    {
        for attempt in 1..=Self::APPLY_RETRY_ATTEMPTS {
            match apply() {
                Ok(()) => return true,
                Err(e) if attempt < Self::APPLY_RETRY_ATTEMPTS => {
                    warn!(
                        "[WAL-COMMITTED seq={}] in-memory apply of '{}' key='{}' (ns={}) failed on attempt {}/{}: {}. Retrying.",
                        seq,
                        op_name,
                        display_key(key),
                        namespace_id,
                        attempt,
                        Self::APPLY_RETRY_ATTEMPTS,
                        e
                    );
                }
                Err(e) => {
                    error!(
                        "[WAL-COMMITTED seq={}] in-memory apply of '{}' key='{}' (ns={}) failed after {} attempts: {}. \
                         Write IS durable in the WAL and will be replayed on the next open.",
                        seq,
                        op_name,
                        display_key(key),
                        namespace_id,
                        Self::APPLY_RETRY_ATTEMPTS,
                        e
                    );
                }
            }
        }
        false
    }

    pub fn put_ns(&self, namespace_id: u32, key: &[u8], value: &[u8]) -> Result<()> {
        self.check_closed()?;

        // Step 1: Write to shared WAL.
        // Allocate the sequence number *inside* the WAL lock so the global
        // sequence order is identical to the WAL physical (append) order. If the
        // seq were allocated before taking the lock, two concurrent writers could
        // acquire seqs in one order but append in the other, leaving the on-disk
        // WAL out of sequence order — which recovery relies on to replay in the
        // original order.
        let mut wal_metadata = self.wal_metadata.write();
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        assert!(seq != u64::MAX, "WAL global sequence number exhausted");
        let wal_entry = WalEntry::new_upsert_ns(namespace_id, key.to_vec(), value.to_vec()).with_sequence(seq);
        let wal_pointer = self.wal.append_entry(&wal_entry, &mut wal_metadata.tail, true)?;
        let segment_id = self.wal.segment_id_for_offset(wal_pointer.offset);
        wal_metadata.add_segment_total(segment_id, 1);
        wal_metadata.total_entries += 1;
        drop(wal_metadata);

        crate::db::metrics::Metrics::bump(&self.metrics.puts);
        crate::db::metrics::Metrics::bump(&self.metrics.wal_fsyncs);
        crate::db::metrics::Metrics::add(&self.metrics.wal_bytes_appended, wal_pointer.size as u64 + 4);

        // Step 2: Apply to the namespace's in-memory store. The write is already
        // durable in the WAL, so this is best-effort with bounded retry: on
        // persistent failure we log and still return Ok, because recovery will
        // replay the entry on the next open. Surfacing an error here would be
        // misleading — the data is already committed.
        let kv_store = self.get_store(namespace_id)?;
        if !Self::apply_with_retry("put", namespace_id, key, seq, || kv_store.put_to_storage_seq(key, value, seq)) {
            crate::db::metrics::Metrics::bump(&self.metrics.apply_failures);
        }

        // Step 3: Maybe sync
        if kv_store.should_sync() && kv_store.sync_value_log().is_ok() {
            let start = *self.last_persisted_wal_offset.read();
            let tail = self.wal_metadata.read().tail;
            self.wal_flush_observer.mark_persisted_range(start, tail);
        }

        Ok(())
    }

    /// Write a key-value pair **without** appending to the WAL.
    ///
    /// This skips the fsync that normally accompanies every WAL entry, giving
    /// much higher throughput at the cost of crash safety: any data written
    /// via this method is unrecoverable if the process crashes before the
    /// value-log page is flushed.  Only use this for bulk-loading scenarios
    /// where re-running the load is acceptable.
    pub fn put_ns_no_wal(&self, namespace_id: u32, key: &[u8], value: &[u8]) -> Result<()> {
        self.check_closed()?;
        crate::db::metrics::Metrics::bump(&self.metrics.no_wal_puts);
        let kv_store = self.get_store(namespace_id)?;
        kv_store.put_to_storage(key, value)
    }

    pub fn get_ns(&self, namespace_id: u32, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.check_closed()?;
        let kv_store = self.get_store(namespace_id)?;
        kv_store.get(key)
    }

    pub fn delete_ns(&self, namespace_id: u32, key: &[u8]) -> Result<()> {
        self.check_closed()?;

        // Step 1: Write DELETE to shared WAL.
        // Allocate the sequence inside the WAL lock so sequence order == WAL
        // append order (see `put_ns` for the rationale).
        let mut wal_metadata = self.wal_metadata.write();
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        assert!(seq != u64::MAX, "WAL global sequence number exhausted");
        let wal_entry = WalEntry::new_delete_ns(namespace_id, key.to_vec()).with_sequence(seq);
        let wal_pointer = self.wal.append_entry(&wal_entry, &mut wal_metadata.tail, true)?;
        let segment_id = self.wal.segment_id_for_offset(wal_pointer.offset);
        wal_metadata.add_segment_total(segment_id, 1);
        wal_metadata.total_entries += 1;
        drop(wal_metadata);

        crate::db::metrics::Metrics::bump(&self.metrics.deletes);
        crate::db::metrics::Metrics::bump(&self.metrics.wal_fsyncs);
        crate::db::metrics::Metrics::add(&self.metrics.wal_bytes_appended, wal_pointer.size as u64 + 4);

        // Step 2: Apply the delete to the in-memory store (best-effort with
        // bounded retry; durable in the WAL — see `put_ns`).
        let kv_store = self.get_store(namespace_id)?;
        if !Self::apply_with_retry("delete", namespace_id, key, seq, || kv_store.delete_from_storage_seq(key, seq)) {
            crate::db::metrics::Metrics::bump(&self.metrics.apply_failures);
        }

        // Step 3: Maybe sync
        if kv_store.should_sync() && kv_store.sync_value_log().is_ok() {
            let start = *self.last_persisted_wal_offset.read();
            let tail = self.wal_metadata.read().tail;
            self.wal_flush_observer.mark_persisted_range(start, tail);
        }

        Ok(())
    }

    // ── Namespace management (admin API) ───────────────────────────────

    /// Create a new namespace. Returns the namespace ID.
    pub fn create_namespace(&self, name: &str) -> Result<u32> {
        self.check_closed()?;

        let ns_id = self.registry.write().create(name)?;

        let ns_path = self.db_path.join(format!("ns_{}", name));
        let kv_store = KVStore::open(ns_id, name, &ns_path, self.config.lsm_config.clone(), self.config.sync_config)?;
        kv_store.set_verify_checksums_on_read(self.config.verify_checksums_on_read);
        kv_store.set_seq_counter(self.next_seq.clone());
        kv_store.set_metrics(self.metrics.clone());

        // Wire up flush observer
        let observer: Arc<dyn LsmFlushObserver> = Arc::new(LsmFlushObserverHub::new(
            Arc::clone(&self.wal_flush_observer),
            Arc::clone(&kv_store.lsm_compaction_trigger),
        ));
        kv_store.set_flush_observer(Some(observer));

        // Wire compaction trigger if a worker is already running
        if let Some(sender) = self.lsm_compaction_sender.read().as_ref() {
            kv_store.set_compaction_trigger(sender.clone());
        }

        self.stores.write().insert(ns_id, Arc::new(kv_store));
        Ok(ns_id)
    }

    /// Create a new namespace with an optional TTL. Returns the namespace ID.
    /// The TTL is recorded on the store; the single global TTL worker (started by
    /// the async facade's `namespace_with_ttl`) expires records older than it.
    pub fn create_namespace_with_ttl(&self, name: &str, ttl: Option<Duration>) -> Result<u32> {
        self.check_closed()?;

        let ns_id = self.registry.write().create(name)?;

        let ns_path = self.db_path.join(format!("ns_{}", name));
        let kv_store = KVStore::open_with_ttl(ns_id, name, &ns_path, self.config.lsm_config.clone(), self.config.sync_config, ttl)?;
        kv_store.set_verify_checksums_on_read(self.config.verify_checksums_on_read);
        kv_store.set_seq_counter(self.next_seq.clone());
        kv_store.set_metrics(self.metrics.clone());

        // Wire up flush observer
        let observer: Arc<dyn LsmFlushObserver> = Arc::new(LsmFlushObserverHub::new(
            Arc::clone(&self.wal_flush_observer),
            Arc::clone(&kv_store.lsm_compaction_trigger),
        ));
        kv_store.set_flush_observer(Some(observer));

        // Wire compaction trigger if a worker is already running
        if let Some(sender) = self.lsm_compaction_sender.read().as_ref() {
            kv_store.set_compaction_trigger(sender.clone());
        }

        self.stores.write().insert(ns_id, Arc::new(kv_store));
        Ok(ns_id)
    }

    /// List all namespaces (name, id)
    pub fn list_namespaces(&self) -> Vec<(String, u32)> {
        self.registry.read().list().into_iter().map(|(name, id)| (name.to_string(), id)).collect()
    }

    /// Get a namespace ID by name
    pub fn get_namespace_id(&self, name: &str) -> Option<u32> {
        self.registry.read().get_id(name)
    }

    /// Check if a namespace exists
    pub fn namespace_exists(&self, name: &str) -> bool {
        self.registry.read().exists(name)
    }

    /// Remove a namespace, close its KVStore, and reclaim its on-disk storage.
    ///
    /// The step ordering is chosen so a crash at any point leaves a consistent
    /// state and, in particular, never confuses WAL recovery:
    ///
    /// 1. **Persist the registry deletion first.** From this point the namespace
    ///    is logically gone, and all of its on-disk files *and* WAL entries are
    ///    unreferenced. Recovery skips WAL entries whose namespace is absent from
    ///    the registry (see [`recover_from_wal`](Self::recover_from_wal)), so the
    ///    later steps can fail or only partially complete without ever
    ///    resurrecting the deleted namespace's data.
    /// 2. **Flush, shut down, and drop the KVStore** so every file handle is
    ///    released before the files are removed.
    /// 3. **Mark the namespace's WAL entries Persisted** so WAL GC can reclaim
    ///    their segments — the clean-shutdown counterpart to recovery's skip.
    /// 4. **Delete the on-disk files.** This touches only `{db_path}/ns_{name}`
    ///    and `{db_path}/index/{ns_id}` — never the shared WAL, the WAL metadata,
    ///    or the registry — so WAL recovery for the surviving namespaces is
    ///    completely unaffected.
    pub fn remove_namespace(&self, name: &str) -> Result<u32> {
        self.check_closed()?;

        // (1) Durable point: once this returns, the namespace is gone from the
        // persisted registry (including its TTL config, so the global TTL worker
        // stops scanning it) and recovery will skip its WAL entries.
        let ns_id = self.registry.write().remove(name)?;

        // (2) Flush + close, then drop so the data directory's file handles are
        // released before we delete it.
        if let Some(store) = self.stores.write().remove(&ns_id) {
            store.set_flush_observer(None);
            let _ = store.shutdown();
            drop(store);
        }

        // (3) The KVStore has been flushed and shut down, so all WAL entries for
        // this namespace are now durable. Mark them persisted so WAL GC can
        // reclaim the segments they occupy without waiting for a flush that will
        // never come.
        let start = *self.last_persisted_wal_offset.read();
        let tail = self.wal_metadata.read().tail;
        self.wal_flush_observer.mark_namespace_persisted(ns_id, start, tail);

        // (4) Reclaim disk. Best-effort and independent of the WAL.
        self.remove_namespace_storage(ns_id, name);

        Ok(ns_id)
    }

    /// Delete a dropped namespace's on-disk files: its data directory
    /// (`{db_path}/ns_{name}`, holding the LSM SSTables and value log) and its
    /// index subtree (`{db_path}/index/{ns_id}`).
    ///
    /// Best-effort: failures are logged, not propagated. By the time this runs
    /// the registry entry is already gone, so any file left behind is
    /// unreferenced — it wastes disk but cannot affect correctness or recovery.
    /// This deliberately never touches the shared WAL or its metadata, so WAL
    /// replay for other namespaces is unaffected.
    fn remove_namespace_storage(&self, ns_id: u32, name: &str) {
        let ns_path = self.db_path.join(format!("ns_{}", name));
        match std::fs::remove_dir_all(&ns_path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => warn!("[NAMESPACE] Failed to remove data dir {:?} for ns={}: {:?}", ns_path, ns_id, e),
        }
        if let Err(e) = self.index_manager.remove_namespace_path(ns_id) {
            warn!("[NAMESPACE] Failed to remove index dir for ns={}: {:?}", ns_id, e);
        }
    }

    // ── Index field registry ───────────────────────────────────────────

    /// Register an indexed field for a namespace.
    ///
    /// Assigns a monotonic `FieldId` via the namespace schema, then creates
    /// `{db_path}/index/{namespace_id}/{field_id}/` on disk.
    /// The field name must be unique within the namespace.
    ///
    /// The `value_type` is stored in the schema and validated when
    /// [`activate_field_index`] is called, so type mismatches are caught at
    /// activation time rather than at query time.
    pub fn register_index_field(&self, namespace_id: u32, field_name: &str, value_type: IndexValueType) -> Result<FieldId> {
        let field_id = self.registry.write().register_schema_field(namespace_id, field_name, value_type)?;
        self.index_manager.ensure_field_path(namespace_id, field_id)?;
        Ok(field_id)
    }

    /// Register a custom row-ID function (and optionally its inverse) for a namespace.
    ///
    /// When set, every key written to or deleted from `namespace_id` will be
    /// assigned a row ID produced by `row_id_fn` rather than the default
    /// Murmur3 hash.  Providing `row_to_key_fn` (the exact inverse) additionally
    /// enables O(|hits|) query resolution with zero memory overhead: each matching
    /// key is reconstructed directly from its row ID without any in-memory map.
    ///
    /// This must be called **before** [`activate_field_index`] so that the WAL
    /// replay during activation uses the same row IDs that future writes will
    /// produce.
    ///
    /// [`query_keys`]: Self::query_keys
    /// [`activate_field_index`]: Self::activate_field_index
    pub fn set_row_id_fn(
        &self,
        namespace_id: u32,
        row_id_fn: crate::db::namespace_index::RowIdFn,
        row_to_key_fn: Option<crate::db::namespace_index::RowToKeyFn>,
    ) -> Result<()> {
        self.get_store(namespace_id)?.set_row_id_fn(row_id_fn, row_to_key_fn)
    }

    /// Return all indexed fields registered for a namespace, sorted by `FieldId`.
    pub fn list_index_fields(&self, namespace_id: u32) -> Vec<FieldMeta> {
        self.registry.read().schema(namespace_id).map(|s| s.list_fields()).unwrap_or_default()
    }

    /// Return the number of distinct indexed values for a field.
    ///
    /// Returns `None` when the field is not active (not yet registered via
    /// [`activate_field_index`]).
    pub fn field_index_distinct_count(&self, namespace_id: u32, field_id: FieldId) -> Option<usize> {
        let store = self.get_store(namespace_id).ok()?;
        let ns_index = store.namespace_index.read();
        ns_index.get(field_id).map(|e| e.index.read().distinct_count())
    }

    /// Reclaimable dead-space ratios (`0.0..1.0`) for a field's two append-only
    /// stores, as `(bitmap_waste, keymap_waste)`. The bitmap store grows with
    /// per-document churn; the keymap store grows under distinct-value churn.
    /// Compaction reclaims either at the index checkpoint once it crosses
    /// [`index_blob_waste_threshold`](Self::index_blob_waste_threshold).
    ///
    /// Returns `None` when the field is not active.
    pub fn field_index_waste(&self, namespace_id: u32, field_id: FieldId) -> Option<(f64, f64)> {
        let store = self.get_store(namespace_id).ok()?;
        let ns_index = store.namespace_index.read();
        ns_index.get(field_id).map(|e| {
            let idx = e.index.read();
            (idx.bitmap_waste_ratio(), idx.keymap_waste_ratio())
        })
    }

    /// On-disk blob growth/waste metrics for an active field index — the bitmap
    /// and keymap store sizes (logical vs. live bytes) and waste ratios. Use this
    /// to monitor the append-only write amplification that low-cardinality fields
    /// suffer (a value rewritten per document leaves a stale blob copy each time),
    /// e.g. to alert before disk fills between compactions.
    ///
    /// Returns `None` when the field is not active.
    pub fn field_index_blob_stats(&self, namespace_id: u32, field_id: FieldId) -> Option<index::IndexBlobStats> {
        let store = self.get_store(namespace_id).ok()?;
        let ns_index = store.namespace_index.read();
        ns_index.get(field_id).map(|e| e.index.read().blob_stats())
    }

    /// The configured field-index compaction threshold as a fraction
    /// (`0.0..1.0`) — a store is compacted at the next checkpoint once its waste
    /// ratio reaches this. Mirrors the clamp applied in `run_index_checkpoint`.
    pub fn index_blob_waste_threshold(&self) -> f64 {
        (self.config.threshold_config.index_blob_waste_threshold / 100.0).clamp(0.0, 1.0)
    }

    /// Wire up a live extractor for a previously-registered field.
    ///
    /// Loads the on-disk index state (BlobStore + keymap mmap store) and replays any
    /// WAL entries written since the last checkpoint, so the index is fully
    /// current before it becomes visible.  After this call, every `put` and
    /// `delete` on the namespace's KVStore will keep the in-memory
    /// [`DynFieldIndex`] up to date automatically.
    ///
    /// # Arguments
    /// * `namespace_id` – the namespace the field belongs to
    /// * `field_id`     – the [`FieldId`] returned by [`register_index_field`]
    /// * `value_type`   – runtime type for the index entries
    /// * `extractor`    – closure that extracts a typed value from raw document bytes
    pub fn activate_field_index(&self, namespace_id: u32, field_id: FieldId, value_type: IndexValueType, extractor: ExtractorFn) -> Result<()> {
        // Validate: field_id must be registered and its stored type must match.
        {
            let registry = self.registry.read();
            let schema = registry
                .schema(namespace_id)
                .ok_or_else(|| KVError::Serialization(format!("Namespace {} not found", namespace_id)))?;
            let meta = schema
                .get_field(field_id)
                .ok_or_else(|| KVError::Serialization(format!("Field id {} is not registered in namespace {}", field_id, namespace_id)))?;
            if meta.field_type != value_type {
                return Err(KVError::Serialization(format!(
                    "Type mismatch for field '{}' (id {}): registered as {:?}, activation supplies {:?}",
                    meta.field_name, field_id, meta.field_type, value_type
                )));
            }
        }

        let store = self.get_store(namespace_id)?;

        // Ensure the namespace's dense row-ID map is loaded before any
        // resolution happens — the write and replay paths resolve through it.
        store.ensure_rowmap(&self.index_manager.rowmap_path(namespace_id))?;

        // Open (or create) the file-backed mmap index in the field directory.
        let field_path = self.index_manager.field_path(namespace_id, field_id);
        let mut dyn_index = DynFieldIndex::open(value_type, &field_path).map_err(KVError::Io)?;

        // Replay the WAL tail into dyn_index *before* registering the entry.
        // Recovering on the unshared dyn_index (not yet wrapped in Arc<RwLock<>>)
        // guarantees no concurrent put can race with recovery and overwrite a
        // newer value with a stale one from the WAL scan.
        {
            let wal_tail = self.wal_metadata.read().tail;
            if wal_tail > 0 {
                let checkpoint_offset = self.index_manager.read_checkpoint(namespace_id, field_id, wal_tail);
                if checkpoint_offset < wal_tail {
                    let wal_head = self.wal_metadata.read().head;
                    let entries = self.wal.scan_entries(wal_head.max(checkpoint_offset), wal_tail)?;
                    let mut affected_keys = std::collections::BTreeSet::<Vec<u8>>::new();
                    for (_, wal_entry) in &entries {
                        if wal_entry.namespace_id == namespace_id {
                            affected_keys.insert(wal_entry.key.clone());
                        }
                    }
                    debug!(
                        "[activate_field_index] ns={} field={} WAL replay: \
                         {} entries scanned, {} keys affected",
                        namespace_id,
                        field_id,
                        entries.len(),
                        affected_keys.len()
                    );

                    // Defense-in-depth tripwire for the `set_row_id_fn` ordering
                    // contract. If we are about to rebuild an already-populated
                    // field index through the dense RowMap fallback (no RowIdFn
                    // registered) while that RowMap has never allocated an ID,
                    // the existing on-disk entries were almost certainly built
                    // under a custom RowIdFn that should have been registered
                    // BEFORE activation — replaying now would mix row-ID schemes.
                    // Legitimate RowMap-based namespaces don't trip this: their
                    // RowMap is non-empty once any write has occurred.
                    if !affected_keys.is_empty() {
                        let scheme_mismatch = store.rowmap_active() && store.rowmap_is_empty() && dyn_index.distinct_count() > 0;
                        debug_assert!(
                            !scheme_mismatch,
                            "activate_field_index ns={namespace_id} field={field_id}: replaying a non-empty field index \
                             through the dense RowMap fallback with no RowIdFn registered and an unused RowMap — register \
                             set_row_id_fn BEFORE activate_field_index (see its ordering contract); replaying now mixes row-ID schemes"
                        );
                        if scheme_mismatch {
                            warn!(
                                "[activate_field_index] ns={} field={}: possible row-ID scheme mismatch — \
                                 rebuilding a non-empty index via the RowMap fallback with no RowIdFn set; \
                                 register set_row_id_fn before activation",
                                namespace_id, field_id
                            );
                        }
                    }

                    for key in affected_keys {
                        let current_value = store.get(&key)?;
                        match current_value {
                            // Live key: allocate/resolve its dense ID, then
                            // rebuild its index entry from the current value.
                            Some(ref bytes) => {
                                let row_id = store.resolve_row_id_alloc(&key);
                                dyn_index.remove_all_for_row(row_id);
                                if let Some(v) = extractor(bytes)
                                    && let Err(e) = dyn_index.insert(&v, row_id)
                                {
                                    warn!(
                                        "[activate_field_index] index update rejected \
                                         ns={} field={}: {}",
                                        namespace_id, field_id, e
                                    );
                                }
                            }
                            // Deleted key: clear any existing entry without
                            // allocating a (dead) ID for a key that no longer exists.
                            None => {
                                if let Some(row_id) = store.resolve_row_id_get(&key) {
                                    dyn_index.remove_all_for_row(row_id);
                                }
                            }
                        }
                    }
                    dyn_index.flush(&field_path).map_err(KVError::Io)?;
                }
            }
        }

        let entry = IndexEntry {
            field_id,
            extractor,
            index: Arc::new(parking_lot::RwLock::new(dyn_index)),
        };
        store.namespace_index.write().register(entry);
        Ok(())
    }

    /// Remove a previously-activated field index from the in-memory registry.
    ///
    /// After this call the field's bitmap is dropped and any predicate query
    /// that references it returns [`KVError::Serialization`] wrapping
    /// [`index::query::QueryError::InactiveField`].  The on-disk checkpoint
    /// files are left untouched; callers are responsible for removing them.
    pub fn deactivate_field_index(&self, namespace_id: u32, field_id: FieldId) -> Result<()> {
        self.get_store(namespace_id)?.namespace_index.write().deregister(field_id);
        Ok(())
    }

    // ── Query execution ────────────────────────────────────────────────

    /// Evaluate a query string against the active field indices of a namespace
    /// and return the raw keys of all matching documents.
    ///
    /// # Arguments
    /// * `namespace_id` — the namespace to query
    /// * `query_str`    — query string, e.g. `"age > 30 AND status = 'active'"`
    ///
    /// # How it works
    /// 1. Parses and evaluates `query_str` against the in-memory field indices,
    ///    producing a bitmap of matching row IDs.
    /// 2. Scans all keys in the namespace's KVStore and returns those whose
    ///    derived row ID (`key_to_row_id`) appears in the bitmap.
    ///
    /// # Limitations
    /// Only fields activated via [`activate_field_index`] are queryable.
    /// Unindexed fields in the predicate produce a [`KVError::Serialization`]
    /// wrapping a [`index::query::QueryError::InactiveField`].
    pub fn query_keys(&self, namespace_id: u32, query_str: &str) -> Result<Vec<Vec<u8>>> {
        use index::query::{SchemaMap, parse_and_evaluate};

        let store = self.get_store(namespace_id)?;

        // Build the schema map: field_name → field_id, restricted to fields
        // that have an active in-memory index. Dropped fields remain in the
        // registry for field_id reuse but must not appear as queryable fields.
        let schema_map: SchemaMap = {
            let registry = self.registry.read();
            let ns_index = store.namespace_index.read();
            registry
                .schema(namespace_id)
                .map(|s| {
                    s.list_fields()
                        .into_iter()
                        .filter(|f| ns_index.get(f.field_id).is_some())
                        .map(|f| (f.field_name, f.field_id))
                        .collect()
                })
                .unwrap_or_default()
        };

        // Closure: look up a live DynFieldIndex by field_id
        let get_index = |field_id: u32| {
            let ns_index = store.namespace_index.read();
            ns_index.get(field_id).map(|e| Arc::clone(&e.index))
        };

        // Evaluate the query → bitmap of matching row IDs
        let bitmap = parse_and_evaluate(query_str, &schema_map, &get_index).map_err(|e| KVError::Serialization(e.to_string()))?;

        if bitmap.is_empty() {
            return Ok(Vec::new());
        }

        // Fast path: when a RowToKeyFn inverse is registered, reconstruct each
        // matching key directly from its row ID — O(|hits|), zero memory overhead,
        // crash-safe (no map to rebuild on restart).
        if let Some(ref inv) = *store.row_to_key_fn.read() {
            return Ok(bitmap.iter().map(|row_id| inv(row_id)).collect());
        }

        // Fast path: the dense row map resolves each hit's key directly — O(|hits|).
        if store.rowmap_active() {
            return Ok(bitmap.iter().filter_map(|row_id| store.rowmap_key_for(row_id)).collect());
        }

        // Fallback (no inverse function): scan all keys and check bitmap membership.
        // Pre-existing O(n_keys) path retained for backward compatibility.
        let all_keys = store.keys()?;
        let matching = all_keys
            .into_iter()
            .filter(|key| store.resolve_row_id_get(key).is_some_and(|id| bitmap.contains(id)))
            .collect();

        Ok(matching)
    }

    /// Evaluate a query and return `(page_keys, total)` where `total` is the
    /// full match count (bitmap cardinality) and `page_keys` contains at most
    /// `limit` keys starting from `offset` in iteration order.
    ///
    /// More efficient than [`query_keys`] when only a page of results is needed:
    /// - With a registered [`RowToKeyFn`]: O(offset + limit) key resolutions.
    /// - Fallback (no inverse): O(n_keys) scan but no full match list allocated.
    ///
    /// [`query_keys`]: Self::query_keys
    pub fn query_keys_paginated(&self, namespace_id: u32, query_str: &str, offset: usize, limit: usize) -> Result<(Vec<Vec<u8>>, usize)> {
        use index::query::{SchemaMap, parse_and_evaluate};

        let store = self.get_store(namespace_id)?;

        let schema_map: SchemaMap = {
            let registry = self.registry.read();
            let ns_index = store.namespace_index.read();
            registry
                .schema(namespace_id)
                .map(|s| {
                    s.list_fields()
                        .into_iter()
                        .filter(|f| ns_index.get(f.field_id).is_some())
                        .map(|f| (f.field_name, f.field_id))
                        .collect()
                })
                .unwrap_or_default()
        };

        let get_index = |field_id: u32| {
            let ns_index = store.namespace_index.read();
            ns_index.get(field_id).map(|e| Arc::clone(&e.index))
        };

        let bitmap = parse_and_evaluate(query_str, &schema_map, &get_index).map_err(|e| KVError::Serialization(e.to_string()))?;

        let total = bitmap.len();

        if total == 0 {
            return Ok((Vec::new(), 0));
        }

        // Fast path: RowToKeyFn registered — resolve only the page window.
        if let Some(ref inv) = *store.row_to_key_fn.read() {
            let keys: Vec<Vec<u8>> = bitmap.iter().skip(offset).take(limit).map(|row_id| inv(row_id)).collect();
            return Ok((keys, total));
        }

        // Fast path: dense row map — resolve only the page window.
        if store.rowmap_active() {
            let keys: Vec<Vec<u8>> = bitmap
                .iter()
                .skip(offset)
                .take(limit)
                .filter_map(|row_id| store.rowmap_key_for(row_id))
                .collect();
            return Ok((keys, total));
        }

        // Fallback: scan all keys, filter by bitmap membership, then window.
        let all_keys = store.keys()?;
        let keys: Vec<Vec<u8>> = all_keys
            .into_iter()
            .filter(|key| store.resolve_row_id_get(key).is_some_and(|id| bitmap.contains(id)))
            .skip(offset)
            .take(limit)
            .collect();
        Ok((keys, total))
    }

    // ── Store access ───────────────────────────────────────────────────

    /// Get a KVStore by namespace ID
    fn get_store(&self, namespace_id: u32) -> Result<Arc<KVStore>> {
        self.stores
            .read()
            .get(&namespace_id)
            .cloned()
            .ok_or_else(|| KVError::Serialization(format!("Namespace with ID {} not found", namespace_id)))
    }

    /// Get a KVStore by namespace name
    pub fn get_store_by_name(&self, name: &str) -> Result<Arc<KVStore>> {
        let ns_id = self
            .registry
            .read()
            .get_id(name)
            .ok_or_else(|| KVError::Serialization(format!("Namespace '{}' not found", name)))?;
        self.get_store(ns_id)
    }

    /// Get the default namespace's KVStore
    pub fn default_store(&self) -> Result<Arc<KVStore>> {
        self.get_store(DEFAULT_NAMESPACE_ID)
    }

    // ── WAL helpers ────────────────────────────────────────────────────

    fn check_closed(&self) -> Result<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(KVError::DatabaseClosed);
        }
        Ok(())
    }

    fn flush_wal_metadata_internal(&self) -> Result<()> {
        let wal_metadata = self.wal_metadata.read();
        let bytes = wal_metadata.to_file_bytes()?;
        crate::support::write_atomic_durable(&self.wal_metadata_path, &bytes)?;
        Ok(())
    }

    // ── Recovery ───────────────────────────────────────────────────────

    fn recover_from_wal(&self) -> Result<()> {
        let wal_metadata = self.wal_metadata.read();
        let head = wal_metadata.head;
        let tail = wal_metadata.tail;
        let total_entries = wal_metadata.total_entries;
        let persisted_entries = wal_metadata.persisted_entries;
        drop(wal_metadata);

        if persisted_entries >= total_entries || tail == 0 {
            debug!("[RECOVERY] No recovery needed.");
            return Ok(());
        }

        info!(
            "[RECOVERY] Starting WAL recovery. Total: {}, Persisted: {}",
            total_entries, persisted_entries
        );

        let all_entries = self.wal.scan_entries(head, tail)?;

        // Collect every eligible entry into a single flat list, then replay
        // strictly in global sequence order.
        //
        // Replaying in sequence order is essential for correctness: the *last*
        // write to a key by sequence must win after recovery, exactly as it did
        // before the crash. Sequence == WAL append order (writers allocate the
        // sequence under the WAL lock), so sorting by sequence reproduces the
        // original order.
        //
        // Eligibility:
        //   • Already-Persisted entries are skipped (nothing to replay).
        //   • Entries for a namespace that no longer exists are skipped. This
        //     happens when a crash interrupts `remove_namespace` after it has
        //     persisted the registry deletion but before it could mark the
        //     namespace's WAL entries Persisted (see `mark_namespace_persisted`).
        //     The namespace is intentionally gone, so replaying its writes would
        //     be wrong — and there is no KVStore to replay them into. We drop
        //     them with a WARN rather than routing them to the fail log, which is
        //     reserved for writes that genuinely failed to apply. Namespace IDs
        //     are monotonic and never reused, so a missing store is unambiguous.
        use std::collections::HashSet;
        let known_namespaces: HashSet<u32> = self.stores.read().keys().copied().collect();
        let mut skipped_unknown_ns_count = 0u64;
        let mut warned_namespaces: HashSet<u32> = HashSet::new();
        let mut eligible: Vec<WalEntry> = Vec::new();
        for (_, entry) in &all_entries {
            if entry.status == WalEntryStatus::Persisted {
                continue;
            }
            if !known_namespaces.contains(&entry.namespace_id) {
                // Log once per namespace to avoid flooding the log when a deleted
                // namespace had many un-persisted entries at crash time.
                if warned_namespaces.insert(entry.namespace_id) {
                    warn!(
                        "[RECOVERY] Skipping WAL entries for unknown namespace ns={} \
                         (likely a namespace deleted just before the crash). Its writes \
                         will not be replayed.",
                        entry.namespace_id
                    );
                }
                skipped_unknown_ns_count += 1;
                continue;
            }
            eligible.push(entry.clone());
        }
        // Stable sort by sequence so entries replay in their original order.
        eligible.sort_by_key(|e| e.sequence);

        // Apply each entry in sequence order, retrying once *immediately* on
        // failure. Immediate (rather than deferred) retry is what preserves the
        // ordering guarantee: a failed entry has no effect, so retrying it before
        // moving on means a later write to the same key still lands afterwards.
        let mut recovered_count = 0u64;
        let mut error_count = 0u64;
        let mut failed_entries: Vec<(WalEntry, String)> = Vec::new();

        for entry in eligible {
            match self.apply_wal_entry(&entry) {
                Ok(()) => recovered_count += 1,
                Err(e) => {
                    warn!(
                        "[RECOVERY] Apply failed for op '{}' key='{}' (ns={}): {}. Retrying once.",
                        entry.op_name,
                        display_key(&entry.key),
                        entry.namespace_id,
                        e
                    );
                    match self.apply_wal_entry(&entry) {
                        Ok(()) => recovered_count += 1,
                        Err(e2) => {
                            error!(
                                "[RECOVERY] Retry also failed for op '{}' key='{}' (ns={}): {}. \
                                 Writing to fail log.",
                                entry.op_name,
                                display_key(&entry.key),
                                entry.namespace_id,
                                e2
                            );
                            error_count += 1;
                            failed_entries.push((entry, e2.to_string()));
                        }
                    }
                }
            }
        }

        // Write any persistent failures to the fail log — one record per failed op.
        if !failed_entries.is_empty() {
            let failures: Vec<(&WalEntry, String)> = failed_entries.iter().map(|(e, msg)| (e, msg.clone())).collect();
            crate::db::fail_log::write_fail_log(&self.fail_log_dir, &self.db_path, &failures);
        }

        // Flush all namespaces that had recovered entries.
        if recovered_count > 0 {
            let stores = self.stores.read();
            for (_, kv_store) in stores.iter() {
                let _ = kv_store.flush_and_compact_all();
            }
        }

        // Mark all scanned entries as Persisted — including entries that failed
        // after retry.  WAL GC can now reclaim their segments; the fail log is
        // the operator's recovery path.
        let entries = self.wal.scan_entries(head, tail)?;
        let mut per_segment: BTreeMap<u64, u64> = BTreeMap::new();
        for (pointer, entry) in entries {
            if entry.status != WalEntryStatus::Persisted
                && let Err(e) = self.wal.update_entry_status(pointer.offset, WalEntryStatus::Persisted)
            {
                warn!("[RECOVERY] Failed to update WAL entry status: {:?}", e);
            }
            let segment_id = self.wal.segment_id_for_offset(pointer.offset);
            *per_segment.entry(segment_id).or_insert(0) += 1;
        }

        {
            let mut wal_metadata = self.wal_metadata.write();
            wal_metadata.reconcile_segment_lengths();
            for segment_id in wal_metadata.tracked_segments().collect::<Vec<_>>() {
                let total = wal_metadata.segment_total(segment_id);
                let persisted = per_segment.get(&segment_id).copied().unwrap_or(0);
                wal_metadata.set_segment_persisted(segment_id, persisted.min(total));
            }
            wal_metadata.persisted_entries = wal_metadata.total_entries;
        }
        *self.last_persisted_wal_offset.write() = tail;
        self.flush_wal_metadata_internal()?;

        info!(
            "[RECOVERY] Complete. Recovered: {}, \
             Skipped (unknown/deleted namespaces): {}, Failed after retry (see fail log): {}",
            recovered_count, skipped_unknown_ns_count, error_count
        );
        Ok(())
    }

    /// Apply a single WAL entry to its KV store.  Returns `Err` if the
    /// namespace is missing or the underlying storage write fails.
    ///
    /// `recover_from_wal` filters out entries for deleted namespaces before
    /// calling this, so in the recovery path a missing namespace here indicates
    /// a genuine inconsistency rather than an ordinary post-deletion entry.
    fn apply_wal_entry(&self, entry: &WalEntry) -> Result<()> {
        let kv_store = self.get_store(entry.namespace_id)?;
        let seq = entry.sequence;
        match entry.operation {
            WalOperationType::Upsert => {
                if let Some(value) = &entry.value {
                    kv_store.replay_upsert(&entry.key, value, seq)
                } else {
                    Ok(())
                }
            }
            WalOperationType::Delete => kv_store.replay_delete(&entry.key, seq),
        }
    }

    fn rebuild_wal_persisted_state(&self) -> Result<u64> {
        let wal_metadata = self.wal_metadata.read();
        let head = wal_metadata.head;
        let tail = wal_metadata.tail;
        drop(wal_metadata);

        if tail == 0 {
            return Ok(head);
        }

        let entries = self.wal.scan_entries(head, tail)?;
        let mut persisted_entries = 0u64;
        let mut per_segment: BTreeMap<u64, u64> = BTreeMap::new();
        let mut last_persisted_offset = head;

        for (pointer, entry) in entries {
            if entry.status != WalEntryStatus::Persisted {
                break;
            }
            persisted_entries += 1;
            let segment_id = self.wal.segment_id_for_offset(pointer.offset);
            *per_segment.entry(segment_id).or_insert(0) += 1;
            last_persisted_offset = pointer.offset + 4 + pointer.size as u64;
        }

        {
            let mut wal_metadata = self.wal_metadata.write();
            wal_metadata.persisted_entries = persisted_entries;
            wal_metadata.reconcile_segment_lengths();
            wal_metadata.reset_segment_persisted();
            for (segment_id, count) in per_segment {
                let total = wal_metadata.segment_total(segment_id);
                wal_metadata.set_segment_persisted(segment_id, count.min(total));
            }
        }
        self.flush_wal_metadata_internal()?;
        Ok(last_persisted_offset)
    }

    // ── WAL GC ─────────────────────────────────────────────────────────

    pub fn get_wal_gc_stats(&self) -> (u64, u64) {
        let wal_metadata = self.wal_metadata.read();
        (wal_metadata.total_entries, wal_metadata.persisted_entries)
    }

    /// Returns `true` when at least one non-current WAL segment is fully persisted
    /// and ready to be deleted.  Entries in the active segment are not yet eligible
    /// — they will be marked persisted after the next memtable flush or clean shutdown.
    pub fn has_deletable_wal_segments(&self) -> bool {
        let wal_metadata = self.wal_metadata.read();
        if wal_metadata.tail == 0 {
            return false;
        }
        let current = self.wal.segment_id_for_offset(wal_metadata.tail - 1);
        wal_metadata.tracked_segments().take_while(|&s| s < current).any(|s| {
            let t = wal_metadata.segment_total(s);
            t > 0 && t == wal_metadata.segment_persisted(s)
        })
    }

    pub fn garbage_collect_wal(&self) -> Result<(u64, u64)> {
        if self
            .wal_gc_in_progress
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            let (total, persisted) = self.get_wal_gc_stats();
            return Ok((0, total.saturating_sub(persisted)));
        }

        struct WalGcGuard<'a>(&'a AtomicBool);
        impl<'a> Drop for WalGcGuard<'a> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        let _guard = WalGcGuard(&self.wal_gc_in_progress);

        let mut wal_metadata = self.wal_metadata.write();
        let current_segment_id = if wal_metadata.tail == 0 {
            0
        } else {
            self.wal.segment_id_for_offset(wal_metadata.tail - 1)
        };

        // Persist the current sequence high-water mark BEFORE deleting any segment.
        // This ensures recover_sequence always finds a hint >= the max sequence
        // in any segment that survives, even if those segments are later deleted.
        wal_metadata.last_sequence = self.next_seq.load(Ordering::Relaxed).saturating_sub(1);

        let mut bytes_reclaimed = 0u64;
        let mut segments_deleted = 0u64;
        let candidates: Vec<u64> = wal_metadata.tracked_segments().take_while(|&s| s < current_segment_id).collect();
        for segment_id in candidates {
            let total = wal_metadata.segment_total(segment_id);
            let persisted = wal_metadata.segment_persisted(segment_id);
            if total > 0 && total == persisted && self.wal.delete_segment_file(segment_id).is_ok() {
                bytes_reclaimed = bytes_reclaimed.saturating_add(self.wal.segment_size());
                segments_deleted += 1;
                wal_metadata.total_entries = wal_metadata.total_entries.saturating_sub(total);
                wal_metadata.persisted_entries = wal_metadata.persisted_entries.saturating_sub(persisted);
                wal_metadata.clear_segment(segment_id);
            }
        }

        // Advance head past all consecutively deleted/empty segments so that a
        // subsequent scan_entries(head, tail) never tries to open a deleted file.
        // Segments with segment_total == 0 were either deleted in this run or in
        // a previous one; either way their files are gone.
        let head_segment = wal_metadata.head / self.wal.segment_size();
        let first_live = (head_segment..=current_segment_id)
            .find(|&sid| sid == current_segment_id || wal_metadata.segment_total(sid) > 0)
            .unwrap_or(current_segment_id);
        let new_head = first_live * self.wal.segment_size();
        if new_head > wal_metadata.head {
            wal_metadata.head = new_head;
        }
        // Trim per-segment counters below the new head so the dense vecs track
        // only the live segment window instead of growing with every segment
        // ever created (keeps `base_segment_id == head`'s segment).
        wal_metadata.trim_segments_before(first_live);

        wal_metadata.total_gc_runs = wal_metadata.total_gc_runs.saturating_add(1);
        wal_metadata.total_bytes_reclaimed = wal_metadata.total_bytes_reclaimed.saturating_add(bytes_reclaimed);
        let remaining = wal_metadata.total_entries.saturating_sub(wal_metadata.persisted_entries);
        drop(wal_metadata);
        self.flush_wal_metadata_internal()?;

        crate::db::metrics::Metrics::bump(&self.metrics.wal_gc_runs);
        crate::db::metrics::Metrics::add(&self.metrics.wal_segments_deleted, segments_deleted);

        Ok((bytes_reclaimed, remaining))
    }

    // ── Value log GC (per namespace) ───────────────────────────────────

    /// Run value log GC on a specific namespace
    pub fn garbage_collect_namespace(&self, namespace_id: u32) -> Result<GCStats> {
        let kv_store = self.get_store(namespace_id)?;
        let threshold = self.config.threshold_config.value_log_waste_threshold;
        kv_store.garbage_collect_with_threshold(threshold)
    }

    /// Run value log GC on the default namespace
    pub fn garbage_collect(&self) -> Result<GCStats> {
        self.garbage_collect_namespace(DEFAULT_NAMESPACE_ID)
    }

    // ── LSM compaction ─────────────────────────────────────────────────

    pub fn compact_lsm(&self) -> Result<()> {
        self.check_closed()?;
        let stores = self.stores.read();
        for (_, kv_store) in stores.iter() {
            kv_store.compact_lsm()?;
        }
        Ok(())
    }

    pub fn has_lsm_compaction_work(&self) -> bool {
        let stores = self.stores.read();
        stores.values().any(|s| s.has_lsm_compaction_work())
    }

    // ── Stats ──────────────────────────────────────────────────────────

    pub fn stats(&self) -> Stats {
        self.default_store().map(|s| s.stats()).unwrap_or_else(|_| Stats {
            head: 0,
            tail: 0,
            garbage_size: 0,
            waste_ratio: 0.0,
            free_space_ratio: 0.0,
            total_gc_runs: 0,
            total_bytes_reclaimed: 0,
            live_bytes: 0,
        })
    }

    pub fn get_waste_ratio(&self) -> f64 {
        self.default_store().map(|s| s.get_waste_ratio()).unwrap_or(0.0)
    }

    /// Returns per-bucket value-log metadata for every active namespace.
    ///
    /// Each entry is `(namespace_name, Vec<(bucket_id, ValueLogMetadata)>)`.
    pub fn value_log_shard_stats(&self) -> Vec<(String, Vec<(u32, ValueLogMetadata)>)> {
        let namespaces = self.list_namespaces();
        let stores = self.stores.read();
        namespaces
            .into_iter()
            .filter_map(|(name, ns_id)| stores.get(&ns_id).map(|s| (name, s.value_log.get_all_bucket_stats())))
            .collect()
    }

    /// Physical (on-disk `st_blocks`) vs logical (file length) value-log
    /// footprint per shard for every namespace. Cheap: one `stat` per shard.
    pub fn value_log_physical_stats(&self) -> Vec<(String, Vec<crate::store::value_log::sharded::ShardPhysicalStats>)> {
        let namespaces = self.list_namespaces();
        let stores = self.stores.read();
        namespaces
            .into_iter()
            .filter_map(|(name, ns_id)| stores.get(&ns_id).map(|s| (name, s.value_log.physical_stats())))
            .collect()
    }

    /// Per-page garbage breakdown for one namespace's value-log shards. Cost is
    /// O(pages × records), so it is scoped to a single namespace (by name).
    pub fn value_log_page_stats(&self, namespace: &str) -> Result<Vec<(u32, Vec<crate::store::value_log::PageGarbageStats>)>> {
        let store = self.get_store_by_name(namespace)?;
        let mut out = Vec::new();
        for bucket in 0..store.value_log.num_buckets() as u32 {
            let pages = store.value_log.page_stats(bucket).map_err(KVError::from)?;
            out.push((bucket, pages));
        }
        Ok(out)
    }

    /// Run value-log GC on every namespace and return per-namespace results.
    pub fn garbage_collect_all(&self) -> Vec<(String, GCStats)> {
        self.list_namespaces()
            .into_iter()
            .filter_map(|(name, ns_id)| self.garbage_collect_namespace(ns_id).ok().map(|stats| (name, stats)))
            .collect()
    }

    /// Returns a snapshot of the current WAL metadata.
    pub fn wal_metadata(&self) -> WalMetadata {
        self.wal_metadata.read().clone()
    }

    /// Snapshot of the engine-wide operational counters (runtime metrics).
    pub fn metrics_snapshot(&self) -> crate::db::metrics::MetricsSnapshot {
        self.metrics.snapshot()
    }

    /// Returns a live manifest snapshot for every active namespace.
    ///
    /// Each entry is `(namespace_name, manifest)`.  The manifest is built
    /// from the in-memory LSM state so it does not touch disk.
    pub fn lsm_manifests(&self) -> Vec<(String, crate::store::lsm::lsm_manifest::LsmManifest)> {
        let namespaces = self.list_namespaces();
        let stores = self.stores.read();
        namespaces
            .into_iter()
            .filter_map(|(name, ns_id)| stores.get(&ns_id).and_then(|s| s.lsm.build_manifest_snapshot().ok()).map(|m| (name, m)))
            .collect()
    }

    /// Returns the in-memory (non-SSTable) LSM stats for every active namespace,
    /// as `(namespace_name, stats)`. Complements [`lsm_manifests`](Self::lsm_manifests),
    /// which only reflects on-disk SSTables.
    pub fn lsm_runtime_stats(&self) -> Vec<(String, crate::store::lsm::lsm_tree::LSMStats)> {
        let namespaces = self.list_namespaces();
        let stores = self.stores.read();
        namespaces
            .into_iter()
            .filter_map(|(name, ns_id)| stores.get(&ns_id).map(|s| (name, s.lsm.stats())))
            .collect()
    }

    // ── Shutdown ──────────────────────────────────────────────────────────

    pub fn shutdown(&self) -> Result<()> {
        if self.closed.swap(true, Ordering::SeqCst) {
            return Err(KVError::DatabaseClosed);
        }

        // Shutdown all KVStores
        let stores = self.stores.read();
        for (_, kv_store) in stores.iter() {
            kv_store.set_flush_observer(None);
            kv_store.shutdown()?;
        }
        drop(stores);

        // Flush all active field indices (keymap + bitmap data) to disk.
        // This persists in-memory index state accumulated during the session,
        // including entries written via put_no_wal which have no WAL to replay.
        self.run_index_checkpoint()?;

        // Mark remaining WAL entries as persisted
        let start = *self.last_persisted_wal_offset.read();
        let tail = self.wal_metadata.read().tail;
        self.wal_flush_observer.mark_persisted_range(start, tail);

        // Flush WAL metadata
        self.flush_wal_metadata_internal()?;

        // Sync WAL
        self.wal.sync()?;

        Ok(())
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    // ── Test-only helpers ─────────────────────────────────────────────────

    #[cfg(test)]
    pub(crate) fn open_with_wal_segment_size(db_path: &Path, mut config: DbConfig, wal_segment_size: u64) -> Result<Self> {
        std::fs::create_dir_all(db_path)?;

        let default_vlog_dir = db_path.join("ns_default").join("value_logs");
        if default_vlog_dir.exists() {
            let existing_count = Self::detect_bucket_count(&default_vlog_dir);
            if existing_count > 0 && existing_count != config.num_buckets {
                config.num_buckets = existing_count;
            }
        }

        config.lsm_config.num_buckets = config.num_buckets;
        config.lsm_config.skip_list_capacity = config.skip_list_capacity;

        let wal_path = db_path.join("wal.log");
        let wal_metadata_path = db_path.join("wal_metadata");

        let wal = Arc::new(Wal::open_with_options_and_segment_size(&wal_path, false, wal_segment_size)?);

        let mut wal_metadata = if wal_metadata_path.exists() {
            let data = std::fs::read(&wal_metadata_path)?;
            match WalMetadata::from_file_bytes(&data) {
                Ok(m) => m,
                Err(_) => {
                    let backup = wal_metadata_path.with_extension("corrupt");
                    let _ = std::fs::rename(&wal_metadata_path, &backup);
                    WalMetadata::new()
                }
            }
        } else {
            WalMetadata::new()
        };
        wal_metadata.reconcile_segment_lengths();

        // WAL entries are fsynced on every write, but `wal_metadata` (which holds
        // the tail) is only flushed periodically — so after a crash the persisted
        // tail can lag the durable end of the log, and recovery scanning only up
        // to the stale tail would silently drop fsynced entries. Reconstruct the
        // true tail from the self-describing WAL and fold the durable-but-
        // unaccounted entries into the counters so recovery actually replays them
        // (and is not short-circuited by the stale total/persisted counts). With
        // a lost/corrupt metadata file (tail = 0) this rebuilds the tail wholesale.
        {
            let persisted_tail = wal_metadata.tail;
            let true_tail = wal.recover_tail(persisted_tail);
            if true_tail > persisted_tail {
                let extra = wal.scan_entries(persisted_tail, true_tail).unwrap_or_default();
                warn!(
                    "[RECOVERY] WAL metadata tail ({}) lagged the durable log end ({}); \
                     recovering {} entry(ies) appended since the last metadata flush",
                    persisted_tail,
                    true_tail,
                    extra.len()
                );
                for (pointer, _) in &extra {
                    let segment_id = wal.segment_id_for_offset(pointer.offset);
                    wal_metadata.add_segment_total(segment_id, 1);
                }
                wal_metadata.total_entries = wal_metadata.total_entries.saturating_add(extra.len() as u64);
                wal_metadata.tail = true_tail;
            }
        }

        let next_seq_start = wal.recover_sequence(wal_metadata.head, wal_metadata.tail, wal_metadata.last_sequence);

        let wal_metadata = Arc::new(RwLock::new(wal_metadata));
        let pending_wal_flushes = Arc::new(RwLock::new(BTreeMap::new()));
        let last_persisted_wal_offset = Arc::new(RwLock::new(0u64));
        let wal_flush_observer = Arc::new(WalPersistObserver::new(
            Arc::clone(&wal),
            Arc::clone(&wal_metadata),
            wal_metadata_path.clone(),
            Arc::clone(&pending_wal_flushes),
            Arc::clone(&last_persisted_wal_offset),
        ));

        let registry = NamespaceRegistry::open(db_path)?;
        let index_manager = IndexManager::open(db_path)?;

        let mut stores = HashMap::new();
        for (name, ns_id) in registry.list() {
            let ns_path = db_path.join(format!("ns_{}", name));
            let kv_store = KVStore::open(ns_id, name, &ns_path, config.lsm_config.clone(), config.sync_config)?;
            kv_store.set_verify_checksums_on_read(config.verify_checksums_on_read);
            kv_store.cleanup_old_files_on_startup()?;
            stores.insert(ns_id, Arc::new(kv_store));
        }

        let old_wal = db_path.join("wal.log.old");
        if old_wal.exists() {
            let _ = std::fs::remove_file(&old_wal);
        }

        let fail_log_dir = config.fail_log_dir.clone().unwrap_or_else(|| db_path.join("fail_logs"));

        let db = Self {
            db_path: db_path.to_path_buf(),
            config,
            wal,
            wal_path,
            wal_metadata_path,
            wal_metadata,
            wal_flush_observer,
            pending_wal_flushes,
            last_persisted_wal_offset,
            wal_gc_in_progress: Arc::new(AtomicBool::new(false)),
            registry: RwLock::new(registry),
            stores: RwLock::new(stores),
            closed: Arc::new(AtomicBool::new(false)),
            wal_gc_worker: Arc::new(tokio::sync::RwLock::new(None)),
            lsm_compaction_worker: Arc::new(tokio::sync::RwLock::new(None)),
            value_log_gc_worker: Arc::new(tokio::sync::RwLock::new(None)),
            lsm_compaction_sender: Arc::new(parking_lot::RwLock::new(None)),
            ttl_worker: Arc::new(tokio::sync::RwLock::new(None)),
            index_manager,
            index_checkpoint_worker: Arc::new(tokio::sync::RwLock::new(None)),
            next_seq: Arc::new(AtomicU64::new(next_seq_start)),
            fail_log_dir,
            metrics: Arc::new(crate::db::metrics::Metrics::default()),
        };

        // Share the global WAL sequence counter with every store (see the other
        // open path); must happen before recovery replays entries. Share the
        // operational counters too.
        {
            let stores = db.stores.read();
            for kv_store in stores.values() {
                kv_store.set_seq_counter(db.next_seq.clone());
                kv_store.set_metrics(db.metrics.clone());
            }
        }

        db.recover_from_wal()?;
        let last_persisted = db.rebuild_wal_persisted_state()?;
        *db.last_persisted_wal_offset.write() = last_persisted;
        db.wire_up_flush_observers();

        Ok(db)
    }

    #[cfg(test)]
    pub(crate) fn simulate_crash_with_wal_entries(&self, entries: Vec<WalEntry>) -> Result<()> {
        let mut wal_metadata = self.wal_metadata.write();
        for entry in entries {
            let _ = self.wal.append_entry(&entry, &mut wal_metadata.tail, false)?;
            let segment_id = self.wal.segment_id_for_offset(wal_metadata.tail.saturating_sub(1));
            wal_metadata.add_segment_total(segment_id, 1);
            wal_metadata.total_entries += 1;
        }
        drop(wal_metadata);
        self.flush_wal_metadata_internal()?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn flush_all_namespaces(&self) -> Result<()> {
        let stores = self.stores.read();
        for (_, kv_store) in stores.iter() {
            kv_store.flush_and_compact_all()?;
        }
        Ok(())
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        // Skip if shutdown() already ran cleanly
        if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }

        // Flush and sync each KVStore
        let stores = self.stores.read();
        for (_, kv_store) in stores.iter() {
            let _ = kv_store.shutdown();
        }
        drop(stores);

        // Flush all active field indices to disk (keymap + bitmap data).
        let _ = self.run_index_checkpoint();

        // Flush WAL metadata and sync WAL
        let _ = self.flush_wal_metadata_internal();
        let _ = self.wal.sync();
    }
}

// ── IndexCheckpointTarget impl ────────────────────────────────────────────

impl IndexCheckpointTarget for Database {
    fn is_closed(&self) -> bool {
        self.is_closed()
    }

    fn run_index_checkpoint(&self) -> Result<usize> {
        let wal_tail = self.wal_metadata.read().tail;
        let fields = self.registry.read().all_indexed_fields();

        // Flush mmap bitmap data for each active field index, and collect only
        // those that are still active (dropped fields are deregistered from
        // namespace_index but remain in the registry schema).
        let waste_threshold = (self.config.threshold_config.index_blob_waste_threshold / 100.0).clamp(0.0, 1.0);
        let mut active_fields = Vec::with_capacity(fields.len());
        let stores = self.stores.read();

        // Flush each namespace's dense row map FIRST and advance its marker, so
        // the map is at least as durable as every field bitmap that references
        // its IDs (a crash must never leave a persisted bit whose row ID is not
        // reproducible from the row map). See `index::RowMap` durability docs.
        let mut flushed_ns = std::collections::HashSet::new();
        for &(ns_id, _) in &fields {
            if flushed_ns.insert(ns_id)
                && let Some(store) = stores.get(&ns_id)
            {
                store.flush_rowmap(wal_tail)?;
            }
        }

        for &(ns_id, field_id) in &fields {
            if let Some(store) = stores.get(&ns_id) {
                let ns_index = store.namespace_index.read();
                if let Some(entry) = ns_index.get(field_id) {
                    let field_path = self.index_manager.field_path(ns_id, field_id);
                    // Flush under a read lock, then check waste cheaply. Only take
                    // the write lock (which serialises with index writers) when the
                    // bitmap store OR the keymap store has crossed the compaction
                    // threshold (the keymap accumulates dead space under
                    // distinct-value churn).
                    let (over_threshold, stats) = {
                        let idx = entry.index.read();
                        idx.flush(&field_path).map_err(KVError::Io)?;
                        let stats = idx.blob_stats();
                        let over = stats.bitmap_waste_ratio >= waste_threshold || stats.keymap_waste_ratio >= waste_threshold;
                        (over, stats)
                    };
                    // Guardrail: low-cardinality fields suffer append-only write
                    // amplification — a value rewritten per document leaves a stale
                    // bitmap copy each time (see index/CLAUDE.md). The compaction
                    // below reclaims it, but warn when a field's bitmap blob has
                    // grown large with a small live footprint so operators can spot
                    // runaway growth between checkpoints.
                    const LARGE_BITMAP_LOGICAL_BYTES: u64 = 64 * 1024 * 1024;
                    if stats.bitmap_logical_bytes >= LARGE_BITMAP_LOGICAL_BYTES && stats.bitmap_waste_ratio >= 0.5 {
                        warn!(
                            "[IndexCheckpoint] ns={ns_id} field={field_id}: bitmap blob logical={} MiB live={} MiB \
                             waste={:.0}% across {} distinct value(s) — append-only write amplification (likely a \
                             low-cardinality, high-churn field); compaction will reclaim it now, but review the field's update rate",
                            stats.bitmap_logical_bytes / (1024 * 1024),
                            stats.bitmap_live_bytes / (1024 * 1024),
                            stats.bitmap_waste_ratio * 100.0,
                            stats.distinct_values,
                        );
                    }
                    if over_threshold {
                        let mut idx = entry.index.write();
                        if idx.maybe_compact(waste_threshold).map_err(KVError::Io)? {
                            idx.flush(&field_path).map_err(KVError::Io)?;
                            debug!("[IndexCheckpoint] compacted field-index stores ns={ns_id} field={field_id}");
                        }
                    }
                    active_fields.push((ns_id, field_id));
                }
            }
        }

        self.index_manager.checkpoint_fields(wal_tail, &active_fields)?;
        Ok(active_fields.len())
    }
}

// ── WalGcTarget impl ──────────────────────────────────────────────────

impl WalGcTarget for Database {
    fn is_closed(&self) -> bool {
        self.is_closed()
    }

    fn get_wal_gc_stats(&self) -> (u64, u64) {
        self.get_wal_gc_stats()
    }

    fn has_deletable_wal_segments(&self) -> bool {
        self.has_deletable_wal_segments()
    }

    fn garbage_collect_wal(&self) -> Result<(u64, u64)> {
        self.garbage_collect_wal()
    }
}

// ── LsmCompactionTarget impl ──────────────────────────────────────────

impl LsmCompactionTarget for Database {
    fn is_closed(&self) -> bool {
        self.is_closed()
    }

    fn has_lsm_compaction_work(&self) -> bool {
        self.has_lsm_compaction_work()
    }

    fn compact_lsm(&self) -> Result<()> {
        self.compact_lsm()
    }
}

// ── ValueLogGcTarget impl ─────────────────────────────────────────────

impl ValueLogGcTarget for Database {
    fn is_closed(&self) -> bool {
        self.is_closed()
    }

    fn run_gc_if_needed(&self, waste_threshold: f64) {
        let stores = self.stores.read();
        info!(
            "[GCWorker] tick — checking {} namespace(s) against {:.2}% waste threshold",
            stores.len(),
            waste_threshold
        );
        for (ns_id, kv_store) in stores.iter() {
            let waste_ratio = kv_store.get_waste_ratio();
            if waste_ratio < waste_threshold {
                debug!("[GCWorker] ns_id={} waste {:.2}% below threshold, skipping", ns_id, waste_ratio);
                continue;
            }
            let (garbage_bytes, written_bytes) = kv_store.waste_bytes();
            info!(
                "[GCWorker] ns_id={} waste {:.2}% ({} garbage / {} written bytes; waste = garbage / (live + garbage)) \
                 exceeds threshold {:.2}%, starting GC",
                ns_id, waste_ratio, garbage_bytes, written_bytes, waste_threshold
            );
            let start = std::time::Instant::now();
            match kv_store.garbage_collect_with_threshold(waste_threshold) {
                Ok(stats) => info!(
                    "[GCWorker] ns_id={} GC complete in {:?} — reclaimed {} bytes, live {} bytes, \
                     total reclaimed {} bytes across {} run(s)",
                    ns_id,
                    start.elapsed(),
                    stats.bytes_reclaimed,
                    stats.bytes_live,
                    stats.total_bytes_reclaimed,
                    stats.gc_run_count,
                ),
                Err(e) => error!("[GCWorker] ns_id={} GC failed: {:?}", ns_id, e),
            }
        }
    }
}

// ── TtlTarget impl ─────────────────────────────────────────────────────

impl TtlTarget for Database {
    fn is_closed(&self) -> bool {
        self.is_closed()
    }

    fn run_ttl_pass(&self) {
        // Snapshot the durable TTL config so the (brief) scan/delete work doesn't
        // hold the registry lock; new registrations take effect on the next pass.
        let ttl_configs = self.registry.read().ttl_configs();
        info!("[TtlWorker] tick — scanning {} TTL-enabled namespace(s)", ttl_configs.len());
        for (ns_id, (ttl, max_deletes)) in ttl_configs {
            // The namespace may have been dropped between snapshot and lookup.
            let Some(store) = self.stores.read().get(&ns_id).cloned() else {
                continue;
            };
            match store.expire_records(ttl, max_deletes) {
                Ok(deleted) => {
                    if deleted > 0 {
                        info!("[TtlWorker] ns_id={} expired {} record(s)", ns_id, deleted);
                    } else {
                        debug!("[TtlWorker] ns_id={} no records expired", ns_id);
                    }
                }
                Err(e) => warn!("[TtlWorker] ns_id={} TTL cleanup failed: {:?}", ns_id, e),
            }
        }
    }
}

// ── AsyncDatabase ──────────────────────────────────────────────────────

/// Async wrapper around Database for use with Tokio.
/// Provides the same multi-namespace API with async/await support.
#[derive(Clone)]
pub struct AsyncDatabase {
    inner: Arc<Database>,
}

impl AsyncDatabase {
    pub fn new(db: Database) -> Self {
        Self { inner: Arc::new(db) }
    }

    /// Open a database with background workers enabled
    pub async fn open_with_workers(db_path: &Path, config: DbConfig) -> Result<Self> {
        let db_path = db_path.to_path_buf();
        let cfg = config.clone();
        let db = tokio::task::spawn_blocking(move || Database::open(&db_path, cfg))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))??;
        let async_db = AsyncDatabase::new(db);

        // Enable background workers
        let scheduled = config.scheduled_task_config;
        async_db.enable_wal_gc_worker(scheduled.wal_gc_interval).await?;
        async_db.enable_index_checkpoint_worker(DEFAULT_CHECKPOINT_INTERVAL).await?;

        Ok(async_db)
    }

    // ── Core data operations (default namespace) ───────────────────────

    pub async fn put(&self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.put(&key, &value))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn get(&self, key: Vec<u8>) -> Result<Option<Vec<u8>>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.get(&key))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn delete(&self, key: Vec<u8>) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.delete(&key))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    // ── Namespace-aware data operations ────────────────────────────────

    pub async fn put_ns(&self, namespace_id: u32, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.put_ns(namespace_id, &key, &value))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn get_ns(&self, namespace_id: u32, key: Vec<u8>) -> Result<Option<Vec<u8>>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.get_ns(namespace_id, &key))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn delete_ns(&self, namespace_id: u32, key: Vec<u8>) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.delete_ns(namespace_id, &key))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    // ── Namespace management ───────────────────────────────────────────

    pub fn create_namespace(&self, name: &str) -> Result<u32> {
        self.inner.create_namespace(name)
    }

    pub fn list_namespaces(&self) -> Vec<(String, u32)> {
        self.inner.list_namespaces()
    }

    pub fn get_namespace_id(&self, name: &str) -> Option<u32> {
        self.inner.get_namespace_id(name)
    }

    pub fn namespace_exists(&self, name: &str) -> bool {
        self.inner.namespace_exists(name)
    }

    pub fn remove_namespace(&self, name: &str) -> Result<u32> {
        self.inner.remove_namespace(name)
    }

    /// Ensure the single global TTL worker is running. Idempotent — a no-op if
    /// it has already been started. The worker scans every TTL-registered
    /// namespace on each tick (driven by `ttl_cleanup_interval`).
    pub(crate) async fn ensure_ttl_worker(&self) {
        let mut slot = self.inner.ttl_worker.write().await;
        if slot.is_none() {
            let interval = self.inner.config.scheduled_task_config.ttl_cleanup_interval;
            let worker = TtlWorker::new(Arc::clone(&self.inner), interval);
            *slot = Some(Arc::new(worker));
            info!("[AsyncDatabase] Global TTL worker enabled (interval={}s)", interval.as_secs());
        }
    }

    /// Create a namespace with a TTL. The namespace is registered with the
    /// single global TTL worker, which expires its records older than `ttl`
    /// (capped at `max_deletes_per_run` per pass).
    pub async fn create_namespace_with_ttl(&self, name: &str, ttl: Duration, max_deletes_per_run: usize) -> Result<u32> {
        let ns_id = self.inner.create_namespace_with_ttl(name, Some(ttl))?;
        self.inner.registry.write().set_ttl_config(ns_id, ttl, max_deletes_per_run)?;
        self.ensure_ttl_worker().await;
        info!(
            "[AsyncDatabase] TTL registered for namespace '{}' (ttl={}s, max_deletes={})",
            name,
            ttl.as_secs(),
            max_deletes_per_run
        );
        Ok(ns_id)
    }

    /// Trigger an immediate TTL cleanup pass.
    ///
    /// `namespace_id` is accepted for API compatibility but the single global
    /// worker runs a full pass over every TTL-registered namespace; if that
    /// namespace is registered it will be expired as part of the pass.
    pub async fn trigger_ttl_cleanup(&self, _namespace_id: u32) -> Result<()> {
        if let Some(worker) = self.inner.ttl_worker.read().await.as_ref() {
            worker.trigger().map_err(|e| KVError::Io(std::io::Error::other(e.to_string())))?;
        }
        Ok(())
    }

    /// Stop expiring records for a namespace by removing its persisted TTL
    /// config. The worker task itself keeps running for other namespaces; the
    /// namespace's `store.ttl` metadata is left intact.
    pub async fn shutdown_ttl_worker(&self, namespace_id: u32) {
        if let Err(e) = self.inner.registry.write().remove_ttl_config(namespace_id) {
            warn!("[AsyncDatabase] Failed to remove TTL config for ns_id={}: {:?}", namespace_id, e);
        }
    }

    // ── WAL GC worker ──────────────────────────────────────────────────

    pub async fn enable_wal_gc_worker(&self, check_interval: Duration) -> Result<()> {
        let worker = WalGcWorker::new(self.inner.clone(), check_interval);
        *self.inner.wal_gc_worker.write().await = Some(Arc::new(worker));
        info!("[AsyncDatabase] WAL GC worker enabled with {}ms interval", check_interval.as_millis());
        Ok(())
    }

    pub async fn trigger_wal_gc_worker(&self) -> Result<()> {
        if let Some(worker) = self.inner.wal_gc_worker.read().await.as_ref() {
            worker.trigger_gc().map_err(|e| KVError::Io(std::io::Error::other(e.to_string())))?;
        }
        Ok(())
    }

    pub async fn shutdown_wal_gc_worker(&self) {
        if let Some(worker) = self.inner.wal_gc_worker.write().await.take() {
            worker.shutdown().await;
        }
    }

    pub async fn is_wal_gc_worker_enabled(&self) -> bool {
        self.inner.wal_gc_worker.read().await.is_some()
    }

    // ── Index checkpoint worker ────────────────────────────────────────

    /// Start the index checkpoint worker with the given interval.
    ///
    /// The worker periodically serialises in-memory field indices to
    /// `{db_path}/index/{namespace_id}/{field_id}/` so that crash recovery
    /// only needs to replay a bounded WAL tail.
    pub async fn enable_index_checkpoint_worker(&self, interval: Duration) -> Result<()> {
        let worker = IndexCheckpointWorker::new(Arc::clone(&self.inner), interval);
        *self.inner.index_checkpoint_worker.write().await = Some(Arc::new(worker));
        info!("[AsyncDatabase] Index checkpoint worker enabled with {}s interval", interval.as_secs());
        Ok(())
    }

    /// Trigger an immediate index checkpoint outside of the normal schedule.
    pub async fn trigger_index_checkpoint(&self) -> Result<()> {
        if let Some(worker) = self.inner.index_checkpoint_worker.read().await.as_ref() {
            worker.trigger().map_err(|e| KVError::Io(std::io::Error::other(e.to_string())))?;
        }
        Ok(())
    }

    /// Shut down the index checkpoint worker gracefully.
    pub async fn shutdown_index_checkpoint_worker(&self) {
        if let Some(worker) = self.inner.index_checkpoint_worker.write().await.take() {
            worker.shutdown().await;
        }
    }

    // ── LSM compaction worker ──────────────────────────────────────────

    /// Start the LSM compaction worker.
    ///
    /// Wires all existing KVStores' compaction triggers to the worker so that
    /// memtable flushes immediately schedule a compaction check.  Namespaces
    /// opened after this call are wired up at creation time.
    pub async fn enable_lsm_compaction_worker(&self, interval: Duration) -> Result<()> {
        let worker = LsmCompactionWorker::new(Arc::clone(&self.inner), interval);
        let sender = worker.sender();
        // Wire existing namespaces — drop the read-guard before the async write below.
        {
            let stores = self.inner.stores.read();
            for kv_store in stores.values() {
                kv_store.set_compaction_trigger(sender.clone());
            }
        }
        *self.inner.lsm_compaction_sender.write() = Some(sender);
        *self.inner.lsm_compaction_worker.write().await = Some(Arc::new(worker));
        info!("[AsyncDatabase] LSM compaction worker enabled with {}ms interval", interval.as_millis());
        Ok(())
    }

    pub async fn shutdown_lsm_compaction_worker(&self) {
        if let Some(worker) = self.inner.lsm_compaction_worker.write().await.take() {
            worker.shutdown().await;
        }
        *self.inner.lsm_compaction_sender.write() = None;
    }

    // ── Value-log GC worker ────────────────────────────────────────────

    /// Start the value-log GC worker.  Runs GC on every namespace whose waste
    /// ratio exceeds `waste_threshold` on each `interval` tick.
    pub async fn enable_value_log_gc_worker(&self, interval: Duration, waste_threshold: f64) -> Result<()> {
        let worker = GCWorker::new(Arc::clone(&self.inner), interval, waste_threshold);
        *self.inner.value_log_gc_worker.write().await = Some(Arc::new(worker));
        info!(
            "[AsyncDatabase] Value-log GC worker enabled with {}s interval, threshold {:.1}%",
            interval.as_secs(),
            waste_threshold
        );
        Ok(())
    }

    pub async fn shutdown_value_log_gc_worker(&self) {
        if let Some(worker) = self.inner.value_log_gc_worker.write().await.take() {
            worker.shutdown().await;
        }
    }

    /// Register an indexed field for a namespace.  Returns the assigned `FieldId`.
    pub fn register_index_field(&self, namespace_id: u32, field_name: &str, value_type: IndexValueType) -> Result<FieldId> {
        self.inner.register_index_field(namespace_id, field_name, value_type)
    }

    /// Return all indexed fields registered for a namespace, sorted by `FieldId`.
    pub fn list_index_fields(&self, namespace_id: u32) -> Vec<FieldMeta> {
        self.inner.list_index_fields(namespace_id)
    }

    /// Return the number of distinct indexed values for a field.
    ///
    /// Returns `None` when the field is not active.
    pub fn field_index_distinct_count(&self, namespace_id: u32, field_id: FieldId) -> Option<usize> {
        self.inner.field_index_distinct_count(namespace_id, field_id)
    }

    /// Register a custom row-ID function (and optionally its inverse) for a namespace.
    ///
    /// See [`Database::set_row_id_fn`] for full documentation.
    pub fn set_row_id_fn(
        &self,
        namespace_id: u32,
        row_id_fn: crate::db::namespace_index::RowIdFn,
        row_to_key_fn: Option<crate::db::namespace_index::RowToKeyFn>,
    ) -> Result<()> {
        self.inner.set_row_id_fn(namespace_id, row_id_fn, row_to_key_fn)
    }

    /// Wire up a live extractor for a registered field and load its snapshot.
    ///
    /// See [`Database::activate_field_index`] for full documentation.
    pub fn activate_field_index(&self, namespace_id: u32, field_id: FieldId, value_type: IndexValueType, extractor: ExtractorFn) -> Result<()> {
        self.inner.activate_field_index(namespace_id, field_id, value_type, extractor)
    }

    /// Evaluate a query string and return matching document keys.
    ///
    /// See [`Database::query_keys`] for full documentation.
    pub async fn query_keys(&self, namespace_id: u32, query_str: String) -> Result<Vec<Vec<u8>>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.query_keys(namespace_id, &query_str))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    // ── Shutdown ───────────────────────────────────────────────────────

    #[allow(dead_code)]
    pub(crate) async fn shutdown(&self) -> Result<()> {
        // Shutdown the global TTL worker. The TTL config stays persisted in the
        // registry so it is restored on the next open.
        if let Some(worker) = self.inner.ttl_worker.write().await.take() {
            info!("[AsyncDatabase] Shutting down TTL worker...");
            worker.shutdown().await;
        }

        // Shutdown WAL GC worker
        if self.is_wal_gc_worker_enabled().await {
            info!("[AsyncDatabase] Shutting down WAL GC worker...");
            self.shutdown_wal_gc_worker().await;
        }

        // Flush index state to disk before stopping the checkpoint worker so
        // that snapshot.idx files are always consistent with the last write.
        if self.inner.index_checkpoint_worker.read().await.is_some() {
            info!("[AsyncDatabase] Running final index checkpoint before shutdown...");
            let db = self.inner.clone();
            if let Err(e) = tokio::task::spawn_blocking(move || db.run_index_checkpoint())
                .await
                .map_err(|e| KVError::Io(std::io::Error::other(e)))?
            {
                log::warn!("[AsyncDatabase] Final index checkpoint failed: {:?}", e);
            }
            info!("[AsyncDatabase] Shutting down index checkpoint worker...");
            self.shutdown_index_checkpoint_worker().await;
        }

        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || db.shutdown())
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    #[allow(dead_code)]
    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::config::{ScheduledTaskConfig, SyncConfig, ThresholdConfig};
    use crate::store::lsm::lsm_tree::LSMConfig;
    use tempfile::TempDir;

    fn create_db_config() -> DbConfig {
        let gc_interval = Duration::from_secs(5);
        let wal_gc_interval = Duration::from_secs(5);
        let lsm_compaction_interval = Duration::from_secs(5);

        let sync_config = SyncConfig::default();
        let threshold_config = ThresholdConfig::new(2.5);
        let scheduled_task_config = ScheduledTaskConfig::new(gc_interval, wal_gc_interval, lsm_compaction_interval);
        let lsm_config = LSMConfig::default();
        DbConfig::new(threshold_config, scheduled_task_config, sync_config, lsm_config)
    }

    #[test]
    fn test_database_basic_operations() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        db.put(b"key1", b"value1").unwrap();
        db.put(b"key2", b"value2").unwrap();

        assert_eq!(db.get(b"key1").unwrap(), Some(b"value1".to_vec()));
        assert_eq!(db.get(b"key2").unwrap(), Some(b"value2".to_vec()));

        db.delete(b"key2").unwrap();
        assert_eq!(db.get(b"key2").unwrap(), None);

        db.shutdown().unwrap();
    }

    #[test]
    fn test_database_multiple_namespaces() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        // Create additional namespace — ID 0 = default, 1 = system, so users gets 2.
        let users_id = db.create_namespace("users").unwrap();
        assert_eq!(users_id, 2);

        // Write to default namespace
        db.put(b"global_key", b"global_value").unwrap();

        // Write to users namespace
        db.put_ns(users_id, b"user:1", b"alice").unwrap();
        db.put_ns(users_id, b"user:2", b"bob").unwrap();

        // Read from default — should not see users data
        assert_eq!(db.get(b"global_key").unwrap(), Some(b"global_value".to_vec()));
        assert_eq!(db.get(b"user:1").unwrap(), None);

        // Read from users namespace
        assert_eq!(db.get_ns(users_id, b"user:1").unwrap(), Some(b"alice".to_vec()));
        assert_eq!(db.get_ns(users_id, b"user:2").unwrap(), Some(b"bob".to_vec()));
        assert_eq!(db.get_ns(users_id, b"global_key").unwrap(), None);

        db.shutdown().unwrap();
    }

    #[test]
    fn test_database_namespace_listing() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        db.create_namespace("users").unwrap();
        db.create_namespace("orders").unwrap();

        let mut namespaces = db.list_namespaces();
        namespaces.sort_by_key(|(_, id)| *id);
        // default(0), system(1), users(2), orders(3)
        assert_eq!(namespaces.len(), 4);
        assert_eq!(namespaces[0], ("default".to_string(), 0));
        assert_eq!(namespaces[1], ("system".to_string(), 1));
        assert_eq!(namespaces[2], ("users".to_string(), 2));
        assert_eq!(namespaces[3], ("orders".to_string(), 3));

        db.shutdown().unwrap();
    }

    #[test]
    fn test_database_namespace_persistence() {
        let dir = TempDir::new().unwrap();

        // Create database with namespaces and data
        {
            let db = Database::open(dir.path(), create_db_config()).unwrap();
            let users_id = db.create_namespace("users").unwrap();
            db.put(b"default_key", b"default_val").unwrap();
            db.put_ns(users_id, b"user_key", b"user_val").unwrap();
            db.shutdown().unwrap();
        }

        // Reopen and verify
        {
            let db = Database::open(dir.path(), create_db_config()).unwrap();
            assert!(db.namespace_exists("users"));
            let users_id = db.get_namespace_id("users").unwrap();

            assert_eq!(db.get(b"default_key").unwrap(), Some(b"default_val".to_vec()));
            assert_eq!(db.get_ns(users_id, b"user_key").unwrap(), Some(b"user_val".to_vec()));

            db.shutdown().unwrap();
        }
    }

    #[test]
    fn test_database_remove_namespace() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        let users_id = db.create_namespace("users").unwrap();
        db.put_ns(users_id, b"key", b"val").unwrap();

        db.remove_namespace("users").unwrap();
        assert!(!db.namespace_exists("users"));
        assert!(db.get_ns(users_id, b"key").is_err());

        // Cannot remove default
        assert!(db.remove_namespace("default").is_err());

        db.shutdown().unwrap();
    }

    #[test]
    fn test_wal_gc_eligible_after_namespace_drop() {
        // Regression test: WAL entries for a dropped namespace must be marked
        // persisted on drop so that WAL GC can reclaim the segments they occupy.
        // Previously, removing a namespace left its WAL entries as Inserted forever,
        // blocking GC for any segment that contained those entries.
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        let ns_id = db.create_namespace("temp").unwrap();
        db.put_ns(ns_id, b"k1", b"v1").unwrap();
        db.put_ns(ns_id, b"k2", b"v2").unwrap();

        let (total_before, persisted_before) = db.get_wal_gc_stats();
        assert!(total_before >= 2, "expected at least 2 WAL entries");

        db.remove_namespace("temp").unwrap();

        let (total_after, persisted_after) = db.get_wal_gc_stats();
        // After dropping the namespace, the persisted count must have increased
        // to cover the entries we just wrote, so GC is not blocked on them.
        assert!(
            persisted_after > persisted_before,
            "persisted count should increase after namespace drop (before={}, after={})",
            persisted_before,
            persisted_after,
        );
        assert_eq!(total_after, total_before, "total entry count should be unchanged");

        db.shutdown().unwrap();
    }

    #[test]
    fn test_activate_type_mismatch_rejected() {
        use crate::db::namespace_index::ExtractorFn;
        use index::IndexValueType;
        use std::sync::Arc;

        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        let ns = DEFAULT_NAMESPACE_ID;
        // Register as Int
        let field_id = db.register_index_field(ns, "age", IndexValueType::Int).unwrap();

        // Activating with the correct type succeeds
        let extractor: ExtractorFn = Arc::new(|_| None);
        assert!(db.activate_field_index(ns, field_id, IndexValueType::Int, Arc::clone(&extractor)).is_ok());

        // Activating with a different type must fail
        let err = db.activate_field_index(ns, field_id, IndexValueType::Str, extractor).unwrap_err();
        assert!(err.to_string().contains("Type mismatch"), "unexpected error: {}", err);
    }

    #[test]
    fn test_activate_unknown_field_id_rejected() {
        use crate::db::namespace_index::ExtractorFn;
        use index::IndexValueType;
        use std::sync::Arc;

        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        let extractor: ExtractorFn = Arc::new(|_| None);
        // field_id 99 was never registered
        let err = db
            .activate_field_index(DEFAULT_NAMESPACE_ID, 99, IndexValueType::Int, extractor)
            .unwrap_err();
        assert!(err.to_string().contains("not registered"), "unexpected error: {}", err);
    }

    #[test]
    fn test_duplicate_register_field_idempotent() {
        use index::IndexValueType;

        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        let id1 = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str).unwrap();
        // Same name + same type: idempotent, returns existing id
        let id2 = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str).unwrap();
        assert_eq!(id1, id2);
        // Same name, different type: error
        let err = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Int).unwrap_err();
        assert!(err.to_string().contains("already registered"), "unexpected error: {}", err);
    }

    #[test]
    fn test_schema_survives_restart_and_index_activates_without_re_register() {
        use crate::db::namespace_index::ExtractorFn;
        use index::{IndexValue, IndexValueType};
        use std::sync::Arc;

        let dir = TempDir::new().unwrap();

        // ── First open: register fields and write some data ──────────────
        let status_field_id;
        let age_field_id;
        {
            let db = Database::open(dir.path(), create_db_config()).unwrap();

            status_field_id = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str).unwrap();
            age_field_id = db.register_index_field(DEFAULT_NAMESPACE_ID, "age", IndexValueType::Int).unwrap();

            // Activate both indices with extractors
            let status_extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
                let s = std::str::from_utf8(bytes).ok()?;
                let v: serde_json::Value = serde_json::from_str(s).ok()?;
                Some(IndexValue::Str(v["status"].as_str()?.to_string()))
            });
            let age_extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
                let s = std::str::from_utf8(bytes).ok()?;
                let v: serde_json::Value = serde_json::from_str(s).ok()?;
                Some(IndexValue::Int(v["age"].as_i64()?))
            });
            db.activate_field_index(DEFAULT_NAMESPACE_ID, status_field_id, IndexValueType::Str, status_extractor)
                .unwrap();
            db.activate_field_index(DEFAULT_NAMESPACE_ID, age_field_id, IndexValueType::Int, age_extractor)
                .unwrap();

            db.put(b"user:1", br#"{"status":"active","age":30}"#).unwrap();
            db.put(b"user:2", br#"{"status":"inactive","age":25}"#).unwrap();
            db.put(b"user:3", br#"{"status":"active","age":40}"#).unwrap();

            db.shutdown().unwrap();
        }

        // ── Second open: do NOT call register_index_field ────────────────
        // Schema must be loaded from config.json automatically.
        {
            let db = Database::open(dir.path(), create_db_config()).unwrap();

            // Confirm schema is present without re-registering
            let fields = db.list_index_fields(DEFAULT_NAMESPACE_ID);
            assert_eq!(fields.len(), 2);
            assert!(fields.iter().any(|f| f.field_name == "status" && f.field_type == IndexValueType::Str));
            assert!(fields.iter().any(|f| f.field_name == "age" && f.field_type == IndexValueType::Int));

            // Activate indices with extractors (closures can't be persisted — caller always supplies these)
            let status_extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
                let s = std::str::from_utf8(bytes).ok()?;
                let v: serde_json::Value = serde_json::from_str(s).ok()?;
                Some(IndexValue::Str(v["status"].as_str()?.to_string()))
            });
            let age_extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
                let s = std::str::from_utf8(bytes).ok()?;
                let v: serde_json::Value = serde_json::from_str(s).ok()?;
                Some(IndexValue::Int(v["age"].as_i64()?))
            });
            // These must succeed using the field IDs recovered from config.json
            db.activate_field_index(DEFAULT_NAMESPACE_ID, status_field_id, IndexValueType::Str, status_extractor)
                .unwrap();
            db.activate_field_index(DEFAULT_NAMESPACE_ID, age_field_id, IndexValueType::Int, age_extractor)
                .unwrap();

            // Queries must return correct results from the warmed index
            let mut active_keys = db.query_keys(DEFAULT_NAMESPACE_ID, "status = \"active\"").unwrap();
            active_keys.sort();
            assert_eq!(active_keys, vec![b"user:1".to_vec(), b"user:3".to_vec()]);

            let inactive_keys = db.query_keys(DEFAULT_NAMESPACE_ID, "status = \"inactive\"").unwrap();
            assert_eq!(inactive_keys, vec![b"user:2".to_vec()]);

            db.shutdown().unwrap();
        }
    }

    /// Verify that `activate_field_index` replays the WAL tail into the index
    /// on reopen, covering any writes that happened after the last checkpoint.
    ///
    /// Sequence:
    ///   1. Open DB, activate index, write batch A.
    ///   2. Checkpoint (flush BlobStore + keymap mmap store + checkpoint file).
    ///   3. Write batch B (live index updated, checkpoint is now stale).
    ///   4. Drop DB without shutdown — simulates a crash.
    ///   5. Reopen DB, activate index (WAL replay runs for batch B).
    ///   6. Query must return all of batch A and batch B.
    #[test]
    fn test_activate_field_index_replays_wal_after_crash() {
        use crate::db::namespace_index::ExtractorFn;
        use index::{IndexValue, IndexValueType};
        use std::sync::Arc;

        let dir = TempDir::new().unwrap();
        let ns = DEFAULT_NAMESPACE_ID;

        let make_extractor = || -> ExtractorFn {
            Arc::new(|bytes: &[u8]| {
                let s = std::str::from_utf8(bytes).ok()?;
                let v: serde_json::Value = serde_json::from_str(s).ok()?;
                Some(IndexValue::Str(v["status"].as_str()?.to_string()))
            })
        };

        let field_id;
        {
            let db = Database::open(dir.path(), create_db_config()).unwrap();
            field_id = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();
            db.activate_field_index(ns, field_id, IndexValueType::Str, make_extractor()).unwrap();

            // Batch A — will be on disk after checkpoint.
            db.put(b"user:1", br#"{"status":"active"}"#).unwrap();
            db.put(b"user:2", br#"{"status":"inactive"}"#).unwrap();

            // Checkpoint: flush BlobStore + keymap mmap store + checkpoint marker.
            db.run_index_checkpoint().unwrap();

            // Batch B — live index only; checkpoint is now stale.
            db.put(b"user:3", br#"{"status":"active"}"#).unwrap();
            db.put(b"user:4", br#"{"status":"inactive"}"#).unwrap();

            // Drop without shutdown — WAL has batch B but checkpoint does not.
        }

        {
            let db = Database::open(dir.path(), create_db_config()).unwrap();
            // activate_field_index must replay batch B from the WAL tail.
            db.activate_field_index(ns, field_id, IndexValueType::Str, make_extractor()).unwrap();

            let mut active_keys = db.query_keys(ns, "status = \"active\"").unwrap();
            active_keys.sort();
            assert_eq!(
                active_keys,
                vec![b"user:1".to_vec(), b"user:3".to_vec()],
                "WAL replay must include batch-B writes",
            );

            let mut inactive_keys = db.query_keys(ns, "status = \"inactive\"").unwrap();
            inactive_keys.sort();
            assert_eq!(inactive_keys, vec![b"user:2".to_vec(), b"user:4".to_vec()],);

            db.shutdown().unwrap();
        }
    }

    /// Item 13 regression: replaying an **update** (same key, changed field
    /// value) must reconcile via the targeted O(1) path — during replay
    /// `lsm.get` returns the replay-so-far value, so the second put moves the
    /// row off the old value. A bug here would leave the key matching BOTH the
    /// old and new value after recovery.
    #[test]
    fn test_field_index_replays_updates_after_crash() {
        use crate::db::namespace_index::ExtractorFn;
        use index::{IndexValue, IndexValueType};
        use std::sync::Arc;

        let dir = TempDir::new().unwrap();
        let ns = DEFAULT_NAMESPACE_ID;
        let make_extractor = || -> ExtractorFn {
            Arc::new(|bytes: &[u8]| {
                let s = std::str::from_utf8(bytes).ok()?;
                let v: serde_json::Value = serde_json::from_str(s).ok()?;
                Some(IndexValue::Str(v["status"].as_str()?.to_string()))
            })
        };

        let field_id;
        {
            let db = Database::open(dir.path(), create_db_config()).unwrap();
            field_id = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();
            db.activate_field_index(ns, field_id, IndexValueType::Str, make_extractor()).unwrap();

            // Insert, then UPDATE the same key to a different value — all in the
            // WAL, no checkpoint, so recovery must replay both writes in order.
            db.put(b"u:1", br#"{"status":"active"}"#).unwrap();
            db.put(b"u:1", br#"{"status":"archived"}"#).unwrap(); // update
            db.put(b"u:2", br#"{"status":"active"}"#).unwrap();
            // Drop without shutdown — index lives only in the WAL.
        }

        {
            let db = Database::open(dir.path(), create_db_config()).unwrap();
            db.activate_field_index(ns, field_id, IndexValueType::Str, make_extractor()).unwrap();

            // u:1 must match ONLY its latest value after replay, not the old one.
            assert_eq!(
                db.query_keys(ns, "status = \"archived\"").unwrap(),
                vec![b"u:1".to_vec()],
                "replayed update must land u:1 under its new value"
            );
            assert_eq!(
                db.query_keys(ns, "status = \"active\"").unwrap(),
                vec![b"u:2".to_vec()],
                "replayed update must remove u:1 from its old value (no stale bucket)"
            );
            db.shutdown().unwrap();
        }
    }

    /// Item 13: a put that adds/removes the indexed field (absent↔present) must
    /// update the index correctly via the targeted path — the row joins the new
    /// value's bucket and leaves whatever it was in (including "nothing").
    #[test]
    fn test_field_index_update_field_appears_and_disappears() {
        use crate::db::namespace_index::ExtractorFn;
        use index::{IndexValue, IndexValueType};
        use std::sync::Arc;

        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = DEFAULT_NAMESPACE_ID;
        let field_id = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();
        let extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Str(v["status"].as_str()?.to_string()))
        });
        db.activate_field_index(ns, field_id, IndexValueType::Str, extractor).unwrap();

        // Field absent → not indexed.
        db.put(b"d:1", br#"{"other":1}"#).unwrap();
        assert!(db.query_keys(ns, "status = \"active\"").unwrap().is_empty());

        // Field appears (None → Some): row joins the "active" bucket.
        db.put(b"d:1", br#"{"status":"active"}"#).unwrap();
        assert_eq!(db.query_keys(ns, "status = \"active\"").unwrap(), vec![b"d:1".to_vec()]);

        // Field disappears (Some → None): row must leave the bucket.
        db.put(b"d:1", br#"{"other":2}"#).unwrap();
        assert!(
            db.query_keys(ns, "status = \"active\"").unwrap().is_empty(),
            "row must leave its bucket when the indexed field is removed"
        );

        db.shutdown().unwrap();
    }

    #[test]
    fn test_ops_metrics_counters_move() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        db.put(b"k1", b"v1").unwrap();
        db.put(b"k2", b"v2").unwrap();
        assert_eq!(db.get(b"k1").unwrap(), Some(b"v1".to_vec())); // hit
        assert_eq!(db.get(b"missing").unwrap(), None); // miss
        db.delete(b"k2").unwrap();

        let m = db.metrics_snapshot();
        assert_eq!(m.puts, 2, "two WAL-backed puts");
        assert_eq!(m.deletes, 1, "one delete");
        assert_eq!(m.reads, 2, "two user reads");
        assert_eq!(m.read_hits, 1, "one hit");
        assert_eq!(m.read_misses, 1, "one miss");
        // Every WAL-backed write fsyncs once (2 puts + 1 delete).
        assert_eq!(m.wal_fsyncs, 3);
        assert!(m.wal_bytes_appended > 0);
        // Reads go through the LSM point-lookup path.
        assert!(m.lookups >= 2, "lookups should cover the user reads, got {}", m.lookups);

        db.shutdown().unwrap();
    }

    #[test]
    fn test_value_log_physical_and_page_stats() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        for i in 0..50u32 {
            db.put(format!("k{i}").as_bytes(), &vec![b'v'; 256]).unwrap();
        }

        // Physical stats: present for the default namespace, with real on-disk bytes.
        let physical = db.value_log_physical_stats();
        let (_, shards) = physical.iter().find(|(ns, _)| ns == "default").expect("default ns present");
        assert!(!shards.is_empty(), "expected value-log shards");
        let total_physical: u64 = shards.iter().map(|s| s.physical_bytes).sum();
        assert!(total_physical > 0, "physical bytes should be non-zero after writes");

        // Page stats: per-bucket page breakdown, with live records counted.
        let pages = db.value_log_page_stats("default").unwrap();
        assert_eq!(pages.len(), db.config.num_buckets as usize, "one entry per bucket");
        let total_records: u32 = pages.iter().flat_map(|(_, ps)| ps.iter()).map(|p| p.total_records).sum();
        assert!(total_records >= 50, "every written record should appear in a page, got {}", total_records);

        // Unknown namespace errors rather than panicking.
        assert!(db.value_log_page_stats("nope").is_err());

        db.shutdown().unwrap();
    }

    #[test]
    fn test_database_wal_gc() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        db.put(b"key", b"value").unwrap();

        let (total, _persisted) = db.get_wal_gc_stats();
        assert!(total >= 1);

        db.shutdown().unwrap();
    }

    #[tokio::test]
    async fn test_async_database_basic() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let async_db = AsyncDatabase::new(db);

        async_db.put(b"key1".to_vec(), b"value1".to_vec()).await.unwrap();
        let val = async_db.get(b"key1".to_vec()).await.unwrap();
        assert_eq!(val, Some(b"value1".to_vec()));

        async_db.delete(b"key1".to_vec()).await.unwrap();
        let val = async_db.get(b"key1".to_vec()).await.unwrap();
        assert_eq!(val, None);

        async_db.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_async_database_wal_gc_worker() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let async_db = AsyncDatabase::new(db);

        // Enable WAL GC worker
        async_db.enable_wal_gc_worker(Duration::from_secs(1)).await.unwrap();
        assert!(async_db.is_wal_gc_worker_enabled().await);

        // Write some data so the WAL has entries
        async_db.put(b"key1".to_vec(), b"value1".to_vec()).await.unwrap();
        async_db.put(b"key2".to_vec(), b"value2".to_vec()).await.unwrap();

        // Trigger WAL GC manually
        async_db.trigger_wal_gc_worker().await.unwrap();

        // Give it a moment to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Shutdown should cleanly stop the worker
        async_db.shutdown().await.unwrap();
        assert!(!async_db.is_wal_gc_worker_enabled().await);
    }

    #[tokio::test]
    async fn test_async_database_open_with_workers() {
        let dir = TempDir::new().unwrap();
        let dir_path = dir.path().to_path_buf();

        let async_db = AsyncDatabase::open_with_workers(&dir_path, create_db_config()).await.unwrap();
        assert!(async_db.is_wal_gc_worker_enabled().await);

        async_db.put(b"k".to_vec(), b"v".to_vec()).await.unwrap();
        assert_eq!(async_db.get(b"k".to_vec()).await.unwrap(), Some(b"v".to_vec()));

        async_db.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_async_database_multi_namespace() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let async_db = AsyncDatabase::new(db);

        let users_id = async_db.create_namespace("users").unwrap();

        async_db.put(b"global".to_vec(), b"g_val".to_vec()).await.unwrap();
        async_db.put_ns(users_id, b"user:1".to_vec(), b"alice".to_vec()).await.unwrap();

        // Cross-namespace isolation
        assert_eq!(async_db.get(b"user:1".to_vec()).await.unwrap(), None);
        assert_eq!(async_db.get_ns(users_id, b"user:1".to_vec()).await.unwrap(), Some(b"alice".to_vec()));
        assert_eq!(async_db.get_ns(users_id, b"global".to_vec()).await.unwrap(), None);

        async_db.shutdown().await.unwrap();
    }

    // ── delete_ns tests ─────────────────────────────────────────────────

    #[test]
    fn test_delete_ns_removes_key_from_namespace() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        let ns = db.create_namespace("items").unwrap();

        db.put_ns(ns, b"k1", b"v1").unwrap();
        db.put_ns(ns, b"k2", b"v2").unwrap();
        assert_eq!(db.get_ns(ns, b"k1").unwrap(), Some(b"v1".to_vec()));

        db.delete_ns(ns, b"k1").unwrap();
        assert_eq!(db.get_ns(ns, b"k1").unwrap(), None);
        // Sibling key is untouched
        assert_eq!(db.get_ns(ns, b"k2").unwrap(), Some(b"v2".to_vec()));

        db.shutdown().unwrap();
    }

    #[test]
    fn test_delete_ns_does_not_affect_other_namespaces() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        let ns_a = db.create_namespace("ns_a").unwrap();
        let ns_b = db.create_namespace("ns_b").unwrap();

        // Same key in three namespaces
        db.put(b"shared", b"default_val").unwrap();
        db.put_ns(ns_a, b"shared", b"a_val").unwrap();
        db.put_ns(ns_b, b"shared", b"b_val").unwrap();

        // Delete only from ns_a
        db.delete_ns(ns_a, b"shared").unwrap();

        assert_eq!(db.get_ns(ns_a, b"shared").unwrap(), None);
        assert_eq!(db.get(b"shared").unwrap(), Some(b"default_val".to_vec()));
        assert_eq!(db.get_ns(ns_b, b"shared").unwrap(), Some(b"b_val".to_vec()),);

        db.shutdown().unwrap();
    }

    #[test]
    fn test_delete_ns_nonexistent_key_is_ok() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        let ns = db.create_namespace("empty").unwrap();

        // Deleting a key that was never written should succeed silently.
        db.delete_ns(ns, b"ghost").unwrap();
        assert_eq!(db.get_ns(ns, b"ghost").unwrap(), None);

        db.shutdown().unwrap();
    }

    #[test]
    fn test_delete_ns_advances_wal() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        let ns = db.create_namespace("wal_check").unwrap();
        db.put_ns(ns, b"k", b"v").unwrap();

        let tail_before = db.wal_metadata().tail;
        db.delete_ns(ns, b"k").unwrap();
        let tail_after = db.wal_metadata().tail;

        assert!(
            tail_after > tail_before,
            "WAL tail must advance after delete_ns (before={}, after={})",
            tail_before,
            tail_after,
        );

        db.shutdown().unwrap();
    }

    #[test]
    fn test_delete_ns_invalid_namespace_returns_error() {
        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();

        // Namespace 9999 was never created.
        let result = db.delete_ns(9999, b"k");
        assert!(result.is_err(), "delete_ns on unknown namespace should fail");

        db.shutdown().unwrap();
    }

    /// Item 1 regression: several single-op writes to one key, laid down in the
    /// WAL with sequences *out of physical order*, must resolve to the
    /// highest-sequence value. This directly exercises the sequence sort in
    /// recovery (a naive scan-order replay would pick the physically-last entry,
    /// which here has a lower sequence).
    #[test]
    fn test_recovery_applies_same_key_writes_in_sequence_order() {
        use crate::db::wal::{Wal, WalEntry, WalMetadata};

        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("wal.log");
        let wal_meta_path = dir.path().join("wal_metadata");
        {
            let wal = Wal::open(&wal_path).unwrap();
            let mut tail = 0u64;

            // Physical order: seq 5, seq 9, seq 7. Sequence order says seq 9 wins.
            for (seq, val) in [(5u64, b"v5".to_vec()), (9, b"v9".to_vec()), (7, b"v7".to_vec())] {
                wal.append_entry(&WalEntry::new_upsert(b"k".to_vec(), val).with_sequence(seq), &mut tail, false)
                    .unwrap();
            }
            wal.sync().unwrap();

            let mut meta = WalMetadata::new();
            meta.tail = tail;
            meta.total_entries = 3;
            meta.segment_total_entries = vec![3];
            meta.segment_persisted_entries = vec![0];
            std::fs::write(&wal_meta_path, meta.to_file_bytes().unwrap()).unwrap();
        }

        let db = Database::open(dir.path(), create_db_config()).unwrap();
        assert_eq!(
            db.get(b"k").unwrap(),
            Some(b"v9".to_vec()),
            "recovery must apply same-key writes in sequence order (seq 9 is newest)"
        );
        db.shutdown().unwrap();
    }

    /// Critical regression: WAL entries fsynced *after* the last `wal_metadata`
    /// flush must not be lost. We craft the post-crash state — three durable WAL
    /// entries but a metadata file whose `tail`/counts cover only the first —
    /// and assert recovery reconstructs the true tail and replays all three
    /// (without the fix, recovery scans only up to the stale tail and the last
    /// two are silently dropped).
    #[test]
    fn test_recovery_reconstructs_wal_tail_past_stale_metadata() {
        use crate::db::wal::{Wal, WalEntry, WalMetadata};

        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("wal.log");
        let wal_meta_path = dir.path().join("wal_metadata");

        let stale_tail;
        {
            let wal = Wal::open(&wal_path).unwrap();
            let mut tail = 0u64;

            // Entry 1 — covered by the (earlier) metadata flush.
            wal.append_entry(&WalEntry::new_upsert(b"k1".to_vec(), b"v1".to_vec()).with_sequence(1), &mut tail, false)
                .unwrap();
            stale_tail = tail;

            // Entries 2 and 3 — fsynced after the flush; metadata never recorded them.
            wal.append_entry(&WalEntry::new_upsert(b"k2".to_vec(), b"v2".to_vec()).with_sequence(2), &mut tail, false)
                .unwrap();
            wal.append_entry(&WalEntry::new_upsert(b"k3".to_vec(), b"v3".to_vec()).with_sequence(3), &mut tail, false)
                .unwrap();
            wal.sync().unwrap();

            // Persist metadata as it stood at the earlier flush: it sees only entry 1.
            let mut meta = WalMetadata::new();
            meta.tail = stale_tail;
            meta.total_entries = 1;
            meta.segment_total_entries = vec![1];
            meta.segment_persisted_entries = vec![0];
            std::fs::write(&wal_meta_path, meta.to_file_bytes().unwrap()).unwrap();
        }

        let db = Database::open(dir.path(), create_db_config()).unwrap();
        assert_eq!(db.get(b"k1").unwrap(), Some(b"v1".to_vec()));
        assert_eq!(
            db.get(b"k2").unwrap(),
            Some(b"v2".to_vec()),
            "entry past the stale tail must be recovered"
        );
        assert_eq!(
            db.get(b"k3").unwrap(),
            Some(b"v3".to_vec()),
            "entry past the stale tail must be recovered"
        );
        db.shutdown().unwrap();
    }

    /// Coordinator-level companion to
    /// [`test_recovery_reconstructs_wal_tail_past_stale_metadata`]: rather than
    /// hand-crafting the WAL, it drives the **real** `Database::put` write path so
    /// it catches counter/short-circuit regressions in `Database::open` +
    /// `recover_from_wal` + `Wal::recover_tail` — e.g. a stale `total_entries`
    /// that wrongly trips the `persisted >= total` short-circuit, or a tail fold
    /// that miscounts `total`/`segment_total` and so under- or over-replays.
    ///
    /// Scenario (the "crash just after a metadata flush" window):
    ///   1. open Db, `put` key A,
    ///   2. flush WAL metadata so the durable snapshot covers only A,
    ///   3. `put` key B — fsynced into the WAL, but its in-memory metadata bump
    ///      never reaches disk (no further flush, `records_per_sync = 1000` so the
    ///      write path never auto-syncs/persists for two puts),
    ///   4. abandon the Db **without** shutdown via `mem::forget`, faithfully
    ///      simulating a crash: the in-memory metadata that knows about B and the
    ///      unflushed memtable holding A and B are both lost, leaving both keys
    ///      only in the WAL. (A plain `drop` would run `Database::drop`, which does
    ///      a *clean* flush — memtable→SSTable plus metadata — making B durable and
    ///      defeating the test.)
    ///   5. restore the stale (A-only) metadata on disk,
    ///   6. reopen and assert BOTH A and B are recovered.
    #[test]
    fn test_recovery_from_stale_metadata_via_real_write_path() {
        let dir = TempDir::new().unwrap();
        let wal_meta_path = dir.path().join("wal_metadata");

        let stale_metadata: Vec<u8>;
        {
            let db = Database::open(dir.path(), create_db_config()).unwrap();
            db.put(b"A", b"va").unwrap();

            // The on-disk metadata as it stood at the last flush before the crash:
            // its tail/total cover only A.
            db.flush_wal_metadata_internal().unwrap();
            stale_metadata = std::fs::read(&wal_meta_path).unwrap();

            // B is durable in the WAL (every put fsyncs it) but its metadata bump
            // lives only in memory — exactly the post-flush window.
            db.put(b"B", b"vb").unwrap();

            // Abandon without shutdown to simulate a crash: `mem::forget` skips
            // `Database::drop`, which would otherwise cleanly flush the memtable
            // and metadata (persisting B and defeating the test). The leaked
            // handles are released at process exit; everything durable (the WAL)
            // is already fsynced.
            std::mem::forget(db);
        }

        // Force the on-disk metadata back to the A-only snapshot, simulating the
        // crash landing after A's flush but before B's would have been recorded.
        // (Belt-and-suspenders: nothing should have rewritten it, but this makes
        // the staleness explicit regardless of write-path sync timing.)
        std::fs::write(&wal_meta_path, &stale_metadata).unwrap();

        // Reopen: recover_tail must notice the durable WAL extends past the stale
        // tail, fold B into total/segment counters, and recover_from_wal must
        // replay both A (lost from the memtable) and B (past the stale tail).
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        assert_eq!(
            db.get(b"A").unwrap(),
            Some(b"va".to_vec()),
            "A lived only in the WAL and must be replayed"
        );
        assert_eq!(
            db.get(b"B").unwrap(),
            Some(b"vb".to_vec()),
            "B was appended after the last metadata flush; recover_tail must fold it in and replay it"
        );
        db.shutdown().unwrap();
    }

    /// Item 3: `apply_with_retry` rides out transient apply failures and stops
    /// as soon as one attempt succeeds.
    #[test]
    fn test_apply_with_retry_succeeds_after_transient_failure() {
        use std::cell::Cell;

        let attempts = Cell::new(0);
        Database::apply_with_retry("put", 0, b"k", 1, || {
            attempts.set(attempts.get() + 1);
            // Fail on the first attempt, succeed on the second.
            if attempts.get() < 2 { Err(KVError::KeyNotFound) } else { Ok(()) }
        });
        assert_eq!(attempts.get(), 2, "should stop retrying once an attempt succeeds");
    }

    /// Item 3: `apply_with_retry` gives up after a bounded number of attempts
    /// (it never surfaces the error — the write is durable in the WAL).
    #[test]
    fn test_apply_with_retry_gives_up_after_max_attempts() {
        use std::cell::Cell;

        let attempts = Cell::new(0);
        Database::apply_with_retry("put", 0, b"k", 1, || {
            attempts.set(attempts.get() + 1);
            Err(KVError::KeyNotFound)
        });
        assert_eq!(
            attempts.get(),
            Database::APPLY_RETRY_ATTEMPTS,
            "should attempt exactly APPLY_RETRY_ATTEMPTS times before giving up"
        );
    }

    /// Deactivating a field index removes the in-memory bitmap so that any
    /// subsequent predicate query on that field returns an UnknownField error
    /// (the field is filtered out of the queryable schema map).
    #[test]
    fn test_deactivate_field_index_makes_field_unqueryable() {
        use crate::db::namespace_index::ExtractorFn;
        use index::{IndexValue, IndexValueType};
        use std::sync::Arc;

        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = DEFAULT_NAMESPACE_ID;

        let field_id = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();
        let extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Str(v["status"].as_str()?.to_string()))
        });
        db.activate_field_index(ns, field_id, IndexValueType::Str, extractor).unwrap();

        db.put(b"doc:1", br#"{"status":"active"}"#).unwrap();
        db.put(b"doc:2", br#"{"status":"inactive"}"#).unwrap();

        // Sanity: query works before deactivation.
        let keys = db.query_keys(ns, "status = \"active\"").unwrap();
        assert_eq!(keys, vec![b"doc:1".to_vec()]);

        db.deactivate_field_index(ns, field_id).unwrap();

        // Dropped fields are excluded from the queryable schema map, so the
        // query fails with "unknown field" rather than "no active index".
        let err = db.query_keys(ns, "status = \"active\"").unwrap_err();
        assert!(
            err.to_string().contains("unknown field"),
            "expected UnknownField error after deactivation, got: {err}"
        );

        db.shutdown().unwrap();
    }

    /// Regression for the O(1) targeted field-index update/delete (item 13): a
    /// document update must move its row from the old value's bucket to the new
    /// one (the prior value must stop matching), and a delete must remove it —
    /// driven by the prior document bytes read in the put/delete path, not a
    /// full bucket scan.
    #[test]
    fn test_field_index_targeted_update_and_delete() {
        use crate::db::namespace_index::ExtractorFn;
        use index::{IndexValue, IndexValueType};
        use std::sync::Arc;

        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = DEFAULT_NAMESPACE_ID;

        let field_id = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();
        let extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Str(v["status"].as_str()?.to_string()))
        });
        db.activate_field_index(ns, field_id, IndexValueType::Str, extractor).unwrap();

        db.put(b"doc:1", br#"{"status":"active"}"#).unwrap();
        db.put(b"doc:2", br#"{"status":"active"}"#).unwrap();
        assert_eq!(db.query_keys(ns, "status = \"active\"").unwrap().len(), 2);

        // Update doc:1's status active -> archived. The targeted update must move
        // the row, so it no longer matches "active" and now matches "archived".
        db.put(b"doc:1", br#"{"status":"archived"}"#).unwrap();
        assert_eq!(db.query_keys(ns, "status = \"active\"").unwrap(), vec![b"doc:2".to_vec()]);
        assert_eq!(db.query_keys(ns, "status = \"archived\"").unwrap(), vec![b"doc:1".to_vec()]);

        // Updating to the same value is a no-op and keeps the row queryable.
        db.put(b"doc:2", br#"{"status":"active"}"#).unwrap();
        assert_eq!(db.query_keys(ns, "status = \"active\"").unwrap(), vec![b"doc:2".to_vec()]);

        // Delete doc:2 → it must leave the "active" bucket.
        db.delete(b"doc:2").unwrap();
        assert!(db.query_keys(ns, "status = \"active\"").unwrap().is_empty());
        assert_eq!(db.query_keys(ns, "status = \"archived\"").unwrap(), vec![b"doc:1".to_vec()]);

        db.shutdown().unwrap();
    }

    /// Deactivating a field index and deleting its on-disk directory must not
    /// cause shutdown to fail with ENOENT when run_index_checkpoint is called.
    #[test]
    fn test_shutdown_after_drop_index_does_not_error() {
        use crate::db::namespace_index::ExtractorFn;
        use index::{IndexValue, IndexValueType};
        use std::sync::Arc;

        let dir = TempDir::new().unwrap();
        let db = Database::open(dir.path(), create_db_config()).unwrap();
        let ns = DEFAULT_NAMESPACE_ID;

        let field_id = db.register_index_field(ns, "status", IndexValueType::Str).unwrap();
        let extractor: ExtractorFn = Arc::new(|bytes: &[u8]| {
            let s = std::str::from_utf8(bytes).ok()?;
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            Some(IndexValue::Str(v["status"].as_str()?.to_string()))
        });
        db.activate_field_index(ns, field_id, IndexValueType::Str, extractor).unwrap();
        db.put(b"doc:1", br#"{"status":"active"}"#).unwrap();

        // Simulate drop_index: deactivate in-memory then delete on-disk directory.
        db.deactivate_field_index(ns, field_id).unwrap();
        let index_dir = dir.path().join("index").join(ns.to_string()).join(field_id.to_string());
        if index_dir.exists() {
            std::fs::remove_dir_all(&index_dir).unwrap();
        }

        // Shutdown must succeed even though the field's directory is gone.
        db.shutdown().unwrap();
    }

    // ── WAL GC segment deletion tests ────────────────────────────────────

    #[test]
    fn test_wal_gc_deletes_fully_persisted_segments() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 256)?;

        for i in 0..120u32 {
            let key = format!("wal_gc_key_{:03}", i).into_bytes();
            let value = vec![b'v'; 64];
            db.put(&key, &value)?;
        }

        db.flush_all_namespaces()?;

        let segment_size = db.wal.segment_size();
        let mut tail = db.wal_metadata.read().tail;
        let mut remaining = segment_size.saturating_sub(tail % segment_size);
        if remaining == segment_size {
            db.put(b"wal_gc_pad", b"pad")?;
            db.flush_all_namespaces()?;
            tail = db.wal_metadata.read().tail;
            remaining = segment_size.saturating_sub(tail % segment_size);
        }

        let mut pad_value_len = 1usize;
        loop {
            let pad_entry = WalEntry::new_upsert(b"wal_gc_pad_crash".to_vec(), vec![0u8; pad_value_len]);
            let entry_len = 4u64 + pad_entry.to_bytes()?.len() as u64;
            if entry_len > remaining && entry_len < segment_size {
                db.simulate_crash_with_wal_entries(vec![pad_entry])?;
                break;
            }
            pad_value_len = pad_value_len.saturating_add(8);
        }

        let segment1_path = temp_dir.path().join("wal.log.seg000001");
        assert!(segment1_path.exists());

        let crash_entries = (0..3u32)
            .map(|i| WalEntry::new_upsert(format!("crash_{}", i).into_bytes(), vec![b'x'; 16]))
            .collect();
        db.simulate_crash_with_wal_entries(crash_entries)?;

        let (bytes_reclaimed, _) = db.garbage_collect_wal()?;
        assert!(bytes_reclaimed > 0);
        assert!(!segment1_path.exists());

        Ok(())
    }

    // Regression: WAL GC can delete a fully-persisted segment out of segment
    // order (e.g. a dropped namespace persists a trailing segment while an
    // earlier one stays live), leaving a "hole". Recovery scans head→tail, so it
    // must survive the missing middle segment. Before the `scan_entries` hole-skip
    // fix, the open-time scan aborted with NotFound and `Database::open` failed.
    #[test]
    fn test_recovery_survives_deleted_middle_wal_segment() -> Result<()> {
        let temp_dir = TempDir::new()?;
        {
            let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 256)?;
            // Inject un-persisted entries spanning several WAL segments so reopen
            // actually runs recovery (persisted < total).
            let entries: Vec<WalEntry> = (0..60u32)
                .map(|i| WalEntry::new_upsert(format!("rk{:03}", i).into_bytes(), vec![b'v'; 32]))
                .collect();
            db.simulate_crash_with_wal_entries(entries)?;
            assert!(
                temp_dir.path().join("wal.log.seg000002").exists(),
                "need >= 3 segments to have a middle one"
            );

            // Punch a hole: delete a middle segment while segment 0 stays live
            // (so head is not advanced past it) — exactly what WAL GC would leave.
            db.wal.delete_segment_file(1)?;
            drop(db);
        }

        // Reopen: recovery must NOT crash on the hole, and the entries from the
        // surviving segments must be recovered (the deleted segment's are lost).
        let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 256)?;
        assert_eq!(db.get(b"rk000")?, Some(vec![b'v'; 32]), "segment 0 entry must survive the hole");
        assert_eq!(db.get(b"rk059")?, Some(vec![b'v'; 32]), "last-segment entry must survive the hole");
        db.shutdown()?;
        Ok(())
    }

    // Regression: the per-segment counter vecs must be trimmed as segments are
    // reclaimed, so the wal_metadata file stays proportional to the *live*
    // segment window instead of growing ~16 bytes for every segment ever created.
    #[test]
    fn test_wal_gc_trims_segment_counters_so_metadata_stays_bounded() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 256)?;

        let mut max_vec_len = 0usize;
        // Many write→persist→GC rounds create and reclaim far more segments than
        // are ever live at once.
        for round in 0..50u32 {
            for i in 0..10u32 {
                db.put(format!("k{}_{}", round, i).as_bytes(), &[b'v'; 32])?;
            }
            db.flush_all_namespaces()?;
            db.garbage_collect_wal()?;
            max_vec_len = max_vec_len.max(db.wal_metadata.read().segment_total_entries.len());
        }

        let m = db.wal_metadata.read();
        // The base advanced as old segments were reclaimed and trimmed...
        assert!(
            m.base_segment_id > 10,
            "base_segment_id should advance as segments are reclaimed, got {}",
            m.base_segment_id
        );
        // ...while the counter vecs stayed bounded by the live window rather than
        // growing to ~= total segments ever created (base + len).
        assert!(max_vec_len <= 8, "segment counter vec should stay bounded, peaked at {}", max_vec_len);
        assert_eq!(m.segment_total_entries.len(), m.segment_persisted_entries.len());
        drop(m);
        db.shutdown()?;
        Ok(())
    }

    #[test]
    fn test_wal_gc_shrinks_wal_directory_once_persisted() -> Result<()> {
        let temp_dir = TempDir::new()?;
        // Small segments so a modest number of writes fills several of them.
        let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 4096)?;

        for i in 0..400u32 {
            db.put(format!("wal_{:04}", i).as_bytes(), &[b'v'; 64])?;
        }
        // Persist the entries (flush memtables → SSTables), marking their WAL
        // segments fully persisted and therefore reclaimable.
        db.flush_all_namespaces()?;

        let wal_dir_bytes = |dir: &std::path::Path| -> u64 {
            std::fs::read_dir(dir)
                .unwrap()
                .flatten()
                .filter(|e| e.file_name().to_string_lossy().starts_with("wal.log.seg"))
                .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
                .sum()
        };

        let before = wal_dir_bytes(temp_dir.path());
        assert!(before > 4096, "expected several WAL segments before GC, got {before} bytes");

        let (reclaimed, _) = db.garbage_collect_wal()?;
        assert!(reclaimed > 0, "WAL GC should reclaim fully-persisted segments");

        let after = wal_dir_bytes(temp_dir.path());
        assert!(after < before, "WAL directory should shrink: {before} -> {after} bytes");

        // Persisted data is still readable after the segments were deleted.
        assert_eq!(db.get(b"wal_0000")?, Some(vec![b'v'; 64]));
        Ok(())
    }

    #[test]
    fn test_wal_gc_global_counters_stay_consistent_after_restart() -> Result<()> {
        let temp_dir = TempDir::new()?;

        {
            let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 256)?;

            for i in 0..120u32 {
                db.put(&format!("k{:03}", i).into_bytes(), &[b'v'; 32])?;
            }
            db.flush_all_namespaces()?;

            db.put(b"sentinel", b"v")?;
            db.flush_all_namespaces()?;

            let (bytes_reclaimed, _) = db.garbage_collect_wal()?;
            assert!(bytes_reclaimed > 0, "expected segments to be GC'd");

            let (total, persisted) = db.get_wal_gc_stats();
            assert_eq!(
                total.saturating_sub(persisted),
                0,
                "pending should be 0 immediately after GC (total={total}, persisted={persisted})"
            );
        }

        {
            let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 256)?;
            let (total, persisted) = db.get_wal_gc_stats();
            assert_eq!(
                total.saturating_sub(persisted),
                0,
                "pending should still be 0 after reopen (total={total}, persisted={persisted})"
            );
        }

        Ok(())
    }

    // ── WAL sequence number tests ─────────────────────────────────────────

    #[test]
    fn test_sequence_increments_on_writes() -> Result<()> {
        use std::sync::atomic::Ordering;
        let temp_dir = TempDir::new()?;
        let db = Database::open(temp_dir.path(), create_db_config())?;

        let seq_before = db.next_seq.load(Ordering::Relaxed);
        db.put(b"a", b"1")?;
        db.put(b"b", b"2")?;
        db.delete(b"a")?;
        let seq_after = db.next_seq.load(Ordering::Relaxed);

        assert_eq!(seq_after - seq_before, 3);
        Ok(())
    }

    #[test]
    fn test_sequence_stamps_wal_entries() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let db = Database::open(temp_dir.path(), create_db_config())?;

        db.put(b"x", b"v1")?;
        db.put(b"y", b"v2")?;

        let wal_metadata = db.wal_metadata.read();
        let entries = db.wal.scan_entries(0, wal_metadata.tail)?;
        drop(wal_metadata);

        let seqs: Vec<u64> = entries.iter().map(|(_, e)| e.sequence).collect();
        assert!(!seqs.is_empty());
        for seq in &seqs {
            assert!(*seq > 0, "sequence must be non-zero");
        }
        for w in seqs.windows(2) {
            assert!(w[1] > w[0], "sequences must be strictly increasing");
        }
        Ok(())
    }

    #[test]
    fn test_sequence_recovered_after_reopen() -> Result<()> {
        use std::sync::atomic::Ordering;
        let temp_dir = TempDir::new()?;

        let seq_at_close = {
            let db = Database::open(temp_dir.path(), create_db_config())?;
            db.put(b"k1", b"v1")?;
            db.put(b"k2", b"v2")?;
            db.next_seq.load(Ordering::Relaxed)
        };

        let db2 = Database::open(temp_dir.path(), create_db_config())?;
        let seq_after_reopen = db2.next_seq.load(Ordering::Relaxed);
        assert!(
            seq_after_reopen >= seq_at_close,
            "recovered seq ({}) must be >= seq at close ({})",
            seq_after_reopen,
            seq_at_close
        );

        let seq_before_new_write = db2.next_seq.load(Ordering::Relaxed);
        db2.put(b"k3", b"v3")?;
        let wal_metadata = db2.wal_metadata.read();
        let entries = db2.wal.scan_entries(0, wal_metadata.tail)?;
        drop(wal_metadata);
        let max_old_seq = seq_before_new_write - 1;
        let new_entry_seq = entries.last().map(|(_, e)| e.sequence).unwrap_or(0);
        assert!(
            new_entry_seq > max_old_seq,
            "new write seq ({}) must exceed previous max ({})",
            new_entry_seq,
            max_old_seq
        );
        Ok(())
    }

    #[test]
    fn test_gc_updates_last_sequence_before_deleting_segments() -> Result<()> {
        use std::sync::atomic::Ordering;
        let temp_dir = TempDir::new()?;
        let db = Database::open_with_wal_segment_size(temp_dir.path(), create_db_config(), 256)?;

        for i in 0..60u32 {
            let key = format!("gc_key_{}", i).into_bytes();
            db.put(&key, b"some_value_padding_xxx")?;
        }

        let seq_before_gc = db.next_seq.load(Ordering::Relaxed).saturating_sub(1);

        {
            let wal_metadata = db.wal_metadata.read();
            let tail = wal_metadata.tail;
            let head = wal_metadata.head;
            drop(wal_metadata);
            db.wal_flush_observer.mark_persisted_range(head, tail);
        }
        {
            let mut wal_metadata = db.wal_metadata.write();
            for i in 0..wal_metadata.segment_total_entries.len() {
                wal_metadata.segment_persisted_entries[i] = wal_metadata.segment_total_entries[i];
            }
        }

        db.garbage_collect_wal()?;

        let last_seq = db.wal_metadata.read().last_sequence;
        assert!(
            last_seq >= seq_before_gc,
            "last_sequence ({}) must be >= highest written seq ({}) after GC",
            last_seq,
            seq_before_gc
        );
        Ok(())
    }

    // ── WAL metadata corruption recovery test ─────────────────────────────

    #[test]
    fn test_metadata_checksum_corruption_recovery() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let path = temp_dir.path().to_path_buf();

        {
            let db = Database::open(&path, create_db_config())?;
            db.put(b"key1", b"value1")?;
            db.shutdown()?;
        }

        {
            let wal_metadata_path = path.join("wal_metadata");
            let mut data = std::fs::read(&wal_metadata_path)?;
            if let Some(byte) = data.get_mut(16) {
                *byte ^= 0xFF;
            }
            std::fs::write(&wal_metadata_path, data)?;
        }

        let db = Database::open(&path, create_db_config())?;
        assert!(path.join("wal_metadata.corrupt").exists());
        assert_eq!(db.get(b"key1")?, Some(b"value1".to_vec()));

        Ok(())
    }
}
