//! Database - Multi-namespace coordinator
//!
//! Manages a shared WAL, namespace registry, and multiple KVStore instances.
//! Each namespace has independent LSM + value log storage.

mod flush_observers;
pub(crate) mod key_locks;
mod namespaces;
mod open;
mod shutdown;
mod stats;
#[cfg(test)]
mod test_helpers;
mod wal_ops;
mod workers;
mod writes;

pub(crate) use flush_observers::*;

use crate::db::config::DbConfig;
use crate::db::database::key_locks::KeyLocks;
use crate::db::error::{KVError, Result};
use crate::db::indexing::manager::IndexManager;
use crate::db::kv_store::KVStore;
use crate::db::namespace::{DEFAULT_NAMESPACE_ID, FieldId, NamespaceRegistry};
use crate::db::stats::{GCStats, Stats};
use crate::db::wal::segmented_log::LogError;
use crate::db::wal::vector_index_wal::{VectorIndexWal, VectorIndexWalStats};
use crate::db::wal::{Wal, WalEntry, WalError, WalMetadata};
use crate::db::workers::index_checkpoint::{IndexCheckpointTarget, IndexCheckpointTrigger, IndexCheckpointWorker};
use crate::db::workers::lsm_compaction::{LsmCompactionCommand, LsmCompactionTarget, LsmCompactionWorker};
use crate::db::workers::ttl::{TtlTarget, TtlWorker};
use crate::db::workers::value_log_gc::{GCWorker, ValueLogGcTarget};
use crate::db::workers::wal_gc::{WalGcTarget, WalGcWorker};
use crate::storage::lsm::tree::LsmFlushObserver;
use crate::storage::value_log::ValueLogMetadata;

use log::{debug, error, info, warn};
use parking_lot::{Mutex, RwLock};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

/// Format a key for log output: UTF-8 string if valid, otherwise `hex:<hex>`.
pub(in crate::db) fn display_key(key: &[u8]) -> String {
    match std::str::from_utf8(key) {
        Ok(s) => s.to_owned(),
        Err(_) => format!("hex:{}", key.iter().map(|b| format!("{:02x}", b)).collect::<String>()),
    }
}

/// What a WAL-backed write achieved.
///
/// The write path has two distinct successes, and collapsing them into `Ok(())`
/// is what let a merge compute from a stale base. Once the WAL fsync returns the
/// write is **durable** and will take effect — but the in-memory apply that makes
/// it *readable* is best-effort (`Database::apply_with_retry`), so it can fail
/// while the write remains committed.
///
/// `put`/`delete` discard this: for a blind write, durable-but-not-yet-readable
/// is a success, and there is nothing useful the caller could do with the
/// distinction. `merge_ns` does not, because a merge whose result never became
/// readable poisons the next merge on that key.
pub(crate) struct WriteOutcome {
    /// The global sequence this write was assigned, for correlating with the
    /// ERROR log line the failed apply emitted.
    pub(crate) seq: u64,
    /// `true` if the write reached the in-memory store, so a read sees it now.
    /// `false` if it is durable in the WAL only, until the next open replays it.
    pub(crate) applied: bool,
}

/// What [`Database::lookup_open_store`] found for a namespace name.
pub(crate) enum OpenStoreLookup {
    /// Registered, and its store is open.
    Open(u32, Arc<KVStore>),
    /// No namespace has this name.
    Absent,
    /// Registered, but its store is not in the store map: being opened by
    /// `create_namespace`, or being removed.
    NotReady,
}

// ── Database coordinator ───────────────────────────────────────────────

/// The main database coordinator that manages:
/// - A shared WAL across all namespaces
/// - A namespace registry (name → id mapping)
/// - A collection of KVStore instances (one per namespace)
pub struct Database {
    pub(in crate::db) db_path: PathBuf,
    pub(crate) config: DbConfig,

    // Shared WAL
    pub(crate) wal: Arc<Wal>,
    pub(crate) wal_metadata: Arc<RwLock<WalMetadata>>,
    pub(crate) wal_flush_observer: Arc<WalPersistObserver>,
    #[allow(dead_code)]
    pending_wal_flushes: Arc<RwLock<BTreeMap<(u32, u64), WalFlushState>>>,
    pub(crate) wal_gc_in_progress: Arc<AtomicBool>,

    /// The vector index WAL: the vector worker's payload writes, one per
    /// database, shared by every namespace (`db/wal/vector_index_wal/mod.rs`).
    pub(crate) vector_index_wal: Arc<VectorIndexWal>,

    // Namespace registry
    pub(crate) registry: RwLock<NamespaceRegistry>,

    /// Serialises namespace creation, and lets a reader that lands mid-creation
    /// wait it out.
    ///
    /// Creation is not atomic across the two maps below: the registry entry is
    /// published (and persisted) before the `KVStore` is opened and inserted into
    /// `stores`, so in between, a name resolves to an id that `get_store` does
    /// not know. This lock is what makes creation appear atomic — held across the
    /// whole of `create_namespace` and `remove_namespace`, and taken by
    /// [`Database::get_store`] on a miss so it blocks until any in-flight creation
    /// has finished.
    ///
    /// **Lock ordering: this is the outermost lock.** Never acquire it while
    /// holding `registry` or `stores`. Making the two maps update atomically the
    /// obvious way instead — holding `stores.write()` across the registry publish
    /// — would deadlock against `metrics_snapshot_by_namespace`, which holds
    /// `registry.read()` while taking `stores.read()`.
    namespace_create_lock: Mutex<()>,

    /// Serialises field-index activation.
    ///
    /// Activation *creates* memory-mapped files — the field's `BlobStore` pair,
    /// its `keymap/` pair, and the namespace's `RowMap` — through a check-then-act
    /// (`if exists { open } else { create }`), and `GrowableMmap::create_file`
    /// opens with `truncate(true)`. Two threads activating the same field
    /// therefore truncate a file the other has already mapped, and the next
    /// access to that mapping faults: **SIGBUS, which kills the process** — no
    /// unwinding, no `Result`, the whole database goes down. A different
    /// interleaving surfaces it as a panic writing a 64-byte header into a
    /// zero-length map.
    ///
    /// Not hypothetical, and not confined to tests: any client that retries a
    /// `POST /stores` after a timeout can produce two concurrent creates of the
    /// same namespace, and each activates the same field. Measured before this
    /// lock: 8 racing creates killed the server on every attempt.
    ///
    /// **Lock ordering: taken only AFTER `get_store` returns**, never around it —
    /// `get_store` may wait on `namespace_create_lock`, which is strictly
    /// outermost (see above).
    pub(in crate::db) index_activate_lock: Mutex<()>,

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

    // Shared index-checkpoint backpressure valve — stored so newly-opened
    // namespaces inherit it. `None` until the checkpoint worker is enabled.
    pub(crate) index_checkpoint_trigger: Arc<parking_lot::RwLock<Option<Arc<IndexCheckpointTrigger>>>>,
    /// Memory budget shared by every field index's write buffer (its overlay
    /// of changed containers). Set on each index when it is activated, before WAL
    /// replay, so replay is bounded by it too.
    pub(crate) index_overlay_budget: Arc<crate::field_index::IndexOverlayBudget>,
    /// Rejected field-index updates awaiting the next checkpoint, which turns
    /// them into durable gap records. Shared with every `KVStore`.
    pub(crate) rejected_index_updates: Arc<crate::db::indexing::manager::RejectedUpdateBuffer>,
    /// Namespaces whose durable "no-WAL writes outstanding" marker is set.
    /// Debounces the marker write to once per checkpoint interval.
    pub(crate) no_wal_pending: parking_lot::Mutex<std::collections::HashSet<u32>>,

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
    pub(crate) next_seq: Arc<AtomicU64>,

    // Striped per-key write locks — the OUTERMOST lock on every WAL-backed
    // write path. `merge_ns` needs read-modify-write to be indivisible, and the
    // two locks the write path already takes cannot provide that: the WAL
    // metadata lock is global (so a user closure under it stalls every
    // namespace), and the value-log bucket lock is taken only *after* the WAL
    // lock is released, so making the merge atomic with it would invert the
    // established order. The stripe sits above both, giving one uniform order
    // `stripe -> wal_metadata -> (released) -> bucket`. See `key_locks`.
    pub(in crate::db) key_locks: KeyLocks,

    // Directory for recovery fail-log files.
    pub(in crate::db) fail_log_dir: PathBuf,

    // Global operational counters that belong to no single namespace: the
    // WAL-GC counters plus a fold of every dropped namespace's final totals.
    // Per-namespace counters live on each KVStore's own Metrics instance; the
    // engine-wide view (`metrics_snapshot`) sums those with this global one.
    pub(crate) metrics: Arc<crate::support::metrics::Metrics>,
}

impl Database {
    /// Get a KVStore by namespace ID.
    ///
    /// A miss is not immediately an error. `create_namespace` publishes the
    /// registry entry before it opens the store, so a caller that has just
    /// resolved a *name* to this id — which is what every get-or-create path
    /// does — can arrive while the store is still being opened. If the registry
    /// knows the id, a creation is or was in flight: wait on the creation lock
    /// and look once more. Ids the registry does not know fail straight away.
    pub(crate) fn get_store(&self, namespace_id: u32) -> Result<Arc<KVStore>> {
        if let Some(store) = self.stores.read().get(&namespace_id).cloned() {
            return Ok(store);
        }

        if self.registry.read().get_name(namespace_id).is_some() {
            // Blocks until any in-flight `create_namespace` has inserted its
            // store. Taken only after both reads above are released — this lock
            // is the outermost one (see `namespace_create_lock`).
            let _create = self.namespace_create_lock.lock();
            if let Some(store) = self.stores.read().get(&namespace_id).cloned() {
                return Ok(store);
            }
        }

        Err(KVError::Serialization(format!("Namespace with ID {} not found", namespace_id)))
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

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }
}

/// Unit tests for the coordinator.
#[cfg(test)]
mod tests;
