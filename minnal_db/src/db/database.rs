//! Database - Multi-namespace coordinator
//!
//! Manages a shared WAL, namespace registry, and multiple KVStore instances.
//! Each namespace has independent LSM + value log storage.

use crate::db::config::DbConfig;
use crate::db::error::{KVError, Result};
use crate::db::index_checkpoint_worker::{IndexCheckpointTarget, IndexCheckpointTrigger, IndexCheckpointWorker};
use crate::db::index_manager::IndexManager;
use crate::db::key_locks::KeyLocks;
use crate::db::kv_store::KVStore;
use crate::db::namespace::{DEFAULT_NAMESPACE_ID, FieldId, FieldMeta, FieldReindexOutcome, NamespaceRegistry};
use crate::db::namespace_index::{ExtractorFn, IndexEntry};
use crate::db::stats::{GCStats, Stats};
use crate::db::ttl_worker::{TtlTarget, TtlWorker};
use crate::db::wal::{Wal, WalEntry, WalEntryStatus, WalError, WalMetadata, WalOperationType};
use crate::db::wal_worker::{WalGcTarget, WalGcWorker};
use crate::index::{DynFieldIndex, IndexValueType};
use crate::store::gc_value_log_worker::{GCWorker, ValueLogGcTarget};
use crate::store::lsm::lsm_tree::LsmFlushObserver;
use crate::store::lsm_worker::{LsmCompactionCommand, LsmCompactionTarget, LsmCompactionWorker};
use crate::store::value_log::ValueLogMetadata;

use log::{debug, error, info, warn};
use parking_lot::{Mutex, RwLock};
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

/// How far one namespace's writes have made it onto disk.
///
/// The WAL is global but memtables are per-namespace, so "this entry is durable
/// in an SSTable" is a *per-namespace* question and the global watermark can only
/// be the slowest namespace's answer.
#[derive(Default, Clone, Copy)]
struct NsFlushProgress {
    /// WAL offset up to which every write by this namespace is in an SSTable.
    safe_offset: u64,
    /// WAL tail just after this namespace's most recent WAL append.
    last_write_offset: u64,
}

impl NsFlushProgress {
    /// Whether this namespace still holds WAL-backed writes that are not yet in
    /// an SSTable, and so must hold the global watermark back.
    fn has_unflushed(&self) -> bool {
        self.last_write_offset > self.safe_offset
    }
}

/// Observes LSM flush events and marks WAL entries as persisted.
/// Shared across all namespaces since the WAL is global.
pub(crate) struct WalPersistObserver {
    wal: Arc<Wal>,
    wal_metadata: Arc<RwLock<WalMetadata>>,
    wal_metadata_path: PathBuf,
    /// Keyed by `(namespace_id, memtable version)`. Memtable versions are
    /// allocated **per KVStore**, so keying by version alone made namespace A's
    /// version 7 and namespace B's version 7 the same entry — A's flush would
    /// then satisfy B's pending record and advance the watermark past B's
    /// un-flushed writes.
    pending: Arc<RwLock<BTreeMap<(u32, u64), WalFlushState>>>,
    /// Per-namespace flush progress — the input to the global watermark.
    ns_progress: RwLock<std::collections::HashMap<u32, NsFlushProgress>>,
    last_persisted_offset: Arc<RwLock<u64>>,
    /// Serializes every persisted-marking operation (`mark_persisted_range`,
    /// `mark_namespace_persisted`). These do a non-atomic scan → check status →
    /// flip → count over a WAL offset range; run concurrently they would flip and
    /// **count the same entry twice**, driving `persisted_entries` past
    /// `total_entries` and permanently wedging WAL GC (whose deletion gate is
    /// `persisted >= total`). Holding this across the whole operation makes the
    /// on-disk `status == Persisted` check a reliable per-entry dedup.
    persist_lock: Mutex<()>,
    /// Start offsets of writes appended to the WAL but not yet applied to a
    /// memtable — the writes for which "it is in the WAL" and "it is in the LSM"
    /// disagree *right now*.
    ///
    /// **Why this exists.** Every quantity that drives the persisted cut is
    /// derived from a `wal_metadata.tail` snapshot, and a tail snapshot
    /// over-states durability: `put_ns` appends under the WAL lock (advancing
    /// the tail) and applies to the memtable *after* releasing it, so between
    /// those two points an entry is visible in the tail while its key is
    /// nowhere. Marking such an entry `Persisted` tells recovery to skip a write
    /// that only ever existed in a memtable — and a crash before that memtable
    /// flushes loses an acknowledged write. Measured: 4 acknowledged writes lost
    /// in one round of `work/stress/repro/drop_loses_a_write.py`, and 1 in
    /// 34,538 in the 2026-08-02 stress run.
    ///
    /// Registration happens **while the WAL metadata write lock is still held**,
    /// so anyone who can observe the new tail can also observe the registration.
    /// That ordering is the whole guarantee; see [`Self::wal_cut_ceiling`].
    ///
    /// **Lock ordering: `ns_progress` → `wal_metadata` → `in_flight`.** Never
    /// take `wal_metadata` while holding this.
    ///
    /// Refcounted rather than a set. Two registrations can legitimately name the
    /// same offset — an entry starts exactly where the previous tail ended, so
    /// anything that registers a *synthetic* offset (a test simulating a write
    /// in flight) collides with the next real append, and a plain set would let
    /// one `Drop` cancel the other's protection. Counting makes registration
    /// composable and the barrier impossible to un-register by accident.
    in_flight: Mutex<BTreeMap<u64, u32>>,
}

/// Registers a WAL entry as appended-but-not-yet-applied for its lifetime.
///
/// RAII is load-bearing, not stylistic: `put_ns` has fallible steps between the
/// WAL append and the memtable apply (`get_store`), and a leaked registration
/// pins [`WalPersistObserver::wal_cut_ceiling`] forever — which freezes the
/// persisted watermark, so WAL GC's `persisted >= total` gate never fires again
/// and the WAL grows without bound. That is the same failure mode as the dropped
/// namespace in `17a8a0c`, reached a different way.
pub(crate) struct InFlightWrite<'a> {
    observer: &'a WalPersistObserver,
    offset: u64,
}

impl Drop for InFlightWrite<'_> {
    fn drop(&mut self) {
        let mut in_flight = self.observer.in_flight.lock();
        if let std::collections::btree_map::Entry::Occupied(mut e) = in_flight.entry(self.offset) {
            match e.get_mut() {
                1 => {
                    e.remove();
                }
                n => *n -= 1,
            }
        }
    }
}

impl WalPersistObserver {
    fn new(
        wal: Arc<Wal>,
        wal_metadata: Arc<RwLock<WalMetadata>>,
        wal_metadata_path: PathBuf,
        pending: Arc<RwLock<BTreeMap<(u32, u64), WalFlushState>>>,
        last_persisted_offset: Arc<RwLock<u64>>,
    ) -> Self {
        Self {
            wal,
            wal_metadata,
            wal_metadata_path,
            pending,
            ns_progress: RwLock::new(std::collections::HashMap::new()),
            last_persisted_offset,
            persist_lock: Mutex::new(()),
            in_flight: Mutex::new(BTreeMap::new()),
        }
    }

    /// Register `offset` as appended-to-the-WAL-but-not-yet-applied.
    ///
    /// **Call this while still holding the WAL metadata write lock.** The
    /// guarantee [`Self::wal_cut_ceiling`] rests on is that no observer can see a
    /// tail containing this entry without also seeing this registration;
    /// registering after releasing the lock reopens exactly the window this
    /// closes.
    pub(crate) fn begin_write(&self, offset: u64) -> InFlightWrite<'_> {
        *self.in_flight.lock().entry(offset).or_insert(0) += 1;
        InFlightWrite { observer: self, offset }
    }

    /// The highest WAL offset that may be treated as durable right now.
    ///
    /// This is the current tail, lowered to exclude every write that has been
    /// appended but not yet applied to a memtable. Both consumers of "how far
    /// has the WAL got" must go through it:
    ///
    /// * [`Self::try_advance_persisted`], whose cut would otherwise be the live
    ///   tail whenever no namespace reports un-flushed writes;
    /// * [`Self::on_memtable_sealed_ns`], which records "everything up to here is
    ///   in the memtable I am sealing" — false for a write whose apply has not
    ///   landed yet, because that apply goes into the *next* memtable while its
    ///   offset sits below the recorded tail.
    ///
    /// The second one is why this is a shared helper rather than a check bolted
    /// onto the first: it needs no namespace drop to trigger, and a fix that
    /// only guarded the cut would have left it live.
    ///
    /// Lock order is `wal_metadata` then `in_flight`, matching the write path.
    fn wal_cut_ceiling(&self) -> u64 {
        let tail = self.wal_metadata.read().tail;
        match self.in_flight.lock().first_key_value() {
            Some((&oldest, _)) => tail.min(oldest),
            None => tail,
        }
    }

    /// Namespaces holding WAL-backed writes that are not yet in an SSTable.
    ///
    /// These are what hold the global watermark — and therefore WAL GC — back.
    /// A namespace that writes a little and then goes idle would pin the WAL
    /// indefinitely, so the WAL GC worker flushes them on its tick.
    pub(crate) fn namespaces_with_unflushed(&self) -> Vec<u32> {
        self.ns_progress
            .read()
            .iter()
            .filter(|(_, p)| p.has_unflushed())
            .map(|(ns, _)| *ns)
            .collect()
    }

    /// Record that `namespace_id` appended a WAL entry ending at `tail`.
    ///
    /// This is what lets the watermark tell "namespace has nothing outstanding"
    /// apart from "namespace has not flushed yet": a namespace that never writes
    /// must not hold the watermark — and therefore WAL GC — back forever.
    pub(crate) fn note_write(&self, namespace_id: u32, tail: u64) {
        let mut progress = self.ns_progress.write();
        let entry = progress.entry(namespace_id).or_default();
        entry.last_write_offset = entry.last_write_offset.max(tail);
    }

    /// Stop tracking a namespace that has been dropped.
    ///
    /// A dropped namespace's store is gone from the registry, so it can never
    /// flush again and its `safe_offset` can never advance. Left in
    /// `ns_progress` it reports [`has_unflushed`](NsFlushProgress::has_unflushed)
    /// forever, which pins the global cut in [`try_advance_persisted`] at its
    /// stale offset — and [`Database::flush_namespaces_pinning_wal`] cannot clear
    /// it, because the store it would flush no longer exists. The watermark then
    /// never advances, `persisted_entries` stops tracking `total_entries`, and
    /// WAL GC's `persisted >= total` gate never fires again: unbounded WAL growth
    /// after any namespace drop.
    ///
    /// Forgetting it is sound because `remove_namespace` has already flushed and
    /// shut the store down and marked its WAL entries persisted
    /// ([`mark_namespace_persisted`]) before calling this — those entries need no
    /// further protection, and recovery skips entries for namespaces missing from
    /// the registry anyway.
    ///
    /// [`try_advance_persisted`]: Self::try_advance_persisted
    /// [`mark_namespace_persisted`]: Self::mark_namespace_persisted
    pub(crate) fn forget_namespace(&self, namespace_id: u32) {
        {
            self.ns_progress.write().remove(&namespace_id);
            // Seal records for memtables that will now never be flushed; left
            // behind they would be folded into a future namespace's scan.
            self.pending.write().retain(|(ns, _), _| *ns != namespace_id);
        }
        // Locks released: the dropped namespace was potentially the one holding
        // the cut back, so re-evaluate it now rather than waiting for some other
        // namespace to happen to flush.
        self.try_advance_persisted();
    }

    pub(crate) fn mark_persisted_range(&self, start: u64, end: u64) {
        // Serialize with every other persisted-marking operation so no entry is
        // scanned+flipped+counted concurrently (see `persist_lock`).
        let _persist = self.persist_lock.lock();

        // Re-read the authoritative watermark under the lock: the caller's `start`
        // may be stale if another persist advanced past it while we waited, and
        // processing an already-counted range would double-count it. `start` is
        // kept as a floor (it only ever equals the watermark at call sites).
        let start = (*self.last_persisted_offset.read()).max(start);
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

        // Serialize with `mark_persisted_range` (same `persist_lock`): both flip and
        // count entries in overlapping windows, so running concurrently they could
        // count the same entry twice. Under the lock, whichever flips an entry first
        // marks it `Persisted` on disk and the other skips it uncounted.
        let _persist = self.persist_lock.lock();

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

    /// Fold `namespace_id`'s completed flushes into its `safe_offset`.
    ///
    /// Versions are consumed in order and only while flushed, so an out-of-order
    /// flush cannot advance the namespace past a still-pending older memtable.
    fn advance_namespace(&self, namespace_id: u32) {
        let mut new_safe = 0u64;
        loop {
            let next = {
                let pending = self.pending.read();
                // BTreeMap ordering makes this the namespace's lowest version.
                let Some((&key, state)) = pending.range((namespace_id, 0)..=(namespace_id, u64::MAX)).next() else {
                    break;
                };
                if !state.flushed {
                    break;
                }
                (key, state.tail)
            };

            {
                let mut pending = self.pending.write();
                let Some(state) = pending.get(&next.0) else {
                    continue;
                };
                if !state.flushed {
                    break;
                }
                pending.remove(&next.0);
            }
            new_safe = new_safe.max(next.1);
        }

        if new_safe > 0 {
            let mut progress = self.ns_progress.write();
            let entry = progress.entry(namespace_id).or_default();
            entry.safe_offset = entry.safe_offset.max(new_safe);
        }
    }

    /// Advance the global watermark to the point every namespace has reached.
    ///
    /// A WAL entry may only be marked `Persisted` once the namespace that owns it
    /// has flushed it to an SSTable, because recovery skips `Persisted` entries.
    /// The WAL is shared, so the safe cut is the **minimum** `safe_offset` over
    /// the namespaces that still hold un-flushed writes. Marking up to one
    /// namespace's own tail — as this did before — declared every *other*
    /// namespace's un-flushed entries persisted too, and a crash then lost them.
    /// Namespaces with nothing outstanding do not constrain the cut; if none do,
    /// everything written so far is durable and the cut is the current tail.
    fn try_advance_persisted(&self) {
        let cut = {
            let progress = self.ns_progress.read();
            let slowest = progress.values().filter(|p| p.has_unflushed()).map(|p| p.safe_offset).min();
            // The ceiling excludes writes that are in the WAL but not yet in any
            // memtable. It is applied to BOTH branches, not just the `None` one:
            // capping the `Some` branch too costs nothing (a namespace's
            // `safe_offset` is already below any later write) and removes the
            // need to re-derive that argument every time this is read.
            //
            // Without it, the `None` branch cut at the live tail, which is how a
            // store drop lost acknowledged writes: `forget_namespace` calls this
            // the instant it removes a namespace from `ns_progress`, and that is
            // precisely when the set of namespaces reporting un-flushed writes
            // can go empty while other namespaces are mid-write.
            let ceiling = self.wal_cut_ceiling();
            match slowest {
                Some(offset) => offset.min(ceiling),
                None => ceiling,
            }
        };

        let start = *self.last_persisted_offset.read();
        if cut > start {
            self.mark_persisted_range(start, cut);
        }
    }
}

impl WalPersistObserver {
    fn on_memtable_sealed_ns(&self, namespace_id: u32, version: u64) {
        // The ceiling, not the raw tail. This record means "every WAL entry below
        // `tail` is in the memtable being sealed", and that is false for a write
        // whose apply has not landed yet: its apply will go into the *next*
        // memtable, while its offset is below this tail. Recording the raw tail
        // therefore let `advance_namespace` push `safe_offset` past an entry that
        // no SSTable holds — the same lost-write bug as the cut, reached without
        // any namespace drop.
        //
        // Computed before taking `pending`: the lock order is
        // `pending` → `ns_progress` → `wal_metadata` → `in_flight`, so the tail
        // must never be read while `pending` is held.
        let tail = self.wal_cut_ceiling();
        let mut pending = self.pending.write();
        pending.entry((namespace_id, version)).or_insert(WalFlushState { tail, flushed: false });
    }

    fn on_ro_memtable_flushed_to_level0_ns(&self, namespace_id: u32, version: u64) {
        // Same reasoning as the seal path, for the case where the flush is
        // observed without a preceding seal record.
        let ceiling = self.wal_cut_ceiling();
        {
            let mut pending = self.pending.write();
            let entry = pending
                .entry((namespace_id, version))
                .or_insert(WalFlushState { tail: 0, flushed: false });
            if entry.tail == 0 {
                entry.tail = ceiling;
            }
            entry.flushed = true;
        }
        self.advance_namespace(namespace_id);
        self.try_advance_persisted();
    }
}

/// Hub that fans out LSM flush events to the WAL observer and compaction trigger.
///
/// One hub per KVStore, which is what supplies the namespace id the shared
/// [`WalPersistObserver`] needs: the LSM itself only knows memtable versions, and
/// those are allocated per store.
struct LsmFlushObserverHub {
    namespace_id: u32,
    wal_observer: Arc<WalPersistObserver>,
    compaction_trigger: Arc<RwLock<Option<tokio::sync::mpsc::UnboundedSender<LsmCompactionCommand>>>>,
}

impl LsmFlushObserverHub {
    fn new(
        namespace_id: u32,
        wal_observer: Arc<WalPersistObserver>,
        compaction_trigger: Arc<RwLock<Option<tokio::sync::mpsc::UnboundedSender<LsmCompactionCommand>>>>,
    ) -> Self {
        Self {
            namespace_id,
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
        self.wal_observer.on_memtable_sealed_ns(self.namespace_id, version);
        self.trigger_compaction();
    }

    fn on_ro_memtable_flushed_to_level0(&self, version: u64) {
        self.wal_observer.on_ro_memtable_flushed_to_level0_ns(self.namespace_id, version);
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

// ── Database coordinator ───────────────────────────────────────────────

/// The main database coordinator that manages:
/// - A shared WAL across all namespaces
/// - A namespace registry (name → id mapping)
/// - A collection of KVStore instances (one per namespace)
pub struct Database {
    db_path: PathBuf,
    pub(crate) config: DbConfig,

    // Shared WAL
    pub(crate) wal: Arc<Wal>,
    #[allow(dead_code)]
    wal_path: PathBuf,
    wal_metadata_path: PathBuf,
    pub(crate) wal_metadata: Arc<RwLock<WalMetadata>>,
    pub(crate) wal_flush_observer: Arc<WalPersistObserver>,
    #[allow(dead_code)]
    pending_wal_flushes: Arc<RwLock<BTreeMap<(u32, u64), WalFlushState>>>,
    last_persisted_wal_offset: Arc<RwLock<u64>>,
    pub(crate) wal_gc_in_progress: Arc<AtomicBool>,

    // Namespace registry
    pub(crate) registry: RwLock<NamespaceRegistry>,

    /// Serialises namespace creation, and lets a reader that lands mid-creation
    /// wait it out.
    ///
    /// Creation is not atomic across the two maps below: the registry entry is
    /// published (and persisted) before the `KVStore` is opened and inserted into
    /// `stores`, so in between, a name resolves to an id that `get_store` does
    /// not know. This lock is what makes creation appear atomic — held across the
    /// whole of `create_namespace`, and taken by [`Database::get_store`] on a miss
    /// so it blocks until any in-flight creation has finished.
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
    index_activate_lock: Mutex<()>,

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
    /// Rejected field-index updates awaiting the next checkpoint, which turns
    /// them into durable gap records. Shared with every `KVStore`.
    pub(crate) rejected_index_updates: Arc<crate::db::index_manager::RejectedUpdateBuffer>,
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
    key_locks: KeyLocks,

    // Directory for recovery fail-log files.
    fail_log_dir: PathBuf,

    // Global operational counters that belong to no single namespace: the
    // WAL-GC counters plus a fold of every dropped namespace's final totals.
    // Per-namespace counters live on each KVStore's own Metrics instance; the
    // engine-wide view (`metrics_snapshot`) sums those with this global one.
    pub(crate) metrics: Arc<crate::db::metrics::Metrics>,
}

impl Database {
    /// Resolve the WAL segment size to open with, locking it at creation.
    ///
    /// Precedence:
    /// 1. An existing `wal_segment_size` marker wins — the size is fixed for the
    ///    life of the WAL, because a segment id is `offset / segment_size` and a
    ///    different size would map stored offsets to the wrong segment files.
    /// 2. No marker but the WAL already holds data (a metadata file, or a
    ///    non-empty `wal.log`): a pre-marker WAL, which was written at the
    ///    historical [`DEFAULT_SEGMENT_SIZE`]. Use that and stamp the marker.
    /// 3. Brand-new WAL: honour `configured` and stamp the marker.
    ///
    /// The marker write is best-effort; if it fails the effective size is still
    /// correct for this run (it just re-resolves the same way next open).
    fn resolve_wal_segment_size(db_path: &Path, wal_path: &Path, wal_metadata_path: &Path, configured: u64) -> u64 {
        let marker = db_path.join("wal_segment_size");

        if let Ok(bytes) = std::fs::read(&marker)
            && let Ok(arr) = <[u8; 8]>::try_from(bytes.as_slice())
        {
            let recorded = u64::from_le_bytes(arr);
            if recorded != 0 {
                if configured != recorded {
                    warn!(
                        "[WAL] Configured wal.segment_size_bytes={} but this WAL was created at {}. \
                         The segment size is fixed at creation and cannot change on existing data; \
                         the configured value will NOT be applied.",
                        configured, recorded,
                    );
                }
                return recorded;
            }
        }

        let preexisting = wal_metadata_path.exists() || std::fs::metadata(wal_path).map(|m| m.len() > 0).unwrap_or(false);
        let size = if preexisting {
            crate::db::wal::DEFAULT_SEGMENT_SIZE
        } else {
            configured.max(crate::db::wal::MIN_SEGMENT_SIZE)
        };

        if let Err(e) = crate::support::write_atomic_durable(&marker, &size.to_le_bytes()) {
            warn!("[WAL] Failed to record WAL segment-size marker at {}: {:?}", marker.display(), e);
        }
        size
    }

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

        // The WAL segment size is fixed at creation and honoured from config only
        // for a brand-new WAL. It cannot change on existing data: a segment id is
        // `offset / segment_size`, so a different size re-buckets every stored
        // offset into the wrong segment file. The chosen size is recorded in a
        // marker so it survives restarts; a pre-marker WAL falls back to the
        // historical 64 MiB it was written at. (`wal.segment_size_bytes` in config.)
        let wal_segment_size = Self::resolve_wal_segment_size(db_path, &wal_path, &wal_metadata_path, config.wal_segment_size);

        // Open WAL
        let wal = Arc::new(Wal::open_with_options_and_segment_size(&wal_path, false, wal_segment_size)?);

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
            let ns_path = crate::db::layout::namespace_data_dir(db_path, name);
            let ttl = registry.ttl_config(ns_id).map(|(ttl, _)| ttl);
            let kv_store = KVStore::open_with_ttl(
                ns_id,
                name,
                &ns_path,
                config.lsm_config.clone(),
                config.sync_config,
                config.segment_size_bytes,
                ttl,
            )?;
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
            namespace_create_lock: Mutex::new(()),
            index_activate_lock: Mutex::new(()),
            stores: RwLock::new(stores),
            closed: Arc::new(AtomicBool::new(false)),
            wal_gc_worker: Arc::new(tokio::sync::RwLock::new(None)),
            lsm_compaction_worker: Arc::new(tokio::sync::RwLock::new(None)),
            value_log_gc_worker: Arc::new(tokio::sync::RwLock::new(None)),
            lsm_compaction_sender: Arc::new(parking_lot::RwLock::new(None)),
            index_checkpoint_trigger: Arc::new(parking_lot::RwLock::new(None)),
            rejected_index_updates: Arc::new(crate::db::index_manager::RejectedUpdateBuffer::default()),
            no_wal_pending: parking_lot::Mutex::new(std::collections::HashSet::new()),
            ttl_worker: Arc::new(tokio::sync::RwLock::new(None)),
            index_manager,
            index_checkpoint_worker: Arc::new(tokio::sync::RwLock::new(None)),
            next_seq: Arc::new(AtomicU64::new(next_seq_start)),
            key_locks: KeyLocks::new(),
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
                kv_store.set_metrics(std::sync::Arc::new(crate::db::metrics::Metrics::default()));
                kv_store.set_index_gap_sink(Some(
                    Arc::clone(&db.rejected_index_updates) as Arc<dyn crate::db::index_manager::IndexGapSink>
                ));
            }
        }

        // Finish any field-index drop a crash interrupted, before recovery can
        // replay into a directory that is meant to be gone.
        db.complete_interrupted_field_drops();

        // Report field indices left incomplete by no-WAL writes that an unclean
        // shutdown caught before any checkpoint covered them.
        db.record_no_wal_gaps_after_unclean_shutdown();

        // Recover from WAL
        db.recover_from_wal()?;

        // Rebuild WAL persisted state and wire up observers
        let last_persisted = db.rebuild_wal_persisted_state()?;
        *db.last_persisted_wal_offset.write() = last_persisted;
        db.wire_up_flush_observers();

        Ok(db)
    }

    /// Detect an existing database's bucket count from its value-log filenames.
    ///
    /// Both the segment files (`value_log_{bucket}.seg000123`) and the per-bucket
    /// metadata (`value_log_{bucket}.metadata`) name their bucket right after the
    /// `value_log_` prefix. A bucket now owns **many** segment files, so a plain file
    /// count over-reports; instead take the highest bucket id seen and add one. Buckets
    /// are numbered `0..num_buckets`, so that is the original count — and it stays
    /// correct even if one bucket's files are momentarily absent.
    fn detect_bucket_count(vlog_dir: &Path) -> usize {
        let Ok(entries) = std::fs::read_dir(vlog_dir) else {
            return 0;
        };
        let highest = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                let rest = name.strip_prefix("value_log_")?;
                rest.split('.').next()?.parse::<u32>().ok()
            })
            .max();
        highest.map_or(0, |b| b as usize + 1)
    }

    /// Wire up LSM flush observers for all KVStores
    fn wire_up_flush_observers(&self) {
        let stores = self.stores.read();
        for (ns_id, kv_store) in stores.iter() {
            let observer: Arc<dyn LsmFlushObserver> = Arc::new(LsmFlushObserverHub::new(
                *ns_id,
                Arc::clone(&self.wal_flush_observer),
                Arc::clone(&kv_store.lsm_compaction_trigger),
            ));
            kv_store.set_flush_observer(Some(observer));
        }
    }

    /// Build the index-checkpoint backpressure valve for `worker` (using the
    /// configured `index_blob_backpressure_bytes` cap), wire it into every
    /// existing store, and record it centrally so namespaces opened later inherit
    /// it. Called when the checkpoint worker is enabled.
    pub(crate) fn wire_index_checkpoint_trigger(&self, worker: &IndexCheckpointWorker) {
        let cap = self.config.threshold_config.index_blob_backpressure_bytes;
        let trigger = worker.backpressure_trigger(cap);
        {
            let stores = self.stores.read();
            for kv_store in stores.values() {
                kv_store.set_index_checkpoint_trigger(Some(Arc::clone(&trigger)));
            }
        }
        *self.index_checkpoint_trigger.write() = Some(trigger);
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

    pub fn merge<F>(&self, key: &[u8], value: &[u8], merge_fn: F) -> Result<Option<Vec<u8>>>
    where
        F: FnOnce(Option<&[u8]>, &[u8]) -> Result<Option<Vec<u8>>>,
    {
        self.merge_ns(DEFAULT_NAMESPACE_ID, key, value, merge_fn)
    }

    // ── Namespace-aware data operations ────────────────────────────────

    /// Number of attempts to apply a WAL-durable write to the in-memory store
    /// before giving up. Once the WAL fsync has succeeded the write is durable
    /// and *will* take effect (via the in-memory apply now, or via WAL replay on
    /// the next open), so this only bounds the retry of transient apply errors.
    const APPLY_RETRY_ATTEMPTS: usize = 3;

    /// Fail-log label carried by WAL entries a `merge` produced.
    ///
    /// (See also [`WriteOutcome`], returned by the `_locked` helpers below.)
    ///
    /// Only merges set `op_name`. An ordinary put or delete leaves it empty, as
    /// it always has: the entry's own `WalOperationType` already says what it is,
    /// so writing a label would add bytes to every record on the hot path to
    /// record nothing new. A merge is worth the exception — its entry is
    /// indistinguishable from a blind write otherwise, and "which op produced
    /// this?" is exactly the question a fail log is read to answer.
    const MERGE_OP_NAME: &'static str = "merge";

    /// Cap on how many keys a field's gap record will carry as a row-scoped
    /// repair worklist before it is downgraded to a full rebuild.
    ///
    /// Row-scoped repair is bounded by the damage, but the damage itself is not
    /// bounded: a wedged checkpoint worker can strand many segments, and a
    /// worklist large enough to rival the store would be both a disk cost and a
    /// slower repair than simply rebuilding the field. At that point a full
    /// rebuild is the cheaper, simpler answer, so the keys are dropped rather
    /// than accumulated.
    pub(crate) const GAP_KEY_WORKLIST_CAP: usize = 100_000;

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
        // The key stripe is the outermost write lock (see `key_locks`). A blind
        // put does not need it for itself, but taking it is what lets `merge_ns`
        // read-modify-write without a concurrent put slipping in between its
        // read and its write. It costs an uncontended mutex acquire on a path
        // that already serialises globally on the WAL fsync.
        let _stripe = self.key_locks.guard(namespace_id, key);
        // Discarded: `put`'s contract is that a WAL-durable write is a success,
        // whether or not it is readable yet. See `WriteOutcome`.
        self.put_ns_locked(namespace_id, key, value, "put").map(|_| ())
    }

    /// The body of [`Self::put_ns`], with the caller holding the key stripe.
    ///
    /// Split out so [`Self::merge_ns`] can reuse the WAL append verbatim rather
    /// than growing a second copy of it. That matters more than it looks:
    /// `WalPersistObserver::begin_write` must be called while the WAL metadata
    /// write lock is still held, and a hand-rolled second copy that got the
    /// ordering wrong lost one acknowledged write in 34,538 (see the *WAL
    /// ownership* section of `minnal_db/CLAUDE.md`).
    ///
    /// `op_name` labels the operation in log and fail-log output; see
    /// [`Self::MERGE_OP_NAME`] for which entries carry it into the WAL.
    ///
    /// Returns the [`WriteOutcome`] so `merge_ns` can tell a write that became
    /// readable from one that is merely durable. `put_ns` discards it, keeping
    /// its own `Ok(())` contract unchanged.
    fn put_ns_locked(&self, namespace_id: u32, key: &[u8], value: &[u8], op_name: &str) -> Result<WriteOutcome> {
        self.check_write_size(key, value)?;

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
        let mut wal_entry = WalEntry::new_upsert_ns(namespace_id, key.to_vec(), value.to_vec()).with_sequence(seq);
        if op_name == Self::MERGE_OP_NAME {
            wal_entry = wal_entry.with_op_name(op_name);
        }
        let wal_pointer = self.wal.append_entry(&wal_entry, &mut wal_metadata.tail, true)?;
        let segment_id = self.wal.segment_id_for_offset(wal_pointer.offset);
        wal_metadata.add_segment_total(segment_id, 1);
        wal_metadata.total_entries += 1;
        let wal_tail = wal_metadata.tail;
        // Registered BEFORE the WAL lock is released, so no observer can see a
        // tail containing this entry without also seeing that it is not yet in
        // any memtable. Held until this function returns, by which point step 2
        // has applied it (or failed loudly). See `WalPersistObserver::in_flight`.
        let _in_flight = self.wal_flush_observer.begin_write(wal_pointer.offset);
        drop(wal_metadata);

        // This namespace now has a WAL-backed write that is not yet in an
        // SSTable, so it holds the global persisted watermark back until it
        // flushes (see `WalPersistObserver::try_advance_persisted`).
        self.wal_flush_observer.note_write(namespace_id, wal_tail);

        // Step 2: Apply to the namespace's in-memory store. The write is already
        // durable in the WAL, so this is best-effort with bounded retry: on
        // persistent failure we log and still return Ok, because recovery will
        // replay the entry on the next open. Surfacing an error here would be
        // misleading — the data is already committed.
        let kv_store = self.get_store(namespace_id)?;

        // Record write counters against this namespace's metrics (the store is
        // already in hand, so this is no extra lookup).
        if let Some(m) = kv_store.metrics() {
            crate::db::metrics::Metrics::bump(&m.puts);
            crate::db::metrics::Metrics::bump(&m.wal_fsyncs);
            crate::db::metrics::Metrics::add(&m.wal_bytes_appended, wal_pointer.size as u64 + 4);
        }

        let applied = Self::apply_with_retry(op_name, namespace_id, key, seq, || kv_store.put_to_storage_seq(key, value, seq));
        if !applied && let Some(m) = kv_store.metrics() {
            crate::db::metrics::Metrics::bump(&m.apply_failures);
        }

        // Step 3: Maybe sync the value log.
        //
        // Syncing the value log does NOT make these WAL entries replaceable by
        // what is on disk: the value is durable, but the key → pointer mapping
        // still lives only in the LSM memtable until that memtable is flushed to
        // an SSTable. Marking the WAL persisted here would tell recovery to skip
        // entries whose keys exist nowhere on disk, silently losing acknowledged
        // writes and stranding their values in the value log. The persisted
        // watermark is advanced solely by `WalPersistObserver`, which waits for
        // the memtable flush (`on_memtable_sealed` → `try_advance_persisted`).
        if kv_store.should_sync() {
            let _ = kv_store.sync_value_log();
        }

        Ok(WriteOutcome { seq, applied })
    }

    /// Write a key-value pair **without** appending to the WAL.
    ///
    /// This skips the fsync that normally accompanies every WAL entry, giving
    /// much higher throughput at the cost of crash safety: any data written
    /// via this method is unrecoverable if the process crashes before the
    /// value-log segment is flushed.  Only use this for bulk-loading scenarios
    /// where re-running the load is acceptable.
    pub fn put_ns_no_wal(&self, namespace_id: u32, key: &[u8], value: &[u8]) -> Result<()> {
        self.check_closed()?;
        self.check_write_size(key, value)?;
        // Outermost write lock — see `put_ns`. No-WAL writes take it too, so a
        // merge is serialised against bulk loading as well as against `put`.
        let _stripe = self.key_locks.guard(namespace_id, key);
        let kv_store = self.get_store(namespace_id)?;
        if let Some(m) = kv_store.metrics() {
            crate::db::metrics::Metrics::bump(&m.no_wal_puts);
        }
        // No WAL entry exists for this write, so until the memtable is flushed
        // it lives only in memory — the background flusher uses this to bound
        // how much a crash can destroy.
        kv_store.note_no_wal_write();
        self.note_no_wal_index_exposure(namespace_id);
        kv_store.put_to_storage(key, value)
    }

    /// Delete a key **without** appending to the WAL.
    ///
    /// The no-WAL counterpart of [`Self::delete_ns`]: it skips the WAL append
    /// and its fsync, so a lost delete is unrecoverable if the process crashes
    /// before the tombstone is flushed. Only safe for derived/regenerable data
    /// where a missed delete self-heals — e.g. a TTL cache, where a surviving
    /// stale entry expires on its own. See [`Self::put_ns_no_wal`] for the
    /// write-side trade-off. The delete still allocates from the global
    /// sequence counter (via `delete_from_storage`), so it resolves against
    /// concurrent same-key writes highest-sequence-wins like any other write.
    pub fn delete_ns_no_wal(&self, namespace_id: u32, key: &[u8]) -> Result<()> {
        self.check_closed()?;
        self.check_write_size(key, &[])?;
        // Outermost write lock — see `put_ns`.
        let _stripe = self.key_locks.guard(namespace_id, key);
        let kv_store = self.get_store(namespace_id)?;
        if let Some(m) = kv_store.metrics() {
            crate::db::metrics::Metrics::bump(&m.no_wal_deletes);
        }
        kv_store.note_no_wal_write();
        self.note_no_wal_index_exposure(namespace_id);
        kv_store.delete_from_storage(key)
    }

    /// Read a key, hand the stored value and `value` to `merge_fn`, and write
    /// the result back — atomically with respect to every other write to that
    /// key.
    ///
    /// `merge_fn` receives `(existing, operand)`, where `existing` is `None`
    /// when the key is absent, and returns:
    ///
    /// - `Ok(Some(new))` — write `new` (an ordinary WAL-backed upsert),
    /// - `Ok(None)` — delete the key; a no-op appending **nothing** to the WAL
    ///   when the key was already absent,
    /// - `Err(e)` — abort. Nothing is written, no WAL entry is appended and no
    ///   sequence number is consumed.
    ///
    /// Returns the value that was written, or `None` for the delete and no-op
    /// cases — or [`KVError::MergeNotApplied`] if the write was committed to the
    /// WAL but did not reach the in-memory store, so its result is not readable
    /// (see [`Self::require_applied`]).
    ///
    /// # Durability and the WAL
    ///
    /// The merge resolves to an ordinary `Upsert` or `Delete` WAL record
    /// carrying the *result*, so recovery, WAL GC and index replay are unchanged
    /// — there is no merge record type and nothing re-runs `merge_fn` at replay.
    /// (A closure cannot be serialised, and recovery runs long before user code
    /// could re-register one.) The usual durability contract applies: the call
    /// returns once the WAL entry is fsynced.
    ///
    /// # The closure runs under a lock
    ///
    /// `merge_fn` is invoked while this key's stripe lock is held, so it must be
    /// quick and must **not write back** into the database — a re-entrant write
    /// to the same key, or to any key sharing its stripe, self-deadlocks. Reads
    /// are safe but pointless: a read of this key returns the same pre-merge
    /// value the closure was already handed.
    ///
    /// TTL expiry is the one writer outside the stripe: it deletes straight
    /// through `KVStore`, so a merge racing an expiry can resurrect a key that
    /// was about to expire. The next TTL pass expires it again.
    pub fn merge_ns<F>(&self, namespace_id: u32, key: &[u8], value: &[u8], merge_fn: F) -> Result<Option<Vec<u8>>>
    where
        F: FnOnce(Option<&[u8]>, &[u8]) -> Result<Option<Vec<u8>>>,
    {
        self.check_closed()?;
        // Held across the read, the closure, the WAL append AND the in-memory
        // apply. This is the whole atomicity argument: no other WAL-backed write
        // to this key can be durable-but-unapplied while we hold it, so the
        // closure never sees a stale base.
        let _stripe = self.key_locks.guard(namespace_id, key);

        let kv_store = self.get_store(namespace_id)?;
        if let Some(m) = kv_store.metrics() {
            crate::db::metrics::Metrics::bump(&m.merges);
        }

        let existing = kv_store.get(key)?;
        // `existing` outlives the call: the delete arm below needs to know
        // whether there was anything there to delete.
        let merged = merge_fn(existing.as_deref(), value)?;

        match merged {
            Some(new_value) => {
                let outcome = self.put_ns_locked(namespace_id, key, &new_value, Self::MERGE_OP_NAME)?;
                Self::require_applied(outcome)?;
                Ok(Some(new_value))
            }
            // Deleting an absent key would cost an fsync and a tombstone to say
            // nothing, so the absent -> absent case writes nothing at all.
            None if existing.is_some() => {
                let outcome = self.delete_ns_locked(namespace_id, key, Self::MERGE_OP_NAME)?;
                Self::require_applied(outcome)?;
                Ok(None)
            }
            None => Ok(None),
        }
    }

    /// Turn a durable-but-unapplied merge write into an error.
    ///
    /// The write is committed either way — this does not undo it, and the WAL
    /// entry replays on the next open. What it prevents is the caller carrying
    /// on as though the merge took effect, because the *next* merge on this key
    /// would then read the unchanged value and accumulate onto a stale base.
    fn require_applied(outcome: WriteOutcome) -> Result<()> {
        if outcome.applied {
            Ok(())
        } else {
            Err(KVError::MergeNotApplied { seq: outcome.seq })
        }
    }

    pub fn get_ns(&self, namespace_id: u32, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.check_closed()?;
        let kv_store = self.get_store(namespace_id)?;
        kv_store.get(key)
    }

    pub fn delete_ns(&self, namespace_id: u32, key: &[u8]) -> Result<()> {
        self.check_closed()?;
        // Outermost write lock — see `put_ns`.
        let _stripe = self.key_locks.guard(namespace_id, key);
        // Discarded, as in `put_ns`.
        self.delete_ns_locked(namespace_id, key, "delete").map(|_| ())
    }

    /// The body of [`Self::delete_ns`], with the caller holding the key stripe.
    /// See [`Self::put_ns_locked`] for why this split exists.
    fn delete_ns_locked(&self, namespace_id: u32, key: &[u8], op_name: &str) -> Result<WriteOutcome> {
        self.check_write_size(key, &[])?; // delete carries only a key

        // Step 1: Write DELETE to shared WAL.
        // Allocate the sequence inside the WAL lock so sequence order == WAL
        // append order (see `put_ns` for the rationale).
        let mut wal_metadata = self.wal_metadata.write();
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        assert!(seq != u64::MAX, "WAL global sequence number exhausted");
        let mut wal_entry = WalEntry::new_delete_ns(namespace_id, key.to_vec()).with_sequence(seq);
        if op_name == Self::MERGE_OP_NAME {
            wal_entry = wal_entry.with_op_name(op_name);
        }
        let wal_pointer = self.wal.append_entry(&wal_entry, &mut wal_metadata.tail, true)?;
        let segment_id = self.wal.segment_id_for_offset(wal_pointer.offset);
        wal_metadata.add_segment_total(segment_id, 1);
        wal_metadata.total_entries += 1;
        let wal_tail = wal_metadata.tail;
        // As in `put_ns`: registered under the WAL lock, released on return.
        // A lost delete is as damaging as a lost put — recovery skipping it
        // resurrects the key.
        let _in_flight = self.wal_flush_observer.begin_write(wal_pointer.offset);
        drop(wal_metadata);

        // As in `put_ns`: this namespace now holds an un-flushed WAL-backed write.
        self.wal_flush_observer.note_write(namespace_id, wal_tail);

        // Step 2: Apply the delete to the in-memory store (best-effort with
        // bounded retry; durable in the WAL — see `put_ns`).
        let kv_store = self.get_store(namespace_id)?;

        if let Some(m) = kv_store.metrics() {
            crate::db::metrics::Metrics::bump(&m.deletes);
            crate::db::metrics::Metrics::bump(&m.wal_fsyncs);
            crate::db::metrics::Metrics::add(&m.wal_bytes_appended, wal_pointer.size as u64 + 4);
        }

        let applied = Self::apply_with_retry(op_name, namespace_id, key, seq, || kv_store.delete_from_storage_seq(key, seq));
        if !applied && let Some(m) = kv_store.metrics() {
            crate::db::metrics::Metrics::bump(&m.apply_failures);
        }

        // Step 3: Maybe sync the value log. As in `put_ns`, syncing the value
        // log must NOT advance the WAL persisted watermark — the tombstone still
        // lives only in the memtable until it is flushed.
        if kv_store.should_sync() {
            let _ = kv_store.sync_value_log();
        }

        Ok(WriteOutcome { seq, applied })
    }

    // ── Namespace management (admin API) ───────────────────────────────

    /// Create a new namespace. Returns the namespace ID.
    pub fn create_namespace(&self, name: &str) -> Result<u32> {
        self.check_closed()?;

        // Held until this function returns, so no one can observe the window
        // between the registry publish below and the `stores` insert at the end.
        let _create = self.namespace_create_lock.lock();

        // Idempotent under the lock. Callers reach this through get-or-create
        // (`Db::namespace`), which checks for the name and creates it if absent —
        // two of them racing a brand-new name both see it absent, and without
        // this the loser would get a spurious "already exists" from the registry.
        if let Some(existing) = self.registry.read().get_id(name) {
            return Ok(existing);
        }

        let ns_id = self.registry.write().create(name)?;

        let ns_path = crate::db::layout::namespace_data_dir(&self.db_path, name);
        let kv_store = KVStore::open(
            ns_id,
            name,
            &ns_path,
            self.config.lsm_config.clone(),
            self.config.sync_config,
            self.config.segment_size_bytes,
        )?;
        kv_store.set_verify_checksums_on_read(self.config.verify_checksums_on_read);
        kv_store.set_seq_counter(self.next_seq.clone());
        // Per-namespace metrics: each store owns its own counters; the engine-wide
        // view is the sum of all stores' snapshots plus the global instance.
        kv_store.set_metrics(std::sync::Arc::new(crate::db::metrics::Metrics::default()));
        kv_store.set_index_gap_sink(Some(
            Arc::clone(&self.rejected_index_updates) as Arc<dyn crate::db::index_manager::IndexGapSink>
        ));

        // Wire up flush observer
        let observer: Arc<dyn LsmFlushObserver> = Arc::new(LsmFlushObserverHub::new(
            ns_id,
            Arc::clone(&self.wal_flush_observer),
            Arc::clone(&kv_store.lsm_compaction_trigger),
        ));
        kv_store.set_flush_observer(Some(observer));

        // Wire compaction trigger if a worker is already running
        if let Some(sender) = self.lsm_compaction_sender.read().as_ref() {
            kv_store.set_compaction_trigger(sender.clone());
        }
        // Inherit the index-checkpoint backpressure valve if the worker is running.
        if let Some(trigger) = self.index_checkpoint_trigger.read().as_ref() {
            kv_store.set_index_checkpoint_trigger(Some(Arc::clone(trigger)));
        }

        self.stores.write().insert(ns_id, Arc::new(kv_store));
        Ok(ns_id)
    }

    /// Create a new namespace with an optional TTL. Returns the namespace ID.
    /// The TTL is recorded on the store; the single global TTL worker (started by
    /// the async facade's `namespace_with_ttl`) expires records older than it.
    pub fn create_namespace_with_ttl(&self, name: &str, ttl: Option<Duration>) -> Result<u32> {
        self.check_closed()?;

        // Same contract as `create_namespace` — see its comments.
        let _create = self.namespace_create_lock.lock();
        if let Some(existing) = self.registry.read().get_id(name) {
            return Ok(existing);
        }

        let ns_id = self.registry.write().create(name)?;

        let ns_path = crate::db::layout::namespace_data_dir(&self.db_path, name);
        let kv_store = KVStore::open_with_ttl(
            ns_id,
            name,
            &ns_path,
            self.config.lsm_config.clone(),
            self.config.sync_config,
            self.config.segment_size_bytes,
            ttl,
        )?;
        kv_store.set_verify_checksums_on_read(self.config.verify_checksums_on_read);
        kv_store.set_seq_counter(self.next_seq.clone());
        // Per-namespace metrics: each store owns its own counters; the engine-wide
        // view is the sum of all stores' snapshots plus the global instance.
        kv_store.set_metrics(std::sync::Arc::new(crate::db::metrics::Metrics::default()));
        kv_store.set_index_gap_sink(Some(
            Arc::clone(&self.rejected_index_updates) as Arc<dyn crate::db::index_manager::IndexGapSink>
        ));

        // Wire up flush observer
        let observer: Arc<dyn LsmFlushObserver> = Arc::new(LsmFlushObserverHub::new(
            ns_id,
            Arc::clone(&self.wal_flush_observer),
            Arc::clone(&kv_store.lsm_compaction_trigger),
        ));
        kv_store.set_flush_observer(Some(observer));

        // Wire compaction trigger if a worker is already running
        if let Some(sender) = self.lsm_compaction_sender.read().as_ref() {
            kv_store.set_compaction_trigger(sender.clone());
        }
        // Inherit the index-checkpoint backpressure valve if the worker is running.
        if let Some(trigger) = self.index_checkpoint_trigger.read().as_ref() {
            kv_store.set_index_checkpoint_trigger(Some(Arc::clone(trigger)));
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
    ///    the registry (see `recover_from_wal`), so the
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
            // Fold this namespace's final counters into the global accumulator so
            // engine-wide aggregates stay monotonic after the per-namespace
            // metrics disappear with the store.
            if let Some(m) = store.metrics() {
                self.metrics.add_snapshot(&m.snapshot());
            }
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

        // (3b) Stop tracking its flush progress. The store is gone, so it can
        // never flush again — left in place it would report un-flushed writes
        // forever, pinning the global persisted watermark (and therefore WAL GC)
        // at a stale offset that nothing can ever advance.
        self.wal_flush_observer.forget_namespace(ns_id);

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
        let ns_path = crate::db::layout::namespace_data_dir(&self.db_path, name);
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
    /// `activate_field_index` is called, so type mismatches are caught at
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
    /// dense row-ID map.  Providing `row_to_key_fn` (the exact inverse) additionally
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
    /// `activate_field_index`).
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
    pub fn field_index_blob_stats(&self, namespace_id: u32, field_id: FieldId) -> Option<crate::index::IndexBlobStats> {
        let store = self.get_store(namespace_id).ok()?;
        let ns_index = store.namespace_index.read();
        ns_index.get(field_id).map(|e| e.index.read().blob_stats())
    }

    /// Reindex a single field for a single key: re-derive the field value from
    /// the key's current stored bytes and rewrite its entry in that field's
    /// index, using the same extractor + `DynFieldIndex` ops as the put path.
    /// Touches only the named field — no value rewrite, no other-field or vector
    /// re-indexing. See [`KVStore::reindex_field`].
    ///
    /// Returns `FieldReindexOutcome::FieldNotActive` when the namespace is not
    /// open or `field_id` has no live index there.
    pub fn reindex_field(&self, namespace_id: u32, field_id: FieldId, key: &[u8]) -> Result<FieldReindexOutcome> {
        let Ok(store) = self.get_store(namespace_id) else {
            return Ok(FieldReindexOutcome::FieldNotActive);
        };
        store.reindex_field(field_id, key)
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
    /// * `field_id`     – the [`FieldId`] returned by `register_index_field`
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
            // A dropped field's directory is gone (or is about to be), so
            // activating it would open an empty index and silently serve
            // incomplete results — `detect_replay_gap` suppresses exactly this
            // shape (`Absent` + empty) as a normal first build. Fail loudly and
            // make the caller re-register, which clears the flag and forces a
            // rebuild. See `FEATURE-REQUEST.md` (FR-001).
            if meta.dropped {
                return Err(KVError::Serialization(format!(
                    "Field '{}' (id {}) in namespace {} was dropped; re-register it before activating so the index is rebuilt",
                    meta.field_name, field_id, namespace_id
                )));
            }
        }

        let store = self.get_store(namespace_id)?;

        // Everything below this point creates or maps files under
        // `index/{ns_id}/`. Two threads doing that for the same field truncate
        // each other's live mappings and the process dies with SIGBUS — see
        // `index_activate_lock`. Taken after `get_store`, which may itself wait
        // on the strictly-outermost `namespace_create_lock`.
        let _activate = self.index_activate_lock.lock();

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
                let checkpoint_state = self.index_manager.read_checkpoint_state(namespace_id, field_id, wal_tail);
                let checkpoint_offset = checkpoint_state.replay_offset();
                if checkpoint_offset < wal_tail {
                    let wal_head = self.wal_metadata.read().head;

                    // Report — but do not act on — a replay window the WAL can no
                    // longer satisfy. WAL GC reclaims a segment once its entries are
                    // persisted to the *LSM*; it never consults these checkpoint
                    // markers, so the segments that would heal this field's index may
                    // already be gone. `scan_entries` skips a missing segment as a
                    // hole and returns what it can, silently, which is exactly how
                    // this stayed invisible. Checking segment *presence* rather than
                    // `checkpoint_offset < wal_head` matters: GC reclaims
                    // fully-persisted segments out of order, so a hole can sit in the
                    // middle of the window while `head` is still below the checkpoint.
                    //
                    // Detection only — the replay below is unchanged, and the missing
                    // updates are unrecoverable from here. Recording, surfacing and
                    // repairing this is tracked in `FEATURE-REQUEST.md` (FR-001).
                    if let Some(gap) = crate::db::index_manager::detect_replay_gap(
                        checkpoint_state,
                        dyn_index.distinct_count() == 0,
                        wal_tail,
                        self.wal.missing_segments(checkpoint_offset, wal_tail),
                    ) {
                        error!(
                            "[activate_field_index] ns={} field={}: field index is INCOMPLETE and cannot be repaired \
                             by WAL replay — {} WAL segment(s) covering the replay window [{}, {}) have been reclaimed \
                             (missing segment ids: {:?}, checkpoint marker: {:?}). Every write recorded in those \
                             segments is absent from this field index, so queries on it will return incomplete \
                             results until the index is rebuilt. Re-index this field to restore full coverage.",
                            namespace_id,
                            field_id,
                            gap.missing_segments.len(),
                            gap.from,
                            gap.to,
                            gap.missing_segments,
                            checkpoint_state,
                        );
                    }

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

                    // Bound the blob growth this replay produces.
                    //
                    // The bitmap store is append-only and `insert` rewrites a
                    // whole bitmap per key, so replaying a **low-cardinality**
                    // field is quadratic: a value shared by N keys leaves N-1
                    // stale copies. The write path bounds exactly this with the
                    // backpressure valve — but the valve signals the checkpoint
                    // *worker*, and the workers are not started until after every
                    // field has been activated. During replay the trigger is
                    // `None` and the valve is a no-op.
                    //
                    // Measured before this compaction existed: replaying one
                    // 5-distinct-value field over 16k documents reached 2.6 GB on
                    // disk and had not finished after ~1.5 hours. It also
                    // compounded — an interrupted replay leaves a bigger blob and
                    // does not advance the checkpoint, so the next start was
                    // worse. The database became unopenable.
                    //
                    // So compact inline, on the same `dead_bytes` cap the write
                    // path uses. We hold `dyn_index` exclusively here (it is not
                    // yet published behind the `RwLock`), so this needs no locks
                    // and cannot race a writer.
                    let waste_threshold = (self.config.threshold_config.index_blob_waste_threshold / 100.0).clamp(0.0, 1.0);
                    // A cap of 0 disables the *write-path* valve, which is a
                    // throughput choice. It must not disable this: replay is not
                    // the write path, and an unbounded replay leaves the database
                    // unopenable rather than merely slow.
                    let cap = match self.config.threshold_config.index_blob_backpressure_bytes {
                        0 => crate::db::config::DEFAULT_INDEX_BLOB_BACKPRESSURE_BYTES,
                        n => n,
                    };
                    let mut compactions = 0usize;

                    // Group the replay by VALUE before writing anything.
                    //
                    // `insert` re-serialises the whole bitmap for a value, and
                    // the blob store is append-only, so inserting key-by-key
                    // leaves one dead copy of the entire bitmap per key. For a
                    // low-cardinality field (few values, many rows) that is the
                    // dominant cost of replay — measured at ~0.1 ms per key and
                    // rising with the store's size, because the cost tracks the
                    // bitmap, not the window.
                    //
                    // Replay is the one caller that can avoid it: it knows its
                    // complete key set up front, so it can resolve every row
                    // first and then write each value's bitmap exactly once.
                    let mut rows_by_value: HashMap<crate::index::IndexValue, Vec<u128>> = HashMap::new();
                    let mut rows_to_clear: Vec<u128> = Vec::new();

                    for key in &affected_keys {
                        match store.get(key)? {
                            // Live key: resolve its dense ID and bucket it under
                            // the value its current bytes extract to.
                            Some(ref bytes) => {
                                let row_id = store.resolve_row_id_alloc(key);
                                rows_to_clear.push(row_id);
                                if let Some(v) = extractor(bytes) {
                                    rows_by_value.entry(v).or_default().push(row_id);
                                }
                            }
                            // Deleted key: clear any existing entry without
                            // allocating a (dead) ID for a key that no longer exists.
                            None => {
                                if let Some(row_id) = store.resolve_row_id_get(key) {
                                    rows_to_clear.push(row_id);
                                }
                            }
                        }
                    }

                    // Clear every affected row first, so the scalar
                    // one-value-per-row invariant holds no matter which bucket
                    // each row used to live in. This must precede the inserts:
                    // clearing afterwards would undo them.
                    //
                    // Batched for the same reason the inserts are: the per-row
                    // form loads every bucket and rewrites the changed one *per
                    // row*, so clearing N rows appends N copies of the bitmap.
                    dyn_index.remove_all_for_rows(&rows_to_clear);

                    for (value, row_ids) in rows_by_value {
                        if dyn_index.reclaimable_dead_bytes() >= cap {
                            dyn_index.flush(&field_path).map_err(KVError::Io)?;
                            if dyn_index.maybe_compact(waste_threshold).map_err(KVError::Io)? {
                                dyn_index.flush(&field_path).map_err(KVError::Io)?;
                                compactions += 1;
                            }
                        }
                        if let Err(e) = dyn_index.insert_many(&value, &row_ids) {
                            warn!(
                                "[activate_field_index] index update rejected \
                                 ns={} field={}: {}",
                                namespace_id, field_id, e
                            );
                        }
                    }
                    dyn_index.flush(&field_path).map_err(KVError::Io)?;
                    if compactions > 0 {
                        debug!(
                            "[activate_field_index] ns={namespace_id} field={field_id}: \
                             compacted the index blob {compactions} time(s) during replay to keep it bounded"
                        );
                    }
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
    /// that references it returns [`KVError::Query`] carrying
    /// [`crate::index::query::QueryError::InactiveField`].  The on-disk checkpoint
    /// files are left untouched; callers are responsible for removing them.
    pub fn deactivate_field_index(&self, namespace_id: u32, field_id: FieldId) -> Result<()> {
        self.get_store(namespace_id)?.namespace_index.write().deregister(field_id);
        Ok(())
    }

    /// Drop a field index: persist the drop, deregister it, and delete its
    /// on-disk directory.
    ///
    /// Unlike [`deactivate_field_index`](Self::deactivate_field_index) — which
    /// is an in-memory deregister a caller may legitimately use while a field is
    /// rebuilt — this is the permanent operation. The field's [`FieldMeta`] is
    /// retained in the schema so a later `register_field` reuses the same
    /// [`FieldId`], but it is marked `dropped`: excluded from the checkpoint
    /// worker and refused by `activate_field_index`.
    ///
    /// **Step order is load-bearing** (mirrors `remove_namespace`):
    ///
    /// 1. persist `dropped` to `config.json`,
    /// 2. deregister the in-memory index,
    /// 3. delete `index/{ns_id}/{field_id}/`.
    ///
    /// A crash anywhere in that sequence leaves the drop *recorded*, so
    /// [`complete_interrupted_field_drops`](Self::complete_interrupted_field_drops)
    /// finishes it at the next open. The reverse order — files first — would
    /// leave a live-looking field with no data, which reads as `Absent` + empty
    /// and is suppressed by `detect_replay_gap` as a normal first build: a
    /// silently incomplete index. See `FEATURE-REQUEST.md` (FR-001).
    ///
    /// Idempotent: dropping an already-dropped or unregistered field still
    /// deregisters and reclaims any leftover directory.
    pub fn drop_field_index(&self, namespace_id: u32, field_id: FieldId) -> Result<()> {
        // (1) Persist the drop before anything becomes unreachable.
        self.registry.write().mark_schema_field_dropped(namespace_id, field_id)?;

        // (2) Deregister in memory. Best-effort on the store lookup: a namespace
        // that is not open has no in-memory index to deregister, but its files
        // still need reclaiming below.
        if let Ok(store) = self.get_store(namespace_id) {
            store.namespace_index.write().deregister(field_id);
        }

        // (3) Reclaim disk.
        self.index_manager.remove_field_path(namespace_id, field_id)
    }

    /// Finish deleting the directories of field indices whose drop was
    /// interrupted by a crash, and reclaim any that were dropped before this
    /// cleanup existed.
    ///
    /// Runs at open, after the registry is loaded and before any field is
    /// activated. Idempotent — `remove_field_path` treats a missing directory as
    /// success — so the common case (nothing dropped, or every drop completed)
    /// costs one `remove_dir_all` per dropped field and touches no disk.
    ///
    /// Failures are logged, not propagated: a leftover directory wastes disk but
    /// cannot affect correctness, and refusing to open the database over it would
    /// be a far worse outcome.
    fn complete_interrupted_field_drops(&self) {
        let dropped: Vec<(u32, FieldId)> = {
            let registry = self.registry.read();
            registry
                .list()
                .into_iter()
                .filter_map(|(_, ns_id)| registry.schema(ns_id).map(|s| (ns_id, s.dropped_field_ids())))
                .flat_map(|(ns_id, fields)| fields.into_iter().map(move |fid| (ns_id, fid)))
                .collect()
        };
        for (ns_id, field_id) in dropped {
            match self.index_manager.remove_field_path(ns_id, field_id) {
                Ok(()) => debug!("[INDEX] reclaimed dropped field index ns={ns_id} field={field_id}"),
                Err(e) => warn!("[INDEX] failed to reclaim dropped field index ns={ns_id} field={field_id}: {e:?}"),
            }
        }
    }

    // ── Store access ───────────────────────────────────────────────────

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

    // ── WAL helpers ────────────────────────────────────────────────────

    fn check_closed(&self) -> Result<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(KVError::DatabaseClosed);
        }
        Ok(())
    }

    /// Reject a write whose key/value cannot be represented by the on-disk
    /// formats, with a clean [`KVError::WriteTooLarge`] instead of silently
    /// truncating a length deep in the store.
    ///
    /// **Two** bounds apply, and both must be checked here.
    ///
    /// 1. **Field widths.** Several persisted lengths are `u32`: the value
    ///    record's `value_len`, the row-ID map's `key_len`, and the whole write
    ///    is framed as one `u32`-sized WAL entry (key + value + op-name + rkyv
    ///    overhead). A single combined check — `key + value + headroom ≤
    ///    u32::MAX` — covers all three (key alone and value alone are subsets of
    ///    the entry). The headroom bounds the per-entry framing.
    /// 2. **Segment capacity.** A value record must fit inside one value-log
    ///    segment, whose size is `DbConfig::segment_size_bytes` (default 256
    ///    MiB, so *far* tighter than the ~4 GiB field-width bound).
    ///
    /// Checking only the first left a gap that produced an acknowledged write
    /// nobody could read: a value between the segment size and ~4 GiB passed
    /// here, got a fsynced WAL entry — so `put` returned `Ok` — and only then hit
    /// `ValueLogError::ValueTooLarge` inside `ValueLog::append`, where the write
    /// path is past its durability barrier and treats failures as best-effort.
    /// All three apply attempts failed identically (the value's size does not
    /// change between retries), `get` kept returning the previous value, and WAL
    /// replay hit the same error on the next open and wrote a fail log. **Both
    /// bounds must be enforced before the WAL append**, which is the only point
    /// at which a rejection can still mean "nothing happened".
    fn check_write_size(&self, key: &[u8], value: &[u8]) -> Result<()> {
        // Generous headroom for WAL-entry framing (short op-name + rkyv) and the
        // value-record header — far larger than any of those.
        const HEADROOM: u64 = 64 * 1024;
        const FIELD_WIDTH_LIMIT: u64 = u32::MAX as u64 - HEADROOM;

        Self::check_total_len(key.len(), value.len(), FIELD_WIDTH_LIMIT, "lengths are stored as u32")?;

        let segment_size = self.config.segment_size_bytes;
        Self::check_total_len(
            key.len(),
            value.len(),
            crate::store::value_log::max_record_payload(segment_size),
            &format!("a record must fit one value-log segment; value_log.segment_size_bytes = {segment_size}"),
        )
    }

    /// Inner of [`check_write_size`] with the limit as a parameter so both
    /// branches are testable without allocating a multi-GiB key/value.
    /// `reason` names which bound was hit, since the two have different fixes.
    fn check_total_len(key_len: usize, value_len: usize, limit: u64, reason: &str) -> Result<()> {
        let total = key_len as u64 + value_len as u64;
        if total > limit {
            return Err(KVError::WriteTooLarge(format!(
                "key {key_len} + value {value_len} = {total} bytes exceeds the {limit}-byte limit ({reason})"
            )));
        }
        Ok(())
    }

    /// Flush every namespace holding no-WAL writes that exist only in memory.
    ///
    /// WAL-backed writes survive a crash because recovery replays them; no-WAL
    /// writes have no such copy, so whatever sits in the memtable when the
    /// process dies is gone. That is the intended trade for the vector index and
    /// the query-embedding cache — both are bulky and re-derivable — but a
    /// memtable is only flushed at capacity, so in practice a crash discarded
    /// *everything* those namespaces had accumulated. Observed: a `SIGKILL` left
    /// `stress_docs_sparse_vector` with 4 entries on disk against 6020 in
    /// memory, and reconciliation re-enqueued ~18000 documents whose embeddings
    /// had already been computed — hours of work against the embedding service.
    ///
    /// Running this on the compaction worker's tick bounds that loss to one tick
    /// of writes instead of a whole memtable. Only namespaces with no-WAL writes
    /// are flushed: WAL-backed ones are recoverable, are already flushed by
    /// [`flush_namespaces_pinning_wal`], and forcing extra flushes on them would
    /// buy no durability while adding L0 files for compaction to merge.
    ///
    /// Returns how many namespaces were flushed.
    ///
    /// [`flush_namespaces_pinning_wal`]: Self::flush_namespaces_pinning_wal
    pub(crate) fn flush_no_wal_memtables(&self) -> usize {
        let stores = self.stores.read();
        let mut flushed = 0;
        for (ns_id, kv_store) in stores.iter() {
            if !kv_store.has_unflushed_no_wal_writes() {
                continue;
            }
            match kv_store.flush_memtable_to_level0() {
                Ok(()) => flushed += 1,
                Err(e) => warn!("[LSM] Failed to flush no-WAL memtable for ns={}: {:?}", ns_id, e),
            }
        }
        flushed
    }

    /// Flush every namespace that is holding the WAL persisted watermark back.
    ///
    /// The watermark can only advance to the slowest namespace's flushed offset,
    /// so a namespace that writes a few records and then goes idle would pin the
    /// whole WAL forever. Flushing those memtables to level 0 is what lets the
    /// watermark — and WAL GC behind it — move again. Returns how many
    /// namespaces were flushed.
    pub(crate) fn flush_namespaces_pinning_wal(&self) -> usize {
        let pinning = self.wal_flush_observer.namespaces_with_unflushed();
        if pinning.is_empty() {
            return 0;
        }
        let stores = self.stores.read();
        let mut flushed = 0;
        for ns_id in pinning {
            let Some(kv_store) = stores.get(&ns_id) else {
                continue;
            };
            match kv_store.flush_memtable_to_level0() {
                Ok(()) => flushed += 1,
                Err(e) => warn!("[WAL] Failed to flush ns={} while unpinning the WAL watermark: {:?}", ns_id, e),
            }
        }
        flushed
    }

    pub(crate) fn flush_wal_metadata_internal(&self) -> Result<()> {
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

    // ── Value log GC (per namespace) ───────────────────────────────────

    /// Run value log GC on a specific namespace.
    ///
    /// Note the threshold passed here is the **per-segment** selection one, not the
    /// bucket-level trigger: an explicit `garbage_collect()` call has already decided
    /// the namespace is worth collecting, so what is left to decide is which *segments*
    /// to rewrite.
    pub fn garbage_collect_namespace(&self, namespace_id: u32) -> Result<GCStats> {
        let kv_store = self.get_store(namespace_id)?;
        let segment_threshold = self.config.threshold_config.segment_gc_threshold;
        let tail_threshold = self.config.threshold_config.effective_tail_gc_min_garbage_pct();
        kv_store.garbage_collect_with_thresholds(segment_threshold, tail_threshold)
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
            segment_count: 0,
            disk_bytes: 0,
            garbage_size: 0,
            waste_ratio: 0.0,
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
            .filter_map(|(name, ns_id)| stores.get(&ns_id).map(|s| (name, s.value_log.all_bucket_metadata())))
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

    /// Per-segment live/garbage breakdown for one namespace's value-log shards.
    ///
    /// Cheap: the counters are maintained in memory by writers, so this reads no
    /// segment data at all (the old per-page version had to scan every record).
    pub fn value_log_segment_stats(&self, namespace: &str) -> Result<Vec<(u32, Vec<crate::store::value_log::SegmentStats>)>> {
        let store = self.get_store_by_name(namespace)?;
        Ok(store.value_log.all_segment_stats())
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
    ///
    /// Counters are recorded per namespace, so the engine-wide view is the sum
    /// of every live namespace's snapshot plus the global instance — which holds
    /// the WAL-GC counters (not attributable to one namespace) and the folded
    /// totals of every dropped namespace (so the aggregate stays monotonic).
    pub fn metrics_snapshot(&self) -> crate::db::metrics::MetricsSnapshot {
        let mut agg = self.metrics.snapshot();
        for store in self.stores.read().values() {
            if let Some(m) = store.metrics() {
                agg.accumulate(&m.snapshot());
            }
        }
        agg
    }

    /// Operational counters for a single namespace, by name.
    ///
    /// Note: the WAL-GC counters (`wal_gc_runs`, `wal_segments_deleted`) belong
    /// to the shared WAL and are not attributed to any namespace — they are
    /// always `0` here and only populated in [`Self::metrics_snapshot`].
    pub fn metrics_snapshot_for(&self, name: &str) -> Result<crate::db::metrics::MetricsSnapshot> {
        let ns_id = self
            .registry
            .read()
            .get_id(name)
            .ok_or_else(|| KVError::Serialization(format!("Namespace '{}' not found", name)))?;
        let stores = self.stores.read();
        let store = stores
            .get(&ns_id)
            .ok_or_else(|| KVError::Serialization(format!("Namespace '{}' not found", name)))?;
        Ok(store.metrics().map(|m| m.snapshot()).unwrap_or_default())
    }

    /// Per-namespace operational counters for every live namespace, keyed by
    /// namespace name. Excludes the global WAL-GC/retired accumulator.
    pub fn metrics_snapshot_by_namespace(&self) -> Vec<(String, crate::db::metrics::MetricsSnapshot)> {
        let registry = self.registry.read();
        let mut out: Vec<(String, crate::db::metrics::MetricsSnapshot)> = self
            .stores
            .read()
            .iter()
            .filter_map(|(ns_id, store)| {
                let name = registry.get_name(*ns_id)?.to_owned();
                Some((name, store.metrics().map(|m| m.snapshot()).unwrap_or_default()))
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
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

        // Record the forced size so a later production `open` of this dir honours
        // it rather than re-resolving to the default (mirrors `open`).
        let _ = crate::support::write_atomic_durable(&db_path.join("wal_segment_size"), &wal_segment_size.to_le_bytes());

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
            let ns_path = crate::db::layout::namespace_data_dir(db_path, name);
            let kv_store = KVStore::open(
                ns_id,
                name,
                &ns_path,
                config.lsm_config.clone(),
                config.sync_config,
                config.segment_size_bytes,
            )?;
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
            namespace_create_lock: Mutex::new(()),
            index_activate_lock: Mutex::new(()),
            stores: RwLock::new(stores),
            closed: Arc::new(AtomicBool::new(false)),
            wal_gc_worker: Arc::new(tokio::sync::RwLock::new(None)),
            lsm_compaction_worker: Arc::new(tokio::sync::RwLock::new(None)),
            value_log_gc_worker: Arc::new(tokio::sync::RwLock::new(None)),
            lsm_compaction_sender: Arc::new(parking_lot::RwLock::new(None)),
            index_checkpoint_trigger: Arc::new(parking_lot::RwLock::new(None)),
            rejected_index_updates: Arc::new(crate::db::index_manager::RejectedUpdateBuffer::default()),
            no_wal_pending: parking_lot::Mutex::new(std::collections::HashSet::new()),
            ttl_worker: Arc::new(tokio::sync::RwLock::new(None)),
            index_manager,
            index_checkpoint_worker: Arc::new(tokio::sync::RwLock::new(None)),
            next_seq: Arc::new(AtomicU64::new(next_seq_start)),
            key_locks: KeyLocks::new(),
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
                kv_store.set_metrics(std::sync::Arc::new(crate::db::metrics::Metrics::default()));
                kv_store.set_index_gap_sink(Some(
                    Arc::clone(&db.rejected_index_updates) as Arc<dyn crate::db::index_manager::IndexGapSink>
                ));
            }
        }

        db.complete_interrupted_field_drops();

        // Report field indices left incomplete by no-WAL writes that an unclean
        // shutdown caught before any checkpoint covered them.
        db.record_no_wal_gaps_after_unclean_shutdown();
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
        // reproducible from the row map). See `crate::index::RowMap` durability docs.
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
        drop(stores);

        // The index is now durable up to `wal_tail`, so any no-WAL writes before
        // this point are covered and their markers can go.
        for ns_id in self.no_wal_pending.lock().drain() {
            if let Err(e) = self.index_manager.clear_no_wal_pending(ns_id) {
                warn!("[IndexCheckpoint] failed to clear the no-WAL marker for ns={ns_id}: {e:?}");
            }
        }

        self.persist_rejected_update_gaps();
        Ok(active_fields.len())
    }
}

// ── WalGcTarget impl ──────────────────────────────────────────────────

impl WalGcTarget for Database {
    fn is_closed(&self) -> bool {
        self.is_closed()
    }

    fn flush_namespaces_pinning_wal(&self) -> usize {
        self.flush_namespaces_pinning_wal()
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

    fn flush_no_wal_memtables(&self) -> usize {
        self.flush_no_wal_memtables()
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

    /// `waste_threshold` decides **whether** a namespace is collected; the separate,
    /// lower `segment_gc_threshold` then decides **which sealed segments** get rewritten.
    /// Passing the trigger through as the selection threshold (as this used to) makes
    /// garbage sitting just under it uncollectable: those segments read as "clean", are
    /// left in place with their garbage intact, and keep the bucket over its trigger
    /// forever.
    fn run_gc_if_needed(&self, waste_threshold: f64) {
        let stores = self.stores.read();
        let segment_threshold = self.config.threshold_config.segment_gc_threshold;
        let tail_threshold = self.config.threshold_config.effective_tail_gc_min_garbage_pct();
        // Fires on every tick whether or not there is anything to collect, so
        // DEBUG. The per-namespace results below stay at INFO.
        debug!(
            "[GCWorker] tick — checking {} namespace(s) against {:.2}% waste threshold \
             (segments rewritten at >= {:.2}% garbage, tail sealed at >= {:.2}%)",
            stores.len(),
            waste_threshold,
            segment_threshold,
            tail_threshold
        );
        for (ns_id, kv_store) in stores.iter() {
            let waste_ratio = kv_store.get_waste_ratio();
            // The namespace-average waste is over the trigger, OR some single bucket is —
            // the average alone hides a hot bucket behind near-empty ones, so a bucket well
            // over the trigger would never be collected. Check both.
            if waste_ratio < waste_threshold && !kv_store.has_bucket_over_waste(waste_threshold) {
                debug!(
                    "[GCWorker] ns_id={} waste {:.2}% below threshold (no bucket over it), skipping",
                    ns_id, waste_ratio
                );
                continue;
            }
            // Over the trigger, but is any of that garbage actually collectable? Garbage in
            // a bucket's active tail is not, until the tail is either filled or sealed — so
            // without this check a small, fully-deleted namespace would sit at 100% waste
            // and make the worker log "starting GC ... reclaimed 0 bytes" on every tick.
            if !kv_store.has_gc_work(segment_threshold, tail_threshold) {
                debug!(
                    "[GCWorker] ns_id={} waste {:.2}% is over the trigger but no segment is collectable yet, skipping",
                    ns_id, waste_ratio
                );
                continue;
            }
            let (garbage_bytes, written_bytes) = kv_store.waste_bytes();
            info!(
                "[GCWorker] ns_id={} waste {:.2}% ({} garbage / {} written bytes; waste = garbage / (live + garbage)) \
                 exceeds threshold {:.2}%, starting GC",
                ns_id, waste_ratio, garbage_bytes, written_bytes, waste_threshold
            );
            let start = std::time::Instant::now();
            match kv_store.garbage_collect_with_thresholds(segment_threshold, tail_threshold) {
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

/// Unit tests for the coordinator.
///
/// Declared here with `#[path]` rather than as a sibling module so `use super::*`
/// still resolves to this file's scope — the tests were written against it, and
/// re-deriving 40 imports to move them would have been change for its own sake.
#[cfg(test)]
#[path = "database_tests.rs"]
mod tests;
