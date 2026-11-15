//! What the LSM's flushes tell the WAL: `WalPersistObserver` tracks each namespace's
//! unflushed entries and moves the cut; `LsmFlushObserverHub` fans the events out.

use super::*;
use crate::namespace::NamespaceId;

pub(super) struct WalFlushState {
    tail: u64,
    pub(super) flushed: bool,
}

/// How far one namespace's writes have made it onto disk.
///
/// The WAL is global but memtables are per-namespace, so "this entry is durable
/// in an SSTable" is a *per-namespace* question and the global watermark can only
/// be the slowest namespace's answer.
#[derive(Default, Clone, Copy)]
pub(super) struct NsFlushProgress {
    /// WAL offset up to which every write by this namespace is in an SSTable.
    pub(super) safe_offset: u64,
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

/// Observes LSM flush events and keeps the WAL's retirement progress: each
/// namespace's safe offset and the cut, saved in the WAL metadata.
/// Shared across all namespaces since the WAL is global.
///
/// Progress moves on a flush's *written* event, when its L0 files are durable
/// but not yet registered, and takes effect only once the metadata recording it
/// is on disk ([`Self::prepare_written`]). So recovery and WAL GC never
/// act on progress a crash could undo, and the saved per-namespace offsets are
/// durable before L1 compaction can merge the files they describe.
pub(crate) struct WalPersistObserver {
    wal: Arc<Wal>,
    wal_metadata: Arc<RwLock<WalMetadata>>,
    wal_metadata_path: PathBuf,
    /// Keyed by `(namespace_id, memtable version)`. Memtable versions are
    /// allocated **per KVStore**, so keying by version alone made namespace A's
    /// version 7 and namespace B's version 7 the same entry — A's flush would
    /// then satisfy B's pending record and advance the cut past B's
    /// un-flushed writes.
    pending: Arc<RwLock<BTreeMap<(NamespaceId, u64), WalFlushState>>>,
    /// The database's sequence counter, whose high-water mark every metadata
    /// save records: open reads the log only from the cut, and this covers the
    /// sequences below it.
    next_seq: Arc<AtomicU64>,
    /// Per-namespace flush progress — the input to the cut.
    pub(super) ns_progress: RwLock<std::collections::HashMap<NamespaceId, NsFlushProgress>>,
    /// Serialises every change to the saved progress and every save of the WAL
    /// metadata file, so two saves cannot land in the opposite order (older
    /// progress overwriting newer). Held across the file write; never taken
    /// while holding `pending`, `ns_progress`, `wal_metadata` or `in_flight`.
    ///
    /// **Lock ordering: `persist_lock` → `pending` → `ns_progress` →
    /// `wal_metadata` → `in_flight`.**
    persist_lock: Mutex<()>,
    /// The field indexes that need the WAL, each from its last checkpoint's
    /// position (the other kind of WAL consumer, beside the namespaces'
    /// memtables). `Some(p)`: the field replays from `p` after a crash. `None`:
    /// it has no checkpoint yet, and holds the WAL from 0 once its index has
    /// entries (a first build is populated from the stored documents, not by
    /// replay, so an empty one holds nothing). Kept current by
    /// [`Self::fields_checkpointed`]; a field is added when it is activated and
    /// removed when it is deactivated or dropped, or its namespace is.
    field_consumers: RwLock<BTreeMap<(NamespaceId, FieldId), Option<u64>>>,
    /// Test hook: outcomes for the next metadata saves, `true` = fail.
    #[cfg(test)]
    save_failures: Mutex<std::collections::VecDeque<bool>>,
    /// Start offsets of writes appended to the WAL but not yet applied to a
    /// memtable — the writes for which "it is in the WAL" and "it is in the LSM"
    /// disagree *right now*.
    ///
    /// **Why this exists.** Every quantity that drives the persisted cut is
    /// derived from a `wal_metadata.tail` snapshot, and a tail snapshot
    /// over-states durability: `put_ns` appends under the WAL lock (advancing
    /// the tail) and applies to the memtable *after* releasing it, so between
    /// those two points an entry is visible in the tail while its key is
    /// nowhere. Moving the cut past such an entry tells recovery to skip a write
    /// that only ever existed in a memtable — and a crash before that memtable
    /// flushes loses an acknowledged write (a stress run without this lost about
    /// 1 write in 35,000).
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
    pub(super) in_flight: Mutex<BTreeMap<u64, u32>>,
}

/// Registers a WAL entry as appended-but-not-yet-applied for its lifetime.
///
/// RAII is load-bearing, not stylistic: `put_ns` has fallible steps between the
/// WAL append and the memtable apply (`get_store`), and a leaked registration
/// pins [`WalPersistObserver::wal_cut_ceiling`] forever — which freezes the
/// cut, so WAL GC never reclaims another segment and the WAL grows without
/// bound, as a dropped namespace that kept holding the cut would.
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
    pub(super) fn new(
        wal: Arc<Wal>,
        wal_metadata: Arc<RwLock<WalMetadata>>,
        wal_metadata_path: PathBuf,
        pending: Arc<RwLock<BTreeMap<(NamespaceId, u64), WalFlushState>>>,
        next_seq: Arc<AtomicU64>,
    ) -> Self {
        Self {
            wal,
            wal_metadata,
            wal_metadata_path,
            pending,
            next_seq,
            ns_progress: RwLock::new(std::collections::HashMap::new()),
            persist_lock: Mutex::new(()),
            field_consumers: RwLock::new(BTreeMap::new()),
            #[cfg(test)]
            save_failures: Mutex::new(std::collections::VecDeque::new()),
            in_flight: Mutex::new(BTreeMap::new()),
        }
    }

    /// Test hook: script the outcome of the next metadata saves, in order
    /// (`true` = the save fails).
    #[cfg(test)]
    pub(crate) fn script_saves(&self, outcomes: &[bool]) {
        self.save_failures.lock().extend(outcomes.iter().copied());
    }

    /// `Err(WalPoisoned)` once the main WAL has been stopped: by a failed
    /// fsync (`Wal::commit`), or by a failed rollback of saved progress
    /// ([`PreparedProgress::abort`]), after which the metadata on disk may claim
    /// entries are in L0 that are not. Reopening runs recovery, which rebuilds
    /// the truth from the log and L0.
    pub(crate) fn check_poisoned(&self) -> Result<()> {
        self.wal.check_poisoned().map_err(wal_write_error)
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
    /// * [`Self::compute_cut`], whose cut would otherwise be the live
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
    pub(super) fn wal_cut_ceiling(&self) -> u64 {
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
    pub(crate) fn namespaces_with_unflushed(&self) -> Vec<NamespaceId> {
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
    pub(crate) fn note_write(&self, namespace_id: NamespaceId, tail: u64) {
        let mut progress = self.ns_progress.write();
        let entry = progress.entry(namespace_id).or_default();
        entry.last_write_offset = entry.last_write_offset.max(tail);
    }

    /// Field `(namespace_id, field_id)` is active and needs the WAL from
    /// `position` (see `field_consumers`).
    pub(crate) fn add_field_consumer(&self, namespace_id: NamespaceId, field_id: FieldId, position: Option<u64>) {
        self.field_consumers.write().insert((namespace_id, field_id), position);
    }

    /// The field is no longer active: it stops holding the WAL.
    pub(crate) fn remove_field_consumer(&self, namespace_id: NamespaceId, field_id: FieldId) {
        self.field_consumers.write().remove(&(namespace_id, field_id));
    }

    /// The *checkpointed* event: every field in `fields` has a durable
    /// checkpoint marker at `position`. Reported only after the markers are on
    /// disk; reported early, WAL GC could delete a segment a crash would send
    /// the field back to. A field no longer active is not re-added.
    pub(crate) fn fields_checkpointed(&self, fields: &[(NamespaceId, FieldId)], position: u64) {
        let mut consumers = self.field_consumers.write();
        for key in fields {
            if let Some(p) = consumers.get_mut(key) {
                *p = Some(p.map_or(position, |old| old.max(position)));
            }
        }
    }

    /// Every field consumer and its position.
    pub(crate) fn field_consumers(&self) -> Vec<((NamespaceId, FieldId), Option<u64>)> {
        self.field_consumers.read().iter().map(|(&k, &v)| (k, v)).collect()
    }

    /// Stop tracking a namespace that has been dropped.
    ///
    /// A dropped namespace's store is gone from the registry, so it can never
    /// flush again and its `safe_offset` can never advance. Left in
    /// `ns_progress` it reports [`has_unflushed`](NsFlushProgress::has_unflushed)
    /// forever, which pins the cut at its stale offset — and
    /// [`Database::flush_namespaces_pinning_wal`] cannot clear it, because the
    /// store it would flush no longer exists: unbounded WAL growth after any
    /// namespace drop.
    ///
    /// Forgetting it is sound because recovery skips entries for namespaces
    /// missing from the registry, which `remove_namespace` has already updated.
    /// The new cut is saved; a failed save only leaves the saved cut behind,
    /// which costs replay, not data.
    pub(crate) fn forget_namespace(&self, namespace_id: NamespaceId) {
        self.field_consumers.write().retain(|(ns, _), _| *ns != namespace_id);
        let _persist = self.persist_lock.lock();
        self.ns_progress.write().remove(&namespace_id);
        // Seal records for memtables that will now never be flushed; left
        // behind they would be folded into a future namespace's scan.
        self.pending.write().retain(|(ns, _), _| *ns != namespace_id);
        let progress = self.ns_progress.read().clone();
        if let Err(e) = self.commit_progress(&progress) {
            warn!("[WAL] Failed to save WAL progress after dropping ns={namespace_id}: {e:?}");
        }
    }

    /// The cut `progress` implies: the lowest safe offset among namespaces with
    /// un-flushed writes, capped at the oldest in-flight write, and never below
    /// the cut already saved.
    ///
    /// The WAL is shared, so the cut is the **minimum** over the namespaces that
    /// still hold un-flushed writes; one namespace's own progress says nothing
    /// about another's un-flushed entries. Namespaces with nothing outstanding
    /// do not constrain it; if none do, everything written so far is in L0 and
    /// the cut is the ceiling.
    ///
    /// The ceiling excludes writes that are in the WAL but not yet in any
    /// memtable. It is applied to BOTH branches: capping the `Some` branch costs
    /// nothing (a namespace's `safe_offset` is already below any later write)
    /// and removes the need to re-derive that argument every time this is read.
    /// Without it, the `None` branch cut at the live tail, which is how a store
    /// drop lost acknowledged writes: `forget_namespace` recomputes the cut the
    /// instant it removes a namespace, which is precisely when the set of
    /// namespaces reporting un-flushed writes can go empty while other
    /// namespaces are mid-write.
    fn compute_cut(&self, progress: &std::collections::HashMap<NamespaceId, NsFlushProgress>) -> u64 {
        let slowest = progress.values().filter(|p| p.has_unflushed()).map(|p| p.safe_offset).min();
        let ceiling = self.wal_cut_ceiling();
        let cut = match slowest {
            Some(offset) => offset.min(ceiling),
            None => ceiling,
        };
        cut.max(self.wal_metadata.read().cut)
    }

    /// The metadata `progress` implies: the current metadata with its cut and
    /// per-namespace offsets replaced.
    fn progress_snapshot(&self, progress: &std::collections::HashMap<NamespaceId, NsFlushProgress>) -> WalMetadata {
        let cut = self.compute_cut(progress);
        let mut safe: Vec<(NamespaceId, u64)> = progress
            .iter()
            .filter(|(_, p)| p.safe_offset > cut)
            .map(|(&ns, p)| (ns, p.safe_offset))
            .collect();
        safe.sort_unstable();
        let mut snapshot = self.wal_metadata.read().clone();
        snapshot.cut = cut;
        snapshot.safe = safe;
        snapshot
    }

    /// Make a saved snapshot's progress the live one.
    fn apply_progress(&self, snapshot: &WalMetadata) {
        let mut live = self.wal_metadata.write();
        live.cut = snapshot.cut;
        live.safe = snapshot.safe.clone();
    }

    /// Save the progress `progress` implies, then make it live. Caller holds
    /// `persist_lock`. On a failed save nothing changes in memory: recovery and
    /// WAL GC read the in-memory `cut` / `safe`, and they must never act on
    /// progress that a crash could undo.
    fn commit_progress(&self, progress: &std::collections::HashMap<NamespaceId, NsFlushProgress>) -> std::io::Result<()> {
        let snapshot = self.progress_snapshot(progress);
        self.write_metadata(&snapshot)?;
        self.apply_progress(&snapshot);
        Ok(())
    }

    /// Write `metadata` to the metadata file, durably. Caller holds
    /// `persist_lock`.
    fn write_metadata(&self, metadata: &WalMetadata) -> std::io::Result<()> {
        #[cfg(test)]
        if self.save_failures.lock().pop_front() == Some(true) {
            return Err(std::io::Error::other("injected WAL metadata save failure"));
        }
        // Never a tail beyond what is fsynced: recovery scans forward from the
        // saved tail, and a tail past a hole a crash left would make it append
        // after the hole and skip acknowledged writes behind it.
        let mut metadata = metadata.clone();
        // Every sequence handed out so far (an over-estimate is harmless).
        metadata.last_sequence = metadata.last_sequence.max(self.next_seq.load(Ordering::Acquire).saturating_sub(1));
        let bytes = metadata.to_file_bytes();
        crate::support::write_atomic_durable(&self.wal_metadata_path, &bytes)?;
        // The live copy shows what is saved (`Db::wal_stats`).
        let mut live = self.wal_metadata.write();
        live.last_sequence = live.last_sequence.max(metadata.last_sequence);
        Ok(())
    }

    /// Save the WAL metadata as it is (counters, head, tail), serialised with
    /// every progress save.
    pub(crate) fn save_metadata(&self) -> std::io::Result<()> {
        let _persist = self.persist_lock.lock();
        let snapshot = self.wal_metadata.read().clone();
        self.write_metadata(&snapshot)
    }

    /// Retire everything written so far: cut at the ceiling, no per-namespace
    /// offsets. Only for when every namespace's writes are known to be in L0
    /// (clean shutdown after flushing every store, the end of recovery).
    pub(crate) fn retire_all(&self) -> std::io::Result<()> {
        let _persist = self.persist_lock.lock();
        let mut snapshot = self.wal_metadata.read().clone();
        snapshot.cut = self.wal_cut_ceiling().max(snapshot.cut);
        snapshot.safe.clear();
        self.write_metadata(&snapshot)?;
        let mut live = self.wal_metadata.write();
        live.cut = snapshot.cut;
        live.safe.clear();
        drop(live);
        for p in self.ns_progress.write().values_mut() {
            p.safe_offset = p.safe_offset.max(snapshot.cut);
        }
        Ok(())
    }

    pub(super) fn on_memtable_sealed_ns(&self, namespace_id: NamespaceId, version: u64) {
        // The ceiling, not the raw tail. This record means "every WAL entry below
        // `tail` is in the memtable being sealed", and that is false for a write
        // whose apply has not landed yet: its apply will go into the *next*
        // memtable, while its offset is below this tail. Recording the raw tail
        // therefore let the written event push `safe_offset` past an entry that
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

    /// `namespace_id`'s memtable `version` has its L0 files durable, not yet
    /// registered. Fold the namespace's completed flushes (in version order)
    /// into its safe offset, recompute the cut and **save** both, but do not make
    /// them live yet: the flush can still be undone by another observer of the
    /// same event (the vector index WAL), which deletes the L0 files. The
    /// returned [`PreparedProgress`] holds `persist_lock` until it is committed
    /// or aborted.
    ///
    /// An error means the save failed: nothing changed, and the LSM undoes the
    /// flush (nothing registered, the memtable kept for a retry).
    pub(super) fn prepare_written(&self, namespace_id: NamespaceId, version: u64) -> std::io::Result<PreparedProgress<'_>> {
        // Same reasoning as the seal path, for a flush observed without a
        // preceding seal record.
        let ceiling = self.wal_cut_ceiling();
        let guard = self.persist_lock.lock();
        let previous = self.wal_metadata.read().clone();
        let (fresh, folded, new_safe) = {
            let mut pending = self.pending.write();
            let fresh = !pending.contains_key(&(namespace_id, version));
            let entry = pending
                .entry((namespace_id, version))
                .or_insert(WalFlushState { tail: 0, flushed: false });
            if entry.tail == 0 {
                entry.tail = ceiling;
            }
            entry.flushed = true;
            // Versions are consumed in order and only while flushed, so an
            // out-of-order flush cannot advance the namespace past a
            // still-pending older memtable.
            let mut folded = Vec::new();
            let mut new_safe = 0u64;
            for (&key, state) in pending.range((namespace_id, 0)..=(namespace_id, u64::MAX)) {
                if !state.flushed {
                    break;
                }
                folded.push(key);
                new_safe = new_safe.max(state.tail);
            }
            (fresh, folded, new_safe)
        };

        let mut progress = self.ns_progress.read().clone();
        if new_safe > 0 {
            let entry = progress.entry(namespace_id).or_default();
            entry.safe_offset = entry.safe_offset.max(new_safe);
        }
        let snapshot = self.progress_snapshot(&progress);
        let prepared = PreparedProgress {
            observer: self,
            _guard: guard,
            namespace_id,
            version,
            fresh,
            folded,
            new_safe,
            previous,
            snapshot,
        };
        if let Err(e) = self.write_metadata(&prepared.snapshot) {
            prepared.withdraw();
            return Err(e);
        }
        Ok(prepared)
    }
}

/// A flush's progress, saved by [`WalPersistObserver::prepare_written`] but not
/// yet live. Holds `persist_lock`, so no other save can land in between.
pub(crate) struct PreparedProgress<'a> {
    observer: &'a WalPersistObserver,
    _guard: parking_lot::MutexGuard<'a, ()>,
    namespace_id: NamespaceId,
    version: u64,
    /// The flush had no seal record before this event.
    fresh: bool,
    /// Pending seal records this flush completes, in version order.
    folded: Vec<(NamespaceId, u64)>,
    new_safe: u64,
    /// The metadata before this flush, re-saved by [`Self::abort`].
    previous: WalMetadata,
    snapshot: WalMetadata,
}

impl PreparedProgress<'_> {
    /// The flush is registered: make its progress live.
    pub(super) fn commit(self) {
        let o = self.observer;
        {
            let mut pending = o.pending.write();
            for key in &self.folded {
                pending.remove(key);
            }
        }
        if self.new_safe > 0 {
            let mut live = o.ns_progress.write();
            let entry = live.entry(self.namespace_id).or_default();
            entry.safe_offset = entry.safe_offset.max(self.new_safe);
        }
        o.apply_progress(&self.snapshot);
    }

    /// The flush is being undone and its L0 files deleted: put the saved
    /// progress back. If that save fails too, the file on disk may claim
    /// entries are in L0 that will not be, so the main WAL is stopped.
    fn abort(self) {
        if let Err(e) = self.observer.write_metadata(&self.previous) {
            self.observer.wal.poison(format!(
                "could not restore the WAL progress after ns={} memtable {} failed to flush: {e}",
                self.namespace_id, self.version
            ));
        }
        self.withdraw();
    }

    /// Take back the flush mark this event set.
    fn withdraw(self) {
        let mut pending = self.observer.pending.write();
        if self.fresh {
            pending.remove(&(self.namespace_id, self.version));
        } else if let Some(state) = pending.get_mut(&(self.namespace_id, self.version)) {
            state.flushed = false;
        }
    }
}

/// A main-WAL error on the write path: a poisoned log becomes
/// [`KVError::WalPoisoned`], an entry too large for a segment
/// [`KVError::WriteTooLarge`], the rest pass through.
pub(super) fn wal_write_error(e: WalError) -> KVError {
    match e {
        WalError::Log(LogError::Poisoned(why)) => KVError::WalPoisoned(why),
        // Refused before anything was written: the same kind of error as the
        // other write-size bounds (`check_write_size`).
        WalError::Log(LogError::RecordTooLarge { len, max }) => KVError::WriteTooLarge(format!(
            "the WAL entry is {len} bytes, more than a WAL segment holds ({max}); raise wal.segment_size_bytes"
        )),
        other => KVError::from(other),
    }
}

/// Hub that fans out LSM flush events to the WAL observer, the vector index
/// WAL and the compaction trigger.
///
/// One hub per KVStore, which is what supplies the namespace id the shared
/// [`WalPersistObserver`] and [`VectorIndexWal`] need: the LSM itself only knows
/// memtable versions, and those are allocated per store. Every namespace's hub
/// reports to the vector index WAL; it ignores namespaces that never appended.
pub(in crate::db) struct LsmFlushObserverHub {
    namespace_id: NamespaceId,
    wal_observer: Arc<WalPersistObserver>,
    vector_index_wal: Arc<VectorIndexWal>,
    compaction_trigger: Arc<RwLock<Option<tokio::sync::mpsc::UnboundedSender<LsmCompactionCommand>>>>,
}

impl LsmFlushObserverHub {
    pub(in crate::db) fn new(
        namespace_id: NamespaceId,
        wal_observer: Arc<WalPersistObserver>,
        vector_index_wal: Arc<VectorIndexWal>,
        compaction_trigger: Arc<RwLock<Option<tokio::sync::mpsc::UnboundedSender<LsmCompactionCommand>>>>,
    ) -> Self {
        Self {
            namespace_id,
            wal_observer,
            vector_index_wal,
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
        self.vector_index_wal.on_memtable_sealed(self.namespace_id, version);
        self.trigger_compaction();
    }

    // Before any L0 file is written: a namespace with records in the vector
    // index WAL past its durable tail syncs it first, so no L0 file can hold a
    // logged write the log could still lose.
    fn on_ro_memtable_flushing(&self, _version: u64) -> std::io::Result<()> {
        self.vector_index_wal.before_memtable_written(self.namespace_id).map_err(|e| match e {
            crate::db::wal::vector_index_wal::VectorIndexWalError::Log(LogError::Io(io)) => io,
            other => std::io::Error::other(other.to_string()),
        })
    }

    // Before the memtable's L0 files are registered, and through
    // `SyncValueLogBeforePersist`, so the values the retired records point at
    // are fsynced first. Both logs save their progress here; an error from
    // either undoes the flush, deleting its L0 files. See `LsmFlushObserver`
    // for the order of events.
    //
    // The main WAL saves first and makes its progress live only once the
    // vector index WAL has succeeded too: had it gone live, the undo would
    // leave it claiming entries are in L0 files that were just deleted. If the
    // vector index WAL fails, the main WAL puts its saved progress back.
    fn on_ro_memtable_written(&self, version: u64) -> std::io::Result<()> {
        let prepared = self.wal_observer.prepare_written(self.namespace_id, version)?;
        match self.vector_index_wal.on_memtable_written(self.namespace_id, version) {
            Ok(()) => {
                prepared.commit();
                Ok(())
            }
            Err(e) => {
                prepared.abort();
                Err(match e {
                    crate::db::wal::vector_index_wal::VectorIndexWalError::Log(LogError::Io(io)) => io,
                    other => std::io::Error::other(other.to_string()),
                })
            }
        }
    }

    fn on_ro_memtable_flushed_to_level0(&self, _version: u64) {}
}
