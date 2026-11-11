//! Reads and writes through the WAL, and the no-WAL and vector-log forms of each.

use super::*;
use crate::namespace::NamespaceId;

impl Database {
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

    /// Number of attempts to apply a WAL-durable write to the in-memory store
    /// before giving up. Once the WAL fsync has succeeded the write is durable
    /// and *will* take effect (via the in-memory apply now, or via WAL replay on
    /// the next open), so this only bounds the retry of transient apply errors.
    pub(super) const APPLY_RETRY_ATTEMPTS: usize = 3;

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
    pub(super) fn apply_with_retry<F>(op_name: &str, namespace_id: NamespaceId, key: &[u8], seq: u64, mut apply: F) -> bool
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

    pub fn put_ns(&self, namespace_id: NamespaceId, key: &[u8], value: &[u8]) -> Result<()> {
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
    fn put_ns_locked(&self, namespace_id: NamespaceId, key: &[u8], value: &[u8], op_name: &str) -> Result<WriteOutcome> {
        self.check_write_size(key, value)?;
        self.wal_flush_observer.check_poisoned()?;

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
        let wal_pointer = self.wal.append_entry(&wal_entry, &mut wal_metadata.tail).map_err(wal_write_error)?;
        let wal_tail = wal_metadata.tail;
        // Registered BEFORE the WAL lock is released, so no observer can see a
        // tail containing this entry without also seeing that it is not yet in
        // any memtable. Held until this function returns, by which point step 2
        // has applied it (or failed loudly). See `WalPersistObserver::in_flight`.
        let _in_flight = self.wal_flush_observer.begin_write(wal_pointer.offset);
        drop(wal_metadata);

        // This namespace now has a WAL-backed write that is not yet in an
        // SSTable, so it holds the cut back until it flushes (see
        // `WalPersistObserver::compute_cut`).
        self.wal_flush_observer.note_write(namespace_id, wal_tail);

        // Wait for an fsync that covers the entry (group commit), outside the
        // WAL lock so other writers keep appending meanwhile. Only then is it
        // applied: a reader must never see a write a crash could still undo.
        self.commit_wal(wal_tail)?;

        // Step 2: Apply to the namespace's in-memory store. The write is already
        // durable in the WAL, so this is best-effort with bounded retry: on
        // persistent failure we log and still return Ok, because recovery will
        // replay the entry on the next open. Surfacing an error here would be
        // misleading — the data is already committed.
        let kv_store = self.get_store(namespace_id)?;

        // Record write counters against this namespace's metrics (the store is
        // already in hand, so this is no extra lookup).
        if let Some(m) = kv_store.metrics() {
            crate::support::metrics::Metrics::bump(&m.puts);
            crate::support::metrics::Metrics::bump(&m.wal_writes);
            crate::support::metrics::Metrics::add(
                &m.wal_bytes_appended,
                wal_pointer.size as u64 + u64::from(crate::db::wal::segmented_log::FRAME_HEADER_LEN),
            );
        }

        let applied = Self::apply_with_retry(op_name, namespace_id, key, seq, || kv_store.put_to_storage_seq(key, value, seq));
        if !applied && let Some(m) = kv_store.metrics() {
            crate::support::metrics::Metrics::bump(&m.apply_failures);
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
        // the memtable flush (the seal, then `prepare_written` on the written
        // event).
        if kv_store.should_sync() {
            let _ = kv_store.sync_value_log();
        }

        Ok(WriteOutcome { seq, applied })
    }

    /// Wait until the main WAL is durable up to `end` (`Wal::commit`), as a
    /// commit round's leader or follower, and count it.
    fn commit_wal(&self, end: u64) -> Result<()> {
        let delay = Duration::from_micros(self.config.sync_config.commit_delay_us);
        let led = self.wal.commit(end, delay).map_err(wal_write_error)?;
        crate::support::metrics::Metrics::bump(&self.metrics.wal_commit_writes);
        if led {
            crate::support::metrics::Metrics::bump(&self.metrics.wal_commit_rounds);
        }
        Ok(())
    }

    /// Write a key-value pair **without** appending to the WAL.
    ///
    /// This skips the fsync that normally accompanies every WAL entry, giving
    /// much higher throughput at the cost of crash safety: any data written
    /// via this method is unrecoverable if the process crashes before the
    /// value-log segment is flushed.  Only use this for bulk-loading scenarios
    /// where re-running the load is acceptable.
    pub fn put_ns_no_wal(&self, namespace_id: NamespaceId, key: &[u8], value: &[u8]) -> Result<()> {
        self.check_closed()?;
        // Outermost write lock — see `put_ns`. No-WAL writes take it too, so a
        // merge is serialised against bulk loading as well as against `put`.
        let _stripe = self.key_locks.guard(namespace_id, key);
        self.put_ns_no_wal_locked(namespace_id, key, value)
    }

    /// The body of [`Self::put_ns_no_wal`], for a caller already holding the
    /// key's stripe ([`Self::merge_ns_no_wal`]).
    fn put_ns_no_wal_locked(&self, namespace_id: NamespaceId, key: &[u8], value: &[u8]) -> Result<()> {
        self.check_write_size(key, value)?;
        let kv_store = self.get_store(namespace_id)?;
        if let Some(m) = kv_store.metrics() {
            crate::support::metrics::Metrics::bump(&m.no_wal_puts);
        }
        // No WAL entry exists for this write, so until the memtable is flushed
        // it lives only in memory — the background flusher uses this to bound
        // how much a crash can destroy.
        kv_store.note_no_wal_write();
        self.note_no_wal_index_exposure(namespace_id);
        kv_store.put_to_storage(key, value)
    }

    /// Write a key-value pair logged in the **vector index WAL** instead of the
    /// main WAL: no main-WAL append, no per-write fsync. The write is durable
    /// once a later [`Self::sync_vector_index_wal`] returns; after a crash,
    /// recovery replays it with its original seq unless the memtable holding it
    /// was already written to L0.
    ///
    /// For the vector worker's payloads (chunk codes, whole-document codes) and
    /// partition maintenance's per-document writes: one durability barrier per
    /// batch rather than per write. Order, as on every write path: key stripe,
    /// seq, append (registered as not yet applied), memtable apply, release.
    pub fn put_ns_logged(&self, namespace_id: NamespaceId, key: &[u8], value: &[u8]) -> Result<()> {
        self.check_closed()?;
        self.check_write_size(key, value)?;
        let _stripe = self.key_locks.guard(namespace_id, key);
        let kv_store = self.get_store(namespace_id)?;
        let seq = kv_store.alloc_seq();
        let logged = self.vector_index_wal.append(namespace_id, key, value, seq)?;
        // The main WAL never sees this write, so a field index on the namespace
        // could not replay it from there (FR-001's no-WAL marker).
        self.note_no_wal_index_exposure(namespace_id);
        // Applied with the logged seq; one retry, as the WAL path's apply. On a
        // second failure the caller gets the error and does not complete its
        // work (the vector worker redoes the entry). The record stays in the
        // log but is in no memtable: it replays only if the database stops
        // before the namespace's next seal, which otherwise retires it.
        kv_store
            .put_to_storage_seq(key, value, seq)
            .or_else(|_| kv_store.put_to_storage_seq(key, value, seq))?;
        drop(logged);
        Ok(())
    }

    /// Delete a key logged in the **vector index WAL** instead of the main WAL:
    /// the counterpart of [`Self::put_ns_logged`], durable once a later
    /// [`Self::sync_vector_index_wal`] returns. Replay applies it with its
    /// original seq, as the main WAL replays a delete.
    pub fn delete_ns_logged(&self, namespace_id: NamespaceId, key: &[u8]) -> Result<()> {
        self.check_closed()?;
        self.check_write_size(key, &[])?;
        let _stripe = self.key_locks.guard(namespace_id, key);
        let kv_store = self.get_store(namespace_id)?;
        let seq = kv_store.alloc_seq();
        let logged = self.vector_index_wal.append_delete(namespace_id, key, seq)?;
        self.note_no_wal_index_exposure(namespace_id);
        // One retry, as `put_ns_logged`.
        kv_store
            .delete_from_storage_seq(key, seq)
            .or_else(|_| kv_store.delete_from_storage_seq(key, seq))?;
        drop(logged);
        Ok(())
    }

    /// Make every write [`Self::put_ns_logged`] returned from before this call
    /// durable: one fsync of the vector index WAL (none if nothing was appended
    /// since the last).
    pub fn sync_vector_index_wal(&self) -> Result<()> {
        Ok(self.vector_index_wal.sync()?)
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
    pub fn delete_ns_no_wal(&self, namespace_id: NamespaceId, key: &[u8]) -> Result<()> {
        self.check_closed()?;
        // Outermost write lock — see `put_ns`.
        let _stripe = self.key_locks.guard(namespace_id, key);
        self.delete_ns_no_wal_locked(namespace_id, key)
    }

    /// The body of [`Self::delete_ns_no_wal`], for a caller already holding the
    /// key's stripe ([`Self::merge_ns_no_wal`]).
    fn delete_ns_no_wal_locked(&self, namespace_id: NamespaceId, key: &[u8]) -> Result<()> {
        self.check_write_size(key, &[])?;
        let kv_store = self.get_store(namespace_id)?;
        if let Some(m) = kv_store.metrics() {
            crate::support::metrics::Metrics::bump(&m.no_wal_deletes);
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
    /// (see `require_applied`).
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
    /// TTL expiry takes the same stripe and deletes by compare-and-set, so a
    /// merge that rewrites a key about to expire keeps its result; the next TTL
    /// pass judges the new value on its own epoch.
    pub fn merge_ns<F>(&self, namespace_id: NamespaceId, key: &[u8], value: &[u8], merge_fn: F) -> Result<Option<Vec<u8>>>
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
            crate::support::metrics::Metrics::bump(&m.merges);
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

    /// [`Self::merge_ns`] whose resolved write skips the WAL: the no-WAL
    /// counterpart of `merge_ns`, as [`Self::put_ns_no_wal`] and
    /// [`Self::delete_ns_no_wal`] are of `put_ns` and `delete_ns`.
    ///
    /// The read, the closure and the write run under the key's stripe, exactly
    /// as in `merge_ns`, so the result is atomic against every other write to the
    /// key; the closure contract, the abort and the absent → absent no-op are the
    /// same. What differs is durability: the resolved put or delete appends no
    /// WAL entry and costs no fsync, so it is lost if the process crashes before
    /// the namespace's memtable is flushed (the LSM worker flushes no-WAL writes
    /// on its next tick). It still draws a seq from the global counter, so it
    /// resolves against concurrent writes to the key by seq.
    ///
    /// Only for data whose lost write repairs itself — the vector worker's queue
    /// completions, where a lost removal brings the entry back and the document
    /// is embedded again. Like the other no-WAL writes it is never replayed into
    /// a field index (see [`Self::put_ns_no_wal`]).
    pub fn merge_ns_no_wal<F>(&self, namespace_id: NamespaceId, key: &[u8], value: &[u8], merge_fn: F) -> Result<Option<Vec<u8>>>
    where
        F: FnOnce(Option<&[u8]>, &[u8]) -> Result<Option<Vec<u8>>>,
    {
        self.check_closed()?;
        // Held across the read, the closure and the write: see `merge_ns`.
        let _stripe = self.key_locks.guard(namespace_id, key);

        let kv_store = self.get_store(namespace_id)?;
        if let Some(m) = kv_store.metrics() {
            crate::support::metrics::Metrics::bump(&m.merges);
        }

        let existing = kv_store.get(key)?;
        let merged = merge_fn(existing.as_deref(), value)?;

        match merged {
            Some(new_value) => {
                self.put_ns_no_wal_locked(namespace_id, key, &new_value)?;
                Ok(Some(new_value))
            }
            // As in `merge_ns`: an absent key stays absent with no tombstone.
            None if existing.is_some() => {
                self.delete_ns_no_wal_locked(namespace_id, key)?;
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

    pub fn get_ns(&self, namespace_id: NamespaceId, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.check_closed()?;
        let kv_store = self.get_store(namespace_id)?;
        kv_store.get(key)
    }

    pub fn delete_ns(&self, namespace_id: NamespaceId, key: &[u8]) -> Result<()> {
        self.check_closed()?;
        // Outermost write lock — see `put_ns`.
        let _stripe = self.key_locks.guard(namespace_id, key);
        // Discarded, as in `put_ns`.
        self.delete_ns_locked(namespace_id, key, "delete").map(|_| ())
    }

    /// The body of [`Self::delete_ns`], with the caller holding the key stripe.
    /// See [`Self::put_ns_locked`] for why this split exists.
    fn delete_ns_locked(&self, namespace_id: NamespaceId, key: &[u8], op_name: &str) -> Result<WriteOutcome> {
        self.check_write_size(key, &[])?; // delete carries only a key
        self.wal_flush_observer.check_poisoned()?;

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
        let wal_pointer = self.wal.append_entry(&wal_entry, &mut wal_metadata.tail).map_err(wal_write_error)?;
        let wal_tail = wal_metadata.tail;
        // As in `put_ns`: registered under the WAL lock, released on return.
        // A lost delete is as damaging as a lost put — recovery skipping it
        // resurrects the key.
        let _in_flight = self.wal_flush_observer.begin_write(wal_pointer.offset);
        drop(wal_metadata);

        // As in `put_ns`: this namespace now holds an un-flushed WAL-backed
        // write, and the entry is applied only once an fsync covers it.
        self.wal_flush_observer.note_write(namespace_id, wal_tail);
        self.commit_wal(wal_tail)?;

        // Step 2: Apply the delete to the in-memory store (best-effort with
        // bounded retry; durable in the WAL — see `put_ns`).
        let kv_store = self.get_store(namespace_id)?;

        if let Some(m) = kv_store.metrics() {
            crate::support::metrics::Metrics::bump(&m.deletes);
            crate::support::metrics::Metrics::bump(&m.wal_writes);
            crate::support::metrics::Metrics::add(
                &m.wal_bytes_appended,
                wal_pointer.size as u64 + u64::from(crate::db::wal::segmented_log::FRAME_HEADER_LEN),
            );
        }

        let applied = Self::apply_with_retry(op_name, namespace_id, key, seq, || kv_store.delete_from_storage_seq(key, seq));
        if !applied && let Some(m) = kv_store.metrics() {
            crate::support::metrics::Metrics::bump(&m.apply_failures);
        }

        // Step 3: Maybe sync the value log. As in `put_ns`, syncing the value
        // log must NOT advance the WAL persisted watermark — the tombstone still
        // lives only in the memtable until it is flushed.
        if kv_store.should_sync() {
            let _ = kv_store.sync_value_log();
        }

        Ok(WriteOutcome { seq, applied })
    }
}
