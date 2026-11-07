//! Creating, listing and removing namespaces.

use crate::db::database::{Database, LsmFlushObserverHub, OpenStoreLookup};
use crate::db::error::Result;
use crate::db::kv_store::KVStore;
use crate::storage::lsm::tree::LsmFlushObserver;

use log::warn;
use std::sync::Arc;
use std::time::Duration;

impl Database {
    /// Create a new namespace. Returns the namespace ID.
    pub fn create_namespace(&self, name: &str) -> Result<u32> {
        self.create_namespace_with_ttl(name, None)
    }

    /// Create a new namespace with an optional TTL. Returns the namespace ID.
    /// The TTL is recorded on the store; the single global TTL worker (started by
    /// the async facade's `namespace_with_ttl`) expires records older than it.
    pub fn create_namespace_with_ttl(&self, name: &str, ttl: Option<Duration>) -> Result<u32> {
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
        kv_store.set_metrics(std::sync::Arc::new(crate::support::metrics::Metrics::default()));
        kv_store.set_index_gap_sink(Some(
            Arc::clone(&self.rejected_index_updates) as Arc<dyn crate::db::indexing::manager::IndexGapSink>
        ));

        // Wire up flush observer
        let observer: Arc<dyn LsmFlushObserver> = Arc::new(LsmFlushObserverHub::new(
            ns_id,
            Arc::clone(&self.wal_flush_observer),
            Arc::clone(&self.vector_index_wal),
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

    /// `name`'s id and open store, from the registry and the store map alone:
    /// two in-memory reads that never wait. The one state it cannot answer is a
    /// namespace whose name is published but whose store `create_namespace` is
    /// still opening (or `remove_namespace` is taking down); that is
    /// [`OpenStoreLookup::NotReady`], and [`get_store`](Self::get_store) resolves
    /// it by waiting on `namespace_create_lock`.
    pub(crate) fn lookup_open_store(&self, name: &str) -> OpenStoreLookup {
        let Some(ns_id) = self.get_namespace_id(name) else {
            return OpenStoreLookup::Absent;
        };
        match self.stores.read().get(&ns_id) {
            Some(store) => OpenStoreLookup::Open(ns_id, Arc::clone(store)),
            None => OpenStoreLookup::NotReady,
        }
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
    ///    and `{db_path}/field-index/{ns_id}` — never the shared WAL, the WAL metadata,
    ///    or the registry — so WAL recovery for the surviving namespaces is
    ///    completely unaffected.
    pub fn remove_namespace(&self, name: &str) -> Result<u32> {
        self.check_closed()?;

        // Held to the end. Removing a namespace `create_namespace` is still
        // opening found no store to close, deleted the directory under the open,
        // and the create then published a store the registry does not list. A
        // create of the same name after step 1 opens the same path, so this also
        // has to cover the deletion in step 4.
        let _create = self.namespace_create_lock.lock();

        // (1) Durable point: once this returns, the namespace is gone from the
        // persisted registry (including its TTL config, so the global TTL worker
        // stops scanning it) and recovery will skip its WAL entries.
        let ns_id = self.registry.write().remove(name)?;

        // (2) Flush + close, then drop so the data directory's file handles are
        // released before we delete it.
        // On its own line: in an `if let` scrutinee the write guard would live to the
        // end of the block, and every namespace lookup in the database would wait out
        // this store's flush and compaction.
        let removed = self.stores.write().remove(&ns_id);
        if let Some(store) = removed {
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

        // (3) Stop tracking its flush progress. The store is gone, so it can
        // never flush again — left in place it would report un-flushed writes
        // forever, pinning the cut (and therefore WAL GC) at a stale offset that
        // nothing can ever advance. Its entries above the cut need no flush:
        // recovery skips entries of a namespace missing from the registry.
        self.wal_flush_observer.forget_namespace(ns_id);
        // Likewise for the vector index WAL, whose replay skips the dropped
        // namespace's records.
        self.vector_index_wal.forget_namespace(ns_id);

        // (4) Reclaim disk. Best-effort and independent of the WAL.
        self.remove_namespace_storage(ns_id, name);

        Ok(ns_id)
    }

    /// Delete the data directory of `name` left by an interrupted drop. A name
    /// the registry lists is left alone: its directory holds a live store's files.
    pub fn remove_unregistered_namespace_dir(&self, name: &str) -> Result<()> {
        self.check_closed()?;
        // The lock keeps a create of `name` from opening the directory meanwhile.
        let _create = self.namespace_create_lock.lock();
        if self.registry.read().exists(name) {
            return Ok(());
        }
        match std::fs::remove_dir_all(crate::db::layout::namespace_data_dir(&self.db_path, name)) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
            _ => Ok(()),
        }
    }

    /// Delete a dropped namespace's on-disk files: its data directory
    /// (`{db_path}/ns_{name}`, holding the LSM SSTables and value log) and its
    /// index subtree (`{db_path}/field-index/{ns_id}`).
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
}
