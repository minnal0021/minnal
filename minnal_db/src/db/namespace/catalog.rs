//! The `Database` side of the system catalog: opening its store first,
//! replaying its WAL records before anything else, loading the registry from
//! it, and writing every registry change to it.
//!
//! The record layout is in `crate::namespace::catalog`.

use std::path::Path;
use std::sync::Arc;

use log::info;

use crate::db::config::DbConfig;
use crate::db::database::Database;
use crate::db::error::Result;
use crate::kv_store::KVStore;
use crate::namespace::catalog::{CATALOG_NAMESPACE, CatalogWrite};
use crate::namespace::{NamespaceId, NamespaceRegistry};

impl Database {
    /// Open one namespace's store from its directory, with the settings every
    /// store shares.
    pub(in crate::db) fn open_store_files(
        db_path: &Path,
        config: &DbConfig,
        ns_id: NamespaceId,
        name: &str,
        ttl: Option<std::time::Duration>,
    ) -> Result<KVStore> {
        let kv_store = KVStore::open_with_ttl(
            ns_id,
            name,
            &crate::layout::namespace_data_dir(db_path, name),
            config.lsm_config.clone(),
            config.sync_config,
            config.segment_size_bytes,
            ttl,
        )?;
        kv_store.set_verify_checksums_on_read(config.verify_checksums_on_read);
        Ok(kv_store)
    }

    /// Open the catalog's store: a fixed name and id, so it opens before
    /// anything is read.
    pub(in crate::db) fn open_catalog_store(db_path: &Path, config: &DbConfig) -> Result<KVStore> {
        Self::open_store_files(db_path, config, NamespaceId::CATALOG, CATALOG_NAMESPACE, None)
    }

    /// Share the engine's sequence counter, a store's own counters and the
    /// index-gap sink with a store opened at open time. Must happen before
    /// recovery replays into it.
    fn prepare_opened_store(&self, kv_store: &KVStore) {
        kv_store.set_seq_counter(self.next_seq.clone());
        kv_store.set_metrics(Arc::new(crate::support::metrics::Metrics::default()));
        kv_store.set_index_gap_sink(Some(
            Arc::clone(&self.rejected_index_updates) as Arc<dyn crate::kv_store::index_set::IndexGapSink>
        ));
    }

    /// The rest of opening, shared by every constructor, once the database
    /// holds the catalog's store and nothing else:
    ///
    /// 1. replay the catalog's records of the main WAL;
    /// 2. load the registry from the catalog, and open every namespace's store;
    /// 3. finish interrupted field drops, note no-WAL gaps, and recover both
    ///    WALs into the other namespaces;
    /// 4. wire the flush observers, then save what loading left pending (a new
    ///    database's first records, repairs).
    ///
    /// The pending writes wait for step 4: recovery retires the whole log when
    /// it is done, which is only right while every write it covers has been
    /// replayed and flushed, and the observers are what track a write made
    /// after it.
    pub(in crate::db) fn finish_open(self) -> Result<Self> {
        if let Some(catalog) = self.stores.read().get(&NamespaceId::CATALOG) {
            self.prepare_opened_store(catalog);
        }
        let catalog_replayed = self.recover_catalog_from_wal()?;
        self.load_registry()?;
        self.open_registered_stores()?;
        {
            let registry = self.registry.read();
            for ns in self.vector_index_wal.tracked_namespaces() {
                // Replay skips a dropped namespace's records, so they must not
                // hold the retired cut.
                if registry.get_name(ns).is_none() {
                    self.vector_index_wal.forget_namespace(ns);
                }
            }
        }
        // Finish any field-index drop a crash interrupted, before recovery can
        // replay into a directory that is meant to be gone.
        self.complete_interrupted_field_drops();
        // Report field indices left incomplete by no-WAL writes that an unclean
        // shutdown caught before any checkpoint covered them.
        self.record_no_wal_gaps_after_unclean_shutdown();
        self.recover_from_vector_index_wal()?;
        self.recover_from_wal(catalog_replayed)?;
        self.wire_up_flush_observers();
        let pending = self.registry.write().take_writes();
        if !pending.is_empty() {
            self.write_catalog(pending)?;
            self.flush_catalog();
        }
        Ok(self)
    }

    /// Replace the registry with the one the catalog describes.
    fn load_registry(&self) -> Result<()> {
        let catalog = self.get_store(NamespaceId::CATALOG)?;
        let registry = NamespaceRegistry::from_catalog(catalog.scan_prefix_batch(&[])?)?;
        info!("[catalog] {} namespaces", registry.list().len());
        *self.registry.write() = registry;
        Ok(())
    }

    /// Open the store of every namespace the registry lists (the catalog's is
    /// open already), each with its saved TTL.
    fn open_registered_stores(&self) -> Result<()> {
        let namespaces: Vec<(String, NamespaceId, Option<std::time::Duration>)> = {
            let registry = self.registry.read();
            registry
                .list()
                .into_iter()
                .filter(|&(_, id)| id != NamespaceId::CATALOG)
                .map(|(name, id)| (name.to_owned(), id, registry.ttl_config(id).map(|(ttl, _)| ttl)))
                .collect()
        };
        for (name, ns_id, ttl) in namespaces {
            let kv_store = Self::open_store_files(&self.db_path, &self.config, ns_id, &name, ttl)?;
            self.prepare_opened_store(&kv_store);
            self.stores.write().insert(ns_id, Arc::new(kv_store));
        }
        Ok(())
    }

    /// Change the registry: run `change` on a copy, write the catalog records
    /// it produced (each durable on return, in order), and only then swap the
    /// copy in. A change that fails, or whose writes fail, leaves the registry
    /// as it was, so the registry is always what the catalog holds.
    ///
    /// The registry's write lock is held throughout, so changes are applied in
    /// the order they are saved. It never waits for a lock the catalog write
    /// takes while holding one of its own: the catalog's store is always in
    /// `stores`, so the write does not read the registry.
    pub(crate) fn update_registry<R>(&self, change: impl FnOnce(&mut NamespaceRegistry) -> Result<R>) -> Result<R> {
        let (out, wrote) = {
            let mut registry = self.registry.write();
            let mut next = registry.clone();
            let out = change(&mut next)?;
            let writes = next.take_writes();
            let wrote = !writes.is_empty();
            self.write_catalog(writes)?;
            *registry = next;
            (out, wrote)
        };
        if wrote {
            self.flush_catalog();
        }
        Ok(out)
    }

    /// Write `writes` to the catalog, in order, each WAL-backed and durable on
    /// return.
    fn write_catalog(&self, writes: Vec<CatalogWrite>) -> Result<()> {
        for w in writes {
            match w {
                CatalogWrite::Put { key, value } => self.put_ns(NamespaceId::CATALOG, &key, &value)?,
                CatalogWrite::Delete { key } => self.delete_ns(NamespaceId::CATALOG, &key)?,
            }
        }
        Ok(())
    }

    /// Flush the catalog's memtable to L0, so the catalog never holds back the
    /// main WAL: its few writes would otherwise sit in a memtable that rarely
    /// fills, keeping their WAL segments until WAL GC flushes them, and every
    /// open would replay them first. Best-effort: the writes are durable in the
    /// WAL already, and WAL GC flushes a namespace holding the cut.
    fn flush_catalog(&self) {
        #[cfg(test)]
        if self.catalog_flush_paused.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        if let Ok(catalog) = self.get_store(NamespaceId::CATALOG)
            && let Err(e) = catalog.flush_memtable_to_level0()
        {
            log::warn!("[catalog] flushing the system catalog failed; WAL GC will flush it: {e:?}");
        }
    }
}
