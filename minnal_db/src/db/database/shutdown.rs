//! Closing a database: `shutdown` and `Drop`.

use super::*;

impl Database {
    pub fn shutdown(&self) -> Result<()> {
        if self.closed.swap(true, Ordering::SeqCst) {
            return Err(KVError::DatabaseClosed);
        }

        // Every step runs even if an earlier one failed (a failed shutdown
        // cannot be retried: the database is already closed); the first error
        // is returned.
        fn note(first: &mut Option<KVError>, step: &str, r: Result<()>) {
            if let Err(e) = r {
                error!("[SHUTDOWN] {step} failed: {e:?}");
                first.get_or_insert(e);
            }
        }
        let mut first_error: Option<KVError> = None;

        // Shutdown all KVStores
        let stores = self.stores.read();
        // Flush first, with the observers still attached, so the vector index
        // WAL retires what reaches L0 — its marker is written before the files
        // are registered. Each store's shutdown then compacts into L1, which
        // drops tombstones; with the observers already detached, a stale marker
        // would make the next open replay records whose keys were deleted
        // (`a_clean_shutdown_does_not_bring_back_deleted_keys`). The main WAL
        // sees these flushes too; it marks everything persisted below anyway.
        let mut flush_failed = std::collections::HashSet::new();
        for (ns_id, kv_store) in stores.iter() {
            if let Err(e) = kv_store.flush_memtable_to_level0() {
                flush_failed.insert(*ns_id);
                note(&mut first_error, &format!("flushing ns={ns_id}"), Err(e));
            }
        }
        for (ns_id, kv_store) in stores.iter() {
            // A store whose flush failed keeps its observers: its shutdown
            // flushes again before compacting, and that flush must be seen by
            // the vector index WAL (which vetoes it if its marker cannot be
            // written), so nothing it holds is merged unrecorded.
            if !flush_failed.contains(ns_id) {
                kv_store.set_flush_observer(None);
            }
            note(&mut first_error, &format!("shutting down ns={ns_id}"), kv_store.shutdown());
        }
        drop(stores);
        // An entry may be marked persisted only once its key is in an SSTable.
        let stores_on_disk = first_error.is_none();

        // Flush all active field indices (keymap + bitmap data) to disk.
        // This persists in-memory index state accumulated during the session,
        // including entries written via put_no_wal which have no WAL to replay.
        note(&mut first_error, "the index checkpoint", self.run_index_checkpoint().map(|_| ()));

        // Every store is flushed, so every WAL entry is in L0: retire the whole
        // log (this also saves the WAL metadata). Otherwise just save it, and the
        // next open replays what the flush did not cover.
        if stores_on_disk {
            note(
                &mut first_error,
                "retiring the WAL",
                self.wal_flush_observer.retire_all().map_err(KVError::from),
            );
        } else {
            note(&mut first_error, "writing the WAL metadata", self.flush_wal_metadata_internal());
        }

        // Sync WAL
        note(&mut first_error, "syncing the WAL", self.wal.sync().map_err(KVError::from));

        first_error.map_or(Ok(()), Err)
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
        for kv_store in stores.values() {
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
