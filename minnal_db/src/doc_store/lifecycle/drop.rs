//! Dropping a store: move it to `Deleting`, wait out its writes, remove its
//! namespaces, its index files and its maintenance records, then its record.

use super::super::*;
use crate::namespace::NamespaceId;
use crate::namespace_lifecycle::{Governor, Task, TaskDone};

impl DocStore {
    /// Destroy a document store completely, and return once it is gone.
    ///
    /// Moves the store to [`NamespaceState::Deleting`] first, durably: from then on
    /// it refuses every request but the admin views. Then its writes in progress
    /// are waited out, its namespaces, index files and maintenance records are
    /// removed, and its record last. A store already being dropped (a drop a
    /// crash interrupted, or another caller's) is finished. A store still being
    /// created cannot be dropped ([`DocStoreError::StoreCreating`]), unless its
    /// creation was given up on (it is parked): the drop then removes what the
    /// creation made, and frees the name.
    ///
    /// Fails with [`DocStoreError::GovernorUnavailable`] when the governor is
    /// not running, before anything changes. This is irreversible.
    pub async fn remove(&self, namespace: &str) -> Result<(), DocStoreError> {
        info!("dropping namespace '{}'", namespace);
        let done = self.start_drop(namespace, StoreType::Doc)?;
        Governor::wait(namespace, done).await?;
        info!("namespace '{}' dropped", namespace);
        Ok(())
    }

    /// [`remove`](Self::remove), returning once the store is
    /// [`NamespaceState::Deleting`]: the lifecycle governor removes it in the
    /// background, and it is gone once it is no longer listed.
    pub fn begin_remove(&self, namespace: &str) -> Result<(), DocStoreError> {
        self.start_drop(namespace, StoreType::Doc).map(drop)
    }

    /// Destroy a KV store completely (irreversible); see [`remove`](Self::remove).
    pub async fn remove_kv(&self, namespace: &str) -> Result<(), DocStoreError> {
        info!("dropping KV namespace '{}'", namespace);
        let done = self.start_drop(namespace, StoreType::Kv)?;
        Governor::wait(namespace, done).await?;
        info!("KV namespace '{}' dropped", namespace);
        Ok(())
    }

    /// [`remove_kv`](Self::remove_kv), returning once the store is `Deleting`;
    /// see [`begin_remove`](Self::begin_remove).
    pub fn begin_remove_kv(&self, namespace: &str) -> Result<(), DocStoreError> {
        self.start_drop(namespace, StoreType::Kv).map(drop)
    }

    /// Start dropping `namespace`, a store of type `kind`: check the governor
    /// runs, move the store to `Deleting`, durably, and hand its teardown to
    /// the governor. One already `Deleting` stays so (the drop is finished);
    /// one `Creating` is refused, unless its creation was given up on. No
    /// `.await`: a caller that goes away cannot leave the store `Deleting`
    /// with no teardown handed over.
    fn start_drop(&self, namespace: &str, kind: StoreType) -> Result<TaskDone, DocStoreError> {
        let governor = self.running_governor()?;
        match kind {
            StoreType::Doc => self.schemas.doc(namespace).map(drop)?,
            StoreType::Kv => self.schemas.kv(namespace).map(drop)?,
        }
        let mut from = vec![NamespaceState::Created, NamespaceState::Deleting];
        if governor.parked_in_creation(namespace) {
            from.push(NamespaceState::Creating);
        }
        let was = self.schemas.transition(namespace, &from, NamespaceState::Deleting)?;
        if was != NamespaceState::Deleting {
            self.audit(namespace, AuditEvent::Deleting);
        }
        Ok(governor.submit(namespace, Task::Teardown)?)
    }

    /// Remove everything of `namespace`, whose record is `Deleting`, and the
    /// record last. Idempotent: a teardown cut short is finished by running it
    /// again. Does nothing for a store that is not `Deleting` (or is gone).
    pub(in crate::doc_store) async fn teardown(&self, namespace: &str) -> Result<(), DocStoreError> {
        #[cfg(test)]
        self.inject_lifecycle_fault(namespace)?;
        let Some(schema) = self.schemas.get(namespace) else {
            return Ok(());
        };
        if self.schemas.state(namespace) != Some(NamespaceState::Deleting) {
            return Ok(());
        }
        let Some(ns_id) = schema.ns_id() else {
            return self.teardown_unprovisioned(namespace).await;
        };
        self.without_writes(namespace, async {
            self.without_partition_maintenance(namespace, self.cleanup_store_namespaces(namespace, ns_id))
                .await
        })
        .await?;
        self.audit(namespace, AuditEvent::Deleted);
        info!("store '{namespace}' is gone (ns_id={ns_id})");
        Ok(())
    }

    /// The teardown of a store whose creation was given up on before its
    /// namespace was recorded: all it can have left is an empty namespace of
    /// its name, created by a run cut short. That goes (one holding keys was
    /// written directly, and is not the store's to delete), then the record.
    async fn teardown_unprovisioned(&self, namespace: &str) -> Result<(), DocStoreError> {
        let _lifecycle = self.lifecycle.lock(namespace).await;
        if namespace != crate::namespace::DEFAULT_NAMESPACE
            && self.db.existing_namespace(namespace.to_owned()).await?.is_some()
            && self.refuse_a_namespace_holding_keys(namespace).await.is_ok()
        {
            self.db.remove_namespace(namespace.to_owned()).await?;
        }
        self.schemas.remove(namespace)?;
        self.audit(namespace, AuditEvent::Deleted);
        info!("store '{namespace}' is gone (it was never set up)");
        Ok(())
    }

    /// Run `drop`, which removes store `namespace` (already `Deleting`, so no
    /// write is let in), once the writes in progress have finished (see
    /// `write_gate`): a write after the removal would recreate the namespace
    /// or queue an embedding for it.
    async fn without_writes<T>(
        &self,
        namespace: &str,
        drop: impl std::future::Future<Output = Result<T, DocStoreError>>,
    ) -> Result<T, DocStoreError> {
        if let Some(writes) = self.schemas.writes(namespace) {
            writes.drain().await;
        }
        drop.await
    }

    /// Start a write to `namespace`: refused unless its state takes requests
    /// (see `write_gate`).
    pub(in crate::doc_store) fn enter_write(&self, namespace: &str) -> Result<WriteGuard, DocStoreError> {
        self.schemas.enter_write(namespace)
    }
}

impl DocStore {
    /// Remove store `namespace` (id `ns_id`, already `Deleting`) from the
    /// engine: its namespace and vector companions, their directories, its
    /// index files and maintenance records, and its record last.
    async fn cleanup_store_namespaces(&self, namespace: &str, ns_id: NamespaceId) -> Result<(), DocStoreError> {
        let db = &self.db;
        // Clear this namespace's pending embed-queue entries FIRST.
        //
        // The queue outlives the store otherwise, and a store recreated under the
        // same name would inherit entries for documents it does not hold.
        //
        // Best-effort, and deliberately not the only guard: the worker may already be
        // mid-entry, and a crash here would strand the rest. `VecIndexWorker` also
        // discards queue entries whose namespace has gone.
        #[cfg(feature = "semantic-search")]
        clear_queue_entries(db, namespace).await;

        // Absent only when an earlier drop was interrupted after removing it (the
        // schema goes last): this drop finishes that one. Vector companions are
        // optional — ignore errors on removal.
        if db.existing_namespace(namespace.to_owned()).await?.is_some() {
            db.remove_namespace(namespace.to_owned()).await?;
        }
        // Again, after the removal: no write is admitted any more, but the vector
        // worker may have rewritten an entry it read before the first clear.
        #[cfg(feature = "semantic-search")]
        clear_queue_entries(db, namespace).await;

        #[cfg_attr(not(feature = "semantic-search"), allow(unused_mut))]
        let mut ns_names = vec![namespace.to_owned()];
        #[cfg(feature = "semantic-search")]
        {
            for companion in vector_kv::companion_namespaces(namespace) {
                if self.schemas.contains(&companion) {
                    continue; // a store saved under that name, not a companion
                }
                let _ = db.remove_namespace(companion.clone()).await;
                ns_names.push(companion);
            }
        }

        for ns_name in &ns_names {
            db.remove_unregistered_namespace_dir(ns_name.clone()).await?;
        }

        let index_dir = crate::layout::namespace_index_dir(&crate::layout::index_root(&self.db_path), ns_id);
        if index_dir.exists() {
            std::fs::remove_dir_all(&index_dir)?;
        }
        self.maintenance.delete_store(ns_id)?;

        // Last: until here the store still exists, so a failed or interrupted
        // cleanup is finished by running the teardown again.
        self.schemas.remove(namespace)?;

        // Drop any in-memory vector corruption counters for this namespace so a
        // dropped store stops appearing in /admin/indices/vector/corruption-metrics.
        #[cfg(feature = "semantic-search")]
        crate::semantic_search::metrics::reset(namespace);

        Ok(())
    }
}

/// Remove every embed-queue entry of `namespace`. Best-effort: a failure is
/// logged, and the vector worker discards entries whose namespace has gone.
#[cfg(feature = "semantic-search")]
async fn clear_queue_entries(db: &AsyncDb, namespace: &str) {
    if let Err(e) = vector_kv::clear_queue_for(db, namespace).await {
        warn!("cleanup_store_namespaces: could not clear the embed queue for '{namespace}': {e}");
    }
}
