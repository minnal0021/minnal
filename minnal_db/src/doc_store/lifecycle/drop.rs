//! Dropping a store: move it to `Deleting`, wait out its writes, remove its
//! namespaces, its index files and its maintenance records, then its record.

use super::super::*;
use crate::namespace::NamespaceId;

impl DocStore {
    /// Destroy a document store completely, and return once it is gone.
    ///
    /// Moves the store to [`StoreState::Deleting`] first, durably: from then on
    /// it refuses every request but the admin views. Then its writes in progress
    /// are waited out, its namespaces, index files and maintenance records are
    /// removed, and its record last. A store already being dropped (a drop a
    /// crash interrupted, or another caller's) is finished; a store still being
    /// created cannot be dropped ([`DocStoreError::StoreCreating`]).
    ///
    /// This is irreversible.
    pub async fn remove(&self, namespace: &str) -> Result<(), DocStoreError> {
        info!("dropping namespace '{}'", namespace);
        self.schemas.doc(namespace)?;
        self.begin_drop_store(namespace)?;
        self.teardown(namespace).await?;
        info!("namespace '{}' dropped", namespace);
        Ok(())
    }

    /// Destroy a KV store completely (irreversible); see [`remove`](Self::remove).
    pub async fn remove_kv(&self, namespace: &str) -> Result<(), DocStoreError> {
        info!("dropping KV namespace '{}'", namespace);
        self.schemas.kv(namespace)?;
        self.begin_drop_store(namespace)?;
        self.teardown(namespace).await?;
        info!("KV namespace '{}' dropped", namespace);
        Ok(())
    }

    /// Move `namespace` from `Created` to `Deleting`, durably. One already
    /// `Deleting` stays so (the drop is finished); one `Creating` is refused.
    fn begin_drop_store(&self, namespace: &str) -> Result<(), DocStoreError> {
        self.schemas
            .transition(namespace, &[StoreState::Created, StoreState::Deleting], StoreState::Deleting)?;
        Ok(())
    }

    /// Remove everything of `namespace`, whose record is `Deleting`, and the
    /// record last. Idempotent: a teardown cut short is finished by running it
    /// again. Does nothing for a store that is not `Deleting` (or is gone).
    pub(in crate::doc_store) async fn teardown(&self, namespace: &str) -> Result<(), DocStoreError> {
        let Some(schema) = self.schemas.get(namespace) else {
            return Ok(());
        };
        if self.schemas.state(namespace) != Some(StoreState::Deleting) {
            return Ok(());
        }
        let ns_id = schema.ns_id().ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;
        self.without_writes(namespace, async {
            self.without_partition_maintenance(
                namespace,
                cleanup_store_namespaces(&self.db, &self.db_path, &self.schemas, &self.maintenance, namespace, ns_id),
            )
            .await
        })
        .await
    }

    /// Run `drop`, which removes store `namespace`, with no write to it in
    /// progress and none let in until it is done ([`WriteGate`]): a write after
    /// the removal would recreate the namespace or queue an embedding for it.
    async fn without_writes<T>(
        &self,
        namespace: &str,
        drop: impl std::future::Future<Output = Result<T, DocStoreError>>,
    ) -> Result<T, DocStoreError> {
        // A guard, because this future can be dropped at any await (the client of
        // a route disconnecting): left closed, the store refuses writes for good.
        struct Reopen<'a>(&'a WriteGate, &'a str);
        impl Drop for Reopen<'_> {
            fn drop(&mut self) {
                self.0.reopen(self.1);
            }
        }
        let _reopen = Reopen(&self.writes, namespace);
        self.writes.close(namespace).await;
        drop.await
    }
}

// ── Namespace cleanup helper ──────────────────────────────────────────────────

/// Remove a namespace (and its vector-index companions) from the engine registry
/// and delete all on-disk directories and the schema file.
///
/// Shared by [`DocStore::drop`] and [`DocStore::drop_kv`] to
/// eliminate the near-identical cleanup sequences in each method.
async fn cleanup_store_namespaces(
    db: &AsyncDb,
    db_path: &Path,
    schemas: &SchemaStore,
    maintenance: &MaintenanceStore,
    namespace: &str,
    ns_id: NamespaceId,
) -> Result<(), DocStoreError> {
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
    // Again, now that no write can reach the store: a `put` that landed
    // between the first clear and the removal enqueued its embedding.
    #[cfg(feature = "semantic-search")]
    clear_queue_entries(db, namespace).await;

    #[cfg_attr(not(feature = "semantic-search"), allow(unused_mut))]
    let mut ns_names = vec![namespace.to_owned()];
    #[cfg(feature = "semantic-search")]
    {
        for companion in vector_kv::companion_namespaces(namespace) {
            if schemas.contains(&companion) {
                continue; // a store saved under that name, not a companion
            }
            let _ = db.remove_namespace(companion.clone()).await;
            ns_names.push(companion);
        }
    }

    for ns_name in &ns_names {
        db.remove_unregistered_namespace_dir(ns_name.clone()).await?;
    }

    let index_dir = crate::layout::namespace_index_dir(&crate::layout::index_root(db_path), ns_id);
    if index_dir.exists() {
        std::fs::remove_dir_all(&index_dir)?;
    }
    maintenance.delete_store(ns_id)?;

    // Last, as the schema file used to be: until here the store still exists,
    // so a failed or interrupted cleanup can be retried by dropping it again.
    schemas.remove(namespace)?;

    // Drop any in-memory vector corruption counters for this namespace so a
    // dropped store stops appearing in /admin/indices/vector/corruption-metrics.
    #[cfg(feature = "semantic-search")]
    crate::semantic_search::metrics::reset(namespace);

    Ok(())
}

/// Remove every embed-queue entry of `namespace`. Best-effort: a failure is
/// logged, and the vector worker discards entries whose namespace has gone.
#[cfg(feature = "semantic-search")]
async fn clear_queue_entries(db: &AsyncDb, namespace: &str) {
    if let Err(e) = vector_kv::clear_queue_for(db, namespace).await {
        warn!("cleanup_store_namespaces: could not clear the embed queue for '{namespace}': {e}");
    }
}
