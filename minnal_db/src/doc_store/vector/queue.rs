//! The embed queue through the doc store: clearing a document's vectors,
//! counting, listing, retrying and deleting queue entries.

use crate::doc_store::*;

impl DocStore {
    /// Remove a document from the vector index: it was deleted, or its embedding
    /// text is now empty.
    ///
    /// With a worker running, this writes a `Clear` tombstone before deleting the
    /// vectors ([`vector_kv::clear_vectors`]), so a worker already embedding the
    /// document's older text removes what it writes when it completes, and wakes
    /// the worker to retire the tombstone. With no worker nothing can be in
    /// flight, so the queue entry and the vectors are removed directly.
    #[cfg(feature = "semantic-search")]
    pub(in crate::doc_store) async fn clear_doc_vectors(&self, namespace: &str, key: &[u8]) -> Result<(), DocStoreError> {
        // The partition first, so the count change below always reaches it. A
        // change made while no partition is loaded is dropped, which is right
        // only if the next load counts after the delete; a load already
        // counting (and not yet installed) would keep the deleted document for
        // good. Loading here installs it before the delete.
        let partition = self.loaded_partition(namespace).await;
        let delta = match &self.notify {
            Some(notify) => {
                let delta = vector_kv::clear_vectors(&self.db, namespace, key).await?;
                notify.notify_one();
                delta
            }
            None => {
                vector_kv::remove_queue_entry(&self.db, namespace, key).await?;
                vector_kv::delete_vector(&self.db, namespace, key).await?
            }
        };
        if let Some(ctx) = &self.semantic_ctx {
            match partition {
                Some(p) => {
                    p.snapshot().apply_delta(&delta);
                    if p.needs_maintenance(&delta) {
                        ctx.request_maintenance(namespace);
                    }
                }
                None => ctx.apply_posting_delta(namespace, &delta),
            }
        }
        Ok(())
    }

    /// `namespace`'s partition, loading it if needed; `None` when the namespace
    /// has none to load (no semantic search, or it is being dropped).
    #[cfg(feature = "semantic-search")]
    async fn loaded_partition(&self, namespace: &str) -> Option<Arc<crate::semantic_search::PartitionHandle>> {
        let ctx = self.semantic_ctx.as_ref()?;
        let (settings, ns_id) = self.schemas.vector_settings(namespace).ok()?;
        ctx.for_namespace(&self.db, namespace, ns_id, &settings).await.ok().map(|ns| ns.partition)
    }

    /// Record a write to a semantic-search store's document until the returned
    /// guard drops (see [`InFlightWrites`]). Taken before the write's queue entry
    /// and held until its document write returns; the drop wakes the worker.
    #[cfg(feature = "semantic-search")]
    pub(in crate::doc_store) fn begin_vector_write(&self, namespace: &str, key: &[u8]) -> InFlightWrite {
        self.in_flight.begin(namespace, key, self.notify.clone())
    }

    /// For an upsert whose embedding text is empty: clear the document's vectors
    /// and any pending embed of its older text — but only if it has some, so a
    /// bulk load of documents without embedding text pays no extra writes.
    #[cfg(feature = "semantic-search")]
    pub(in crate::doc_store) async fn clear_doc_vectors_if_any(&self, namespace: &str, key: &[u8]) -> Result<(), DocStoreError> {
        if vector_kv::has_any_vector_state(&self.db, namespace, key).await? {
            self.clear_doc_vectors(namespace, key).await?;
        }
        Ok(())
    }

    /// Number of entries currently waiting in the async vector-index queue.
    ///
    /// This is the count of documents that have been written but whose
    /// vector index entry has not yet been committed — either because the
    /// worker has not processed them yet or the embedding service is
    /// temporarily unavailable.
    #[cfg(feature = "semantic-search")]
    pub async fn pending_vector_index_count(&self) -> Result<usize, DocStoreError> {
        Ok(vector_kv::count_queue_entries(&self.db).await?)
    }

    /// How many documents of `namespace` have vector-index entries, counted
    /// from the keys of `{namespace}_sparse_vector_meta` (one per indexed
    /// document; no value is read). Recent writes count at once, and a deleted
    /// document's entry stops counting as soon as it is removed. 0 when nothing
    /// in the namespace has been indexed.
    #[cfg(feature = "semantic-search")]
    pub async fn count_indexed_docs(&self, namespace: &str) -> Result<u64, DocStoreError> {
        let meta_ns = vector_kv::sparse_vectors_meta_ns(namespace);
        Ok(match self.db.existing_namespace(meta_ns).await? {
            Some(ns) => ns.count().await? as u64,
            None => 0,
        })
    }

    /// Return all entries currently in the async vector-index queue.
    ///
    /// Includes entries that are still pending, currently being retried, and
    /// those that have exceeded the configured `max_retries` threshold and are
    /// waiting for manual removal via [`delete_queue_entry`].
    ///
    /// [`delete_queue_entry`]: DocStore::delete_queue_entry
    #[cfg(feature = "semantic-search")]
    pub async fn list_queue_entries(&self) -> Vec<vector_kv::QueueEntry> {
        vector_kv::list_queue_entries(&self.db).await.unwrap_or_default()
    }

    /// Look up a single pending embedding queue entry by namespace and document ID.
    ///
    /// Returns `None` when no entry exists for the pair.  This is an O(1) key
    /// lookup — it does not scan the whole queue.
    #[cfg(feature = "semantic-search")]
    pub async fn get_queue_entry(&self, namespace: &str, doc_id_bytes: &[u8]) -> Option<vector_kv::QueueEntry> {
        vector_kv::get_queue_entry(&self.db, namespace, doc_id_bytes).await.unwrap_or(None)
    }

    /// Remove a specific entry from the async vector-index queue.
    ///
    /// This is an admin operation intended for entries that have exceeded
    /// `max_retries` or that need to be manually cleared.  Returns `Ok` even
    /// if no matching entry exists.
    #[cfg(feature = "semantic-search")]
    pub async fn delete_queue_entry(&self, namespace: &str, doc_id_bytes: &[u8]) -> Result<(), DocStoreError> {
        vector_kv::remove_queue_entry(&self.db, namespace, doc_id_bytes).await?;
        Ok(())
    }

    /// Reset the retry count to zero for a specific queue entry, allowing the
    /// worker to attempt embedding it again on its next pass.
    ///
    /// Returns the entry as it was before the reset (so the caller can inspect
    /// its previous `retry_count`), or `None` when no matching entry exists.
    /// The worker is notified immediately after the reset so processing starts
    /// without waiting for the next scheduled wake-up.
    #[cfg(feature = "semantic-search")]
    pub async fn retry_queue_entry(&self, namespace: &str, doc_id_bytes: &[u8]) -> Result<Option<vector_kv::QueueEntry>, DocStoreError> {
        // Atomic reset that keeps the entry's kind and text: re-enqueueing a text
        // read a moment earlier could overwrite a newer upsert, or turn a `Clear`
        // tombstone into an embed.
        let Some(entry) = vector_kv::reset_queue_entry(&self.db, namespace, doc_id_bytes).await? else {
            return Ok(None);
        };
        if let Some(notify) = &self.notify {
            notify.notify_one();
        }
        Ok(Some(entry))
    }

    /// Reset the retry count to zero for every exhausted queue entry in
    /// `namespace` (those whose `retry_count` has reached `max_retries`), so
    /// they are picked up by the worker on its next pass.
    ///
    /// Returns the number of entries that were reset.  The worker is notified
    /// immediately when at least one entry was reset.
    #[cfg(feature = "semantic-search")]
    pub async fn retry_all_failed_queue_entries(&self, namespace: &str) -> Result<usize, DocStoreError> {
        // Rewriting entries while the cleanup clears them could leave one behind.
        if let Some(ns_id) = self.schemas.get(namespace).and_then(|s| s.ns_id())
            && self
                .jobs
                .get(JobKey::new(ns_id, crate::namespace_lifecycle::JobKind::VectorCleanup))?
                .is_some_and(|r| r.is_live())
        {
            return Err(DocStoreError::JobConflict {
                namespace: namespace.to_owned(),
                job: "retry of failed queue entries".to_owned(),
                running: crate::namespace_lifecycle::JobKind::VectorCleanup.name().to_owned(),
            });
        }
        let count = vector_kv::reset_exhausted_queue_entries(&self.db, namespace, self.vector_index_config.max_retries).await?;
        if count > 0
            && let Some(notify) = &self.notify
        {
            notify.notify_one();
        }
        Ok(count)
    }

    /// Delete every entry in the async vector-index queue that belongs to
    /// `namespace`, regardless of retry count.
    ///
    /// Returns the number of entries removed.  Use this to completely drain the
    /// queue for a namespace before dropping it or when forcing a clean slate.
    #[cfg(feature = "semantic-search")]
    pub async fn delete_all_queue_entries(&self, namespace: &str) -> Result<usize, DocStoreError> {
        Ok(vector_kv::clear_queue_for(&self.db, namespace).await?)
    }
}
