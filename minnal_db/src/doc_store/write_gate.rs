//! What keeps a store's writes and lifecycle apart: per-store lifecycle locks, the
//! gate a drop drains to wait out writes in progress, and the set of documents
//! being written that the vector worker skips.

use std::sync::Arc;

use crate::doc_store::error::DocStoreError;

/// One lock per store name, held across whatever creates or removes that store's
/// namespaces as a whole: a create, an amendment (which can turn semantic search
/// on and seed the partition) and a drop. Interleaved, a drop removes the partition
/// an enable just seeded, leaving semantic search on with no partition.
///
/// Per store because a drop can wait minutes for a rebuild's k-means, and other
/// stores must not wait with it. Each operation touches only its own store's
/// namespaces, and none holds two of these locks.
#[derive(Default)]
pub(crate) struct LifecycleLocks {
    locks: parking_lot::Mutex<std::collections::HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl LifecycleLocks {
    /// Wait for `namespace`'s lock. Taken after the write gate.
    pub(crate) async fn lock(&self, namespace: &str) -> tokio::sync::OwnedMutexGuard<()> {
        let lock = {
            let mut locks = self.locks.lock();
            // Nobody holds or waits for a lock only this map refers to: holders
            // and waiters got their clone under this mutex. Keeps the map to the
            // stores in use, whatever names come and go.
            locks.retain(|_, lock| Arc::strong_count(lock) > 1);
            Arc::clone(locks.entry(namespace.to_owned()).or_default())
        };
        lock.lock_owned().await
    }
}

/// Fences a store's writes off from its drop. Every write to a store (a
/// document or KV write, a diagnostic reindex) holds a [`WriteGuard`] for its
/// whole duration, entered only while the store's state takes requests. A drop
/// first moves the store to `Deleting`, then [`drain`](Self::drain)s the gate:
/// it waits for the writes already in, before it removes anything. A write
/// that went on would resolve the store's namespace after its removal, or
/// leave an embed-queue entry the drop already cleared.
///
/// The state check and the count are one step under the gate's lock, and the
/// drop reads the count under it after its state change is in place: a write
/// either saw the old state and is counted, or sees `Deleting` and is refused.
#[derive(Default)]
pub(crate) struct WriteGate {
    writers: parking_lot::Mutex<std::collections::HashMap<String, usize>>,
    left: tokio::sync::Notify,
}

/// A write in progress on one store; see [`WriteGate`].
pub(crate) struct WriteGuard {
    gate: Arc<WriteGate>,
    namespace: String,
}

impl WriteGate {
    /// Start a write to `namespace` if `admit` (its state check) passes;
    /// `admit` runs under the gate's lock.
    pub(crate) fn enter(self: &Arc<Self>, namespace: &str, admit: impl FnOnce() -> Result<(), DocStoreError>) -> Result<WriteGuard, DocStoreError> {
        let mut writers = self.writers.lock();
        admit()?;
        *writers.entry(namespace.to_owned()).or_default() += 1;
        Ok(WriteGuard {
            gate: Arc::clone(self),
            namespace: namespace.to_owned(),
        })
    }

    /// Wait until no write to `namespace` is in progress. The caller has
    /// moved the store to a state that admits none, so none starts after.
    pub(crate) async fn drain(&self, namespace: &str) {
        loop {
            let left = self.left.notified();
            tokio::pin!(left);
            // Registered before the check, so a write leaving between the two is seen.
            left.as_mut().enable();
            if !self.writers.lock().contains_key(namespace) {
                return;
            }
            left.await;
        }
    }
}

impl Drop for WriteGuard {
    fn drop(&mut self) {
        let mut writers = self.gate.writers.lock();
        if let Some(n) = writers.get_mut(&self.namespace) {
            *n -= 1;
            if *n == 0 {
                writers.remove(&self.namespace);
            }
        }
        drop(writers);
        self.gate.left.notify_waiters();
    }
}

/// Documents of semantic-search stores with a write in progress: a write adds
/// its document before it queues the embed and removes it after the document
/// write (see `vector_kv`'s *Queue first*).
///
/// Nothing waits on it. The vector worker only looks: it compares a queue entry
/// with its document only when no write to the document is in progress, so a
/// mismatch it finds is a write a crash or an error cut short, or a reindex
/// entry from an older snapshot — never a write half done. A document with a
/// write in progress is skipped; that write's own entry and wake-up bring it
/// back. Without this, the worker could read the document between a writer's
/// two writes and turn the writer's new entry back into the old text, or embed a
/// reindex's stale text after it had completed the writer's entry.
///
/// Keyed by a 64-bit hash of (store, key) with a count of concurrent writes, so
/// it holds about 16 bytes per write in progress and nothing once writes finish.
/// A hash collision only makes the worker skip an entry for a pass.
#[cfg(feature = "semantic-search")]
#[derive(Default)]
pub(crate) struct InFlightWrites {
    writes: parking_lot::Mutex<std::collections::HashMap<u64, u32>>,
}

/// One write in progress (see [`InFlightWrites`]); removed when dropped, so an
/// error or a cancelled request (its future dropped) cannot leave it behind.
/// Dropping it then wakes the vector worker, which may have skipped the
/// document while the write was in progress.
#[cfg(feature = "semantic-search")]
pub(crate) struct InFlightWrite {
    writes: Arc<InFlightWrites>,
    doc: u64,
    wake: Option<Arc<tokio::sync::Notify>>,
}

#[cfg(feature = "semantic-search")]
impl InFlightWrites {
    fn doc(namespace: &str, key: &[u8]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        namespace.hash(&mut hasher);
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Record a write to `namespace`'s document `key` until the returned guard
    /// drops; the drop then wakes `wake` (the vector worker), if any.
    pub(crate) fn begin(self: &Arc<Self>, namespace: &str, key: &[u8], wake: Option<Arc<tokio::sync::Notify>>) -> InFlightWrite {
        let doc = Self::doc(namespace, key);
        *self.writes.lock().entry(doc).or_default() += 1;
        InFlightWrite {
            writes: Arc::clone(self),
            doc,
            wake,
        }
    }

    /// Whether a write to `namespace`'s document `key` is in progress.
    pub(crate) fn contains(&self, namespace: &str, key: &[u8]) -> bool {
        self.writes.lock().contains_key(&Self::doc(namespace, key))
    }
}

#[cfg(feature = "semantic-search")]
impl Drop for InFlightWrite {
    fn drop(&mut self) {
        let mut writes = self.writes.writes.lock();
        if let Some(n) = writes.get_mut(&self.doc) {
            *n -= 1;
            if *n == 0 {
                writes.remove(&self.doc);
            }
        }
        drop(writes);
        if let Some(wake) = &self.wake {
            wake.notify_one();
        }
    }
}
