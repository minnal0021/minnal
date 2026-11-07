//! What keeps a store's writes and lifecycle apart: per-store lifecycle locks, the
//! gate a drop closes to wait out writes in progress, and the set of documents
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
/// document or KV write, a reindex) holds a [`WriteGuard`] for its whole
/// duration; a drop [`close`](Self::close)s the gate, which refuses new writes
/// and waits for the ones in progress, before it removes anything. A write
/// that went on would resolve the store's namespace after its removal and
/// recreate it, or leave an embed-queue entry the drop already cleared.
#[derive(Default)]
pub(crate) struct WriteGate {
    state: parking_lot::Mutex<WriteGateState>,
    left: tokio::sync::Notify,
}

#[derive(Default)]
struct WriteGateState {
    closed: std::collections::HashSet<String>,
    writers: std::collections::HashMap<String, usize>,
}

/// A write in progress on one store; see [`WriteGate`]. Owns its gate, so a
/// background task (an index build) can hold it.
pub(crate) struct WriteGuard {
    gate: Arc<WriteGate>,
    namespace: String,
}

impl WriteGuard {
    /// Whether the store is being dropped: a long write stops when it is, so
    /// the drop waiting for it is not held up.
    pub(crate) fn store_dropping(&self) -> bool {
        self.gate.is_closed(&self.namespace)
    }
}

impl WriteGate {
    /// Start a write to `namespace`; fails with [`DocStoreError::NotFound`]
    /// while it is being dropped.
    pub(crate) fn enter(self: &Arc<Self>, namespace: &str) -> Result<WriteGuard, DocStoreError> {
        let mut state = self.state.lock();
        if state.closed.contains(namespace) {
            return Err(DocStoreError::NotFound {
                namespace: namespace.to_owned(),
            });
        }
        *state.writers.entry(namespace.to_owned()).or_default() += 1;
        Ok(WriteGuard {
            gate: Arc::clone(self),
            namespace: namespace.to_owned(),
        })
    }

    /// Whether `namespace` is being dropped.
    pub(crate) fn is_closed(&self, namespace: &str) -> bool {
        self.state.lock().closed.contains(namespace)
    }

    /// Refuse new writes to `namespace` and wait for those in progress.
    pub(crate) async fn close(&self, namespace: &str) {
        self.state.lock().closed.insert(namespace.to_owned());
        loop {
            let left = self.left.notified();
            tokio::pin!(left);
            // Registered before the check, so a write leaving between the two is seen.
            left.as_mut().enable();
            if !self.state.lock().writers.contains_key(namespace) {
                return;
            }
            left.await;
        }
    }

    /// Accept writes to `namespace` again: the drop is over (its schema is
    /// gone, so they now fail on that, or it failed and the store remains).
    pub(crate) fn reopen(&self, namespace: &str) {
        self.state.lock().closed.remove(namespace);
    }
}

impl Drop for WriteGuard {
    fn drop(&mut self) {
        let mut state = self.gate.state.lock();
        if let Some(n) = state.writers.get_mut(&self.namespace) {
            *n -= 1;
            if *n == 0 {
                state.writers.remove(&self.namespace);
            }
        }
        drop(state);
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
