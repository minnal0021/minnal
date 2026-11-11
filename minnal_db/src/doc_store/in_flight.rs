//! `InFlightWrites`: the documents of semantic-search stores with a write in
//! progress, which the vector worker skips.

use std::sync::Arc;

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
#[derive(Default)]
pub(crate) struct InFlightWrites {
    writes: parking_lot::Mutex<std::collections::HashMap<u64, u32>>,
}

/// One write in progress (see [`InFlightWrites`]); removed when dropped, so an
/// error or a cancelled request (its future dropped) cannot leave it behind.
/// Dropping it then wakes the vector worker, which may have skipped the
/// document while the write was in progress.
pub(crate) struct InFlightWrite {
    writes: Arc<InFlightWrites>,
    doc: u64,
    wake: Option<Arc<tokio::sync::Notify>>,
}

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
