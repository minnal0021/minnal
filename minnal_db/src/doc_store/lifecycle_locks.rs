//! `LifecycleLocks`: one lock per store name, which keeps a store's create,
//! amendments and drop apart.

use std::sync::Arc;

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
