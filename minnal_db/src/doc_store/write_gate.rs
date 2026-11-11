//! The write gate: what fences a store's writes off from its drop.
//!
//! Every write to a store (a document or KV write, a diagnostic reindex)
//! holds a [`WriteGuard`] for its whole duration, entered only while the
//! store's state takes requests. A drop first moves the store to `Deleting`,
//! then [`drain`](StoreWrites::drain)s the store's writes: it waits for the
//! writes already in, before it removes anything. A write that went on would
//! resolve the store's namespace after its removal, or leave an embed-queue
//! entry the drop already cleared.
//!
//! **Per store, no lock.** Each store's count of writes lives in its schema
//! cache entry ([`SchemaStore::enter_write`](crate::doc_store::schema::store)),
//! for as long as the store exists: a store dropped and created again under
//! its name gets a new one. A write counts itself first, then reads the
//! store's state; a drop moves the state first, then reads the count. The
//! state is read and written under the schema cache's lock, so either the
//! write read the old state before the move, and the drop's later read of the
//! count sees it, or the write sees `Deleting`, and takes itself back.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// One store's writes in progress; see the module docs.
#[derive(Default, Debug)]
pub(crate) struct StoreWrites {
    count: AtomicUsize,
    left: tokio::sync::Notify,
}

/// A write in progress on one store; see the module docs.
pub(crate) struct WriteGuard {
    writes: Arc<StoreWrites>,
}

impl StoreWrites {
    /// Count a write until the returned guard drops. The caller checks the
    /// store's state *after* this.
    pub(crate) fn enter(self: &Arc<Self>) -> WriteGuard {
        self.count.fetch_add(1, Ordering::SeqCst);
        WriteGuard { writes: Arc::clone(self) }
    }

    /// Wait until no write is in progress. The caller has moved the store to
    /// a state that admits none, so none starts after.
    pub(crate) async fn drain(&self) {
        loop {
            let left = self.left.notified();
            tokio::pin!(left);
            // Registered before the check, so a write leaving between the two is seen.
            left.as_mut().enable();
            if self.count.load(Ordering::SeqCst) == 0 {
                return;
            }
            left.await;
        }
    }
}

impl Drop for WriteGuard {
    fn drop(&mut self) {
        if self.writes.count.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.writes.left.notify_waiters();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[tokio::test]
    async fn a_drain_waits_for_the_writes_in_progress() {
        let writes = Arc::new(StoreWrites::default());
        writes.drain().await; // none: returns at once
        let a = writes.enter();
        let b = writes.enter();
        let drained = tokio::spawn({
            let writes = Arc::clone(&writes);
            async move { writes.drain().await }
        });
        drop(a);
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!drained.is_finished(), "one write is still in progress");
        drop(b);
        tokio::time::timeout(Duration::from_secs(5), drained).await.unwrap().unwrap();
    }
}
