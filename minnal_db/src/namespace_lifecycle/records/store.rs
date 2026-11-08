//! `MaintenanceStore`: maintenance records, persisted in
//! [`MAINTENANCE_NAMESPACE`] and served from memory.
//!
//! The same write-through rule as the doc store's schema store
//! (`doc_store/schema/store.rs`): each
//! write takes `writer`, persists the record (a WAL-backed put or delete,
//! durable on return), then updates the cached copy. So the cache always holds
//! what is on disk, in write order, and a read costs no I/O, which matters
//! because a query checks the build record of every field it touches.
//! `writer` is a `parking_lot` lock, never held across an `.await`.

use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::key::TaskKey;
use crate::namespace::NamespaceId;
use crate::namespace_lifecycle::LifecycleError;
use crate::{AsyncDb, AsyncNamespace};

/// The system namespace that holds every maintenance record.
pub const MAINTENANCE_NAMESPACE: &str = "system_namespace_maintenance";

/// Every maintenance record, persisted in [`MAINTENANCE_NAMESPACE`] and cached
/// in memory as its stored JSON.
pub(crate) struct MaintenanceStore {
    ns: AsyncNamespace,
    map: RwLock<BTreeMap<TaskKey, Arc<[u8]>>>,
    /// Held by every write across its persist and its cache update.
    writer: Mutex<()>,
}

impl MaintenanceStore {
    /// Open the namespace (creating it in a new database) and load every record.
    /// A key that does not decode is logged and left alone: nothing written
    /// here produces one.
    pub(crate) async fn open(db: &AsyncDb) -> Result<Self, LifecycleError> {
        let ns = db.namespace(MAINTENANCE_NAMESPACE.to_owned()).await?;
        let mut map = BTreeMap::new();
        for (key, value) in ns.scan_prefix(Vec::new()).await? {
            match TaskKey::from_bytes(&key) {
                Some(k) => {
                    map.insert(k, Arc::from(value));
                }
                None => log::warn!("{MAINTENANCE_NAMESPACE}: ignoring a record with an unknown key {key:02x?}"),
            }
        }
        Ok(Self {
            ns,
            map: RwLock::new(map),
            writer: Mutex::new(()),
        })
    }

    /// The record at `key`, if there is one.
    ///
    /// A record that does not parse as `T` is an error, not an absence: records
    /// are never torn, so it means a record of another shape, and treating it as
    /// missing could restart or skip a task.
    pub(crate) fn get<T: DeserializeOwned>(&self, key: TaskKey) -> Result<Option<T>, LifecycleError> {
        let Some(bytes) = self.map.read().get(&key).cloned() else {
            return Ok(None);
        };
        serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|e| LifecycleError::Record(format!("{key:?} does not parse: {e}")))
    }

    /// Save `record` at `key`, durably.
    pub(crate) fn put<T: Serialize>(&self, key: TaskKey, record: &T) -> Result<(), LifecycleError> {
        let _w = self.writer.lock();
        self.persist(key, record)
    }

    /// Remove the record at `key`, durably. Removing an absent record is a no-op.
    pub(crate) fn delete(&self, key: TaskKey) -> Result<(), LifecycleError> {
        let _w = self.writer.lock();
        self.remove(key)
    }

    /// Remove every record of `store`, durably: the last step of dropping it.
    pub(crate) fn delete_store(&self, store: NamespaceId) -> Result<(), LifecycleError> {
        let _w = self.writer.lock();
        for key in self.keys_of(store) {
            self.remove(key)?;
        }
        Ok(())
    }

    /// The keys of `store`'s records, in key order.
    pub(crate) fn keys_of(&self, store: NamespaceId) -> Vec<TaskKey> {
        self.map.read().keys().filter(|k| k.store == store).copied().collect()
    }

    /// Persist and cache. Caller holds `writer`.
    fn persist<T: Serialize>(&self, key: TaskKey, record: &T) -> Result<(), LifecycleError> {
        let json = serde_json::to_vec(record).map_err(|e| LifecycleError::Record(e.to_string()))?;
        crate::support::run_blocking_pass(|| self.ns.put_blocking(&key.to_bytes(), &json))?;
        self.map.write().insert(key, Arc::from(json));
        Ok(())
    }

    /// Delete and uncache. Caller holds `writer`.
    fn remove(&self, key: TaskKey) -> Result<(), LifecycleError> {
        if !self.map.read().contains_key(&key) {
            return Ok(());
        }
        crate::support::run_blocking_pass(|| self.ns.delete_blocking(&key.to_bytes()))?;
        self.map.write().remove(&key);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Note {
        n: u64,
    }

    async fn open(dir: &TempDir) -> (AsyncDb, MaintenanceStore) {
        let db = AsyncDb::open_with_config(dir.path().to_path_buf(), crate::support::test_db_config())
            .await
            .unwrap();
        let store = MaintenanceStore::open(&db).await.unwrap();
        (db, store)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn records_survive_a_crash_and_deletes_do_too() {
        let dir = TempDir::new().unwrap();
        let a = NamespaceId::random().unwrap();
        let b = NamespaceId::random().unwrap();
        {
            let (db, store) = open(&dir).await;
            store.put(TaskKey::field_build(a, 1), &Note { n: 1 }).unwrap();
            store.put(TaskKey::reindex_all(a), &Note { n: 2 }).unwrap();
            store.put(TaskKey::vector_reindex(b), &Note { n: 3 }).unwrap();
            store.put(TaskKey::field_build(a, 1), &Note { n: 11 }).unwrap();
            store.delete(TaskKey::reindex_all(a)).unwrap();
            // A write returns once it is durable: nothing is flushed here.
            drop(store);
            db.crash().await;
        }
        let (_db, store) = open(&dir).await;
        assert_eq!(store.get::<Note>(TaskKey::field_build(a, 1)).unwrap(), Some(Note { n: 11 }));
        assert_eq!(store.get::<Note>(TaskKey::reindex_all(a)).unwrap(), None);
        assert_eq!(store.keys_of(b), vec![TaskKey::vector_reindex(b)]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn deleting_a_store_leaves_the_others() {
        let dir = TempDir::new().unwrap();
        let (a, b) = (NamespaceId::random().unwrap(), NamespaceId::random().unwrap());
        let (_db, store) = open(&dir).await;
        store.put(TaskKey::field_build(a, 1), &Note { n: 1 }).unwrap();
        store.put(TaskKey::field_build(a, 2), &Note { n: 2 }).unwrap();
        store.put(TaskKey::field_build(b, 1), &Note { n: 3 }).unwrap();
        store.delete_store(a).unwrap();
        assert!(store.keys_of(a).is_empty());
        assert_eq!(store.keys_of(b), vec![TaskKey::field_build(b, 1)]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_record_of_another_shape_is_an_error_not_an_absence() {
        let dir = TempDir::new().unwrap();
        let (_db, store) = open(&dir).await;
        let key = TaskKey::reindex_all(NamespaceId::random().unwrap());
        store.put(key, &"text").unwrap();
        assert!(store.get::<Note>(key).is_err());
    }
}
