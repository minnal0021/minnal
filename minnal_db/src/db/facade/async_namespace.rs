//! [`AsyncNamespace`]: a scoped handle to one namespace of an [`AsyncDb`](super::AsyncDb).

use super::*;
use crate::namespace::NamespaceId;

// ── AsyncNamespace ─────────────────────────────────────────────────────

/// Async scoped handle to a single namespace.
#[derive(Clone)]
pub struct AsyncNamespace {
    pub(super) ns_id: NamespaceId,
    pub(super) store: Arc<KVStore>,
    pub(super) db: Arc<Db>,
}

impl AsyncNamespace {
    pub fn id(&self) -> NamespaceId {
        self.ns_id
    }

    /// The TTL for this namespace, if configured.
    pub fn ttl(&self) -> Option<Duration> {
        self.store.ttl
    }

    /// Insert or update a key-value pair.
    ///
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](AsyncDb#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub async fn put(&self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        let db = self.db.clone();
        let ns_id = self.ns_id;
        tokio::task::spawn_blocking(move || db.inner.put_ns(ns_id, &key, &value))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Write a key-value pair without WAL — see [`Database::put_ns_no_wal`] for
    /// the crash-safety trade-off.
    pub async fn put_no_wal(&self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        let db = self.db.clone();
        let ns_id = self.ns_id;
        tokio::task::spawn_blocking(move || db.inner.put_ns_no_wal(ns_id, &key, &value))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Write a key-value pair logged in the vector index WAL instead of the
    /// main WAL — durable after the next `AsyncDb::sync_vector_index_wal`. See
    /// [`Database::put_ns_logged`].
    #[cfg(feature = "semantic-search")]
    pub(crate) async fn put_logged(&self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        let db = self.db.clone();
        let ns_id = self.ns_id;
        tokio::task::spawn_blocking(move || db.inner.put_ns_logged(ns_id, &key, &value))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Delete a key logged in the vector index WAL instead of the main WAL —
    /// durable after the next `AsyncDb::sync_vector_index_wal`. See
    /// [`Database::delete_ns_logged`].
    #[cfg(feature = "semantic-search")]
    pub(crate) async fn delete_logged(&self, key: Vec<u8>) -> Result<()> {
        let db = self.db.clone();
        let ns_id = self.ns_id;
        tokio::task::spawn_blocking(move || db.inner.delete_ns_logged(ns_id, &key))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// [`put`](Self::put) on the calling thread, for crate code that must stay
    /// synchronous (the doc store's schema store). Wrap a call made on a runtime
    /// thread in `support::run_blocking_pass`.
    #[cfg(feature = "doc-store")]
    pub(crate) fn put_blocking(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.db.inner.put_ns(self.ns_id, key, value)
    }

    /// [`delete`](Self::delete) on the calling thread; see [`put_blocking`](Self::put_blocking).
    #[cfg(feature = "doc-store")]
    pub(crate) fn delete_blocking(&self, key: &[u8]) -> Result<()> {
        self.db.inner.delete_ns(self.ns_id, key)
    }

    pub async fn get(&self, key: Vec<u8>) -> Result<Option<Vec<u8>>> {
        let db = self.db.clone();
        let ns_id = self.ns_id;
        tokio::task::spawn_blocking(move || db.inner.get_ns(ns_id, &key))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Scan multiple 4-byte BE u32 prefixes in a single blocking task.
    ///
    /// Internally performs one sequential LSM scan across all prefix IDs, then reads
    /// all value-log entries in exactly `num_buckets` threads —
    /// dramatically cheaper than one `spawn_blocking` per prefix.
    ///
    /// Returns every `(prefix_id, raw_key_bytes, value_bytes)` in key order, packed
    /// into a few buffers (see [`PrefixScan`](crate::PrefixScan)). A storage error is
    /// returned as such, never as an empty result: a caller must be able to tell
    /// it from "no entries under these prefixes".
    pub async fn scan_prefixes_batch(&self, prefix_ids: Vec<u32>) -> Result<crate::kv_store::PrefixScan> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || store.scan_prefixes_batch(&prefix_ids))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Fetch multiple keys in a single blocking task.
    ///
    /// Returns one `Option<Vec<u8>>` per input key in the same order; `None` means
    /// absent or deleted. A storage error fails the whole call, never reads as
    /// `None`: a caller must be able to tell it from a missing key.
    pub async fn get_multiple(&self, keys: Vec<Vec<u8>>) -> Result<Vec<Option<Vec<u8>>>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || store.get_multiple(&keys))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Delete a key.
    ///
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](AsyncDb#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub async fn delete(&self, key: Vec<u8>) -> Result<()> {
        let db = self.db.clone();
        let ns_id = self.ns_id;
        tokio::task::spawn_blocking(move || db.inner.delete_ns(ns_id, &key))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Delete a key without WAL — see [`Database::delete_ns_no_wal`] for the
    /// crash-safety trade-off. The no-WAL counterpart of [`Self::put_no_wal`],
    /// intended for derived/regenerable data such as TTL caches.
    pub async fn delete_no_wal(&self, key: Vec<u8>) -> Result<()> {
        let db = self.db.clone();
        let ns_id = self.ns_id;
        tokio::task::spawn_blocking(move || db.inner.delete_ns_no_wal(ns_id, &key))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Atomically read-modify-write a key in this namespace.
    ///
    /// See [`AsyncDb::merge`] for the full contract.
    pub async fn merge<F>(&self, key: Vec<u8>, value: Vec<u8>, merge_fn: F) -> Result<Option<Vec<u8>>>
    where
        F: FnOnce(Option<&[u8]>, &[u8]) -> Result<Option<Vec<u8>>> + Send + 'static,
    {
        let db = self.db.clone();
        let ns_id = self.ns_id;
        tokio::task::spawn_blocking(move || db.inner.merge_ns(ns_id, &key, &value, merge_fn))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// [`Self::merge`] whose resolved write skips the WAL — see
    /// `Database::merge_ns_no_wal` for the crash-safety trade-off. The no-WAL
    /// counterpart of `merge`, for data whose lost write repairs itself.
    pub async fn merge_no_wal<F>(&self, key: Vec<u8>, value: Vec<u8>, merge_fn: F) -> Result<Option<Vec<u8>>>
    where
        F: FnOnce(Option<&[u8]>, &[u8]) -> Result<Option<Vec<u8>>> + Send + 'static,
    {
        let db = self.db.clone();
        let ns_id = self.ns_id;
        tokio::task::spawn_blocking(move || db.inner.merge_ns_no_wal(ns_id, &key, &value, merge_fn))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn iter(&self) -> Result<Vec<KeyValue>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || store.scan_range_batch(&[], None))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn keys(&self) -> Result<Vec<Vec<u8>>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || store.keys())
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// The number of live keys. Every key is read, a page at a time, so memory
    /// stays at one page whatever the namespace's size; no value is read.
    pub async fn count(&self) -> Result<usize> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || store.count_keys())
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// At most `limit` live keys in `[cursor, end)`, in key order, and the key the
    /// next page starts at (`None` after the last page): [`scan`](Self::scan)
    /// without the values. `None` for `cursor` starts at the first key.
    pub async fn key_page(&self, cursor: Option<Vec<u8>>, end: Option<Vec<u8>>, limit: usize) -> Result<crate::kv_store::KeyPage> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || store.key_page(cursor.as_deref(), end.as_deref(), limit))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn range(&self, start: Vec<u8>, end: Option<Vec<u8>>) -> Result<Vec<KeyValue>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || store.scan_range_batch(&start, end.as_deref()))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub async fn scan_prefix(&self, prefix: Vec<u8>) -> Result<Vec<KeyValue>> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || store.scan_prefix_batch(&prefix))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    /// Paginated cursor scan. See [`AsyncDb::scan`] for details.
    pub async fn scan(&self, cursor: Option<Vec<u8>>, end: Option<Vec<u8>>, limit: usize) -> Result<ScanPage> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || store.scan_page_batch(cursor.as_deref(), end.as_deref(), limit))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    pub fn stats(&self) -> Stats {
        self.store.stats()
    }

    pub async fn garbage_collect(&self) -> Result<GCStats> {
        let db = self.db.clone();
        let ns_id = self.ns_id;
        tokio::task::spawn_blocking(move || db.inner.garbage_collect_namespace(ns_id))
            .await
            .map_err(|e| KVError::Io(std::io::Error::other(e)))?
    }

    // ── Typed CRUD (rkyv ser/de) ──────────────────────────────────────

    /// Insert a typed key-value pair, serialized via rkyv.
    ///
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](AsyncDb#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub async fn put_typed<K, V>(&self, key: &K, value: &V) -> Result<()>
    where
        K: RkyvEncode,
        V: RkyvEncode,
    {
        let kb = rkyv_serialize(key)?;
        let vb = rkyv_serialize(value)?;
        self.put(kb, vb).await
    }

    /// Look up a typed value by typed key.
    pub async fn get_typed<K, V>(&self, key: &K) -> Result<Option<V>>
    where
        K: RkyvEncode,
        V: RkyvDecode,
    {
        let kb = rkyv_serialize(key)?;
        match self.get(kb).await? {
            Some(vb) => Ok(Some(rkyv_deserialize::<V>(&vb)?)),
            None => Ok(None),
        }
    }

    /// Delete by typed key.
    ///
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](AsyncDb#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub async fn delete_typed<K>(&self, key: &K) -> Result<()>
    where
        K: RkyvEncode,
    {
        let kb = rkyv_serialize(key)?;
        self.delete(kb).await
    }

    /// Atomically merge a typed value, serialized via rkyv.
    ///
    /// The typed twin of [`merge`](Self::merge); see [`AsyncDb::merge_typed`] for an example.
    pub async fn merge_typed<K, V, F>(&self, key: &K, value: &V, merge_fn: F) -> Result<Option<V>>
    where
        K: RkyvEncode,
        V: RkyvEncode + RkyvDecode + Send + 'static,
        F: FnOnce(Option<V>, V) -> Result<Option<V>> + Send + 'static,
    {
        let kb = rkyv_serialize(key)?;
        let vb = rkyv_serialize(value)?;
        // Unlike the sync twin, the new `V` cannot be carried out of the closure
        // in a borrowed slot: the closure crosses a `spawn_blocking` boundary and
        // so must be `'static`. Deserializing the returned bytes is the price of
        // that, and it is paid only when the merge actually wrote.
        let written = self
            .merge(kb, vb, move |existing, operand| {
                let current = match existing {
                    Some(bytes) => Some(rkyv_deserialize::<V>(bytes)?),
                    None => None,
                };
                let operand = rkyv_deserialize::<V>(operand)?;
                match merge_fn(current, operand)? {
                    Some(new) => Ok(Some(rkyv_serialize(&new)?)),
                    None => Ok(None),
                }
            })
            .await?;
        match written {
            Some(bytes) => Ok(Some(rkyv_deserialize::<V>(&bytes)?)),
            None => Ok(None),
        }
    }

    // ── Typed Iteration (rkyv ser/de) ─────────────────────────────────

    /// Iterate over all typed key-value pairs.
    pub async fn iter_typed<K, V>(&self) -> Result<Vec<(K, V)>>
    where
        K: RkyvDecode,
        V: RkyvDecode,
    {
        deserialize_pairs(self.iter().await?)
    }

    /// Iterate over all typed keys.
    pub async fn keys_typed<K>(&self) -> Result<Vec<K>>
    where
        K: RkyvDecode,
    {
        deserialize_keys(self.keys().await?)
    }

    /// Iterate over typed key-value pairs in `[start, end)`.
    pub async fn range_typed<K, V>(&self, start: &K, end: Option<&K>) -> Result<Vec<(K, V)>>
    where
        K: RkyvEncode + RkyvDecode,
        V: RkyvDecode,
    {
        let start_bytes = rkyv_serialize(start)?;
        let end_bytes = end.map(|e| rkyv_serialize(e)).transpose()?;
        let raw = self.range(start_bytes, end_bytes).await?;
        deserialize_pairs(raw)
    }

    /// Iterate over typed key-value pairs whose keys start with `prefix`.
    pub async fn scan_prefix_typed<K, V>(&self, prefix: Vec<u8>) -> Result<Vec<(K, V)>>
    where
        K: RkyvDecode,
        V: RkyvDecode,
    {
        deserialize_pairs(self.scan_prefix(prefix).await?)
    }
}
