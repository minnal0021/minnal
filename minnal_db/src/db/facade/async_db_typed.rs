//! [`AsyncDb`]'s typed methods (rkyv ser/de) on the default namespace: each
//! forwards to its twin on that namespace's handle.

use super::*;

impl AsyncDb {
    // ── Typed CRUD (rkyv ser/de) ──────────────────────────────────────
    //
    // On the default namespace: each forwards to its twin on that namespace's
    // handle, which holds the one implementation.

    async fn default_namespace(&self) -> Result<AsyncNamespace> {
        self.namespace(crate::namespace::DEFAULT_NAMESPACE.to_owned()).await
    }

    /// Insert a typed key-value pair, serialized via rkyv.
    ///
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](AsyncDb#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub async fn put_typed<K, V>(&self, key: &K, value: &V) -> Result<()>
    where
        K: RkyvEncode,
        V: RkyvEncode,
    {
        self.default_namespace().await?.put_typed(key, value).await
    }

    /// Look up a typed value by typed key.
    pub async fn get_typed<K, V>(&self, key: &K) -> Result<Option<V>>
    where
        K: RkyvEncode,
        V: RkyvDecode,
    {
        self.default_namespace().await?.get_typed(key).await
    }

    /// Delete by typed key.
    ///
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](AsyncDb#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub async fn delete_typed<K>(&self, key: &K) -> Result<()>
    where
        K: RkyvEncode,
    {
        self.default_namespace().await?.delete_typed(key).await
    }

    /// Atomically merge a typed value, serialized via rkyv.
    ///
    /// The typed twin of [`merge`](Self::merge): `merge_fn` sees the stored
    /// value and the operand already deserialized, and returns the new value
    /// (or `None` to delete the key). Every caveat on [`merge`](Self::merge)
    /// applies — in particular the closure runs under a lock and must not call
    /// back into the database.
    ///
    /// ```rust,no_run
    /// # async fn run() -> Result<(), minnal_db::KVError> {
    /// # let db = minnal_db::AsyncDb::open("/tmp/merge_typed_doc").await?;
    /// let total: Option<u64> = db
    ///     .merge_typed(&"visits".to_string(), &1u64, |current, step| Ok(Some(current.unwrap_or(0) + step)))
    ///     .await?;
    /// # let _ = total;
    /// # db.shutdown().await?;
    /// # Ok(()) }
    /// ```
    pub async fn merge_typed<K, V, F>(&self, key: &K, value: &V, merge_fn: F) -> Result<Option<V>>
    where
        K: RkyvEncode,
        V: RkyvEncode + RkyvDecode + Send + 'static,
        F: FnOnce(Option<V>, V) -> Result<Option<V>> + Send + 'static,
    {
        self.default_namespace().await?.merge_typed(key, value, merge_fn).await
    }

    // ── Typed Iteration (rkyv ser/de) ─────────────────────────────────

    /// Iterate over all typed key-value pairs.
    pub async fn iter_typed<K, V>(&self) -> Result<Vec<(K, V)>>
    where
        K: RkyvDecode,
        V: RkyvDecode,
    {
        self.default_namespace().await?.iter_typed().await
    }

    /// Iterate over all typed keys.
    pub async fn keys_typed<K>(&self) -> Result<Vec<K>>
    where
        K: RkyvDecode,
    {
        self.default_namespace().await?.keys_typed().await
    }

    /// Iterate over typed key-value pairs in `[start, end)`.
    pub async fn range_typed<K, V>(&self, start: &K, end: Option<&K>) -> Result<Vec<(K, V)>>
    where
        K: RkyvEncode + RkyvDecode,
        V: RkyvDecode,
    {
        self.default_namespace().await?.range_typed(start, end).await
    }

    /// Iterate over typed key-value pairs whose keys start with `prefix`.
    pub async fn scan_prefix_typed<K, V>(&self, prefix: Vec<u8>) -> Result<Vec<(K, V)>>
    where
        K: RkyvDecode,
        V: RkyvDecode,
    {
        self.default_namespace().await?.scan_prefix_typed(prefix).await
    }
}
