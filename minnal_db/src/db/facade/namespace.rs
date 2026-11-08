//! [`Namespace`]: a scoped handle to one namespace of a [`Db`](super::Db).

use super::*;
use crate::namespace::NamespaceId;

// ── Namespace (scoped handle) ──────────────────────────────────────────

/// A scoped handle to a single namespace within the database.
///
/// Provides the same CRUD and iteration methods as `Db`, but all
/// operations are pinned to the namespace this handle was created for.
pub struct Namespace<'db> {
    pub(super) ns_id: NamespaceId,
    pub(super) store: Arc<KVStore>,
    pub(super) db: &'db Database,
}

impl<'db> Namespace<'db> {
    /// The namespace ID.
    pub fn id(&self) -> NamespaceId {
        self.ns_id
    }

    /// The TTL for this namespace, if configured.
    pub fn ttl(&self) -> Option<Duration> {
        self.store.ttl
    }

    // ── CRUD ──────────────────────────────────────────────────────────

    /// Insert or update a key-value pair in this namespace.
    ///
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](Db#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.db.put_ns(self.ns_id, key, value)
    }

    /// Look up a value by key. Returns `None` if the key does not exist.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.db.get_ns(self.ns_id, key)
    }

    /// Delete a key from this namespace.
    ///
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](Db#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub fn delete(&self, key: &[u8]) -> Result<()> {
        self.db.delete_ns(self.ns_id, key)
    }

    /// Write a key-value pair without the WAL — see [`Database::put_ns_no_wal`]
    /// for the crash-safety trade-off. Mirrors [`AsyncNamespace::put_no_wal`].
    pub fn put_no_wal(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.db.put_ns_no_wal(self.ns_id, key, value)
    }

    /// Delete a key without the WAL — for derived or regenerable data such as a
    /// TTL cache. Mirrors [`AsyncNamespace::delete_no_wal`].
    pub fn delete_no_wal(&self, key: &[u8]) -> Result<()> {
        self.db.delete_ns_no_wal(self.ns_id, key)
    }

    /// Fetch several keys in one pass, one `Option` per key in input order
    /// (`None` = absent or deleted). A storage error fails the whole call.
    /// Mirrors [`AsyncNamespace::get_multiple`].
    pub fn get_multiple(&self, keys: &[Vec<u8>]) -> Result<Vec<Option<Vec<u8>>>> {
        self.store.get_multiple(keys)
    }

    /// Scan several 4-byte big-endian `u32` prefixes in one pass, returning every
    /// `(prefix_id, key, value)` in key order. Mirrors
    /// [`AsyncNamespace::scan_prefixes_batch`].
    pub fn scan_prefixes_batch(&self, prefix_ids: &[u32]) -> Result<crate::kv_store::PrefixScan> {
        self.store.scan_prefixes_batch(prefix_ids)
    }

    /// Atomically read-modify-write a key in this namespace.
    ///
    /// See [`Db::merge`] for the full contract — the closure's return values, the
    /// durability guarantee, and the rule that `merge_fn` must not call back into
    /// the database.
    pub fn merge<F>(&self, key: &[u8], value: &[u8], merge_fn: F) -> Result<Option<Vec<u8>>>
    where
        F: FnOnce(Option<&[u8]>, &[u8]) -> Result<Option<Vec<u8>>>,
    {
        self.db.merge_ns(self.ns_id, key, value, merge_fn)
    }

    // ── Iteration ─────────────────────────────────────────────────────

    /// Iterate over all key-value pairs in key order.
    pub fn iter(&self) -> Result<Vec<KeyValue>> {
        self.store.scan_range_batch(&[], None)
    }

    /// Iterate over all keys.
    pub fn keys(&self) -> Result<Vec<Vec<u8>>> {
        self.store.keys()
    }

    /// Iterate over key-value pairs in `[start, end)`.
    pub fn range(&self, start: &[u8], end: Option<&[u8]>) -> Result<Vec<KeyValue>> {
        self.store.scan_range_batch(start, end)
    }

    /// Iterate over key-value pairs whose keys start with `prefix`.
    pub fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<KeyValue>> {
        self.store.scan_prefix_batch(prefix)
    }

    /// Paginated cursor scan over `[cursor, end)` on this namespace.
    ///
    /// Returns up to `limit` key-value pairs starting from `cursor`
    /// (or the beginning if `None`) and ending before `end` (or the last key if
    /// `None`), plus an optional next cursor.
    pub fn scan(&self, cursor: Option<&[u8]>, end: Option<&[u8]>, limit: usize) -> Result<ScanPage> {
        self.store.scan_page_batch(cursor, end, limit)
    }

    // ── Maintenance ───────────────────────────────────────────────────

    /// Returns snapshot statistics for this namespace.
    pub fn stats(&self) -> Stats {
        self.store.stats()
    }

    /// Run value-log garbage collection on this namespace.
    pub fn garbage_collect(&self) -> Result<GCStats> {
        self.db.garbage_collect_namespace(self.ns_id)
    }

    // ── Typed CRUD (rkyv ser/de) ──────────────────────────────────────

    /// Insert a typed key-value pair, serialized via rkyv.
    ///
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](Db#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub fn put_typed<K, V>(&self, key: &K, value: &V) -> Result<()>
    where
        K: RkyvEncode,
        V: RkyvEncode,
    {
        let kb = rkyv_serialize(key)?;
        let vb = rkyv_serialize(value)?;
        self.put(&kb, &vb)
    }

    /// Look up a typed value by typed key.
    pub fn get_typed<K, V>(&self, key: &K) -> Result<Option<V>>
    where
        K: RkyvEncode,
        V: RkyvDecode,
    {
        let kb = rkyv_serialize(key)?;
        match self.db.get_ns(self.ns_id, &kb)? {
            Some(vb) => Ok(Some(rkyv_deserialize::<V>(&vb)?)),
            None => Ok(None),
        }
    }

    /// Delete by typed key.
    ///
    /// Returns `Ok(())` once the write is durable. See [*Write durability*](Db#write-durability)
    /// for what that does and does not guarantee about immediate readability.
    pub fn delete_typed<K>(&self, key: &K) -> Result<()>
    where
        K: RkyvEncode,
    {
        let kb = rkyv_serialize(key)?;
        self.delete(&kb)
    }

    /// Atomically merge a typed value, serialized via rkyv.
    ///
    /// The typed twin of [`Db::merge`]; see [`Db::merge_typed`] for an example.
    pub fn merge_typed<K, V, F>(&self, key: &K, value: &V, merge_fn: F) -> Result<Option<V>>
    where
        K: RkyvEncode,
        V: RkyvEncode + RkyvDecode,
        F: FnOnce(Option<V>, V) -> Result<Option<V>>,
    {
        let kb = rkyv_serialize(key)?;
        let vb = rkyv_serialize(value)?;
        // The new `V` is carried out of the closure rather than deserialized back
        // from the bytes `merge` returns — it already exists in there, and a
        // second round-trip on every call would be pure waste.
        let mut produced: Option<V> = None;
        {
            let slot = &mut produced;
            self.merge(&kb, &vb, move |existing, operand| {
                let current = match existing {
                    Some(bytes) => Some(rkyv_deserialize::<V>(bytes)?),
                    None => None,
                };
                let operand = rkyv_deserialize::<V>(operand)?;
                match merge_fn(current, operand)? {
                    Some(new) => {
                        let bytes = rkyv_serialize(&new)?;
                        *slot = Some(new);
                        Ok(Some(bytes))
                    }
                    None => Ok(None),
                }
            })?;
        }
        Ok(produced)
    }

    // ── Typed Iteration (rkyv ser/de) ─────────────────────────────────

    /// Iterate over all typed key-value pairs in key order.
    pub fn iter_typed<K, V>(&self) -> Result<Vec<(K, V)>>
    where
        K: RkyvDecode,
        V: RkyvDecode,
    {
        deserialize_pairs(self.iter()?)
    }

    /// Iterate over all typed keys.
    pub fn keys_typed<K>(&self) -> Result<Vec<K>>
    where
        K: RkyvDecode,
    {
        deserialize_keys(self.keys()?)
    }

    /// Iterate over typed key-value pairs in `[start, end)`.
    pub fn range_typed<K, V>(&self, start: &K, end: Option<&K>) -> Result<Vec<(K, V)>>
    where
        K: RkyvEncode + RkyvDecode,
        V: RkyvDecode,
    {
        let start_bytes = rkyv_serialize(start)?;
        let end_bytes = end.map(|e| rkyv_serialize(e)).transpose()?;
        deserialize_pairs(self.range(&start_bytes, end_bytes.as_deref())?)
    }

    /// Iterate over typed key-value pairs whose keys start with `prefix`.
    pub fn scan_prefix_typed<K, V>(&self, prefix: &[u8]) -> Result<Vec<(K, V)>>
    where
        K: RkyvDecode,
        V: RkyvDecode,
    {
        deserialize_pairs(self.scan_prefix(prefix)?)
    }
}
