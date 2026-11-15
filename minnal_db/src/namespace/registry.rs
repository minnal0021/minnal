//! `NamespaceRegistry`: every namespace's name, id, TTL settings and field
//! indexes, in memory, read from and saved to the system catalog
//! (`catalog.rs`).
//!
//! `default`, `system` and `system_catalog` always exist, with fixed ids.
//! Every other id comes from the catalog's counter (`next_id`), which is saved
//! before the id is used, so an id is never handed out twice: a name dropped
//! and created again gets a new id. Field ids come from the same counter.
//!
//! The registry does no I/O. Each change records the catalog writes it needs
//! ([`take_writes`](NamespaceRegistry::take_writes)); `Database::update_registry`
//! makes a change on a copy, writes them durably, and only then swaps the copy
//! in.

use std::collections::HashMap;
use std::time::Duration;

use super::NamespaceId;
use super::catalog::{self, CATALOG_NAMESPACE, CatalogEntry, CatalogWrite};
use super::schema::{NamespaceSchema, Registered};
use crate::error::{KVError, Result};
use crate::field_index::{FieldId, IndexValueType};

/// Default namespace name, always present with the fixed id [`NamespaceId::DEFAULT`].
pub const DEFAULT_NAMESPACE: &str = "default";
pub const DEFAULT_NAMESPACE_ID: NamespaceId = NamespaceId::DEFAULT;

/// System namespace name, always present with the fixed id [`NamespaceId::SYSTEM`].
///
/// Used for system-wide internal stores such as the query-embedding cache
/// shared across all doc-store namespaces.
pub const SYSTEM_NAMESPACE: &str = "system";
pub const SYSTEM_NAMESPACE_ID: NamespaceId = NamespaceId::SYSTEM;

/// The suffixes that mark a namespace as holding another's vector index: its
/// vector data and its partition (centres, postings and maintenance journal).
/// Here, not in `vector_kv`, because store names must avoid them in every
/// build, with or without semantic search.
///
/// Listed longest-first so a suffix search cannot mistake `x_sparse_vector_meta`
/// for `x_sparse_vector`.
pub const VECTOR_COMPANION_SUFFIXES: [&str; 6] = [
    "_sparse_vector_meta",
    "_sparse_vector",
    "_dense_vector",
    "_ivf_centres",
    "_ivf_postings",
    "_ivf_journal",
];

/// The namespaces every database has, with their fixed ids. The catalog's own
/// record is implicit; the others are saved like any namespace (they can have
/// TTL settings and field indexes).
const FIXED: [(&str, NamespaceId); 3] = [
    (DEFAULT_NAMESPACE, NamespaceId::DEFAULT),
    (SYSTEM_NAMESPACE, NamespaceId::SYSTEM),
    (CATALOG_NAMESPACE, NamespaceId::CATALOG),
];

/// Every namespace's name, id, TTL settings and field indexes. See the module docs.
#[derive(Debug, Clone)]
pub struct NamespaceRegistry {
    /// name -> id
    namespaces: HashMap<String, NamespaceId>,
    /// id -> name (for reverse lookups)
    id_to_name: HashMap<NamespaceId, String>,
    /// namespace id -> its field indexes
    schemas: HashMap<NamespaceId, NamespaceSchema>,
    /// namespace id -> (ttl, max_deletes_per_run) for TTL-enabled namespaces.
    ttl_configs: HashMap<NamespaceId, (Duration, usize)>,
    /// The next id the counter hands out.
    next_id: u64,
    /// Catalog writes of changes not yet saved.
    pending: Vec<CatalogWrite>,
}

impl NamespaceRegistry {
    /// A new database's registry: the fixed namespaces only. Their records and
    /// the counter are pending, to be saved once the database is open.
    pub(crate) fn new() -> Self {
        let mut r = Self {
            namespaces: HashMap::new(),
            id_to_name: HashMap::new(),
            schemas: HashMap::new(),
            ttl_configs: HashMap::new(),
            next_id: NamespaceId::FIRST_ALLOCATED,
            pending: Vec::new(),
        };
        r.pending.push(catalog::next_id_write(r.next_id));
        for (name, id) in FIXED {
            r.insert(name, id);
            if id != NamespaceId::CATALOG {
                r.pending.push(catalog::namespace_write(name, id, None));
            }
        }
        r
    }

    /// The registry the catalog's records describe; [`new`](Self::new) for an
    /// empty catalog. Records a database cannot use — a namespace whose name
    /// cannot be a directory, a field of a namespace that does not exist —
    /// are left out, and their deletes are pending.
    pub(crate) fn from_catalog(records: impl IntoIterator<Item = (Vec<u8>, Vec<u8>)>) -> Result<Self> {
        let mut entries = Vec::new();
        for (key, value) in records {
            entries.push((key.clone(), catalog::decode(&key, &value)?));
        }
        if entries.is_empty() {
            return Ok(Self::new());
        }
        let mut r = Self {
            namespaces: HashMap::new(),
            id_to_name: HashMap::new(),
            schemas: HashMap::new(),
            ttl_configs: HashMap::new(),
            next_id: 0,
            pending: Vec::new(),
        };
        r.insert(CATALOG_NAMESPACE, NamespaceId::CATALOG);
        let mut next_id = None;
        let mut fields = Vec::new();
        for (key, entry) in entries {
            match entry {
                CatalogEntry::NextId(n) => next_id = Some(n),
                CatalogEntry::Namespace { name, id, ttl } => {
                    if let Some(problem) = crate::layout::namespace_name_problem(&name) {
                        // A failed create once left such an entry, and opening
                        // it failed every later start of the database.
                        log::error!("[catalog] removing namespace {id}, which can hold no data: {problem}");
                        r.pending.push(CatalogWrite::Delete { key });
                        continue;
                    }
                    r.insert(&name, id);
                    if let Some(ttl) = ttl {
                        r.ttl_configs.insert(id, ttl);
                    }
                }
                CatalogEntry::Field { ns_id, meta } => fields.push((key, ns_id, meta)),
            }
        }
        for (key, ns_id, meta) in fields {
            match r.schemas.get_mut(&ns_id) {
                Some(schema) => schema.insert(meta),
                None => {
                    log::warn!(
                        "[catalog] removing field '{}' of namespace {ns_id}, which no longer exists",
                        meta.field_name
                    );
                    r.pending.push(CatalogWrite::Delete { key });
                }
            }
        }
        for (name, id) in FIXED {
            if !r.namespaces.contains_key(name) {
                r.insert(name, id);
                r.pending.push(catalog::namespace_write(name, id, None));
            }
        }
        // The counter is saved before any id it hands out is used, so it is
        // above every id on record; this only repairs a missing counter.
        let above_all = r
            .id_to_name
            .keys()
            .map(|id| id.as_u64())
            .chain(r.schemas.values().flat_map(|s| s.field_ids()))
            .max()
            .map_or(0, |m| m + 1)
            .max(NamespaceId::FIRST_ALLOCATED);
        r.next_id = match next_id {
            Some(n) if n >= above_all => n,
            other => {
                log::warn!("[catalog] the id counter is {other:?}, below an id in use; moving it to {above_all}");
                r.pending.push(catalog::next_id_write(above_all));
                above_all
            }
        };
        Ok(r)
    }

    fn insert(&mut self, name: &str, id: NamespaceId) {
        self.namespaces.insert(name.to_string(), id);
        self.id_to_name.insert(id, name.to_string());
        self.schemas.insert(id, NamespaceSchema::default());
    }

    /// The next id of the counter. Its new value is the first pending write,
    /// so it is saved before anything that uses the id.
    fn allocate_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.pending.push(catalog::next_id_write(self.next_id));
        id
    }

    /// The catalog writes of the changes made since the last call, in order.
    pub(crate) fn take_writes(&mut self) -> Vec<CatalogWrite> {
        std::mem::take(&mut self.pending)
    }

    /// Create a new namespace and return its id, a new one from the counter.
    /// Returns an error if the name already exists.
    pub fn create(&mut self, name: &str) -> Result<NamespaceId> {
        if let Some(problem) = crate::layout::namespace_name_problem(name) {
            return Err(KVError::Serialization(problem));
        }
        if self.namespaces.contains_key(name) {
            return Err(KVError::Serialization(format!("Namespace '{}' already exists", name)));
        }
        let id = NamespaceId::from_u64(self.allocate_id());
        self.insert(name, id);
        self.pending.push(catalog::namespace_write(name, id, None));
        Ok(id)
    }

    /// Get the ID for a namespace by name
    pub fn get_id(&self, name: &str) -> Option<NamespaceId> {
        self.namespaces.get(name).copied()
    }

    /// Get the name for a namespace by ID
    pub fn get_name(&self, id: NamespaceId) -> Option<&str> {
        self.id_to_name.get(&id).map(|s| s.as_str())
    }

    /// List all active namespace names and their IDs
    pub fn list(&self) -> Vec<(&str, NamespaceId)> {
        self.namespaces.iter().map(|(name, &id)| (name.as_str(), id)).collect()
    }

    /// Check if a namespace exists
    pub fn exists(&self, name: &str) -> bool {
        self.namespaces.contains_key(name)
    }

    /// Remove a namespace, with its TTL settings and field records. The
    /// namespaces with fixed ids cannot be removed.
    pub fn remove(&mut self, name: &str) -> Result<NamespaceId> {
        if FIXED.iter().any(|&(fixed, _)| fixed == name) {
            return Err(KVError::Serialization(format!("Cannot remove the {name} namespace")));
        }
        let id = self
            .namespaces
            .remove(name)
            .ok_or_else(|| KVError::Serialization(format!("Namespace '{}' does not exist", name)))?;
        self.id_to_name.remove(&id);
        self.ttl_configs.remove(&id);
        // The namespace's record first: once it is gone the namespace is, and
        // a field record left by a crash is removed at the next open.
        self.pending.push(CatalogWrite::Delete {
            key: catalog::namespace_key(name),
        });
        if let Some(schema) = self.schemas.remove(&id) {
            for f in schema.list_fields() {
                self.pending.push(CatalogWrite::Delete {
                    key: catalog::field_key(id, &f.field_name),
                });
            }
        }
        Ok(id)
    }

    fn save_namespace(&mut self, ns_id: NamespaceId) -> Result<()> {
        let name = self
            .id_to_name
            .get(&ns_id)
            .ok_or_else(|| KVError::Serialization(format!("Namespace {ns_id} not found")))?;
        if ns_id == NamespaceId::CATALOG {
            return Err(KVError::Serialization("the system catalog has no settings to change".to_owned()));
        }
        self.pending
            .push(catalog::namespace_write(name, ns_id, self.ttl_configs.get(&ns_id).copied()));
        Ok(())
    }

    /// Record the TTL configuration for a namespace, so the global TTL worker
    /// expires its records after `ttl` (capped at `max_deletes` per pass) and
    /// the configuration is restored on the next open.
    pub fn set_ttl_config(&mut self, ns_id: NamespaceId, ttl: Duration, max_deletes: usize) -> Result<()> {
        self.ttl_configs.insert(ns_id, (ttl, max_deletes));
        self.save_namespace(ns_id)
    }

    /// Stop expiring a namespace by removing its TTL config. A no-op if the
    /// namespace has no TTL configured.
    pub fn remove_ttl_config(&mut self, ns_id: NamespaceId) -> Result<()> {
        if self.ttl_configs.remove(&ns_id).is_some() {
            self.save_namespace(ns_id)?;
        }
        Ok(())
    }

    /// The TTL configuration for a namespace, if one is set.
    pub fn ttl_config(&self, ns_id: NamespaceId) -> Option<(Duration, usize)> {
        self.ttl_configs.get(&ns_id).copied()
    }

    /// Snapshot of every namespace's TTL configuration. Used by the global TTL
    /// worker each tick and to restore TTL stores on open.
    pub fn ttl_configs(&self) -> Vec<(NamespaceId, (Duration, usize))> {
        self.ttl_configs.iter().map(|(id, cfg)| (*id, *cfg)).collect()
    }

    // ── Schema accessors ──────────────────────────────────────────────────

    /// Borrow the schema for a namespace.
    pub fn schema(&self, ns_id: NamespaceId) -> Option<&NamespaceSchema> {
        self.schemas.get(&ns_id)
    }

    /// Register an indexed field of a namespace. A new field takes its id from
    /// the counter.
    ///
    /// Idempotent: if the field already exists with the same type the existing
    /// [`FieldId`] is returned without error.
    pub fn register_schema_field(&mut self, ns_id: NamespaceId, field_name: &str, field_type: IndexValueType) -> Result<FieldId> {
        let mut schema = self
            .schemas
            .remove(&ns_id)
            .ok_or_else(|| KVError::Serialization(format!("Namespace {} not found", ns_id)))?;
        let registered = schema.register_field(field_name, field_type, || self.allocate_id());
        self.schemas.insert(ns_id, schema);
        match registered? {
            Registered::Unchanged(id) => Ok(id),
            Registered::Changed(meta) => {
                self.pending.push(catalog::field_write(ns_id, &meta));
                Ok(meta.field_id)
            }
        }
    }

    /// Mark a field's index as dropped. Saved **before** any file is deleted.
    ///
    /// Returns `false` if no live field has `field_id` (nothing to mark).
    /// Saving first is what makes an interrupted drop completable at the next
    /// open (`complete_interrupted_field_drops`).
    pub fn mark_schema_field_dropped(&mut self, ns_id: NamespaceId, field_id: FieldId) -> Result<bool> {
        let schema = self
            .schemas
            .get_mut(&ns_id)
            .ok_or_else(|| KVError::Serialization(format!("Namespace {} not found", ns_id)))?;
        match schema.mark_field_dropped(field_id) {
            Some(meta) => {
                self.pending.push(catalog::field_write(ns_id, &meta));
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Return all `(namespace_id, field_id)` pairs across every namespace,
    /// suitable for use by the index checkpoint worker.
    ///
    /// **Excludes dropped fields.** Their definitions are kept for id reuse
    /// but their on-disk directories are gone, so checkpointing them would fail;
    /// and a dropped field's frozen checkpoint marker must never hold up WAL GC.
    pub fn all_indexed_fields(&self) -> Vec<(NamespaceId, FieldId)> {
        self.schemas
            .iter()
            .flat_map(|(ns_id, schema)| schema.live_field_ids().into_iter().map(|fid| (*ns_id, fid)))
            .collect()
    }
}

#[cfg(test)]
mod tests;
