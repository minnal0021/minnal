use crc32fast::Hasher;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
/// Namespace Registry
///
/// Persistent mapping of namespace names to unique u32 IDs.
/// The "default" namespace always exists with ID 0.
/// IDs are never reused even after deletion.
///
/// Each namespace may also carry a [`NamespaceConfig`] on disk — a JSON file
/// at `{db_path}/ns_{name}/config.json` — that stores the field schema for
/// namespaces that use indexed fields.  KV-only namespaces have no config file.
/// The schema is loaded automatically on [`NamespaceRegistry::open`] so field
/// definitions survive restarts without the caller re-registering them.
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::db::error::{KVError, Result};
use index::IndexValueType;

// ── Namespace config (persisted per namespace directory) ───────────────────

/// A single field definition as stored in `config.json`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FieldDef {
    pub field_id: FieldId,
    pub field_name: String,
    pub field_type: IndexValueType,
}

/// The schema section of a namespace config: all registered field definitions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SchemaConfig {
    pub fields: Vec<FieldDef>,
    pub next_field_id: FieldId,
}

/// Persisted configuration for a namespace, stored at
/// `{db_path}/ns_{name}/config.json`.
///
/// A KV-only namespace has no config file (all fields `None`).
/// The presence of `schema` indicates at least one index field has been
/// registered, making this namespace a structured doc store.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct NamespaceConfig {
    pub schema: Option<SchemaConfig>,
}

// ── Field / schema types ───────────────────────────────────────────────────

/// Unique identifier for an indexed field within a namespace.
/// Assigned monotonically; never reused after a field is removed.
pub type FieldId = u32;

/// Metadata for a single indexed field within a namespace.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldMeta {
    /// Unique field identifier within the namespace.
    pub field_id: FieldId,
    /// Human-readable field name (e.g. `"status"`, `"user_id"`).
    pub field_name: String,
    /// The value type this field was registered with.
    ///
    /// `activate_field_index` validates that the caller-supplied type matches
    /// this so mismatches are caught at activation time rather than at query time.
    pub field_type: IndexValueType,
}

/// Outcome of a targeted single-field reindex ([`crate::Db::reindex_field`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldReindexOutcome {
    /// The field index entry for the key was re-derived from its current value
    /// and rewritten.
    Reindexed,
    /// The key has no stored value (absent or deleted), so there was nothing to
    /// index.
    KeyNotFound,
    /// No active index exists for the given field id in the namespace (it was
    /// never registered, or is not currently activated — e.g. still building).
    FieldNotActive,
}

/// Schema of a single namespace: the in-memory map of indexed fields.
///
/// Loaded from `config.json` on startup when present, so field definitions
/// survive restarts without the caller re-registering them.
pub struct NamespaceSchema {
    /// Primary map: field name → FieldMeta.
    fields: HashMap<String, FieldMeta>,
    /// Monotonically increasing; never reused.
    next_field_id: FieldId,
}

impl NamespaceSchema {
    fn new() -> Self {
        Self {
            fields: HashMap::new(),
            next_field_id: 0,
        }
    }

    /// Restore a schema from a persisted [`SchemaConfig`].
    fn from_config(cfg: SchemaConfig) -> Self {
        let mut fields = HashMap::new();
        for f in cfg.fields {
            fields.insert(
                f.field_name.clone(),
                FieldMeta {
                    field_id: f.field_id,
                    field_name: f.field_name,
                    field_type: f.field_type,
                },
            );
        }
        Self {
            fields,
            next_field_id: cfg.next_field_id,
        }
    }

    /// Register a new indexed field and return its assigned [`FieldId`].
    ///
    /// **Idempotent**: if a field with the same name and type already exists
    /// the existing [`FieldId`] is returned without error.  A name collision
    /// with a *different* type is an error.
    pub fn register_field(&mut self, field_name: &str, field_type: IndexValueType) -> Result<FieldId> {
        if let Some(existing) = self.fields.get(field_name) {
            if existing.field_type != field_type {
                return Err(KVError::Serialization(format!(
                    "Field '{}' is already registered as {:?}, cannot re-register as {:?}",
                    field_name, existing.field_type, field_type
                )));
            }
            return Ok(existing.field_id);
        }
        let field_id = self.next_field_id;
        self.next_field_id += 1;
        self.fields.insert(
            field_name.to_string(),
            FieldMeta {
                field_id,
                field_name: field_name.to_string(),
                field_type,
            },
        );
        Ok(field_id)
    }

    /// Look up a field by name — the primary query-time access pattern, O(1).
    pub fn get_field_by_name(&self, field_name: &str) -> Option<&FieldMeta> {
        self.fields.get(field_name)
    }

    /// Look up a field by ID — used for index path resolution, O(n).
    pub fn get_field(&self, field_id: FieldId) -> Option<&FieldMeta> {
        self.fields.values().find(|f| f.field_id == field_id)
    }

    /// Return all fields sorted by `FieldId`.
    pub fn list_fields(&self) -> Vec<FieldMeta> {
        let mut fields: Vec<FieldMeta> = self.fields.values().cloned().collect();
        fields.sort_by_key(|f| f.field_id);
        fields
    }

    /// Return all registered `FieldId`s sorted ascending.
    pub fn field_ids(&self) -> Vec<FieldId> {
        let mut ids: Vec<FieldId> = self.fields.values().map(|f| f.field_id).collect();
        ids.sort();
        ids
    }
}

const REGISTRY_MAGIC: [u8; 4] = *b"NSRG";
const REGISTRY_VERSION: u32 = 1;
const REGISTRY_FILENAME: &str = "namespace_registry";
const CONFIG_FILENAME: &str = "config.json";

/// Default namespace name, always present with ID 0
pub const DEFAULT_NAMESPACE: &str = "default";
pub const DEFAULT_NAMESPACE_ID: u32 = 0;

/// System namespace name, always present with ID 1.
///
/// Used for system-wide internal stores such as the query-embedding cache
/// shared across all doc-store namespaces.
pub const SYSTEM_NAMESPACE: &str = "system";
pub const SYSTEM_NAMESPACE_ID: u32 = 1;

/// Persisted TTL configuration for one namespace (part of [`RegistryData`]).
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
struct TtlConfigData {
    ns_id: u32,
    /// Record lifetime in milliseconds.
    ttl_millis: u64,
    /// Cap on records tombstoned per TTL pass.
    max_deletes: u64,
}

/// Persistent namespace registry data (the `namespace_registry` binary file).
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
struct RegistryData {
    /// Next ID to assign (monotonically increasing, never reused)
    next_id: u32,
    /// Active namespace mappings: name -> id
    names: Vec<String>,
    ids: Vec<u32>,
    /// Per-namespace TTL configuration (only TTL-enabled namespaces appear here).
    ttl_configs: Vec<TtlConfigData>,
}

impl RegistryData {
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            next_id: 2, // 0 is reserved for "default", 1 for "system"
            names: vec![DEFAULT_NAMESPACE.to_string(), SYSTEM_NAMESPACE.to_string()],
            ids: vec![DEFAULT_NAMESPACE_ID, SYSTEM_NAMESPACE_ID],
            ttl_configs: Vec::new(),
        }
    }
}

/// In-memory namespace registry backed by a binary file on disk.
///
/// Also owns the [`NamespaceSchema`] for each namespace.  Schemas are loaded
/// from per-namespace `config.json` files on startup and persisted back
/// whenever a field is registered, so field definitions survive restarts.
pub struct NamespaceRegistry {
    /// Root database directory — used to locate per-namespace config files.
    db_path: PathBuf,
    /// Path to the `namespace_registry` binary file.
    path: PathBuf,
    /// name -> id
    namespaces: HashMap<String, u32>,
    /// id -> name (for reverse lookups)
    id_to_name: HashMap<u32, String>,
    next_id: u32,
    /// namespace_id -> schema (loaded from config.json on open; persisted on mutation)
    schemas: HashMap<u32, NamespaceSchema>,
    /// namespace_id -> (ttl, max_deletes_per_run) for TTL-enabled namespaces.
    /// Persisted in the registry file and reloaded on open, so TTL survives
    /// restarts without the caller re-declaring it.
    ttl_configs: HashMap<u32, (Duration, usize)>,
}

impl NamespaceRegistry {
    /// Open or create a namespace registry rooted at `db_path`.
    pub fn open(db_path: &Path) -> Result<Self> {
        let path = db_path.join(REGISTRY_FILENAME);

        if path.exists() {
            let bytes =
                fs::read(&path).map_err(|e| KVError::Io(std::io::Error::new(e.kind(), format!("Failed to read namespace registry: {}", e))))?;
            let data = Self::deserialize_file(&bytes)?;
            let mut namespaces = HashMap::new();
            let mut id_to_name = HashMap::new();
            let mut schemas = HashMap::new();
            for (name, id) in data.names.iter().zip(data.ids.iter()) {
                namespaces.insert(name.clone(), *id);
                id_to_name.insert(*id, name.clone());
                let schema = match Self::load_config(db_path, name)? {
                    Some(cfg) if cfg.schema.is_some() => NamespaceSchema::from_config(cfg.schema.unwrap()),
                    _ => NamespaceSchema::new(),
                };
                schemas.insert(*id, schema);
            }
            let ttl_configs = data
                .ttl_configs
                .iter()
                .map(|c| (c.ns_id, (Duration::from_millis(c.ttl_millis), c.max_deletes as usize)))
                .collect();
            Ok(Self {
                db_path: db_path.to_path_buf(),
                path,
                namespaces,
                id_to_name,
                next_id: data.next_id,
                schemas,
                ttl_configs,
            })
        } else {
            let mut registry = Self {
                db_path: db_path.to_path_buf(),
                path,
                namespaces: HashMap::new(),
                id_to_name: HashMap::new(),
                next_id: 2, // 0 = default, 1 = system
                schemas: HashMap::new(),
                ttl_configs: HashMap::new(),
            };
            registry.namespaces.insert(DEFAULT_NAMESPACE.to_string(), DEFAULT_NAMESPACE_ID);
            registry.id_to_name.insert(DEFAULT_NAMESPACE_ID, DEFAULT_NAMESPACE.to_string());
            registry.schemas.insert(DEFAULT_NAMESPACE_ID, NamespaceSchema::new());
            registry.namespaces.insert(SYSTEM_NAMESPACE.to_string(), SYSTEM_NAMESPACE_ID);
            registry.id_to_name.insert(SYSTEM_NAMESPACE_ID, SYSTEM_NAMESPACE.to_string());
            registry.schemas.insert(SYSTEM_NAMESPACE_ID, NamespaceSchema::new());
            registry.persist()?;
            Ok(registry)
        }
    }

    /// Return the directory path used for namespace `name`: `{db_path}/ns_{name}`.
    pub fn ns_dir(&self, name: &str) -> PathBuf {
        self.db_path.join(format!("ns_{}", name))
    }

    /// Create a new namespace and return its ID.
    /// Returns an error if the name already exists.
    pub fn create(&mut self, name: &str) -> Result<u32> {
        if self.namespaces.contains_key(name) {
            return Err(KVError::Serialization(format!("Namespace '{}' already exists", name)));
        }
        let id = self.next_id;
        self.next_id += 1;
        self.namespaces.insert(name.to_string(), id);
        self.id_to_name.insert(id, name.to_string());
        self.schemas.insert(id, NamespaceSchema::new());
        self.persist()?;
        Ok(id)
    }

    /// Get the ID for a namespace by name
    pub fn get_id(&self, name: &str) -> Option<u32> {
        self.namespaces.get(name).copied()
    }

    /// Get the name for a namespace by ID
    #[allow(dead_code)]
    pub fn get_name(&self, id: u32) -> Option<&str> {
        self.id_to_name.get(&id).map(|s| s.as_str())
    }

    /// List all active namespace names and their IDs
    pub fn list(&self) -> Vec<(&str, u32)> {
        self.namespaces.iter().map(|(name, &id)| (name.as_str(), id)).collect()
    }

    /// Check if a namespace exists
    pub fn exists(&self, name: &str) -> bool {
        self.namespaces.contains_key(name)
    }

    /// Get all namespace IDs
    #[allow(dead_code)]
    pub fn all_ids(&self) -> Vec<u32> {
        self.namespaces.values().copied().collect()
    }

    /// Remove a namespace. Cannot remove the default or system namespace.
    pub fn remove(&mut self, name: &str) -> Result<u32> {
        if name == DEFAULT_NAMESPACE {
            return Err(KVError::Serialization("Cannot remove the default namespace".to_string()));
        }
        if name == SYSTEM_NAMESPACE {
            return Err(KVError::Serialization("Cannot remove the system namespace".to_string()));
        }
        match self.namespaces.remove(name) {
            Some(id) => {
                self.id_to_name.remove(&id);
                self.schemas.remove(&id);
                self.ttl_configs.remove(&id);
                self.persist()?;
                Ok(id)
            }
            None => Err(KVError::Serialization(format!("Namespace '{}' does not exist", name))),
        }
    }

    /// Record (and persist) the TTL configuration for a namespace, so the global
    /// TTL worker expires its records after `ttl` (capped at `max_deletes` per
    /// pass) and the configuration is restored on the next open.
    pub fn set_ttl_config(&mut self, ns_id: u32, ttl: Duration, max_deletes: usize) -> Result<()> {
        self.ttl_configs.insert(ns_id, (ttl, max_deletes));
        self.persist()
    }

    /// Stop expiring a namespace by removing (and persisting) its TTL config.
    /// A no-op if the namespace has no TTL configured.
    pub fn remove_ttl_config(&mut self, ns_id: u32) -> Result<()> {
        if self.ttl_configs.remove(&ns_id).is_some() {
            self.persist()?;
        }
        Ok(())
    }

    /// The TTL configuration for a namespace, if one is set.
    pub fn ttl_config(&self, ns_id: u32) -> Option<(Duration, usize)> {
        self.ttl_configs.get(&ns_id).copied()
    }

    /// Snapshot of every namespace's TTL configuration. Used by the global TTL
    /// worker each tick and to restore TTL stores on open.
    pub fn ttl_configs(&self) -> Vec<(u32, (Duration, usize))> {
        self.ttl_configs.iter().map(|(id, cfg)| (*id, *cfg)).collect()
    }

    // ── Schema accessors ──────────────────────────────────────────────────

    /// Borrow the schema for a namespace.
    pub fn schema(&self, ns_id: u32) -> Option<&NamespaceSchema> {
        self.schemas.get(&ns_id)
    }

    /// Mutably borrow the schema for a namespace.
    pub fn schema_mut(&mut self, ns_id: u32) -> Option<&mut NamespaceSchema> {
        self.schemas.get_mut(&ns_id)
    }

    /// Register an indexed field in the namespace schema and persist the change
    /// to `config.json`.
    ///
    /// Idempotent: if the field already exists with the same type the existing
    /// [`FieldId`] is returned without error.
    pub fn register_schema_field(&mut self, ns_id: u32, field_name: &str, field_type: IndexValueType) -> Result<FieldId> {
        let field_id = self
            .schemas
            .get_mut(&ns_id)
            .ok_or_else(|| KVError::Serialization(format!("Namespace {} not found", ns_id)))?
            .register_field(field_name, field_type)?;
        self.persist_schema(ns_id)?;
        Ok(field_id)
    }

    /// Return all `(namespace_id, field_id)` pairs across every namespace,
    /// suitable for use by the index checkpoint worker.
    pub fn all_indexed_fields(&self) -> Vec<(u32, FieldId)> {
        self.schemas
            .iter()
            .flat_map(|(ns_id, schema)| schema.field_ids().into_iter().map(|fid| (*ns_id, fid)))
            .collect()
    }

    // ── Config persistence ────────────────────────────────────────────────

    /// Read `{db_path}/ns_{name}/config.json`, returning `None` if absent.
    fn load_config(db_path: &Path, name: &str) -> Result<Option<NamespaceConfig>> {
        let config_path = db_path.join(format!("ns_{}", name)).join(CONFIG_FILENAME);
        if !config_path.exists() {
            return Ok(None);
        }
        let bytes =
            fs::read(&config_path).map_err(|e| KVError::Io(std::io::Error::new(e.kind(), format!("Failed to read namespace config: {}", e))))?;
        let config: NamespaceConfig = serde_json::from_slice(&bytes)
            .map_err(|e| KVError::Serialization(format!("Failed to deserialize namespace config for '{}': {}", name, e)))?;
        Ok(Some(config))
    }

    /// Write the current schema for `ns_id` to `{db_path}/ns_{name}/config.json`.
    ///
    /// Uses an atomic tmp-then-rename write.  No-ops for namespaces with no
    /// registered fields (KV-only namespaces get no config file).
    fn persist_schema(&self, ns_id: u32) -> Result<()> {
        let name = self
            .id_to_name
            .get(&ns_id)
            .ok_or_else(|| KVError::Serialization(format!("Namespace {} not found", ns_id)))?;
        let schema = self
            .schemas
            .get(&ns_id)
            .ok_or_else(|| KVError::Serialization(format!("Schema for namespace {} not found", ns_id)))?;

        if schema.fields.is_empty() {
            return Ok(());
        }

        let config = NamespaceConfig {
            schema: Some(SchemaConfig {
                fields: schema
                    .list_fields()
                    .into_iter()
                    .map(|f| FieldDef {
                        field_id: f.field_id,
                        field_name: f.field_name,
                        field_type: f.field_type,
                    })
                    .collect(),
                next_field_id: schema.next_field_id,
            }),
        };

        let config_dir = self.db_path.join(format!("ns_{}", name));
        fs::create_dir_all(&config_dir)?;
        let config_path = config_dir.join(CONFIG_FILENAME);

        let json =
            serde_json::to_string_pretty(&config).map_err(|e| KVError::Serialization(format!("Failed to serialize namespace config: {}", e)))?;

        crate::support::write_atomic_durable(&config_path, json.as_bytes())?;

        Ok(())
    }

    /// Persist the registry name↔ID table to the `namespace_registry` binary file.
    fn persist(&self) -> Result<()> {
        let data = RegistryData {
            next_id: self.next_id,
            names: self.namespaces.keys().cloned().collect(),
            ids: self.namespaces.values().copied().collect(),
            ttl_configs: self
                .ttl_configs
                .iter()
                .map(|(ns_id, (ttl, max_deletes))| TtlConfigData {
                    ns_id: *ns_id,
                    ttl_millis: ttl.as_millis() as u64,
                    max_deletes: *max_deletes as u64,
                })
                .collect(),
        };

        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&data)
            .map_err(|e| KVError::Serialization(format!("Failed to serialize namespace registry: {}", e)))?;

        let mut hasher = Hasher::new();
        hasher.update(&payload);
        let checksum = hasher.finalize();

        let mut out = Vec::with_capacity(16 + payload.len());
        out.extend_from_slice(&REGISTRY_MAGIC);
        out.extend_from_slice(&REGISTRY_VERSION.to_le_bytes());
        out.extend_from_slice(&checksum.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&payload);

        crate::support::write_atomic_durable(&self.path, &out)
            .map_err(|e| KVError::Io(std::io::Error::new(e.kind(), format!("Failed to persist namespace registry: {}", e))))?;

        Ok(())
    }

    /// Deserialize registry data from file bytes (magic + version + crc + payload)
    fn deserialize_file(bytes: &[u8]) -> Result<RegistryData> {
        if bytes.len() < 16 {
            return Err(KVError::Serialization("Namespace registry file too small".to_string()));
        }

        if bytes[0..4] != REGISTRY_MAGIC {
            return Err(KVError::Serialization("Invalid namespace registry magic".to_string()));
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != REGISTRY_VERSION {
            return Err(KVError::Serialization(format!("Unsupported namespace registry version: {}", version)));
        }

        let expected_checksum = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let payload_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;

        if bytes.len() < 16 + payload_len {
            return Err(KVError::Serialization("Namespace registry file truncated".to_string()));
        }

        let payload = &bytes[16..16 + payload_len];

        let mut hasher = Hasher::new();
        hasher.update(payload);
        let actual_checksum = hasher.finalize();
        if actual_checksum != expected_checksum {
            return Err(KVError::Serialization(format!(
                "Namespace registry checksum mismatch: expected {}, got {}",
                expected_checksum, actual_checksum
            )));
        }

        let archived = rkyv::access::<rkyv::Archived<RegistryData>, rkyv::rancor::Error>(payload)
            .map_err(|e| KVError::Serialization(format!("Namespace registry validation failed: {}", e)))?;
        let mut names = Vec::new();
        for name in archived.names.iter() {
            names.push(name.as_str().to_string());
        }
        let mut ids = Vec::new();
        for id in archived.ids.iter() {
            ids.push((*id).into());
        }
        let next_id: u32 = archived.next_id.into();

        let mut ttl_configs = Vec::new();
        for c in archived.ttl_configs.iter() {
            ttl_configs.push(TtlConfigData {
                ns_id: c.ns_id.into(),
                ttl_millis: c.ttl_millis.into(),
                max_deletes: c.max_deletes.into(),
            });
        }

        Ok(RegistryData {
            next_id,
            names,
            ids,
            ttl_configs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_ttl_config_persists_across_reopen() {
        let dir = TempDir::new().unwrap();
        let ns_id = {
            let mut registry = NamespaceRegistry::open(dir.path()).unwrap();
            let ns_id = registry.create("cache").unwrap();
            registry.set_ttl_config(ns_id, Duration::from_secs(3600), 500).unwrap();
            ns_id
        };

        // Reopen from disk: the TTL config must be restored.
        let registry = NamespaceRegistry::open(dir.path()).unwrap();
        assert_eq!(registry.ttl_config(ns_id), Some((Duration::from_secs(3600), 500)));
    }

    #[test]
    fn test_remove_ttl_config_persists() {
        let dir = TempDir::new().unwrap();
        let mut registry = NamespaceRegistry::open(dir.path()).unwrap();
        let ns_id = registry.create("cache").unwrap();
        registry.set_ttl_config(ns_id, Duration::from_secs(60), 10).unwrap();
        registry.remove_ttl_config(ns_id).unwrap();
        assert_eq!(registry.ttl_config(ns_id), None);

        let reopened = NamespaceRegistry::open(dir.path()).unwrap();
        assert_eq!(reopened.ttl_config(ns_id), None);
    }

    #[test]
    fn test_remove_namespace_drops_ttl_config() {
        let dir = TempDir::new().unwrap();
        let mut registry = NamespaceRegistry::open(dir.path()).unwrap();
        let ns_id = registry.create("cache").unwrap();
        registry.set_ttl_config(ns_id, Duration::from_secs(60), 10).unwrap();
        registry.remove("cache").unwrap();
        assert_eq!(registry.ttl_config(ns_id), None);
    }

    #[test]
    fn test_new_registry_has_default() {
        let dir = TempDir::new().unwrap();
        let registry = NamespaceRegistry::open(dir.path()).unwrap();
        assert_eq!(registry.get_id(DEFAULT_NAMESPACE), Some(DEFAULT_NAMESPACE_ID));
        assert_eq!(registry.get_name(DEFAULT_NAMESPACE_ID), Some(DEFAULT_NAMESPACE));
        assert!(registry.exists(DEFAULT_NAMESPACE));
    }

    #[test]
    fn test_create_namespace() {
        let dir = TempDir::new().unwrap();
        let mut registry = NamespaceRegistry::open(dir.path()).unwrap();
        // IDs 0 (default) and 1 (system) are reserved; first user namespace gets 2.
        let id = registry.create("users").unwrap();
        assert_eq!(id, 2);
        assert_eq!(registry.get_id("users"), Some(2));
        assert_eq!(registry.get_name(2), Some("users"));
    }

    #[test]
    fn test_duplicate_create_fails() {
        let dir = TempDir::new().unwrap();
        let mut registry = NamespaceRegistry::open(dir.path()).unwrap();
        registry.create("users").unwrap();
        assert!(registry.create("users").is_err());
    }

    #[test]
    fn test_ids_never_reused() {
        let dir = TempDir::new().unwrap();
        let mut registry = NamespaceRegistry::open(dir.path()).unwrap();
        let id1 = registry.create("ns1").unwrap();
        let _id2 = registry.create("ns2").unwrap();
        registry.remove("ns1").unwrap();
        let id3 = registry.create("ns3").unwrap();
        assert!(id3 > id1);
    }

    #[test]
    fn test_cannot_remove_default() {
        let dir = TempDir::new().unwrap();
        let mut registry = NamespaceRegistry::open(dir.path()).unwrap();
        assert!(registry.remove(DEFAULT_NAMESPACE).is_err());
    }

    #[test]
    fn test_cannot_remove_system() {
        let dir = TempDir::new().unwrap();
        let mut registry = NamespaceRegistry::open(dir.path()).unwrap();
        assert!(registry.remove(SYSTEM_NAMESPACE).is_err());
    }

    #[test]
    fn test_persistence_roundtrip() {
        let dir = TempDir::new().unwrap();
        {
            let mut registry = NamespaceRegistry::open(dir.path()).unwrap();
            registry.create("users").unwrap();
            registry.create("orders").unwrap();
        }
        let registry = NamespaceRegistry::open(dir.path()).unwrap();
        assert!(registry.exists(DEFAULT_NAMESPACE));
        assert!(registry.exists(SYSTEM_NAMESPACE));
        assert!(registry.exists("users"));
        assert!(registry.exists("orders"));
        // 0 = default, 1 = system, 2 = users, 3 = orders
        assert_eq!(registry.get_id("users"), Some(2));
        assert_eq!(registry.get_id("orders"), Some(3));
    }

    #[test]
    fn test_list_namespaces() {
        let dir = TempDir::new().unwrap();
        let mut registry = NamespaceRegistry::open(dir.path()).unwrap();
        registry.create("alpha").unwrap();
        registry.create("beta").unwrap();
        let mut list = registry.list();
        list.sort_by_key(|(_, id)| *id);
        // default(0), system(1), alpha(2), beta(3)
        assert_eq!(list.len(), 4);
        assert_eq!(list[0], ("default", 0));
        assert_eq!(list[1], ("system", 1));
        assert_eq!(list[2], ("alpha", 2));
        assert_eq!(list[3], ("beta", 3));
    }

    #[test]
    fn test_schema_persists_across_restart() {
        let dir = TempDir::new().unwrap();
        // Create the ns_default directory so config.json has somewhere to live
        std::fs::create_dir_all(dir.path().join("ns_default")).unwrap();
        {
            let mut registry = NamespaceRegistry::open(dir.path()).unwrap();
            registry
                .register_schema_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str)
                .unwrap();
            registry.register_schema_field(DEFAULT_NAMESPACE_ID, "age", IndexValueType::Int).unwrap();
        }
        // Reopen — schema must be restored from config.json without re-registering
        let registry = NamespaceRegistry::open(dir.path()).unwrap();
        let schema = registry.schema(DEFAULT_NAMESPACE_ID).unwrap();
        let status = schema.get_field_by_name("status").unwrap();
        assert_eq!(status.field_type, IndexValueType::Str);
        assert_eq!(status.field_id, 0);
        let age = schema.get_field_by_name("age").unwrap();
        assert_eq!(age.field_type, IndexValueType::Int);
        assert_eq!(age.field_id, 1);
    }

    #[test]
    fn test_register_field_idempotent() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("ns_default")).unwrap();
        let mut registry = NamespaceRegistry::open(dir.path()).unwrap();
        let id1 = registry
            .register_schema_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str)
            .unwrap();
        // Same name + type: returns existing id
        let id2 = registry
            .register_schema_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str)
            .unwrap();
        assert_eq!(id1, id2);
        // Same name, different type: error
        assert!(
            registry
                .register_schema_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Int)
                .is_err()
        );
    }

    #[test]
    fn test_kv_only_namespace_has_no_config_file() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("ns_default")).unwrap();
        let _registry = NamespaceRegistry::open(dir.path()).unwrap();
        // No fields registered — config.json must not exist
        assert!(!dir.path().join("ns_default").join("config.json").exists());
    }
}
