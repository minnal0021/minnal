use std::path::Path;

use json_dotpath::DotPaths;
use serde::{Deserialize, Serialize};

use crate::error::SchemaError;

/// Maximum number of indices allowed per document store.
pub const MAX_INDICES: usize = 5;

/// The kind of store a schema describes.
///
/// This is the **explicit, mandatory** discriminant between the two schema
/// families — a document store ([`DocStoreSchema`]) versus a raw key-value store
/// ([`KvStoreSchema`]). It replaces the historical practice of inferring the kind
/// from the (incidentally disjoint) `key_type` values, which was fragile and
/// accidental. Every persisted schema and every create/import payload must carry
/// a `store_type` that parses into this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StoreType {
    /// A JSON document store (`DocStoreSchema`): indices, attributes, doc CRUD.
    Doc,
    /// A raw key-value store (`KvStoreSchema`): opaque values, no indices.
    Kv,
}

impl StoreType {
    /// Human-readable label for error messages.
    pub fn label(self) -> &'static str {
        match self {
            StoreType::Doc => "doc",
            StoreType::Kv => "kv",
        }
    }
}

/// Read just the `store_type` discriminant from a raw schema JSON document,
/// without committing to either full schema struct.
///
/// Returns `None` if the JSON is unparseable or carries no parseable
/// `store_type`. This is the authoritative way to tell a doc-store schema from a
/// KV-store schema on disk — used by the loaders to dispatch to the right struct.
pub(crate) fn peek_store_type(json: &str) -> Option<StoreType> {
    #[derive(Deserialize)]
    struct Discriminant {
        store_type: StoreType,
    }
    serde_json::from_str::<Discriminant>(json).ok().map(|d| d.store_type)
}

/// The key type for document identifiers in the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KeyType {
    /// 128-bit UUID (stored as 16 bytes, big-endian for key ordering).
    Uuid,
    /// Unsigned 64-bit integer.
    U64,
    /// Unsigned 128-bit integer.
    U128,
}

/// Key type for a KV store namespace.
///
/// `Bool` is intentionally absent — a boolean is not a useful lookup key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvKeyType {
    #[serde(rename = "str")]
    Str,
    #[serde(rename = "int")]
    Int,
}

/// Value type for a KV store namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvValueType {
    #[serde(rename = "int")]
    Int,
    #[serde(rename = "str")]
    Str,
    #[serde(rename = "f32")]
    F32,
    #[serde(rename = "vec_f32")]
    VecF32,
}

/// Schema definition for a KV store namespace.
///
/// One `KvStoreSchema` maps to exactly one namespace in the underlying
/// `minnal_db`.  Values are stored as raw bytes according to `value_type`.
/// Indices and attribute declarations are not supported; use `DocStoreSchema`
/// if you need them.
///
/// On disk the JSON is distinguished from `DocStoreSchema` by the mandatory
/// [`store_type`](Self::store_type) field, which must be `"kv"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvStoreSchema {
    pub namespace: String,
    /// Mandatory store-kind discriminant; must be [`StoreType::Kv`].
    pub store_type: StoreType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ns_id: Option<u32>,
    pub key_type: KvKeyType,
    pub value_type: KvValueType,
    /// Enable ANN semantic search on this namespace.  Only valid when
    /// `value_type = str`; the stored string is the text that gets embedded.
    #[serde(default)]
    pub semantic_search_enabled: bool,
}

impl KvStoreSchema {
    /// Validate the schema without saving.
    pub fn validate(&self) -> Result<(), SchemaError> {
        if self.store_type != StoreType::Kv {
            return Err(SchemaError::WrongStoreType {
                namespace: self.namespace.clone(),
                expected: "kv",
                found: "doc",
            });
        }
        if self.namespace.is_empty() || !self.namespace.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            return Err(SchemaError::InvalidNamespace);
        }
        if self.semantic_search_enabled && self.value_type != KvValueType::Str {
            return Err(SchemaError::KvSemanticSearchOnlyForStr);
        }
        Ok(())
    }

    /// Returns `true` when semantic search is enabled and the value type is `Str`.
    pub fn is_semantic_search_enabled(&self) -> bool {
        self.semantic_search_enabled && self.value_type == KvValueType::Str
    }

    /// Persist the schema to `schema_dir/<namespace>.json` (atomic write).
    pub fn save(&self, schema_dir: &Path) -> Result<(), SchemaError> {
        let path = schema_dir.join(format!("{}.json", self.namespace));
        let tmp = path.with_extension("tmp");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&tmp, &json)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    /// Load the KV schema for `namespace` from `schema_dir/<namespace>.json`.
    ///
    /// Returns [`SchemaError::WrongStoreType`] if the file on disk is a document
    /// store rather than a KV store.
    pub fn load(schema_dir: &Path, namespace: &str) -> Result<Self, SchemaError> {
        let path = schema_dir.join(format!("{namespace}.json"));
        if !path.exists() {
            return Err(SchemaError::NotFound {
                namespace: namespace.to_owned(),
            });
        }
        let json = std::fs::read_to_string(path)?;
        if let Some(found) = peek_store_type(&json)
            && found != StoreType::Kv
        {
            return Err(SchemaError::WrongStoreType {
                namespace: namespace.to_owned(),
                expected: "kv",
                found: found.label(),
            });
        }
        serde_json::from_str(&json).map_err(SchemaError::Serialize)
    }
}

impl KvKeyType {
    /// Serialize a JSON value as raw key bytes for storage in minnal_db.
    ///
    /// Int keys use big-endian encoding so that lexicographic byte order
    /// matches numeric order, enabling range scans.
    pub fn serialize_key(&self, key: &serde_json::Value) -> Result<Vec<u8>, SchemaError> {
        match self {
            KvKeyType::Str => {
                let s = key.as_str().ok_or(SchemaError::KvKeyTypeMismatch { expected: "string" })?;
                Ok(s.as_bytes().to_vec())
            }
            KvKeyType::Int => {
                let n = key.as_i64().ok_or(SchemaError::KvKeyTypeMismatch { expected: "integer" })?;
                Ok(n.to_be_bytes().to_vec())
            }
        }
    }

    /// Parse a raw URL path segment and serialize it as key bytes.
    pub fn serialize_key_from_str(&self, raw: &str) -> Result<Vec<u8>, SchemaError> {
        match self {
            KvKeyType::Str => Ok(raw.as_bytes().to_vec()),
            KvKeyType::Int => {
                let n: i64 = raw.parse().map_err(|_| SchemaError::KvKeyTypeMismatch { expected: "integer" })?;
                Ok(n.to_be_bytes().to_vec())
            }
        }
    }

    /// Deserialize raw key bytes back to a JSON value.
    pub fn deserialize_key(&self, bytes: &[u8]) -> Result<serde_json::Value, SchemaError> {
        match self {
            KvKeyType::Str => {
                let s = std::str::from_utf8(bytes).map_err(|_| SchemaError::KvValueCorrupt)?;
                Ok(serde_json::Value::String(s.to_owned()))
            }
            KvKeyType::Int => {
                let arr: [u8; 8] = bytes.try_into().map_err(|_| SchemaError::KvValueCorrupt)?;
                Ok(serde_json::Value::from(i64::from_be_bytes(arr)))
            }
        }
    }
}

impl KvValueType {
    /// Serialize a JSON value to raw bytes for storage.
    pub fn serialize_value(&self, value: &serde_json::Value) -> Result<Vec<u8>, SchemaError> {
        match self {
            KvValueType::Int => {
                let n = value.as_i64().ok_or(SchemaError::KvValueTypeMismatch { expected: "integer" })?;
                Ok(n.to_le_bytes().to_vec())
            }
            KvValueType::Str => {
                let s = value.as_str().ok_or(SchemaError::KvValueTypeMismatch { expected: "string" })?;
                Ok(s.as_bytes().to_vec())
            }
            KvValueType::F32 => {
                let n = value.as_f64().ok_or(SchemaError::KvValueTypeMismatch { expected: "number (f32)" })?;
                Ok((n as f32).to_le_bytes().to_vec())
            }
            KvValueType::VecF32 => {
                let arr = value.as_array().ok_or(SchemaError::KvValueTypeMismatch {
                    expected: "array of numbers",
                })?;
                let mut bytes = Vec::with_capacity(arr.len() * 4);
                for v in arr {
                    let n = v.as_f64().ok_or(SchemaError::KvValueTypeMismatch {
                        expected: "array element must be a number",
                    })?;
                    bytes.extend_from_slice(&(n as f32).to_le_bytes());
                }
                Ok(bytes)
            }
        }
    }

    /// Deserialize raw bytes back to a JSON value.
    pub fn deserialize_value(&self, bytes: &[u8]) -> Result<serde_json::Value, SchemaError> {
        match self {
            KvValueType::Int => {
                let arr: [u8; 8] = bytes.try_into().map_err(|_| SchemaError::KvValueCorrupt)?;
                Ok(serde_json::Value::from(i64::from_le_bytes(arr)))
            }
            KvValueType::Str => {
                let s = std::str::from_utf8(bytes).map_err(|_| SchemaError::KvValueCorrupt)?;
                Ok(serde_json::Value::String(s.to_owned()))
            }
            KvValueType::F32 => {
                let arr: [u8; 4] = bytes.try_into().map_err(|_| SchemaError::KvValueCorrupt)?;
                let f = f32::from_le_bytes(arr);
                Ok(serde_json::json!(f))
            }
            KvValueType::VecF32 => {
                if !bytes.len().is_multiple_of(4) {
                    return Err(SchemaError::KvValueCorrupt);
                }
                let floats: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                serde_json::to_value(&floats).map_err(SchemaError::Serialize)
            }
        }
    }
}

/// The value type for an indexed field, matching the underlying index engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IndexType {
    /// Boolean field (`true` / `false`).
    Bool,
    /// Signed 64-bit integer field.
    Int,
    /// UTF-8 string field.
    Str,
}

/// The technology backing an index: a RoaringBitmap field index or a vector ANN index.
///
/// Serialises as `"attribute"` / `"vector"` in JSON so REST API responses are
/// self-describing without requiring callers to parse `IndexId` internals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IndexKind {
    /// RoaringBitmap field index — updated synchronously on every write.
    Attribute,
    /// Vector (IVF + RaBitQ) ANN index — updated asynchronously via the embedding queue.
    Vector,
}

/// The value type for a non-indexed document attribute.
///
/// Mirrors [`IndexType`] so that attribute definitions are self-contained.
/// Non-indexed attributes are tracked in the schema for documentation and
/// to prevent accidentally indexing a field with the wrong type later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AttributeType {
    Bool,
    Int,
    Str,
}

/// A non-indexed field declared in the schema.
///
/// Non-indexed attributes live in the stored JSON document but do not have a
/// live field index in the underlying `minnal_db` namespace.  They can be
/// added, removed, or type-updated freely via [`SchemaAmendment`] as long as
/// no [`IndexSpec`] references the same field name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttributeDef {
    /// The JSON field name (e.g. `"email"`, `"created_at"`).
    pub name: String,
    /// Declared value type — used for documentation and future validation.
    pub attr_type: AttributeType,
    /// Optional human-readable description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Specification for a single index on a JSON document attribute.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexSpec {
    /// The top-level JSON attribute name to index (e.g. `"status"`, `"age"`).
    pub field: String,
    /// The value type used by the index engine for this field.
    pub index_type: IndexType,
}

/// A single schema change applied by `DocStore::amend`.
///
/// Only **non-indexed** attributes may be amended.  Attempting to remove or
/// update an attribute that is currently indexed returns
/// `DocStoreError::AttributeIsIndexed`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaAmendment {
    /// Add a new non-indexed attribute.  Fails if the name already exists in
    /// `attributes` or `indices`.
    AddAttribute {
        name: String,
        attr_type: AttributeType,
        description: Option<String>,
    },
    /// Remove a non-indexed attribute.  Fails if the name has an active index.
    RemoveAttribute { name: String },
    /// Change the declared type (and optional description) of a non-indexed
    /// attribute.  Fails if the name has an active index.
    UpdateAttribute {
        name: String,
        attr_type: AttributeType,
        description: Option<String>,
    },
    /// Add a new `str` attribute and register it as an embedding field for
    /// semantic search.  Sets `semantic_search_enabled = true`.  Fails if the
    /// name already exists in `attributes` or `indices`, or if a vector index is
    /// already present (drop it first — a namespace has at most one).
    AddEmbeddingAttribute { name: String, description: Option<String> },
    /// Enable the namespace's (single) vector index over **one or more**
    /// embedding fields in a single call.  Each name is declared as a `str`
    /// attribute and registered as an embedding field, and
    /// `semantic_search_enabled` is set.  Fails if a vector index is already
    /// present (drop it first), if `fields` is empty, or if any name conflicts
    /// with an existing index/attribute or is duplicated within the list.
    ///
    /// This is the post-create way to (re)create a multi-field vector index:
    /// `DELETE /stores/{ns}/indices/vector` then enable with the full field set.
    EnableVectorIndex { fields: Vec<String> },
}

/// Schema definition for a single document store instance.
///
/// One `DocStoreSchema` maps to exactly one namespace in the underlying
/// `minnal_db`.  The `ns_id` is `None` until the store is created via
/// `DocStore::create`, after which it is persisted in the schema JSON so
/// that drop operations can locate and clean up index directories.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocStoreSchema {
    /// The `minnal_db` namespace this store maps to.
    pub namespace: String,
    /// Mandatory store-kind discriminant; must be [`StoreType::Doc`].
    pub store_type: StoreType,
    /// The minnal_db internal namespace ID — set after creation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ns_id: Option<u32>,
    /// The type used as the document key.
    pub key_type: KeyType,
    /// Non-indexed attribute declarations (documentation / future validation).
    #[serde(default)]
    pub attributes: Vec<AttributeDef>,
    /// Field indices (0 to [`MAX_INDICES`] inclusive).
    pub indices: Vec<IndexSpec>,
    /// Whether semantic (vector) search is enabled for this store.
    ///
    /// When `true`, `embedding_fields` must be non-empty.  Both conditions
    /// together are required before a vector index KV store is set up by
    /// `DocStore::create`.
    #[serde(default)]
    pub semantic_search_enabled: bool,
    /// The top-level JSON fields whose string values are concatenated and
    /// embedded for semantic search.
    ///
    /// Each field must be declared as a `Str` attribute in `attributes`.
    /// Must be non-empty when `semantic_search_enabled` is `true`.
    /// Ignored if `semantic_search_enabled` is `false`.
    ///
    /// When building the embedding, fields are joined as:
    /// `"field_name: field_value\nfield_name: field_value\n…"`.
    /// Fields absent from the document are omitted from the concatenation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub embedding_fields: Vec<String>,
}

impl DocStoreSchema {
    /// Validate the schema and save it to `schema_dir/<namespace>.json`.
    ///
    /// The directory `schema_dir` must already exist.
    pub fn validate_and_save(&self, schema_dir: &Path) -> Result<(), SchemaError> {
        self.validate()?;
        self.save(schema_dir)
    }

    /// Validate the schema without saving.
    ///
    /// Rules:
    /// - `namespace` must be non-empty; only ASCII alphanumerics, `_`, or `-`.
    /// - At most [`MAX_INDICES`] indices.
    /// - Every index field name must be non-empty and unique.
    /// - Non-indexed attribute names must be non-empty, unique among themselves,
    ///   and must not overlap with index field names.
    pub fn validate(&self) -> Result<(), SchemaError> {
        if self.store_type != StoreType::Doc {
            return Err(SchemaError::WrongStoreType {
                namespace: self.namespace.clone(),
                expected: "doc",
                found: "kv",
            });
        }
        if self.namespace.is_empty() || !self.namespace.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            return Err(SchemaError::InvalidNamespace);
        }

        if self.indices.len() > MAX_INDICES {
            return Err(SchemaError::TooManyIndices {
                count: self.indices.len(),
                max: MAX_INDICES,
            });
        }

        let mut seen = std::collections::HashSet::new();
        for (i, spec) in self.indices.iter().enumerate() {
            if spec.field.is_empty() {
                return Err(SchemaError::EmptyFieldName { index: i });
            }
            if !seen.insert(spec.field.clone()) {
                return Err(SchemaError::DuplicateFieldName { field: spec.field.clone() });
            }
        }

        for attr in &self.attributes {
            if attr.name.is_empty() {
                return Err(SchemaError::EmptyAttributeName);
            }
            if seen.contains(&attr.name) {
                return Err(SchemaError::DuplicateFieldName { field: attr.name.clone() });
            }
            seen.insert(attr.name.clone());
        }

        if self.semantic_search_enabled {
            if self.embedding_fields.is_empty() {
                return Err(SchemaError::SemanticSearchMissingField);
            }

            let indexed_fields = self.indexed_field_names();
            // Build a set of attributes declared as Str for quick lookup.
            let str_attributes: std::collections::HashSet<&str> = self
                .attributes
                .iter()
                .filter(|a| a.attr_type == AttributeType::Str)
                .map(|a| a.name.as_str())
                .collect();

            let mut seen_embedding = std::collections::HashSet::new();
            for f in &self.embedding_fields {
                if f.is_empty() {
                    return Err(SchemaError::SemanticSearchMissingField);
                }
                if !seen_embedding.insert(f.as_str()) {
                    return Err(SchemaError::DuplicateFieldName { field: f.clone() });
                }
                if indexed_fields.contains(f.as_str()) {
                    return Err(SchemaError::EmbeddingFieldConflict { field: f.clone() });
                }
                if !str_attributes.contains(f.as_str()) {
                    return Err(SchemaError::EmbeddingFieldNotString { field: f.clone() });
                }
            }
        }

        Ok(())
    }

    /// Returns `true` when `semantic_search_enabled` is set and at least one
    /// `embedding_fields` entry is configured.
    ///
    /// Use this guard before setting up the vector index KV store.
    pub fn is_semantic_search_enabled(&self) -> bool {
        self.semantic_search_enabled && !self.embedding_fields.is_empty()
    }

    /// Save the schema to `schema_dir/<namespace>.json` (atomic write).
    pub fn save(&self, schema_dir: &Path) -> Result<(), SchemaError> {
        let path = schema_dir.join(format!("{}.json", self.namespace));
        let tmp = path.with_extension("tmp");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&tmp, &json)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    /// Load a schema for `namespace` from `schema_dir/<namespace>.json`.
    ///
    /// Returns [`SchemaError::WrongStoreType`] if the file on disk is a KV store
    /// rather than a document store.
    pub fn load(schema_dir: &Path, namespace: &str) -> Result<Self, SchemaError> {
        let path = schema_dir.join(format!("{namespace}.json"));
        if !path.exists() {
            return Err(SchemaError::NotFound {
                namespace: namespace.to_owned(),
            });
        }
        let json = std::fs::read_to_string(path)?;
        if let Some(found) = peek_store_type(&json)
            && found != StoreType::Doc
        {
            return Err(SchemaError::WrongStoreType {
                namespace: namespace.to_owned(),
                expected: "doc",
                found: found.label(),
            });
        }
        let schema: Self = serde_json::from_str(&json)?;
        Ok(schema)
    }

    /// Return the names of all indexed fields.
    pub fn indexed_field_names(&self) -> std::collections::HashSet<&str> {
        self.indices.iter().map(|s| s.field.as_str()).collect()
    }

    /// Apply a [`SchemaAmendment`] in-place, enforcing the constraint that
    /// indexed fields cannot be removed or type-changed.
    ///
    /// Returns an error (via [`SchemaError`]) rather than panicking on
    /// constraint violations so that callers can propagate the error cleanly.
    pub fn apply_amendment(&mut self, amendment: SchemaAmendment) -> Result<(), SchemaError> {
        let indexed: std::collections::HashSet<&str> = self.indexed_field_names();
        match amendment {
            SchemaAmendment::AddAttribute {
                name,
                attr_type,
                description,
            } => {
                if indexed.contains(name.as_str()) {
                    return Err(SchemaError::AttributeIsIndexed { name });
                }
                if self.attributes.iter().any(|a| a.name == name) {
                    return Err(SchemaError::DuplicateFieldName { field: name });
                }
                self.attributes.push(AttributeDef {
                    name,
                    attr_type,
                    description,
                });
                Ok(())
            }
            SchemaAmendment::RemoveAttribute { name } => {
                if indexed.contains(name.as_str()) {
                    return Err(SchemaError::AttributeIsIndexed { name });
                }
                let before = self.attributes.len();
                self.attributes.retain(|a| a.name != name);
                if self.attributes.len() == before {
                    return Err(SchemaError::AttributeNotFound { name });
                }
                Ok(())
            }
            SchemaAmendment::UpdateAttribute {
                name,
                attr_type,
                description,
            } => {
                if indexed.contains(name.as_str()) {
                    return Err(SchemaError::AttributeIsIndexed { name });
                }
                match self.attributes.iter_mut().find(|a| a.name == name) {
                    Some(attr) => {
                        attr.attr_type = attr_type;
                        attr.description = description;
                        Ok(())
                    }
                    None => Err(SchemaError::AttributeNotFound { name }),
                }
            }
            SchemaAmendment::AddEmbeddingAttribute { name, description } => {
                // A namespace has at most one vector index. Adding an embedding
                // attribute is how the vector index is created; once it exists,
                // reject further adds — the caller must drop the vector index
                // first (which also clears its backing data).
                if self.semantic_search_enabled {
                    return Err(SchemaError::SemanticSearchAlreadyEnabled {
                        namespace: self.namespace.clone(),
                    });
                }
                if name.is_empty() {
                    return Err(SchemaError::EmptyAttributeName);
                }
                if indexed.contains(name.as_str()) {
                    return Err(SchemaError::EmbeddingFieldConflict { field: name });
                }
                if self.attributes.iter().any(|a| a.name == name) {
                    return Err(SchemaError::DuplicateFieldName { field: name });
                }
                if self.embedding_fields.contains(&name) {
                    return Err(SchemaError::DuplicateFieldName { field: name });
                }
                self.attributes.push(AttributeDef {
                    name: name.clone(),
                    attr_type: AttributeType::Str,
                    description,
                });
                self.embedding_fields.push(name);
                self.semantic_search_enabled = true;
                Ok(())
            }
            SchemaAmendment::EnableVectorIndex { fields } => {
                // One vector index per namespace — reject if already present.
                if self.semantic_search_enabled {
                    return Err(SchemaError::SemanticSearchAlreadyEnabled {
                        namespace: self.namespace.clone(),
                    });
                }
                if fields.is_empty() {
                    return Err(SchemaError::SemanticSearchMissingField);
                }
                // Validate every field up front so a bad entry leaves the schema
                // untouched (no partial application).
                let mut seen = std::collections::HashSet::new();
                for name in &fields {
                    if name.is_empty() {
                        return Err(SchemaError::SemanticSearchMissingField);
                    }
                    if !seen.insert(name.as_str()) {
                        return Err(SchemaError::DuplicateFieldName { field: name.clone() });
                    }
                    if indexed.contains(name.as_str()) {
                        return Err(SchemaError::EmbeddingFieldConflict { field: name.clone() });
                    }
                    if self.attributes.iter().any(|a| &a.name == name) {
                        return Err(SchemaError::DuplicateFieldName { field: name.clone() });
                    }
                }
                for name in fields {
                    self.attributes.push(AttributeDef {
                        name: name.clone(),
                        attr_type: AttributeType::Str,
                        description: None,
                    });
                    self.embedding_fields.push(name);
                }
                self.semantic_search_enabled = true;
                Ok(())
            }
        }
    }
}

// ── Doc validation helpers ─────────────────────────────────────────────────

fn json_type_name(val: &serde_json::Value) -> &'static str {
    match val {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

impl IndexType {
    fn expected_json_type(self) -> &'static str {
        match self {
            IndexType::Bool => "bool",
            IndexType::Int => "integer",
            IndexType::Str => "string",
        }
    }

    fn matches_value(self, val: &serde_json::Value) -> bool {
        match self {
            IndexType::Bool => val.is_boolean(),
            IndexType::Int => val.as_i64().is_some(),
            IndexType::Str => val.is_string(),
        }
    }
}

impl AttributeType {
    fn expected_json_type(self) -> &'static str {
        match self {
            AttributeType::Bool => "bool",
            AttributeType::Int => "integer",
            AttributeType::Str => "string",
        }
    }

    fn matches_value(self, val: &serde_json::Value) -> bool {
        match self {
            AttributeType::Bool => val.is_boolean(),
            AttributeType::Int => val.as_i64().is_some(),
            AttributeType::Str => val.is_string(),
        }
    }
}

impl DocStoreSchema {
    /// Validate that a document conforms to this schema.
    ///
    /// Rules:
    /// - `doc` must be a JSON object.
    /// - For every [`IndexSpec`]: if the field is present its JSON type must
    ///   match the declared [`IndexType`] (`bool`, integer number, or string).
    /// - For every [`AttributeDef`]: if the field is present its JSON type must
    ///   match the declared [`AttributeType`].
    ///
    /// Fields that are absent from the document are not an error — they are
    /// simply omitted from the index / treated as missing attributes.
    pub fn validate_doc(&self, doc: &serde_json::Value) -> Result<(), SchemaError> {
        if !doc.is_object() {
            return Err(SchemaError::DocNotObject);
        }

        for spec in &self.indices {
            if let Some(val) = doc.dot_get::<serde_json::Value>(&spec.field).unwrap_or(None)
                && !spec.index_type.matches_value(&val)
            {
                return Err(SchemaError::FieldTypeMismatch {
                    field: spec.field.clone(),
                    expected: spec.index_type.expected_json_type(),
                    actual: json_type_name(&val),
                });
            }
        }

        for attr in &self.attributes {
            if let Some(val) = doc.dot_get::<serde_json::Value>(&attr.name).unwrap_or(None)
                && !attr.attr_type.matches_value(&val)
            {
                return Err(SchemaError::FieldTypeMismatch {
                    field: attr.name.clone(),
                    expected: attr.attr_type.expected_json_type(),
                    actual: json_type_name(&val),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn valid_schema() -> DocStoreSchema {
        DocStoreSchema {
            store_type: StoreType::Doc,
            namespace: "products".to_owned(),
            ns_id: None,
            key_type: KeyType::Uuid,
            attributes: vec![],
            indices: vec![
                IndexSpec {
                    field: "status".to_owned(),
                    index_type: IndexType::Str,
                },
                IndexSpec {
                    field: "price".to_owned(),
                    index_type: IndexType::Int,
                },
            ],
            semantic_search_enabled: false,
            embedding_fields: vec![],
        }
    }

    #[test]
    fn valid_schema_passes_validation() {
        assert!(valid_schema().validate().is_ok());
    }

    #[test]
    fn empty_namespace_is_rejected() {
        let mut s = valid_schema();
        s.namespace = String::new();
        assert!(matches!(s.validate(), Err(SchemaError::InvalidNamespace)));
    }

    #[test]
    fn namespace_with_spaces_is_rejected() {
        let mut s = valid_schema();
        s.namespace = "my namespace".to_owned();
        assert!(matches!(s.validate(), Err(SchemaError::InvalidNamespace)));
    }

    #[test]
    fn namespace_with_dots_is_rejected() {
        let mut s = valid_schema();
        s.namespace = "my.namespace".to_owned();
        assert!(matches!(s.validate(), Err(SchemaError::InvalidNamespace)));
    }

    #[test]
    fn namespace_with_hyphens_and_underscores_is_valid() {
        let mut s = valid_schema();
        s.namespace = "my-name_space_01".to_owned();
        assert!(s.validate().is_ok());
    }

    #[test]
    fn too_many_indices_is_rejected() {
        let mut s = valid_schema();
        s.indices = (0..=MAX_INDICES)
            .map(|i| IndexSpec {
                field: format!("field_{i}"),
                index_type: IndexType::Int,
            })
            .collect();
        assert!(matches!(s.validate(), Err(SchemaError::TooManyIndices { count: 6, max: 5 })));
    }

    #[test]
    fn exactly_max_indices_is_valid() {
        let mut s = valid_schema();
        s.indices = (0..MAX_INDICES)
            .map(|i| IndexSpec {
                field: format!("field_{i}"),
                index_type: IndexType::Bool,
            })
            .collect();
        assert!(s.validate().is_ok());
    }

    #[test]
    fn empty_field_name_is_rejected() {
        let mut s = valid_schema();
        s.indices.push(IndexSpec {
            field: String::new(),
            index_type: IndexType::Str,
        });
        assert!(matches!(s.validate(), Err(SchemaError::EmptyFieldName { .. })));
    }

    #[test]
    fn duplicate_field_names_are_rejected() {
        let mut s = valid_schema();
        s.indices.push(IndexSpec {
            field: "status".to_owned(),
            index_type: IndexType::Bool,
        });
        assert!(matches!(s.validate(), Err(SchemaError::DuplicateFieldName { .. })));
    }

    #[test]
    fn all_key_types_are_valid() {
        for key_type in [KeyType::Uuid, KeyType::U64, KeyType::U128] {
            let s = DocStoreSchema {
                store_type: StoreType::Doc,
                namespace: "ns".to_owned(),
                ns_id: None,
                key_type,
                attributes: vec![],
                indices: vec![],
                semantic_search_enabled: false,
                embedding_fields: vec![],
            };
            assert!(s.validate().is_ok());
        }
    }

    #[test]
    fn all_index_types_roundtrip_json() {
        let s = DocStoreSchema {
            store_type: StoreType::Doc,
            namespace: "ns".to_owned(),
            ns_id: None,
            key_type: KeyType::U64,
            attributes: vec![],
            indices: vec![
                IndexSpec {
                    field: "active".to_owned(),
                    index_type: IndexType::Bool,
                },
                IndexSpec {
                    field: "count".to_owned(),
                    index_type: IndexType::Int,
                },
                IndexSpec {
                    field: "tag".to_owned(),
                    index_type: IndexType::Str,
                },
            ],
            semantic_search_enabled: false,
            embedding_fields: vec![],
        };
        let json = serde_json::to_string(&s).unwrap();
        let restored: DocStoreSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(s, restored);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let schema = valid_schema();
        schema.validate_and_save(dir.path()).unwrap();
        let loaded = DocStoreSchema::load(dir.path(), "products").unwrap();
        assert_eq!(schema, loaded);
    }

    #[test]
    fn load_missing_schema_returns_not_found() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(DocStoreSchema::load(dir.path(), "missing"), Err(SchemaError::NotFound { .. })));
    }

    #[test]
    fn save_creates_file_named_after_namespace() {
        let dir = tempfile::tempdir().unwrap();
        let schema = valid_schema();
        schema.save(dir.path()).unwrap();
        assert!(dir.path().join("products.json").exists());
    }

    #[test]
    fn index_field_names_are_unique_across_types() {
        let s = DocStoreSchema {
            store_type: StoreType::Doc,
            namespace: "ns".to_owned(),
            ns_id: None,
            key_type: KeyType::U128,
            attributes: vec![],
            indices: vec![
                IndexSpec {
                    field: "score".to_owned(),
                    index_type: IndexType::Int,
                },
                IndexSpec {
                    field: "score".to_owned(),
                    index_type: IndexType::Str,
                },
            ],
            semantic_search_enabled: false,
            embedding_fields: vec![],
        };
        assert!(matches!(
            s.validate(),
            Err(SchemaError::DuplicateFieldName { field }) if field == "score"
        ));
    }

    #[test]
    fn zero_indices_is_valid() {
        let s = DocStoreSchema {
            store_type: StoreType::Doc,
            namespace: "empty".to_owned(),
            ns_id: None,
            key_type: KeyType::Uuid,
            attributes: vec![],
            indices: vec![],
            semantic_search_enabled: false,
            embedding_fields: vec![],
        };
        assert!(s.validate().is_ok());
    }

    #[test]
    fn field_names_are_unique_in_valid_schema() {
        let s = valid_schema();
        let names: HashSet<_> = s.indices.iter().map(|i| &i.field).collect();
        assert_eq!(names.len(), s.indices.len());
    }

    #[test]
    fn attribute_name_conflicts_with_index_field_is_rejected() {
        let s = DocStoreSchema {
            store_type: StoreType::Doc,
            namespace: "ns".to_owned(),
            ns_id: None,
            key_type: KeyType::Uuid,
            attributes: vec![AttributeDef {
                name: "status".to_owned(), // same as index field
                attr_type: AttributeType::Str,
                description: None,
            }],
            indices: vec![IndexSpec {
                field: "status".to_owned(),
                index_type: IndexType::Str,
            }],
            semantic_search_enabled: false,
            embedding_fields: vec![],
        };
        assert!(matches!(s.validate(), Err(SchemaError::DuplicateFieldName { .. })));
    }

    #[test]
    fn amendment_add_attribute() {
        let mut s = valid_schema();
        s.apply_amendment(SchemaAmendment::AddAttribute {
            name: "email".to_owned(),
            attr_type: AttributeType::Str,
            description: Some("user email".to_owned()),
        })
        .unwrap();
        assert_eq!(s.attributes.len(), 1);
        assert_eq!(s.attributes[0].name, "email");
    }

    #[test]
    fn amendment_cannot_add_indexed_field_as_attribute() {
        let mut s = valid_schema();
        // "status" is already an index field
        let result = s.apply_amendment(SchemaAmendment::AddAttribute {
            name: "status".to_owned(),
            attr_type: AttributeType::Str,
            description: None,
        });
        assert!(matches!(result, Err(SchemaError::AttributeIsIndexed { .. })));
    }

    #[test]
    fn amendment_remove_attribute() {
        let mut s = valid_schema();
        s.attributes.push(AttributeDef {
            name: "email".to_owned(),
            attr_type: AttributeType::Str,
            description: None,
        });
        s.apply_amendment(SchemaAmendment::RemoveAttribute { name: "email".to_owned() }).unwrap();
        assert!(s.attributes.is_empty());
    }

    #[test]
    fn amendment_cannot_remove_indexed_attribute() {
        let mut s = valid_schema();
        // "status" is an index field, not a plain attribute
        let result = s.apply_amendment(SchemaAmendment::RemoveAttribute { name: "status".to_owned() });
        assert!(matches!(result, Err(SchemaError::AttributeIsIndexed { .. })));
    }

    #[test]
    fn amendment_update_attribute() {
        let mut s = valid_schema();
        s.attributes.push(AttributeDef {
            name: "score".to_owned(),
            attr_type: AttributeType::Int,
            description: None,
        });
        s.apply_amendment(SchemaAmendment::UpdateAttribute {
            name: "score".to_owned(),
            attr_type: AttributeType::Str,
            description: Some("updated".to_owned()),
        })
        .unwrap();
        let attr = s.attributes.iter().find(|a| a.name == "score").unwrap();
        assert_eq!(attr.attr_type, AttributeType::Str);
        assert_eq!(attr.description.as_deref(), Some("updated"));
    }

    // ── Semantic search configuration tests ───────────────────────────────

    #[test]
    fn semantic_search_disabled_by_default() {
        let s = valid_schema();
        assert!(!s.semantic_search_enabled);
        assert!(s.embedding_fields.is_empty());
        assert!(!s.is_semantic_search_enabled());
    }

    #[test]
    fn semantic_search_enabled_with_single_str_attribute_is_valid() {
        let mut s = valid_schema();
        s.attributes.push(AttributeDef {
            name: "description".to_owned(),
            attr_type: AttributeType::Str,
            description: None,
        });
        s.semantic_search_enabled = true;
        s.embedding_fields = vec!["description".to_owned()];
        assert!(s.validate().is_ok());
        assert!(s.is_semantic_search_enabled());
    }

    #[test]
    fn semantic_search_enabled_with_multiple_str_attributes_is_valid() {
        let mut s = valid_schema();
        s.attributes.push(AttributeDef {
            name: "title".to_owned(),
            attr_type: AttributeType::Str,
            description: None,
        });
        s.attributes.push(AttributeDef {
            name: "body".to_owned(),
            attr_type: AttributeType::Str,
            description: None,
        });
        s.semantic_search_enabled = true;
        s.embedding_fields = vec!["title".to_owned(), "body".to_owned()];
        assert!(s.validate().is_ok());
        assert!(s.is_semantic_search_enabled());
    }

    #[test]
    fn semantic_search_enabled_without_fields_is_rejected() {
        let mut s = valid_schema();
        s.semantic_search_enabled = true;
        assert!(matches!(s.validate(), Err(SchemaError::SemanticSearchMissingField)));
    }

    #[test]
    fn semantic_search_enabled_with_empty_string_in_fields_is_rejected() {
        let mut s = valid_schema();
        s.semantic_search_enabled = true;
        s.embedding_fields = vec![String::new()];
        assert!(matches!(s.validate(), Err(SchemaError::SemanticSearchMissingField)));
    }

    #[test]
    fn embedding_field_conflicting_with_index_is_rejected() {
        let mut s = valid_schema();
        s.semantic_search_enabled = true;
        s.embedding_fields = vec!["status".to_owned()]; // "status" is already an index field
        assert!(matches!(
            s.validate(),
            Err(SchemaError::EmbeddingFieldConflict { field }) if field == "status"
        ));
    }

    #[test]
    fn embedding_field_must_be_declared_as_str_attribute() {
        let mut s = valid_schema();
        s.semantic_search_enabled = true;
        s.embedding_fields = vec!["body".to_owned()]; // not declared in attributes
        assert!(matches!(
            s.validate(),
            Err(SchemaError::EmbeddingFieldNotString { field }) if field == "body"
        ));
    }

    #[test]
    fn embedding_field_declared_as_non_str_attribute_is_rejected() {
        let mut s = valid_schema();
        s.attributes.push(AttributeDef {
            name: "count".to_owned(),
            attr_type: AttributeType::Int, // Int, not Str
            description: None,
        });
        s.semantic_search_enabled = true;
        s.embedding_fields = vec!["count".to_owned()];
        assert!(matches!(
            s.validate(),
            Err(SchemaError::EmbeddingFieldNotString { field }) if field == "count"
        ));
    }

    #[test]
    fn duplicate_embedding_fields_are_rejected() {
        let mut s = valid_schema();
        s.attributes.push(AttributeDef {
            name: "body".to_owned(),
            attr_type: AttributeType::Str,
            description: None,
        });
        s.semantic_search_enabled = true;
        s.embedding_fields = vec!["body".to_owned(), "body".to_owned()];
        assert!(matches!(
            s.validate(),
            Err(SchemaError::DuplicateFieldName { field }) if field == "body"
        ));
    }

    #[test]
    fn is_semantic_search_enabled_false_when_disabled() {
        let mut s = valid_schema();
        s.attributes.push(AttributeDef {
            name: "body".to_owned(),
            attr_type: AttributeType::Str,
            description: None,
        });
        s.semantic_search_enabled = false;
        s.embedding_fields = vec!["body".to_owned()];
        assert!(!s.is_semantic_search_enabled());
    }

    #[test]
    fn semantic_search_schema_roundtrips_json() {
        let mut s = valid_schema();
        s.attributes.push(AttributeDef {
            name: "body".to_owned(),
            attr_type: AttributeType::Str,
            description: None,
        });
        s.semantic_search_enabled = true;
        s.embedding_fields = vec!["body".to_owned()];
        let json = serde_json::to_string(&s).unwrap();
        let restored: DocStoreSchema = serde_json::from_str(&json).unwrap();
        assert!(restored.semantic_search_enabled);
        assert_eq!(restored.embedding_fields, vec!["body"]);
        assert!(restored.is_semantic_search_enabled());
    }

    #[test]
    fn semantic_search_disabled_schema_omits_fields_in_json() {
        let s = valid_schema(); // semantic_search_enabled defaults to false
        let json = serde_json::to_string(&s).unwrap();
        // embedding_fields is empty so skip_serializing_if applies
        assert!(!json.contains("embedding_fields"));
        assert!(!json.contains("embedding_field"));
    }

    // ── AddEmbeddingAttribute amendment tests ─────────────────────────────────

    #[test]
    fn amendment_add_embedding_attribute_happy_path() {
        let mut s = valid_schema();
        s.apply_amendment(SchemaAmendment::AddEmbeddingAttribute {
            name: "body".to_owned(),
            description: Some("doc body".to_owned()),
        })
        .unwrap();
        let attr = s.attributes.iter().find(|a| a.name == "body").unwrap();
        assert_eq!(attr.attr_type, AttributeType::Str);
        assert_eq!(attr.description.as_deref(), Some("doc body"));
        assert!(s.embedding_fields.contains(&"body".to_owned()));
        assert!(s.semantic_search_enabled);
    }

    #[test]
    fn amendment_add_embedding_attribute_enables_semantic_search() {
        let mut s = valid_schema();
        assert!(!s.semantic_search_enabled);
        s.apply_amendment(SchemaAmendment::AddEmbeddingAttribute {
            name: "summary".to_owned(),
            description: None,
        })
        .unwrap();
        assert!(s.semantic_search_enabled);
    }

    #[test]
    fn amendment_add_embedding_attribute_empty_name_rejected() {
        let mut s = valid_schema();
        let result = s.apply_amendment(SchemaAmendment::AddEmbeddingAttribute {
            name: String::new(),
            description: None,
        });
        assert!(matches!(result, Err(SchemaError::EmptyAttributeName)));
    }

    #[test]
    fn amendment_add_embedding_attribute_conflicts_with_index_field() {
        let mut s = valid_schema();
        // "status" is already an index field
        let result = s.apply_amendment(SchemaAmendment::AddEmbeddingAttribute {
            name: "status".to_owned(),
            description: None,
        });
        assert!(matches!(
            result,
            Err(SchemaError::EmbeddingFieldConflict { field }) if field == "status"
        ));
    }

    #[test]
    fn amendment_add_embedding_attribute_duplicate_attribute_rejected() {
        let mut s = valid_schema();
        s.attributes.push(AttributeDef {
            name: "body".to_owned(),
            attr_type: AttributeType::Str,
            description: None,
        });
        let result = s.apply_amendment(SchemaAmendment::AddEmbeddingAttribute {
            name: "body".to_owned(),
            description: None,
        });
        assert!(matches!(
            result,
            Err(SchemaError::DuplicateFieldName { field }) if field == "body"
        ));
    }

    /// A namespace has at most one vector index: once an embedding attribute has
    /// enabled semantic search, adding another is rejected (drop the vector index
    /// first). The caller may instead declare multiple embedding fields up front
    /// at create time.
    #[test]
    fn amendment_add_embedding_attribute_rejected_when_already_enabled() {
        let mut s = valid_schema();
        s.apply_amendment(SchemaAmendment::AddEmbeddingAttribute {
            name: "title".to_owned(),
            description: None,
        })
        .unwrap();
        assert!(s.semantic_search_enabled);

        let result = s.apply_amendment(SchemaAmendment::AddEmbeddingAttribute {
            name: "body".to_owned(),
            description: None,
        });
        assert!(
            matches!(result, Err(SchemaError::SemanticSearchAlreadyEnabled { .. })),
            "second embedding add must be rejected, got {result:?}"
        );
        // The rejected field left no trace.
        assert_eq!(s.embedding_fields, vec!["title"]);
        assert!(!s.attributes.iter().any(|a| a.name == "body"));
    }

    #[test]
    fn amendment_enable_vector_index_multi_field() {
        let mut s = valid_schema();
        s.apply_amendment(SchemaAmendment::EnableVectorIndex {
            fields: vec!["title".to_owned(), "body".to_owned()],
        })
        .unwrap();
        assert!(s.semantic_search_enabled);
        assert_eq!(s.embedding_fields, vec!["title", "body"]);
        assert!(s.attributes.iter().any(|a| a.name == "title" && a.attr_type == AttributeType::Str));
        assert!(s.attributes.iter().any(|a| a.name == "body"));
        assert!(s.validate().is_ok());
    }

    #[test]
    fn amendment_enable_vector_index_rejected_when_already_enabled() {
        let mut s = valid_schema();
        s.apply_amendment(SchemaAmendment::EnableVectorIndex {
            fields: vec!["title".to_owned()],
        })
        .unwrap();
        let result = s.apply_amendment(SchemaAmendment::EnableVectorIndex {
            fields: vec!["body".to_owned()],
        });
        assert!(matches!(result, Err(SchemaError::SemanticSearchAlreadyEnabled { .. })));
    }

    #[test]
    fn amendment_enable_vector_index_empty_fields_rejected() {
        let mut s = valid_schema();
        let result = s.apply_amendment(SchemaAmendment::EnableVectorIndex { fields: vec![] });
        assert!(matches!(result, Err(SchemaError::SemanticSearchMissingField)));
        assert!(!s.semantic_search_enabled);
    }

    #[test]
    fn amendment_enable_vector_index_duplicate_in_list_rejected() {
        let mut s = valid_schema();
        let result = s.apply_amendment(SchemaAmendment::EnableVectorIndex {
            fields: vec!["dup".to_owned(), "dup".to_owned()],
        });
        assert!(matches!(result, Err(SchemaError::DuplicateFieldName { field }) if field == "dup"));
        // Nothing applied on failure.
        assert!(!s.semantic_search_enabled);
        assert!(s.embedding_fields.is_empty());
    }

    #[test]
    fn amendment_enable_vector_index_conflict_with_index_field_rejected() {
        let mut s = valid_schema(); // has an index on "status"
        let result = s.apply_amendment(SchemaAmendment::EnableVectorIndex {
            fields: vec!["status".to_owned()],
        });
        assert!(matches!(result, Err(SchemaError::EmbeddingFieldConflict { field }) if field == "status"));
    }

    // ── validate_doc tests ────────────────────────────────────────────────────

    fn schema_with_attrs() -> DocStoreSchema {
        DocStoreSchema {
            store_type: StoreType::Doc,
            namespace: "things".to_owned(),
            ns_id: None,
            key_type: KeyType::Uuid,
            attributes: vec![
                AttributeDef {
                    name: "label".to_owned(),
                    attr_type: AttributeType::Str,
                    description: None,
                },
                AttributeDef {
                    name: "count".to_owned(),
                    attr_type: AttributeType::Int,
                    description: None,
                },
                AttributeDef {
                    name: "active".to_owned(),
                    attr_type: AttributeType::Bool,
                    description: None,
                },
            ],
            indices: vec![
                IndexSpec {
                    field: "status".to_owned(),
                    index_type: IndexType::Str,
                },
                IndexSpec {
                    field: "score".to_owned(),
                    index_type: IndexType::Int,
                },
                IndexSpec {
                    field: "flag".to_owned(),
                    index_type: IndexType::Bool,
                },
            ],
            semantic_search_enabled: false,
            embedding_fields: vec![],
        }
    }

    #[test]
    fn valid_doc_with_all_fields_passes() {
        let s = schema_with_attrs();
        let doc = serde_json::json!({
            "status": "active",
            "score": 42,
            "flag": true,
            "label": "hello",
            "count": 7,
            "active": false,
            "extra": "ignored"
        });
        assert!(s.validate_doc(&doc).is_ok());
    }

    #[test]
    fn valid_doc_with_absent_fields_passes() {
        let s = schema_with_attrs();
        // All schema fields absent — absent fields are not an error.
        let doc = serde_json::json!({ "unrelated": 1 });
        assert!(s.validate_doc(&doc).is_ok());
    }

    #[test]
    fn non_object_doc_is_rejected() {
        let s = schema_with_attrs();
        assert!(matches!(s.validate_doc(&serde_json::json!([1, 2])), Err(SchemaError::DocNotObject)));
        assert!(matches!(s.validate_doc(&serde_json::json!("string")), Err(SchemaError::DocNotObject)));
        assert!(matches!(s.validate_doc(&serde_json::json!(42)), Err(SchemaError::DocNotObject)));
        assert!(matches!(s.validate_doc(&serde_json::json!(null)), Err(SchemaError::DocNotObject)));
    }

    #[test]
    fn wrong_type_on_str_index_field_is_rejected() {
        let s = schema_with_attrs();
        let doc = serde_json::json!({ "status": 99 }); // Int instead of Str
        assert!(matches!(
            s.validate_doc(&doc),
            Err(SchemaError::FieldTypeMismatch { field, expected, actual })
                if field == "status" && expected == "string" && actual == "number"
        ));
    }

    #[test]
    fn wrong_type_on_int_index_field_is_rejected() {
        let s = schema_with_attrs();
        let doc = serde_json::json!({ "score": "ninety" }); // Str instead of Int
        assert!(matches!(
            s.validate_doc(&doc),
            Err(SchemaError::FieldTypeMismatch { field, expected, actual })
                if field == "score" && expected == "integer" && actual == "string"
        ));
    }

    #[test]
    fn float_on_int_index_field_is_rejected() {
        let s = schema_with_attrs();
        let doc = serde_json::json!({ "score": 1.5 }); // float is not an integer
        assert!(matches!(
            s.validate_doc(&doc),
            Err(SchemaError::FieldTypeMismatch { field, .. }) if field == "score"
        ));
    }

    #[test]
    fn wrong_type_on_bool_index_field_is_rejected() {
        let s = schema_with_attrs();
        let doc = serde_json::json!({ "flag": "yes" }); // Str instead of Bool
        assert!(matches!(
            s.validate_doc(&doc),
            Err(SchemaError::FieldTypeMismatch { field, expected, actual })
                if field == "flag" && expected == "bool" && actual == "string"
        ));
    }

    #[test]
    fn wrong_type_on_str_attribute_is_rejected() {
        let s = schema_with_attrs();
        let doc = serde_json::json!({ "label": false }); // Bool instead of Str
        assert!(matches!(
            s.validate_doc(&doc),
            Err(SchemaError::FieldTypeMismatch { field, expected, actual })
                if field == "label" && expected == "string" && actual == "bool"
        ));
    }

    #[test]
    fn wrong_type_on_int_attribute_is_rejected() {
        let s = schema_with_attrs();
        let doc = serde_json::json!({ "count": "many" }); // Str instead of Int
        assert!(matches!(
            s.validate_doc(&doc),
            Err(SchemaError::FieldTypeMismatch { field, .. }) if field == "count"
        ));
    }

    #[test]
    fn wrong_type_on_bool_attribute_is_rejected() {
        let s = schema_with_attrs();
        let doc = serde_json::json!({ "active": 1 }); // Int instead of Bool
        assert!(matches!(
            s.validate_doc(&doc),
            Err(SchemaError::FieldTypeMismatch { field, .. }) if field == "active"
        ));
    }

    #[test]
    fn negative_integer_is_valid_for_int_index() {
        let s = schema_with_attrs();
        let doc = serde_json::json!({ "score": -100 });
        assert!(s.validate_doc(&doc).is_ok());
    }

    // ── Dot-path (nested field) tests ─────────────────────────────────────────

    fn schema_with_nested_fields() -> DocStoreSchema {
        DocStoreSchema {
            store_type: StoreType::Doc,
            namespace: "nested".to_owned(),
            ns_id: None,
            key_type: KeyType::Uuid,
            attributes: vec![AttributeDef {
                name: "meta.created_by".to_owned(),
                attr_type: AttributeType::Str,
                description: None,
            }],
            indices: vec![
                IndexSpec {
                    field: "address.city".to_owned(),
                    index_type: IndexType::Str,
                },
                IndexSpec {
                    field: "stats.count".to_owned(),
                    index_type: IndexType::Int,
                },
            ],
            semantic_search_enabled: false,
            embedding_fields: vec![],
        }
    }

    #[test]
    fn nested_index_field_with_correct_type_passes() {
        let s = schema_with_nested_fields();
        let doc = serde_json::json!({
            "address": { "city": "London" },
            "stats": { "count": 10 }
        });
        assert!(s.validate_doc(&doc).is_ok());
    }

    #[test]
    fn nested_index_field_with_wrong_type_is_rejected() {
        let s = schema_with_nested_fields();
        let doc = serde_json::json!({
            "address": { "city": 42 }  // Int instead of Str
        });
        assert!(matches!(
            s.validate_doc(&doc),
            Err(SchemaError::FieldTypeMismatch { field, .. }) if field == "address.city"
        ));
    }

    #[test]
    fn nested_attribute_with_correct_type_passes() {
        let s = schema_with_nested_fields();
        let doc = serde_json::json!({ "meta": { "created_by": "alice" } });
        assert!(s.validate_doc(&doc).is_ok());
    }

    #[test]
    fn nested_attribute_with_wrong_type_is_rejected() {
        let s = schema_with_nested_fields();
        let doc = serde_json::json!({ "meta": { "created_by": false } }); // Bool instead of Str
        assert!(matches!(
            s.validate_doc(&doc),
            Err(SchemaError::FieldTypeMismatch { field, .. }) if field == "meta.created_by"
        ));
    }

    #[test]
    fn absent_nested_field_is_not_an_error() {
        let s = schema_with_nested_fields();
        // "address" key is entirely absent
        let doc = serde_json::json!({ "other": "value" });
        assert!(s.validate_doc(&doc).is_ok());
    }

    // ── KvStoreSchema validation ──────────────────────────────────────────────

    fn valid_kv_schema() -> KvStoreSchema {
        KvStoreSchema {
            store_type: StoreType::Kv,
            namespace: "cache".to_owned(),
            ns_id: None,
            key_type: KvKeyType::Str,
            value_type: KvValueType::Str,
            semantic_search_enabled: false,
        }
    }

    #[test]
    fn kv_schema_valid_passes() {
        assert!(valid_kv_schema().validate().is_ok());
    }

    #[test]
    fn kv_schema_invalid_namespace_rejected() {
        let mut s = valid_kv_schema();
        s.namespace = "my namespace".to_owned();
        assert!(matches!(s.validate(), Err(SchemaError::InvalidNamespace)));
    }

    #[test]
    fn kv_schema_empty_namespace_rejected() {
        let mut s = valid_kv_schema();
        s.namespace = String::new();
        assert!(matches!(s.validate(), Err(SchemaError::InvalidNamespace)));
    }

    #[test]
    fn kv_schema_semantic_search_on_non_str_value_rejected() {
        for vt in [KvValueType::Int, KvValueType::F32, KvValueType::VecF32] {
            let s = KvStoreSchema {
                store_type: StoreType::Kv,
                namespace: "ns".to_owned(),
                ns_id: None,
                key_type: KvKeyType::Str,
                value_type: vt,
                semantic_search_enabled: true,
            };
            assert!(
                matches!(s.validate(), Err(SchemaError::KvSemanticSearchOnlyForStr)),
                "expected error for value_type={vt:?}"
            );
        }
    }

    #[test]
    fn kv_schema_semantic_search_on_str_is_valid() {
        let s = KvStoreSchema {
            store_type: StoreType::Kv,
            namespace: "ns".to_owned(),
            ns_id: None,
            key_type: KvKeyType::Str,
            value_type: KvValueType::Str,
            semantic_search_enabled: true,
        };
        assert!(s.validate().is_ok());
        assert!(s.is_semantic_search_enabled());
    }

    #[test]
    fn kv_schema_semantic_search_disabled_by_default() {
        assert!(!valid_kv_schema().is_semantic_search_enabled());
    }

    #[test]
    fn kv_schema_save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let schema = KvStoreSchema {
            store_type: StoreType::Kv,
            namespace: "tokens".to_owned(),
            ns_id: Some(7),
            key_type: KvKeyType::Int,
            value_type: KvValueType::F32,
            semantic_search_enabled: false,
        };
        schema.save(dir.path()).unwrap();
        let loaded = KvStoreSchema::load(dir.path(), "tokens").unwrap();
        assert_eq!(schema, loaded);
    }

    #[test]
    fn kv_schema_load_missing_returns_not_found() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(KvStoreSchema::load(dir.path(), "nope"), Err(SchemaError::NotFound { .. })));
    }

    // ── On-disk type discrimination ───────────────────────────────────────────
    // The authoritative discriminant is the mandatory `store_type` field; the
    // loaders dispatch on it via `peek_store_type`, and the typed `load`/`validate`
    // reject the wrong kind. (The key_type sets remain incidentally disjoint, but
    // that is no longer what tells the two schemas apart.)

    #[test]
    fn peek_store_type_reads_the_discriminant() {
        assert_eq!(peek_store_type(&serde_json::to_string(&valid_schema()).unwrap()), Some(StoreType::Doc));
        assert_eq!(peek_store_type(&serde_json::to_string(&valid_kv_schema()).unwrap()), Some(StoreType::Kv));
        assert_eq!(peek_store_type("not json"), None);
        assert_eq!(peek_store_type(r#"{"namespace":"x"}"#), None);
    }

    #[test]
    fn store_type_is_mandatory_on_the_wire() {
        // A schema JSON without store_type must fail to deserialize.
        let json = r#"{"namespace":"x","key_type":"u64","indices":[]}"#;
        assert!(serde_json::from_str::<DocStoreSchema>(json).is_err());
        let json = r#"{"namespace":"x","key_type":"str","value_type":"str"}"#;
        assert!(serde_json::from_str::<KvStoreSchema>(json).is_err());
    }

    #[test]
    fn validate_rejects_wrong_store_type() {
        let mut doc = valid_schema();
        doc.store_type = StoreType::Kv;
        assert!(matches!(doc.validate(), Err(SchemaError::WrongStoreType { expected: "doc", .. })));

        let mut kv = valid_kv_schema();
        kv.store_type = StoreType::Doc;
        assert!(matches!(kv.validate(), Err(SchemaError::WrongStoreType { expected: "kv", .. })));
    }

    #[test]
    fn load_rejects_wrong_store_type() {
        let dir = tempfile::tempdir().unwrap();
        // Persist a doc schema, then try to load it as a KV schema (same path).
        let doc = DocStoreSchema {
            namespace: "shared".to_owned(),
            ..valid_schema()
        };
        doc.save(dir.path()).unwrap();
        assert!(matches!(
            KvStoreSchema::load(dir.path(), "shared"),
            Err(SchemaError::WrongStoreType { expected: "kv", found: "doc", .. })
        ));
        // ...and the reverse direction.
        let dir2 = tempfile::tempdir().unwrap();
        let kv = KvStoreSchema {
            namespace: "shared".to_owned(),
            ..valid_kv_schema()
        };
        kv.save(dir2.path()).unwrap();
        assert!(matches!(
            DocStoreSchema::load(dir2.path(), "shared"),
            Err(SchemaError::WrongStoreType { expected: "doc", found: "kv", .. })
        ));
    }

    // ── KvKeyType serialisation ───────────────────────────────────────────────

    #[test]
    fn kv_key_str_serialize_roundtrip() {
        let key = serde_json::Value::String("hello world".to_owned());
        let bytes = KvKeyType::Str.serialize_key(&key).unwrap();
        let restored = KvKeyType::Str.deserialize_key(&bytes).unwrap();
        assert_eq!(key, restored);
    }

    #[test]
    fn kv_key_int_serialize_roundtrip() {
        for n in [0i64, 1, -1, i64::MIN, i64::MAX, 42] {
            let key = serde_json::Value::from(n);
            let bytes = KvKeyType::Int.serialize_key(&key).unwrap();
            let restored = KvKeyType::Int.deserialize_key(&bytes).unwrap();
            assert_eq!(key, restored, "roundtrip failed for n={n}");
        }
    }

    #[test]
    fn kv_key_int_ordering_preserved_by_big_endian() {
        // Big-endian encoding means lexicographic byte order == numeric order.
        let keys: Vec<i64> = vec![0, 1, 42, 100, i64::MAX];
        let encoded: Vec<Vec<u8>> = keys
            .iter()
            .map(|&n| KvKeyType::Int.serialize_key(&serde_json::Value::from(n)).unwrap())
            .collect();
        let mut sorted = encoded.clone();
        sorted.sort();
        assert_eq!(encoded, sorted);
    }

    #[test]
    fn kv_key_str_rejects_non_string() {
        let err = KvKeyType::Str.serialize_key(&serde_json::json!(42));
        assert!(matches!(err, Err(SchemaError::KvKeyTypeMismatch { .. })));
    }

    #[test]
    fn kv_key_int_rejects_non_integer() {
        let err = KvKeyType::Int.serialize_key(&serde_json::json!("not-a-number"));
        assert!(matches!(err, Err(SchemaError::KvKeyTypeMismatch { .. })));
    }

    #[test]
    fn kv_key_int_rejects_float() {
        let err = KvKeyType::Int.serialize_key(&serde_json::json!(1.5));
        assert!(matches!(err, Err(SchemaError::KvKeyTypeMismatch { .. })));
    }

    #[test]
    fn kv_key_from_str_str_key() {
        let bytes = KvKeyType::Str.serialize_key_from_str("mykey").unwrap();
        assert_eq!(bytes, b"mykey");
    }

    #[test]
    fn kv_key_from_str_int_key() {
        let bytes = KvKeyType::Int.serialize_key_from_str("42").unwrap();
        let expected = 42i64.to_be_bytes().to_vec();
        assert_eq!(bytes, expected);
    }

    #[test]
    fn kv_key_from_str_invalid_int_rejected() {
        assert!(matches!(
            KvKeyType::Int.serialize_key_from_str("not-a-number"),
            Err(SchemaError::KvKeyTypeMismatch { .. })
        ));
    }

    // ── KvValueType serialisation ─────────────────────────────────────────────

    #[test]
    fn kv_value_int_roundtrip() {
        for n in [0i64, -1, 1, i64::MIN, i64::MAX] {
            let v = serde_json::Value::from(n);
            let bytes = KvValueType::Int.serialize_value(&v).unwrap();
            let restored = KvValueType::Int.deserialize_value(&bytes).unwrap();
            assert_eq!(v, restored, "roundtrip failed for n={n}");
        }
    }

    #[test]
    fn kv_value_str_roundtrip() {
        let v = serde_json::Value::String("the quick brown fox".to_owned());
        let bytes = KvValueType::Str.serialize_value(&v).unwrap();
        let restored = KvValueType::Str.deserialize_value(&bytes).unwrap();
        assert_eq!(v, restored);
    }

    #[test]
    fn kv_value_f32_roundtrip() {
        // Use values exactly representable as f32 to avoid precision surprises.
        for f in [0.0f32, 1.0, -1.0, 0.5, 1024.25] {
            let v = serde_json::json!(f);
            let bytes = KvValueType::F32.serialize_value(&v).unwrap();
            let restored = KvValueType::F32.deserialize_value(&bytes).unwrap();
            let got = restored.as_f64().unwrap() as f32;
            assert!((got - f).abs() < f32::EPSILON, "roundtrip failed for f={f}");
        }
    }

    #[test]
    fn kv_value_vec_f32_roundtrip() {
        let v = serde_json::json!([1.0f32, -0.5f32, 0.25f32]);
        let bytes = KvValueType::VecF32.serialize_value(&v).unwrap();
        let restored = KvValueType::VecF32.deserialize_value(&bytes).unwrap();
        let arr = restored.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        let expected = [1.0f32, -0.5, 0.25];
        for (got, exp) in arr.iter().zip(expected.iter()) {
            assert!((got.as_f64().unwrap() as f32 - exp).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn kv_value_vec_f32_empty_roundtrip() {
        let v = serde_json::json!([]);
        let bytes = KvValueType::VecF32.serialize_value(&v).unwrap();
        assert!(bytes.is_empty());
        let restored = KvValueType::VecF32.deserialize_value(&bytes).unwrap();
        assert_eq!(restored.as_array().unwrap().len(), 0);
    }

    #[test]
    fn kv_value_int_rejects_str() {
        assert!(matches!(
            KvValueType::Int.serialize_value(&serde_json::json!("text")),
            Err(SchemaError::KvValueTypeMismatch { .. })
        ));
    }

    #[test]
    fn kv_value_str_rejects_number() {
        assert!(matches!(
            KvValueType::Str.serialize_value(&serde_json::json!(42)),
            Err(SchemaError::KvValueTypeMismatch { .. })
        ));
    }

    #[test]
    fn kv_value_f32_rejects_string() {
        assert!(matches!(
            KvValueType::F32.serialize_value(&serde_json::json!("bad")),
            Err(SchemaError::KvValueTypeMismatch { .. })
        ));
    }

    #[test]
    fn kv_value_vec_f32_rejects_non_array() {
        assert!(matches!(
            KvValueType::VecF32.serialize_value(&serde_json::json!(42)),
            Err(SchemaError::KvValueTypeMismatch { .. })
        ));
    }

    #[test]
    fn kv_value_vec_f32_rejects_mixed_array() {
        assert!(matches!(
            KvValueType::VecF32.serialize_value(&serde_json::json!([1.0, "bad"])),
            Err(SchemaError::KvValueTypeMismatch { .. })
        ));
    }

    #[test]
    fn kv_value_int_corrupt_bytes_rejected() {
        assert!(matches!(
            KvValueType::Int.deserialize_value(&[0u8; 5]), // wrong length
            Err(SchemaError::KvValueCorrupt)
        ));
    }

    #[test]
    fn kv_value_f32_corrupt_bytes_rejected() {
        assert!(matches!(
            KvValueType::F32.deserialize_value(&[0u8; 3]), // wrong length
            Err(SchemaError::KvValueCorrupt)
        ));
    }

    #[test]
    fn kv_value_vec_f32_unaligned_bytes_rejected() {
        assert!(matches!(
            KvValueType::VecF32.deserialize_value(&[0u8; 7]), // not multiple of 4
            Err(SchemaError::KvValueCorrupt)
        ));
    }

    #[test]
    fn kv_value_str_invalid_utf8_rejected() {
        assert!(matches!(
            KvValueType::Str.deserialize_value(&[0xFF, 0xFE]),
            Err(SchemaError::KvValueCorrupt)
        ));
    }
}
