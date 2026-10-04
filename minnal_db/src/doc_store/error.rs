use std::path::PathBuf;

use thiserror::Error;

// ── Schema-level errors (validation and persistence) ──────────────────────

/// Errors produced by schema validation and persistence.
#[derive(Debug, Error)]
pub enum SchemaError {
    #[error("namespace must be non-empty and contain only alphanumerics, underscores, or hyphens")]
    InvalidNamespace,

    #[error("too many indices: {count} specified, maximum is {max}")]
    TooManyIndices { count: usize, max: usize },

    #[error("index field name must be non-empty (index {index})")]
    EmptyFieldName { index: usize },

    #[error("attribute name must be non-empty")]
    EmptyAttributeName,

    #[error("duplicate field name: '{field}'")]
    DuplicateFieldName { field: String },

    #[error("field name '{field}' is not usable: {reason}")]
    InvalidFieldName { field: String, reason: &'static str },

    #[error("attribute '{name}' is used by an active index — drop the index first")]
    AttributeIsIndexed { name: String },

    #[error("attribute '{name}' not found in schema")]
    AttributeNotFound { name: String },

    #[error("failed to serialize schema: {0}")]
    Serialize(#[from] serde_json::Error),

    /// A schema in a request does not parse: a missing or misspelt key, or a
    /// value of the wrong type.
    #[error("invalid schema: {0}")]
    Malformed(serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("schema not found for namespace '{namespace}'")]
    NotFound { namespace: String },

    #[error("semantic_search_enabled is true but no embedding_fields were specified")]
    SemanticSearchMissingField,

    #[error("embedding field '{field}' conflicts with an existing index field name")]
    EmbeddingFieldConflict { field: String },

    #[error("namespace '{namespace}' already has a vector index — drop it before adding another")]
    SemanticSearchAlreadyEnabled { namespace: String },

    #[error("embedding field '{field}' must be declared as a string (Str) attribute")]
    EmbeddingFieldNotString { field: String },

    #[error("document must be a JSON object")]
    DocNotObject,

    #[error("field '{field}': expected {expected}, got {actual}")]
    FieldTypeMismatch {
        field: String,
        expected: &'static str,
        actual: &'static str,
    },

    #[error("KV key type mismatch: expected {expected}")]
    KvKeyTypeMismatch { expected: &'static str },

    #[error("string key is too long: {len} bytes, maximum is {max}")]
    StrKeyTooLong { max: usize, len: usize },

    #[error("string key must not be empty")]
    EmptyStrKey,

    #[error("string key is not valid UTF-8")]
    StrKeyNotUtf8,

    #[error("KV value type mismatch: expected {expected}")]
    KvValueTypeMismatch { expected: &'static str },

    #[error("KV value is corrupted or has an unexpected length")]
    KvValueCorrupt,

    #[error("semantic search is only supported for KV stores with value_type = str")]
    KvSemanticSearchOnlyForStr,

    /// A `vector_index` setting is out of range or malformed.
    #[error("vector_index.{field} {reason}")]
    InvalidVectorSetting { field: &'static str, reason: String },

    /// A request tried to change a vector-index setting that is fixed for the
    /// namespace's life (model, dimension, chunking) or read-only (code widths).
    #[error("vector_index.{field} is {current} for this namespace and cannot be changed (requested {requested})")]
    VectorSettingFixed {
        field: &'static str,
        current: String,
        requested: String,
    },

    /// The namespace has never had semantic search enabled, so it has no
    /// vector-index settings to update.
    #[error("namespace '{namespace}' has no vector index settings; enable semantic search first")]
    VectorIndexNotConfigured { namespace: String },

    #[error("wrong store type for namespace '{namespace}': expected {expected}, found {found}")]
    WrongStoreType {
        namespace: String,
        expected: &'static str,
        found: &'static str,
    },
}

// ── Doc-store-level errors ─────────────────────────────────────────────────

/// Errors produced by [`DocStore`] operations.
///
/// [`DocStore`]: crate::doc_store::store::DocStore
#[derive(Debug, Error)]
pub enum DocStoreError {
    /// A namespace names an embedding model this server has no centroids for.
    #[error("embedding model '{model}' is not supported by this server (supported: {supported})")]
    UnsupportedEmbeddingModel { model: String, supported: String },

    /// The embedding service does not serve a namespace's model at its
    /// dimension (it answered "unknown model", or returned another dimension).
    #[error("the embedding service cannot serve model '{model}' at dimension {dim}: {reason}")]
    EmbeddingModelUnavailable { model: String, dim: u32, reason: String },

    /// A namespace's embedding dimension does not match its model's centroids.
    #[error("embedding_dim {dim} does not match the {centroid_dim}-dimensional centroids of model '{model}'")]
    EmbeddingDimMismatch { model: String, dim: u32, centroid_dim: usize },

    /// The underlying schema was invalid.
    #[error("schema error: {0}")]
    Schema(#[from] SchemaError),

    /// The underlying minnal_db returned an error.
    #[error("database error: {0}")]
    Db(#[from] crate::KVError),

    /// A semantic-search / vector-index operation was requested, but this build
    /// was compiled without the `semantic-search` cargo feature.
    #[cfg(not(feature = "semantic-search"))]
    #[error("semantic search is not available: this build was compiled without the `semantic-search` feature")]
    SemanticSearchNotCompiled,

    /// A doc store with that namespace already exists.
    #[error("doc store '{namespace}' already exists")]
    AlreadyExists { namespace: String },

    /// No doc store with that namespace was found.
    #[error("doc store '{namespace}' not found")]
    NotFound { namespace: String },

    /// The specified index field does not exist in the schema.
    #[error("index field '{field}' not found in namespace '{namespace}'")]
    IndexNotFound { namespace: String, field: String },

    /// The field already has an active index.
    #[error("field '{field}' in namespace '{namespace}' is already indexed")]
    IndexAlreadyExists { namespace: String, field: String },

    /// A background index build for this field is already running.
    #[error("an index build for field '{field}' in namespace '{namespace}' is already in progress")]
    IndexBuildInProgress { namespace: String, field: String },

    /// Attempt to drop or amend an attribute that is currently indexed.
    /// Drop the index first, then retry the operation.
    #[error("attribute '{field}' in namespace '{namespace}' is used by an index — drop the index first")]
    AttributeIsIndexed { namespace: String, field: String },

    /// The `ns_id` is missing from the schema — store was not created via `DocStore::create`.
    #[error("namespace '{namespace}' has no stored ID — was the doc store created via DocStore::create?")]
    MissingNsId { namespace: String },

    /// A document ID could not be serialized or deserialized.
    #[error("invalid document ID: {0}")]
    InvalidId(String),

    /// An I/O error occurred (e.g. during file cleanup on drop).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The database directory is held open by another **live** process.
    ///
    /// The lock is an advisory `flock(2)` on `{db_path}/.lock`, which the kernel
    /// releases when its owner exits — including on a crash. So this error means
    /// a second instance really is running, not that a previous run left a file
    /// behind; deleting the lock file will not help and risks two writers.
    #[error("database at '{path}' is already open by another running instance ({})", match owner_pid {
        Some(pid) => format!("pid {pid}"),
        None => "pid unknown".to_owned(),
    })]
    StoreLocked { path: PathBuf, owner_pid: Option<u32> },

    /// The index build background task failed or was cancelled.
    #[error("index build failed: {0}")]
    BuildFailed(String),

    /// The embedding service call failed during a semantic-search-enabled write.
    #[error("embedding failed: {0}")]
    EmbeddingFailed(String),

    /// An operation that requires semantic search was requested on a namespace
    /// that does not have it enabled.
    #[error("semantic search is not enabled for namespace '{namespace}'")]
    SemanticSearchNotEnabled { namespace: String },

    /// A vector-index reindex (`index_all`) is already running for this namespace.
    /// Poll `GET /admin/indices/{namespace}/progress` and retry when it finishes.
    #[error("a vector index reindex for namespace '{namespace}' is already in progress")]
    VecReindexInProgress { namespace: String },

    /// The vector index for this namespace is currently being dropped (background cleanup).
    /// Wait for the cleanup to complete before re-enabling semantic search.
    #[error("vector index cleanup for namespace '{namespace}' is already in progress")]
    VecIndexCleanupInProgress { namespace: String },

    /// An exclusive attribute-index operation is already running for this namespace.
    #[error("an attribute index operation is already in progress for namespace '{namespace}'")]
    AttrIndexOpInProgress { namespace: String },
}

/// Shared conversion: serde_json errors become `DocStoreError::InvalidJson`.
impl From<serde_json::Error> for DocStoreError {
    fn from(e: serde_json::Error) -> Self {
        DocStoreError::InvalidId(e.to_string())
    }
}
