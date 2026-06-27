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

    #[error("attribute '{name}' is used by an active index — drop the index first")]
    AttributeIsIndexed { name: String },

    #[error("attribute '{name}' not found in schema")]
    AttributeNotFound { name: String },

    #[error("failed to serialize schema: {0}")]
    Serialize(#[from] serde_json::Error),

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

    #[error("KV value type mismatch: expected {expected}")]
    KvValueTypeMismatch { expected: &'static str },

    #[error("KV value is corrupted or has an unexpected length")]
    KvValueCorrupt,

    #[error("semantic search is only supported for KV stores with value_type = str")]
    KvSemanticSearchOnlyForStr,
}

// ── Doc-store-level errors ─────────────────────────────────────────────────

/// Errors produced by [`DocStore`] operations.
///
/// [`DocStore`]: crate::store::DocStore
#[derive(Debug, Error)]
pub enum DocStoreError {
    /// The underlying schema was invalid.
    #[error("schema error: {0}")]
    Schema(#[from] SchemaError),

    /// The underlying minnal_db returned an error.
    #[error("database error: {0}")]
    Db(#[from] minnal_db::KVError),

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

    /// The database directory is locked by another process.
    ///
    /// A `.lock` file exists at `{db_path}/.lock`, which means another
    /// `DocStore` instance is already open against this path, or a previous
    /// run did not shut down cleanly.  Remove the file manually to recover
    /// from an unclean shutdown.
    #[error(
        "database at '{path}' is locked — another instance may be running, or the previous run did not shut down cleanly (remove '{path}/.lock' to recover)"
    )]
    StoreLocked { path: PathBuf },

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
