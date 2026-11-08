use std::path::PathBuf;

use thiserror::Error;

// ── Schema-level errors (validation and persistence) ──────────────────────

/// Errors produced by schema validation and persistence.
#[derive(Debug, Error)]
pub enum SchemaError {
    #[error("namespace must be non-empty and contain only alphanumerics, underscores, or hyphens")]
    InvalidNamespace,

    #[error("store name is {len} bytes; the limit is {max}")]
    StoreNameTooLong { len: usize, max: usize },

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

    /// A schema stored in `system_schema` does not parse.
    #[error("stored schema of namespace '{namespace}' is unreadable: {reason}")]
    Corrupt { namespace: String, reason: String },

    /// The name is taken by a system namespace (`system` or `system_*`).
    #[error("namespace '{namespace}' is reserved: 'system' and names starting with 'system_' belong to system namespaces")]
    ReservedNamespace { namespace: String },

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

impl From<crate::vector_index::vector_settings::VectorSettingsError> for SchemaError {
    fn from(e: crate::vector_index::vector_settings::VectorSettingsError) -> Self {
        use crate::vector_index::vector_settings::VectorSettingsError as E;
        match e {
            E::Invalid { field, reason } => Self::InvalidVectorSetting { field, reason },
            E::Fixed { field, current, requested } => Self::VectorSettingFixed { field, current, requested },
            E::NotConfigured { namespace } => Self::VectorIndexNotConfigured { namespace },
        }
    }
}

#[cfg(feature = "semantic-search")]
impl From<crate::semantic_search::context::ContextError> for DocStoreError {
    fn from(e: crate::semantic_search::context::ContextError) -> Self {
        use crate::semantic_search::context::ContextError as E;
        match e {
            E::UnsupportedEmbeddingModel { model, supported } => Self::UnsupportedEmbeddingModel { model, supported },
            E::EmbeddingDimMismatch { model, dim, expected } => Self::EmbeddingDimMismatch { model, dim, expected },
            E::VectorIndexNotSeeded { namespace } => Self::VectorIndexNotSeeded { namespace },
            E::SemanticSearchNotEnabled { namespace } => Self::SemanticSearchNotEnabled { namespace },
            E::VecRebuildInProgress { namespace } => Self::VecRebuildInProgress { namespace },
            E::EmbeddingFailed(reason) => Self::EmbeddingFailed(reason),
            E::InvalidProbeTrace { namespace, reason } => Self::InvalidProbeTrace { namespace, reason },
            E::Settings(e) => e.into(),
            E::Db(e) => Self::Db(e),
        }
    }
}

impl From<crate::vector_index::vector_settings::VectorSettingsError> for DocStoreError {
    fn from(e: crate::vector_index::vector_settings::VectorSettingsError) -> Self {
        Self::Schema(e.into())
    }
}

/// Errors produced by [`DocStore`] operations.
///
/// [`DocStore`]: crate::doc_store::DocStore
#[derive(Debug, Error)]
pub enum DocStoreError {
    /// A namespace names an embedding model this server's `supported_models`
    /// does not list.
    #[error("embedding model '{model}' is not supported by this server (supported: {supported})")]
    UnsupportedEmbeddingModel { model: String, supported: String },

    /// The embedding service does not serve a namespace's model at its
    /// dimension (it answered "unknown model", or returned another dimension).
    #[error("the embedding service cannot serve model '{model}' at dimension {dim}: {reason}")]
    EmbeddingModelUnavailable { model: String, dim: u32, reason: String },

    /// A semantic namespace has no centres and postings to search or index with.
    #[error("namespace '{namespace}' has no seeded centres; restart to seed them, or drop its vector index and enable it again")]
    VectorIndexNotSeeded { namespace: String },

    /// A namespace's embedding dimension is not the one the server lists for
    /// its model (`supported_models`).
    #[error("embedding_dim {dim} does not match model '{model}', which this server runs at {expected} dimensions")]
    EmbeddingDimMismatch { model: String, dim: u32, expected: usize },

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

    /// The store is still being set up; it takes requests once it is created.
    #[error("store '{namespace}' is being created")]
    StoreCreating { namespace: String },

    /// The store is being dropped.
    #[error("namespace '{namespace}' is being deleted")]
    StoreDeleting { namespace: String },

    /// The store has index jobs running, and cannot be dropped until they end.
    #[error("namespace '{namespace}' has index jobs running; wait for them to end, or cancel them, before dropping it")]
    StoreInMaintenance { namespace: String },

    /// An index job cannot start while a job it conflicts with runs.
    #[error("namespace '{namespace}': job {job} cannot start while job {running} runs")]
    JobConflict { namespace: String, job: String, running: String },

    /// The store has no running job of that name.
    #[error("namespace '{namespace}' has no running job {job}")]
    JobNotFound { namespace: String, job: String },

    /// The job cannot be cancelled: stopping it part-way would leave its index
    /// unusable.
    #[error("namespace '{namespace}': job {job} cannot be cancelled")]
    JobNotCancellable { namespace: String, job: String },

    /// A lifecycle move the store's state does not allow (a drop of a store
    /// that is not created, say).
    #[error("store '{namespace}' cannot move from {from:?} to {to:?}")]
    InvalidTransition {
        namespace: String,
        from: crate::namespace_lifecycle::NamespaceState,
        to: crate::namespace_lifecycle::NamespaceState,
    },

    /// A store's creation or drop failed every run the governor gave it; the
    /// store is left in its state until the process restarts.
    #[error("store '{namespace}': its lifecycle task failed {attempts} time(s), last: {error}")]
    LifecycleTaskFailed { namespace: String, attempts: u32, error: String },

    /// The lifecycle governor, which creates and drops stores, is not running
    /// (it stopped after too many restarts). Stores that exist keep working;
    /// none can be created or dropped until the process is restarted.
    #[error("store lifecycle is unavailable: the lifecycle governor is not running")]
    GovernorUnavailable,

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

    /// A store cannot take this name: a namespace of that name, with no store,
    /// holds keys written directly through [`Db`](crate::Db).
    #[error("namespace '{namespace}' already holds keys and is not a store; remove it or choose another name")]
    NamespaceNotEmpty { namespace: String },

    /// The namespace's vector index is still being removed. Enabling it again
    /// now would lose what is written before the removal finishes.
    #[error("the vector index of namespace '{namespace}' is still being removed; retry when it finishes")]
    VectorCleanupInProgress { namespace: String },

    /// A partition rebuild is already running for this namespace.
    #[error("a partition rebuild for namespace '{namespace}' is already running")]
    VecRebuildInProgress { namespace: String },

    /// The vector index for this namespace is currently being dropped (background cleanup).
    /// Wait for the cleanup to complete before re-enabling semantic search.
    #[error("vector index cleanup for namespace '{namespace}' is already in progress")]
    VecIndexCleanupInProgress { namespace: String },

    /// A probe trace was asked for something it cannot show: its window
    /// (`start..end` over the namespace's probe order) is empty, runs past the
    /// namespace's postings, or is wider than
    /// [`MAX_TRACE_PROBES`](crate::semantic_search::service::MAX_TRACE_PROBES),
    /// or its `top_k` is 0 or above
    /// [`MAX_TRACE_TOP_K`](crate::semantic_search::service::MAX_TRACE_TOP_K).
    #[cfg(feature = "semantic-search")]
    #[error("invalid probe trace for namespace '{namespace}': {reason}")]
    InvalidProbeTrace { namespace: String, reason: String },

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

impl From<crate::namespace_lifecycle::LifecycleError> for DocStoreError {
    /// Each lifecycle error becomes the flat variant of the same meaning, so a
    /// caller matches one error type.
    fn from(e: crate::namespace_lifecycle::LifecycleError) -> Self {
        use crate::namespace_lifecycle::LifecycleError as L;
        match e {
            L::Creating { namespace } => Self::StoreCreating { namespace },
            L::Deleting { namespace } => Self::StoreDeleting { namespace },
            L::InvalidTransition { namespace, from, to } => Self::InvalidTransition { namespace, from, to },
            L::TaskFailed { namespace, attempts, error } => Self::LifecycleTaskFailed { namespace, attempts, error },
            L::GovernorUnavailable => Self::GovernorUnavailable,
            L::InMaintenance { namespace } => Self::StoreInMaintenance { namespace },
            L::NotFound { namespace } => Self::NotFound { namespace },
            L::JobConflict { namespace, job, running } => Self::JobConflict { namespace, job, running },
            L::JobNotFound { namespace, job } => Self::JobNotFound { namespace, job },
            L::JobNotCancellable { namespace, job } => Self::JobNotCancellable { namespace, job },
            L::Db(e) => Self::Db(e),
            L::Record(e) => Self::Io(std::io::Error::other(format!("maintenance record: {e}"))),
        }
    }
}
