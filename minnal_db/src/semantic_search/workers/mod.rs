//! The background workers of the vector index: `vec_index` carries out the embed
//! queue, and `ivf_maintenance` splits and merges postings. The engine's workers
//! are in `db::workers`, whose module docs list them all.
//!
//! They need to know a little about the stores above them, and ask through
//! [`WorkerHost`], so this layer does not depend on the document store. They are
//! compiled with `doc-store`, the only layer that starts them.

pub(crate) mod ivf_maintenance;
pub(crate) mod vec_index;

use std::future::Future;

use crate::semantic_search::context::ContextError;
use crate::vector_settings::VectorIndexSettings;

/// What the workers ask of the stores above them.
pub(crate) trait WorkerHost: Send + Sync + 'static {
    /// `namespace`'s vector-index settings and id: each entry is embedded with
    /// its namespace's model, dimension and chunking.
    fn vector_settings(&self, namespace: &str) -> Result<(VectorIndexSettings, u32), WorkerError>;

    /// Whether a store called `namespace` exists.
    fn store_exists(&self, namespace: &str) -> bool;

    /// Whether a write to `namespace`'s document `key` is in progress, in which
    /// case the worker skips it; that write's end wakes the worker again.
    fn write_in_progress(&self, namespace: &str, key: &[u8]) -> bool;

    /// What `namespace`'s document `key` holds now. The caller has seen no write
    /// to it in progress.
    fn document_text(&self, namespace: &str, key: &[u8]) -> impl Future<Output = Result<DocumentText, WorkerError>> + Send;
}

/// Why a worker could not carry out a step.
#[derive(Debug, thiserror::Error)]
pub(crate) enum WorkerError {
    #[error(transparent)]
    Context(#[from] ContextError),

    #[error("database error: {0}")]
    Db(#[from] crate::KVError),

    #[error("embedding failed: {0}")]
    EmbeddingFailed(String),

    /// The namespace has no store: it was dropped, or never existed. Holds the
    /// host's description.
    #[error("{0}")]
    StoreGone(String),

    /// The host's own error, as it describes itself.
    #[error("{0}")]
    Host(String),
}

/// What a document (or KV value) holds now, for the vector worker's check of a
/// queue entry against it ([`vector_kv::check_entry_against_document`]).
pub(crate) enum DocumentText {
    /// The store is gone (dropped, or being dropped): leave the entry to the drop.
    StoreGone,
    /// The document is absent, or has no embedding text: it must have no vectors.
    Nothing,
    /// The document's embedding text.
    Text(String),
}

impl DocumentText {
    /// The text the document's vectors must be embedded from, if any.
    pub(crate) fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(t) => Some(t),
            Self::StoreGone | Self::Nothing => None,
        }
    }
}
