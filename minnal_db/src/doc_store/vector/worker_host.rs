//! What the vector workers ask of the document store.

use std::sync::Arc;

use crate::AsyncDb;
use crate::doc_store::InFlightWrites;
use crate::doc_store::error::{DocStoreError, SchemaError};
use crate::doc_store::schema::store::SchemaStore;
use crate::namespace_lifecycle::NamespaceState;
use crate::semantic_search::workers::{DocumentText, WorkerError, WorkerHost};
use crate::vector_index::vector_settings::VectorIndexSettings;

/// The document store's side of [`WorkerHost`].
pub(crate) struct DocStoreWorkerHost {
    pub(crate) db: Arc<AsyncDb>,
    pub(crate) schemas: Arc<SchemaStore>,
    pub(crate) in_flight: Arc<InFlightWrites>,
}

impl WorkerHost for DocStoreWorkerHost {
    /// A store being dropped is gone to the workers: its entries are discarded
    /// and its work dropped, while its data is removed.
    fn vector_settings(&self, namespace: &str) -> Result<(VectorIndexSettings, crate::namespace::NamespaceId), WorkerError> {
        if self.schemas.state(namespace) == Some(NamespaceState::Deleting) {
            return Err(WorkerError::StoreGone(format!("store '{namespace}' is being deleted")));
        }
        self.schemas.vector_settings(namespace).map_err(|e| match e {
            DocStoreError::Schema(SchemaError::NotFound { .. }) => WorkerError::StoreGone(e.to_string()),
            e => WorkerError::Host(e.to_string()),
        })
    }

    /// Whether the store exists and is not being dropped. A store being
    /// created has no entries yet (it takes no writes until it is created).
    fn store_exists(&self, namespace: &str) -> bool {
        self.schemas.state(namespace).is_some_and(|s| s != NamespaceState::Deleting)
    }

    fn write_in_progress(&self, namespace: &str, key: &[u8]) -> bool {
        self.in_flight.contains(namespace, key)
    }

    async fn document_text(&self, namespace: &str, key: &[u8]) -> Result<DocumentText, WorkerError> {
        super::document_text(&self.db, &self.schemas, namespace, key)
            .await
            .map_err(|e| WorkerError::Host(e.to_string()))
    }
}
