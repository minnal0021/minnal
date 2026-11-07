//! What the vector workers ask of the document store.

use std::sync::Arc;

use crate::AsyncDb;
use crate::doc_store::error::{DocStoreError, SchemaError};
use crate::doc_store::schema::store::SchemaStore;
use crate::doc_store::write_gate::InFlightWrites;
use crate::semantic_search::workers::{DocumentText, WorkerError, WorkerHost};
use crate::vector_settings::VectorIndexSettings;

/// The document store's side of [`WorkerHost`].
pub(crate) struct DocStoreWorkerHost {
    pub(crate) db: Arc<AsyncDb>,
    pub(crate) schemas: Arc<SchemaStore>,
    pub(crate) in_flight: Arc<InFlightWrites>,
}

impl WorkerHost for DocStoreWorkerHost {
    fn vector_settings(&self, namespace: &str) -> Result<(VectorIndexSettings, u32), WorkerError> {
        self.schemas.vector_settings(namespace).map_err(|e| match e {
            DocStoreError::Schema(SchemaError::NotFound { .. }) => WorkerError::StoreGone(e.to_string()),
            e => WorkerError::Host(e.to_string()),
        })
    }

    fn store_exists(&self, namespace: &str) -> bool {
        self.schemas.contains(namespace)
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
