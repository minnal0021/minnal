use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use minnal_doc_store::{DocStoreError, SchemaError};
use tracing::error;

/// Wraps [`DocStoreError`] so it can be returned from axum handlers.
pub struct AppError {
    pub inner: DocStoreError,
    pub namespace: Option<String>,
    pub id: Option<String>,
}

impl AppError {
    pub fn with_ns(mut self, ns: &str) -> Self {
        self.namespace = Some(ns.to_owned());
        self
    }

    pub fn with_id(mut self, id: &str) -> Self {
        self.id = Some(id.to_owned());
        self
    }
}

impl From<DocStoreError> for AppError {
    fn from(e: DocStoreError) -> Self {
        AppError {
            inner: e,
            namespace: None,
            id: None,
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match &self.inner {
            DocStoreError::NotFound { .. } => StatusCode::NOT_FOUND,

            DocStoreError::AlreadyExists { .. }
            | DocStoreError::IndexAlreadyExists { .. }
            | DocStoreError::IndexBuildInProgress { .. }
            | DocStoreError::AttributeIsIndexed { .. }
            | DocStoreError::VecReindexInProgress { .. }
            | DocStoreError::VecIndexCleanupInProgress { .. }
            | DocStoreError::AttrIndexOpInProgress { .. }
            | DocStoreError::Schema(SchemaError::SemanticSearchAlreadyEnabled { .. }) => StatusCode::CONFLICT,

            DocStoreError::InvalidId(_)
            | DocStoreError::Schema(SchemaError::InvalidNamespace)
            | DocStoreError::Schema(SchemaError::TooManyIndices { .. })
            | DocStoreError::Schema(SchemaError::EmptyFieldName { .. })
            | DocStoreError::Schema(SchemaError::EmptyAttributeName)
            | DocStoreError::Schema(SchemaError::DuplicateFieldName { .. })
            | DocStoreError::Schema(SchemaError::AttributeIsIndexed { .. })
            | DocStoreError::Schema(SchemaError::AttributeNotFound { .. })
            | DocStoreError::Schema(SchemaError::SemanticSearchMissingField)
            | DocStoreError::Schema(SchemaError::EmbeddingFieldConflict { .. })
            | DocStoreError::Schema(SchemaError::EmbeddingFieldNotString { .. })
            | DocStoreError::Schema(SchemaError::KvKeyTypeMismatch { .. })
            | DocStoreError::Schema(SchemaError::KvValueTypeMismatch { .. })
            | DocStoreError::Schema(SchemaError::KvSemanticSearchOnlyForStr) => StatusCode::BAD_REQUEST,

            // A key/value too large for the storage format's u32 length fields is
            // user-actionable: report 413 rather than a generic 500.
            DocStoreError::Db(minnal_db::KVError::WriteTooLarge(_)) => StatusCode::PAYLOAD_TOO_LARGE,

            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };

        // 4xx errors are user-actionable, so the descriptive message is returned
        // to the client. 5xx errors may carry internal details (paths, internal
        // state), so the full error is only logged server-side and the client
        // receives a generic message.
        let message = if status.is_server_error() {
            error!(
                namespace = self.namespace.as_deref(),
                id = self.id.as_deref(),
                error = %self.inner,
                "internal server error"
            );
            "internal server error".to_owned()
        } else {
            self.inner.to_string()
        };
        let body = Json(serde_json::json!({ "error": message }));
        (status, body).into_response()
    }
}
