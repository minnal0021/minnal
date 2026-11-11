use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use minnal_db::{DocStoreError, SchemaError};
use tracing::error;

/// Wraps [`DocStoreError`] so it can be returned from axum handlers.
///
/// `Debug` is for tests and tracing only — what a client sees is decided by
/// [`IntoResponse`], which withholds the message for 5xx.
#[derive(Debug)]
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
            DocStoreError::NotFound { .. }
            | DocStoreError::IndexNotFound { .. }
            | DocStoreError::JobNotFound { .. }
            | DocStoreError::Schema(SchemaError::NotFound { .. }) => {
                StatusCode::NOT_FOUND
            }

            DocStoreError::AlreadyExists { .. }
            | DocStoreError::IndexAlreadyExists { .. }
            | DocStoreError::AttributeIsIndexed { .. }
            | DocStoreError::NamespaceNotEmpty { .. }
            | DocStoreError::Schema(SchemaError::SemanticSearchAlreadyEnabled { .. })
            | DocStoreError::Schema(SchemaError::VectorSettingFixed { .. })
            | DocStoreError::Schema(SchemaError::WrongStoreType { .. })
            // The store is being created or deleted: the request conflicts with
            // its lifecycle state, and the message says which.
            | DocStoreError::StoreCreating { .. }
            | DocStoreError::StoreDeleting { .. }
            | DocStoreError::StoreInMaintenance { .. }
            | DocStoreError::JobConflict { .. }
            | DocStoreError::JobNotCancellable { .. }
            | DocStoreError::InvalidTransition { .. } => StatusCode::CONFLICT,

            DocStoreError::InvalidId(_)
            | DocStoreError::Schema(SchemaError::Serialize(_))
            | DocStoreError::Schema(SchemaError::Malformed(_))
            | DocStoreError::Schema(SchemaError::InvalidNamespace)
            | DocStoreError::Schema(SchemaError::StoreNameTooLong { .. })
            | DocStoreError::Schema(SchemaError::DocNotObject)
            | DocStoreError::Schema(SchemaError::FieldTypeMismatch { .. })
            | DocStoreError::Schema(SchemaError::ReservedNamespace { .. })
            | DocStoreError::Schema(SchemaError::TooManyIndices { .. })
            | DocStoreError::Schema(SchemaError::EmptyFieldName { .. })
            | DocStoreError::Schema(SchemaError::EmptyAttributeName)
            | DocStoreError::Schema(SchemaError::DuplicateFieldName { .. })
            | DocStoreError::Schema(SchemaError::InvalidFieldName { .. })
            | DocStoreError::Schema(SchemaError::AttributeIsIndexed { .. })
            | DocStoreError::Schema(SchemaError::AttributeNotFound { .. })
            | DocStoreError::Schema(SchemaError::SemanticSearchMissingField)
            | DocStoreError::Schema(SchemaError::EmbeddingFieldConflict { .. })
            | DocStoreError::Schema(SchemaError::EmbeddingFieldNotString { .. })
            | DocStoreError::Schema(SchemaError::KvKeyTypeMismatch { .. })
            | DocStoreError::Schema(SchemaError::KvValueTypeMismatch { .. })
            | DocStoreError::Schema(SchemaError::KvSemanticSearchOnlyForStr)
            | DocStoreError::Schema(SchemaError::StrKeyTooLong { .. })
            | DocStoreError::Schema(SchemaError::EmptyStrKey)
            | DocStoreError::Schema(SchemaError::StrKeyNotUtf8)
            | DocStoreError::InvalidProbeTrace { .. }
            | DocStoreError::Schema(SchemaError::InvalidVectorSetting { .. }) => StatusCode::BAD_REQUEST,

            // A key/value too large for the storage format's u32 length fields is
            // user-actionable: report 413 rather than a generic 500.
            DocStoreError::Db(minnal_db::KVError::WriteTooLarge(_)) => StatusCode::PAYLOAD_TOO_LARGE,

            // A malformed predicate, an unknown or un-indexed field, a type
            // mismatch, or a query past the parser's complexity limits are all
            // faults in the *request*. Reporting them as 500 told the caller the
            // database had failed and withheld the one thing that would let them
            // fix it — the parser's message.
            DocStoreError::Db(minnal_db::KVError::Query(_)) => StatusCode::BAD_REQUEST,

            // Semantic search on a store that does not have it enabled is a
            // request the server understood but cannot apply to this store —
            // the same 422 the admin vector endpoints return for it.
            DocStoreError::SemanticSearchNotEnabled { .. } => StatusCode::UNPROCESSABLE_ENTITY,

            // The request is well-formed but names a model this server, or its
            // embedding service, cannot serve; or updates settings a store
            // without semantic search does not have.
            DocStoreError::UnsupportedEmbeddingModel { .. }
            | DocStoreError::EmbeddingDimMismatch { .. }
            | DocStoreError::EmbeddingModelUnavailable { .. }
            | DocStoreError::Schema(SchemaError::VectorIndexNotConfigured { .. }) => StatusCode::UNPROCESSABLE_ENTITY,

            // The embedding service is down or failing: the request can work
            // once it is back, so 503 rather than 500.
            DocStoreError::EmbeddingFailed(_) => StatusCode::SERVICE_UNAVAILABLE,

            // No store can be created or dropped until the process restarts;
            // stores that exist keep working.
            DocStoreError::GovernorUnavailable => StatusCode::SERVICE_UNAVAILABLE,

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
            if matches!(self.inner, DocStoreError::EmbeddingFailed(_)) {
                "the embedding service is unavailable; retry later".to_owned()
            } else if matches!(self.inner, DocStoreError::GovernorUnavailable) {
                self.inner.to_string()
            } else {
                "internal server error".to_owned()
            }
        } else if let DocStoreError::Db(minnal_db::KVError::Query(query_error)) = &self.inner {
            // Report the parser's own message without the "database error:"
            // framing `DocStoreError::Db` adds — the fault is in the caller's
            // query, and saying "database error" for a 400 misdirects them.
            query_error.to_string()
        } else {
            self.inner.to_string()
        };
        let body = Json(serde_json::json!({ "error": message }));
        (status, body).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minnal_db::KVError;
    use minnal_db::field_index::query::QueryError;

    async fn render(inner: DocStoreError) -> (StatusCode, String) {
        let response = AppError::from(inner).into_response();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        (status, json["error"].as_str().unwrap().to_owned())
    }

    /// A missing index is the caller's mistake, not a server fault: 404 with the
    /// reason, matching the routes that already mapped it by hand.
    #[tokio::test]
    async fn index_not_found_is_a_404() {
        let (status, msg) = render(DocStoreError::IndexNotFound {
            namespace: "users".into(),
            field: "nope".into(),
        })
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(msg.contains("nope"), "got: {msg}");
    }

    /// A bad predicate is the caller's fault. Reporting it as 500 with the text
    /// withheld — which is what every query error did before `KVError::Query`
    /// existed — leaves the client unable to tell an invalid query from a broken
    /// database, and with nothing to act on.
    #[tokio::test]
    async fn query_errors_are_400_and_carry_the_parsers_message() {
        let cases = [
            QueryError::UnknownField { name: "title".into() },
            QueryError::Syntax {
                pos: 12,
                msg: "expected a value".into(),
            },
            QueryError::TooComplex {
                msg: "nesting depth exceeded".into(),
            },
            QueryError::InactiveField { field: "status".into() },
        ];

        for case in cases {
            let expected = case.to_string();
            let (status, message) = render(DocStoreError::Db(KVError::Query(case))).await;

            assert_eq!(status, StatusCode::BAD_REQUEST, "query errors must not be 5xx");
            assert_eq!(message, expected, "the parser's own message must reach the client, unframed");
            assert!(
                !message.contains("database error"),
                "a client error must not be framed as a database failure: {message:?}"
            );
        }
    }

    /// Key-validation failures are the caller's fault and must carry their
    /// message. A `SchemaError` variant missing from the 400 arm falls through
    /// to the catch-all and becomes a 500 with the text suppressed — which
    /// would tell a client that a 51-byte key had broken the database.
    #[tokio::test]
    async fn string_key_validation_errors_are_400_and_explain_themselves() {
        // `SchemaError` is not `Clone`, so each case is a constructor.
        let cases: [fn() -> SchemaError; 3] = [
            || SchemaError::StrKeyTooLong { max: 50, len: 51 },
            || SchemaError::EmptyStrKey,
            || SchemaError::StrKeyNotUtf8,
        ];

        for make in cases {
            let expected = DocStoreError::Schema(make()).to_string();
            let (status, message) = render(DocStoreError::Schema(make())).await;

            assert_eq!(status, StatusCode::BAD_REQUEST, "key validation must not be 5xx: {expected}");
            assert_eq!(message, expected, "the client needs to be told what was wrong with the key");
        }
    }

    /// The other side of the split: a genuine engine failure still withholds its
    /// details, which may name internal paths or state.
    #[tokio::test]
    async fn non_query_engine_errors_stay_500_and_opaque() {
        let (status, message) = render(DocStoreError::Db(KVError::DatabaseClosed)).await;

        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(message, "internal server error");
    }

    /// A document that is not a JSON object, or a field of the wrong type, is the
    /// caller's mistake: 400 with the reason, not 500.
    #[tokio::test]
    async fn a_document_the_schema_rejects_is_a_400() {
        let (status, msg) = render(DocStoreError::Schema(SchemaError::DocNotObject)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("object"), "got: {msg}");
    }

    /// The embedding service being down is not a server fault the caller can do
    /// nothing about: 503 says to retry, and the internal detail stays in the log.
    #[tokio::test]
    async fn a_failed_embedding_is_a_503_without_internal_detail() {
        let (status, msg) = render(DocStoreError::EmbeddingFailed("HTTP 503 from http://10.0.0.1:8001/embedding".into())).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(!msg.contains("10.0.0.1"), "got: {msg}");
    }
}
