use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use minnal_db::{DocStoreError, SchemaError};
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
            | DocStoreError::Schema(SchemaError::SemanticSearchAlreadyEnabled { .. })
            | DocStoreError::Schema(SchemaError::WrongStoreType { .. }) => StatusCode::CONFLICT,

            DocStoreError::InvalidId(_)
            | DocStoreError::Schema(SchemaError::Serialize(_))
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

            // A malformed predicate, an unknown or un-indexed field, a type
            // mismatch, or a query past the parser's complexity limits are all
            // faults in the *request*. Reporting them as 500 told the caller the
            // database had failed and withheld the one thing that would let them
            // fix it — the parser's message.
            DocStoreError::Db(minnal_db::KVError::Query(_)) => StatusCode::BAD_REQUEST,

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
    use minnal_db::index::query::QueryError;

    async fn render(inner: DocStoreError) -> (StatusCode, String) {
        let response = AppError::from(inner).into_response();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        (status, json["error"].as_str().unwrap().to_owned())
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

    /// The other side of the split: a genuine engine failure still withholds its
    /// details, which may name internal paths or state.
    #[tokio::test]
    async fn non_query_engine_errors_stay_500_and_opaque() {
        let (status, message) = render(DocStoreError::Db(KVError::DatabaseClosed)).await;

        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(message, "internal server error");
    }
}
