//! The public error surface is actually usable from outside the crate.
//!
//! `KVError`'s subsystem variants wrap `LSMError`, `ValueLogError` and
//! `ShardedValueLogError`, which live inside the private `store` module. Before
//! they were re-exported a caller could match `KVError::LsmError(_)` but had no
//! way to *name* the value inside it, so the payload was unreachable — a variant
//! you can detect but not inspect.
//!
//! This file is an integration test on purpose: it compiles as an external
//! consumer of the crate, so it fails if any of these types stops being
//! reachable through the public path. A unit test inside the crate could not
//! tell the difference.

use minnal_db::{KVError, LSMError, ShardedValueLogError, ValueLogError, WalError};

/// Naming each type in a signature is the assertion — this must compile.
fn _payloads_are_namable(lsm: LSMError, vl: ValueLogError, svl: ShardedValueLogError, wal: WalError) -> Vec<String> {
    vec![lsm.to_string(), vl.to_string(), svl.to_string(), wal.to_string()]
}

#[test]
fn kv_error_payloads_can_be_bound_and_inspected() {
    // The payload of a subsystem variant can be bound to a named type and read,
    // not merely matched on.
    let err: KVError = KVError::LsmError(LSMError::Corruption("bad frame length".into()));

    let detail = match err {
        KVError::LsmError(inner) => {
            let named: LSMError = inner;
            named.to_string()
        }
        other => panic!("expected LsmError, got {other:?}"),
    };

    assert!(detail.contains("bad frame length"), "payload should be inspectable, got {detail:?}");
}

#[test]
fn subsystem_errors_convert_into_kv_error() {
    // The `#[from]` conversions are part of the contract too: a caller writing
    // a helper that returns Result<_, KVError> can use `?` on a subsystem error.
    fn helper() -> Result<(), KVError> {
        Err(LSMError::Corruption("propagated".into()))?;
        unreachable!()
    }

    let err = helper().unwrap_err();
    assert!(matches!(err, KVError::LsmError(_)), "got {err:?}");
}

/// An invalid predicate is reported as `KVError::Query`, carrying the parser's
/// own error rather than a stringified one.
///
/// It used to be flattened into `KVError::Serialization(String)`, which the API
/// layer could only treat as a generic engine failure — every bad query became a
/// `500 {"error":"internal server error"}` with the useful text dropped. The
/// variant is what lets a caller distinguish "your query is wrong" from "the
/// database broke", so these assertions are on the *type*, not just the text.
#[test]
fn invalid_predicates_surface_as_query_errors() {
    use minnal_db::index::query::QueryError;
    use minnal_db::{DEFAULT_NAMESPACE_ID, Db, DbConfig, IndexValue, IndexValueType};
    use std::sync::Arc;

    let dir = tempfile::TempDir::new().unwrap();
    let config = DbConfig {
        num_buckets: 2,
        ..DbConfig::default()
    };
    let db = Db::open_with_config(dir.path(), config).unwrap();

    let field = db.register_index_field(DEFAULT_NAMESPACE_ID, "status", IndexValueType::Str).unwrap();
    db.activate_field_index(
        DEFAULT_NAMESPACE_ID,
        field,
        IndexValueType::Str,
        Arc::new(|bytes: &[u8]| Some(IndexValue::Str(String::from_utf8_lossy(bytes).into_owned()))),
    )
    .unwrap();
    db.put(b"k1", b"active").unwrap();

    // A field with no index cannot be queried...
    let err = db.query_index(DEFAULT_NAMESPACE_ID, r#"nosuchfield = "x""#).unwrap_err();
    match err {
        KVError::Query(inner) => {
            let named: QueryError = inner;
            assert!(matches!(named, QueryError::UnknownField { .. }), "expected UnknownField, got {named:?}");
            assert!(named.to_string().contains("nosuchfield"), "message should name the field");
        }
        other => panic!("expected KVError::Query, got {other:?}"),
    }

    // ...nor is a syntactically broken predicate an engine failure.
    let err = db.query_index(DEFAULT_NAMESPACE_ID, "status === ").unwrap_err();
    assert!(matches!(err, KVError::Query(QueryError::Syntax { .. })), "got {err:?}");

    // The complexity bound reports itself as a query fault too, so the API can
    // answer 400 instead of appearing to have crashed.
    let deep = format!("{}status = \"active\"{}", "(".repeat(50_000), ")".repeat(50_000));
    let err = db.query_index(DEFAULT_NAMESPACE_ID, &deep).unwrap_err();
    assert!(matches!(err, KVError::Query(QueryError::TooComplex { .. })), "got {err:?}");

    db.shutdown().unwrap();
}
