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
