//! Db — single entry-point facade for MinnalDB.
//!
//! Wraps the internal `Database` coordinator and exposes a clean,
//! user-facing API for CRUD, iteration, namespaces, and maintenance.

mod async_db;
mod async_namespace;
mod db;
mod namespace;

pub use async_db::AsyncDb;
pub use async_namespace::AsyncNamespace;
pub use db::Db;
pub use namespace::Namespace;

use log::info;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use crate::db::config::DbConfig;
use crate::db::config::toml::MinnalTomlConfig;
use crate::db::database::{Database, OpenStoreLookup};
use crate::db::error::{KVError, Result};
use crate::db::indexing::manager::FieldIndexHealth;
use crate::db::kv_store::index_set::ExtractorFn;
use crate::db::kv_store::{KVStore, KeyValue, ScanPage};
use crate::db::stats::{GCStats, Stats};
use crate::db::workers::index_checkpoint::{IndexCheckpointTarget, IndexCheckpointWorker};
use crate::db::workers::lsm_compaction::{LsmCompactionTarget, LsmCompactionWorker};
use crate::db::workers::ttl::{TtlTarget, TtlWorker};
use crate::db::workers::value_log_gc::{GCWorker, ValueLogGcTarget};
use crate::db::workers::wal_gc::WalGcWorker;
use crate::field_index::IndexValueType;
use crate::field_index::{FieldId, FieldReindexOutcome, FieldRepairOutcome, QueryOutcome};

// ── rkyv trait alias ───────────────────────────────────────────────────
//
// Bounds required for a type to be used with the `_typed` methods.
// Users derive these via `#[derive(Archive, RkyvSerialize, RkyvDeserialize)]`.

use rkyv::api::high::{HighSerializer, HighValidator};
use rkyv::rancor::Error as RkyvError;
use rkyv::rancor::Strategy;
use rkyv::ser::allocator::ArenaHandle;

/// Serialize a value to bytes via rkyv.
fn rkyv_serialize<T>(value: &T) -> Result<Vec<u8>>
where
    T: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>>,
{
    rkyv::to_bytes::<RkyvError>(value)
        .map(|buf| buf.to_vec())
        .map_err(|e| KVError::Serialization(format!("rkyv serialize: {}", e)))
}

/// Deserialize bytes into a value via rkyv, **validating them first**.
///
/// The bytes are whatever is stored under the key, and nothing ties them to `T`:
/// a caller can `put` raw bytes and then `get_typed` the key, `put_typed` one type
/// and `get_typed` another, or run a typed range scan over keys written any way at
/// all. This used `access_unchecked` on the assumption that the bytes always came
/// from `rkyv_serialize` of the same `T`, which made a safe public function able
/// to follow archived relative pointers out of bounds — undefined behaviour. Every
/// typed entry point already requires `T::Archived: CheckBytes`, so validating
/// costs no API change; mismatched bytes are now a `Serialization` error.
fn rkyv_deserialize<T>(bytes: &[u8]) -> Result<T>
where
    T: rkyv::Archive,
    T::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<T, Strategy<rkyv::de::Pool, RkyvError>>,
{
    rkyv::from_bytes::<T, RkyvError>(bytes).map_err(|e| KVError::Serialization(format!("rkyv deserialize: {:?}", e)))
}

// ── Shared helpers ─────────────────────────────────────────────────────

/// Deserialize a list of raw key-value byte pairs into typed pairs.
fn deserialize_pairs<K, V>(pairs: Vec<(Vec<u8>, Vec<u8>)>) -> Result<Vec<(K, V)>>
where
    K: rkyv::Archive,
    K::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<K, Strategy<rkyv::de::Pool, RkyvError>>,
    V: rkyv::Archive,
    V::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<V, Strategy<rkyv::de::Pool, RkyvError>>,
{
    pairs
        .into_iter()
        .map(|(kb, vb)| {
            let key = rkyv_deserialize::<K>(&kb)?;
            let val = rkyv_deserialize::<V>(&vb)?;
            Ok((key, val))
        })
        .collect()
}

/// Deserialize a list of raw key bytes into typed keys.
fn deserialize_keys<K>(keys: Vec<Vec<u8>>) -> Result<Vec<K>>
where
    K: rkyv::Archive,
    K::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<K, Strategy<rkyv::de::Pool, RkyvError>>,
{
    keys.into_iter().map(|kb| rkyv_deserialize::<K>(&kb)).collect()
}

#[cfg(test)]
mod tests;

/// Tests of the facade that reach into internal state.
#[cfg(test)]
mod integration_tests;
