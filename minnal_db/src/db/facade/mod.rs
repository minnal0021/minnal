//! Db — single entry-point facade for MinnalDB.
//!
//! Wraps the internal `Database` coordinator and exposes a clean,
//! user-facing API for CRUD, iteration, namespaces, and maintenance.

mod async_db;
mod async_db_index;
mod async_db_maintenance;
mod async_db_typed;
mod async_namespace;
mod db;

pub use async_db::AsyncDb;
pub use async_namespace::AsyncNamespace;
pub(crate) use db::Db;
use log::info;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use crate::db::config::DbConfig;
use crate::db::database::{Database, OpenStoreLookup};
use crate::db::error::{KVError, Result};
use crate::db::indexing::manager::FieldIndexHealth;
use crate::db::stats::{GCStats, Stats};
use crate::db::workers::index_checkpoint::{IndexCheckpointTarget, IndexCheckpointWorker};
use crate::db::workers::lsm_compaction::{LsmCompactionTarget, LsmCompactionWorker};
use crate::db::workers::ttl::{TtlTarget, TtlWorker};
use crate::db::workers::value_log_gc::{GCWorker, ValueLogGcTarget};
use crate::db::workers::wal_gc::WalGcWorker;
use crate::field_index::IndexValueType;
use crate::field_index::{FieldId, FieldReindexOutcome, FieldRepairOutcome, QueryOutcome};
use crate::kv_store::index_set::ExtractorFn;
use crate::kv_store::{KVStore, KeyValue, ScanPage};

mod typed;

pub use typed::{RkyvDecode, RkyvEncode};
use typed::{deserialize_keys, deserialize_pairs, rkyv_deserialize, rkyv_serialize};

/// Not API (feature `bench-internals`): what the benches need to time the
/// engine directly. Nothing here is stable.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod bench_internals {
    pub use super::db::Db;
    pub use super::typed::{deserialize_keys, deserialize_pairs, rkyv_deserialize, rkyv_serialize};
}

#[cfg(test)]
mod tests;

/// Tests of the facade that reach into internal state.
#[cfg(test)]
mod integration_tests;
#[cfg(test)]
mod iterator_tests;
