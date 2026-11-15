//! The engine: `AsyncDb` and its namespaces (`facade`), the `Database`
//! coordinator, its configuration, both write-ahead logs, field-index upkeep
//! and the background workers.

pub mod config;
pub mod database;
pub use crate::error;
pub mod facade;
pub mod indexing;
mod namespace;
pub use crate::kv_store::stats;
pub mod wal;
pub(crate) mod workers;

/// Shared fixtures for this module's unit tests.
#[cfg(test)]
pub(crate) mod test_support;
