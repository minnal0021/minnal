pub mod config;
pub mod database;
pub mod error;
pub mod facade;
pub(crate) mod fail_log;
pub(crate) mod index_checkpoint_worker;
pub(crate) mod index_health;
pub mod index_manager;
pub(crate) mod key_locks;
pub mod kv_store;
pub(crate) mod layout;
pub mod metrics;
pub mod namespace;
pub mod namespace_index;
pub(crate) mod query;
pub mod stats;
pub mod toml_config;
pub(crate) mod ttl_worker;
pub mod wal;
pub(crate) mod wal_gc;
pub(crate) mod wal_worker;

/// Shared fixtures for this module's unit tests.
#[cfg(test)]
pub(crate) mod test_support;

/// Integration tests for the `Db` / `AsyncDb` facade.
///
/// Test-only, and named for what it holds. It was `db::db` — which read as a
/// module of database code rather than 25 tests for a type defined elsewhere,
/// and needed an `allow(clippy::module_inception)` to say so.
#[cfg(test)]
mod facade_tests;
