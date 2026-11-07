pub mod config;
pub mod database;
pub mod error;
pub mod facade;
pub mod indexing;
pub(crate) mod key_locks;
pub(crate) mod layout;
pub mod namespace;
pub mod stats;
pub mod toml_config;
pub mod wal;
pub(crate) mod workers;

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
