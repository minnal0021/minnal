//! Maintenance tasks and their progress, kept in the system namespace
//! `system_namespace_maintenance`.
//!
//! A long-running task on a store (a field-index build, a `reindex-all`, a
//! vector reindex) writes its progress here, and a restart finds it here. Each
//! record is a WAL-backed write, durable when the call returns and never torn,
//! so there is no file to sync and no half-written record to recover from.
//!
//! `key` names a record: the store's [`NamespaceId`](crate::NamespaceId), then
//! the task. `store` owns the namespace: nothing else reads or writes it.

mod key;
mod store;

pub(crate) use key::TaskKey;
pub use store::MAINTENANCE_NAMESPACE;
pub(crate) use store::MaintenanceStore;
