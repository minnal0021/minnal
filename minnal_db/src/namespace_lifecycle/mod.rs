//! A namespace's lifecycle: its states, the governor that moves it between
//! them, and the maintenance records that keep its long-running tasks.
//!
//! ```text
//! Creating ──► Created ⇄ IndexMaintenance
//!                 │
//!                 └──► Deleting ──► (gone)
//! ```
//!
//! A namespace is visible from the moment its record is written (`Creating`),
//! but takes no requests until everything it needs exists (`Created`). While
//! an index job runs it is in `IndexMaintenance`, and cannot be dropped. A
//! drop moves it to `Deleting`, which refuses every request but the admin
//! views while its data is removed; its record goes last, the moment it is
//! gone.
//!
//! This layer sits above the engine (`db`) and below the doc store, which owns
//! the namespaces: it keeps their states and does the work of each step,
//! through [`LifecycleHost`]. Nothing here imports the doc store.
//!
//! - `state`: [`NamespaceState`] and which requests each state takes.
//! - `governor`: one task that runs every creation, drop and index job,
//!   supervised.
//! - `jobs`: index jobs: their keys, records and conflicts, and [`Jobs`].
//! - `host`: [`LifecycleHost`], what the governor needs from the owner.
//! - `records`: the system namespace `system_namespace_maintenance`.
//! - `audit`: the lifecycle audit log, `system_lifecycle_audit`.
//! - `error`: [`LifecycleError`].

mod audit;
mod error;
mod governor;
mod host;
mod jobs;
mod records;
mod state;
#[cfg(test)]
mod tests;

pub use audit::{AUDIT_NAMESPACE, AuditEvent, AuditLog, AuditRecord, DEFAULT_AUDIT_RETENTION, MAX_AUDIT_LIMIT};
pub use error::LifecycleError;
pub(crate) use governor::Governor;
pub use governor::{LifecycleConfig, Task};
pub use host::LifecycleHost;
pub use jobs::{BuildStatus, JobKey, JobKind, JobRecord, Jobs, conflicts};
pub use records::MAINTENANCE_NAMESPACE;
pub(crate) use records::MaintenanceStore;
pub use state::NamespaceState;
