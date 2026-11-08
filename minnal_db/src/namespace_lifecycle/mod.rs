//! A namespace's lifecycle: its states, the governor that moves it between
//! them, and the maintenance records that keep its long-running tasks.
//!
//! ```text
//! Creating ──► Created ──► Deleting ──► (gone)
//! ```
//!
//! A namespace is visible from the moment its record is written (`Creating`),
//! but takes no requests until everything it needs exists (`Created`). A drop
//! moves it to `Deleting`, which refuses every request but the admin views
//! while its data is removed; its record goes last, the moment it is gone.
//!
//! This layer sits above the engine (`db`) and below the doc store, which owns
//! the namespaces: it keeps their states and does the work of each step,
//! through [`LifecycleHost`]. Nothing here imports the doc store.
//!
//! - `state`: [`NamespaceState`] and which requests each state takes.
//! - `governor`: one task that runs every creation and drop, supervised.
//! - `host`: [`LifecycleHost`], what the governor needs from the owner.
//! - `records`: the system namespace `system_namespace_maintenance`.
//! - `error`: [`LifecycleError`].

mod error;
mod governor;
mod host;
mod records;
mod state;

pub use error::LifecycleError;
pub(crate) use governor::Governor;
pub use governor::{LifecycleConfig, Task};
pub use host::LifecycleHost;
pub use records::{BuildStatus, FieldBuildRecord, MAINTENANCE_NAMESPACE};
pub(crate) use records::{MaintenanceStore, TaskKey};
pub use state::NamespaceState;
