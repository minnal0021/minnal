//! Index jobs: the operator-started, bounded work on a namespace's indexes
//! (build or drop a field index, reindex every field, turn semantic search on
//! or off, reindex or re-cluster the vector index).
//!
//! Each job has a record in `system_namespace_maintenance` ([`JobRecord`],
//! under its [`JobKey`]), written before the job starts, and the governor
//! runs it as a lifecycle task ([`Task::Job`](super::Task::Job)), supervised
//! like a creation or a drop. While a namespace has a live job it is in
//! `IndexMaintenance`: it takes every request, but cannot be dropped. The job
//! bodies are the owner's (the doc store's); this module only decides which
//! jobs may run and keeps the state in step.
//!
//! - `key`: [`JobKey`] and [`JobKind`].
//! - `record`: [`JobRecord`].
//! - `status`: [`BuildStatus`].
//! - `conflicts`: which jobs cannot run together.
//! - `registry`: [`Jobs`], which starts, finishes and cancels them.

mod conflicts;
mod key;
mod record;
mod registry;
mod status;

pub use conflicts::conflicts;
pub use key::{JobKey, JobKind};
pub use record::JobRecord;
pub use registry::Jobs;
pub use status::BuildStatus;
