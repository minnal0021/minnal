//! Building and tracking indexes: field-index lifecycle, reindex-all, and the
//! progress and handles shared by field-index and vector-index builds.

mod build;
pub mod build_manager;
pub mod observer;
pub mod progress;
mod reindex_all;
mod running;

pub(crate) use running::RunningBuilds;
