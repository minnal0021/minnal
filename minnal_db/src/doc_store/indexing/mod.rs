//! Field indexes: adding and dropping one (`build`, each an index job),
//! `reindex_all` (drop and rebuild them all, one job), and the progress
//! types the build jobs save and the progress views report.

mod build;
pub mod progress;
mod reindex_all;

pub(in crate::doc_store) use build::record_error;
