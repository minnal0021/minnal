//! `BuildStatus`: where an index job is, as its record and the admin views
//! report it.

use serde::{Deserialize, Serialize};

/// Status of an index job (a field-index build, a reindex, …).
///
/// Stored as a snake_case string in JSON, in API responses and in the
/// maintenance records. A job is live while pending, running or cancelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuildStatus {
    /// Queued but not yet started.
    Pending,
    /// Actively building.
    Running,
    /// Finished successfully.
    Complete,
    /// Terminated with an error.
    Failed,
    /// Asked to stop; the job stops at its next batch and undoes what it did.
    Cancelling,
    /// Stopped by a cancel, its work undone.
    Cancelled,
}

impl BuildStatus {
    /// Returns `true` when the job has reached a terminal state.
    pub fn is_terminal(self) -> bool {
        matches!(self, BuildStatus::Complete | BuildStatus::Failed | BuildStatus::Cancelled)
    }

    /// Returns `true` while the job is still to finish.
    pub fn is_live(self) -> bool {
        !self.is_terminal()
    }
}
