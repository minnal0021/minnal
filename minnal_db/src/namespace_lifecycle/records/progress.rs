//! The progress records of index tasks: what a field-index build saves, and
//! the status every index task reports.

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

/// One field-index build's progress, kept in the maintenance namespace
/// (key `JobKey::field_build`).
///
/// Written when the build is set up (before the schema lists the index), every
/// 1,000 documents while it runs, and when it ends. A restart resumes a
/// `Running` build from `resume_after`. Each write is WAL-backed and durable on
/// return, so a record is either there whole or not there.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldBuildRecord {
    /// `Running` until the build ends, then `Complete` or `Failed`.
    pub status: BuildStatus,
    /// Total document count (0 until the initial scan finishes).
    pub total: u64,
    /// Documents processed so far.
    pub indexed: u64,
    /// Hex-encoded key a restart resumes after. Never ahead of what the
    /// field index has made durable: the build names a key only once an index
    /// checkpoint covers it. `None`: start from the first key.
    pub resume_after_hex: Option<String>,
    /// Why the build failed, when `status == Failed`.
    pub error: Option<String>,
}

impl FieldBuildRecord {
    /// A build that has started and processed nothing yet.
    pub fn started() -> Self {
        Self {
            status: BuildStatus::Running,
            total: 0,
            indexed: 0,
            resume_after_hex: None,
            error: None,
        }
    }

    /// Whether the build is still to finish (running, or cut short by a crash).
    pub fn is_running(&self) -> bool {
        self.status == BuildStatus::Running
    }
}
