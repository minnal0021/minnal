//! `JobRecord`: a job's record in `system_namespace_maintenance`, under its
//! [`JobKey`](super::JobKey).
//!
//! The record is the job: a live record (pending, running or cancelling) is
//! what puts its namespace in `IndexMaintenance`, and what a restart resumes.
//! Its `progress` belongs to the job's body (how far a build got, say), which
//! reads and writes it; the jobs registry never looks inside.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::namespace_lifecycle::BuildStatus;

/// A job's record. See the module docs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JobRecord {
    /// Where the job is: live while pending, running or cancelling.
    pub status: BuildStatus,
    /// What the job works on, in the owner's words (a field job's field
    /// name), for the admin views.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    /// Unix epoch milliseconds when the job started.
    pub started_at_ms: u64,
    /// Unix epoch milliseconds of the record's last change.
    pub updated_at_ms: u64,
    /// Why the job failed, when it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Whether the body got past the point a cancel can stop it
    /// ([`Jobs::refuse_cancel`](super::Jobs::refuse_cancel)): a cancel is then
    /// refused.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub past_cancel: bool,
    /// The job body's own progress, opaque here.
    #[serde(default)]
    pub progress: serde_json::Value,
}

impl JobRecord {
    /// A job starting now, its body's progress `progress` (what the body
    /// needs to begin, such as the index spec of a build).
    pub fn started(subject: Option<String>, progress: serde_json::Value) -> Self {
        let now = crate::support::now_ms();
        Self {
            status: BuildStatus::Running,
            subject,
            started_at_ms: now,
            updated_at_ms: now,
            error: None,
            past_cancel: false,
            progress,
        }
    }

    /// Whether the job is still to finish.
    pub fn is_live(&self) -> bool {
        self.status.is_live()
    }

    /// The body's progress as `P`; `None` before the body saved any, or if it
    /// saved another shape.
    pub fn progress<P: DeserializeOwned>(&self) -> Option<P> {
        serde_json::from_value(self.progress.clone()).ok()
    }
}
