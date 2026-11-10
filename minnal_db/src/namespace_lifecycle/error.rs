//! `LifecycleError`: what a lifecycle step or a refused request reports.
//!
//! The layers above turn it into their own error type; the doc store maps each
//! variant onto a flat `DocStoreError` variant of the same meaning, so its
//! callers match one error type.

use crate::namespace_lifecycle::NamespaceState;

/// A lifecycle error. See the module docs.
#[derive(Debug, thiserror::Error)]
pub enum LifecycleError {
    /// The namespace is being created and takes no requests yet.
    #[error("namespace '{namespace}' is being created")]
    Creating {
        /// The namespace.
        namespace: String,
    },

    /// The namespace is being dropped.
    #[error("namespace '{namespace}' is being deleted")]
    Deleting {
        /// The namespace.
        namespace: String,
    },

    /// The namespace has index jobs running, so it cannot be dropped until
    /// they end or are cancelled.
    #[error("namespace '{namespace}' has index jobs running; wait for them to end, or cancel them, before dropping it")]
    InMaintenance {
        /// The namespace.
        namespace: String,
    },

    /// There is no namespace of that name.
    #[error("namespace '{namespace}' not found")]
    NotFound {
        /// The namespace.
        namespace: String,
    },

    /// An index job cannot start while another job of the namespace runs that
    /// touches the same index.
    #[error("namespace '{namespace}': job {job} cannot start while job {running} runs")]
    JobConflict {
        /// The namespace.
        namespace: String,
        /// The job asked for.
        job: String,
        /// The running job it conflicts with.
        running: String,
    },

    /// No live job of that name.
    #[error("namespace '{namespace}' has no running job {job}")]
    JobNotFound {
        /// The namespace.
        namespace: String,
        /// The job asked for.
        job: String,
    },

    /// The job cannot be cancelled: stopping it part-way would leave its index
    /// unusable.
    #[error("namespace '{namespace}': job {job} cannot be cancelled")]
    JobNotCancellable {
        /// The namespace.
        namespace: String,
        /// The job.
        job: String,
    },

    /// A move its state does not allow (a drop of a namespace that is not
    /// created, say).
    #[error("namespace '{namespace}' cannot move from {from:?} to {to:?}")]
    InvalidTransition {
        /// The namespace.
        namespace: String,
        /// Its state.
        from: NamespaceState,
        /// The state asked for.
        to: NamespaceState,
    },

    /// A lifecycle task failed every run the governor allows; the namespace is
    /// parked in its state until the process restarts.
    #[error("namespace '{namespace}': lifecycle task failed after {attempts} attempts: {error}")]
    TaskFailed {
        /// The namespace.
        namespace: String,
        /// Runs made.
        attempts: u32,
        /// The last run's error.
        error: String,
    },

    /// A task's run ended with an error a run again cannot mend: the
    /// governor ends the task with it at once, without its runs again.
    #[error("{reason}")]
    Abandoned {
        /// Why.
        reason: String,
    },

    /// The governor is not running (it stopped after too many restarts).
    #[error("the namespace lifecycle governor is not running; restart the process to create or drop namespaces")]
    GovernorUnavailable,

    /// The storage engine failed.
    #[error(transparent)]
    Db(#[from] crate::KVError),

    /// A maintenance record could not be encoded or decoded.
    #[error("maintenance record: {0}")]
    Record(String),
}
