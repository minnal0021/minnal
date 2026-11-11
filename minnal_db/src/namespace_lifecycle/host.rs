//! `LifecycleHost`: what the governor and the jobs registry need from the
//! layer that owns the namespaces (the doc store): their states, the moves
//! between them, and the work of each task.

use std::future::Future;

use crate::namespace::NamespaceId;
use crate::namespace_lifecycle::{JobKind, LifecycleError, NamespaceState, Task};

/// The owner of a set of namespaces, as the lifecycle layer sees it. The
/// governor decides when a task runs and how often it is run again, and the
/// jobs registry which jobs may run; the host does the work, and keeps each
/// namespace's state.
pub trait LifecycleHost: Send + Sync + 'static {
    /// The host's error type; lifecycle errors become it.
    type Error: std::fmt::Display + Send + From<LifecycleError>;

    /// Every namespace, with its state.
    fn states(&self) -> Vec<(String, NamespaceState)>;

    /// `namespace`'s state, if it exists.
    fn state(&self, namespace: &str) -> Option<NamespaceState>;

    /// `namespace`'s id, once it has one (a namespace being created gets it
    /// part-way).
    fn namespace_id(&self, namespace: &str) -> Option<NamespaceId>;

    /// Move `namespace` from one of `from` to `to`, durably, and return the
    /// state it left. In any other state, the error
    /// [`NamespaceState::refusal`](crate::namespace_lifecycle::NamespaceState)
    /// gives. Moving to the state it is in, when listed in `from`, is a no-op
    /// that succeeds.
    fn transition(&self, namespace: &str, from: &[NamespaceState], to: NamespaceState) -> Result<NamespaceState, Self::Error>;

    /// How `namespace`'s job `kind` is named to people (in errors): by default
    /// its kind with the field's id; a host that knows field names uses them.
    fn job_name(&self, _namespace: &str, kind: JobKind) -> String {
        kind.to_string()
    }

    /// Whether `error`, returned by a run, ends its task at once: running it
    /// again cannot mend it ([`LifecycleError::Abandoned`] converted, for a
    /// host that maps it). By default every error is run again.
    fn is_final(_error: &Self::Error) -> bool {
        false
    }

    /// Run `namespace`'s `task` once. It must be idempotent: a run cut short
    /// (an error, a panic, a crash) is finished by running it again.
    fn run(&self, namespace: &str, task: Task) -> impl Future<Output = Result<(), Self::Error>> + Send;
}
