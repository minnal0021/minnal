//! `LifecycleHost`: what the governor needs from the layer that owns the
//! namespaces (the doc store): their states, and the tasks to run.

use std::future::Future;

use crate::namespace_lifecycle::{NamespaceState, Task};

/// The owner of a set of namespaces, as the governor sees it. The governor
/// decides when a task runs and how often it is run again; the host does the
/// work, and keeps each namespace's state.
pub trait LifecycleHost: Send + Sync + 'static {
    /// Why a task failed.
    type Error: std::fmt::Display + Send;

    /// Every namespace, with its state.
    fn states(&self) -> Vec<(String, NamespaceState)>;

    /// `namespace`'s state, if it exists.
    fn state(&self, namespace: &str) -> Option<NamespaceState>;

    /// Run `namespace`'s `task` once. It must be idempotent: a run cut short
    /// (an error, a panic, a crash) is finished by running it again.
    fn run(&self, namespace: &str, task: Task) -> impl Future<Output = Result<(), Self::Error>> + Send;
}
