//! The doc store as the lifecycle governor's host: its stores' states come
//! from their records, and their tasks are `provision`, `teardown` and the
//! index jobs' bodies (`jobs.rs`). Also
//! the `DocStore` methods that hand the governor tasks.

use super::super::*;
use crate::namespace::NamespaceId;
use crate::namespace_lifecycle::{Governor, LifecycleHost, Task};

impl LifecycleHost for DocStore {
    type Error = DocStoreError;

    fn states(&self) -> Vec<(String, NamespaceState)> {
        self.schemas.states()
    }

    fn state(&self, namespace: &str) -> Option<NamespaceState> {
        self.schemas.state(namespace)
    }

    fn namespace_id(&self, namespace: &str) -> Option<NamespaceId> {
        self.schemas.get(namespace).and_then(|s| s.ns_id())
    }

    fn transition(&self, namespace: &str, from: &[NamespaceState], to: NamespaceState) -> Result<NamespaceState, DocStoreError> {
        self.schemas.transition(namespace, from, to)
    }

    fn job_name(&self, namespace: &str, kind: crate::namespace_lifecycle::JobKind) -> String {
        match self.schemas.get(namespace).and_then(|s| s.ns_id()) {
            Some(id) => super::jobs::job_name(&self.db.list_index_fields(id), kind),
            None => kind.to_string(),
        }
    }

    fn is_final(error: &DocStoreError) -> bool {
        matches!(error, DocStoreError::LifecycleTaskAbandoned(_))
    }

    async fn run(&self, namespace: &str, task: Task) -> Result<(), DocStoreError> {
        match task {
            Task::Provision => self.provision(namespace).await,
            Task::Teardown => self.teardown(namespace).await,
            Task::Job(key) => self.job_body(namespace, key).await,
        }
    }
}

impl DocStore {
    /// The governor, if it is running (it stops only after failing too often).
    pub(in crate::doc_store) fn running_governor(&self) -> Result<&Governor, DocStoreError> {
        self.governor.get().filter(|g| g.is_running()).ok_or(DocStoreError::GovernorUnavailable)
    }

    /// Stores whose lifecycle task the governor gave up on, with the last
    /// error. They stay in their state until the process restarts.
    pub fn lifecycle_parked(&self) -> Vec<(String, String)> {
        self.governor.get().map(Governor::parked).unwrap_or_default()
    }
}
