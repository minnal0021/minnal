//! The doc store as the lifecycle governor's host: its stores' states come
//! from their records, and their tasks are `provision` and `teardown`. Also
//! the `DocStore` methods that start the governor and hand it tasks.

use std::sync::Arc;

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

    async fn run(&self, namespace: &str, task: Task) -> Result<(), DocStoreError> {
        match task {
            Task::Provision => self.provision(namespace).await,
            Task::Teardown => self.teardown(namespace).await,
            // No job is started by this store yet, so none is left to resume.
            Task::Job(key) => Err(DocStoreError::Io(std::io::Error::other(format!("no body for index job {}", key.kind)))),
        }
    }
}

impl DocStore {
    /// Start the lifecycle governor, which from now on runs every store's
    /// creation and drop in the background, supervised (see `namespace_lifecycle`),
    /// and finishes those a crash left part-way. Call once, after the store is
    /// configured; a second call does nothing.
    ///
    /// Without it, [`create`](Self::create) and [`remove`](Self::remove) run
    /// their tasks inline (once, unsupervised), and the `begin_*` forms fail
    /// with [`DocStoreError::GovernorUnavailable`].
    pub fn start_lifecycle(self: &Arc<Self>, config: LifecycleConfig) {
        self.governor
            .get_or_init(|| Governor::start(Arc::downgrade(self), Arc::clone(&self.jobs), config));
    }

    /// The governor, if it is started and running.
    pub(in crate::doc_store) fn running_governor(&self) -> Result<&Governor, DocStoreError> {
        self.governor.get().filter(|g| g.is_running()).ok_or(DocStoreError::GovernorUnavailable)
    }

    /// Run `namespace`'s lifecycle `task` and wait for it: through the
    /// governor when it is started, inline otherwise.
    pub(in crate::doc_store) async fn run_lifecycle_task(&self, namespace: &str, task: Task) -> Result<(), DocStoreError> {
        match self.governor.get() {
            Some(governor) => Ok(governor.run(namespace, task).await?),
            None => LifecycleHost::run(self, namespace, task).await,
        }
    }

    /// Stores whose lifecycle task the governor gave up on, with the last
    /// error. They stay in their state until the process restarts.
    pub fn lifecycle_parked(&self) -> Vec<(String, String)> {
        self.governor.get().map(Governor::parked).unwrap_or_default()
    }

    /// Whether the lifecycle governor is started and running.
    pub fn lifecycle_running(&self) -> bool {
        self.running_governor().is_ok()
    }
}
