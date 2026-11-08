//! Index jobs on the doc store's side: starting one and running it (through
//! the governor, or inline without one), which body each job runs, and the
//! admin views: a store's jobs by name, cancelling one, waiting for one.
//!
//! A job's name is its kind, and for a field job the field's name:
//! `field_build:status`, `reindex_all`.

use std::time::Duration;

use super::super::*;
use crate::namespace_lifecycle::{JobKind, JobRecord, Task};

impl DocStore {
    /// Whether a job started now can run: inline without the governor, or
    /// through it while it runs. Checked before a job is recorded, so a dead
    /// governor does not leave a store in `IndexMaintenance` with a job
    /// nobody runs.
    pub(in crate::doc_store) fn job_runner_ready(&self) -> Result<(), DocStoreError> {
        match self.governor.get() {
            Some(g) if !g.is_running() => Err(DocStoreError::GovernorUnavailable),
            _ => Ok(()),
        }
    }

    /// Record job `key` of `namespace` as started, its body to begin from
    /// `progress` (see `Jobs::start`). `subject` names what it works on.
    pub(in crate::doc_store) fn start_job(
        &self,
        namespace: &str,
        key: JobKey,
        subject: &str,
        progress: &impl serde::Serialize,
    ) -> Result<(), DocStoreError> {
        let progress = serde_json::to_value(progress)?;
        self.jobs.start(self, namespace, key, Some(subject.to_owned()), progress)
    }

    /// Save job `key`'s body's `progress`. A save that fails is logged and
    /// the job goes on: the record keeps an earlier point, which only costs
    /// work after a restart.
    pub(in crate::doc_store) fn save_job_progress(&self, key: JobKey, progress: &impl serde::Serialize) {
        let saved = serde_json::to_value(progress)
            .map_err(|e| e.to_string())
            .and_then(|p| self.jobs.save_progress(key, p).map_err(|e| e.to_string()));
        if let Err(e) = saved {
            log::warn!("job {}: could not save its progress: {e}", key.kind);
        }
    }

    /// Run started job `key` and wait for it to end: through the governor
    /// when it is started (supervised: run again after a failure), inline
    /// otherwise (once; its end is recorded here).
    pub(in crate::doc_store) async fn run_job(&self, namespace: &str, key: JobKey) -> Result<(), DocStoreError> {
        if let Some(governor) = self.governor.get() {
            return Ok(governor.run(namespace, Task::Job(key)).await?);
        }
        let outcome = self.job_body(namespace, key).await;
        self.jobs
            .finish(self, namespace, key, outcome.as_ref().map(|_| ()).map_err(ToString::to_string))?;
        outcome
    }

    /// Hand started job `key` to the governor, to run in the background.
    pub(in crate::doc_store) fn submit_job(&self, namespace: &str, key: JobKey) -> Result<(), DocStoreError> {
        self.running_governor()?.submit(namespace, Task::Job(key))?;
        Ok(())
    }

    /// The body of job `key`: what the governor runs for it. Each is
    /// idempotent and resumes from the job's record.
    pub(in crate::doc_store) async fn job_body(&self, namespace: &str, key: JobKey) -> Result<(), DocStoreError> {
        match key.kind {
            JobKind::FieldBuild(_) => self.field_build_job(namespace, key).await,
            JobKind::FieldDrop(_) => self.field_drop_job(namespace, key).await,
            JobKind::ReindexAll => self.reindex_all_job(namespace, key).await,
            JobKind::VectorEnable => self.vector_enable_job(namespace, key).await,
            JobKind::VectorCleanup => self.vector_cleanup_job(namespace, key).await,
            #[cfg(feature = "semantic-search")]
            JobKind::VectorReindex => self.vector_reindex_job(namespace, key).await,
            #[cfg(feature = "semantic-search")]
            JobKind::PartitionRebuild => self.partition_rebuild_job(namespace, key).await,
            #[cfg(not(feature = "semantic-search"))]
            JobKind::VectorReindex | JobKind::PartitionRebuild => Err(DocStoreError::SemanticSearchNotCompiled),
        }
    }

    /// Run every job a crash left live, inline and one after the other, and
    /// wait for them; each store's state is first put in step with its jobs'
    /// records. For a doc store without the governor, which does this itself
    /// when it starts.
    pub async fn run_pending_jobs(&self) -> Result<(), DocStoreError> {
        for (namespace, state) in self.schemas.states() {
            if !state.is_ready() {
                continue;
            }
            let Some(id) = self.schemas.get(&namespace).and_then(|s| s.ns_id()) else {
                continue;
            };
            for key in self.jobs.repair(self, &namespace, id)? {
                info!("resuming job {} of '{namespace}'", key.kind);
                self.run_job(&namespace, key).await?;
            }
        }
        Ok(())
    }

    // ── Admin views ────────────────────────────────────────────────────────

    /// `namespace`'s index jobs with a record, live or ended, by name.
    pub fn jobs(&self, namespace: &str) -> Result<Vec<(String, JobRecord)>, DocStoreError> {
        let Some(id) = self.store_id(namespace)? else {
            return Ok(Vec::new());
        };
        let fields = self.db.list_index_fields(id);
        Ok(self
            .jobs
            .list(id)?
            .into_iter()
            .map(|(key, record)| (job_name(&fields, key.kind), record))
            .collect())
    }

    /// Ask `namespace`'s live job `job` (named as [`jobs`](Self::jobs) names
    /// it) to stop: it stops at its next batch and undoes its work, and is
    /// recorded cancelled. Refused with [`DocStoreError::JobNotCancellable`]
    /// for a job that cannot be stopped part-way.
    pub fn cancel_job(&self, namespace: &str, job: &str) -> Result<(), DocStoreError> {
        let key = self.job_key(namespace, job)?;
        Ok(self.jobs.cancel(namespace, key)?)
    }

    /// Wait up to `timeout` for `namespace`'s job `job` to end, and return
    /// its record (still live if the time ran out).
    pub async fn wait_for_job(&self, namespace: &str, job: &str, timeout: Duration) -> Result<JobRecord, DocStoreError> {
        let key = self.job_key(namespace, job)?;
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let record = self.jobs.get(key)?.ok_or_else(|| DocStoreError::JobNotFound {
                namespace: namespace.to_owned(),
                job: job.to_owned(),
            })?;
            if !record.is_live() || tokio::time::Instant::now() >= deadline {
                return Ok(record);
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// `namespace`'s id, `None` for a store not given one yet (being
    /// created); `NotFound` for no store.
    fn store_id(&self, namespace: &str) -> Result<Option<crate::namespace::NamespaceId>, DocStoreError> {
        match self.schemas.get(namespace) {
            Some(schema) => Ok(schema.ns_id()),
            None => Err(DocStoreError::NotFound {
                namespace: namespace.to_owned(),
            }),
        }
    }

    /// The key of `namespace`'s job named `job`.
    fn job_key(&self, namespace: &str, job: &str) -> Result<JobKey, DocStoreError> {
        let not_found = || DocStoreError::JobNotFound {
            namespace: namespace.to_owned(),
            job: job.to_owned(),
        };
        let id = self.store_id(namespace)?.ok_or_else(not_found)?;
        let (name, field) = match job.split_once(':') {
            Some((name, field)) => (name, Some(field)),
            None => (job, None),
        };
        let field_id = match field {
            Some(field) => {
                let fields = self.db.list_index_fields(id);
                Some(fields.iter().find(|f| f.field_name == field).ok_or_else(not_found)?.field_id)
            }
            None => None,
        };
        let kind = match (name, field_id) {
            ("field_build", Some(f)) => JobKind::FieldBuild(f),
            ("field_drop", Some(f)) => JobKind::FieldDrop(f),
            ("reindex_all", None) => JobKind::ReindexAll,
            ("vector_enable", None) => JobKind::VectorEnable,
            ("vector_cleanup", None) => JobKind::VectorCleanup,
            ("vector_reindex", None) => JobKind::VectorReindex,
            ("partition_rebuild", None) => JobKind::PartitionRebuild,
            _ => return Err(not_found()),
        };
        Ok(JobKey::new(id, kind))
    }
}

/// A job's name: its kind, and for a field job the field's name (its id if
/// it has none).
fn job_name(fields: &[crate::field_index::FieldMeta], kind: JobKind) -> String {
    match kind.field() {
        Some(id) => match fields.iter().find(|f| f.field_id == id) {
            Some(meta) => format!("{}:{}", kind.name(), meta.field_name),
            None => kind.to_string(),
        },
        None => kind.name().to_owned(),
    }
}
