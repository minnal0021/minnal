//! Index jobs on the doc store's side: starting one and running it (through
//! the governor), which body each job runs, and the
//! admin views: a store's jobs by name, cancelling one, waiting for one.
//!
//! A job's name is its kind, and for a field job the field's name:
//! `field_build:status`, `reindex_all`.

use std::time::Duration;

use super::super::*;
use crate::namespace_lifecycle::{AuditEvent, AuditRecord, JobKind, JobRecord, Task};

impl DocStore {
    /// Whether a job started now can run: the governor is running. Checked
    /// before a job is recorded, so a dead governor does not leave a store in
    /// `IndexMaintenance` with a job nobody runs.
    pub(in crate::doc_store) fn job_runner_ready(&self) -> Result<(), DocStoreError> {
        self.running_governor().map(|_| ())
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

    /// Run started job `key` through the governor (supervised: run again
    /// after a failure) and wait for it to end.
    pub(in crate::doc_store) async fn run_job(&self, namespace: &str, key: JobKey) -> Result<(), DocStoreError> {
        Ok(self.running_governor()?.run(namespace, Task::Job(key)).await?)
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

    /// Test hook: hand every live job to the governor and wait for each, one
    /// after the other (a job it already runs is joined). For a test that
    /// recorded a job, or ran one part-way, without handing it over.
    #[cfg(test)]
    pub(crate) async fn run_live_jobs(&self) -> Result<(), DocStoreError> {
        for (namespace, state) in self.schemas.states() {
            if !state.is_ready() {
                continue;
            }
            let Some(id) = self.schemas.get(&namespace).and_then(|s| s.ns_id()) else {
                continue;
            };
            for key in self.jobs.repair(self, &namespace, id)? {
                self.run_job(&namespace, key).await?;
            }
        }
        Ok(())
    }

    /// Record a create or drop `event` of `namespace` in the audit log.
    pub(in crate::doc_store) fn audit(&self, namespace: &str, event: AuditEvent) {
        self.jobs.audit().record(namespace, event, None, None, None);
    }

    /// The lifecycle audit log, newest first: every create, drop and index job
    /// event, and every task the governor gave up on, of `namespace` only
    /// when given (a store dropped since too). At most `limit` records from
    /// `cursor` (a page's next cursor) on; returns the next page's cursor,
    /// `None` at the end.
    pub async fn lifecycle_audit(
        &self,
        namespace: Option<&str>,
        cursor: Option<Vec<u8>>,
        limit: usize,
    ) -> Result<(Vec<AuditRecord>, Option<Vec<u8>>), DocStoreError> {
        Ok(self.jobs.audit().read(namespace, cursor, limit).await?)
    }

    /// `namespace`'s lifecycle state, `None` for no store.
    pub fn store_state(&self, namespace: &str) -> Option<NamespaceState> {
        self.schemas.state(namespace)
    }

    /// Every store's live jobs, by store then name.
    pub fn live_jobs(&self) -> Vec<(String, String, JobRecord)> {
        let mut out = Vec::new();
        for (namespace, _) in self.schemas.states() {
            if let Ok(jobs) = self.jobs(&namespace) {
                out.extend(jobs.into_iter().filter(|(_, r)| r.is_live()).map(|(n, r)| (namespace.clone(), n, r)));
            }
        }
        out.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
        out
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
        use crate::namespace_lifecycle::LifecycleError;
        let key = self.job_key(namespace, job)?;
        // The registry names the job by its field's id; say it as asked.
        self.jobs.cancel(namespace, key).map_err(|e| match e {
            LifecycleError::JobNotFound { namespace, .. } => DocStoreError::JobNotFound {
                namespace,
                job: job.to_owned(),
            },
            LifecycleError::JobNotCancellable { namespace, .. } => DocStoreError::JobNotCancellable {
                namespace,
                job: job.to_owned(),
            },
            other => other.into(),
        })
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
pub(super) fn job_name(fields: &[crate::field_index::FieldMeta], kind: JobKind) -> String {
    match kind.field() {
        Some(id) => match fields.iter().find(|f| f.field_id == id) {
            Some(meta) => format!("{}:{}", kind.name(), meta.field_name),
            None => kind.to_string(),
        },
        None => kind.name().to_owned(),
    }
}
