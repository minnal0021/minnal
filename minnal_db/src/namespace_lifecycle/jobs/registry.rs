//! `Jobs`: starting, finishing and cancelling index jobs, and keeping each
//! namespace's state in step with them.
//!
//! **Order.** A start writes the job's record first, then moves the namespace
//! to `IndexMaintenance`; a finish marks the record ended, then moves the
//! namespace back to `Created` if no live job is left. A crash between the two
//! writes leaves a state that disagrees with the records, which
//! [`Jobs::repair`] puts right at the next start (the records win).
//!
//! **One lock** covers every start and finish, across both writes: otherwise a
//! finish could see no live job, a start then write its record and find the
//! namespace still in maintenance, and the finish move it to `Created` with a
//! job running. It is a `parking_lot` lock, never held across an `.await`.
//!
//! A drop and a start race through the state alone: a drop moves `Created` to
//! `Deleting`, a start `Created` (or `IndexMaintenance`) to
//! `IndexMaintenance`; whichever moves first wins, and the other is refused.

use std::sync::Arc;

use parking_lot::Mutex;

use super::conflicts::conflicts;
use super::key::JobKey;
use super::record::JobRecord;
use crate::namespace::NamespaceId;
use crate::namespace_lifecycle::{BuildStatus, LifecycleError, LifecycleHost, MaintenanceStore, NamespaceState};

/// The index jobs of every namespace. See the module docs.
pub struct Jobs {
    records: Arc<MaintenanceStore>,
    /// Held by every start and finish, across the record and the state write.
    lock: Mutex<()>,
}

impl Jobs {
    /// The jobs whose records `records` holds.
    pub(crate) fn new(records: Arc<MaintenanceStore>) -> Self {
        Self {
            records,
            lock: Mutex::new(()),
        }
    }

    /// Start job `key` on `namespace`: refused unless the namespace is ready
    /// and no live job of it conflicts; then its record is written (running)
    /// and the namespace moved to `IndexMaintenance`. The caller then hands
    /// the job to the governor. `subject` names what the job works on, for the
    /// admin views.
    pub fn start<H: LifecycleHost>(&self, host: &H, namespace: &str, key: JobKey, subject: Option<String>) -> Result<(), H::Error> {
        let _g = self.lock.lock();
        let state = host.state(namespace).ok_or_else(|| LifecycleError::NotFound {
            namespace: namespace.to_owned(),
        })?;
        state.admit(namespace)?;
        if let Some((running, _)) = self.live(key.namespace)?.into_iter().find(|(k, _)| conflicts(k.kind, key.kind)) {
            return Err(LifecycleError::JobConflict {
                namespace: namespace.to_owned(),
                job: key.kind.to_string(),
                running: running.kind.to_string(),
            }
            .into());
        }
        self.records.put(key, &JobRecord::started(subject))?;
        let ready = [NamespaceState::Created, NamespaceState::IndexMaintenance];
        if let Err(e) = host.transition(namespace, &ready, NamespaceState::IndexMaintenance) {
            // A drop moved first: the job never started.
            self.records.delete(key)?;
            return Err(e);
        }
        Ok(())
    }

    /// End job `key` of `namespace`: `Ok` completes it (or, if a cancel was
    /// asked for, marks it cancelled), `Err` fails it with that error. Then,
    /// if no live job of the namespace is left, it goes back to `Created`.
    /// Ending a job with no record, or one already ended, only does the
    /// second part.
    pub fn finish<H: LifecycleHost>(&self, host: &H, namespace: &str, key: JobKey, outcome: Result<(), String>) -> Result<(), H::Error> {
        let _g = self.lock.lock();
        if let Some(mut record) = self.records.get::<JobRecord>(key)?
            && record.is_live()
        {
            record.status = match (&outcome, record.status) {
                (Ok(()), BuildStatus::Cancelling) => BuildStatus::Cancelled,
                (Ok(()), _) => BuildStatus::Complete,
                (Err(_), _) => BuildStatus::Failed,
            };
            record.error = outcome.err();
            record.updated_at_ms = crate::support::now_ms();
            self.records.put(key, &record)?;
        }
        self.leave_if_idle(host, namespace, key.namespace)
    }

    /// Ask live job `key` to stop. Its body sees the request at its next batch
    /// ([`is_cancelling`](Self::is_cancelling)), undoes its work and returns,
    /// and the job ends cancelled.
    pub fn cancel(&self, namespace: &str, key: JobKey) -> Result<(), LifecycleError> {
        let _g = self.lock.lock();
        let job = key.kind.to_string();
        let Some(mut record) = self.records.get::<JobRecord>(key)?.filter(JobRecord::is_live) else {
            return Err(LifecycleError::JobNotFound {
                namespace: namespace.to_owned(),
                job,
            });
        };
        if !key.kind.cancellable() {
            return Err(LifecycleError::JobNotCancellable {
                namespace: namespace.to_owned(),
                job,
            });
        }
        record.status = BuildStatus::Cancelling;
        record.updated_at_ms = crate::support::now_ms();
        self.records.put(key, &record)
    }

    /// Whether job `key` has been asked to stop. A memory read, for a job body
    /// to check between batches.
    pub fn is_cancelling(&self, key: JobKey) -> bool {
        matches!(self.records.get::<JobRecord>(key), Ok(Some(r)) if r.status == BuildStatus::Cancelling)
    }

    /// Job `key`'s record, live or ended.
    pub fn get(&self, key: JobKey) -> Result<Option<JobRecord>, LifecycleError> {
        self.records.get(key)
    }

    /// Save the body's `progress` in job `key`'s record, leaving its status
    /// alone (a cancel may have changed it meanwhile). Nothing to save to once
    /// the job has ended.
    pub fn save_progress(&self, key: JobKey, progress: serde_json::Value) -> Result<(), LifecycleError> {
        let _g = self.lock.lock();
        let Some(mut record) = self.records.get::<JobRecord>(key)?.filter(JobRecord::is_live) else {
            return Ok(());
        };
        record.progress = progress;
        record.updated_at_ms = crate::support::now_ms();
        self.records.put(key, &record)
    }

    /// Every job of `namespace` with a record, live or ended, in key order.
    pub fn list(&self, namespace: NamespaceId) -> Result<Vec<(JobKey, JobRecord)>, LifecycleError> {
        self.records
            .keys_of(namespace)
            .into_iter()
            .filter_map(|k| self.records.get::<JobRecord>(k).transpose().map(|r| r.map(|r| (k, r))))
            .collect()
    }

    /// The live jobs of `namespace`, in key order.
    pub fn live(&self, namespace: NamespaceId) -> Result<Vec<(JobKey, JobRecord)>, LifecycleError> {
        Ok(self.list(namespace)?.into_iter().filter(|(_, r)| r.is_live()).collect())
    }

    /// Put `namespace`'s state in step with its records after a restart, and
    /// return its live jobs, to be resumed: in maintenance with none, it goes
    /// back to `Created`; created with some, it goes to `IndexMaintenance`.
    pub fn repair<H: LifecycleHost>(&self, host: &H, namespace: &str, id: NamespaceId) -> Result<Vec<JobKey>, H::Error> {
        let _g = self.lock.lock();
        let live: Vec<JobKey> = self.live(id)?.into_iter().map(|(k, _)| k).collect();
        match (host.state(namespace), live.is_empty()) {
            (Some(NamespaceState::IndexMaintenance), true) => {
                host.transition(namespace, &[NamespaceState::IndexMaintenance], NamespaceState::Created)?;
            }
            (Some(NamespaceState::Created), false) => {
                host.transition(namespace, &[NamespaceState::Created], NamespaceState::IndexMaintenance)?;
            }
            _ => {}
        }
        Ok(live)
    }

    /// Move `namespace` back to `Created` if it is in maintenance with no live
    /// job. Caller holds `lock`.
    fn leave_if_idle<H: LifecycleHost>(&self, host: &H, namespace: &str, id: NamespaceId) -> Result<(), H::Error> {
        if host.state(namespace) == Some(NamespaceState::IndexMaintenance) && self.live(id)?.is_empty() {
            host.transition(namespace, &[NamespaceState::IndexMaintenance], NamespaceState::Created)?;
        }
        Ok(())
    }
}
