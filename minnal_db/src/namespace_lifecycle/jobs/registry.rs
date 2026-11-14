//! `Jobs`: starting, finishing and cancelling index jobs, and keeping each
//! namespace's state in step with them.
//!
//! **Order.** A start writes the job's record first, then moves the namespace
//! to `IndexMaintenance`; a finish marks the record ended, then moves the
//! namespace back to `Created` if no live job is left. A crash between the two
//! writes leaves a state that disagrees with the records, which
//! [`Jobs::repair`] puts right at the next start (the records win).
//!
//! **A lock per namespace** covers every start and finish, across both
//! writes: otherwise a finish could see no live job, a start then write its
//! record and find the namespace still in maintenance, and the finish move it
//! to `Created` with a job running. The locks are striped by namespace id, so
//! jobs of different namespaces seldom wait for each other's writes. Each is a
//! `parking_lot` lock, never held across an `.await`; audit records are
//! written after it is released.
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
use crate::namespace_lifecycle::{AuditEvent, AuditLog, BuildStatus, LifecycleError, LifecycleHost, MaintenanceStore, NamespaceState};

/// The index jobs of every namespace. See the module docs.
pub struct Jobs {
    records: Arc<MaintenanceStore>,
    /// Where every start, cancel and end is recorded.
    audit: Arc<AuditLog>,
    /// Held by every start and finish, across the record and the state
    /// write; one per stripe of namespace ids ([`Jobs::lock`]).
    locks: Box<[Mutex<()>]>,
}

/// Stripes of [`Jobs::locks`]: a store's operations take the lock at the
/// hash of its id modulo this.
///
/// The lock only keeps one store's start, finish, cancel and progress save
/// apart (nothing it guards spans two stores), so stores need not share one.
/// Striping keeps that without a map of per-store locks to look up, allocate
/// and prune as stores come and go. Two stores on one stripe only wait for
/// each other, as every store would with a single lock.
///
/// 64 is a judgment, not a measurement: with a few dozen stores running jobs
/// at once, most land on different stripes, and the locks cost a few hundred
/// bytes. Any value from about 16 up would serve; the cost of a job is its
/// fsyncs (and `MaintenanceStore`'s one writer), not this lock.
const LOCK_STRIPES: usize = 64;

impl Jobs {
    /// The jobs whose records `records` holds, their events recorded in
    /// `audit`.
    pub(crate) fn new(records: Arc<MaintenanceStore>, audit: Arc<AuditLog>) -> Self {
        Self {
            records,
            audit,
            locks: (0..LOCK_STRIPES).map(|_| Mutex::new(())).collect(),
        }
    }

    /// The lock of `namespace`'s jobs.
    fn lock(&self, namespace: NamespaceId) -> parking_lot::MutexGuard<'_, ()> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        namespace.hash(&mut hasher);
        self.locks[(hasher.finish() as usize) % LOCK_STRIPES].lock()
    }

    /// The lifecycle audit log the jobs record their events in.
    pub fn audit(&self) -> &AuditLog {
        &self.audit
    }

    /// Start job `key` on `namespace`: refused unless the namespace is ready
    /// and no live job of it conflicts; then its record is written (running,
    /// holding `progress`, what the body starts from) and the namespace moved
    /// to `IndexMaintenance`. The caller then hands the job to the governor.
    /// `subject` names what the job works on, for the admin views. A record
    /// of an ended job under the same key is replaced.
    pub fn start<H: LifecycleHost>(
        &self,
        host: &H,
        namespace: &str,
        key: JobKey,
        subject: Option<String>,
        progress: serde_json::Value,
    ) -> Result<(), H::Error> {
        {
            let _g = self.lock(key.namespace);
            self.check_start_locked(host, namespace, key)?;
            self.records.put(key, &JobRecord::started(subject.clone(), progress))?;
            let ready = [NamespaceState::Created, NamespaceState::IndexMaintenance];
            if let Err(e) = host.transition(namespace, &ready, NamespaceState::IndexMaintenance) {
                // A drop moved first: the job never started. (Any ended record
                // under the key went with it, which only shortens the history.)
                self.records.delete(key)?;
                return Err(e);
            }
        }
        self.audit
            .record(namespace, AuditEvent::JobStarted, Some(key.kind.name()), subject.as_deref(), None);
        Ok(())
    }

    /// Whether job `key` would be started now on `namespace`: the error
    /// [`start`](Self::start) would refuse it with, if any. For a caller with
    /// work to do before the start that a refusal would leave behind.
    pub fn check_start<H: LifecycleHost>(&self, host: &H, namespace: &str, key: JobKey) -> Result<(), H::Error> {
        let _g = self.lock(key.namespace);
        self.check_start_locked(host, namespace, key)
    }

    /// [`check_start`](Self::check_start), its caller holding the lock.
    fn check_start_locked<H: LifecycleHost>(&self, host: &H, namespace: &str, key: JobKey) -> Result<(), H::Error> {
        let state = host.state(namespace).ok_or_else(|| LifecycleError::NotFound {
            namespace: namespace.to_owned(),
        })?;
        state.admit(namespace)?;
        if let Some((running, _)) = self.live(key.namespace)?.into_iter().find(|(k, _)| conflicts(k.kind, key.kind)) {
            return Err(LifecycleError::JobConflict {
                namespace: namespace.to_owned(),
                job: host.job_name(namespace, key.kind),
                running: host.job_name(namespace, running.kind),
            }
            .into());
        }
        Ok(())
    }

    /// Take back job `key` of `namespace`, started but refused before any of
    /// its work ran (the request it belongs to failed a check that needed the
    /// job started first): its record is removed, as if it never started,
    /// and the namespace goes back to `Created` if no live job is left.
    pub fn withdraw<H: LifecycleHost>(&self, host: &H, namespace: &str, key: JobKey, reason: &str) -> Result<(), H::Error> {
        {
            let _g = self.lock(key.namespace);
            if self.records.get::<JobRecord>(key)?.is_some_and(|r| r.is_live()) {
                self.records.delete(key)?;
            }
            self.leave_if_idle(host, namespace, key.namespace)?;
        }
        self.audit
            .record(namespace, AuditEvent::JobWithdrawn, Some(key.kind.name()), None, Some(reason));
        Ok(())
    }

    /// End job `key` of `namespace`: `Ok` completes it (or, if a cancel was
    /// asked for, marks it cancelled), `Err` fails it with that error. Then,
    /// if no live job of the namespace is left, it goes back to `Created`.
    /// Ending a job with no record, or one already ended, only does the
    /// second part.
    pub fn finish<H: LifecycleHost>(&self, host: &H, namespace: &str, key: JobKey, outcome: Result<(), String>) -> Result<(), H::Error> {
        let ended = {
            let _g = self.lock(key.namespace);
            let ended = match self.records.get::<JobRecord>(key)? {
                Some(mut record) if record.is_live() => {
                    record.status = match (&outcome, record.status) {
                        (Ok(()), BuildStatus::Cancelling) => BuildStatus::Cancelled,
                        (Ok(()), _) => BuildStatus::Complete,
                        (Err(_), _) => BuildStatus::Failed,
                    };
                    record.error = outcome.err();
                    record.updated_at_ms = crate::support::now_ms();
                    self.records.put(key, &record)?;
                    Some(record)
                }
                _ => None,
            };
            self.leave_if_idle(host, namespace, key.namespace)?;
            ended
        };
        if let Some(record) = ended {
            let event = match record.status {
                BuildStatus::Cancelled => AuditEvent::JobCancelled,
                BuildStatus::Failed => AuditEvent::JobFailed,
                _ => AuditEvent::JobCompleted,
            };
            self.audit.record(
                namespace,
                event,
                Some(key.kind.name()),
                record.subject.as_deref(),
                record.error.as_deref(),
            );
        }
        Ok(())
    }

    /// Ask live job `key` to stop. Its body sees the request at its next batch
    /// ([`is_cancelling`](Self::is_cancelling)), undoes its work and returns,
    /// and the job ends cancelled.
    pub fn cancel(&self, namespace: &str, key: JobKey) -> Result<(), LifecycleError> {
        let subject = {
            let _g = self.lock(key.namespace);
            let job = key.kind.to_string();
            let Some(mut record) = self.records.get::<JobRecord>(key)?.filter(JobRecord::is_live) else {
                return Err(LifecycleError::JobNotFound {
                    namespace: namespace.to_owned(),
                    job,
                });
            };
            if !key.kind.cancellable() || record.past_cancel {
                return Err(LifecycleError::JobNotCancellable {
                    namespace: namespace.to_owned(),
                    job,
                });
            }
            record.status = BuildStatus::Cancelling;
            record.updated_at_ms = crate::support::now_ms();
            self.records.put(key, &record)?;
            record.subject
        };
        self.audit
            .record(namespace, AuditEvent::JobCancelRequested, Some(key.kind.name()), subject.as_deref(), None);
        Ok(())
    }

    /// Mark job `key` past the point a cancel can stop it: its body finishes
    /// from here. A cancel asked for already is taken back (the job then ends
    /// complete, not cancelled), and a later one is refused with
    /// [`LifecycleError::JobNotCancellable`].
    pub fn refuse_cancel(&self, key: JobKey) -> Result<(), LifecycleError> {
        let _g = self.lock(key.namespace);
        if let Some(mut record) = self.records.get::<JobRecord>(key)?.filter(JobRecord::is_live) {
            if record.status == BuildStatus::Cancelling {
                record.status = BuildStatus::Running;
            }
            record.past_cancel = true;
            record.updated_at_ms = crate::support::now_ms();
            self.records.put(key, &record)?;
        }
        Ok(())
    }

    /// Whether job `key` has been asked to stop. A memory read, for a job body
    /// to check between batches.
    pub fn is_cancelling(&self, key: JobKey) -> bool {
        matches!(self.records.status(key), Ok(Some(BuildStatus::Cancelling)))
    }

    /// Job `key`'s status, `None` with no record. A memory read that parses
    /// nothing, for the checks a query makes.
    pub fn status(&self, key: JobKey) -> Result<Option<BuildStatus>, LifecycleError> {
        self.records.status(key)
    }

    /// Job `key`'s record, live or ended.
    pub fn get(&self, key: JobKey) -> Result<Option<JobRecord>, LifecycleError> {
        self.records.get(key)
    }

    /// Save the body's `progress` in job `key`'s record, leaving its status
    /// alone (a cancel may have changed it meanwhile). Nothing to save to once
    /// the job has ended.
    pub fn save_progress(&self, key: JobKey, progress: serde_json::Value) -> Result<(), LifecycleError> {
        let _g = self.lock(key.namespace);
        let Some(mut record) = self.records.get::<JobRecord>(key)?.filter(JobRecord::is_live) else {
            return Ok(());
        };
        record.progress = progress;
        record.updated_at_ms = crate::support::now_ms();
        self.records.put(key, &record)
    }

    /// Delete job `key`'s record if the job has ended: its history goes, for a
    /// record that would describe something no longer there (the build of an
    /// index since dropped). A live record is kept.
    pub fn forget(&self, key: JobKey) -> Result<(), LifecycleError> {
        let _g = self.lock(key.namespace);
        if self.records.get::<JobRecord>(key)?.is_some_and(|r| !r.is_live()) {
            self.records.delete(key)?;
        }
        Ok(())
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
        let _g = self.lock(id);
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
