//! The lifecycle audit log: one record per lifecycle event of any namespace
//! (created, dropped, an index job started or ended, work a restart picked up
//! again, a task given up on), in
//! the TTL namespace `system_lifecycle_audit`.
//!
//! It is what an operator reads to answer "what happened to this store, and
//! when": the namespace records go with their namespace, and the job records
//! keep only each job's latest run, but the audit log keeps every event until
//! it expires (30 days unless the server sets another retention), including
//! those of namespaces since dropped.
//!
//! Key: `u64::MAX - at_ms` then `u64::MAX - n` (both big-endian), where `n`
//! counts this process's records from a random start: a scan from the first
//! key reads the newest record first, and two records in one millisecond keep
//! their order. Value: the [`AuditRecord`] as JSON. Writes are WAL-backed.
//! A record that cannot be written is logged and lost: an audit failure never
//! fails the operation it describes.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::{AsyncDb, AsyncNamespace};

/// The TTL namespace holding the audit log.
pub const AUDIT_NAMESPACE: &str = "system_lifecycle_audit";

/// How long a record is kept unless the server sets another retention.
pub const DEFAULT_AUDIT_RETENTION: Duration = Duration::from_secs(30 * 24 * 3600);

/// Expired records the TTL worker deletes per pass.
const MAX_DELETES_PER_RUN: usize = 10_000;

/// Records a read scans at most, to fill one page of a filtered read.
const MAX_SCANNED: usize = 20_000;

/// The most records one read returns.
pub const MAX_AUDIT_LIMIT: usize = 1_000;

/// What happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditEvent {
    /// A create was accepted; the namespace is being set up.
    Creating,
    /// The namespace was set up and takes requests.
    Created,
    /// A drop was accepted; the namespace refuses requests.
    Deleting,
    /// The namespace and everything of it is gone.
    Deleted,
    /// An index job started.
    JobStarted,
    /// A cancel of a running job was asked for.
    JobCancelRequested,
    /// A job finished.
    JobCompleted,
    /// A job failed (after the governor's runs again); `detail` says why.
    JobFailed,
    /// A job stopped on a cancel, its work undone.
    JobCancelled,
    /// A creation, drop or job a restart found part-way was picked up again.
    Resumed,
    /// A creation or drop failed every run the governor allows; the namespace
    /// is left in its state until the process restarts.
    TaskParked,
    /// The governor itself failed too often and stopped; no namespace can be
    /// created, dropped or have a job until the process restarts.
    GovernorStopped,
}

/// One audit record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditRecord {
    /// When (Unix epoch milliseconds).
    pub at_ms: u64,
    /// The namespace (empty for [`AuditEvent::GovernorStopped`]).
    pub namespace: String,
    /// What happened.
    pub event: AuditEvent,
    /// The job (`field_build`, `vector_cleanup`, …), or the task (`creation`,
    /// `drop`) of a parked namespace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job: Option<String>,
    /// What the job works on, as its record names it (a field, say).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    /// An error or other detail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// The audit log. See the module docs.
pub struct AuditLog {
    ns: AsyncNamespace,
    next: AtomicU64,
}

impl AuditLog {
    /// Open the log (creating its namespace in a new database), its records
    /// kept for `retention`.
    pub(crate) async fn open(db: &AsyncDb, retention: Duration) -> Result<Self, crate::KVError> {
        let ns = db.namespace_with_ttl(AUDIT_NAMESPACE.to_owned(), retention, MAX_DELETES_PER_RUN).await?;
        let start = u64::from_le_bytes(crate::namespace::NamespaceId::random()?.to_bytes()[..8].try_into().expect("8 bytes")) >> 1;
        Ok(Self {
            ns,
            next: AtomicU64::new(start),
        })
    }

    /// Keep records for `retention` from now on (the server's setting). Records
    /// already written expire by the retention they were written with.
    pub(crate) async fn set_retention(db: &AsyncDb, retention: Duration) -> Result<(), crate::KVError> {
        db.namespace_with_ttl(AUDIT_NAMESPACE.to_owned(), retention, MAX_DELETES_PER_RUN)
            .await
            .map(|_| ())
    }

    /// Record `event` of `namespace`. Never fails: a record that cannot be
    /// written is logged.
    pub fn record(&self, namespace: &str, event: AuditEvent, job: Option<&str>, subject: Option<&str>, detail: Option<&str>) {
        let record = AuditRecord {
            at_ms: crate::support::now_ms(),
            namespace: namespace.to_owned(),
            event,
            job: job.map(str::to_owned),
            subject: subject.map(str::to_owned),
            detail: detail.map(str::to_owned),
        };
        let mut key = Vec::with_capacity(16);
        key.extend_from_slice(&(u64::MAX - record.at_ms).to_be_bytes());
        key.extend_from_slice(&(u64::MAX - self.next.fetch_add(1, Ordering::Relaxed)).to_be_bytes());
        let written = serde_json::to_vec(&record)
            .map_err(|e| crate::KVError::Io(std::io::Error::other(e)))
            .and_then(|value| crate::support::run_blocking_pass(|| self.ns.put_blocking(&key, &value)));
        if let Err(e) = written {
            log::warn!("{AUDIT_NAMESPACE}: could not record {event:?} of '{namespace}': {e}");
        }
    }

    /// Records newest first, of `namespace` only when given: at most `limit`
    /// (capped at [`MAX_AUDIT_LIMIT`]), from `cursor` (a page's `next`) on.
    /// Returns the cursor of the next page, `None` at the end. A filtered read
    /// scans at most a bounded number of records, so a page can come back
    /// short with a cursor to go on from.
    pub async fn read(
        &self,
        namespace: Option<&str>,
        cursor: Option<Vec<u8>>,
        limit: usize,
    ) -> Result<(Vec<AuditRecord>, Option<Vec<u8>>), crate::KVError> {
        let limit = limit.clamp(1, MAX_AUDIT_LIMIT);
        let mut out = Vec::new();
        let mut cursor = cursor;
        let mut scanned = 0;
        while out.len() < limit && scanned < MAX_SCANNED {
            let (page, next) = self.ns.scan(cursor.take(), None, limit - out.len()).await?;
            scanned += page.len();
            for (_, value) in page {
                // A record that does not decode is skipped: one bad record
                // must not hide the rest.
                if let Ok(r) = serde_json::from_slice::<AuditRecord>(&value)
                    && namespace.is_none_or(|n| r.namespace == n)
                {
                    out.push(r);
                }
            }
            match next {
                Some(c) => cursor = Some(c),
                None => return Ok((out, None)),
            }
        }
        Ok((out, cursor))
    }
}
