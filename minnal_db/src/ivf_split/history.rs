//! A record of what a namespace's partition looked like over time (FR-007).
//!
//! Code health (`?codes=true`) and the recall check each read every chunk code
//! or search many times, so they are too costly to compute on every request.
//! Each run is recorded here instead, with finished rebuilds, in one system-wide
//! TTL namespace, so the latest result and the trend (drift and overlap before
//! and after a rebuild, say) can be read back at any time without running
//! anything.
//!
//! Key: `namespace ‖ 0x00 ‖ recorded_at_ms (u64 BE) ‖ kind (u8)`, so a prefix scan
//! returns one namespace's records oldest first. Value: the record as JSON.
//! Writes are WAL-backed: they are rare and small, and a history that loses its
//! newest entry to a crash would misreport the state an operator acted on.
//! Records expire after a TTL the server sets (`[semantic_search]
//! partition_history_ttl_secs`, default [`DEFAULT_HISTORY_TTL`]). Dropping a namespace leaves its records
//! to expire; a namespace re-created under the same name shows them until then,
//! each stamped with when it was taken.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::AsyncDb;

/// The TTL namespace holding every namespace's records.
pub const HISTORY_NS: &str = "system_partition_history";

/// How long a record is kept unless configured otherwise: 30 days.
pub const DEFAULT_HISTORY_TTL: Duration = Duration::from_secs(30 * 86_400);

/// Most records the TTL worker deletes per pass.
const HISTORY_MAX_DELETES: usize = 10_000;

/// The most records one read returns.
pub const MAX_HISTORY_LIMIT: usize = 1_000;

/// What a record holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordKind {
    /// A partition summary with code metrics ([`super::health::PartitionHealth`]).
    Codes,
    /// A recall check ([`super::recall_check::RecallCheck`]).
    RecallCheck,
    /// A finished or failed rebuild ([`super::rebuild::RebuildStatus`]).
    Rebuild,
}

impl RecordKind {
    fn tag(self) -> u8 {
        match self {
            Self::Codes => 1,
            Self::RecallCheck => 2,
            Self::Rebuild => 3,
        }
    }
}

/// One record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoryRecord {
    /// When it was recorded (Unix ms).
    pub recorded_at_ms: u64,
    /// What it holds.
    pub kind: RecordKind,
    /// The result, as the endpoint that produced it returned it.
    pub data: serde_json::Value,
}

fn prefix(namespace: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(namespace.len() + 1);
    k.extend_from_slice(namespace.as_bytes());
    k.push(0);
    k
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

/// Record `data` for `namespace`, to expire after `ttl`. Callers treat a failure
/// as a lost record, never as a failure of what they measured.
pub async fn record(db: &AsyncDb, namespace: &str, kind: RecordKind, data: &impl Serialize, ttl: Duration) -> Result<(), crate::KVError> {
    let value = serde_json::to_vec(&HistoryRecord {
        recorded_at_ms: now_ms(),
        kind,
        data: serde_json::to_value(data).map_err(|e| crate::KVError::Io(std::io::Error::other(e)))?,
    })
    .map_err(|e| crate::KVError::Io(std::io::Error::other(e)))?;
    let mut key = prefix(namespace);
    key.extend_from_slice(&now_ms().to_be_bytes());
    key.push(kind.tag());
    db.namespace_with_ttl(HISTORY_NS.to_string(), ttl, HISTORY_MAX_DELETES)
        .await?
        .put(key, value)
        .await
}

/// `namespace`'s records, newest first: at most `limit` (capped at
/// [`MAX_HISTORY_LIMIT`]), of `kind` only when given. `ttl` is the namespace's
/// TTL, as for [`record`] (a different one would re-register it).
pub async fn read(
    db: &AsyncDb,
    namespace: &str,
    kind: Option<RecordKind>,
    limit: usize,
    ttl: Duration,
) -> Result<Vec<HistoryRecord>, crate::KVError> {
    let rows = db
        .namespace_with_ttl(HISTORY_NS.to_string(), ttl, HISTORY_MAX_DELETES)
        .await?
        .scan_prefix(prefix(namespace))
        .await?;
    let mut out = Vec::new();
    for (k, v) in rows.into_iter().rev() {
        if out.len() >= limit.min(MAX_HISTORY_LIMIT) {
            break;
        }
        if kind.is_some_and(|want| k.last() != Some(&want.tag())) {
            continue;
        }
        // A record that does not decode is skipped: the history is a convenience,
        // and one bad record must not hide the rest.
        if let Ok(r) = serde_json::from_slice::<HistoryRecord>(&v) {
            out.push(r);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn records_read_back_newest_first_by_namespace_and_kind() {
        let dir = tempfile::TempDir::new().unwrap();
        let db = AsyncDb::open_with_config(dir.path().to_owned(), crate::support::test_db_config())
            .await
            .unwrap();
        record(&db, "a", RecordKind::Codes, &serde_json::json!({"n": 1}), DEFAULT_HISTORY_TTL)
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(3)).await;
        record(&db, "a", RecordKind::RecallCheck, &serde_json::json!({"n": 2}), DEFAULT_HISTORY_TTL)
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(3)).await;
        record(&db, "a", RecordKind::Codes, &serde_json::json!({"n": 3}), DEFAULT_HISTORY_TTL)
            .await
            .unwrap();
        // A namespace whose name extends another's is not mixed in.
        record(&db, "ab", RecordKind::Codes, &serde_json::json!({"n": 9}), DEFAULT_HISTORY_TTL)
            .await
            .unwrap();
        let all = read(&db, "a", None, 10, DEFAULT_HISTORY_TTL).await.unwrap();
        let ns: Vec<i64> = all.iter().map(|r| r.data["n"].as_i64().unwrap()).collect();
        assert_eq!(ns, [3, 2, 1]);
        let codes = read(&db, "a", Some(RecordKind::Codes), 1, DEFAULT_HISTORY_TTL).await.unwrap();
        assert_eq!((codes.len(), codes[0].data["n"].as_i64()), (1, Some(3)));
        assert!(read(&db, "none", None, 10, DEFAULT_HISTORY_TTL).await.unwrap().is_empty());
        db.shutdown().await.unwrap();
    }
}
