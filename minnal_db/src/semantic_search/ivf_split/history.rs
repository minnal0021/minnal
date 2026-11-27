//! A record of what a namespace's partition looked like over time.
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
//! partition_history_ttl_secs`, default [`crate::defaults::semantic_search::PARTITION_HISTORY_TTL`]). Dropping a namespace leaves its records
//! to expire; a namespace re-created under the same name shows them until then,
//! each stamped with when it was taken.

use std::collections::VecDeque;
use std::time::Duration;

use crate::support::now_ms;

use serde::{Deserialize, Serialize};

use crate::AsyncDb;

/// The TTL namespace holding every namespace's records.
pub const HISTORY_NS: &str = "system_partition_history";

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
///
/// The keys are read a page at a time, oldest first, keeping only the newest
/// `limit` that match; then only those records' values are read. Memory stays at
/// one page plus `limit` records however long the history is.
pub async fn read(
    db: &AsyncDb,
    namespace: &str,
    kind: Option<RecordKind>,
    limit: usize,
    ttl: Duration,
) -> Result<Vec<HistoryRecord>, crate::KVError> {
    let limit = limit.min(MAX_HISTORY_LIMIT);
    if limit == 0 {
        return Ok(Vec::new());
    }
    let ns = db.namespace_with_ttl(HISTORY_NS.to_string(), ttl, HISTORY_MAX_DELETES).await?;
    let start = prefix(namespace);
    // `namespace ‖ 0x01`: the first key past every `namespace ‖ 0x00 ‖ …` key.
    let mut end = start.clone();
    *end.last_mut().expect("the prefix ends with 0x00") = 1;
    let mut newest = VecDeque::with_capacity(limit);
    let mut cursor = Some(start);
    loop {
        let (keys, next) = ns.key_page(cursor, Some(end.clone()), crate::kv_store::KEY_PAGE).await?;
        for k in keys {
            if kind.is_some_and(|want| k.last() != Some(&want.tag())) {
                continue;
            }
            if newest.len() == limit {
                newest.pop_front();
            }
            newest.push_back(k);
        }
        match next {
            Some(n) => cursor = Some(n),
            None => break,
        }
    }
    let values = ns.get_multiple(newest.into_iter().rev().collect()).await?;
    // A record that expired since its key was read is gone; one that does not
    // decode is skipped: the history is a convenience, and one bad record must
    // not hide the rest.
    Ok(values
        .into_iter()
        .flatten()
        .filter_map(|v| serde_json::from_slice::<HistoryRecord>(&v).ok())
        .collect())
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
        record(
            &db,
            "a",
            RecordKind::Codes,
            &serde_json::json!({"n": 1}),
            crate::defaults::semantic_search::PARTITION_HISTORY_TTL,
        )
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(3)).await;
        record(
            &db,
            "a",
            RecordKind::RecallCheck,
            &serde_json::json!({"n": 2}),
            crate::defaults::semantic_search::PARTITION_HISTORY_TTL,
        )
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(3)).await;
        record(
            &db,
            "a",
            RecordKind::Codes,
            &serde_json::json!({"n": 3}),
            crate::defaults::semantic_search::PARTITION_HISTORY_TTL,
        )
        .await
        .unwrap();
        // A namespace whose name extends another's is not mixed in.
        record(
            &db,
            "ab",
            RecordKind::Codes,
            &serde_json::json!({"n": 9}),
            crate::defaults::semantic_search::PARTITION_HISTORY_TTL,
        )
        .await
        .unwrap();
        let all = read(&db, "a", None, 10, crate::defaults::semantic_search::PARTITION_HISTORY_TTL)
            .await
            .unwrap();
        let ns: Vec<i64> = all.iter().map(|r| r.data["n"].as_i64().unwrap()).collect();
        assert_eq!(ns, [3, 2, 1]);
        let codes = read(
            &db,
            "a",
            Some(RecordKind::Codes),
            1,
            crate::defaults::semantic_search::PARTITION_HISTORY_TTL,
        )
        .await
        .unwrap();
        assert_eq!((codes.len(), codes[0].data["n"].as_i64()), (1, Some(3)));
        assert!(
            read(&db, "none", None, 10, crate::defaults::semantic_search::PARTITION_HISTORY_TTL)
                .await
                .unwrap()
                .is_empty()
        );
        db.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn the_newest_records_are_found_across_key_pages() {
        let dir = tempfile::TempDir::new().unwrap();
        let db = AsyncDb::open_with_config(dir.path().to_owned(), crate::support::test_db_config())
            .await
            .unwrap();
        let ttl = crate::defaults::semantic_search::PARTITION_HISTORY_TTL;
        let ns = db.namespace_with_ttl(HISTORY_NS.to_string(), ttl, HISTORY_MAX_DELETES).await.unwrap();
        // More records than one key page, one millisecond apart, every tenth a
        // recall check; written directly so their timestamps are distinct.
        let n = crate::kv_store::KEY_PAGE as u64 * 2 + 7;
        for t in 0..n {
            let kind = if t % 10 == 0 { RecordKind::RecallCheck } else { RecordKind::Codes };
            let mut key = prefix("a");
            key.extend_from_slice(&t.to_be_bytes());
            key.push(kind.tag());
            let value = serde_json::to_vec(&HistoryRecord {
                recorded_at_ms: t,
                kind,
                data: serde_json::json!({}),
            })
            .unwrap();
            ns.put(key, value).await.unwrap();
        }
        let at = |records: Vec<HistoryRecord>| records.iter().map(|r| r.recorded_at_ms).collect::<Vec<_>>();
        assert_eq!(at(read(&db, "a", None, 3, ttl).await.unwrap()), [n - 1, n - 2, n - 3]);
        let last_check = (n - 1) / 10 * 10;
        assert_eq!(
            at(read(&db, "a", Some(RecordKind::RecallCheck), 2, ttl).await.unwrap()),
            [last_check, last_check - 10]
        );
        assert_eq!(read(&db, "a", None, MAX_HISTORY_LIMIT + 50, ttl).await.unwrap().len(), MAX_HISTORY_LIMIT);
        assert!(read(&db, "a", None, 0, ttl).await.unwrap().is_empty());
        db.shutdown().await.unwrap();
    }
}
