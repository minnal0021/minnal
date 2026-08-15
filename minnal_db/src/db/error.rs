use crate::db::wal::WalError;
use crate::store::lsm::lsm_tree::LSMError;
use crate::store::value_log::ValueLogError;
use crate::store::value_log::sharded::ShardedValueLogError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum KVError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Key not found")]
    KeyNotFound,
    #[error("LSM error: {0}")]
    LsmError(#[from] LSMError),
    #[error("Value log error: {0}")]
    ValueLogError(#[from] ValueLogError),
    #[error("Sharded value log error: {0}")]
    ShardedValueLogError(#[from] ShardedValueLogError),
    #[error("WAL error: {0}")]
    WalError(#[from] WalError),
    #[error("Database is closed")]
    DatabaseClosed,
    #[error("write too large: {0}")]
    WriteTooLarge(String),
    /// A [`merge`](crate::Db::merge) was committed to the WAL but could not be
    /// applied to the in-memory store, so its result is not readable until the
    /// next open replays it.
    ///
    /// `put` and `delete` deliberately return `Ok` in this situation — the write
    /// *is* durable, and there is nothing useful the caller can do with the
    /// knowledge. A merge is different: its whole premise is that the new value
    /// is a function of the old one, so a merge whose result never became
    /// readable makes the *next* merge on that key compute from a stale base.
    /// Failing loudly is how the caller learns to stop rather than silently
    /// accumulating onto the wrong value.
    ///
    /// The write is not lost — the WAL entry replays on the next open — so this
    /// is "not readable yet", not "gone". `apply_failures` on
    /// [`Db::ops_metrics`](crate::Db::ops_metrics) counts these.
    #[error("merge is durable in the WAL (seq {seq}) but was not applied in memory; it becomes readable after the next open")]
    MergeNotApplied { seq: u64 },
    /// A `merge` closure declined to produce a value.
    ///
    /// The one error the *caller* raises rather than the database: `merge`
    /// hands the closure the stored value and takes back whatever it decides,
    /// including "don't write this" (a counter already at its cap, a value that
    /// failed the caller's own validation). Nothing is written and no WAL entry
    /// is appended when it is returned.
    ///
    /// A closure is free to return any other `KVError` instead — this variant
    /// exists so that deliberately declining does not have to borrow a variant
    /// that means something else.
    #[error("merge aborted: {0}")]
    MergeAborted(String),
    /// The caller's query string is invalid — bad syntax, an unknown or
    /// un-indexed field, a type mismatch, or past the parser's complexity
    /// limits.
    ///
    /// Kept as its own variant rather than folded into [`Serialization`] so
    /// callers can tell "the request was wrong" from "the database failed".
    /// The API layer maps it to `400` and returns the message; a generic
    /// internal error would be a `500` with the text withheld, leaving the
    /// caller nothing to act on.
    ///
    /// [`Serialization`]: KVError::Serialization
    #[error("{0}")]
    Query(#[from] crate::index::query::QueryError),
}

pub type Result<T> = std::result::Result<T, KVError>;
