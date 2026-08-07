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
