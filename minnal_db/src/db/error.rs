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
    #[allow(dead_code)]
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
    #[error("Value log corrupted")]
    #[allow(dead_code)]
    CorruptedLog,
    #[error("Database is closed")]
    DatabaseClosed,
    #[error("write too large: {0}")]
    WriteTooLarge(String),
}

pub type Result<T> = std::result::Result<T, KVError>;
