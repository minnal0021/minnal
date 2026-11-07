//! The engine's write-ahead logs and what they share.
//!
//! Two logs sit on one format: the main WAL ([`Wal`]) takes every WAL-backed
//! write, and the vector index WAL takes the vector worker's payload writes.
//! `segmented_log` is the format itself, with `commit_rounds` (concurrent
//! callers sharing one fsync) and `segment_prep` (segments zero-filled ahead)
//! beneath it. `recovery` replays both logs at open, `gc` reclaims the main WAL's
//! segments, and `wal_fail_log` records the writes that failed to replay. See
//! `minnal_db/docs/WAL-Architecture.md`.

pub(crate) mod commit_rounds;
pub(crate) mod gc;
mod main_wal;
mod recovery;
pub(crate) mod segment_prep;
pub(crate) mod segmented_log;
pub(crate) mod vector_index_wal;
pub(crate) mod wal_fail_log;

pub use main_wal::*;

/// Power-loss and kill tests for the main WAL's write path (group commit,
/// retirement, checksummed frames).
#[cfg(test)]
mod power_loss_tests;
