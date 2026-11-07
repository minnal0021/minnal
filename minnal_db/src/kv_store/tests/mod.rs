//! Tests for `KVStore`, with the fixtures they share.

mod basic;
mod gc;
mod gc_concurrency;
mod gc_crash;
mod scans;

use super::*;
use crate::storage::value_log::DEFAULT_SEGMENT_SIZE_BYTES;
use tempfile::TempDir;

fn default_lsm_config() -> LSMConfig {
    LSMConfig {
        num_buckets: crate::support::TEST_NUM_BUCKETS,
        ..LSMConfig::default()
    }
}
