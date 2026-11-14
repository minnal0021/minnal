//! The config file's engine tables, `[storage]` to `[recovery]`, and the one
//! mapping from them to [`DbConfig`].
//!
//! Every key's default is the constant of the same name in
//! [`crate::defaults`]; each table rejects keys it does not know.

use std::path::PathBuf;
use std::time::Duration;

use serde::Deserialize;

use crate::db::config::{DbConfig, ScheduledTaskConfig, SyncConfig, ThresholdConfig};
use crate::defaults;
use crate::storage::lsm::tree::LSMConfig;

/// `[storage]`: where the database lives. `log_dir` is read by the API
/// server for its log files; the engine ignores it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StorageSection {
    /// The database directory.
    pub db_path: String,
    /// Where the API server writes its rolling log files.
    pub log_dir: String,
}

impl Default for StorageSection {
    fn default() -> Self {
        Self {
            db_path: defaults::storage::DB_PATH.into(),
            log_dir: defaults::storage::LOG_DIR.into(),
        }
    }
}

/// `[memtable]`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MemtableSection {
    /// Entries, tombstones included, a memtable holds before it is sealed.
    pub max_capacity: usize,
}

impl Default for MemtableSection {
    fn default() -> Self {
        Self {
            max_capacity: defaults::memtable::MAX_CAPACITY,
        }
    }
}

/// `[sharding]`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ShardingSection {
    /// Buckets the value log and the LSM tree are each split into. Fixed when
    /// the database is created: an existing database keeps its own count.
    pub num_buckets: usize,
}

impl Default for ShardingSection {
    fn default() -> Self {
        Self {
            num_buckets: defaults::sharding::NUM_BUCKETS,
        }
    }
}

/// `[lsm]`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LsmSection {
    /// How full (0–100) a memtable may get before it is sealed.
    pub compaction_threshold_percent: usize,
}

impl Default for LsmSection {
    fn default() -> Self {
        Self {
            compaction_threshold_percent: defaults::lsm::COMPACTION_THRESHOLD_PERCENT,
        }
    }
}

/// `[sync]`. See [`SyncConfig`].
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SyncSection {
    /// Value-log writes between fsyncs.
    pub records_per_sync: usize,
    /// The main WAL's commit delay, in microseconds.
    pub commit_delay_us: u64,
}

impl Default for SyncSection {
    fn default() -> Self {
        Self {
            records_per_sync: defaults::sync::RECORDS_PER_SYNC,
            commit_delay_us: defaults::sync::COMMIT_DELAY_US,
        }
    }
}

/// `[thresholds]`. See [`ThresholdConfig`] for what each one does.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ThresholdSection {
    /// Percentage of a bucket that may be garbage before value-log GC runs on it.
    pub value_log_waste_threshold: f64,
    /// Percentage of a sealed segment that may be garbage before GC rewrites it.
    pub segment_gc_threshold: f64,
    /// Garbage percentage at which a bucket's active tail is sealed for GC;
    /// unset, `value_log_waste_threshold`.
    pub tail_gc_min_garbage_pct: Option<f64>,
    /// Percentage of a field-index bitmap file that may be dead space before
    /// a checkpoint compacts it.
    pub index_blob_waste_threshold: f64,
    /// Dead bytes one field index may hold before a write asks for an early
    /// checkpoint; 0 turns this off.
    pub index_blob_backpressure_bytes: u64,
    /// Field-index write-buffer memory that requests an early checkpoint.
    pub index_overlay_soft_bytes: u64,
    /// Field-index write-buffer memory past which a write spills its field.
    pub index_overlay_hard_bytes: u64,
    /// WAL segments the index-replay watermark may keep from WAL GC; 0 turns
    /// the limit off.
    pub max_pinned_wal_segments: u32,
}

impl Default for ThresholdSection {
    fn default() -> Self {
        Self {
            value_log_waste_threshold: defaults::thresholds::VALUE_LOG_WASTE_THRESHOLD,
            segment_gc_threshold: defaults::thresholds::SEGMENT_GC_THRESHOLD,
            tail_gc_min_garbage_pct: defaults::thresholds::TAIL_GC_MIN_GARBAGE_PCT,
            index_blob_waste_threshold: defaults::thresholds::INDEX_BLOB_WASTE_THRESHOLD,
            index_blob_backpressure_bytes: defaults::thresholds::INDEX_BLOB_BACKPRESSURE_BYTES,
            index_overlay_soft_bytes: defaults::thresholds::INDEX_OVERLAY_SOFT_BYTES,
            index_overlay_hard_bytes: defaults::thresholds::INDEX_OVERLAY_HARD_BYTES,
            max_pinned_wal_segments: defaults::thresholds::MAX_PINNED_WAL_SEGMENTS,
        }
    }
}

/// `[scheduled_tasks]`: how often each background worker runs.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ScheduledTaskSection {
    /// Value-log GC, in seconds.
    pub value_log_gc_interval_secs: u64,
    /// WAL GC, in seconds.
    pub wal_gc_interval_secs: u64,
    /// LSM compaction, in seconds.
    pub lsm_compaction_interval_secs: u64,
    /// TTL expiry, in seconds.
    pub ttl_cleanup_interval_secs: u64,
    /// Index checkpoint, in milliseconds: the crash-replay window.
    pub index_checkpoint_interval_ms: u64,
}

impl Default for ScheduledTaskSection {
    fn default() -> Self {
        use defaults::scheduled_tasks as d;
        Self {
            value_log_gc_interval_secs: d::VALUE_LOG_GC_INTERVAL.as_secs(),
            wal_gc_interval_secs: d::WAL_GC_INTERVAL.as_secs(),
            lsm_compaction_interval_secs: d::LSM_COMPACTION_INTERVAL.as_secs(),
            ttl_cleanup_interval_secs: d::TTL_CLEANUP_INTERVAL.as_secs(),
            index_checkpoint_interval_ms: d::INDEX_CHECKPOINT_INTERVAL.as_millis() as u64,
        }
    }
}

/// `[wal]`: the main write-ahead log.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WalSection {
    /// Size of each new segment, in bytes.
    pub segment_size_bytes: u64,
}

impl Default for WalSection {
    fn default() -> Self {
        Self {
            segment_size_bytes: defaults::wal::SEGMENT_SIZE_BYTES,
        }
    }
}

/// `[value_log]`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ValueLogSection {
    /// Size at which a segment is sealed, in bytes.
    pub segment_size_bytes: u64,
    /// Re-check each value's CRC on every read.
    pub verify_checksums_on_read: bool,
}

impl Default for ValueLogSection {
    fn default() -> Self {
        Self {
            segment_size_bytes: defaults::value_log::SEGMENT_SIZE_BYTES,
            verify_checksums_on_read: defaults::value_log::VERIFY_CHECKSUMS_ON_READ,
        }
    }
}

/// `[recovery]`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RecoverySection {
    /// Where recovery writes a fail-log file for each WAL record it cannot
    /// apply after a retry; unset, `<db_path>/fail_logs`.
    pub fail_log_dir: Option<String>,
}

impl Default for RecoverySection {
    fn default() -> Self {
        Self {
            fail_log_dir: defaults::recovery::FAIL_LOG_DIR.map(String::from),
        }
    }
}

impl super::ConfigFile {
    /// The engine's configuration: every field of [`DbConfig`] comes from
    /// its key here, and nowhere else. `lsm_config`'s directory, bucket count
    /// and memtable capacity are filled in when the database opens.
    pub fn db_config(&self) -> DbConfig {
        let t = &self.thresholds;
        let s = &self.scheduled_tasks;
        DbConfig {
            threshold_config: ThresholdConfig {
                value_log_waste_threshold: t.value_log_waste_threshold,
                segment_gc_threshold: t.segment_gc_threshold,
                tail_gc_min_garbage_pct: t.tail_gc_min_garbage_pct,
                index_blob_waste_threshold: t.index_blob_waste_threshold,
                index_blob_backpressure_bytes: t.index_blob_backpressure_bytes,
                index_overlay_soft_bytes: t.index_overlay_soft_bytes,
                index_overlay_hard_bytes: t.index_overlay_hard_bytes,
                max_pinned_wal_segments: t.max_pinned_wal_segments,
            },
            sync_config: SyncConfig {
                records_per_sync: self.sync.records_per_sync,
                commit_delay_us: self.sync.commit_delay_us,
            },
            scheduled_task_config: ScheduledTaskConfig {
                value_log_gc_interval: Duration::from_secs(s.value_log_gc_interval_secs),
                wal_gc_interval: Duration::from_secs(s.wal_gc_interval_secs),
                lsm_compaction_interval: Duration::from_secs(s.lsm_compaction_interval_secs),
                ttl_cleanup_interval: Duration::from_secs(s.ttl_cleanup_interval_secs),
                index_checkpoint_interval: Duration::from_millis(s.index_checkpoint_interval_ms),
            },
            lsm_config: LSMConfig::new(self.lsm.compaction_threshold_percent, PathBuf::from("lsm_data")),
            num_buckets: self.sharding.num_buckets,
            skip_list_capacity: self.memtable.max_capacity,
            wal_segment_size: self.wal.segment_size_bytes,
            segment_size_bytes: self.value_log.segment_size_bytes,
            fail_log_dir: self.recovery.fail_log_dir.as_deref().map(PathBuf::from),
            verify_checksums_on_read: self.value_log.verify_checksums_on_read,
        }
    }

    /// Check the engine settings a typed value cannot rule out.
    pub(super) fn validate_engine(&self) -> Result<(), String> {
        let n = self.sharding.num_buckets;
        if n == 0 || n > u32::MAX as usize + 1 {
            return Err(format!("sharding.num_buckets must be between 1 and 2^32, got {n}"));
        }
        Ok(())
    }
}
