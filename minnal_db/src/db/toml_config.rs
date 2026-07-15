use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::db::config::{DbConfig, ScheduledTaskConfig, SyncConfig, ThresholdConfig};
use crate::db::error::{KVError, Result};
use crate::store::lsm::lsm_tree::LSMConfig;
use crate::support::DEFAULT_NUM_BUCKETS;

/// Root structure mirroring `minnal.toml`.
#[derive(Debug, Deserialize)]
pub struct MinnalTomlConfig {
    pub storage: StorageSection,
    #[serde(default)]
    pub memtable: MemtableSection,
    #[serde(default)]
    pub sharding: ShardingSection,
    #[serde(default)]
    pub lsm: LsmSection,
    #[serde(default)]
    pub sync: SyncSection,
    #[serde(default)]
    pub thresholds: ThresholdSection,
    #[serde(default)]
    pub scheduled_tasks: ScheduledTaskSection,
    #[serde(default)]
    pub wal: WalSection,
    #[serde(default)]
    pub value_log: ValueLogSection,
    #[serde(default)]
    pub recovery: RecoverySection,
}

#[derive(Debug, Deserialize)]
pub struct StorageSection {
    pub db_path: String,
}

#[derive(Debug, Deserialize)]
pub struct MemtableSection {
    #[serde(default = "default_max_capacity")]
    pub max_capacity: usize,
}

impl Default for MemtableSection {
    fn default() -> Self {
        Self {
            max_capacity: default_max_capacity(),
        }
    }
}

fn default_max_capacity() -> usize {
    100_000
}

#[derive(Debug, Deserialize)]
pub struct ShardingSection {
    #[serde(default = "default_num_buckets")]
    pub num_buckets: usize,
}

impl Default for ShardingSection {
    fn default() -> Self {
        Self {
            num_buckets: default_num_buckets(),
        }
    }
}

fn default_num_buckets() -> usize {
    DEFAULT_NUM_BUCKETS
}

#[derive(Debug, Deserialize)]
pub struct LsmSection {
    #[serde(default = "default_compaction_threshold_percent")]
    pub compaction_threshold_percent: usize,
}

impl Default for LsmSection {
    fn default() -> Self {
        Self {
            compaction_threshold_percent: default_compaction_threshold_percent(),
        }
    }
}

fn default_compaction_threshold_percent() -> usize {
    95
}

#[derive(Debug, Deserialize)]
pub struct SyncSection {
    #[serde(default = "default_records_per_sync")]
    pub records_per_sync: usize,
}

impl Default for SyncSection {
    fn default() -> Self {
        Self {
            records_per_sync: default_records_per_sync(),
        }
    }
}

fn default_records_per_sync() -> usize {
    1000
}

#[derive(Debug, Deserialize)]
pub struct ThresholdSection {
    #[serde(default = "default_value_log_waste_threshold")]
    pub value_log_waste_threshold: f64,
    #[serde(default = "default_segment_gc_threshold")]
    pub segment_gc_threshold: f64,
    /// Garbage share at which a bucket's active tail is sealed for GC. Omit to track
    /// `value_log_waste_threshold` (the recommended default).
    #[serde(default)]
    pub tail_gc_min_garbage_pct: Option<f64>,
    #[serde(default = "default_index_blob_waste_threshold")]
    pub index_blob_waste_threshold: f64,
    #[serde(default = "default_index_blob_backpressure_bytes")]
    pub index_blob_backpressure_bytes: u64,
}

impl Default for ThresholdSection {
    fn default() -> Self {
        Self {
            value_log_waste_threshold: default_value_log_waste_threshold(),
            segment_gc_threshold: default_segment_gc_threshold(),
            tail_gc_min_garbage_pct: None,
            index_blob_waste_threshold: default_index_blob_waste_threshold(),
            index_blob_backpressure_bytes: default_index_blob_backpressure_bytes(),
        }
    }
}

fn default_value_log_waste_threshold() -> f64 {
    30.0
}

fn default_segment_gc_threshold() -> f64 {
    crate::db::config::DEFAULT_SEGMENT_GC_THRESHOLD
}

fn default_index_blob_waste_threshold() -> f64 {
    crate::db::config::DEFAULT_INDEX_BLOB_WASTE_THRESHOLD
}

fn default_index_blob_backpressure_bytes() -> u64 {
    crate::db::config::DEFAULT_INDEX_BLOB_BACKPRESSURE_BYTES
}

#[derive(Debug, Deserialize)]
pub struct ScheduledTaskSection {
    #[serde(default = "default_gc_interval_secs")]
    pub value_log_gc_interval_secs: u64,
    #[serde(default = "default_gc_interval_secs")]
    pub wal_gc_interval_secs: u64,
    #[serde(default = "default_gc_interval_secs")]
    pub lsm_compaction_interval_secs: u64,
    #[serde(default = "default_ttl_cleanup_interval_secs")]
    pub ttl_cleanup_interval_secs: u64,
}

impl Default for ScheduledTaskSection {
    fn default() -> Self {
        Self {
            value_log_gc_interval_secs: default_gc_interval_secs(),
            wal_gc_interval_secs: default_gc_interval_secs(),
            lsm_compaction_interval_secs: default_gc_interval_secs(),
            ttl_cleanup_interval_secs: default_ttl_cleanup_interval_secs(),
        }
    }
}

fn default_gc_interval_secs() -> u64 {
    60
}

fn default_ttl_cleanup_interval_secs() -> u64 {
    3600
}

#[derive(Debug, Deserialize)]
pub struct WalSection {
    #[serde(default = "default_wal_segment_size_bytes")]
    pub segment_size_bytes: u64,
}

impl Default for WalSection {
    fn default() -> Self {
        Self {
            segment_size_bytes: default_wal_segment_size_bytes(),
        }
    }
}

fn default_wal_segment_size_bytes() -> u64 {
    64 * 1024 * 1024
}

#[derive(Debug, Deserialize)]
pub struct ValueLogSection {
    #[serde(default = "default_segment_size_bytes")]
    pub segment_size_bytes: u64,
    /// Re-verify each value's CRC32 on every read. Defaults to `false`
    /// (latency first); see `DbConfig::verify_checksums_on_read`.
    #[serde(default)]
    pub verify_checksums_on_read: bool,
}

impl Default for ValueLogSection {
    fn default() -> Self {
        Self {
            segment_size_bytes: default_segment_size_bytes(),
            verify_checksums_on_read: false,
        }
    }
}

fn default_segment_size_bytes() -> u64 {
    crate::store::value_log::DEFAULT_SEGMENT_SIZE_BYTES
}

/// `[recovery]` — WAL recovery settings.
#[derive(Debug, Default, Deserialize)]
pub struct RecoverySection {
    /// Directory for timestamped fail-log JSON files written when a WAL entry
    /// cannot be applied even after one retry.  Defaults to `<db_path>/fail_logs`.
    pub fail_log_dir: Option<String>,
}

impl MinnalTomlConfig {
    /// Parse a `minnal.toml` file from the given path.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            KVError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to read config file '{}': {}", path.display(), e),
            ))
        })?;
        let config: MinnalTomlConfig = toml::from_str(&content).map_err(|e| {
            KVError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to parse config file '{}': {}", path.display(), e),
            ))
        })?;
        Ok(config)
    }

    /// Return the database path from the config.
    pub fn db_path(&self) -> PathBuf {
        PathBuf::from(&self.storage.db_path)
    }

    /// Convert to the internal `DbConfig` used throughout the engine.
    pub fn to_db_config(&self) -> DbConfig {
        DbConfig {
            threshold_config: ThresholdConfig::new(self.thresholds.value_log_waste_threshold)
                .with_segment_gc_threshold(self.thresholds.segment_gc_threshold)
                .with_tail_gc_min_garbage_pct(self.thresholds.tail_gc_min_garbage_pct)
                .with_index_blob_waste_threshold(self.thresholds.index_blob_waste_threshold)
                .with_index_blob_backpressure_bytes(self.thresholds.index_blob_backpressure_bytes),
            sync_config: SyncConfig::new(self.sync.records_per_sync),
            scheduled_task_config: ScheduledTaskConfig {
                value_log_gc_interval: Duration::from_secs(self.scheduled_tasks.value_log_gc_interval_secs),
                wal_gc_interval: Duration::from_secs(self.scheduled_tasks.wal_gc_interval_secs),
                lsm_compaction_interval: Duration::from_secs(self.scheduled_tasks.lsm_compaction_interval_secs),
                ttl_cleanup_interval: Duration::from_secs(self.scheduled_tasks.ttl_cleanup_interval_secs),
            },
            lsm_config: LSMConfig::new(
                self.lsm.compaction_threshold_percent,
                PathBuf::from("lsm_data"), // overridden at open-time
            ),
            num_buckets: self.sharding.num_buckets,
            skip_list_capacity: self.memtable.max_capacity,
            wal_segment_size: self.wal.segment_size_bytes,
            segment_size_bytes: self.value_log.segment_size_bytes,
            fail_log_dir: self.recovery.fail_log_dir.as_deref().map(PathBuf::from),
            verify_checksums_on_read: self.value_log.verify_checksums_on_read,
        }
    }
}
