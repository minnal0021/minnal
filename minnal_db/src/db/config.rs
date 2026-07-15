use crate::store::lsm::lsm_tree::LSMConfig;
use crate::support::DEFAULT_NUM_BUCKETS;
use std::path::PathBuf;
use std::time::Duration;

/// How many insert/update/delete operations to batch before syncing the
/// value log and marking WAL entries as persisted.
///
/// * `records_per_sync = 0` — only sync on close or GC (maximum throughput,
///   least durability between explicit syncs).
/// * `records_per_sync = 1` — sync after every single write (maximum
///   durability, lowest throughput).
/// * Any other value N — sync every N writes (tunable trade-off).
///
/// The WAL is always fsynced after every append regardless of this setting,
/// so crash recovery is always possible.  This setting controls how quickly
/// value-log data and WAL-persisted status are flushed to disk.
#[derive(Debug, Clone, Copy)]
pub struct SyncConfig {
    pub records_per_sync: usize,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self { records_per_sync: 1000 }
    }
}

impl SyncConfig {
    pub fn new(records_per_sync: usize) -> Self {
        Self { records_per_sync }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScheduledTaskConfig {
    pub(crate) value_log_gc_interval: Duration,
    pub(crate) wal_gc_interval: Duration,
    pub(crate) lsm_compaction_interval: Duration,
    pub(crate) ttl_cleanup_interval: Duration,
}

/// Default percentage of a field-index bitmap value region that may be dead
/// space before checkpoint compacts it.
pub const DEFAULT_INDEX_BLOB_WASTE_THRESHOLD: f64 = 50.0;

/// Default absolute cap (bytes) on a single field index's reclaimable dead blob
/// bytes before the write path requests an early index checkpoint. 64 MiB. See
/// [`ThresholdConfig::index_blob_backpressure_bytes`].
pub const DEFAULT_INDEX_BLOB_BACKPRESSURE_BYTES: u64 = 64 * 1024 * 1024;

/// Default percentage of a *value-log page* that may be garbage before GC
/// rewrites that page. Deliberately **lower** than the bucket-level
/// [`value_log_waste_threshold`](ThresholdConfig::value_log_waste_threshold) —
/// see [`ThresholdConfig::page_gc_threshold`] for why.
pub const DEFAULT_PAGE_GC_THRESHOLD: f64 = 10.0;

#[derive(Debug, Clone, Copy)]
pub struct ThresholdConfig {
    /// Percentage (`0..100`) of a **bucket** that may be garbage before GC runs
    /// on it at all. This is the *trigger*: it answers "is this namespace worth
    /// collecting yet?"
    pub value_log_waste_threshold: f64,
    /// Percentage (`0..100`) of an individual **page** that may be garbage before
    /// GC rewrites that page. This is the *selection* rule, and it is a different
    /// question from the trigger above.
    ///
    /// Keep it **well below** `value_log_waste_threshold`. A page under this
    /// threshold is treated as "clean" and copied byte-for-byte into the
    /// compacted file — carrying its garbage with it, unreclaimed. If the two
    /// values were equal, garbage sitting just under the trigger could never be
    /// collected at all: every pass would copy those pages across intact, report
    /// success, and reclaim almost nothing, while the same bytes keep the bucket
    /// over its trigger — GC on a treadmill. A lower page threshold rewrites more
    /// survivors per pass but actually finishes the job.
    pub page_gc_threshold: f64,
    /// Percentage (`0..100`) of a field-index bitmap value region that may be
    /// dead space before the index checkpoint compacts it. The bitmap store is
    /// append-only, so each per-document insert leaves a stale copy of that
    /// field-value's bitmap behind; compaction reclaims it.
    pub index_blob_waste_threshold: f64,
    /// Absolute cap (bytes) on a single field index's reclaimable dead blob
    /// bytes before the write path proactively requests an index checkpoint,
    /// instead of waiting for the periodic (~15 min) tick.
    ///
    /// This is **backpressure**, and it is an absolute byte cap on purpose — a
    /// *ratio* trigger is useless here because a low-cardinality, high-churn
    /// field (e.g. a boolean over many docs) crosses any ratio almost
    /// immediately and stays pinned near 100%, so it would fire on nearly every
    /// write. Capping absolute dead bytes bounds the transient on-disk
    /// amplification to roughly this value per field, and it self-debounces:
    /// compaction resets the field's dead-byte count to 0, so the next request
    /// only fires after another `index_blob_backpressure_bytes` accumulate.
    /// The dead-byte count is O(1) to read, so the check is cheap on the hot
    /// write path (unlike `index_blob_waste_threshold`, which scans every slot).
    pub index_blob_backpressure_bytes: u64,
}

impl ThresholdConfig {
    pub fn new(waste_threshold: f64) -> Self {
        Self {
            value_log_waste_threshold: waste_threshold,
            page_gc_threshold: DEFAULT_PAGE_GC_THRESHOLD,
            index_blob_waste_threshold: DEFAULT_INDEX_BLOB_WASTE_THRESHOLD,
            index_blob_backpressure_bytes: DEFAULT_INDEX_BLOB_BACKPRESSURE_BYTES,
        }
    }

    /// Override the per-page GC threshold (percentage `0..100`).
    pub fn with_page_gc_threshold(mut self, threshold: f64) -> Self {
        self.page_gc_threshold = threshold;
        self
    }

    /// Override the field-index bitmap compaction threshold (percentage `0..100`).
    pub fn with_index_blob_waste_threshold(mut self, threshold: f64) -> Self {
        self.index_blob_waste_threshold = threshold;
        self
    }

    /// Override the index-checkpoint backpressure cap (bytes). See
    /// [`index_blob_backpressure_bytes`](Self::index_blob_backpressure_bytes).
    pub fn with_index_blob_backpressure_bytes(mut self, bytes: u64) -> Self {
        self.index_blob_backpressure_bytes = bytes;
        self
    }
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            value_log_waste_threshold: 30.0,
            page_gc_threshold: DEFAULT_PAGE_GC_THRESHOLD,
            index_blob_waste_threshold: DEFAULT_INDEX_BLOB_WASTE_THRESHOLD,
            index_blob_backpressure_bytes: DEFAULT_INDEX_BLOB_BACKPRESSURE_BYTES,
        }
    }
}

impl Default for ScheduledTaskConfig {
    fn default() -> Self {
        Self {
            value_log_gc_interval: Duration::from_secs(60),
            wal_gc_interval: Duration::from_secs(60),
            lsm_compaction_interval: Duration::from_secs(60),
            ttl_cleanup_interval: Duration::from_secs(3600),
        }
    }
}

impl ScheduledTaskConfig {
    pub fn new(value_log_gc_interval: Duration, wal_gc_interval: Duration, lsm_compaction_interval: Duration) -> Self {
        Self {
            value_log_gc_interval,
            wal_gc_interval,
            lsm_compaction_interval,
            ttl_cleanup_interval: Duration::from_secs(3600),
        }
    }

    /// Set the TTL cleanup worker interval.
    pub fn with_ttl_cleanup_interval(mut self, interval: Duration) -> Self {
        self.ttl_cleanup_interval = interval;
        self
    }
}

#[derive(Debug, Clone)]
pub struct DbConfig {
    pub threshold_config: ThresholdConfig,
    pub sync_config: SyncConfig,
    pub scheduled_task_config: ScheduledTaskConfig,
    pub lsm_config: LSMConfig,
    /// Number of sharding buckets for value log and LSM.
    pub num_buckets: usize,
    /// Maximum number of entries (including tombstones) in the in-memory skip list.
    pub skip_list_capacity: usize,
    /// WAL segment size in bytes (default 64 MiB).
    ///
    /// **Fixed at creation.** Unlike [`segment_size_bytes`](Self::segment_size_bytes)
    /// for the value log, a WAL segment id is `offset / wal_segment_size`, so this
    /// cannot change once the WAL holds data — a different size would map stored
    /// offsets to the wrong segment files. It is honoured only for a brand-new WAL
    /// and then recorded in a `wal_segment_size` marker; on an existing WAL the
    /// recorded size wins and a differing config value is ignored with a warning.
    pub wal_segment_size: u64,
    /// Size at which a value-log segment is sealed and a new one opened
    /// (default 256 MiB).
    ///
    /// Unlike the page size it replaces, this is **not** fixed at creation: a
    /// segment's size is not encoded in any value pointer (only its id and a byte
    /// offset are), so existing segments keep the size they were written at and new
    /// ones simply use whatever is configured now.
    ///
    /// Must be a multiple of 4096, at least 64 KiB, and at most 4 GiB (a record's
    /// offset within its segment is a `u32`).
    pub segment_size_bytes: u64,
    /// Directory for recovery fail-log files.
    /// `None` defaults to `<db_path>/fail_logs` at open time.
    pub fail_log_dir: Option<PathBuf>,
    /// Verify the per-record CRC32 of each value on **every read**.
    ///
    /// Values always carry a CRC on disk (written on the cheap write path);
    /// this flag controls whether reads re-verify it. Verifying catches silent
    /// corruption before a value is served, but adds a full CRC pass over the
    /// value to the read hot path — measurable for large values. Defaults to
    /// `false` (latency first); the SSTable-level CRC is always verified
    /// regardless of this flag.
    pub verify_checksums_on_read: bool,
}

impl Default for DbConfig {
    fn default() -> Self {
        Self {
            threshold_config: ThresholdConfig::default(),
            sync_config: SyncConfig::default(),
            scheduled_task_config: ScheduledTaskConfig::default(),
            lsm_config: LSMConfig::default(),
            num_buckets: DEFAULT_NUM_BUCKETS,
            skip_list_capacity: 100_000,
            wal_segment_size: 64 * 1024 * 1024,
            segment_size_bytes: crate::store::value_log::DEFAULT_SEGMENT_SIZE_BYTES,
            fail_log_dir: None,
            verify_checksums_on_read: false,
        }
    }
}

impl DbConfig {
    pub fn new(
        threshold_config: ThresholdConfig,
        scheduled_task_config: ScheduledTaskConfig,
        sync_config: SyncConfig,
        lsm_config: LSMConfig,
    ) -> Self {
        Self {
            threshold_config,
            sync_config,
            scheduled_task_config,
            lsm_config,
            num_buckets: DEFAULT_NUM_BUCKETS,
            skip_list_capacity: 100_000,
            wal_segment_size: 64 * 1024 * 1024,
            segment_size_bytes: crate::store::value_log::DEFAULT_SEGMENT_SIZE_BYTES,
            fail_log_dir: None,
            verify_checksums_on_read: false,
        }
    }
}
