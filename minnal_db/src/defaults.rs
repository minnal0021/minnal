//! Every configurable setting's default value, in one place.
//!
//! There is one submodule per table of the config file (`config/sample.toml`)
//! and one constant per key, named after the key. [`schema_vector_index`]
//! holds the defaults of a namespace's own vector-index settings, which live
//! in its schema rather than in the config file.
//!
//! Nothing else defines a default. The `Default` impls and constructors of the
//! config types and the config-file parser ([`crate::config_file`]) all read
//! these constants, so a key left out of the file takes the same value as a
//! config built in code, however the database is opened.
//!
//! Limits (the smallest or largest value a setting accepts) and internal
//! tuning constants that no setting exposes stay beside the code they bound.
//!
//! This module imports nothing from the crate, so every layer may use it.

/// `[storage]`: where the database and the server's logs live.
pub mod storage {
    /// `db_path`: the database directory.
    pub const DB_PATH: &str = "./data/db";
    /// `log_dir`: where the API server writes its rolling log files. The
    /// engine does not read it.
    pub const LOG_DIR: &str = "./data/log";
}

/// `[memtable]`: the in-memory write buffer.
pub mod memtable {
    /// `max_capacity`: entries, tombstones included, a memtable holds before
    /// it is sealed and flushed.
    pub const MAX_CAPACITY: usize = 100_000;
}

/// `[sharding]`: how keys are spread across buckets.
pub mod sharding {
    /// `num_buckets`: buckets the value log and the LSM tree are each split
    /// into. Fixed when the database is created: an existing database keeps
    /// the count it was created with, whatever is configured.
    ///
    /// Each namespace holds two open files per bucket (one LSM file and one
    /// value-log file), so the count is also a file-descriptor cost.
    ///
    /// At most `u32::MAX + 1`: a value pointer stores its bucket as a `u32`.
    pub const NUM_BUCKETS: usize = 8;

    const _: () = assert!(
        NUM_BUCKETS >= 1 && NUM_BUCKETS <= (u32::MAX as usize + 1),
        "a value pointer stores its bucket as a u32"
    );
}

/// `[lsm]`: the LSM tree.
pub mod lsm {
    /// `compaction_threshold_percent`: how full (0–100) a memtable may get
    /// before it is sealed.
    pub const COMPACTION_THRESHOLD_PERCENT: usize = 95;
}

/// `[sync]`: when writes reach the disk.
pub mod sync {
    /// `records_per_sync`: value-log writes between fsyncs. A throughput
    /// setting only: every write returns once a WAL fsync covers it.
    pub const RECORDS_PER_SYNC: usize = 1000;
    /// `commit_delay_us`: how long a main-WAL commit round's leader waits for
    /// more writers before its fsync, only when others are already waiting.
    /// 0: under load rounds form without a delay, and a delay lowers
    /// throughput when each client waits for its write before sending the
    /// next.
    pub const COMMIT_DELAY_US: u64 = 0;
}

/// `[thresholds]`: when garbage collection, compaction and back-pressure act.
pub mod thresholds {
    /// `value_log_waste_threshold`: percentage of a bucket that may be
    /// garbage before value-log GC runs on it.
    pub const VALUE_LOG_WASTE_THRESHOLD: f64 = 30.0;
    /// `segment_gc_threshold`: percentage of a sealed value-log segment that
    /// may be garbage before GC rewrites it. Lower than the bucket trigger
    /// above on purpose: with the two equal, garbage just under the trigger
    /// would never be collected.
    pub const SEGMENT_GC_THRESHOLD: f64 = 10.0;
    /// `tail_gc_min_garbage_pct`: garbage percentage at which a bucket's
    /// active tail segment is sealed for GC. `None` uses
    /// `value_log_waste_threshold`.
    pub const TAIL_GC_MIN_GARBAGE_PCT: Option<f64> = None;
    /// `index_blob_waste_threshold`: percentage of a field-index bitmap file
    /// that may be dead space before an index checkpoint compacts it.
    pub const INDEX_BLOB_WASTE_THRESHOLD: f64 = 50.0;
    /// `index_blob_backpressure_bytes`: dead bytes one field index may hold
    /// before a write asks for an early index checkpoint (64 MiB).
    pub const INDEX_BLOB_BACKPRESSURE_BYTES: u64 = 64 * 1024 * 1024;
    /// `index_overlay_soft_bytes`: memory all field-index write buffers of a
    /// database may hold before an early index checkpoint is requested
    /// (32 MiB).
    pub const INDEX_OVERLAY_SOFT_BYTES: u64 = 32 * 1024 * 1024;
    /// `index_overlay_hard_bytes`: memory all field-index write buffers may
    /// hold before the write that crosses it writes its field's buffer out
    /// (64 MiB).
    pub const INDEX_OVERLAY_HARD_BYTES: u64 = 64 * 1024 * 1024;
    /// `max_pinned_wal_segments`: WAL segments the index-replay watermark may
    /// keep from WAL GC before GC reclaims the oldest anyway.
    pub const MAX_PINNED_WAL_SEGMENTS: u32 = 32;
}

/// `[scheduled_tasks]`: how often the background workers run.
pub mod scheduled_tasks {
    use std::time::Duration;

    /// `value_log_gc_interval_secs`.
    pub const VALUE_LOG_GC_INTERVAL: Duration = Duration::from_secs(60);
    /// `wal_gc_interval_secs`.
    pub const WAL_GC_INTERVAL: Duration = Duration::from_secs(60);
    /// `lsm_compaction_interval_secs`.
    pub const LSM_COMPACTION_INTERVAL: Duration = Duration::from_secs(60);
    /// `ttl_cleanup_interval_secs`.
    pub const TTL_CLEANUP_INTERVAL: Duration = Duration::from_secs(3600);
    /// `index_checkpoint_interval_ms`: 1750 ms.
    ///
    /// Short on purpose. The interval is the crash-replay window: after a
    /// crash every field index replays the WAL from its last checkpoint, at
    /// roughly 0.1 ms per distinct key, so a long interval makes a restart
    /// take minutes. It also bounds WAL retention, since the index-replay
    /// watermark keeps segments until a checkpoint moves it. Each tick costs a
    /// flush of every field index and one marker write (temporary file,
    /// rename, two fsyncs) per field, small beside the WAL's fsync on every
    /// write.
    pub const INDEX_CHECKPOINT_INTERVAL: Duration = Duration::from_millis(1750);
}

/// `[wal]`: the main write-ahead log.
pub mod wal {
    /// `segment_size_bytes`: size of each new main-WAL segment (64 MiB).
    pub const SEGMENT_SIZE_BYTES: u64 = 64 * 1024 * 1024;
}

/// `[value_log]`: where values are stored.
pub mod value_log {
    /// `segment_size_bytes`: size at which a value-log segment is sealed
    /// (256 MiB).
    pub const SEGMENT_SIZE_BYTES: u64 = 256 * 1024 * 1024;
    /// `verify_checksums_on_read`: re-check each value's CRC on every read.
    /// Off: latency first; SSTable checksums are always checked.
    pub const VERIFY_CHECKSUMS_ON_READ: bool = false;
}

/// `[recovery]`: what recovery does with a WAL record it cannot apply.
pub mod recovery {
    /// `fail_log_dir`: where fail-log files are written. `None` means
    /// `<db_path>/fail_logs`.
    pub const FAIL_LOG_DIR: Option<&str> = None;
}

/// `[lifecycle]`: how the lifecycle governor runs a store's create, drop and
/// index jobs again after a failure, and how long it keeps its audit log.
pub mod lifecycle {
    use std::time::Duration;

    /// `max_restarts`: runs again of a failed task before it is given up on.
    pub const MAX_RESTARTS: u32 = 3;
    /// `restart_backoff_ms`: wait before each run again.
    pub const RESTART_BACKOFF: Duration = Duration::from_secs(1);
    /// Failures of the governor itself further apart than this are counted
    /// separately. Not in the config file.
    pub const RESTART_WINDOW: Duration = Duration::from_secs(600);
    /// `audit_retention_days`: how long a lifecycle audit record is kept
    /// (30 days).
    pub const AUDIT_RETENTION: Duration = Duration::from_secs(30 * 86_400);
}

/// `[vector_index]`: the background worker that embeds documents and writes
/// their vectors.
pub mod vector_index {
    /// `retry_wait_secs`: wait after a pass with a failure before the queue
    /// is read again.
    pub const RETRY_WAIT_SECS: u64 = 2;
    /// `max_retries`: embedding attempts per queue entry before it is skipped
    /// and left for an operator.
    pub const MAX_RETRIES: u32 = 5;
    /// `concurrency`: embedding calls in flight at once.
    pub const CONCURRENCY: usize = 6;
}

/// `[semantic_search]`: the embedding service and the system-wide caches.
pub mod semantic_search {
    use std::time::Duration;

    /// `embedding_service_url`.
    pub const EMBEDDING_SERVICE_URL: &str = "http://localhost:8001";
    /// `query_embedding_cache_ttl_secs`: how long a cached query embedding
    /// is kept (1 day).
    pub const QUERY_EMBEDDING_CACHE_TTL: Duration = Duration::from_secs(86_400);
    /// `partition_history_ttl_secs`: how long a partition-history record
    /// (code-health read, recall check, rebuild) is kept (30 days).
    pub const PARTITION_HISTORY_TTL: Duration = Duration::from_secs(30 * 86_400);
    /// `embedding_request_timeout_secs`: the whole of one embedding request.
    pub const EMBEDDING_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
    /// `embedding_connect_timeout_secs`: the TCP connect alone; shorter than
    /// the request timeout so an unreachable host fails fast.
    pub const EMBEDDING_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

    /// `[semantic_search.maintenance]`: how each namespace's partition is
    /// split, reassigned and merged. All but `threads` and `rebuild_threads`
    /// are copied into a namespace's schema when it enables semantic search.
    pub mod maintenance {
        /// `threads`: partition-maintenance threads per process.
        pub const THREADS: usize = 2;
        /// `rebuild_threads`: threads a partition rebuild's k-means uses.
        pub const REBUILD_THREADS: usize = 2;
        /// `target_posting_size`: chunks per posting aimed at; a posting
        /// splits above twice this.
        pub const TARGET_POSTING_SIZE: u32 = 128;
        /// `reassign_range`: nearest postings re-checked after a split.
        pub const REASSIGN_RANGE: u32 = 64;
        /// `merge_ratio`: the merge limit as a share of the split limit,
        /// SPFresh's 10 / 118.
        pub const MERGE_RATIO: f64 = 10.0 / 118.0;
        /// `skip_uncertain_moves`: move a chunk only when its code shows it
        /// is clearly closer to the new posting.
        pub const SKIP_UNCERTAIN_MOVES: bool = false;
        /// `split_samples`: members sampled per balanced 2-means iteration
        /// (SPTAG's value).
        pub const SPLIT_SAMPLES: u32 = 1000;
        /// `split_init_trials`: random starting pairs balanced 2-means tries
        /// (SPTAG's value).
        pub const SPLIT_INIT_TRIALS: u32 = 3;
        /// `split_max_iters`: iteration cap of balanced 2-means (SPTAG's
        /// value).
        pub const SPLIT_MAX_ITERS: u32 = 100;
        /// `split_lambda_factor`: balance factor of balanced 2-means,
        /// `λ ≤ 1 / (factor · samples)` (SPTAG's value).
        pub const SPLIT_LAMBDA_FACTOR: f64 = 100.0;
    }

    /// `[semantic_search.search_defaults]`: search settings copied into a
    /// namespace's schema when it enables semantic search; a request can
    /// override them.
    pub mod search_defaults {
        /// `min_probes`: floor on the postings a query reads, for requests
        /// that lower `max_probes`.
        pub const MIN_PROBES: u32 = 1;
        /// `max_probes`: postings a query reads, its nearest, or every
        /// posting of a namespace with fewer. 256 is the smallest count
        /// measured within 0.001 nDCG of reading every posting, at every
        /// cutoff from 10 to 100, on the largest corpus tried (FiQA: 173k
        /// chunks, about 960 postings); it reads 28% of the entries.
        pub const MAX_PROBES: u32 = 256;
        /// `first_pass_top_k`: candidates Pass 1 hands to Pass 2.
        pub const FIRST_PASS_TOP_K: u32 = 1000;
        /// `top_k`: results a search returns.
        pub const TOP_K: u32 = 100;
    }
}

/// A namespace's own vector-index settings (its schema's `vector_index`),
/// used when the namespace enables semantic search without giving them. The
/// model, dimension, chunking and Pass-1 code width are fixed after that.
pub mod schema_vector_index {
    /// `embedding_model`.
    pub const EMBEDDING_MODEL: &str = "gemma";
    /// `embedding_dim`.
    pub const EMBEDDING_DIM: u32 = 768;
    /// `chunking.window_size`: sentences per document chunk.
    pub const WINDOW_SIZE: u32 = 4;
    /// `chunking.sliding_size`: sentences the chunk window moves by.
    pub const SLIDING_SIZE: u32 = 2;
    /// `quantisation.pass1_bits`: bits per dimension of chunk (Pass-1) codes.
    pub const PASS1_BITS: u8 = 1;
    /// `quantisation.pass2_bits`: bits per dimension of whole-document
    /// (Pass-2) codes. Read-only: every namespace uses this width.
    pub const PASS2_BITS: u8 = 8;
    /// `quantisation.rotation_seed`: seed of the random rotation applied
    /// before quantising. Changing it affects only namespaces that enable
    /// semantic search afterwards.
    pub const ROTATION_SEED: u64 = 0x6d69_6e6e_616c_0001;
}
