//! Sharded Value Log
//!
//! Fans the value log across `num_buckets` buckets, chosen by hashing the key.
//! Each bucket owns an independent series of [`ValueLog`] segment files and its
//! own write lock, so writes and GC on different buckets never contend.

use super::{BatchValue, SegmentStats, ValueLocation, ValueLog, ValueLogMetadata, ValueRecordMeta};
use crate::support;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ShardedValueLogError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Value log error: {0}")]
    ValueLogError(#[from] super::ValueLogError),
    #[error("Invalid bucket index: {0}")]
    InvalidBucket(u32),
}

impl ShardedValueLogError {
    /// True only for the **transient** reclaimed-segment case: GC unlinked the segment
    /// after relocating the record, so re-resolving the pointer through the LSM (which
    /// now holds the new one) and retrying is the correct response. Every other error —
    /// corruption, IO faults — is real and must be surfaced, not retried and not masked
    /// as a missing key. See [`super::ValueLogError::SegmentMissing`].
    pub fn is_segment_missing(&self) -> bool {
        matches!(self, ShardedValueLogError::ValueLogError(super::ValueLogError::SegmentMissing(_)))
    }
}

pub type Result<T> = std::result::Result<T, ShardedValueLogError>;

/// A value's full address: its bucket plus its location within that bucket's
/// segment files.
///
/// # Encoding
///
/// The LSM stores this as one `u128`:
///
/// ```text
///  127            96 95            64 63            32 31             0
/// ┌────────────────┬────────────────┬────────────────┬────────────────┐
/// │  bucket (32)   │ segment_id(32) │ rec_offset(32) │  value_len(32) │
/// └────────────────┴────────────────┴────────────────┴────────────────┘
/// ```
///
/// `value_len` rides along so a read is **one `pread` of exactly
/// `header + value_len` bytes** — there is no page header to consult and no slot
/// table to walk. `segment_id == 0` is the reserved "no value" sentinel.
///
/// Segment ids are **never reused**, so an address identifies one record for the
/// life of the database: a pointer GC has superseded still resolves to the same
/// key's bytes, and one whose segment GC has deleted fails loudly rather than
/// landing on some other key's record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardedValuePointer {
    pub bucket: u32,
    pub location: ValueLocation,
}

impl ShardedValuePointer {
    pub fn new(bucket: u32, location: ValueLocation, num_buckets: usize) -> Result<Self> {
        if bucket >= num_buckets as u32 {
            return Err(ShardedValueLogError::InvalidBucket(bucket));
        }
        Ok(Self { bucket, location })
    }

    pub fn to_u128(self) -> u128 {
        ((self.bucket as u128) << 96)
            | ((self.location.segment_id as u128) << 64)
            | ((self.location.rec_offset as u128) << 32)
            | (self.location.value_len as u128)
    }

    pub fn from_u128(encoded: u128) -> Self {
        Self {
            bucket: ((encoded >> 96) & 0xFFFF_FFFF) as u32,
            location: ValueLocation {
                segment_id: ((encoded >> 64) & 0xFFFF_FFFF) as u32,
                rec_offset: ((encoded >> 32) & 0xFFFF_FFFF) as u32,
                value_len: (encoded & 0xFFFF_FFFF) as u32,
            },
        }
    }

    /// The on-disk size of the record this points at, given its key length.
    /// Used for garbage accounting when a write displaces it — no I/O needed.
    pub fn record_len(&self, key_len: usize) -> u64 {
        super::ValueRecordHeader::record_len(key_len, self.location.value_len as usize)
    }
}

/// Physical on-disk footprint of one value-log shard.
#[derive(Debug, Clone)]
pub struct ShardPhysicalStats {
    pub bucket: u32,
    /// Sum of the shard's segment files. Segments are dense append-only files, so
    /// this is simply what the shard occupies — there are no sparse holes.
    pub physical_bytes: u64,
    /// Record bytes the shard is tracking (live + garbage), excluding file headers.
    pub logical_bytes: u64,
}

/// Sharded value log manager: one [`ValueLog`] per bucket.
pub struct ShardedValueLog {
    logs: Vec<Arc<ValueLog>>,
    /// Per-bucket write/GC lock. Held across a write's value-log append *and* its
    /// LSM insert, and across GC's re-point, so GC and writers never interleave on
    /// one key.
    bucket_write_locks: Vec<Arc<Mutex<()>>>,
    num_buckets: usize,
    #[allow(dead_code)] // kept for diagnostics: the size new segments are sealed at
    segment_size: u64,
    #[allow(dead_code)]
    base_path: PathBuf,
}

impl ShardedValueLog {
    /// Open (or create) the value log for every bucket under `base_path`.
    pub fn open<P: AsRef<Path>>(base_path: P, num_buckets: usize, segment_size: u64) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_path)?;

        let mut logs = Vec::with_capacity(num_buckets);
        let mut bucket_write_locks = Vec::with_capacity(num_buckets);
        for bucket in 0u32..num_buckets as u32 {
            logs.push(Arc::new(ValueLog::open(&base_path, bucket, segment_size)?));
            bucket_write_locks.push(Arc::new(Mutex::new(())));
        }

        Ok(Self {
            logs,
            bucket_write_locks,
            num_buckets,
            segment_size,
            base_path,
        })
    }

    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }

    #[allow(dead_code)]
    pub fn segment_size(&self) -> u64 {
        self.segment_size
    }

    #[allow(dead_code)]
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    pub fn bucket_for_key(&self, key: &[u8]) -> u32 {
        support::get_bucket_for_key(key, self.num_buckets)
    }

    pub fn get_bucket_log(&self, bucket: u32) -> Result<Arc<ValueLog>> {
        self.logs
            .get(bucket as usize)
            .map(Arc::clone)
            .ok_or(ShardedValueLogError::InvalidBucket(bucket))
    }

    /// Take a bucket's write lock. Writers hold it across their value-log append and
    /// LSM insert; GC holds it across its re-point, so a key cannot be overwritten
    /// between GC deciding to relocate it and GC re-pointing it.
    pub fn lock_bucket_for_write(&self, bucket: u32) -> Result<parking_lot::MutexGuard<'_, ()>> {
        self.bucket_write_locks
            .get(bucket as usize)
            .map(|l| l.lock())
            .ok_or(ShardedValueLogError::InvalidBucket(bucket))
    }

    /// Any bucket whose live/garbage accounting could not be loaded and needs an
    /// exact rebuild from the LSM's pointers.
    pub fn buckets_needing_stat_rebuild(&self) -> Vec<u32> {
        (0..self.num_buckets as u32)
            .filter(|&b| self.logs[b as usize].stats_need_rebuild())
            .collect()
    }

    pub fn rebuild_bucket_stats(&self, bucket: u32, live: &HashMap<u32, u64>) -> Result<()> {
        self.get_bucket_log(bucket)?.rebuild_stats(live);
        Ok(())
    }

    // ── Writes ────────────────────────────────────────────────────────────

    /// Append a record to `bucket`. The caller must already hold that bucket's write
    /// lock (see [`lock_bucket_for_write`](Self::lock_bucket_for_write)).
    pub fn append_to_locked_bucket(&self, bucket: u32, key: &[u8], value: &[u8], meta: ValueRecordMeta, sync: bool) -> Result<ShardedValuePointer> {
        let location = self.get_bucket_log(bucket)?.append(key, value, meta, sync)?;
        ShardedValuePointer::new(bucket, location, self.num_buckets)
    }

    /// Record that a write or delete displaced `pointer`'s record: its bytes become
    /// garbage in its segment. No I/O.
    pub fn note_displaced(&self, pointer: ShardedValuePointer, key_len: usize) {
        if let Ok(log) = self.get_bucket_log(pointer.bucket) {
            log.note_displaced(pointer.location, key_len);
        }
    }

    // ── Reads ─────────────────────────────────────────────────────────────

    pub fn read_value(&self, pointer: ShardedValuePointer) -> Result<Vec<u8>> {
        Ok(self.get_bucket_log(pointer.bucket)?.read_value(pointer.location)?)
    }

    pub fn read_record_meta(&self, pointer: ShardedValuePointer) -> Result<ValueRecordMeta> {
        Ok(self.get_bucket_log(pointer.bucket)?.read_record_meta(pointer.location)?)
    }

    /// Batch-read values that all live in one bucket.
    pub fn read_values_batch(&self, bucket: u32, entries: &[(usize, ValueLocation)]) -> Vec<BatchValue> {
        match self.get_bucket_log(bucket) {
            Ok(log) => log.read_values_batch(entries),
            Err(_) => entries.iter().map(|(idx, _)| (*idx, None)).collect(),
        }
    }

    // ── Durability & stats ────────────────────────────────────────────────

    pub fn sync_all(&self) -> Result<()> {
        for log in &self.logs {
            log.sync()?;
        }
        Ok(())
    }

    pub fn flush_all_metadata(&self) -> Result<()> {
        for log in &self.logs {
            log.flush_metadata()?;
        }
        Ok(())
    }

    pub fn set_verify_checksums_on_read(&self, verify: bool) {
        for log in &self.logs {
            log.set_verify_checksums_on_read(verify);
        }
    }

    /// `(bucket, metadata)` for every bucket.
    pub fn all_bucket_metadata(&self) -> Vec<(u32, ValueLogMetadata)> {
        self.logs.iter().enumerate().map(|(b, log)| (b as u32, log.metadata_snapshot())).collect()
    }

    /// `(bucket, segments)` for every bucket — the per-segment live/garbage view GC
    /// selects from, and what `/admin/storage/value-log/{ns}/segments` reports.
    pub fn all_segment_stats(&self) -> Vec<(u32, Vec<SegmentStats>)> {
        self.logs.iter().enumerate().map(|(b, log)| (b as u32, log.segment_stats())).collect()
    }

    pub fn physical_stats(&self) -> Vec<ShardPhysicalStats> {
        self.logs
            .iter()
            .enumerate()
            .map(|(b, log)| ShardPhysicalStats {
                bucket: b as u32,
                physical_bytes: log.disk_bytes(),
                logical_bytes: log.metadata_snapshot().total_bytes(),
            })
            .collect()
    }

    /// Live bytes across all buckets.
    pub fn total_live_bytes(&self) -> u64 {
        self.logs.iter().map(|l| l.metadata_snapshot().live_bytes()).sum()
    }

    /// Garbage bytes across all buckets.
    pub fn total_garbage_bytes(&self) -> u64 {
        self.logs.iter().map(|l| l.metadata_snapshot().garbage_bytes()).sum()
    }

    /// Waste ratio across all buckets: `garbage / (live + garbage)`, as a percentage.
    /// This is the **trigger** GC compares against `value_log_waste_threshold`.
    pub fn total_garbage_ratio(&self) -> f64 {
        let garbage = self.total_garbage_bytes();
        let total = garbage + self.total_live_bytes();
        if total == 0 {
            return 0.0;
        }
        (garbage as f64 / total as f64) * 100.0
    }

    /// True if **any single bucket** is at or above `threshold_pct` waste. The
    /// cross-bucket [`total_garbage_ratio`](Self::total_garbage_ratio) is an average, so a
    /// bucket well over the trigger can hide behind near-empty ones; this catches that.
    pub fn any_bucket_over_waste(&self, threshold_pct: f64) -> bool {
        self.logs.iter().any(|l| l.waste_ratio_pct() >= threshold_pct)
    }
}
