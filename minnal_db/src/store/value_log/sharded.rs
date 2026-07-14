//! Sharded Value Log Module
//!
//! Implements a sharded value log that stores values in NUM_BUCKETS buckets
//! based on the hash of the key. Each bucket is backed by a single
//! value log file. The bucket determination is dynamic via hashing.

use super::{DEFAULT_PAGE_SIZE_BYTES, PageGarbageStats, ValueLocation, ValueLog, ValueLogMetadata, ValueRecordMeta};
use crate::support;
use parking_lot::{Mutex, RwLock};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

#[allow(dead_code)]
fn current_epoch_millis() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

#[derive(Error, Debug)]
pub enum ShardedValueLogError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Value log error: {0}")]
    ValueLogError(#[from] super::ValueLogError),
    #[error("Invalid bucket index: {0}")]
    InvalidBucket(u32),
}

pub type Result<T> = std::result::Result<T, ShardedValueLogError>;

/// Pointer to a value in the sharded value log
/// Includes bucket information for routing reads
#[derive(Debug, Clone, Copy)]
pub struct ShardedValuePointer {
    pub bucket: u32,
    pub location: ValueLocation,
}

impl ShardedValuePointer {
    pub fn new(bucket: u32, page_offset: u64, segment_id: u32, num_buckets: usize) -> Result<Self> {
        if bucket >= num_buckets as u32 {
            return Err(ShardedValueLogError::InvalidBucket(bucket));
        }
        Ok(Self {
            bucket,
            location: ValueLocation { page_offset, segment_id },
        })
    }

    /// Encode as u128 for efficient storage in skip_list-based LSM
    /// Format: [bucket: 32 bits][page_offset: 64 bits][segment_id: 32 bits]
    pub fn to_u128(self) -> u128 {
        let bucket_bits = (self.bucket as u128) << 96;
        let page_bits = (self.location.page_offset as u128) << 32;
        let segment_bits = self.location.segment_id as u128;
        bucket_bits | page_bits | segment_bits
    }

    /// Decode from u128
    pub fn from_u128(encoded: u128) -> Result<Self> {
        let bucket = ((encoded >> 96) & 0xFFFF_FFFF) as u32;
        let page_offset = ((encoded >> 32) & 0xFFFF_FFFF_FFFF_FFFF) as u64;
        let segment_id = (encoded & 0xFFFF_FFFF) as u32;

        Ok(Self {
            bucket,
            location: ValueLocation { page_offset, segment_id },
        })
    }

    /// Serialize to bytes for storage in LSM
    #[allow(dead_code)]
    pub fn to_bytes(self) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(16);
        bytes.extend_from_slice(&self.bucket.to_le_bytes());
        bytes.extend_from_slice(&self.location.page_offset.to_le_bytes());
        bytes.extend_from_slice(&self.location.segment_id.to_le_bytes());
        Ok(bytes)
    }

    /// Deserialize from bytes
    #[allow(dead_code)]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 16 {
            return Err(ShardedValueLogError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Empty bytes",
            )));
        }

        let bucket = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let page_offset = u64::from_le_bytes(bytes[4..12].try_into().unwrap());
        let segment_id = u32::from_le_bytes(bytes[12..16].try_into().unwrap());

        Ok(Self {
            bucket,
            location: ValueLocation { page_offset, segment_id },
        })
    }
}

/// Physical on-disk footprint of one value-log shard. `physical_bytes` is the
/// blocks actually allocated (sparse holes from copying GC excluded);
/// `logical_bytes` is the file length (holes included).
#[derive(Debug, Clone)]
pub struct ShardPhysicalStats {
    pub bucket: u32,
    pub physical_bytes: u64,
    pub logical_bytes: u64,
}

/// Sharded value log manager
///
/// Maintains `num_buckets` value log files, one per bucket.
/// Bucket assignment is determined dynamically via key hashing.
pub struct ShardedValueLog {
    // Vec of value logs, one per bucket
    logs: Vec<Arc<ValueLog>>,

    // Vec of metadata, one per bucket
    metadata: Vec<Arc<RwLock<ValueLogMetadata>>>,

    // Per-bucket write/GC lock to prevent file swaps during appends
    bucket_write_locks: Vec<Arc<Mutex<()>>>,

    // Number of sharding buckets
    num_buckets: usize,

    // Page size of every bucket file. Fixed at creation (see `ValueLog::page_size`).
    page_size: u64,

    // Base directory path
    base_path: PathBuf,
}

impl ShardedValueLog {
    /// Open or create a sharded value log with the given number of buckets and
    /// the default page size.
    #[allow(dead_code)] // test/convenience constructor; production opens with the configured size
    pub fn open<P: AsRef<Path>>(base_path: P, num_buckets: usize) -> Result<Self> {
        Self::open_with_page_size(base_path, num_buckets, DEFAULT_PAGE_SIZE_BYTES)
    }

    /// Open or create a sharded value log whose bucket files use `page_size` pages.
    ///
    /// Creates `num_buckets` value log files (one per bucket) in the given directory.
    /// The page size is fixed at creation: reopening buckets that already hold data
    /// with a different size fails with [`ValueLogError::PageSizeMismatch`].
    pub fn open_with_page_size<P: AsRef<Path>>(base_path: P, num_buckets: usize, page_size: u64) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_path)?;

        let mut logs: Vec<Arc<ValueLog>> = Vec::with_capacity(num_buckets);
        let mut metadata: Vec<Arc<RwLock<ValueLogMetadata>>> = Vec::with_capacity(num_buckets);
        let mut bucket_write_locks: Vec<Arc<Mutex<()>>> = Vec::with_capacity(num_buckets);

        for bucket in 0u32..num_buckets as u32 {
            let bucket_path = base_path.join(format!("value_log_{}.log", bucket));
            let metadata_path = base_path.join(format!("value_log_{}.metadata", bucket));

            let log = ValueLog::open_with_page_size(&bucket_path, page_size)?;

            // Load metadata, or rebuild it from the value-log pages when it is
            // missing or corrupt. Resetting to `ValueLogMetadata::new(DEFAULT_PAGE_SIZE_BYTES)` would
            // zero the live/garbage counters (GC would never trigger) and the
            // page cursors (a later write could land a new page over existing
            // data) — `reconstruct_metadata` recovers both by scanning the file.
            let mut meta = if metadata_path.exists() {
                let data = std::fs::read(&metadata_path)?;
                match ValueLogMetadata::from_file_bytes(&data, page_size) {
                    Ok(metadata) => metadata,
                    Err(_) => {
                        let backup_path = metadata_path.with_extension("corrupt");
                        let _ = std::fs::rename(&metadata_path, &backup_path);
                        log.reconstruct_metadata().map_err(ShardedValueLogError::ValueLogError)?
                    }
                }
            } else {
                log.reconstruct_metadata().map_err(ShardedValueLogError::ValueLogError)?
            };

            log.ensure_current_page(&mut meta).map_err(ShardedValueLogError::ValueLogError)?;

            logs.push(Arc::new(log));
            metadata.push(Arc::new(RwLock::new(meta)));
            bucket_write_locks.push(Arc::new(Mutex::new(())));
        }

        Ok(Self {
            logs,
            metadata,
            bucket_write_locks,
            num_buckets,
            page_size,
            base_path,
        })
    }

    /// Page size of every bucket file (fixed at creation).
    #[inline]
    pub fn page_size(&self) -> u64 {
        self.page_size
    }

    /// Write a value to the appropriate bucket
    /// Returns a ShardedValuePointer that can be stored in LSM
    #[allow(dead_code)]
    pub fn write_value(&self, key: &[u8], value: &[u8], sync: bool) -> Result<ShardedValuePointer> {
        let bucket = support::get_bucket_for_key(key, self.num_buckets);
        let meta = ValueRecordMeta {
            version: 1,
            tombstone: false,
            updated: false,
            epoch: current_epoch_millis(),
            seq: 0,
        };
        self.write_record_with_meta(bucket, value, meta, sync)
    }

    /// Returns the bucket index for a key.
    pub fn bucket_for_key(&self, key: &[u8]) -> u32 {
        support::get_bucket_for_key(key, self.num_buckets)
    }

    pub fn write_record(&self, key: &[u8], value: &[u8], meta: ValueRecordMeta, sync: bool) -> Result<ShardedValuePointer> {
        let bucket = support::get_bucket_for_key(key, self.num_buckets);
        self.write_record_with_meta(bucket, value, meta, sync)
    }

    /// Write a record to a specific bucket without acquiring the bucket lock.
    /// The caller must already hold the lock returned by `lock_bucket_for_write(bucket)`,
    /// so that the value-log write and the subsequent LSM insert are atomic with respect to GC.
    pub fn write_record_to_locked_bucket(&self, bucket: u32, value: &[u8], meta: ValueRecordMeta, sync: bool) -> Result<ShardedValuePointer> {
        let mut metadata = self.metadata[bucket as usize].write();
        let location = self.logs[bucket as usize]
            .write_record(value, meta, &mut metadata, sync)
            .map_err(ShardedValueLogError::ValueLogError)?;
        drop(metadata);
        ShardedValuePointer::new(bucket, location.page_offset, location.segment_id, self.num_buckets)
    }

    fn write_record_with_meta(&self, bucket: u32, value: &[u8], meta: ValueRecordMeta, sync: bool) -> Result<ShardedValuePointer> {
        let _bucket_guard = self.bucket_write_locks[bucket as usize].lock();
        let mut metadata = self.metadata[bucket as usize].write();
        let location = self.logs[bucket as usize]
            .write_record(value, meta, &mut metadata, sync)
            .map_err(ShardedValueLogError::ValueLogError)?;
        drop(metadata);

        ShardedValuePointer::new(bucket, location.page_offset, location.segment_id, self.num_buckets)
    }

    /// Read a value using a ShardedValuePointer
    pub fn read_value(&self, pointer: ShardedValuePointer) -> Result<Vec<u8>> {
        Ok(self.read_value_with_seq(pointer)?.0)
    }

    /// Like [`read_value`](Self::read_value) but also returns the record's stored
    /// write `seq` for the read-time stale-pointer validity check.
    pub fn read_value_with_seq(&self, pointer: ShardedValuePointer) -> Result<(Vec<u8>, u64)> {
        let bucket = pointer.bucket as usize;
        if bucket >= self.num_buckets {
            return Err(ShardedValueLogError::InvalidBucket(pointer.bucket));
        }

        let file = self.logs[bucket].get_file();
        self.logs[bucket]
            .read_value_and_seq_from_file(&file, pointer.location)
            .map_err(ShardedValueLogError::ValueLogError)
    }

    /// Snapshot current file handles for all buckets
    #[allow(dead_code)]
    pub fn snapshot_bucket_files(&self) -> Vec<Arc<File>> {
        let mut files = Vec::with_capacity(self.num_buckets);
        for bucket in 0u32..self.num_buckets as u32 {
            files.push(self.logs[bucket as usize].get_file());
        }
        files
    }

    /// Read a value using a specific file handle (snapshot read)
    pub fn read_value_with_file(&self, pointer: ShardedValuePointer, file: &File) -> Result<Vec<u8>> {
        let bucket = pointer.bucket as usize;
        if bucket >= self.num_buckets {
            return Err(ShardedValueLogError::InvalidBucket(pointer.bucket));
        }

        self.logs[bucket]
            .read_value_from_file(file, pointer.location)
            .map_err(ShardedValueLogError::ValueLogError)
    }

    /// Read multiple values from one bucket using batch pread optimisation.
    ///
    /// All entries must belong to the same bucket; the caller is responsible for
    /// grouping them before calling this.  Uses the file handle from the first entry
    /// (consistent with the snapshot-read model used by `read_value_with_file`).
    pub fn read_values_batch(&self, entries: &[(usize, ShardedValuePointer, std::sync::Arc<File>)]) -> Vec<super::BatchValue> {
        let Some((_, first_ptr, first_file)) = entries.first() else {
            return vec![];
        };
        let bucket = first_ptr.bucket as usize;
        if bucket >= self.num_buckets {
            return entries.iter().map(|(idx, _, _)| (*idx, None)).collect();
        }
        let locations: Vec<(usize, ValueLocation)> = entries.iter().map(|(idx, ptr, _)| (*idx, ptr.location)).collect();
        self.logs[bucket].read_values_batch_from_file(first_file, &locations)
    }

    pub fn read_record_meta(&self, pointer: ShardedValuePointer) -> Result<ValueRecordMeta> {
        let bucket = pointer.bucket as usize;
        if bucket >= self.num_buckets {
            return Err(ShardedValueLogError::InvalidBucket(pointer.bucket));
        }
        let file = self.logs[bucket].get_file();
        self.logs[bucket]
            .read_record_meta_from_file(&file, pointer.location)
            .map_err(ShardedValueLogError::ValueLogError)
    }

    pub fn update_record_meta(
        &self,
        pointer: ShardedValuePointer,
        tombstone: Option<bool>,
        updated: Option<bool>,
        epoch: Option<u64>,
    ) -> Result<ValueRecordMeta> {
        let bucket = pointer.bucket as usize;
        if bucket >= self.num_buckets {
            return Err(ShardedValueLogError::InvalidBucket(pointer.bucket));
        }
        let mut metadata = self.metadata[bucket].write();
        let (previous_meta, updated_meta, record_bytes) = self.logs[bucket]
            .update_record_meta(pointer.location, tombstone, updated, epoch)
            .map_err(ShardedValueLogError::ValueLogError)?;

        let was_obsolete = previous_meta.tombstone || previous_meta.updated;
        let is_obsolete = updated_meta.tombstone || updated_meta.updated;
        if was_obsolete != is_obsolete {
            if is_obsolete {
                metadata.live_bytes = metadata.live_bytes.saturating_sub(record_bytes);
                metadata.garbage_bytes = metadata.garbage_bytes.saturating_add(record_bytes);
            } else {
                metadata.garbage_bytes = metadata.garbage_bytes.saturating_sub(record_bytes);
                metadata.live_bytes = metadata.live_bytes.saturating_add(record_bytes);
            }
        }

        Ok(updated_meta)
    }

    #[allow(dead_code)]
    fn used_capacity_bytes(&self, metadata: &ValueLogMetadata) -> u64 {
        let used_in_current = metadata.current_page_free_offset as u64 + (self.page_size.saturating_sub(metadata.current_page_table_offset as u64));
        metadata.current_page_offset.saturating_add(used_in_current)
    }

    #[allow(dead_code)]
    fn garbage_ratio(metadata: &ValueLogMetadata) -> f64 {
        let total_written = metadata.live_bytes.saturating_add(metadata.garbage_bytes);
        if total_written > 0 {
            (metadata.garbage_bytes as f64 / total_written as f64) * 100.0
        } else {
            0.0
        }
    }

    #[allow(dead_code)]
    fn free_space_ratio(&self, metadata: &ValueLogMetadata) -> f64 {
        let allocated = metadata.tail;
        if allocated == 0 {
            return 0.0;
        }
        let used = self.used_capacity_bytes(metadata).min(allocated);
        let free = allocated.saturating_sub(used);
        (free as f64 / allocated as f64) * 100.0
    }

    /// Get garbage ratio for a specific bucket (for targeted GC)
    #[allow(dead_code)]
    pub fn get_bucket_garbage_ratio(&self, bucket: u32) -> Result<f64> {
        if bucket as usize >= self.num_buckets {
            return Err(ShardedValueLogError::InvalidBucket(bucket));
        }

        let metadata = self.metadata[bucket as usize].read();
        Ok(Self::garbage_ratio(&metadata))
    }

    /// Get free space ratio for a specific bucket
    #[allow(dead_code)]
    pub fn get_bucket_free_space_ratio(&self, bucket: u32) -> Result<f64> {
        if bucket as usize >= self.num_buckets {
            return Err(ShardedValueLogError::InvalidBucket(bucket));
        }

        let metadata = self.metadata[bucket as usize].read();
        Ok(self.free_space_ratio(&metadata))
    }

    /// Backward-compat alias for garbage ratio.
    #[allow(dead_code)]
    pub fn get_bucket_waste_ratio(&self, bucket: u32) -> Result<f64> {
        self.get_bucket_garbage_ratio(bucket)
    }

    /// Get garbage ratio across all buckets (weighted average)
    pub fn get_total_garbage_ratio(&self) -> f64 {
        let mut total_garbage = 0u64;
        let mut total_written = 0u64;

        for bucket in 0u32..self.num_buckets as u32 {
            let metadata = self.metadata[bucket as usize].read();
            total_garbage = total_garbage.saturating_add(metadata.garbage_bytes);
            total_written = total_written.saturating_add(metadata.live_bytes.saturating_add(metadata.garbage_bytes));
        }

        if total_written > 0 {
            (total_garbage as f64 / total_written as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Get free space ratio across all buckets (weighted by allocated capacity)
    #[allow(dead_code)]
    pub fn get_total_free_space_ratio(&self) -> f64 {
        let mut total_free = 0u64;
        let mut total_allocated = 0u64;

        for bucket in 0u32..self.num_buckets as u32 {
            let metadata = self.metadata[bucket as usize].read();
            let allocated = metadata.tail;
            let used = self.used_capacity_bytes(&metadata).min(allocated);
            total_free = total_free.saturating_add(allocated.saturating_sub(used));
            total_allocated = total_allocated.saturating_add(allocated);
        }

        if total_allocated > 0 {
            (total_free as f64 / total_allocated as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Backward-compat alias for garbage ratio.
    #[allow(dead_code)]
    pub fn get_total_waste_ratio(&self) -> f64 {
        self.get_total_garbage_ratio()
    }

    /// Enable or disable re-verifying each value's CRC32 on read across all
    /// buckets (default off). See `DbConfig::verify_checksums_on_read`.
    pub fn set_verify_checksums_on_read(&self, verify: bool) {
        for log in &self.logs {
            log.set_verify_checksums_on_read(verify);
        }
    }

    /// Sync all buckets to disk
    pub fn sync(&self) -> Result<()> {
        for log in &self.logs {
            log.sync().map_err(ShardedValueLogError::ValueLogError)?;
        }
        Ok(())
    }

    /// Flush metadata for all buckets.
    ///
    /// Each bucket's metadata is written crash-atomically via
    /// [`write_atomic_durable`](crate::support::write_atomic_durable): the bytes
    /// go to a fsynced temp file that is renamed over the live file, and the
    /// parent directory is fsynced so the rename itself is durable. A crash
    /// mid-write can never leave a torn/partial metadata file — the reader sees
    /// either the previous complete version or the new one. (A corrupt file
    /// would otherwise force a full reset to `ValueLogMetadata::new(DEFAULT_PAGE_SIZE_BYTES)` on open;
    /// see `open`.)
    pub fn flush_all_metadata(&self) -> Result<()> {
        for bucket in 0u32..self.num_buckets as u32 {
            let metadata = self.metadata[bucket as usize].read();
            let bytes = metadata.to_file_bytes().map_err(ShardedValueLogError::ValueLogError)?;
            let metadata_path = self.base_path.join(format!("value_log_{}.metadata", bucket));
            crate::support::write_atomic_durable(&metadata_path, &bytes)?;
        }
        Ok(())
    }

    /// Get metadata for a specific bucket
    pub fn get_bucket_metadata(&self, bucket: u32) -> Result<ValueLogMetadata> {
        if bucket as usize >= self.num_buckets {
            return Err(ShardedValueLogError::InvalidBucket(bucket));
        }

        let metadata = self.metadata[bucket as usize].read();
        Ok(ValueLogMetadata {
            head: metadata.head,
            tail: metadata.tail,
            current_page_offset: metadata.current_page_offset,
            current_page_free_offset: metadata.current_page_free_offset,
            current_page_table_offset: metadata.current_page_table_offset,
            current_page_next_segment_id: metadata.current_page_next_segment_id,
            total_gc_runs: metadata.total_gc_runs,
            total_bytes_reclaimed: metadata.total_bytes_reclaimed,
            live_bytes: metadata.live_bytes,
            garbage_bytes: metadata.garbage_bytes,
        })
    }

    /// Get all bucket statistics
    pub fn get_all_bucket_stats(&self) -> Vec<(u32, ValueLogMetadata)> {
        let mut stats = Vec::with_capacity(self.num_buckets);
        for bucket in 0u32..self.num_buckets as u32 {
            if let Ok(meta) = self.get_bucket_metadata(bucket) {
                stats.push((bucket, meta));
            }
        }
        stats
    }

    /// Get the base path
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Get number of buckets
    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }

    pub fn get_bucket_log(&self, bucket: u32) -> Result<Arc<ValueLog>> {
        if bucket as usize >= self.num_buckets {
            return Err(ShardedValueLogError::InvalidBucket(bucket));
        }
        Ok(self.logs[bucket as usize].clone())
    }

    /// Physical (on-disk blocks) vs logical (file length) footprint per shard.
    /// The copying GC leaves sparse holes, so `physical_bytes` (from `st_blocks`)
    /// can be far below `logical_bytes` — the physical figure is the true disk
    /// usage. Cheap: one `stat` per shard, no page scan.
    pub fn physical_stats(&self) -> Vec<ShardPhysicalStats> {
        use std::os::unix::fs::MetadataExt;
        (0..self.num_buckets as u32)
            .filter_map(|bucket| {
                let md = std::fs::metadata(self.logs[bucket as usize].path()).ok()?;
                Some(ShardPhysicalStats {
                    bucket,
                    physical_bytes: md.blocks() * 512,
                    logical_bytes: md.len(),
                })
            })
            .collect()
    }

    /// Per-page garbage breakdown for one shard (where waste concentrates).
    /// Cost is O(pages × records) — scope it to a single namespace/bucket.
    pub fn page_stats(&self, bucket: u32) -> Result<Vec<PageGarbageStats>> {
        let log = self.get_bucket_log(bucket)?;
        let file = log.get_file();
        let metadata = self.get_bucket_metadata(bucket)?;
        Ok(log.scan_all_page_stats(&file, &metadata)?)
    }

    /// Current swap generation for a bucket.
    ///
    /// Readers sample this before and after a pointer+value read to detect a
    /// GC file swap that raced with the read. Returns 0 for an out-of-range
    /// bucket (callers validate the bucket elsewhere; a stable 0 simply forces
    /// the consistency check to pass and the read to fail downstream).
    pub fn bucket_generation(&self, bucket: u32) -> u64 {
        if bucket as usize >= self.num_buckets {
            return 0;
        }
        self.logs[bucket as usize].generation()
    }

    pub fn update_bucket_metadata(&self, bucket: u32, new_metadata: ValueLogMetadata) -> Result<()> {
        if bucket as usize >= self.num_buckets {
            return Err(ShardedValueLogError::InvalidBucket(bucket));
        }
        let mut metadata = self.metadata[bucket as usize].write();
        *metadata = new_metadata;
        Ok(())
    }

    pub fn lock_bucket_for_write(&self, bucket: u32) -> Result<parking_lot::MutexGuard<'_, ()>> {
        if bucket as usize >= self.num_buckets {
            return Err(ShardedValueLogError::InvalidBucket(bucket));
        }
        Ok(self.bucket_write_locks[bucket as usize].lock())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::{DEFAULT_NUM_BUCKETS, get_bucket_for_key};
    use tempfile::TempDir;

    #[test]
    fn test_sharded_value_log_creation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let svl = ShardedValueLog::open(temp_dir.path(), DEFAULT_NUM_BUCKETS)?;
        assert_eq!(svl.num_buckets(), DEFAULT_NUM_BUCKETS);
        Ok(())
    }

    #[test]
    fn test_bucket_determination() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let _svl = ShardedValueLog::open(temp_dir.path(), DEFAULT_NUM_BUCKETS)?;

        // Same key should always get same bucket
        let bucket1 = get_bucket_for_key(b"test_key", DEFAULT_NUM_BUCKETS);
        let bucket2 = get_bucket_for_key(b"test_key", DEFAULT_NUM_BUCKETS);
        assert_eq!(bucket1, bucket2);

        // Different keys might get different buckets
        let bucket3 = get_bucket_for_key(b"other_key", DEFAULT_NUM_BUCKETS);
        // bucket3 might equal bucket1, but that's okay (hash collision)
        assert!(bucket3 < DEFAULT_NUM_BUCKETS as u32);

        Ok(())
    }

    #[test]
    fn write_returns_pointer_that_reads_back_original_value() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let svl = ShardedValueLog::open(temp_dir.path(), DEFAULT_NUM_BUCKETS)?;

        let key = b"mykey";
        let value = b"myvalue";

        let pointer = svl.write_value(key, value, false)?;
        let read_value = svl.read_value(pointer)?;

        assert_eq!(read_value, value.to_vec());
        Ok(())
    }

    #[test]
    fn rebuilds_metadata_counters_when_metadata_file_is_lost() -> Result<()> {
        let temp = TempDir::new()?;
        let key = b"some_key";
        let bucket = get_bucket_for_key(key, DEFAULT_NUM_BUCKETS);

        // Write a value, flush metadata, and capture the bucket's live byte count.
        let (pointer, live_before) = {
            let svl = ShardedValueLog::open(temp.path(), DEFAULT_NUM_BUCKETS)?;
            let pointer = svl.write_value(key, &[7u8; 200], true)?;
            svl.flush_all_metadata()?;
            (pointer, svl.get_bucket_metadata(bucket)?.live_bytes)
        };
        assert!(live_before > 0);

        // Simulate metadata loss for that bucket.
        std::fs::remove_file(temp.path().join(format!("value_log_{}.metadata", bucket)))?;

        // Reopen: the counters are rebuilt from the value-log pages (so GC's waste
        // ratio is correct again) and the value is still readable.
        let svl = ShardedValueLog::open(temp.path(), DEFAULT_NUM_BUCKETS)?;
        assert_eq!(svl.get_bucket_metadata(bucket)?.live_bytes, live_before, "live_bytes rebuilt from pages");
        assert_eq!(svl.read_value(pointer)?, vec![7u8; 200], "value still readable after metadata loss");
        Ok(())
    }

    #[test]
    fn test_multiple_values_different_buckets() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let svl = ShardedValueLog::open(temp_dir.path(), DEFAULT_NUM_BUCKETS)?;

        // Write multiple values with different keys
        let mut pointers = Vec::new();
        for i in 0..10 {
            let key = format!("key_{}", i).into_bytes();
            let value = format!("value_{}", i).into_bytes();
            let pointer = svl.write_value(&key, &value, false)?;
            pointers.push((key, value, pointer));
        }

        // Verify all can be read back correctly
        for (_key, expected_value, pointer) in pointers {
            let read_value = svl.read_value(pointer)?;
            assert_eq!(read_value, expected_value);
        }

        Ok(())
    }

    #[test]
    fn test_waste_ratio() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let svl = ShardedValueLog::open(temp_dir.path(), DEFAULT_NUM_BUCKETS)?;

        let garbage_ratio = svl.get_total_garbage_ratio();
        let free_ratio = svl.get_total_free_space_ratio();
        assert!((0.0..=100.0).contains(&garbage_ratio));
        assert!((0.0..=100.0).contains(&free_ratio));

        svl.write_value(b"key1", b"value1", false)?;
        svl.write_value(b"key2", b"value2", false)?;

        let garbage_ratio = svl.get_total_garbage_ratio();
        let free_ratio = svl.get_total_free_space_ratio();
        assert!((0.0..100.0).contains(&garbage_ratio));
        assert!((0.0..=100.0).contains(&free_ratio));

        Ok(())
    }

    #[test]
    fn test_sharded_value_pointer_serialization() -> Result<()> {
        let pointer = ShardedValuePointer::new(5, 128, 42, DEFAULT_NUM_BUCKETS)?;

        let bytes = pointer.to_bytes()?;
        let restored = ShardedValuePointer::from_bytes(&bytes)?;

        assert_eq!(restored.bucket, pointer.bucket);
        assert_eq!(restored.location.page_offset, pointer.location.page_offset);
        assert_eq!(restored.location.segment_id, pointer.location.segment_id);

        Ok(())
    }

    #[test]
    fn test_invalid_bucket() {
        let result = ShardedValuePointer::new(DEFAULT_NUM_BUCKETS as u32, 0, 0, DEFAULT_NUM_BUCKETS);
        assert!(result.is_err());
    }

    #[test]
    fn test_garbage_ratio_increases_on_update_and_delete() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let svl = ShardedValueLog::open(temp_dir.path(), DEFAULT_NUM_BUCKETS)?;

        let ptr1 = svl.write_value(b"k1", b"value1", false)?;
        let ratio_initial = svl.get_total_garbage_ratio();

        svl.update_record_meta(ptr1, None, Some(true), Some(current_epoch_millis()))?;
        let ratio_after_update = svl.get_total_garbage_ratio();
        assert!(ratio_after_update > ratio_initial);

        let ptr2 = svl.write_value(b"k2", b"value2", false)?;
        let ratio_before_delete = svl.get_total_garbage_ratio();

        svl.update_record_meta(ptr2, Some(true), None, Some(current_epoch_millis()))?;
        let ratio_after_delete = svl.get_total_garbage_ratio();
        assert!(ratio_after_delete > ratio_before_delete);

        Ok(())
    }

    #[test]
    fn test_bucket_for_key_deterministic() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let svl = ShardedValueLog::open(temp_dir.path(), DEFAULT_NUM_BUCKETS)?;

        // Same key must always map to the same bucket.
        let b1 = svl.bucket_for_key(b"hello");
        let b2 = svl.bucket_for_key(b"hello");
        assert_eq!(b1, b2);
        assert!((b1 as usize) < DEFAULT_NUM_BUCKETS);
        Ok(())
    }

    #[test]
    fn test_write_record_to_locked_bucket_readable() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let svl = ShardedValueLog::open(temp_dir.path(), DEFAULT_NUM_BUCKETS)?;

        let key = b"locked_key";
        let value = b"locked_value";
        let bucket = svl.bucket_for_key(key);

        let _guard = svl.lock_bucket_for_write(bucket)?;
        let meta = ValueRecordMeta {
            version: 1,
            tombstone: false,
            updated: false,
            epoch: 0,
            seq: 0,
        };
        let ptr = svl.write_record_to_locked_bucket(bucket, value, meta, false)?;
        drop(_guard);

        assert_eq!(ptr.bucket, bucket);
        assert_eq!(svl.read_value(ptr)?, value.to_vec());
        Ok(())
    }

    #[test]
    fn test_write_record_locked_matches_write_record() -> Result<()> {
        // write_record_to_locked_bucket and write_record must produce
        // records readable through the same path.
        let temp_dir = TempDir::new()?;
        let svl = ShardedValueLog::open(temp_dir.path(), DEFAULT_NUM_BUCKETS)?;

        let meta = ValueRecordMeta {
            version: 1,
            tombstone: false,
            updated: false,
            epoch: 0,
            seq: 0,
        };

        let ptr_unlocked = svl.write_record(b"key_a", b"val_a", meta, false)?;

        let bucket = svl.bucket_for_key(b"key_b");
        let _guard = svl.lock_bucket_for_write(bucket)?;
        let ptr_locked = svl.write_record_to_locked_bucket(bucket, b"val_b", meta, false)?;
        drop(_guard);

        assert_eq!(svl.read_value(ptr_unlocked)?, b"val_a".to_vec());
        assert_eq!(svl.read_value(ptr_locked)?, b"val_b".to_vec());
        Ok(())
    }
}
