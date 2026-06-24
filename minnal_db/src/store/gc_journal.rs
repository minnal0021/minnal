//! GC Journal — crash-recovery log for value log garbage collection.
//!
//! Written **before** the value log file swap during GC and deleted **after**
//! LSM pointer updates are applied. If the process crashes between the file
//! swap and the LSM updates, the journal is replayed at startup to fix any
//! dangling pointers.
//!
//! Format (simple binary, no rkyv — crash recovery must be self-contained):
//! ```text
//! [magic: "GCJN" (4 bytes)]
//! [version: u32  (4 bytes)]
//! [bucket:  u32  (4 bytes)]
//! [checksum: u32 (4 bytes)]  — CRC32 over the entries payload
//! [num_entries: u32 (4 bytes)]
//! entries: [{
//!     key_len:         u32   (4 bytes)
//!     key:             [u8]  (key_len bytes)
//!     new_bucket:      u32   (4 bytes)
//!     new_page_offset: u64   (8 bytes)
//!     new_segment_id:  u32   (4 bytes)
//! }]
//! ```

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const GC_JOURNAL_MAGIC: [u8; 4] = *b"GCJN";
const GC_JOURNAL_VERSION: u32 = 1;
const GC_JOURNAL_HEADER_SIZE: usize = 4 + 4 + 4 + 4 + 4; // magic + version + bucket + checksum + num_entries

/// fsync a directory so that a create/rename/unlink of one of its entries is
/// durable (the file's own `sync_all` only persists its *contents*, not the
/// directory entry that makes it visible after a crash).
pub fn fsync_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::File::open(dir)?.sync_all()
}

/// A single entry in the GC journal: maps a key to its new value-log pointer.
#[derive(Debug, Clone)]
pub struct GCJournalEntry {
    pub key: Vec<u8>,
    pub new_bucket: u32,
    pub new_page_offset: u64,
    pub new_segment_id: u32,
}

/// Writes and reads GC journal files for crash recovery.
pub struct GCJournal;

impl GCJournal {
    /// Path to the journal file for a given bucket.
    pub fn journal_path(base_path: &Path, bucket: u32) -> PathBuf {
        base_path.join(format!("gc_journal_{}.bin", bucket))
    }

    /// Write a journal file for the given bucket. The file is fsynced before returning.
    pub fn write(
        base_path: &Path,
        bucket: u32,
        entries: &[(Vec<u8>, u32, u64, u32)], // (key, new_bucket, new_page_offset, new_segment_id)
    ) -> std::io::Result<PathBuf> {
        let path = Self::journal_path(base_path, bucket);

        // Serialize entries payload first so we can compute checksum
        let mut payload = Vec::new();
        for (key, new_bucket, new_page_offset, new_segment_id) in entries {
            payload.extend_from_slice(&(key.len() as u32).to_le_bytes());
            payload.extend_from_slice(key);
            payload.extend_from_slice(&new_bucket.to_le_bytes());
            payload.extend_from_slice(&new_page_offset.to_le_bytes());
            payload.extend_from_slice(&new_segment_id.to_le_bytes());
        }

        let checksum = {
            let mut hasher = crc32fast::Hasher::new();
            hasher.update(&payload);
            hasher.finalize()
        };

        let mut file = OpenOptions::new().write(true).create(true).truncate(true).open(&path)?;

        file.write_all(&GC_JOURNAL_MAGIC)?;
        file.write_all(&GC_JOURNAL_VERSION.to_le_bytes())?;
        file.write_all(&bucket.to_le_bytes())?;
        file.write_all(&checksum.to_le_bytes())?;
        file.write_all(&(entries.len() as u32).to_le_bytes())?;
        file.write_all(&payload)?;
        file.sync_all()?;

        Ok(path)
    }

    /// Read and validate a journal file. Returns `None` if the file is corrupt or invalid.
    pub fn read(path: &Path) -> Option<(u32, Vec<GCJournalEntry>)> {
        let data = fs::read(path).ok()?;
        if data.len() < GC_JOURNAL_HEADER_SIZE {
            return None;
        }

        // Validate magic
        if data[0..4] != GC_JOURNAL_MAGIC {
            return None;
        }

        // Validate version
        let version = u32::from_le_bytes(data[4..8].try_into().ok()?);
        if version != GC_JOURNAL_VERSION {
            return None;
        }

        let bucket = u32::from_le_bytes(data[8..12].try_into().ok()?);
        let stored_checksum = u32::from_le_bytes(data[12..16].try_into().ok()?);
        let num_entries = u32::from_le_bytes(data[16..20].try_into().ok()?);

        let payload = &data[GC_JOURNAL_HEADER_SIZE..];

        // Validate checksum over the entries payload
        let computed_checksum = {
            let mut hasher = crc32fast::Hasher::new();
            hasher.update(payload);
            hasher.finalize()
        };
        if computed_checksum != stored_checksum {
            return None;
        }

        // Parse entries
        let mut entries = Vec::with_capacity(num_entries as usize);
        let mut offset = 0usize;

        for _ in 0..num_entries {
            if offset + 4 > payload.len() {
                return None;
            }
            let key_len = u32::from_le_bytes(payload[offset..offset + 4].try_into().ok()?) as usize;
            offset += 4;

            if offset + key_len + 4 + 8 + 4 > payload.len() {
                return None;
            }
            let key = payload[offset..offset + key_len].to_vec();
            offset += key_len;

            let new_bucket = u32::from_le_bytes(payload[offset..offset + 4].try_into().ok()?);
            offset += 4;

            let new_page_offset = u64::from_le_bytes(payload[offset..offset + 8].try_into().ok()?);
            offset += 8;

            let new_segment_id = u32::from_le_bytes(payload[offset..offset + 4].try_into().ok()?);
            offset += 4;

            entries.push(GCJournalEntry {
                key,
                new_bucket,
                new_page_offset,
                new_segment_id,
            });
        }

        Some((bucket, entries))
    }

    /// Delete the journal file for a given bucket.
    pub fn delete(base_path: &Path, bucket: u32) -> std::io::Result<()> {
        let path = Self::journal_path(base_path, bucket);
        match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Path to the swap-commit marker for a given bucket.
    ///
    /// The marker is the **commit point** of a value-log GC swap: present means
    /// "the value-log file swap committed but its LSM pointer updates are not yet
    /// durable, so the journal must be applied (or, if unreadable, reverted)".
    /// It is written (fsynced, dir fsynced) before the file rename and deleted
    /// only after the LSM pointer updates have been flushed to disk.
    pub fn commit_marker_path(base_path: &Path, bucket: u32) -> PathBuf {
        base_path.join(format!("gc_commit_{}.marker", bucket))
    }

    /// Write the swap-commit marker for a bucket, fsyncing both the file and the
    /// containing directory so its presence survives a crash.
    pub fn write_commit_marker(base_path: &Path, bucket: u32) -> std::io::Result<()> {
        let path = Self::commit_marker_path(base_path, bucket);
        let file = OpenOptions::new().write(true).create(true).truncate(true).open(&path)?;
        file.sync_all()?;
        drop(file);
        fsync_dir(base_path)
    }

    /// Whether the swap-commit marker exists for a bucket.
    pub fn commit_marker_exists(base_path: &Path, bucket: u32) -> bool {
        Self::commit_marker_path(base_path, bucket).exists()
    }

    /// Delete the swap-commit marker for a bucket (idempotent), fsyncing the
    /// directory so the deletion is durable.
    pub fn delete_commit_marker(base_path: &Path, bucket: u32) -> std::io::Result<()> {
        let path = Self::commit_marker_path(base_path, bucket);
        match fs::remove_file(&path) {
            Ok(()) => fsync_dir(base_path),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Find all journal files in the base directory.
    pub fn find_journals(base_path: &Path) -> Vec<PathBuf> {
        let mut journals = Vec::new();
        if let Ok(entries) = fs::read_dir(base_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str())
                    && name.starts_with("gc_journal_")
                    && name.ends_with(".bin")
                {
                    journals.push(path);
                }
            }
        }
        journals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_journal_write_read_roundtrip() {
        let dir = TempDir::new().unwrap();
        let entries = vec![
            (b"key1".to_vec(), 0u32, 0u64, 1u32),
            (b"key2".to_vec(), 3u32, 67108864u64, 5u32),
            (b"longer_key_here".to_vec(), 15u32, 134217728u64, 42u32),
        ];

        let path = GCJournal::write(dir.path(), 7, &entries).unwrap();
        assert!(path.exists());

        let (bucket, read_entries) = GCJournal::read(&path).unwrap();
        assert_eq!(bucket, 7);
        assert_eq!(read_entries.len(), 3);

        assert_eq!(read_entries[0].key, b"key1");
        assert_eq!(read_entries[0].new_bucket, 0);
        assert_eq!(read_entries[0].new_page_offset, 0);
        assert_eq!(read_entries[0].new_segment_id, 1);

        assert_eq!(read_entries[1].key, b"key2");
        assert_eq!(read_entries[1].new_bucket, 3);
        assert_eq!(read_entries[1].new_page_offset, 67108864);
        assert_eq!(read_entries[1].new_segment_id, 5);

        assert_eq!(read_entries[2].key, b"longer_key_here".as_slice());
        assert_eq!(read_entries[2].new_bucket, 15);
        assert_eq!(read_entries[2].new_segment_id, 42);
    }

    #[test]
    fn test_journal_corrupt_checksum_rejected() {
        let dir = TempDir::new().unwrap();
        let entries = vec![(b"key".to_vec(), 0u32, 0u64, 1u32)];
        let path = GCJournal::write(dir.path(), 0, &entries).unwrap();

        // Corrupt one byte in the payload
        let mut data = fs::read(&path).unwrap();
        let last = data.len() - 1;
        data[last] ^= 0xFF;
        fs::write(&path, &data).unwrap();

        assert!(GCJournal::read(&path).is_none());
    }

    #[test]
    fn test_journal_truncated_rejected() {
        let dir = TempDir::new().unwrap();
        let entries = vec![(b"key".to_vec(), 0u32, 0u64, 1u32)];
        let path = GCJournal::write(dir.path(), 0, &entries).unwrap();

        // Truncate the file
        let data = fs::read(&path).unwrap();
        fs::write(&path, &data[..10]).unwrap();

        assert!(GCJournal::read(&path).is_none());
    }

    #[test]
    fn test_journal_empty_entries() {
        let dir = TempDir::new().unwrap();
        let entries: Vec<(Vec<u8>, u32, u64, u32)> = vec![];
        let path = GCJournal::write(dir.path(), 2, &entries).unwrap();

        let (bucket, read_entries) = GCJournal::read(&path).unwrap();
        assert_eq!(bucket, 2);
        assert!(read_entries.is_empty());
    }

    #[test]
    fn test_journal_delete() {
        let dir = TempDir::new().unwrap();
        let entries = vec![(b"key".to_vec(), 0u32, 0u64, 1u32)];
        GCJournal::write(dir.path(), 5, &entries).unwrap();

        GCJournal::delete(dir.path(), 5).unwrap();
        assert!(!GCJournal::journal_path(dir.path(), 5).exists());

        // Double delete is fine
        GCJournal::delete(dir.path(), 5).unwrap();
    }

    #[test]
    fn test_find_journals() {
        let dir = TempDir::new().unwrap();
        let e = vec![(b"k".to_vec(), 0u32, 0u64, 1u32)];
        GCJournal::write(dir.path(), 0, &e).unwrap();
        GCJournal::write(dir.path(), 3, &e).unwrap();

        // Create a non-journal file to ensure it's filtered out
        fs::write(dir.path().join("other_file.bin"), b"nope").unwrap();

        let journals = GCJournal::find_journals(dir.path());
        assert_eq!(journals.len(), 2);
    }
}
