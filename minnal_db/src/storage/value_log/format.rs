//! The value log's on-disk formats: the segment header, the record header,
//! a record's location and metadata, and the per-bucket metadata file.

use super::*;

pub(super) const SEGMENT_MAGIC: [u8; 4] = *b"VSG1";
/// Bumped whenever the segment or record layout changes. Open refuses a segment
/// written in another format: read with this layout, its records would not parse,
/// and the tail repair would truncate them away.
pub(super) const SEGMENT_FORMAT_VERSION: u32 = 2;
/// magic(4) + version(4) + segment_id(4) + reserved(4)
pub(super) const SEGMENT_HEADER_SIZE: u64 = 16;

pub(super) const VALUE_LOG_METADATA_MAGIC: [u8; 4] = *b"VLOG";
pub(super) const VALUE_LOG_METADATA_VERSION: u32 = 3;

/// Where a value lives: which segment, where in it, and how long it is.
/// `segment_id == 0` is the reserved "no value" sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueLocation {
    pub segment_id: u32,
    pub rec_offset: u32,
    pub value_len: u32,
}

/// Per-record metadata that is not the value itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueRecordMeta {
    /// Creation time in Unix millis. TTL expiry reads this straight from the record.
    pub epoch: u64,
    /// The global write sequence that produced this record. A GC copy carries the
    /// fresh seq GC wrote it under. Reads do not check it: a segment id is never reused,
    /// so a pointer can never resolve to a different write.
    pub seq: u64,
}

pub(super) fn value_checksum(value: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(value);
    hasher.finalize()
}

/// 32 bytes, followed by the value bytes and then the key bytes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ValueRecordHeader {
    pub(super) total_len: u32,
    pub(super) epoch: u64,
    pub(super) seq: u64,
    pub(super) key_len: u32,
    pub(super) value_len: u32,
    /// CRC32 over the value payload.
    pub(super) checksum: u32,
}

impl ValueRecordHeader {
    pub(crate) const SIZE: usize = 32;

    pub(crate) fn record_len(key_len: usize, value_len: usize) -> u64 {
        Self::SIZE as u64 + key_len as u64 + value_len as u64
    }

    pub(super) fn to_bytes(self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        out[0..4].copy_from_slice(&self.total_len.to_le_bytes());
        out[4..12].copy_from_slice(&self.epoch.to_le_bytes());
        out[12..20].copy_from_slice(&self.seq.to_le_bytes());
        out[20..24].copy_from_slice(&self.key_len.to_le_bytes());
        out[24..28].copy_from_slice(&self.value_len.to_le_bytes());
        out[28..32].copy_from_slice(&self.checksum.to_le_bytes());
        out
    }

    pub(super) fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::SIZE {
            return None;
        }
        let total_len = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
        let epoch = u64::from_le_bytes(bytes[4..12].try_into().ok()?);
        let seq = u64::from_le_bytes(bytes[12..20].try_into().ok()?);
        let key_len = u32::from_le_bytes(bytes[20..24].try_into().ok()?);
        let value_len = u32::from_le_bytes(bytes[24..28].try_into().ok()?);
        let checksum = u32::from_le_bytes(bytes[28..32].try_into().ok()?);
        // The length must agree with its parts, or these bytes are not a record
        // header — a torn tail, or a pointer into the middle of a record.
        if total_len as u64 != Self::record_len(key_len as usize, value_len as usize) {
            return None;
        }
        Some(Self {
            total_len,
            epoch,
            seq,
            key_len,
            value_len,
            checksum,
        })
    }

    pub(super) fn meta(&self) -> ValueRecordMeta {
        ValueRecordMeta {
            epoch: self.epoch,
            seq: self.seq,
        }
    }
}

/// Live/garbage accounting for one segment. `live + garbage == total`.
#[derive(Debug, Clone, Copy, PartialEq, Archive, RkyvSerialize, RkyvDeserialize, serde::Serialize)]
pub struct SegmentStats {
    pub id: u32,
    /// Record bytes written into this segment (excludes the 16-byte file header).
    pub total_bytes: u64,
    /// Bytes still referenced by the LSM, as tracked by writers.
    pub live_bytes: u64,
    /// Bytes displaced by an overwrite or a delete.
    pub garbage_bytes: u64,
    /// Sealed segments are immutable. Exactly one segment per bucket is unsealed —
    /// the active tail — and GC never selects it.
    pub sealed: bool,
}

impl SegmentStats {
    /// The segment file's size on disk: its 16-byte header plus its records. Segments
    /// are dense append-only files, so this is exact — no `stat` needed, and there are
    /// no sparse holes to reason about.
    pub fn file_bytes(&self) -> u64 {
        SEGMENT_HEADER_SIZE + self.total_bytes
    }

    pub fn garbage_ratio_pct(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        (self.garbage_bytes as f64 / self.total_bytes as f64) * 100.0
    }
}

/// Durable per-bucket state: the segment inventory and the id high-water mark.
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, serde::Serialize)]
pub struct ValueLogMetadata {
    /// High-water mark of segment ids handed out, so ids stay unique for the life of
    /// the database across restarts too — see `ValueLog::open`.
    pub next_segment_id: u64,
    pub active_segment_id: u32,
    pub segments: Vec<SegmentStats>,
    pub total_gc_runs: u64,
    pub total_bytes_reclaimed: u64,
}

impl Default for ValueLogMetadata {
    fn default() -> Self {
        Self::new()
    }
}

impl ValueLogMetadata {
    pub fn new() -> Self {
        Self {
            next_segment_id: 1,
            active_segment_id: 0,
            segments: Vec::new(),
            total_gc_runs: 0,
            total_bytes_reclaimed: 0,
        }
    }

    pub fn live_bytes(&self) -> u64 {
        self.segments.iter().map(|s| s.live_bytes).sum()
    }

    pub fn garbage_bytes(&self) -> u64 {
        self.segments.iter().map(|s| s.garbage_bytes).sum()
    }

    pub fn total_bytes(&self) -> u64 {
        self.segments.iter().map(|s| s.total_bytes).sum()
    }

    pub(super) fn to_file_bytes(&self) -> Result<Vec<u8>> {
        let payload =
            rkyv::to_bytes::<rkyv::rancor::Error>(self).map_err(|e| ValueLogError::Serialization(format!("Failed to serialize metadata: {}", e)))?;
        let mut hasher = Hasher::new();
        hasher.update(&payload);
        let checksum = hasher.finalize();

        let mut out = Vec::with_capacity(16 + payload.len());
        out.extend_from_slice(&VALUE_LOG_METADATA_MAGIC);
        out.extend_from_slice(&VALUE_LOG_METADATA_VERSION.to_le_bytes());
        out.extend_from_slice(&checksum.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Decode metadata. A file of any other version is **rejected, never
    /// upgraded**; the caller then rebuilds the accounting from the segment files
    /// and the LSM's pointers.
    pub(super) fn from_file_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 16 || bytes.get(0..4) != Some(&VALUE_LOG_METADATA_MAGIC) {
            return Err(ValueLogError::Serialization("value log metadata: bad magic".into()));
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != VALUE_LOG_METADATA_VERSION {
            return Err(ValueLogError::Serialization(format!(
                "value log metadata version {version} unsupported (expected {VALUE_LOG_METADATA_VERSION})"
            )));
        }
        let checksum = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let payload_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let end = 16usize.saturating_add(payload_len);
        if bytes.len() < end {
            return Err(ValueLogError::Serialization("value log metadata: truncated payload".into()));
        }
        let payload = &bytes[16..end];
        let mut hasher = Hasher::new();
        hasher.update(payload);
        if hasher.finalize() != checksum {
            return Err(ValueLogError::Serialization("value log metadata: checksum mismatch".into()));
        }
        rkyv::from_bytes::<Self, rkyv::rancor::Error>(payload)
            .map_err(|e| ValueLogError::Serialization(format!("Failed to deserialize metadata: {}", e)))
    }
}
