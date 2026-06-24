//! LSM manifest snapshot for recovery and debugging.
//!
//! This is a minimal, serialized view of L0/L1 files per bucket.

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static MANIFEST_WRITE_SEQ: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, serde::Serialize)]
pub struct LsmManifest {
    pub version: u32,
    pub created_at_ms: u64,
    pub levels: Vec<ManifestLevel>,
}

#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, serde::Serialize)]
pub struct ManifestLevel {
    pub level: u8,
    pub buckets: Vec<ManifestBucket>,
}

#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, serde::Serialize)]
pub struct ManifestBucket {
    pub bucket: u32,
    pub files: Vec<ManifestFile>,
}

#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, serde::Serialize)]
pub struct ManifestFile {
    pub path: String,
    pub created_at_ms: u128,
    pub entry_count: u64,
}

impl LsmManifest {
    pub(crate) fn new(levels: Vec<ManifestLevel>, created_at_ms: u64) -> Self {
        Self {
            version: 1,
            created_at_ms,
            levels,
        }
    }

    pub(crate) fn to_bytes(&self) -> Result<Vec<u8>, String> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .map_err(|e| format!("Failed to serialize LSM manifest: {}", e))
            .map(|buf| buf.to_vec())
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        rkyv::from_bytes::<LsmManifest, rkyv::rancor::Error>(bytes).map_err(|e| format!("Failed to deserialize LSM manifest: {}", e))
    }

    pub(crate) fn write_to_path(&self, path: &Path) -> std::io::Result<()> {
        let bytes = self.to_bytes().map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        // Use a unique per-call sequence number so concurrent writers don't
        // race on the same "manifest.tmp" filename (which would cause the
        // second rename to fail with ENOENT after the first already moved it).
        let seq = MANIFEST_WRITE_SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp = path.with_extension(format!("tmp.{}", seq));
        {
            let file = std::fs::File::create(&tmp)?;
            let mut writer = BufWriter::new(&file);
            writer.write_all(&bytes)?;
            writer.flush()?;
            file.sync_all()?;
        }
        std::fs::rename(&tmp, path)
    }

    pub(crate) fn read_from_path(path: &Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}
