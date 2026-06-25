//! LSM manifest snapshot for recovery and debugging.
//!
//! This is a minimal, serialized view of L0/L1 files per bucket.

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use std::path::Path;

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
        // `write_atomic_durable` generates a per-call unique temp name (so
        // concurrent writers don't race on a shared "manifest.tmp"), fsyncs the
        // temp file, renames over `path`, and fsyncs the parent directory.
        crate::support::write_atomic_durable(path, &bytes)
    }

    pub(crate) fn read_from_path(path: &Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}
