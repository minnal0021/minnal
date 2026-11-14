//! A snapshot of a namespace's LSM files: each level's files per bucket, with
//! their entry counts. Built on request for the storage admin endpoints; never
//! written to disk.

#[derive(Debug, Clone, serde::Serialize)]
pub struct LsmManifest {
    pub version: u32,
    pub created_at_ms: u64,
    pub levels: Vec<ManifestLevel>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ManifestLevel {
    pub level: u8,
    pub buckets: Vec<ManifestBucket>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ManifestBucket {
    pub bucket: u32,
    pub files: Vec<ManifestFile>,
}

#[derive(Debug, Clone, serde::Serialize)]
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
}
