//! The config file (`config/sample.toml` is the reference): parsing it, and
//! the one mapping from it to the configuration each part of the database
//! takes.
//!
//! [`ConfigFile`] holds every table the database reads, and is the only
//! parser of them: the API server, the benchmarks and an embedded
//! [`AsyncDb`](crate::AsyncDb) or doc store all go through it, so a key means
//! the same everywhere. A key left out takes its default from
//! [`crate::defaults`], and a key or table it does not know is an error, so a
//! misspelt key fails at startup instead of silently taking its default.
//!
//! A program that keeps tables of its own in the same file (the API server's
//! `[api]` and `[logging]`) names them in [`ConfigFile::parse`]; they are
//! handed back untouched for it to read.

mod engine;
mod lifecycle;
mod semantic_search;
#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};

use serde::Deserialize;

pub use engine::{
    LsmSection, MemtableSection, RecoverySection, ScheduledTaskSection, ShardingSection, StorageSection, SyncSection, ThresholdSection,
    ValueLogSection, WalSection,
};
pub use lifecycle::LifecycleSection;
pub use semantic_search::{MaintenanceSection, SearchDefaultsSection, SemanticSearchSection, SupportedModel, VectorIndexSection};

/// The tables the API server keeps in the same file, for a program that
/// opens the server's file itself (a benchmark, an embedded store).
pub const API_SERVER_TABLES: &[&str] = &["api", "logging"];

/// Why a config file could not be used.
#[derive(Debug, thiserror::Error)]
pub enum ConfigFileError {
    /// The file could not be read.
    #[error("cannot read config file '{path}': {source}")]
    Read {
        /// The file.
        path: PathBuf,
        /// What went wrong.
        source: std::io::Error,
    },
    /// The file is not valid TOML, or a key is unknown or of the wrong type.
    #[error("cannot parse config file{}: {source}", at(.path))]
    Parse {
        /// The file, when read from one.
        path: Option<PathBuf>,
        /// What went wrong.
        source: toml::de::Error,
    },
    /// A value is out of range.
    #[error("invalid config file{}: {reason}", at(.path))]
    Invalid {
        /// The file, when read from one.
        path: Option<PathBuf>,
        /// What is wrong.
        reason: String,
    },
}

fn at(path: &Option<PathBuf>) -> String {
    path.as_ref().map(|p| format!(" '{}'", p.display())).unwrap_or_default()
}

impl ConfigFileError {
    fn with_path(self, file: &Path) -> Self {
        match self {
            Self::Parse { source, .. } => Self::Parse {
                path: Some(file.to_path_buf()),
                source,
            },
            Self::Invalid { reason, .. } => Self::Invalid {
                path: Some(file.to_path_buf()),
                reason,
            },
            read => read,
        }
    }
}

/// Every table of the config file the database reads. `Default` is the file
/// with nothing in it.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConfigFile {
    /// `[storage]`.
    pub storage: StorageSection,
    /// `[memtable]`.
    pub memtable: MemtableSection,
    /// `[sharding]`.
    pub sharding: ShardingSection,
    /// `[lsm]`.
    pub lsm: LsmSection,
    /// `[sync]`.
    pub sync: SyncSection,
    /// `[thresholds]`.
    pub thresholds: ThresholdSection,
    /// `[scheduled_tasks]`.
    pub scheduled_tasks: ScheduledTaskSection,
    /// `[wal]`.
    pub wal: WalSection,
    /// `[value_log]`.
    pub value_log: ValueLogSection,
    /// `[recovery]`.
    pub recovery: RecoverySection,
    /// `[lifecycle]`.
    pub lifecycle: LifecycleSection,
    /// `[vector_index]`.
    pub vector_index: VectorIndexSection,
    /// `[semantic_search]`.
    pub semantic_search: SemanticSearchSection,
}

impl ConfigFile {
    /// Read and check the file at `path`. Tables named in `host_tables`
    /// belong to the calling program: they are returned for it to read and
    /// are not checked here.
    pub fn from_file(path: &Path, host_tables: &[&str]) -> Result<(Self, toml::Table), ConfigFileError> {
        let text = std::fs::read_to_string(path).map_err(|source| ConfigFileError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse(&text, host_tables).map_err(|e| e.with_path(path))
    }

    /// Parse and check config text. Tables named in `host_tables` are taken
    /// out first and returned for the caller to read; every other table and
    /// key must be one this type knows.
    pub fn parse(text: &str, host_tables: &[&str]) -> Result<(Self, toml::Table), ConfigFileError> {
        let parse = |source| ConfigFileError::Parse { path: None, source };
        let mut table: toml::Table = text.parse().map_err(parse)?;
        let mut host = toml::Table::new();
        for name in host_tables {
            if let Some(value) = table.remove(*name) {
                host.insert((*name).to_string(), value);
            }
        }
        let config = Self::deserialize(table).map_err(parse)?;
        config.validate().map_err(|reason| ConfigFileError::Invalid { path: None, reason })?;
        Ok((config, host))
    }

    /// Check what the types alone cannot.
    fn validate(&self) -> Result<(), String> {
        self.validate_engine()?;
        self.lifecycle.validate()?;
        self.semantic_search.validate()
    }

    /// The database directory (`storage.db_path`).
    pub fn db_path(&self) -> PathBuf {
        PathBuf::from(&self.storage.db_path)
    }
}
