//! Configuration of the minnal_db_api server.
//!
//! The file is the database's config file (`config/sample.toml`), read by
//! [`minnal_db::ConfigFile`] exactly as an embedded store reads it, plus two
//! tables of the server's own: `[api]` (the HTTP address) and `[logging]`.
//! Every table rejects keys it does not know, so a misspelt key stops the
//! server at startup instead of silently taking its default. With no file,
//! every value is its default.

use std::path::{Path, PathBuf};

use minnal_db::ConfigFile;
use minnal_db::config_file::API_SERVER_TABLES;
use serde::Deserialize;

/// The address the server listens on when `[api] listen_addr` is not set.
pub const DEFAULT_LISTEN_ADDR: &str = "0.0.0.0:8080";

/// The log level used when neither `RUST_LOG` nor `[logging] level` is set.
pub const DEFAULT_LOG_LEVEL: &str = "info";

/// The server's configuration.
#[derive(Debug, Default)]
pub struct DocStoreApiConfig {
    /// Every table the database reads.
    pub file: ConfigFile,
    /// `[api]`.
    pub api: ApiSection,
    /// `[logging]`.
    pub logging: LoggingSection,
}

impl DocStoreApiConfig {
    /// Read and check the config file at `path`.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let (file, host) = ConfigFile::from_file(path, API_SERVER_TABLES).map_err(|e| e.to_string())?;
        Self::with_server_tables(file, host).map_err(|e| format!("invalid config file '{}': {e}", path.display()))
    }

    /// Parse and check config text.
    #[cfg(test)]
    pub fn parse(text: &str) -> Result<Self, String> {
        let (file, host) = ConfigFile::parse(text, API_SERVER_TABLES).map_err(|e| e.to_string())?;
        Self::with_server_tables(file, host)
    }

    fn with_server_tables(file: ConfigFile, host: toml::Table) -> Result<Self, String> {
        fn table<T: for<'de> Deserialize<'de> + Default>(host: &toml::Table, name: &str) -> Result<T, String> {
            match host.get(name) {
                Some(value) => value.clone().try_into().map_err(|e| format!("[{name}]: {e}")),
                None => Ok(T::default()),
            }
        }
        Ok(Self {
            api: table(&host, "api")?,
            logging: table(&host, "logging")?,
            file,
        })
    }

    /// The database directory.
    pub fn db_path(&self) -> PathBuf {
        self.file.db_path()
    }

    /// Where the rolling log files are written (`storage.log_dir`).
    pub fn log_dir(&self) -> PathBuf {
        PathBuf::from(&self.file.storage.log_dir)
    }

    /// The HTTP address to listen on.
    pub fn listen_addr(&self) -> &str {
        &self.api.listen_addr
    }

    /// The log level used when `RUST_LOG` is not set.
    pub fn log_level(&self) -> &str {
        &self.logging.level
    }
}

/// `[api]`: the HTTP server.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ApiSection {
    /// The address to listen on.
    pub listen_addr: String,
}

impl Default for ApiSection {
    fn default() -> Self {
        Self {
            listen_addr: DEFAULT_LISTEN_ADDR.into(),
        }
    }
}

/// `[logging]`.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoggingSection {
    /// The lowest level logged when `RUST_LOG` is not set: `"error"`,
    /// `"warn"`, `"info"`, `"debug"` or `"trace"`. `RUST_LOG` always wins.
    pub level: String,
}

impl Default for LoggingSection {
    fn default() -> Self {
        Self {
            level: DEFAULT_LOG_LEVEL.into(),
        }
    }
}

#[cfg(test)]
mod tests;
