//! Unit tests for the database coordinator, one file per topic.
//!
//! Everything here exercises the coordinator itself; tests for the pieces in
//! sibling modules live with them.

use super::*;
use crate::db::config::SyncConfig;
use crate::db::test_support::*;
use crate::field_index::{IndexValue, IndexValueType};
use crate::kv_store::index_set::ExtractorFn;
use tempfile::TempDir;

/// The saved metadata file, as recovery would read it (a fresh database has
/// not saved one yet).
fn saved_wal_metadata(dir: &TempDir) -> WalMetadata {
    match std::fs::read(wal_metadata_path(dir.path())) {
        Ok(bytes) => WalMetadata::from_file_bytes(&bytes).unwrap(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => WalMetadata::new(),
        Err(e) => panic!("reading the WAL metadata: {e}"),
    }
}

mod async_db;
mod crash;
mod field_index;
mod in_flight;
mod merge;
mod namespaces;
mod no_wal;
mod recovery;
mod wal_progress;
mod writes;
