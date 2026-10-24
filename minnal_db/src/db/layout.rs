//! The one place that knows how a database is laid out on disk.
//!
//! Every directory name and nesting rule lives here. Nothing above this module
//! should call `join("index")` or `format!("ns_{}", ..)` — not the engine, not
//! the document store, and not the API server.
//!
//! This is not tidiness. The layout used to be spelled out in three layers, and
//! it caused a real defect: `doc_store::drop_index` built the field-index path
//! itself and deleted the directory *before* saving its schema, which is the
//! opposite of the order the layer that owns the path uses. A crash in that
//! window left a registered field with no data — an index that came up silently
//! incomplete. Duplicated layout knowledge means duplicated (and diverging)
//! rules about how to use it.
//!
//! ```text
//! {db_path}/
//!   ns_{name}/            ← per-namespace data: LSM SSTables, value log, config.json
//!   index/
//!     {ns_id}/
//!       {field_id}/       ← field index: bitmap blob, keymap, checkpoint, gap.json
//!       rowmap/           ← dense row-ID map sidecar
//!       no_wal_pending    ← marker: uncheckpointed no-WAL writes (FR-001)
//!   wal/                  ← the main WAL's segments ({segment_id}.log) and metadata
//!   vector_index_wal/     ← the vector index WAL's segments ({segment_id}.log) and retired_cut
//!   fail_logs/            ← recovery failures
//! ```

use std::path::{Path, PathBuf};

/// Prefix for a namespace's data directory.
const NAMESPACE_DIR_PREFIX: &str = "ns_";

/// Name of the index subtree at the database root.
const INDEX_DIR: &str = "index";

/// A namespace's data directory: `{db_path}/ns_{name}`.
///
/// Holds the LSM SSTables, the value-log segments, and `config.json`. Keyed by
/// **name**, not id — this is the one path that is, for historical reasons.
pub(crate) fn namespace_data_dir(db_path: &Path, name: &str) -> PathBuf {
    db_path.join(format!("{NAMESPACE_DIR_PREFIX}{name}"))
}

/// The index subtree root: `{db_path}/index`.
pub(crate) fn index_root(db_path: &Path) -> PathBuf {
    db_path.join(INDEX_DIR)
}

/// A namespace's index subtree, given the index root: `{index_root}/{ns_id}`.
///
/// Keyed by **id**, not name, so a rename would not orphan it.
///
/// Takes the root rather than the database path so `IndexManager` — which holds
/// the root, not the path — can use the same function instead of keeping its own
/// copy of the rule. Callers starting from a database path compose:
/// `namespace_index_dir(&index_root(db_path), ns_id)`.
pub(crate) fn namespace_index_dir(index_root: &Path, ns_id: u32) -> PathBuf {
    index_root.join(ns_id.to_string())
}

/// Name of the main WAL's directory at the database root.
const MAIN_WAL_DIR: &str = "wal";

/// The main WAL's directory: `{db_path}/wal`, holding its segments and its
/// `metadata` (the saved retirement progress).
pub(crate) fn main_wal_dir(db_path: &Path) -> PathBuf {
    db_path.join(MAIN_WAL_DIR)
}

/// Name of the vector index WAL's directory at the database root.
const VECTOR_INDEX_WAL_DIR: &str = "vector_index_wal";

/// The vector index WAL's directory: `{db_path}/vector_index_wal`.
///
/// One per database, shared by every vector namespace (see
/// `semantic_search/report/Vector-Index-WAL-Design.md`).
pub(crate) fn vector_index_wal_dir(db_path: &Path) -> PathBuf {
    db_path.join(VECTOR_INDEX_WAL_DIR)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_nest_as_documented() {
        let root = Path::new("/db");
        assert_eq!(namespace_data_dir(root, "users"), Path::new("/db/ns_users"));
        assert_eq!(index_root(root), Path::new("/db/index"));
        assert_eq!(main_wal_dir(root), Path::new("/db/wal"));
        assert_eq!(namespace_index_dir(&index_root(root), 7), Path::new("/db/index/7"));
        assert_eq!(vector_index_wal_dir(root), Path::new("/db/vector_index_wal"));
    }

    /// The namespace data directory is keyed by name and the index subtree by
    /// id. They are deliberately different keys; a change to either would strand
    /// existing data, so this pins both.
    #[test]
    fn data_and_index_dirs_use_different_keys() {
        let root = Path::new("/db");
        assert!(namespace_data_dir(root, "users").to_string_lossy().contains("ns_users"));
        assert!(!namespace_index_dir(&index_root(root), 7).to_string_lossy().contains("users"));
    }
}
