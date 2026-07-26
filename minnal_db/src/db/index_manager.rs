//! Index Manager
//!
//! Manages the on-disk index directory structure and checkpoint file I/O.
//! Field metadata (names, types, etc.) lives in `NamespaceSchema` inside
//! the `NamespaceRegistry`; this module is only concerned with paths and
//! the checkpoint marker files used for crash recovery.
//!
//! Directory layout:
//!
//! ```text
//! {db_path}/index/
//!   {namespace_id}/
//!     {field_id}/
//!       blobs.keys     ← BlobStore key file (mmap hash table, slot_id → offset)
//!       blobs.vals     ← BlobStore value file (serialised RoaringBitmap blobs)
//!       keymap/        ← mmap-backed keymap store (value → slot_id mapping)
//!         blobs.keys
//!         blobs.vals
//!       checkpoint     ← WAL write-offset at last flush (8 bytes, LE u64)
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::db::error::{KVError, Result};
use crate::db::namespace::FieldId;

/// What a field's on-disk `checkpoint` marker says.
///
/// [`read_checkpoint`](IndexManager::read_checkpoint) collapses all three into a
/// replay offset, since every case replays from the earliest available point.
/// The distinction matters only when deciding whether a short replay window is
/// *suspicious*: a field that has never been checkpointed is expected to have no
/// marker, whereas a missing marker on a field that already holds data means the
/// recorded position was lost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointState {
    /// Marker present and usable — the field's persisted index reflects the WAL
    /// up to this offset.
    At(u64),
    /// No marker on disk. Expected for a field that has never been checkpointed.
    Absent,
    /// Marker present but unusable: too short, or recording an offset ahead of
    /// the WAL tail (only possible from a torn write).
    Unusable,
}

impl CheckpointState {
    /// The offset to replay from: the recorded one, else 0 ("replay everything
    /// still available").
    pub fn replay_offset(self) -> u64 {
        match self {
            CheckpointState::At(offset) => offset,
            CheckpointState::Absent | CheckpointState::Unusable => 0,
        }
    }
}

/// A field-index replay window the WAL can no longer satisfy.
///
/// The field index is missing every update recorded in the absent segments, and
/// **cannot recover them** — the entries are gone. Restoring full coverage needs
/// a re-index; see `docs/feature-requests/index-replay-gap-remediation.md`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayGap {
    /// Offset replay was to start from (the field's recorded checkpoint, or 0).
    pub from: u64,
    /// Offset replay was to run to (the WAL tail).
    pub to: u64,
    /// Segment ids inside `[from, to)` whose files are gone.
    pub missing_segments: Vec<u64>,
}

/// Decide whether an incomplete replay window is a real gap worth reporting.
///
/// Detection is deliberately *reporting-only* — it changes no replay behaviour.
/// The one judgement it makes is suppressing the expected case: a field being
/// activated for the very first time has no checkpoint marker and no data, so it
/// replays from 0 and every already-reclaimed segment reads as "missing". That
/// is normal (a brand-new index is populated by a build, not by WAL replay) and
/// must not be reported, or every `add_index` on an established database would
/// log a spurious error.
///
/// Every other combination is reportable:
/// - [`CheckpointState::At`] — the field recorded a position and the WAL behind
///   it has since been reclaimed. This is the real defect.
/// - [`CheckpointState::Unusable`] — the recorded position was lost to a torn
///   marker, so coverage cannot be proven either way.
/// - [`CheckpointState::Absent`] with a **non-empty** index — the field holds
///   data but has no marker, so its position was lost rather than never set.
pub fn detect_replay_gap(state: CheckpointState, index_is_empty: bool, wal_tail: u64, missing_segments: Vec<u64>) -> Option<ReplayGap> {
    if missing_segments.is_empty() {
        return None;
    }
    if state == CheckpointState::Absent && index_is_empty {
        return None;
    }
    Some(ReplayGap {
        from: state.replay_offset(),
        to: wal_tail,
        missing_segments,
    })
}

/// Manages the on-disk index directory structure and checkpoint files.
///
/// This struct holds no field registry state — that lives in
/// `NamespaceSchema` inside `NamespaceRegistry`.
pub struct IndexManager {
    /// Root index directory: `{db_path}/index/`
    pub index_base_path: PathBuf,
}

impl IndexManager {
    /// Open (or create) the index manager rooted at `{db_path}/index/`.
    pub fn open(db_path: &Path) -> Result<Arc<Self>> {
        let index_base_path = db_path.join("index");
        std::fs::create_dir_all(&index_base_path)?;
        Ok(Arc::new(Self { index_base_path }))
    }

    /// Ensure the per-field index directory exists.
    ///
    /// Path: `{index_base}/{namespace_id}/{field_id}/`
    pub fn ensure_field_path(&self, namespace_id: u32, field_id: FieldId) -> Result<()> {
        let path = self.field_path(namespace_id, field_id);
        std::fs::create_dir_all(&path)?;
        Ok(())
    }

    /// Compute the on-disk path for a field index directory.
    ///
    /// Path: `{index_base}/{namespace_id}/{field_id}/`
    ///
    /// Does not create the directory or verify the field is registered.
    pub fn field_path(&self, namespace_id: u32, field_id: FieldId) -> PathBuf {
        self.index_base_path.join(namespace_id.to_string()).join(field_id.to_string())
    }

    /// Compute the on-disk path for a namespace's dense row-ID map.
    ///
    /// Path: `{index_base}/{namespace_id}/rowmap/` — a sibling of the per-field
    /// directories. (`rowmap` can never collide with a `FieldId`, which is
    /// numeric.) The `RowMap` creates the directory on first use.
    pub fn rowmap_path(&self, namespace_id: u32) -> PathBuf {
        self.index_base_path.join(namespace_id.to_string()).join("rowmap")
    }

    /// Remove the entire on-disk index subtree for a namespace.
    ///
    /// Deletes `{index_base}/{namespace_id}/` and everything under it (all field
    /// directories, blob stores, keymaps, and checkpoint markers). Called when a
    /// namespace is dropped. A missing directory is treated as success, so this
    /// is safe to call when the namespace had no indexed fields.
    pub fn remove_namespace_path(&self, namespace_id: u32) -> Result<()> {
        let path = self.index_base_path.join(namespace_id.to_string());
        match std::fs::remove_dir_all(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(KVError::Io(e)),
        }
    }

    /// Write a checkpoint marker for each `(namespace_id, field_id)` pair.
    ///
    /// Records `wal_tail` as an 8-byte little-endian value in
    /// `{field_path}/checkpoint` using a durable tmp-then-rename write: the tmp
    /// file is fsynced before the rename (so its bytes are on disk) and the field
    /// directory is fsynced after (so the rename itself survives a crash). This
    /// matches [`RowMap::write_marker`](crate::index::RowMap) — without it a crash could
    /// expose a renamed but torn checkpoint whose garbage offset, if read back as
    /// greater than the WAL tail, would make recovery *skip* replay and silently
    /// drop index entries.
    ///
    /// The mmap bitmap data and keymap store must be flushed by the caller before
    /// calling this so the checkpoint offset is consistent with the on-disk index
    /// state.
    ///
    /// The list of fields to checkpoint is supplied by the caller (typically
    /// obtained from `NamespaceRegistry::all_indexed_fields`).
    pub fn checkpoint_fields(&self, wal_tail: u64, fields: &[(u32, FieldId)]) -> Result<()> {
        use std::io::Write;
        for &(namespace_id, field_id) in fields {
            let field_path = self.field_path(namespace_id, field_id);
            let checkpoint_file = field_path.join("checkpoint");
            let tmp = checkpoint_file.with_extension("tmp");

            // Write + fsync the tmp file so its bytes are durable before the rename.
            {
                let mut f = std::fs::File::create(&tmp)?;
                f.write_all(&wal_tail.to_le_bytes())?;
                f.sync_all()?;
            }
            std::fs::rename(&tmp, &checkpoint_file).map_err(|e| {
                KVError::Io(std::io::Error::new(
                    e.kind(),
                    format!("Failed to write checkpoint for ns={} field={}: {}", namespace_id, field_id, e),
                ))
            })?;
            // fsync the field directory so the rename survives a crash.
            std::fs::File::open(&field_path)?.sync_all()?;
        }
        Ok(())
    }

    /// Read the WAL offset recorded by the last checkpoint for a field.
    ///
    /// Returns `0` (meaning "uncheckpointed" — replay the full available WAL
    /// window) when no checkpoint file exists, the file is too short, or the
    /// recorded offset is **ahead of `wal_tail`**. The last case can only happen
    /// from a corrupt/torn marker — a real checkpoint records a *past* tail and
    /// the tail only grows — and returning the raw value would let recovery's
    /// `offset < wal_tail` check skip replay and silently drop index entries.
    /// Clamping to `0` instead forces a safe (idempotent) full replay.
    pub fn read_checkpoint(&self, namespace_id: u32, field_id: FieldId, wal_tail: u64) -> u64 {
        self.read_checkpoint_state(namespace_id, field_id, wal_tail).replay_offset()
    }

    /// Read a field's checkpoint marker, distinguishing "absent" and "unusable"
    /// from a real recorded offset.
    ///
    /// Same clamping rules as [`read_checkpoint`](Self::read_checkpoint) — which
    /// is this function with the distinction thrown away — but callers that need
    /// to judge whether a short replay window is expected or alarming can see
    /// which case they are in.
    pub fn read_checkpoint_state(&self, namespace_id: u32, field_id: FieldId, wal_tail: u64) -> CheckpointState {
        let path = self.field_path(namespace_id, field_id).join("checkpoint");
        let Ok(bytes) = std::fs::read(&path) else {
            return CheckpointState::Absent;
        };
        if bytes.len() < 8 {
            return CheckpointState::Unusable;
        }
        let offset = u64::from_le_bytes(bytes[..8].try_into().unwrap_or([0u8; 8]));
        if offset > wal_tail {
            log::warn!(
                "[index] checkpoint offset {offset} for ns={namespace_id} field={field_id} exceeds WAL tail {wal_tail}; \
                 treating as uncheckpointed (full replay)"
            );
            return CheckpointState::Unusable;
        }
        CheckpointState::At(offset)
    }
}

// ── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_open_creates_index_dir() {
        let dir = TempDir::new().unwrap();
        let _mgr = IndexManager::open(dir.path()).unwrap();
        assert!(dir.path().join("index").exists());
    }

    #[test]
    fn test_ensure_field_path() {
        let dir = TempDir::new().unwrap();
        let mgr = IndexManager::open(dir.path()).unwrap();
        // ensure_field_path creates the namespace dir as a side-effect
        mgr.ensure_field_path(0, 3).unwrap();
        assert!(dir.path().join("index").join("0").exists());
        assert!(dir.path().join("index").join("0").join("3").exists());
    }

    #[test]
    fn test_remove_namespace_path() {
        let dir = TempDir::new().unwrap();
        let mgr = IndexManager::open(dir.path()).unwrap();
        mgr.ensure_field_path(7, 1).unwrap();
        mgr.ensure_field_path(7, 2).unwrap();
        let ns_index_dir = dir.path().join("index").join("7");
        assert!(ns_index_dir.exists());

        mgr.remove_namespace_path(7).unwrap();
        assert!(!ns_index_dir.exists());

        // Idempotent: removing an already-absent namespace is a no-op, not an error.
        mgr.remove_namespace_path(7).unwrap();
    }

    // ── replay-gap detection ────────────────────────────────────────────

    #[test]
    fn read_checkpoint_state_distinguishes_absent_from_recorded_zero() {
        let dir = TempDir::new().unwrap();
        let mgr = IndexManager::open(dir.path()).unwrap();
        mgr.ensure_field_path(0, 1).unwrap();

        // No marker written yet.
        assert_eq!(mgr.read_checkpoint_state(0, 1, 100), CheckpointState::Absent);

        // A marker recording 0 is a *recorded* position, not an absent one — both
        // replay from 0, which is exactly why read_checkpoint alone cannot tell
        // a fresh field from one whose marker was lost.
        mgr.checkpoint_fields(0, &[(0, 1)]).unwrap();
        assert_eq!(mgr.read_checkpoint_state(0, 1, 100), CheckpointState::At(0));
        assert_eq!(mgr.read_checkpoint(0, 1, 100), 0);

        mgr.checkpoint_fields(64, &[(0, 1)]).unwrap();
        assert_eq!(mgr.read_checkpoint_state(0, 1, 100), CheckpointState::At(64));

        // Ahead of the tail — only reachable from a torn write.
        assert_eq!(mgr.read_checkpoint_state(0, 1, 10), CheckpointState::Unusable);
        assert_eq!(mgr.read_checkpoint(0, 1, 10), 0, "unusable still replays from 0");
    }

    #[test]
    fn read_checkpoint_state_reports_a_short_marker_as_unusable() {
        let dir = TempDir::new().unwrap();
        let mgr = IndexManager::open(dir.path()).unwrap();
        mgr.ensure_field_path(0, 1).unwrap();
        std::fs::write(mgr.field_path(0, 1).join("checkpoint"), [1u8, 2, 3]).unwrap();
        assert_eq!(mgr.read_checkpoint_state(0, 1, 100), CheckpointState::Unusable);
    }

    #[test]
    fn no_gap_reported_when_every_segment_is_present() {
        // The common case: replay window fully covered, nothing to report,
        // whatever the marker says.
        for state in [
            CheckpointState::At(0),
            CheckpointState::At(64),
            CheckpointState::Absent,
            CheckpointState::Unusable,
        ] {
            for empty in [true, false] {
                assert_eq!(detect_replay_gap(state, empty, 500, vec![]), None, "state={state:?} empty={empty}");
            }
        }
    }

    #[test]
    fn a_fresh_field_activating_on_an_established_database_reports_no_gap() {
        // The one case detection must stay quiet about. A brand-new index has no
        // marker and no data, so it replays from 0 and every already-reclaimed
        // segment looks "missing" — but a new index is populated by a build, not
        // by WAL replay. Reporting here would fire on every add_index.
        assert_eq!(detect_replay_gap(CheckpointState::Absent, true, 500, vec![0, 1]), None);
    }

    #[test]
    fn a_checkpointed_field_missing_its_wal_reports_a_gap() {
        // The real defect: the field recorded a position, and WAL GC has since
        // reclaimed segments covering the window between it and the tail.
        let gap = detect_replay_gap(CheckpointState::At(64), false, 500, vec![1, 2]).expect("gap");
        assert_eq!(
            gap,
            ReplayGap {
                from: 64,
                to: 500,
                missing_segments: vec![1, 2],
            }
        );
    }

    #[test]
    fn a_populated_field_with_no_marker_reports_a_gap() {
        // Has data but no marker: the position was lost, not never set.
        let gap = detect_replay_gap(CheckpointState::Absent, false, 500, vec![0]).expect("gap");
        assert_eq!(gap.from, 0, "replays from the start of what survives");
    }

    #[test]
    fn an_unusable_marker_reports_a_gap_even_on_an_empty_index() {
        // A torn marker means coverage cannot be proven either way, so report it
        // regardless of whether the index happens to be empty.
        assert!(detect_replay_gap(CheckpointState::Unusable, true, 500, vec![3]).is_some());
        assert!(detect_replay_gap(CheckpointState::Unusable, false, 500, vec![3]).is_some());
    }

    #[test]
    fn test_field_path_is_pure_computation() {
        let dir = TempDir::new().unwrap();
        let mgr = IndexManager::open(dir.path()).unwrap();
        let path = mgr.field_path(5, 2);
        assert_eq!(path, dir.path().join("index").join("5").join("2"));
    }

    #[test]
    fn test_checkpoint_and_read_back() {
        let dir = TempDir::new().unwrap();
        let mgr = IndexManager::open(dir.path()).unwrap();
        mgr.ensure_field_path(0, 0).unwrap();

        assert_eq!(mgr.read_checkpoint(0, 0, u64::MAX), 0); // no file yet

        mgr.checkpoint_fields(55555, &[(0, 0)]).unwrap();
        assert_eq!(mgr.read_checkpoint(0, 0, u64::MAX), 55555);

        mgr.checkpoint_fields(99999, &[(0, 0)]).unwrap();
        assert_eq!(mgr.read_checkpoint(0, 0, u64::MAX), 99999);
    }

    #[test]
    fn test_checkpoint_multiple_fields() {
        let dir = TempDir::new().unwrap();
        let mgr = IndexManager::open(dir.path()).unwrap();
        for field_id in 0u32..3 {
            mgr.ensure_field_path(1, field_id).unwrap();
        }

        mgr.checkpoint_fields(42, &[(1, 0), (1, 1), (1, 2)]).unwrap();
        assert_eq!(mgr.read_checkpoint(1, 0, u64::MAX), 42);
        assert_eq!(mgr.read_checkpoint(1, 1, u64::MAX), 42);
        assert_eq!(mgr.read_checkpoint(1, 2, u64::MAX), 42);
    }

    #[test]
    fn test_read_checkpoint_missing_returns_zero() {
        let dir = TempDir::new().unwrap();
        let mgr = IndexManager::open(dir.path()).unwrap();
        assert_eq!(mgr.read_checkpoint(99, 99, u64::MAX), 0);
    }

    #[test]
    fn test_read_checkpoint_ahead_of_wal_tail_is_uncheckpointed() {
        let dir = TempDir::new().unwrap();
        let mgr = IndexManager::open(dir.path()).unwrap();
        mgr.ensure_field_path(0, 0).unwrap();
        mgr.checkpoint_fields(99_999, &[(0, 0)]).unwrap();

        // In range (≤ wal_tail) → returned verbatim.
        assert_eq!(mgr.read_checkpoint(0, 0, 100_000), 99_999);
        assert_eq!(mgr.read_checkpoint(0, 0, 99_999), 99_999);

        // Ahead of the WAL tail (only possible from a corrupt/torn marker) → 0,
        // forcing a safe full replay instead of skipping it.
        assert_eq!(mgr.read_checkpoint(0, 0, 50_000), 0);
    }
}
