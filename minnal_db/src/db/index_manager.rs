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
/// a re-index; see `FEATURE-REQUEST.md` (FR-001).
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

/// How a field index with an outstanding gap has to be repaired.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum RepairMode {
    /// Repair only the listed keys — the normal case. Work is proportional to
    /// the damage, not to the size of the store.
    ///
    /// Keys are hex-encoded so the record stays human-readable and greppable;
    /// they are raw key bytes, which are not necessarily UTF-8.
    RowScoped { keys: Vec<String> },
    /// The affected key set could not be captured (it exceeded the worklist cap,
    /// or was never knowable — see [`GapCause::NoWalWrites`]). The whole field
    /// index must be rebuilt from current data.
    FullRebuild,
}

/// Why a field index is incomplete.
///
/// Recorded so an operator can tell a wedged checkpoint worker (`BackstopReclaim`)
/// from a crash after bulk loading (`NoWalWrites`) from a genuine index fault
/// (`RejectedUpdate`) — the three have completely different fixes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GapCause {
    /// WAL GC's backstop reclaimed a segment this field still needed, because the
    /// index-replay watermark had held more than `max_pinned_wal_segments`.
    BackstopReclaim,
    /// The database came up after an unclean shutdown with no-WAL writes
    /// outstanding. Those writes have no WAL entries, so the affected keys were
    /// never recoverable and the repair is necessarily a full rebuild.
    NoWalWrites,
    /// A field index rejected an update on the write path. The key is known
    /// exactly, so this is always row-scoped.
    RejectedUpdate,
}

/// A durable record that a field index is missing updates, and what it would
/// take to repair it.
///
/// Written beside the field's `checkpoint` marker as `gap.json`. Survives
/// restart; cleared only by a successful repair. See `FEATURE-REQUEST.md`
/// (FR-001).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GapRecord {
    pub namespace_id: u32,
    pub field_id: FieldId,
    /// Why the index is incomplete.
    pub cause: GapCause,
    /// Start of the WAL range that could not be replayed (the field's recorded
    /// checkpoint, or 0). Zero for causes with no WAL range.
    pub from: u64,
    /// End of that range (the WAL tail at detection). Zero for causes with no
    /// WAL range.
    pub to: u64,
    /// Segment ids inside `[from, to)` whose files are gone.
    #[serde(default)]
    pub missing_segments: Vec<u64>,
    /// Milliseconds since the epoch when this gap was first recorded.
    pub detected_at_ms: u64,
    /// What repair has to do.
    pub repair: RepairMode,
}

impl GapRecord {
    /// Merge a newly detected gap into an existing record for the same field.
    ///
    /// Gaps accumulate: a second backstop reclaim before anyone repairs the first
    /// must not discard the first one's worklist. The merged record keeps the
    /// **earliest** `from` and the **latest** `to`, unions the segment ids and the
    /// key worklists, and keeps the original detection time (when the index first
    /// became untrustworthy).
    ///
    /// `FullRebuild` is absorbing: once either side needs a full rebuild, so does
    /// the merged record, and the accumulated keys are dropped — they are no
    /// longer needed and would only cost disk.
    pub fn merge(&mut self, other: GapRecord, key_cap: usize) {
        self.from = self.from.min(other.from);
        self.to = self.to.max(other.to);
        self.missing_segments.extend(other.missing_segments);
        self.missing_segments.sort_unstable();
        self.missing_segments.dedup();
        // Keep the more severe cause visible; a full rebuild subsumes the rest.
        if matches!(other.repair, RepairMode::FullRebuild) {
            self.cause = other.cause;
        }

        let merged = match (std::mem::replace(&mut self.repair, RepairMode::FullRebuild), other.repair) {
            (RepairMode::RowScoped { mut keys }, RepairMode::RowScoped { keys: more }) => {
                keys.extend(more);
                keys.sort_unstable();
                keys.dedup();
                if keys.len() > key_cap {
                    RepairMode::FullRebuild
                } else {
                    RepairMode::RowScoped { keys }
                }
            }
            _ => RepairMode::FullRebuild,
        };
        self.repair = merged;
    }
}

/// Health of one field index: where its persisted state reaches, and whether it
/// is known to be incomplete.
///
/// Returned by `Db::index_health` and surfaced by the admin API so the condition
/// is alertable instead of buried in a log line.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct FieldIndexHealth {
    pub namespace_id: u32,
    pub field_id: FieldId,
    pub field_name: String,
    /// WAL offset the field's persisted index reflects, if it has ever been
    /// checkpointed.
    pub checkpoint_offset: Option<u64>,
    /// Whether the field is currently activated in memory. An inactive field is
    /// not queryable, which is a different condition from a degraded one.
    pub active: bool,
    /// Set when the index is missing updates it cannot recover on its own.
    /// `None` ⇒ healthy.
    pub gap: Option<GapRecord>,
}

impl FieldIndexHealth {
    /// Whether this field is missing updates and needs repair.
    pub fn is_degraded(&self) -> bool {
        self.gap.is_some()
    }
}

/// Write-path sink for field-index updates that were **rejected**.
///
/// A rejected update leaves the row silently absent from that field forever. The
/// write path knows the namespace, field and key exactly, so it is the cheapest
/// gap in FR-001 to capture — but it must not pay for durability inline, so the
/// sink only buffers and the index checkpoint persists what it collected.
///
/// That is sound because the buffer and the index have the *same* durability
/// point: if a crash loses the buffered key, it also lost the index update, and
/// WAL replay re-runs the write, re-rejects it, and re-buffers the key.
///
/// Mirrors the WAL/LSM observer pattern: the write path holds a cheap handle and
/// signals, rather than knowing how a gap is recorded.
pub trait IndexGapSink: Send + Sync {
    /// Note that `field_id` in `namespace_id` refused to index `key`.
    fn note_rejected_update(&self, namespace_id: u32, field_id: FieldId, key: &[u8]);
}

/// Buffers rejected field-index updates until the next index checkpoint turns
/// them into durable [`GapRecord`]s.
///
/// Shared between [`Database`](crate::db::database::Database) and every
/// `KVStore`, so the write path needs no back-reference to the coordinator.
///
/// Bounded per field: past `cap` keys the field is flagged `overflowed` and its
/// keys are dropped, downgrading repair to a full rebuild. A field rejecting
/// every write (a type mismatch against the extractor, say) would otherwise
/// accumulate one key per write in memory — turning a reporting mechanism into
/// its own outage.
#[derive(Default)]
pub struct RejectedUpdateBuffer {
    inner: parking_lot::Mutex<std::collections::HashMap<(u32, FieldId), RejectedField>>,
}

/// Per-field accumulation inside a [`RejectedUpdateBuffer`].
///
/// Keys are a set: a hot key rejected on every write must cost one entry, not
/// one per write, and repair order is irrelevant to a worklist.
#[derive(Default)]
pub struct RejectedField {
    /// Distinct keys this field refused.
    pub keys: std::collections::HashSet<Vec<u8>>,
    /// Set once `keys` hit the cap; `keys` is then cleared and stays empty.
    pub overflowed: bool,
}

impl RejectedUpdateBuffer {
    /// Take everything buffered so far, leaving the buffer empty.
    pub fn drain(&self) -> Vec<((u32, FieldId), RejectedField)> {
        self.inner.lock().drain().collect()
    }

    /// Whether anything is buffered — an O(1) check so the checkpoint path can
    /// skip the work entirely in the overwhelmingly common case.
    pub fn is_empty(&self) -> bool {
        self.inner.lock().is_empty()
    }

    /// Record a rejection, bounded by `cap` keys per field.
    fn note(&self, namespace_id: u32, field_id: FieldId, key: &[u8], cap: usize) {
        let mut guard = self.inner.lock();
        let entry = guard.entry((namespace_id, field_id)).or_default();
        if entry.overflowed {
            return;
        }
        if entry.keys.len() >= cap {
            entry.overflowed = true;
            entry.keys = std::collections::HashSet::new();
            return;
        }
        entry.keys.insert(key.to_vec());
    }
}

impl IndexGapSink for RejectedUpdateBuffer {
    fn note_rejected_update(&self, namespace_id: u32, field_id: FieldId, key: &[u8]) {
        self.note(namespace_id, field_id, key, REJECTED_UPDATE_BUFFER_CAP);
    }
}

/// Per-field cap on buffered rejected-update keys. Matches
/// `Database::GAP_KEY_WORKLIST_CAP`, since crossing either downgrades repair to
/// a full rebuild anyway.
const REJECTED_UPDATE_BUFFER_CAP: usize = 100_000;

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
        let index_base_path = crate::db::layout::index_root(db_path);
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
        self.namespace_path(namespace_id).join(field_id.to_string())
    }

    /// Compute the on-disk path for a namespace's whole index subtree:
    /// `{index_base}/{namespace_id}/`.
    pub fn namespace_path(&self, namespace_id: u32) -> PathBuf {
        crate::db::layout::namespace_index_dir(&self.index_base_path, namespace_id)
    }

    /// Compute the on-disk path for a namespace's dense row-ID map.
    ///
    /// Path: `{index_base}/{namespace_id}/rowmap/` — a sibling of the per-field
    /// directories. (`rowmap` can never collide with a `FieldId`, which is
    /// numeric.) The `RowMap` creates the directory on first use.
    pub fn rowmap_path(&self, namespace_id: u32) -> PathBuf {
        self.namespace_path(namespace_id).join("rowmap")
    }

    /// Remove the entire on-disk index subtree for a namespace.
    ///
    /// Deletes `{index_base}/{namespace_id}/` and everything under it (all field
    /// directories, blob stores, keymaps, and checkpoint markers). Called when a
    /// namespace is dropped. A missing directory is treated as success, so this
    /// is safe to call when the namespace had no indexed fields.
    pub fn remove_namespace_path(&self, namespace_id: u32) -> Result<()> {
        let path = self.namespace_path(namespace_id);
        match std::fs::remove_dir_all(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(KVError::Io(e)),
        }
    }

    /// Remove the on-disk index subtree for a single field.
    ///
    /// Deletes `{index_base}/{namespace_id}/{field_id}/` and everything under it
    /// (bitmap blob store, keymap store, `checkpoint` marker). Called when a
    /// field index is dropped, and again at open to finish a drop that a crash
    /// interrupted — so it is **idempotent**: a missing directory is success.
    ///
    /// Leaves the namespace directory and its sibling `rowmap/` untouched.
    ///
    /// The caller must have persisted the field's `dropped` flag first; see
    /// `FEATURE-REQUEST.md` (FR-001) — *Dropped-index cleanup*.
    pub fn remove_field_path(&self, namespace_id: u32, field_id: FieldId) -> Result<()> {
        let path = self.field_path(namespace_id, field_id);
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

    /// Path of a field's durable gap record.
    fn gap_path(&self, namespace_id: u32, field_id: FieldId) -> PathBuf {
        self.field_path(namespace_id, field_id).join("gap.json")
    }

    /// Read a field's outstanding gap record, if any.
    ///
    /// An unreadable or malformed record returns `None` — it is a *report* about
    /// the index, not index data, so a corrupt one must not fail the open. The
    /// index is still queryable; the worst case is that an operator has to notice
    /// the condition another way.
    pub fn read_gap(&self, namespace_id: u32, field_id: FieldId) -> Option<GapRecord> {
        let bytes = std::fs::read(self.gap_path(namespace_id, field_id)).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    /// Record a gap for a field, merging into any record already there.
    ///
    /// Written through [`write_atomic_durable`](crate::support::write_atomic_durable)
    /// — a gap record that did not survive the crash it describes would be
    /// worthless.
    ///
    /// `key_cap` bounds the merged row-scoped worklist; crossing it downgrades
    /// the record to [`RepairMode::FullRebuild`].
    pub fn record_gap(&self, gap: GapRecord, key_cap: usize) -> Result<()> {
        let namespace_id = gap.namespace_id;
        let field_id = gap.field_id;
        let field_path = self.field_path(namespace_id, field_id);
        // A field whose directory is gone (dropped, mid-cleanup) has no index to
        // repair, so there is nothing to record.
        if !field_path.is_dir() {
            return Ok(());
        }

        let merged = match self.read_gap(namespace_id, field_id) {
            Some(mut existing) => {
                existing.merge(gap, key_cap);
                existing
            }
            None => gap,
        };

        let bytes = serde_json::to_vec_pretty(&merged).map_err(|e| KVError::Serialization(format!("Failed to serialise gap record: {}", e)))?;
        crate::support::write_atomic_durable(&self.gap_path(namespace_id, field_id), &bytes).map_err(|e| {
            KVError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to write gap record for ns={} field={}: {}", namespace_id, field_id, e),
            ))
        })
    }

    /// Clear a field's gap record after a successful repair.
    ///
    /// Idempotent — a missing record is success.
    pub fn clear_gap(&self, namespace_id: u32, field_id: FieldId) -> Result<()> {
        match std::fs::remove_file(self.gap_path(namespace_id, field_id)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(KVError::Io(e)),
        }
    }

    /// Path of a namespace's "no-WAL writes are outstanding" marker.
    fn no_wal_marker_path(&self, namespace_id: u32) -> PathBuf {
        self.namespace_path(namespace_id).join("no_wal_pending")
    }

    /// Durably record that this namespace has taken no-WAL writes which no index
    /// checkpoint has covered yet.
    ///
    /// No-WAL writes update the field index in memory but write **no WAL entry**,
    /// so replay cannot heal a lost update, the index-replay watermark has nothing
    /// to pin, and `detect_replay_gap` sees an entirely intact checkpoint. Under
    /// bulk load the memtable flushes to L0 far more often than the ~15 min index
    /// checkpoint, so a crash readily leaves the value durable and the index
    /// update gone, with no replay path and nothing to notice it.
    ///
    /// This marker is what makes that noticeable. `Database::shutdown` runs an
    /// index checkpoint before anything else, which clears it — so a marker found
    /// at open means the previous run ended **uncleanly** with no-WAL writes
    /// outstanding.
    ///
    /// The caller must write this **before** the storage write it describes.
    /// Marker-then-write can only produce a spurious gap (a full rebuild is
    /// idempotent); write-then-marker can lose both and leave the index silently
    /// incomplete, which is the failure being closed.
    pub fn set_no_wal_pending(&self, namespace_id: u32) -> Result<()> {
        let dir = self.namespace_path(namespace_id);
        std::fs::create_dir_all(&dir)?;
        crate::support::write_atomic_durable(&self.no_wal_marker_path(namespace_id), b"1").map_err(KVError::Io)
    }

    /// Whether a namespace's no-WAL marker is present.
    pub fn no_wal_pending(&self, namespace_id: u32) -> bool {
        self.no_wal_marker_path(namespace_id).exists()
    }

    /// Clear a namespace's no-WAL marker — called once a checkpoint has made its
    /// index state durable. Idempotent.
    pub fn clear_no_wal_pending(&self, namespace_id: u32) -> Result<()> {
        match std::fs::remove_file(self.no_wal_marker_path(namespace_id)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(KVError::Io(e)),
        }
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
