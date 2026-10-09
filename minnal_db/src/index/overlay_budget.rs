//! The memory budget for field-index write buffers.
//!
//! Each [`FieldIndex`](crate::index::FieldIndex) keeps the bitmaps it changed
//! since its last write-out in memory (its *overlay*) and writes them to its
//! files at the next index checkpoint. One `IndexOverlayBudget` is shared by
//! every field of a database and counts the heap bytes all their overlays hold:
//!
//! - past the **soft** limit the owner asks for an early checkpoint;
//! - past the **hard** limit the writer that crossed it writes its own field's
//!   overlay out before returning (a *spill*), so the bound holds even when the
//!   checkpoint worker is slow or not running.
//!
//! A spill appends the changed bitmaps to the field's memory-mapped files and
//! syncs them in crash order (values, then slots), so the memory moves to page
//! cache, which the kernel can evict. The writer that triggers a hard-limit
//! spill pays those two syncs while holding its field's write lock; the
//! checkpoint path instead syncs under a read lock.
//!
//! The bound: total overlay bytes stay below the hard limit plus, per field
//! written since the limit was crossed, the size of one changed bitmap (each
//! such write spills its own field again).

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Default soft limit: request an early checkpoint (32 MiB).
pub const DEFAULT_INDEX_OVERLAY_SOFT_BYTES: u64 = 32 * 1024 * 1024;
/// Default hard limit: the writer spills its field before returning (64 MiB).
pub const DEFAULT_INDEX_OVERLAY_HARD_BYTES: u64 = 64 * 1024 * 1024;

/// Shared byte counter for field-index overlays, with a soft and a hard limit.
/// See the module docs.
#[derive(Debug)]
pub struct IndexOverlayBudget {
    used: AtomicU64,
    soft: u64,
    hard: u64,
    /// Highest `used` seen since the budget was created.
    peak: AtomicU64,
    /// Times `used` rose from below the soft limit to at or above it.
    soft_crossings: AtomicU64,
    /// Spills a writer ran because the hard limit was exceeded, and their
    /// total duration.
    hard_spills: AtomicU64,
    hard_spill_nanos: AtomicU64,
}

/// A snapshot of an [`IndexOverlayBudget`]: current and peak use, its limits,
/// and how often each limit fired. Served under `index_overlay` in
/// `GET /admin/storage/ops-metrics`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct IndexOverlayStats {
    /// Bytes every field's write buffer holds now.
    pub used_bytes: u64,
    /// Highest `used_bytes` since the database opened.
    pub peak_bytes: u64,
    /// The soft limit (requests an early checkpoint).
    pub soft_limit_bytes: u64,
    /// The hard limit (the writer spills its own field).
    pub hard_limit_bytes: u64,
    /// Times use crossed the soft limit going up.
    pub soft_crossings: u64,
    /// Writes that spilled their field because use was over the hard limit.
    pub hard_spills: u64,
    /// Total time those writes spent spilling, in microseconds.
    pub hard_spill_micros: u64,
}

impl IndexOverlayBudget {
    /// A budget with the given limits. A `soft` above `hard` is clamped to
    /// `hard`, and a `hard` of 0 makes every write spill at once (no buffering).
    pub fn new(soft: u64, hard: u64) -> Self {
        Self {
            used: AtomicU64::new(0),
            soft: soft.min(hard),
            hard,
            peak: AtomicU64::new(0),
            soft_crossings: AtomicU64::new(0),
            hard_spills: AtomicU64::new(0),
            hard_spill_nanos: AtomicU64::new(0),
        }
    }

    /// Bytes currently held by every overlay sharing this budget.
    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }

    /// The soft limit in bytes.
    pub fn soft_limit(&self) -> u64 {
        self.soft
    }

    /// The hard limit in bytes.
    pub fn hard_limit(&self) -> u64 {
        self.hard
    }

    /// True once overlays hold at least the soft limit: time to checkpoint.
    pub fn over_soft(&self) -> bool {
        self.used() >= self.soft
    }

    /// True once overlays exceed the hard limit: the writer must spill.
    pub fn over_hard(&self) -> bool {
        self.used() > self.hard
    }

    /// Apply a change in one overlay's size.
    pub(crate) fn charge(&self, delta: i64) {
        if delta >= 0 {
            let before = self.used.fetch_add(delta as u64, Ordering::Relaxed);
            let after = before + delta as u64;
            self.peak.fetch_max(after, Ordering::Relaxed);
            if before < self.soft && after >= self.soft {
                self.soft_crossings.fetch_add(1, Ordering::Relaxed);
            }
        } else {
            let d = delta.unsigned_abs();
            // Saturate rather than wrap: a release can never exceed what was
            // charged, but a wrapped counter would read as permanently full.
            let _ = self
                .used
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |u| Some(u.saturating_sub(d)));
        }
    }
}

impl IndexOverlayBudget {
    /// Record a spill a writer ran because the hard limit was exceeded.
    pub(crate) fn note_hard_spill(&self, took: Duration) {
        self.hard_spills.fetch_add(1, Ordering::Relaxed);
        self.hard_spill_nanos.fetch_add(took.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Current use, peak, limits and counters.
    pub fn stats(&self) -> IndexOverlayStats {
        IndexOverlayStats {
            used_bytes: self.used(),
            peak_bytes: self.peak.load(Ordering::Relaxed),
            soft_limit_bytes: self.soft,
            hard_limit_bytes: self.hard,
            soft_crossings: self.soft_crossings.load(Ordering::Relaxed),
            hard_spills: self.hard_spills.load(Ordering::Relaxed),
            hard_spill_micros: self.hard_spill_nanos.load(Ordering::Relaxed) / 1_000,
        }
    }
}

impl Default for IndexOverlayBudget {
    fn default() -> Self {
        Self::new(DEFAULT_INDEX_OVERLAY_SOFT_BYTES, DEFAULT_INDEX_OVERLAY_HARD_BYTES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_and_charges() {
        let b = IndexOverlayBudget::new(100, 200);
        b.charge(150);
        assert!(b.over_soft() && !b.over_hard());
        b.charge(51);
        assert!(b.over_hard());
        b.charge(-1000);
        assert_eq!(b.used(), 0, "a release saturates at zero");
        assert_eq!(IndexOverlayBudget::new(500, 200).soft_limit(), 200, "soft is clamped to hard");
    }

    #[test]
    fn stats_track_peak_and_soft_crossings() {
        let b = IndexOverlayBudget::new(100, 200);
        b.charge(60);
        b.charge(60); // crosses soft
        b.charge(30);
        b.charge(-150);
        b.charge(120); // crosses soft again
        b.note_hard_spill(Duration::from_micros(250));
        let s = b.stats();
        assert_eq!((s.used_bytes, s.peak_bytes, s.soft_crossings), (120, 150, 2));
        assert_eq!((s.hard_spills, s.hard_spill_micros), (1, 250));
    }
}
