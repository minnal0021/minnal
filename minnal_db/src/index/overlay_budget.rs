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
//! A spill copies bitmaps into the field's memory-mapped files; it does not
//! fsync. The memory moves to page cache, which the kernel can write back and
//! evict, and the next checkpoint makes it durable.
//!
//! The bound: total overlay bytes stay below the hard limit plus, per field
//! written since the limit was crossed, the size of one changed bitmap (each
//! such write spills its own field again).

use std::sync::atomic::{AtomicU64, Ordering};

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
}

impl IndexOverlayBudget {
    /// A budget with the given limits. A `soft` above `hard` is clamped to
    /// `hard`, and a `hard` of 0 makes every write spill at once (no buffering).
    pub fn new(soft: u64, hard: u64) -> Self {
        Self {
            used: AtomicU64::new(0),
            soft: soft.min(hard),
            hard,
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
            self.used.fetch_add(delta as u64, Ordering::Relaxed);
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
}
