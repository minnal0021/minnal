//! Thread-safe shared bitmap.
//!
//! `SharedRoaringBitmap` wraps a `RoaringBitmap` in an `Arc<RwLock<...>>`:
//!
//! - **Reads** (`contains`, `cardinality`, `min`, `max`, …) acquire a shared
//!   read lock — many threads can read concurrently.
//! - **Writes** (`insert`, `remove`, `clear`, …) acquire an exclusive write lock.
//! - **Clone** is O(1) — it bumps the `Arc` reference count, not the bitmap.
//! - **Set operations** (`and`, `or`, `and_not`) acquire read locks on both
//!   operands and return a *new* independent `SharedRoaringBitmap`.
//! - **In-place set operations** return `Result<(), LockTimeout>`.  They use a
//!   bounded wait for the write lock so that passing `self` as both operand and
//!   target (which would deadlock with a blocking lock) times out cleanly instead.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::index::bitmap::RoaringBitmap;

/// Default timeout for write-lock acquisition in inplace ops.
pub const DEFAULT_WRITE_TIMEOUT: Duration = Duration::from_secs(5);

/// Returned when a write lock could not be acquired within the timeout.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("write lock not acquired within {timeout:?}")]
pub struct LockTimeout {
    /// The timeout that was exceeded.
    pub timeout: Duration,
}

/// A cheaply-cloneable, thread-safe handle to a `RoaringBitmap`.
///
/// # Example
/// ```rust
/// use minnal_db::index::sync::SharedRoaringBitmap;
///
/// let bm = SharedRoaringBitmap::new();
/// bm.insert(42);
/// bm.insert(100);
/// assert!(bm.contains(42));
/// assert_eq!(bm.cardinality(), 2);
///
/// let bm2 = bm.clone(); // cheap Arc clone — shares the same bitmap
/// assert!(bm2.contains(100));
/// ```
#[derive(Clone, Debug)]
pub struct SharedRoaringBitmap(Arc<RwLock<RoaringBitmap>>);

impl SharedRoaringBitmap {
    /// Create a new empty shared bitmap.
    pub fn new() -> Self {
        Self(Arc::new(RwLock::new(RoaringBitmap::new())))
    }

    /// Wrap an existing `RoaringBitmap` (e.g. one built via bulk load).
    pub fn from_bitmap(bm: RoaringBitmap) -> Self {
        Self(Arc::new(RwLock::new(bm)))
    }

    /// Returns `true` if `self` and `other` point to the same underlying bitmap.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    // ── Low-level lock access ─────────────────────────────────────────

    /// Acquire a shared read guard. Many threads may hold this concurrently.
    pub fn read(&self) -> RwLockReadGuard<'_, RoaringBitmap> {
        self.0.read()
    }

    /// Acquire an exclusive write guard, blocking until available.
    pub fn write(&self) -> RwLockWriteGuard<'_, RoaringBitmap> {
        self.0.write()
    }

    /// Try to acquire an exclusive write guard within `timeout`.
    /// Returns `None` if the lock could not be acquired in time.
    pub fn try_write_for(&self, timeout: Duration) -> Option<RwLockWriteGuard<'_, RoaringBitmap>> {
        self.0.try_write_for(timeout)
    }

    // ── Mutations (exclusive write lock) ─────────────────────────────

    /// Insert a value. Returns `true` if the value was not already present.
    pub fn insert(&self, value: u128) -> bool {
        self.0.write().insert(value)
    }

    /// Remove a value. Returns `true` if the value was present.
    pub fn remove(&self, value: u128) -> bool {
        self.0.write().remove(value)
    }

    /// Remove all values.
    pub fn clear(&self) {
        self.0.write().clear();
    }

    /// Re-evaluate container types (Array ↔ Bitset ↔ Run).
    pub fn optimize(&self) {
        self.0.write().optimize();
    }

    // ── Queries (shared read lock) ────────────────────────────────────

    /// Returns `true` if the bitmap contains `value`.
    pub fn contains(&self, value: u128) -> bool {
        self.0.read().contains(value)
    }

    /// Number of elements. O(1).
    pub fn cardinality(&self) -> usize {
        self.0.read().cardinality()
    }

    /// Alias for `cardinality()`.
    pub fn len(&self) -> usize {
        self.cardinality()
    }

    pub fn is_empty(&self) -> bool {
        self.0.read().is_empty()
    }

    pub fn min(&self) -> Option<u128> {
        self.0.read().min()
    }

    pub fn max(&self) -> Option<u128> {
        self.0.read().max()
    }

    pub fn num_containers(&self) -> usize {
        self.0.read().num_containers()
    }

    // ── Set operations (read locks on both operands) ──────────────────

    /// AND: returns a new `SharedRoaringBitmap` with the intersection.
    pub fn and(&self, other: &Self) -> Self {
        let a = self.read();
        let b = other.read();
        Self::from_bitmap(a.and(&b))
    }

    /// OR: returns a new `SharedRoaringBitmap` with the union.
    pub fn or(&self, other: &Self) -> Self {
        let a = self.read();
        let b = other.read();
        Self::from_bitmap(a.or(&b))
    }

    /// AND NOT: returns a new `SharedRoaringBitmap` with elements in `self` but not `other`.
    pub fn and_not(&self, other: &Self) -> Self {
        let a = self.read();
        let b = other.read();
        Self::from_bitmap(a.and_not(&b))
    }

    // ── In-place set operations (timeout-bounded write lock) ──────────
    //
    // These acquire a read lock on `other` first (to compute the RHS), then
    // a write lock on `self`.  If `other` is the same handle as `self` the
    // read lock is still held when we request the write lock, which would
    // deadlock with a blocking acquire.  `parking_lot::RwLock::try_write_for`
    // gives up after `timeout` and returns `Err(LockTimeout)` instead.

    /// In-place AND with a custom write-lock timeout.
    ///
    /// Returns `Err(LockTimeout)` if the write lock on `self` could not be
    /// acquired within `timeout` — including the case where `other` is the
    /// same handle as `self`.
    pub fn and_inplace_timeout(&self, other: &Self, timeout: Duration) -> Result<(), LockTimeout> {
        let b = other.read();
        self.try_write_for(timeout).ok_or(LockTimeout { timeout })?.and_inplace(&b);
        Ok(())
    }

    /// In-place OR with a custom write-lock timeout.
    pub fn or_inplace_timeout(&self, other: &Self, timeout: Duration) -> Result<(), LockTimeout> {
        let b = other.read();
        self.try_write_for(timeout).ok_or(LockTimeout { timeout })?.or_inplace(&b);
        Ok(())
    }

    /// In-place AND NOT with a custom write-lock timeout.
    pub fn and_not_inplace_timeout(&self, other: &Self, timeout: Duration) -> Result<(), LockTimeout> {
        let b = other.read();
        self.try_write_for(timeout).ok_or(LockTimeout { timeout })?.and_not_inplace(&b);
        Ok(())
    }

    /// In-place AND using [`DEFAULT_WRITE_TIMEOUT`].
    pub fn and_inplace(&self, other: &Self) -> Result<(), LockTimeout> {
        self.and_inplace_timeout(other, DEFAULT_WRITE_TIMEOUT)
    }

    /// In-place OR using [`DEFAULT_WRITE_TIMEOUT`].
    pub fn or_inplace(&self, other: &Self) -> Result<(), LockTimeout> {
        self.or_inplace_timeout(other, DEFAULT_WRITE_TIMEOUT)
    }

    /// In-place AND NOT using [`DEFAULT_WRITE_TIMEOUT`].
    pub fn and_not_inplace(&self, other: &Self) -> Result<(), LockTimeout> {
        self.and_not_inplace_timeout(other, DEFAULT_WRITE_TIMEOUT)
    }

    // ── Atomic multi-step access ──────────────────────────────────────

    /// Call `f` with a read-locked reference to the inner bitmap.
    ///
    /// Use this to iterate or perform multiple read operations atomically
    /// without releasing the lock between calls.
    ///
    /// ```rust
    /// use minnal_db::index::sync::SharedRoaringBitmap;
    ///
    /// let bm = SharedRoaringBitmap::new();
    /// bm.insert(1); bm.insert(2); bm.insert(3);
    ///
    /// bm.with_read(|inner| {
    ///     let values: Vec<u128> = inner.iter().collect();
    ///     assert_eq!(values, vec![1, 2, 3]);
    /// });
    /// ```
    pub fn with_read<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&RoaringBitmap) -> R,
    {
        f(&self.read())
    }

    /// Call `f` with a write-locked reference to the inner bitmap.
    ///
    /// Use this to perform multiple mutations atomically.
    pub fn with_write<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut RoaringBitmap) -> R,
    {
        f(&mut self.write())
    }
}

impl Default for SharedRoaringBitmap {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<u128> for SharedRoaringBitmap {
    fn from_iter<I: IntoIterator<Item = u128>>(iter: I) -> Self {
        Self::from_bitmap(iter.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn basic_insert_contains_remove() {
        let bm = SharedRoaringBitmap::new();
        assert!(bm.insert(42));
        assert!(!bm.insert(42));
        assert!(bm.contains(42));
        assert_eq!(bm.cardinality(), 1);
        assert!(bm.remove(42));
        assert!(!bm.contains(42));
        assert_eq!(bm.cardinality(), 0);
    }

    #[test]
    fn clone_shares_state() {
        let bm = SharedRoaringBitmap::new();
        bm.insert(10);
        let bm2 = bm.clone();
        bm.insert(20);
        assert!(bm2.contains(20));
        assert_eq!(bm2.cardinality(), 2);
    }

    #[test]
    fn ptr_eq_detects_same_handle() {
        let bm = SharedRoaringBitmap::new();
        let bm2 = bm.clone();
        let bm3 = SharedRoaringBitmap::new();
        assert!(bm.ptr_eq(&bm2));
        assert!(!bm.ptr_eq(&bm3));
    }

    #[test]
    fn set_operations() {
        let a: SharedRoaringBitmap = [1u128, 3, 5, 7].into_iter().collect();
        let b: SharedRoaringBitmap = [3u128, 7, 9].into_iter().collect();

        let and_vals: Vec<u128> = a.and(&b).with_read(|bm| bm.iter().collect());
        assert_eq!(and_vals, vec![3, 7]);

        let or_vals: Vec<u128> = a.or(&b).with_read(|bm| bm.iter().collect());
        assert_eq!(or_vals, vec![1, 3, 5, 7, 9]);

        let andnot_vals: Vec<u128> = a.and_not(&b).with_read(|bm| bm.iter().collect());
        assert_eq!(andnot_vals, vec![1, 5]);
    }

    #[test]
    fn inplace_distinct_handles_succeeds() {
        let a: SharedRoaringBitmap = [1u128, 3, 5].into_iter().collect();
        let b: SharedRoaringBitmap = [3u128, 5, 7].into_iter().collect();
        a.and_inplace(&b).expect("should succeed with distinct handles");
        let vals: Vec<u128> = a.with_read(|bm| bm.iter().collect());
        assert_eq!(vals, vec![3, 5]);
    }

    #[test]
    fn inplace_same_handle_times_out() {
        let a: SharedRoaringBitmap = [1u128, 3, 5].into_iter().collect();
        // Using a short timeout so the test doesn't stall
        let short = Duration::from_millis(100);
        let result = a.and_inplace_timeout(&a.clone(), short);
        // clone() shares the Arc — ptr_eq is true — write lock cannot be acquired
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().timeout, short);
    }

    #[test]
    fn concurrent_reads() {
        let bm = Arc::new(SharedRoaringBitmap::new());
        for i in 0u128..100 {
            bm.insert(i);
        }
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let bm = Arc::clone(&bm);
                thread::spawn(move || {
                    assert_eq!(bm.cardinality(), 100);
                    for i in 0u128..100 {
                        assert!(bm.contains(i));
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn concurrent_writes_are_serialised() {
        let bm = Arc::new(SharedRoaringBitmap::new());
        let handles: Vec<_> = (0u128..4)
            .map(|t| {
                let bm = Arc::clone(&bm);
                thread::spawn(move || {
                    for i in 0u128..250 {
                        bm.insert(t * 1_000_000 + i);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(bm.cardinality(), 1000);
    }

    #[test]
    fn with_read_atomic_iteration() {
        let bm: SharedRoaringBitmap = (0u128..10).collect();
        let values = bm.with_read(|inner| inner.iter().collect::<Vec<_>>());
        assert_eq!(values, (0u128..10).collect::<Vec<_>>());
    }

    #[test]
    fn with_write_atomic_bulk_insert() {
        let bm = SharedRoaringBitmap::new();
        bm.with_write(|inner| {
            for i in 0u128..50 {
                inner.insert(i);
            }
        });
        assert_eq!(bm.cardinality(), 50);
    }

    #[test]
    fn from_bitmap_wraps_existing() {
        let inner = RoaringBitmap::from_sorted_iter(0u128..500);
        let bm = SharedRoaringBitmap::from_bitmap(inner);
        assert_eq!(bm.cardinality(), 500);
        assert!(bm.contains(499));
    }

    #[test]
    fn lock_timeout_display() {
        let e = LockTimeout {
            timeout: Duration::from_millis(250),
        };
        assert!(e.to_string().contains("250ms"));
    }
}
