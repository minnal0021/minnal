//! Striped per-key write locks.
//!
//! The outermost lock on every WAL-backed write path. It exists for
//! [`Database::merge_ns`](crate::db::database::Database::merge_ns), which has to
//! read a key, run a user closure over the value, and write the result back as
//! one indivisible step.
//!
//! Why a new lock rather than one of the two the write path already takes:
//!
//! - `Database::wal_metadata` is a single global `RwLock` held across the WAL
//!   fsync. Holding it across the merge closure would run arbitrary user code
//!   under the lock that every namespace's writes queue behind.
//! - The value-log bucket mutex is taken *after* the WAL lock is released
//!   (`put_ns` → `KVStore::put_to_storage_seq`). Making a merge atomic with it
//!   would mean taking the bucket lock first and the WAL lock second, inverting
//!   the established order and constraining every future WAL-touching path.
//!
//! So the stripe sits strictly above both. The order is
//! **`key stripe → wal_metadata → (released) → value-log bucket`**, uniformly,
//! and nothing else in the crate takes a stripe — which is what makes a cycle
//! impossible rather than merely unlikely.
//!
//! The cost on `put`/`delete` is negligible: those writes already serialise
//! globally on `wal_metadata.write()` across an fsync, so an uncontended
//! `parking_lot::Mutex` acquire disappears into the noise.

use mm3h::Murmur3Hasher;
use parking_lot::{Mutex, MutexGuard};
use std::hash::Hasher;

/// Number of stripes. Fixed rather than configurable: it only trades a byte per
/// stripe against how often two unrelated keys collide onto one lock.
const STRIPES: usize = 1024;

const _: () = assert!(STRIPES.is_power_of_two(), "STRIPES must be a power of two for the mask in `guard`");

/// Distinct from the value log's seed so a key's stripe and its storage bucket
/// are independent.
const SEED: u32 = 0x5D9E_2A17;

/// A fixed array of mutexes, one of which is picked per `(namespace, key)`.
pub(crate) struct KeyLocks {
    stripes: Box<[Mutex<()>]>,
}

impl KeyLocks {
    pub(crate) fn new() -> Self {
        let mut stripes = Vec::with_capacity(STRIPES);
        stripes.resize_with(STRIPES, || Mutex::new(()));
        Self {
            stripes: stripes.into_boxed_slice(),
        }
    }

    /// Lock the stripe guarding `key` in `namespace_id`.
    ///
    /// Not reentrant — `parking_lot::Mutex` deadlocks on a second acquire from
    /// the same thread. Callers must never take two stripes at once, and a
    /// merge closure must not call back into the database.
    pub(crate) fn guard(&self, namespace_id: u32, key: &[u8]) -> MutexGuard<'_, ()> {
        self.stripes[self.stripe_of(namespace_id, key)].lock()
    }

    /// Hashes the **whole** key, deliberately not
    /// [`get_bucket_for_key`](crate::support::get_bucket_for_key), which hashes
    /// only the first 8 bytes: reusing it would collapse every key sharing an
    /// 8-byte prefix — the common shape for string keys — onto one stripe.
    fn stripe_of(&self, namespace_id: u32, key: &[u8]) -> usize {
        let mut hasher = Murmur3Hasher::new_with_seed(SEED);
        hasher.write(&namespace_id.to_le_bytes());
        hasher.write(key);
        (hasher.finish() as usize) & (STRIPES - 1)
    }
}

impl Default for KeyLocks {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[test]
    fn the_same_key_always_maps_to_the_same_stripe() {
        let locks = KeyLocks::new();
        assert_eq!(locks.stripe_of(7, b"alpha"), locks.stripe_of(7, b"alpha"));
    }

    #[test]
    fn the_namespace_participates_in_the_stripe() {
        // Not a correctness requirement (a collision is legal), just a check
        // that the namespace is actually mixed in rather than ignored.
        let locks = KeyLocks::new();
        let differs = (0..64u32).any(|ns| locks.stripe_of(ns, b"k") != locks.stripe_of(ns + 1, b"k"));
        assert!(differs, "namespace_id is not affecting the stripe");
    }

    /// The property `get_bucket_for_key` cannot provide: keys that share their
    /// first 8 bytes must still spread across stripes.
    #[test]
    fn keys_sharing_an_eight_byte_prefix_spread_across_stripes() {
        let locks = KeyLocks::new();
        let stripes: std::collections::HashSet<usize> = (0..64).map(|i| locks.stripe_of(0, format!("job_content_{}", i).as_bytes())).collect();
        assert!(stripes.len() > 32, "expected keys to spread, got {} distinct stripes", stripes.len());
    }

    #[test]
    fn the_stripe_serialises_concurrent_holders() {
        let locks = Arc::new(KeyLocks::new());
        let counter = Arc::new(AtomicU64::new(0));
        let observed_overlap = Arc::new(AtomicU64::new(0));
        let live = Arc::new(AtomicU64::new(0));

        std::thread::scope(|s| {
            for _ in 0..8 {
                let (locks, counter, live, overlap) = (Arc::clone(&locks), Arc::clone(&counter), Arc::clone(&live), Arc::clone(&observed_overlap));
                s.spawn(move || {
                    for _ in 0..500 {
                        let _g = locks.guard(0, b"contended");
                        if live.fetch_add(1, Ordering::SeqCst) != 0 {
                            overlap.fetch_add(1, Ordering::SeqCst);
                        }
                        // Non-atomic read-modify-write: only the stripe makes it safe.
                        let seen = counter.load(Ordering::SeqCst);
                        std::thread::yield_now();
                        counter.store(seen + 1, Ordering::SeqCst);
                        live.fetch_sub(1, Ordering::SeqCst);
                    }
                });
            }
        });

        assert_eq!(observed_overlap.load(Ordering::SeqCst), 0, "two threads held the same stripe at once");
        assert_eq!(counter.load(Ordering::SeqCst), 8 * 500, "lost update under the stripe lock");
    }
}
