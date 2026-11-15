//! Striped per-key write locks, one set per namespace.
//!
//! The outermost lock on every write path. Each namespace's [`KVStore`] owns
//! 1,024 mutexes and a key takes the one its hash picks (its *stripe*). Every
//! writer holds its key's stripe from the moment it draws its write sequence
//! until the write is applied to the memtable, so the writes to one key apply
//! in sequence order, and a writer that draws a sequence after acquiring the
//! stripe is newer than every copy of the key already stored.
//!
//! Who takes a stripe: every user write — WAL-backed, the vector index WAL's
//! logged writes, the no-WAL writes, and `merge` (which reads a key, runs a user
//! closure and writes the result back as one indivisible step) — and the field
//! index's out-of-band reindex. Reads and scans never take one, and neither do
//! the system actions, value-log GC and TTL expiry: they keep the user seq of the
//! copy they replace and raise its system seq (`storage::lsm::version`), so a
//! user write that drew its seq before them is newer whenever it lands. Only a
//! copy whose system seq is used up makes them fall back to the stripe.
//!
//! Why a lock of its own rather than one of the two the write path already
//! takes:
//!
//! - `Database::wal_metadata` is a single global lock held across the WAL
//!   append. Holding it across a merge closure would run user code under the
//!   lock every namespace's writes queue behind.
//! - The value-log bucket mutex is taken *after* the WAL lock is released
//!   (`put_ns` → `KVStore::put_to_storage_seq`). Making a merge atomic with it
//!   would mean taking the bucket lock first and the WAL lock second, inverting
//!   the established order.
//!
//! So the stripe sits strictly above both. The order is
//! **`key stripe → wal_metadata → (released) → value-log bucket`**, uniformly:
//! nothing waits for a stripe while holding either of the others, so a cycle is
//! impossible rather than merely unlikely.
//!
//! The stripes are per namespace so that a writer only ever waits on work in
//! its own namespace: a slow merge closure in one namespace never holds up
//! writes to another. A stripe is one byte (an
//! uncontended acquire is one atomic compare-and-swap), so 1,024 per namespace
//! is about 1 KiB.
//!
//! [`KVStore`]: crate::kv_store::KVStore

use mm3h::Murmur3Hasher;
use parking_lot::{Mutex, MutexGuard};
use std::hash::Hasher;

/// Number of stripes per namespace. Fixed rather than configurable: it only
/// trades a byte per stripe against how often two unrelated keys of one
/// namespace collide onto one lock.
const STRIPES: usize = 1024;

const _: () = assert!(STRIPES.is_power_of_two(), "STRIPES must be a power of two for the mask in `stripe_of`");

/// Distinct from the value log's seed so a key's stripe and its storage bucket
/// are independent.
const SEED: u32 = 0x5D9E_2A17;

/// A fixed array of mutexes, one of which is picked per key.
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

    /// Lock the stripe guarding `key`.
    ///
    /// Not reentrant — `parking_lot::Mutex` deadlocks on a second acquire from
    /// the same thread. Callers must never block on two stripes at once, and a
    /// merge closure must not call back into the database.
    pub(crate) fn guard(&self, key: &[u8]) -> MutexGuard<'_, ()> {
        self.stripes[self.stripe_of(key)].lock()
    }

    /// Hashes the **whole** key, deliberately not
    /// [`get_bucket_for_key`](crate::support::get_bucket_for_key), which hashes
    /// only the first 8 bytes: reusing it would collapse every key sharing an
    /// 8-byte prefix — the common shape for string keys — onto one stripe.
    pub(crate) fn stripe_of(&self, key: &[u8]) -> usize {
        let mut hasher = Murmur3Hasher::new_with_seed(SEED);
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
        assert_eq!(locks.stripe_of(b"alpha"), locks.stripe_of(b"alpha"));
    }

    /// The property `get_bucket_for_key` cannot provide: keys that share their
    /// first 8 bytes must still spread across stripes.
    #[test]
    fn keys_sharing_an_eight_byte_prefix_spread_across_stripes() {
        let locks = KeyLocks::new();
        let stripes: std::collections::HashSet<usize> = (0..64).map(|i| locks.stripe_of(format!("job_content_{}", i).as_bytes())).collect();
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
                        let _g = locks.guard(b"contended");
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
