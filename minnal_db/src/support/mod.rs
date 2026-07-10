use mm3h::Murmur3Hasher;
use std::hash::Hasher;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) mod simd_support;

/// fsync a directory so that a create/rename/unlink of one of its entries is
/// durable. A file's own `sync_all` only persists its *contents*, not the
/// directory entry that makes those contents visible by name after a crash.
pub fn fsync_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::File::open(dir)?.sync_all()
}

static ATOMIC_WRITE_SEQ: AtomicU64 = AtomicU64::new(0);

/// Atomically and durably replace the file at `path` with `bytes`.
///
/// The single helper every metadata writer should funnel through. In order:
/// 1. write `bytes` to a unique sibling temp file,
/// 2. `sync_all` the temp file (its contents reach stable storage),
/// 3. `rename` it over `path` (an atomic replace),
/// 4. fsync the parent directory (so the rename itself survives a crash).
///
/// The temp filename carries a process- and call-unique suffix, so concurrent
/// writers targeting the same `path` never collide on the temp file — a shared
/// temp name would let the second writer's `rename` fail with `ENOENT` after
/// the first already moved it.
///
/// A bare `write` + `rename` (without the two fsyncs) leaves both the temp
/// contents and the rename sitting in the page cache, so a power loss could
/// resurrect stale or truncated data even after the call returned `Ok`.
pub fn write_atomic_durable(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let seq = ATOMIC_WRITE_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = path.with_extension(format!("tmp.{}.{}", std::process::id(), seq));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fsync_dir(parent)?;
    }
    Ok(())
}

/// Default bucket count used by both the LSM tree and the sharded value log.
/// All bucket-routing logic in the codebase must use a single bucket count so
/// that a key always lands in the same bucket regardless of which layer handles it.
///
/// Must not exceed `u32::MAX + 1`: bucket indices are stored as `u32` in
/// `ShardedValuePointer` and occupy the upper 32 bits of the encoded `u128`
/// pointer, so a value above that limit would overflow the field.
pub const DEFAULT_NUM_BUCKETS: usize = 16;

const _: () = assert!(
    DEFAULT_NUM_BUCKETS <= (u32::MAX as usize + 1),
    "DEFAULT_NUM_BUCKETS exceeds u32::MAX + 1: bucket indices are stored as u32 in ShardedValuePointer and would overflow"
);

/// Bucket count used by tests that open a `Db`/`Database`/`KVStore` with the
/// generic (non-sharding-specific) config.
///
/// Each namespace eagerly opens `2 × num_buckets` file descriptors (one LSM L1
/// file plus one value-log file per bucket), held for its lifetime. With the
/// production default of 16 buckets a single fresh `Db` (which opens the
/// `default` + `system` namespaces) holds ~65 fds; a namespace-heavy test can
/// hold several hundred. `cargo test` runs one test per core, so on a
/// many-core machine the aggregate fd demand blows past the typical 1024 soft
/// limit ("Too many open files"). Keeping test databases at 2 buckets cuts the
/// per-namespace fd cost ~8× while still exercising the multi-bucket routing,
/// scan-over-buckets, and per-bucket GC paths. Tests that specifically need a
/// single bucket set `num_buckets: 1` explicitly.
#[cfg(test)]
pub(crate) const TEST_NUM_BUCKETS: usize = 2;

/// Test-only `DbConfig` with the small [`TEST_NUM_BUCKETS`] bucket count.
///
/// Used by the `vector_kv` tests so the fd mitigation above applies there too.
/// See [`TEST_NUM_BUCKETS`] for the why.
#[cfg(all(test, feature = "semantic-search"))]
pub(crate) fn test_db_config() -> crate::DbConfig {
    crate::DbConfig {
        num_buckets: TEST_NUM_BUCKETS,
        ..Default::default()
    }
}

/// Validate that a runtime bucket count is within the u32 limit.
#[allow(dead_code)]
pub fn validate_num_buckets(num_buckets: usize) {
    assert!(
        num_buckets > 0 && num_buckets <= (u32::MAX as usize + 1),
        "num_buckets must be between 1 and u32::MAX + 1, got {}",
        num_buckets,
    );
}

const SEED: u32 = 0xFEACBE01;

/// Determine which bucket a key belongs to.
/// Uses Murmur3 hash of the key prefix.
pub fn get_bucket_for_key(key: &[u8], num_buckets: usize) -> u32 {
    let mut hasher = Murmur3Hasher::new_with_seed(SEED);
    hasher.write(key_prefix_of(key).to_le_bytes().as_slice());
    let hash = hasher.finish();
    (hash as usize % num_buckets) as u32
}

/// Extract first 8 bytes of key as u64 for fast prefix comparison
pub fn key_prefix_of(bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let n = bytes.len().min(8);
    buf[..n].copy_from_slice(&bytes[..n]);
    u64::from_be_bytes(buf)
}
