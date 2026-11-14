//! Shared utilities with no better home: key bucketing and prefixes, durable
//! file writes, running a worker's blocking pass, and the `file_lock`, `hex`,
//! `metrics` and SIMD helpers.

use mm3h::Murmur3Hasher;
use std::hash::Hasher;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "doc-store")]
pub(crate) mod file_lock;
pub mod hex;
pub mod metrics;
#[cfg(test)]
pub(crate) mod runtime_probe;
pub(crate) mod simd_support;

/// Run a background worker's blocking pass (fsyncs, flushes, compaction, GC)
/// without holding up the async runtime.
///
/// On a multi-thread runtime, `block_in_place` hands this thread's queued tasks
/// to another thread for the duration. Run plainly, the pass holds up every
/// task scheduled behind it: the system benchmark traced every get over 15 ms
/// to an index checkpoint (~55 ms of fsyncs every 1.75 s; get p99 4-10 ms
/// against a 1.3 ms median). The pass still runs inside the worker's own task,
/// so nothing it uses outlives it (`spawn_blocking` would keep the database alive past its
/// last handle while a pass ran). A current-thread runtime cannot hand its tasks
/// off, so there the pass runs inline.
pub(crate) fn run_blocking_pass<R>(pass: impl FnOnce() -> R) -> R {
    match tokio::runtime::Handle::try_current().map(|h| h.runtime_flavor()) {
        Ok(tokio::runtime::RuntimeFlavor::MultiThread) => tokio::task::block_in_place(pass),
        _ => pass(),
    }
}

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

/// Bucket count used by tests that open a `Db`/`Database`/`KVStore` with the
/// generic (non-sharding-specific) config.
///
/// Each namespace eagerly opens `2 × num_buckets` file descriptors (one LSM L1
/// file plus one value-log file per bucket), held for its lifetime. With the
/// default of 8 buckets that is 16 per namespace, and a fresh database opens
/// several (`default`, `system`, `system_catalog`); a namespace-heavy test can
/// hold several hundred. `cargo test` runs one test per core, so on a
/// many-core machine the aggregate fd demand blows past the typical 1024 soft
/// limit ("Too many open files"). Keeping test databases at 2 buckets cuts the
/// per-namespace fd cost 4× while still exercising the multi-bucket routing,
/// scan-over-buckets, and per-bucket GC paths. Tests that specifically need a
/// single bucket set `num_buckets: 1` explicitly.
#[cfg(test)]
pub(crate) const TEST_NUM_BUCKETS: usize = 2;

/// Test-only `DbConfig` with the small [`TEST_NUM_BUCKETS`] bucket count.
///
/// Used by the document-store and vector tests so the fd mitigation above applies
/// there too. See [`TEST_NUM_BUCKETS`] for the why.
#[cfg(all(test, any(feature = "doc-store", feature = "semantic-search")))]
pub(crate) fn test_db_config() -> crate::DbConfig {
    crate::DbConfig {
        num_buckets: TEST_NUM_BUCKETS,
        ..Default::default()
    }
}

/// Test-only: the files this process has open under `dir`, deleted ones included.
///
/// It looks only under `dir`, so tests running alongside cannot disturb the answer.
#[cfg(all(test, target_os = "linux"))]
pub(crate) fn open_files_under(dir: &std::path::Path) -> Vec<String> {
    let dir = dir.to_string_lossy().into_owned();
    let inside = format!("{dir}/");
    std::fs::read_dir("/proc/self/fd")
        .unwrap()
        .filter_map(|entry| std::fs::read_link(entry.ok()?.path()).ok())
        .map(|target| target.to_string_lossy().into_owned())
        .filter(|target| target.starts_with(&inside))
        .collect()
}

/// Test-only: wait for every file under `dir` to be closed, and fail naming those
/// still open. It waits because a worker's last pass can hold a database for a
/// moment after its last handle goes, and it yields to the runtime so that pass
/// can finish.
#[cfg(all(test, target_os = "linux"))]
pub(crate) async fn assert_no_open_files_under(dir: &std::path::Path, what: &str) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let open = open_files_under(dir);
        if open.is_empty() {
            return;
        }
        assert!(std::time::Instant::now() < deadline, "{what}: {} files still open: {open:#?}", open.len());
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
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

/// Smallest key strictly greater than every key beginning with `prefix` — the
/// exclusive upper bound for a prefix scan expressed as a range `[prefix, …)`.
///
/// Increments the last byte that is not `0xFF` and drops the all-`0xFF` tail.
/// Returns `None` when `prefix` is empty or all `0xFF`: there is no finite
/// exclusive bound, so the scan runs open-ended to the last key.
pub fn prefix_upper_bound(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut upper = prefix.to_vec();
    while let Some(last) = upper.last_mut() {
        if *last != 0xFF {
            *last += 1;
            return Some(upper);
        }
        upper.pop();
    }
    None
}

/// Current wall-clock time as milliseconds since the Unix epoch.
#[cfg(any(feature = "doc-store", feature = "semantic-search"))]
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

#[cfg(test)]
mod cursor_tests {
    use super::prefix_upper_bound;

    #[test]
    fn upper_bound_increments_last_byte() {
        assert_eq!(prefix_upper_bound(b"abc"), Some(b"abd".to_vec()));
    }

    #[test]
    fn upper_bound_carries_over_trailing_ff() {
        assert_eq!(prefix_upper_bound(&[b'a', 0xFF, 0xFF]), Some(vec![b'b']));
    }

    #[test]
    fn upper_bound_all_ff_is_open_ended() {
        assert_eq!(prefix_upper_bound(&[0xFF, 0xFF]), None);
    }

    #[test]
    fn upper_bound_empty_is_open_ended() {
        assert_eq!(prefix_upper_bound(b""), None);
    }
}
