use mm3h::Murmur3Hasher;
use std::hash::Hasher;

pub(crate) mod simd_support;

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
