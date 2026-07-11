pub mod array_merge;
pub mod bitwise;
pub mod extract;
pub mod popcount;
pub mod run_bitset;
pub mod sum;

/// Returns true if the CPU supports AVX-512 VPOPCNTDQ (native u64 popcount via VPOPCNTQ).
#[inline]
pub fn has_avx512_popcount() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Returns true if the CPU supports AVX-512F (needed for zero-extend / reduce ops).
#[inline]
pub fn has_avx512f() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Returns true if the CPU supports AVX2 (256-bit integer SIMD).
#[inline]
pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
