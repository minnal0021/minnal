//! SIMD kernels for container operations: popcount, bitwise set operations,
//! sorted-array merge, position extraction, sums and run-to-bitset conversion.
//! Each kernel has AVX-512 / AVX2 paths (x86_64, chosen at run time), a NEON
//! path (aarch64) and a scalar fallback, tested against the scalar version.

pub mod array_merge;
pub mod bitwise;
pub mod extract;
pub mod popcount;
pub mod run_bitset;
pub mod sum;
