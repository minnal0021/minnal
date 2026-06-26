use simsimd::SpatialSimilarity;

/// RaBitQ's error-bound confidence multiplier ε₀ for the **multi-bit** path: the
/// estimated dot product deviates from the true value by at most `EPSILON ·
/// l2_norm_distance_to_centroid · √error` with high probability, where ε₀ sets the
/// confidence level (larger ⇒ looser but safer bound). 1.9 is the value used by the
/// RaBitQ reference implementation. Identical to `EPSILON_0` in the single-bit path
/// (`quantisation/rabitq/mod.rs`). See the RaBitQ papers referenced in that module.
const EPSILON: f32 = 1.9;

pub struct Quantisation {
    pub quantised_embedding: Vec<u8>,
    pub addition_factor: f32,
    pub scaling_factor: f32,
    pub error_bound: f32,
}

pub struct IndexCalculationData {
    pub shifted_quantization_vector: Vec<f32>,
    pub l2_sqr_distance_from_residual_to_centroid: f32,
    pub l2_norm_distance_from_residual_to_centroid: f32,
    pub ip_residual_shifted_code: f32,
    pub ip_centroid_shifted_code: f32,
    pub dot_product_residual_and_centroid: f32,
}

impl IndexCalculationData {
    pub fn new(
        shifted_quantization_vector: Vec<f32>,
        l2_sqr_distance_from_residual_to_centroid: f32,
        l2_norm_distance_from_residual_to_centroid: f32,
        ip_residual_shifted_code: f32,
        ip_centroid_shifted_code: f32,
        dot_product_residual_and_centroid: f32,
    ) -> Self {
        IndexCalculationData {
            shifted_quantization_vector,
            l2_sqr_distance_from_residual_to_centroid,
            l2_norm_distance_from_residual_to_centroid,
            ip_residual_shifted_code,
            ip_centroid_shifted_code,
            dot_product_residual_and_centroid,
        }
    }
}

pub fn calculate_error_bound(
    embedding_size: usize,
    l2_sqr_distance_to_centroid: f32,
    l2_norm_distance_to_centroid: f32,
    shifted_quantization_vector: &[f32],
    ip_residual_shifted_code: f32,
) -> f32 {
    let scale_multiplier = l2_norm_distance_to_centroid * EPSILON;
    let ip_shifted_code_self = SpatialSimilarity::dot(shifted_quantization_vector, shifted_quantization_vector).unwrap() as f32;
    let ip_multiply_l2_sqr_data = l2_sqr_distance_to_centroid * ip_shifted_code_self;
    let ip_residual_sqr = ip_residual_shifted_code * ip_residual_shifted_code;

    let mut error: f32 = ((ip_multiply_l2_sqr_data / ip_residual_sqr) - 1.0) / (embedding_size - 1) as f32;
    if error < 0.0 {
        error = 0.0001; // Ensure error is non-negative
    }

    let sqrt_error = f32::sqrt(error);

    scale_multiplier * sqrt_error
}

#[allow(dead_code)]
#[inline]
pub fn subtract_inplace(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x -= y;
    }
}

/// Pack a binary vector (`true` = 1, `false` = 0) into a `Vec<u64>`.
/// The first bit of `bits[0]` corresponds to dimension 0.
/// Dimensions are packed little-endian within each `u64` word.
///
/// `dim` need not be a multiple of 64; the last word is zero-padded.
pub fn pack_bits(binary: &[u8]) -> Vec<u64> {
    let words = binary.len().div_ceil(64);
    let mut packed = vec![0u64; words];
    for (i, &b) in binary.iter().enumerate() {
        if b == 1 {
            packed[i / 64] |= 1u64 << (i % 64);
        }
    }
    packed
}

/// Unpack a packed bit-vector back to `Vec<bool>` of length `dim`.
#[allow(dead_code)]
pub fn unpack_bits(packed: &[u64], dim: usize) -> Vec<bool> {
    let mut out = Vec::with_capacity(dim);
    for i in 0..dim {
        out.push((packed[i / 64] >> (i % 64)) & 1 == 1);
    }
    out
}

/// Pack a multi-bit quantised vector (one `u8` value per dimension) into `Vec<u64>`.
/// Eight bytes are stored per word in little-endian order.
/// `dim` need not be a multiple of 8; the last word is zero-padded.
pub fn pack_bytes(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks(8)
        .map(|chunk| {
            let mut arr = [0u8; 8];
            arr[..chunk.len()].copy_from_slice(chunk);
            u64::from_le_bytes(arr)
        })
        .collect()
}

/// Unpack a byte-packed `Vec<u64>` back to `Vec<u8>` of length `dim`.
#[allow(dead_code)]
pub fn unpack_bytes(packed: &[u64], dim: usize) -> Vec<u8> {
    let mut out: Vec<u8> = packed.iter().flat_map(|w| w.to_le_bytes()).collect();
    out.truncate(dim);
    out
}

#[cfg(test)]
mod tests {
    use super::{pack_bits, subtract_inplace, unpack_bits};

    #[test]
    fn subtracts_element_wise() {
        let mut a = vec![5.0_f32, 3.0, 8.0];
        subtract_inplace(&mut a, &[1.0, 2.0, 3.0]);
        assert_eq!(a, vec![4.0, 1.0, 5.0]);
    }

    #[test]
    fn equal_vectors_produce_zeros() {
        let mut a = vec![1.5_f32, -2.0, 0.5];
        let b = a.clone();
        subtract_inplace(&mut a, &b);
        for v in &a {
            assert!(*v == 0.0, "expected 0.0, got {v}");
        }
    }

    #[test]
    fn subtract_zero_vector_is_identity() {
        let original = vec![1.0_f32, -3.0, 7.5];
        let mut a = original.clone();
        subtract_inplace(&mut a, &[0.0, 0.0, 0.0]);
        assert_eq!(a, original);
    }

    #[test]
    fn empty_slices_are_no_op() {
        let mut a: Vec<f32> = vec![];
        subtract_inplace(&mut a, &[]);
        assert!(a.is_empty());
    }

    #[test]
    fn stops_at_shorter_slice() {
        // zip stops at b.len(); a[2] must remain unchanged
        let mut a = vec![10.0_f32, 20.0, 30.0];
        subtract_inplace(&mut a, &[1.0, 2.0]);
        assert_eq!(a, vec![9.0, 18.0, 30.0]);
    }

    // pack_bits / unpack_bits tests

    #[test]
    fn pack_empty() {
        assert!(pack_bits(&[]).is_empty());
    }

    #[test]
    fn pack_all_zeros_is_zero_word() {
        let packed = pack_bits(&[0u8; 64]);
        assert_eq!(packed, vec![0u64]);
    }

    #[test]
    fn pack_all_ones_is_all_ones_word() {
        let packed = pack_bits(&[1u8; 64]);
        assert_eq!(packed, vec![u64::MAX]);
    }

    #[test]
    fn pack_single_bits_land_in_correct_positions() {
        // bit 0 → word[0] bit 0; bit 63 → word[0] bit 63
        assert_eq!(pack_bits(&[1u8, 0u8])[0], 1u64);
        let mut bits = vec![0u8; 64];
        bits[63] = 1;
        assert_eq!(pack_bits(&bits)[0], 1u64 << 63);
    }

    #[test]
    fn pack_spans_two_words() {
        // 65 bits: bit 64 is the LSB of word[1]
        let mut bits = vec![0u8; 65];
        bits[64] = 1;
        let packed = pack_bits(&bits);
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0], 0u64);
        assert_eq!(packed[1], 1u64);
    }

    #[test]
    fn pack_non_multiple_of_64_zero_pads_last_word() {
        let packed = pack_bits(&[1u8, 0u8, 1u8]);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0b101u64);
    }

    #[test]
    fn unpack_roundtrips_arbitrary_pattern() {
        let bits: Vec<u8> = (0u8..130).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
        let packed = pack_bits(&bits);
        let restored = unpack_bits(&packed, bits.len());
        let expected: Vec<bool> = bits.iter().map(|&b| b == 1).collect();
        assert_eq!(restored, expected);
    }

    #[test]
    fn unpack_exact_word_boundary() {
        let bits: Vec<u8> = (0u8..128).map(|i| i % 2).collect();
        let packed = pack_bits(&bits);
        assert_eq!(packed.len(), 2);
        let expected: Vec<bool> = bits.iter().map(|&b| b == 1).collect();
        assert_eq!(unpack_bits(&packed, 128), expected);
    }

    #[test]
    fn unpack_dim_zero_returns_empty() {
        assert!(unpack_bits(&[], 0).is_empty());
    }
}
