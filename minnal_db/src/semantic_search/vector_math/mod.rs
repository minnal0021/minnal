//! Vector maths utilities: L2 normalisation, residuals, and the RaBitQ
//! quantisation/bit-packing primitives shared by the quantisation modules.

use std::error::Error;

const EPS: f32 = 1e-5;

/// Normalises `embedding` to unit L2 length in-place.
///
/// The L2 norm is computed via [`simsimd`]'s dot-product kernel, which
/// automatically selects the best SIMD backend available at runtime
/// (AVX-512, AVX2, NEON, etc.).  If the norm is smaller than `1e-5` the
/// vector is left unchanged to avoid division by near-zero.
pub fn normalise_l2(embedding: &mut [f32]) {
    use simsimd::SpatialSimilarity;

    let norm_sq = match SpatialSimilarity::dot(embedding, embedding) {
        Some(v) => v as f32,
        None => return,
    };

    if norm_sq < EPS * EPS {
        return;
    }

    let inv_norm = norm_sq.sqrt().recip();
    for x in embedding.iter_mut() {
        *x *= inv_norm;
    }
}

pub fn residual_vector(x: &[f32], centroid: &[f32]) -> Vec<f32> {
    x.iter().zip(centroid.iter()).map(|(xi, ci)| xi - ci).collect()
}

pub fn normalise(embedding: &[f32], distance: f32) -> Vec<f32> {
    let mut normalised_embedding = Vec::new();
    for value in embedding {
        normalised_embedding.push(value / distance);
    }
    normalised_embedding
}
pub fn binary_quantize(residual_vector: &[f32]) -> Vec<u8> {
    residual_vector.iter().map(|&x| if x >= 0.0 { 1u8 } else { 0u8 }).collect()
}

pub fn from_centroid(centroid: &[f32], embeddings: &[f32]) -> Vec<f32> {
    let mut from_centroid_embeddings = Vec::new();
    for (i, value) in embeddings.iter().enumerate() {
        from_centroid_embeddings.push(value - centroid[i]);
    }
    from_centroid_embeddings
}

pub fn total_number_of_one_bits(embeddings: &[u8]) -> u32 {
    let mut total_number_of_one_bits = 0;
    for value in embeddings {
        total_number_of_one_bits += value.count_ones();
    }
    total_number_of_one_bits
}

pub fn snap_to_grid(embedding: &[u8], grid: usize) -> Vec<f32> {
    let mut snap_to_grid_embeddings = Vec::with_capacity(embedding.len());
    for value in embedding.iter() {
        let quantized_value = (2.0 * *value as f32 - 1.0) / f32::sqrt(grid as f32);
        snap_to_grid_embeddings.push(quantized_value);
    }
    snap_to_grid_embeddings
}

#[allow(dead_code)]
pub(crate) fn total(embeddings: &[u8]) -> u32 {
    let mut total = 0;
    for value in embeddings {
        total += *value as u32;
    }
    total
}

/// Smallest component of `embedding` — the lower bound of the scalar-quantisation grid.
pub fn grid_min(embedding: &[f32]) -> f32 {
    let mut min = f32::MAX;
    for value in embedding {
        if *value < min {
            min = *value;
        }
    }
    min
}

/// Largest component of `embedding` — the upper bound of the scalar-quantisation grid.
pub fn grid_max(embedding: &[f32]) -> f32 {
    let mut max = f32::MIN;
    for value in embedding {
        if *value > max {
            max = *value;
        }
    }
    max
}

/// Width of one scalar-quantisation bucket for `embedding` at `bits_count` bits:
/// `(max - min) / (2^bits - 1)`.
pub fn grid_width(embedding: &[f32], bits_count: i32) -> f32 {
    let min = grid_min(embedding);
    let max = grid_max(embedding);
    (max - min) / (2.0f32.powi(bits_count) - 1.0)
}

pub fn scalar_quantize(embedding: &[f32], bits_count: i32) -> Vec<u8> {
    let width = grid_width(embedding, bits_count);
    let min = grid_min(embedding);
    let mut scalar_quantized_embeddings = Vec::new();
    for value in embedding {
        let quantized_value = ((value - min) / width).round() as u8;
        scalar_quantized_embeddings.push(quantized_value);
    }
    scalar_quantized_embeddings
}

pub fn pack(binary_quantized_embeddings: &[u8]) -> Result<Vec<u128>, Box<dyn Error>> {
    if !binary_quantized_embeddings.len().is_multiple_of(16) {
        return Err(Box::from("Input length must be divisible by 16".to_string()));
    }

    Ok(binary_quantized_embeddings
        .chunks_exact(16)
        .map(|chunk| {
            let mut array = [0u8; 16];
            array.copy_from_slice(chunk);
            u128::from_le_bytes(array)
        })
        .collect())
}

pub fn pack_to_signed_128(binary_quantized_embeddings: &[u8]) -> Result<Vec<i128>, Box<dyn Error>> {
    if !binary_quantized_embeddings.len().is_multiple_of(16) {
        return Err(Box::from("Input length must be divisible by 16".to_string()));
    }

    Ok(binary_quantized_embeddings
        .chunks_exact(16)
        .map(|chunk| {
            let mut array = [0u8; 16];
            array.copy_from_slice(chunk);
            i128::from_le_bytes(array)
        })
        .collect())
}

pub fn pack_to_signed_64(binary_quantized_embeddings: &[u8]) -> Result<Vec<i64>, Box<dyn Error>> {
    if !binary_quantized_embeddings.len().is_multiple_of(8) {
        return Err(Box::from("Input length must be divisible by 8".to_string()));
    }

    Ok(binary_quantized_embeddings
        .chunks_exact(8)
        .map(|chunk| {
            let mut array = [0u8; 8];
            array.copy_from_slice(chunk);
            i64::from_le_bytes(array)
        })
        .collect())
}

pub fn unpack(packed_embeddings: &[u128]) -> Vec<u8> {
    packed_embeddings.iter().flat_map(|val| val.to_le_bytes()).collect()
}

pub fn unpack_from_signed_64(packed_embeddings: &[i64]) -> Vec<u8> {
    packed_embeddings.iter().flat_map(|val| val.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    #[test]
    fn test_pack_as_16_bytes() {
        let bits: Vec<u8> = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let packed_bits = pack(&bits).expect("Error packing bits");
        let unpacked_bits = unpack(&packed_bits);
        assert_eq!(packed_bits.len(), 1);
        assert_eq!(unpacked_bits, bits);
    }

    #[test]
    fn test_pack_as_16_bytes_large_embeddings() {
        let mut bits: Vec<u8> = Vec::with_capacity(768);
        for i in 0..768 {
            bits.push((i % 2) as u8);
        }
        let packed_bits = pack(&bits).expect("Error packing bits");
        let unpacked_bits = unpack(&packed_bits);
        assert_eq!(packed_bits.len(), 48);
        assert_eq!(unpacked_bits, bits);
    }

    #[test]
    fn test_snap_to_grid() {
        let embedding = vec![1, 0];
        let snap_to_grid_embeddings = snap_to_grid(&embedding, 2);
        assert_relative_eq!(snap_to_grid_embeddings[0], 0.70710677, epsilon = 0.00001);
        assert_relative_eq!(snap_to_grid_embeddings[1], -0.70710677, epsilon = 0.00001);
    }

    #[test]
    fn test_scalar_quantize() {
        let embedding = vec![0.37, -0.92];
        let scalar_quantized_embeddings = scalar_quantize(&embedding, 4);
        assert_eq!(scalar_quantized_embeddings, vec![15, 0]);
    }

    #[test]
    fn quantisation_bucket_width_scales_with_value_range_and_bit_depth() {
        let embedding = vec![1.0, 2.0, 3.0];
        let width = grid_width(&embedding, 4);
        assert_relative_eq!(width, 0.13333334, epsilon = 0.00001);
    }

    #[test]
    fn grid_min_returns_smallest_component_of_embedding() {
        let embedding = vec![1.0, 2.0, 3.0];
        let min = grid_min(&embedding);
        assert_eq!(min, 1.0);
    }

    #[test]
    fn grid_max_returns_largest_component_of_embedding() {
        let embedding = vec![1.0, 2.0, 3.0];
        let max = grid_max(&embedding);
        assert_eq!(max, 3.0);
    }

    #[test]
    fn total_sums_all_elements() {
        let embeddings = vec![1, 0, 1];
        let total = total(&embeddings);
        assert_eq!(total, 2);
    }

    #[test]
    fn test_total_number_of_one_bits() {
        let embeddings = vec![1, 0, 1];
        let total_number_of_one_bits = total_number_of_one_bits(&embeddings);
        assert_eq!(total_number_of_one_bits, 2);
    }

    #[test]
    fn test_from_centroid() {
        let centroid = vec![2.5, 3.5, 4.5];
        let embeddings = vec![1.0, 2.0, 3.0];
        let from_centroid_embeddings = from_centroid(&centroid, &embeddings);
        assert_eq!(from_centroid_embeddings, vec![-1.5, -1.5, -1.5]);
    }

    #[test]
    fn test_normalise() {
        let embeddings = vec![1.0, 2.0, 3.0];
        let distance = 5.196152;
        let normalised_embeddings = normalise(&embeddings, distance);
        assert_eq!(normalised_embeddings, vec![0.19245009, 0.38490018, 0.5773503]);
    }

    #[test]
    fn test_binary_quantize() {
        let embeddings = vec![1.0, -2.0, 3.0];
        let binary_quantized_embeddings = binary_quantize(&embeddings);
        assert_eq!(binary_quantized_embeddings, vec![1, 0, 1]);
    }

    #[test]
    fn test_normalise_l2_basic() {
        // [3, 4] has L2 norm 5; expected result [0.6, 0.8]
        let mut embedding = vec![3.0_f32, 4.0_f32];
        normalise_l2(&mut embedding);
        assert_relative_eq!(embedding[0], 0.6, epsilon = 1e-6);
        assert_relative_eq!(embedding[1], 0.8, epsilon = 1e-6);
    }

    #[test]
    fn test_normalise_l2_unit_norm_after() {
        let mut embedding: Vec<f32> = (1..=768).map(|i| i as f32).collect();
        normalise_l2(&mut embedding);
        let norm_sq: f32 = embedding.iter().map(|x| x * x).sum();
        assert_relative_eq!(norm_sq, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_normalise_l2_zero_vector_unchanged() {
        let mut embedding = vec![0.0_f32, 0.0_f32, 0.0_f32];
        normalise_l2(&mut embedding);
        assert_eq!(embedding, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalise_l2_already_unit() {
        let mut embedding = vec![0.6_f32, 0.8_f32];
        normalise_l2(&mut embedding);
        assert_relative_eq!(embedding[0], 0.6, epsilon = 1e-6);
        assert_relative_eq!(embedding[1], 0.8, epsilon = 1e-6);
    }
}
