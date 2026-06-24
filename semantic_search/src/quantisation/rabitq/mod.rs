pub(crate) mod quantisation_support;

use crate::cluster::{Cluster, find_closest_cluster_id};
use log::debug;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use simsimd::SpatialSimilarity;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::index::vector_index::VectorIndex;
use crate::quantisation::rabitq::quantisation_support::{IndexCalculationData, Quantisation, calculate_error_bound};
use crate::vector_math::binary_quantize;

const EPS: f32 = 1e-5;
const NO_ENUMERATIONS: u32 = 10;
const START: [f32; 9] = [0.0, 0.15, 0.20, 0.52, 0.59, 0.71, 0.75, 0.77, 0.81];

struct ExtraBitsQuantisation {
    extra_bits_quantised_embedding: Vec<i32>,
    inverse_of_inner_product: f32,
}

struct TotalQuantisedEmbedding {
    total_quantised_embedding: Vec<u8>,
    inverse_of_inner_product: f32,
}

#[derive(Debug, Clone)]
struct QueueItem {
    value: f32,
    index: usize,
}

impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for QueueItem {}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other.value.partial_cmp(&self.value).unwrap_or(Ordering::Equal)
    }
}

pub fn index_embedding(cluster_map: &HashMap<u32, Cluster>, embeddings: &[f32], style: crate::index::vector_index::QuantisationStyle) -> VectorIndex {
    let closest_cluster_id = find_closest_cluster_id(cluster_map, embeddings);
    let cluster = cluster_map.get(&closest_cluster_id).unwrap();
    index_embedding_to_cluster(embeddings, cluster, style)
}

pub fn index_embedding_to_cluster(embedding: &[f32], cluster: &Cluster, style: crate::index::vector_index::QuantisationStyle) -> VectorIndex {
    use crate::index::vector_index::QuantisationStyle;
    use crate::quantisation::rabitq::quantisation_support::{pack_bits, pack_bytes};
    let n_bits = style.number_of_bits();
    let quantisation = quantise(embedding, &cluster.centroid, n_bits);
    let packed = match &style {
        QuantisationStyle::SingleBit => pack_bits(&quantisation.quantised_embedding),
        QuantisationStyle::MultiBit { .. } => pack_bytes(&quantisation.quantised_embedding),
    };
    VectorIndex::new(
        cluster.cluster_id,
        style,
        quantisation.addition_factor,
        quantisation.scaling_factor,
        quantisation.error_bound,
        packed,
    )
}

#[inline]
fn setup_initial_computation_values(normalised_residual_vector: &[f32], rescale_factor_start: f32) -> (Vec<i32>, f32, f32) {
    let dimension = normalised_residual_vector.len();
    let mut normalised_residual_vector_bar: Vec<i32> = Vec::with_capacity(dimension);
    let mut denominator: f32 = dimension as f32 * 0.25;
    let mut numerator: f32 = 0.0;

    for (index, &value) in normalised_residual_vector.iter().enumerate() {
        let current_rescale_factor = ((rescale_factor_start * value) + EPS) as i32;
        normalised_residual_vector_bar.insert(index, current_rescale_factor);
        denominator += (current_rescale_factor * current_rescale_factor + current_rescale_factor) as f32;
        numerator += (current_rescale_factor as f32 + 0.5) * value;
    }

    (normalised_residual_vector_bar, denominator, numerator)
}

#[inline]
fn best_rescale_factor(normalised_residual_vector: &[f32], number_of_bits_for_quantisation: usize) -> f32 {
    let dimension = normalised_residual_vector.len();

    let max_value = normalised_residual_vector
        .into_par_iter()
        .fold(|| f32::NEG_INFINITY, |acc, &x| acc.max(x))
        .reduce(|| f32::NEG_INFINITY, |a, b| a.max(b));

    let rescale_factor_end = ((1 << number_of_bits_for_quantisation) - 1 + NO_ENUMERATIONS) as f32 / max_value;
    let rescale_factor_start = rescale_factor_end * START[number_of_bits_for_quantisation];

    let (mut normalised_residual_vector_bar, mut sqr_of_denominator, mut numerator) =
        setup_initial_computation_values(normalised_residual_vector, rescale_factor_start);

    let mut rescaling_factor_binary_heap = BinaryHeap::with_capacity(dimension);

    for (index, &value) in normalised_residual_vector.iter().enumerate() {
        let next_value = (normalised_residual_vector_bar[index] + 1) as f32 / value;
        rescaling_factor_binary_heap.push(QueueItem { value: next_value, index });
    }

    let mut max_inner_product = 0.0;
    let mut best_rescaling_factor = 0.0;
    let max_normalised_residual_vector_bar = (1 << number_of_bits_for_quantisation) - 1;

    while let Some(item) = rescaling_factor_binary_heap.pop() {
        let current_t = item.value;
        let update_id = item.index;

        normalised_residual_vector_bar[update_id] += 1;
        let update_normalised_residual_vector_bar = normalised_residual_vector_bar[update_id];

        // Update accumulators
        sqr_of_denominator += 2.0 * update_normalised_residual_vector_bar as f32;
        numerator += normalised_residual_vector[update_id];

        let current_inner_product = numerator / sqr_of_denominator.sqrt();

        if current_inner_product > max_inner_product {
            max_inner_product = current_inner_product;
            best_rescaling_factor = current_t;
        }

        if update_normalised_residual_vector_bar < max_normalised_residual_vector_bar {
            let rescaling_factor_next = (update_normalised_residual_vector_bar + 1) as f32 / normalised_residual_vector[update_id];
            if rescaling_factor_next < rescale_factor_end {
                rescaling_factor_binary_heap.push(QueueItem {
                    value: rescaling_factor_next,
                    index: update_id,
                });
            }
        }
    }

    best_rescaling_factor
}

fn quantise_extra_bits(normalised_residual_vector: &[f32], number_of_bits_for_quantisation: usize) -> ExtraBitsQuantisation {
    let dimension = normalised_residual_vector.len();
    let mut extra_bits_quantised_embedding = Vec::with_capacity(dimension);

    let rescale_factor = best_rescale_factor(normalised_residual_vector, number_of_bits_for_quantisation);
    debug!("Best rescale factor: {:?}", rescale_factor);

    let mut inner_product = 0.0;
    let max_embedding_value = 1 << number_of_bits_for_quantisation;
    let embedding_value_ceiling = max_embedding_value - 1;

    for (index, &embedding) in normalised_residual_vector.iter().enumerate() {
        let mut quantised_embedding_value = ((rescale_factor * embedding) + EPS) as i32;
        if quantised_embedding_value >= max_embedding_value {
            quantised_embedding_value = embedding_value_ceiling;
        }

        extra_bits_quantised_embedding.insert(index, quantised_embedding_value);
        inner_product += (quantised_embedding_value as f32 + 0.5) * embedding;
    }

    let mut inverse_of_inner_product = 1.0 / inner_product;
    inverse_of_inner_product = if inverse_of_inner_product.is_normal() {
        inverse_of_inner_product
    } else {
        1.0
    };

    ExtraBitsQuantisation {
        extra_bits_quantised_embedding,
        inverse_of_inner_product,
    }
}

fn quantise_multi_bits(residual_vector_data_to_centroid: &[f32], number_of_bits_for_quantisation: usize) -> TotalQuantisedEmbedding {
    let extra_number_of_bits = number_of_bits_for_quantisation - 1;

    let dimension = residual_vector_data_to_centroid.len();

    // Calculate L2 norm
    let norm_squared: f32 = residual_vector_data_to_centroid.iter().map(|&x| x * x).sum();
    let norm = norm_squared.sqrt();

    // Create normalized residual
    let normalised_residual: Vec<f32> = residual_vector_data_to_centroid.iter().map(|&x| (x / norm).abs()).collect();

    // Quantize extra bits into i32 code
    let extra_bits_quantised = quantise_extra_bits(&normalised_residual, extra_number_of_bits);

    let mut extra_bits_quantised_embedding = extra_bits_quantised.extra_bits_quantised_embedding;
    let inverse_of_inner_product = extra_bits_quantised.inverse_of_inner_product;

    // Revert embedding for negative dimensions
    let mask = (1_i32 << extra_number_of_bits) - 1;
    for (index, &residual_val) in residual_vector_data_to_centroid.iter().enumerate() {
        if residual_val < 0.0 {
            let complemented = !extra_bits_quantised_embedding[index];
            let tmp_code = complemented & mask;
            extra_bits_quantised_embedding[index] = tmp_code;
        }
    }

    let mut total_quantised_embedding: Vec<u8> = Vec::with_capacity(dimension);
    for index in 0..dimension {
        let one_bit_quantisation = if residual_vector_data_to_centroid[index] >= 0.0 {
            1i32 << extra_number_of_bits
        } else {
            0i32
        };
        let total_quantised_value = extra_bits_quantised_embedding[index] + one_bit_quantisation;
        total_quantised_embedding.insert(index, total_quantised_value as u8);
    }

    TotalQuantisedEmbedding {
        total_quantised_embedding,
        inverse_of_inner_product,
    }
}

fn get_index_calculation_data(
    dimension: usize,
    quantised_embedding: &[u8],
    residual_vector_data_to_centroid: &[f32],
    centroid: &[f32],
    number_of_bits_for_quantisation: usize,
) -> IndexCalculationData {
    let extra_number_of_bits = number_of_bits_for_quantisation - 1;
    // centroid_bias — Centroid Bias / Binary Code Offset
    // Scalar offset that shifts the unsigned binary representation into a signed, centered range.
    // For 1-bit quantization: -(2^1 - 0.5) = -0.5; for multi-bit: -(2^ex_bits - 0.5).
    // Centers the quantization grid around zero.
    let centroid_bias: f32 = -((1i32 << extra_number_of_bits) as f32 - 0.5);

    // shifted_quantization_vector — Shifted Quantization Vector (Centered Code)
    // x_u + centroid_bias: the quantization code shifted by the centroid bias to produce a
    // signed, centered vector. Approximates the direction of the true residual and is used as
    // a proxy for computing inner products. Semantically: the centered quantized residual direction.
    let mut shifted_quantization_vector: Vec<f32> = Vec::with_capacity(dimension);

    //Quantised embedding + centroid_bias
    for (index, &value) in quantised_embedding.iter().enumerate() {
        let quantised_value = value as f32 + centroid_bias;
        shifted_quantization_vector.insert(index, quantised_value);
    }

    // Calculate l2 norm squared and l2 norm
    let l2_sqr_distance_to_centroid = SpatialSimilarity::dot(residual_vector_data_to_centroid, residual_vector_data_to_centroid).unwrap() as f32;
    let l2_norm_distance_to_centroid = l2_sqr_distance_to_centroid.sqrt();

    let ip_residual_shifted_code = SpatialSimilarity::dot(&shifted_quantization_vector, residual_vector_data_to_centroid).unwrap() as f32;
    let mut ip_centroid_shifted_code = SpatialSimilarity::dot(centroid, &shifted_quantization_vector).unwrap() as f32;
    if ip_centroid_shifted_code == 0.0 {
        ip_centroid_shifted_code = f32::INFINITY; // Avoid division by zero
    }

    let dot_product_residual_centroid = SpatialSimilarity::dot(residual_vector_data_to_centroid, centroid).unwrap() as f32;

    IndexCalculationData::new(
        shifted_quantization_vector,
        l2_sqr_distance_to_centroid,
        l2_norm_distance_to_centroid,
        ip_residual_shifted_code,
        ip_centroid_shifted_code,
        dot_product_residual_centroid,
    )
}

fn quantise_using_single_bit(embedding: &[f32], centroid: &[f32]) -> Quantisation {
    let dim = embedding.len();

    let residual: Vec<f32> = embedding.iter().zip(centroid.iter()).map(|(e, c)| e - c).collect();

    // ‖r‖ via SIMD dot product
    let residual_norm: f32 = SpatialSimilarity::dot(&residual, &residual).unwrap().sqrt() as f32;
    let safe_norm = residual_norm.max(1e-10);

    // Normalize to unit vector o = r / ‖r‖
    let residual_unit_vector: Vec<f32> = residual.iter().map(|x| x / safe_norm).collect();

    // 1-bit code: sign of each component
    let binary_quantised_embedding = binary_quantize(&residual_unit_vector);

    // o_bar: quantized approximation in {±1/√D}
    let inv_sqrt_d = 1.0 / (dim as f32).sqrt();
    let o_bar: Vec<f32> = binary_quantised_embedding
        .iter()
        .map(|&b| if b == 1 { inv_sqrt_d } else { -inv_sqrt_d })
        .collect();

    // ⟨o_bar, o⟩ via SIMD
    let ip_obar_o = SpatialSimilarity::dot(&o_bar, &residual_unit_vector).unwrap() as f32;
    let ip_obar_o_safe = if ip_obar_o.abs() < 1e-10 { 1e-10 } else { ip_obar_o };

    // f_rescale = ‖r‖ / (⟨o_bar, o⟩ · √D)
    // est_ip ≈ f_rescale · (2·popcount − D)
    let scaling_factor = residual_norm / (ip_obar_o_safe * (dim as f32).sqrt());

    // f_error = ‖r‖ · √((1 − ⟨o_bar,o⟩²) / ⟨o_bar,o⟩²) · ε₀ / √(D−1)
    const EPSILON_0: f32 = 1.9;
    let obar_o_sq = ip_obar_o * ip_obar_o;
    let error_bound = if dim > 1 {
        let ratio = ((1.0 - obar_o_sq) / obar_o_sq.max(1e-20)).max(0.0).sqrt();
        residual_norm * ratio * EPSILON_0 / ((dim - 1) as f32).sqrt()
    } else {
        0.0
    };

    Quantisation {
        quantised_embedding: binary_quantised_embedding,
        addition_factor: 0.0,
        scaling_factor,
        error_bound,
    }
}

fn quantise_using_multi_bits(embedding: &[f32], centroid: &[f32], number_of_bits_for_quantisation: usize) -> Quantisation {
    assert!(number_of_bits_for_quantisation > 1);

    let dimension = embedding.len();

    let residual_vector_data_to_centroid: Vec<f32> = embedding.iter().zip(centroid.iter()).map(|(e, c)| e - c).collect();
    let total_quantised_embedding = quantise_multi_bits(&residual_vector_data_to_centroid, number_of_bits_for_quantisation);
    let index_calculation_data = get_index_calculation_data(
        dimension,
        &total_quantised_embedding.total_quantised_embedding,
        &residual_vector_data_to_centroid,
        centroid,
        number_of_bits_for_quantisation,
    );

    let addition_factor = 1.0 - index_calculation_data.dot_product_residual_and_centroid
        + index_calculation_data.l2_sqr_distance_from_residual_to_centroid * index_calculation_data.ip_centroid_shifted_code
            / index_calculation_data.ip_residual_shifted_code;
    let scaling_factor = total_quantised_embedding.inverse_of_inner_product * -index_calculation_data.l2_norm_distance_from_residual_to_centroid;

    let error_bound = calculate_error_bound(
        dimension,
        index_calculation_data.l2_sqr_distance_from_residual_to_centroid,
        index_calculation_data.l2_norm_distance_from_residual_to_centroid,
        &index_calculation_data.shifted_quantization_vector,
        index_calculation_data.ip_residual_shifted_code,
    );

    Quantisation {
        quantised_embedding: total_quantised_embedding.total_quantised_embedding,
        addition_factor,
        scaling_factor,
        error_bound,
    }
}

pub fn quantise(embedding: &[f32], centroid: &[f32], number_of_bits_for_quantisation: usize) -> Quantisation {
    if number_of_bits_for_quantisation == 1 {
        quantise_using_single_bit(embedding, centroid)
    } else {
        quantise_using_multi_bits(embedding, centroid, number_of_bits_for_quantisation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_math::{pack, pack_to_signed_64, pack_to_signed_128};
    use approx::assert_relative_eq;

    /// Parse a JSON array of `f32` from a `test_data` fixture file.
    fn load_vec_f32(json: &str) -> Vec<f32> {
        serde_json::from_str(json).expect("fixture must be a JSON array of f32")
    }

    /// Parse a JSON array of `u8` from a `test_data` fixture file.
    fn load_vec_u8(json: &str) -> Vec<u8> {
        serde_json::from_str(json).expect("fixture must be a JSON array of u8")
    }

    /// Parse a JSON array of `i64` from a `test_data` fixture file.
    fn load_vec_i64(json: &str) -> Vec<i64> {
        serde_json::from_str(json).expect("fixture must be a JSON array of i64")
    }

    /// Parse a JSON array of decimal `u128` strings from a `test_data` fixture file.
    ///
    /// 128-bit integers are stored as strings because `serde_json` cannot parse
    /// numeric literals wider than 64 bits.
    fn load_vec_u128(json: &str) -> Vec<u128> {
        serde_json::from_str::<Vec<String>>(json)
            .expect("fixture must be a JSON array of u128 strings")
            .iter()
            .map(|s| s.parse().expect("fixture element must be a valid u128"))
            .collect()
    }

    /// Parse a JSON array of decimal `i128` strings from a `test_data` fixture file.
    ///
    /// 128-bit integers are stored as strings because `serde_json` cannot parse
    /// numeric literals wider than 64 bits.
    fn load_vec_i128(json: &str) -> Vec<i128> {
        serde_json::from_str::<Vec<String>>(json)
            .expect("fixture must be a JSON array of i128 strings")
            .iter()
            .map(|s| s.parse().expect("fixture element must be a valid i128"))
            .collect()
    }

    #[test]
    fn test_find_best_rescale_factor() {
        let embedding: Vec<f32> = vec![0.56333, 0.83332, 0.54333, -0.54333, 0.12345, -0.67890, 0.98765, -0.43210];
        let centroid: Vec<f32> = vec![-0.49333, 1.22333, 0.54333, -0.12345, 0.67890, -0.98765, 0.43210, -0.54321];
        let residual_vector_data_to_centroid: Vec<f32> = embedding.iter().zip(centroid.iter()).map(|(e, c)| e - c).collect();
        let number_of_bits = 6;

        let best_rescale_factor = best_rescale_factor(&residual_vector_data_to_centroid, number_of_bits);
        assert_relative_eq!(best_rescale_factor, 47.632656, epsilon = 0.0001);
    }

    #[test]
    fn test_quantise_extra_bits() {
        let embedding: Vec<f32> = vec![0.56333, 0.83332, 0.54333, -0.54333, 0.12345, -0.67890, 0.98765, -0.43210];
        let centroid: Vec<f32> = vec![-0.49333, 1.22333, 0.54333, -0.12345, 0.67890, -0.98765, 0.43210, -0.54321];
        let residual_vector_data_to_centroid: Vec<f32> = embedding.iter().zip(centroid.iter()).map(|(e, c)| e - c).collect();
        let number_of_bits = 6;

        let extra_bits_quantisation = quantise_extra_bits(&residual_vector_data_to_centroid, number_of_bits);
        assert_relative_eq!(extra_bits_quantisation.inverse_of_inner_product, 0.009810816, epsilon = 0.0001);
        assert_eq!(extra_bits_quantisation.extra_bits_quantised_embedding, [50, -18, 0, -19, -26, 14, 26, 5]);
    }

    #[test]
    fn test_quantise_multi_bits() {
        let embedding: Vec<f32> = vec![0.56333, 0.83332, 0.54333, -0.54333, 0.12345, -0.67890, 0.98765, -0.43210];
        let centroid: Vec<f32> = vec![-0.49333, 1.22333, 0.54333, -0.12345, 0.67890, -0.98765, 0.43210, -0.54321];
        let residual_vector_data_to_centroid: Vec<f32> = embedding.iter().zip(centroid.iter()).map(|(e, c)| e - c).collect();
        let number_of_bits = 7;

        let total_quantised_embedding = quantise_multi_bits(&residual_vector_data_to_centroid, number_of_bits);
        assert_relative_eq!(total_quantised_embedding.inverse_of_inner_product, 0.011276863, epsilon = 0.0001);
        assert_eq!(total_quantised_embedding.total_quantised_embedding, [127, 40, 64, 38, 30, 82, 97, 70]);
    }

    #[test]
    fn test_quantise_using_multi_bits() {
        let embedding: Vec<f32> = vec![0.56333, 0.83332, 0.54333, -0.54333, 0.12345, -0.67890, 0.98765, -0.43210];
        let centroid: Vec<f32> = vec![-0.49333, 1.22333, 0.54333, -0.12345, 0.67890, -0.98765, 0.43210, -0.54321];
        let number_of_bits = 7;

        let quantisation = quantise(&embedding, &centroid, number_of_bits);
        assert_relative_eq!(quantisation.addition_factor, 1.0083025, epsilon = 0.0001);
        assert_relative_eq!(quantisation.scaling_factor, -0.016610974, epsilon = 0.0001);
        assert_relative_eq!(quantisation.error_bound, 0.007194197, epsilon = 0.0001);
        assert_eq!(quantisation.quantised_embedding, [127, 40, 64, 38, 30, 82, 97, 70]);
    }

    #[test]
    fn test_quantise_using_single_bits() {
        let embedding: Vec<f32> = vec![0.56333, 0.83332, 0.54333, -0.54333, 0.12345, -0.67890, 0.98765, -0.43210];
        let centroid: Vec<f32> = vec![-0.49333, 1.22333, 0.54333, -0.12345, 0.67890, -0.98765, 0.43210, -0.54321];
        let number_of_bits = 1;

        let quantisation = quantise(&embedding, &centroid, number_of_bits);
        assert_relative_eq!(quantisation.addition_factor, 0.0, epsilon = 0.0001);
        assert_relative_eq!(quantisation.scaling_factor, 0.63865, epsilon = 0.001);
        assert_relative_eq!(quantisation.error_bound, 0.75050, epsilon = 0.001);
        assert_eq!(quantisation.quantised_embedding, [1, 0, 1, 0, 0, 1, 1, 1]);
    }

    fn real_centroid() -> Vec<f32> {
        load_vec_f32(include_str!("../../../test_data/rabitq_real_centroid.json"))
    }

    fn real_embedding() -> Vec<f32> {
        load_vec_f32(include_str!("../../../test_data/real_embedding.json"))
    }

    #[test]
    fn test_quantisation_using_real_data() {
        let centroid = real_centroid();
        let embedding = real_embedding();

        assert_multi_bit_quantisation(&centroid, &embedding);

        assert_single_bit_quantisation(&centroid, &embedding)
    }

    fn assert_single_bit_quantisation(centroid: &[f32], embedding: &[f32]) {
        let number_of_bits: usize = 1;

        let quantisation = quantise(embedding, centroid, number_of_bits);
        let packed_quantisation = pack(&quantisation.quantised_embedding).expect("Failed to pack quantisation");
        let packed_to_signed_quantisation_i128 = pack_to_signed_128(&quantisation.quantised_embedding).expect("Failed to pack signed quantisation");
        let packed_to_signed_quantisation_i64 = pack_to_signed_64(&quantisation.quantised_embedding).expect("Failed to pack signed quantisation");
        assert_relative_eq!(quantisation.addition_factor, 0.0, epsilon = 0.0001);
        assert_relative_eq!(quantisation.scaling_factor, 0.054529752, epsilon = 0.001);
        assert_relative_eq!(quantisation.error_bound, 0.06784632, epsilon = 0.001);
        assert_eq!(
            quantisation.quantised_embedding,
            load_vec_u8(include_str!("../../../test_data/rabitq_singlebit_codes.json"))
        );
        assert_eq!(
            packed_quantisation,
            load_vec_u128(include_str!("../../../test_data/rabitq_singlebit_packed_u128.json"))
        );
        assert_eq!(
            packed_to_signed_quantisation_i128,
            load_vec_i128(include_str!("../../../test_data/rabitq_singlebit_signed_i128.json"))
        );
        assert_eq!(
            packed_to_signed_quantisation_i64,
            load_vec_i64(include_str!("../../../test_data/rabitq_singlebit_signed_i64.json"))
        );
    }

    fn assert_multi_bit_quantisation(centroid: &[f32], embedding: &[f32]) {
        let number_of_bits: usize = 8;

        let quantisation = quantise(embedding, centroid, number_of_bits);
        let packed_quantisation = pack(&quantisation.quantised_embedding).expect("Failed to pack quantisation");
        let packed_to_signed_quantisation_i128 = pack_to_signed_128(&quantisation.quantised_embedding).expect("Failed to pack signed quantisation");
        let packed_to_signed_quantisation_i64 = pack_to_signed_64(&quantisation.quantised_embedding).expect("Failed to pack signed quantisation");
        assert_relative_eq!(quantisation.addition_factor, 0.9996985, epsilon = 0.0001);
        assert_relative_eq!(quantisation.scaling_factor, -0.0012493228, epsilon = 0.0001);
        assert_relative_eq!(quantisation.error_bound, 0.0006662832, epsilon = 0.0001);
        assert_eq!(
            quantisation.quantised_embedding,
            load_vec_u8(include_str!("../../../test_data/rabitq_multibit_codes.json"))
        );
        assert_eq!(
            packed_quantisation,
            load_vec_u128(include_str!("../../../test_data/rabitq_multibit_packed_u128.json"))
        );
        assert_eq!(
            packed_to_signed_quantisation_i128,
            load_vec_i128(include_str!("../../../test_data/rabitq_multibit_signed_i128.json"))
        );
        assert_eq!(
            packed_to_signed_quantisation_i64,
            load_vec_i64(include_str!("../../../test_data/rabitq_multibit_signed_i64.json"))
        );
    }

    #[test]
    fn test_index_embedding_to_cluster_returns_correct_cluster_id() {
        use crate::index::vector_index::QuantisationStyle;
        let centroid: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let embedding: Vec<f32> = (0..16).map(|i| i as f32 * 0.1 + 0.05).collect();
        let cluster = Cluster::new(7, centroid);
        let result = index_embedding_to_cluster(&embedding, &cluster, QuantisationStyle::MultiBit { number_of_bits: 4 });
        assert_eq!(result.cluster_id, 7);
    }

    #[test]
    fn test_index_embedding_to_cluster_produces_non_empty_quantised_vector() {
        use crate::index::vector_index::QuantisationStyle;
        let centroid: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let embedding: Vec<f32> = (0..16).map(|i| i as f32 * 0.1 + 0.05).collect();
        let cluster = Cluster::new(1, centroid);
        let result = index_embedding_to_cluster(&embedding, &cluster, QuantisationStyle::MultiBit { number_of_bits: 4 });
        assert!(!result.packed_vector.is_empty());
    }

    #[test]
    fn test_index_picks_closest_cluster() {
        use crate::index::vector_index::QuantisationStyle;
        let cluster1 = Cluster::new(1, vec![0.0f32; 16]);
        let cluster2 = Cluster::new(2, vec![1.0f32; 16]);
        let mut cluster_map = HashMap::new();
        cluster_map.insert(1, cluster1);
        cluster_map.insert(2, cluster2);

        let embedding = vec![0.9f32; 16];
        let result = index_embedding(&cluster_map, &embedding, QuantisationStyle::MultiBit { number_of_bits: 4 });
        assert_eq!(result.cluster_id, 2);
    }
}
