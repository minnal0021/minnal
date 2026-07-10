//! Cross-container set operations.
//!
//! When two different container types need to interact (e.g., Array AND Bitset),
//! we implement the operation here. The strategy is typically to convert to the
//! type that makes the operation cheapest, or iterate over the smaller container.

use super::array::ArrayContainer;
use super::bitset::{AlignedBits, BitsetContainer};
use super::run::RunContainer;
use crate::index::simd_support::{bitwise, run_bitset};

// ── Array x Bitset ──────────────────────────────────────────────────

/// AND: Array ∩ Bitset → Array (iterate array, probe bitset)
pub fn array_and_bitset(array: &ArrayContainer, bitset: &BitsetContainer) -> ArrayContainer {
    let values: Vec<u16> = array.iter().filter(|&v| bitset.contains(v)).collect();
    ArrayContainer::from_sorted(values)
}

/// OR: Array ∪ Bitset → Bitset (copy bitset, batch-set array bits)
pub fn array_or_bitset(array: &ArrayContainer, bitset: &BitsetContainer) -> BitsetContainer {
    let mut result = bitset.clone();
    run_bitset::set_bits_from_array(result.bits_mut(), array.values());
    result.recompute_cardinality();
    result
}

/// ANDNOT: Array \ Bitset → Array (iterate array, exclude if in bitset)
pub fn array_andnot_bitset(array: &ArrayContainer, bitset: &BitsetContainer) -> ArrayContainer {
    let values: Vec<u16> = array.iter().filter(|&v| !bitset.contains(v)).collect();
    ArrayContainer::from_sorted(values)
}

/// ANDNOT: Bitset \ Array → Bitset (copy bitset, batch-clear array bits)
pub fn bitset_andnot_array(bitset: &BitsetContainer, array: &ArrayContainer) -> BitsetContainer {
    let mut result = bitset.clone();
    run_bitset::clear_bits_from_array(result.bits_mut(), array.values());
    result.recompute_cardinality();
    result
}

// ── Array x Run ─────────────────────────────────────────────────────

/// AND: Array ∩ Run → Array (iterate array, probe run)
pub fn array_and_run(array: &ArrayContainer, run: &RunContainer) -> ArrayContainer {
    let values: Vec<u16> = array.iter().filter(|&v| run.contains(v)).collect();
    ArrayContainer::from_sorted(values)
}

/// OR: Array ∪ Run → merge values, build best container.
/// Returns as a sorted `Vec<u16>` — caller wraps into the right container type.
pub fn array_or_run_values(array: &ArrayContainer, run: &RunContainer) -> Vec<u16> {
    let mut result = Vec::with_capacity(array.cardinality() + run.cardinality());
    let mut ai = array.iter().peekable();
    let mut ri = run.iter().peekable();
    loop {
        match (ai.peek(), ri.peek()) {
            (Some(&a), Some(&b)) => match a.cmp(&b) {
                std::cmp::Ordering::Less => {
                    result.push(a);
                    ai.next();
                }
                std::cmp::Ordering::Greater => {
                    result.push(b);
                    ri.next();
                }
                std::cmp::Ordering::Equal => {
                    result.push(a);
                    ai.next();
                    ri.next();
                }
            },
            (Some(&a), None) => {
                result.push(a);
                ai.next();
            }
            (None, Some(&b)) => {
                result.push(b);
                ri.next();
            }
            (None, None) => break,
        }
    }
    result
}

/// ANDNOT: Array \ Run → Array
pub fn array_andnot_run(array: &ArrayContainer, run: &RunContainer) -> ArrayContainer {
    let values: Vec<u16> = array.iter().filter(|&v| !run.contains(v)).collect();
    ArrayContainer::from_sorted(values)
}

/// ANDNOT: Run \ Array → expand run, filter out array values
pub fn run_andnot_array(run: &RunContainer, array: &ArrayContainer) -> Vec<u16> {
    let mut result = Vec::with_capacity(run.cardinality());
    for v in run.iter() {
        if !array.contains(v) {
            result.push(v);
        }
    }
    result
}

// ── Bitset x Run ────────────────────────────────────────────────────

/// AND: Bitset ∩ Run → build run bitmask, AND with bitset via SIMD
pub fn bitset_and_run(bitset: &BitsetContainer, run: &RunContainer) -> BitsetContainer {
    let run_bits = run_bitset::runs_to_bitmask(run.runs());
    let r = bitwise::and(bitset.bits(), &run_bits);
    BitsetContainer::from_raw_bits(AlignedBits(*r.bits), r.cardinality)
}

/// OR: Bitset ∪ Run → build run bitmask, OR with bitset via SIMD
pub fn bitset_or_run(bitset: &BitsetContainer, run: &RunContainer) -> BitsetContainer {
    let run_bits = run_bitset::runs_to_bitmask(run.runs());
    let r = bitwise::or(bitset.bits(), &run_bits);
    BitsetContainer::from_raw_bits(AlignedBits(*r.bits), r.cardinality)
}

/// ANDNOT: Bitset \ Run → build run bitmask, ANDNOT with bitset via SIMD
pub fn bitset_andnot_run(bitset: &BitsetContainer, run: &RunContainer) -> BitsetContainer {
    let run_bits = run_bitset::runs_to_bitmask(run.runs());
    let r = bitwise::and_not(bitset.bits(), &run_bits);
    BitsetContainer::from_raw_bits(AlignedBits(*r.bits), r.cardinality)
}

/// ANDNOT: Run \ Bitset → expand run, exclude bitset bits
pub fn run_andnot_bitset(run: &RunContainer, bitset: &BitsetContainer) -> Vec<u16> {
    let mut result = Vec::with_capacity(run.cardinality());
    for v in run.iter() {
        if !bitset.contains(v) {
            result.push(v);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_bitset_and() {
        let a = ArrayContainer::from_sorted(vec![1, 5, 10, 100]);
        let b = BitsetContainer::from_values(&[5, 10, 50, 100, 200]);
        let r = array_and_bitset(&a, &b);
        assert_eq!(r.values(), &[5, 10, 100]);
    }

    #[test]
    fn array_bitset_or() {
        let a = ArrayContainer::from_sorted(vec![1, 5]);
        let b = BitsetContainer::from_values(&[5, 10]);
        let r = array_or_bitset(&a, &b);
        assert!(r.contains(1));
        assert!(r.contains(5));
        assert!(r.contains(10));
        assert_eq!(r.cardinality(), 3);
    }

    #[test]
    fn array_run_and() {
        let a = ArrayContainer::from_sorted(vec![1, 5, 10, 15]);
        let r = RunContainer::from_sorted_values(&[3, 4, 5, 6, 7, 8, 9, 10, 11]);
        let result = array_and_run(&a, &r);
        assert_eq!(result.values(), &[5, 10]);
    }

    #[test]
    fn bitset_run_and() {
        let b = BitsetContainer::from_values(&[1, 5, 10, 100]);
        let r = RunContainer::from_sorted_values(&[5, 6, 7, 8, 9, 10]);
        let result = bitset_and_run(&b, &r);
        assert_eq!(result.to_values(), vec![5, 10]);
    }

    #[test]
    fn array_bitset_andnot() {
        let a = ArrayContainer::from_sorted(vec![1, 5, 10, 100]);
        let b = BitsetContainer::from_values(&[5, 10, 50]);
        let r = array_andnot_bitset(&a, &b);
        assert_eq!(r.values(), &[1, 100]);
    }

    #[test]
    fn bitset_array_andnot() {
        let b = BitsetContainer::from_values(&[5, 10, 50, 100]);
        let a = ArrayContainer::from_sorted(vec![10, 100]);
        let r = bitset_andnot_array(&b, &a);
        assert_eq!(r.to_values(), vec![5, 50]);
        assert_eq!(r.cardinality(), 2);
    }

    #[test]
    fn array_run_or() {
        let a = ArrayContainer::from_sorted(vec![1, 5, 10]);
        let run = RunContainer::from_sorted_values(&[4, 5, 6, 7]);
        let values = array_or_run_values(&a, &run);
        assert_eq!(values, vec![1, 4, 5, 6, 7, 10]);
    }

    #[test]
    fn array_run_andnot() {
        let a = ArrayContainer::from_sorted(vec![1, 5, 10, 15]);
        let run = RunContainer::from_sorted_values(&[4, 5, 6, 7, 8, 9, 10, 11]);
        let r = array_andnot_run(&a, &run);
        assert_eq!(r.values(), &[1, 15]);
    }

    #[test]
    fn run_array_andnot() {
        let run = RunContainer::from_sorted_values(&[3, 4, 5, 6, 7]);
        let a = ArrayContainer::from_sorted(vec![4, 6]);
        let values = run_andnot_array(&run, &a);
        assert_eq!(values, vec![3, 5, 7]);
    }

    #[test]
    fn bitset_run_or() {
        let b = BitsetContainer::from_values(&[1, 100]);
        let run = RunContainer::from_sorted_values(&[5, 6, 7]);
        let r = bitset_or_run(&b, &run);
        assert_eq!(r.to_values(), vec![1, 5, 6, 7, 100]);
        assert_eq!(r.cardinality(), 5);
    }

    #[test]
    fn bitset_run_andnot() {
        let b = BitsetContainer::from_values(&[5, 6, 7, 8, 100]);
        let run = RunContainer::from_sorted_values(&[6, 7]);
        let r = bitset_andnot_run(&b, &run);
        assert_eq!(r.to_values(), vec![5, 8, 100]);
        assert_eq!(r.cardinality(), 3);
    }

    #[test]
    fn run_bitset_andnot() {
        let run = RunContainer::from_sorted_values(&[10, 11, 12, 13]);
        let b = BitsetContainer::from_values(&[11, 13]);
        let values = run_andnot_bitset(&run, &b);
        assert_eq!(values, vec![10, 12]);
    }
}
