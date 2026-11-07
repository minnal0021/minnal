//! Cross-container set operations.
//!
//! When two different container types need to interact (e.g., Array AND Bitset),
//! we implement the operation here. The strategy is typically to convert to the
//! type that makes the operation cheapest, or iterate over the smaller container.

use super::array::ArrayContainer;
use super::bitset::{AlignedBits, BitsetContainer};
use super::run::{Run, RunContainer};
use crate::field_index::bitmap::simd_support::{bitwise, run_bitset};

/// The values of `values` (at most `len` of them) whose bit in `bitset` equals
/// `keep`. Branchless: whether a value is kept is unpredictable, and a mispredicted
/// branch costs more than always writing it.
fn filter_by_bitset(values: impl Iterator<Item = u16>, len: usize, bitset: &BitsetContainer, keep: bool) -> Vec<u16> {
    let bits = bitset.bits();
    let mut out = vec![0u16; len];
    let mut n = 0;
    for v in values {
        out[n] = v;
        n += (((bits[usize::from(v >> 6)] >> (v & 63)) & 1 == 1) == keep) as usize;
    }
    out.truncate(n);
    out
}

/// From `from`, the first of `runs` that ends at or after `value` (`runs.len()`
/// if none). Walks forward in growing steps, so a merge over a sorted array costs
/// O(array + runs) at worst and stays logarithmic per value for a small array over
/// many runs.
#[inline]
fn seek_run(runs: &[Run], from: usize, value: u16) -> usize {
    match runs.get(from) {
        Some(r) if r.end() < value => {}
        _ => return from,
    }
    let (mut lo, mut step) = (from + 1, 1);
    while lo + step < runs.len() && runs[lo + step - 1].end() < value {
        lo += step;
        step *= 2;
    }
    let hi = (lo + step).min(runs.len());
    lo + runs[lo..hi].partition_point(|r| r.end() < value)
}

// ── Array x Bitset ──────────────────────────────────────────────────

/// AND: Array ∩ Bitset → Array (iterate array, probe bitset)
pub fn array_and_bitset(array: &ArrayContainer, bitset: &BitsetContainer) -> ArrayContainer {
    ArrayContainer::from_sorted(filter_by_bitset(array.iter(), array.cardinality(), bitset, true))
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
    ArrayContainer::from_sorted(filter_by_bitset(array.iter(), array.cardinality(), bitset, false))
}

/// ANDNOT: Bitset \ Array → Bitset (copy bitset, batch-clear array bits)
pub fn bitset_andnot_array(bitset: &BitsetContainer, array: &ArrayContainer) -> BitsetContainer {
    let mut result = bitset.clone();
    run_bitset::clear_bits_from_array(result.bits_mut(), array.values());
    result.recompute_cardinality();
    result
}

// ── Array x Run ─────────────────────────────────────────────────────

/// AND: Array ∩ Run → Array (merge the two sorted lists)
pub fn array_and_run(array: &ArrayContainer, run: &RunContainer) -> ArrayContainer {
    let runs = run.runs();
    let mut values = Vec::with_capacity(array.cardinality());
    let mut at = 0;
    for v in array.iter() {
        at = seek_run(runs, at, v);
        if at == runs.len() {
            break;
        }
        if runs[at].start <= v {
            values.push(v);
        }
    }
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
    let runs = run.runs();
    let mut values = Vec::with_capacity(array.cardinality());
    let mut at = 0;
    for v in array.iter() {
        at = seek_run(runs, at, v);
        if at == runs.len() || v < runs[at].start {
            values.push(v);
        }
    }
    ArrayContainer::from_sorted(values)
}

/// ANDNOT: Run \ Array → expand run, skipping array values (merge the two sorted lists)
pub fn run_andnot_array(run: &RunContainer, array: &ArrayContainer) -> Vec<u16> {
    let exclude = array.values();
    let mut result = Vec::with_capacity(run.cardinality());
    let mut at = 0;
    for v in run.iter() {
        while at < exclude.len() && exclude[at] < v {
            at += 1;
        }
        if at == exclude.len() || exclude[at] != v {
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
    filter_by_bitset(run.iter(), run.cardinality(), bitset, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each operation against the plain per-value definition, over containers of
    /// many shapes, including values and runs at both ends of the u16 range.
    #[test]
    fn merged_and_branchless_operations_match_their_definitions() {
        let mut state = 11u64;
        let mut next = move |n: u64| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let z = (state ^ (state >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            (z ^ (z >> 27)) % n
        };
        for case in 0..400 {
            let n_values = [0, 1, 2, 5, 100, 4000][case % 6];
            let mut values: Vec<u16> = (0..n_values).map(|_| next(65536) as u16).collect();
            if case % 5 == 0 {
                values.extend([0, u16::MAX]);
            }
            values.sort_unstable();
            values.dedup();
            let array = ArrayContainer::from_sorted(values.clone());

            let n_runs = [0, 1, 2, 16, 2000][case % 5];
            let gap = 65536 / (n_runs.max(1) as u64 + 1);
            let mut runs: Vec<Run> = (0..n_runs)
                .map(|i| {
                    let start = i as u64 * gap + next(gap / 4 + 1);
                    Run::new(start as u16, next(gap / 2 + 1) as u16)
                })
                .collect();
            if case % 7 == 0 && n_runs > 1 {
                *runs.first_mut().unwrap() = Run::new(0, 3);
                let last = runs.last().unwrap().start;
                *runs.last_mut().unwrap() = Run::new(last, u16::MAX - last);
            }
            let run = RunContainer::from_runs(runs);
            let in_run: Vec<u16> = run.iter().collect();

            let set: Vec<u16> = (0..=u16::MAX).filter(|_| next(100) < [1, 50, 99][case % 3]).collect();
            let bitset = BitsetContainer::from_values(&set);

            let where_run = |v: &u16, member: bool| in_run.binary_search(v).is_ok() == member;
            let where_bits = |v: &u16, member: bool| set.binary_search(v).is_ok() == member;
            let kept = |pred: &dyn Fn(&u16) -> bool| values.iter().copied().filter(|v| pred(v)).collect::<Vec<u16>>();

            assert_eq!(
                array_and_run(&array, &run).values(),
                kept(&|v| where_run(v, true)),
                "array and run, case {case}"
            );
            assert_eq!(
                array_andnot_run(&array, &run).values(),
                kept(&|v| where_run(v, false)),
                "array andnot run, case {case}"
            );
            assert_eq!(
                array_and_bitset(&array, &bitset).values(),
                kept(&|v| where_bits(v, true)),
                "array and bitset, case {case}"
            );
            assert_eq!(
                array_andnot_bitset(&array, &bitset).values(),
                kept(&|v| where_bits(v, false)),
                "array andnot bitset, case {case}"
            );

            let from_run = |pred: &dyn Fn(&u16) -> bool| in_run.iter().copied().filter(|v| pred(v)).collect::<Vec<u16>>();
            assert_eq!(
                run_andnot_array(&run, &array),
                from_run(&|v| values.binary_search(v).is_err()),
                "run andnot array, case {case}"
            );
            assert_eq!(
                run_andnot_bitset(&run, &bitset),
                from_run(&|v| where_bits(v, false)),
                "run andnot bitset, case {case}"
            );
        }
    }

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
