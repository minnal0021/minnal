pub mod array;
pub mod bitset;
pub mod ops;
pub mod run;

use array::ARRAY_TO_BITSET_THRESHOLD;
use array::ArrayContainer;
use bitset::BITSET_WORDS;
use bitset::BitsetContainer;
use run::Run;
use run::RunContainer;

use rkyv::Archive;

/// Enum dispatching to the three container types.
#[derive(Debug, Clone, PartialEq, Eq, Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum Container {
    Array(ArrayContainer),
    Bitset(BitsetContainer),
    Run(RunContainer),
}

impl Container {
    pub fn new_array() -> Self {
        Container::Array(ArrayContainer::new())
    }

    pub fn insert(&mut self, value: u16) -> bool {
        match self {
            Container::Array(a) => {
                let inserted = a.insert(value);
                if inserted && a.should_promote() {
                    *self = Container::Bitset(BitsetContainer::from_values(a.values()));
                }
                inserted
            }
            Container::Bitset(b) => b.insert(value),
            Container::Run(r) => r.insert(value),
        }
    }

    pub fn remove(&mut self, value: u16) -> bool {
        match self {
            Container::Array(a) => a.remove(value),
            Container::Bitset(b) => {
                let removed = b.remove(value);
                if removed && b.should_demote() {
                    *self = Container::Array(ArrayContainer::from_sorted(b.to_values()));
                }
                removed
            }
            Container::Run(r) => r.remove(value),
        }
    }

    pub fn contains(&self, value: u16) -> bool {
        match self {
            Container::Array(a) => a.contains(value),
            Container::Bitset(b) => b.contains(value),
            Container::Run(r) => r.contains(value),
        }
    }

    pub fn cardinality(&self) -> usize {
        match self {
            Container::Array(a) => a.cardinality(),
            Container::Bitset(b) => b.cardinality(),
            Container::Run(r) => r.cardinality(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.cardinality() == 0
    }

    pub fn min(&self) -> Option<u16> {
        match self {
            Container::Array(a) => a.min(),
            Container::Bitset(b) => b.min(),
            Container::Run(r) => r.min(),
        }
    }

    pub fn max(&self) -> Option<u16> {
        match self {
            Container::Array(a) => a.max(),
            Container::Bitset(b) => b.max(),
            Container::Run(r) => r.max(),
        }
    }

    /// Re-evaluate which container type is best for the current data.
    pub fn optimize(&mut self) {
        match self {
            Container::Array(a) => {
                // Check if run encoding would be better
                let run = RunContainer::from_sorted_values(a.values());
                if run.is_efficient() {
                    *self = Container::Run(run);
                } else if a.should_promote() {
                    *self = Container::Bitset(BitsetContainer::from_values(a.values()));
                }
            }
            Container::Bitset(b) => {
                if b.should_demote() {
                    let values = b.to_values();
                    let run = RunContainer::from_sorted_values(&values);
                    if run.is_efficient() {
                        *self = Container::Run(run);
                    } else {
                        *self = Container::Array(ArrayContainer::from_sorted(values));
                    }
                } else {
                    // Check if run encoding is better even at high cardinality
                    let values = b.to_values();
                    let run = RunContainer::from_sorted_values(&values);
                    // Run is better if it uses less than 8KB (bitset size)
                    if run.num_runs() * 4 < 8192 {
                        *self = Container::Run(run);
                    }
                }
            }
            Container::Run(r) => {
                if !r.is_efficient() {
                    let values = r.to_values();
                    if values.len() >= array::ARRAY_TO_BITSET_THRESHOLD {
                        *self = Container::Bitset(BitsetContainer::from_values(&values));
                    } else {
                        *self = Container::Array(ArrayContainer::from_sorted(values));
                    }
                }
            }
        }
    }

    /// Collect values into a Vec<u16>.
    pub fn to_values(&self) -> Vec<u16> {
        match self {
            Container::Array(a) => a.values().to_vec(),
            Container::Bitset(b) => b.to_values(),
            Container::Run(r) => r.to_values(),
        }
    }

    /// Count of elements ≤ `value` in this container.
    pub fn rank(&self, value: u16) -> usize {
        match self {
            Container::Array(a) => a.rank(value),
            Container::Bitset(b) => b.rank(value),
            Container::Run(r) => r.rank(value),
        }
    }

    /// The `rank`-th element (0-indexed) in sorted order, or `None` if out of bounds.
    pub fn select(&self, rank: usize) -> Option<u16> {
        match self {
            Container::Array(a) => a.select(rank),
            Container::Bitset(b) => b.select(rank),
            Container::Run(r) => r.select(rank),
        }
    }

    /// Complement all bits in the inclusive range [`lo`, `hi_inclusive`] in place.
    ///
    /// Non-Bitset containers are promoted to a `BitsetContainer` for the flip, then
    /// demoted back to `ArrayContainer` if the result has fewer than
    /// [`ARRAY_TO_BITSET_THRESHOLD`] elements.
    pub fn flip_range(&mut self, lo: u16, hi_inclusive: u16) {
        // Take ownership of the BitsetContainer (or build one from current values),
        // flip in place, then assign back the best container type.
        let mut b = match self {
            Container::Bitset(b) => std::mem::take(b),
            _ => BitsetContainer::from_sorted_values(&self.to_values()),
        };
        b.flip_range(lo, hi_inclusive);
        let card = b.cardinality();
        *self = if card == 0 {
            Container::new_array()
        } else if card < ARRAY_TO_BITSET_THRESHOLD {
            Container::Array(ArrayContainer::from_sorted(b.to_values()))
        } else {
            Container::Bitset(b)
        };
    }

    /// Retain only values in [`lo`, `hi_inclusive`], discarding all others.
    ///
    /// Returns an empty `Array` container when no values survive.
    /// Used internally by [`crate::bitmap::RoaringBitmap::range_and`] and
    /// [`crate::bitmap::RoaringBitmap::range_or`].
    pub(crate) fn clip_to_range(self, lo: u16, hi_inclusive: u16) -> Self {
        if lo == 0 && hi_inclusive == u16::MAX {
            return self; // no clipping needed
        }
        match self {
            Container::Array(a) => {
                let vals = a.into_values();
                let lo_pos = vals.partition_point(|&v| v < lo);
                let hi_pos = vals.partition_point(|&v| v <= hi_inclusive);
                Container::Array(ArrayContainer::from_sorted(vals[lo_pos..hi_pos].to_vec()))
            }
            Container::Bitset(mut b) => {
                // Zero words before lo
                if lo > 0 {
                    let lo_word = (lo >> 6) as usize;
                    let lo_bit = lo & 63;
                    for w in 0..lo_word {
                        b.bits_mut()[w] = 0;
                    }
                    b.bits_mut()[lo_word] &= u64::MAX << lo_bit;
                }
                // Zero words after hi_inclusive
                let hi_word = (hi_inclusive >> 6) as usize;
                let hi_bit = hi_inclusive & 63;
                for w in (hi_word + 1)..BITSET_WORDS {
                    b.bits_mut()[w] = 0;
                }
                if hi_bit < 63 {
                    b.bits_mut()[hi_word] &= (1u64 << (hi_bit + 1)) - 1;
                }
                b.recompute_cardinality();
                let card = b.cardinality();
                if card == 0 {
                    Container::new_array()
                } else if b.should_demote() {
                    Container::Array(ArrayContainer::from_sorted(b.to_values()))
                } else {
                    Container::Bitset(b)
                }
            }
            Container::Run(r) => {
                let clipped: Vec<Run> = r
                    .runs()
                    .iter()
                    .filter_map(|run| {
                        let start = run.start.max(lo);
                        let end = run.end().min(hi_inclusive);
                        if start <= end { Some(Run::new(start, end - start)) } else { None }
                    })
                    .collect();
                if clipped.is_empty() {
                    return Container::new_array();
                }
                Container::Run(RunContainer::from_runs(clipped))
            }
        }
    }

    pub fn iter(&self) -> Box<dyn Iterator<Item = u16> + '_> {
        match self {
            Container::Array(a) => Box::new(a.iter()),
            Container::Bitset(b) => Box::new(b.iter()),
            Container::Run(r) => Box::new(r.iter()),
        }
    }

    // ── Set operations (all 9 combinations) ─────────────────────────

    pub fn and(&self, other: &Container) -> Container {
        match (self, other) {
            (Container::Array(a), Container::Array(b)) => Container::Array(a.and(b)),
            (Container::Bitset(a), Container::Bitset(b)) => {
                let r = a.and(b);
                if r.should_demote() {
                    Container::Array(ArrayContainer::from_sorted(r.to_values()))
                } else {
                    Container::Bitset(r)
                }
            }
            (Container::Run(a), Container::Run(b)) => Container::Run(a.and(b)),
            (Container::Array(a), Container::Bitset(b)) => Container::Array(ops::array_and_bitset(a, b)),
            (Container::Bitset(b), Container::Array(a)) => Container::Array(ops::array_and_bitset(a, b)),
            (Container::Array(a), Container::Run(r)) => Container::Array(ops::array_and_run(a, r)),
            (Container::Run(r), Container::Array(a)) => Container::Array(ops::array_and_run(a, r)),
            (Container::Bitset(b), Container::Run(r)) => {
                let result = ops::bitset_and_run(b, r);
                if result.should_demote() {
                    Container::Array(ArrayContainer::from_sorted(result.to_values()))
                } else {
                    Container::Bitset(result)
                }
            }
            (Container::Run(r), Container::Bitset(b)) => {
                let result = ops::bitset_and_run(b, r);
                if result.should_demote() {
                    Container::Array(ArrayContainer::from_sorted(result.to_values()))
                } else {
                    Container::Bitset(result)
                }
            }
        }
    }

    pub fn or(&self, other: &Container) -> Container {
        match (self, other) {
            (Container::Array(a), Container::Array(b)) => {
                let result = a.or(b);
                if result.should_promote() {
                    Container::Bitset(BitsetContainer::from_values(result.values()))
                } else {
                    Container::Array(result)
                }
            }
            (Container::Bitset(a), Container::Bitset(b)) => Container::Bitset(a.or(b)),
            (Container::Run(a), Container::Run(b)) => Container::Run(a.or(b)),
            (Container::Array(a), Container::Bitset(b)) | (Container::Bitset(b), Container::Array(a)) => {
                Container::Bitset(ops::array_or_bitset(a, b))
            }
            (Container::Array(a), Container::Run(r)) | (Container::Run(r), Container::Array(a)) => {
                let values = ops::array_or_run_values(a, r);
                if values.len() >= array::ARRAY_TO_BITSET_THRESHOLD {
                    Container::Bitset(BitsetContainer::from_values(&values))
                } else {
                    Container::Array(ArrayContainer::from_sorted(values))
                }
            }
            (Container::Bitset(b), Container::Run(r)) | (Container::Run(r), Container::Bitset(b)) => Container::Bitset(ops::bitset_or_run(b, r)),
        }
    }

    pub fn and_not(&self, other: &Container) -> Container {
        match (self, other) {
            (Container::Array(a), Container::Array(b)) => Container::Array(a.and_not(b)),
            (Container::Bitset(a), Container::Bitset(b)) => {
                let r = a.and_not(b);
                if r.should_demote() {
                    Container::Array(ArrayContainer::from_sorted(r.to_values()))
                } else {
                    Container::Bitset(r)
                }
            }
            (Container::Run(a), Container::Run(b)) => {
                let r = a.and_not(b);
                if !r.is_efficient() {
                    let values = r.to_values();
                    if values.len() >= array::ARRAY_TO_BITSET_THRESHOLD {
                        Container::Bitset(BitsetContainer::from_values(&values))
                    } else {
                        Container::Array(ArrayContainer::from_sorted(values))
                    }
                } else {
                    Container::Run(r)
                }
            }
            (Container::Array(a), Container::Bitset(b)) => Container::Array(ops::array_andnot_bitset(a, b)),
            (Container::Bitset(b), Container::Array(a)) => {
                let r = ops::bitset_andnot_array(b, a);
                if r.should_demote() {
                    Container::Array(ArrayContainer::from_sorted(r.to_values()))
                } else {
                    Container::Bitset(r)
                }
            }
            (Container::Array(a), Container::Run(r)) => Container::Array(ops::array_andnot_run(a, r)),
            (Container::Run(r), Container::Array(a)) => {
                let values = ops::run_andnot_array(r, a);
                if values.len() >= array::ARRAY_TO_BITSET_THRESHOLD {
                    Container::Bitset(BitsetContainer::from_values(&values))
                } else {
                    Container::Array(ArrayContainer::from_sorted(values))
                }
            }
            (Container::Bitset(b), Container::Run(r)) => {
                let result = ops::bitset_andnot_run(b, r);
                if result.should_demote() {
                    Container::Array(ArrayContainer::from_sorted(result.to_values()))
                } else {
                    Container::Bitset(result)
                }
            }
            (Container::Run(r), Container::Bitset(b)) => {
                let values = ops::run_andnot_bitset(r, b);
                if values.len() >= array::ARRAY_TO_BITSET_THRESHOLD {
                    Container::Bitset(BitsetContainer::from_values(&values))
                } else {
                    Container::Array(ArrayContainer::from_sorted(values))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_promote_array_to_bitset() {
        let mut c = Container::new_array();
        for i in 0..4096u16 {
            c.insert(i);
        }
        assert!(matches!(c, Container::Bitset(_)));
    }

    #[test]
    fn auto_demote_bitset_to_array() {
        let mut c = Container::Bitset(BitsetContainer::from_values(&(0..4096u16).collect::<Vec<_>>()));
        // Remove enough to drop below threshold
        for i in 0..100u16 {
            c.remove(i);
        }
        assert!(matches!(c, Container::Array(_)));
    }

    #[test]
    fn cross_container_and() {
        let a = Container::Array(ArrayContainer::from_sorted(vec![1, 5, 10, 100]));
        let b = Container::Bitset(BitsetContainer::from_values(&[5, 10, 50, 100, 200]));
        let result = a.and(&b);
        assert_eq!(result.to_values(), vec![5, 10, 100]);
    }

    #[test]
    fn optimize_to_run() {
        // Consecutive values should become a RunContainer
        let mut c = Container::Array(ArrayContainer::from_sorted((0..100).collect()));
        c.optimize();
        assert!(matches!(c, Container::Run(_)));
    }
}
