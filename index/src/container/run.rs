use rkyv::Archive;

/// A single run: represents all values in [start, start + length] inclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct Run {
    pub start: u16,
    pub length: u16, // number of values beyond start, so the run has length+1 elements
}

impl Run {
    pub fn new(start: u16, length: u16) -> Self {
        Self { start, length }
    }

    pub fn end(&self) -> u16 {
        self.start + self.length
    }

    pub fn contains(&self, value: u16) -> bool {
        value >= self.start && value <= self.end()
    }

    pub fn cardinality(&self) -> usize {
        self.length as usize + 1
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct RunContainer {
    /// Sorted, non-overlapping, non-adjacent runs.
    runs: Vec<Run>,
    /// Cached total element count — maintained on every mutation; O(1) to read.
    cardinality: usize,
}

impl RunContainer {
    pub fn new() -> Self {
        Self {
            runs: Vec::new(),
            cardinality: 0,
        }
    }

    pub fn from_runs(runs: Vec<Run>) -> Self {
        debug_assert!(runs.windows(2).all(|w| w[0].end() < w[1].start.saturating_sub(0)));
        let cardinality = runs.iter().map(|r| r.cardinality()).sum();
        Self { runs, cardinality }
    }

    /// Build a RunContainer from a sorted slice of values.
    pub fn from_sorted_values(values: &[u16]) -> Self {
        if values.is_empty() {
            return Self::new();
        }
        let mut runs = Vec::new();
        let mut start = values[0];
        let mut prev = values[0];
        for &v in &values[1..] {
            if v == prev + 1 {
                prev = v;
            } else {
                runs.push(Run::new(start, prev - start));
                start = v;
                prev = v;
            }
        }
        runs.push(Run::new(start, prev - start));
        let cardinality = values.len();
        Self { runs, cardinality }
    }

    /// Build directly from a pre-validated runs vec and known cardinality.
    fn from_runs_with_cardinality(runs: Vec<Run>, cardinality: usize) -> Self {
        Self { runs, cardinality }
    }

    /// Find the run index that could contain `value`.
    fn find_run(&self, value: u16) -> Result<usize, usize> {
        self.runs.binary_search_by(|run| {
            if value < run.start {
                std::cmp::Ordering::Greater
            } else if value > run.end() {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
    }

    pub fn insert(&mut self, value: u16) -> bool {
        match self.find_run(value) {
            Ok(_) => false, // already in a run
            Err(pos) => {
                let extends_prev = pos > 0 && self.runs[pos - 1].end() + 1 == value;
                let extends_next = pos < self.runs.len() && self.runs[pos].start == value + 1;

                match (extends_prev, extends_next) {
                    (true, true) => {
                        let next_end = self.runs[pos].end();
                        self.runs[pos - 1].length = next_end - self.runs[pos - 1].start;
                        self.runs.remove(pos);
                    }
                    (true, false) => {
                        self.runs[pos - 1].length += 1;
                    }
                    (false, true) => {
                        self.runs[pos].start = value;
                        self.runs[pos].length += 1;
                    }
                    (false, false) => {
                        self.runs.insert(pos, Run::new(value, 0));
                    }
                }
                self.cardinality += 1;
                true
            }
        }
    }

    pub fn remove(&mut self, value: u16) -> bool {
        match self.find_run(value) {
            Ok(idx) => {
                let run = self.runs[idx];
                if run.length == 0 {
                    self.runs.remove(idx);
                } else if value == run.start {
                    self.runs[idx].start += 1;
                    self.runs[idx].length -= 1;
                } else if value == run.end() {
                    self.runs[idx].length -= 1;
                } else {
                    let new_run = Run::new(value + 1, run.end() - value - 1);
                    self.runs[idx].length = value - run.start - 1;
                    self.runs.insert(idx + 1, new_run);
                }
                self.cardinality -= 1;
                true
            }
            Err(_) => false,
        }
    }

    pub fn contains(&self, value: u16) -> bool {
        self.find_run(value).is_ok()
    }

    /// O(1) — returns the cached count.
    pub fn cardinality(&self) -> usize {
        self.cardinality
    }

    /// Alias for `cardinality()`.
    pub fn popcount(&self) -> usize {
        self.cardinality
    }

    pub fn is_empty(&self) -> bool {
        self.cardinality == 0
    }

    pub fn num_runs(&self) -> usize {
        self.runs.len()
    }

    pub fn runs(&self) -> &[Run] {
        &self.runs
    }

    pub fn min(&self) -> Option<u16> {
        self.runs.first().map(|r| r.start)
    }

    pub fn max(&self) -> Option<u16> {
        self.runs.last().map(|r| r.end())
    }

    /// Expand all runs into a sorted Vec<u16>.
    pub fn to_values(&self) -> Vec<u16> {
        let mut result = Vec::with_capacity(self.cardinality);
        for run in &self.runs {
            for v in run.start..=run.end() {
                result.push(v);
            }
        }
        result
    }

    pub fn iter(&self) -> RunIter<'_> {
        RunIter {
            runs: &self.runs,
            run_idx: 0,
            current: 0,
            initialized: false,
        }
    }

    /// Heuristic: run container is efficient when it uses less memory than an ArrayContainer.
    /// Each run costs 4 bytes; an array element costs 2 bytes.
    pub fn is_efficient(&self) -> bool {
        self.runs.len() * 4 <= self.cardinality * 2
    }

    /// Count of elements ≤ `value` in this container.
    ///
    /// Walks runs in order — O(n_runs).
    pub fn rank(&self, value: u16) -> usize {
        let mut count = 0;
        for run in &self.runs {
            if run.start > value {
                break;
            }
            if run.end() <= value {
                count += run.cardinality();
            } else {
                // value falls inside this run
                count += (value - run.start + 1) as usize;
                break;
            }
        }
        count
    }

    /// The `rank`-th element (0-indexed) in sorted order, or `None` if out of bounds.
    ///
    /// Walks runs, subtracting each run's cardinality until `rank` falls within a run.
    pub fn select(&self, mut rank: usize) -> Option<u16> {
        for run in &self.runs {
            let card = run.cardinality();
            if rank < card {
                return Some(run.start + rank as u16);
            }
            rank -= card;
        }
        None
    }

    // ── Set operations ──────────────────────────────────────────────

    pub fn and(&self, other: &RunContainer) -> RunContainer {
        let mut result = Vec::new();
        let (mut i, mut j) = (0, 0);
        while i < self.runs.len() && j < other.runs.len() {
            let a = self.runs[i];
            let b = other.runs[j];
            let start = a.start.max(b.start);
            let end = a.end().min(b.end());
            if start <= end {
                result.push(Run::new(start, end - start));
            }
            if a.end() < b.end() {
                i += 1;
            } else {
                j += 1;
            }
        }
        let cardinality = result.iter().map(|r| r.cardinality()).sum();
        RunContainer::from_runs_with_cardinality(result, cardinality)
    }

    pub fn or(&self, other: &RunContainer) -> RunContainer {
        let mut merged = Vec::with_capacity(self.runs.len() + other.runs.len());
        let (mut i, mut j) = (0, 0);
        while i < self.runs.len() && j < other.runs.len() {
            if self.runs[i].start <= other.runs[j].start {
                merged.push(self.runs[i]);
                i += 1;
            } else {
                merged.push(other.runs[j]);
                j += 1;
            }
        }
        merged.extend_from_slice(&self.runs[i..]);
        merged.extend_from_slice(&other.runs[j..]);

        if merged.is_empty() {
            return RunContainer::new();
        }
        let mut result = vec![merged[0]];
        for &run in &merged[1..] {
            let last = result.last_mut().unwrap();
            if run.start <= last.end().saturating_add(1) {
                let new_end = last.end().max(run.end());
                last.length = new_end - last.start;
            } else {
                result.push(run);
            }
        }
        let cardinality = result.iter().map(|r| r.cardinality()).sum();
        RunContainer::from_runs_with_cardinality(result, cardinality)
    }

    pub fn and_not(&self, other: &RunContainer) -> RunContainer {
        let mut result = Vec::new();
        let mut j = 0;
        for &a in &self.runs {
            let mut start = a.start;
            let end = a.end();
            while j < other.runs.len() && other.runs[j].end() < start {
                j += 1;
            }
            let mut k = j;
            while k < other.runs.len() && other.runs[k].start <= end {
                let b = other.runs[k];
                if start < b.start {
                    result.push(Run::new(start, b.start - 1 - start));
                }
                start = b.end().saturating_add(1);
                k += 1;
            }
            if start <= end {
                result.push(Run::new(start, end - start));
            }
        }
        let cardinality = result.iter().map(|r| r.cardinality()).sum();
        RunContainer::from_runs_with_cardinality(result, cardinality)
    }
}

impl Default for RunContainer {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RunIter<'a> {
    runs: &'a [Run],
    run_idx: usize,
    current: u16,
    initialized: bool,
}

impl Iterator for RunIter<'_> {
    type Item = u16;

    fn next(&mut self) -> Option<u16> {
        if !self.initialized {
            self.initialized = true;
            if self.run_idx < self.runs.len() {
                self.current = self.runs[self.run_idx].start;
            }
        }
        loop {
            if self.run_idx >= self.runs.len() {
                return None;
            }
            let run = &self.runs[self.run_idx];
            if self.current <= run.end() {
                let val = self.current;
                self.current = self.current.wrapping_add(1);
                return Some(val);
            }
            self.run_idx += 1;
            if self.run_idx < self.runs.len() {
                self.current = self.runs[self.run_idx].start;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_contains() {
        let mut c = RunContainer::new();
        assert!(c.insert(5));
        assert!(c.insert(6));
        assert!(c.insert(7));
        assert!(!c.insert(6)); // duplicate
        assert_eq!(c.num_runs(), 1);
        assert_eq!(c.cardinality(), 3);
        assert!(c.contains(5));
        assert!(c.contains(6));
        assert!(c.contains(7));
        assert!(!c.contains(4));
    }

    #[test]
    fn insert_merges_runs() {
        let mut c = RunContainer::new();
        c.insert(1);
        c.insert(3);
        assert_eq!(c.num_runs(), 2);
        c.insert(2); // bridges the gap
        assert_eq!(c.num_runs(), 1);
        assert_eq!(c.cardinality(), 3);
    }

    #[test]
    fn remove_splits_run() {
        let mut c = RunContainer::from_sorted_values(&[10, 11, 12, 13, 14]);
        assert_eq!(c.num_runs(), 1);
        c.remove(12);
        assert_eq!(c.num_runs(), 2);
        assert_eq!(c.cardinality(), 4);
        assert!(!c.contains(12));
        assert!(c.contains(10));
        assert!(c.contains(14));
    }

    #[test]
    fn from_sorted_values() {
        let c = RunContainer::from_sorted_values(&[1, 2, 3, 10, 11, 20]);
        assert_eq!(c.num_runs(), 3);
        assert_eq!(c.cardinality(), 6);
    }

    #[test]
    fn cardinality_cached_after_mutations() {
        let mut c = RunContainer::new();
        for i in 0u16..100 {
            c.insert(i);
        }
        assert_eq!(c.cardinality(), 100);
        for i in 0u16..50 {
            c.remove(i);
        }
        assert_eq!(c.cardinality(), 50);
    }

    #[test]
    fn min_max() {
        let c = RunContainer::from_sorted_values(&[10, 11, 12, 50, 51]);
        assert_eq!(c.min(), Some(10));
        assert_eq!(c.max(), Some(51));
    }

    #[test]
    fn and_operation() {
        let a = RunContainer::from_sorted_values(&[1, 2, 3, 4, 5, 10, 11, 12]);
        let b = RunContainer::from_sorted_values(&[3, 4, 5, 6, 7, 11, 12, 13]);
        let result = a.and(&b);
        assert_eq!(result.to_values(), vec![3, 4, 5, 11, 12]);
        assert_eq!(result.cardinality(), 5);
    }

    #[test]
    fn or_operation() {
        let a = RunContainer::from_sorted_values(&[1, 2, 3]);
        let b = RunContainer::from_sorted_values(&[3, 4, 5]);
        let result = a.or(&b);
        assert_eq!(result.to_values(), vec![1, 2, 3, 4, 5]);
        assert_eq!(result.num_runs(), 1);
        assert_eq!(result.cardinality(), 5);
    }

    #[test]
    fn and_not_operation() {
        let a = RunContainer::from_sorted_values(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let b = RunContainer::from_sorted_values(&[3, 4, 5, 8, 9]);
        let result = a.and_not(&b);
        assert_eq!(result.to_values(), vec![1, 2, 6, 7, 10]);
        assert_eq!(result.cardinality(), 5);
    }

    #[test]
    fn iter() {
        let c = RunContainer::from_sorted_values(&[1, 2, 3, 10, 11]);
        let collected: Vec<u16> = c.iter().collect();
        assert_eq!(collected, vec![1, 2, 3, 10, 11]);
    }

    #[test]
    fn rank_basic() {
        // runs: [10..=12, 20..=22]
        let c = RunContainer::from_sorted_values(&[10, 11, 12, 20, 21, 22]);
        assert_eq!(c.rank(5), 0); // before first run
        assert_eq!(c.rank(10), 1); // start of first run
        assert_eq!(c.rank(11), 2);
        assert_eq!(c.rank(12), 3); // end of first run
        assert_eq!(c.rank(15), 3); // between runs
        assert_eq!(c.rank(20), 4); // start of second run
        assert_eq!(c.rank(22), 6); // end of second run
        assert_eq!(c.rank(99), 6); // beyond all runs
    }

    #[test]
    fn rank_empty() {
        let c = RunContainer::new();
        assert_eq!(c.rank(0), 0);
        assert_eq!(c.rank(u16::MAX), 0);
    }

    #[test]
    fn select_basic() {
        // runs: [10..=12, 20..=21]
        let c = RunContainer::from_sorted_values(&[10, 11, 12, 20, 21]);
        assert_eq!(c.select(0), Some(10));
        assert_eq!(c.select(1), Some(11));
        assert_eq!(c.select(2), Some(12));
        assert_eq!(c.select(3), Some(20));
        assert_eq!(c.select(4), Some(21));
        assert_eq!(c.select(5), None);
    }

    #[test]
    fn rank_select_round_trip() {
        let vals: Vec<u16> = (0..50u16).flat_map(|i| [i * 4, i * 4 + 1, i * 4 + 2]).collect();
        let c = RunContainer::from_sorted_values(&vals);
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(c.select(i), Some(v), "select({i})");
            assert_eq!(c.rank(v), i + 1, "rank({v})");
        }
    }
}
