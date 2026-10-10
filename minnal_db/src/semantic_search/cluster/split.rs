//! Balanced 2-means: how a posting is split in two (design doc M3, *M3 split
//! algorithm (LIRE)*, step 3).
//!
//! This is SPTAG's `KmeansClustering` / `TryClustering` with k = 2, the
//! clustering SPFresh's LIRE protocol splits postings with
//! (`AnnService/inc/Core/Common/BKTree.h`), with one deviation: the two centres
//! are kept as the **means** of their members, where SPTAG replaces each with
//! its nearest member. In minnal the members are reconstructions of 1-bit codes,
//! and a single member's reconstruction carries its full code error; the mean
//! averages it away (`report/m3-pre-simulation/spfresh-comparison.md`).
//!
//! Each step:
//!
//! 1. **Start.** `init_trials` times, take two random members as centres and
//!    assign a random sample of `S = min(samples, n)` members to the nearer;
//!    keep the trial with the smallest total squared distance, and from it
//!    `λ_spread = (largest − mean distance in its larger cluster) / S`.
//! 2. **Balance weight.** `λ = min(λ_spread, 1 / (lambda_factor · S))`.
//! 3. **Iterate** at most `max_iters` times: draw a fresh sample of S members,
//!    assign each to the centre `k` minimising `‖x − c_k‖² + λ · size_k` (sizes
//!    from the previous iteration), move each centre to the mean of its members
//!    (an empty cluster takes the farthest member of the larger one). Stop when
//!    the centres move less than 0.001 in total (squared) or after 5 iterations
//!    without a lower total distance.
//! 4. **Assign** every member to the nearer centre without the balance term.
//!    `None` if one side is empty.

/// The parameters of a split (the namespace's `maintenance` settings).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitParams {
    /// Members sampled per iteration.
    pub samples: usize,
    /// Random starting pairs tried.
    pub init_trials: usize,
    /// Iteration cap.
    pub max_iters: usize,
    /// Balance factor: `λ ≤ 1 / (factor · samples)`.
    pub lambda_factor: f64,
}

impl Default for SplitParams {
    /// SPTAG's values.
    fn default() -> Self {
        Self {
            samples: 1000,
            init_trials: 3,
            max_iters: 100,
            lambda_factor: 100.0,
        }
    }
}

/// The outcome of a split.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitResult {
    /// The two new centres (means of their members).
    pub centres: [Vec<f32>; 2],
    /// For every input point, the centre (0 or 1) it is assigned to.
    pub labels: Vec<u8>,
}

/// Iterations without a lower total distance before stopping (SPTAG).
const NO_IMPROVEMENT_LIMIT: usize = 5;
/// Total squared centre movement below which the iteration has converged (SPTAG).
const CONVERGED: f64 = 1e-3;

/// A small seeded generator (SplitMix64): splits are reproducible for a given
/// seed, and no dependency is needed.
#[derive(Debug, Clone)]
pub struct SplitRng(u64);

impl SplitRng {
    /// A generator seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (`n > 0`).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// `k` distinct indices from `0..n`, in random order (partial Fisher–Yates).
    fn sample(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in 0..k.min(n) {
            let j = i + self.below(n - i);
            idx.swap(i, j);
        }
        idx.truncate(k.min(n));
        idx
    }
}

fn sq_dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = f64::from(*x) - f64::from(*y);
            d * d
        })
        .sum()
}

/// Assign `members` to the nearer of `centres` with balance weight `lambda` on
/// cluster sizes `sizes`. Returns each member's label and its (weighted)
/// distance.
fn assign(points: &[Vec<f32>], members: &[usize], centres: &[Vec<f32>; 2], lambda: f64, sizes: [f64; 2]) -> (Vec<u8>, Vec<f64>) {
    members
        .iter()
        .map(|&i| {
            let d0 = sq_dist(&points[i], &centres[0]) + lambda * sizes[0];
            let d1 = sq_dist(&points[i], &centres[1]) + lambda * sizes[1];
            if d1 < d0 { (1u8, d1) } else { (0u8, d0) }
        })
        .unzip()
}

fn sizes_of(labels: &[u8]) -> [f64; 2] {
    let ones = labels.iter().filter(|&&l| l == 1).count();
    [(labels.len() - ones) as f64, ones as f64]
}

/// Split `points` into two balanced groups (see the module docs). `None` when
/// there are fewer than two points or one side ends empty (all points alike).
pub fn balanced_two_means(points: &[Vec<f32>], params: &SplitParams, rng: &mut SplitRng) -> Option<SplitResult> {
    let n = points.len();
    if n < 2 {
        return None;
    }
    let dim = points[0].len();
    let s = params.samples.clamp(2, n);

    // 1. Start: the best of `init_trials` random pairs on one sample.
    let batch = rng.sample(n, s);
    let mut best = f64::INFINITY;
    let mut centres = [points[0].clone(), points[1].clone()];
    let mut sizes = [0.0f64; 2];
    let mut lambda_spread = 0.0f64;
    for _ in 0..params.init_trials.max(1) {
        let cand = [points[rng.below(n)].clone(), points[rng.below(n)].clone()];
        let (labels, dists) = assign(points, &batch, &cand, 0.0, [0.0, 0.0]);
        let total: f64 = dists.iter().sum();
        if total < best {
            best = total;
            sizes = sizes_of(&labels);
            let larger = u8::from(sizes[1] > sizes[0]);
            let inside: Vec<f64> = labels.iter().zip(&dists).filter(|(l, _)| **l == larger).map(|(_, d)| *d).collect();
            lambda_spread = if inside.is_empty() {
                0.0
            } else {
                let max = inside.iter().copied().fold(f64::MIN, f64::max);
                let mean = inside.iter().sum::<f64>() / inside.len() as f64;
                ((max - mean) / s as f64).max(0.0)
            };
            centres = cand;
        }
    }
    // 2. Balance weight.
    let lambda = lambda_spread.min(1.0 / (params.lambda_factor * s as f64));

    // 3. Iterate on fresh samples.
    let mut min_total = f64::INFINITY;
    let mut no_improvement = 0;
    for _ in 0..params.max_iters.max(1) {
        let batch = rng.sample(n, s);
        let (labels, dists) = assign(points, &batch, &centres, lambda, sizes);
        sizes = sizes_of(&labels);
        let total: f64 = dists.iter().sum();
        if total < min_total {
            min_total = total;
            no_improvement = 0;
        } else {
            no_improvement += 1;
        }
        let mut sums = [vec![0.0f64; dim], vec![0.0f64; dim]];
        for (&i, &l) in batch.iter().zip(&labels) {
            for (acc, &x) in sums[l as usize].iter_mut().zip(&points[i]) {
                *acc += f64::from(x);
            }
        }
        let mut next = centres.clone();
        for k in 0..2 {
            if sizes[k] > 0.0 {
                next[k] = sums[k].iter().map(|v| (v / sizes[k]) as f32).collect();
            } else {
                // Empty: take the farthest member of the larger cluster.
                let larger = (1 - k) as u8;
                if let Some((&i, _)) = batch
                    .iter()
                    .zip(labels.iter().zip(&dists))
                    .filter(|(_, (l, _))| **l == larger)
                    .max_by(|a, b| a.1.1.total_cmp(b.1.1))
                {
                    next[k] = points[i].clone();
                }
            }
        }
        let moved = sq_dist(&next[0], &centres[0]) + sq_dist(&next[1], &centres[1]);
        centres = next;
        if moved < CONVERGED || no_improvement >= NO_IMPROVEMENT_LIMIT {
            break;
        }
    }

    // 4. Every member to the nearer mean, without the balance term.
    let all: Vec<usize> = (0..n).collect();
    let (labels, _) = assign(points, &all, &centres, 0.0, [0.0, 0.0]);
    let ones = labels.iter().filter(|&&l| l == 1).count();
    if ones == 0 || ones == n {
        return None;
    }
    Some(SplitResult { centres, labels })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` points around `centre` with spread `noise`.
    fn blob(rng: &mut SplitRng, centre: &[f32], n: usize, noise: f32) -> Vec<Vec<f32>> {
        (0..n)
            .map(|_| {
                centre
                    .iter()
                    .map(|&c| c + noise * ((rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn two_clear_groups_are_found_and_the_centres_are_their_means() {
        let mut g = SplitRng::new(1);
        let mut pts = blob(&mut g, &[1.0, 0.0, 0.0, 0.0], 60, 0.2);
        pts.extend(blob(&mut g, &[0.0, 1.0, 0.0, 0.0], 60, 0.2));
        let r = balanced_two_means(&pts, &SplitParams::default(), &mut SplitRng::new(7)).unwrap();
        // Every point of a blob shares a label, and the blobs differ.
        assert!(r.labels[..60].iter().all(|&l| l == r.labels[0]));
        assert!(r.labels[60..].iter().all(|&l| l == r.labels[60]));
        assert_ne!(r.labels[0], r.labels[60]);
        // Each centre is (close to) the mean of its members.
        for k in 0..2u8 {
            let members: Vec<&Vec<f32>> = pts.iter().zip(&r.labels).filter(|(_, l)| **l == k).map(|(p, _)| p).collect();
            let mean: Vec<f32> = (0..4).map(|d| members.iter().map(|p| p[d]).sum::<f32>() / members.len() as f32).collect();
            assert!(sq_dist(&mean, &r.centres[k as usize]) < 1e-3, "{mean:?} vs {:?}", r.centres[k as usize]);
        }
    }

    #[test]
    fn a_structureless_cloud_is_split_into_roughly_equal_halves() {
        // With no gap to follow, the balance term keeps the halves even. (It is
        // deliberately weak, as in SPTAG: a group of real outliers far away
        // still becomes its own half.)
        let mut g = SplitRng::new(3);
        let pts = blob(&mut g, &[0.0; 8], 400, 1.0);
        for seed in 0..5 {
            let r = balanced_two_means(&pts, &SplitParams::default(), &mut SplitRng::new(seed)).unwrap();
            let ones = r.labels.iter().filter(|&&l| l == 1).count();
            assert!((120..=280).contains(&ones), "seed {seed}: sides {ones} / {}", pts.len() - ones);
        }
    }

    #[test]
    fn identical_points_cannot_be_split() {
        let pts = vec![vec![0.5f32; 4]; 30];
        assert_eq!(balanced_two_means(&pts, &SplitParams::default(), &mut SplitRng::new(1)), None);
        assert_eq!(balanced_two_means(&pts[..1], &SplitParams::default(), &mut SplitRng::new(1)), None);
    }

    #[test]
    fn the_same_seed_gives_the_same_split() {
        let mut g = SplitRng::new(11);
        let pts = blob(&mut g, &[0.0; 16], 300, 1.0);
        let p = SplitParams {
            samples: 100,
            ..SplitParams::default()
        };
        let a = balanced_two_means(&pts, &p, &mut SplitRng::new(42)).unwrap();
        let b = balanced_two_means(&pts, &p, &mut SplitRng::new(42)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn sampling_draws_distinct_indices() {
        let mut g = SplitRng::new(9);
        let s = g.sample(50, 20);
        let mut sorted = s.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!((s.len(), sorted.len()), (20, 20));
        assert!(s.iter().all(|&i| i < 50));
        assert_eq!(g.sample(5, 10).len(), 5);
    }
}
