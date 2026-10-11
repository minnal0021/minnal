//! K-means over a namespace's chunks: how a partition is rebuilt from its codes
//! (design doc M4).
//!
//! The same procedure as the M3-pre simulation's rebuild (`sim.kmeans`), which
//! measured a partition fitted on 1-bit reconstructions as good as one fitted
//! on the floats (FiQA recall@10 0.978 against 0.976):
//!
//! 1. **Seed** with k-means++ on a random sample of at most `seeding_sample`
//!    points: the first centre at random, each next one drawn with probability
//!    proportional to its squared distance from the nearest centre so far.
//! 2. **Iterate** `iterations` times (Lloyd): assign every point to its nearest
//!    centre, move each centre to the mean of its points. A centre left empty
//!    takes a random point. Stops early once no point changes centre.
//!
//! Points are never held all at once. The caller supplies a function that
//! writes point `i` into a buffer (for codes, its reconstruction `c + s·σ`), so
//! memory is the codes plus `k` centres, not every chunk as floats. Assignment
//! runs in parallel on the current rayon pool; the means are summed in point
//! order, so a seed always gives the same centres.

use rayon::prelude::*;

use super::split::SplitRng;

/// How k-means runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KmeansParams {
    /// Centres wanted (fewer when there are fewer points).
    pub k: usize,
    /// Lloyd iterations at most.
    pub iterations: usize,
    /// Points k-means++ seeds from.
    pub seeding_sample: usize,
}

impl KmeansParams {
    /// `k` centres with the simulation's settings: 20 iterations, seeded from a
    /// sample of 50,000.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            iterations: 20,
            seeding_sample: 50_000,
        }
    }
}

/// The fitted centres.
#[derive(Debug, Clone, PartialEq)]
pub struct KmeansResult {
    /// The centres (means of their points).
    pub centres: Vec<Vec<f32>>,
    /// Points nearest each centre at the last assignment.
    pub sizes: Vec<u64>,
    /// Lloyd iterations run.
    pub iterations: usize,
}

fn sq_dist(a: &[f32], b: &[f32]) -> f64 {
    use simsimd::SpatialSimilarity;
    f32::sqeuclidean(a, b).expect("equal dimensions")
}

/// The nearest of `centres` to `x` (ties to the lower index) and its distance.
fn nearest(x: &[f32], centres: &[Vec<f32>]) -> (u32, f64) {
    let mut best = (0u32, f64::INFINITY);
    for (i, c) in centres.iter().enumerate() {
        let d = sq_dist(x, c);
        if d < best.1 {
            best = (i as u32, d);
        }
    }
    best
}

/// Points per parallel task.
const BLOCK: usize = 1024;

/// Fit `params.k` centres to `n` points of dimension `dim`; `point(i, buf)`
/// writes point `i` into `buf` (handed over empty). `None` when there are no points.
pub fn kmeans<F>(n: usize, dim: usize, point: F, params: &KmeansParams, rng: &mut SplitRng) -> Option<KmeansResult>
where
    F: Fn(usize, &mut Vec<f32>) + Sync,
{
    if n == 0 || params.k == 0 {
        return None;
    }
    let k = params.k.min(n);
    let point = |i: usize, buf: &mut Vec<f32>| {
        buf.clear();
        point(i, buf);
    };
    let get = |i: usize| {
        let mut buf = Vec::with_capacity(dim);
        point(i, &mut buf);
        buf
    };

    // 1. k-means++ on a sample.
    let sample = rng.sample(n, params.seeding_sample.max(k).min(n));
    let mut centres = vec![get(sample[rng.below(sample.len())])];
    let mut d2: Vec<f64> = vec![f64::INFINITY; sample.len()];
    while centres.len() < k {
        let last = centres.last().expect("one centre");
        d2.par_chunks_mut(BLOCK).enumerate().for_each(|(b, out)| {
            let mut buf = Vec::with_capacity(dim);
            for (j, d) in out.iter_mut().enumerate() {
                point(sample[b * BLOCK + j], &mut buf);
                *d = d.min(sq_dist(&buf, last));
            }
        });
        let total: f64 = d2.iter().sum();
        let next = if total > 0.0 {
            // A draw in [0, total), walked in sample order.
            let mut r = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * total;
            d2.iter()
                .position(|&d| {
                    r -= d;
                    r < 0.0
                })
                .unwrap_or(d2.len() - 1)
        } else {
            // Every sampled point sits on a centre; any will do.
            rng.below(sample.len())
        };
        centres.push(get(sample[next]));
    }

    // 2. Lloyd.
    let mut labels = vec![u32::MAX; n];
    let mut sizes = vec![0u64; k];
    let mut iterations = 0;
    for _ in 0..params.iterations.max(1) {
        iterations += 1;
        let changed: usize = labels
            .par_chunks_mut(BLOCK)
            .enumerate()
            .map(|(b, out)| {
                let mut buf = Vec::with_capacity(dim);
                let mut changed = 0;
                for (j, label) in out.iter_mut().enumerate() {
                    point(b * BLOCK + j, &mut buf);
                    let (c, _) = nearest(&buf, &centres);
                    changed += usize::from(*label != c);
                    *label = c;
                }
                changed
            })
            .sum();
        sizes = vec![0u64; k];
        for &l in &labels {
            sizes[l as usize] += 1;
        }
        if changed == 0 {
            break;
        }
        // Means, summed in point order.
        let mut sums = vec![vec![0.0f64; dim]; k];
        let mut buf = Vec::with_capacity(dim);
        for (i, &l) in labels.iter().enumerate() {
            point(i, &mut buf);
            for (acc, &x) in sums[l as usize].iter_mut().zip(&buf) {
                *acc += f64::from(x);
            }
        }
        for (c, (sum, &size)) in centres.iter_mut().zip(sums.iter().zip(&sizes)) {
            *c = if size > 0 {
                sum.iter().map(|v| (v / size as f64) as f32).collect()
            } else {
                get(rng.below(n))
            };
        }
    }
    Some(KmeansResult { centres, sizes, iterations })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `per` points around each of `centres`, spread `noise`, in blob order.
    fn blobs(rng: &mut SplitRng, centres: &[Vec<f32>], per: usize, noise: f32) -> Vec<Vec<f32>> {
        centres
            .iter()
            .flat_map(|c| {
                (0..per)
                    .map(|_| {
                        c.iter()
                            .map(|&v| v + noise * ((rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5))
                            .collect::<Vec<f32>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn run(points: &[Vec<f32>], params: &KmeansParams, seed: u64) -> KmeansResult {
        kmeans(
            points.len(),
            points[0].len(),
            |i, buf| buf.extend_from_slice(&points[i]),
            params,
            &mut SplitRng::new(seed),
        )
        .unwrap()
    }

    fn corners(k: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..k).map(|i| (0..dim).map(|d| if d == i { 4.0 } else { 0.0 }).collect()).collect()
    }

    #[test]
    fn separated_groups_each_get_their_own_centre_at_their_mean() {
        let truth = corners(6, 8);
        let pts = blobs(&mut SplitRng::new(1), &truth, 200, 0.5);
        let r = run(&pts, &KmeansParams::new(6), 3);
        assert_eq!(r.sizes.iter().copied().collect::<std::collections::BTreeSet<_>>(), [200].into());
        for t in &truth {
            let (c, d) = nearest(t, &r.centres);
            assert!(d < 0.05, "centre {c} is {d} from {t:?}");
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_centres_on_any_pool() {
        let pts = blobs(&mut SplitRng::new(2), &corners(5, 16), 3000, 3.0);
        let params = KmeansParams {
            seeding_sample: 2000,
            ..KmeansParams::new(12)
        };
        let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let four = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let a = one.install(|| run(&pts, &params, 9));
        let b = four.install(|| run(&pts, &params, 9));
        assert_eq!(a, b);
        assert_ne!(a, run(&pts, &params, 10), "another seed, other centres");
    }

    #[test]
    fn fewer_points_than_centres_gives_one_centre_per_point() {
        let pts = corners(3, 4);
        let r = run(&pts, &KmeansParams::new(10), 1);
        assert_eq!(r.centres.len(), 3);
        assert_eq!(r.sizes, vec![1, 1, 1]);
        assert_eq!(kmeans(0, 4, |_, _| {}, &KmeansParams::new(3), &mut SplitRng::new(1)), None);
    }

    #[test]
    fn identical_points_end_without_a_division_by_zero() {
        let pts = vec![vec![0.25f32; 4]; 50];
        let r = run(&pts, &KmeansParams::new(4), 1);
        assert_eq!(r.sizes.iter().sum::<u64>(), 50);
        assert!(r.centres.iter().flatten().all(|v| v.is_finite()));
    }
}

#[cfg(test)]
mod timing {
    use super::*;

    /// FiQA-sized: 173k 768-d points, 988 centres; prints seconds per Lloyd
    /// iteration. `cargo test --release -- --ignored kmeans_at_fiqa_size --nocapture`
    #[test]
    #[ignore]
    fn kmeans_at_fiqa_size() {
        let (n, dim, k) = (172_998, 768, 988);
        let mut g = SplitRng::new(5);
        let pts: Vec<f32> = (0..n * dim).map(|_| (g.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5).collect();
        for (threads, iters) in [(2, 1), (2, 3), (8, 1), (8, 3)] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let params = KmeansParams {
                iterations: iters,
                ..KmeansParams::new(k)
            };
            let t = std::time::Instant::now();
            let r = pool.install(|| {
                kmeans(
                    n,
                    dim,
                    |i, buf| buf.extend_from_slice(&pts[i * dim..(i + 1) * dim]),
                    &params,
                    &mut SplitRng::new(1),
                )
                .unwrap()
            });
            let total = t.elapsed().as_secs_f64();
            println!("threads {threads}: {total:.1} s for seeding + {} iterations", r.iterations);
        }
    }
}
