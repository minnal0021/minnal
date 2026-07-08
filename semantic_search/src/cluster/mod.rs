use log::info;
use simsimd::SpatialSimilarity;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Cluster {
    pub cluster_id: u32,
    pub centroid: Vec<f32>,
}

impl Cluster {
    pub fn euclidean_distance(&self, other: &[f32]) -> f32 {
        // A dimension mismatch makes `l2sq` return `None`, which we mask to
        // `f64::MAX` ("infinitely far") so the caller never panics — but a masked
        // distance silently corrupts nearest-cluster selection (every cluster ties
        // at MAX, so `find_closest_cluster_id` can return `u32::MAX`). That must
        // never happen in production: `ClusterIndex::load_with_dim` validates the
        // centroids against `embedding_dim`, and the embedding service validates
        // every returned vector against the same dim, so the two operands are
        // guaranteed equal-length. The assert turns any remaining mismatch into a
        // loud dev-time failure instead of a silent wrong answer.
        debug_assert_eq!(
            self.centroid.len(),
            other.len(),
            "euclidean_distance dimension mismatch (centroid {} vs operand {}); \
             load_with_dim and the embed boundary should make this impossible",
            self.centroid.len(),
            other.len(),
        );
        f32::l2sq(&self.centroid, other).unwrap_or(f64::MAX) as f32
    }

    pub fn new(cluster_id: u32, centroid: Vec<f32>) -> Cluster {
        Cluster { cluster_id, centroid }
    }
}

/// Errors raised when validating a freshly-loaded cluster index.
///
/// These are all "the centroid file is unusable" conditions caught at load time,
/// before any query or document can hit a dimension mismatch deep in the search
/// path (where it would surface as a `u32::MAX` cluster id and a downstream
/// `unwrap` panic).
#[derive(Debug, thiserror::Error)]
pub enum ClusterIndexError {
    /// The file parsed but yielded no clusters.
    #[error("cluster file contains no clusters")]
    Empty,
    /// An embedding could not be assigned because the cluster map is empty.
    /// Startup validation ([`ClusterIndex::load`]) normally prevents this; it is
    /// the runtime guard for callers that pass an empty map directly.
    #[error("cannot assign embedding: the cluster index is empty")]
    EmptyClusterMap,
    /// A centroid vector had zero dimensions.
    #[error("cluster {cluster_id} has an empty centroid vector")]
    EmptyCentroid { cluster_id: u32 },
    /// The same `cluster_id` appeared on more than one line.
    #[error("duplicate cluster_id {cluster_id} in cluster file")]
    DuplicateClusterId { cluster_id: u32 },
    /// Centroids are not all the same length.
    #[error("cluster {cluster_id} centroid has dimension {found}, but other centroids have {expected} (centroids must be uniform)")]
    InconsistentDimension { cluster_id: u32, expected: usize, found: usize },
    /// Centroids are uniform but do not match the configured `embedding_dim`.
    #[error("centroid dimension {found} does not match the configured embedding_dim {expected}")]
    DimensionMismatch { expected: usize, found: usize },
}

/// Validate a centroid map and return the (uniform) centroid dimension.
///
/// Rejects an empty map, any empty centroid, and centroids of mixed length. When
/// `expected_dim` is `Some`, also rejects a uniform dimension that disagrees with
/// it. Duplicate cluster IDs are caught earlier, in [`read_clusters_from_file`].
fn validate_centroids(map: &HashMap<u32, Vec<f32>>, expected_dim: Option<usize>) -> Result<usize, ClusterIndexError> {
    let mut dim: Option<usize> = None;
    for (&cluster_id, centroid) in map {
        if centroid.is_empty() {
            return Err(ClusterIndexError::EmptyCentroid { cluster_id });
        }
        match dim {
            None => dim = Some(centroid.len()),
            Some(d) if d != centroid.len() => {
                return Err(ClusterIndexError::InconsistentDimension {
                    cluster_id,
                    expected: d,
                    found: centroid.len(),
                });
            }
            Some(_) => {}
        }
    }
    let dim = dim.ok_or(ClusterIndexError::Empty)?;
    if let Some(expected) = expected_dim
        && dim != expected
    {
        return Err(ClusterIndexError::DimensionMismatch { expected, found: dim });
    }
    Ok(dim)
}

/// The set of IVF cluster centroids, loaded once at startup and shared read-only
/// across all requests.
///
/// Clusters are probed by **exact** nearest-centroid distance. Coarse assignment is an
/// exhaustive scan: [`find_top_n_cluster_ids_batch`](ClusterIndex::find_top_n_cluster_ids_batch)
/// computes the distance from every query chunk to every centroid and returns the
/// `n_probes` nearest per chunk (see [`crate::service::search`]). There is deliberately
/// **no precomputed neighbour graph** — coarse-assignment cost is `T·C·D` (query chunks ×
/// centroids × dim), which at the cluster counts in use (a few hundred) is microseconds
/// per chunk; a graph is approximate on the most recall-sensitive stage and does nothing
/// about the per-chunk `T` factor. Revisit graph-over-centroids only if `C` reaches tens
/// of thousands. (Parallelising the per-chunk scans was benchmarked and *regressed* —
/// the work is too small to amortise a thread pool; see `find_top_n_cluster_ids_batch`.)
///
/// To make that scan cache-friendly the centroids are held **twice**: once as the
/// `clusters` map (id lookup) and once as a row-major `centroids` matrix with a parallel
/// `centroid_ids` (row → id). The map's `Cluster` values each own a separately
/// heap-allocated `Vec<f32>`, so scanning them pointer-chases scattered buffers; the
/// matrix is one contiguous allocation the scan streams sequentially (a measured ~12%
/// win). The two are built together and never mutated after construction, so they cannot
/// drift.
#[derive(Debug)]
pub struct ClusterIndex {
    /// All clusters keyed by their ID.
    pub clusters: HashMap<u32, Cluster>,
    /// Row-major `C×dim` centroid matrix — the same centroids as `clusters`, laid out
    /// contiguously so the coarse-assignment scan streams sequential memory. Row `r`
    /// holds `centroids[r*dim .. (r+1)*dim]` and belongs to cluster `centroid_ids[r]`.
    centroids: Vec<f32>,
    /// Cluster id for each row of `centroids` (same length as the row count).
    centroid_ids: Vec<u32>,
    /// The uniform centroid dimension, validated at load time. Every query and
    /// document embedding compared against these centroids must share it.
    dim: usize,
}

impl ClusterIndex {
    /// Load clusters from `cluster_file_path`, validating only the file's internal
    /// consistency.
    ///
    /// Rejects an empty file, empty centroids, duplicate cluster IDs, and
    /// centroids of mixed dimension. Use [`load_with_dim`](Self::load_with_dim) to
    /// additionally pin the centroid dimension to a configured `embedding_dim`.
    ///
    /// Returns an error if the file cannot be read, parsed, or fails validation.
    pub fn load(cluster_file_path: &str) -> Result<Self, Box<dyn Error>> {
        Self::load_inner(cluster_file_path, None)
    }

    /// Like [`load`](Self::load), but also rejects a cluster file whose (uniform)
    /// centroid dimension does not equal `embedding_dim`.
    ///
    /// This is the entry point the server uses: catching a centroid/embedding
    /// dimension mismatch here disables semantic search with a clear error at
    /// startup, instead of letting it surface as a `u32::MAX` nearest-cluster id
    /// and a downstream `unwrap` panic on the first document insert or query.
    pub fn load_with_dim(cluster_file_path: &str, embedding_dim: usize) -> Result<Self, Box<dyn Error>> {
        Self::load_inner(cluster_file_path, Some(embedding_dim))
    }

    fn load_inner(cluster_file_path: &str, expected_dim: Option<usize>) -> Result<Self, Box<dyn Error>> {
        let centroid_map = read_clusters_from_file(cluster_file_path)?;
        let dim = validate_centroids(&centroid_map, expected_dim)?;
        let clusters: HashMap<u32, Cluster> = centroid_map.into_iter().map(|(id, centroid)| (id, Cluster::new(id, centroid))).collect();
        Ok(Self::from_parts(clusters, dim))
    }

    /// Build the index from a validated cluster map of known dimension, deriving the
    /// contiguous `centroids` matrix / `centroid_ids` from the map in one pass.
    ///
    /// This is the single place the map and the matrix are materialised together, so
    /// they are consistent by construction. Row order follows the map's iteration
    /// order — arbitrary but irrelevant, since every row carries its own id in
    /// `centroid_ids` and results are ranked by distance.
    fn from_parts(clusters: HashMap<u32, Cluster>, dim: usize) -> Self {
        let mut centroids = Vec::with_capacity(clusters.len() * dim);
        let mut centroid_ids = Vec::with_capacity(clusters.len());
        for cluster in clusters.values() {
            centroids.extend_from_slice(&cluster.centroid);
            centroid_ids.push(cluster.cluster_id);
        }
        Self {
            clusters,
            centroids,
            centroid_ids,
            dim,
        }
    }

    /// Build an index directly from in-memory clusters, inferring the dimension
    /// from the first centroid (`0` if empty).
    ///
    /// For callers that already hold centroids in memory rather than a file. The
    /// per-centroid validation done by [`load`](Self::load) is the file path's
    /// concern; in-memory callers are trusted to pass uniform centroids.
    pub fn from_clusters(clusters: HashMap<u32, Cluster>) -> Self {
        let dim = clusters.values().next().map(|c| c.centroid.len()).unwrap_or(0);
        Self::from_parts(clusters, dim)
    }

    /// The uniform centroid dimension every embedding must match.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The number of centroids (clusters) in the index.
    pub fn len(&self) -> usize {
        self.centroid_ids.len()
    }

    /// Whether the index holds no centroids.
    pub fn is_empty(&self) -> bool {
        self.centroid_ids.is_empty()
    }

    /// Batched coarse assignment: for each query chunk in `queries`, the ids of the `n`
    /// centroids closest to it (ascending distance), returned as one `Vec<u32>` per
    /// query in input order.
    ///
    /// This is the Pass-1 cluster-probing primitive, called once per search over all
    /// query chunks. It is exact and exhaustive — every query is compared against every
    /// centroid — and differs from the historical per-query [`find_top_n_cluster_ids`]
    /// loop over the `clusters` map only in that each query streams the contiguous,
    /// row-major `centroids` matrix in order instead of pointer-chasing the map's
    /// scattered per-`Cluster` `Vec`s. At the cluster counts in use (a few hundred) that
    /// contiguous layout is a measured ~12% win at 100 query chunks (`bench_distance_estimation`,
    /// `coarse_assignment`).
    ///
    /// The scan is deliberately **serial** across query chunks. Parallelising it with
    /// rayon was benchmarked and *regressed* (~2.4× slower at 100 chunks): each per-query
    /// scan is only microseconds, so the pool's dispatch/sync overhead dwarfs the work.
    /// Batching only becomes worth parallelising via a blocked GEMM (reusing each
    /// centroid tile across queries), which is worth revisiting only at far larger `C`.
    ///
    /// Distances use the same `l2sq` primitive as [`find_top_n_cluster_ids`], so the
    /// per-query result is identical to calling that function on each query (the
    /// distance-ranked `n` nearest ids). When `n >= len()` every id is returned.
    pub fn find_top_n_cluster_ids_batch(&self, queries: &[Vec<f32>], n: usize) -> Vec<Vec<u32>> {
        queries.iter().map(|q| self.top_n_over_matrix(q, n)).collect()
    }

    /// Top-`n` nearest centroid ids for a single query, computed over the contiguous
    /// `centroids` matrix. Selection mirrors [`find_top_n_cluster_ids`]: partition the
    /// `n` smallest by distance in ~O(C) with `select_nth_unstable`, then sort only
    /// those `n`.
    fn top_n_over_matrix(&self, embedding: &[f32], n: usize) -> Vec<u32> {
        let mut distances: Vec<(u32, f32)> = self
            .centroid_ids
            .iter()
            .enumerate()
            .map(|(row, &id)| {
                let base = row * self.dim;
                let centroid = &self.centroids[base..base + self.dim];
                // Masked like Cluster::euclidean_distance — load_with_dim and the embed
                // boundary guarantee equal lengths, so this never actually fires.
                let distance = f32::l2sq(centroid, embedding).unwrap_or(f64::MAX) as f32;
                (id, distance)
            })
            .collect();

        let by_distance = |a: &(u32, f32), b: &(u32, f32)| a.1.total_cmp(&b.1);
        if n < distances.len() {
            distances.select_nth_unstable_by(n, by_distance);
            distances.truncate(n);
        }
        distances.sort_unstable_by(by_distance);
        distances.into_iter().map(|(id, _)| id).collect()
    }
}

pub fn read_clusters_from_file(cluster_file_path: &str) -> Result<HashMap<u32, Vec<f32>>, Box<dyn Error>> {
    let centroid_file = File::open(Path::new(cluster_file_path))?;
    let reader = BufReader::new(centroid_file);

    let mut map: HashMap<u32, Vec<f32>> = HashMap::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line)?;
        let id = v["cluster_id"].as_u64().ok_or("missing cluster_id")? as u32;
        let centroid: Vec<f32> = serde_json::from_value(v["centroid"].clone())?;
        // Detect duplicates here: the map would otherwise silently keep only the
        // last centroid for a repeated cluster_id.
        if map.insert(id, centroid).is_some() {
            return Err(Box::new(ClusterIndexError::DuplicateClusterId { cluster_id: id }));
        }
    }

    info!("Read {} clusters from file: {}", map.len(), cluster_file_path);

    Ok(map)
}

pub fn find_closest_cluster_id(clusters: &HashMap<u32, Cluster>, embedding: &[f32]) -> u32 {
    let mut closest_cluster_id: u32 = u32::MAX;
    let mut min_distance = f32::MAX;
    for cluster in clusters {
        let distance = cluster.1.euclidean_distance(embedding);
        if distance < min_distance {
            min_distance = distance;
            closest_cluster_id = *cluster.0;
        }
    }

    closest_cluster_id
}

/// Return the IDs of the `n` clusters closest to `embedding`, sorted by ascending distance.
///
/// When `n` exceeds the number of clusters, all cluster IDs are returned.
///
/// For `n < cluster_count` this selects the `n` nearest with
/// [`select_nth_unstable_by`](slice::select_nth_unstable_by) (introselect, ~O(C))
/// and sorts only those `n` (O(n log n)), rather than fully sorting all `C` clusters
/// (O(C log C)). The result is identical — the `n` closest in ascending distance.
///
/// This is the single-query form over a plain cluster map, kept for callers that hold a
/// `HashMap<u32, Cluster>` directly (and as the reference the batched path is tested
/// against). The search path uses
/// [`ClusterIndex::find_top_n_cluster_ids_batch`], which produces the same per-query
/// result but scans a contiguous centroid matrix in parallel across query chunks.
pub fn find_top_n_cluster_ids(clusters: &HashMap<u32, Cluster>, embedding: &[f32], n: usize) -> Vec<u32> {
    let mut distances: Vec<(u32, f32)> = clusters.values().map(|c| (c.cluster_id, c.euclidean_distance(embedding))).collect();
    let by_distance = |a: &(u32, f32), b: &(u32, f32)| a.1.total_cmp(&b.1);

    // Partition the n smallest into [0..n) in linear time, then drop the rest so the
    // final sort only orders n elements. Skipped when n >= len (we return all anyway,
    // and select_nth_unstable_by would panic on an out-of-range index).
    if n < distances.len() {
        distances.select_nth_unstable_by(n, by_distance);
        distances.truncate(n);
    }
    distances.sort_unstable_by(by_distance);
    distances.into_iter().map(|(id, _)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_closest_cluster_id() {
        let cluster1 = Cluster::new(1, vec![0.243, 0.453, 0.7644]);
        let cluster2 = Cluster::new(2, vec![0.564, 0.432, 0.654]);
        let cluster3 = Cluster::new(3, vec![1.23, -0.45, -0.90]);
        let mut clusters = HashMap::new();
        clusters.insert(1, cluster1);
        clusters.insert(2, cluster2);
        clusters.insert(3, cluster3);
        let embedding = vec![1.1, -1.2, 0.98];
        let closest_cluster_id = find_closest_cluster_id(&clusters, &embedding);
        assert_eq!(closest_cluster_id, 2);
    }

    #[test]
    fn test_create_cluster() {
        let cluster = Cluster::new(1, vec![1.0, 2.0, 3.0]);
        assert_eq!(cluster.cluster_id, 1);
        assert_eq!(cluster.centroid, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cluster_euclidean_distance() {
        let cluster = Cluster::new(1, vec![0.0, 0.0, 0.0]);
        let other = vec![1.0, 0.0, 0.0];
        let distance = cluster.euclidean_distance(&other);
        assert_eq!(distance, 1.0);
    }

    #[test]
    fn test_cluster_euclidean_distance_same_point() {
        let cluster = Cluster::new(1, vec![1.0, 2.0, 3.0]);
        let distance = cluster.euclidean_distance(&[1.0, 2.0, 3.0]);
        assert_eq!(distance, 0.0);
    }

    #[test]
    fn test_read_clusters_from_file() {
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, r#"{{"cluster_id":1,"centroid":[0.1,0.2,0.3]}}"#).unwrap();
        writeln!(tmp, r#"{{"cluster_id":2,"centroid":[0.4,0.5,0.6]}}"#).unwrap();
        let clusters = read_clusters_from_file(tmp.path().to_str().unwrap()).unwrap();
        assert_eq!(clusters.len(), 2);
        assert!(clusters.contains_key(&1));
        assert!(clusters.contains_key(&2));
        assert_eq!(clusters[&1], vec![0.1, 0.2, 0.3]);
        assert_eq!(clusters[&2], vec![0.4, 0.5, 0.6]);
    }

    #[test]
    fn test_find_top_n_cluster_ids_returns_n_closest_in_order() {
        // cluster 1: L2sq 0.0, cluster 2: L2sq 0.5, cluster 3: L2sq 2.0 from [1,0,0,0]
        let mut clusters = HashMap::new();
        clusters.insert(1, Cluster::new(1, vec![1.0, 0.0, 0.0, 0.0]));
        clusters.insert(2, Cluster::new(2, vec![0.5, 0.5, 0.5, 0.5]));
        clusters.insert(3, Cluster::new(3, vec![0.0, 1.0, 0.0, 0.0]));
        let ids = find_top_n_cluster_ids(&clusters, &[1.0, 0.0, 0.0, 0.0], 2);
        assert_eq!(ids.len(), 2, "must return exactly 2 ids");
        // Closest is cluster 1 (distance 0), then cluster 2.
        assert_eq!(ids[0], 1, "cluster 1 must be first (distance 0)");
        assert!(ids.contains(&2), "cluster 2 must be in the top-2");
        assert!(!ids.contains(&3), "cluster 3 must be excluded");
    }

    #[test]
    fn test_find_top_n_cluster_ids_returns_all_when_n_exceeds_count() {
        let mut clusters = HashMap::new();
        clusters.insert(1, Cluster::new(1, vec![1.0, 0.0]));
        clusters.insert(2, Cluster::new(2, vec![0.0, 1.0]));
        let ids = find_top_n_cluster_ids(&clusters, &[1.0, 0.0], 100);
        assert_eq!(ids.len(), 2, "must return all clusters when n exceeds cluster count");
    }

    #[test]
    fn test_find_top_n_cluster_ids_n_equals_1_matches_find_closest_cluster_id() {
        let mut clusters = HashMap::new();
        clusters.insert(1, Cluster::new(1, vec![1.0, 0.0, 0.0]));
        clusters.insert(2, Cluster::new(2, vec![0.0, 1.0, 0.0]));
        clusters.insert(3, Cluster::new(3, vec![-1.0, 0.0, 0.0]));
        let embedding = vec![0.8, 0.2, 0.0];
        let top_1 = find_top_n_cluster_ids(&clusters, &embedding, 1);
        let closest = find_closest_cluster_id(&clusters, &embedding);
        assert_eq!(top_1.len(), 1);
        assert_eq!(top_1[0], closest, "top-1 result must match find_closest_cluster_id");
    }

    #[test]
    fn test_find_top_n_cluster_ids_empty_clusters_returns_empty() {
        let clusters = HashMap::new();
        let ids = find_top_n_cluster_ids(&clusters, &[1.0, 0.0], 5);
        assert!(ids.is_empty(), "must return empty vec when cluster map is empty");
    }

    #[test]
    fn test_find_top_n_cluster_ids_n_zero_returns_empty() {
        let mut clusters = HashMap::new();
        clusters.insert(1, Cluster::new(1, vec![1.0, 0.0]));
        clusters.insert(2, Cluster::new(2, vec![0.0, 1.0]));
        assert!(find_top_n_cluster_ids(&clusters, &[1.0, 0.0], 0).is_empty());
    }

    #[test]
    fn test_find_top_n_cluster_ids_matches_full_sort_reference() {
        // The select_nth-based path must produce exactly the same result as a full
        // sort + truncate, across every n from 0..=count and a few beyond. Distances
        // are made distinct (per-dim values) so there is no tie ambiguity.
        let mut clusters = HashMap::new();
        for id in 0..32u32 {
            // Distinct 4-D centroids → distinct distances from the query below.
            let c = vec![id as f32 * 0.1, (id as f32 * 0.03).sin(), (id as f32).cos(), id as f32 * -0.02];
            clusters.insert(id, Cluster::new(id, c));
        }
        let query = vec![0.5f32, -0.2, 0.7, 0.1];

        // Reference: full sort by ascending distance, then take n.
        let reference = |n: usize| -> Vec<u32> {
            let mut d: Vec<(u32, f32)> = clusters.values().map(|c| (c.cluster_id, c.euclidean_distance(&query))).collect();
            d.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            d.truncate(n);
            d.into_iter().map(|(id, _)| id).collect()
        };

        for n in 0..=clusters.len() + 3 {
            assert_eq!(find_top_n_cluster_ids(&clusters, &query, n), reference(n), "mismatch at n={n}");
        }
    }

    // ── Batched coarse assignment (ClusterIndex::find_top_n_cluster_ids_batch) ──

    /// Deterministic XorShift64 → f32 in [-1, 1], for reproducible random vectors.
    fn prng(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as i64 as f32) / (i64::MAX as f32)
        }
    }

    fn random_clusters(count: u32, dim: usize, seed: u64) -> HashMap<u32, Cluster> {
        let mut next = prng(seed);
        (0..count).map(|id| (id, Cluster::new(id, (0..dim).map(|_| next()).collect()))).collect()
    }

    fn random_queries(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut next = prng(seed);
        (0..count).map(|_| (0..dim).map(|_| next()).collect()).collect()
    }

    fn sorted(mut v: Vec<u32>) -> Vec<u32> {
        v.sort_unstable();
        v
    }

    #[test]
    fn batch_matches_serial_free_fn_probe_set() {
        // The batched path must probe exactly the clusters the historical per-query
        // serial path does, for every query chunk and a range of n. Compared as sets:
        // recall depends on *which* clusters are probed, and the caller unions them.
        let clusters = random_clusters(64, 48, 0x1234_5678_9abc_def0);
        let index = ClusterIndex::from_clusters(clusters.clone());
        let queries = random_queries(16, 48, 0x0fed_cba9_8765_4321);

        for &n in &[1usize, 4, 16, 63, 64, 100] {
            let batched = index.find_top_n_cluster_ids_batch(&queries, n);
            assert_eq!(batched.len(), queries.len(), "one result per query, in order");
            for (q, got) in queries.iter().zip(&batched) {
                let expected = find_top_n_cluster_ids(&clusters, q, n);
                assert_eq!(
                    sorted(got.clone()),
                    sorted(expected),
                    "batched probe set diverged from serial free-fn at n={n}"
                );
            }
        }
    }

    #[test]
    fn batch_preserves_query_order_and_ranking() {
        // Two well-separated centroids; each query sits on top of one of them. The
        // result vec must align with input order, and each row must be distance-ranked
        // (nearest centroid first).
        let mut clusters = HashMap::new();
        clusters.insert(10, Cluster::new(10, vec![1.0, 0.0, 0.0]));
        clusters.insert(20, Cluster::new(20, vec![0.0, 1.0, 0.0]));
        let index = ClusterIndex::from_clusters(clusters);

        let queries = vec![vec![0.9, 0.1, 0.0], vec![0.1, 0.9, 0.0]];
        let out = index.find_top_n_cluster_ids_batch(&queries, 2);

        assert_eq!(out[0][0], 10, "query 0 is nearest cluster 10");
        assert_eq!(out[1][0], 20, "query 1 is nearest cluster 20");
        assert_eq!(out[0], vec![10, 20], "row 0 ranked nearest-first");
        assert_eq!(out[1], vec![20, 10], "row 1 ranked nearest-first");
    }

    #[test]
    fn batch_n_exceeds_count_returns_all_per_query() {
        let index = ClusterIndex::from_clusters(random_clusters(5, 8, 42));
        let queries = random_queries(3, 8, 99);
        let out = index.find_top_n_cluster_ids_batch(&queries, 100);
        for row in &out {
            assert_eq!(sorted(row.clone()), vec![0, 1, 2, 3, 4], "n >= count returns every cluster id");
        }
    }

    #[test]
    fn batch_empty_queries_returns_empty() {
        let index = ClusterIndex::from_clusters(random_clusters(4, 8, 7));
        assert!(index.find_top_n_cluster_ids_batch(&[], 2).is_empty());
    }

    #[test]
    fn batch_single_cluster() {
        let mut clusters = HashMap::new();
        clusters.insert(3, Cluster::new(3, vec![0.5, 0.5]));
        let index = ClusterIndex::from_clusters(clusters);
        let out = index.find_top_n_cluster_ids_batch(&[vec![0.0, 0.0], vec![1.0, 1.0]], 4);
        assert_eq!(out, vec![vec![3], vec![3]]);
    }

    #[test]
    fn contiguous_matrix_is_consistent_with_clusters_map() {
        // The matrix is a second copy of the centroids; each row must equal the mapped
        // cluster's centroid, and len()/is_empty() must track the cluster count.
        let clusters = random_clusters(20, 12, 555);
        let index = ClusterIndex::from_clusters(clusters.clone());

        assert_eq!(index.len(), 20);
        assert!(!index.is_empty());
        assert_eq!(index.centroid_ids.len(), 20);
        assert_eq!(index.centroids.len(), 20 * 12);

        for (row, &id) in index.centroid_ids.iter().enumerate() {
            let base = row * index.dim;
            let matrix_row = &index.centroids[base..base + index.dim];
            assert_eq!(
                matrix_row,
                clusters[&id].centroid.as_slice(),
                "matrix row {row} != mapped centroid for id {id}"
            );
        }
    }

    #[test]
    fn empty_index_len_and_is_empty() {
        let index = ClusterIndex::from_clusters(HashMap::new());
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert!(index.find_top_n_cluster_ids_batch(&[vec![1.0, 2.0]], 3)[0].is_empty());
    }

    // ── Centroid validation (ClusterIndexError) ──────────────────────────────

    /// Write the given lines to a temp file and return it (kept alive by caller).
    fn cluster_file(lines: &[&str]) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(tmp, "{line}").unwrap();
        }
        tmp
    }

    #[test]
    fn validate_centroids_accepts_uniform_and_returns_dim() {
        let mut map = HashMap::new();
        map.insert(1, vec![0.1, 0.2, 0.3]);
        map.insert(2, vec![0.4, 0.5, 0.6]);
        assert_eq!(validate_centroids(&map, None).unwrap(), 3);
        assert_eq!(validate_centroids(&map, Some(3)).unwrap(), 3);
    }

    #[test]
    fn validate_centroids_rejects_empty_map() {
        let map: HashMap<u32, Vec<f32>> = HashMap::new();
        assert!(matches!(validate_centroids(&map, None), Err(ClusterIndexError::Empty)));
    }

    #[test]
    fn validate_centroids_rejects_empty_centroid() {
        let mut map = HashMap::new();
        map.insert(7, Vec::<f32>::new());
        assert!(matches!(
            validate_centroids(&map, None),
            Err(ClusterIndexError::EmptyCentroid { cluster_id: 7 })
        ));
    }

    #[test]
    fn validate_centroids_rejects_mixed_dimensions() {
        let mut map = HashMap::new();
        map.insert(1, vec![0.1, 0.2, 0.3]);
        map.insert(2, vec![0.4, 0.5]); // shorter
        assert!(matches!(
            validate_centroids(&map, None),
            Err(ClusterIndexError::InconsistentDimension { .. })
        ));
    }

    #[test]
    fn validate_centroids_rejects_dim_mismatch_against_expected() {
        let mut map = HashMap::new();
        map.insert(1, vec![0.1, 0.2, 0.3]);
        assert!(matches!(
            validate_centroids(&map, Some(768)),
            Err(ClusterIndexError::DimensionMismatch { expected: 768, found: 3 })
        ));
    }

    #[test]
    fn read_clusters_rejects_duplicate_cluster_id() {
        let tmp = cluster_file(&[
            r#"{"cluster_id":1,"centroid":[0.1,0.2,0.3]}"#,
            r#"{"cluster_id":1,"centroid":[0.4,0.5,0.6]}"#,
        ]);
        let err = read_clusters_from_file(tmp.path().to_str().unwrap()).unwrap_err();
        assert!(err.to_string().contains("duplicate cluster_id 1"), "got: {err}");
    }

    #[test]
    fn load_validates_internal_consistency() {
        let tmp = cluster_file(&[
            r#"{"cluster_id":1,"centroid":[0.1,0.2,0.3]}"#,
            r#"{"cluster_id":2,"centroid":[0.4,0.5,0.6]}"#,
        ]);
        let idx = ClusterIndex::load(tmp.path().to_str().unwrap()).unwrap();
        assert_eq!(idx.clusters.len(), 2);
        assert_eq!(idx.dim(), 3);
    }

    #[test]
    fn load_rejects_mixed_dimension_file() {
        let tmp = cluster_file(&[r#"{"cluster_id":1,"centroid":[0.1,0.2,0.3]}"#, r#"{"cluster_id":2,"centroid":[0.4,0.5]}"#]);
        assert!(ClusterIndex::load(tmp.path().to_str().unwrap()).is_err());
    }

    #[test]
    fn load_with_dim_accepts_matching_dim() {
        let tmp = cluster_file(&[r#"{"cluster_id":1,"centroid":[0.1,0.2,0.3]}"#]);
        let idx = ClusterIndex::load_with_dim(tmp.path().to_str().unwrap(), 3).unwrap();
        assert_eq!(idx.dim(), 3);
    }

    #[test]
    fn load_with_dim_rejects_mismatched_dim() {
        let tmp = cluster_file(&[r#"{"cluster_id":1,"centroid":[0.1,0.2,0.3]}"#]);
        let err = ClusterIndex::load_with_dim(tmp.path().to_str().unwrap(), 768).unwrap_err();
        assert!(err.to_string().contains("does not match"), "got: {err}");
    }

    #[test]
    fn load_rejects_empty_file() {
        let tmp = cluster_file(&[]);
        assert!(ClusterIndex::load(tmp.path().to_str().unwrap()).is_err());
    }
}
