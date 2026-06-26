use log::info;
use simsimd::SpatialSimilarity;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Neighbour {
    pub cluster_id: u32,
    pub distance: f32,
}

#[derive(Debug, Clone)]
pub struct Cluster {
    pub cluster_id: u32,
    pub centroid: Vec<f32>,
}

impl Neighbour {
    pub fn new(cluster_id: u32, distance: f32) -> Neighbour {
        Neighbour { cluster_id, distance }
    }

    pub fn sort_by_distance(neighbours: &[Neighbour]) -> Vec<Neighbour> {
        let mut sorted_neighbours = neighbours.to_vec();
        sorted_neighbours.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        sorted_neighbours
    }

    pub fn find_closest_neighbours(neighbours: &[Neighbour], maximum_neighbours: usize) -> Vec<Neighbour> {
        let mut closest_neighbours: Vec<Neighbour> = Vec::with_capacity(maximum_neighbours);

        for neighbour in Neighbour::sort_by_distance(neighbours) {
            if closest_neighbours.len() == maximum_neighbours {
                break;
            }
            closest_neighbours.push(neighbour);
        }

        closest_neighbours
    }
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

/// A fully built cluster index: the centroid map and the pre-computed
/// neighbour graph, loaded once at startup and shared read-only across
/// all requests.
#[derive(Debug)]
pub struct ClusterIndex {
    /// All clusters keyed by their ID.
    pub clusters: HashMap<u32, Cluster>,
    /// Pre-computed nearest-neighbour graph, keyed by cluster ID.
    pub neighbours: HashMap<u32, Vec<Neighbour>>,
    /// The uniform centroid dimension, validated at load time. Every query and
    /// document embedding compared against these centroids must share it.
    dim: usize,
}

impl ClusterIndex {
    /// Load clusters from `cluster_file_path` and build the neighbour graph,
    /// validating only the file's internal consistency.
    ///
    /// Rejects an empty file, empty centroids, duplicate cluster IDs, and
    /// centroids of mixed dimension. Use [`load_with_dim`](Self::load_with_dim) to
    /// additionally pin the centroid dimension to a configured `embedding_dim`.
    ///
    /// Returns an error if the file cannot be read, parsed, or fails validation.
    pub fn load(cluster_file_path: &str, max_neighbours: usize) -> Result<Self, Box<dyn Error>> {
        Self::load_inner(cluster_file_path, max_neighbours, None)
    }

    /// Like [`load`](Self::load), but also rejects a cluster file whose (uniform)
    /// centroid dimension does not equal `embedding_dim`.
    ///
    /// This is the entry point the server uses: catching a centroid/embedding
    /// dimension mismatch here disables semantic search with a clear error at
    /// startup, instead of letting it surface as a `u32::MAX` nearest-cluster id
    /// and a downstream `unwrap` panic on the first document insert or query.
    pub fn load_with_dim(cluster_file_path: &str, max_neighbours: usize, embedding_dim: usize) -> Result<Self, Box<dyn Error>> {
        Self::load_inner(cluster_file_path, max_neighbours, Some(embedding_dim))
    }

    fn load_inner(cluster_file_path: &str, max_neighbours: usize, expected_dim: Option<usize>) -> Result<Self, Box<dyn Error>> {
        let centroid_map = read_clusters_from_file(cluster_file_path)?;
        let dim = validate_centroids(&centroid_map, expected_dim)?;
        let clusters: HashMap<u32, Cluster> = centroid_map.into_iter().map(|(id, centroid)| (id, Cluster::new(id, centroid))).collect();
        let neighbours = build_neighbours_graph(&clusters, max_neighbours);
        Ok(Self { clusters, neighbours, dim })
    }

    /// Build an index directly from in-memory clusters, computing the neighbour
    /// graph and inferring the dimension from the first centroid (`0` if empty).
    ///
    /// For callers that already hold centroids in memory rather than a file. The
    /// per-centroid validation done by [`load`](Self::load) is the file path's
    /// concern; in-memory callers are trusted to pass uniform centroids.
    pub fn from_clusters(clusters: HashMap<u32, Cluster>, max_neighbours: usize) -> Self {
        let dim = clusters.values().next().map(|c| c.centroid.len()).unwrap_or(0);
        let neighbours = build_neighbours_graph(&clusters, max_neighbours);
        Self { clusters, neighbours, dim }
    }

    /// The uniform centroid dimension every embedding must match.
    pub fn dim(&self) -> usize {
        self.dim
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

pub fn build_neighbours_graph(clusters: &HashMap<u32, Cluster>, max_neighbours: usize) -> HashMap<u32, Vec<Neighbour>> {
    let mut neighbours_map: HashMap<u32, Vec<Neighbour>> = HashMap::new();
    for cluster in clusters {
        let mut neighbours: Vec<Neighbour> = Vec::with_capacity(max_neighbours);
        for c in clusters {
            if cluster.1.cluster_id != c.1.cluster_id {
                let distance = f32::l2sq(&cluster.1.centroid, &c.1.centroid).unwrap_or(f64::MAX) as f32;
                neighbours.push(Neighbour::new(c.1.cluster_id, distance));
            }
        }

        let capped_neighbours = Neighbour::find_closest_neighbours(&neighbours, max_neighbours);
        neighbours_map.insert(cluster.1.cluster_id, capped_neighbours);
    }

    neighbours_map
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
/// (O(C log C)). The result is identical — the `n` closest in ascending distance —
/// but the cost stays close to linear in the cluster count as the cluster file grows
/// (it is called once per query chunk).
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
    fn test_build_neighbours_graph() {
        let cluster1 = Cluster::new(1, vec![-1.2, 2.23, 3.12]);
        let cluster2 = Cluster::new(2, vec![4.23, -2.98, 3.12]);
        let cluster3 = Cluster::new(3, vec![-7.0, 8.1, 9.2]);
        let mut clusters: HashMap<u32, Cluster> = HashMap::new();
        clusters.insert(1, cluster1);
        clusters.insert(2, cluster2);
        clusters.insert(3, cluster3);

        let neighbours_map: HashMap<u32, Vec<Neighbour>> = build_neighbours_graph(&clusters, 2);
        assert_eq!(neighbours_map.len(), 3);

        for (cluster_id, neighbours) in neighbours_map.iter() {
            assert_eq!(neighbours.len(), 2);
            neighbours.iter().for_each(|neighbour| assert_ne!(neighbour.cluster_id, *cluster_id));
            neighbours.iter().for_each(|neighbour| assert_ne!(neighbour.distance, 0.0));

            let neighbour1 = neighbours.first().unwrap();
            let neighbour2 = neighbours.get(1).unwrap();

            assert!(neighbour1.distance < neighbour2.distance);
        }
    }

    #[test]
    fn test_neighbour_sort_by_distance() {
        let neighbour1 = Neighbour::new(1, 0.1);
        let neighbour2 = Neighbour::new(2, 0.2);
        let neighbour3 = Neighbour::new(3, 0.3);
        let neighbour4 = Neighbour::new(4, 0.4);
        let neighbour5 = Neighbour::new(5, 0.5);
        let mut neighbours = vec![neighbour5, neighbour4, neighbour3, neighbour2, neighbour1];
        neighbours = Neighbour::sort_by_distance(&neighbours);
        assert_eq!(neighbours.len(), 5);
        assert_eq!(neighbours.first().unwrap().distance, 0.1);
        assert_eq!(neighbours.get(1).unwrap().distance, 0.2);
        assert_eq!(neighbours.get(2).unwrap().distance, 0.3);
        assert_eq!(neighbours.get(3).unwrap().distance, 0.4);
        assert_eq!(neighbours.get(4).unwrap().distance, 0.5);
    }

    #[test]
    fn test_find_closest_neighbours() {
        let neighbour1 = Neighbour::new(1, 0.1);
        let neighbour2 = Neighbour::new(2, 0.2);
        let neighbour3 = Neighbour::new(3, 0.3);
        let neighbour4 = Neighbour::new(4, 0.4);
        let neighbour5 = Neighbour::new(5, 0.5);
        let neighbours = vec![neighbour5, neighbour4, neighbour3, neighbour2, neighbour1];
        let closest_neighbours = Neighbour::find_closest_neighbours(&neighbours, 3);
        assert_eq!(closest_neighbours.len(), 3);
        assert_eq!(closest_neighbours.first().unwrap().cluster_id, 1);
        assert_eq!(closest_neighbours.get(1).unwrap().cluster_id, 2);
        assert_eq!(closest_neighbours.get(2).unwrap().cluster_id, 3);
    }

    #[test]
    fn test_create_neighbour() {
        let neighbour = Neighbour::new(42, 0.75);
        assert_eq!(neighbour.cluster_id, 42);
        assert_eq!(neighbour.distance, 0.75);
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
    fn test_find_closest_neighbours_fewer_than_max() {
        let neighbours = vec![Neighbour::new(1, 0.5), Neighbour::new(2, 0.2)];
        let closest = Neighbour::find_closest_neighbours(&neighbours, 10);
        assert_eq!(closest.len(), 2);
        assert_eq!(closest.first().unwrap().cluster_id, 2);
        assert_eq!(closest.get(1).unwrap().cluster_id, 1);
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
        let idx = ClusterIndex::load(tmp.path().to_str().unwrap(), 1).unwrap();
        assert_eq!(idx.clusters.len(), 2);
        assert_eq!(idx.dim(), 3);
    }

    #[test]
    fn load_rejects_mixed_dimension_file() {
        let tmp = cluster_file(&[r#"{"cluster_id":1,"centroid":[0.1,0.2,0.3]}"#, r#"{"cluster_id":2,"centroid":[0.4,0.5]}"#]);
        assert!(ClusterIndex::load(tmp.path().to_str().unwrap(), 1).is_err());
    }

    #[test]
    fn load_with_dim_accepts_matching_dim() {
        let tmp = cluster_file(&[r#"{"cluster_id":1,"centroid":[0.1,0.2,0.3]}"#]);
        let idx = ClusterIndex::load_with_dim(tmp.path().to_str().unwrap(), 1, 3).unwrap();
        assert_eq!(idx.dim(), 3);
    }

    #[test]
    fn load_with_dim_rejects_mismatched_dim() {
        let tmp = cluster_file(&[r#"{"cluster_id":1,"centroid":[0.1,0.2,0.3]}"#]);
        let err = ClusterIndex::load_with_dim(tmp.path().to_str().unwrap(), 1, 768).unwrap_err();
        assert!(err.to_string().contains("does not match"), "got: {err}");
    }

    #[test]
    fn load_rejects_empty_file() {
        let tmp = cluster_file(&[]);
        assert!(ClusterIndex::load(tmp.path().to_str().unwrap(), 1).is_err());
    }
}
