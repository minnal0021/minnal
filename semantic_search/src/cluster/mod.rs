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
        f32::l2sq(&self.centroid, other).unwrap_or(f64::MAX) as f32
    }

    pub fn new(cluster_id: u32, centroid: Vec<f32>) -> Cluster {
        Cluster { cluster_id, centroid }
    }
}

/// A fully built cluster index: the centroid map and the pre-computed
/// neighbour graph, loaded once at startup and shared read-only across
/// all requests.
pub struct ClusterIndex {
    /// All clusters keyed by their ID.
    pub clusters: HashMap<u32, Cluster>,
    /// Pre-computed nearest-neighbour graph, keyed by cluster ID.
    pub neighbours: HashMap<u32, Vec<Neighbour>>,
}

impl ClusterIndex {
    /// Load clusters from `cluster_file_path` and build the neighbour graph.
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn load(cluster_file_path: &str, max_neighbours: usize) -> Result<Self, Box<dyn Error>> {
        let centroid_map = read_clusters_from_file(cluster_file_path)?;
        let clusters: HashMap<u32, Cluster> = centroid_map.into_iter().map(|(id, centroid)| (id, Cluster::new(id, centroid))).collect();
        let neighbours = build_neighbours_graph(&clusters, max_neighbours);
        Ok(Self { clusters, neighbours })
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
        map.insert(id, centroid);
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
pub fn find_top_n_cluster_ids(clusters: &HashMap<u32, Cluster>, embedding: &[f32], n: usize) -> Vec<u32> {
    let mut distances: Vec<(u32, f32)> = clusters.values().map(|c| (c.cluster_id, c.euclidean_distance(embedding))).collect();
    distances.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    distances.truncate(n);
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
}
