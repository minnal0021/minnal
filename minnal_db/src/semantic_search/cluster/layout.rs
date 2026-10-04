//! How vectors are partitioned and encoded: the [`IvfLayout`] a search and the
//! indexer work against (design doc M2b).
//!
//! Two roles that one centroid file used to play at once are kept apart:
//!
//! - **Routing.** A *posting* is the set of chunk entries stored under one key
//!   prefix. Each has a routing centroid: a vector is filed under the nearest one,
//!   and a query probes the nearest few. Routing may change (M3 splits postings).
//! - **Encoding.** A *centre* is what a code is encoded against (`x − c`). A code
//!   records its `centre_id` and is always decoded through it, never through the
//!   posting it sits in, so a centre must never change once a code points at it.
//!
//! [`ClusterIndex`] is the identity layout (posting = centre = cluster, default
//! seed), which tests, benches and raw `vector_kv` users build from a centroid
//! file. [`NamespaceIvf`] is a namespace's own: its stored postings and centres,
//! rotated with the seed its schema records.

use std::collections::HashMap;
use std::error::Error;

use super::{Cluster, ClusterIndex, DEFAULT_ROTATION_SEED, find_closest_cluster_id};

/// The centre whole-document (Pass-2) codes are encoded against: the origin.
/// Reserved: no stored centre may use this id. A zero centre makes Pass 2
/// independent of the partition, so postings and centres can change under it
/// (design doc M2c).
pub const ZERO_CENTRE: u32 = u32::MAX;

/// What indexing and search need to know about a partition.
pub trait IvfLayout: Sync {
    /// The embedding dimension.
    fn dim(&self) -> usize;
    /// `v` in the space codes are computed in (`Pᵀv`).
    fn rotate(&self, v: &[f32]) -> Vec<f32>;
    /// The posting a vector is filed under (nearest routing centroid); `None`
    /// when there are no postings.
    fn route(&self, embedding: &[f32]) -> Option<u32>;
    /// For each query vector, the `n` postings nearest it, nearest first.
    fn probe(&self, queries: &[Vec<f32>], n: usize) -> Vec<Vec<u32>>;
    /// The centre a code written to `posting` is encoded against.
    fn centre_of(&self, posting: u32) -> Option<u32>;
    /// A centre in the original space (for `⟨q, c⟩`).
    fn centre(&self, centre_id: u32) -> Option<&[f32]>;
    /// A centre in the space codes are computed in (`Pᵀc`).
    fn rotated_centre(&self, centre_id: u32) -> Option<&[f32]>;
    /// Every centre id.
    fn centre_ids(&self) -> Vec<u32>;
}

impl IvfLayout for ClusterIndex {
    fn dim(&self) -> usize {
        ClusterIndex::dim(self)
    }
    fn rotate(&self, v: &[f32]) -> Vec<f32> {
        ClusterIndex::rotate(self, v)
    }
    fn route(&self, embedding: &[f32]) -> Option<u32> {
        (!self.clusters.is_empty()).then(|| find_closest_cluster_id(&self.clusters, embedding))
    }
    fn probe(&self, queries: &[Vec<f32>], n: usize) -> Vec<Vec<u32>> {
        self.find_top_n_cluster_ids_batch(queries, n)
    }
    fn centre_of(&self, posting: u32) -> Option<u32> {
        self.clusters.contains_key(&posting).then_some(posting)
    }
    fn centre(&self, centre_id: u32) -> Option<&[f32]> {
        self.clusters.get(&centre_id).map(|c| c.centroid.as_slice())
    }
    fn rotated_centre(&self, centre_id: u32) -> Option<&[f32]> {
        self.rotated_centroid(centre_id)
    }
    fn centre_ids(&self) -> Vec<u32> {
        self.clusters.keys().copied().collect()
    }
}

impl<T: IvfLayout + Send + ?Sized> IvfLayout for std::sync::Arc<T> {
    fn dim(&self) -> usize {
        (**self).dim()
    }
    fn rotate(&self, v: &[f32]) -> Vec<f32> {
        (**self).rotate(v)
    }
    fn route(&self, embedding: &[f32]) -> Option<u32> {
        (**self).route(embedding)
    }
    fn probe(&self, queries: &[Vec<f32>], n: usize) -> Vec<Vec<u32>> {
        (**self).probe(queries, n)
    }
    fn centre_of(&self, posting: u32) -> Option<u32> {
        (**self).centre_of(posting)
    }
    fn centre(&self, centre_id: u32) -> Option<&[f32]> {
        (**self).centre(centre_id)
    }
    fn rotated_centre(&self, centre_id: u32) -> Option<&[f32]> {
        (**self).rotated_centre(centre_id)
    }
    fn centre_ids(&self) -> Vec<u32> {
        (**self).centre_ids()
    }
}

/// One posting as a namespace stores it.
#[derive(Debug, Clone, PartialEq)]
pub struct Posting {
    /// Key prefix of the posting's chunk entries.
    pub posting_id: u32,
    /// What vectors are routed and queries probed by.
    pub routing_centroid: Vec<f32>,
    /// The centre new codes in this posting are encoded against.
    pub centre_id: u32,
}

/// A namespace's own partition: its postings, its centres, and its rotation.
#[derive(Debug)]
pub struct NamespaceIvf {
    /// Routing centroids keyed by posting id (its rotation is unused).
    routing: ClusterIndex,
    /// The centre each posting encodes new codes against.
    posting_centre: HashMap<u32, u32>,
    /// Centres keyed by centre id, rotated with the namespace's seed.
    centres: ClusterIndex,
}

impl NamespaceIvf {
    /// Build from stored postings and centres. Fails if the two are empty, a
    /// posting names a centre that does not exist, or dimensions differ.
    pub fn new(postings: Vec<Posting>, centres: HashMap<u32, Vec<f32>>, seed: u64) -> Result<Self, String> {
        if postings.is_empty() || centres.is_empty() {
            return Err("a namespace index needs at least one posting and one centre".into());
        }
        let dim = centres.values().next().map(Vec::len).unwrap_or(0);
        if centres.contains_key(&ZERO_CENTRE) {
            return Err(format!("centre id {ZERO_CENTRE} is reserved for the zero centre"));
        }
        if let Some((id, c)) = centres.iter().find(|(_, c)| c.len() != dim) {
            return Err(format!("centre {id} has {} dimensions, expected {dim}", c.len()));
        }
        let mut posting_centre = HashMap::with_capacity(postings.len());
        let mut routing = HashMap::with_capacity(postings.len());
        for p in postings {
            if !centres.contains_key(&p.centre_id) {
                return Err(format!("posting {} names centre {}, which does not exist", p.posting_id, p.centre_id));
            }
            if p.routing_centroid.len() != dim {
                return Err(format!(
                    "posting {} has a {}-dimensional routing centroid, expected {dim}",
                    p.posting_id,
                    p.routing_centroid.len()
                ));
            }
            posting_centre.insert(p.posting_id, p.centre_id);
            routing.insert(p.posting_id, Cluster::new(p.posting_id, p.routing_centroid));
        }
        Ok(Self {
            routing: ClusterIndex::from_clusters_with_seed(routing, seed),
            posting_centre,
            centres: ClusterIndex::from_clusters_with_seed(centres.into_iter().map(|(id, c)| (id, Cluster::new(id, c))).collect(), seed),
        })
    }

    /// The layout a centroid set seeds: one posting per centroid, each encoding
    /// against itself (posting id = centre id = cluster id).
    pub fn seeded(centroids: &HashMap<u32, Vec<f32>>, seed: u64) -> Result<Self, String> {
        let postings = centroids
            .iter()
            .map(|(&id, c)| Posting {
                posting_id: id,
                routing_centroid: c.clone(),
                centre_id: id,
            })
            .collect();
        Self::new(postings, centroids.clone(), seed)
    }

    /// [`seeded`](Self::seeded) from a centroid file, with the default seed: what
    /// benches and evaluations index against.
    pub fn from_cluster_file(path: &str) -> Result<Self, Box<dyn Error>> {
        let centroids = super::read_clusters_from_file(path)?;
        Ok(Self::seeded(&centroids, DEFAULT_ROTATION_SEED)?)
    }

    /// The rotation seed (`None` for dimensions too small to rotate).
    pub fn rotation_seed(&self) -> Option<u64> {
        self.centres.rotation_seed()
    }

    /// Number of postings.
    pub fn postings(&self) -> usize {
        self.posting_centre.len()
    }

    /// Number of centres.
    pub fn centres(&self) -> usize {
        self.centres.len()
    }
}

impl IvfLayout for NamespaceIvf {
    fn dim(&self) -> usize {
        self.centres.dim()
    }
    fn rotate(&self, v: &[f32]) -> Vec<f32> {
        self.centres.rotate(v)
    }
    fn route(&self, embedding: &[f32]) -> Option<u32> {
        IvfLayout::route(&self.routing, embedding)
    }
    fn probe(&self, queries: &[Vec<f32>], n: usize) -> Vec<Vec<u32>> {
        self.routing.find_top_n_cluster_ids_batch(queries, n)
    }
    fn centre_of(&self, posting: u32) -> Option<u32> {
        self.posting_centre.get(&posting).copied()
    }
    fn centre(&self, centre_id: u32) -> Option<&[f32]> {
        IvfLayout::centre(&self.centres, centre_id)
    }
    fn rotated_centre(&self, centre_id: u32) -> Option<&[f32]> {
        self.centres.rotated_centroid(centre_id)
    }
    fn centre_ids(&self) -> Vec<u32> {
        self.centres.clusters.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn centroids(dim: usize) -> HashMap<u32, Vec<f32>> {
        (0..4u32)
            .map(|id| {
                let mut v = vec![0.0f32; dim];
                v[id as usize % dim] = 1.0;
                (id, v)
            })
            .collect()
    }

    #[test]
    fn seeded_layout_matches_the_identity_layout() {
        let c = centroids(16);
        let ivf = NamespaceIvf::seeded(&c, DEFAULT_ROTATION_SEED).unwrap();
        let identity = ClusterIndex::from_clusters(c.iter().map(|(&id, v)| (id, Cluster::new(id, v.clone()))).collect());
        // Distinct distances to every centroid, so no tie is broken by map order.
        let q: Vec<f32> = (0..16).map(|i| 1.0 / (i as f32 + 1.0)).collect();
        assert_eq!(IvfLayout::route(&ivf, &q), IvfLayout::route(&identity, &q));
        assert_eq!(ivf.probe(std::slice::from_ref(&q), 3), identity.probe(std::slice::from_ref(&q), 3));
        assert_eq!(ivf.rotate(&q), identity.rotate(&q));
        for id in 0..4 {
            assert_eq!(ivf.centre_of(id), Some(id));
            assert_eq!(IvfLayout::rotated_centre(&ivf, id), identity.rotated_centroid(id));
        }
    }

    #[test]
    fn the_seed_picks_the_rotation() {
        let c = centroids(16);
        let a = NamespaceIvf::seeded(&c, 1).unwrap();
        let b = NamespaceIvf::seeded(&c, 2).unwrap();
        let v = vec![0.5f32; 16];
        assert_ne!(a.rotate(&v), b.rotate(&v));
        assert_eq!(a.rotation_seed(), Some(1));
    }

    #[test]
    fn routing_and_centres_are_separate() {
        let c = centroids(16);
        let postings = vec![
            Posting {
                posting_id: 10,
                routing_centroid: c[&0].clone(),
                centre_id: 0,
            },
            Posting {
                posting_id: 11,
                routing_centroid: c[&1].clone(),
                centre_id: 0,
            },
        ];
        let ivf = NamespaceIvf::new(postings, c.clone(), DEFAULT_ROTATION_SEED).unwrap();
        assert_eq!(IvfLayout::route(&ivf, &c[&1]), Some(11));
        assert_eq!(ivf.centre_of(11), Some(0));
        assert_eq!(ivf.centre_of(12), None);
        assert_eq!((ivf.postings(), ivf.centres()), (2, 4));
    }

    #[test]
    fn inconsistent_parts_are_rejected() {
        let c = centroids(16);
        let bad = vec![Posting {
            posting_id: 1,
            routing_centroid: c[&0].clone(),
            centre_id: 99,
        }];
        assert!(NamespaceIvf::new(bad, c.clone(), 1).unwrap_err().contains("centre 99"));
        assert!(NamespaceIvf::new(vec![], c.clone(), 1).is_err());
        let mut reserved = c.clone();
        reserved.insert(ZERO_CENTRE, vec![0.0; 16]);
        let posting = vec![Posting {
            posting_id: 1,
            routing_centroid: c[&0].clone(),
            centre_id: 0,
        }];
        assert!(NamespaceIvf::new(posting, reserved, 1).unwrap_err().contains("reserved"));
    }
}
