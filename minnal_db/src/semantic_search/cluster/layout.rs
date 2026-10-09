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
use std::sync::atomic::{AtomicU64, Ordering};

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
    /// How many entries `posting` holds, when the layout knows. Probing by entry
    /// budget ([`select_probes`]) counts an unknown size as 0.
    fn posting_entries(&self, _posting: u32) -> Option<u64> {
        None
    }
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
    fn posting_entries(&self, posting: u32) -> Option<u64> {
        (**self).posting_entries(posting)
    }
}

/// How many postings a search probes (design doc M2d).
///
/// Each query vector walks its postings nearest first and stops once it has
/// probed at least `min_probes` **and** their entries add up to
/// `budget_entries`, and never goes past `max_probes`. The posting that crosses
/// the budget is probed. The cost of Pass 1 is the entries it reads, so a budget
/// keeps that cost fixed as postings are split and change size, where a fixed
/// probe count would not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbeSettings {
    /// Entries to read per query vector before stopping.
    pub budget_entries: u64,
    /// Postings always probed per query vector, whatever the budget.
    pub min_probes: usize,
    /// Postings never exceeded per query vector, whatever the budget.
    pub max_probes: usize,
}

impl ProbeSettings {
    /// Exactly `n` postings per query vector, whatever their size.
    pub const fn fixed(n: usize) -> Self {
        Self {
            budget_entries: 0,
            min_probes: n,
            max_probes: n,
        }
    }
}

/// The postings a search probes under `probe`: per query vector, the nearest
/// postings within its limits ([`ProbeSettings`]); then their union, in
/// first-seen order. Also returns the entries the union holds, as far as the
/// layout knows.
pub fn select_probes<L: IvfLayout + ?Sized>(layout: &L, queries: &[Vec<f32>], probe: &ProbeSettings) -> (Vec<u32>, u64) {
    let mut seen = std::collections::HashSet::new();
    let mut ids = Vec::new();
    let mut planned = 0u64;
    for ranked in layout.probe(queries, probe.max_probes) {
        let mut entries = 0u64;
        for (taken, id) in ranked.into_iter().enumerate() {
            if taken >= probe.min_probes && entries >= probe.budget_entries {
                break;
            }
            let size = layout.posting_entries(id).unwrap_or(0);
            entries += size;
            if seen.insert(id) {
                ids.push(id);
                planned += size;
            }
        }
    }
    (ids, planned)
}

/// A layout with entry counts attached: for layouts that keep none
/// ([`ClusterIndex`]), so they can be probed by budget. Benches and tests use it
/// with counts taken from the store.
pub struct WithEntryCounts<L> {
    /// The layout.
    pub layout: L,
    /// Entries per posting id.
    pub entries: HashMap<u32, u64>,
}

impl<L: IvfLayout> IvfLayout for WithEntryCounts<L> {
    fn dim(&self) -> usize {
        self.layout.dim()
    }
    fn rotate(&self, v: &[f32]) -> Vec<f32> {
        self.layout.rotate(v)
    }
    fn route(&self, embedding: &[f32]) -> Option<u32> {
        self.layout.route(embedding)
    }
    fn probe(&self, queries: &[Vec<f32>], n: usize) -> Vec<Vec<u32>> {
        self.layout.probe(queries, n)
    }
    fn centre_of(&self, posting: u32) -> Option<u32> {
        self.layout.centre_of(posting)
    }
    fn centre(&self, centre_id: u32) -> Option<&[f32]> {
        self.layout.centre(centre_id)
    }
    fn rotated_centre(&self, centre_id: u32) -> Option<&[f32]> {
        self.layout.rotated_centre(centre_id)
    }
    fn centre_ids(&self) -> Vec<u32> {
        self.layout.centre_ids()
    }
    fn posting_entries(&self, posting: u32) -> Option<u64> {
        self.entries.get(&posting).copied()
    }
}

/// How one document's write changed posting sizes: a posting gains an entry
/// for each id in `added` and loses one for each id in `removed`. Returned by
/// the vector writes in `vector_kv` and applied with
/// [`NamespaceIvf::apply_delta`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PostingDelta {
    /// Postings that gained this document's entry.
    pub added: Vec<u32>,
    /// Postings that lost this document's entry.
    pub removed: Vec<u32>,
}

impl PostingDelta {
    /// Add `other`'s changes to these.
    pub fn extend(&mut self, other: PostingDelta) {
        self.added.extend(other.added);
        self.removed.extend(other.removed);
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
    /// Entries per posting, for probing by budget. **An estimate:** counted
    /// from the store when the partition is loaded, then kept up to date by the
    /// vector worker's [`PostingDelta`]s. A write racing the load, or no-WAL
    /// entries lost in a crash, can leave it slightly off until the next load
    /// recounts. It steers how much a search reads, never what is correct.
    entries: HashMap<u32, AtomicU64>,
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
        let entries = posting_centre.keys().map(|&id| (id, AtomicU64::new(0))).collect();
        Ok(Self {
            routing: ClusterIndex::from_clusters_with_seed(routing, seed),
            entries,
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

    /// Set the entry counts, typically from a count of the stored keys.
    /// Postings not named are set to 0; ids that are not postings are ignored.
    pub fn set_entry_counts(&self, counts: &HashMap<u32, u64>) {
        for (id, n) in &self.entries {
            n.store(counts.get(id).copied().unwrap_or(0), Ordering::Relaxed);
        }
    }

    /// Apply one document's [`PostingDelta`]. A count never goes below 0.
    pub fn apply_delta(&self, delta: &PostingDelta) {
        for id in &delta.added {
            if let Some(n) = self.entries.get(id) {
                n.fetch_add(1, Ordering::Relaxed);
            }
        }
        for id in &delta.removed {
            if let Some(n) = self.entries.get(id) {
                let _ = n.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| Some(v.saturating_sub(1)));
            }
        }
    }

    /// Entries across every posting (an estimate; see the field's notes).
    pub fn total_entries(&self) -> u64 {
        self.entries.values().map(|n| n.load(Ordering::Relaxed)).sum()
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
    fn posting_entries(&self, posting: u32) -> Option<u64> {
        self.entries.get(&posting).map(|n| n.load(Ordering::Relaxed))
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

    /// Four postings on the axes; a query nearest posting 0, then 1, 2, 3.
    fn ranked_layout(sizes: [u64; 4]) -> (WithEntryCounts<ClusterIndex>, Vec<f32>) {
        let c = centroids(16);
        let layout = ClusterIndex::from_clusters(c.iter().map(|(&id, v)| (id, Cluster::new(id, v.clone()))).collect());
        let mut q = vec![0.0f32; 16];
        q[..4].copy_from_slice(&[0.8, 0.4, 0.2, 0.1]);
        let entries = (0..4u32).map(|id| (id, sizes[id as usize])).collect();
        (WithEntryCounts { layout, entries }, q)
    }

    fn budget(budget_entries: u64, min_probes: usize, max_probes: usize) -> ProbeSettings {
        ProbeSettings {
            budget_entries,
            min_probes,
            max_probes,
        }
    }

    #[test]
    fn a_budget_stops_at_the_posting_that_reaches_it() {
        let (layout, q) = ranked_layout([10, 20, 30, 40]);
        let qs = [q];
        assert_eq!(select_probes(&layout, &qs, &budget(25, 1, 4)), (vec![0, 1], 30));
        assert_eq!(select_probes(&layout, &qs, &budget(30, 1, 4)), (vec![0, 1], 30));
        assert_eq!(select_probes(&layout, &qs, &budget(31, 1, 4)), (vec![0, 1, 2], 60));
        assert_eq!(select_probes(&layout, &qs, &budget(1, 1, 4)), (vec![0], 10));
        assert_eq!(select_probes(&layout, &qs, &budget(1_000, 1, 4)), (vec![0, 1, 2, 3], 100));
    }

    #[test]
    fn min_and_max_probes_bound_the_budget() {
        let (layout, q) = ranked_layout([10, 20, 30, 40]);
        let qs = [q];
        assert_eq!(select_probes(&layout, &qs, &budget(1, 3, 4)).0, vec![0, 1, 2]);
        assert_eq!(select_probes(&layout, &qs, &budget(1_000, 1, 2)).0, vec![0, 1]);
    }

    #[test]
    fn a_fixed_count_ignores_sizes_and_matches_the_plain_probe() {
        let (layout, q) = ranked_layout([10, 20, 30, 40]);
        let qs = [q];
        for n in 1..=4 {
            assert_eq!(select_probes(&layout, &qs, &ProbeSettings::fixed(n)).0, layout.probe(&qs, n)[0]);
        }
    }

    #[test]
    fn unknown_sizes_count_as_zero() {
        let (mut layout, q) = ranked_layout([10, 20, 30, 40]);
        layout.entries.clear();
        assert_eq!(select_probes(&layout, &[q], &budget(5, 1, 3)), (vec![0, 1, 2], 0));
    }

    #[test]
    fn each_query_vector_has_its_own_budget_and_the_union_is_probed() {
        let (layout, q) = ranked_layout([10, 20, 30, 40]);
        let mut q2 = vec![0.0f32; 16];
        q2[..4].copy_from_slice(&[0.1, 0.2, 0.4, 0.8]); // nearest 3, then 2
        // q: {0, 1}; q2: {3}; posting 1 is not counted twice.
        assert_eq!(select_probes(&layout, &[q.clone(), q2.clone()], &budget(25, 1, 4)), (vec![0, 1, 3], 70));
        assert_eq!(select_probes(&layout, &[q, q2], &budget(25, 2, 4)), (vec![0, 1, 3, 2], 100));
    }

    #[test]
    fn deltas_keep_entry_counts_and_never_go_below_zero() {
        let ivf = NamespaceIvf::seeded(&centroids(16), 1).unwrap();
        assert_eq!(ivf.posting_entries(0), Some(0));
        ivf.set_entry_counts(&HashMap::from([(0, 5), (1, 2), (99, 7)]));
        assert_eq!(
            (ivf.posting_entries(0), ivf.posting_entries(2), ivf.posting_entries(99)),
            (Some(5), Some(0), None)
        );
        ivf.apply_delta(&PostingDelta {
            added: vec![2, 99],
            removed: vec![0, 3],
        });
        assert_eq!([0, 1, 2, 3].map(|id| ivf.posting_entries(id).unwrap()), [4, 2, 1, 0]);
        assert_eq!(ivf.total_entries(), 7);
    }
}
