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
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{Cluster, ClusterIndex, find_closest_cluster_id};
use crate::semantic_search::quantisation::rotation::FhtKacRotator;

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
    /// The largest centre id (`None` without centres). Lets a search size a
    /// table by id without listing every centre.
    fn max_centre_id(&self) -> Option<u32> {
        self.centre_ids().into_iter().max()
    }
    /// How many entries `posting` holds, when the layout knows.
    /// [`select_probes`] counts an unknown size as 0.
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
    fn max_centre_id(&self) -> Option<u32> {
        (**self).max_centre_id()
    }
    fn posting_entries(&self, posting: u32) -> Option<u64> {
        (**self).posting_entries(posting)
    }
}

/// How many postings (clusters) a search probes.
///
/// Each query vector probes its `max_probes` nearest postings, or every active
/// posting when the namespace has fewer, and never fewer than `min_probes`
/// (a floor for overrides that lower `max_probes`). Postings are kept between
/// the merge and split limits, so the count bounds the cost of Pass 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbeSettings {
    /// Postings always probed per query vector (when the namespace has them).
    pub min_probes: usize,
    /// Postings probed per query vector at most.
    pub max_probes: usize,
}

impl ProbeSettings {
    /// Exactly `n` postings per query vector.
    pub const fn fixed(n: usize) -> Self {
        Self {
            min_probes: n,
            max_probes: n,
        }
    }

    /// Postings each query vector probes, before the layout caps it at its
    /// active postings.
    pub fn per_query(&self) -> usize {
        self.max_probes.max(self.min_probes)
    }
}

/// The postings a search probes under `probe`: per query vector, its nearest
/// postings ([`ProbeSettings`]); then their union, in first-seen order. Also
/// returns the entries the union holds, as far as the layout knows.
pub fn select_probes<L: IvfLayout + ?Sized>(layout: &L, queries: &[Vec<f32>], probe: &ProbeSettings) -> (Vec<u32>, u64) {
    let mut seen = std::collections::HashSet::new();
    let mut ids = Vec::new();
    let mut planned = 0u64;
    for ranked in layout.probe(queries, probe.per_query()) {
        for id in ranked {
            if seen.insert(id) {
                ids.push(id);
                planned += layout.posting_entries(id).unwrap_or(0);
            }
        }
    }
    (ids, planned)
}

/// A layout with entry counts attached: for layouts that keep none
/// ([`ClusterIndex`]), so a search can report the entries it plans to read.
/// Benches and tests use it with counts taken from the store.
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
    fn max_centre_id(&self) -> Option<u32> {
        self.layout.max_centre_id()
    }
    fn posting_entries(&self, posting: u32) -> Option<u64> {
        self.entries.get(&posting).copied()
    }
}

/// How vector writes changed posting sizes. Each element is one entry (one
/// document's chunks in one posting) with its chunk count: an entry in `added`
/// was written, one in `removed` was deleted. A document whose entry in a
/// posting is rewritten appears in both, so its chunk count is exact. Returned by
/// the vector writes in `vector_kv` and applied with [`CountTable::apply`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PostingDelta {
    /// `(posting, chunks)` of each entry written.
    pub added: Vec<(u32, u32)>,
    /// `(posting, chunks)` of each entry deleted.
    pub removed: Vec<(u32, u32)>,
}

impl PostingDelta {
    /// Add `other`'s changes to these.
    pub fn extend(&mut self, other: PostingDelta) {
        self.added.extend(other.added);
        self.removed.extend(other.removed);
    }

    /// `true` when nothing changed.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty()
    }
}

/// One posting's size.
#[derive(Debug, Default)]
pub struct PostingCounts {
    /// Entries: one per document with chunks in the posting.
    pub entries: AtomicU64,
    /// Chunks across those entries.
    pub chunks: AtomicU64,
}

/// Every posting's size, **exact**: counted from the namespace's document
/// records when its partition loads, then kept current by every vector write's
/// [`PostingDelta`]. Shared by every snapshot of a namespace's partition, so a
/// count never has to be copied when a split publishes a new snapshot. The
/// split trigger reads it.
#[derive(Debug, Default)]
pub struct CountTable {
    map: parking_lot::RwLock<HashMap<u32, Arc<PostingCounts>>>,
}

impl CountTable {
    /// A table holding `counts` (`posting → (entries, chunks)`).
    pub fn from_counts(counts: &HashMap<u32, (u64, u64)>) -> Self {
        let t = Self::default();
        t.set_all(counts);
        t
    }

    /// Replace every count.
    pub fn set_all(&self, counts: &HashMap<u32, (u64, u64)>) {
        let mut map = self.map.write();
        map.clear();
        for (&id, &(e, c)) in counts {
            map.insert(
                id,
                Arc::new(PostingCounts {
                    entries: AtomicU64::new(e),
                    chunks: AtomicU64::new(c),
                }),
            );
        }
    }

    fn slot(&self, id: u32) -> Arc<PostingCounts> {
        if let Some(c) = self.map.read().get(&id) {
            return Arc::clone(c);
        }
        Arc::clone(self.map.write().entry(id).or_default())
    }

    /// Apply a delta. A count never goes below 0.
    pub fn apply(&self, delta: &PostingDelta) {
        for &(id, chunks) in &delta.added {
            let c = self.slot(id);
            c.entries.fetch_add(1, Ordering::Relaxed);
            c.chunks.fetch_add(u64::from(chunks), Ordering::Relaxed);
        }
        for &(id, chunks) in &delta.removed {
            let c = self.slot(id);
            let _ = c.entries.try_update(Ordering::Relaxed, Ordering::Relaxed, |v| Some(v.saturating_sub(1)));
            let _ = c
                .chunks
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |v| Some(v.saturating_sub(u64::from(chunks))));
        }
    }

    /// `(entries, chunks)` of `posting` (0, 0 for one never counted).
    pub fn get(&self, posting: u32) -> (u64, u64) {
        self.map
            .read()
            .get(&posting)
            .map_or((0, 0), |c| (c.entries.load(Ordering::Relaxed), c.chunks.load(Ordering::Relaxed)))
    }

    /// Every posting's `(entries, chunks)`.
    pub fn snapshot(&self) -> HashMap<u32, (u64, u64)> {
        self.map
            .read()
            .iter()
            .map(|(&id, c)| (id, (c.entries.load(Ordering::Relaxed), c.chunks.load(Ordering::Relaxed))))
            .collect()
    }
}

/// Lifecycle of a posting (design doc M3a, *Maintenance journal*).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PostingState {
    /// Takes new chunks and is probed.
    Active = 0,
    /// Being split or merged away: probed (it still holds chunks), but no new
    /// chunk is routed to it.
    Draining = 1,
    /// Empty and gone from routing and probing; kept as a record (its centre
    /// may still be what moved codes decode against).
    Retired = 2,
}

impl PostingState {
    /// The state a stored byte names.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Active),
            1 => Some(Self::Draining),
            2 => Some(Self::Retired),
            _ => None,
        }
    }
}

/// How a posting came to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PostingOrigin {
    /// The single posting a namespace starts with.
    Root = 0,
    /// Copied from a centroid file (namespaces enabled before M3; tests and
    /// benches seed static baselines this way).
    Seed = 1,
    /// One half of a split.
    Split = 2,
    /// Fitted by a rebuild (design doc M4).
    Rebuild = 3,
}

impl PostingOrigin {
    /// The origin a stored byte names.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Root),
            1 => Some(Self::Seed),
            2 => Some(Self::Split),
            3 => Some(Self::Rebuild),
            _ => None,
        }
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
    /// Where it is in its lifecycle.
    pub state: PostingState,
    /// The posting it was split from.
    pub parent_id: Option<u32>,
    /// How it was created.
    pub origin: PostingOrigin,
}

impl Posting {
    /// An `Active` posting seeded from a centroid file (no parent).
    pub fn seeded(posting_id: u32, routing_centroid: Vec<f32>, centre_id: u32) -> Self {
        Self {
            posting_id,
            routing_centroid,
            centre_id,
            state: PostingState::Active,
            parent_id: None,
            origin: PostingOrigin::Seed,
        }
    }
}

/// What a namespace keeps about a posting besides its routing centroid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PostingInfo {
    /// The centre new codes in this posting are encoded against.
    pub centre_id: u32,
    /// Where it is in its lifecycle.
    pub state: PostingState,
    /// The posting it was split from.
    pub parent_id: Option<u32>,
    /// How it was created.
    pub origin: PostingOrigin,
}

/// One centre, in the original space and rotated.
#[derive(Debug)]
struct Centre {
    original: Vec<f32>,
    rotated: Vec<f32>,
}

/// A namespace's centres, each rotated once. Centres are append-only and never
/// change, so the next snapshot shares every existing one (an `Arc` per centre)
/// and rotates only the new: building a snapshot does not grow with the
/// centres a namespace has accumulated (splits and rebuilds add them, nothing
/// removes them, since codes keep their centre).
#[derive(Debug, Clone)]
struct CentreSet {
    dim: usize,
    rotator: Option<Arc<FhtKacRotator>>,
    map: HashMap<u32, Arc<Centre>>,
    max_id: Option<u32>,
}

impl CentreSet {
    fn new(centres: HashMap<u32, Vec<f32>>, seed: u64) -> Self {
        let dim = centres.values().next().map(Vec::len).unwrap_or(0);
        let rotator = (dim >= 8 && dim.is_multiple_of(2)).then(|| Arc::new(FhtKacRotator::new(dim, seed)));
        let mut set = Self {
            dim,
            rotator,
            map: HashMap::with_capacity(centres.len()),
            max_id: None,
        };
        for (id, c) in centres {
            set.insert(id, c);
        }
        set
    }

    fn insert(&mut self, id: u32, original: Vec<f32>) {
        let rotated = self.rotate(&original);
        self.map.insert(id, Arc::new(Centre { original, rotated }));
        self.max_id = Some(self.max_id.map_or(id, |m| m.max(id)));
    }

    /// These centres and `new` ones, which must be new ids of this dimension.
    fn with_added(&self, new: &[(u32, Vec<f32>)]) -> Result<Self, String> {
        let mut next = self.clone();
        for (id, c) in new {
            if next.map.contains_key(id) {
                return Err(format!("centre {id} already exists; centres are append-only"));
            }
            if c.len() != self.dim {
                return Err(format!("centre {id} has {} dimensions, expected {}", c.len(), self.dim));
            }
            next.insert(*id, c.clone());
        }
        Ok(next)
    }

    fn rotate(&self, v: &[f32]) -> Vec<f32> {
        let mut out = v.to_vec();
        if let Some(r) = &self.rotator {
            r.rotate_inplace(&mut out);
        }
        out
    }

    fn unrotate(&self, v: &[f32]) -> Vec<f32> {
        let mut out = v.to_vec();
        if let Some(r) = &self.rotator {
            r.unrotate_inplace(&mut out);
        }
        out
    }

    fn original(&self, id: u32) -> Option<&[f32]> {
        self.map.get(&id).map(|c| c.original.as_slice())
    }

    fn rotated(&self, id: u32) -> Option<&[f32]> {
        self.map.get(&id).map(|c| c.rotated.as_slice())
    }
}

/// A namespace's own partition at one moment: its postings, its centres, and its
/// rotation. Immutable; a split publishes a new one through the namespace's
/// [`PartitionHandle`]. Counts live in the shared [`CountTable`].
#[derive(Debug)]
pub struct NamespaceIvf {
    /// Routing centroids of `Active` postings: where new chunks go.
    routing: ClusterIndex,
    /// Routing centroids of `Active` and `Draining` postings: what queries probe.
    probing: ClusterIndex,
    /// Every stored posting, retired ones included.
    postings: HashMap<u32, PostingInfo>,
    /// Routing centroids of every posting that is not retired (for building the
    /// next snapshot).
    centroids: HashMap<u32, Vec<f32>>,
    /// Centres keyed by centre id, rotated with the namespace's seed.
    centres: CentreSet,
    /// Sizes, shared with every other snapshot of this namespace.
    counts: Arc<CountTable>,
    /// The namespace's rotation seed.
    seed: u64,
}

impl NamespaceIvf {
    /// Build from stored postings and centres, with a fresh count table. Fails if
    /// there is no posting or centre, a posting names a centre that does not
    /// exist, or dimensions differ.
    pub fn new(postings: Vec<Posting>, centres: HashMap<u32, Vec<f32>>, seed: u64) -> Result<Self, String> {
        Self::with_counts(postings, centres, seed, Arc::new(CountTable::default()))
    }

    /// [`new`](Self::new) sharing `counts`.
    pub fn with_counts(postings: Vec<Posting>, centres: HashMap<u32, Vec<f32>>, seed: u64, counts: Arc<CountTable>) -> Result<Self, String> {
        let dim = centres.values().next().map(Vec::len).unwrap_or(0);
        if let Some((id, c)) = centres.iter().find(|(_, c)| c.len() != dim) {
            return Err(format!("centre {id} has {} dimensions, expected {dim}", c.len()));
        }
        Self::from_parts(postings, CentreSet::new(centres, seed), seed, counts)
    }

    fn from_parts(postings: Vec<Posting>, centres: CentreSet, seed: u64, counts: Arc<CountTable>) -> Result<Self, String> {
        if postings.is_empty() || centres.map.is_empty() {
            return Err("a namespace index needs at least one posting and one centre".into());
        }
        let dim = centres.dim;
        if centres.map.contains_key(&ZERO_CENTRE) {
            return Err(format!("centre id {ZERO_CENTRE} is reserved for the zero centre"));
        }
        let mut infos = HashMap::with_capacity(postings.len());
        let mut centroids = HashMap::with_capacity(postings.len());
        let (mut routing, mut probing) = (HashMap::new(), HashMap::new());
        for p in postings {
            if !centres.map.contains_key(&p.centre_id) {
                return Err(format!("posting {} names centre {}, which does not exist", p.posting_id, p.centre_id));
            }
            if p.state != PostingState::Retired && p.routing_centroid.len() != dim {
                return Err(format!(
                    "posting {} has a {}-dimensional routing centroid, expected {dim}",
                    p.posting_id,
                    p.routing_centroid.len()
                ));
            }
            infos.insert(
                p.posting_id,
                PostingInfo {
                    centre_id: p.centre_id,
                    state: p.state,
                    parent_id: p.parent_id,
                    origin: p.origin,
                },
            );
            if p.state == PostingState::Active {
                routing.insert(p.posting_id, Cluster::new(p.posting_id, p.routing_centroid.clone()));
            }
            if p.state != PostingState::Retired {
                probing.insert(p.posting_id, Cluster::new(p.posting_id, p.routing_centroid.clone()));
                centroids.insert(p.posting_id, p.routing_centroid);
            }
        }
        if probing.is_empty() {
            return Err("a namespace index needs at least one posting that is not retired".into());
        }
        // Routing and probing compare in the original space: no rotation.
        Ok(Self {
            routing: ClusterIndex::from_clusters_unrotated(routing),
            probing: ClusterIndex::from_clusters_unrotated(probing),
            postings: infos,
            centroids,
            centres,
            counts,
            seed,
        })
    }

    /// The layout a centroid set seeds: one posting per centroid, each encoding
    /// against itself (posting id = centre id = cluster id).
    pub fn seeded(centroids: &HashMap<u32, Vec<f32>>, seed: u64) -> Result<Self, String> {
        let postings = centroids.iter().map(|(&id, c)| Posting::seeded(id, c.clone(), id)).collect();
        Self::new(postings, centroids.clone(), seed)
    }

    /// The partition after a change: `new_centres` appended and every posting in
    /// `changed` added or replaced; the counts stay shared.
    pub fn evolve(&self, new_centres: &[(u32, Vec<f32>)], changed: &[Posting]) -> Result<Self, String> {
        let centres = self.centres.with_added(new_centres)?;
        let mut postings: HashMap<u32, Posting> = self
            .postings
            .iter()
            .map(|(&id, info)| {
                (
                    id,
                    Posting {
                        posting_id: id,
                        routing_centroid: self.centroids.get(&id).cloned().unwrap_or_default(),
                        centre_id: info.centre_id,
                        state: info.state,
                        parent_id: info.parent_id,
                        origin: info.origin,
                    },
                )
            })
            .collect();
        for p in changed {
            postings.insert(p.posting_id, p.clone());
        }
        Self::from_parts(postings.into_values().collect(), centres, self.seed, Arc::clone(&self.counts))
    }

    /// `v` from the space codes are computed in back to the original space.
    pub fn unrotate(&self, v: &[f32]) -> Vec<f32> {
        self.centres.unrotate(v)
    }

    /// The rotation seed (`None` for dimensions too small to rotate).
    pub fn rotation_seed(&self) -> Option<u64> {
        self.centres.rotator.as_ref().map(|r| r.seed())
    }

    /// Number of postings that are not retired.
    pub fn postings(&self) -> usize {
        self.probing.len()
    }

    /// Number of centres.
    pub fn centres(&self) -> usize {
        self.centres.map.len()
    }

    /// Every stored posting, retired ones included.
    pub fn posting_infos(&self) -> &HashMap<u32, PostingInfo> {
        &self.postings
    }

    /// A posting's routing centroid (`None` once retired).
    pub fn routing_centroid(&self, posting: u32) -> Option<&[f32]> {
        self.centroids.get(&posting).map(Vec::as_slice)
    }

    /// The largest posting id and centre id in use (new ids go above them).
    pub fn max_ids(&self) -> (u32, u32) {
        (self.postings.keys().copied().max().unwrap_or(0), self.centres.max_id.unwrap_or(0))
    }

    /// The shared size table.
    pub fn counts(&self) -> &Arc<CountTable> {
        &self.counts
    }

    /// Apply a vector write's [`PostingDelta`] to the shared counts.
    pub fn apply_delta(&self, delta: &PostingDelta) {
        self.counts.apply(delta);
    }

    /// Entries across every posting.
    pub fn total_entries(&self) -> u64 {
        self.postings.keys().map(|&id| self.counts.get(id).0).sum()
    }

    /// Chunks in `posting`.
    pub fn posting_chunks(&self, posting: u32) -> u64 {
        self.counts.get(posting).1
    }

    /// The rotated centre of `posting` (what its codes and maintenance compare
    /// against; a posting's centre is its routing centroid).
    pub fn rotated_posting_centre(&self, posting: u32) -> Option<&[f32]> {
        self.centres.rotated(self.postings.get(&posting)?.centre_id)
    }

    /// Every `Active` posting with its rotated centre, in posting-id order (so
    /// maintenance breaks distance ties the same way every time).
    pub fn active_centres(&self) -> Vec<(u32, &[f32])> {
        let mut out: Vec<(u32, &[f32])> = self
            .postings
            .iter()
            .filter(|(_, info)| info.state == PostingState::Active)
            .filter_map(|(&id, info)| Some((id, self.centres.rotated(info.centre_id)?)))
            .collect();
        out.sort_unstable_by_key(|&(id, _)| id);
        out
    }
}

/// What a namespace's maintenance has done since its partition was loaded
/// (design doc M3b). Reported by partition health and the vector bench.
#[derive(Debug, Default)]
pub struct MaintenanceCounters {
    /// Postings split.
    pub splits: AtomicU64,
    /// Postings merged away.
    pub merges: AtomicU64,
    /// Reassignments run (one after each split).
    pub reassigns: AtomicU64,
    /// Chunks a reassignment found to be candidates.
    pub reassign_candidates: AtomicU64,
    /// Chunks a reassignment moved.
    pub reassign_moves: AtomicU64,
    /// Chunks copied out of a draining posting (a split's or a merge's).
    pub drain_moves: AtomicU64,
    /// Rebuilds run.
    pub rebuilds: AtomicU64,
    /// Chunks a rebuild copied to a new posting.
    pub rebuild_moves: AtomicU64,
}

impl MaintenanceCounters {
    /// Add `n` to `counter`.
    pub fn add(counter: &AtomicU64, n: u64) {
        counter.fetch_add(n, Ordering::Relaxed);
    }

    /// The current values.
    pub fn read(&self) -> MaintenanceTotals {
        let r = |c: &AtomicU64| c.load(Ordering::Relaxed);
        MaintenanceTotals {
            splits: r(&self.splits),
            merges: r(&self.merges),
            reassigns: r(&self.reassigns),
            reassign_candidates: r(&self.reassign_candidates),
            reassign_moves: r(&self.reassign_moves),
            drain_moves: r(&self.drain_moves),
            rebuilds: r(&self.rebuilds),
            rebuild_moves: r(&self.rebuild_moves),
        }
    }
}

/// A reading of [`MaintenanceCounters`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct MaintenanceTotals {
    /// Postings split.
    pub splits: u64,
    /// Postings merged away.
    pub merges: u64,
    /// Reassignments run.
    pub reassigns: u64,
    /// Chunks a reassignment found to be candidates.
    pub reassign_candidates: u64,
    /// Chunks a reassignment moved.
    pub reassign_moves: u64,
    /// Chunks copied out of a draining posting.
    pub drain_moves: u64,
    /// Rebuilds run.
    pub rebuilds: u64,
    /// Chunks a rebuild copied to a new posting.
    pub rebuild_moves: u64,
}

impl MaintenanceTotals {
    /// Every chunk key move of ongoing maintenance: drains (splits and merges)
    /// plus reassignments. Each is a put and, once the source goes, a delete or
    /// rewrite. A rebuild's moves are counted apart (`rebuild_moves`).
    pub fn chunk_moves(&self) -> u64 {
        self.drain_moves + self.reassign_moves
    }
}

impl IvfLayout for NamespaceIvf {
    fn dim(&self) -> usize {
        self.centres.dim
    }
    fn rotate(&self, v: &[f32]) -> Vec<f32> {
        self.centres.rotate(v)
    }
    fn route(&self, embedding: &[f32]) -> Option<u32> {
        IvfLayout::route(&self.routing, embedding)
    }
    fn probe(&self, queries: &[Vec<f32>], n: usize) -> Vec<Vec<u32>> {
        self.probing.find_top_n_cluster_ids_batch(queries, n)
    }
    fn centre_of(&self, posting: u32) -> Option<u32> {
        self.postings.get(&posting).map(|p| p.centre_id)
    }
    fn centre(&self, centre_id: u32) -> Option<&[f32]> {
        self.centres.original(centre_id)
    }
    fn rotated_centre(&self, centre_id: u32) -> Option<&[f32]> {
        self.centres.rotated(centre_id)
    }
    fn centre_ids(&self) -> Vec<u32> {
        self.centres.map.keys().copied().collect()
    }
    fn max_centre_id(&self) -> Option<u32> {
        self.centres.max_id
    }
    fn posting_entries(&self, posting: u32) -> Option<u64> {
        self.postings.contains_key(&posting).then(|| self.counts.get(posting).0)
    }
}

/// Grace periods for the routing epoch (an RCU-style barrier): readers
/// register in the current generation and never wait; a barrier starts a new
/// generation and waits until no reader of an earlier one is left.
#[derive(Debug, Default)]
struct Grace {
    state: parking_lot::Mutex<GraceState>,
    changed: tokio::sync::Notify,
}

#[derive(Debug, Default)]
struct GraceState {
    generation: u64,
    /// Readers still running, by the generation they started in.
    readers: std::collections::BTreeMap<u64, usize>,
}

/// A vector write or search inside the routing epoch. Dropping it leaves.
#[derive(Debug)]
pub struct EpochGuard<'a> {
    grace: &'a Grace,
    generation: u64,
}

impl Drop for EpochGuard<'_> {
    fn drop(&mut self) {
        {
            let mut s = self.grace.state.lock();
            if let Some(n) = s.readers.get_mut(&self.generation) {
                *n -= 1;
                if *n == 0 {
                    s.readers.remove(&self.generation);
                }
            }
        }
        self.grace.changed.notify_waiters();
    }
}

impl Grace {
    fn enter(&self) -> EpochGuard<'_> {
        let mut s = self.state.lock();
        let generation = s.generation;
        *s.readers.entry(generation).or_default() += 1;
        EpochGuard { grace: self, generation }
    }

    /// Wait until every reader that entered before this call has left.
    async fn wait(&self) {
        let last = {
            let mut s = self.state.lock();
            let last = s.generation;
            s.generation += 1;
            last
        };
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            // Registered before the check, so a leave between the two is seen.
            notified.as_mut().enable();
            if self.state.lock().readers.range(..=last).next().is_none() {
                return;
            }
            notified.await;
        }
    }
}

/// A namespace's live partition: the current snapshot, the shared counts, and
/// the locks maintenance relies on.
///
/// - **Routing epoch.** Every vector write (route, encode, write, apply its
///   delta) and every search (choose postings, scan them) runs inside it
///   ([`read_epoch`](Self::read_epoch)). It is a grace period, not a lock:
///   entering never waits. [`publish`](Self::publish) installs a snapshot and
///   then waits until every write and search that entered before it has left,
///   so once it returns nothing is still filing chunks by the old routing.
///   [`epoch_barrier`](Self::epoch_barrier) waits the same way, so a split can
///   delete a posting's old keys without a search that began before the copy
///   missing them (design doc M3a, rule 4). Neither holds up a write or search
///   that starts after it, so a slow search delays maintenance, never other
///   searches.
/// - **Maintenance lock.** One maintenance operation per namespace at a time.
/// - **Recovery.** A handle starts out needing its journal recovered (a crash
///   may have left an operation half done), and is marked so again after any
///   maintenance error; [`crate::ivf_split::ensure_recovered`] runs recovery
///   before anything else uses it.
/// - **Dropped.** Set when its namespace's vector index is dropped: maintenance
///   then stops, so it never writes (and so recreates) a dropped namespace.
#[derive(Debug)]
pub struct PartitionHandle {
    current: parking_lot::RwLock<Arc<NamespaceIvf>>,
    epoch: Grace,
    maintenance: tokio::sync::Mutex<()>,
    needs_recovery: std::sync::atomic::AtomicBool,
    dropped: std::sync::atomic::AtomicBool,
    unsplittable: parking_lot::Mutex<HashMap<u32, u64>>,
    unmergeable: parking_lot::Mutex<HashMap<u32, u64>>,
    counters: MaintenanceCounters,
    /// `(split_limit, merge_limit)` in chunks; `(u64::MAX, 0)` until set.
    limits: parking_lot::Mutex<(u64, u64)>,
}

impl PartitionHandle {
    /// A handle starting at `ivf`. It needs recovery before first use.
    pub fn new(ivf: NamespaceIvf) -> Self {
        Self {
            current: parking_lot::RwLock::new(Arc::new(ivf)),
            epoch: Grace::default(),
            maintenance: tokio::sync::Mutex::new(()),
            needs_recovery: std::sync::atomic::AtomicBool::new(true),
            dropped: std::sync::atomic::AtomicBool::new(false),
            unsplittable: parking_lot::Mutex::new(HashMap::new()),
            unmergeable: parking_lot::Mutex::new(HashMap::new()),
            counters: MaintenanceCounters::default(),
            limits: parking_lot::Mutex::new((u64::MAX, 0)),
        }
    }

    /// Record the namespace's current split and merge limits (in chunks), which
    /// [`needs_maintenance`](Self::needs_maintenance) checks writes against.
    pub fn set_limits(&self, split_limit: u64, merge_limit: u64) {
        *self.limits.lock() = (split_limit, merge_limit);
    }

    /// Whether a write that made `delta` (already applied) left a posting to
    /// split (one it added to is over the split limit) or to merge (one it
    /// removed from is at or under the merge limit, while another posting is
    /// active).
    pub fn needs_maintenance(&self, delta: &PostingDelta) -> bool {
        let (split_limit, merge_limit) = *self.limits.lock();
        let ivf = self.snapshot();
        let active = |p: u32| ivf.posting_infos().get(&p).map(|i| i.state) == Some(PostingState::Active);
        delta.added.iter().any(|&(p, _)| ivf.posting_chunks(p) > split_limit)
            || (ivf.routing.len() > 1 && delta.removed.iter().any(|&(p, _)| active(p) && ivf.posting_chunks(p) <= merge_limit))
    }

    /// What maintenance has done since this handle was made.
    pub fn counters(&self) -> &MaintenanceCounters {
        &self.counters
    }

    /// Wait until no vector write or search that started before this call is
    /// still running. Maintenance calls it between copying chunks and deleting
    /// their old keys, so a search that could have missed a copy has finished
    /// before the original goes. Writes and searches that start meanwhile do not
    /// wait.
    pub async fn epoch_barrier(&self) {
        self.epoch.wait().await;
    }

    /// Remember that `posting` found no neighbour with room to merge into at
    /// `chunks` chunks; it is tried again only once its size changes.
    pub fn mark_unmergeable(&self, posting: u32, chunks: u64) {
        self.unmergeable.lock().insert(posting, chunks);
    }

    /// Whether `posting`, now at `chunks` chunks, found no merge at this size.
    pub fn is_unmergeable(&self, posting: u32, chunks: u64) -> bool {
        self.unmergeable.lock().get(&posting) == Some(&chunks)
    }

    /// The current snapshot.
    pub fn snapshot(&self) -> Arc<NamespaceIvf> {
        Arc::clone(&self.current.read())
    }

    /// Enter the routing epoch as a writer of vectors or a search. Never waits.
    pub async fn read_epoch(&self) -> EpochGuard<'_> {
        self.epoch.enter()
    }

    /// Install `ivf` as the current snapshot, then wait until every vector
    /// write and search that entered the epoch before it has left: on return,
    /// nothing routes or scans by an older snapshot. Only maintenance (holding
    /// the maintenance lock) publishes. Marks of postings no longer active are
    /// dropped.
    pub async fn publish(&self, ivf: NamespaceIvf) {
        let active = |id: &u32| ivf.posting_infos().get(id).map(|i| i.state) == Some(PostingState::Active);
        self.unsplittable.lock().retain(|id, _| active(id));
        self.unmergeable.lock().retain(|id, _| active(id));
        *self.current.write() = Arc::new(ivf);
        self.epoch.wait().await;
    }

    /// Serialise maintenance on this namespace.
    pub async fn lock_maintenance(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.maintenance.lock().await
    }

    /// Whether the journal must be recovered before this partition is used.
    pub fn needs_recovery(&self) -> bool {
        self.needs_recovery.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Mark whether the journal must be recovered before further use.
    pub fn set_needs_recovery(&self, needed: bool) {
        self.needs_recovery.store(needed, std::sync::atomic::Ordering::Release);
    }

    /// Stop maintenance for good: the namespace's vector index is being
    /// dropped. Maintenance checks it after taking the maintenance lock and
    /// between batches.
    pub fn mark_dropped(&self) {
        self.dropped.store(true, std::sync::atomic::Ordering::Release);
    }

    /// Whether [`mark_dropped`](Self::mark_dropped) was called.
    pub fn is_dropped(&self) -> bool {
        self.dropped.load(std::sync::atomic::Ordering::Acquire)
    }

    /// The shared counts.
    pub fn counts(&self) -> Arc<CountTable> {
        Arc::clone(self.snapshot().counts())
    }

    /// Remember that `posting` could not be split at `chunks` chunks; it is
    /// tried again only once its size changes.
    pub fn mark_unsplittable(&self, posting: u32, chunks: u64) {
        self.unsplittable.lock().insert(posting, chunks);
    }

    /// Whether `posting`, now at `chunks` chunks, failed to split at this size.
    pub fn is_unsplittable(&self, posting: u32, chunks: u64) -> bool {
        self.unsplittable.lock().get(&posting) == Some(&chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic_search::cluster::DEFAULT_ROTATION_SEED;

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
        let postings = vec![Posting::seeded(10, c[&0].clone(), 0), Posting::seeded(11, c[&1].clone(), 0)];
        let ivf = NamespaceIvf::new(postings, c.clone(), DEFAULT_ROTATION_SEED).unwrap();
        assert_eq!(IvfLayout::route(&ivf, &c[&1]), Some(11));
        assert_eq!(ivf.centre_of(11), Some(0));
        assert_eq!(ivf.centre_of(12), None);
        assert_eq!((ivf.postings(), ivf.centres()), (2, 4));
    }

    #[test]
    fn inconsistent_parts_are_rejected() {
        let c = centroids(16);
        let bad = vec![Posting::seeded(1, c[&0].clone(), 99)];
        assert!(NamespaceIvf::new(bad, c.clone(), 1).unwrap_err().contains("centre 99"));
        assert!(NamespaceIvf::new(vec![], c.clone(), 1).is_err());
        let mut reserved = c.clone();
        reserved.insert(ZERO_CENTRE, vec![0.0; 16]);
        let posting = vec![Posting::seeded(1, c[&0].clone(), 0)];
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

    fn probes(min_probes: usize, max_probes: usize) -> ProbeSettings {
        ProbeSettings { min_probes, max_probes }
    }

    #[test]
    fn each_query_vector_probes_its_max_probes_nearest_postings() {
        let (layout, q) = ranked_layout([10, 20, 30, 40]);
        let qs = [q];
        assert_eq!(select_probes(&layout, &qs, &probes(1, 1)), (vec![0], 10));
        assert_eq!(select_probes(&layout, &qs, &probes(1, 2)), (vec![0, 1], 30));
        assert_eq!(select_probes(&layout, &qs, &probes(1, 3)), (vec![0, 1, 2], 60));
    }

    #[test]
    fn a_namespace_with_fewer_postings_than_max_probes_is_read_in_full() {
        let (layout, q) = ranked_layout([10, 20, 30, 40]);
        assert_eq!(select_probes(&layout, &[q], &probes(1, 1024)), (vec![0, 1, 2, 3], 100));
    }

    #[test]
    fn min_probes_is_a_floor_under_max_probes() {
        let (layout, q) = ranked_layout([10, 20, 30, 40]);
        assert_eq!(select_probes(&layout, &[q], &probes(3, 2)).0, vec![0, 1, 2]);
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
        assert_eq!(select_probes(&layout, &[q], &probes(1, 3)), (vec![0, 1, 2], 0));
    }

    #[test]
    fn each_query_vector_probes_its_own_postings_and_the_union_is_probed() {
        let (layout, q) = ranked_layout([10, 20, 30, 40]);
        let mut q2 = vec![0.0f32; 16];
        q2[..4].copy_from_slice(&[0.1, 0.2, 0.4, 0.8]); // nearest 3, then 2
        // q: {0, 1}; q2: {3, 2}; nothing is counted twice.
        assert_eq!(select_probes(&layout, &[q.clone(), q2.clone()], &probes(1, 1)), (vec![0, 3], 50));
        assert_eq!(select_probes(&layout, &[q, q2], &probes(1, 2)), (vec![0, 1, 3, 2], 100));
    }

    #[test]
    fn deltas_keep_exact_counts_and_never_go_below_zero() {
        let ivf = NamespaceIvf::seeded(&centroids(16), 1).unwrap();
        assert_eq!(ivf.posting_entries(0), Some(0));
        ivf.counts().set_all(&HashMap::from([(0, (5, 9)), (1, (2, 2)), (99, (7, 7))]));
        assert_eq!(
            (ivf.posting_entries(0), ivf.posting_entries(2), ivf.posting_entries(99)),
            (Some(5), Some(0), None)
        );
        // Doc rewritten in posting 1 (2 chunks → 3) and moved out of 0 into 2.
        ivf.apply_delta(&PostingDelta {
            added: vec![(1, 3), (2, 4)],
            removed: vec![(1, 2), (0, 4), (3, 1)],
        });
        assert_eq!([0, 1, 2, 3].map(|id| ivf.counts().get(id)), [(4, 5), (2, 3), (1, 4), (0, 0)]);
        assert_eq!(ivf.total_entries(), 7);
        assert_eq!(ivf.posting_chunks(1), 3);
    }

    #[test]
    fn draining_postings_are_probed_but_not_routed_and_retired_ones_neither() {
        let c = centroids(16);
        let mut postings: Vec<Posting> = (0..4u32).map(|id| Posting::seeded(id, c[&id].clone(), id)).collect();
        postings[0].state = PostingState::Draining;
        postings[1].state = PostingState::Retired;
        let ivf = NamespaceIvf::new(postings, c.clone(), 1).unwrap();
        assert_ne!(IvfLayout::route(&ivf, &c[&0]), Some(0));
        assert_ne!(IvfLayout::route(&ivf, &c[&1]), Some(1));
        let probed = ivf.probe(std::slice::from_ref(&c[&0]), 4).remove(0);
        assert!(probed.contains(&0) && !probed.contains(&1), "{probed:?}");
        assert_eq!(ivf.postings(), 3);
        // A retired posting still names its centre: moved codes decode through it.
        assert_eq!(ivf.centre_of(1), Some(1));
    }

    #[test]
    fn evolve_appends_centres_replaces_postings_and_shares_the_counts() {
        let c = centroids(16);
        let ivf = NamespaceIvf::seeded(&c, 7).unwrap();
        ivf.counts().set_all(&HashMap::from([(0, (3, 6))]));
        let mut drained = Posting::seeded(0, c[&0].clone(), 0);
        drained.state = PostingState::Draining;
        let child = Posting {
            posting_id: 10,
            routing_centroid: c[&1].clone(),
            centre_id: 10,
            state: PostingState::Active,
            parent_id: Some(0),
            origin: PostingOrigin::Split,
        };
        let next = ivf.evolve(&[(10, c[&1].clone())], &[drained, child]).unwrap();
        assert_eq!((next.centres(), next.postings()), (5, 5));
        assert_eq!(next.posting_infos()[&0].state, PostingState::Draining);
        assert_eq!(next.posting_infos()[&10].parent_id, Some(0));
        assert_eq!(next.rotation_seed(), Some(7));
        assert!(Arc::ptr_eq(next.counts(), ivf.counts()));
        assert_eq!(next.max_ids(), (10, 10));
        assert!(ivf.evolve(&[(1, c[&1].clone())], &[]).unwrap_err().contains("append-only"));
    }

    /// A publish builds the next snapshot from the current one. Centres never
    /// change, so it keeps the rotated ones it has instead of rotating every
    /// centre again (measured 9.5 ms at FiQA's 1,957 centres, 144 ms at 50,000,
    /// paid on every split's publishes).
    #[test]
    fn evolving_keeps_the_existing_rotated_centres() {
        let c = centroids(16);
        let ivf = NamespaceIvf::seeded(&c, DEFAULT_ROTATION_SEED).unwrap();
        let added: Vec<f32> = (0..16).map(|i| (i as f32 - 7.5) / 10.0).collect();
        let next = ivf.evolve(&[(9, added.clone())], &[]).unwrap();
        for id in 0..4u32 {
            let (a, b) = (ivf.rotated_centre(id).unwrap(), next.rotated_centre(id).unwrap());
            assert!(std::ptr::eq(a.as_ptr(), b.as_ptr()), "centre {id} was rotated again");
        }
        assert_eq!(next.rotated_centre(9).unwrap(), ivf.rotate(&added).as_slice(), "a new centre is rotated");
        assert_eq!(next.centre(9).unwrap(), added.as_slice());
        assert_eq!(next.unrotate(&next.rotate(&added)).len(), 16);
    }

    /// A barrier waits for the readers (searches, vector writes) that started
    /// before it ...
    #[tokio::test]
    async fn a_barrier_waits_for_readers_that_started_before_it() {
        let h = Arc::new(PartitionHandle::new(NamespaceIvf::seeded(&centroids(16), DEFAULT_ROTATION_SEED).unwrap()));
        let old = h.read_epoch().await;
        let barrier = tokio::spawn({
            let h = Arc::clone(&h);
            async move { h.epoch_barrier().await }
        });
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert!(!barrier.is_finished(), "the barrier passed a reader that started before it");
        drop(old);
        tokio::time::timeout(std::time::Duration::from_secs(5), barrier).await.unwrap().unwrap();
    }

    /// ... but never holds up a reader that starts after it: a search must not
    /// queue behind a split waiting for an older, slow search.
    #[tokio::test]
    async fn a_new_reader_does_not_wait_behind_a_barrier() {
        let h = Arc::new(PartitionHandle::new(NamespaceIvf::seeded(&centroids(16), DEFAULT_ROTATION_SEED).unwrap()));
        let old = h.read_epoch().await;
        let barrier = tokio::spawn({
            let h = Arc::clone(&h);
            async move { h.epoch_barrier().await }
        });
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let new = tokio::time::timeout(std::time::Duration::from_secs(1), h.read_epoch()).await;
        assert!(new.is_ok(), "a new reader waited behind a barrier");
        drop(new);
        drop(old);
        tokio::time::timeout(std::time::Duration::from_secs(5), barrier).await.unwrap().unwrap();
    }
}
