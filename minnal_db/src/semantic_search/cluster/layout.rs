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
use std::sync::Arc;
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
/// split trigger and probing by budget both read it.
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
            let _ = c
                .entries
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| Some(v.saturating_sub(1)));
            let _ = c
                .chunks
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| Some(v.saturating_sub(u64::from(chunks))));
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
    /// Copied from a centroid file (namespaces enabled before M3).
    Seed = 1,
    /// One half of a split.
    Split = 2,
}

impl PostingOrigin {
    /// The origin a stored byte names.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Root),
            1 => Some(Self::Seed),
            2 => Some(Self::Split),
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
    centres: ClusterIndex,
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
        let mut infos = HashMap::with_capacity(postings.len());
        let mut centroids = HashMap::with_capacity(postings.len());
        let (mut routing, mut probing) = (HashMap::new(), HashMap::new());
        for p in postings {
            if !centres.contains_key(&p.centre_id) {
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
        Ok(Self {
            routing: ClusterIndex::from_clusters_with_seed(routing, seed),
            probing: ClusterIndex::from_clusters_with_seed(probing, seed),
            postings: infos,
            centroids,
            centres: ClusterIndex::from_clusters_with_seed(centres.into_iter().map(|(id, c)| (id, Cluster::new(id, c))).collect(), seed),
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

    /// [`seeded`](Self::seeded) from a centroid file, with the default seed: what
    /// benches and evaluations index against.
    pub fn from_cluster_file(path: &str) -> Result<Self, Box<dyn Error>> {
        let centroids = super::read_clusters_from_file(path)?;
        Ok(Self::seeded(&centroids, DEFAULT_ROTATION_SEED)?)
    }

    /// The partition after a change: `new_centres` appended and every posting in
    /// `changed` added or replaced; the counts stay shared.
    pub fn evolve(&self, new_centres: &[(u32, Vec<f32>)], changed: &[Posting]) -> Result<Self, String> {
        let seed = self.seed;
        let mut centres: HashMap<u32, Vec<f32>> = self.centres.clusters.iter().map(|(&id, c)| (id, c.centroid.clone())).collect();
        for (id, c) in new_centres {
            if centres.insert(*id, c.clone()).is_some() {
                return Err(format!("centre {id} already exists; centres are append-only"));
            }
        }
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
        Self::with_counts(postings.into_values().collect(), centres, seed, Arc::clone(&self.counts))
    }

    /// The rotation seed (`None` for dimensions too small to rotate).
    pub fn rotation_seed(&self) -> Option<u64> {
        self.centres.rotation_seed()
    }

    /// Number of postings that are not retired.
    pub fn postings(&self) -> usize {
        self.probing.len()
    }

    /// Number of centres.
    pub fn centres(&self) -> usize {
        self.centres.len()
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
        (
            self.postings.keys().copied().max().unwrap_or(0),
            self.centres.clusters.keys().copied().max().unwrap_or(0),
        )
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
        self.probing.find_top_n_cluster_ids_batch(queries, n)
    }
    fn centre_of(&self, posting: u32) -> Option<u32> {
        self.postings.get(&posting).map(|p| p.centre_id)
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
        self.postings.contains_key(&posting).then(|| self.counts.get(posting).0)
    }
}

/// A namespace's live partition: the current snapshot, the shared counts, and
/// the two locks maintenance relies on.
///
/// - **Routing epoch.** Every vector write (route, encode, write, apply its
///   delta) and every search (choose postings, scan them) holds a read guard;
///   publishing a new snapshot takes the write guard. So once a publish returns,
///   no write is still filing chunks under the old routing and no search is still
///   scanning by it: a split can then delete the old posting's keys without a
///   search missing them (design doc M3a, rule 4).
/// - **Maintenance lock.** One maintenance operation per namespace at a time.
#[derive(Debug)]
pub struct PartitionHandle {
    current: parking_lot::RwLock<Arc<NamespaceIvf>>,
    epoch: tokio::sync::RwLock<()>,
    maintenance: tokio::sync::Mutex<()>,
    unsplittable: parking_lot::Mutex<HashMap<u32, u64>>,
}

impl PartitionHandle {
    /// A handle starting at `ivf`.
    pub fn new(ivf: NamespaceIvf) -> Self {
        Self {
            current: parking_lot::RwLock::new(Arc::new(ivf)),
            epoch: tokio::sync::RwLock::new(()),
            maintenance: tokio::sync::Mutex::new(()),
            unsplittable: parking_lot::Mutex::new(HashMap::new()),
        }
    }

    /// The current snapshot.
    pub fn snapshot(&self) -> Arc<NamespaceIvf> {
        Arc::clone(&self.current.read())
    }

    /// Enter the routing epoch as a writer of vectors or a search.
    pub async fn read_epoch(&self) -> tokio::sync::RwLockReadGuard<'_, ()> {
        self.epoch.read().await
    }

    /// Exclude every vector write and search, to publish a snapshot.
    pub async fn write_epoch(&self) -> tokio::sync::RwLockWriteGuard<'_, ()> {
        self.epoch.write().await
    }

    /// Install `ivf` as the current snapshot. The caller holds the
    /// [`write_epoch`](Self::write_epoch) guard (passed as proof).
    pub fn publish(&self, _guard: &tokio::sync::RwLockWriteGuard<'_, ()>, ivf: NamespaceIvf) {
        *self.current.write() = Arc::new(ivf);
    }

    /// Serialise maintenance on this namespace.
    pub async fn lock_maintenance(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.maintenance.lock().await
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
}
