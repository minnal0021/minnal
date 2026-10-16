//! Reassignment after a split (design doc *M3 split algorithm (LIRE)*, step 5;
//! M3b).
//!
//! A split moves the centre its chunks were routed by, so some chunks are now
//! nearer another posting: chunks of the two halves that were nearer the old
//! centre than to their half's, and chunks of nearby postings that the new
//! centres now beat. LIRE checks exactly those:
//!
//! - a chunk in a half `Pi` is a **candidate** if `D(x̂, c_P) < D(x̂, c_Pi)`;
//! - a chunk in one of the `reassign_range` postings nearest `c_P` is a
//!   candidate if the nearer half beats both the old centre and the chunk's own
//!   posting: `min_i D(x̂, c_Pi) < D(x̂, c_P)` and `< D(x̂, c_cur)`.
//!
//! A candidate moves to its nearest active posting, unless that is where it is.
//! `x̂` is the chunk's reconstruction from its 1-bit code; with
//! `skip_uncertain_moves` every comparison must win by more than the code's
//! error allows (see `nearer`).
//!
//! The postings involved stay active, so a move copies the chunk to its
//! destination (no-WAL, after listing the destination in the document's meta)
//! and, after a barrier flush, shrinks the source key with a WAL-backed write.
//! A lost no-WAL shrink would leave a duplicate and a wrong count for good. The
//! journal record names the scope (the halves and the neighbours), and the rule
//! is re-evaluated on a redo: under the maintenance lock nothing else changes
//! the postings, so it decides the same way, and a source gives up a chunk only
//! once its destination holds it.

use std::collections::{BTreeMap, HashMap};

use super::*;

/// The reassignment to record after `parent`'s split into `children`: the
/// halves plus the `reassign_range` active postings nearest the old centre.
/// `None` if a centre is unknown.
pub(super) fn plan(ivf: &NamespaceIvf, ops: &MaintenanceOps, parent: u32, children: [u32; 2]) -> Option<Record> {
    let parent_centre = ivf.posting_infos().get(&parent)?.centre_id;
    let c_p = ivf.rotated_centre(parent_centre)?;
    for c in children {
        ivf.rotated_posting_centre(c)?;
    }
    let others: Vec<(u32, &[f32])> = ivf.active_centres().into_iter().filter(|(id, _)| !children.contains(id)).collect();
    let mut scope = children.to_vec();
    scope.extend(neighbours(c_p, &others, ops.reassign_range));
    Some(Record {
        state: JournalState::Planned,
        op: Op::Reassign {
            parent,
            parent_centre,
            children,
            scope,
        },
    })
}

/// `⟨σ, v⟩` for a 1-bit code's sign vector `σ` (`+1` where its bit is set,
/// `−1` elsewhere), by table lookup: one 256-entry table per byte of the packed
/// bits, holding the sum of `v` over the bits set in that byte. 96 lookups for
/// 768 dimensions, against a full reconstruction and a distance.
struct SignDot {
    tables: Vec<[f32; 256]>,
    total: f32,
}

impl SignDot {
    fn new(v: &[f32]) -> Self {
        let tables = v
            .chunks(8)
            .map(|byte| {
                let mut t = [0.0f32; 256];
                for b in 1..256usize {
                    // The table of `b` is that of `b` without its lowest bit,
                    // plus the dimension of that bit.
                    t[b] = t[b & (b - 1)] + byte.get(b.trailing_zeros() as usize).copied().unwrap_or(0.0);
                }
                t
            })
            .collect();
        Self {
            tables,
            total: v.iter().sum(),
        }
    }

    fn dot(&self, packed: &[u64]) -> f64 {
        let mut set = 0.0f32;
        for (j, t) in self.tables.iter().enumerate() {
            set += t[((packed[j / 8] >> (8 * (j % 8))) & 0xff) as usize];
        }
        f64::from(2.0 * set - self.total)
    }
}

/// Distances from a code's reconstruction to a few fixed centres, without
/// reconstructing it. With `x̂ = c + s·σ` (its own centre `c`, scale `s`, signs
/// `σ`), `D(x̂, t) − ‖x̂‖² = ‖t‖² − 2⟨c, t⟩ − 2s⟨σ, t⟩`: comparing that *score*
/// across targets is comparing distances. `⟨c, t⟩` is cached per (own centre,
/// target), since a scope's codes share few centres. Targets are keyed by
/// centre id.
struct Scorer<'a> {
    ivf: &'a NamespaceIvf,
    targets: parking_lot::Mutex<HashMap<u32, std::sync::Arc<(f64, SignDot)>>>,
    own: parking_lot::Mutex<HashMap<(u32, u32), f64>>,
}

impl<'a> Scorer<'a> {
    fn new(ivf: &'a NamespaceIvf) -> Self {
        Self {
            ivf,
            targets: parking_lot::Mutex::new(HashMap::new()),
            own: parking_lot::Mutex::new(HashMap::new()),
        }
    }

    /// `D(x̂, t) − ‖x̂‖²` for target centre `t`, or `None` if a centre is unknown.
    /// The code must be 1-bit and match the dimension (checked by the caller).
    fn score(&self, code: &VectorIndex, t: u32) -> Option<f64> {
        let target = {
            let mut targets = self.targets.lock();
            match targets.get(&t) {
                Some(x) => std::sync::Arc::clone(x),
                None => {
                    let c = self.ivf.rotated_centre(t)?;
                    let made = std::sync::Arc::new((c.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>(), SignDot::new(c)));
                    targets.insert(t, std::sync::Arc::clone(&made));
                    made
                }
            }
        };
        let key = (code.centre_id, t);
        // Read, release, then compute: a guard held across the `match` would
        // deadlock on the insert below (the lock is not reentrant).
        let cached = self.own.lock().get(&key).copied();
        let own = match cached {
            Some(d) => d,
            None => {
                use simsimd::SpatialSimilarity;
                let d = f32::dot(self.ivf.rotated_centre(code.centre_id)?, self.ivf.rotated_centre(t)?)?;
                self.own.lock().insert(key, d);
                d
            }
        };
        Some(target.0 - 2.0 * (own + f64::from(code.scaling_factor) * target.1.dot(&code.packed_vector)))
    }

    /// Whether centre `a` is nearer to the code than centre `b`, as
    /// [`nearer`] decides it on the reconstruction.
    fn nearer(&self, code: &VectorIndex, a: u32, b: u32, gate: bool) -> Option<bool> {
        let (sa, sb) = (self.score(code, a)?, self.score(code, b)?);
        if sa >= sb {
            return Some(false);
        }
        Some(!gate || sb - sa > 2.0 * f64::from(code.error_bound) * sq_dist(self.ivf.rotated_centre(a)?, self.ivf.rotated_centre(b)?).sqrt())
    }
}

/// LIRE's conditions for one reassignment.
struct Rule<'a> {
    parent_centre: u32,
    children: [(u32, u32); 2],
    active: Vec<(u32, &'a [f32])>,
    scorer: Scorer<'a>,
    gate: bool,
}

impl<'a> Rule<'a> {
    fn new(ivf: &'a NamespaceIvf, ops: &MaintenanceOps, parent_centre: u32, children: [u32; 2]) -> Option<Self> {
        ivf.rotated_centre(parent_centre)?;
        let centre_of = |p: u32| {
            let c = ivf.posting_infos().get(&p)?.centre_id;
            ivf.rotated_centre(c).map(|_| c)
        };
        Some(Self {
            parent_centre,
            children: [(children[0], centre_of(children[0])?), (children[1], centre_of(children[1])?)],
            active: ivf.active_centres(),
            scorer: Scorer::new(ivf),
            gate: ops.skip_uncertain_moves,
        })
    }

    /// For a chunk in posting `src`: whether it is a candidate, and where it
    /// moves (`None` to stay). `x` is scratch space for the reconstruction a
    /// candidate needs. LIRE's conditions are decided from the code without
    /// reconstructing it (`Scorer`); only a candidate is reconstructed, to find
    /// its nearest active posting.
    fn decide(&self, ivf: &NamespaceIvf, src: u32, src_centre: &[f32], code: &VectorIndex, x: &mut Vec<f32>) -> (bool, Option<u32>) {
        use crate::semantic_search::index::vector_index::QuantisationStyle;
        let dim = src_centre.len();
        let Some(src_c) = ivf.posting_infos().get(&src).map(|i| i.centre_id) else {
            return (false, None);
        };
        if code.quantisation_style != QuantisationStyle::SingleBit
            || code.packed_vector.len() < dim.div_ceil(64)
            || ivf.rotated_centre(code.centre_id).is_none_or(|c| c.len() != dim)
        {
            return (false, None);
        }
        let s = &self.scorer;
        let candidate = if self.children.iter().any(|&(id, _)| id == src) {
            s.nearer(code, self.parent_centre, src_c, self.gate)
        } else {
            // The nearer half (the first on a tie, as `nearest` picks).
            let [(_, c0), (_, c1)] = self.children;
            let c_min = match (s.score(code, c0), s.score(code, c1)) {
                (Some(a), Some(b)) => Some(if b < a { c1 } else { c0 }),
                _ => None,
            };
            c_min.and_then(|c| Some(s.nearer(code, c, self.parent_centre, self.gate)? && s.nearer(code, c, src_c, self.gate)?))
        };
        if candidate != Some(true) || !reconstruct_into(ivf, code, x) {
            return (false, None);
        }
        let x = x.as_slice();
        match nearest(x, &self.active) {
            Some((n, c)) if n != src && nearer(x, c, src_centre, code, self.gate) => (true, Some(n)),
            _ => (true, None),
        }
    }

    /// [`decide`](Self::decide) on the reconstruction, as it was written before
    /// the scorer: the oracle the scorer is tested against.
    #[cfg(test)]
    fn decide_reconstructed(&self, ivf: &NamespaceIvf, src: u32, src_centre: &[f32], code: &VectorIndex) -> (bool, Option<u32>) {
        let Some(x) = reconstruct(ivf, code) else { return (false, None) };
        let parent = ivf.rotated_centre(self.parent_centre).unwrap();
        let children: Vec<(u32, &[f32])> = self.children.iter().map(|&(p, c)| (p, ivf.rotated_centre(c).unwrap())).collect();
        let candidate = if self.children.iter().any(|&(id, _)| id == src) {
            nearer(&x, parent, src_centre, code, self.gate)
        } else {
            let (_, c_min) = nearest(&x, &children).expect("two halves");
            nearer(&x, c_min, parent, code, self.gate) && nearer(&x, c_min, src_centre, code, self.gate)
        };
        if !candidate {
            return (false, None);
        }
        match nearest(&x, &self.active) {
            Some((n, c)) if n != src && nearer(&x, c, src_centre, code, self.gate) => (true, Some(n)),
            _ => (true, None),
        }
    }
}

/// Run (or, after a crash, finish) a recorded reassignment.
pub(super) async fn run(
    db: &AsyncDb,
    namespace: &str,
    handle: &PartitionHandle,
    ops: &MaintenanceOps,
    op: u64,
    record: Record,
) -> Result<(), crate::KVError> {
    let Op::Reassign {
        parent_centre,
        children,
        ref scope,
        ..
    } = record.op
    else {
        unreachable!("not a reassignment")
    };
    let ivf = handle.snapshot();
    let Some(rule) = Rule::new(&ivf, ops, parent_centre, children) else {
        return Err(corrupt(namespace, format!("reassignment {op}: a centre it names is missing")));
    };
    // Sources that are still active (a redo after a crash finds them so).
    let sources: Vec<(u32, Vec<f32>)> = scope
        .iter()
        .filter(|p| ivf.posting_infos().get(p).map(|i| i.state) == Some(PostingState::Active))
        .filter_map(|&p| Some((p, ivf.rotated_posting_centre(p)?.to_vec())))
        .collect();

    // The entries with a moving chunk. A live run decides once and shrinks the
    // same entries it copied from: a chunk written since by an upsert was routed
    // by its float and is no candidate a shrink could take (and a shrink only
    // takes chunks the destination holds). A redo from `Copied` decides again.
    let mut reuse = None;
    if record.state == JournalState::Planned {
        let (selected, skipped) = select(db, namespace, &ivf, ops, parent_centre, children, &sources).await?;
        let sparse = db.namespace(sparse_vectors_ns(namespace)).await?;
        let (rule, ivf, sparse) = (&rule, &ivf, &sparse);
        let counts = in_batches(&selected, |batch| async move {
            let _docs = lock_docs_vectors(namespace, batch.iter().map(|(doc, _)| doc.as_slice())).await;
            // Re-read under the locks: an upsert or delete may have moved them on.
            let keys = batch
                .iter()
                .flat_map(|(doc, srcs)| srcs.iter().map(move |(src, _)| composite_key::encode(*src, doc)))
                .collect();
            let mut values = sparse.get_multiple(keys).await?.into_iter();
            let (mut x, mut candidates, mut items) = (Vec::new(), 0u64, Vec::new());
            for (doc, srcs) in batch {
                // Every source's moves together, so two sources moving into one
                // key are copied as one.
                let mut moving: BTreeMap<u32, Vec<VectorIndex>> = BTreeMap::new();
                for (src, src_centre) in srcs {
                    let Some(bytes) = values.next().expect("one value per key") else {
                        continue;
                    };
                    for code in VectorIndex::list_from_bytes(&bytes).map_err(|e| corrupt(namespace, e))? {
                        let (candidate, dest) = rule.decide(ivf, *src, src_centre, &code, &mut x);
                        candidates += u64::from(candidate);
                        if let Some(dest) = dest {
                            moving.entry(dest).or_default().push(VectorIndex { cluster_id: dest, ..code });
                        }
                    }
                }
                if !moving.is_empty() {
                    items.push((doc.clone(), moving));
                }
            }
            let (delta, n) = copy_into_batch(db, namespace, items).await?;
            handle.snapshot().apply_delta(&delta);
            Ok::<(u64, u64), crate::KVError>((candidates, n))
        })
        .await?;
        let (candidates, moved) = counts.into_iter().fold((skipped, 0), |(c, m), (dc, dm)| (c + dc, m + dm));
        MaintenanceCounters::add(&handle.counters().reassign_candidates, candidates);
        MaintenanceCounters::add(&handle.counters().reassign_moves, moved);
        // Barrier: the copies are durable before any source shrinks (one fsync
        // of the vector index WAL, which holds them).
        crash_point!();
        db.sync_vector_index_wal().await?;
        write_record(db, namespace, op, &record.with_state(JournalState::Copied)).await?;
        reuse = Some(selected);
    }
    handle.epoch_barrier().await;
    let selected = match reuse {
        Some(selected) => selected,
        None => select(db, namespace, &ivf, ops, parent_centre, children, &sources).await?.0,
    };
    let (rule, ivf) = (&rule, &ivf);
    in_batches(&selected, |batch| async move {
        let _docs = lock_docs_vectors(namespace, batch.iter().map(|(doc, _)| doc.as_slice())).await;
        shrink_sources(db, namespace, handle, ivf, rule, batch).await
    })
    .await?;
    MaintenanceCounters::add(&handle.counters().reassigns, 1);
    delete_record(db, namespace, op).await
}

/// A document with a moving chunk, and the sources (posting, rotated centre)
/// it has one in.
type Selected = (Vec<u8>, Vec<(u32, Vec<f32>)>);

/// The documents holding a chunk that moves, each with its sources, and how
/// many candidates the entries that move nothing hold (those that move are
/// counted again, exactly, under their lock). Reads the sources' entries, then decides
/// on the blocking pool: every chunk of the scope is reconstructed, and each
/// candidate is compared with every active centre.
async fn select(
    db: &AsyncDb,
    namespace: &str,
    ivf: &std::sync::Arc<NamespaceIvf>,
    ops: &MaintenanceOps,
    parent_centre: u32,
    children: [u32; 2],
    sources: &[(u32, Vec<f32>)],
) -> Result<(Vec<Selected>, u64), crate::KVError> {
    let ids: Vec<u32> = sources.iter().map(|(p, _)| *p).collect();
    let mut read = postings_entries(db, namespace, &ids).await?;
    let entries: Vec<_> = sources
        .iter()
        .map(|(src, centre)| (*src, centre.clone(), read.remove(src).unwrap_or_default()))
        .collect();
    let (ivf, ops) = (std::sync::Arc::clone(ivf), *ops);
    tokio::task::spawn_blocking(move || {
        let rule = Rule::new(&ivf, &ops, parent_centre, children).expect("checked by the caller");
        let mut x = Vec::new();
        let mut selected: BTreeMap<Vec<u8>, Vec<(u32, Vec<f32>)>> = BTreeMap::new();
        let mut candidates = 0;
        for (src, centre, docs) in entries {
            for (doc, codes) in docs {
                let decisions: Vec<(bool, Option<u32>)> = codes.iter().map(|c| rule.decide(&ivf, src, &centre, c, &mut x)).collect();
                if decisions.iter().any(|d| d.1.is_some()) {
                    selected.entry(doc).or_default().push((src, centre.clone()));
                } else {
                    candidates += decisions.iter().filter(|d| d.0).count() as u64;
                }
            }
        }
        (selected.into_iter().collect(), candidates)
    })
    .await
    .map_err(|e| crate::KVError::Io(std::io::Error::other(e)))
}

/// For each document of `batch`, remove from each source key every chunk the
/// rule moves and whose destination holds it: per document, the meta first
/// (WAL, one write for all its sources), then each source key, rewritten or
/// deleted (WAL). The caller holds every document's vector lock.
async fn shrink_sources(
    db: &AsyncDb,
    namespace: &str,
    handle: &PartitionHandle,
    ivf: &NamespaceIvf,
    rule: &Rule<'_>,
    batch: &[Selected],
) -> Result<(), crate::KVError> {
    let sparse = db.namespace(sparse_vectors_ns(namespace)).await?;
    let meta_ns = db.namespace(sparse_vectors_meta_ns(namespace)).await?;
    // Each (document, source) key, its codes, and which go where.
    struct Entry {
        doc: usize,
        src: u32,
        codes: Vec<VectorIndex>,
        by_dest: BTreeMap<u32, Vec<usize>>,
    }
    let keys: Vec<(usize, u32)> = batch
        .iter()
        .enumerate()
        .flat_map(|(d, (_, srcs))| srcs.iter().map(move |(src, _)| (d, *src)))
        .collect();
    let values = sparse
        .get_multiple(keys.iter().map(|&(d, src)| composite_key::encode(src, &batch[d].0)).collect())
        .await?;
    let mut x = Vec::new();
    let mut entries = Vec::new();
    for (&(d, src), value) in keys.iter().zip(values) {
        let Some(bytes) = value else { continue };
        let codes = VectorIndex::list_from_bytes(&bytes).map_err(|e| corrupt(namespace, e))?;
        let centre = &batch[d].1.iter().find(|(p, _)| *p == src).expect("listed").1;
        let mut by_dest: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (i, code) in codes.iter().enumerate() {
            if let (_, Some(dest)) = rule.decide(ivf, src, centre, code, &mut x) {
                by_dest.entry(dest).or_default().push(i);
            }
        }
        if !by_dest.is_empty() {
            entries.push(Entry { doc: d, src, codes, by_dest });
        }
    }
    // What the destinations hold, each read once.
    let dests: Vec<(usize, u32)> = entries
        .iter()
        .flat_map(|e| e.by_dest.keys().map(move |&dest| (e.doc, dest)))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let held_values = sparse
        .get_multiple(dests.iter().map(|&(d, dest)| composite_key::encode(dest, &batch[d].0)).collect())
        .await?;
    let mut held: HashMap<(usize, u32), (Vec<VectorIndex>, Vec<bool>)> = HashMap::new();
    for (&k, value) in dests.iter().zip(held_values) {
        if let Some(bytes) = value {
            let codes = VectorIndex::list_from_bytes(&bytes).map_err(|e| corrupt(namespace, e))?;
            let used = vec![false; codes.len()];
            held.insert(k, (codes, used));
        }
    }
    // Per entry, the codes its destinations hold go; the rest remain.
    let mut shrunk: BTreeMap<usize, Vec<(u32, Vec<VectorIndex>)>> = BTreeMap::new();
    for e in entries {
        let mut gone = vec![false; e.codes.len()];
        for (dest, idx) in &e.by_dest {
            let Some((codes, used)) = held.get_mut(&(e.doc, *dest)) else { continue };
            let wanted: Vec<&VectorIndex> = idx.iter().map(|&i| &e.codes[i]).collect();
            for (&i, ok) in idx.iter().zip(matched_with(&wanted, codes, used)) {
                gone[i] = ok;
            }
        }
        if gone.iter().any(|&g| g) {
            let remaining = e.codes.into_iter().zip(&gone).filter(|(_, g)| !**g).map(|(c, _)| c).collect();
            shrunk.entry(e.doc).or_default().push((e.src, remaining));
        }
    }
    if shrunk.is_empty() {
        return Ok(());
    }
    let metas = meta_ns.get_multiple(shrunk.keys().map(|&d| batch[d].0.clone()).collect()).await?;
    for ((d, sources), meta) in shrunk.into_iter().zip(metas) {
        let doc = &batch[d].0;
        // The meta first: a redo walks the sources' keys, so it comes back to a
        // key whose meta already shrank (the drain's rule 2 exception, for the
        // same reason). A document with no meta is being deleted: leave it to
        // the delete.
        let Some(mut meta) = meta.and_then(|b| decode_sparse_meta(&b)) else {
            continue;
        };
        let mut delta = PostingDelta::default();
        for (src, remaining) in &sources {
            let n = remaining.len() as u32;
            if let Some(i) = meta.postings.iter().position(|&(id, _)| id == *src) {
                let had = meta.postings[i].1;
                if had != n {
                    delta.removed.push((*src, had));
                    if n == 0 {
                        meta.postings.remove(i);
                    } else {
                        meta.postings[i].1 = n;
                        delta.added.push((*src, n));
                    }
                }
            }
        }
        if !delta.is_empty() {
            crash_point!();
            meta_ns.put(doc.clone(), encode_sparse_meta(meta.text_hash, &meta.postings)).await?;
            handle.snapshot().apply_delta(&delta);
        }
        for (src, remaining) in sources {
            let key = composite_key::encode(src, doc);
            crash_point!();
            if remaining.is_empty() {
                sparse.delete(key).await?;
            } else {
                sparse.put(key, VectorIndex::list_to_bytes(&remaining)).await?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{Scenario, full, handle, prepare};
    use super::*;

    #[test]
    fn sign_dot_matches_the_explicit_inner_product() {
        let mut state = 7u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state >> 11
        };
        for dim in [8usize, 16, 70, 768] {
            let v: Vec<f32> = (0..dim).map(|_| (next() % 2000) as f32 / 1000.0 - 1.0).collect();
            let packed: Vec<u64> = (0..dim.div_ceil(64)).map(|_| next()).collect();
            let explicit: f64 = (0..dim)
                .map(|i| f64::from(v[i]) * if (packed[i / 64] >> (i % 64)) & 1 == 1 { 1.0 } else { -1.0 })
                .sum();
            let got = SignDot::new(&v).dot(&packed);
            assert!((got - explicit).abs() < 1e-3, "dim {dim}: {got} vs {explicit}");
        }
    }

    /// The scorer decides exactly as the reconstruction does: for every split
    /// in a grown partition, every stored chunk gets the same decision both
    /// ways, with and without the error-bound gate.
    #[tokio::test]
    async fn the_scorer_decides_as_the_reconstruction_does() {
        let dir = tempfile::TempDir::new().unwrap();
        let (db, _) = prepare(&dir, Scenario::Reassign).await;
        let h = handle(&db).await;
        maintain_partition(&db, super::super::tests::NS, &h, &full(40, 0, 64)).await.unwrap();
        let ivf = h.snapshot();
        let mut checked = 0;
        let mut candidates = 0;
        for (&p, info) in ivf.posting_infos() {
            if info.state != PostingState::Retired {
                continue;
            }
            let children: Vec<u32> = ivf
                .posting_infos()
                .iter()
                .filter(|(_, i)| i.parent_id == Some(p) && i.state == PostingState::Active)
                .map(|(&id, _)| id)
                .collect();
            let [c0, c1] = children[..] else { continue };
            for gate in [false, true] {
                let ops = MaintenanceOps {
                    skip_uncertain_moves: gate,
                    ..full(40, 0, 64)
                };
                let rule = Rule::new(&ivf, &ops, info.centre_id, [c0, c1]).unwrap();
                let mut x = Vec::new();
                for &(src, centre) in &ivf.active_centres() {
                    for (doc, codes) in posting_entries(&db, super::super::tests::NS, src).await.unwrap() {
                        for code in &codes {
                            let fast = rule.decide(&ivf, src, centre, code, &mut x);
                            let slow = rule.decide_reconstructed(&ivf, src, centre, code);
                            assert_eq!(fast, slow, "posting {src}, doc {doc:?}, split of {p}, gate {gate}");
                            checked += 1;
                            candidates += usize::from(fast.0);
                        }
                    }
                }
                // Random codes too: random signs and scales, own centres drawn
                // from every centre, filed under every active posting.
                let centres = crate::semantic_search::IvfLayout::centre_ids(&*ivf);
                let active = ivf.active_centres();
                let mut state = 11u64 ^ u64::from(p);
                let mut next = || {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    state >> 11
                };
                for _ in 0..2000 {
                    let (src, centre) = active[next() as usize % active.len()];
                    let code = VectorIndex {
                        cluster_id: src,
                        centre_id: centres[next() as usize % centres.len()],
                        quantisation_style: crate::semantic_search::index::vector_index::QuantisationStyle::SingleBit,
                        addition_factor: 0.0,
                        scaling_factor: (next() % 1000) as f32 / 4000.0,
                        error_bound: (next() % 1000) as f32 / 20000.0,
                        packed_vector: vec![next() ^ (next() << 40); centre.len().div_ceil(64)],
                    };
                    let fast = rule.decide(&ivf, src, centre, &code, &mut x);
                    assert_eq!(
                        fast,
                        rule.decide_reconstructed(&ivf, src, centre, &code),
                        "random code, split of {p}, gate {gate}"
                    );
                    checked += 1;
                    candidates += usize::from(fast.0);
                }
            }
        }
        assert!(checked > 4000 && candidates > 10, "{checked} decisions, {candidates} candidates");
        db.shutdown().await.unwrap();
    }
}
