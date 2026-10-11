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

use std::collections::BTreeMap;

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

/// LIRE's conditions for one reassignment.
struct Rule<'a> {
    parent: &'a [f32],
    children: [(u32, &'a [f32]); 2],
    active: Vec<(u32, &'a [f32])>,
    gate: bool,
}

impl<'a> Rule<'a> {
    fn new(ivf: &'a NamespaceIvf, ops: &MaintenanceOps, parent_centre: u32, children: [u32; 2]) -> Option<Self> {
        Some(Self {
            parent: ivf.rotated_centre(parent_centre)?,
            children: [
                (children[0], ivf.rotated_posting_centre(children[0])?),
                (children[1], ivf.rotated_posting_centre(children[1])?),
            ],
            active: ivf.active_centres(),
            gate: ops.skip_uncertain_moves,
        })
    }

    /// For a chunk in posting `src` (centre `src_centre`): whether it is a
    /// candidate, and where it moves (`None` to stay).
    /// `x` is scratch space for the code's reconstruction.
    fn decide(&self, ivf: &NamespaceIvf, src: u32, src_centre: &[f32], code: &VectorIndex, x: &mut Vec<f32>) -> (bool, Option<u32>) {
        if !reconstruct_into(ivf, code, x) {
            return (false, None);
        }
        let x = x.as_slice();
        let candidate = if self.children.iter().any(|&(id, _)| id == src) {
            nearer(x, self.parent, src_centre, code, self.gate)
        } else {
            let (_, c_min) = nearest(x, &self.children).expect("two halves");
            nearer(x, c_min, self.parent, code, self.gate) && nearer(x, c_min, src_centre, code, self.gate)
        };
        if !candidate {
            return (false, None);
        }
        match nearest(x, &self.active) {
            Some((n, c)) if n != src && nearer(x, c, src_centre, code, self.gate) => (true, Some(n)),
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
        let (selected, mut candidates) = select(db, namespace, &ivf, ops, parent_centre, children, &sources).await?;
        let mut x = Vec::new();
        let mut moved = 0;
        for (src, src_centre, doc) in &selected {
            let _doc = lock_doc_vectors(namespace, doc).await;
            // Re-read under the lock: an upsert or delete may have moved it on.
            let Some(bytes) = db
                .namespace(sparse_vectors_ns(namespace))
                .await?
                .get(composite_key::encode(*src, doc))
                .await?
            else {
                continue;
            };
            let mut moving: BTreeMap<u32, Vec<VectorIndex>> = BTreeMap::new();
            for code in VectorIndex::list_from_bytes(&bytes).map_err(|e| corrupt(namespace, e))? {
                let (candidate, dest) = rule.decide(&ivf, *src, src_centre, &code, &mut x);
                candidates += u64::from(candidate);
                if let Some(dest) = dest {
                    moving.entry(dest).or_default().push(VectorIndex { cluster_id: dest, ..code });
                }
            }
            if moving.is_empty() {
                continue;
            }
            let (delta, n) = copy_into(db, namespace, doc, moving).await?;
            handle.snapshot().apply_delta(&delta);
            moved += n;
        }
        MaintenanceCounters::add(&handle.counters().reassign_candidates, candidates);
        MaintenanceCounters::add(&handle.counters().reassign_moves, moved);
        // Barrier: the copies are durable before any source shrinks.
        crash_point!();
        db.flush_namespaces(vec![sparse_vectors_ns(namespace), sparse_vectors_meta_ns(namespace)])
            .await?;
        write_record(db, namespace, op, &record.with_state(JournalState::Copied)).await?;
        reuse = Some(selected);
    }
    handle.epoch_barrier().await;
    let selected = match reuse {
        Some(selected) => selected,
        None => select(db, namespace, &ivf, ops, parent_centre, children, &sources).await?.0,
    };
    for (src, src_centre, doc) in &selected {
        let _doc = lock_doc_vectors(namespace, doc).await;
        shrink_source(db, namespace, handle, &ivf, &rule, *src, src_centre, doc).await?;
    }
    MaintenanceCounters::add(&handle.counters().reassigns, 1);
    delete_record(db, namespace, op).await
}

/// A source posting, its rotated centre, and one document in it.
type Selected = (u32, Vec<f32>, Vec<u8>);

/// The (source, document) entries holding a chunk that moves, and how many
/// candidates the entries that move nothing hold (those that move are counted
/// again, exactly, under their lock). Reads the sources' entries, then decides
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
    let mut entries = Vec::with_capacity(sources.len());
    for (src, centre) in sources {
        entries.push((*src, centre.clone(), posting_entries(db, namespace, *src).await?));
    }
    let (ivf, ops) = (std::sync::Arc::clone(ivf), *ops);
    tokio::task::spawn_blocking(move || {
        let rule = Rule::new(&ivf, &ops, parent_centre, children).expect("checked by the caller");
        let mut x = Vec::new();
        let mut selected = Vec::new();
        let mut candidates = 0;
        for (src, centre, docs) in entries {
            for (doc, codes) in docs {
                let decisions: Vec<(bool, Option<u32>)> = codes.iter().map(|c| rule.decide(&ivf, src, &centre, c, &mut x)).collect();
                if decisions.iter().any(|d| d.1.is_some()) {
                    selected.push((src, centre.clone(), doc));
                } else {
                    candidates += decisions.iter().filter(|d| d.0).count() as u64;
                }
            }
        }
        (selected, candidates)
    })
    .await
    .map_err(|e| crate::KVError::Io(std::io::Error::other(e)))
}

/// Remove from `src‖doc` every chunk the rule moves and whose destination holds
/// it: the meta first (WAL), then the key, rewritten or deleted (WAL). Runs
/// under the document's vector lock.
#[allow(clippy::too_many_arguments)]
async fn shrink_source(
    db: &AsyncDb,
    namespace: &str,
    handle: &PartitionHandle,
    ivf: &NamespaceIvf,
    rule: &Rule<'_>,
    src: u32,
    src_centre: &[f32],
    doc: &[u8],
) -> Result<(), crate::KVError> {
    let sparse = db.namespace(sparse_vectors_ns(namespace)).await?;
    let meta_ns = db.namespace(sparse_vectors_meta_ns(namespace)).await?;
    let key = composite_key::encode(src, doc);
    let Some(bytes) = sparse.get(key.clone()).await? else { return Ok(()) };
    let codes = VectorIndex::list_from_bytes(&bytes).map_err(|e| corrupt(namespace, e))?;
    // Which codes go where, then which of them their destination holds.
    let mut by_dest: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    let mut x = Vec::new();
    for (i, code) in codes.iter().enumerate() {
        if let (_, Some(dest)) = rule.decide(ivf, src, src_centre, code, &mut x) {
            by_dest.entry(dest).or_default().push(i);
        }
    }
    let mut gone = vec![false; codes.len()];
    for (dest, idx) in by_dest {
        let held = match sparse.get(composite_key::encode(dest, doc)).await? {
            Some(b) => VectorIndex::list_from_bytes(&b).map_err(|e| corrupt(namespace, e))?,
            None => continue,
        };
        let wanted: Vec<&VectorIndex> = idx.iter().map(|&i| &codes[i]).collect();
        for (&i, ok) in idx.iter().zip(matched(&wanted, &held)) {
            gone[i] = ok;
        }
    }
    if !gone.iter().any(|&g| g) {
        return Ok(());
    }
    let remaining: Vec<VectorIndex> = codes.into_iter().zip(&gone).filter(|(_, g)| !**g).map(|(c, _)| c).collect();
    // The meta first: a redo walks the source's keys, so it comes back to a key
    // whose meta already shrank (the drain's rule 2 exception, for the same
    // reason). A document with no meta is being deleted: leave it to the delete.
    let Some(mut meta) = meta_ns.get(doc.to_vec()).await?.and_then(|b| decode_sparse_meta(&b)) else {
        return Ok(());
    };
    let mut delta = PostingDelta::default();
    let n = remaining.len() as u32;
    if let Some(i) = meta.postings.iter().position(|&(id, _)| id == src) {
        let had = meta.postings[i].1;
        if had != n {
            delta.removed.push((src, had));
            if n == 0 {
                meta.postings.remove(i);
            } else {
                meta.postings[i].1 = n;
                delta.added.push((src, n));
            }
            crash_point!();
            meta_ns.put(doc.to_vec(), encode_sparse_meta(meta.text_hash, &meta.postings)).await?;
            handle.snapshot().apply_delta(&delta);
        }
    }
    crash_point!();
    if remaining.is_empty() {
        sparse.delete(key).await
    } else {
        sparse.put(key, VectorIndex::list_to_bytes(&remaining)).await
    }
}
