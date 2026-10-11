//! Maintaining a namespace's postings: splits, reassignment and merges, the
//! maintenance journal they run through, and recovery after a crash (design doc
//! *M3 split algorithm (LIRE)*, *Maintenance journal*; M3a and M3b).
//!
//! Three operations change which posting holds a chunk:
//!
//! - **Split** (M3a): a posting holding more than `split_limit` chunks is
//!   divided by balanced 2-means on its codes' reconstructions
//!   ([`crate::semantic_search::cluster::split`]) into two new postings, and
//!   retired.
//! - **Reassign** (M3b, [`reassign`]): after every split, the chunks of the two
//!   new postings and of the `reassign_range` postings nearest the old centre
//!   that LIRE's conditions select move to their nearest active posting.
//! - **Merge** (M3b, [`merge`]): a posting left with `merge_limit` chunks or
//!   fewer is folded into a nearby posting with room, and retired.
//!
//! A split and a merge both **drain** one posting: mark it `Draining` so nothing
//! new is routed to it, copy each chunk to where it belongs, delete the
//! originals, and retire it. They differ only in where a chunk goes
//! ([`DrainRule`]). A reassignment moves chunks between active postings, so its
//! source keys shrink instead of going away.
//!
//! The store has no transactions, so every operation is a sequence of
//! single-key writes recorded in a WAL-backed journal (`{ns}_ivf_journal`), and
//! a crash at any point is finished, never left half done. A drain:
//!
//! | Step | Writes | Durable how | Record after |
//! |---|---|---|---|
//! | 1. Plan | split: recount P, 2-means; write the record | WAL | `Planned` |
//! | 2. Prepare | split: append two centres, write P1, P2 `Active`; mark P `Draining` | WAL | `Planned` |
//! | 3. Publish | under the routing epoch's write guard: record `Published`, then the new snapshot | WAL | `Published` |
//! | 4. Copy | per document, under its vector lock: list each destination in its meta, write its chunks there | meta WAL, keys no-WAL | `Published` |
//! | 5. Barrier | flush the sparse and meta namespaces | flush | `Copied` |
//! | 6. Delete | after an epoch barrier, per document under its lock: drop P from its meta, delete `P‖doc` | WAL | `Copied` |
//! | 7. Retire | P `Retired`; publish; (a split) record its reassignment; delete the record | WAL | (gone) |
//!
//! A reassignment has no publish (routing does not change): `Planned` → copy →
//! barrier → `Copied` → epoch barrier → shrink each source key (WAL) → gone.
//!
//! Rules the steps depend on:
//!
//! 1. **Barrier before delete.** The copies are no-WAL; deletes and shrinks
//!    are WAL-backed and durable at once. Without the flush, a crash after a
//!    delete could lose the copy and the chunk would be gone for good.
//! 2. **The meta is always a superset of the live postings.** A document's
//!    meta (WAL-backed, as since M0-2) is the only record of which postings hold
//!    its chunks; a destination is added to it before the copy. A source is the
//!    exception: it loses its meta entry (or count) just before its key is
//!    deleted or shrunk, because a redo walks the source's own keys and finishes
//!    every one, listed or not.
//! 3. **Publish before copy, and record before publish.** Once `Published` is on
//!    disk recovery redoes the copy; until then nothing routes to new postings,
//!    so abandoning a `Planned` drain never strands an entry.
//! 4. **An epoch barrier before any original goes.** A search that started
//!    before a copy landed may scan the destination before the copy and the
//!    source after its delete; waiting out every search in flight
//!    ([`PartitionHandle::epoch_barrier`]) closes that window.
//! 5. **A source loses a chunk only if its destination holds it** (compared
//!    code by code, counting duplicates), so a redo never deletes a chunk whose
//!    copy did not land.
//! 6. **Recovery comes first**: it runs before the vector worker's first pass.
//!
//! Codes are never re-encoded: a moved code keeps its `centre_id` and only its
//! key changes. While a chunk is in both places a search scores it twice, and
//! MaxSim takes the maximum, so the result is unchanged.

use std::collections::{BTreeMap, HashMap};

use crate::AsyncDb;
use crate::semantic_search::cluster::split::{SplitParams, SplitRng, balanced_two_means};
use crate::semantic_search::index::composite_key;
use crate::semantic_search::index::vector_index::VectorIndex;
use crate::semantic_search::quantisation::rabitq::{reconstruct_single_bit, reconstruct_single_bit_into};
use crate::semantic_search::{IvfLayout, MaintenanceCounters, NamespaceIvf, PartitionHandle, Posting, PostingDelta, PostingOrigin, PostingState};
use crate::vector_kv::{
    decode_posting, decode_sparse_meta, encode_posting, encode_sparse_meta, f32s_to_bytes, ivf_centres_ns, ivf_journal_ns, ivf_postings_ns,
    lock_doc_vectors, sparse_vectors_meta_ns, sparse_vectors_ns,
};

/// A point where the crash tests stop the process: before a write.
macro_rules! crash_point {
    () => {
        #[cfg(test)]
        crate::vector_kv::crash_point::check()?;
    };
}

pub mod merge;
pub mod reassign;

/// What a namespace's maintenance runs with (its `maintenance` settings).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaintenanceOps {
    /// A posting holding more chunks than this splits (`2 · target_posting_size`).
    pub split_limit: u64,
    /// A posting holding this many chunks or fewer merges.
    pub merge_limit: u64,
    /// Nearest postings re-checked after a split (besides its two halves).
    pub reassign_range: usize,
    /// Move a chunk only when its code shows it clearly closer: the gain must
    /// exceed what the code's `error_bound` allows.
    pub skip_uncertain_moves: bool,
    /// Balanced 2-means parameters.
    pub params: SplitParams,
}

impl MaintenanceOps {
    /// Splits at `split_limit`, nothing else: no merges and no reassignment
    /// beyond the split's own halves.
    pub fn split_only(split_limit: u64) -> Self {
        Self {
            split_limit,
            merge_limit: 0,
            reassign_range: 0,
            skip_uncertain_moves: false,
            params: SplitParams::default(),
        }
    }
}

/// What [`split_posting`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitOutcome {
    /// The posting is within the limit (or is not active).
    NotNeeded,
    /// 2-means could not divide it (all its chunks alike); it is not tried again
    /// until its size changes.
    Unsplittable,
    /// Split into these two postings.
    Split {
        /// The new postings.
        children: [u32; 2],
    },
}

/// What [`maintain_partition`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MaintenanceReport {
    /// Postings split (each followed by a reassignment).
    pub splits: usize,
    /// Postings merged away.
    pub merges: usize,
}

const RECORD_VERSION: u8 = 1;
const KIND_SPLIT: u8 = 0;
const KIND_MERGE: u8 = 1;
const KIND_REASSIGN: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JournalState {
    Planned = 1,
    Published = 2,
    Copied = 3,
}

/// One maintenance operation, as the journal stores it. The plan is a rule
/// (where each chunk goes is decided from its code), so a record stays small
/// however many chunks it moves.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Op {
    /// Drain `posting` into `children`, each chunk to the nearer of `centres`.
    Split { posting: u32, centres: [u32; 2], children: [u32; 2] },
    /// Drain `posting` into `kept` (or, for a chunk nearer its old centre than
    /// to `kept`'s, its nearest active posting).
    Merge { posting: u32, kept: u32 },
    /// After `parent`'s split into `children`, move the chunks of `scope` that
    /// LIRE's conditions select to their nearest active posting.
    Reassign {
        parent: u32,
        parent_centre: u32,
        children: [u32; 2],
        scope: Vec<u32>,
    },
}

impl Op {
    /// The posting this operation drains, if it drains one.
    fn drained(&self) -> Option<u32> {
        match *self {
            Op::Split { posting, .. } | Op::Merge { posting, .. } => Some(posting),
            Op::Reassign { .. } => None,
        }
    }
}

/// One operation in the journal.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Record {
    state: JournalState,
    op: Op,
}

impl Record {
    fn with_state(&self, state: JournalState) -> Self {
        Self { state, op: self.op.clone() }
    }

    /// `version ‖ kind ‖ state ‖ ids` (ids 4B BE). A split's ids are
    /// `posting, centre × 2, child × 2`; a merge's `posting, kept`; a
    /// reassignment's `parent, parent centre, child × 2, scope…`.
    fn encode(&self) -> Vec<u8> {
        let (kind, ids): (u8, Vec<u32>) = match &self.op {
            Op::Split { posting, centres, children } => (KIND_SPLIT, vec![*posting, centres[0], centres[1], children[0], children[1]]),
            Op::Merge { posting, kept } => (KIND_MERGE, vec![*posting, *kept]),
            Op::Reassign {
                parent,
                parent_centre,
                children,
                scope,
            } => {
                let mut ids = vec![*parent, *parent_centre, children[0], children[1]];
                ids.extend(scope);
                (KIND_REASSIGN, ids)
            }
        };
        let mut v = vec![RECORD_VERSION, kind, self.state as u8];
        for id in ids {
            v.extend_from_slice(&id.to_be_bytes());
        }
        v
    }

    fn decode(v: &[u8]) -> Option<Self> {
        if v.len() < 3 || v[0] != RECORD_VERSION || !(v.len() - 3).is_multiple_of(4) {
            return None;
        }
        let state = match v[2] {
            1 => JournalState::Planned,
            2 => JournalState::Published,
            3 => JournalState::Copied,
            _ => return None,
        };
        let ids: Vec<u32> = v[3..].chunks_exact(4).map(|b| u32::from_be_bytes(b.try_into().unwrap())).collect();
        let op = match (v[1], ids.len()) {
            (KIND_SPLIT, 5) => Op::Split {
                posting: ids[0],
                centres: [ids[1], ids[2]],
                children: [ids[3], ids[4]],
            },
            (KIND_MERGE, 2) => Op::Merge {
                posting: ids[0],
                kept: ids[1],
            },
            (KIND_REASSIGN, n) if n >= 4 => Op::Reassign {
                parent: ids[0],
                parent_centre: ids[1],
                children: [ids[2], ids[3]],
                scope: ids[4..].to_vec(),
            },
            _ => return None,
        };
        Some(Self { state, op })
    }
}

fn corrupt(namespace: &str, what: impl std::fmt::Display) -> crate::KVError {
    crate::KVError::Serialization(format!("namespace '{namespace}': {what}"))
}

async fn write_record(db: &AsyncDb, namespace: &str, op: u64, record: &Record) -> Result<(), crate::KVError> {
    crash_point!();
    db.namespace(ivf_journal_ns(namespace))
        .await?
        .put(op.to_be_bytes().to_vec(), record.encode())
        .await
}

async fn delete_record(db: &AsyncDb, namespace: &str, op: u64) -> Result<(), crate::KVError> {
    crash_point!();
    db.namespace(ivf_journal_ns(namespace)).await?.delete(op.to_be_bytes().to_vec()).await
}

/// Every unfinished operation in `namespace`'s journal, in `op_id` order.
async fn records(db: &AsyncDb, namespace: &str) -> Result<Vec<(u64, Record)>, crate::KVError> {
    let mut out = Vec::new();
    for (k, v) in db.namespace(ivf_journal_ns(namespace)).await?.scan_prefix(Vec::new()).await? {
        let op = u64::from_be_bytes(k.as_slice().try_into().map_err(|_| corrupt(namespace, "malformed journal key"))?);
        out.push((
            op,
            Record::decode(&v).ok_or_else(|| corrupt(namespace, format!("malformed journal record {op}")))?,
        ));
    }
    out.sort_by_key(|(op, _)| *op);
    Ok(out)
}

/// The id a new operation takes: above every unfinished one (finished records
/// are deleted, so an id only has to beat those).
async fn next_op_id(db: &AsyncDb, namespace: &str) -> Result<u64, crate::KVError> {
    Ok(records(db, namespace).await?.last().map_or(0, |(op, _)| op + 1))
}

async fn write_posting(db: &AsyncDb, namespace: &str, p: &Posting) -> Result<(), crate::KVError> {
    crash_point!();
    db.namespace(ivf_postings_ns(namespace))
        .await?
        .put(p.posting_id.to_be_bytes().to_vec(), encode_posting(p))
        .await
}

async fn stored_posting(db: &AsyncDb, namespace: &str, id: u32) -> Result<Option<Posting>, crate::KVError> {
    match db.namespace(ivf_postings_ns(namespace)).await?.get(id.to_be_bytes().to_vec()).await? {
        Some(v) => decode_posting(id, &v)
            .map(Some)
            .ok_or_else(|| corrupt(namespace, format!("malformed posting {id}"))),
        None => Ok(None),
    }
}

async fn stored_centre(db: &AsyncDb, namespace: &str, id: u32) -> Result<Option<Vec<f32>>, crate::KVError> {
    match db.namespace(ivf_centres_ns(namespace)).await?.get(id.to_be_bytes().to_vec()).await? {
        Some(v) => crate::vector_kv::bytes_to_f32s(&v)
            .map(Some)
            .ok_or_else(|| corrupt(namespace, format!("malformed centre {id}"))),
        None => Ok(None),
    }
}

/// `posting`'s entries: `(doc id, codes)`.
async fn posting_entries(db: &AsyncDb, namespace: &str, posting: u32) -> Result<Vec<(Vec<u8>, Vec<VectorIndex>)>, crate::KVError> {
    let rows = db
        .namespace(sparse_vectors_ns(namespace))
        .await?
        .scan_prefix(posting.to_be_bytes().to_vec())
        .await?;
    let mut out = Vec::with_capacity(rows.len());
    for (k, v) in rows {
        let Some((_, doc)) = composite_key::decode(&k) else { continue };
        let codes = VectorIndex::list_from_bytes(&v).map_err(|e| corrupt(namespace, format!("posting {posting}: {e}")))?;
        out.push((doc.to_vec(), codes));
    }
    Ok(out)
}

/// Several postings' entries in one read (each LSM layer is read once for the
/// whole set, as a search's probe is): `posting → [(doc id, codes)]`.
async fn postings_entries(db: &AsyncDb, namespace: &str, postings: &[u32]) -> Result<HashMap<u32, Vec<(Vec<u8>, Vec<VectorIndex>)>>, crate::KVError> {
    let rows = db
        .namespace(sparse_vectors_ns(namespace))
        .await?
        .scan_prefixes_batch(postings.to_vec())
        .await?;
    let mut out = HashMap::with_capacity(rows.len());
    for (posting, rows) in rows {
        let mut entries = Vec::with_capacity(rows.len());
        for (k, v) in rows {
            let Some((_, doc)) = composite_key::decode(&k) else { continue };
            let codes = VectorIndex::list_from_bytes(&v).map_err(|e| corrupt(namespace, format!("posting {posting}: {e}")))?;
            entries.push((doc.to_vec(), codes));
        }
        out.insert(posting, entries);
    }
    Ok(out)
}

/// The posting as stored, as it is in `ivf` (for writing it back changed).
fn posting_of(ivf: &NamespaceIvf, id: u32) -> Option<Posting> {
    let info = ivf.posting_infos().get(&id)?;
    Some(Posting {
        posting_id: id,
        routing_centroid: ivf.routing_centroid(id).map(<[f32]>::to_vec).unwrap_or_default(),
        centre_id: info.centre_id,
        state: info.state,
        parent_id: info.parent_id,
        origin: info.origin,
    })
}

// ── Geometry, all in the rotated space codes live in ──

/// The reconstruction of `code`, or `None` for a code that cannot be
/// reconstructed (its centre is unknown, or it is not 1-bit).
fn reconstruct(ivf: &NamespaceIvf, code: &VectorIndex) -> Option<Vec<f32>> {
    reconstruct_single_bit(code, ivf.rotated_centre(code.centre_id)?)
}

/// [`reconstruct`] into `buf`, for loops over many codes. `false` for a code
/// that cannot be reconstructed.
fn reconstruct_into(ivf: &NamespaceIvf, code: &VectorIndex, buf: &mut Vec<f32>) -> bool {
    ivf.rotated_centre(code.centre_id)
        .is_some_and(|c| reconstruct_single_bit_into(code, c, buf))
}

/// Squared Euclidean distance (SIMD). Maintenance compares millions of these
/// per FiQA-sized namespace; a scalar f64 loop was most of reassignment's cost.
fn sq_dist(a: &[f32], b: &[f32]) -> f64 {
    use simsimd::SpatialSimilarity;
    f32::sqeuclidean(a, b).expect("equal dimensions")
}

/// Whether centre `a` is nearer to `x` than centre `b`. With `gate`
/// (`skip_uncertain_moves`), by more than `2 · error_bound · ‖a − b‖`, the most
/// the code's estimate of that comparison can be off (design doc step 6).
fn nearer(x: &[f32], a: &[f32], b: &[f32], code: &VectorIndex, gate: bool) -> bool {
    let (da, db) = (sq_dist(x, a), sq_dist(x, b));
    if da >= db {
        return false;
    }
    !gate || db - da > 2.0 * f64::from(code.error_bound) * sq_dist(a, b).sqrt()
}

/// The posting in `candidates` nearest `x` (ties to the first, so to the lower
/// id when `candidates` is in id order).
fn nearest<'a>(x: &[f32], candidates: &[(u32, &'a [f32])]) -> Option<(u32, &'a [f32])> {
    let mut best: Option<(f64, u32, &'a [f32])> = None;
    for &(id, c) in candidates {
        let d = sq_dist(x, c);
        if best.is_none_or(|(bd, _, _)| d < bd) {
            best = Some((d, id, c));
        }
    }
    best.map(|(_, id, c)| (id, c))
}

/// The `n` postings of `candidates` nearest `centre`, nearest first.
fn neighbours(centre: &[f32], candidates: &[(u32, &[f32])], n: usize) -> Vec<u32> {
    let mut by_dist: Vec<(f64, u32)> = candidates.iter().map(|&(id, c)| (sq_dist(centre, c), id)).collect();
    by_dist.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    by_dist.into_iter().take(n).map(|(_, id)| id).collect()
}

/// Whether two codes are the same chunk: everything but the posting they are
/// filed under.
fn same_code(a: &VectorIndex, b: &VectorIndex) -> bool {
    a.centre_id == b.centre_id
        && a.quantisation_style == b.quantisation_style
        && a.addition_factor.to_bits() == b.addition_factor.to_bits()
        && a.scaling_factor.to_bits() == b.scaling_factor.to_bits()
        && a.error_bound.to_bits() == b.error_bound.to_bits()
        && a.packed_vector == b.packed_vector
}

/// Match each of `wanted` against a distinct element of `held`: for each,
/// whether `held` has a copy not already matched. Duplicates count, so three
/// identical chunks need three copies.
fn matched(wanted: &[&VectorIndex], held: &[VectorIndex]) -> Vec<bool> {
    let mut used = vec![false; held.len()];
    wanted
        .iter()
        .map(|w| match (0..held.len()).find(|&i| !used[i] && same_code(w, &held[i])) {
            Some(i) => {
                used[i] = true;
                true
            }
            None => false,
        })
        .collect()
}

/// Where a draining posting's chunks go.
enum DrainRule<'a> {
    /// A split: the nearer of the two new centres.
    Split { children: [u32; 2], centres: [&'a [f32]; 2] },
    /// A merge into `kept`; a chunk nearer its old centre (`from`) than to
    /// `kept`'s goes to its nearest active posting instead.
    Merge {
        kept: u32,
        kept_centre: &'a [f32],
        from_centre: &'a [f32],
        active: Vec<(u32, &'a [f32])>,
        gate: bool,
    },
}

impl DrainRule<'_> {
    fn destination(&self, ivf: &NamespaceIvf, code: &VectorIndex) -> u32 {
        match self {
            DrainRule::Split { children, centres } => match reconstruct(ivf, code) {
                // Side 0 for a code that cannot be reconstructed, so it is kept.
                Some(x) => children[usize::from(sq_dist(&x, centres[1]) < sq_dist(&x, centres[0]))],
                None => children[0],
            },
            DrainRule::Merge {
                kept,
                kept_centre,
                from_centre,
                active,
                gate,
            } => {
                let Some(x) = reconstruct(ivf, code) else { return *kept };
                if !nearer(&x, from_centre, kept_centre, code, *gate) {
                    return *kept;
                }
                match nearest(&x, active) {
                    Some((n, c)) if n != *kept && nearer(&x, c, kept_centre, code, *gate) => n,
                    _ => *kept,
                }
            }
        }
    }
}

/// Add `moving` (`destination → codes`, each already filed under its
/// destination) to document `doc`'s keys and meta: the meta first (WAL), then
/// the keys (no-WAL), each key gaining only the codes it does not already hold.
/// Runs under the document's vector lock. A document with no meta is being
/// deleted and is skipped (rule 2). Returns the count change and how many codes
/// were newly written.
async fn copy_into(
    db: &AsyncDb,
    namespace: &str,
    doc: &[u8],
    moving: BTreeMap<u32, Vec<VectorIndex>>,
) -> Result<(PostingDelta, u64), crate::KVError> {
    let sparse = db.namespace(sparse_vectors_ns(namespace)).await?;
    let meta_ns = db.namespace(sparse_vectors_meta_ns(namespace)).await?;
    let Some(mut meta) = meta_ns.get(doc.to_vec()).await?.and_then(|b| decode_sparse_meta(&b)) else {
        return Ok((PostingDelta::default(), 0));
    };
    let mut writes = Vec::new();
    let mut copied = 0;
    for (dest, codes) in moving {
        let key = composite_key::encode(dest, doc);
        let held = match sparse.get(key.clone()).await? {
            Some(b) => VectorIndex::list_from_bytes(&b).map_err(|e| corrupt(namespace, e))?,
            None => Vec::new(),
        };
        let refs: Vec<&VectorIndex> = codes.iter().collect();
        let present = matched(&refs, &held);
        let mut merged = held.clone();
        merged.extend(codes.iter().zip(&present).filter(|(_, p)| !**p).map(|(c, _)| c.clone()));
        let new = merged.len() - held.len();
        copied += new as u64;
        writes.push((dest, key, merged, new > 0));
    }
    // The meta: each destination at the size its key will have.
    let mut delta = PostingDelta::default();
    for (dest, _, merged, _) in &writes {
        let n = merged.len() as u32;
        match meta.postings.iter_mut().find(|(id, _)| id == dest) {
            Some((_, have)) if *have == n => {}
            Some((_, have)) => {
                delta.removed.push((*dest, *have));
                delta.added.push((*dest, n));
                *have = n;
            }
            None => {
                meta.postings.push((*dest, n));
                delta.added.push((*dest, n));
            }
        }
    }
    if !delta.is_empty() {
        crash_point!();
        meta_ns.put(doc.to_vec(), encode_sparse_meta(meta.text_hash, &meta.postings)).await?;
    }
    for (_, key, merged, changed) in writes {
        if changed {
            crash_point!();
            sparse.put_no_wal(key, VectorIndex::list_to_bytes(&merged)).await?;
        }
    }
    Ok((delta, copied))
}

/// Split `posting` if it holds more than `ops.split_limit` chunks, then reassign
/// around it. Takes the namespace's maintenance lock, so maintenance of one
/// namespace never overlaps.
pub async fn split_posting(
    db: &AsyncDb,
    namespace: &str,
    handle: &PartitionHandle,
    posting: u32,
    ops: &MaintenanceOps,
) -> Result<SplitOutcome, crate::KVError> {
    let _maintenance = handle.lock_maintenance().await;
    split_locked(db, namespace, handle, posting, ops).await
}

/// Split every active posting over the split limit and merge every one at or
/// under the merge limit, until none is left (each split followed by its
/// reassignment, which may in turn leave postings to split or merge). Postings
/// that cannot be split or merged at their current size are not retried.
pub async fn maintain_partition(
    db: &AsyncDb,
    namespace: &str,
    handle: &PartitionHandle,
    ops: &MaintenanceOps,
) -> Result<MaintenanceReport, crate::KVError> {
    let _maintenance = handle.lock_maintenance().await;
    let mut report = MaintenanceReport::default();
    // A guard against splits and merges undoing each other for ever; far above
    // anything a real partition needs.
    let cap = 16 * (handle.snapshot().postings() + 64);
    let mut done = 0;
    loop {
        let ivf = handle.snapshot();
        let active: Vec<(u64, u32)> = ivf
            .posting_infos()
            .iter()
            .filter(|(_, info)| info.state == PostingState::Active)
            .map(|(&id, _)| (ivf.posting_chunks(id), id))
            .collect();
        let mut over: Vec<(u64, u32)> = active
            .iter()
            .copied()
            .filter(|&(chunks, id)| chunks > ops.split_limit && !handle.is_unsplittable(id, chunks))
            .collect();
        over.sort_unstable_by(|a, b| b.cmp(a)); // largest first
        let mut progressed = false;
        for (_, id) in over {
            if let SplitOutcome::Split { .. } = split_locked(db, namespace, handle, id, ops).await? {
                report.splits += 1;
                progressed = true;
                done += 1;
            }
        }
        let ivf = handle.snapshot();
        let mut under: Vec<(u64, u32)> = ivf
            .posting_infos()
            .iter()
            .filter(|(_, info)| info.state == PostingState::Active)
            .map(|(&id, _)| (ivf.posting_chunks(id), id))
            .filter(|&(chunks, id)| chunks <= ops.merge_limit && !handle.is_unmergeable(id, chunks))
            .collect();
        under.sort_unstable(); // smallest first
        for (_, id) in under {
            if merge::merge_locked(db, namespace, handle, id, ops).await? {
                report.merges += 1;
                progressed = true;
                done += 1;
            }
        }
        if !progressed {
            return Ok(report);
        }
        if done >= cap {
            log::warn!(
                "partition maintenance: namespace='{namespace}' stopped after {done} operations ({} splits, {} merges); resumes on the next request",
                report.splits,
                report.merges
            );
            return Ok(report);
        }
    }
}

async fn split_locked(
    db: &AsyncDb,
    namespace: &str,
    handle: &PartitionHandle,
    posting: u32,
    ops: &MaintenanceOps,
) -> Result<SplitOutcome, crate::KVError> {
    let ivf = handle.snapshot();
    if ivf.posting_infos().get(&posting).map(|i| i.state) != Some(PostingState::Active) {
        return Ok(SplitOutcome::NotNeeded);
    }
    // 1. Plan: recount from the stored codes and divide them.
    let entries = posting_entries(db, namespace, posting).await?;
    let chunks: u64 = entries.iter().map(|(_, c)| c.len() as u64).sum();
    if chunks <= ops.split_limit {
        return Ok(SplitOutcome::NotNeeded);
    }
    if handle.is_unsplittable(posting, chunks) {
        return Ok(SplitOutcome::Unsplittable);
    }
    let points: Vec<Vec<f32>> = entries
        .iter()
        .flat_map(|(_, codes)| codes.iter().filter_map(|c| reconstruct(&ivf, c)))
        .collect();
    let op = next_op_id(db, namespace).await?;
    let mut rng = SplitRng::new(u64::from(posting).rotate_left(32) ^ op ^ chunks);
    let params = ops.params;
    // Off the async threads (and off the rayon pool searches share).
    let result = tokio::task::spawn_blocking(move || balanced_two_means(&points, &params, &mut rng))
        .await
        .map_err(|e| crate::KVError::Io(std::io::Error::other(e)))?;
    // A split whose smaller half is at or under the merge limit is undone by the
    // next merge: the half is merged away, and its chunks, nearer their own old
    // centre than the merge target's, are routed back to the larger half, which
    // is over the limit again. Measured on SciFact (gemma): 2-means isolated 3
    // outlier chunks from a 262-chunk posting, and splits and merges took turns
    // 900 times in one maintenance pass. So it counts as unsplittable at this
    // size, like a posting 2-means cannot divide at all.
    let smaller = result.as_ref().map_or(0, |r| {
        let ones = r.labels.iter().filter(|&&l| l == 1).count() as u64;
        ones.min(r.labels.len() as u64 - ones)
    });
    let Some(result) = result.filter(|_| smaller > ops.merge_limit) else {
        handle.mark_unsplittable(posting, chunks);
        return Ok(SplitOutcome::Unsplittable);
    };
    let (max_posting, max_centre) = ivf.max_ids();
    let centre_ids = [max_centre + 1, max_centre + 2];
    let children_ids = [max_posting + 1, max_posting + 2];
    let record = Record {
        state: JournalState::Planned,
        op: Op::Split {
            posting,
            centres: centre_ids,
            children: children_ids,
        },
    };
    write_record(db, namespace, op, &record).await?;

    // 2. Prepare: centres and children (not yet routable), P draining.
    let centres = [ivf.unrotate(&result.centres[0]), ivf.unrotate(&result.centres[1])];
    let centres_ns = db.namespace(ivf_centres_ns(namespace)).await?;
    for (id, c) in centre_ids.iter().zip(&centres) {
        crash_point!();
        centres_ns.put(id.to_be_bytes().to_vec(), f32s_to_bytes(c)).await?;
    }
    let mut changed: Vec<Posting> = (0..2)
        .map(|k| Posting {
            posting_id: children_ids[k],
            routing_centroid: centres[k].clone(),
            centre_id: centre_ids[k],
            state: PostingState::Active,
            parent_id: Some(posting),
            origin: PostingOrigin::Split,
        })
        .collect();
    for child in &changed {
        write_posting(db, namespace, child).await?;
    }
    let mut parent = posting_of(&ivf, posting).ok_or_else(|| corrupt(namespace, format!("posting {posting} vanished")))?;
    parent.state = PostingState::Draining;
    write_posting(db, namespace, &parent).await?;
    changed.push(parent);

    // 3. Publish: recorded first, so recovery redoes the copy once anything can
    //    have routed to the children.
    let new_centres = [(centre_ids[0], centres[0].clone()), (centre_ids[1], centres[1].clone())];
    publish_drain(db, namespace, handle, op, &record, &new_centres, &changed).await?;
    finish_drain(db, namespace, handle, ops, op, record.with_state(JournalState::Published)).await?;
    Ok(SplitOutcome::Split { children: children_ids })
}

/// Step 3 of a drain: record `Published`, then install the snapshot with
/// `new_centres` and `changed`, both under the routing epoch's write guard.
async fn publish_drain(
    db: &AsyncDb,
    namespace: &str,
    handle: &PartitionHandle,
    op: u64,
    record: &Record,
    new_centres: &[(u32, Vec<f32>)],
    changed: &[Posting],
) -> Result<(), crate::KVError> {
    let guard = handle.write_epoch().await;
    write_record(db, namespace, op, &record.with_state(JournalState::Published)).await?;
    let next = handle.snapshot().evolve(new_centres, changed).map_err(|e| corrupt(namespace, e))?;
    handle.publish(&guard, next);
    Ok(())
}

/// Steps 4–7 of a published drain (also what recovery runs). A split ends by
/// recording and running its reassignment.
async fn finish_drain(
    db: &AsyncDb,
    namespace: &str,
    handle: &PartitionHandle,
    ops: &MaintenanceOps,
    op: u64,
    record: Record,
) -> Result<(), crate::KVError> {
    let drained = record.op.drained().expect("a drain");
    if record.state == JournalState::Published {
        let copied = drain_copy(db, namespace, handle, ops, &record.op).await?;
        MaintenanceCounters::add(&handle.counters().drain_moves, copied);
        // 5. Barrier: the copies are durable before any delete.
        crash_point!();
        db.flush_namespaces(vec![sparse_vectors_ns(namespace), sparse_vectors_meta_ns(namespace)])
            .await?;
        write_record(db, namespace, op, &record.with_state(JournalState::Copied)).await?;
    }
    handle.epoch_barrier().await;
    drain_delete(db, namespace, handle, drained).await?;
    // 7. Retire.
    let mut posting = stored_posting(db, namespace, drained)
        .await?
        .ok_or_else(|| corrupt(namespace, format!("drained posting {drained} is missing")))?;
    posting.state = PostingState::Retired;
    write_posting(db, namespace, &posting).await?;
    {
        let guard = handle.write_epoch().await;
        let next = handle.snapshot().evolve(&[], &[posting]).map_err(|e| corrupt(namespace, e))?;
        handle.publish(&guard, next);
    }
    match record.op {
        Op::Split { posting, children, .. } => {
            MaintenanceCounters::add(&handle.counters().splits, 1);
            // The reassignment is recorded before the split's record goes, so a
            // crash between the two still reassigns.
            let reassign = reassign::plan(&handle.snapshot(), ops, posting, children)
                .ok_or_else(|| corrupt(namespace, format!("split of posting {posting}: its centre or a child's is missing")))?;
            let next = op + 1;
            write_record(db, namespace, next, &reassign).await?;
            delete_record(db, namespace, op).await?;
            reassign::run(db, namespace, handle, ops, next, reassign).await
        }
        Op::Merge { .. } => {
            MaintenanceCounters::add(&handle.counters().merges, 1);
            delete_record(db, namespace, op).await
        }
        Op::Reassign { .. } => unreachable!("not a drain"),
    }
}

/// 4. Copy each document's chunks from the drained posting to where the rule
///    sends them. Returns how many codes were newly written.
async fn drain_copy(db: &AsyncDb, namespace: &str, handle: &PartitionHandle, ops: &MaintenanceOps, op: &Op) -> Result<u64, crate::KVError> {
    let ivf = handle.snapshot();
    let missing = |what: &str| corrupt(namespace, format!("{op:?}: {what} is missing"));
    let rule = match *op {
        Op::Split { centres, children, .. } => {
            let c0 = ivf.rotated_centre(centres[0]).ok_or_else(|| missing("a new centre"))?;
            let c1 = ivf.rotated_centre(centres[1]).ok_or_else(|| missing("a new centre"))?;
            DrainRule::Split { children, centres: [c0, c1] }
        }
        Op::Merge { posting, kept } => DrainRule::Merge {
            kept,
            kept_centre: ivf.rotated_posting_centre(kept).ok_or_else(|| missing("the kept posting's centre"))?,
            from_centre: ivf
                .rotated_posting_centre(posting)
                .ok_or_else(|| missing("the merged posting's centre"))?,
            active: ivf.active_centres(),
            gate: ops.skip_uncertain_moves,
        },
        Op::Reassign { .. } => unreachable!("not a drain"),
    };
    let drained = op.drained().expect("a drain");
    let sparse = db.namespace(sparse_vectors_ns(namespace)).await?;
    let mut copied = 0;
    for (doc, _) in posting_entries(db, namespace, drained).await? {
        let _doc = lock_doc_vectors(namespace, &doc).await;
        // Re-read under the lock: an upsert or delete may have moved it on.
        let Some(bytes) = sparse.get(composite_key::encode(drained, &doc)).await? else {
            continue;
        };
        let mut moving: BTreeMap<u32, Vec<VectorIndex>> = BTreeMap::new();
        for code in VectorIndex::list_from_bytes(&bytes).map_err(|e| corrupt(namespace, e))? {
            let dest = rule.destination(&ivf, &code);
            moving.entry(dest).or_default().push(VectorIndex { cluster_id: dest, ..code });
        }
        let (delta, n) = copy_into(db, namespace, &doc, moving).await?;
        handle.snapshot().apply_delta(&delta);
        copied += n;
    }
    Ok(copied)
}

/// 6. Delete the drained posting's keys, and the posting from each document's
///    meta.
async fn drain_delete(db: &AsyncDb, namespace: &str, handle: &PartitionHandle, drained: u32) -> Result<(), crate::KVError> {
    let sparse = db.namespace(sparse_vectors_ns(namespace)).await?;
    let meta_ns = db.namespace(sparse_vectors_meta_ns(namespace)).await?;
    // Nothing routes to the posting any more, so its key set only shrinks; one
    // more pass covers anything a concurrent writer left.
    for _ in 0..2 {
        let keys = sparse.scan_prefix(drained.to_be_bytes().to_vec()).await?;
        if keys.is_empty() {
            return Ok(());
        }
        for (key, _) in keys {
            let Some((_, doc)) = composite_key::decode(&key) else { continue };
            let doc = doc.to_vec();
            let _doc = lock_doc_vectors(namespace, &doc).await;
            // The meta first, then the key. The other order lost track of the
            // meta's entry: a crash between the two left a meta naming the
            // posting with no key, and the redo, which walks the posting's keys,
            // never came back to it. This order is safe only here: the posting
            // is draining, and the redo deletes every key still under it whether
            // or not a meta lists it.
            if let Some(mut meta) = meta_ns.get(doc.clone()).await?.and_then(|b| decode_sparse_meta(&b))
                && let Some(i) = meta.postings.iter().position(|&(id, _)| id == drained)
            {
                let (_, chunks) = meta.postings.remove(i);
                crash_point!();
                meta_ns.put(doc.clone(), encode_sparse_meta(meta.text_hash, &meta.postings)).await?;
                handle.snapshot().apply_delta(&PostingDelta {
                    added: vec![],
                    removed: vec![(drained, chunks)],
                });
            }
            crash_point!();
            sparse.delete(key).await?;
        }
    }
    Ok(())
}

/// Finish or abandon every operation a crash left in `namespace`'s journal.
/// Runs before anything else writes the namespace's vectors. Returns how many
/// operations it found.
pub async fn recover(db: &AsyncDb, namespace: &str, handle: &PartitionHandle, ops: &MaintenanceOps) -> Result<usize, crate::KVError> {
    let _maintenance = handle.lock_maintenance().await;
    let found = records(db, namespace).await?;
    for (op, record) in &found {
        // A split's record may already be gone if its reassignment, recorded
        // after it, was found too; finishing a drain re-checks the store.
        match (&record.op, record.state) {
            (Op::Reassign { .. }, _) => reassign::run(db, namespace, handle, ops, *op, record.clone()).await?,
            (_, JournalState::Planned) => abandon(db, namespace, handle, *op, record).await?,
            (_, JournalState::Published | JournalState::Copied) => finish_drain(db, namespace, handle, ops, *op, record.clone()).await?,
        }
    }
    Ok(found.len())
}

/// A drain that never published: nothing was routed to new postings, so they
/// are retired empty and the drained posting becomes active again. Centres it
/// appended stay, unreferenced.
async fn abandon(db: &AsyncDb, namespace: &str, handle: &PartitionHandle, op: u64, record: &Record) -> Result<(), crate::KVError> {
    let mut changed = Vec::new();
    let (new_ids, centres) = match record.op {
        Op::Split { centres, children, .. } => (children.to_vec(), centres.to_vec()),
        _ => (vec![], vec![]),
    };
    for id in new_ids {
        if let Some(mut child) = stored_posting(db, namespace, id).await? {
            child.state = PostingState::Retired;
            write_posting(db, namespace, &child).await?;
            changed.push(child);
        }
    }
    let drained = record.op.drained().expect("a drain");
    if let Some(mut posting) = stored_posting(db, namespace, drained).await?
        && posting.state != PostingState::Active
    {
        posting.state = PostingState::Active;
        write_posting(db, namespace, &posting).await?;
        changed.push(posting);
    }
    let mut new_centres = Vec::new();
    let ivf = handle.snapshot();
    for id in centres {
        if ivf.centre(id).is_none()
            && let Some(c) = stored_centre(db, namespace, id).await?
        {
            new_centres.push((id, c));
        }
    }
    if !changed.is_empty() || !new_centres.is_empty() {
        let guard = handle.write_epoch().await;
        let next = handle.snapshot().evolve(&new_centres, &changed).map_err(|e| corrupt(namespace, e))?;
        handle.publish(&guard, next);
    }
    delete_record(db, namespace, op).await
}

pub mod health;

#[cfg(test)]
mod tests;

/// Postings and their stored chunk counts, for tests and health checks:
/// `posting → chunks`, counted from the stored keys' values.
pub async fn recount_chunks(db: &AsyncDb, namespace: &str) -> Result<HashMap<u32, u64>, crate::KVError> {
    let mut out = HashMap::new();
    for (k, v) in db.namespace(sparse_vectors_ns(namespace)).await?.scan_prefix(Vec::new()).await? {
        if let Some((p, _)) = composite_key::decode(&k) {
            let n = VectorIndex::list_from_bytes(&v).map_err(|e| corrupt(namespace, e))?.len() as u64;
            *out.entry(p).or_insert(0) += n;
        }
    }
    Ok(out)
}
