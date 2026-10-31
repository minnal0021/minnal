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
//! (`DrainRule`). A reassignment moves chunks between active postings, so its
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
//! | 3. Publish | record `Published`, then install the new snapshot and wait out every write and search that entered by the old one | WAL | `Published` |
//! | 4. Copy | per document, under its vector lock: list each destination in its meta, write its chunks there | vector index WAL, one sync per batch | `Published` |
//! | 5. Barrier | one fsync of the vector index WAL (a no-op after the last batch's) | fsync | `Copied` |
//! | 6. Delete | after an epoch barrier, per document under its lock: drop P from its meta, delete `P‖doc` | vector index WAL, one sync per batch | `Copied` |
//! | 7. Retire | P `Retired`; publish; (a split) record its reassignment; delete the record | WAL | (gone) |
//!
//! A reassignment has no publish (routing does not change): `Planned` → copy →
//! barrier → `Copied` → epoch barrier → shrink each source key → gone.
//!
//! Every per-document write (meta and keys, both namespaces) is logged in the
//! vector index WAL, and each batch syncs it before releasing its documents'
//! locks (`sync_batch`). The journal records, centres and postings are
//! main-WAL writes, durable on return. A crash keeps a prefix of the vector
//! index WAL (a flush syncs it before writing L0 files), so the orders below
//! hold across the two namespaces exactly as if each write were durable on
//! return.
//!
//! Rules the steps depend on:
//!
//! 1. **Barrier before delete.** The copies must be durable before the record
//!    says `Copied`; after it, recovery redoes only the delete, and a lost copy
//!    would leave the chunk gone for good.
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
//! 6. **Recovery comes first.** Nothing uses a partition whose journal is
//!    unfinished: a new handle needs recovery, and so does one after any
//!    maintenance error; [`ensure_recovered`] runs it before the partition is
//!    handed out, and every maintenance entry point before it starts
//!    (`prepare`). An operation started on top of an unfinished one would
//!    take ids the unfinished one already used.
//! 7. **A dropped namespace is never written.** Maintenance resolves namespaces
//!    with get-or-create, so a write after a drop brings them back; the drop
//!    stops maintenance and waits for the operation in progress
//!    ([`PartitionHandle::mark_dropped`]).
//!
//! Codes are never re-encoded: a moved code keeps its `centre_id` and only its
//! key changes. While a chunk is in both places a search scores it twice, and
//! MaxSim takes the maximum, so the result is unchanged.

use std::collections::{BTreeMap, HashMap};

use crate::AsyncDb;
use crate::semantic_search::cluster::split::{SplitParams, SplitRng, balanced_two_means};
use crate::semantic_search::index::composite_key;
use crate::semantic_search::index::vector_index::VectorIndex;
use crate::semantic_search::quantisation::rabitq::{reconstruct_code, reconstruct_code_into};
use crate::semantic_search::{IvfLayout, MaintenanceCounters, NamespaceIvf, PartitionHandle, Posting, PostingDelta, PostingOrigin, PostingState};
use crate::vector_kv::{
    decode_posting, decode_sparse_meta, encode_posting, encode_sparse_meta, f32s_to_bytes, ivf_centres_ns, ivf_journal_ns, ivf_postings_ns,
    lock_docs_vectors, sparse_vectors_meta_ns, sparse_vectors_ns,
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
pub mod rebuild;

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
const KIND_REBUILD: u8 = 3;

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
    /// Drain every posting of `drained` into `postings` new postings with ids
    /// from `first_posting`, encoded against centres with ids from
    /// `first_centre`; each chunk to the nearest new centre (M4).
    Rebuild {
        first_centre: u32,
        first_posting: u32,
        postings: u32,
        drained: Vec<u32>,
    },
}

impl Op {
    /// The posting this operation drains, if it drains one.
    fn drained(&self) -> Option<u32> {
        match *self {
            Op::Split { posting, .. } | Op::Merge { posting, .. } => Some(posting),
            Op::Reassign { .. } | Op::Rebuild { .. } => None,
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
    /// reassignment's `parent, parent centre, child × 2, scope…`; a rebuild's
    /// `first centre, first posting, postings, drained…`.
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
            Op::Rebuild {
                first_centre,
                first_posting,
                postings,
                drained,
            } => {
                let mut ids = vec![*first_centre, *first_posting, *postings];
                ids.extend(drained);
                (KIND_REBUILD, ids)
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
        let ids: Vec<u32> = v[3..].as_chunks::<4>().0.iter().map(|b| u32::from_be_bytes(*b)).collect();
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
            (KIND_REBUILD, n) if n >= 3 => Op::Rebuild {
                first_centre: ids[0],
                first_posting: ids[1],
                postings: ids[2],
                drained: ids[3..].to_vec(),
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
    let journal = ivf_journal_ns(namespace);
    // Resolving a namespace creates it: a namespace that never maintained its
    // partition has no journal, and reading must not make one.
    if !db.list_namespaces().iter().any(|(name, _)| *name == journal) {
        return Ok(out);
    }
    for (k, v) in db.namespace(journal).await?.scan_prefix(Vec::new()).await? {
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
    let mut out = HashMap::<u32, Vec<_>>::with_capacity(postings.len());
    for (posting, k, v) in rows.iter() {
        let Some((_, doc)) = composite_key::decode(k) else { continue };
        let codes = VectorIndex::list_from_bytes(v).map_err(|e| corrupt(namespace, format!("posting {posting}: {e}")))?;
        out.entry(posting).or_default().push((doc.to_vec(), codes));
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
/// reconstructed (its centre is unknown, or it is not a Pass-1 code).
fn reconstruct(ivf: &NamespaceIvf, code: &VectorIndex) -> Option<Vec<f32>> {
    reconstruct_code(code, ivf.rotated_centre(code.centre_id)?)
}

/// [`reconstruct`] into `buf`, for loops over many codes. `false` for a code
/// that cannot be reconstructed.
fn reconstruct_into(ivf: &NamespaceIvf, code: &VectorIndex, buf: &mut Vec<f32>) -> bool {
    ivf.rotated_centre(code.centre_id).is_some_and(|c| reconstruct_code_into(code, c, buf))
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
    matched_with(wanted, held, &mut vec![false; held.len()])
}

/// [`matched`], with the copies already claimed in `used` (one flag per element
/// of `held`): several sources of one document moving chunks into the same key
/// share it, so two of them never count the same copy.
fn matched_with(wanted: &[&VectorIndex], held: &[VectorIndex], used: &mut [bool]) -> Vec<bool> {
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
    /// A rebuild: the nearest of the new postings.
    Nearest { postings: Vec<(u32, &'a [f32])> },
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
            // The first new posting for a code that cannot be reconstructed.
            DrainRule::Nearest { postings } => reconstruct(ivf, code)
                .and_then(|x| nearest(&x, postings))
                .map_or(postings[0].0, |(id, _)| id),
        }
    }
}

/// Documents a maintenance step handles under one set of locks. Each batch
/// reads its keys and metas with `get_multiple` (one read per kind, not one per
/// key), then writes per document, so it holds its documents' locks for about
/// `BATCH_DOCS` meta writes (WAL fsyncs, ~2.3 ms each here). Batches run one at
/// a time: writers and deletes wait for any document lock a batch holds, and for
/// the WAL lock its writes queue on, so a wide batch, or several at once, is
/// foreground tail latency. Four lanes of 64 held up to 256 documents' locks (as
/// many as there were stripes) and put a delete's p99 at 580 ms under a
/// delete-and-insert load while partitions split; 16 at a time, with 4,096
/// stripes, 110 ms (M5 investigation).
const BATCH_DOCS: usize = 16;

/// Run `work` on `items` in batches of [`BATCH_DOCS`], one batch at a time, in
/// this task (never spawned, so a simulated crash in one stops every later
/// write: the crash point is per thread and sticky). The first error ends the
/// call, and so does a drop of the namespace's vector index (checked before
/// each batch, so a long operation stops soon after). Returns each batch's
/// result, in order.
async fn in_batches<'a, T, R, F, Fut>(namespace: &str, handle: &PartitionHandle, items: &'a [T], work: F) -> Result<Vec<R>, crate::KVError>
where
    F: Fn(&'a [T]) -> Fut,
    Fut: std::future::Future<Output = Result<R, crate::KVError>>,
{
    let mut out = Vec::with_capacity(items.len().div_ceil(BATCH_DOCS));
    for batch in items.chunks(BATCH_DOCS) {
        check_live(namespace, handle)?;
        out.push(work(batch).await?);
    }
    Ok(out)
}

/// An error once `namespace`'s vector index is dropped: its maintenance must
/// stop, since every write resolves its namespace with get-or-create and would
/// bring the dropped namespaces back.
fn check_live(namespace: &str, handle: &PartitionHandle) -> Result<(), crate::KVError> {
    if handle.is_dropped() {
        return Err(crate::KVError::Io(std::io::Error::other(format!(
            "namespace '{namespace}': its vector index was dropped; maintenance stopped"
        ))));
    }
    Ok(())
}

/// What every maintenance entry point does once it holds the maintenance lock:
/// refuse a dropped namespace, and finish or abandon whatever the journal holds
/// if the partition needs it (a new handle, or an earlier error). An operation
/// must never start on top of an unfinished one: it would allocate ids from a
/// snapshot that does not know the unfinished one's, and recovery would later
/// retire postings the new operation made live.
async fn prepare(db: &AsyncDb, namespace: &str, handle: &PartitionHandle, ops: &MaintenanceOps) -> Result<(), crate::KVError> {
    check_live(namespace, handle)?;
    if handle.needs_recovery() {
        recover_locked(db, namespace, handle, ops).await?;
    }
    Ok(())
}

/// Recover `namespace`'s journal if its partition needs it, before anything
/// else uses the partition (the vector worker's writes, searches, deletes).
/// Takes the maintenance lock only when there is something to do. Returns how
/// many unfinished operations it found.
pub async fn ensure_recovered(db: &AsyncDb, namespace: &str, handle: &PartitionHandle, ops: &MaintenanceOps) -> Result<usize, crate::KVError> {
    if !handle.needs_recovery() {
        return Ok(0);
    }
    let _maintenance = handle.lock_maintenance().await;
    if !handle.needs_recovery() {
        return Ok(0);
    }
    check_live(namespace, handle)?;
    recover_locked(db, namespace, handle, ops).await
}

/// One document's chunks to copy: `destination → codes`, each code already
/// filed under its destination.
type Moving = BTreeMap<u32, Vec<VectorIndex>>;

/// Add each document's `moving` codes (`destination → codes`, each already
/// filed under its destination) to its keys and meta: per document, the meta
/// first, then the keys, each key gaining only the codes it does not already
/// hold. All are logged in the vector index WAL, in that order, and synced once
/// at the end ([`sync_batch`]). The caller holds every document's vector lock.
/// A document with no meta is being deleted and is skipped (rule 2). Returns
/// the count change and how many codes were newly written.
async fn copy_into_batch(db: &AsyncDb, namespace: &str, items: Vec<(Vec<u8>, Moving)>) -> Result<(PostingDelta, u64), crate::KVError> {
    if items.is_empty() {
        return Ok((PostingDelta::default(), 0));
    }
    let sparse = db.namespace(sparse_vectors_ns(namespace)).await?;
    let meta_ns = db.namespace(sparse_vectors_meta_ns(namespace)).await?;
    let metas = meta_ns.get_multiple(items.iter().map(|(doc, _)| doc.clone()).collect()).await?;
    let keys: Vec<Vec<u8>> = items
        .iter()
        .flat_map(|(doc, moving)| moving.keys().map(|&dest| composite_key::encode(dest, doc)))
        .collect();
    let mut held_all = sparse.get_multiple(keys.clone()).await?.into_iter();
    let mut keys = keys.into_iter();
    let mut delta = PostingDelta::default();
    let mut copied = 0;
    for ((doc, moving), meta) in items.into_iter().zip(metas) {
        // Take this document's keys off the shared lists even if it is skipped.
        let mut writes = Vec::with_capacity(moving.len());
        for (dest, codes) in moving {
            let key = keys.next().expect("one key per destination");
            let held = match held_all.next().expect("one value per key") {
                Some(b) => VectorIndex::list_from_bytes(&b).map_err(|e| corrupt(namespace, e))?,
                None => Vec::new(),
            };
            let refs: Vec<&VectorIndex> = codes.iter().collect();
            let present = matched(&refs, &held);
            let mut merged = held.clone();
            merged.extend(codes.iter().zip(&present).filter(|(_, p)| !**p).map(|(c, _)| c.clone()));
            let new = merged.len() - held.len();
            writes.push((dest, key, merged, new));
        }
        let Some(mut meta) = meta.and_then(|b| decode_sparse_meta(&b)) else {
            continue;
        };
        // The meta: each destination at the size its key will have.
        let mut doc_delta = PostingDelta::default();
        for (dest, _, merged, _) in &writes {
            let n = merged.len() as u32;
            match meta.postings.iter_mut().find(|(id, _)| id == dest) {
                Some((_, have)) if *have == n => {}
                Some((_, have)) => {
                    doc_delta.removed.push((*dest, *have));
                    doc_delta.added.push((*dest, n));
                    *have = n;
                }
                None => {
                    meta.postings.push((*dest, n));
                    doc_delta.added.push((*dest, n));
                }
            }
        }
        if !doc_delta.is_empty() {
            crash_point!();
            meta_ns
                .put_logged(doc.clone(), encode_sparse_meta(meta.text_hash, &meta.postings))
                .await?;
        }
        for (_, key, merged, new) in writes {
            if new > 0 {
                crash_point!();
                sparse.put_logged(key, VectorIndex::list_to_bytes(&merged)).await?;
                copied += new as u64;
            }
        }
        delta.extend(doc_delta);
    }
    sync_batch(db).await?;
    Ok((delta, copied))
}

/// Make a batch's logged writes durable: one fsync of the vector index WAL.
/// Called before the batch releases its documents' vector locks, so no
/// foreground write (a re-embed, a delete) builds on a maintenance write a
/// crash could still lose. What survives a crash is a prefix of the log
/// (`db/vector_index_wal.rs`), so within a batch the meta-before-key order
/// holds across the two namespaces.
async fn sync_batch(db: &AsyncDb) -> Result<(), crate::KVError> {
    crash_point!();
    db.sync_vector_index_wal().await
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
    prepare(db, namespace, handle, ops).await?;
    split_locked(db, namespace, handle, posting, ops)
        .await
        .inspect_err(|_| handle.set_needs_recovery(true))
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
    maintain_partition_until(db, namespace, handle, ops, &std::sync::atomic::AtomicBool::new(false)).await
}

/// [`maintain_partition`], returning early once `stop` is set: it is checked
/// before each split or merge, so the one in progress finishes and the rest are
/// left for the next request. Every operation is journaled on its own, so
/// stopping between two leaves the partition consistent. Server shutdown sets
/// `stop`: growing a large namespace from its root posting is thousands of
/// splits, and shutdown would otherwise wait for all of them.
pub async fn maintain_partition_until(
    db: &AsyncDb,
    namespace: &str,
    handle: &PartitionHandle,
    ops: &MaintenanceOps,
    stop: &std::sync::atomic::AtomicBool,
) -> Result<MaintenanceReport, crate::KVError> {
    let _maintenance = handle.lock_maintenance().await;
    prepare(db, namespace, handle, ops).await?;
    maintain_locked(db, namespace, handle, ops, stop)
        .await
        .inspect_err(|_| handle.set_needs_recovery(true))
}

async fn maintain_locked(
    db: &AsyncDb,
    namespace: &str,
    handle: &PartitionHandle,
    ops: &MaintenanceOps,
    stop: &std::sync::atomic::AtomicBool,
) -> Result<MaintenanceReport, crate::KVError> {
    use std::sync::atomic::Ordering;
    let stopping = || stop.load(Ordering::Acquire) || handle.is_dropped();
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
            if stopping() {
                return Ok(report);
            }
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
            if stopping() {
                return Ok(report);
            }
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
/// `new_centres` and `changed` and wait out every write and search that
/// entered by the old one ([`PartitionHandle::publish`]). Only maintenance
/// publishes (under its lock), so the snapshot cannot change in between, and
/// nothing is excluded while the record is written or the snapshot built.
async fn publish_drain(
    db: &AsyncDb,
    namespace: &str,
    handle: &PartitionHandle,
    op: u64,
    record: &Record,
    new_centres: &[(u32, Vec<f32>)],
    changed: &[Posting],
) -> Result<(), crate::KVError> {
    write_record(db, namespace, op, &record.with_state(JournalState::Published)).await?;
    let next = handle.snapshot().evolve(new_centres, changed).map_err(|e| corrupt(namespace, e))?;
    handle.publish(next).await;
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
        let ivf = handle.snapshot();
        let rule = drain_rule(namespace, &ivf, ops, &record.op)?;
        let copied = drain_copy(db, namespace, handle, &ivf, &rule, drained).await?;
        MaintenanceCounters::add(&handle.counters().drain_moves, copied);
        // 5. Barrier: the copies are durable before any delete (one fsync of
        // the vector index WAL, which holds them; the metas are WAL-backed).
        crash_point!();
        db.sync_vector_index_wal().await?;
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
    let next = handle.snapshot().evolve(&[], &[posting]).map_err(|e| corrupt(namespace, e))?;
    handle.publish(next).await;
    match record.op {
        Op::Split { posting, children, .. } => {
            MaintenanceCounters::add(&handle.counters().splits, 1);
            // The reassignment is recorded before the split's record goes, so a
            // crash between the two still reassigns. If that crash happened, it
            // is recorded already: run that one, never a second.
            let recorded = records(db, namespace)
                .await?
                .into_iter()
                .find(|(_, r)| matches!(r.op, Op::Reassign { parent, .. } if parent == posting));
            let (next, reassign) = match recorded {
                Some(found) => found,
                None => {
                    let reassign = reassign::plan(&handle.snapshot(), ops, posting, children)
                        .ok_or_else(|| corrupt(namespace, format!("split of posting {posting}: its centre or a child's is missing")))?;
                    let next = next_op_id(db, namespace).await?;
                    write_record(db, namespace, next, &reassign).await?;
                    (next, reassign)
                }
            };
            delete_record(db, namespace, op).await?;
            reassign::run(db, namespace, handle, ops, next, reassign).await
        }
        Op::Merge { .. } => {
            MaintenanceCounters::add(&handle.counters().merges, 1);
            delete_record(db, namespace, op).await
        }
        Op::Reassign { .. } | Op::Rebuild { .. } => unreachable!("not a single-posting drain"),
    }
}

/// Where `op` sends a drained posting's chunks, against snapshot `ivf`.
fn drain_rule<'a>(namespace: &str, ivf: &'a NamespaceIvf, ops: &MaintenanceOps, op: &Op) -> Result<DrainRule<'a>, crate::KVError> {
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
        Op::Rebuild { first_posting, postings, .. } => {
            let new: Vec<(u32, &[f32])> = (first_posting..first_posting + postings)
                .map(|id| ivf.rotated_posting_centre(id).map(|c| (id, c)).ok_or_else(|| missing("a new posting")))
                .collect::<Result<_, _>>()?;
            if new.is_empty() {
                return Err(missing("every new posting"));
            }
            DrainRule::Nearest { postings: new }
        }
        Op::Reassign { .. } => unreachable!("not a drain"),
    };
    Ok(rule)
}

/// 4. Copy each document's chunks from the `drained` posting to where `rule`
///    sends them (`ivf`: the snapshot the rule was built from). Returns how
///    many codes were newly written.
async fn drain_copy(
    db: &AsyncDb,
    namespace: &str,
    handle: &PartitionHandle,
    ivf: &NamespaceIvf,
    rule: &DrainRule<'_>,
    drained: u32,
) -> Result<u64, crate::KVError> {
    let sparse = db.namespace(sparse_vectors_ns(namespace)).await?;
    let docs: Vec<Vec<u8>> = posting_entries(db, namespace, drained).await?.into_iter().map(|(doc, _)| doc).collect();
    let sparse = &sparse;
    let copied = in_batches(namespace, handle, &docs, |batch| async move {
        let _docs = lock_docs_vectors(namespace, batch.iter().map(Vec::as_slice)).await;
        // Re-read under the locks: an upsert or delete may have moved them on.
        let keys = batch.iter().map(|doc| composite_key::encode(drained, doc)).collect();
        let mut items = Vec::with_capacity(batch.len());
        for (doc, bytes) in batch.iter().zip(sparse.get_multiple(keys).await?) {
            let Some(bytes) = bytes else { continue };
            let mut moving: BTreeMap<u32, Vec<VectorIndex>> = BTreeMap::new();
            for code in VectorIndex::list_from_bytes(&bytes).map_err(|e| corrupt(namespace, e))? {
                let dest = rule.destination(ivf, &code);
                moving.entry(dest).or_default().push(VectorIndex { cluster_id: dest, ..code });
            }
            items.push((doc.clone(), moving));
        }
        let (delta, n) = copy_into_batch(db, namespace, items).await?;
        handle.snapshot().apply_delta(&delta);
        Ok::<u64, crate::KVError>(n)
    })
    .await?;
    Ok(copied.into_iter().sum())
}

/// 6. Delete the drained posting's keys, and the posting from each document's
///    meta (both logged, synced per batch).
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
        let docs: Vec<Vec<u8>> = keys
            .into_iter()
            .filter_map(|(key, _)| composite_key::decode(&key).map(|(_, doc)| doc.to_vec()))
            .collect();
        let (sparse, meta_ns) = (&sparse, &meta_ns);
        in_batches(namespace, handle, &docs, |batch| async move {
            let _docs = lock_docs_vectors(namespace, batch.iter().map(Vec::as_slice)).await;
            let metas = meta_ns.get_multiple(batch.to_vec()).await?;
            for (doc, meta) in batch.iter().zip(metas) {
                // The meta first, then the key. The other order lost track of
                // the meta's entry: a crash between the two left a meta naming
                // the posting with no key, and the redo, which walks the
                // posting's keys, never came back to it. This order is safe
                // only here: the posting is draining, and the redo deletes
                // every key still under it whether or not a meta lists it.
                if let Some(mut meta) = meta.and_then(|b| decode_sparse_meta(&b))
                    && let Some(i) = meta.postings.iter().position(|&(id, _)| id == drained)
                {
                    let (_, chunks) = meta.postings.remove(i);
                    crash_point!();
                    meta_ns
                        .put_logged(doc.clone(), encode_sparse_meta(meta.text_hash, &meta.postings))
                        .await?;
                    handle.snapshot().apply_delta(&PostingDelta {
                        added: vec![],
                        removed: vec![(drained, chunks)],
                    });
                }
                crash_point!();
                sparse.delete_logged(composite_key::encode(drained, doc)).await?;
            }
            sync_batch(db).await
        })
        .await?;
    }
    Ok(())
}

/// Finish or abandon every operation a crash (or an error) left in
/// `namespace`'s journal. Runs before anything else uses the partition
/// ([`ensure_recovered`], and every maintenance entry point). Returns how many
/// operations it found.
pub async fn recover(db: &AsyncDb, namespace: &str, handle: &PartitionHandle, ops: &MaintenanceOps) -> Result<usize, crate::KVError> {
    let _maintenance = handle.lock_maintenance().await;
    recover_locked(db, namespace, handle, ops).await
}

/// [`recover`] for a caller holding the maintenance lock. Takes the oldest
/// unfinished operation, finishes or abandons it (which deletes its record,
/// and may record and finish a follow-up), and reads the journal again, until
/// it is empty: a list read once would run a follow-up a finished operation
/// already ran. Clears the partition's need for recovery on success.
async fn recover_locked(db: &AsyncDb, namespace: &str, handle: &PartitionHandle, ops: &MaintenanceOps) -> Result<usize, crate::KVError> {
    let mut found = 0;
    let mut last: Option<(u64, Record)> = None;
    while let Some((op, record)) = records(db, namespace).await?.into_iter().next() {
        if last.as_ref() == Some(&(op, record.clone())) {
            return Err(corrupt(namespace, format!("recovery of operation {op} left its record unchanged")));
        }
        found += 1;
        match (&record.op, record.state) {
            (Op::Reassign { .. }, _) => reassign::run(db, namespace, handle, ops, op, record.clone()).await?,
            (Op::Rebuild { .. }, JournalState::Planned) => abandon(db, namespace, handle, op, &record).await?,
            (Op::Rebuild { .. }, _) => rebuild::finish(db, namespace, handle, ops, op, record.clone()).await?,
            (_, JournalState::Planned) => abandon(db, namespace, handle, op, &record).await?,
            (_, JournalState::Published | JournalState::Copied) => finish_drain(db, namespace, handle, ops, op, record.clone()).await?,
        }
        last = Some((op, record));
    }
    handle.set_needs_recovery(false);
    Ok(found)
}

/// A drain that never published: nothing was routed to new postings, so they
/// are retired empty and the drained posting becomes active again. Centres it
/// appended stay, unreferenced.
async fn abandon(db: &AsyncDb, namespace: &str, handle: &PartitionHandle, op: u64, record: &Record) -> Result<(), crate::KVError> {
    let mut changed = Vec::new();
    let (new_ids, centres, drained) = match &record.op {
        Op::Split {
            posting, centres, children, ..
        } => (children.to_vec(), centres.to_vec(), vec![*posting]),
        Op::Rebuild {
            first_centre,
            first_posting,
            postings,
            drained,
        } => (
            (*first_posting..first_posting + postings).collect(),
            (*first_centre..first_centre + postings).collect(),
            drained.clone(),
        ),
        Op::Merge { posting, .. } => (vec![], vec![], vec![*posting]),
        Op::Reassign { .. } => unreachable!("a reassignment is redone, never abandoned"),
    };
    for id in new_ids {
        if let Some(mut child) = stored_posting(db, namespace, id).await? {
            child.state = PostingState::Retired;
            write_posting(db, namespace, &child).await?;
            changed.push(child);
        }
    }
    for id in drained {
        if let Some(mut posting) = stored_posting(db, namespace, id).await?
            && posting.state != PostingState::Active
        {
            posting.state = PostingState::Active;
            write_posting(db, namespace, &posting).await?;
            changed.push(posting);
        }
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
        let next = handle.snapshot().evolve(&new_centres, &changed).map_err(|e| corrupt(namespace, e))?;
        handle.publish(next).await;
    }
    delete_record(db, namespace, op).await
}

pub mod health;
pub mod history;
pub mod recall_check;

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
