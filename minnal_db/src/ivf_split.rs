//! Splitting a namespace's postings: the maintenance journal, the split
//! executor, and recovery after a crash (design doc M3a, *M3 split algorithm
//! (LIRE)* steps 1–4, *Maintenance journal*).
//!
//! A posting that holds more than `split_limit` chunks is split in two by
//! balanced 2-means on its codes' reconstructions
//! ([`crate::semantic_search::cluster::split`]). The store has no transactions,
//! so a split is a sequence of single-key writes, each recorded in a WAL-backed
//! journal (`{ns}_ivf_journal`) so that a crash at any point is finished, never
//! left half done:
//!
//! | Step | Writes | Durable how | Record after |
//! |---|---|---|---|
//! | 1. Plan | recount P, 2-means; write the record | WAL | `Planned` |
//! | 2. Prepare | append the two centres; write P1, P2 `Active`; mark P `Draining` | WAL | `Planned` |
//! | 3. Publish | under the routing epoch's write guard: record `Published`, then the new snapshot | WAL | `Published` |
//! | 4. Copy | per document, under its vector lock: add P1/P2 to its meta, write its chunks under `P1‖doc` / `P2‖doc` | meta WAL, keys no-WAL | `Published` |
//! | 5. Barrier | flush the sparse and meta namespaces | flush | `Copied` |
//! | 6. Delete | per document, under its lock: drop P from its meta, delete `P‖doc` | WAL | `Copied` |
//! | 7. Retire | P `Retired`; publish; delete the record | WAL | (gone) |
//!
//! Rules the steps depend on:
//!
//! 1. **Barrier before delete.** The copies are no-WAL; the deletes are
//!    WAL-backed and durable at once. Without the flush of step 5, a crash after
//!    a delete could lose the copy and the chunk would be gone for good.
//! 2. **The meta is always a superset of the live postings.** A document's
//!    meta (WAL-backed, as since M0-2) is the only record of which postings hold
//!    its chunks; a child is added to it before the copy. P is the one exception:
//!    it leaves the meta just before its key is deleted, because a redo of step 6
//!    walks P's own keys and deletes every one, listed or not.
//! 3. **Publish before copy, and record before publish.** Once `Published` is on
//!    disk recovery redoes the copy; until then nothing routes to P1/P2, so
//!    abandoning a `Planned` split never strands an entry. The publish takes the
//!    routing epoch's write guard, so afterwards no write still routes to P and
//!    no search still scans by the routing that lacks P1/P2.
//! 4. **Recovery comes first**: it runs before the vector worker's first pass.
//!
//! Codes are never re-encoded: a moved code keeps its `centre_id` and only its
//! key changes. During a split a chunk can be in both P and P1/P2; a search
//! scores it twice, and MaxSim takes the maximum, so the result is unchanged.

use std::collections::HashMap;

use crate::AsyncDb;
use crate::semantic_search::cluster::split::{SplitParams, SplitRng, balanced_two_means};
use crate::semantic_search::index::composite_key;
use crate::semantic_search::index::vector_index::VectorIndex;
use crate::semantic_search::quantisation::rabitq::reconstruct_single_bit;
use crate::semantic_search::{IvfLayout, NamespaceIvf, PartitionHandle, Posting, PostingDelta, PostingOrigin, PostingState};
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

/// What a namespace's splits run with (its `maintenance` settings).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitSettings {
    /// A posting holding more chunks than this splits (`2 · target_posting_size`).
    pub split_limit: u64,
    /// Balanced 2-means parameters.
    pub params: SplitParams,
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

const RECORD_VERSION: u8 = 1;
const KIND_SPLIT: u8 = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JournalState {
    Planned = 1,
    Published = 2,
    Copied = 3,
}

/// One split in the journal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SplitRecord {
    state: JournalState,
    posting: u32,
    centres: [u32; 2],
    children: [u32; 2],
}

impl SplitRecord {
    /// `version ‖ kind ‖ state ‖ posting ‖ centre × 2 ‖ child × 2` (ids 4B BE).
    fn encode(&self) -> Vec<u8> {
        let mut v = vec![RECORD_VERSION, KIND_SPLIT, self.state as u8];
        for id in [self.posting, self.centres[0], self.centres[1], self.children[0], self.children[1]] {
            v.extend_from_slice(&id.to_be_bytes());
        }
        v
    }

    fn decode(v: &[u8]) -> Option<Self> {
        if v.len() != 23 || v[0] != RECORD_VERSION || v[1] != KIND_SPLIT {
            return None;
        }
        let state = match v[2] {
            1 => JournalState::Planned,
            2 => JournalState::Published,
            3 => JournalState::Copied,
            _ => return None,
        };
        let id = |i: usize| u32::from_be_bytes(v[3 + 4 * i..7 + 4 * i].try_into().unwrap());
        Some(Self {
            state,
            posting: id(0),
            centres: [id(1), id(2)],
            children: [id(3), id(4)],
        })
    }
}

fn corrupt(namespace: &str, what: impl std::fmt::Display) -> crate::KVError {
    crate::KVError::Serialization(format!("namespace '{namespace}': {what}"))
}

async fn write_record(db: &AsyncDb, namespace: &str, op: u64, record: &SplitRecord) -> Result<(), crate::KVError> {
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
async fn records(db: &AsyncDb, namespace: &str) -> Result<Vec<(u64, SplitRecord)>, crate::KVError> {
    let mut out = Vec::new();
    for (k, v) in db.namespace(ivf_journal_ns(namespace)).await?.scan_prefix(Vec::new()).await? {
        let op = u64::from_be_bytes(k.as_slice().try_into().map_err(|_| corrupt(namespace, "malformed journal key"))?);
        out.push((
            op,
            SplitRecord::decode(&v).ok_or_else(|| corrupt(namespace, format!("malformed journal record {op}")))?,
        ));
    }
    out.sort_by_key(|(op, _)| *op);
    Ok(out)
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

/// The reconstruction of `code` in the rotated space, or `None` for a code that
/// cannot be reconstructed (its centre is unknown, or it is not 1-bit).
fn reconstruct(ivf: &NamespaceIvf, code: &VectorIndex) -> Option<Vec<f32>> {
    reconstruct_single_bit(code, ivf.rotated_centre(code.centre_id)?)
}

fn sq_dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = f64::from(*x) - f64::from(*y);
            d * d
        })
        .sum()
}

/// Which side a code goes to: the nearer of the two rotated centres (side 0
/// for a code that cannot be reconstructed, so it is still kept).
fn side(ivf: &NamespaceIvf, code: &VectorIndex, rotated: &[Vec<f32>; 2]) -> usize {
    match reconstruct(ivf, code) {
        Some(x) => usize::from(sq_dist(&x, &rotated[1]) < sq_dist(&x, &rotated[0])),
        None => 0,
    }
}

/// Split `posting` if it holds more than `settings.split_limit` chunks. Takes
/// the namespace's maintenance lock, so splits of one namespace never overlap.
pub async fn split_posting(
    db: &AsyncDb,
    namespace: &str,
    handle: &PartitionHandle,
    posting: u32,
    settings: &SplitSettings,
) -> Result<SplitOutcome, crate::KVError> {
    let _maintenance = handle.lock_maintenance().await;
    split_locked(db, namespace, handle, posting, settings).await
}

/// Split every active posting over the limit, and their halves while they still
/// are, until none is. Returns how many splits were made.
pub async fn split_oversized(db: &AsyncDb, namespace: &str, handle: &PartitionHandle, settings: &SplitSettings) -> Result<usize, crate::KVError> {
    let _maintenance = handle.lock_maintenance().await;
    let mut splits = 0;
    loop {
        let ivf = handle.snapshot();
        let mut over: Vec<(u64, u32)> = ivf
            .posting_infos()
            .iter()
            .filter(|(_, info)| info.state == PostingState::Active)
            .map(|(&id, _)| (ivf.posting_chunks(id), id))
            .filter(|&(chunks, id)| chunks > settings.split_limit && !handle.is_unsplittable(id, chunks))
            .collect();
        if over.is_empty() {
            return Ok(splits);
        }
        over.sort_unstable_by(|a, b| b.cmp(a)); // largest first
        let mut progressed = false;
        for (_, id) in over {
            match split_locked(db, namespace, handle, id, settings).await? {
                SplitOutcome::Split { .. } => {
                    splits += 1;
                    progressed = true;
                }
                SplitOutcome::Unsplittable | SplitOutcome::NotNeeded => {}
            }
        }
        if !progressed {
            return Ok(splits);
        }
    }
}

async fn split_locked(
    db: &AsyncDb,
    namespace: &str,
    handle: &PartitionHandle,
    posting: u32,
    settings: &SplitSettings,
) -> Result<SplitOutcome, crate::KVError> {
    let ivf = handle.snapshot();
    if ivf.posting_infos().get(&posting).map(|i| i.state) != Some(PostingState::Active) {
        return Ok(SplitOutcome::NotNeeded);
    }
    // 1. Plan: recount from the stored codes and divide them.
    let entries = posting_entries(db, namespace, posting).await?;
    let chunks: u64 = entries.iter().map(|(_, c)| c.len() as u64).sum();
    if chunks <= settings.split_limit {
        return Ok(SplitOutcome::NotNeeded);
    }
    if handle.is_unsplittable(posting, chunks) {
        return Ok(SplitOutcome::Unsplittable);
    }
    let points: Vec<Vec<f32>> = entries
        .iter()
        .flat_map(|(_, codes)| codes.iter().filter_map(|c| reconstruct(&ivf, c)))
        .collect();
    // Finished records are deleted, so an id only has to beat unfinished ones.
    let op = records(db, namespace).await?.last().map_or(0, |(op, _)| op + 1);
    let mut rng = SplitRng::new(u64::from(posting).rotate_left(32) ^ op ^ chunks);
    let params = settings.params;
    // Off the async threads (and off the rayon pool searches share).
    let result = tokio::task::spawn_blocking(move || balanced_two_means(&points, &params, &mut rng))
        .await
        .map_err(|e| crate::KVError::Io(std::io::Error::other(e)))?;
    let Some(result) = result else {
        handle.mark_unsplittable(posting, chunks);
        return Ok(SplitOutcome::Unsplittable);
    };
    let (max_posting, max_centre) = ivf.max_ids();
    let record = SplitRecord {
        state: JournalState::Planned,
        posting,
        centres: [max_centre + 1, max_centre + 2],
        children: [max_posting + 1, max_posting + 2],
    };
    write_record(db, namespace, op, &record).await?;

    // 2. Prepare: centres and children (not yet routable), P draining.
    let centres = [ivf.unrotate(&result.centres[0]), ivf.unrotate(&result.centres[1])];
    let centres_ns = db.namespace(ivf_centres_ns(namespace)).await?;
    for (id, c) in record.centres.iter().zip(&centres) {
        crash_point!();
        centres_ns.put(id.to_be_bytes().to_vec(), f32s_to_bytes(c)).await?;
    }
    let children: Vec<Posting> = (0..2)
        .map(|k| Posting {
            posting_id: record.children[k],
            routing_centroid: centres[k].clone(),
            centre_id: record.centres[k],
            state: PostingState::Active,
            parent_id: Some(posting),
            origin: PostingOrigin::Split,
        })
        .collect();
    for child in &children {
        write_posting(db, namespace, child).await?;
    }
    let parent = Posting {
        posting_id: posting,
        routing_centroid: ivf.routing_centroid(posting).map(<[f32]>::to_vec).unwrap_or_default(),
        centre_id: ivf.posting_infos()[&posting].centre_id,
        state: PostingState::Draining,
        parent_id: ivf.posting_infos()[&posting].parent_id,
        origin: ivf.posting_infos()[&posting].origin,
    };
    write_posting(db, namespace, &parent).await?;

    // 3. Publish: recorded first, so recovery redoes the copy once anything can
    //    have routed to the children.
    {
        let guard = handle.write_epoch().await;
        let record = SplitRecord {
            state: JournalState::Published,
            ..record
        };
        write_record(db, namespace, op, &record).await?;
        let mut changed = children.clone();
        changed.push(parent.clone());
        let next = ivf
            .evolve(
                &[(record.centres[0], centres[0].clone()), (record.centres[1], centres[1].clone())],
                &changed,
            )
            .map_err(|e| corrupt(namespace, e))?;
        handle.publish(&guard, next);
    }
    finish_split(
        db,
        namespace,
        handle,
        op,
        SplitRecord {
            state: JournalState::Published,
            ..record
        },
    )
    .await?;
    Ok(SplitOutcome::Split { children: record.children })
}

/// Steps 4–7 of a published split (also what recovery runs).
async fn finish_split(db: &AsyncDb, namespace: &str, handle: &PartitionHandle, op: u64, record: SplitRecord) -> Result<(), crate::KVError> {
    if record.state == JournalState::Published {
        copy(db, namespace, handle, &record).await?;
        // 5. Barrier: the copies are durable before any delete.
        crash_point!();
        db.flush_namespaces(vec![sparse_vectors_ns(namespace), sparse_vectors_meta_ns(namespace)])
            .await?;
        write_record(
            db,
            namespace,
            op,
            &SplitRecord {
                state: JournalState::Copied,
                ..record
            },
        )
        .await?;
    }
    delete_from_parent(db, namespace, handle, &record).await?;
    // 7. Retire.
    let mut parent = stored_posting(db, namespace, record.posting)
        .await?
        .ok_or_else(|| corrupt(namespace, format!("split posting {} is missing", record.posting)))?;
    parent.state = PostingState::Retired;
    write_posting(db, namespace, &parent).await?;
    {
        let guard = handle.write_epoch().await;
        let next = handle.snapshot().evolve(&[], &[parent]).map_err(|e| corrupt(namespace, e))?;
        handle.publish(&guard, next);
    }
    delete_record(db, namespace, op).await
}

/// 4. Copy each document's chunks from P to the child they are nearer.
async fn copy(db: &AsyncDb, namespace: &str, handle: &PartitionHandle, record: &SplitRecord) -> Result<(), crate::KVError> {
    let ivf = handle.snapshot();
    let rotated = [0, 1].map(|k| ivf.rotated_centre(record.centres[k]).map(<[f32]>::to_vec));
    let [Some(r0), Some(r1)] = rotated else {
        return Err(corrupt(
            namespace,
            format!("split of posting {}: a new centre is missing", record.posting),
        ));
    };
    let rotated = [r0, r1];
    let sparse = db.namespace(sparse_vectors_ns(namespace)).await?;
    let meta_ns = db.namespace(sparse_vectors_meta_ns(namespace)).await?;
    for (doc, _) in posting_entries(db, namespace, record.posting).await? {
        let _doc = lock_doc_vectors(namespace, &doc).await;
        // Re-read under the lock: an upsert or delete may have moved it on.
        let Some(bytes) = sparse.get(composite_key::encode(record.posting, &doc)).await? else {
            continue;
        };
        let codes = VectorIndex::list_from_bytes(&bytes).map_err(|e| corrupt(namespace, e))?;
        let mut sides: [Vec<VectorIndex>; 2] = [Vec::new(), Vec::new()];
        for code in codes {
            let k = side(&ivf, &code, &rotated);
            sides[k].push(VectorIndex {
                cluster_id: record.children[k],
                ..code
            });
        }
        // The meta first (WAL): it must list every key that can be on disk.
        // A key no meta lists is an orphan: not copied (step 6 deletes it).
        let Some(mut meta) = meta_ns.get(doc.clone()).await?.and_then(|b| decode_sparse_meta(&b)) else {
            continue;
        };
        let mut delta = PostingDelta::default();
        for (side, &child) in sides.iter().zip(&record.children) {
            if !side.is_empty() && !meta.postings.iter().any(|&(id, _)| id == child) {
                meta.postings.push((child, side.len() as u32));
                delta.added.push((child, side.len() as u32));
            }
        }
        if !delta.is_empty() {
            crash_point!();
            meta_ns.put(doc.clone(), encode_sparse_meta(meta.text_hash, &meta.postings)).await?;
        }
        for (side, &child) in sides.iter().zip(&record.children) {
            if side.is_empty() {
                continue;
            }
            let key = composite_key::encode(child, &doc);
            let new = VectorIndex::list_to_bytes(side);
            if sparse.get(key.clone()).await?.as_deref() == Some(new.as_slice()) {
                continue; // a redo of a copy that already landed
            }
            crash_point!();
            sparse.put_no_wal(key, new).await?;
        }
        handle.snapshot().apply_delta(&delta);
    }
    Ok(())
}

/// 6. Delete P's keys, and P from each document's meta.
async fn delete_from_parent(db: &AsyncDb, namespace: &str, handle: &PartitionHandle, record: &SplitRecord) -> Result<(), crate::KVError> {
    let sparse = db.namespace(sparse_vectors_ns(namespace)).await?;
    let meta_ns = db.namespace(sparse_vectors_meta_ns(namespace)).await?;
    // Nothing routes to P any more, so its key set only shrinks; one more pass
    // covers anything a concurrent writer left.
    for _ in 0..2 {
        let keys = sparse.scan_prefix(record.posting.to_be_bytes().to_vec()).await?;
        if keys.is_empty() {
            return Ok(());
        }
        for (key, _) in keys {
            let Some((_, doc)) = composite_key::decode(&key) else { continue };
            let doc = doc.to_vec();
            let _doc = lock_doc_vectors(namespace, &doc).await;
            // The meta first, then the key. The other order lost track of the
            // meta's entry: a crash between the two left a meta naming P with no
            // key, and the redo, which walks P's keys, never came back to it.
            // This order is safe only here: P is draining, and the redo deletes
            // every key still under P whether or not a meta lists it.
            if let Some(mut meta) = meta_ns.get(doc.clone()).await?.and_then(|b| decode_sparse_meta(&b))
                && let Some(i) = meta.postings.iter().position(|&(id, _)| id == record.posting)
            {
                let (_, chunks) = meta.postings.remove(i);
                crash_point!();
                meta_ns.put(doc.clone(), encode_sparse_meta(meta.text_hash, &meta.postings)).await?;
                handle.snapshot().apply_delta(&PostingDelta {
                    added: vec![],
                    removed: vec![(record.posting, chunks)],
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
pub async fn recover(db: &AsyncDb, namespace: &str, handle: &PartitionHandle) -> Result<usize, crate::KVError> {
    let _maintenance = handle.lock_maintenance().await;
    let found = records(db, namespace).await?;
    for &(op, record) in &found {
        match record.state {
            JournalState::Planned => abandon(db, namespace, handle, op, &record).await?,
            JournalState::Published | JournalState::Copied => finish_split(db, namespace, handle, op, record).await?,
        }
    }
    Ok(found.len())
}

/// A split that never published: nothing was routed to its children, so they
/// are retired empty and P becomes active again. Centres it appended stay,
/// unreferenced.
async fn abandon(db: &AsyncDb, namespace: &str, handle: &PartitionHandle, op: u64, record: &SplitRecord) -> Result<(), crate::KVError> {
    let mut changed = Vec::new();
    for id in record.children {
        if let Some(mut child) = stored_posting(db, namespace, id).await? {
            child.state = PostingState::Retired;
            write_posting(db, namespace, &child).await?;
            changed.push(child);
        }
    }
    if let Some(mut parent) = stored_posting(db, namespace, record.posting).await?
        && parent.state != PostingState::Active
    {
        parent.state = PostingState::Active;
        write_posting(db, namespace, &parent).await?;
        changed.push(parent);
    }
    let mut new_centres = Vec::new();
    let ivf = handle.snapshot();
    for id in record.centres {
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
