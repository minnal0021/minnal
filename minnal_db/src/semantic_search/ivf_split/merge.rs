//! Merging a posting that deletes or moves have shrunk (design doc *M3 split
//! algorithm (LIRE)*, *Merge posting P*; M3b).
//!
//! A posting with `merge_limit` chunks or fewer costs a probe for little to
//! read. It is folded into the first of the 64 active postings nearest its
//! centre that has room (`chunks(S) + chunks(Q) < split_limit`, so the result
//! does not split straight away). The larger of the two is kept and the smaller
//! is drained into it through the same journal steps as a split. A chunk whose
//! old centre is nearer than the kept posting's goes to its nearest active
//! posting instead, which is the design's "route the far ones again" folded
//! into the same pass. The last active posting never merges.

use super::*;

/// How many of the nearest postings a merge looks through for room (SPTAG's
/// value, as for reassignment).
const MERGE_CANDIDATES: usize = 64;

/// Merge `posting` if it is active, at or under `ops.merge_limit` chunks, and
/// a nearby posting has room. Returns whether a merge was made. The caller
/// holds the maintenance lock.
pub(super) async fn merge_locked(
    db: &AsyncDb,
    namespace: &str,
    handle: &PartitionHandle,
    posting: u32,
    ops: &MaintenanceOps,
) -> Result<bool, crate::KVError> {
    let ivf = handle.snapshot();
    if ivf.posting_infos().get(&posting).map(|i| i.state) != Some(PostingState::Active) {
        return Ok(false);
    }
    let chunks = ivf.posting_chunks(posting);
    if chunks > ops.merge_limit || handle.is_unmergeable(posting, chunks) {
        return Ok(false);
    }
    let active = ivf.active_centres();
    let Some(centre) = ivf.rotated_posting_centre(posting) else {
        return Err(corrupt(namespace, format!("posting {posting} has no centre")));
    };
    let others: Vec<(u32, &[f32])> = active.iter().copied().filter(|&(id, _)| id != posting).collect();
    let partner = neighbours(centre, &others, MERGE_CANDIDATES)
        .into_iter()
        .find(|&q| chunks + ivf.posting_chunks(q) < ops.split_limit);
    let Some(partner) = partner else {
        handle.mark_unmergeable(posting, chunks);
        return Ok(false);
    };
    // Keep the larger; drain the smaller.
    let (drained, kept) = if ivf.posting_chunks(partner) >= chunks {
        (posting, partner)
    } else {
        (partner, posting)
    };
    let op = next_op_id(db, namespace).await?;
    let record = Record {
        state: JournalState::Planned,
        op: Op::Merge { posting: drained, kept },
    };
    write_record(db, namespace, op, &record).await?;
    let mut changed = posting_of(&ivf, drained).ok_or_else(|| corrupt(namespace, format!("posting {drained} vanished")))?;
    changed.state = PostingState::Draining;
    write_posting(db, namespace, &changed).await?;
    publish_drain(db, namespace, handle, op, &record, &[], &[changed]).await?;
    finish_drain(db, namespace, handle, ops, op, record.with_state(JournalState::Published)).await?;
    Ok(true)
}
