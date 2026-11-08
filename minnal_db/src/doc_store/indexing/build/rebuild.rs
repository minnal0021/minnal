//! The scan that fills a field index: every document of the store is indexed
//! in the field, from where the build's saved progress says the last run got
//! to. Shared by a `field_build` job and a `reindex-all`.

use std::collections::VecDeque;

use log::info;

use crate::doc_store::DocStoreError;
use crate::doc_store::indexing::progress::FieldBuildProgress;
use crate::namespace::NamespaceId;
use crate::support::hex::{bytes_to_hex, hex_to_bytes};
use crate::{AsyncDb, FieldId};

/// `(key, value)` pairs fetched per cursor page. Bounds memory to about one
/// page of documents, and sets how often the scan saves its progress and looks
/// for a cancel.
pub(super) const PAGE_SIZE: usize = 1_000;

/// Documents between two saves of the progress.
const SAVE_EVERY: u64 = 1_000;

/// How a scan ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::doc_store) enum Scan {
    /// Every document is indexed and the index checkpointed.
    Finished,
    /// `cancelled` said stop; the index holds part of the store.
    Cancelled,
}

/// Smallest key strictly greater than `key`: the scan cursor is inclusive, so
/// a resumed scan starts here.
fn successor_key(key: &[u8]) -> Vec<u8> {
    let mut next = Vec::with_capacity(key.len() + 1);
    next.extend_from_slice(key);
    next.push(0);
    next
}

/// Index every document of `ns_name` in field `field_id` (activated already),
/// resuming after `progress.resume_after_hex`. `save` is handed the progress
/// every [`SAVE_EVERY`] documents and at the end; `cancelled` is asked before
/// each page.
///
/// Each key is indexed at its current value under its key stripe
/// (`Db::reindex_field`), not by writing the document again: a write of
/// unchanged bytes is a no-op for the index, and writing back bytes read a
/// page earlier would overwrite a newer write of the key.
pub(in crate::doc_store) async fn rebuild_field(
    db: &AsyncDb,
    ns_name: &str,
    ns_id: NamespaceId,
    field_id: FieldId,
    progress: &mut FieldBuildProgress,
    mut save: impl FnMut(&FieldBuildProgress),
    cancelled: impl Fn() -> bool,
) -> Result<Scan, DocStoreError> {
    // Never get-or-create: a build outliving its store would bring it back.
    let ns = db.existing_namespace(ns_name.to_owned()).await?.ok_or_else(|| DocStoreError::NotFound {
        namespace: ns_name.to_owned(),
    })?;
    let resume_after: Option<Vec<u8>> = progress.resume_after_hex.as_deref().and_then(hex_to_bytes);

    // Pass 1: count the keys, a page at a time (the total drives the percent
    // complete), and how many a resumed build has done already.
    let (mut total, mut already_done) = (0u64, 0u64);
    let mut cursor: Option<Vec<u8>> = None;
    loop {
        if cancelled() {
            return Ok(Scan::Cancelled);
        }
        let (pairs, next) = ns.scan(cursor, None, PAGE_SIZE).await?;
        for (key, _) in &pairs {
            total += 1;
            if resume_after.as_ref().is_some_and(|r| key <= r) {
                already_done += 1;
            }
        }
        match next {
            Some(c) => cursor = Some(c),
            None => break,
        }
    }
    info!("index build started: namespace='{ns_name}' field_id={field_id} total={total} docs");
    progress.total = total;
    progress.indexed = already_done;
    save(progress);

    // The rows this build sets live in memory until an index checkpoint writes
    // them out; they are not in the WAL, so a crash before then loses them. The
    // saved resume point therefore names a key only once a checkpoint that
    // started after the key was indexed has finished. Naming it sooner skipped,
    // on resume, every key a crash had taken back out of the index.
    let mut unsaved: VecDeque<(u64, Vec<u8>)> = VecDeque::new();

    // Pass 2: index, a page at a time, from just past the resume point.
    let mut cursor: Option<Vec<u8>> = resume_after.as_deref().map(successor_key);
    loop {
        if cancelled() {
            save(progress);
            return Ok(Scan::Cancelled);
        }
        let (pairs, next) = ns.scan(cursor, None, PAGE_SIZE).await?;
        if pairs.is_empty() {
            break;
        }
        for (key, _) in pairs {
            let outcome = db.reindex_field(ns_id, field_id, key.clone()).await?;
            if outcome == crate::field_index::FieldReindexOutcome::FieldNotActive {
                return Err(crate::db::error::KVError::Serialization(format!(
                    "index build for namespace '{ns_name}' field_id={field_id}: the field was deactivated mid-build"
                ))
                .into());
            }
            progress.indexed += 1;
            if progress.indexed.is_multiple_of(SAVE_EVERY) {
                unsaved.push_back((db.index_checkpoints_started(), key));
            }
            while unsaved.front().is_some_and(|(started, _)| db.index_checkpoint_covers(*started)) {
                if let Some((_, k)) = unsaved.pop_front() {
                    progress.resume_after_hex = Some(bytes_to_hex(&k));
                }
            }
            if progress.indexed.is_multiple_of(SAVE_EVERY) {
                save(progress);
                info!(
                    "index build progress: namespace='{ns_name}' field_id={field_id} {}/{total} docs ({:.1}%)",
                    progress.indexed,
                    if total > 0 {
                        progress.indexed as f64 / total as f64 * 100.0
                    } else {
                        0.0
                    }
                );
                tokio::task::yield_now().await;
            }
        }
        match next {
            Some(c) => cursor = Some(c),
            None => break,
        }
    }

    // Write the built rows out before the job is recorded complete: a complete
    // build is never resumed, so rows a crash took back would stay missing.
    db.checkpoint_index().await?;
    save(progress);
    info!(
        "index build complete: namespace='{ns_name}' field_id={field_id} indexed={} docs",
        progress.indexed
    );
    Ok(Scan::Finished)
}
