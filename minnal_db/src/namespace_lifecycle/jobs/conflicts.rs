//! Which index jobs of one namespace cannot run at the same time.
//!
//! A job is refused at its start while a job it conflicts with is live, so
//! two jobs never work on the same index at once:
//!
//! - a job and itself (the same key: one build of a field at a time);
//! - a build and a drop of the same field;
//! - `reindex_all`, which drops and rebuilds every field, and any field job;
//! - turning semantic search on and any other vector job: it runs on a store
//!   with no vector index, and every other vector job needs one;
//! - removing the vector index (`vector_cleanup`) and any other vector job.
//!
//! A vector reindex and a partition rebuild may run together, as they may
//! today: the rebuild moves codes, the reindex writes new ones through the
//! partition's usual routing.

use super::key::JobKind;

/// Whether jobs `a` and `b` of one namespace may not run at the same time.
/// Symmetric.
pub fn conflicts(a: JobKind, b: JobKind) -> bool {
    use JobKind::*;
    if a == b {
        return true;
    }
    match (a, b) {
        (FieldBuild(x) | FieldDrop(x), FieldBuild(y) | FieldDrop(y)) => x == y,
        (ReindexAll, FieldBuild(_) | FieldDrop(_)) | (FieldBuild(_) | FieldDrop(_), ReindexAll) => true,
        (VectorEnable | VectorCleanup, VectorEnable | VectorCleanup | VectorReindex | PartitionRebuild)
        | (VectorReindex | PartitionRebuild, VectorEnable | VectorCleanup) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::super::key::EVERY_KIND;
    use super::*;
    use JobKind::*;

    #[test]
    fn conflicts_are_symmetric() {
        for a in EVERY_KIND {
            for b in EVERY_KIND {
                assert_eq!(conflicts(a, b), conflicts(b, a), "{a} / {b}");
            }
        }
    }

    /// Every pair, spelled out: the pairs that may run together.
    #[test]
    fn only_these_pairs_run_together() {
        let together = |a: JobKind, b: JobKind| {
            matches!(
                (a, b),
                (FieldBuild(7) | FieldDrop(7), FieldBuild(8) | FieldDrop(8))
                    | (FieldBuild(8) | FieldDrop(8), FieldBuild(7) | FieldDrop(7))
                    | (
                        FieldBuild(_) | FieldDrop(_) | ReindexAll,
                        VectorEnable | VectorCleanup | VectorReindex | PartitionRebuild
                    )
                    | (
                        VectorEnable | VectorCleanup | VectorReindex | PartitionRebuild,
                        FieldBuild(_) | FieldDrop(_) | ReindexAll
                    )
                    | (VectorReindex, PartitionRebuild)
                    | (PartitionRebuild, VectorReindex)
            )
        };
        for a in EVERY_KIND {
            for b in EVERY_KIND {
                assert_eq!(conflicts(a, b), !together(a, b), "{a} / {b}");
            }
        }
    }
}
