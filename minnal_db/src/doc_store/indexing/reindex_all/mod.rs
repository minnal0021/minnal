//! `reindex-all` and drop-all: drop every field index of a store, free its
//! row IDs, and (for a reindex) rebuild every index. One `reindex_all` job.
//!
//! Dropping an index demotes it to a plain attribute in the saved schema, so a
//! crash between the drops and the re-adds would leave the store with no
//! indexes. The job's progress ([`Intent`]) lists the index specs and how far
//! the run got, and a restart resumes it. A run that failed is taken over by
//! the next one, which starts again from its list.
//!
//! Between the drops and the re-adds no field index of the namespace exists, so
//! no row ID is referenced anywhere: that is when the row map is reset
//! ([`AsyncDb::reset_rowmap`]), and the rebuilt indexes get dense IDs for the
//! live documents only. The indexes are then rebuilt one after the
//! other. A `reindex-all` cannot be cancelled: its first step drops the old
//! indexes.

#[cfg(test)]
mod tests;

use crate::doc_store::indexing::build::{Scan, rebuild_field, record_error};
use crate::doc_store::indexing::progress::FieldBuildProgress;
use crate::doc_store::*;
use crate::namespace::NamespaceId;
use crate::namespace_lifecycle::{BuildStatus, JobRecord};

/// How far a run got. Each phase is finished before the record moves to the
/// next, so a resumed run repeats at most one idempotent step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum Phase {
    /// Dropping the indexes; the row map has not been reset yet.
    Dropping,
    /// Indexes dropped and the row map reset; listing them again.
    Readding,
    /// Every index listed again; building them in turn.
    Building,
}

/// The job's progress.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Intent {
    /// The indexes, as the schema listed them before the run.
    specs: Vec<IndexSpec>,
    /// Rebuild them (`reindex-all`), or only drop them (drop-all).
    rebuild: bool,
    phase: Phase,
    /// How many of `specs` are built (in `Building`).
    built: usize,
    /// The build of `specs[built]`, once started.
    current: Option<FieldBuildProgress>,
}

impl Intent {
    /// The fields whose index does not hold every document yet.
    fn unbuilt(&self) -> impl Iterator<Item = &str> {
        let built = if self.phase == Phase::Building { self.built } else { 0 };
        self.specs.iter().skip(built).filter(|_| self.rebuild).map(|s| s.field.as_str())
    }
}

impl DocStore {
    /// Drop every field index of `namespace`, free its row IDs, and rebuild
    /// every index; returns once every index is built again. Crash-safe: a
    /// run cut short is resumed at the next start. A store with no indexes
    /// changes nothing.
    pub async fn reindex_all_attribute_indices(&self, namespace: &str) -> Result<(), DocStoreError> {
        match self.start_reindex_all(namespace, true)? {
            Some((key, _)) => self.run_job(namespace, key).await,
            None => Ok(()),
        }
    }

    /// [`reindex_all_attribute_indices`](Self::reindex_all_attribute_indices),
    /// returning the job's name once it is handed to the governor; `None` for a
    /// store with no indexes.
    pub fn begin_reindex_all(&self, namespace: &str) -> Result<Option<String>, DocStoreError> {
        self.begin_reindex_or_drop_all(namespace, true)
    }

    /// Drop every field index of `namespace` and return their specs; with no
    /// field index left, its row map is reset, freeing every row ID. Each
    /// field is demoted to a plain attribute (its data stays in the
    /// documents). The specs can be passed back to [`add_index`] to rebuild
    /// the indexes; to drop and rebuild in one crash-safe step, use
    /// [`reindex_all_attribute_indices`](Self::reindex_all_attribute_indices).
    ///
    /// [`add_index`]: DocStore::add_index
    pub async fn drop_all_attribute_indices(&self, namespace: &str) -> Result<Vec<IndexSpec>, DocStoreError> {
        match self.start_reindex_all(namespace, false)? {
            Some((key, specs)) => {
                self.run_job(namespace, key).await?;
                Ok(specs)
            }
            None => Ok(Vec::new()),
        }
    }

    /// [`drop_all_attribute_indices`](Self::drop_all_attribute_indices),
    /// returning the job's name once it is handed to the governor; `None` for a
    /// store with no indexes.
    pub fn begin_drop_all_attribute_indices(&self, namespace: &str) -> Result<Option<String>, DocStoreError> {
        self.begin_reindex_or_drop_all(namespace, false)
    }

    fn begin_reindex_or_drop_all(&self, namespace: &str, rebuild: bool) -> Result<Option<String>, DocStoreError> {
        self.running_governor()?;
        let Some((key, _)) = self.start_reindex_all(namespace, rebuild)? else {
            return Ok(None);
        };
        self.submit_job(namespace, key)?;
        Ok(Some(key.kind.name().to_owned()))
    }

    /// Start the `reindex_all` job, with the specs it works on; `None` if
    /// there is none.
    ///
    /// The specs are read again once the job has started: until then a field
    /// build may finish and list its index, and a list read before would
    /// leave that index out (its field still active when the row map is
    /// reset). Once started, the job's conflicts keep every field job out.
    fn start_reindex_all(&self, namespace: &str, rebuild: bool) -> Result<Option<(JobKey, Vec<IndexSpec>)>, DocStoreError> {
        self.job_runner_ready()?;
        let ns_id = self.load_schema(namespace)?.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;
        let key = JobKey::reindex_all(ns_id);
        let specs = self.reindex_all_specs(namespace, key)?;
        if specs.is_empty() {
            return Ok(None);
        }
        let mut intent = Intent {
            specs,
            rebuild,
            phase: Phase::Dropping,
            built: 0,
            current: None,
        };
        let subject = if rebuild { "every index" } else { "every index, drop only" };
        self.start_job(namespace, key, subject, &intent)?;
        let specs = self.reindex_all_specs(namespace, key)?;
        if specs != intent.specs {
            intent.specs = specs;
            self.save_job_progress(key, &intent);
        }
        info!(
            "{} '{namespace}': {} index(es)",
            if rebuild { "reindex-all" } else { "drop-all" },
            intent.specs.len()
        );
        Ok(Some((key, intent.specs)))
    }

    /// The indexes a `reindex_all` job (`key`) of `namespace` works on: the
    /// schema's, and those of an earlier run that failed part-way. That run
    /// may have dropped some already; its list has them all, and leaving
    /// them out would lose them for good.
    fn reindex_all_specs(&self, namespace: &str, key: JobKey) -> Result<Vec<IndexSpec>, DocStoreError> {
        let mut specs = self.load_schema(namespace)?.indices.clone();
        if let Some(earlier) = self
            .jobs
            .get(key)?
            .filter(|r| matches!(r.status, BuildStatus::Failed | BuildStatus::Running))
            .and_then(|r| r.progress::<Intent>())
        {
            for spec in earlier.specs {
                if !specs.iter().any(|s| s.field == spec.field) {
                    specs.push(spec);
                }
            }
        }
        Ok(specs)
    }

    /// The body of a `reindex_all` job.
    pub(in crate::doc_store) async fn reindex_all_job(&self, namespace: &str, key: JobKey) -> Result<(), DocStoreError> {
        self.drive_reindex_all(namespace, key, None).await
    }

    /// Run (or resume) job `key` from its record, stopping after `stop_after`
    /// steps (each drop, the reset, each re-add and each build is one) to
    /// simulate a crash in tests.
    async fn drive_reindex_all(&self, namespace: &str, key: JobKey, stop_after: Option<usize>) -> Result<(), DocStoreError> {
        let Some(record) = self.jobs.get(key)?.filter(JobRecord::is_live) else {
            return Ok(());
        };
        let mut intent: Intent = record.progress().ok_or_else(|| record_error(key, "no reindex-all progress"))?;
        let ns_id = key.namespace;
        let mut steps = 0usize;
        let stop = |steps: usize| stop_after.is_some_and(|n| steps >= n);
        let listed = |field: &str| -> Result<bool, DocStoreError> { Ok(self.load_schema(namespace)?.indices.iter().any(|s| s.field == field)) };

        if intent.phase == Phase::Dropping {
            for spec in &intent.specs {
                if stop(steps) {
                    return Ok(());
                }
                self.unlist_index(namespace, ns_id, &spec.field)?;
                steps += 1;
            }
            if stop(steps) {
                return Ok(());
            }
            // No field index of the namespace exists now, so no row ID is
            // referenced: free them all before the rebuild allocates new ones.
            self.db.reset_rowmap(ns_id)?;
            if !intent.rebuild {
                return Ok(());
            }
            intent.phase = Phase::Readding;
            self.save_job_progress(key, &intent);
            steps += 1;
        }

        if intent.phase == Phase::Readding {
            for spec in &intent.specs {
                if stop(steps) {
                    return Ok(());
                }
                if !listed(&spec.field)? {
                    self.list_index(namespace, ns_id, spec).await?;
                }
                steps += 1;
            }
            intent.phase = Phase::Building;
            self.save_job_progress(key, &intent);
        }

        while intent.built < intent.specs.len() {
            if stop(steps) {
                return Ok(());
            }
            let spec = intent.specs[intent.built].clone();
            let field_id = self
                .field_id(ns_id, &spec.field)
                .ok_or_else(|| record_error(key, "a re-added field is not registered"))?;
            let mut progress = intent.current.take().unwrap_or_else(|| FieldBuildProgress::new(spec));
            let base = intent.clone();
            let save = |p: &FieldBuildProgress| {
                let mut snapshot = base.clone();
                snapshot.current = Some(p.clone());
                self.save_job_progress(key, &snapshot);
            };
            // Not cancellable, so never stopped part-way.
            let scan = rebuild_field(&self.db, namespace, ns_id, field_id, &mut progress, save, || false).await?;
            debug_assert_eq!(scan, Scan::Finished);
            intent.built += 1;
            self.save_job_progress(key, &intent);
            steps += 1;
        }
        info!("reindex-all '{namespace}': every index rebuilt");
        Ok(())
    }

    /// The fields of `ns_id` a live or failed `reindex-all` has not rebuilt
    /// yet: their indexes hold part of the store, or nothing.
    pub(in crate::doc_store) fn fields_reindexing(&self, ns_id: NamespaceId) -> Result<Vec<String>, DocStoreError> {
        // The status first, from memory: on every query, and most stores have
        // no unfinished reindex-all to parse.
        let key = JobKey::reindex_all(ns_id);
        if !matches!(
            self.jobs.status(key)?,
            Some(BuildStatus::Running | BuildStatus::Pending | BuildStatus::Failed)
        ) {
            return Ok(Vec::new());
        }
        let Some(record) = self.jobs.get(key)? else {
            return Ok(Vec::new());
        };
        let intent: Intent = record
            .progress()
            .ok_or_else(|| record_error(JobKey::reindex_all(ns_id), "no reindex-all progress"))?;
        Ok(intent.unbuilt().map(str::to_owned).collect())
    }

    /// The build in progress of a live `reindex-all` of `ns_id`, and the
    /// fields it has still to build after it, for the progress views.
    pub(in crate::doc_store) fn reindex_all_builds(&self, ns_id: NamespaceId) -> Option<(JobRecord, Option<FieldBuildProgress>, Vec<String>)> {
        let record = self.jobs.get(JobKey::reindex_all(ns_id)).ok()??;
        if !record.is_live() {
            return None;
        }
        let intent: Intent = record.progress()?;
        let mut waiting: Vec<String> = intent.unbuilt().map(str::to_owned).collect();
        let current = intent.current.clone().filter(|_| intent.phase == Phase::Building);
        if current.is_some() && !waiting.is_empty() {
            waiting.remove(0);
        }
        Some((record, current, waiting))
    }

    /// The id of `ns_id`'s registered field `field`.
    fn field_id(&self, ns_id: NamespaceId, field: &str) -> Option<FieldId> {
        self.db
            .list_index_fields(ns_id)
            .iter()
            .find(|f| f.field_name == field)
            .map(|f| f.field_id)
    }

    /// Start a `reindex-all` (or drop-all) and run it for `stop_after` steps,
    /// then return without ending the job, as a crash would leave it.
    #[cfg(test)]
    pub(super) async fn reindex_all_until(&self, namespace: &str, rebuild: bool, stop_after: usize) -> Result<(), DocStoreError> {
        if let Some((key, _)) = self.start_reindex_all(namespace, rebuild)? {
            self.drive_reindex_all(namespace, key, Some(stop_after)).await?;
        }
        Ok(())
    }
}
