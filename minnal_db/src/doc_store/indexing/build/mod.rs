//! Field-index jobs: adding an index (a `field_build` job, which activates it
//! and then indexes every document) and dropping one (a `field_drop` job).
//!
//! Each starts as a job of the lifecycle layer (`namespace_lifecycle::Jobs`):
//! its record is written and the store moved to `IndexMaintenance` before any
//! work, and the governor runs the body below, supervised. A build's progress
//! is kept in its record ([`FieldBuildProgress`]), so a restart resumes it
//! where it got to; a cancel stops it at its next page and drops the
//! half-built index.

mod rebuild;
#[cfg(test)]
mod tests;

pub(in crate::doc_store) use rebuild::{Scan, rebuild_field};

use crate::doc_store::indexing::progress::FieldBuildProgress;
use crate::doc_store::*;
use crate::namespace::NamespaceId;
use crate::namespace_lifecycle::{JobKind, JobRecord};

impl DocStore {
    // ── Add ────────────────────────────────────────────────────────────────

    /// Add an index to `namespace` and build it; returns once the build has
    /// ended. See [`begin_add_index`](Self::begin_add_index) for the form
    /// that returns once the build has started.
    ///
    /// The index is activated before its build starts, so writes are indexed
    /// from then on; the build then indexes every document already stored.
    /// Queries on the field report it in `degraded_fields` until the build is
    /// complete.
    ///
    /// Refused with [`DocStoreError::IndexAlreadyExists`] if the field is
    /// indexed, and with [`DocStoreError::JobConflict`] while another job on
    /// the field (or a `reindex-all`) runs.
    pub async fn add_index(&self, namespace: &str, spec: IndexSpec) -> Result<(), DocStoreError> {
        let key = self.start_field_build(namespace, spec).await?;
        self.run_job(namespace, key).await
    }

    /// [`add_index`](Self::add_index), returning the job's name
    /// (`field_build:{field}`) once the index is activated and its build
    /// handed to the lifecycle governor, which the store starts when it opens.
    pub async fn begin_add_index(&self, namespace: &str, spec: IndexSpec) -> Result<String, DocStoreError> {
        self.running_governor()?;
        let name = format!("{}:{}", JobKind::FieldBuild(0).name(), spec.field);
        let key = self.start_field_build(namespace, spec).await?;
        self.submit_job(namespace, key)?;
        Ok(name)
    }

    /// Start a `field_build` job for `spec`: check it, record the job, and
    /// activate and list the index. The job's body then does the build.
    pub(in crate::doc_store) async fn start_field_build(&self, namespace: &str, spec: IndexSpec) -> Result<JobKey, DocStoreError> {
        self.job_runner_ready()?;
        let schema = self.load_schema(namespace)?;
        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;
        // Saving the schema does not run `validate`, so check the name here.
        crate::doc_store::schema::validate_index_field_name(&spec.field)?;
        if schema.indices.iter().any(|s| s.field == spec.field) {
            return Err(DocStoreError::IndexAlreadyExists {
                namespace: namespace.to_owned(),
                field: spec.field.clone(),
            });
        }
        // The per-namespace cap: saving the schema does not validate, so
        // without this the store could pass MAX_INDICES one field at a time.
        if schema.indices.len() >= crate::doc_store::schema::MAX_INDICES {
            return Err(DocStoreError::Schema(SchemaError::TooManyIndices {
                count: schema.indices.len() + 1,
                max: crate::doc_store::schema::MAX_INDICES,
            }));
        }
        // A field indexed before keeps its id. It is registered again (which
        // clears its `dropped` flag) only once the job has started: a drop of
        // it may still be running, and the start is what refuses that.
        let field_id = match self.db.list_index_fields(ns_id).iter().find(|f| f.field_name == spec.field) {
            Some(meta) => meta.field_id,
            None => self.db.register_index_field(ns_id, &spec.field, to_ivt(spec.index_type))?,
        };
        let key = JobKey::field_build(ns_id, field_id);
        self.start_job(namespace, key, &spec.field, &FieldBuildProgress::new(spec.clone()))?;
        info!(
            "adding index '{}' (type={:?}, field_id={field_id}) to namespace '{namespace}' (ns_id={ns_id})",
            spec.field, spec.index_type
        );
        if let Err(e) = self.list_index(namespace, ns_id, &spec).await {
            // The job never got going: record it failed, and take back what
            // was done (an index activated but not listed).
            self.jobs.finish(self, namespace, key, Err(e.to_string()))?;
            self.unlist_index(namespace, ns_id, &spec.field)?;
            return Err(e);
        }
        Ok(key)
    }

    /// The body of a `field_build` job: list the index if a crash cut the
    /// start short, then index every document from where the record says the
    /// build got to. On a cancel, the half-built index is dropped.
    pub(in crate::doc_store) async fn field_build_job(&self, namespace: &str, key: JobKey) -> Result<(), DocStoreError> {
        let JobKind::FieldBuild(field_id) = key.kind else {
            unreachable!("a field build's key")
        };
        let Some(record) = self.jobs.get(key)?.filter(JobRecord::is_live) else {
            return Ok(());
        };
        let mut progress: FieldBuildProgress = record.progress().ok_or_else(|| record_error(key, "no build progress"))?;
        let ns_id = key.namespace;
        if !self.load_schema(namespace)?.indices.iter().any(|s| s.field == progress.spec.field) {
            self.list_index(namespace, ns_id, &progress.spec).await?;
        }
        if progress.resume_after_hex.is_some() {
            info!(
                "resuming the index build of '{}' in '{namespace}' at {}/{}",
                progress.spec.field, progress.indexed, progress.total
            );
        }
        let save = |p: &FieldBuildProgress| self.save_job_progress(key, p);
        let cancelled = || self.jobs.is_cancelling(key);
        match rebuild_field(&self.db, namespace, ns_id, field_id, &mut progress, save, cancelled).await? {
            Scan::Finished => Ok(()),
            Scan::Cancelled => {
                info!("index build of '{}' in '{namespace}' cancelled; dropping the index", progress.spec.field);
                self.unlist_index(namespace, ns_id, &progress.spec.field)
            }
        }
    }

    /// Activate `spec`'s index, empty, for a build to fill, and list it in the
    /// schema (moving the field out of the plain attributes). Done already,
    /// nothing changes.
    pub(in crate::doc_store) async fn list_index(&self, namespace: &str, ns_id: NamespaceId, spec: &IndexSpec) -> Result<(), DocStoreError> {
        let ivt = to_ivt(spec.index_type);
        let field_id = self.db.register_index_field(ns_id, &spec.field, ivt)?;
        if self.db.field_index_distinct_count(ns_id, field_id).is_none() {
            let extractor = json_extractor(spec.field.clone(), spec.index_type);
            self.db.activate_field_index_for_build(ns_id, field_id, ivt, extractor).await?;
        }
        self.schemas.update_doc(namespace, |schema| {
            if schema.indices.iter().any(|s| s.field == spec.field) {
                return Ok(());
            }
            if schema.indices.len() >= crate::doc_store::schema::MAX_INDICES {
                return Err(DocStoreError::Schema(SchemaError::TooManyIndices {
                    count: schema.indices.len() + 1,
                    max: crate::doc_store::schema::MAX_INDICES,
                }));
            }
            schema.attributes.retain(|a| a.name != spec.field);
            schema.indices.push(spec.clone());
            Ok(())
        })
    }

    // ── Drop ───────────────────────────────────────────────────────────────

    /// Drop `field`'s index from `namespace`, as a `field_drop` job; returns
    /// once it is gone.
    ///
    /// The field becomes a plain attribute of the same type: its data stays
    /// in every document, and only the index files
    /// (`field-index/{ns_id}/{field_id}/`) are deleted. The field keeps its
    /// id, so adding the index again reuses it. Refused with
    /// [`DocStoreError::JobConflict`] while the field's build runs (cancel it
    /// first).
    pub async fn drop_index(&self, namespace: &str, field: &str) -> Result<(), DocStoreError> {
        let key = self.start_field_drop(namespace, field)?;
        self.run_job(namespace, key).await
    }

    /// [`drop_index`](Self::drop_index), returning the job's name
    /// (`field_drop:{field}`) once it is handed to the governor.
    pub fn begin_drop_index(&self, namespace: &str, field: &str) -> Result<String, DocStoreError> {
        self.running_governor()?;
        let key = self.start_field_drop(namespace, field)?;
        self.submit_job(namespace, key)?;
        Ok(format!("{}:{field}", JobKind::FieldDrop(0).name()))
    }

    fn start_field_drop(&self, namespace: &str, field: &str) -> Result<JobKey, DocStoreError> {
        self.job_runner_ready()?;
        let schema = self.load_schema(namespace)?;
        let ns_id = schema.ns_id.ok_or_else(|| DocStoreError::MissingNsId {
            namespace: namespace.to_owned(),
        })?;
        let not_found = || DocStoreError::IndexNotFound {
            namespace: namespace.to_owned(),
            field: field.to_owned(),
        };
        if !schema.indices.iter().any(|s| s.field == field) {
            return Err(not_found());
        }
        let fields = self.db.list_index_fields(ns_id);
        let meta = fields.iter().find(|f| f.field_name == field).ok_or_else(not_found)?;
        let key = JobKey::field_drop(ns_id, meta.field_id);
        self.start_job(namespace, key, field, &serde_json::Value::Null)?;
        info!("dropping index '{field}' from namespace '{namespace}'");
        Ok(key)
    }

    /// The body of a `field_drop` job.
    pub(in crate::doc_store) async fn field_drop_job(&self, namespace: &str, key: JobKey) -> Result<(), DocStoreError> {
        let JobKind::FieldDrop(field_id) = key.kind else {
            unreachable!("a field drop's key")
        };
        let fields = self.db.list_index_fields(key.namespace);
        if let Some(meta) = fields.iter().find(|f| f.field_id == field_id) {
            self.unlist_index(namespace, key.namespace, &meta.field_name)?;
        }
        Ok(())
    }

    /// Take `field`'s index out of the schema, demoted to a plain attribute,
    /// then drop it in the engine. Done already, nothing changes.
    ///
    /// The schema goes first. The other order had a crash window in which the
    /// files were gone but the schema still listed the index: the field was
    /// activated again at the next open, found no checkpoint and no data, and
    /// read as a new, empty index — silently incomplete (FR-001). The engine
    /// persists the field's `dropped` flag before it deletes anything, so a
    /// crash part-way through its drop is finished at the next open.
    pub(in crate::doc_store) fn unlist_index(&self, namespace: &str, ns_id: NamespaceId, field: &str) -> Result<(), DocStoreError> {
        self.schemas.update_doc(namespace, |schema| {
            if let Some(pos) = schema.indices.iter().position(|s| s.field == field) {
                let spec = schema.indices.remove(pos);
                if !schema.attributes.iter().any(|a| a.name == spec.field) {
                    schema.attributes.push(crate::doc_store::schema::AttributeDef {
                        name: spec.field.clone(),
                        attr_type: index_type_to_attr_type(spec.index_type),
                        description: Some("previously indexed; field data is still present in stored documents".to_owned()),
                    });
                }
            }
            Ok(())
        })?;
        let fields = self.db.list_index_fields(ns_id);
        if let Some(meta) = fields.iter().find(|f| f.field_name == field) {
            self.db.drop_field_index(ns_id, meta.field_id)?;
            // An ended build's record would describe the index a later add
            // creates, which is a new one.
            self.jobs.forget(JobKey::field_build(ns_id, meta.field_id))?;
        }
        info!("index '{field}' dropped from namespace '{namespace}'");
        Ok(())
    }
}

/// A job record that does not hold what its body needs.
pub(in crate::doc_store) fn record_error(key: JobKey, what: &str) -> DocStoreError {
    crate::namespace_lifecycle::LifecycleError::Record(format!("job {}: {what}", key.kind)).into()
}
