//! Unified build-handle registry for both field-index and vector-index builds.
//!
//! [`IndexBuildManager`] replaces the ad-hoc
//! `Arc<Mutex<HashMap<(String, String), IndexBuildHandle>>>` in `AppState`,
//! providing a uniform surface for:
//! - Tracking live builds by [`IndexId`]
//! - Polling progress via [`IndexBuildSnapshot`]
//! - Awaiting completion on graceful shutdown

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use tokio::task::JoinHandle;

use crate::doc_store::error::DocStoreError;
use crate::doc_store::index_progress::{BuildStatus, IndexBuildSnapshot, IndexId, now_ms};
use crate::doc_store::store::IndexBuildHandle;

// ── BuildHandle ───────────────────────────────────────────────────────────────

/// Uniform handle for any in-flight index build.
pub struct ManagedBuildHandle {
    /// Produces a live snapshot without blocking the build task.
    progress_fn: Box<dyn Fn() -> IndexBuildSnapshot + Send + Sync>,
    /// The underlying tokio task.
    join: JoinHandle<Result<(), DocStoreError>>,
}

impl ManagedBuildHandle {
    /// Return a live progress snapshot.
    pub fn snapshot(&self) -> IndexBuildSnapshot {
        (self.progress_fn)()
    }
}

// ── IndexBuildManager ─────────────────────────────────────────────────────────

/// Registry of active index build tasks.
///
/// Holds at most one live handle per [`IndexId`].  Completed handles are not
/// automatically removed — call [`drain_all`] on shutdown to await them all.
///
/// [`drain_all`]: IndexBuildManager::drain_all
pub struct IndexBuildManager {
    handles: Mutex<HashMap<IndexId, ManagedBuildHandle>>,
}

impl Default for IndexBuildManager {
    fn default() -> Self {
        Self {
            handles: Mutex::new(HashMap::new()),
        }
    }
}

impl IndexBuildManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a field-index build handle from `DocStore::add_index`.
    ///
    /// Clones the `Arc<InMemoryProgress>` out of the handle before spawning a
    /// wrapper task so that the progress closure can read live counters without
    /// holding the handle itself.
    pub fn insert_field_build(&self, handle: IndexBuildHandle) {
        let id = IndexId::Field {
            namespace: handle.namespace.clone(),
            field: handle.field.clone(),
        };
        let started = now_ms();
        let mem = Arc::clone(&handle.mem);
        let id2 = id.clone();

        let join = tokio::spawn(async move { handle.wait().await });

        let progress_fn: Box<dyn Fn() -> IndexBuildSnapshot + Send + Sync> = Box::new(move || {
            let total = mem.total.load(Ordering::Relaxed);
            let indexed = mem.indexed.load(Ordering::Relaxed);
            let done = mem.done.load(Ordering::Relaxed);
            let error = mem.error.lock().unwrap().clone();
            let status = if done {
                if error.is_some() { BuildStatus::Failed } else { BuildStatus::Complete }
            } else {
                BuildStatus::Running
            };
            IndexBuildSnapshot {
                kind: id2.kind(),
                id: id2.clone(),
                status,
                total,
                indexed,
                failed: 0,
                started_at_ms: started,
                updated_at_ms: now_ms(),
                completed_at_ms: if done { Some(now_ms()) } else { None },
                last_error: error,
                extra: None,
            }
        });

        self.handles.lock().unwrap().insert(id, ManagedBuildHandle { progress_fn, join });
    }

    /// Insert a build with a custom progress closure and join handle.
    ///
    /// Used for vector-index reindex tracking where the caller composes the
    /// snapshot function directly.
    pub fn insert_with_snapshot(
        &self,
        id: IndexId,
        snapshot_fn: impl Fn() -> IndexBuildSnapshot + Send + Sync + 'static,
        join: JoinHandle<Result<(), DocStoreError>>,
    ) {
        let handle = ManagedBuildHandle {
            progress_fn: Box::new(snapshot_fn),
            join,
        };
        self.handles.lock().unwrap().insert(id, handle);
    }

    /// Return a progress snapshot for the given build, or `None` when no live
    /// handle exists for that index.
    pub fn progress(&self, id: &IndexId) -> Option<IndexBuildSnapshot> {
        let map = self.handles.lock().unwrap();
        map.get(id).map(|h: &ManagedBuildHandle| h.snapshot())
    }

    /// Return snapshots for every tracked build, sorted by namespace name.
    pub fn list(&self) -> Vec<IndexBuildSnapshot> {
        let map = self.handles.lock().unwrap();
        let mut snaps: Vec<IndexBuildSnapshot> = map.values().map(|h: &ManagedBuildHandle| h.snapshot()).collect();
        drop(map);
        snaps.sort_by(|a, b| a.id.namespace().cmp(b.id.namespace()));
        snaps
    }

    /// Await every in-flight build task.  Used during graceful shutdown.
    pub async fn drain_all(&self) {
        let handles: Vec<(IndexId, ManagedBuildHandle)> = {
            let mut map = self.handles.lock().unwrap();
            std::mem::take(&mut *map).into_iter().collect()
        };
        for (id, h) in handles {
            match h.join.await {
                Ok(Ok(())) => log::info!("index build {:?} completed on shutdown", id),
                Ok(Err(e)) => log::error!("index build {:?} failed on shutdown: {e}", id),
                Err(e) => log::error!("index build {:?} panicked on shutdown: {e}", id),
            }
        }
    }
}
