//! Background maintenance of every namespace's partition (design doc M3a, M3b).
//!
//! One task per process. The vector worker wakes it with a namespace's name
//! whenever a write leaves a posting over its namespace's split limit, or a
//! delete leaves one at or under its merge limit; it then splits, reassigns and
//! merges that namespace's postings ([`ivf_split::maintain_partition`]) with
//! the namespace's own `maintenance` settings. At most `threads`
//! namespaces are maintained at once, and one namespace is never maintained by
//! two at once (its partition's maintenance lock). The 2-means runs on the
//! blocking pool, never on the rayon pool searches share (the `SCORING_GATE`
//! rules in `semantic_search`).
//!
//! [`recover_all`] finishes or abandons the operations a crash interrupted. The
//! vector worker runs it before its first pass, so nothing writes a namespace's
//! vectors while its journal is unfinished.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use log::{info, warn};
use tokio::sync::{Notify, Semaphore, mpsc};

use crate::AsyncDb;
use crate::doc_store::schema_store::SchemaStore;
use crate::doc_store::store::SemanticSearchContext;
use crate::ivf_split;

/// Default maintenance threads per process.
pub const DEFAULT_MAINTENANCE_THREADS: usize = 2;

/// A running maintenance task. [`shutdown`](Self::shutdown) stops it after the
/// split or merge in progress in each namespace finishes, leaving the rest for
/// the next start (startup recovery asks for maintenance of every namespace);
/// dropping it stops it too.
pub(crate) struct MaintenanceHandle {
    stop: Arc<Notify>,
    /// Set on shutdown; running jobs check it between operations.
    stopping: Arc<AtomicBool>,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl MaintenanceHandle {
    /// Stop the task and wait for it.
    pub(crate) async fn shutdown(mut self) {
        self.stopping.store(true, Ordering::Release);
        self.stop.notify_one();
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }
}

impl Drop for MaintenanceHandle {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::Release);
        self.stop.notify_one();
    }
}

/// Start the task; `ctx.request_maintenance` reaches it from then on.
pub(crate) fn start(db: Arc<AsyncDb>, ctx: Arc<SemanticSearchContext>, schemas: Arc<SchemaStore>) -> MaintenanceHandle {
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    ctx.set_maintenance_sender(tx);
    let stop = Arc::new(Notify::new());
    let stopping = Arc::new(AtomicBool::new(false));
    let permits = Arc::new(Semaphore::new(ctx.maintenance_threads.max(1)));
    let task = {
        let stop = Arc::clone(&stop);
        let stopping = Arc::clone(&stopping);
        tokio::spawn(async move {
            let mut running = tokio::task::JoinSet::new();
            loop {
                let first = tokio::select! {
                    _ = stop.notified() => break,
                    msg = rx.recv() => match msg { Some(ns) => ns, None => break },
                };
                // Several wake-ups for one namespace are one job.
                let mut pending: HashSet<String> = HashSet::from([first]);
                while let Ok(ns) = rx.try_recv() {
                    pending.insert(ns);
                }
                for namespace in pending {
                    let Ok(permit) = Arc::clone(&permits).acquire_owned().await else { break };
                    if stopping.load(Ordering::Acquire) {
                        break;
                    }
                    let (db, ctx, schemas, stopping) = (Arc::clone(&db), Arc::clone(&ctx), Arc::clone(&schemas), Arc::clone(&stopping));
                    running.spawn(async move {
                        maintain(&db, &ctx, &schemas, &namespace, &stopping).await;
                        drop(permit);
                    });
                }
                while running.try_join_next().is_some() {}
            }
            while running.join_next().await.is_some() {}
        })
    };
    MaintenanceHandle {
        stop,
        stopping,
        task: Some(task),
    }
}

/// Split, reassign and merge `namespace`'s postings with its own settings,
/// until none is left or `stopping` is set.
async fn maintain(db: &AsyncDb, ctx: &SemanticSearchContext, schemas: &SchemaStore, namespace: &str, stopping: &AtomicBool) {
    let result = async {
        let (settings, ns_id) = schemas.vector_settings(namespace)?;
        let ns = ctx.for_namespace(db, namespace, ns_id, &settings).await?;
        let report = ivf_split::maintain_partition_until(db, namespace, &ns.partition, &settings.maintenance.ops(), stopping).await?;
        Ok::<ivf_split::MaintenanceReport, crate::doc_store::error::DocStoreError>(report)
    }
    .await;
    match result {
        Ok(r) if r == ivf_split::MaintenanceReport::default() => {}
        Ok(r) => info!(
            "partition maintenance: namespace='{namespace}' split {} posting(s), merged {}",
            r.splits, r.merges
        ),
        Err(e) => warn!("partition maintenance: namespace='{namespace}' failed: {e}"),
    }
    if stopping.load(Ordering::Acquire) {
        info!("partition maintenance: namespace='{namespace}' stopped for shutdown; resumes at the next start");
    }
}

/// Finish or abandon every operation a crash left unfinished, in every
/// namespace with a journal, then ask for maintenance of every namespace with
/// a partition (a crash may have left postings to split or merge). Loading a
/// partition recovers its journal ([`SemanticSearchContext::for_namespace`]),
/// so this only does it early; a namespace whose recovery fails here is tried
/// again by its next user, and nothing uses it before that succeeds.
pub(crate) async fn recover_all(db: &AsyncDb, ctx: &SemanticSearchContext, schemas: &SchemaStore) {
    let names: Vec<String> = db.list_namespaces().into_iter().map(|(n, _)| n).collect();
    for name in &names {
        let Some(namespace) = name.strip_suffix("_ivf_journal") else { continue };
        let result = async {
            let (settings, ns_id) = schemas.vector_settings(namespace)?;
            ctx.for_namespace(db, namespace, ns_id, &settings).await?;
            Ok::<(), crate::doc_store::error::DocStoreError>(())
        }
        .await;
        if let Err(e) = result {
            warn!("partition recovery: namespace='{namespace}' failed: {e}; retried by its next use");
        }
    }
    for name in &names {
        if let Some(namespace) = name.strip_suffix("_ivf_postings")
            && schemas.vector_settings(namespace).is_ok()
        {
            ctx.request_maintenance(namespace);
        }
    }
}
