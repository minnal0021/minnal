//! Background maintenance of every namespace's partition (design doc M3a).
//!
//! One task per process. The vector worker wakes it with a namespace's name
//! whenever a write leaves a posting over its namespace's split limit; it then
//! splits that namespace's oversized postings ([`ivf_split::split_oversized`])
//! with the namespace's own `maintenance` settings. At most `threads`
//! namespaces are maintained at once, and one namespace is never maintained by
//! two at once (its partition's maintenance lock). The 2-means runs on the
//! blocking pool, never on the rayon pool searches share (the `SCORING_GATE`
//! rules in `semantic_search`).
//!
//! [`recover_all`] finishes or abandons the splits a crash interrupted. The
//! vector worker runs it before its first pass, so nothing writes a namespace's
//! vectors while its journal is unfinished.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use log::{info, warn};
use tokio::sync::{Notify, Semaphore, mpsc};

use crate::AsyncDb;
use crate::doc_store::store::{SemanticSearchContext, load_vector_settings};
use crate::ivf_split;

/// Default maintenance threads per process.
pub const DEFAULT_MAINTENANCE_THREADS: usize = 2;

/// A running maintenance task. [`shutdown`](Self::shutdown) stops it after the
/// splits in progress finish; dropping it stops it too.
pub(crate) struct MaintenanceHandle {
    stop: Arc<Notify>,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl MaintenanceHandle {
    /// Stop the task and wait for it.
    pub(crate) async fn shutdown(mut self) {
        self.stop.notify_one();
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }
}

impl Drop for MaintenanceHandle {
    fn drop(&mut self) {
        self.stop.notify_one();
    }
}

/// Start the task; `ctx.request_maintenance` reaches it from then on.
pub(crate) fn start(db: Arc<AsyncDb>, ctx: Arc<SemanticSearchContext>, schema_dir: PathBuf) -> MaintenanceHandle {
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    ctx.set_maintenance_sender(tx);
    let stop = Arc::new(Notify::new());
    let permits = Arc::new(Semaphore::new(ctx.maintenance_threads.max(1)));
    let schema_dir = Arc::new(schema_dir);
    let task = {
        let stop = Arc::clone(&stop);
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
                    let (db, ctx, schema_dir) = (Arc::clone(&db), Arc::clone(&ctx), Arc::clone(&schema_dir));
                    running.spawn(async move {
                        maintain(&db, &ctx, &schema_dir, &namespace).await;
                        drop(permit);
                    });
                }
                while running.try_join_next().is_some() {}
            }
            while running.join_next().await.is_some() {}
        })
    };
    MaintenanceHandle { stop, task: Some(task) }
}

/// Split `namespace`'s oversized postings with its own settings.
async fn maintain(db: &AsyncDb, ctx: &SemanticSearchContext, schema_dir: &std::path::Path, namespace: &str) {
    let result = async {
        let (settings, ns_id) = load_vector_settings(schema_dir, namespace)?;
        let ns = ctx.for_namespace(db, namespace, ns_id, &settings).await?;
        let made = ivf_split::split_oversized(db, namespace, &ns.partition, &settings.maintenance.split_settings()).await?;
        Ok::<usize, crate::doc_store::error::DocStoreError>(made)
    }
    .await;
    match result {
        Ok(0) => {}
        Ok(n) => info!("partition maintenance: namespace='{namespace}' split {n} posting(s)"),
        Err(e) => warn!("partition maintenance: namespace='{namespace}' failed: {e}"),
    }
}

/// Finish or abandon every split a crash left unfinished, in every namespace,
/// then ask for maintenance of every namespace with a partition (a crash may
/// have left oversized postings unsplit). Returns how many unfinished splits
/// were found.
pub(crate) async fn recover_all(db: &AsyncDb, ctx: &SemanticSearchContext, schema_dir: &std::path::Path) -> usize {
    let names: Vec<String> = db.list_namespaces().into_iter().map(|(n, _)| n).collect();
    let mut found = 0;
    for name in &names {
        let Some(namespace) = name.strip_suffix("_ivf_journal") else { continue };
        let result = async {
            let (settings, ns_id) = load_vector_settings(schema_dir, namespace)?;
            let ns = ctx.for_namespace(db, namespace, ns_id, &settings).await?;
            Ok::<usize, crate::doc_store::error::DocStoreError>(ivf_split::recover(db, namespace, &ns.partition).await?)
        }
        .await;
        match result {
            Ok(0) => {}
            Ok(n) => {
                info!("partition recovery: namespace='{namespace}' finished {n} interrupted split(s)");
                found += n;
            }
            Err(e) => warn!("partition recovery: namespace='{namespace}' failed: {e}"),
        }
    }
    for name in &names {
        if let Some(namespace) = name.strip_suffix("_ivf_postings")
            && load_vector_settings(schema_dir, namespace).is_ok()
        {
            ctx.request_maintenance(namespace);
        }
    }
    found
}
