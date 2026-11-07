//! Shared corpus indexing for the real-embedding harnesses.
//!
//! Used by `semantic_search::eval::beir`: embeds each text
//! through the real embedding service and upserts its vectors under a `u64` BE doc id.

use crate::AsyncDb;
use crate::semantic_search::ClusterIndex;
use crate::semantic_search::service::{SemanticSearchConfig, embed_document};
use crate::semantic_search::vector_kv::upsert_vectors;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Embed and index `docs` (`(doc_id, text)`) into `ns` with at most `concurrency`
/// embedding requests in flight. Returns `(indexed, failed, first_error)`.
pub(crate) async fn index_texts(
    db: &Arc<AsyncDb>,
    ns: &str,
    config: &Arc<SemanticSearchConfig>,
    index: &Arc<ClusterIndex>,
    docs: Vec<(u64, String)>,
    concurrency: usize,
) -> (usize, usize, Option<String>) {
    // `upsert_vectors` silently skips a namespace without its vector namespaces
    // (so a dropped store is never resurrected). Without this every upsert is
    // an `Ok` no-op and the index stays empty.
    if let Err(e) = crate::semantic_search::vector_kv::create_vector_namespaces(db, ns).await {
        return (0, docs.len(), Some(format!("create namespace '{ns}': {e}")));
    }
    let sem = Arc::new(Semaphore::new(concurrency));
    let mut set = tokio::task::JoinSet::new();
    for (id, text) in docs {
        let permit = sem.clone().acquire_owned().await.unwrap();
        let (cfg, idx, db, ns) = (config.clone(), index.clone(), db.clone(), ns.to_string());
        set.spawn(async move {
            let _permit = permit;
            let doc_id = id.to_be_bytes().to_vec();
            match embed_document(&cfg, &idx, &text).await {
                Ok(vis) => upsert_vectors(&db, &ns, &doc_id, "text", &vis).await.map_err(|e| e.to_string()),
                Err(e) => Err(e.to_string()),
            }
        });
    }
    let (mut ok, mut fail) = (0usize, 0usize);
    let mut first_err: Option<String> = None;
    while let Some(res) = set.join_next().await {
        match res.unwrap() {
            Ok(_) => ok += 1,
            Err(e) => {
                fail += 1;
                first_err.get_or_insert(e);
            }
        }
    }
    (ok, fail, first_err)
}
