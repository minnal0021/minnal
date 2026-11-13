//! Query embedding cache: the system-wide TTL namespace of whole-query embeddings.

use std::time::Duration;

use crate::AsyncDb;

/// System-wide namespace for caching query embeddings.
///
/// A single shared TTL-enabled store under the `system` namespace; all doc-store
/// namespaces with semantic search enabled write to and read from this cache
/// regardless of which namespace the query originated from.
///
/// Keys are `model ‖ 0x00 ‖ query` (see [`query_cache_key`]); values are
/// big-endian `f32` vectors.
/// The namespace is TTL-enabled; the expiry duration is supplied by the caller
/// (configurable via `[semantic_search] query_embedding_cache_ttl_secs`), so
/// stale embeddings are evicted automatically.
///
/// **Durability:** cache populates use the **no-WAL** path (`put_no_wal`). The
/// cache is best-effort and fully regenerable — every entry can be recomputed
/// by calling the embedding service — so paying a per-write WAL fsync would add
/// query latency for no durability benefit. A crash that drops un-flushed
/// populates just produces future cache misses.
///
/// The administrative clear ([`clear_cached_query_embeddings`]) is the
/// exception: its deletes are **WAL-backed**. Cached entries are periodically
/// flushed to disk, so a no-WAL tombstone lost to a crash before the next flush
/// tick would resurrect an entry the operator explicitly cleared — and keep
/// serving it until the TTL expires.
const SYSTEM_QUERY_EMB_CACHE_NS: &str = "system_qemb_cache";

/// Maximum records the TTL worker will delete per pass.
const QUERY_EMBEDDING_CACHE_MAX_DELETES: usize = 10_000;

/// The cache key for `query_text` embedded by `model` at `dim` dimensions:
/// `model ‖ 0x00 ‖ dim (u32 BE) ‖ query`.
///
/// The model is part of the key because one embedding service can serve
/// several models, and the same text embeds to unrelated vectors in each; the
/// dimension because each namespace picks its own, and the same model at two
/// dimensions returns different vectors. Without them, namespaces with
/// different models or dimensions would share cached query vectors. The NUL
/// separator keeps `("ab", "c")` and `("a", "bc")` apart.
pub(super) fn query_cache_key(model: &str, dim: usize, query_text: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(model.len() + 5 + query_text.len());
    key.extend_from_slice(model.as_bytes());
    key.push(0);
    key.extend_from_slice(&(dim as u32).to_be_bytes());
    key.extend_from_slice(query_text.as_bytes());
    key
}

/// Look up a cached query embedding in the system-wide TTL cache.
///
/// Keyed by embedding model, dimension and query text. Returns the whole-query embedding, which serves both
/// search passes (queries are not chunked; see
/// [`embed_query`](crate::semantic_search::service::embed_query)).
///
/// Returns `None` on cache miss, dimension mismatch, or any I/O error so that
/// the caller always falls back to the embedding service transparently.
pub async fn get_cached_query_embedding(db: &AsyncDb, model: &str, query_text: &str, expected_dim: usize, ttl: Duration) -> Option<Vec<f32>> {
    let cache_ns = db
        .namespace_with_ttl(SYSTEM_QUERY_EMB_CACHE_NS.to_string(), ttl, QUERY_EMBEDDING_CACHE_MAX_DELETES)
        .await
        .ok()?;

    let bytes = cache_ns.get(query_cache_key(model, expected_dim, query_text)).await.ok()??;
    bytes_to_f32_vec_list(&bytes, expected_dim)?.into_iter().next()
}

/// Store `model`'s whole-query embedding of `query_text` in the system-wide TTL
/// cache (see [`get_cached_query_embedding`]). Failures are silently ignored — the cache
/// is best-effort and must never block or fail a query.
pub async fn put_cached_query_embedding(db: &AsyncDb, model: &str, query_text: &str, dense: &[f32], ttl: Duration) {
    let Ok(cache_ns) = db
        .namespace_with_ttl(SYSTEM_QUERY_EMB_CACHE_NS.to_string(), ttl, QUERY_EMBEDDING_CACHE_MAX_DELETES)
        .await
    else {
        return;
    };

    // No-WAL: the cache is best-effort, TTL-bounded and fully regenerable by
    // re-calling the embedding service on a miss, so a WAL fsync per populate
    // would be pure query-path latency with no durability benefit. A crash that
    // drops the entry simply turns into a future cache miss.
    let _ = cache_ns
        .put_no_wal(query_cache_key(model, dense.len(), query_text), f32_vec_list_to_bytes(&[dense.to_vec()]))
        .await;
}

/// Up to `limit` cached whole-query embeddings of `model` at `dim` dimensions:
/// real queries recently searched, which the recall check (FR-007) prefers to
/// queries made from stored chunks. One bounded page read, starting at a point
/// in the key space chosen by `seed` and wrapping once, so a large cache is not
/// read in full and the page does not always begin at the same query text.
pub async fn sample_cached_query_embeddings(
    db: &AsyncDb,
    model: &str,
    dim: usize,
    ttl: Duration,
    limit: usize,
    seed: u64,
) -> Result<Vec<Vec<f32>>, crate::KVError> {
    let cache_ns = db
        .namespace_with_ttl(SYSTEM_QUERY_EMB_CACHE_NS.to_string(), ttl, QUERY_EMBEDDING_CACHE_MAX_DELETES)
        .await?;
    let prefix = query_cache_key(model, dim, "");
    // The key just past every key with this prefix: the prefix ends in the
    // dimension's last byte, so add one to it with carry.
    let mut end = prefix.clone();
    while let Some(last) = end.pop() {
        if last < u8::MAX {
            end.push(last + 1);
            break;
        }
    }
    let end = (!end.is_empty()).then_some(end);
    let mut start = prefix.clone();
    start.push((seed % 256) as u8);
    let mut out = Vec::new();
    for (from, to) in [(start.clone(), end.clone()), (prefix, Some(start))] {
        if out.len() >= limit {
            break;
        }
        let (rows, _) = cache_ns.scan(Some(from), to, limit - out.len()).await?;
        out.extend(rows.into_iter().filter_map(|(_, v)| bytes_to_f32_vec_list(&v, dim)?.into_iter().next()));
    }
    Ok(out)
}

/// Delete every entry from the system-wide query-embedding cache.
///
/// Returns the number of cached entries removed. This is an explicit
/// administrative operation. Entries are keyed by model and query text, so
/// neither switching models nor changing chunking settings (queries are not
/// chunked) needs a clear. Clear it when the service starts producing different
/// vectors under the same model name (a different model build behind the name),
/// or stale vectors are served until the configured TTL expires.
pub async fn clear_cached_query_embeddings(db: &AsyncDb, ttl: Duration) -> Result<usize, crate::KVError> {
    let cache_ns = db
        .namespace_with_ttl(SYSTEM_QUERY_EMB_CACHE_NS.to_string(), ttl, QUERY_EMBEDDING_CACHE_MAX_DELETES)
        .await?;

    let mut deleted = 0usize;
    let mut cursor = None;
    loop {
        let (keys, next) = cache_ns.key_page(cursor, None, crate::kv_store::KEY_PAGE).await?;
        for key in keys {
            // WAL-backed, unlike the populate path: the entries being cleared may
            // already be flushed to disk, so a no-WAL tombstone lost to a crash
            // before the next flush tick would bring them back. A clear is a rare
            // admin operation; the per-delete fsync is off the query path.
            cache_ns.delete(key).await?;
            deleted += 1;
        }
        match next {
            Some(next) => cursor = Some(next),
            None => return Ok(deleted),
        }
    }
}

/// Encode a list of `Vec<f32>` embeddings as bytes.
///
/// Format: `[count: 4B BE u32][f32s for embedding 0][f32s for embedding 1]...`
/// Each f32 is encoded as 4 big-endian bytes.
pub(super) fn f32_vec_list_to_bytes(list: &[Vec<f32>]) -> Vec<u8> {
    let count = list.len() as u32;
    let mut out = Vec::with_capacity(4 + list.iter().map(|v| v.len() * 4).sum::<usize>());
    out.extend_from_slice(&count.to_be_bytes());
    for v in list {
        out.extend(v.iter().flat_map(|f| f.to_be_bytes()));
    }
    out
}

/// Decode bytes back into a list of `Vec<f32>` embeddings.
///
/// Returns `None` if the bytes are malformed or any embedding has the wrong dimension.
pub(super) fn bytes_to_f32_vec_list(bytes: &[u8], expected_dim: usize) -> Option<Vec<Vec<f32>>> {
    if bytes.len() < 4 {
        return None;
    }
    let count = u32::from_be_bytes(bytes[..4].try_into().ok()?) as usize;
    let rest = &bytes[4..];
    if rest.len() != count * expected_dim * 4 {
        return None;
    }
    Some(
        rest.chunks_exact(expected_dim * 4)
            .map(|chunk| chunk.as_chunks::<4>().0.iter().map(|c| f32::from_be_bytes(*c)).collect())
            .collect(),
    )
}
