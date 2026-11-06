//! Real KV-backed search() profile (phase breakdown).
//!
//! The `semantic_search` `end_to_end_multichunk` bench runs `search()` over an
//! IN-MEMORY mock `VectorKvStore` whose scan deep-clones entries. This harness runs the
//! same two-pass `search()` over a **real minnal_db-backed `DbVectorStore`** so the
//! sparse scan and dense fetch pay actual LSM + value-log (WiscKey `pread`) cost. It
//! phase-times coarse assignment, the sparse cluster scan (Pass-1 I/O), the dense batch
//! fetch (Pass-2 I/O), and the full `search()`, so we can confirm the scan/fetch share
//! vs. the ~18-20% the isolation bench attributes to the dot arithmetic.
//!
//! `#[ignore]`d — it builds a real on-disk store; run on demand:
//!   cargo test -p minnal_db --all-features --lib real_kv_search_profile -- --ignored --nocapture

use super::super::*;
use crate::AsyncDb;
use crate::semantic_search::cluster::{Cluster, find_closest_cluster_id};
use crate::semantic_search::index::vector_index::VectorKvStore;
use crate::semantic_search::service::{SemanticSearchConfig, search};
use crate::semantic_search::{ClusterIndex, QuantisationStyle, index_embedding_to_cluster};
use std::collections::HashMap;
use std::time::Instant;
use tempfile::TempDir;

const DIM: usize = 768;
const N_BITS_MULTI: usize = 8;

/// Deterministic XorShift64 f32 stream in [-1, 1).
fn prng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as i64 as f32) / (i64::MAX as f32)
    }
}

fn synthetic_chunks(t: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut next = prng(seed);
    (0..t).map(|_| (0..DIM).map(|_| next()).collect()).collect()
}

/// Median of a set of durations in microseconds.
fn median_us(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn real_kv_search_profile() {
    const N_DOCS: u64 = 5_000;
    const T: usize = 4;
    const ITERS: usize = 25;

    // Sweep n_probes to compare the default (64) against the earlier defaults 10 and 32
    // and the upstream 128. The store depends only on cpd (not n_probes), so it is built
    // once per cpd and every n_probes value is profiled against the same on-disk data.
    const N_PROBES_SWEEP: [usize; 4] = [10, 32, 64, 128];

    // 256 synthetic centres, like the store's synthetic chunks.
    let cluster_map: HashMap<u32, Cluster> = synthetic_chunks(256, 0xc0ff_ee00_d15c_0de5)
        .into_iter()
        .enumerate()
        .map(|(id, c)| (id as u32, Cluster::new(id as u32, c)))
        .collect();
    let index = ClusterIndex::from_clusters(cluster_map.clone());
    let base_config = SemanticSearchConfig::default(); // first_pass top_k 1000
    let ns = "profile";

    let dense_query = synthetic_chunks(1, 0xfeed_face_cafe_dead).remove(0);
    let sparse_query = synthetic_chunks(T, 0x00c0_ffee_1234_5678);
    let no_filter: Option<fn(&[u8]) -> bool> = None;

    eprintln!("\n=== real KV-backed search() profile: {N_DOCS} docs, T={T}, dim {DIM}, n_probes sweep {N_PROBES_SWEEP:?} ===");

    for &cpd in &[1usize, 4, 8] {
        let dir = TempDir::new().unwrap();
        let db = AsyncDb::open_with_config(dir.path().to_owned(), crate::support::test_db_config())
            .await
            .unwrap();

        // `upsert_vectors` silently skips a namespace that is not registered (so a
        // dropped store is never resurrected), so the parent must exist first.
        db.namespace(ns.to_string()).await.unwrap();
        create_vector_namespaces(&db, ns).await.unwrap();

        // Populate: one MultiBit dense whole-doc vector + `cpd` SingleBit chunks per
        // doc, quantised against the real centroids and written through the
        // production `upsert_vectors` path (no-WAL, groups sparse by cluster).
        let mut emb = prng(0xdead_beef_0bad_f00d ^ cpd as u64);
        for d in 0..N_DOCS {
            let doc_id = d.to_be_bytes().to_vec();
            let dense_emb: Vec<f32> = (0..DIM).map(|_| emb()).collect();
            let dcid = find_closest_cluster_id(&cluster_map, &dense_emb);
            let mut vis = vec![index_embedding_to_cluster(
                &dense_emb,
                &cluster_map[&dcid],
                QuantisationStyle::MultiBit {
                    number_of_bits: N_BITS_MULTI,
                },
            )];
            for _ in 0..cpd {
                let e: Vec<f32> = (0..DIM).map(|_| emb()).collect();
                let cid = find_closest_cluster_id(&cluster_map, &e);
                vis.push(index_embedding_to_cluster(&e, &cluster_map[&cid], QuantisationStyle::SingleBit));
            }
            upsert_vectors(&db, ns, &doc_id, "text", &vis).await.unwrap();
        }

        let store = DbVectorStore::new(&db, ns).await.unwrap();
        // Refuse to profile an empty index: a silent no-op upsert once made this
        // harness report microsecond "searches" over nothing.
        let all_clusters: Vec<u32> = cluster_map.keys().copied().collect();
        let stored: usize = store.scan_sparse_clusters_batch(&all_clusters).await.unwrap().len();
        assert!(stored > 0, "profile store is empty — upsert_vectors wrote nothing");

        // Pass-2 fetch set is n_probes-independent: first_pass top_k doc ids (bounded by N_DOCS).
        let dense_ids: Vec<Vec<u8>> = (0..N_DOCS.min(base_config.first_pass_sparse_search_top_k as u64))
            .map(|d| d.to_be_bytes().to_vec())
            .collect();

        eprintln!("\n  ┌─ chunks_per_doc = {cpd} ───────────────────────────────────────────");

        for &n_probes in &N_PROBES_SWEEP {
            let config = SemanticSearchConfig {
                probe: crate::semantic_search::ProbeSettings::fixed(n_probes),
                ..base_config.clone()
            };

            // The probe set the harness reuses for the standalone scan-phase timing —
            // identical to what search() computes internally for this n_probes.
            let probe_clusters: Vec<u32> = {
                let mut seen = std::collections::HashSet::new();
                let mut ids = Vec::new();
                for chunk_ids in index.find_top_n_cluster_ids_batch(&sparse_query, n_probes) {
                    for id in chunk_ids {
                        if seen.insert(id) {
                            ids.push(id);
                        }
                    }
                }
                ids
            };

            // Warm the OS page cache / mmap so we measure steady-state, not first-touch.
            for _ in 0..3 {
                let _ = store.scan_sparse_clusters_batch(&probe_clusters).await.unwrap();
                let _ = store.get_dense_entries_batch(&dense_ids).await.unwrap();
                let _ = search(&config, ns, &index, &sparse_query, &dense_query, &store, no_filter, None)
                    .await
                    .unwrap();
            }

            let mut coarse = Vec::with_capacity(ITERS);
            let mut scan = Vec::with_capacity(ITERS);
            let mut dense = Vec::with_capacity(ITERS);
            let mut total = Vec::with_capacity(ITERS);
            let mut result_len = 0usize;
            let mut scanned_entries = 0usize;
            for _ in 0..ITERS {
                let t = Instant::now();
                let probes = index.find_top_n_cluster_ids_batch(&sparse_query, n_probes);
                let mut seen = std::collections::HashSet::new();
                let mut pc = Vec::new();
                for chunk_ids in probes {
                    for id in chunk_ids {
                        if seen.insert(id) {
                            pc.push(id);
                        }
                    }
                }
                coarse.push(t.elapsed().as_nanos() as f64 / 1e3);

                let t = Instant::now();
                let scanned = store.scan_sparse_clusters_batch(&pc).await.unwrap();
                scan.push(t.elapsed().as_nanos() as f64 / 1e3);
                scanned_entries = scanned.len();

                let t = Instant::now();
                let _ = store.get_dense_entries_batch(&dense_ids).await.unwrap();
                dense.push(t.elapsed().as_nanos() as f64 / 1e3);

                let t = Instant::now();
                let results = search(&config, ns, &index, &sparse_query, &dense_query, &store, no_filter, None)
                    .await
                    .unwrap();
                total.push(t.elapsed().as_nanos() as f64 / 1e3);
                result_len = results.len();
            }

            let (c, s, dd, tot) = (median_us(coarse), median_us(scan), median_us(dense), median_us(total));
            let other = (tot - c - s - dd).max(0.0);
            let pass1 = c + s; // coarse cluster pick + sparse scan I/O
            eprintln!(
                "  n_probes={n_probes:>3}: {} clusters, {scanned_entries} sparse entries scanned, {result_len} results",
                probe_clusters.len(),
            );
            eprintln!("     Pass-1 coarse pick     {c:>9.1} µs  ({:>4.1}%)", 100.0 * c / tot);
            eprintln!("     Pass-1 sparse scan I/O {s:>9.1} µs  ({:>4.1}%)", 100.0 * s / tot);
            eprintln!("       └ Pass-1 I/O total   {pass1:>9.1} µs  ({:>4.1}%)", 100.0 * pass1 / tot);
            eprintln!("     Pass-2 dense fetch I/O {dd:>9.1} µs  ({:>4.1}%)", 100.0 * dd / tot);
            eprintln!(
                "     scoring: MaxSim+rerank {other:>9.1} µs  ({:>4.1}%)  [total − above]",
                100.0 * other / tot
            );
            eprintln!("     ── entire search()     {tot:>9.1} µs  (100%)");
        }

        db.shutdown().await.unwrap();
    }
}
