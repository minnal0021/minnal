//! Pass-1 search over a real on-disk store: the sparse scan alone and the whole
//! two-pass `search()`, at three shares of the postings probed.
//!
//!   cargo bench -p minnal_db --features semantic-search --bench bench_pass1_search
//!
//! `MINNAL_BENCH_PASS1_BITS` (1, 2 or 4; default 1) sets the chunk code width;
//! its group names then end in `@{bits}b`.
//!
//! The store holds `N_DOCS` documents with one chunk code each, filed under
//! the nearest of `N_POSTINGS` centres, plus one 8-bit whole-document code each,
//! written the way production stores them (`{ns}_sparse_vector` keyed
//! `posting ‖ doc_id`, `{ns}_dense_vector` keyed `doc_id`) and compacted into L1.
//! Two value-log layouts are measured:
//!
//! - `grouped`: written posting by posting, so a posting's values are adjacent in
//!   the value log (what a split or rebuild leaves).
//! - `scattered`: written document by document, so a posting's values are spread
//!   over the log (what ordinary inserts leave).
//!
//! Embeddings are seeded pseudo-random vectors, so scores are meaningless; the
//! store's shape (entries per posting, bytes per entry) is what matters here.
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use minnal_db::semantic_search::{
    Cluster, ClusterIndex, ProbeSettings, QuantisationStyle, VectorIndex, composite_key,
    index::vector_index::VectorKvStore,
    index_embedding_rotated,
    service::{SemanticSearchConfig, search},
};
use minnal_db::vector_kv::{DbVectorStore, dense_vectors_ns, sparse_vectors_ns};
use minnal_db::{AsyncDb, DbConfig};
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Duration;

const DIM: usize = 768;
const N_DOCS: usize = 100_000;
const N_POSTINGS: usize = 1_024;
/// Probed postings: 18%, 73% and 100% of them, the shares 256, 1,024 and every
/// posting probe on a 1,397-posting store.
const PROBES: [usize; 3] = [184, 750, N_POSTINGS];
const NS: &str = "bench_pass1";

/// Seeded xorshift; values in (-1, 1).
fn prng(mut s: u64) -> impl FnMut() -> f32 {
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as i64 as f32) / (i64::MAX as f32)
    }
}

fn unit(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= n);
}

struct Corpus {
    index: ClusterIndex,
    /// Per document: (posting, sparse value, dense value).
    docs: Vec<(u32, Vec<u8>, Vec<u8>)>,
    query: Vec<f32>,
}

/// Chunk code width, from `MINNAL_BENCH_PASS1_BITS`.
fn pass1_bits() -> usize {
    std::env::var("MINNAL_BENCH_PASS1_BITS").map_or(1, |b| b.parse().expect("bench: MINNAL_BENCH_PASS1_BITS"))
}

fn corpus() -> Corpus {
    let mut next = prng(0x6d69_6e6e_616c_0001);
    let mut vector = || {
        let mut v: Vec<f32> = (0..DIM).map(|_| next()).collect();
        unit(&mut v);
        v
    };
    let centres: HashMap<u32, Cluster> = (0..N_POSTINGS as u32).map(|id| (id, Cluster::new(id, vector()))).collect();
    let index = ClusterIndex::from_clusters(centres);
    let docs = (0..N_DOCS)
        .map(|_| {
            let emb = vector();
            let sparse = index_embedding_rotated(&index, &emb, QuantisationStyle::pass1(pass1_bits())).expect("bench: route");
            let posting = sparse.cluster_id;
            let dense = index_embedding_rotated(&index, &emb, QuantisationStyle::MultiBit { number_of_bits: 8 }).expect("bench: route");
            (posting, VectorIndex::list_to_bytes(&[sparse]), VectorIndex::list_to_bytes(&[dense]))
        })
        .collect();
    Corpus {
        index,
        docs,
        query: vector(),
    }
}

/// Memtable size in entries: small, so the corpus seals into read-only memtables
/// as it is written and `compact` can move all of it into L1.
const MEMTABLE_ENTRIES: usize = 8_192;

/// Eight buckets, as the measured `musics_200k` store has.
fn bench_config() -> DbConfig {
    DbConfig {
        num_buckets: 8,
        skip_list_capacity: MEMTABLE_ENTRIES,
        ..DbConfig::default()
    }
}

/// Write the corpus into a fresh database in the given order, so that all of it
/// ends up in L1 with no L0 file. A memtable's worth of filler under a posting no
/// search probes (`u32::MAX`) pushes the last corpus entries out of the active
/// memtable; `compact` then merges every sealed memtable into L1. (A shutdown and
/// reopen would do it too, but the closing instance's final pass can still be
/// removing files while the new one opens them.)
async fn build_store(dir: &std::path::Path, corpus: &Corpus, grouped: bool) -> AsyncDb {
    let db = AsyncDb::open_with_config(dir.to_owned(), bench_config()).await.expect("bench: open");
    db.namespace(NS.to_string()).await.expect("bench: parent namespace");
    let sparse = db.namespace(sparse_vectors_ns(NS)).await.expect("bench: sparse namespace");
    let dense = db.namespace(dense_vectors_ns(NS)).await.expect("bench: dense namespace");
    let mut order: Vec<usize> = (0..corpus.docs.len()).collect();
    if grouped {
        order.sort_by_key(|&d| corpus.docs[d].0);
    }
    for d in order {
        let (posting, sparse_value, dense_value) = &corpus.docs[d];
        let doc_id = (d as u64).to_be_bytes();
        sparse
            .put_no_wal(composite_key::encode(*posting, &doc_id), sparse_value.clone())
            .await
            .expect("bench: sparse put");
        dense.put_no_wal(doc_id.to_vec(), dense_value.clone()).await.expect("bench: dense put");
    }
    for i in 0..MEMTABLE_ENTRIES as u64 {
        let key = composite_key::encode(u32::MAX, &i.to_be_bytes());
        sparse.put_no_wal(key, Vec::new()).await.expect("bench: filler put");
        dense
            .put_no_wal([&[0xff; 8][..], &i.to_be_bytes()].concat(), Vec::new())
            .await
            .expect("bench: filler put");
    }
    db.compact().await.expect("bench: compact");
    db
}

fn bench_pass1(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("bench: runtime");
    let corpus = corpus();
    let probe_sets: Vec<Vec<u32>> = PROBES
        .iter()
        .map(|&n| {
            let mut ids = corpus
                .index
                .find_top_n_cluster_ids_batch(std::slice::from_ref(&corpus.query), n)
                .remove(0);
            ids.sort_unstable();
            ids
        })
        .collect();
    let no_filter: Option<fn(&[u8]) -> bool> = None;
    let bits = pass1_bits();
    let suffix = if bits == 1 { String::new() } else { format!("@{bits}b") };

    for layout in ["grouped", "scattered"] {
        let dir = tempfile::TempDir::new().expect("bench: tempdir");
        let db = rt.block_on(build_store(dir.path(), &corpus, layout == "grouped"));
        let store = rt.block_on(DbVectorStore::new(&db, NS)).expect("bench: store");
        let stored = rt.block_on(store.scan_sparse_clusters_batch(&probe_sets[2])).expect("bench: scan").len();
        assert_eq!(stored, N_DOCS, "bench store must hold every document");

        let mut group = c.benchmark_group(format!("pass1_scan/{layout}{suffix}"));
        group.measurement_time(Duration::from_secs(10));
        for (probes, ids) in PROBES.iter().zip(&probe_sets) {
            let entries = rt.block_on(store.scan_sparse_clusters_batch(ids)).expect("bench: scan").len();
            group.throughput(Throughput::Elements(entries as u64));
            group.bench_with_input(BenchmarkId::from_parameter(probes), ids, |b, ids| {
                b.iter(|| black_box(rt.block_on(store.scan_sparse_clusters_batch(black_box(ids))).expect("bench: scan")))
            });
        }
        group.finish();

        let mut group = c.benchmark_group(format!("pass1_search/{layout}{suffix}"));
        group.measurement_time(Duration::from_secs(10));
        for &probes in &PROBES {
            let config = SemanticSearchConfig {
                probe: ProbeSettings::fixed(probes),
                first_pass_sparse_search_top_k: 1_000,
                top_k_results: 10,
                pass1_bits: bits,
                ..SemanticSearchConfig::default()
            };
            group.bench_with_input(BenchmarkId::from_parameter(probes), &config, |b, config| {
                b.iter(|| {
                    black_box(
                        rt.block_on(search(
                            black_box(config),
                            NS,
                            &corpus.index,
                            std::slice::from_ref(&corpus.query),
                            &corpus.query,
                            &store,
                            no_filter,
                            None,
                        ))
                        .expect("bench: search"),
                    )
                })
            });
        }
        group.finish();
        rt.block_on(db.shutdown()).expect("bench: shutdown");
    }
}

criterion_group!(benches, bench_pass1);
criterion_main!(benches);
