# Pass-1 query embedding: whole query vs chunks — BEIR evaluation

*Run 2026-09-25. Embedding model `google/embeddinggemma-300m` through the PyTorch
embedding service (full pipeline, bfloat16); gemma centroids regenerated for that
service (ELI5, k = 256). Harness: [`beir_eval.rs`](beir_eval.rs).*

## Decision

**Queries are no longer chunked.** One embedding of the whole query now serves
both passes. It is Pass 2's dense vector and Pass 1's single ColBERT MaxSim
vector, so the Pass-1 score is `max_j ⟨q, d_j⟩` over document chunks. The
`query_chunking` option that was prototyped for the comparison was removed.
`window_size` / `sliding_size` now apply to documents only.

- **Never worse than 4-word query chunks, often better.** With a tight
  first-pass cut, whole-query probing kept far more relevant documents
  (ArguAna candidate recall 0.991 vs 0.870; nDCG@10 +0.038, p = 0.0001).
- **Much cheaper:**
  - one query embedding instead of 1 + N (ArguAna: 1 vs 96);
  - exactly `n_probes` clusters probed instead of the union over every
    fragment (ArguAna: 32 vs 171);
  - search up to 74% faster.
- **Sentence-window query chunks gained nothing either,** even for 174-word
  queries. Quality was identical to the whole query, at 3.8× the vectors and
  more latency. So a length-based rule (`auto`) was not kept.

## Method

For each dataset, index the corpus once, then run every judged test query
through `search()` under each Pass-1 query mode:

| Mode | Pass-1 query vectors |
|---|---|
| `words` (previous behaviour) | 4-word sliding windows (window 4, slide 2), each embedded separately |
| `whole` (adopted) | the whole-query embedding (reused from Pass 2, no extra embedding) |
| `sentences` | document-style 4-sentence windows; whole-query vector if the query fits in one window |
| `auto` | `whole` up to 32 words, `sentences` above |

Final ranking is by the Pass-2 dense score, so Pass-1 query vectors decide
which candidates reach Pass 2 and how much Pass 1 costs. Two measurements:

- **End to end:** nDCG@10 (primary), MRR@10, Recall@100.
- **Pass 1:** candidate recall (share of relevant docs among the first-pass
  candidates), at the production `first_pass_sparse_search_top_k` = 1000 and at
  100. The tight cut exposes MaxSim's own ranking quality.

Also recorded: clusters probed, query vectors, embedding ms/query, and median
search ms, at `n_probes` 32 (production) and 256 (exhaustive). Significance is a
two-sided paired randomization test of per-query nDCG@10 vs `words`.

| Dataset | Docs | Test queries | Query words (median / p90) | Queries > 32 words |
|---|---:|---:|---:|---:|
| SciFact | 5,183 | 300 | 12 / 21 | 0% |
| NFCorpus | 3,633 | 323 | 2 / 7 | 0% |
| ArguAna | 8,674 | 1,406 | 174 / 299 | 100% |

On SciFact and NFCorpus, `auto` behaves as `whole`, and `sentences` also
behaves as `whole` (every query fits in one window), so those three rows are
identical there. ArguAna is the long-query test.

## Results (`n_probes` = 32)

### SciFact

| Mode | First pass | nDCG@10 | Cand. recall | p vs words | Clusters probed | Query vectors | Search ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| words | 1000 | 0.7890 | 0.9933 | — | 70.9 | 5.5 | 5.95 |
| **whole** | 1000 | **0.7897** | **0.9933** | 0.81 | **32** | **1** | **4.38** |
| words | 100 | 0.7814 | 0.9410 | — | 70.9 | 5.5 | 5.27 |
| **whole** | 100 | **0.7897** | **0.9583** | 0.10 | **32** | **1** | **3.72** |

### NFCorpus

Most queries are ≤ 4 words, so `words` mode already produced a single chunk for
most of them.

| Mode | First pass | nDCG@10 | Cand. recall | p vs words | Clusters probed | Query vectors | Search ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| words | 1000 | 0.3745 | 0.5866 | — | 37.2 | 1.45 | 3.11 |
| **whole** | 1000 | 0.3743 | 0.5866 | 0.62 | **32** | **1** | **2.97** |
| words | 100 | 0.3699 | 0.3154 | — | 37.2 | 1.45 | 2.44 |
| **whole** | 100 | **0.3702** | **0.3187** | 0.85 | **32** | **1** | **2.32** |

### ArguAna (long queries)

| Mode | First pass | nDCG@10 | Cand. recall | p vs words | W / L vs words | Clusters probed | Query vectors | Embed ms/q | Search ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| words | 1000 | 0.4686 | 0.9858 | — | — | 171.2 | 96.0 | 132.5 | 16.91 |
| **whole** | 1000 | **0.4688** | **0.9936** | 0.90 | 16 / 23 | **32** | **1** | **16.7** | **4.38** |
| sentences = auto | 1000 | 0.4686 | 0.9950 | 1.00 | 10 / 20 | 46.0 | 3.8 | 27.2 | 5.51 |
| words | 100 | 0.4308 | 0.8698 | — | — | 171.2 | 96.0 | 132.5 | 16.02 |
| **whole** | 100 | **0.4688** | **0.9908** | **0.0001** | **135 / 85** | **32** | **1** | **16.7** | **3.79** |
| sentences = auto | 100 | 0.4689 | 0.9858 | 0.0001 | 132 / 84 | 46.0 | 3.8 | 27.2 | 4.72 |

## Reading the results

1. **Fragments were the weak signal.** A 4-word fragment carries little meaning
   on its own. Summing per-fragment maxima let off-topic documents in:
   ArguAna's 96 fragments probed 171 clusters, and with a 100-candidate cut
   13% of relevant documents never reached Pass 2. The whole-query vector
   keeps them (0.991).
2. **At the production cut the difference is mostly cost.** With 1,000
   candidates on corpora of 4–9k documents, even the weak signal usually keeps
   the relevant documents (SciFact 0.993 either way), so end-to-end nDCG ties.
   The whole query does the same job with one vector and one `n_probes`
   probe set. The quality gap appears when candidates are scarce relative to
   the corpus, i.e. larger corpora or a smaller `first_pass_sparse_search_top_k`.
3. **Sentence windows don't help long queries.** For 174-word arguments, one
   whole-query embedding matched 4-sentence windows on quality (0.4688 vs
   0.4686) and beat them on candidate recall at the tight cut, at a fraction of
   the cost. There is no case for length-based switching.
4. **`n_probes` is still the bigger lever on recall-limited corpora.**
   NFCorpus: 32 → 256 probes lifts nDCG@10 from 0.374 to 0.391 and candidate
   recall from 0.587 to 0.664, far more than any query-side choice.

## What changed in the code

- `embed_query` sends one payload (the query) and returns it as both the
  dense and the single sparse vector. `chunk_query`, `split_words` and
  `ChunkBoundary` were removed; `chunking` is document-only.
- The query-embedding cache stores only the whole-query vector. Chunking
  settings no longer affect cached entries, so changing
  `window_size`/`sliding_size` needs a corpus re-index but no cache clear.
- `search()` still accepts several Pass-1 query vectors (general MaxSim); only
  the production caller passes one.

## Reproduce

```sh
service/scripts/fetch_beir.sh scifact nfcorpus arguana
cd minnal_db
MINNAL_EMBED_URL=http://localhost:8001 MINNAL_BEIR_DATASET=scifact MINNAL_BEIR_DB=$PWD/../work/beir/db_scifact \
  cargo test -p minnal_db --all-features --release --lib beir_eval -- --ignored --nocapture
```

The mode comparison used a temporary `query_chunking` switch, since removed;
the committed harness evaluates the production pipeline only.
