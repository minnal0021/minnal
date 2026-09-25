# Rank fusion for two-pass semantic search — BEIR evaluation

*Run 2026-09-25 on branch `rrf`. Embedding model `google/embeddinggemma-300m` (served
locally), bundled `gemma` centroids (256 × 768). Harness:
[`beir_eval.rs`](beir_eval.rs).*

## TL;DR

- **The hypothesis doesn't hold on these benchmarks.** The dense-only final order is
  not measurably wrong. The ColBERT MaxSim signal from Pass 1 is much *weaker* than the
  dense score (−0.148 nDCG@10 on SciFact test, −0.045 on NFCorpus test), and fusing it
  in never beat dense by a statistically significant margin on both datasets.
- **Best hyperparameters** (tuned on SciFact `train` + NFCorpus `dev`, confirmed on
  `test`): **`mode = "rrf"`, `rrf_k = 1`, `sparse_weight = 0.1`**. That gains +0.0007
  nDCG@10 on the tuning splits and −0.0001 / +0.0001 on the two test splits, which is a
  tie with `dense`. With a weight that small, MaxSim only breaks near-ties in the dense
  order.
- **Shipped defaults:** `mode` stays **`dense`**. The fusion parameter defaults change
  from the textbook `rrf_k = 60`, `sparse_weight = 0.5` to the tuned **`1` / `0.1`**,
  because the textbook values measured **2–4.5 nDCG@10 points worse** than dense
  (p < 0.001).
- **Your original idea**, an RRF re-order of only the dense top-k (`rrf-dense100`),
  is significantly worse: −0.040 (SciFact) and −0.019 (NFCorpus) at `k = 60, w = 0.5`.
- **Follow-up experiment (§6): querying Pass 1 with the whole query as one sparse
  chunk** (instead of 4-word fragments) confirms the root cause. MaxSim alone
  improves by +0.11 nDCG@10 on SciFact test (0.640 → 0.752), shrinking its gap to
  dense from −0.148 to −0.036. Fusion still only ties dense on SciFact, but on
  NFCorpus with all clusters probed the whole-query setting tuned on the training
  split, `rrf k=10 w=0.2`, gives a small, significant gain: +0.004, p = 0.002.
- **The bigger lever is the candidate set, not the ranking.** On NFCorpus, Pass 1 at
  `n_probes = 32` keeps only 60% of the relevant documents. Probing all 256 clusters
  lifts dense nDCG@10 from 0.371 to 0.385 (+0.014), about 10× the best fusion gain.

## 1. Question

Before this change, `search()` used the Pass-1 ColBERT MaxSim score only to pick
~1000 candidates and then ranked them purely by the Pass-2 dense score. The
hypothesis was that this discards useful passage-level evidence, and that reciprocal
rank fusion (RRF) of the two rankings would order results better.

The branch adds configurable fusion (`[semantic_search.ranking]`, overridable per
request), with modes `dense`, `sparse`, `rrf`, and `zscore`. Results report both
scores and both ranks. This report measures whether any fusion setting improves
relevance.

## 2. Method

**Pipeline under test.** These are the production defaults:

| Setting | Value |
|---|---|
| Chunking | window 4, slide 2 (4 **sentences** per document chunk, 4 **words** per query chunk) |
| Dense quantisation | 8 bits |
| Pass-1 candidates | `first_pass_sparse_search_top_k = 1000` |
| `n_probes` | 32 (production) and 256 (exhaustive), both reported |

**Datasets** ([BEIR](https://github.com/beir-cellar/beir)):

| Dataset | Docs | Doc length (words, median) | Tuning split | Test split | Query length (words, median) | Queries ≤ 4 words |
|---|---:|---:|---|---|---:|---:|
| SciFact | 5,183 | 204 | `train`, 809 queries | `test`, 300 queries | 12 | 1% |
| NFCorpus | 3,633 | 237 | `dev`, 324 queries | `test`, 323 queries | 2 | 70% |

**Protocol.**
- Hyperparameters are chosen on the tuning splits only and then confirmed on `test`.
  Picking the best of 86 grid cells on `test` directly would report the selection
  noise as a gain.
- Selection rule: the highest mean ΔnDCG@10 vs `dense` across both tuning sets, at
  the production `n_probes = 32`.
- Each query runs one live `search()` that returns all ~1000 candidates with both
  scores. Every variant is then computed offline through the production `fuse()`.
  The harness asserts, for 20 queries per run, that a live `rrf` search returns the
  same top 10 as the offline fusion.

**Grid (86 variants):**
- `dense` and `sparse`
- `rrf` over the full candidate set, `k ∈ {1, 5, 10, 20, 30, 60, 100, 200}` ×
  `w ∈ {0.1 … 0.9}`
- `zscore` with `w ∈ {0.1 … 0.9}`
- `rrf-dense100` (RRF restricted to the dense top-100) with `w ∈ {0.3, 0.5, 0.7}`,
  `k = 60`

**Metrics:**
- nDCG@10 with graded gains (the primary metric), MRR@10, and Recall@100.
- W/L: the number of queries where a variant's nDCG@10 is higher or lower than
  `dense`.
- p: a two-sided paired randomization test of per-query nDCG@10 vs `dense`
  (10,000 sign flips).

## 3. Results

### 3.1 Test split (held out)

ΔnDCG@10 is measured against `dense`.

**SciFact test** (300 queries):

| Variant | nDCG@10 (np 32) | Δ | MRR@10 | R@100 | W / L | p | nDCG@10 (np 256) | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dense` | **0.7881** | — | 0.7565 | 0.9727 | — | — | **0.7914** | — |
| **`rrf k=1 w=0.1`** (selected) | 0.7880 | −0.0001 | 0.7566 | 0.9727 | 1 / 3 | 0.62 | 0.7914 | −0.0001 |
| `rrf k=5 w=0.1` | 0.7881 | +0.0000 | 0.7566 | 0.9727 | 3 / 2 | 0.88 | 0.7916 | +0.0002 |
| `zscore w=0.1` | 0.7895 | +0.0015 | 0.7579 | 0.9727 | 19 / 18 | 0.70 | 0.7933 | +0.0019 |
| `zscore w=0.5` | 0.7639 | −0.0241 | 0.7385 | 0.9620 | 31 / 50 | 0.018 | 0.7669 | −0.0245 |
| `rrf k=60 w=0.5` (textbook) | 0.7437 | −0.0444 | 0.7151 | 0.9660 | 32 / 58 | <0.001 | 0.7463 | −0.0451 |
| `rrf-dense100 k=60 w=0.5` | 0.7478 | −0.0403 | 0.7187 | 0.9727 | 31 / 56 | <0.001 | 0.7491 | −0.0423 |
| `sparse` (MaxSim only) | 0.6398 | −0.1483 | 0.6032 | 0.9257 | 18 / 109 | <0.001 | 0.6435 | −0.1479 |

**NFCorpus test** (323 queries):

| Variant | nDCG@10 (np 32) | Δ | MRR@10 | R@100 | W / L | p | nDCG@10 (np 256) | Δ | p (np 256) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dense` | **0.3712** | — | 0.5778 | 0.3356 | — | — | **0.3851** | — | — |
| **`rrf k=1 w=0.1`** (selected) | 0.3713 | +0.0001 | 0.5774 | 0.3343 | 23 / 24 | 0.94 | 0.3863 | +0.0011 | 0.18 |
| `rrf k=5 w=0.1` | 0.3712 | +0.0000 | 0.5779 | 0.3343 | 22 / 18 | 1.00 | 0.3863 | +0.0012 | 0.12 |
| `zscore w=0.1` | 0.3726 | +0.0014 | 0.5774 | 0.3333 | 66 / 43 | 0.31 | 0.3891 | +0.0039 | 0.008 |
| `zscore w=0.5` | 0.3563 | −0.0149 | 0.5571 | 0.3291 | 75 / 100 | <0.001 | 0.3742 | −0.0110 | 0.012 |
| `rrf k=60 w=0.5` (textbook) | 0.3512 | −0.0200 | 0.5483 | 0.3299 | 69 / 104 | <0.001 | 0.3720 | −0.0132 | 0.004 |
| `rrf-dense100 k=60 w=0.5` | 0.3519 | −0.0193 | 0.5484 | 0.3356 | 67 / 104 | <0.001 | 0.3727 | −0.0124 | 0.006 |
| `sparse` (MaxSim only) | 0.3262 | −0.0450 | 0.5196 | 0.3100 | 65 / 137 | <0.001 | 0.3465 | −0.0387 | <0.001 |

For reference, the single best cell chosen with hindsight on `test` (an optimistic
upper bound, not a fair estimate): SciFact `rrf k=20 w=0.1` = 0.7904 (+0.0023), and
NFCorpus `zscore w=0.1` = 0.3726 / 0.3891. Even with the benefit of hindsight, fusion
is worth at most about +0.002 to +0.004 nDCG@10.

### 3.2 Hyperparameter selection (tuning splits, `n_probes = 32`)

ΔnDCG@10 vs `dense` over the RRF grid. The only non-negative region is `w = 0.1` with
small `k`. Everything else loses, and the loss grows steadily with the sparse weight.

**SciFact train** (dense = 0.8733):

| k \ w | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **+0.000** | −0.002 | −0.009 | −0.020 | −0.043 | −0.082 | −0.114 | −0.137 | −0.163 |
| 5 | +0.000 | −0.007 | −0.018 | −0.040 | −0.057 | −0.086 | −0.117 | −0.146 | −0.170 |
| 10 | −0.001 | −0.012 | −0.031 | −0.049 | −0.065 | −0.095 | −0.119 | −0.145 | −0.168 |
| 20 | −0.009 | −0.024 | −0.040 | −0.059 | −0.073 | −0.098 | −0.121 | −0.141 | −0.162 |
| 30 | −0.013 | −0.028 | −0.046 | −0.065 | −0.079 | −0.100 | −0.120 | −0.139 | −0.160 |
| 60 | −0.020 | −0.037 | −0.053 | −0.073 | −0.080 | −0.100 | −0.117 | −0.137 | −0.156 |
| 100 | −0.023 | −0.041 | −0.057 | −0.075 | −0.081 | −0.099 | −0.116 | −0.136 | −0.154 |
| 200 | −0.027 | −0.044 | −0.060 | −0.075 | −0.082 | −0.100 | −0.117 | −0.135 | −0.153 |
| zscore | −0.007 | −0.017 | −0.028 | −0.040 | −0.056 | −0.074 | −0.101 | −0.126 | −0.151 |

**NFCorpus dev** (dense = 0.3403):

| k \ w | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **+0.001** | +0.000 | −0.000 | −0.003 | −0.005 | −0.012 | −0.017 | −0.023 | −0.027 |
| 5 | +0.001 | +0.000 | +0.000 | −0.002 | −0.004 | −0.012 | −0.016 | −0.022 | −0.027 |
| 10 | +0.000 | +0.001 | −0.002 | −0.002 | −0.004 | −0.011 | −0.016 | −0.021 | −0.028 |
| 20 | −0.000 | −0.000 | −0.003 | −0.002 | −0.005 | −0.009 | −0.013 | −0.020 | −0.026 |
| 30 | +0.000 | −0.000 | −0.003 | −0.002 | −0.006 | −0.009 | −0.013 | −0.019 | −0.025 |
| 60 | +0.000 | −0.001 | −0.004 | −0.004 | −0.005 | −0.010 | −0.013 | −0.018 | −0.023 |
| 100 | −0.000 | −0.003 | −0.004 | −0.004 | −0.004 | −0.010 | −0.013 | −0.018 | −0.022 |
| 200 | −0.002 | −0.002 | −0.005 | −0.005 | −0.005 | −0.010 | −0.012 | −0.017 | −0.021 |
| zscore | +0.003 | +0.001 | −0.002 | −0.002 | −0.005 | −0.008 | −0.011 | −0.016 | −0.023 |

**Joint ranking** (mean Δ across both tuning sets, `n_probes = 32`):

| Rank | Variant | Mean Δ | Δ SciFact train (p) | Δ NFCorpus dev (p) |
|---:|---|---:|---:|---:|
| 1 | **`rrf k=1 w=0.1`** | **+0.0007** | +0.0004 (0.62) | +0.0009 (0.35) |
| 2 | `rrf k=5 w=0.1` | +0.0005 | +0.0001 (0.90) | +0.0010 (0.14) |
| 3 | `dense` | 0 | — | — |
| 4 | `rrf k=10 w=0.1` | −0.0004 | −0.0013 (0.32) | +0.0005 (0.50) |
| 6 | `zscore w=0.1` | −0.0018 | **−0.0066 (0.004)** | +0.0030 (0.046) |
| — | `rrf k=60 w=0.5` | −0.0427 | −0.0804 (<0.001) | −0.0049 |

`zscore w=0.1` is the one setting that changes direction between corpora: it is
significantly worse on SciFact `train` (p = 0.004) and slightly better on NFCorpus
(significant only at `n_probes = 256` on `test`, p = 0.008). A corpus-dependent
effect of about ±0.005 isn't a safe default, so it wasn't selected.

### 3.3 The candidate set (Pass 1) matters more than the final order

Candidate-set recall is the share of relevant documents that reach the fused
candidate set. It is the ceiling for any re-ranking.

| Dataset (test) | `n_probes` | Candidate-set recall | Dense nDCG@10 | Best fusion Δ |
|---|---:|---:|---:|---:|
| SciFact | 32 | 0.993 | 0.7881 | +0.0015 |
| SciFact | 256 | 0.997 | 0.7914 (+0.003) | +0.0019 |
| NFCorpus | 32 | 0.595 | 0.3712 | +0.0014 |
| NFCorpus | 256 | 0.653 | 0.3851 (**+0.014**) | +0.0039 |

On SciFact, Pass 1 already recovers nearly every relevant document, so ordering is
all that's left, and dense orders it best. On NFCorpus, 40% of relevant documents
never reach Pass 2 at `n_probes = 32`. Probing more clusters buys roughly 10× what
any fusion does.

## 4. Why fusion doesn't help here

1. **The sparse signal is much weaker, not complementary.** `sparse` alone loses to
   `dense` on 109 of 300 SciFact queries and wins on only 18. Fusion helps when two
   rankers are comparably accurate and make different mistakes. Here one ranker is
   far worse, so any real weight on it adds noise. That explains the smooth decline
   with `w` in every grid row.
2. **Query and document chunks don't match in size (the root cause, confirmed in §6).**
   Documents are chunked into 4-*sentence* windows and queries into 4-*word*
   windows. A 12-word SciFact claim becomes about five 4-word fragments (for example
   "0-dimensional biomaterials show …"), and each is embedded on its own, without
   context. MaxSim then sums how well each fragment matches a 4-sentence passage.
   This is not ColBERT's contextual token-level late interaction: fragment
   embeddings carry little of the claim's meaning. The pattern in the data fits:
   - On NFCorpus, 70% of queries are ≤ 4 words, so the "sparse" query is a single
     chunk equal to the whole query. The gap to dense shrinks from −0.148 to
     −0.045, and low-weight fusion there is slightly positive.
   - On SciFact, with multi-fragment queries, every fusion setting except the tiny
     `w = 0.1` tie-breaker loses significantly.
3. **1-bit quantisation of the chunks adds estimation noise** on top of (2). The
   dense side is 8-bit. These runs can't separate the two effects (see §6).
4. **The documents are short.** The case for passage-level matching is that a
   single embedding dilutes one strong passage inside a long document. SciFact and
   NFCorpus are abstracts (a median of about 200–240 words, about 4–5 chunks per
   document), which fit comfortably in one embedding, so there is little dilution to
   recover.
5. **Probe truncation is not the cause.** MaxSim alone barely improves when all
   clusters are probed (SciFact test 0.640 → 0.644), so unprobed chunks counting as
   0 isn't what makes it weak.

## 5. Recommendation

| Decision | Setting | Why |
|---|---|---|
| Default ranking | `mode = "dense"` (unchanged) | No fusion setting beat it significantly on both corpora. |
| Default fusion params | `rrf_k = 1`, `sparse_weight = 0.1` (**changed** from 60 / 0.5) | They are the tuned optimum and roughly neutral. The old defaults cost 2–4.5 points for anyone who sent `{"mode":"rrf"}`. |
| Avoid | `rrf k≥20` or any `w ≥ 0.3`, and `rrf-dense100` | Significantly worse on both corpora. |
| Worth pursuing | a latency/recall sweep of `n_probes` (e.g. 48 / 64 / 128) | On short-query corpora it is the largest measured lever: +0.014 nDCG@10 at 256. |

The fusion machinery is still useful. The per-request `ranking` override and the
`dense_rank` / `sparse_rank` fields make it cheap to re-test this if the sparse
signal improves.

## 6. Follow-up: whole-query sparse chunk

§4.2 blamed the query fragments. To test that, `beir_eval.rs` gained an eval-only
switch, `MINNAL_BEIR_QUERY_CHUNKING`, which controls the Pass-1 query chunks:

- **`window`**: production behaviour, 4-word sliding windows.
- **`whole`**: a single sparse chunk equal to the whole-query embedding. This is
  the same `/embedding/query` vector of the full query text that Pass 2 already
  uses, so no extra embedding call is needed.
- **`both`**: the whole-query chunk plus the fragments.

Documents are still chunked as before, so the same indexes were reused. The cluster
probes are derived from the query chunks, so the candidate set changes too, not
just the MaxSim score.

### 6.1 MaxSim alone

The fragments were the main cause. nDCG@10 at `n_probes = 32`:

| Split | Dense | `sparse` with `window` (production) | `sparse` with `whole` | `sparse` with `both` |
|---|---:|---:|---:|---:|
| SciFact train | 0.873 | 0.696 (−0.178) | **0.819 (−0.054)** | 0.756 (−0.121) |
| SciFact test | 0.788 | 0.640 (−0.148) | **0.752 (−0.036)** | 0.699 (−0.089) |
| NFCorpus dev | 0.340 | 0.310 (−0.031) | **0.329 (−0.012)** | 0.320 (−0.021) |
| NFCorpus test | 0.371 | 0.326 (−0.045) | **0.344 (−0.027)** | 0.335 (−0.036) |

- **The whole query closes 70–75% of the MaxSim gap to dense on SciFact**, where
  queries are 12 words and so were split into several fragments.
- On NFCorpus the gap shrinks by about 40–60%, even though 70% of its queries were
  already a single chunk. The remaining 30% of queries are longer, and they benefit.
- Adding the fragments back (`both`) is clearly worse than `whole`. Summing
  per-fragment maxima actively harms MaxSim; the fragments aren't just redundant.
- What remains of the gap (−0.036 on SciFact test) is 1-bit quantisation plus
  passage-level vs. whole-document matching, with the whole-document embedding
  ahead on these short abstracts.

Candidate-set recall at `n_probes = 32` is unchanged or marginally higher with
`whole`: SciFact test 0.993 → 0.993, NFCorpus test 0.595 → 0.599. At 256 probes it
is SciFact test 0.997 → 1.000 and NFCorpus test 0.653 → 0.661. This holds even
though `whole` probes only the `n_probes` clusters of one vector instead of the
union across 4–6 fragments. The harness timings were taken with other runs on the
same machine and aren't reliable enough to claim a latency change (per-query Pass
1 time was about 5–9 ms in every mode).

### 6.2 Fusion with the whole-query chunk

Selection followed the same protocol: tune on SciFact train + NFCorpus dev at
`n_probes = 32`, then confirm on test. ΔnDCG@10 is against `dense`.

| Query chunking | Tuned setting | Tuning mean Δ | SciFact test Δ (np 32 / 256) | NFCorpus test Δ (np 32 / 256) |
|---|---|---:|---:|---:|
| `window` (production) | `rrf k=1 w=0.1` | +0.0007 | −0.0001 / −0.0001 | +0.0001 / +0.0011 (p 0.18) |
| **`whole`** | **`rrf k=10 w=0.2`** | +0.0010 | −0.0003 / −0.0003 | +0.0010 (p 0.44) / **+0.0042 (p 0.002)** |
| `both` | `rrf k=5 w=0.1` | +0.0003 | −0.0001 / −0.0001 | +0.0001 / +0.0012 (p 0.08) |

- **Fusion now tolerates a real sparse weight.** Under `whole`, the tuned setting
  gives MaxSim twice the weight (`w = 0.2`, `k = 10`), and even the textbook
  `rrf k=60 w=0.5` drops to only −0.011 / −0.009 on test, down from −0.044 / −0.020.
  On NFCorpus dev it is actually positive (+0.003).
- **The gain is still small and corpus-dependent.** On SciFact it is a tie
  (11 wins, 13 losses). On NFCorpus it is significant only with all clusters probed
  (+0.004, 69 wins / 41 losses, p = 0.002). It isn't significant at the production
  `n_probes = 32` (+0.001, p = 0.44).
- **The best-possible cell per test split** (an optimistic bound) is about +0.004
  in every chunking mode, so a better sparse signal hasn't raised the ceiling on
  these short-document corpora. The dense embedding already captures nearly
  everything MaxSim knows about 200-word abstracts.

### 6.3 What this means

| Question | Answer |
|---|---|
| Is query fragmentation the root cause of the weak MaxSim? | **Yes.** The whole-query chunk recovers most of the gap (§6.1). |
| Should production Pass 1 use the whole query? | **Probably, yes.** It is a much better sparse signal with the same or better candidate recall and fewer probed clusters. But the payoff is invisible while `mode = "dense"`, because MaxSim then only selects candidates and recall barely changes. It becomes worthwhile together with fusion (below) or for long-document corpora. Not changed in this branch. |
| Should the default become fusion? | **Not on this evidence.** The best result is +0.004 on one corpus at 256 probes, and a tie elsewhere. `dense` stays the default. |
| If fusion is enabled with whole-query chunking, which parameters? | `rrf`, `rrf_k = 10`, `sparse_weight = 0.2`. The shipped defaults (`1` / `0.1`) were tuned for production `window` chunking. Under `whole` they tie (SciFact −0.0001, NFCorpus +0.0011 at 256 probes) and are roughly neutral. |

## 7. Next steps (to make the ColBERT signal worth fusing)

1. **Adopt whole-query sparse chunking in production** (§6), with fusion
   re-tuned as in §6.3. It is a query-side change in `embed_query` / `search`,
   needs no re-index, and would make the query-embedding cache smaller (no chunk
   vectors to store).
2. **Store multi-bit (e.g. 4-bit) sparse chunks** for one corpus to size the
   remaining quantisation cost (§4.3). With fragmentation fixed (§6.1), 1-bit
   quantisation is the main suspect for the −0.036 SciFact gap that remains.
3. **Evaluate on corpora with long documents** (e.g. `trec-covid`, `fiqa`,
   `robust04`), where passage-level matching has the most to offer.
4. **Sweep `n_probes`** for latency vs. candidate recall (§3.3).

## 8. Issues found during the evaluation

- **The harness indexed nothing on the first attempt, and `real_recall_vs_nprobes`
  had the same bug.** `upsert_vectors` deliberately skips a namespace that isn't
  registered (so a dropped store isn't silently recreated). The eval harnesses never
  registered their parent namespace, so every upsert was an `Ok` no-op: 8,816
  documents "indexed" in about 40 minutes and stored nothing. Fixed in the shared
  helper (`vector_kv::eval_indexing::index_texts` now registers the namespace
  first), and `beir_rank_fusion_eval` now fails loudly if every query returns zero
  candidates. `real_recall_vs_nprobes` uses the same helper, so it is fixed too.
- **One SciFact document (1 of 5,183) failed to embed** during the final index
  build. It affects every variant equally, and the index was marked complete by
  hand so the `test` run could reuse it without spending another 42 minutes
  re-embedding.
- Indexing throughput was about 3.5 documents per second, limited by the embedding
  service. SciFact took 42 minutes and NFCorpus 35 minutes. Evaluating a split
  against a built index takes about 30 seconds.

## 9. Reproduce

```sh
service/scripts/fetch_beir.sh scifact nfcorpus
# Tuning splits (the first run per dataset builds and keeps the index):
MINNAL_EMBED_URL=http://localhost:8001 MINNAL_BEIR_DATASET=scifact  MINNAL_BEIR_SPLIT=train \
  MINNAL_BEIR_DB=$PWD/work/beir/db_scifact \
  cargo test -p minnal_db --all-features --release --lib beir_rank_fusion_eval -- --ignored --nocapture
MINNAL_EMBED_URL=http://localhost:8001 MINNAL_BEIR_DATASET=nfcorpus MINNAL_BEIR_SPLIT=dev \
  MINNAL_BEIR_DB=$PWD/work/beir/db_nfcorpus \
  cargo test -p minnal_db --all-features --release --lib beir_rank_fusion_eval -- --ignored --nocapture
# Confirmation: rerun both with MINNAL_BEIR_SPLIT=test (reuses the index).
# §6 query-chunking experiment: add MINNAL_BEIR_QUERY_CHUNKING=whole (or both);
# results go to fusion_eval_gemma_<split>_q<mode>.{md,tsv}.
```

Full per-variant tables (all 86 variants × 2 `n_probes`) are written to
`work/beir/<dataset>/fusion_eval_gemma_<split>.{md,tsv}` (gitignored).
