# Pass 1 keeps "80%" of the exact candidates: what is lost, and does it matter?

The vector bench reports a **Pass-1 recall** of about 0.80 at the production
settings: of the 1,000 documents an exact Pass 1 would hand to Pass 2, the index
hands over about 800. This report finds where the other ~200 go, why, and what
they cost in final ranking quality. It also checks Pass 2 on its own.

*Measured on frozen embeddings of BEIR SciFact (5,183 docs, 300 queries) and
FiQA (57,600 docs, 648 queries) from two models served by llama.cpp: gemma
(`embeddinggemma-300m`, Q8) and qwen (`Qwen3-Embedding-8B`, Q4_K_M, truncated to
768 dimensions), each with its bundled centroids (ELI5, k = 256). 64 probes and a
cut of 1,000 unless stated. Commit `946814b`. Harness:
[`vector_bench/study.rs`](../vector_bench/study.rs) (`vector_bench_pass1_study`).*

## Answer

- **The missing ~20% is churn at the bottom of the list, not lost results.**
  81–93% of the missed documents rank 501–1000 in the exact list, and almost none
  rank in the top 100. The index keeps 96.7–99.8% of the exact top 100 and
  98.5–99.97% of the exact top 10.
- **Most misses are near-ties.** Their exact score is within the 1-bit
  estimator's own error bound of the 1,000th score, so 1-bit noise decides which
  side of the cut they land on. Probing all 256 clusters does not change this
  (recall 0.81–0.84). On SciFact a further 24–50% of misses come from probing:
  the document's best chunk is in a cluster that was not probed. The estimator
  alone accounts for about 1%.
- **The cut costs nothing.** Re-ranking the top 4,000 instead of the top 1,000
  gives the same nDCG@10 to four decimals in every case.
- **The index loses little or nothing in ranking quality.** The largest gap
  between the index (with an exact rerank) and the exact two-pass ranking is
  0.0023 nDCG@10 (gemma FiQA), and it disappears at 256 probes, so it is a
  probing cost. Qwen loses nothing on either dataset. Pass 2's 8-bit codes move
  nDCG@10 by at most 0.0010.
- **The two-pass design itself loses nothing.** The exact two-pass ranking equals
  exact dense scoring of every document in all four cases. Pass 2 earns its place
  on FiQA (Pass-1 MaxSim alone: 0.4244 vs 0.4760 on gemma, 0.5003 vs 0.5919 on
  qwen). On qwen SciFact, MaxSim alone scores higher (0.7905 vs 0.7817), a single
  300-query result that is worth re-checking on more datasets.
- **No bug found.** The simulated Pass 1 reproduces `search()` exactly: the same
  candidate set and the same final top 10 on every query of every case.
- **Recommendation:** judge Pass-1 quality by the exact top-10/top-100 coverage,
  not by top-1,000 overlap. The 0.80 figure mostly measures how noisy the bottom
  of two long lists is, which has no effect on what a user sees. For quality, the
  lever is `n_probes` (or a better partition, design doc M2–M3), not
  `first_pass_sparse_search_top_k`.

## How it is measured

The study rebuilds every chunk's 1-bit code in memory with the production
quantiser and scores Pass 1 with the production estimator. It then compares the
result with the **exact Pass 1**: MaxSim over the full-precision chunks of every
document, top 1,000 (`vector_bench/exact.rs`).

**Pass-1 recall** is `|index's 1,000 ∩ exact 1,000| / 1,000`. To confirm the
simulation is the real thing, the study also indexes the corpus into a database
and runs `search()`. The candidate sets and final top 10s are identical on every
query of all four model–dataset cases.

Every exact top-1,000 document the index misses is put into the first matching
class:

| Cause | Meaning |
|---|---|
| not probed | none of the document's chunks is in a probed cluster |
| best chunk not probed | some chunk was probed, but not the one that gives its exact score, so it was scored by a weaker chunk |
| near tie | the best chunk was probed, and the document's exact score is within that chunk's `error_bound` of the exact 1,000th score |
| estimator | the best chunk was probed and the margin exceeds the bound: the 1-bit estimate alone pushed it out |

`error_bound` is the paper's per-code error half-width, which holds for ~94% of
estimates (see `rabitq-rotation-audit.md`).

## Where the misses rank

Share of the exact top-k that the index keeps in its 1,000:

| Model | Dataset | Probes | top 10 | top 50 | top 100 | top 500 | top 1000 |
|---|---|---:|---:|---:|---:|---:|---:|
| gemma | SciFact | 16 | 0.9560 | 0.9219 | 0.9029 | 0.8298 | 0.7200 |
| gemma | SciFact | 64 | 0.9853 | 0.9751 | 0.9672 | 0.9226 | 0.7959 |
| gemma | SciFact | 256 (all) | 1.0000 | 1.0000 | 1.0000 | 0.9811 | 0.8351 |
| gemma | FiQA | 16 | 0.9850 | 0.9793 | 0.9751 | 0.9344 | 0.7878 |
| gemma | FiQA | 64 | 0.9966 | 0.9961 | 0.9958 | 0.9634 | 0.8063 |
| gemma | FiQA | 256 (all) | 1.0000 | 1.0000 | 0.9998 | 0.9691 | 0.8094 |
| qwen | SciFact | 16 | 0.9813 | 0.9645 | 0.9528 | 0.8929 | 0.7668 |
| qwen | SciFact | 64 | 0.9967 | 0.9925 | 0.9895 | 0.9537 | 0.8084 |
| qwen | SciFact | 256 (all) | 1.0000 | 1.0000 | 0.9999 | 0.9760 | 0.8225 |
| qwen | FiQA | 16 | 0.9869 | 0.9781 | 0.9720 | 0.9314 | 0.7956 |
| qwen | FiQA | 64 | 0.9997 | 0.9985 | 0.9979 | 0.9740 | 0.8225 |
| qwen | FiQA | 256 (all) | 1.0000 | 1.0000 | 0.9999 | 0.9784 | 0.8252 |

Coverage falls off only at the bottom of the list. With every cluster probed
the top 100 is complete, and top-1,000 recall is still only 0.81–0.84: the
remaining misses are purely the 1-bit estimate reordering near-equal scores.

Missed documents by exact rank and cause, 64 probes:

| Model | Dataset | Misses per query | rank 1–10 | 11–100 | 101–500 | 501–1000 | not probed | best chunk not probed | near tie | estimator |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma | SciFact | 204 | 0.1% | 1.5% | 17.4% | 81.0% | 37.5% | 12.9% | 49.2% | 0.3% |
| gemma | FiQA | 194 | 0.0% | 0.2% | 9.2% | 90.6% | 3.2% | 2.1% | 93.6% | 1.1% |
| qwen | SciFact | 192 | 0.0% | 0.5% | 11.5% | 87.9% | 16.6% | 7.0% | 76.0% | 0.4% |
| qwen | FiQA | 178 | 0.0% | 0.1% | 7.2% | 92.7% | 2.7% | 2.4% | 94.0% | 1.0% |

SciFact loses more of its misses to probing than FiQA does. This fits what is
known about the bundled centroids: they were fitted on general text (ELI5), and
SciFact's scientific abstracts pile into a few of them (31–36% of SciFact's
chunks land in one cluster, against 12–15% for FiQA). With all 256 clusters
probed, every case is at 99% near-ties.

## At what depth they are cut off

A missed document is not far below the cut. In the approximate ranking its
median position is 1,250–1,320 and its 90th percentile 1,830–2,130. Share of the
exact top-k inside the first N approximate results, 64 probes (∞ = every
document in the probed clusters):

| Model | Dataset | Exact top | 250 | 500 | 1000 | 1500 | 2000 | 3000 | 4000 | ∞ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma | SciFact | 10 | 0.9850 | 0.9853 | 0.9853 | 0.9857 | 0.9857 | 0.9857 | 0.9857 | 0.9857 |
| gemma | SciFact | 100 | 0.9440 | 0.9641 | 0.9672 | 0.9682 | 0.9686 | 0.9689 | 0.9690 | 0.9690 |
| gemma | SciFact | 1000 | 0.2490 | 0.4841 | 0.7959 | 0.8921 | 0.9138 | 0.9217 | 0.9233 | 0.9234 |
| gemma | FiQA | 10 | 0.9966 | 0.9966 | 0.9966 | 0.9966 | 0.9966 | 0.9966 | 0.9966 | 0.9968 |
| gemma | FiQA | 100 | 0.9700 | 0.9928 | 0.9958 | 0.9960 | 0.9962 | 0.9963 | 0.9964 | 0.9966 |
| gemma | FiQA | 1000 | 0.2487 | 0.4827 | 0.8063 | 0.9308 | 0.9702 | 0.9881 | 0.9911 | 0.9937 |
| qwen | SciFact | 10 | 0.9967 | 0.9967 | 0.9967 | 0.9967 | 0.9967 | 0.9967 | 0.9967 | 0.9967 |
| qwen | SciFact | 100 | 0.9651 | 0.9871 | 0.9895 | 0.9897 | 0.9898 | 0.9899 | 0.9900 | 0.9900 |
| qwen | SciFact | 1000 | 0.2492 | 0.4850 | 0.8084 | 0.9277 | 0.9581 | 0.9670 | 0.9681 | 0.9682 |
| qwen | FiQA | 10 | 0.9995 | 0.9997 | 0.9997 | 0.9997 | 0.9997 | 0.9997 | 0.9997 | 0.9997 |
| qwen | FiQA | 100 | 0.9806 | 0.9967 | 0.9979 | 0.9981 | 0.9981 | 0.9982 | 0.9983 | 0.9984 |
| qwen | FiQA | 1000 | 0.2494 | 0.4879 | 0.8225 | 0.9427 | 0.9766 | 0.9908 | 0.9932 | 0.9953 |

The exact top 10 and top 100 are settled by depth 500–1,000; going deeper adds
nothing, because what is still missing at ∞ was never probed. A cut of 2,000
would raise the top-1,000 figure to 0.91–0.98 and double Pass 2's work, for no
gain in the results below.

## What it costs in ranking quality

nDCG@10 for each pipeline:

| Pipeline | gemma SciFact | gemma FiQA | qwen SciFact | qwen FiQA |
|---|---:|---:|---:|---:|
| exact dense over every document (no Pass 1) | 0.7900 | 0.4760 | 0.7817 | 0.5919 |
| exact two-pass: exact MaxSim top 1,000, exact dense rerank | 0.7900 | 0.4760 | 0.7817 | 0.5919 |
| exact two-pass, top 4,000 | 0.7900 | 0.4760 | 0.7817 | 0.5919 |
| index Pass 1 (16 probes), exact dense rerank | 0.7900 | 0.4710 | 0.7800 | 0.5915 |
| index Pass 1 (64 probes), exact dense rerank of its 1,000 | 0.7900 | 0.4737 | 0.7817 | 0.5920 |
| index Pass 1 (64 probes), exact dense rerank of its 4,000 | 0.7900 | 0.4737 | 0.7817 | 0.5920 |
| index Pass 1 (256 probes), exact dense rerank | 0.7900 | 0.4760 | 0.7817 | 0.5919 |
| **production: index Pass 1 (64 probes) + 8-bit Pass 2** | **0.7906** | **0.4729** | **0.7807** | **0.5925** |
| exact MaxSim only (no Pass 2) | 0.7814 | 0.4244 | 0.7905 | 0.5003 |

- The cut is free: 1,000 and 4,000 give the same nDCG in every case.
- Gemma FiQA's −0.0023 at 64 probes disappears at 256, so it is a probing cost.
  The other cases lose nothing at 64 probes.
- Pass 2's 8-bit codes agree with exact dense scoring on 99.7–99.8% of top-10
  positions (identical top 10 on 260/300 and 262/300 SciFact queries, 525/648 and
  563/648 FiQA queries, gemma and qwen). The resulting nDCG@10 differences
  (−0.0010 to +0.0006) are within noise.

Judged-relevant documents: share that reach Pass 2.

| Model | Dataset | Exact Pass 1 | Index, 16 probes | 64 probes | 256 probes |
|---|---|---:|---:|---:|---:|
| gemma | SciFact | 1.0000 | 0.9927 | 0.9967 | 1.0000 |
| gemma | FiQA | 0.9340 | 0.9112 | 0.9204 | 0.9234 |
| qwen | SciFact | 0.9967 | 0.9900 | 0.9833 | 0.9900 |
| qwen | FiQA | 0.9602 | 0.9471 | 0.9529 | 0.9539 |

On FiQA 7.8% (gemma) and 4.8% (qwen) of relevant documents fall outside even
the *exact* MaxSim top 1,000. That is a limit of the best-chunk objective rather
than of the index, and those documents rank too low in the dense rerank to reach
the top 10 anyway: the exact two-pass equals exact dense over every document.

## Things tried that did not help

All at 64 probes, measured on the same codes. Pass-1 recall per case; nDCG@10
(exact rerank) was unchanged by every variant in every case.

| Change | gemma SciFact | gemma FiQA | qwen SciFact | qwen FiQA |
|---|---:|---:|---:|---:|
| production | 0.7959 | 0.8063 | 0.8084 | 0.8225 |
| rank Pass 1 by `estimate + error_bound` (optimistic) | 0.7949 | 0.8027 | 0.8072 | 0.8199 |
| estimate against `q − c`, as the paper does | 0.7858 | 0.8056 | 0.8122 | 0.8338 |
| no rotation | 0.7971 | 0.8050 | 0.7966 | 0.8085 |
| dense Haar rotation | 0.7954 | 0.8055 | 0.8090 | 0.8230 |

The `q − c` form and the rotation do change Pass-1 recall on qwen (the rotation
by +1.2–1.4 points), but not the exact top 100 or the final ranking. Both are
discussed in `rabitq-rotation-audit.md`.

## Reproduce

From `minnal_db/`, after a `vector_bench` run has created the frozen
embeddings for the dataset and model:

```sh
MINNAL_BENCH_MODEL=qwen MINNAL_BENCH_DATASET=fiqa MINNAL_BENCH_LABEL=study cargo test -p minnal_db \
  --all-features --release --lib vector_bench_pass1_study -- --ignored --nocapture
```

FiQA takes about a minute on 32 cores and writes
`work/bench/results/study/fiqa-qwen-study.md`.
