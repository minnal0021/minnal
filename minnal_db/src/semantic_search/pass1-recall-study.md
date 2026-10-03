# Pass 1 keeps "80%" of the exact candidates: what is lost, and does it matter?

The vector bench reports a **Pass-1 recall** of about 0.80 at the production
settings: of the 1,000 documents an exact Pass 1 would hand to Pass 2, the index
hands over about 800. This report finds where the other ~200 go, why, and what
they cost in final ranking quality. It also checks Pass 2 on its own.

*Measured on gemma (`embeddinggemma-300m`, Q8, llama.cpp) frozen embeddings of
BEIR SciFact (5,183 docs, 300 queries) and FiQA (57,600 docs, 648 queries),
bundled gemma centroids (256), 64 probes and a cut of 1,000 unless stated. Commit
`f163050`. Harness: [`vector_bench/study.rs`](vector_bench/study.rs)
(`vector_bench_pass1_study`).*

## Answer

- **The missing ~20% is churn at the bottom of the list, not lost results.**
  81–91% of the missed documents rank 501–1000 in the exact list, and almost
  none rank in the top 100. The index keeps **99.6% of the exact top 100 on
  FiQA** (96.7% on SciFact) and 98.5–99.7% of the exact top 10.
- **Most misses are near-ties.** Their exact score is within the 1-bit
  estimator's own error bound of the 1,000th score, so 1-bit noise decides which
  side of the cut they land on. Probing all 256 clusters does not change this
  (recall 0.81 / 0.84). On SciFact a further 50% of misses come from
  probing: the document's best chunk is in a cluster that was not probed.
  The estimator alone accounts for about 1%.
- **The cut costs nothing.** Re-ranking the top 4,000 instead of the top 1,000
  gives the same nDCG@10 to four decimals on both datasets.
- **The whole quality gap is probing, and it is small.** FiQA: exact two-pass
  0.4760, index with an exact Pass-2 rerank 0.4737 at 64 probes and 0.4760 at
  256. SciFact: no gap. Pass 2's 8-bit codes cost 0.0008 more on FiQA (production
  0.4729).
- **The two-pass design itself loses nothing.** Exact two-pass equals exact
  dense scoring of every document on both datasets (0.4760 / 0.7900). Pass 2 is
  needed: Pass-1 MaxSim alone scores 0.4244 on FiQA.
- **No bug found.** The simulated Pass 1 reproduces `search()` exactly: the same
  candidate set and the same final top 10 on every query of both datasets.
- **Recommendation:** read Pass-1 quality from the exact top-10/top-100 coverage,
  not from top-1,000 overlap. The 0.80 figure mostly measures how noisy the
  bottom of two long lists is, which has no effect on what a user sees. For
  quality, the lever is `n_probes` (or a better partition, design doc M2–M3), not
  `first_pass_sparse_search_top_k`.

## How it is measured

The study rebuilds every chunk's 1-bit code in memory with the production
quantiser and scores Pass 1 with the production estimator. It then compares the
result with the **exact Pass 1**: MaxSim over the full-precision chunks of every
document, top 1,000 (`vector_bench/exact.rs`).

**Pass-1 recall** is `|index's 1,000 ∩ exact 1,000| / 1,000`. To confirm the
simulation is the real thing, the study also indexes the corpus into a database
and runs `search()`. The candidate sets and final top 10s are identical on
300/300 SciFact and 648/648 FiQA queries.

Every exact top-1,000 document the index misses is put into the first matching
class:

| Cause | Meaning |
|---|---|
| not probed | none of the document's chunks is in a probed cluster |
| best chunk not probed | some chunk was probed, but not the one that gives its exact score, so it was scored by a weaker chunk |
| near tie | the best chunk was probed, and the document's exact score is within that chunk's `error_bound` of the exact 1,000th score |
| estimator | the best chunk was probed and the margin exceeds the bound: the 1-bit estimate alone pushed it out |

`error_bound` is the paper's per-code error half-width, which holds for ~94%
of estimates (see `rabitq-rotation-audit.md`).

## Where the misses rank

Share of the exact top-k that the index keeps in its 1,000:

| Dataset | Probes | top 10 | top 50 | top 100 | top 500 | top 1000 |
|---|---:|---:|---:|---:|---:|---:|
| SciFact | 16 | 0.9560 | 0.9219 | 0.9029 | 0.8298 | 0.7200 |
| SciFact | 64 | 0.9853 | 0.9751 | 0.9672 | 0.9226 | 0.7959 |
| SciFact | 256 (all) | 1.0000 | 1.0000 | 1.0000 | 0.9811 | 0.8351 |
| FiQA | 16 | 0.9850 | 0.9793 | 0.9751 | 0.9344 | 0.7878 |
| FiQA | 64 | 0.9966 | 0.9961 | 0.9958 | 0.9634 | 0.8063 |
| FiQA | 256 (all) | 1.0000 | 1.0000 | 0.9998 | 0.9691 | 0.8094 |

Coverage falls off only at the bottom of the list. With every cluster probed
the top 100 is complete, and top-1,000 recall is still only 0.81–0.84: the
remaining misses are purely the 1-bit estimate reordering near-equal scores.

Missed documents by exact rank and cause, 64 probes:

| Dataset | Misses per query | rank 1–10 | 11–100 | 101–500 | 501–1000 | not probed | best chunk not probed | near tie | estimator |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SciFact | 204 | 0.1% | 1.5% | 17.4% | 81.0% | 37.5% | 12.9% | 49.2% | 0.3% |
| FiQA | 194 | 0.0% | 0.2% | 9.2% | 90.6% | 3.2% | 2.1% | 93.6% | 1.1% |

SciFact loses half its misses to probing, and FiQA almost none. This fits what
is known about the bundled centroids: they were fitted on general text (ELI5),
and SciFact's scientific abstracts pile into a few of them (43% of its
documents in one cluster). With all 256 clusters probed, both datasets are at
99% near-ties.

## At what depth they are cut off

A missed document is not far below the cut. In the approximate ranking its
median position is ~1,250 (SciFact) / ~1,315 (FiQA), and its 90th percentile
~1,860 / ~2,130. Share of the exact top-k inside the first N approximate
results, 64 probes (∞ = every document in the probed clusters):

| Dataset | Exact top | 250 | 500 | 1000 | 1500 | 2000 | 3000 | 4000 | ∞ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SciFact | 10 | 0.9850 | 0.9853 | 0.9853 | 0.9857 | 0.9857 | 0.9857 | 0.9857 | 0.9857 |
| SciFact | 100 | 0.9440 | 0.9641 | 0.9672 | 0.9682 | 0.9686 | 0.9689 | 0.9690 | 0.9690 |
| SciFact | 1000 | 0.2490 | 0.4841 | 0.7959 | 0.8921 | 0.9138 | 0.9217 | 0.9233 | 0.9234 |
| FiQA | 10 | 0.9966 | 0.9966 | 0.9966 | 0.9966 | 0.9966 | 0.9966 | 0.9966 | 0.9968 |
| FiQA | 100 | 0.9700 | 0.9928 | 0.9958 | 0.9960 | 0.9962 | 0.9963 | 0.9964 | 0.9966 |
| FiQA | 1000 | 0.2487 | 0.4827 | 0.8063 | 0.9308 | 0.9702 | 0.9881 | 0.9911 | 0.9937 |

The exact top 10 and top 100 are settled by depth 500–1,000; going deeper
adds nothing, because what is still missing at ∞ was never probed. A cut of
2,000 would raise the top-1,000 figure to 0.91–0.97 and double Pass 2's work,
for no gain in the results below.

## What it costs in ranking quality

nDCG@10 for each pipeline:

| Pipeline | SciFact | FiQA |
|---|---:|---:|
| exact dense over every document (no Pass 1) | 0.7900 | 0.4760 |
| exact two-pass: exact MaxSim top 1,000, exact dense rerank | 0.7900 | 0.4760 |
| exact two-pass, top 4,000 | 0.7900 | 0.4760 |
| index Pass 1 (64 probes), exact dense rerank of its 1,000 | 0.7900 | 0.4737 |
| index Pass 1 (64 probes), exact dense rerank of its 4,000 | 0.7900 | 0.4737 |
| index Pass 1 (256 probes), exact dense rerank | 0.7900 | 0.4760 |
| **production: index Pass 1 (64 probes) + 8-bit Pass 2** | **0.7906** | **0.4729** |
| exact MaxSim only (no Pass 2) | 0.7814 | 0.4244 |

- The cut is free: 1,000 and 4,000 give the same nDCG.
- FiQA's −0.0023 at 64 probes disappears at 256, so it is a probing cost.
- Pass 2's 8-bit codes agree with exact dense scoring on 99.7% of top-10
  positions (identical top 10 on 260/300 SciFact and 525/648 FiQA queries).
  On FiQA that costs 0.0008 nDCG; SciFact's +0.0006 is noise.

Judged-relevant documents: how many reach Pass 2.

| Dataset | Exact Pass 1 | Index, 16 probes | 64 probes | 256 probes |
|---|---:|---:|---:|---:|
| SciFact | 1.0000 | 0.9927 | 0.9967 | 1.0000 |
| FiQA | 0.9340 | 0.9112 | 0.9204 | 0.9234 |

On FiQA 7.8% of relevant documents fall outside even the *exact* MaxSim
top 1,000 (5.3% at ranks 1,001–4,000, 2.5% beyond). That is a limit of the
best-chunk objective rather than of the index, and those documents rank too low
in the dense rerank to reach the top 10 anyway: the exact two-pass equals
exact dense over every document.

## Things tried that did not help

All at 64 probes, measured on the same codes:

| Change | SciFact Pass-1 recall | FiQA Pass-1 recall | nDCG@10 (exact rerank) |
|---|---:|---:|---|
| production | 0.7959 | 0.8063 | 0.7900 / 0.4737 |
| rank Pass 1 by `estimate + error_bound` (optimistic) | 0.7949 | 0.8027 | unchanged |
| estimate against `q − c`, as the paper does | 0.7858 | 0.8056 | unchanged |
| no rotation | 0.7971 | 0.8050 | unchanged |
| dense Haar rotation | 0.7954 | 0.8055 | unchanged |

None moves the exact top 10 or top 100 either. The last three are discussed in
`rabitq-rotation-audit.md`.

## Reproduce

From `minnal_db/`, after a `vector_bench` run has created the frozen
embeddings for the dataset:

```sh
MINNAL_BENCH_DATASET=fiqa MINNAL_BENCH_LABEL=study cargo test -p minnal_db \
  --all-features --release --lib vector_bench_pass1_study -- --ignored --nocapture
```

FiQA takes about a minute on 32 cores and writes
`work/bench/results/study/fiqa-gemma-study.md`.
