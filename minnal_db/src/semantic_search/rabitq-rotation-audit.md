# RaBitQ in minnal: audit against the papers and the reference library

This report checks whether minnal's RaBitQ code (the random rotation, the 1-bit
and multi-bit codes, and the similarity both search passes compute) does what
the RaBitQ papers and the reference library do. Where it differs, the report says
whether the difference matters, and the measurements behind that judgement.

*Sources: RaBitQ, Gao & Long, SIGMOD 2024 ([arXiv 2405.12497](https://arxiv.org/abs/2405.12497));
Extended RaBitQ, Gao et al., SIGMOD 2025 ([arXiv 2409.09913](https://arxiv.org/abs/2409.09913));
[RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library) at commit
`d929e30`. Measurements on gemma (`embeddinggemma-300m`, Q8, llama.cpp) frozen
embeddings of BEIR SciFact and FiQA, with the bundled gemma centroids, at commit
`f163050`. Harness: [`vector_bench/study.rs`](vector_bench/study.rs).*

## Verdict

- **The rotation is right.** It matches the library's `FhtKacRotator` step for
  step, its inverse is exact, and every code-facing term uses the rotated query.
- **Both similarity formulas are right.** The Pass-1 estimate is the paper's
  unbiased estimator (measured bias −0.00003 on FiQA). The Pass-2 score is
  algebraically the library's inner-product estimate.
- **One deliberate difference, kept:** Pass 1 estimates the residual's inner
  product against the raw query `q`. The paper and library use the query
  residual `q − c`. That form is only more accurate when the query sits close
  to the centroid. Gemma queries do not: `‖q − c‖` averages 1.10–1.15 against
  `‖q‖ = 1`, and the paper's form measured slightly *worse* (see row 9).
- **One bug, fixed (`9fb62a9`):** the multi-bit rescale-factor search could
  score codes above the top level for 2 and 4 total bits. The default (8 bits)
  was never affected.
- **The rotation buys nothing on gemma.** A dense Haar rotation, FhtKac and no
  rotation at all give the same Pass-1 recall and nDCG. Gemma residuals are
  already spread evenly over dimensions, so there is nothing for the rotation to
  fix. It stays: it costs ~1.5 µs per query and protects a model whose residuals
  are concentrated in a few dimensions.

## Terms used here

- **Residual** `r = x − c`: a vector minus the centroid of the cluster it is
  stored under. `o = r/‖r‖` is the normalised residual.
- **Code** `ō`: the quantised `o`. A 1-bit code keeps the sign of each rotated
  coordinate, `ō = sign(Pᵀo)/√D`. A B-bit code places each coordinate on a grid
  of `2^B` levels. `⟨ō, o⟩` measures how well the code represents `o` (≈0.80 for
  1 bit; close to 1 for 8 bits).
- **Rotation `P`**: a random orthogonal matrix. Codes are computed on `Pᵀx` and
  `Pᵀc`, and queries are rotated once, `q' = Pᵀq`.
- **FhtKac**: the fast rotation the library uses in place of a dense random
  matrix. Four rounds of random sign flips, a Walsh–Hadamard transform and one
  Kac butterfly (`rotation.rs`).
- **Haar rotation**: a rotation drawn uniformly from all rotations. The papers'
  theorems assume one.
- **`error_bound`**: the per-vector half-width the paper gives for the
  estimator's error, `‖r‖·√((1 − ⟨ō,o⟩²)/⟨ō,o⟩²)·ε₀/√(D−1)`, with `ε₀ = 1.9`.
- **Pass 1 / Pass 2**: search first scores 1-bit chunk codes in the probed
  clusters and keeps the best 1,000 documents, then re-ranks those with each
  document's 8-bit whole-text code (`service/mod.rs`, `search`).

## Checklist

✅ matches · ⚠️ differs on purpose (impact stated) · ❌ bug

### The rotation

| # | Check | Result |
|---|---|---|
| 1 | Structure | ✅ Same as `rotator_kernels.hpp`: 4 rounds of sign flip → unnormalised FWHT on a power-of-two block (front block in rounds 0 and 2, back block in rounds 1 and 3) → ×`1/√trunc` → Kac butterfly; then ×0.25. Power-of-two dims skip the Kac step and the 0.25 in both. For D = 768 both use blocks `[0, 512)` and `[256, 768)`. |
| 2 | Signs and inverse | ✅ Independent fair sign bits per round. minnal draws them from splitmix64 with a fixed seed, the library from mt19937 with a random seed it saves. Same distribution, different bits. The inverse applies each self-inverse step in reverse order and is pinned by `preserves_inner_products_and_inverts` and `matrix_is_orthogonal`. |
| 3 | Padding | ⚠️ The library pads D to a multiple of 64; minnal requires an even D ≥ 8 and does not pad. Both bundled models use D = 768, a multiple of 64, so nothing differs today. |
| 4 | What is rotated | ✅ Both rotate the data vector and the centroid separately, then form the residual in rotated space (`index_embedding_in_cluster`; library `ivf.hpp`). The rotation is linear, so this equals rotating the residual. Nothing nonlinear runs first. |
| 5 | One `P` for everything | ✅ As in both papers (Algorithm 1) and the library: one rotation for all clusters. |
| 6 | Mixing the spaces | ✅ Codes, `Σq'` and the packed dot products use `q'`; probing and `⟨q, c⟩` use `q` (rotation preserves both). Pinned by `rotated_scoring_matches_scoring_fully_rotated_inputs`, and end to end by the study: the simulated candidate lists equal `search()`'s on every query. |

### The similarity calculation

| # | Check | Result |
|---|---|---|
| 7 | 1-bit code | ✅ `ō = sign/√D`; `scaling_factor = ‖r‖/(⟨ō,o⟩√D)`; `error_bound` equals the library's `tmp_error` (same 1.9, same `D − 1`). |
| 8 | Pass-1 estimate | ✅ `⟨x,q⟩ ≈ ⟨q,c⟩ + scaling·(2·Σ_{bit=1} q'ᵢ − Σq')`, which is `⟨q,c⟩ + ‖r‖·⟨ō,q'⟩/⟨ō,o⟩`: the paper's unbiased estimator (Theorem 3.2) applied to `⟨r, q⟩`. Measured over 11.1 M (query, chunk) pairs on FiQA: RMSE 0.0239, bias −0.00003. Scores from different clusters share one scale (each includes its own `⟨q,c⟩`), which the cross-cluster MaxSim needs. |
| 9 | Raw `q` vs `q − c` | ⚠️ Kept. See *The one deliberate difference* below. |
| 10 | Pass-2 estimate | ✅ `score = 1 − (−⟨q,c⟩ + addition_factor + scaling·(⟨code,q'⟩ + c_b·Σq'))`, with `c_b = −(2^{B−1} − ½)`. Expanding minnal's own factors gives `⟨q,c⟩ + ⟨r,c⟩ + ‖r‖²⟨s, q−c⟩/⟨r,s⟩`, where `s` is the centred code: the library's `METRIC_IP` estimate, term for term, and `1 − estimate` is `⟨x,q⟩` with the right sign. Pass 2 already uses `q − c`. Measured RMSE 0.00024, bias 0.00000. |
| 11 | Multi-bit codes | ✅ Sign-magnitude grid `sign(rᵢ)·(kᵢ + ½)` built as in the library; factors identical to `rabitq_impl.hpp` `METRIC_IP`. |
| 12 | Rescale-factor search | ❌ → fixed in `9fb62a9`. See *The bug* below. |
| 13 | SIMD kernels | ✅ `pack_bits` puts dimension `i` in bit `i % 64` of word `i / 64`; every `packed_ip_*` path reads that order and is tested against an independent bit-test reference. `pack_bytes` is little-endian, which the x86-64 and aarch64 kernels assume (both targets are little-endian). Small gap: no test feeds `pack_bytes` output to `multi_bit_dot_best` against an independent reference. |
| 14 | Query quantisation | ⚠️ The paper quantises `q'` to 4 bits with randomised rounding, for speed only (Theorem 3.2 is stated for the exact query). minnal keeps the float query, which is slower per entry but adds no error. |
| 15 | Use of `error_bound` | ⚠️ The paper drops a candidate whose lower bound (estimate − bound) is worse than the current k-th result. minnal keeps a fixed top 1,000 and never reads the bound (Pass 2 only passes it through to the API). Ranking Pass 1 by the optimistic `estimate + error_bound` instead was measured and did not help (Pass-1 recall 0.8027 vs 0.8063 on FiQA, nDCG unchanged). |

### The one deliberate difference: estimating against `q` rather than `q − c`

Pass 1 needs `⟨r, q⟩` for each chunk. There are two ways to get it:

| Form | Estimated term | Exact term | Error scales with |
|---|---|---|---|
| minnal | `⟨r, q⟩` | — | `‖r‖ · ‖q‖` |
| paper, library | `⟨r, q − c⟩` | `⟨r, c⟩`, stored at index time | `‖r‖ · ‖q − c‖` |

The paper's form is better exactly when `‖q − c‖ < ‖q‖`, that is, when the query
is close to the cluster's centroid. Gemma embeds queries and documents with
different prompts, so queries do not sit inside document clusters. Over the
probed clusters `‖q − c‖` averages **1.15 on SciFact and 1.10 on FiQA**, against
`‖q‖ = 1`. Measured with both forms on the same codes:

| Dataset | Form | RMSE | Within `error_bound` × query norm | Pass-1 recall | nDCG@10 (exact rerank) |
|---|---|---:|---:|---:|---:|
| SciFact | raw `q` (minnal) | 0.0274 | 93.8% | 0.7959 | 0.7900 |
| SciFact | `q − c` (paper) | 0.0298 | 94.7% | 0.7858 | 0.7900 |
| FiQA | raw `q` (minnal) | 0.0239 | 94.3% | 0.8063 | 0.4737 |
| FiQA | `q − c` (paper) | 0.0255 | 94.6% | 0.8056 | 0.4737 |

So minnal's form stays. The paper's form needs one more stored float per chunk
and a per-cluster query residual, and here it buys nothing. This is a property
of the embedding model, so it is worth re-checking for any new model. The
study prints both forms.

The same table checks the error bound. Under the paper's analysis the error
is close to Gaussian with standard deviation `error_bound/1.9`, so the bound
should hold for P(|Z| ≤ 1.9) = **94.3%** of pairs. Measured: 93.8–94.6%. The
bound is calibrated, and "95% within bound" is not a sign of a problem. For
the raw form the query-norm factor is `‖q‖ = 1`, so the stored bound needs no
query-time scaling. For the `q − c` form it would need `× ‖q − c‖`, as the
library applies.

### The bug: the multi-bit rescale search scored impossible codes

An 8-bit code is built by scaling `|o|` by a factor `t` and rounding each
coordinate down to a level in `0 … 2^{B−1} − 1`. `best_rescale_factor` sweeps
`t` from `t_start` to `t_end` and keeps the `t` with the largest `⟨ō, o⟩`. Its
starting codes were not clamped to the top level. A coordinate already at the
top was also still pushed one level past it. For 3 magnitude bits (B = 4) the
largest coordinate started at ⌊17 × 0.52⌋ = 8, above the maximum of 7. For 1
magnitude bit (B = 2) the push went past the top. The search then scored codes
the quantiser can never emit, and could settle on a worse `t`. A new test,
`best_rescale_factor_matches_an_exhaustive_sweep_on_clamped_codes`, compares the
search with an exhaustive sweep of every critical `t`. Before the fix, B = 2 fell
up to **0.035 short** of the best achievable `⟨ō,o⟩`; after it, every bit-width
from 2 to 8 matches.

The stored factors were always computed from the final, clamped codes, so
estimates stayed consistent; only code quality suffered. The default B = 8 starts
at level 105 of 127, so it was never affected: the gemma SciFact bench is
byte-identical before and after.

## Does the rotation help?

The papers' guarantees assume a Haar rotation; FhtKac approximates one cheaply.
Two measurements:

**How close FhtKac is to Haar (E1).** Rotate every basis vector (the hardest
input: all its weight in one coordinate) and look at the output coordinates,
scaled by `√D`. A Haar rotation makes them nearly standard normal.

| Rotation | Excess kurtosis (0 = normal) | Largest coordinate |
|---|---:|---:|
| none | +765 | 27.7 |
| FhtKac, 4 rounds (production and library) | +1.07 | 8.95 |
| FhtKac twice, 8 rounds | −0.01 | 4.97 |
| dense Haar | −0.00 | 4.95 |

Four rounds leave heavier tails than Haar on worst-case inputs; eight rounds are
indistinguishable from Haar. (`vector_bench_rotation_spread`.)

**Whether that matters on real data (E2).** The same Pass-1 search built three
ways, 64 probes:

| Dataset | Rotation | Exact top 10 kept | Exact top 100 kept | Pass-1 recall | nDCG@10 (exact rerank) |
|---|---|---:|---:|---:|---:|
| SciFact | none | 0.9853 | 0.9676 | 0.7971 | 0.7900 |
| SciFact | FhtKac | 0.9853 | 0.9672 | 0.7959 | 0.7900 |
| SciFact | dense Haar | 0.9850 | 0.9673 | 0.7954 | 0.7900 |
| FiQA | none | 0.9966 | 0.9959 | 0.8050 | 0.4737 |
| FiQA | FhtKac | 0.9966 | 0.9958 | 0.8063 | 0.4737 |
| FiQA | dense Haar | 0.9966 | 0.9957 | 0.8055 | 0.4737 |

No difference beyond noise. Gemma's residuals are already spread over all
dimensions (`⟨ō, o⟩ ≈ 0.80` before rotation, the value a random vector gets;
`vector_bench_residual_spread`), so FhtKac's extra tail weight on spiky inputs
never comes into play. A model with spikier residuals could behave differently,
which is why the qwen run repeats E2.

## Reproduce

From `minnal_db/`, with frozen embeddings in `work/bench/{dataset}/{model}`
(the first `vector_bench` run creates them):

```sh
MINNAL_BENCH_DATASET=fiqa cargo test -p minnal_db --all-features --release --lib \
  vector_bench_pass1_study -- --ignored --nocapture        # rows 8–9, 15, E2
cargo test -p minnal_db --all-features --release --lib \
  vector_bench_rotation_spread -- --ignored --nocapture    # E1
cargo test -p minnal_db --all-features --lib best_rescale_factor_matches   # row 12
```

The study writes `work/bench/results/{label}/{dataset}-{model}-study.md`.
