# M3 split algorithm: SPFresh's LIRE with codes instead of vectors

M3 grows each namespace's partition by splitting postings (option C, chosen by
the M3-pre simulation: [`gemma`](gemma/m3-pre-simulation.md),
[`qwen`](qwen/m3-pre-simulation.md)). This report settles *how* a posting
splits and what happens around it. The starting point is the published design
that does this at scale, SPFresh's LIRE protocol, followed as closely as
minnal allows. minnal stores 1-bit codes, not the vectors SPFresh keeps, so the
question is which parts of LIRE survive that change and what has to be done
differently.

*Measured on frozen gemma and qwen embeddings of BEIR FiQA (57,600 documents,
172,998 chunks, 648 queries; both models) and SciFact (5,183 documents, 22,787
chunks, 300 queries; qwen). Simulator: `work/bench/analysis/m3pre/spfresh.py`,
built on the M3-pre simulator `sim.py` (same rotation, codes, search and
ground truth). Results: `work/bench/results/m3pre/spfresh_{dataset}_{model}.json`.
Branch `dynamic-rabitq-index` at `2c5dd35`.*

## Answer

Build M3 as LIRE, with **one deviation**: a new posting's centre is the
**mean** of its members' code reconstructions, not SPFresh's medoid. Use
**1-bit codes** for search and maintenance, **no replicas**, and leave the
error-bound gate **off** by default.

| Choice | Decision | Evidence (FiQA, both models, both insertion orders) |
|---|---|---|
| Split centre | **Mean** of the reconstructions (deviation) | The medoid collapses with 1-bit codes (12,000–24,000 postings, recall@10 0.58–0.69 at 1% read) and loses to the mean even with exact vectors (0.84–0.89 against 0.89–0.93) |
| What maintenance reads | **1-bit codes** | 2-bit maintenance codes add at most about a point of recall, and only when a query reads 2% or less (+0.3 to +1.0 with the gate on in both; −0.5 to +0.7 with it off) |
| Pass-1 search code | **1 bit, unchanged** | A 2-bit search code raises Pass-1 recall by 1–2 points but leaves final recall unchanged (static k-means: 0.952 against 0.951 at 1% read on qwen) |
| Replicas | **None** | With mean centres SPFresh's replica rule keeps almost none (1.00 copies per chunk); its 4–5 copies with medoid centres lose to an unreplicated partition of the same size |
| Error-bound gate | **Optional, default off** | 12–17% fewer key moves; recall within 0.4 points on shuffled order, 0.7–0.9 points lower at 1% read on drifting order |

With these choices the grown partition trails a k-means partition fitted on
the whole corpus, at the same posting count, by 2–4 points of recall@10 when a
query reads 1% of the chunks, 1–2.4 points at 2%, and under 1 point from 5%.
nDCG moves much less than recall: within 0.7 points of the static partition at
every cutoff from @10 to @100 when a query reads 2% or 5% of the chunks, and
within 0.3 points at 10%. The documents it misses are the ones that barely make
the exact top 10. At today's probe budget (70,000 entries, about 65% of FiQA)
every variant returns the exact top 10.

## Terms used here

- **Posting**, **entry**, **K**, **E**, **target posting size**: as in the
  M3-pre reports. A posting is one partition of a namespace's chunk codes under
  one key prefix; an entry is one key per (posting, document); K counts
  postings, E entries. Here the split limit counts **chunks**, as SPFresh
  counts vectors: a posting splits when it holds more than twice the target.
- **LIRE**: SPFresh's "lightweight incremental rebalancing" protocol: split an
  oversized posting in two, then reassign the vectors near the split that may
  now belong elsewhere, and merge postings that become too small.
- **Reconstruction** of a code: `x̂ = c + ‖r‖ · ȳ / ⟨ȳ, o⟩`, the unbiased
  estimate of the chunk's vector from its code, where `c` is the centre the code
  was encoded against and `ȳ` the normalised code.
- **Medoid**: the member of a group nearest to its mean, an actual vector of the
  group. **Mean** (centroid): the average of the members, usually not a member.
- **Replica**: a copy of a vector stored in a second posting. SPANN and SPFresh
  keep up to 8.
- **Static k-means at the same K**: k-means fitted once on all the corpus's
  float embeddings, with exactly as many postings as the variant it is compared
  with. Posting count changes recall per entry read on its own (smaller
  postings read entries more selectively), so every variant is compared with
  its own static baseline.
- **Recall@10 at x% read**: the share of the exact top 10 that the index
  returns when a query reads x% of the namespace's chunk count in entries
  (FiQA: 1% ≈ 1,800 entries, 2% ≈ 3,530, 5% ≈ 8,720, 10% ≈ 17,370). This is an
  absolute count, the same for every variant, so a replicated index is not
  credited for reading a small share of its larger E.
- **Insertion orders**: *shuffled* (random) and *drifting* (one topic at a
  time, 8 topics), as in the M3-pre reports.

**Noise.** With 648 FiQA queries the standard error of recall@10 is roughly
±0.4 points; differences under about a point are not evidence on their own.

## What SPFresh does

From the paper (Xu et al., *SPFresh: Incremental In-Place Update for
Billion-Scale Vector Search*, SOSP 2023, arXiv 2410.14452) and its reference
implementation in Microsoft's SPTAG (`AnnService/inc/Core/SPANN/ExtraDynamicSearcher.h`,
`AnnService/inc/Core/Common/BKTree.h`, defaults from
`SPANN/ParameterDefinitionList.h`):

1. **Insert.** A new vector goes to its nearest postings: the 64 nearest
   centres are searched and up to 8 are chosen by a relative-neighbourhood rule
   (a further centre is skipped if it is closer to an already chosen centre
   than to the vector).
2. **Split** a posting that exceeds its limit (118 vectors). Deleted vectors are
   dropped first; if the posting is still over the limit it is divided by
   **balanced 2-means**: three trials from two random members, keeping the one
   with the lowest total distance, then up to 100 iterations over at most 1,000
   sampled members. Each member goes to the centre minimising
   `distance + λ · (that cluster's size)`, which keeps the halves even, with
   `λ = min(λ_spread, 1 / (100 · samples))`. Iterations stop when the centres
   move less than 0.001 or 5 iterations bring no improvement. Each centre is then
   replaced by the member nearest it (a medoid), and every member is assigned
   to the nearer of the two.
3. **Reassign** after a split. A vector of the split posting is a candidate if
   the old centre is strictly closer than its new one. A vector in one of the
   64 postings nearest the old centre is a candidate if a new centre is closer
   than the old centre *and* closer than its current one. Each candidate is
   routed again as in step 1; if its current posting is among those chosen it
   stays, otherwise all its copies move.
4. **Merge** a posting with 10 or fewer vectors into the first of its nearest
   postings whose combined size stays under the limit. The shorter posting is
   removed, and its vectors move if the surviving centre is farther from them
   than their old one.

SPFresh stores the full vector in every posting, so every distance in steps
2–4 is exact.

## What the simulator copies, and what it cannot

`spfresh.py` implements steps 1–4 with SPTAG's parameters: 64 candidate
centres, up to 8 replicas, reassignment over the 64 nearest postings, balanced
2-means with three random-pair trials, ≤ 1,000 samples, λ factor 100, ≤ 100
iterations, and the merge rule. The limits are minnal's target with SPFresh's
ratio: split above 2 × target chunks (target 128, so 256), merge at
`round(256 · 10/118)` = 22 chunks or fewer.

What changes with codes:

- **Maintenance sees reconstructions.** Every distance in steps 2–4 uses `x̂`
  from the stored code (1, 2 or 4 bits), or the float in the reference runs
  that model SPFresh exactly.
- **Codes keep their centre when they move** (the M2 design). In the float
  runs a moved copy is encoded afresh against its new posting, as SPFresh's
  stored vectors allow.
- **New chunks arrive as floats**, so each replica of a new chunk is encoded
  against its own posting's centre.
- **Not modelled**: SPANN's exclusion of head vectors from postings (a minnal
  centre is only a routing point, never a search result), deferred deletion
  (deletes here are immediate, so the split-time clean-up has nothing to do),
  concurrency, and SPTAG's approximate centre search (the simulator scans every
  centre exactly).

## Runs

| Run | Replicas | Maintenance sees | Split centre | Gate | Search code |
|---|---|---|---|---|---|
| paper | up to 8 | float | medoid | — | 1 bit |
| float, medoid | 1 | float | medoid | — | 1 bit |
| 1-bit, medoid | 1 | 1-bit code | medoid | off | 1 bit |
| 1-bit, mean | 1 | 1-bit code | mean | off | 1 bit |
| 1-bit, mean, gate | 1 | 1-bit code | mean | on | 1 bit |
| 2-bit maintenance, mean, gate | 1 | 2-bit code | mean | on | 1 bit |
| 1-bit, mean, 8 replicas | up to 8 | 1-bit code | mean | off | 1 bit |
| 2-bit search, mean | 1 | 2-bit code | mean | off | 2 bits |
| 2-bit search, medoid | 1 | 2-bit code | medoid | off | 2 bits |
| static k-means | 1 (or up to 8) | — | — | — | 1 or 2 bits |

The **gate** uses the `error_bound` minnal already stores with every code: it
bounds the error of the code's estimate of `⟨r, u⟩` for a unit vector `u`, so a
comparison of `D(x, A)` with `D(x, B)` is uncertain by at most
`2 · error_bound · ‖A − B‖`. With the gate on, a chunk becomes a reassignment
candidate, or moves, only when the estimated margin exceeds that bound.

## Results

### qwen, FiQA, shuffled order

| Run | K | Copies per chunk | Key moves per insert | 1% | 2% | 5% | 10% | nDCG@10 at 2% |
|---|---|---|---|---|---|---|---|---|
| paper | 5,572 | 5.06 | 20.1 | 0.939 | 0.975 | 0.993 | 0.997 | 0.588 |
| float, medoid | 1,128 | 1.00 | 1.0 | 0.890 | 0.940 | 0.975 | 0.991 | 0.570 |
| 1-bit, medoid | 12,580 | 1.00 | 1.0 | 0.685 | 0.790 | 0.891 | 0.942 | 0.529 |
| **1-bit, mean** | 1,045 | 1.00 | 2.0 | **0.926** | **0.962** | **0.984** | 0.993 | **0.580** |
| 1-bit, mean, gate | 1,076 | 1.00 | 1.8 | 0.922 | 0.961 | 0.984 | 0.993 | 0.579 |
| 2-bit maintenance, mean, gate | 973 | 1.00 | 1.7 | 0.932 | 0.964 | 0.986 | 0.993 | 0.583 |
| 1-bit, mean, 8 replicas | 1,036 | 1.00 | 2.0 | 0.929 | 0.963 | 0.985 | 0.994 | 0.579 |
| 2-bit search, mean | 948 | 1.00 | 1.9 | 0.933 | 0.966 | 0.988 | 0.995 | 0.582 |
| 2-bit search, medoid | 1,566 | 1.00 | 1.2 | 0.869 | 0.927 | 0.974 | 0.990 | 0.573 |
| static k-means, K 1,045 | 1,045 | 1.00 | 0 | 0.952 | 0.975 | 0.991 | 0.996 | 0.586 |
| static k-means, K 948, 2-bit search | 948 | 1.00 | 0 | 0.951 | 0.975 | 0.991 | 0.996 | 0.584 |
| static k-means, K 5,572 | 5,572 | 1.03 | 0 | 0.977 | 0.989 | 0.996 | 0.998 | 0.587 |

Exact nDCG@10 (full scan) is 0.592.

### All four FiQA runs, against static k-means at the same K

Recall@10 at 1% / 2% / 5% read, as points behind static k-means, and the
nDCG@10 difference at 2% read:

| | 1-bit, mean | 1-bit, mean, gate | 2-bit maintenance, mean, gate |
|---|---|---|---|
| qwen, shuffled | −2.6 / −1.4 / −0.6, nDCG −0.5 | −3.0 / −1.5 / −0.6, nDCG −0.6 | −2.0 / −1.1 / −0.5, nDCG −0.2 |
| qwen, drifting | −2.1 / −1.1 / −0.5, nDCG −0.2 | −2.9 / −1.6 / −0.7, nDCG −0.3 | −2.5 / −1.1 / −0.4, nDCG −0.2 |
| gemma, shuffled | −3.8 / −2.4 / −0.8, nDCG −0.1 | −3.8 / −2.6 / −0.8, nDCG +0.1 | −3.0 / −1.7 / −0.8, nDCG +0.3 |
| gemma, drifting | −3.2 / −2.0 / −0.9, nDCG +0.3 | −4.2 / −3.0 / −1.4, nDCG −0.1 | −3.6 / −2.6 / −1.2, nDCG +0.3 |

| | Medoid, 1-bit: K / recall@10 at 1% | Medoid, float: recall@10 at 1% | Mean, 1-bit: recall@10 at 1% | paper: copies per chunk / moves per insert |
|---|---|---|---|---|
| qwen, shuffled | 12,580 / 0.685 | 0.890 | 0.926 | 5.06 / 20.1 |
| qwen, drifting | 11,477 / 0.648 | 0.874 | 0.932 | 4.60 / 17.1 |
| gemma, shuffled | 23,199 / 0.576 | 0.839 | 0.890 | 4.56 / 16.8 |
| gemma, drifting | 23,939 / 0.594 | 0.842 | 0.896 | 4.17 / 14.8 |

### nDCG at every cutoff, against static k-means at the same K

Points of nDCG@10 / @20 / @30 / @40 / @50 / @100, variant minus static k-means
(the 2-bit search code against static k-means with the same code):

| Variant | Read | qwen, shuffled | qwen, drifting | gemma, shuffled | gemma, drifting |
|---|---|---|---|---|---|
| 1-bit, mean | 2% | −0.52 / −0.61 / −0.67 / −0.59 / −0.66 / −0.69 | −0.17 / −0.18 / −0.25 / −0.18 / −0.30 / −0.38 | −0.09 / −0.04 / −0.02 / −0.08 / −0.08 / −0.11 | +0.25 / +0.22 / +0.15 / +0.18 / +0.23 / +0.15 |
| | 5% | −0.16 / −0.18 / −0.21 / −0.21 / −0.18 / −0.25 | −0.06 / −0.05 / −0.04 / −0.07 / −0.07 / −0.11 | −0.39 / −0.44 / −0.43 / −0.50 / −0.54 / −0.54 | −0.15 / −0.21 / −0.19 / −0.18 / −0.24 / −0.27 |
| | 10% | +0.09 / +0.10 / +0.09 / +0.07 / +0.07 / +0.04 | +0.06 / +0.06 / +0.09 / +0.07 / +0.08 / +0.04 | −0.15 / −0.26 / −0.21 / −0.25 / −0.22 / −0.24 | +0.12 / +0.07 / +0.11 / +0.10 / +0.08 / +0.01 |
| 1-bit, mean, skip uncertain moves | 2% | −0.64 / −0.86 / −0.97 / −0.96 / −1.00 / −1.06 | −0.32 / −0.26 / −0.39 / −0.40 / −0.43 / −0.42 | +0.06 / +0.12 / +0.05 / +0.02 / +0.06 / +0.01 | −0.09 / −0.22 / −0.30 / −0.27 / −0.24 / −0.23 |
| | 5% | −0.19 / −0.24 / −0.32 / −0.34 / −0.33 / −0.38 | +0.13 / +0.11 / +0.14 / +0.12 / +0.15 / +0.11 | −0.54 / −0.57 / −0.62 / −0.59 / −0.63 / −0.61 | −0.18 / −0.28 / −0.32 / −0.35 / −0.36 / −0.35 |
| 2-bit maintenance, mean, skip uncertain moves | 2% | −0.22 / −0.32 / −0.47 / −0.43 / −0.39 / −0.46 | −0.21 / −0.27 / −0.30 / −0.26 / −0.38 / −0.40 | +0.29 / +0.34 / +0.36 / +0.28 / +0.33 / +0.31 | +0.25 / +0.37 / +0.24 / +0.28 / +0.31 / +0.22 |
| | 5% | −0.07 / −0.10 / −0.12 / −0.11 / −0.10 / −0.16 | −0.01 / −0.06 / −0.08 / −0.10 / −0.07 / −0.09 | −0.24 / −0.33 / −0.35 / −0.38 / −0.37 / −0.35 | +0.06 / +0.01 / 0.00 / 0.00 / −0.04 / −0.13 |
| 2-bit search, mean | 2% | −0.20 / −0.32 / −0.36 / −0.32 / −0.34 / −0.39 | −0.41 / −0.51 / −0.57 / −0.62 / −0.63 / −0.67 | −0.93 / −0.93 / −0.95 / −0.89 / −0.90 / −0.98 | −0.11 / −0.15 / −0.08 / −0.03 / 0.00 / −0.08 |
| | 5% | −0.12 / −0.13 / −0.12 / −0.13 / −0.13 / −0.12 | −0.13 / −0.15 / −0.12 / −0.16 / −0.13 / −0.14 | −0.04 / −0.02 / −0.09 / −0.05 / −0.05 / −0.10 | −0.37 / −0.34 / −0.37 / −0.36 / −0.35 / −0.38 |

- **1-bit with mean centres stays within 0.7 points at every cutoff** from 2%
  read, and within 0.3 at 10%. The gap is usually a little larger at @100 than
  at @10, so @10 alone understates it slightly; the largest is gemma shuffled at
  5% read (−0.39 at @10, −0.54 at @100).
- **No variant is consistently better.** The 2-bit options are ahead in some
  cells and behind in others by similar amounts; skipping uncertain moves is
  behind in 6 of 8 cells.
- The 1-bit medoid run, not tabled, trails a full scan by 2.2–8.8 points at
  these budgets.

## Findings

**1. The medoid is the one part of LIRE that does not carry over.** With 1-bit
codes a medoid is the reconstruction of a single member, which carries that
member's full code error. In 768 dimensions that places it well off the data:
new chunks route to it badly, postings fragment (12,000–24,000 instead of about
1,000 on FiQA) and recall at 1% read falls to 0.58–0.69. A 2-bit code halves the
error and recovers most of it (0.82–0.87), and only at 4 bits does the medoid
reach its float level (SciFact). The mean averages the reconstruction error of
every member, shrinking it roughly with the square root of the posting size,
and it minimises the squared distance to the members, so it beats the medoid
**even with exact vectors** (0.89–0.93 against 0.84–0.89 at 1% read). SPFresh
needs a medoid because SPANN's centres are real vectors kept in a graph index
and returned as results; minnal's centres are routing points only.

**2. 1-bit codes are enough for maintenance once centres are means.** With
the gate on in both, 2-bit maintenance codes add 0.3–1.0 points of recall at 1%
read, 0.3–0.9 at 2% and 0–0.4 at 5%, on every model and order. With the gate
off (the 2-bit-search runs, whose search width does not change final recall,
finding 3) the difference is −0.5 to +0.7 at 1% read. nDCG@10 at 2% read moves
by at most 0.4 points either way. At most a point of recall at budgets of 2% or
less does not justify a second stored code.

**3. A 2-bit search code does not reach the final ranking.** It raises Pass-1
recall by 1–2 points, but Pass 2 reranks the candidates exactly, and final
recall@10 is the same (static k-means at 1% read: 0.952 / 0.951 on qwen, 0.928 /
0.922 on gemma, with 1-bit / 2-bit search codes). This matches the M2c-pre and
M3-pre width results. A 2-bit code also doubles the bytes Pass 1 reads.

**4. Replicas follow the medoid, and do not pay.** SPFresh's replica rule skips
a further centre that is closer to an already chosen centre than to the
vector. Mean centres sit close together relative to their chunks, so the rule
keeps almost no second copies (1.00 copies per chunk). With medoid centres it
keeps 4.2–5.1, at 15–20 key moves per insert and 4–5 times the storage. That
run reads entries more selectively than the 1-replica runs, but only because
its postings are small: static k-means at its own K, with no replicas, beats
it at every budget (qwen shuffled, 1% read: 0.977 against 0.939).

**5. The error-bound gate is not free.** It halves reassignment candidates and
cuts key moves by 12–17% (most moves are a split rewriting its own posting,
which the gate does not touch). On shuffled order recall does not change; on
drifting order the gated runs lose another 0.7–0.9 points at 1% read on both
models, because some useful moves have margins within the bound. It stays
available for namespaces where write amplification matters more.

**6. What remains is a small-budget gap.** The recommended configuration trails
static k-means at the same K by 2.1–3.8 points of recall at 1% read and 0.5–0.9
at 5%. nDCG moves far less (within 0.7 points at every cutoff from 2% read), so
most misses are near-ties at the edge of the top 10.
The gap matters only if the probe budget falls to a few percent of the
namespace. The M2d default reads about 65% of FiQA's entries; a scaled budget
of 30% of the entries (M3-pre) would read about 19% of FiQA's chunk count,
where every gap here is under 0.3 points.

## Cost

Measured on qwen FiQA, shuffled, with the recommended configuration:

- **Splits:** 1,044 for 172,998 chunks, one per about 166 inserted chunks.
- **Work per split:** about 11,700 chunks re-checked (the 64 neighbouring
  postings) and 125 reassignment candidates, of which 76 moved. In arithmetic,
  about 150 million multiply-adds per split: reconstructing and running 2-means
  on the split posting is about 6 million; checking the neighbours about 45
  million; routing the candidates again against every centre about 96 million.
  Choosing the mean instead of the medoid costs nothing extra. Amortised, about
  1 million multiply-adds per inserted chunk, far below the cost of embedding
  it.
- **Reads per split:** the 64 neighbouring postings, about 1.3 MB of codes.
- **Writes:** 2.0 key moves per inserted chunk (a put and a tombstone each);
  1.8 with the gate. The paper configuration makes 15–20.

Two terms grow with the number of postings: routing candidates again and
finding the 64 nearest postings both scan every centre. At FiQA's 1,000
postings that is negligible; at about 100,000 postings it would call for an
index over the centres.

## Caveats

- **Entries read stand in for latency.** The cost of a probe (an LSM prefix
  seek per posting) is not modelled. This decides the target posting size,
  which M3a measures.
- **Two datasets, two models, one writer.** No concurrency; documents arrive 64
  at a time; FiQA is the evidence, SciFact (qwen only) agrees within its noise.
- **The gate was tested in one form**: margins against `2 · error_bound ·
  ‖A − B‖` at every decision. A looser multiple of the bound would trade the
  two effects differently.
- **The reference float runs model SPFresh's maintenance, not its whole
  system**: no head-vector exclusion, no deferred deletion, exact centre search.
