# Does Pass 2 help? Whole-document reranking against chunk MaxSim

minnal ranks in two passes. Pass 1 scores each document by its best-matching
chunk (MaxSim) and keeps the top 1,000. Pass 2 reranks those by one embedding of
the whole document (dense). On qwen SciFact, MaxSim alone scored a higher nDCG@10
than the two passes together (0.7905 against 0.7817), which raised the question
of whether Pass 2 is wrong for some models. This study answers it on four
datasets and two models.

*Measured on frozen embeddings of BEIR SciFact, FiQA, NFCorpus and ArguAna (test
splits) from two models served by llama.cpp: gemma (`embeddinggemma-300m`, Q8)
and qwen (`Qwen3-Embedding-8B`, Q4_K_M, truncated to 768 dimensions). Documents
are chunked into sentence windows (4 sentences, sliding 2). Everything here is
computed exactly on the float embeddings: no quantisation and no clusters, so
any difference is the scoring objective itself.*

## Answer

- **Pass 2 does not hurt qwen.** The SciFact gap is noise: +0.0089 nDCG@10, 95%
  interval [−0.0066, +0.0246], 38 queries better and 37 worse (sign test
  p = 1.0).
- **Across eight model–dataset pairs, two passes are never significantly worse
  than MaxSim alone, and significantly better in five**, by up to 0.09 nDCG@10
  (qwen FiQA). The other three are ties within noise.
- **Two passes rank as well as dense scoring of every document** in all eight
  pairs (within 0.0004), so the MaxSim cut at 1,000 loses nothing.
- **Why MaxSim alone falls behind:** taking the best of a document's chunks
  favours documents with many chunks, and a whole argument or answer is not
  always well represented by its best passage. Where documents are all about the
  same length (SciFact, NFCorpus abstracts) these effects are small and the two
  objectives tie. Where lengths vary (FiQA, ArguAna) MaxSim loses clearly.
- **Recommendation: keep Pass 2.** MaxSim is a good candidate generator, and
  whole-document dense is the better final ranking.

## Terms used here

- **MaxSim**: a document's score is `max_j ⟨q, d_j⟩` over its chunk embeddings
  `d_j`, as Pass 1 computes it (with one whole-query vector).
- **Dense**: a document's score is `⟨q, d⟩` for one embedding `d` of its whole
  text, as Pass 2 computes it.
- **Two-pass**: MaxSim top 1,000, reranked by dense.
- **nDCG@10**: the BEIR ranking metric, computed as `beir_eval::score` does
  (gain = graded relevance, `log2(rank + 1)` discount).
- **Paired comparison**: per query, the difference in nDCG@10 between two
  rankings. The interval is a 10,000-sample bootstrap of the mean difference; p
  is a two-sided sign test on the queries where the two differ.

## Results

| Model | Dataset | Queries | MaxSim only | Dense, every doc | Two-pass | MaxSim − two-pass [95% interval] | Better / worse / same | Sign-test p |
|---|---|---:|---:|---:|---:|---|---|---:|
| gemma | SciFact | 300 | 0.7814 | 0.7900 | 0.7900 | −0.0086 [−0.0271, +0.0099] | 30 / 50 / 220 | 0.033 |
| gemma | FiQA | 648 | 0.4244 | 0.4760 | 0.4760 | −0.0516 [−0.0663, −0.0365] | 94 / 255 / 299 | 3e−18 |
| gemma | NFCorpus | 323 | 0.3820 | 0.3917 | 0.3917 | −0.0097 [−0.0184, −0.0007] | 70 / 120 / 133 | 0.0004 |
| gemma | ArguAna | 1,406 | 0.6082 | 0.6603 | 0.6606 | −0.0524 [−0.0627, −0.0424] | 220 / 441 / 745 | 6e−18 |
| qwen | SciFact | 300 | 0.7905 | 0.7817 | 0.7817 | +0.0089 [−0.0066, +0.0246] | 38 / 37 / 225 | 1.0 |
| qwen | FiQA | 648 | 0.5003 | 0.5919 | 0.5919 | −0.0915 [−0.1072, −0.0763] | 77 / 304 / 267 | 5e−33 |
| qwen | NFCorpus | 323 | 0.3940 | 0.3981 | 0.3981 | −0.0041 [−0.0126, +0.0042] | 96 / 106 / 121 | 0.53 |
| qwen | ArguAna | 1,406 | 0.6683 | 0.7343 | 0.7342 | −0.0659 [−0.0763, −0.0556] | 152 / 481 / 773 | 1e−40 |

"Better / worse" counts queries where MaxSim alone scores higher / lower than
two-pass. Two-pass is significantly better (interval excludes zero) for gemma
FiQA, NFCorpus and ArguAna and for qwen FiQA and ArguAna. Gemma SciFact is
borderline: the interval includes zero, the sign test does not. Recall@100 tells
the same story (FiQA: MaxSim 0.761 / 0.840 against dense 0.797 / 0.887, gemma /
qwen).

ArguAna follows the BEIR convention of ignoring the document whose id equals the
query id: each ArguAna query is itself an argument in the corpus, and would
otherwise rank first under every objective. The vector bench and the Pass-1
study do not apply that rule, so their ArguAna nDCG@10 figures are lower (gemma
0.4701) and are not comparable with this table.

## Why MaxSim alone falls behind

**For a one-chunk document the two objectives agree.** Its only chunk covers its
whole text, and its chunk vector matches its dense vector (mean cosine at least
0.997 on every dataset and model; a handful of FiQA documents are lower, down to
0.90). Differences therefore come from how multi-chunk documents are scored.

**MaxSim favours documents with more chunks.** A document's MaxSim score is the
best of its chunks, so each extra chunk is another chance at a high score. Mean
chunks per document:

| Model | Dataset | Corpus | One-chunk docs | Relevant docs | MaxSim top 10 | Dense top 10 |
|---|---|---:|---:|---:|---:|---:|
| gemma | SciFact | 4.40 | 2% | 4.38 | 4.49 | 4.31 |
| qwen | SciFact | 4.40 | 2% | 4.38 | 4.60 | 4.31 |
| gemma | NFCorpus | 4.82 | 2% | 4.81 | 5.06 | 4.86 |
| qwen | NFCorpus | 4.82 | 2% | 4.81 | 5.08 | 4.78 |
| gemma | FiQA | 3.00 | 41% | 4.11 | 5.67 | 3.66 |
| qwen | FiQA | 3.00 | 41% | 4.11 | 5.67 | 3.70 |
| gemma | ArguAna | 3.26 | 20% | 2.71 | 3.72 | 3.66 |
| qwen | ArguAna | 3.26 | 20% | 2.71 | 3.81 | 3.42 |

SciFact and NFCorpus are abstracts of similar length. MaxSim's top 10 is only
slightly longer than the relevant documents, and the objectives tie. FiQA's
answers range from one sentence to many paragraphs. MaxSim's top 10 averages 5.7
chunks against 4.1 for the relevant documents, and short relevant answers are
pushed down.

**ArguAna shows a second effect.** For gemma, MaxSim and dense pick documents of
nearly the same length (3.72 against 3.66 chunks), yet MaxSim still loses by
0.052. The task is to find the counter-argument to a whole argument. A
multi-chunk argument's best-matching passage is a weaker match for that than
the argument as a whole.

For each judged-relevant document: which objective ranks it higher, by the
document's chunk count (MaxSim higher / dense higher; ties not shown):

| Model | Dataset | 1 chunk | 2–3 | 4–7 | 8+ |
|---|---|---|---|---|---|
| gemma | SciFact | 0 / 4 | 17 / 29 | 25 / 31 | 4 / 6 |
| qwen | SciFact | 0 / 3 | 24 / 23 | 23 / 28 | 5 / 3 |
| gemma | NFCorpus | 14 / 219 | 733 / 1,717 | 3,819 / 4,720 | 412 / 313 |
| qwen | NFCorpus | 4 / 228 | 794 / 1,665 | 3,947 / 4,559 | 418 / 317 |
| gemma | FiQA | 23 / 348 | 96 / 339 | 131 / 256 | 98 / 86 |
| qwen | FiQA | 25 / 334 | 87 / 334 | 108 / 247 | 55 / 112 |
| gemma | ArguAna | 78 / 63 | 126 / 292 | 39 / 126 | 4 / 15 |
| qwen | ArguAna | 30 / 111 | 93 / 294 | 30 / 107 | 7 / 8 |

Both effects show here. One-chunk relevant documents are almost always ranked
higher by dense (FiQA 348 to 23): equal scores, but MaxSim lifts the long
documents above them. Only for the longest documents (8+ chunks) does MaxSim
sometimes come out ahead. On ArguAna, dense also wins for the 2–7-chunk
arguments that make up most of the relevant set.

## What this means for the design

- **Pass 2 stays.** It is the better final ranking on every dataset where the two
  objectives differ, and it is never significantly worse.
- **MaxSim stays as the candidate generator.** Its length bias matters for the
  order within the top 10, not for recall at 1,000: two-pass matches dense
  scoring of every document in all eight pairs.
- **Nothing to change in code.** The qwen SciFact result that prompted this was
  within noise.

## Caveats

- Eight pairs from four datasets, all English, and test splits only.
- Chunking is fixed at 4-sentence windows sliding by 2. Other window sizes would
  change MaxSim's length bias and are not measured here.
- The ArguAna explanation (best passage against whole argument) is an
  interpretation of the per-document counts, not a separate experiment.

## Reproduce

The frozen embeddings come from the vector bench (`vector_bench`, which dumps a
dataset on its first run; `service/scripts/fetch_beir.sh nfcorpus arguana`
downloads the two extra datasets). The analysis is a standalone Python script
over those files, kept in the gitignored `work/bench/analysis/`:

```sh
python work/bench/analysis/pass2_vs_maxsim.py scifact,fiqa,nfcorpus,arguana gemma,qwen out.json
python work/bench/analysis/chunk_bias.py scifact,fiqa,nfcorpus,arguana gemma,qwen
```

It needs `numpy`. Its SciFact and FiQA numbers equal the Rust Pass-1 study's
exact pipelines to four decimals (`pass1-recall-study.md`).
