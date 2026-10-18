#!/usr/bin/env python3
"""nDCG and recall for `engine_bench quality` output.

    quality.py QRELS OUT_DIR [OUT_DIR ...]

Prints one table row per TREC file (pass) in each OUT_DIR.
"""
import glob
import math
import os
import statistics
import sys
from collections import defaultdict

CUTOFFS = [10, 20, 30, 40, 50, 100]
ORDER = ["before-default", "before-all", "before-all-wide", "rebuilt-default", "rebuilt-all", "rebuilt-all-wide"]

qrels = defaultdict(dict)
for i, line in enumerate(open(sys.argv[1])):
    p = line.rstrip("\n").split("\t")
    if i == 0 and p[0] == "query-id":
        continue
    if len(p) >= 3 and int(p[2]) > 0:
        qrels[p[0]][p[1]] = int(p[2])


def read_trec(path):
    runs = defaultdict(list)
    for line in open(path):
        q, _, d, rank, _, _ = line.split()
        runs[q].append((int(rank), d))
    return {q: [d for _, d in sorted(v)] for q, v in runs.items()}


def ndcg(ranked, rel, k):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(sorted(rel.values(), reverse=True)[:k]))
    return dcg / idcg if idcg else 0.0


print("| database | pass | queries | " + " | ".join(f"nDCG@{k}" for k in CUTOFFS) + " | recall@100 |")
print("|---|---|---:|" + "---:|" * (len(CUTOFFS) + 1))
for out in sys.argv[2:]:
    files = {os.path.basename(f)[:-5]: f for f in glob.glob(f"{out}/*.trec")}
    for name in [n for n in ORDER if n in files] + sorted(set(files) - set(ORDER)):
        trec = read_trec(files[name])
        qs = [q for q in qrels if q in trec] + [q for q in qrels if q not in trec]  # a missing query scores 0
        cells = [statistics.mean(ndcg(trec.get(q, []), qrels[q], k) for q in qs) for k in CUTOFFS]
        recall = statistics.mean(len(set(trec.get(q, [])[:100]) & set(qrels[q])) / len(qrels[q]) for q in qs)
        print(f"| {os.path.basename(out.rstrip('/'))} | {name} | {len(qs)} | "
              + " | ".join(f"{c:.4f}" for c in cells) + f" | {recall:.4f} |")
