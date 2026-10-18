"""Engine benchmark data (minnal_db/benches/engine_bench): FiQA (57,638 documents, gemma) with three synthetic indexed fields,
split so that quality under load stays comparable with quality at rest.

  base.jsonl    40,000 documents loaded and indexed before a run. Every document
                any test query judges relevant is here.
  stream.jsonl  the remaining 17,638: the ingestion stream (none relevant).
  ids.json      stable (gets, updates, scans), deletable (base documents no
                test query judges relevant and that updates never touch).
  queries.jsonl the 648 FiQA test queries; qrels.tsv their judgements.

Fields, seeded per document id: category c0..c11 (Zipf s = 1.1), year
2010-2025 (uniform), score 0-1000 (uniform).

    python3 prepare.py FIQA_DIR OUT_DIR

FIQA_DIR is the BEIR FiQA download (corpus.jsonl, queries.jsonl, qrels/test.tsv).

The starting database is base.jsonl loaded into a store created from
schema.json (gemma, three indexed fields), with the vector queue drained.
"""
import json, os, random, sys

if len(sys.argv) != 3:
    sys.exit(__doc__)
SRC, OUT = sys.argv[1], sys.argv[2]
BASE = 40_000
SEED = 5

os.makedirs(OUT, exist_ok=True)
rng = random.Random(SEED)
docs = [json.loads(l) for l in open(f"{SRC}/corpus.jsonl") if l.strip()]
qrels = [l.rstrip("\n").split("\t") for l in open(f"{SRC}/qrels/test.tsv")][1:]
relevant = {int(r[1]) for r in qrels}
test_q = {r[0] for r in qrels}

weights = [1 / (k + 1) ** 1.1 for k in range(12)]


def fields(i):
    r = random.Random(i * 0x9E3779B1 ^ SEED)
    return {
        "category": f"c{r.choices(range(12), weights)[0]}",
        "year": r.randint(2010, 2025),
        "score": r.randint(0, 1000),
    }


def body(d):
    t, x = d.get("title", ""), d.get("text", "")
    return (f"{t}. {x}" if t else x)[:4000]


ids = [int(d["_id"]) for d in docs]
by_id = {int(d["_id"]): d for d in docs}
rel = [i for i in ids if i in relevant]
rest = [i for i in ids if i not in relevant]
rng.shuffle(rest)
base_ids = sorted(rel + rest[: BASE - len(rel)])
stream_ids = rest[BASE - len(rel):]  # shuffled order = arrival order
non_rel_base = [i for i in base_ids if i not in relevant]
rng.shuffle(non_rel_base)
deletable = sorted(non_rel_base[:10_000])
stable = sorted(set(base_ids) - set(deletable))

with open(f"{OUT}/base.jsonl", "w") as f:
    for i in base_ids:
        f.write(json.dumps({"id": i, "text": body(by_id[i]), **fields(i)}) + "\n")
with open(f"{OUT}/stream.jsonl", "w") as f:
    for i in stream_ids:
        f.write(json.dumps({"id": i, "text": body(by_id[i]), **fields(i)}) + "\n")
json.dump({"stable": stable, "deletable": deletable, "relevant": sorted(relevant & set(base_ids))},
          open(f"{OUT}/ids.json", "w"))
with open(f"{OUT}/queries.jsonl", "w") as f:
    for l in open(f"{SRC}/queries.jsonl"):
        q = json.loads(l)
        if q["_id"] in test_q:
            f.write(json.dumps({"_id": q["_id"], "text": q["text"]}) + "\n")
with open(f"{OUT}/qrels.tsv", "w") as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for r in qrels:
        f.write("\t".join(r) + "\n")
print(f"base {len(base_ids)} ({len(rel)} relevant), stream {len(stream_ids)}, stable {len(stable)}, "
      f"deletable {len(deletable)}, queries {len(test_q)} -> {OUT}")
