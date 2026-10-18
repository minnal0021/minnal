"""Report for an engine_bench run (minnal_db/benches/engine_bench).

    python3 report.py RUN_DIR [QRELS]

Writes RUN_DIR/report.md. QRELS defaults to the run's `data/qrels.tsv`.

Repeats of a cell are summarised as the median of their percentiles, with the
lowest and highest repeat beside it, so run-to-run spread is visible.
"""
import json
import math
import os
import statistics
import sys
from collections import defaultdict

CLASSES = ["get", "insert", "update", "delete", "scan", "field_query", "search", "filtered_search"]
PCTS = [("p50", "p50_ms"), ("p75", "p75_ms"), ("p90", "p90_ms"), ("p95", "p95_ms"), ("p99", "p99_ms"), ("p99.9", "p999_ms")]
CUTOFFS = [10, 20, 30, 40, 50, 100]

run_dir = sys.argv[1]
run = json.load(open(f"{run_dir}/run.json"))
manifest = json.load(open(f"{run_dir}/manifest.json"))
cfg = run["config"]
qrels_path = sys.argv[2] if len(sys.argv) > 2 else f"{cfg['data']}/qrels.tsv"

cells = {}
for f in sorted(os.listdir(f"{run_dir}/cells")):
    if f.endswith(".json"):
        c = json.load(open(f"{run_dir}/cells/{f}"))
        cells[c["cell"]["name"]] = c


def fmt(x, d=2):
    return "—" if x is None else f"{x:.{d}f}"


def spread(values, d=2):
    """median (min–max) of repeats."""
    v = [x for x in values if x is not None]
    if not v:
        return "—"
    if len(v) == 1:
        return fmt(v[0], d)
    return f"{fmt(statistics.median(v), d)} ({fmt(min(v), d)}–{fmt(max(v), d)})"


def repeats(prefix):
    return [c for n, c in sorted(cells.items()) if n.startswith(prefix)]


# ── quality ──────────────────────────────────────────────────────────────────
qrels = defaultdict(dict)
for i, line in enumerate(open(qrels_path)):
    p = line.rstrip("\n").split("\t")
    if i == 0 and p[0] == "query-id":
        continue
    if len(p) >= 3 and int(p[2]) > 0:
        qrels[p[0]][p[1]] = int(p[2])


def read_trec(path):
    runs = defaultdict(list)
    for line in open(path):
        q, _, d, rank, score, _ = line.split()
        runs[q].append((int(rank), d))
    return {q: [d for _, d in sorted(v)] for q, v in runs.items()}


def ndcg(ranked, rel, k):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def quality(trec):
    qs = [q for q in trec if q in qrels]
    if not qs:
        return None
    out = {f"nDCG@{k}": statistics.mean(ndcg(trec[q], qrels[q], k) for q in qs) for k in CUTOFFS}
    out["recall@100"] = statistics.mean(len(set(trec[q][:100]) & set(qrels[q])) / len(qrels[q]) for q in qs)
    out["queries"] = len(qs)
    return out


lines = []
w = lines.append
smoke = run["smoke"]
w(f"# Engine benchmark — {cfg['namespace']}{' (SMOKE RUN: durations too short for conclusions)' if smoke else ''}\n")
allocator = run.get("allocator", "system")
w(f"Commit `{run['commit']}`; allocator {allocator}; {run['host_cores']} cores; started "
  + ", resumed ".join(str(s) for s in manifest["starts"]) + " (Unix s).\n")

done = [n for n, s in manifest["cells"].items() if s == "done"]
invalid = [n for n, s in manifest["cells"].items() if s != "done"]
w(f"Cells: {len(done)} valid of {len(run['plan'])} planned"
  + (f"; **invalid: {', '.join(invalid)}**" if invalid else "")
  + ("; run complete." if manifest.get("complete") else "; **run not complete**.") + "\n")

# ── assumptions ──────────────────────────────────────────────────────────────
e, d, r = run["engine"], run["data"], run["replay"]
w("## Assumptions and constraints\n")
w("What these numbers are, and what they are not. Every value below is read from what the run recorded.\n")
w(f"- **The engine only, in process.** The document store is opened inside the benchmark process (no REST, no HTTP "
  f"serialisation, no network), with the same configuration as the API server (`{os.path.basename(e['server_toml'])}`). "
  "Operations call the library directly: `get`, `put`, `delete`, `scan_range`, `query_resolved`, `search_semantic`, "
  "`search_semantic_filtered` — the calls the REST handlers make.")
w("- **Every background worker runs**: memtable flushes, LSM compaction, value-log GC, WAL GC, vector index WAL GC, "
  "field-index checkpoints (every 1.75 s), the vector-index worker "
  f"({e['vector_worker_concurrency']} documents at a time) and partition maintenance ({e['maintenance_threads']} threads).")
w(f"- **No external call: embeddings are replayed.** The engine's embedding client talks to a replay inside the process "
  f"over localhost, which answers from a table recorded once from the real service ({r['vectors']:,} vectors, models "
  f"{', '.join(r['models'])}, {r['table_bytes'] / 2**20:.0f} MiB). The engine still does all its own work with each "
  "embedding (chunking, quantisation, the vector index WAL, routing, splits, merges). The embedding service's own cost "
  "and throughput are not measured. Every cell is checked for replay misses; a miss makes it invalid.")
w("- **The set of texts is finite** so the table can cover it: inserts past the end of the stream reuse its texts with "
  "a fixed `\" (copy)\"`, and text-changing updates append a fixed `\" Revised.\"`. The engine embeds and indexes each "
  "one regardless. Inserts and deletes are equally likely and deletes take the oldest documents, so the inserts in the "
  "store at once are fewer than the stream's texts and none is there twice (except briefly in a single-insert cell; "
  "counted per cell below).")
w("- **Searches use cached query embeddings.** Every test query is embedded once before the cells; what a query-embedding "
  "cache miss costs is the service's and is not measured.")
w("- **A search's latency is the search alone**: ranked ids and scores, top 100. The REST route also fetches each "
  "result's document, which is not included here.")
w(f"- **Single-operation cells**: one client sends one class back to back for {cfg['single_s']} s, while a trickle of "
  f"{cfg['trickle_rate']:g} writes/s ({', '.join(f'{k} {v:g}' for k, v in cfg['trickle_mix'].items())}) keeps indexing, "
  "flushes and maintenance working. The trickle's latencies are reported separately.")
w(f"- **Concurrent cells are closed loop**: N clients, each sending its next operation when the last returns, for "
  f"{cfg['concurrent_s']} s, drawing from the mix {', '.join(f'{k} {v:g}' for k, v in cfg['mix'].items())}. Latency is "
  "the operation's own time with N in flight; there is no arrival schedule, so time a request would have waited to be "
  "sent is not counted. Concurrency is measured at "
  f"{', '.join(map(str, cfg['clients']))} clients only; nothing is extrapolated beyond {max(cfg['clients'])}.")
w(f"- **Clients and engine share the host**: one process, {run['host_cores']} cores, one Tokio runtime; the clients do "
  "almost no work of their own.")
w(f"- **Every cell**: waits for the vector queue to drain (up to {cfg['drain_max_s']} s), runs {cfg['warm_s']} s "
  "unmeasured, measures, waits for the queue to drain again (so ingestion-lag samples complete), and is checked: no "
  f"failed operation, no replay miss, at least {cfg['min_samples']} operations per measured class, the queue drained, "
  "and if documents were written, the vector worker indexed and a field-index checkpoint advanced. An invalid cell "
  "stops the run.")
w("- **Repeats**: each cell runs once." if cfg['repeats'] == 1 else
  f"- **Repeats**: each cell runs {cfg['repeats']} times (all cells once, then all again); tables give the median of "
  "the repeats with the lowest and highest beside it.")
w(f"- **Data**: FiQA; {d['base']:,} base documents loaded and indexed in the snapshot ({d['stable']:,} never deleted), "
  f"an ingestion stream of {d['stream']:,}, {d['queries']} test queries. One dataset, synthetic `category`/`year`/`score` "
  "fields, one host.")
w(f"- **The store keeps its size**: before each cell, unmeasured writes bring it back to the {d['base']:,} documents it "
  "started with (a single-insert or single-delete cell moves it by thousands). One engine runs the whole plan, so later "
  "cells still run on a store that has seen more maintenance.\n")

# ── single ───────────────────────────────────────────────────────────────────
w("## One operation at a time\n")
w("One client, one class, with the write trickle running. Milliseconds; median of repeats (lowest–highest).\n")
w("| Class | ops/s | " + " | ".join(p for p, _ in PCTS) + " | max |")
w("|---|---:|" + "---:|" * (len(PCTS) + 1))
for c in CLASSES:
    reps = [x["classes"].get(c) for x in repeats(f"single-{c}-")]
    reps = [x for x in reps if x]
    if not reps:
        continue
    w(f"| `{c}` | {spread([x['ops_per_s'] for x in reps], 0)} | "
      + " | ".join(spread([x[k] for x in reps]) for _, k in PCTS) + f" | {spread([x['max_ms'] for x in reps], 1)} |")
w("")

# ── concurrent ───────────────────────────────────────────────────────────────
w("## Concurrent load\n")
w("Closed loop over the mix. Throughput is completed operations per second, all classes; "
  "Little's law check: clients ≈ throughput × mean latency.\n")
w("| Clients | ops/s | mean latency ms | throughput × mean |")
w("|---:|---:|---:|---:|")
for n in cfg["clients"]:
    reps = repeats(f"closed-{n:02d}-")
    if not reps:
        continue
    tput = [x["total_ops_per_s"] for x in reps]
    means = []
    for x in reps:
        cnt = sum(v["count"] for v in x["classes"].values())
        means.append(sum(v["mean_ms"] * v["count"] for v in x["classes"].values()) / cnt if cnt else None)
    little = [t * m / 1e3 for t, m in zip(tput, means) if m is not None]
    w(f"| {n} | {spread(tput, 0)} | {spread(means)} | {spread(little, 1)} |")
w("")
for c in CLASSES:
    if not any(x["classes"].get(c) for n in cfg["clients"] for x in repeats(f"closed-{n:02d}-")):
        continue
    w(f"**`{c}`** (ms)\n")
    w("| Clients | ops/s | " + " | ".join(p for p, _ in PCTS) + " |")
    w("|---:|---:|" + "---:|" * len(PCTS))
    for n in cfg["clients"]:
        reps = [x["classes"].get(c) for x in repeats(f"closed-{n:02d}-")]
        reps = [x for x in reps if x]
        if reps:
            w(f"| {n} | {spread([x['ops_per_s'] for x in reps], 0)} | " + " | ".join(spread([x[k] for x in reps]) for _, k in PCTS) + " |")
    w("")

MEMORY = [("rss_anon_bytes", "Resident heap and stacks"), ("rss_file_bytes", "Resident mapped files"),
          ("heap_live_bytes", "Allocated, not freed"), ("heap_free_bytes", "Freed, kept by the allocator")]
# The heap figures are glibc's own account (mallinfo2); another allocator reports nothing there.
if allocator != "system":
    MEMORY = MEMORY[:2]

# ── soak ─────────────────────────────────────────────────────────────────────
soak = repeats("soak-")
if soak:
    s = soak[0]
    secs = s["measured_s"]
    length = f"{secs / 60:.0f} minutes" if secs >= 120 else f"{secs:.0f} s"
    w(f"## Soak: {s['cell']['clients']} clients, {length}\n")
    w(f"Throughput {s['total_ops_per_s']:.0f} ops/s. p99 by class (ms): "
      + ", ".join(f"`{c}` {v['p99_ms']:.2f}" for c, v in s["classes"].items()) + ".\n")
    tl = [json.loads(l) for l in open(f"{run_dir}/cells/{s['cell']['name']}.timeline.jsonl")]
    if len(tl) >= 2:
        q = len(tl) // 4 or 1
        first, last = tl[:q], tl[-q:]
        avg = lambda xs, k: statistics.mean(x[k] for x in xs)
        w("| Over the soak | first quarter | last quarter |")
        w("|---|---:|---:|")
        w(f"| Disk (MiB) | {avg(first, 'disk_bytes') / 2**20:.0f} | {avg(last, 'disk_bytes') / 2**20:.0f} |")
        w(f"| Resident memory (MiB) | {avg(first, 'rss_bytes') / 2**20:.0f} | {avg(last, 'rss_bytes') / 2**20:.0f} |")
        w(f"| Vector queue | {avg(first, 'queue'):.0f} | {avg(last, 'queue'):.0f} |")
        for k, label in MEMORY:
            if k in tl[0]:
                w(f"| {label} (MiB) | {avg(first, k) / 2**20:.0f} | {avg(last, k) / 2**20:.0f} |")
        w("")
    if s.get("trim") and allocator == "system":
        b, a = s["trim"]
        w("After the soak the allocator was asked to return the memory it keeps free (`malloc_trim`): resident heap "
          f"{b['rss_anon_bytes'] / 2**20:.0f} → {a['rss_anon_bytes'] / 2**20:.0f} MiB; allocated and not freed "
          f"{b['heap_live_bytes'] / 2**20:.0f} MiB.\n")

# ── quality ──────────────────────────────────────────────────────────────────
w("## Search quality\n")
w("nDCG and recall of the top 100 for the FiQA test queries: at the start, under load (the first reply per query in "
  "each phase's `search` operations), and at rest after the last cell. The corpus changes during the run (deletes, "
  "text-changing updates, inserted copies), so later rows measure a different corpus, not only a busier index.\n")
rows = []
for name, path in [("start", "quality-start.trec"), ("at rest", "quality-rest.trec")]:
    if os.path.exists(f"{run_dir}/{path}"):
        rows.append((name, quality(read_trec(f"{run_dir}/{path}"))))
for label, prefix in [("under load: single", "single-search-"), ("under load: concurrent", "closed-"), ("under load: soak", "soak-")]:
    merged = {}
    for n in sorted(cells):
        if n.startswith(prefix) and os.path.exists(f"{run_dir}/cells/{n}.trec"):
            for q, docs in read_trec(f"{run_dir}/cells/{n}.trec").items():
                merged.setdefault(q, docs)
    if merged:
        rows.append((label, quality(merged)))
w("| | queries | " + " | ".join(f"nDCG@{k}" for k in CUTOFFS) + " | recall@100 |")
w("|---|---:|" + "---:|" * (len(CUTOFFS) + 1))
for name, qv in rows:
    if qv:
        w(f"| {name} | {qv['queries']} | " + " | ".join(f"{qv[f'nDCG@{k}']:.4f}" for k in CUTOFFS) + f" | {qv['recall@100']:.4f} |")
w("")

# ── background work and lag ──────────────────────────────────────────────────
w("## Background work and ingestion lag per cell\n")
w("Documents: in the store at the cell's start and end; rebalanced: unmeasured writes before the cell that brought it back "
  "to its starting size (+ inserted, − deleted); repeated texts: inserted texts in the store twice at the cell's end. "
  "Counts during the cell (warm-up, measurement and the drain after it), across all of the engine's namespaces: "
  "`puts` and `deletes` are key-value writes, so one document write is several (document, field index, vector "
  "index). `WAL GC` counts passes that reclaimed WAL segments; the worker ticks every interval, but a pass "
  "with no full segment to reclaim is not counted. Lag: from a write's acknowledgement to its "
  "document being searchable, sampled on every "
  f"{cfg['lag_every']}th insert of each client.\n")
w("| Cell | documents | rebalanced | repeated texts | puts | deletes | flushes | L0→L1 | vlog GC | WAL GC | vec WAL fsyncs | splits | merges | lag p50 ms | lag p99 ms | drain after s |")
w("|---|" + "---:|" * 15)
totals = defaultdict(int)
for n, x in sorted(cells.items(), key=lambda kv: kv[1]["started_unix"]):
    g = x["engine"]
    for k in ("memtable_flushes", "l0_l1_compactions", "vlog_gc_runs", "wal_gc_runs", "vector_wal_fsyncs", "splits"):
        totals[k] += g[k]
    lag = x["ingestion_lag_ms"]
    w(f"| {n} | {x.get('documents_before', '—')}→{x.get('documents_after', '—')} | {x.get('rebalanced', 0):+d} | "
      f"{x.get('repeated_texts', '—')} | {g['puts']} | {g['deletes']} | {g['memtable_flushes']} | {g['l0_l1_compactions']} | {g['vlog_gc_runs']} | "
      f"{g['wal_gc_runs']} | {g['vector_wal_fsyncs']} | {g['splits']} | {g['merges']} | {fmt(lag['p50'], 0)} | "
      f"{fmt(lag['p99'], 0)} | {fmt(x['drain_after_s'], 0)} |")
w("")
idle = [k for k, v in totals.items() if v == 0]
w("Over the whole run: " + ", ".join(f"{k} {v}" for k, v in totals.items()) + ".")
if idle:
    w(f"**Zero over the whole run: {', '.join(idle)}.** Numbers above do not include that work"
      + (" (for `wal_gc_runs`: no WAL segment filled, so there was nothing to reclaim)" if "wal_gc_runs" in idle else "")
      + ".")
w("")

ordered = sorted(cells.items(), key=lambda kv: kv[1]["started_unix"])
if ordered and "rss_anon_bytes" in ordered[0][1]["engine_after"]:
    w("## Memory\n")
    base_mem = run.get("memory_before_engine")
    w("At the end of each cell, MiB. Resident memory split into anonymous pages (heap, stacks) and memory-mapped "
      "files (which the kernel can drop)"
      + ("; and the allocator's own account: what is allocated and not freed (live data, "
         "or a leak if it only grows), and what was freed but is kept (resident until returned)."
         if allocator == "system" else f". The allocator's own account is glibc's and is not available with {allocator}.")
      + (f" Before the engine opened (the replay table alone): {base_mem['rss_anon_bytes'] / 2**20:.0f} MiB resident heap."
         if base_mem else "") + "\n")
    w("| Cell | documents | " + " | ".join(label for _, label in MEMORY) + " |")
    w("|---|" + "---:|" * (len(MEMORY) + 1))
    for n, x in ordered:
        g = x["engine_after"]
        w(f"| {n} | {x.get('documents_after', '—')} | " + " | ".join(f"{g[k] / 2**20:.0f}" for k, _ in MEMORY) + " |")
    w("")

errors = [(n, x["errors"]) for n, x in cells.items() if x["errors"]]
if errors:
    w("## Errors\n")
    for n, es in errors:
        w(f"- {n}: " + "; ".join(es))

open(f"{run_dir}/report.md", "w").write("\n".join(lines) + "\n")
print(f"{run_dir}/report.md")
