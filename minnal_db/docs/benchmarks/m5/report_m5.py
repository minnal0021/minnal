"""M5 report: per mix, the steps, their validity, the sustainable rate, the soak,
the concurrency sweep and quality under load.

    python3 report_m5.py [RESULTS_DIR]   (default work/bench/results/m5)

A step **passes** when every class meets its p99 target, errors stay at or
under 0.1% and nothing was dropped; the **sustainable rate** is the highest
offered rate of a passing ramp or bisection step. The vector queue and
ingestion lag are reported per step but do not decide it (user decision,
2026-10-17: the benchmark measures the engine, and embedding is replayed, see
the assumptions the report prints first).

A step is **valid** when every background task did work during it:
LSM compaction ran, value-log GC ran or no bucket was past its 30% waste
trigger, every field index advanced its checkpoint, WAL GC deleted a segment or
no more than two were held, the vector worker completed work, and the
partition split at least once (or no posting was over its split limit). The
TTL worker has no counter and is not checked.
"""
import glob, json, math, os, sys
from m5_rules import queue_grew, timeline, validity

ASSUMPTIONS = """## Assumptions and constraints

Read these before any number below; each limits what the numbers mean.

1. **What is measured.** The minnal server alone: REST API, document store,
   field indices, LSM tree and value log, the main WAL, and the vector index
   (quantisation, vector index WAL, partition splits and merges, search), on
   one host. The load generator runs on the same host.
2. **No embedding service.** Every embedding request the server makes is
   answered by `embed_replay.py` from a table recorded once from the real
   service ({replay}). The GPU service, its batching and its throughput are not
   in any number here. Writes never wait for embedding (they are acknowledged
   when stored), and searches use cached query embeddings, so the replay is in
   no request's latency; it is in indexing time and ingestion lag only.
3. **A finite set of texts.** So that the table covers every text, inserts use
   the {stream} stream documents and then the same texts with `" (copy)"`
   appended under new ids, and text-changing updates append `" Revised."`.
   From the second pass over the stream, inserted texts repeat exactly, and an
   update of an already-revised document resends the same text: the engine
   still embeds, writes and indexes each one.
4. **Searches use cached query embeddings** ({queries} FiQA test queries; the
   cache is primed before the warm-up). What a query that misses the cache
   costs is not measured.
5. **The sustainable rate does not include indexing.** It is the highest
   offered rate at which every class meets its p99 target, errors stay at or
   under 0.1% and nothing is dropped. The vector queue and ingestion lag are
   reported per step but do not decide it: at a sustainable rate whose queue
   grows, new documents become searchable later and later. Read the queue and
   lag columns with it.
6. **Indexing throughput is the server's vector worker's**
   (`[vector_index] concurrency` = {concurrency} documents at a time).
7. **Data.** FiQA: a base of 40,000 documents, embedded with the real service,
   with its partition settled, copied fresh for each mix. The fields
   `category`, `year` and `score` are synthetic.
8. **Host.** {cores} cores: the server is pinned to cores {server_cores}, the
   load generator and the replay to cores {client_cores}. One machine, so
   network latency is a loopback's.
9. **Step lengths.** Ramp and bisection steps {step_s} s, warm-up {warm_s} s,
   soak {soak_s} s, sweep steps {sweep_s} s. p99 targets (ms): {slo}.
10. **Replay misses.** A payload missing from the table fails its embed
    request; the driver stops a run at the first miss, so a complete report
    has none (column `replay miss`).
"""

R = sys.argv[1] if len(sys.argv) > 1 else "/home/sperumal/minnal_dev/minnal/work/bench/results/m5"
QRELS = "/home/sperumal/minnal_dev/minnal/work/bench/m5/data/qrels.tsv"
CUTS = [10, 20, 30, 40, 50, 100]
CLASSES = ["get", "insert", "update", "delete", "scan", "field_query", "search", "filtered_search"]

qrels = {}
for line in list(open(QRELS))[1:]:
    q, d, s = line.split()
    if int(s) > 0:
        qrels.setdefault(q, set()).add(d)


def ndcg(trec_path, only=None):
    runs = {}
    for line in open(trec_path):
        q, _, d, rank, *_ = line.split()
        runs.setdefault(q, []).append((int(rank), d))
    out = {}
    for k in CUTS:
        vals = []
        for q, rel in qrels.items():
            if q not in runs or (only is not None and q not in only):
                continue
            docs = [d for _, d in sorted(runs[q])][:k]
            dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(docs) if d in rel)
            idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(rel))))
            vals.append(dcg / idcg)
        out[k] = (sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)
    return out


def resources(rows):
    r = [x["resources"] for x in rows if x.get("resources", {}).get("cpu_ticks") is not None]
    if len(r) < 2:
        return {}
    dt = rows[-1]["t"] - rows[0]["t"]
    cpu = (r[-1]["cpu_ticks"] - r[0]["cpu_ticks"]) / 100 / max(dt, 1e-9)
    wb = (r[-1].get("write_bytes") or 0) - (r[0].get("write_bytes") or 0)
    gpu = [x.get("gpu_busy_percent") for x in r if x.get("gpu_busy_percent") is not None]
    return {"cpu_cores": cpu, "rss_mb_end": (r[-1].get("rss_kb") or 0) / 1024, "disk_write_mb": wb / 2**20,
            "gpu_busy_mean": sum(gpu) / len(gpu) if gpu else None, "open_files_end": r[-1].get("open_files")}


printed_assumptions = False
for mix_dir in sorted(glob.glob(f"{R}/*/")):
    mix = os.path.basename(mix_dir.rstrip("/"))
    run = json.load(open(f"{mix_dir}/run.json")) if os.path.exists(f"{mix_dir}/run.json") else {}
    if not printed_assumptions and run.get("replay"):
        r = run["replay"]
        print(ASSUMPTIONS.format(
            replay=f"{r['vectors']:,} vectors, {r['model']}, {r['table_mib']:.0f} MiB, `{r['table']}`",
            stream=f"{run.get('stream_docs', '?'):,}", queries=run.get("queries", "?"),
            concurrency=run.get("vector_worker_concurrency", "?"), cores=run.get("host_cores", "?"),
            server_cores=run.get("server_cores", "?"), client_cores=run.get("client_cores", "?"),
            step_s=run.get("step_s"), warm_s=run.get("warm_s"), soak_s=run.get("soak_s"), sweep_s=run.get("sweep_s"),
            slo=run.get("slo_p99_ms")))
        printed_assumptions = True
    slo = run.get("slo_p99_ms", {})
    rest = ndcg(f"{mix_dir}/rest.trec") if os.path.exists(f"{mix_dir}/rest.trec") else None
    print(f"\n# {mix}  (commit {run.get('commit')}, weights {run.get('weights')})")
    print("\n| step | offered | completed | errors | p99 ms: " + " / ".join(CLASSES) + " | queue | lag p50 / p99 s | replay miss | valid | result |")
    print("|---|---:|---:|---:|---|---|---|---:|---|---|")
    best = None
    for d in sorted(glob.glob(f"{mix_dir}/*/summary.json"), key=os.path.getmtime):
        name = os.path.basename(os.path.dirname(d))
        s = json.load(open(d))
        snaps, rows = timeline(os.path.join(os.path.dirname(d), "timeline.jsonl"))
        v = validity(snaps)
        miss = [c for c in CLASSES if s["classes"][c]["count"] and slo and s["classes"][c]["p99_ms"] > slo[c]]
        passed = not miss and s["error_rate"] <= 0.001 and s["dropped"] == 0
        grew = queue_grew(snaps)
        replay_miss = (s.get("replay") or {}).get("misses", "?")
        if s["mode"] == "open" and name.startswith(("step", "bisect")) and passed:
            best = max(best or 0, s["offered_rate"])
        p99 = " / ".join(("**%.1f**" if c in miss else "%.1f") % s["classes"][c]["p99_ms"] for c in CLASSES)
        load = f"{s['offered_rate']:.0f}/s" if s["mode"] == "open" else f"{s['clients']} clients"
        lag = s["ingestion_lag"]
        print(f"| {name} | {load} | {s['completed_rate']:.1f}/s | {s['error_rate']:.3%} | {p99} | "
              f"{'grows' if grew else 'steady'} | {lag['p50_ms']/1e3:.1f} / {lag['p99_ms']/1e3:.1f} | {replay_miss} | "
              f"{'yes' if all(v.values()) else 'NO: ' + ', '.join(k for k, ok in v.items() if not ok)} | "
              f"{'passes' if passed else 'fails'} |")
    print(f"\n**Sustainable rate:** {best} req/s" if best else "\n**Sustainable rate:** none of the steps")
    soak = glob.glob(f"{mix_dir}/soak-*/")
    if soak:
        snaps, rows = timeline(f"{soak[0]}/timeline.jsonl")
        third = max(1, len(snaps) // 3)
        def lvl(f):
            vals = [f(s) for s in snaps]
            return vals[third], vals[-1]
        l0 = lvl(lambda s: sum(sum(lv.get("total_entries", 0) for lv in ns["levels"] if lv.get("level") == 0) for ns in s.get("lsm") or []))
        wal = lvl(lambda s: (s.get("wal") or {}).get("live_segments", 0))
        waste = lvl(lambda s: max((ns.get("waste_ratio_pct", 0) for ns in s.get("value_log") or []), default=0))
        rss = lvl(lambda s: (s["resources"].get("rss_kb") or 0) / 1024)
        part = lvl(lambda s: (s.get("partition") or {}).get("size_p10_p50_p90_max", [0, 0, 0, 0])[3])
        print(f"\n**Soak** ({os.path.basename(soak[0].rstrip('/'))}), value at a third / at the end: "
              f"L0 entries {l0[0]} / {l0[1]}; WAL segments {wal[0]} / {wal[1]}; worst value-log waste {waste[0]:.1f}% / {waste[1]:.1f}%; "
              f"RSS {rss[0]:.0f} / {rss[1]:.0f} MB; largest posting {part[0]} / {part[1]} chunks")
        print(f"  resources over the soak: {resources(rows)}")
    if rest:
        print("\n**Quality** nDCG@" + "/".join(map(str, CUTS)) + f" at rest: " + " / ".join(f"{rest[k][0]:.4f}" for k in CUTS) + f" ({rest[10][1]} queries)")
        for d in sorted(glob.glob(f"{mix_dir}/*/search.trec"), key=os.path.getmtime):
            name = os.path.basename(os.path.dirname(d))
            if os.path.getsize(d) == 0:
                continue
            asked = {line.split()[0] for line in open(d)}
            q, r = ndcg(d), ndcg(f"{mix_dir}/rest.trec", asked)  # the same queries at rest
            diff = [q[k][0] - r[k][0] for k in CUTS]
            flag = "" if min(diff) >= -0.005 else "  **below at-rest by more than 0.005**"
            print(f"  {name}: " + " / ".join(f"{x:+.4f}" for x in diff) + f" ({q[10][1]} queries){flag}")
