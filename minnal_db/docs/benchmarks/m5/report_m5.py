"""M5 report: per mix, the steps, their validity, the sustainable rate, the soak,
the concurrency sweep and quality under load.

    python3 report_m5.py [RESULTS_DIR]   (default work/bench/results/m5)

A step **passes** when every class meets its p99 target, errors stay at or
under 0.1% and nothing was dropped. It is **sustainable** when it also leaves
the vector queue no deeper than it found it (beyond 500 entries of noise): a
growing queue means writes are accepted faster than they are embedded.

A step is **valid** when every background task did work during it:
LSM compaction ran, value-log GC ran or no bucket was past its 30% waste
trigger, every field index advanced its checkpoint, WAL GC deleted a segment or
no more than two were held, the vector worker completed work, and the
partition split at least once (or no posting was over its split limit). The
TTL worker has no counter and is not checked.
"""
import glob, json, math, os, sys

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


def timeline(path):
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if "ops" in r], rows


def validity(snaps):
    if len(snaps) < 2:
        return {"too short": False}
    a, b = snaps[0], snaps[-1]
    ops = lambda s, *k: (lambda v: v)(s["ops"][k[0]][k[1]]) if s.get("ops") else 0
    checks = {}
    checks["lsm compaction"] = ops(b, "compaction", "l0_l1_compactions") > ops(a, "compaction", "l0_l1_compactions")
    waste = max((ns.get("waste_ratio_pct", 0) for ns in (b.get("value_log") or [])), default=0)
    checks["value-log gc"] = ops(b, "gc", "vlog_gc_runs") > ops(a, "gc", "vlog_gc_runs") or waste < 30
    fa = {f["field_name"]: f["checkpoint_offset"] for f in (a.get("index_health") or {}).get("fields", [])}
    fb = {f["field_name"]: f["checkpoint_offset"] for f in (b.get("index_health") or {}).get("fields", [])}
    checks["index checkpoints"] = bool(fb) and all(fb[k] > fa.get(k, -1) for k in fb)
    checks["wal gc"] = ops(b, "gc", "wal_segments_deleted") > ops(a, "gc", "wal_segments_deleted") or (b.get("wal") or {}).get("live_segments", 99) <= 2
    qa = [s["queue"]["total_pending"] for s in snaps if s.get("queue")]
    worked = any(y < x for x, y in zip(qa, qa[1:])) or (qa and max(qa) == 0)
    checks["vector worker"] = bool(worked)
    pa, pb = a.get("partition") or {}, b.get("partition") or {}
    split_limit = pb.get("split_limit", 1 << 62)
    over = pb.get("size_p10_p50_p90_max", [0, 0, 0, 0])[3] > split_limit
    checks["partition splits"] = (pb.get("maintenance", {}).get("splits", 0) > pa.get("maintenance", {}).get("splits", 0)) or not over
    return checks


def queue_grew(snaps):
    q = [s["queue"]["total_pending"] for s in snaps if s.get("queue")]
    return bool(q) and q[-1] > q[0] + 500


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


for mix_dir in sorted(glob.glob(f"{R}/*/")):
    mix = os.path.basename(mix_dir.rstrip("/"))
    run = json.load(open(f"{mix_dir}/run.json")) if os.path.exists(f"{mix_dir}/run.json") else {}
    slo = run.get("slo_p99_ms", {})
    rest = ndcg(f"{mix_dir}/rest.trec") if os.path.exists(f"{mix_dir}/rest.trec") else None
    print(f"\n# {mix}  (commit {run.get('commit')}, weights {run.get('weights')})")
    print("\n| step | offered | completed | errors | p99 ms: " + " / ".join(CLASSES) + " | queue | lag p50 / p99 s | valid | result |")
    print("|---|---:|---:|---:|---|---|---|---|---|")
    best = None
    for d in sorted(glob.glob(f"{mix_dir}/*/summary.json"), key=os.path.getmtime):
        name = os.path.basename(os.path.dirname(d))
        s = json.load(open(d))
        snaps, rows = timeline(os.path.join(os.path.dirname(d), "timeline.jsonl"))
        v = validity(snaps)
        miss = [c for c in CLASSES if s["classes"][c]["count"] and slo and s["classes"][c]["p99_ms"] > slo[c]]
        passed = not miss and s["error_rate"] <= 0.001 and s["dropped"] == 0
        grew = queue_grew(snaps)
        sustainable = passed and not grew
        if s["mode"] == "open" and name.startswith(("step", "bisect")) and sustainable:
            best = max(best or 0, s["offered_rate"])
        p99 = " / ".join(("**%.1f**" if c in miss else "%.1f") % s["classes"][c]["p99_ms"] for c in CLASSES)
        load = f"{s['offered_rate']:.0f}/s" if s["mode"] == "open" else f"{s['clients']} clients"
        lag = s["ingestion_lag"]
        print(f"| {name} | {load} | {s['completed_rate']:.1f}/s | {s['error_rate']:.3%} | {p99} | "
              f"{'grows' if grew else 'steady'} | {lag['p50_ms']/1e3:.1f} / {lag['p99_ms']/1e3:.1f} | "
              f"{'yes' if all(v.values()) else 'NO: ' + ', '.join(k for k, ok in v.items() if not ok)} | "
              f"{'sustainable' if sustainable else ('passes, queue grows' if passed else 'fails')} |")
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
