"""M5 driver: one mix, end to end (design doc *M5 — System benchmark under load*).

    python3 run_m5.py MIX [--quick]

1. Copy the base snapshot (40,000 FiQA documents, indexed) to a fresh run
   directory and start the release server on cores 4-31; the load generator
   runs on cores 0-3.
2. Prime the query-embedding cache (every test query once, which also records
   quality at the start), then warm up: 5 minutes at 25% of the first rate (not
   measured). Searches in the steps use cached queries only, so they measure
   minnal, not the embedding service (user decision, 2026-10-11): a query that
   misses the cache is embedded by the same service as the ingested documents
   and waits behind them.
3. Steps: 10 minutes each from 50 req/s, x1.5 per step, until a class misses its
   p99 target or errors pass 0.1%; then two bisection steps between the last
   passing and the first failing rate. Before each step the vector queue is
   allowed to drain (up to 20 minutes; how long it took is recorded).
4. Soak: 60 minutes at the best passing rate.
5. Concurrency sweep (closed loop): 1, 2, 4, ..., 64 clients, 5 minutes each.
6. Cold-query penalty: 10 minutes at the sustainable rate with 30% of searches
   missing the cache.
7. Quality at rest: every test query once, alone, against the final state.

Every step writes work/bench/results/m5/{mix}/{name}/ (summary.json,
timeline.jsonl, search.trec, lag.json). `report_m5.py` turns them into tables.
`--quick` shortens every phase (for checking the harness, not for numbers).
"""
import json, os, shutil, signal, subprocess, sys, time, urllib.request

ROOT = "/home/sperumal/minnal_dev/minnal"
M5 = f"{ROOT}/work/bench/m5"
BIN = f"{M5}/bin"
DATA = f"{M5}/data"
URL = "http://127.0.0.1:8095"
NS = "fiqa"

MIXES = {
    "read_heavy": dict(get=40, insert=3, update=4, delete=1, scan=5, field_query=15, search=25, filtered_search=7),
    "balanced": dict(get=25, insert=10, update=10, delete=3, scan=5, field_query=15, search=25, filtered_search=7),
    "ingest_heavy": dict(get=15, insert=30, update=15, delete=5, scan=5, field_query=10, search=15, filtered_search=5),
}
# p99 latency targets (ms), from scheduled time. Writes are bounded below by
# their WAL fsyncs (~2.3 ms each on this host, one WAL lock for the server):
# insert/update ~2 before the reply, delete ~8. So their targets are ~2-3x the
# p99 one client sees at low load (user decision, 2026-10-11): insert/update
# 20-23 ms, delete 35 ms (~120 ms while maintenance runs), scan ~26 ms at
# 50 req/s. The design's starting values (20 ms for every write and scan)
# failed at every rate.
SLO = dict(get=5, insert=50, update=50, delete=150, scan=50, field_query=20, search=100, filtered_search=100)
MAX_ERROR_RATE = 0.001

mix = sys.argv[1]
quick = "--quick" in sys.argv
STEP_S = 60 if quick else 600
WARM_S = 30 if quick else 300
SOAK_S = 120 if quick else 3600
SWEEP_S = 30 if quick else 300
SWEEP = [1, 4, 16] if quick else [1, 2, 4, 8, 16, 32, 64]
FIRST_RATE = 50.0
OUT = f"{ROOT}/work/bench/results/m5{'-quick' if quick else ''}/{mix}"
RUN = f"{M5}/run-{mix}{'-quick' if quick else ''}"


def log(*a):
    line = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(f"{OUT}/driver.log", "a") as f:
        f.write(line + "\n")


def get(path):
    with urllib.request.urlopen(URL + path, timeout=60) as r:
        return json.loads(r.read())


def queue_depth():
    return get("/admin/indices/vector/queue/summary")["total_pending"]


def start_server():
    cfg = open(f"{M5}/server-base.toml").read().replace(f"{M5}/base/", f"{RUN}/")
    open(f"{RUN}/server.toml", "w").write(cfg)
    out = open(f"{RUN}/server.out", "a")
    p = subprocess.Popen(["taskset", "-c", "4-31", f"{BIN}/minnal_db_api", f"{RUN}/server.toml"], stdout=out, stderr=subprocess.STDOUT)
    for _ in range(600):
        try:
            get("/stores")
            return p
        except Exception:
            time.sleep(0.5)
    raise SystemExit("server did not start")


def drain(limit_s=1200):
    t = time.time()
    while time.time() - t < limit_s:
        if queue_depth() == 0:
            return time.time() - t
        time.sleep(5)
    return None


seed_counter = [1000]


def step(name, rate=None, closed=None, duration=STEP_S, cold=0.0):
    seed_counter[0] += 1
    out = f"{OUT}/{name}"
    spec = dict(url=URL, namespace=NS, data_dir=DATA, out_dir=out, mix=MIXES[mix], duration_s=duration,
                seed=seed_counter[0], state_file=f"{OUT}/state.json", server_pid=server.pid,
                poll_interval_s=5, lag_sample_every=20, search_cold_share=cold)
    if closed:
        spec["closed"] = closed
    else:
        spec["rate"] = rate
    os.makedirs(out, exist_ok=True)
    json.dump(spec, open(f"{out}/step.json", "w"), indent=1)
    waited = drain()
    log(f"start {name}: rate {rate} clients {closed}; queue drained in {waited if waited is None else round(waited)} s")
    r = subprocess.run(["taskset", "-c", "0-3", f"{BIN}/minnal_tools", "mixed_load", f"{out}/step.json"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log(f"step {name} failed: {r.stderr[-500:]}")
        raise SystemExit(1)
    s = json.load(open(f"{out}/summary.json"))
    s["queue_drain_s"] = waited
    json.dump(s, open(f"{out}/summary.json", "w"), indent=1)
    misses = {c: s["classes"][c]["p99_ms"] for c in SLO
              if s["classes"][c]["count"] and s["classes"][c]["p99_ms"] > SLO[c]}
    ok = not misses and s["error_rate"] <= MAX_ERROR_RATE and s["dropped"] == 0
    log(f"done  {name}: completed {s['completed_rate']:.1f}/s, errors {s['error_rate']:.4%}, dropped {s['dropped']}, "
        f"late max {s['dispatcher_late_max_ms']:.1f} ms, lag p50 {s['ingestion_lag']['p50_ms']:.0f} ms, "
        f"queue end {queue_depth()}; {'PASS' if ok else 'FAIL ' + json.dumps(misses)}")
    return ok


shutil.rmtree(OUT, ignore_errors=True)  # a run starts fresh, cursors included
os.makedirs(OUT)
if os.path.exists(RUN):
    shutil.rmtree(RUN)
os.makedirs(RUN)
log(f"copy base snapshot -> {RUN}")
for d in ("db", "schemas", "log"):
    if os.path.exists(f"{M5}/base/{d}"):
        shutil.copytree(f"{M5}/base/{d}", f"{RUN}/{d}")
server = start_server()
json.dump(dict(mix=mix, weights=MIXES[mix], slo_p99_ms=SLO, commit=open(f"{BIN}/COMMIT").read().strip(),
               step_s=STEP_S, quick=quick, host_cores=os.cpu_count()), open(f"{OUT}/run.json", "w"), indent=1)
def quality_pass(name):
    subprocess.run(["taskset", "-c", "0-3", f"{BIN}/minnal_tools", "search_load", "-c", "1", "--top-k", "100",
                    "--record", f"{OUT}/{name}.trec", URL, NS, f"{DATA}/queries.jsonl"], check=True,
                   stdout=open(f"{OUT}/{name}.out", "w"), stderr=subprocess.STDOUT)


try:
    log("drain after start:", drain())
    quality_pass("start")
    log("query cache primed; quality at start recorded")
    step("warmup", rate=FIRST_RATE * 0.25, duration=WARM_S)
    rate, passing, failing, k = FIRST_RATE, None, None, 0
    while True:
        k += 1
        if step(f"step-{k:02d}-{rate:.0f}", rate=rate):
            passing = rate
            rate *= 1.5
        else:
            failing = rate
            break
        if rate > 20_000:
            break
    if passing and failing:
        for b in range(2):
            mid = (passing + failing) / 2
            if step(f"bisect-{b + 1}-{mid:.0f}", rate=mid):
                passing = mid
            else:
                failing = mid
    log(f"best passing rate {passing}, first failing {failing}")
    json.dump(dict(passing=passing, failing=failing), open(f"{OUT}/rates.json", "w"))
    if passing:
        step(f"soak-{passing:.0f}", rate=passing, duration=SOAK_S)
    for n in SWEEP:
        step(f"closed-{n:02d}", closed=n, duration=SWEEP_S)
    if passing:
        step(f"cold-penalty-{passing:.0f}", rate=passing, cold=0.3)
    # Quality at rest: every query once, one client, nothing else running.
    log("drain before quality at rest:", drain(3600))
    quality_pass("rest")
    log("quality at rest recorded")
finally:
    server.send_signal(signal.SIGTERM)
    server.wait(timeout=300)
    log("server stopped")
