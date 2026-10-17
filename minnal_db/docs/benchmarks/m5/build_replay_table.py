"""Build M5's embedding replay table (embed_replay.py) once, from the real
embedding service, by making the real engine embed every text a run can send.

    python3 build_replay_table.py [TABLE]

The engine itself turns each document into the payloads it sends (whole text
plus its sentence windows), so recording what it sends, rather than
re-deriving the payloads here, makes the table match a run byte for byte. On a
scratch copy of the base snapshot, with embed_replay.py in record mode in front
of the service, it:

1. starts the server (its startup probe of the service is recorded too);
2. runs every test query once (the query endpoint);
3. rewrites every stable document with its own text (score-only updates
   re-embed it unchanged);
4. rewrites every stable document no test query judges relevant with
   `" Revised."` appended (text-changing updates);
5. inserts every stream document (inserts);
6. inserts every stream text with `" (copy)"` appended under new ids (inserts
   past the end of the stream);

and waits for the vector queue to drain after each phase. Those are all the
texts mixed_load sends (see its `insert` and `update` classes); a replay run
counts any payload missing from the table, and the report treats a run with
misses as invalid.
"""
import concurrent.futures
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = "/home/sperumal/minnal_dev/minnal"
M5 = f"{ROOT}/work/bench/m5"
HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("M5_BIN", f"{M5}/bin")
DATA = f"{M5}/data"
TABLE = sys.argv[1] if len(sys.argv) > 1 else f"{M5}/embed-replay.table"
UPSTREAM = os.environ.get("EMBEDDING_URL", "http://127.0.0.1:8001")
STUB_PORT, SERVER_PORT = 8096, 8097
URL = f"http://127.0.0.1:{SERVER_PORT}"
RUN = f"{M5}/run-record"
NS = "fiqa"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def queue_depth():
    return get(f"{URL}/admin/indices/vector/queue/summary")["total_pending"]


def drain(phase):
    t = time.time()
    while queue_depth():
        time.sleep(5)
    log(f"{phase}: queue drained in {time.time() - t:.0f} s; stub {get(f'http://127.0.0.1:{STUB_PORT}/stats')}")


def put(doc_id, body):
    req = urllib.request.Request(f"{URL}/stores/{NS}/docs/{doc_id}", data=json.dumps(body).encode(), method="PUT",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        if r.status >= 300:
            raise RuntimeError(f"PUT {doc_id}: {r.status}")


def put_all(phase, docs):
    t = time.time()
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        for f in concurrent.futures.as_completed([pool.submit(put, i, b) for i, b in docs]):
            f.result()
    log(f"{phase}: {len(docs)} writes in {time.time() - t:.0f} s")
    drain(phase)


def body(d, text):
    return {"text": text, "category": d["category"], "year": d["year"], "score": d["score"]}


base = {d["id"]: d for d in map(json.loads, open(f"{DATA}/base.jsonl"))}
stream = [json.loads(l) for l in open(f"{DATA}/stream.jsonl")]
ids = json.load(open(f"{DATA}/ids.json"))
relevant = set(ids["relevant"])
stable, stable_text = ids["stable"], [i for i in ids["stable"] if i not in relevant]

shutil.rmtree(RUN, ignore_errors=True)
os.makedirs(RUN)
for d in ("db", "schemas", "log"):
    if os.path.exists(f"{M5}/base/{d}"):
        shutil.copytree(f"{M5}/base/{d}", f"{RUN}/{d}")
cfg = open(f"{M5}/server-base.toml").read().replace(f"{M5}/base/", f"{RUN}/")
cfg = cfg.replace('listen_addr = "127.0.0.1:8095"', f'listen_addr = "127.0.0.1:{SERVER_PORT}"')
cfg = cfg.replace("http://localhost:8001", f"http://127.0.0.1:{STUB_PORT}").replace("http://127.0.0.1:8001", f"http://127.0.0.1:{STUB_PORT}")
# Faster recording only: how many documents are embedded at once, not what is sent.
cfg = cfg.replace("concurrency = 4", "concurrency = 16")
assert f"127.0.0.1:{SERVER_PORT}" in cfg and f"127.0.0.1:{STUB_PORT}" in cfg, "server-base.toml no longer has the expected keys"
open(f"{RUN}/server.toml", "w").write(cfg)

stub = subprocess.Popen(["taskset", "-c", "0-3", sys.executable, f"{HERE}/embed_replay.py", "record", TABLE, str(STUB_PORT), UPSTREAM],
                        stdout=open(f"{RUN}/stub.out", "w"), stderr=subprocess.STDOUT)
time.sleep(2)
server = subprocess.Popen(["taskset", "-c", "4-31", f"{BIN}/minnal_db_api", f"{RUN}/server.toml"],
                          stdout=open(f"{RUN}/server.out", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(600):
        try:
            get(f"{URL}/stores")
            break
        except Exception:
            time.sleep(0.5)
    drain("start")
    subprocess.run(["taskset", "-c", "0-3", f"{BIN}/minnal_tools", "search_load", "-c", "1", "--top-k", "100",
                    URL, NS, f"{DATA}/queries.jsonl"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    log(f"queries: stub {get(f'http://127.0.0.1:{STUB_PORT}/stats')}")
    put_all("stable, own text", [(i, body(base[i], base[i]["text"])) for i in stable])
    put_all("stable, revised", [(i, body(base[i], base[i]["text"] + " Revised.")) for i in stable_text])
    put_all("stream", [(d["id"], body(d, d["text"])) for d in stream])
    put_all("stream copies", [(10_000_000_000 + len(stream) + k, body(d, d["text"] + " (copy)")) for k, d in enumerate(stream)])
    log(f"table {TABLE}: {os.path.getsize(TABLE) / 2**20:.0f} MiB")
finally:
    server.send_signal(signal.SIGTERM)
    server.wait(timeout=300)
    stub.send_signal(signal.SIGTERM)
    stub.wait(timeout=30)
