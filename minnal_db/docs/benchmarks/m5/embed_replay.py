"""Embedding replay for M5: answers the embedding service's API from a table of
embeddings recorded once, so a benchmark run measures the engine and never
waits on the GPU service (whose throughput is outside minnal and varies).

    python3 embed_replay.py record TABLE PORT UPSTREAM_URL
    python3 embed_replay.py replay TABLE PORT

- **record**: forwards every request to the real service at UPSTREAM_URL and
  appends each (model, endpoint, dimensions, text) -> vector it returns to
  TABLE. Answers from the table when it already holds every payload.
- **replay**: answers only from TABLE. A payload missing from it fails the
  whole request (HTTP 500) and is counted: a run with misses is not a valid
  engine measurement, and the report says so.

The API is the service's (`minnal_db/src/semantic_search/service/embedding_service.rs`):
`POST /embedding/{model}/{document|query}` with `{"payloads": [...],
"dimensions": N}` returns `{"embeddings": [[...], ...]}`, one per payload in
order; `GET /healthcheck` lists the models (here: every model in the table)
as `ok`. `GET /stats` returns this process's `requests`, `payloads`, `hits`
and `misses` (the driver reads it before and after every step).

Vectors are stored as little-endian f32, as the engine holds them. TABLE is an
append-only file of records:
`[key_len u32][key utf-8: model \\0 endpoint \\0 dims \\0 text][dims u32][dims x f32]`.
"""
import json
import struct
import sys
import threading
import urllib.request
from array import array
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

mode, table_path, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
upstream = sys.argv[4].rstrip("/") if mode == "record" else None
if mode not in ("record", "replay"):
    raise SystemExit(__doc__)

table: dict[bytes, bytes] = {}
models: set[str] = set()
lock = threading.Lock()
stats = {"requests": 0, "payloads": 0, "hits": 0, "misses": 0}


def key(model: str, endpoint: str, dims: int, text: str) -> bytes:
    return f"{model}\0{endpoint}\0{dims}\0{text}".encode()


def load():
    try:
        data = open(table_path, "rb").read()
    except FileNotFoundError:
        return
    pos = 0
    while pos + 4 <= len(data):
        (klen,) = struct.unpack_from("<I", data, pos)
        k = data[pos + 4 : pos + 4 + klen]
        (dims,) = struct.unpack_from("<I", data, pos + 4 + klen)
        start = pos + 8 + klen
        end = start + 4 * dims
        if end > len(data):
            break  # a record cut short by an interrupted recording: ignored
        table[k] = data[start:end]
        models.add(k.split(b"\0", 1)[0].decode())
        pos = end


load()
out = open(table_path, "ab") if mode == "record" else None


def vector_json(raw: bytes) -> list:
    v = array("f")
    v.frombytes(raw)
    return v.tolist()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def reply(self, status: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/healthcheck":
            names = models or {"gemma"}
            self.reply(200, {"status": "healthy", "models": {m: {"status": "ok"} for m in sorted(names)}})
        elif self.path == "/stats":
            with lock:
                self.reply(200, dict(stats))
        else:
            self.reply(404, {"detail": "not found"})

    def do_POST(self):
        parts = self.path.strip("/").split("/")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if len(parts) != 3 or parts[0] != "embedding" or parts[2] not in ("document", "query"):
            self.reply(404, {"detail": f"unknown path {self.path}"})
            return
        model, endpoint = parts[1], parts[2]
        payloads, dims = body.get("payloads", []), int(body.get("dimensions", 0))
        keys = [key(model, endpoint, dims, p) for p in payloads]
        with lock:
            stats["requests"] += 1
            stats["payloads"] += len(keys)
            found = [table.get(k) for k in keys]
        missing = [i for i, f in enumerate(found) if f is None]
        if missing and mode == "replay":
            with lock:
                stats["misses"] += len(missing)
                stats["hits"] += len(keys) - len(missing)
            self.reply(500, {"detail": f"replay miss: {len(missing)} of {len(keys)} payloads not in the table"})
            return
        if missing:
            req = urllib.request.Request(
                f"{upstream}{self.path}",
                data=json.dumps({"payloads": [payloads[i] for i in missing], "dimensions": dims}).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    got = json.loads(r.read())["embeddings"]
            except urllib.error.HTTPError as e:
                self.reply(e.code, json.loads(e.read() or b"{}") or {"detail": str(e)})
                return
            with lock:
                for i, vec in zip(missing, got):
                    raw = array("f", vec).tobytes()
                    k = keys[i]
                    if k not in table:
                        out.write(struct.pack("<I", len(k)) + k + struct.pack("<I", len(vec)) + raw)
                        table[k] = raw
                        models.add(model)
                    found[i] = table[k]
                out.flush()
        with lock:
            stats["hits"] += len(keys) - len(missing)
        self.reply(200, {"embeddings": [vector_json(f) for f in found]})


print(f"embed_replay {mode}: {len(table)} vectors loaded, models {sorted(models)}, port {port}", flush=True)
ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
