"""Rules the M5 driver (run_m5.py) and report (report_m5.py) share, so a step
the driver passes is one the report calls sustainable."""
import json

# A step may leave the vector queue this many entries deeper than it found it
# (noise); beyond that, writes were accepted faster than they were embedded.
QUEUE_NOISE = 500


def timeline(path):
    """A step's timeline: the admin snapshots (rows with `ops`), and every row."""
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if "ops" in r], rows


def queue_grew(snaps):
    """Whether the vector queue ended the step deeper than it began it."""
    q = [s["queue"]["total_pending"] for s in snaps if s.get("queue")]
    return bool(q) and q[-1] > q[0] + QUEUE_NOISE


def validity(snaps):
    """Whether the background work a step must include happened during it."""
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
