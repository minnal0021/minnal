"""Generates ../vlog-segments.svg: value-log writes and garbage collection of one segment.

    python3 minnal_db/docs/diagrams/vlog_segments.py

Follows `KVStore::compact_bucket` and `garbage_collect_with_threshold`:
GC reads a sealed segment without a lock, decides liveness with one batched
LSM lookup (each record stores its key), appends the survivors to the active
tail and re-points them under the bucket lock with a compare-and-set, flushes
the memtable so the re-point is durable, and only then deletes the file.
Segment ids are never reused.
"""

from pathlib import Path

from anim import BLUE, GREEN, GREY, ORANGE, Anim, caption, document, esc, legend, text

LIVE, GARBAGE, MOVED = "L", "G", "M"
COLOUR = {LIVE: GREEN, GARBAGE: ORANGE, MOVED: BLUE}
SLOTS = 4
SEG_IDS = [12, 13, 14, 15]
LSM_KEYS = ["k1", "k2", "k3", "k4", "k5", "k9"]
PILLS = ["1 · pick a segment", "2 · scan it (no lock)", "3 · move survivors (bucket lock)",
         "4 · flush to L0", "5 · delete the file"]


def steps():
    seg = {12: ["sealed", [["k1", LIVE], ["k2", LIVE], ["k3", LIVE], ["k4", LIVE]]],
           13: ["sealed", [["k5", LIVE], ["k6", LIVE], ["k7", LIVE], ["k8", LIVE]]],
           14: ["active tail", []],
           15: ["absent", []]}
    lsm = {"k1": "seg 12", "k2": "seg 12", "k3": "seg 12", "k4": "seg 12", "k5": "seg 13"}
    held = "holds a pointer to k1 in segment 12, looked up before GC"
    out = []

    def snap(cap, dot, pill=None, reader=("idle", held)):
        out.append({
            "seg": {i: (r, tuple(tuple(x) for x in recs)) for i, (r, recs) in seg.items()},
            "lsm": tuple(lsm.get(k) for k in LSM_KEYS),
            "cap": (dot, cap), "pill": pill, "reader": reader,
        })

    def mark(seg_id, key, state):
        for rec in seg[seg_id][1]:
            if rec[0] == key:
                rec[1] = state

    snap("Values live in numbered segment files. Only the active tail is appended to; sealed segments never change.", GREY)
    seg[14][1].append(["k2′", LIVE]); mark(12, "k2", GARBAGE); lsm["k2"] = "seg 14"
    snap("put(k2) appends the new value to the tail. The old record is not touched; its size is added to segment 12's garbage count.", GREEN)
    mark(12, "k3", GARBAGE); lsm["k3"] = "deleted"
    snap("delete(k3) removes the key from the LSM. Its record becomes garbage too: segment 12 is now half garbage.", ORANGE)
    snap("GC picks the segment with the most garbage, 12. Segment 13 has none, so GC never reads it.", GREY, 0)
    snap("GC reads segment 12 with no lock held. Each record stores its key, so one batched LSM lookup finds k1 and k4 live.", GREY, 1)
    seg[14][1] += [["k1", MOVED], ["k4", MOVED]]; mark(12, "k1", GARBAGE); mark(12, "k4", GARBAGE)
    lsm["k1"] = lsm["k4"] = "seg 14"
    snap("Under the bucket lock, k1 and k4 are appended to the tail and re-pointed, each only if it still points at segment 12.", BLUE, 2)
    snap("The reader uses its old pointer and gets k1's own value from segment 12. The pointer is stale, not wrong.", GREEN, 2,
         ("ok", "reads segment 12: k1's own value, same version"))
    snap("A memtable flush makes the re-pointed entries durable in an SSTable.", GREY, 3)
    seg[12][0] = "deleted"
    snap("Only now is segment 12 deleted. A crash at any earlier step leaves every pointer on a file that exists.", GREY, 4)
    snap("A reader using the old pointer now gets SegmentMissing. It looks k1 up again and reads it from segment 14.", ORANGE, None,
         ("retry", "segment 12 is gone: SegmentMissing, look k1 up again, read segment 14"))
    seg[14][1].append(["k9", LIVE]); lsm["k9"] = "seg 14"
    snap("put(k9) fills segment 14.", GREEN, None, ("idle", "—"))
    seg[14][0] = "sealed"; seg[15][0] = "active tail"
    snap("Segment 14 is sealed and 15 opens. Id 12 is never reused, so an old pointer can never reach another key's record.", GREY, None, ("idle", "—"))
    return out


def main():
    st = steps()
    n = len(st)
    anim = Anim(n, 3.0)
    W, H = 940, 600
    o = [
        text(44, 42, "Value log: garbage collection, one segment at a time", "t1", 19, extra=' font-weight="700"'),
        text(44, 64, "Values live in append-only segment files. GC copies one segment's live records to the tail, "
             "then deletes the file.", "t2", 13),
        text(44, 94, "pointer (u128) =  bucket | segment id | offset in segment | value length", "t2 mono", 11),
        legend(560, 94, [(GREEN, "live"), (ORANGE, "garbage"), (BLUE, "moved by GC")]),
    ]

    # LSM panel.
    o.append('<rect class="panel" x="44" y="112" width="180" height="170" rx="9"/>')
    o.append(text(58, 132, "LSM: key → segment", "t2 mono", 11))
    for i, key in enumerate(LSM_KEYS):
        y = 158 + i * 20
        o.append(anim.states([s["lsm"][i] for s in st], lambda v, y=y, key=key: (
            text(62, y, key, "t1 mono", 13) + text(100, y, "→ " + esc(v), "t2 mono", 13))))

    # Segment files.
    o.append('<rect class="panel" x="240" y="112" width="656" height="170" rx="9"/>')
    o.append(text(254, 132, "bucket 3: segment files", "t2 mono", 11))
    bw, gap, x0, y0 = 152, 8, 254, 146
    for i, sid in enumerate(SEG_IDS):
        x = x0 + i * (bw + gap)

        def render(v, x=x, sid=sid):
            role, recs = v
            name = text(x + 10, y0 + 18, f"seg {sid}", "t2 mono", 11)
            if role in ("absent", "deleted"):
                label = "not created yet" if role == "absent" else "deleted"
                return (f'<rect class="ghost" x="{x}" y="{y0}" width="{bw}" height="120" rx="8" stroke-dasharray="5 4"/>'
                        + name + text(x + bw / 2, y0 + 66, label, "t3", 12, "middle", ' font-style="italic"'))
            parts = [f'<rect class="empty" x="{x}" y="{y0}" width="{bw}" height="120" rx="8"/>', name,
                     text(x + bw - 10, y0 + 18, role, "ink-a" if role == "active tail" else "t3", 10.5, "end")]
            for j in range(SLOTS):
                sx, sy = x + 10 + (j % 2) * 68, y0 + 30 + (j // 2) * 42
                if j < len(recs):
                    key, state = recs[j]
                    parts.append(f'<rect x="{sx}" y="{sy}" width="62" height="36" rx="5" fill="{COLOUR[state]}"/>')
                    parts.append(text(sx + 31, sy + 16, esc(key), "mono", 12, "middle", ' fill="#ffffff" font-weight="700"'))
                    parts.append(text(sx + 31, sy + 29, "+ key", "", 9, "middle", ' fill="#ffffff" opacity="0.85"'))
                else:
                    parts.append(f'<rect class="panel" x="{sx}" y="{sy}" width="62" height="36" rx="5" stroke-dasharray="3 3"/>')
            return "".join(parts)

        o.append(anim.states([s["seg"][sid] for s in st], render))

    # GC steps.
    o.append(text(44, 312, "GC STEPS", "t3", 12, extra=' letter-spacing="0.06em"'))
    px = 44
    for i, label in enumerate(PILLS):
        w = len(label) * 6.1 + 26
        on = [s["pill"] == i for s in st]
        o.append(f'<g>{anim.mask([not v for v in on])}<rect class="pill" x="{px:.0f}" y="322" width="{w:.0f}" height="28" rx="14"/></g>')
        o.append(f'<g>{anim.mask(on)}<rect class="pill-on" x="{px:.0f}" y="322" width="{w:.0f}" height="28" rx="14" stroke-width="2"/></g>')
        o.append(text(round(px + w / 2), 340, esc(label), "t2", 11, "middle"))
        px += w + 10

    # The reader.
    o.append(text(44, 378, "A READER THAT LOOKED UP k1 BEFORE GC STARTED", "t3", 12, extra=' letter-spacing="0.06em"'))
    cls = {"idle": "note-off", "ok": "note-ok", "retry": "note-bad"}
    o.append(anim.states([s["reader"] for s in st], lambda v: (
        f'<rect class="{cls[v[0]]}" x="44" y="388" width="852" height="32" rx="16"/>'
        + text(64, 409, esc(v[1]), "t1" if v[0] != "idle" else "t3", 12))))

    o.append(caption(anim, 44, 456, [s["cap"] for s in st]))

    def counts(s):
        live = garbage = files = 0
        for role, recs in s["seg"].values():
            if role in ("absent", "deleted"):
                continue
            files += 1
            live += sum(1 for _, x in recs if x != GARBAGE)
            garbage += sum(1 for _, x in recs if x == GARBAGE)
        return live, garbage, files

    o.append(text(44, 500, "live records", "t3 mono", 11))
    o.append(text(260, 500, "garbage records", "t3 mono", 11))
    o.append(text(476, 500, "segment files", "t3 mono", 11))
    o.append(anim.states([counts(s) for s in st], lambda v: (
        text(44, 524, v[0], "t1 mono", 18, extra=' font-weight="700"')
        + text(260, 524, v[1], "t1 mono", 18, extra=' font-weight="700"')
        + text(476, 524, v[2], "t1 mono", 18, extra=' font-weight="700"'))))
    o.append(text(44, 572, "Illustrative: 4 records per segment; real segments are 256 MB. "
                  "GC starts when a namespace, or any one bucket, passes 30% garbage, and rewrites segments over 10% garbage.", "t3", 10))

    svg = document(W, H, "Animated diagram: value-log garbage collection. Updates and deletes turn old records into "
                         "garbage; GC scans one sealed segment, moves its live records to the active tail, re-points "
                         "them, flushes, then deletes the file. A reader with an old pointer either reads the same "
                         "value or gets SegmentMissing and looks the key up again. Segment ids are never reused.",
                   "\n".join(o))
    Path(__file__).resolve().parent.parent.joinpath("vlog-segments.svg").write_text(svg)


if __name__ == "__main__":
    main()
