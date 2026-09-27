"""Generates ../wal-lifecycle.svg: how WAL segments fill, get persisted, and are reclaimed.

    python3 minnal_db/docs/diagrams/wal_lifecycle.py

The picture is driven by a small simulation of the real rules:
- a put appends an Inserted entry to the active segment, opening a new segment
  when the active one is full;
- a memtable flush marks every entry written so far Persisted (its keys are now
  in an SSTable);
- WAL GC deletes any segment that is not the active one and whose entries are
  all Persisted; the head is the oldest segment still on disk.
"""

from pathlib import Path

from anim import AMBER, BLUE, GREY, Anim, caption, document, esc, legend, text

SLOTS = 4  # entries per segment, for the drawing (real segments are 64 MB)
NAMES = ["wal.log", "wal.log.seg000001", "wal.log.seg000002"]

SCRIPT = [
    ("put", "put() appends entry 1 to <tspan class='ink-a'>wal.log</tspan> and fsyncs before returning."),
    ("put", "More writes append to the same segment. Each is fsynced on its own."),
    ("put", "Every entry is <tspan class='ink-a'>Inserted</tspan>: the WAL is the only durable copy of these writes."),
    ("put", "The segment is now full."),
    ("put", "The next write opens wal.log.seg000001, which becomes the active segment."),
    ("put", "Nothing can be reclaimed yet: every entry is still needed for crash recovery."),
    ("flush", "A memtable flush writes these keys to an SSTable, so their entries become <tspan class='ink-b'>Persisted</tspan>."),
    ("GC", "WAL GC deletes wal.log: every entry in it is Persisted and it is not the active segment."),
    ("put", "New writes keep appending to the active segment."),
    ("put", "seg000001 is full again, holding a mix of Persisted and Inserted entries."),
    ("put", "The next write opens seg000002."),
    ("GC", "GC finds nothing to delete: seg000001 still has Inserted entries, and seg000002 is active."),
    ("flush", "The next flush persists the remaining entries."),
    ("GC", "GC deletes seg000001. The active segment is never deleted, even when all of it is Persisted."),
]


def simulate():
    segs = []  # each: {"entries": [...], "deleted": bool}
    states = []
    for op, _ in SCRIPT:
        live = [s for s in segs if not s["deleted"]]
        if op == "put":
            if not live or len(live[-1]["entries"]) == SLOTS:
                segs.append({"entries": [], "deleted": False})
            segs[-1]["entries"].append("I")
        elif op == "flush":
            for s in live:
                s["entries"] = ["P"] * len(s["entries"])
        elif op == "GC":
            active = live[-1]
            for s in live:
                if s is not active and all(e == "P" for e in s["entries"]):
                    s["deleted"] = True
        states.append([(tuple(s["entries"]), s["deleted"]) for s in segs])
    return states


def main():
    states = simulate()
    n = len(SCRIPT)
    anim = Anim(n, 2.2)
    W, H = 940, 590
    out = [
        text(44, 42, "WAL lifecycle: append → persist → reclaim", "t1", 19, extra=' font-weight="700"'),
        text(44, 64, "Every write is fsynced into one shared log. Once a flush has put its keys in an SSTable, "
             "WAL GC can delete the segment.", "t2", 13),
        legend(470, 104, [(AMBER, "Inserted: only the WAL holds it"), (BLUE, "Persisted: also in an SSTable")]),
        text(44, 104, "WAL SEGMENTS", "t3", 12, extra=' letter-spacing="0.06em"'),
    ]
    box_w, gap, x0, y0 = 270, 20, 44, 118
    for i, name in enumerate(NAMES):
        x = x0 + i * (box_w + gap)

        def seg_state(step, i=i):
            segs = states[step]
            if i >= len(segs):
                return ("absent",)
            entries, deleted = segs[i]
            if deleted:
                return ("deleted",)
            live = [j for j, (_, d) in enumerate(segs) if not d]
            role = "active" if i == live[-1] else "sealed"
            head = i == live[0]
            return ("present", entries, role, head)

        def render(v, x=x, name=name):
            if v[0] == "absent":
                return (f'<rect class="ghost" x="{x}" y="{y0}" width="{box_w}" height="80" rx="9" stroke-dasharray="5 4"/>'
                        + text(x + 14, y0 + 20, esc(name), "t3 mono", 11)
                        + text(x + box_w / 2, y0 + 52, "not created yet", "t3", 12, "middle", ' font-style="italic"'))
            if v[0] == "deleted":
                return (f'<rect class="ghost" x="{x}" y="{y0}" width="{box_w}" height="80" rx="9" stroke-dasharray="5 4"/>'
                        + text(x + 14, y0 + 20, esc(name), "t3 mono", 11)
                        + text(x + box_w / 2, y0 + 52, "deleted by WAL GC", "t3", 12, "middle", ' font-style="italic"'))
            _, entries, role, head = v
            parts = [f'<rect class="panel" x="{x}" y="{y0}" width="{box_w}" height="80" rx="9"/>',
                     text(x + 14, y0 + 20, esc(name), "t2 mono", 11),
                     text(x + box_w - 14, y0 + 20, role, "ink-a" if role == "active" else "t3", 11, "end")]
            for j in range(SLOTS):
                sx = x + 14 + j * 62
                if j < len(entries):
                    fill = AMBER if entries[j] == "I" else BLUE
                    parts.append(f'<rect x="{sx}" y="{y0 + 34}" width="54" height="32" rx="5" fill="{fill}"/>')
                else:
                    parts.append(f'<rect class="empty" x="{sx}" y="{y0 + 34}" width="54" height="32" rx="5"/>')
            if head:
                cx = x + box_w / 2
                parts.append(f'<path class="arrow" d="M {cx - 6} {y0 + 98} L {cx + 6} {y0 + 98} L {cx} {y0 + 89} Z"/>')
                parts.append(text(cx, y0 + 112, "head: oldest segment on disk", "t3", 11, "middle"))
            return "".join(parts)

        out.append(anim.states([seg_state(s) for s in range(n)], render))

    dot = {"put": AMBER, "flush": BLUE, "GC": GREY}
    out.append(caption(anim, 44, 272, [(dot[op], cap) for op, cap in SCRIPT]))

    # Entries on disk after each step, revealed one bar per step.
    top, bottom, left, right, max_v = 330, 450, 90, 894, 8
    out.append(text(44, 312, "ENTRIES IN WAL FILES ON DISK, AFTER EACH STEP", "t3", 12, extra=' letter-spacing="0.06em"'))
    for v in (0, 4, 8):
        y = bottom - (bottom - top) * v / max_v
        out.append(f'<line class="grid" x1="{left}" y1="{y}" x2="{right}" y2="{y}"/>')
        out.append(text(left - 10, y + 4, str(v), "t3", 10, "end"))
    slot = (right - left) / n
    for i, (op, _) in enumerate(SCRIPT):
        segs = [e for e, deleted in states[i] if not deleted]
        ins = sum(e.count("I") for e in segs)
        per = sum(e.count("P") for e in segs)
        bx = left + i * slot + slot * 0.18
        bw = slot * 0.64
        unit = (bottom - top) / max_v
        bar = ""
        if per:
            bar += f'<rect x="{bx:.1f}" y="{bottom - per * unit:.1f}" width="{bw:.1f}" height="{per * unit:.1f}" fill="{BLUE}"/>'
        if ins:
            bar += f'<rect x="{bx:.1f}" y="{bottom - (per + ins) * unit:.1f}" width="{bw:.1f}" height="{ins * unit:.1f}" fill="{AMBER}"/>'
        out.append(f"<g>{anim.mask([s >= i for s in range(n)])}{bar}</g>")
        out.append(text(round(bx + bw / 2, 1), bottom + 16, op, "t3", 10, "middle"))
    out.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>')

    def counters(step):
        segs = [e for e, deleted in states[step] if not deleted]
        return (sum(len(e) for e in segs), sum(e.count("P") for e in segs), len(segs))

    out.append(text(44, 500, "entries on disk", "t3 mono", 11))
    out.append(text(260, 500, "of which Persisted", "t3 mono", 11))
    out.append(text(476, 500, "segment files", "t3 mono", 11))
    out.append(anim.states([counters(s) for s in range(n)], lambda v: (
        text(44, 524, v[0], "t1 mono", 18, extra=' font-weight="700"')
        + text(260, 524, v[1], "t1 mono", 18, extra=' font-weight="700"')
        + text(476, 524, v[2], "t1 mono", 18, extra=' font-weight="700"'))))
    out.append(text(44, 566, "Illustrative: 4 entries per segment; real segments are 64 MB. "
                    "With field indexes, GC also waits until each index has checkpointed past a segment.", "t3", 10))

    svg = document(W, H, "Animated diagram: WAL entries are appended and fsynced, become Persisted when a memtable "
                         "flush writes their keys to an SSTable, and whole segments are deleted by WAL GC once "
                         "every entry is Persisted; the active segment is never deleted.", "\n".join(out))
    Path(__file__).resolve().parent.parent.joinpath("wal-lifecycle.svg").write_text(svg)


if __name__ == "__main__":
    main()
