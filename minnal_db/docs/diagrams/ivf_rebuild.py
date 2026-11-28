"""Generates ../ivf-rebuild.svg: what a rebuild does to a store's postings, and why it can help a search.

    python3 minnal_db/docs/diagrams/ivf_rebuild.py

Driven by ivf_model.Partition. The store's partition is first grown by the
ordinary rules (inserts, splits, reassignments, merges), from three groups of
chunks that arrive one group at a time. A posting's centre is fixed when the
posting is created, so later chunks leave some centres in the wrong place, and
a search that reads one posting finds only some of its query's nearest chunks.
The rebuild then takes the steps of ivf_split/rebuild.rs: plan (k-means over
every chunk), publish (new postings, the old ones draining), copy (every chunk
to its nearest new posting), retire. No code is rewritten.
"""

import random
from pathlib import Path

from anim import AMBER, BLUE, GREEN, ORANGE, Anim, caption, document, text
from ivf_draw import EXTRA_COLOURS, draw_plane, draw_sizes, op_dot, plane_legend
from ivf_model import Partition, dist2

SPLIT_LIMIT, MERGE_LIMIT = 8, 2
MAX_PROBES = 1  # postings a search reads, for the drawing (the default is 256)
SEED = 162      # the three groups of chunks and their arrival order
QUERY = (40, 40)
NEAREST = 3


def groups(seed):
    """Three groups of 7 chunks around random centres, arriving group by group."""
    rng = random.Random(seed)
    centres = [(rng.uniform(20, 80), rng.uniform(20, 80)) for _ in range(3)]
    pts = [(g, (round(c[0] + rng.gauss(0, 6), 1), round(c[1] + rng.gauss(0, 6), 1)))
           for g, c in enumerate(centres) for _ in range(7)]
    order = sorted(range(len(pts)), key=lambda i: (pts[i][0], rng.random()))
    return [pts[i][1] for i in order]


def build():
    m = Partition(SPLIT_LIMIT, MERGE_LIMIT)
    for n, pos in enumerate(groups(SEED)):
        m.insert(n, pos)
        m.maintain()
    m.moves, m.moved = 0, set()
    first = sorted(m.active())
    assert len(first) == 3, first
    nearest = sorted(m.chunks, key=lambda c: dist2(QUERY, m.chunks[c]["pos"]))[:NEAREST]

    def found():
        read = m.read_postings(QUERY, MAX_PROBES)
        return read, sum(1 for c in nearest if m.chunks[c]["posting"] in read)

    steps = []  # (op, caption, snapshot, planned centres, query, "query finds")

    def step(op, cap, planned=None, query=False):
        q, finds = None, "—"
        if query:
            read, k = found()
            q, finds = (QUERY, read, nearest), f"{k} of {NEAREST}"
        steps.append((op, cap, m.snapshot(), planned, q, finds))

    step("start", "This partition grew one split at a time. A posting's centre is fixed when the posting is created, "
                  "and later chunks never move it.")
    read, k = found()
    assert k == 1, k
    step("search", f"A search reads only the posting nearest its query (1 here). It finds {k} of the query's "
                   f"{NEAREST} nearest chunks; the rest sit in a posting it does not read.",
         query=True)
    planned = m.plan_rebuild()
    assert len(planned) == 3
    step("plan", "Rebuild, plan: k-means over every chunk's code fits 3 new centres. Nothing has moved yet.",
         planned=tuple(planned))
    old, new = m.publish_rebuild(planned)
    step("publish", "Publish: 3 new postings start empty, and the old ones are draining. New chunks now go to the "
                    "new postings.")
    m.copy_rebuild(new)
    step("copy", "Copy: every chunk moves to the new posting whose centre is nearest. No code is rewritten, so "
                 "every chunk is now hollow.")
    m.retire_rebuild(old)
    step("retire", "Retire: the old postings are retired. Their centres stay stored, because the codes still refer "
                   "to them.")
    if m.over_split_limit():
        m.maintain()
        step("split", "Maintenance then splits any posting k-means left over the split limit.")
    read, k = found()
    assert k == NEAREST, k
    step("search", f"The same search reads 1 posting again, and now finds all {NEAREST} of the query's nearest "
                   "chunks.", query=True)
    step("end", "A rebuild changes where chunks are filed, so searches find more. It rewrites no code, so centre "
                "drift goes up. Measure it with a recall check.", query=True)
    colours = dict(zip(first, (AMBER, BLUE, GREEN)))
    colours.update(zip(new, (ORANGE, EXTRA_COLOURS[0], EXTRA_COLOURS[1])))
    return steps, first + new, colours


def main():
    steps, shown, colours = build()
    n = len(steps)
    anim = Anim(n, 3.0)
    W, H = 940, 660
    out = [
        text(44, 42, "Dynamic partition: rebuild", "t1", 19, extra=' font-weight="700"'),
        text(44, 64, "A rebuild refits every posting at once, from the chunks' codes, and moves every chunk to its "
                     "nearest new posting.", "t2", 13),
    ]
    snaps = [s[2] for s in steps]
    out.append(draw_plane(anim, snaps, x=44, y=92, size=420, colours=colours,
                          planned=[s[3] for s in steps], query=[s[4] for s in steps]))
    out.append(plane_legend(44, 540))
    out.append(text(44, 584, "◇ query, with its 3 nearest chunks ringed and the posting it reads outlined",
                    "t2", 12))
    out.append(draw_sizes(anim, snaps, x=500, y=92, w=396, split_limit=SPLIT_LIMIT, merge_limit=MERGE_LIMIT,
                          max_chunks=10, rows=len(shown), colours=colours, only=shown,
                          extra_counter=("query finds", [s[5] for s in steps])))
    out.append(caption(anim, 44, 612, [(op_dot(op), cap) for op, cap, *_ in steps]))
    out.append(text(44, 644, f"Illustrative: searches read {MAX_PROBES} posting (the default max_probes is 256), a "
                    f"split limit of {SPLIT_LIMIT} chunks, 2 dimensions instead of 768. The real rebuild fits its "
                    "centres to the chunks' code reconstructions.", "t3", 10))
    svg = document(W, H, "Animated diagram of a rebuild: a partition grown one split at a time has centres fixed "
                         "before later chunks arrived, so a search that reads one posting finds only one of its "
                         "query's three nearest chunks; the rebuild fits new centres with k-means over every chunk's "
                         "code, publishes new empty postings while the old ones drain, copies every chunk to its "
                         "nearest new posting without rewriting its code, and retires the old postings while keeping "
                         "their centres; the same search then finds all three.", "\n".join(out))
    Path(__file__).resolve().parent.parent.joinpath("ivf-rebuild.svg").write_text(svg)


if __name__ == "__main__":
    main()
