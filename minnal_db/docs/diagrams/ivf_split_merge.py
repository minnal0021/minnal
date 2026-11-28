"""Generates ../ivf-split-merge.svg: how a store's postings split, reassign and merge.

    python3 minnal_db/docs/diagrams/ivf_split_merge.py

The picture is driven by ivf_model.Partition, a small model of the real rules
(see its docstring): chunks are filed under the nearest posting's centre, a
posting over the split limit is split by balanced 2-means, LIRE's reassignment
moves the chunks the new centres now beat, and a posting at the merge limit is
folded into a nearby posting with room. Chunks never move in the plane: what
changes is which posting each one is filed under.
"""

from pathlib import Path

from anim import Anim, caption, document, text
from ivf_draw import POSTING_COLOURS, draw_plane, draw_sizes, plane_legend, op_dot
from ivf_model import Partition

SPLIT_LIMIT, MERGE_LIMIT = 8, 2

LEFT = [(22, 44), (30, 58), (18, 60), (26, 50)]
RIGHT = [(68, 40), (66, 60), (74, 46), (76, 56), (62, 32)]
FIRST = [LEFT[0], RIGHT[0], LEFT[1], RIGHT[1], LEFT[2], RIGHT[2], LEFT[3], RIGHT[3], RIGHT[4]]
EDGE = (42, 20)  # a chunk on the edge of the left posting, inserted later
MORE = [(76, 24), (80, 74), (70, 28), (72, 70)]


def build():
    """Run the script on the model. Returns [(op, caption, snapshot)]."""
    m = Partition(SPLIT_LIMIT, MERGE_LIMIT)
    steps = []

    def step(op, cap):
        steps.append((op, cap, m.snapshot()))

    step("start", "A store starts with one posting, <tspan class='ink-a'>P0</tspan>. "
                  "Its centre is the zero vector.")
    for i in range(3):
        m.insert(i, FIRST[i])
    step("insert", "Documents arrive. Each chunk goes to P0, the only posting, and its code is written against P0's centre.")
    for i in range(3, 6):
        m.insert(i, FIRST[i])
    step("insert", "Every chunk is filed under the posting whose centre is nearest. So far that is always P0.")
    for i in range(6, 9):
        m.insert(i, FIRST[i])
    assert m.over_split_limit() == [0]
    step("insert", f"P0 now holds 9 chunks, more than the split limit ({SPLIT_LIMIT}). The write wakes maintenance.")

    a, b = m.split(0)
    right, left = (a, b) if m.centre_of(a)[0] > m.centre_of(b)[0] else (b, a)
    assert (right, left) == (1, 2), (right, left)
    step("split", "Split: 2-means divides P0's chunks in two. Each half, P1 and P2, is centred on the mean of its "
                  "chunks. P0 is retired.")
    assert not m.reassign(0, [a, b])
    step("reassign", "Reassign finds nothing to move. No code was rewritten: every chunk's code still refers to "
                     "P0's centre, so every chunk is hollow.")

    m.insert(9, MORE[0])
    m.insert(10, MORE[1])
    m.insert(99, EDGE)
    assert m.chunks[99]["posting"] == left
    step("insert", "New chunks go to the nearest centre. The chunk at the top edge is only just nearer P2 than P1.")
    m.insert(11, MORE[2])
    m.insert(12, MORE[3])
    assert m.over_split_limit() == [right]
    step("insert", "P1 now holds 9 chunks: past the split limit again.")

    halves = m.split(right)
    lower, upper = sorted(halves, key=lambda p: -m.centre_of(p)[1])
    assert (lower, upper) == (3, 4), (lower, upper)
    step("split", "Split: P1's chunks are divided between P3 and P4. P1 is retired.")
    moved = m.reassign(right, halves)
    assert moved == {99} and m.chunks[99]["posting"] == upper, moved
    step("reassign", "Reassign: the edge chunk of P2 is now nearer P4's centre than both P1's old centre and P2's "
                     "own, so it moves to P4.")

    gone_lower = sorted(m.postings[lower]["members"])[:2]
    gone_upper = [c for c in sorted(m.postings[upper]["members"]) if c != 99][:1]
    for c in gone_lower + gone_upper:
        m.delete(c)
    assert m.size(lower) == MERGE_LIMIT and m.size(upper) == 5
    step("delete", f"Documents are deleted. P3 is down to {MERGE_LIMIT} chunks: at the merge limit.")

    drained, kept = m.merge(lower)
    assert (drained, kept) == (lower, upper)
    step("merge", f"Merge: P3 folds into P4, the nearest posting with room (2 + 5 &lt; {SPLIT_LIMIT}). The larger, P4, "
                  "keeps its centre. P3 is retired.")
    m.moved = set()
    step("end", "Retired postings are never searched. Their centres stay stored, because the hollow chunks' codes "
                "still refer to them.")
    return steps


def main():
    steps = build()
    n = len(steps)
    anim = Anim(n, 2.8)
    W, H = 940, 640
    out = [
        text(44, 42, "Dynamic partition: split, reassign, merge", "t1", 19, extra=' font-weight="700"'),
        text(44, 64, "Chunks stay where they are. What changes is the posting each one is filed under, and "
                     "the centres that route new chunks.", "t2", 13),
    ]
    snaps = [s for _, _, s in steps]
    out.append(draw_plane(anim, snaps, x=44, y=92, size=420))
    out.append(plane_legend(44, 540))
    out.append(draw_sizes(anim, snaps, x=500, y=92, w=396, split_limit=SPLIT_LIMIT, merge_limit=MERGE_LIMIT,
                          max_chunks=10, rows=5))
    out.append(caption(anim, 44, 584, [(op_dot(op), cap) for op, cap, _ in steps]))
    out.append(text(44, 620, f"Illustrative: a split limit of {SPLIT_LIMIT} and a merge limit of {MERGE_LIMIT} chunks "
                    "(the defaults are 256 and 22), in 2 dimensions instead of 768. The real partition works on the "
                    "chunks' code reconstructions.", "t3", 10))
    assert len(POSTING_COLOURS) >= 5
    svg = document(W, H, "Animated diagram of a store's dynamic partition: chunks are filed under the posting with the "
                         "nearest centre; a posting past the split limit is split in two by 2-means and retired; a "
                         "chunk that a new centre now beats is reassigned; a posting shrunk by deletes to the merge "
                         "limit is folded into a nearby posting with room. No code is rewritten, so moved chunks keep "
                         "the centre they were encoded against, and retired centres stay stored.", "\n".join(out))
    Path(__file__).resolve().parent.parent.joinpath("ivf-split-merge.svg").write_text(svg)


if __name__ == "__main__":
    main()
