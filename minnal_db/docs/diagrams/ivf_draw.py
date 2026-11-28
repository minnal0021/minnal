"""Drawing for the IVF partition diagrams (ivf_split_merge.py, ivf_rebuild.py).

Each function takes the list of ivf_model.Partition snapshots, one per step,
and renders the elements that change through `Anim.states`.

- The **plane**: every active posting's routing area (the points nearer its
  centre than any other active centre) lightly tinted in its colour, its centre
  as a cross, and every chunk as a dot in its posting's colour. A dot is filled
  when the chunk's code was written against its posting's centre, and hollow
  when it was written against another centre (the chunk was moved, and moves
  never rewrite a code). A dashed ring marks the chunks the step moved.
  Centres no active posting uses any more are kept, drawn small and grey.
- The **posting sizes**: one row per posting, with the merge and split limits.
"""

from anim import AMBER, BLUE, GREEN, GREY, ORANGE, esc, text

# One colour per posting, by posting id. The first five are anim.py's
# categorical set; the rest only appear in the rebuild diagram.
POSTING_COLOURS = {0: GREY, 1: AMBER, 2: BLUE, 3: GREEN, 4: ORANGE}
EXTRA_COLOURS = ["#9a6bd6", "#14a3b8", "#d6589a", "#7a8f2b", "#c2410c", "#2563eb"]

PAD = 14

# Labels drawn over the tinted plane get a halo in the panel's colour, so a
# chunk or a boundary behind them never makes them hard to read.
HALO_STYLE = """<style>
  .halo { paint-order: stroke; stroke: #f2f1ec; stroke-width: 4px; stroke-linejoin: round; }
  @media (prefers-color-scheme: dark) { .halo { stroke: #1f1f24; } }
</style>"""

# Where a centre's label may go, relative to the centre: (dx, dy, anchor).
LABEL_SPOTS = [(11, -9, "start"), (11, 17, "start"), (-11, -9, "end"), (-11, 17, "end")]


def label_spot(px, py, width, dots):
    """The spot for a label of `width` px at centre (px, py) that keeps it
    farthest from every chunk dot."""
    def clearance(spot):
        dx, dy, anchor = spot
        x0 = px + dx if anchor == "start" else px + dx - width
        x1, y0, y1 = x0 + width, py + dy - 10, py + dy + 3
        best = 1e9
        for qx, qy in dots:
            ex = max(x0 - qx, 0, qx - x1)
            ey = max(y0 - qy, 0, qy - y1)
            best = min(best, (ex * ex + ey * ey) ** 0.5)
        return best
    return max(LABEL_SPOTS, key=clearance)


def colour(posting, colours=None):
    if colours and posting in colours:
        return colours[posting]
    if posting in POSTING_COLOURS:
        return POSTING_COLOURS[posting]
    return EXTRA_COLOURS[(posting - len(POSTING_COLOURS)) % len(EXTRA_COLOURS)]


def op_dot(op):
    return GREY


def _clip(poly, a, b):
    """Keep the part of `poly` nearer `a` than `b` (half-plane clipping)."""
    nx, ny = b[0] - a[0], b[1] - a[1]
    c = (b[0] ** 2 + b[1] ** 2 - a[0] ** 2 - a[1] ** 2) / 2
    inside = lambda p: p[0] * nx + p[1] * ny <= c
    out = []
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        pin, qin = inside(p), inside(q)
        if pin:
            out.append(p)
        if pin != qin:
            dp = p[0] * nx + p[1] * ny - c
            dq = q[0] * nx + q[1] * ny - c
            t = dp / (dp - dq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def cells(centres):
    """Voronoi cells of `centres` [(posting, (x, y))] inside the 0..100 square."""
    square = [(0, 0), (100, 0), (100, 100), (0, 100)]
    res = []
    for pid, c in centres:
        poly = square
        for qid, d in centres:
            if qid != pid and d != c:
                poly = _clip(poly, c, d)
                if not poly:
                    break
        res.append((pid, poly))
    return res


def draw_plane(anim, snaps, x, y, size, colours=None, op_labels=None, query=None):
    """The plane of chunks and centres, `size` px square at (x, y).

    `query`: optional per-step value None or (pos, read_postings, nearest_ids)
    for a query marker, the postings it reads and its true nearest chunks."""
    inner = size - 2 * PAD
    X = lambda v: round(x + PAD + v / 100 * inner, 1)
    Y = lambda v: round(y + PAD + v / 100 * inner, 1)
    out = [HALO_STYLE, f'<rect class="panel" x="{x}" y="{y}" width="{size}" height="{size}" rx="10"/>']
    dots = sorted({(X(c[0][0]), Y(c[0][1])) for s in snaps for c in s["chunks"].values()})

    # Routing areas of the active postings.
    def active_centres(s):
        return tuple(sorted((p, s["centres"][c]) for p, (c, st, _) in s["postings"].items() if st == "active"))

    def render_cells(v):
        parts = []
        for pid, poly in cells(list(v)):
            if len(poly) < 3:
                continue
            pts = " ".join(f"{X(px)},{Y(py)}" for px, py in poly)
            parts.append(f'<polygon points="{pts}" fill="{colour(pid, colours)}" fill-opacity="0.10" '
                         f'stroke="{colour(pid, colours)}" stroke-opacity="0.45" stroke-dasharray="4 4" stroke-width="1"/>')
        return "".join(parts)

    out.append(anim.states([active_centres(s) for s in snaps], render_cells))

    # Centres: active (a cross in the posting's colour) or kept (small, grey).
    n_centres = max(len(s["centres"]) for s in snaps)
    for cid in range(n_centres):
        def state(s, cid=cid):
            if cid >= len(s["centres"]):
                return None
            users = [(p, st) for p, (c, st, _) in s["postings"].items() if c == cid]
            live = [p for p, st in users if st == "active"]
            draining = [p for p, st in users if st == "draining"]
            if live:
                return ("active", live[0], s["centres"][cid])
            if draining:
                return ("draining", draining[0], s["centres"][cid])
            return ("kept", cid, s["centres"][cid])

        def render(v, cid=cid):
            kind, who, (cx, cy) = v
            px, py = X(cx), Y(cy)
            if kind == "kept":
                r = 4
                lab = f"c{cid}"
                dx, dy, anchor = label_spot(px, py, len(lab) * 6.2, dots)
                return (f'<path d="M {px - r} {py - r} L {px + r} {py + r} M {px - r} {py + r} L {px + r} {py - r}" '
                        f'class="line" stroke-width="1.6"/>'
                        + text(px + dx * 0.7, py + dy * 0.7, lab, "t3 mono halo", 9.5, anchor))
            r = 7
            col = colour(who, colours)
            dash = ' stroke-dasharray="2 2"' if kind == "draining" else ""
            label = f"P{who}" + (" · zero vector" if cid == 0 else "")
            dx, dy, anchor = label_spot(px, py, len(label) * 7.2, dots)
            return (f'<path d="M {px - r} {py - r} L {px + r} {py + r} M {px - r} {py + r} L {px + r} {py - r}" '
                    f'stroke="{col}" stroke-width="3.2" stroke-linecap="round"{dash}/>'
                    + text(px + dx, py + dy, label, "t1 mono halo", 11, anchor, ' font-weight="700"'))

        out.append(anim.states([state(s) for s in snaps], render))

    # Chunks.
    ids = sorted({i for s in snaps for i in s["chunks"]})
    for i in ids:
        def render(v):
            (cx, cy), posting, foreign, moved = v
            px, py = X(cx), Y(cy)
            col = colour(posting, colours)
            ring = (f'<circle cx="{px}" cy="{py}" r="12" fill="none" class="line" stroke-width="1.4" '
                    f'stroke-dasharray="3 2"/>' if moved else "")
            if foreign:
                dot = f'<circle cx="{px}" cy="{py}" r="5.6" class="surface" stroke="{col}" stroke-width="2.6"/>'
            else:
                dot = f'<circle cx="{px}" cy="{py}" r="6.4" fill="{col}"/>'
            return ring + dot

        out.append(anim.states([s["chunks"].get(i) for s in snaps], render))

    if query:
        def render_q(v):
            (qx, qy), _, nearest = v
            px, py = X(qx), Y(qy)
            return (f'<path d="M {px} {py - 9} L {px + 9} {py} L {px} {py + 9} L {px - 9} {py} Z" class="pill-on" '
                    f'stroke-width="2"/>' + text(px + 12, py + 4, "query", "t1", 11, extra=' font-weight="700"'))
        out.append(anim.states(query, render_q))

    if op_labels:
        out.append(anim.states(op_labels, lambda v: (
            f'<rect class="pill-on" x="{x + size - 108}" y="{y + 10}" width="96" height="22" rx="11"/>'
            + text(x + size - 60, y + 25, esc(v), "t1", 11, "middle", ' font-weight="700" letter-spacing="0.06em"'))))
    return "\n".join(out)


def plane_legend(x, y):
    parts = [
        f'<circle cx="{x + 6}" cy="{y - 4}" r="6.4" fill="{GREY}"/>',
        text(x + 18, y, "code written against its posting's centre"),
        f'<circle cx="{x + 290}" cy="{y - 4}" r="5.6" class="surface" stroke="{GREY}" stroke-width="2.6"/>',
        text(x + 302, y, "code written against another centre"),
        f'<circle cx="{x + 6}" cy="{y + 18}" r="10" fill="none" class="line" stroke-width="1.4" stroke-dasharray="3 2"/>',
        text(x + 22, y + 22, "moved in this step"),
        f'<path d="M {x + 286} {y + 14} L {x + 294} {y + 22} M {x + 286} {y + 22} L {x + 294} {y + 14}" class="line" '
        f'stroke-width="1.6"/>',
        text(x + 302, y + 22, "centre kept, no posting routes by it"),
    ]
    return "\n".join(parts)


def draw_sizes(anim, snaps, x, y, w, split_limit, merge_limit, max_chunks, rows, colours=None, row_h=30):
    """One row per posting (in creation order): its size against the limits,
    then the counters."""
    label_w, bar_x0 = 44, x + 44
    bar_w = w - label_w - 70
    unit = bar_w / max_chunks
    out = [text(x, y + 12, "POSTINGS AND THEIR SIZE, IN CHUNKS", "t3", 12, extra=' letter-spacing="0.06em"')]
    top = y + 44
    bottom = top + rows * row_h
    for v, name, anchor in ((merge_limit, f"merge limit {merge_limit}", "end"),
                            (split_limit, f"split limit {split_limit}", "start")):
        lx = round(bar_x0 + v * unit, 1)
        out.append(f'<line class="axis" x1="{lx}" y1="{top - 8}" x2="{lx}" y2="{bottom - 6}" stroke-dasharray="4 3"/>')
        out.append(text(lx + (4 if anchor == "start" else -4), top - 14, name, "t3", 10.5, anchor))
    ids = sorted({p for s in snaps for p in s["postings"]})
    assert len(ids) <= rows, (ids, rows)
    for row, pid in enumerate(ids):
        ry = top + row * row_h

        def state(s, pid=pid):
            if pid not in s["postings"]:
                return None
            _, st, n = s["postings"][pid]
            return (st, n)

        def render(v, ry=ry, pid=pid):
            st, n = v
            col = colour(pid, colours)
            cls = "t1" if st == "active" else "t3"
            parts = [text(x, ry + 15, f"P{pid}", f"{cls} mono", 12, extra=' font-weight="700"')]
            if st == "retired":
                parts.append(text(bar_x0, ry + 15, "retired: never searched, holds no chunks", "t3", 11,
                                  extra=' font-style="italic"'))
                return "".join(parts)
            parts.append(f'<rect class="empty" x="{bar_x0}" y="{ry + 4}" width="{bar_w}" height="16" rx="4"/>')
            if n:
                op = ' fill-opacity="0.45"' if st == "draining" else ""
                parts.append(f'<rect x="{bar_x0}" y="{ry + 4}" width="{round(min(n, max_chunks) * unit, 1)}" height="16" '
                             f'rx="4" fill="{col}"{op}/>')
            note = f"{n}" + ("  draining" if st == "draining" else "")
            over = n > split_limit and st == "active"
            parts.append(text(bar_x0 + bar_w + 8, ry + 16, note, "ink-a mono" if over else "t2 mono", 11.5))
            return "".join(parts)

        out.append(anim.states([state(s) for s in snaps], render))

    cy = bottom + 30

    def counters(s):
        active = sum(1 for _, st, _ in s["postings"].values() if st == "active")
        retired = sum(1 for _, st, _ in s["postings"].values() if st == "retired")
        chunks = len(s["chunks"])
        foreign = sum(1 for v in s["chunks"].values() if v[2])
        return (active, retired, len(s["centres"]), s["moves"], f"{foreign} of {chunks}")

    labels = ["active postings", "retired", "centres stored", "chunk moves", "hollow chunks"]
    xs = [x, x + 112, x + 186, x + 286, x + 0]
    for i, lab in enumerate(labels[:4]):
        out.append(text(xs[i], cy, lab, "t3 mono", 10.5))
    out.append(text(x, cy + 54, labels[4], "t3 mono", 10.5))
    out.append(anim.states([counters(s) for s in snaps], lambda v: "".join(
        text(xs[i], cy + 24, str(v[i]), "t1 mono", 18, extra=' font-weight="700"') for i in range(4))
        + text(x, cy + 78, v[4], "t1 mono", 18, extra=' font-weight="700"')))
    return "\n".join(out)
