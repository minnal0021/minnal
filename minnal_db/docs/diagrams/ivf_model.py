"""A small model of a store's dynamic IVF partition, for the animated diagrams.

It follows the rules in minnal_db/src/semantic_search/ivf_split/ (and
Semantic-Search-Architecture.md §3.3), on 2-D points instead of embeddings:

- **insert**: a chunk is filed under the active posting whose centre is
  nearest, and its code is written against that posting's centre;
- **split** (a posting holds more than the split limit): balanced 2-means over
  its chunks; the two halves become new postings whose centres are the means of
  their members; the old posting is retired. Refused if the smaller half would
  be at or under the merge limit;
- **reassign** (after each split): a chunk of a half is a candidate if the old
  centre is nearer to it than its half's centre; a chunk of one of the
  `reassign_range` postings nearest the old centre is a candidate if the nearer
  half beats both the old centre and its own posting's centre. A candidate moves
  to its nearest active posting;
- **merge** (a posting at or under the merge limit): folded into the first of
  the postings nearest its centre with room (the two together stay under the
  split limit); the larger is kept, with its own centre. A chunk whose old
  centre is nearer than the kept posting's goes to its nearest active posting;
- **rebuild**: k-means over every chunk (k-means++ start, Lloyd iterations), as
  many centres as active postings; every chunk moves to its nearest new posting;
  every old posting is retired.

A move never rewrites a code: a chunk keeps the centre it was encoded against.
Centres are never deleted. The real partition works on the chunks' code
reconstructions; the model uses the points themselves.
"""

import math
import random


def dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def mean(points):
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


class Partition:
    def __init__(self, split_limit, merge_limit, reassign_range=64, zero=(50.0, 50.0)):
        self.split_limit = split_limit
        self.merge_limit = merge_limit
        self.reassign_range = reassign_range
        self.centres = [zero]  # centre 0: the zero vector, the root posting's centre
        self.postings = {0: {"centre": 0, "state": "active", "members": set(), "origin": "root"}}
        self.chunks = {}  # id -> {"pos", "posting", "centre"}
        self.next_posting = 1
        self.moves = 0
        self.moved = set()  # chunks moved by the last operation

    # ── helpers ──────────────────────────────────────────────────────────
    def active(self):
        return [p for p, v in self.postings.items() if v["state"] == "active"]

    def centre_of(self, posting):
        return self.centres[self.postings[posting]["centre"]]

    def nearest_active(self, pos, exclude=()):
        cands = [p for p in self.active() if p not in exclude]
        return min(cands, key=lambda p: (dist2(pos, self.centre_of(p)), p))

    def size(self, posting):
        return len(self.postings[posting]["members"])

    def _new_posting(self, centre_pos, origin):
        self.centres.append(centre_pos)
        pid = self.next_posting
        self.next_posting += 1
        self.postings[pid] = {"centre": len(self.centres) - 1, "state": "active", "members": set(), "origin": origin}
        return pid

    def _move(self, chunk, dest):
        c = self.chunks[chunk]
        if c["posting"] == dest:
            return
        self.postings[c["posting"]]["members"].discard(chunk)
        self.postings[dest]["members"].add(chunk)
        c["posting"] = dest
        self.moves += 1
        self.moved.add(chunk)

    # ── writes ───────────────────────────────────────────────────────────
    def insert(self, chunk, pos):
        self.moved = set()
        p = self.nearest_active(pos)
        self.chunks[chunk] = {"pos": pos, "posting": p, "centre": self.postings[p]["centre"]}
        self.postings[p]["members"].add(chunk)

    def delete(self, chunk):
        self.moved = set()
        c = self.chunks.pop(chunk)
        self.postings[c["posting"]]["members"].discard(chunk)

    # ── maintenance ──────────────────────────────────────────────────────
    def over_split_limit(self):
        return [p for p in self.active() if self.size(p) > self.split_limit]

    def at_merge_limit(self):
        return [p for p in self.active() if self.size(p) <= self.merge_limit and len(self.active()) > 1]

    def split(self, posting, seed=7):
        """Split `posting` with balanced 2-means. Returns the two halves, or
        None if the split is refused."""
        self.moved = set()
        members = sorted(self.postings[posting]["members"])
        pts = [self.chunks[m]["pos"] for m in members]
        halves = balanced_two_means(pts, seed)
        groups = [[m for m, h in zip(members, halves) if h == k] for k in (0, 1)]
        if min(len(g) for g in groups) <= self.merge_limit:
            return None
        children = []
        for g in groups:
            child = self._new_posting(mean([self.chunks[m]["pos"] for m in g]), "split")
            for m in g:
                self.postings[posting]["members"].discard(m)
                self.postings[child]["members"].add(m)
                self.chunks[m]["posting"] = child
            children.append(child)
        self.postings[posting]["state"] = "retired"
        return children

    def reassign(self, parent, children):
        """LIRE's reassignment after `parent` split into `children`. Returns the
        chunks that moved."""
        self.moved = set()
        c_p = self.centre_of(parent)
        others = [p for p in self.active() if p not in children]
        neighbours = sorted(others, key=lambda p: dist2(c_p, self.centre_of(p)))[: self.reassign_range]
        cands = []
        for h in children:
            for m in self.postings[h]["members"]:
                x = self.chunks[m]["pos"]
                if dist2(x, c_p) < dist2(x, self.centre_of(h)):
                    cands.append(m)
        for n in neighbours:
            for m in self.postings[n]["members"]:
                x = self.chunks[m]["pos"]
                best_half = min(dist2(x, self.centre_of(h)) for h in children)
                if best_half < dist2(x, c_p) and best_half < dist2(x, self.centre_of(n)):
                    cands.append(m)
        for m in sorted(cands):
            self._move(m, self.nearest_active(self.chunks[m]["pos"]))
        return set(self.moved)

    def merge(self, posting):
        """Merge `posting` into a nearby posting with room. Returns (drained,
        kept), or None if no posting has room."""
        self.moved = set()
        n = self.size(posting)
        others = sorted((p for p in self.active() if p != posting),
                        key=lambda p: dist2(self.centre_of(posting), self.centre_of(p)))
        partner = next((q for q in others if n + self.size(q) < self.split_limit), None)
        if partner is None:
            return None
        drained, kept = (posting, partner) if self.size(partner) >= n else (partner, posting)
        c_d, c_k = self.centre_of(drained), self.centre_of(kept)
        self.postings[drained]["state"] = "retired"  # no longer a destination
        for m in sorted(self.postings[drained]["members"]):
            x = self.chunks[m]["pos"]
            dest = kept if dist2(x, c_d) >= dist2(x, c_k) else self.nearest_active(x)
            self._move(m, dest)
        return drained, kept

    def rebuild(self, k=None, seed=3, iterations=20):
        """Refit the partition with k-means over every chunk; every chunk moves
        to its nearest new posting. Returns the new postings."""
        self.moved = set()
        old = self.active()
        k = k or len(old)
        ids = sorted(self.chunks)
        pts = [self.chunks[i]["pos"] for i in ids]
        centres = kmeans(pts, k, seed, iterations)
        # A centre no chunk is nearest is dropped.
        used = sorted({min(range(len(centres)), key=lambda j: dist2(p, centres[j])) for p in pts})
        new = [self._new_posting(centres[j], "rebuild") for j in used]
        for p in old:
            self.postings[p]["state"] = "draining"
        for i in ids:
            dest = min(new, key=lambda q: (dist2(self.chunks[i]["pos"], self.centre_of(q)), q))
            self._move(i, dest)
        for p in old:
            self.postings[p]["state"] = "retired"
        return new

    # ── what a picture needs ─────────────────────────────────────────────
    def snapshot(self):
        return {
            "chunks": {i: (c["pos"], c["posting"], c["centre"] != self.postings[c["posting"]]["centre"], i in self.moved)
                       for i, c in self.chunks.items()},
            "postings": {p: (v["centre"], v["state"], len(v["members"])) for p, v in self.postings.items()},
            "centres": list(self.centres),
            "moves": self.moves,
        }


def balanced_two_means(points, seed, trials=3, iters=100, lambda_factor=100):
    """SPTAG's balanced 2-means with mean centres (cluster/split.rs), every
    point sampled. Returns a 0/1 label per point."""
    rng = random.Random(seed)
    n = len(points)
    best = None
    for _ in range(trials):
        a, b = rng.sample(range(n), 2)
        c = [points[a], points[b]]
        lab = [0 if dist2(p, c[0]) <= dist2(p, c[1]) else 1 for p in points]
        total = sum(dist2(p, c[l]) for p, l in zip(points, lab))
        if best is None or total < best[0]:
            best = (total, c, lab)
    _, c, lab = best
    larger = 0 if lab.count(0) >= lab.count(1) else 1
    ds = [math.sqrt(dist2(p, c[larger])) for p, l in zip(points, lab) if l == larger]
    lam = min((max(ds) - sum(ds) / len(ds)) / n if ds else 0.0, 1.0 / (lambda_factor * n))
    for _ in range(iters):
        sizes = [lab.count(0), lab.count(1)]
        new = [min((0, 1), key=lambda k: dist2(p, c[k]) + lam * sizes[k]) for p in points]
        groups = [[p for p, l in zip(points, new) if l == k] for k in (0, 1)]
        if not groups[0] or not groups[1]:
            break
        moved = sum(dist2(c[k], mean(groups[k])) for k in (0, 1))
        c = [mean(groups[0]), mean(groups[1])]
        lab = new
        if moved < 1e-3:
            break
    return [0 if dist2(p, c[0]) <= dist2(p, c[1]) else 1 for p in points]


def kmeans(points, k, seed, iterations):
    """k-means++ start, then Lloyd iterations (cluster/kmeans.rs)."""
    rng = random.Random(seed)
    centres = [points[rng.randrange(len(points))]]
    while len(centres) < k:
        d = [min(dist2(p, c) for c in centres) for p in points]
        r, acc = rng.random() * sum(d), 0.0
        for p, w in zip(points, d):
            acc += w
            if acc >= r:
                centres.append(p)
                break
    for _ in range(iterations):
        groups = [[] for _ in centres]
        for p in points:
            groups[min(range(k), key=lambda j: dist2(p, centres[j]))].append(p)
        centres = [mean(g) if g else centres[j] for j, g in enumerate(groups)]
    return centres
