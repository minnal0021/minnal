"""Draw the M3-pre report's charts for one model from {model}/data.json.

    python3 charts.py gemma      # writes gemma/*.svg
    python3 charts.py qwen       # writes qwen/*.svg

Colour slots, fixed across every chart:
  blue   = shuffled order / 1-bit / the recommended lifecycle
  orange = corpus order   / 2-bit / today's bundled file
  aqua   = drifting order / 4-bit
  blue ramp (light -> dark) = target posting size 1024 -> 128
"""
import json
from pathlib import Path
from svgchart import line_chart, hbar_chart

import sys

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma"
HERE = Path(__file__).parent / MODEL
D = json.load(open(HERE / "data.json"))
ORDERS = [("shuffled", "f0", "shuffled"), ("corpus", "f1", "corpus order"), ("drifting", "f2", "drifting topics")]
LIFE = [("C", 0, "C: grow from one posting"),
        ("A+C", 2000, "A+C: float seed at 2k chunks"), ("A+C", 10000, "A+C: float seed at 10k"), ("A+C", 30000, "A+C: float seed at 30k"),
        ("B+C", 2000, "B+C: code seed at 2k chunks"), ("B+C", 10000, "B+C: code seed at 10k"), ("B+C", 30000, "B+C: code seed at 30k")]


KEY = [("C", "= grow from one posting: split any posting past 2 × target into two (2-means on its codes)."),
       ("A+C", "= keep float embeddings until N chunks, k-means on the floats, drop them, then split as C."),
       ("B+C", "= stay one flat posting until N chunks, k-means on the 1-bit codes, then split as C."),
       ("k", "= neighbouring postings re-checked after each split (chunks move to a closer new posting).")]
KEY_STATIC = [("static k-means", "= fitted once on the whole corpus's floats: the best a partition of this size can do."),
              ("bundled file", f"= today's 256 general-purpose {MODEL} centroids (fitted on ELI5).")]


# Chart titles state a finding, so they are checked per model and dataset. A chart
# not listed here uses the default title in its function (written for gemma).
TITLES = {
    ("qwen", "scifact", "lifecycles"): "SciFact (qwen, 300 queries, noisier): every lifecycle trails static k-means, by 0.7–4.5 points",
    ("qwen", "scifact", "churn"): "SciFact (qwen): after churn C ends within 2.1 points of a fresh fit",
    ("qwen", "fiqa", "lifecycles"): "FiQA (qwen): every lifecycle is within 1.1 points of static k-means",
}


def title(ds, chart, default):
    return TITLES.get((MODEL, ds, chart), default)


def run(ds, **kw):
    return next(r for r in D[ds]["stage1"] if all(r[k] == v for k, v in kw.items()))


def chart_target(ds):
    """Recall against entries read (absolute), one line per target posting size."""
    xmax = 25 if ds == "fiqa" else 2.5
    def pts(r):
        sn = r["snaps"]["full"]
        return [(x * sn["E"] / 1000, y) for x, y, _ in sn["curve"] if x * sn["E"] / 1000 <= xmax * 1.02]
    series = []
    for i, t in enumerate((1024, 512, 256, 128)):
        r = run(ds, variant="C", n_seed=0, target=t, k=8, order="shuffled")
        series.append(dict(name=f"target {t} (K {r['snaps']['full']['K']})", cls=f"lr{i}", pts=pts(r)))
    b = run(ds, variant="bundled", order="shuffled")
    series.append(dict(name="bundled file (K 256)", cls="l1", pts=pts(b)))
    st = run(ds, variant="static", target=128, order="shuffled")
    series.append(dict(name="static k-means, target 128", cls="l0", dash=True, label=False, pts=pts(st)))
    lo = 0.7 if ds == "fiqa" else 0.6
    ticks = [0, 5, 10, 15, 20, 25] if ds == "fiqa" else [0, 0.5, 1, 1.5, 2, 2.5]
    line_chart(HERE / f"{ds}-target-size.svg",
               f"{ds_name(ds)}: smaller postings find more of the right documents per entry read",
               "Grow from one posting, reassign 8 neighbours, shuffled order. Higher and further left is better.",
               series, "entries read per query (thousands; one value-log read each)", "ANN recall@10", (0, xmax), (lo, 1.0),
               xticks=ticks, yfmt="{:.2f}", xfmt="{:g}k", legend_cols=3,
               desc=f"{ds}: recall@10 against entries read for target posting sizes 128 to 1024, the bundled file and static k-means",
               notes=[("target", "= posting size in entries: a posting splits past 2 × target, so K grows with the namespace."),
                      ("ANN recall@10", "= share of the exact (float, unpartitioned) top 10 the index returns."),
                      ("entry", "= one (posting, document) key holding that document's chunk codes in the posting, not its text.")] + KEY_STATIC)


def chart_lifecycle(ds):
    """Every lifecycle against static k-means, at 10% of the namespace scanned, target 128."""
    groups = []
    for v, s, label in LIFE:
        bars = []
        for order, cls, _ in ORDERS:
            r = run(ds, variant=v, n_seed=s, target=128, k=8, order=order)
            st = run(ds, variant="static", target=128, order=order)
            a, b = r["snaps"]["full"]["r10_at"]["0.1"], st["snaps"]["full"]["r10_at"]["0.1"]
            bars.append(dict(v=100 * (a - b), cls=cls, label=None, nan="never seeded: namespace smaller than N (full scan)"))
        groups.append(dict(name=label, bars=bars))
    lo = -3 if ds == "fiqa" else -5
    hbar_chart(HERE / f"{ds}-lifecycles.svg",
               title(ds, "lifecycles", {"fiqa": "FiQA: every lifecycle is within 1.6 points of static k-means",
                                        "scifact": "SciFact (300 queries, noisier): C trails by 3 points when topics arrive in turn"}[ds]),
               "Recall@10 at 10% of entries scanned, minus static k-means fitted on the whole corpus (points). Target 128, k = 8.",
               groups, "difference from static k-means (recall points; 0 = as good)", (lo, 1), fmt="{:+.1f}",
               legend=[(c, t) for _, c, t in ORDERS], refs=[dict(x=-1, label="−1 pt")],
               desc=f"{ds}: difference in recall from static k-means for seven lifecycles and three insertion orders",
               notes=KEY + KEY_STATIC[:1] + [("orders", "= documents arrive shuffled, in corpus order, or one topic at a time (8 topics).")])


def _unused_chart_moves(ds):   # kept as a table in the report: seven near-equal groups
    groups = []
    for v, s, label in LIFE:
        bars = [dict(v=run(ds, variant=v, n_seed=s, target=128, k=k, order="shuffled")["moves"], cls=c, label=None)
                for k, c in ((0, "fr1"), (2, "fr2"), (8, "fr3"))]
        groups.append(dict(name=label, bars=bars))
    hbar_chart(HERE / f"{ds}-moves.svg",
               f"{ds_name(ds)}: each chunk's key moves about twice over the namespace's life",
               "Key moves per inserted chunk (write amplification of maintenance), target 128, shuffled order.",
               groups, "key moves per inserted chunk", (0, 2.2), fmt="{:.2f}",
               legend=[("fr1", "reassign k = 0"), ("fr2", "k = 2"), ("fr3", "k = 8")],
               desc=f"{ds}: key moves per insert for each lifecycle and reassign setting",
               notes=KEY)


POLICY = [("keep", "f6", "keep: a moved code keeps its old centre"),
          ("code", "f4", "code: re-encoded from its own code"),
          ("fresh", "f5", "fresh: re-encoded from the float (stored or re-embedded)")]
WIDTH_KEY = [("width", "= bits per dimension of a chunk code: 1 is today's; 2 and 4 hold more of the vector."),
             ("Pass-1 recall", "= share of the exact top 1,000 candidates Pass 1 hands to Pass 2.")]


def s2(ds, **kw):
    return next(r for r in D[ds]["stage2"] if all(r[k] == v for k, v in kw.items()))


def chart_fidelity(ds):
    rows = {(f["bits"], f["encoded_against"]): f for f in D[ds]["fidelity"]}
    groups = []
    for b in (1, 2, 4):
        f = rows[(b, "bundled")]
        groups.append(dict(name=f"{b}-bit code", bars=[
            dict(v=f["mean_f"], cls="fr1", label="direction kept, ⟨ō, o⟩"),
            dict(v=f["nearest_agree_static"], cls="fr3", label="same nearest centre as the float")]))
    hbar_chart(HERE / f"{ds}-code-fidelity.svg",
               f"{ds_name(ds)}: a 4-bit code is almost the vector; a 1-bit code keeps about 80% of it",
               "Every chunk, coded against its nearest bundled centre; nearest centre checked against a fresh k-means set.",
               groups, "fraction (1 = identical to the float)", (0, 1), fmt="{:.3f}", label_w=110,
               legend=[("fr1", "direction kept: ⟨ō, o⟩, the cosine between code and residual"), ("fr3", "picks the same nearest centre as its float")],
               notes=WIDTH_KEY[:1], desc=f"{ds}: reconstruction fidelity of 1-, 2- and 4-bit codes")


def chart_rebuild(ds):
    rb = {(r["mbits"], r["sbits"], r["policy"], r["fit"]): r for r in D[ds]["rebuild"]}
    target = next(r for r in D[ds]["rebuild"] if r["fit"] == "float")
    today = next(r for r in D[ds]["rebuild"] if r["fit"] is None)
    groups = [dict(name="k-means on floats", bars=[dict(v=target["r10_at"]["0.1"], cls="f5")]),
              dict(name="today's bundled file", bars=[dict(v=today["r10_at"]["0.1"], cls="f1")])]
    for b in (1, 2, 4):
        groups.append(dict(name=f"k-means on {b}-bit codes", bars=[dict(v=rb[(b, 1, "keep", "codes")]["r10_at"]["0.1"], cls="f0")]))
    hbar_chart(HERE / f"{ds}-rebuild.svg",
               f"{ds_name(ds)}: re-clustering from 1-bit codes is as good as from floats",
               f"Full rebuild (M4) of an index coded against the bundled centres. Recall@10 at 10% of entries scanned, K {target['K']}.",
               groups, "ANN recall@10 at 10% of the namespace scanned", (0, 1), fmt="{:.3f}", label_w=170,
               notes=[("codes", "= k-means on each chunk's reconstruction from its code; codes keep their old centre."),
                      ("floats", "= k-means on the original embeddings: needs stored floats or re-embedding.")],
               desc=f"{ds}: recall of partitions rebuilt from floats and from 1-, 2- and 4-bit codes")


def chart_reencode(ds, order="shuffled"):
    groups = []
    for b in (1, 2, 4):
        bars = []
        for pol, cls, _ in POLICY:
            r = s2(ds, variant="C", target=128, k=8, order=order, sbits=b, mbits=b, policy=pol)
            bars.append(dict(v=r["snaps"]["full"]["r10_at"]["0.1"], cls=cls))
        groups.append(dict(name=f"{b}-bit codes", bars=bars))
    hbar_chart(HERE / f"{ds}-reencode.svg",
               f"{ds_name(ds)}: re-encoding a moved code from itself hurts below 4 bits",
               f"Grow from one posting (C), target 128, k = 8, {order} order. Recall@10 at 10% of entries scanned.",
               groups, "ANN recall@10 at 10% of the namespace scanned", (0, 1), fmt="{:.3f}", label_w=110,
               legend=[(c, t) for _, c, t in POLICY], notes=WIDTH_KEY[:1],
               desc=f"{ds}: recall after many splits for three re-encode policies at three code widths")


def chart_search_width(ds, order="shuffled"):
    groups = []
    for b in (1, 2, 4):
        r = s2(ds, variant="C", target=128, k=8, order=order, sbits=b, mbits=b, policy="keep")
        d = r["snaps"]["full"]["default"]
        groups.append(dict(name=f"{b}-bit search", bars=[dict(v=d["p1_recall"], cls="fr1", label="Pass-1 recall"),
                                                          dict(v=d["r100"], cls="fr3", label="final top-100 recall")]))
    hbar_chart(HERE / f"{ds}-search-width.svg",
               f"{ds_name(ds)}: wider codes fix Pass 1's near-ties, not the final ranking",
               "Grow from one posting, target 128, k = 8, codes keep their centre, default budget (70k entries).",
               groups, "recall against the exact pipeline", (0, 1), fmt="{:.3f}", label_w=110,
               legend=[("fr1", "Pass-1 recall (exact top 1,000)"), ("fr3", "final top-100 recall")], notes=WIDTH_KEY[1:],
               desc=f"{ds}: Pass-1 and final recall by search code width")


PROBE = [("default", "f1", "today (70k)"),
         ("pb10_pp30_bc40k_mc1024", "fr1", "10% of entries"),
         ("pb20_pp30_bc40k_mc1024", "fr2", "20% of entries"),
         ("pb30_pp30_bc40k_mc1024", "fr3", "30% of entries")]
PROBE_KEY = [("floor 20k", "= derived, not a separate run: below the floor it reads everything, as today's budget does at those sizes."),
             ("scaled", "= probe_budget_entries = min(p · E, ceiling), max_probes = 30% of K; no ceiling was reached here."),
             ("E", "= entries in the namespace; K = postings. Grow from one posting, target 128, k = 8.")]


def chart_probe(ds, order="shuffled", metric="r10"):
    r = next(x for x in D[ds]["stage3"] if x["order"] == order)
    sizes = [k for k in r["snaps"]]
    xs = {k: (r["snaps"][k]["chunks"]) for k in sizes}
    series = []
    m, div = ("r10", 1) if metric == "r10" else ("entries", 1000)
    for key, cls, name in PROBE:
        pts = [(xs[k], r["snaps"][k]["settings"][key][m] / div) for k in sizes]
        series.append(dict(name=name, cls=cls.replace("f", "l", 1), pts=pts))
    # 30% with a 20k floor: below the floor it reads the whole namespace (today's run at
    # these sizes does exactly that); above it, it is the 30% budget.
    floor = []
    for k in sizes:
        sn = r["snaps"][k]
        key = "default" if 0.3 * sn["E"] < 20_000 else "pb30_pp30_bc40k_mc1024"
        assert key != "default" or sn["E"] <= 70_000
        floor.append((xs[k], sn["settings"][key][m] / div))
    series.append(dict(name="30%, floor 20k", cls="l2", dash=True, pts=floor))
    top = max(xs.values())
    xt = [1000, 5000, 10000, 20000] + ([top] if top > 30000 else [])
    if metric == "r10":
        line_chart(HERE / f"{ds}-probe-recall.svg",
                   f"{ds_name(ds)}: a share-of-namespace budget gives up exactness while the namespace is small",
                   f"Recall@10 as the namespace grows, {order} order. Today's budget scans small namespaces completely.",
                   series, "chunks in the namespace (log scale)", "ANN recall@10", (900, top * 1.1), (0.4 if order == "drifting" else 0.6, 1.0),
                   xlog=True, xticks=xt, xfmt="{:,.0f}", yfmt="{:.2f}", legend_cols=2, notes=PROBE_KEY,
                   desc=f"{ds}: recall@10 against namespace size for today's budget and three scaled budgets")
    else:
        line_chart(HERE / f"{ds}-probe-cost.svg",
                   f"{ds_name(ds)}: entries read per query as the namespace grows",
                   f"Same runs as the recall chart, {order} order. Cost stands in for latency (one value-log read per entry).",
                   series, "chunks in the namespace (log scale)", "entries read per query (thousands)", (900, top * 1.1),
                   (0, 75 if ds == "fiqa" else 9), xlog=True, xticks=xt, xfmt="{:,.0f}", yfmt="{:.1f}", legend_cols=2, notes=PROBE_KEY,
                   desc=f"{ds}: entries read against namespace size for today's budget and three scaled budgets")


CHURN = [("delete-half", "delete a random half"), ("delete-topics", "delete 2 of 8 topics"),
         ("turnover", "replace a random half"), ("topic-turnover", "replace topics 1–4 by 5–8")]


def chart_churn(ds, target=128, k=8):
    groups = []
    for sc, label in CHURN:
        r = next(x for x in D[ds]["stage4"] if x["scenario"] == sc and x["target"] == target and x["k"] == k)
        ph = r["phases"]
        cls = ["fr1", "fr2", "fr3"] if len(ph) == 3 else ["fr1", "fr3"]
        groups.append(dict(name=label, bars=[dict(v=100 * (p["dyn"] - p["static"]), cls=c) for p, c in zip(ph, cls)]))
    hbar_chart(HERE / f"{ds}-churn.svg",
               title(ds, "churn", f"{ds_name(ds)}: deletes and turnover leave C within about a point of a fresh fit"),
               f"Recall@10 at 10% of entries read, minus static k-means fitted on the surviving chunks (points). C, target {target}, k = {k}.",
               groups, "difference from static k-means (recall points; 0 = as good)", (-5, 1), fmt="{:+.1f}", label_w=190,
               legend=[("fr1", "after growing (before churn)"), ("fr2", "half replaced"), ("fr3", "after the deletes / all replaced")],
               refs=[dict(x=-1, label="−1 pt")],
               notes=[("replace", "= delete old documents and insert new ones, 64 at a time, until the old set is gone."),
                      ("merge", "= a posting under target / 4 moves its chunks to its nearest neighbours; codes keep their centre."),
                      ("static", "= k-means fitted on the chunks still present, with the same number of postings.")],
               desc=f"{ds}: recall gap to a fresh static fit before and after four churn scenarios")


def ds_name(ds):
    return {"fiqa": "FiQA", "scifact": "SciFact"}[ds] + f" ({MODEL})"


if __name__ == "__main__":
    for f in HERE.glob("*.svg"):
        f.unlink()
    if "fiqa" in D:
        chart_target("fiqa")
        chart_lifecycle("fiqa")
        chart_fidelity("fiqa")
        chart_rebuild("fiqa")
        chart_reencode("fiqa")
        chart_search_width("fiqa")
        chart_probe("fiqa")
        chart_probe("fiqa", metric="cost")
        chart_churn("fiqa")
    if "scifact" in D:
        chart_lifecycle("scifact")
        chart_probe("scifact")
        if "fiqa" not in D:      # SciFact is the only dataset so far: draw the rest from it too
            chart_target("scifact")
            chart_fidelity("scifact")
            chart_rebuild("scifact")
            chart_reencode("scifact")
            chart_search_width("scifact")
            chart_probe("scifact", metric="cost")
            chart_churn("scifact")
