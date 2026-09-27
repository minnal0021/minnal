#!/usr/bin/env python3
"""Render Criterion means as the gnuplot bar charts in minnal_db/docs/benchmarks/.

Usage: plot.py <tsv> <outdir> <chartspec.json>

chartspec.json: [{"file": "write.png", "title": "...", "ylabel": "...",
                  "unit": "us"|"ns"|"ms", "scale": "log"|"linear",
                  "group_by": "<regex with one capture group>",
                  "ylink": "<name>",
                  "color_by": "<regex with one capture group>"
                              | {"<category>": "<regex>", ...},
                  "legend": {"<captured>": "<legend text>", ...},
                  "cases": ["<regex>", ...]}]

Bars are sorted ASCENDING BY VALUE, whatever order the patterns are written in.

`group_by` keeps a compared category contiguous: bars are grouped by the
captured substring (tier, usually), the groups are ordered by their median, and
bars ascend within each group. Without it, sorting by value alone interleaves
the very categories a chart exists to compare — an l1 bar landing between two
memtable bars destroys the side-by-side read even though the ordering is
technically ascending.

`ylink` gives every chart sharing that name one common y-range, so charts split
apart for readability can still be compared bar-height to bar-height. Splitting
without it silently rescales each half and invites the wrong conclusion.

`color_by` colours each bar by the captured substring and draws a legend, so
the compared categories (in memory against on disk, say) read by colour rather
than by decoding labels. It defaults to `group_by`. As a map of category to
regex, the first regex that matches a benchmark name picks its category. `legend` maps a captured
value to its legend text, and its order fixes the colour slots: list the same
categories in the same order on every chart and each keeps one colour
throughout the report. Colours are the first three slots of a palette validated
for colour-blind separation, so a chart can have at most three categories.

Every bar carries its value, because a PNG has no hover to read it from.

`scale` defaults to "log"; use "linear" for any chart whose bars share a
magnitude, which is the only way a few-percent spread shows up. A chart spanning
more than ~1000x warns that it should be split into per-band charts instead.
"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

UNITS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}


def log_ticks(lo, hi):
    """1-2-5 ticks spanning [lo, hi].

    gnuplot labels only decades on a log axis, so a chart running 0.19 to 9.5
    gets ticks at 1..9 and nothing at all under 1 — every sub-microsecond bar
    then floats with no reference line. Generating the ticks explicitly keeps a
    labelled gridline within a factor of ~2 of every bar.
    """
    ticks, decade = [], 10.0 ** math.floor(math.log10(lo))
    while decade <= hi * 10:
        for mult in (1, 2, 5):
            v = decade * mult
            if lo / 1.5 <= v <= hi * 1.5:
                ticks.append(v)
        decade *= 10
    return ticks

tsv, outdir, spec_path = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
outdir.mkdir(parents=True, exist_ok=True)
work = Path(tsv).parent

rows = {}
for line in Path(tsv).read_text().splitlines():
    name, mean, lo, hi = line.split("\t")
    rows[name] = float(mean)

charts = json.loads(Path(spec_path).read_text())


def resolve(chart):
    """Benchmark names this chart covers, in no particular order."""
    out = []
    for pat in chart["cases"]:
        rx = re.compile(pat)
        out.extend(n for n in rows if rx.fullmatch(n))
    return set(out)


# Charts sharing a `ylink` are drawn on one common y-range so that a split made
# for readability cannot be misread: equal bar heights mean equal values across
# both frames.
links = {}
for chart in charts:
    link = chart.get("ylink")
    if not link:
        continue
    vals = [rows[n] / UNITS[chart["unit"]] for n in resolve(chart)]
    if vals:
        lo, hi = links.get(link, (min(vals), max(vals)))
        links[link] = (min(lo, min(vals)), max(hi, max(vals)))

for chart in charts:
    div = UNITS[chart["unit"]]
    picked = []
    for pat in chart["cases"]:
        rx = re.compile(pat)
        hits = [n for n in rows if rx.fullmatch(n)]
        if not hits:
            print(f"WARN {chart['file']}: no match for {pat!r}", file=sys.stderr)
        picked.extend(hits)

    # Ascending by measured value, never by benchmark name. Alphabetical order
    # scatters comparable magnitudes across the axis and makes a chart that has
    # to be read bar-by-bar against the table; sorted, the shape *is* the
    # finding. Ties keep a stable name order so redeploys don't reshuffle.
    picked = sorted(set(picked), key=lambda n: (rows[n], n))

    PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]
    # `color_by` is a regex whose capture names the category, or a map of
    # category -> regex where the first regex to match a benchmark name wins.
    cby = chart.get("color_by", chart.get("group_by"))
    legend = chart.get("legend", {})

    def category_of(n):
        if isinstance(cby, dict):
            return next((cat for cat, rx in cby.items() if re.search(rx, n)), "")
        m = re.search(cby, n)
        return m.group(1) if m else ""

    # The legend map fixes each category's colour slot, so a category keeps its
    # colour on every chart even when a chart shows only some of them.
    slots = list(legend)
    series = []  # categories present in this chart, in slot order
    if cby:
        found = {category_of(n) for n in picked}
        slots += sorted(k for k in found if k not in slots)
        series = [k for k in slots if k in found]
        if len(slots) > len(PALETTE):
            sys.exit(f"{chart['file']}: {len(slots)} colour categories, palette has {len(PALETTE)}")

    def colour_of(n):
        return slots.index(category_of(n)) if cby else 0

    if chart.get("group_by"):
        grx = re.compile(chart["group_by"])
        def group_of(n):
            m = grx.search(n)
            return m.group(1) if m else ""
        groups = {}
        for n in picked:
            groups.setdefault(group_of(n), []).append(n)
        # Order the groups by their lower median so the chart still reads
        # left-to-right cheapest-to-dearest at the group level, while each group
        # stays whole.
        def median(vals):
            v = sorted(rows[n] for n in vals)
            return v[(len(v) - 1) // 2]
        # When colour marks a different dimension than the grouping (tier within
        # a key count, say), bars inside a group follow the legend order, so
        # "in memory" always stands left of "on disk" and a pair reads the same
        # way everywhere. An inversion then shows as a shorter right-hand bar
        # instead of silently swapping the pair.
        if cby and cby != chart.get("group_by"):
            for names in groups.values():
                names.sort(key=lambda n: (colour_of(n), rows[n], n))
        picked = [n for _, names in sorted(groups.items(), key=lambda kv: median(kv[1])) for n in names]

    span = rows[picked[-1]] / rows[picked[0]] if picked and rows[picked[0]] else 0
    if span > 100 and chart.get("scale", "log") != "log":
        print(f"WARN {chart['file']}: {span:.0f}x span on a linear scale", file=sys.stderr)
    if span > 1000:
        print(f"WARN {chart['file']}: {span:.0f}x span — consider splitting by magnitude band", file=sys.stderr)

    def fmt(v):
        return f"{v:.3g}" if v < 1000 else f"{v:,.0f}"

    label = chart.get("strip", "")
    relabel = chart.get("relabel", [])
    dat = work / (chart["file"].replace(".png", ".dat"))
    labels = []
    with dat.open("w") as f:
        for i, name in enumerate(picked):
            short = name
            for pat, rep in relabel:
                short = re.sub(pat, rep, short)
            if label:
                short = re.sub(label, "", short)
            short = short or name
            labels.append(short)
            f.write(f'{i}\t{rows[name] / div:.6g}\t"{short}"\t{colour_of(name)}\t"{fmt(rows[name] / div)}"\n')

    # Linear is the default for a chart whose bars share a magnitude: it is the
    # only way a few-percent spread is visible at all. Log is for charts that
    # deliberately span bands.
    logscale = "set logscale y" if chart.get("scale", "log") == "log" else "unset logscale y"
    # A linear axis anchored at zero wastes the whole frame when every bar is
    # near the same value, so start the range just below the smallest bar.
    if chart.get("ylink") in links:
        lo, hi = links[chart["ylink"]]
    else:
        lo = min(rows[n] for n in picked) / div
        hi = max(rows[n] for n in picked) / div
    if chart.get("scale", "log") == "log":
        ylo, yhi = lo * 0.7, hi * 1.4
        yrange = f"[{ylo:.6g}:{yhi:.6g}]"
        ticks = ", ".join(f'"{v:g}" {v:g}' for v in log_ticks(lo, hi))
        ytics = f"set ytics ({ticks})" if ticks else "set ytics autofreq"
    else:
        if chart.get("zero_based"):
            # The finding is "no meaningful difference", and a truncated axis
            # would turn sub-1% noise into a staircase the reader reads as a
            # trend. Anchoring at zero makes equal values look equal.
            yrange = f"[0:{hi * 1.18:.6g}]"
        else:
            pad = (hi - lo) * 0.15 or hi * 0.05
            yrange = f"[{max(0, lo - pad):.6g}:{hi + pad:.6g}]"
        ytics = "set ytics autofreq"

    # Rotated labels are read sideways, one character at a time. Where the
    # labels are short enough to fit under their own bars, stack their
    # slash-separated parts on separate lines and leave them horizontal — a
    # two-line "batched_matrix / 4" under the bar reads instantly, the same
    # string rotated 90 degrees does not. Rotate only when the bars are too
    # narrow for that.
    # 7.6 px/char is DejaVu Sans 10 measured against the widest glyphs, and the
    # 0.9 factor leaves a gutter so neighbouring labels cannot touch. Both are
    # deliberately pessimistic: a label that overflows collides with the one
    # next to it and is worse than an honestly rotated one.
    PX_PER_CHAR, PLOT_PX = 7.6, 880.0
    per_bar_px = PLOT_PX / max(1, len(labels))
    widest_part = max((max(len(part) for part in lbl.split("/")) for lbl in labels), default=0)
    horizontal = widest_part * PX_PER_CHAR <= per_bar_px * 0.9

    if horizontal:
        deepest = max(lbl.count("/") for lbl in labels) + 1
        bmargin = max(3, deepest + 2)
        xtics_rotate = "set xtics scale 0 noenhanced"
        with dat.open("w") as f:
            for i, lbl in enumerate(labels):
                v = rows[picked[i]] / div
                f.write(f'{i}\t{v:.6g}\t"{lbl.replace("/", chr(92) + "n")}"\t{colour_of(picked[i])}\t"{fmt(v)}"\n')
    else:
        longest = max(len(lbl) for lbl in labels)
        bmargin = max(4, min(24, round(longest * 0.72)))
        xtics_rotate = "set xtics rotate by -90 scale 0 noenhanced"

    if series:
        key = 'set key at graph 0.5, 1.01 center bottom horizontal reverse Left samplen 2 width 2\nset tmargin 5.5'
        legend_entries = "".join(
            f', \\\n     keyentry with boxes fill solid 1.0 border rgb "black" lc rgb "{PALETTE[i]}" title "{legend.get(k, k)}"'
            for k in series for i in [slots.index(k)])
    else:
        key, legend_entries = "set key off", ""
    # Bar colour comes from column 4 (the series index) through a palette whose
    # integer points are exactly the series colours.
    palette_defs = (f"set palette defined ({', '.join(f'{i} \"{c}\"' for i, c in enumerate(PALETTE))})\n"
                    f"set cbrange [0:{len(PALETTE) - 1}]\nunset colorbox")

    title_offset = "offset 0,1.2" if series else ""

    gp = work / (chart["file"].replace(".png", ".gp"))
    gp.write_text(f"""
set terminal pngcairo size 960,820 font "DejaVu Sans,10"
set output "{outdir / chart["file"]}"
set title "{chart["title"]}" font "DejaVu Sans,12 bold" noenhanced {title_offset}
set ylabel "{chart["ylabel"]}"
set style data histograms
set style fill solid 1.0 border rgb "black"
set boxwidth 0.7
{logscale}
{ytics}
set grid ytics lc rgb "#dddddd"
{xtics_rotate}
{key}
set bmargin {bmargin}
set yrange {yrange}
{palette_defs}
plot "{dat}" using 1:2:4:xtic(3) with boxes lc palette notitle, \
     "{dat}" using 1:2:5 with labels offset 0,0.8 font "DejaVu Sans,9" notitle{legend_entries}
""")
    subprocess.run(["gnuplot", str(gp)], check=True)
    print(f"wrote {outdir / chart['file']} ({len(picked)} bars, {span:.0f}x span, {chart.get('scale','log')})")
