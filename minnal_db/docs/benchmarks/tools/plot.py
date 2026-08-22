#!/usr/bin/env python3
"""Render criterion means as gnuplot bar charts, matching the existing
minnal_db/docs/benchmarks/*.png style (log-scale y, rotated labels).

Usage: plot.py <tsv> <outdir> <chartspec.json>

chartspec.json: [{"file": "write.png", "title": "...", "ylabel": "...",
                  "unit": "us"|"ns"|"ms", "scale": "log"|"linear",
                  "group_by": "<regex with one capture group>",
                  "ylink": "<name>",
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

    if chart.get("group_by"):
        grx = re.compile(chart["group_by"])
        def group_of(n):
            m = grx.search(n)
            return m.group(1) if m else ""
        groups = {}
        for n in picked:
            groups.setdefault(group_of(n), []).append(n)
        # Order the groups by median so the chart still reads left-to-right
        # cheapest-to-dearest at the group level, while each group stays whole.
        def median(vals):
            v = sorted(rows[n] for n in vals)
            return v[len(v) // 2]
        picked = [n for _, names in sorted(groups.items(), key=lambda kv: median(kv[1])) for n in names]

    span = rows[picked[-1]] / rows[picked[0]] if picked and rows[picked[0]] else 0
    if span > 100 and chart.get("scale", "log") != "log":
        print(f"WARN {chart['file']}: {span:.0f}x span on a linear scale", file=sys.stderr)
    if span > 1000:
        print(f"WARN {chart['file']}: {span:.0f}x span — consider splitting by magnitude band", file=sys.stderr)

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
            f.write(f'{i}\t{rows[name] / div:.6g}\t"{short}"\n')

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
            yrange = f"[0:{hi * 1.15:.6g}]"
        else:
            pad = (hi - lo) * 0.15 or hi * 0.05
            yrange = f"[{max(0, lo - pad):.6g}:{hi + pad:.6g}]"
        ytics = "set ytics autofreq"

    # Labels are rotated vertical, so the bottom margin has to clear the
    # longest one; a fixed margin either clips long labels or leaves a band of
    # dead space under short ones.
    longest = max(len(lbl) for lbl in labels)
    bmargin = max(4, min(24, round(longest * 0.72)))

    gp = work / (chart["file"].replace(".png", ".gp"))
    gp.write_text(f"""
set terminal pngcairo size 960,820 font "DejaVu Sans,10"
set output "{outdir / chart["file"]}"
set title "{chart["title"]}" font "DejaVu Sans,12 bold" noenhanced
set ylabel "{chart["ylabel"]}"
set style data histograms
set style fill solid 1.0 border rgb "black"
set boxwidth 0.7
{logscale}
{ytics}
set grid ytics lc rgb "#dddddd"
set xtics rotate by -90 scale 0 noenhanced
set key off
set bmargin {bmargin}
set yrange {yrange}
plot "{dat}" using 2:xtic(3) with boxes lc rgb "#6a8fc4"
""")
    subprocess.run(["gnuplot", str(gp)], check=True)
    print(f"wrote {outdir / chart['file']} ({len(picked)} bars, {span:.0f}x span, {chart.get('scale','log')})")
