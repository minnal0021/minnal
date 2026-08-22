#!/usr/bin/env python3
"""Render criterion means as gnuplot bar charts, matching the existing
minnal_db/docs/benchmarks/*.png style (log-scale y, rotated labels).

Usage: plot.py <tsv> <outdir> <chartspec.json>

chartspec.json: [{"file": "write.png", "title": "...", "ylabel": "...",
                  "unit": "us"|"ns"|"ms", "scale": "log"|"linear",
                  "cases": ["<regex>", ...]}]

Bars are always sorted ASCENDING BY VALUE, whatever order the patterns are
written in. `scale` defaults to "log"; use "linear" for any chart whose bars
share a magnitude, which is the only way a few-percent spread shows up. A chart
spanning more than ~1000x warns that it should be split into per-band charts
instead.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

UNITS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}

tsv, outdir, spec_path = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
outdir.mkdir(parents=True, exist_ok=True)
work = Path(tsv).parent

rows = {}
for line in Path(tsv).read_text().splitlines():
    name, mean, lo, hi = line.split("\t")
    rows[name] = float(mean)

for chart in json.loads(Path(spec_path).read_text()):
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

    span = rows[picked[-1]] / rows[picked[0]] if picked and rows[picked[0]] else 0
    if span > 100 and chart.get("scale", "log") != "log":
        print(f"WARN {chart['file']}: {span:.0f}x span on a linear scale", file=sys.stderr)
    if span > 1000:
        print(f"WARN {chart['file']}: {span:.0f}x span — consider splitting by magnitude band", file=sys.stderr)

    label = chart.get("strip", "")
    dat = work / (chart["file"].replace(".png", ".dat"))
    with dat.open("w") as f:
        for i, name in enumerate(picked):
            short = re.sub(label, "", name) if label else name
            f.write(f'{i}\t{rows[name] / div:.6g}\t"{short}"\n')

    # Linear is the default for a chart whose bars share a magnitude: it is the
    # only way a few-percent spread is visible at all. Log is for charts that
    # deliberately span bands.
    logscale = "set logscale y" if chart.get("scale", "log") == "log" else "unset logscale y"
    # A linear axis anchored at zero wastes the whole frame when every bar is
    # near the same value, so start the range just below the smallest bar.
    if chart.get("scale", "log") == "log":
        yrange = "[*:*]"
    else:
        lo = min(rows[n] for n in picked) / div
        hi = max(rows[n] for n in picked) / div
        pad = (hi - lo) * 0.15 or hi * 0.05
        yrange = f"[{max(0, lo - pad):.6g}:{hi + pad:.6g}]"

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
set grid ytics lc rgb "#dddddd"
set xtics rotate by -90 scale 0 noenhanced
set key off
set bmargin 20
set yrange {yrange}
plot "{dat}" using 2:xtic(3) with boxes lc rgb "#6a8fc4"
""")
    subprocess.run(["gnuplot", str(gp)], check=True)
    print(f"wrote {outdir / chart['file']} ({len(picked)} bars, {span:.0f}x span, {chart.get('scale','log')})")
