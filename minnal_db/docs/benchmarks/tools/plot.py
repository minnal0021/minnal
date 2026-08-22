#!/usr/bin/env python3
"""Render criterion means as gnuplot bar charts, matching the existing
minnal_db/docs/benchmarks/*.png style (log-scale y, rotated labels).

Usage: plot.py <tsv> <outdir> <chartspec.json>

chartspec.json: [{"file": "write.png", "title": "...", "ylabel": "...",
                  "unit": "us"|"ns"|"ms", "cases": ["<regex>", ...]}]
Cases are matched (fullmatch) in order; each pattern may match many rows,
which are emitted sorted by name.
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
        hits = sorted(n for n in rows if rx.fullmatch(n))
        if not hits:
            print(f"WARN {chart['file']}: no match for {pat!r}", file=sys.stderr)
        picked.extend(hits)

    label = chart.get("strip", "")
    dat = work / (chart["file"].replace(".png", ".dat"))
    with dat.open("w") as f:
        for i, name in enumerate(picked):
            short = re.sub(label, "", name) if label else name
            f.write(f'{i}\t{rows[name] / div:.6g}\t"{short}"\n')

    gp = work / (chart["file"].replace(".png", ".gp"))
    gp.write_text(f"""
set terminal pngcairo size 960,820 font "DejaVu Sans,10"
set output "{outdir / chart["file"]}"
set title "{chart["title"]}" font "DejaVu Sans,12 bold" noenhanced
set ylabel "{chart["ylabel"]}"
set style data histograms
set style fill solid 1.0 border rgb "black"
set boxwidth 0.7
set logscale y
set grid ytics lc rgb "#dddddd"
set xtics rotate by -90 scale 0 noenhanced
set key off
set bmargin 20
set yrange [*:*]
plot "{dat}" using 2:xtic(3) with boxes lc rgb "#6a8fc4"
""")
    subprocess.run(["gnuplot", str(gp)], check=True)
    print(f"wrote {outdir / chart['file']} ({len(picked)} bars)")
