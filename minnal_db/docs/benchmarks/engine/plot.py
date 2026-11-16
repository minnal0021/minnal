#!/usr/bin/env python3
"""Charts and tables of an engine_bench run's concurrent cells, for benchmark.md.

    plot.py RUN_DIR OUT_DIR

Reads RUN_DIR/cells/closed-NN-rK.json (every client count, every repeat), the
single-operation cells and the fixed-arrival-rate cells (open-NNNNN-rK.json),
and writes to OUT_DIR:

  engine_throughput.png    operations per second by number of clients
  engine_latency.png       one panel per operation: p50, p90, p95 and p99
                           latency by number of clients
  engine_latency.md        the same figures as Markdown tables, p99.9 included
  engine_open_latency.png  one panel per operation: p50, p90, p95 and p99
                           latency by arrival rate (open-loop cells)
  engine_open_latency.md   the same figures as Markdown tables

Each pair is drawn only when the run has those cells. A rate that saturated the
engine (it shed arrivals) is left off the chart and marked in the table: its
latencies are only a lower bound.

Every point is the median of the cell's repeats; the tables give the lowest and
highest beside it. A percentile keeps one colour on every panel. Needs gnuplot.
"""
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile

OPS = ["get", "scan", "field_query", "search", "filtered_search", "insert", "update", "delete"]
PCTS = [("p50_ms", "p50"), ("p90_ms", "p90"), ("p95_ms", "p95"), ("p99_ms", "p99")]
TABLE_PCTS = [("p50_ms", "p50"), ("p75_ms", "p75"), ("p90_ms", "p90"), ("p95_ms", "p95"), ("p99_ms", "p99"), ("p999_ms", "p99.9")]
# Colour-blind-separable, one colour per percentile on every panel.
COLOURS = ["#2a78d6", "#1baf7a", "#eb6834", "#a13cb3"]
BAR = "#2a78d6"


def load(run):
    """{clients: [cell, ...]} over the valid closed-loop cells, {op: [cell, ...]} single
    ones and {rate: [cell, ...]} open-loop ones."""
    closed, single, open_ = {}, {}, {}
    for path in sorted(glob.glob(f"{run}/cells/*-r[0-9]*.json")):
        cell = json.load(open(path))
        if not cell.get("valid"):
            sys.exit(f"{path}: invalid cell ({cell.get('reasons')}); refusing to chart it")
        name = os.path.basename(path)
        if m := re.match(r"closed-(\d+)-r\d+\.json$", name):
            closed.setdefault(int(m[1]), []).append(cell)
        elif m := re.match(r"single-(\w+)-r\d+\.json$", name):
            single.setdefault(m[1], []).append(cell)
        elif m := re.match(r"open-(\d+)-r\d+\.json$", name):
            open_.setdefault(int(m[1]), []).append(cell)
    if not closed and not open_:
        sys.exit(f"{run}: no concurrent cells")
    return closed, single, open_


def spread(values):
    return statistics.median(values), min(values), max(values)


def fmt(v):
    return f"{v:.2f}" if v < 10 else f"{v:.1f}" if v < 100 else f"{v:.0f}"


def cell_fmt(values):
    med, lo, hi = spread(values)
    return fmt(med) if len(values) == 1 else f"{fmt(med)} ({fmt(lo)}–{fmt(hi)})"


def gnuplot(script):
    subprocess.run(["gnuplot"], input=script, text=True, check=True)


def throughput_chart(closed, out, tmp):
    clients = sorted(closed)
    dat = f"{tmp}/throughput.dat"
    with open(dat, "w") as f:
        for i, n in enumerate(clients):
            med, lo, hi = spread([c["total_ops_per_s"] for c in closed[n]])
            f.write(f"{i} {med} {lo} {hi} {n}\n")
    top = max(spread([c["total_ops_per_s"] for c in closed[n]])[2] for n in clients)
    gnuplot(f"""
set terminal pngcairo size 760,460 font "DejaVu Sans,10"
set output "{out}/engine_throughput.png"
set title "Operations per second, by number of clients"
set xlabel "Clients"
set ylabel "Operations per second"
set yrange [0:{top * 1.15}]
set grid ytics lc rgb "#dddddd"
set boxwidth 0.6
set style fill solid 1.0 border rgb "black"
set xrange [-0.6:{len(clients) - 0.4}]
unset key
plot "{dat}" using 1:2:xtic(5) with boxes lc rgb "{BAR}", \\
     "" using 1:2:3:4 with yerrorbars lc rgb "black" pt 0 ps 0, \\
     "" using 1:4:(sprintf("%.0f", $2)) with labels offset 0,0.9
""")


def latency_chart(closed, out, tmp):
    clients = sorted(closed)
    ops = [op for op in OPS if all(op in c["classes"] for n in clients for c in closed[n])]
    plots = []
    for op in ops:
        dat = f"{tmp}/{op}.dat"
        with open(dat, "w") as f:
            for n in clients:
                row = [statistics.median(c["classes"][op][key] for c in closed[n]) for key, _ in PCTS]
                f.write(f"{n} " + " ".join(f"{v}" for v in row) + "\n")
        top = max(statistics.median(c["classes"][op]["p99_ms"] for c in closed[n]) for n in clients)
        lines = ", \\\n     ".join(
            f'"{dat}" using 1:{i + 2} with linespoints lw 2 pt 7 ps 0.7 lc rgb "{COLOURS[i]}" title "{label}"'
            for i, (_, label) in enumerate(PCTS)
        )
        # The p50 and p99 values at the highest client count, so a reader gets a number off the chart.
        labels = (f'"{dat}" every ::{len(clients) - 1} using 1:2:(sprintf("%.3g", $2)) with labels left offset 0.6,0 tc rgb "{COLOURS[0]}" notitle, \\\n'
                  f'     "{dat}" every ::{len(clients) - 1} using 1:5:(sprintf("%.3g", $5)) with labels left offset 0.6,0 tc rgb "{COLOURS[3]}" notitle')
        plots.append(f"""
set title "{op}" noenhanced
set yrange [0:{top * 1.12}]
plot {lines}, \\
     {labels}
""")
    rows = (len(ops) + 1) // 2
    gnuplot(f"""
set terminal pngcairo size 1000,{280 * rows + 40} font "DejaVu Sans,10"
set output "{out}/engine_latency.png"
set multiplot layout {rows},2 title "Latency by number of clients (ms; closed loop over the mix)"
set grid ytics lc rgb "#dddddd"
set xlabel "Clients"
set ylabel "ms"
set xtics ({", ".join(str(n) for n in clients)})
set xrange [0:{max(clients) * 1.18}]
set key top left horizontal reverse Left font ",9"
""" + "".join(plots) + "unset multiplot\n")
    return ops


def saturated(cells):
    return any(c["open"]["saturated"] for c in cells)


def open_latency_chart(open_, out, tmp):
    """Latency by arrival rate, one panel per operation; saturated rates left off."""
    rates = [r for r in sorted(open_) if not saturated(open_[r])]
    if not rates:
        sys.exit("every open-loop rate saturated the engine; nothing to chart")
    ops = [op for op in OPS if all(op in c["classes"] for r in rates for c in open_[r])]
    plots = []
    for op in ops:
        dat = f"{tmp}/open-{op}.dat"
        with open(dat, "w") as f:
            for r in rates:
                row = [statistics.median(c["classes"][op][key] for c in open_[r]) for key, _ in PCTS]
                f.write(f"{r} " + " ".join(f"{v}" for v in row) + "\n")
        top = max(statistics.median(c["classes"][op]["p99_ms"] for c in open_[r]) for r in rates)
        lines = ", \\\n     ".join(
            f'"{dat}" using 1:{i + 2} with linespoints lw 2 pt 7 ps 0.7 lc rgb "{COLOURS[i]}" title "{label}"'
            for i, (_, label) in enumerate(PCTS)
        )
        # The p50 and p99 values at the highest rate charted.
        labels = (f'"{dat}" every ::{len(rates) - 1} using 1:2:(sprintf("%.3g", $2)) with labels left offset 0.6,0 tc rgb "{COLOURS[0]}" notitle, \\\n'
                  f'     "{dat}" every ::{len(rates) - 1} using 1:5:(sprintf("%.3g", $5)) with labels left offset 0.6,0 tc rgb "{COLOURS[3]}" notitle')
        plots.append(f"""
set title "{op}" noenhanced
set yrange [0:{top * 1.12}]
plot {lines}, \\
     {labels}
""")
    rows = (len(ops) + 1) // 2
    gnuplot(f"""
set terminal pngcairo size 1000,{280 * rows + 40} font "DejaVu Sans,10"
set output "{out}/engine_open_latency.png"
set multiplot layout {rows},2 title "Latency by arrival rate (ms; operations from the mix arriving at a fixed rate)"
set grid ytics lc rgb "#dddddd"
set xlabel "Operations arriving per second"
set ylabel "ms"
set xtics ({", ".join(str(r) for r in rates)}) rotate by -30
set xrange [0:{max(rates) * 1.18}]
set key top left horizontal reverse Left font ",9"
""" + "".join(plots) + "unset multiplot\n")
    return ops


def open_tables(open_, ops, out):
    rates = sorted(open_)
    repeats = max(len(v) for v in open_.values())
    w = ["# Engine latency at a fixed arrival rate", "",
         "The figures behind the chart in `benchmark.md` → *Latency at a fixed arrival rate*, "
         "drawn by `docs/benchmarks/engine/plot.py`. Operations from the mix arrive at the rate "
         "whatever the engine is doing, and each latency runs from the operation's scheduled arrival. "
         "Milliseconds. In flight: operations under way at once, the mean (rate × mean latency) and the most. "
         "A saturated rate shed arrivals; its latencies are only a lower bound and are not given. "
         + ("Each cell ran once." if repeats == 1 else
            f"Median of {repeats} repeats, lowest–highest in brackets."), ""]
    w.append("| Arrivals per second | " + " | ".join(f"{r:,}" for r in rates) + " |")
    w.append("|---|" + "---:|" * len(rates))
    w.append("| Completed per second | " + " | ".join(cell_fmt([c["total_ops_per_s"] for c in open_[r]]) for r in rates) + " |")
    w.append("| Mean in flight | " + " | ".join(cell_fmt([c["open"]["mean_in_flight"] for c in open_[r]]) for r in rates) + " |")
    w.append("| Most in flight | " + " | ".join(cell_fmt([c["open"]["max_in_flight"] for c in open_[r]]) for r in rates) + " |")
    w.append("| Dispatch lag p99 (ms) | " + " | ".join(cell_fmt([c["open"]["dispatch_lag_p99_ms"] for c in open_[r]]) for r in rates) + " |")
    w.append("| Saturated | " + " | ".join("**yes**" if saturated(open_[r]) else "no" for r in rates) + " |")
    w.append("")
    for op in ops:
        w.append(f"**`{op}`** (ms)")
        w.append("")
        w.append("| Arrivals per second | " + " | ".join(label for _, label in TABLE_PCTS) + " |")
        w.append("|---:|" + "---:|" * len(TABLE_PCTS))
        for r in rates:
            if saturated(open_[r]):
                w.append(f"| {r:,} | saturated |" + " |" * (len(TABLE_PCTS) - 1))
            else:
                w.append(f"| {r:,} | " + " | ".join(cell_fmt([c["classes"][op][key] for c in open_[r]]) for key, _ in TABLE_PCTS) + " |")
        w.append("")
    with open(f"{out}/engine_open_latency.md", "w") as f:
        f.write("\n".join(w).rstrip() + "\n")


def tables(closed, single, ops, out):
    clients = sorted(closed)
    repeats = max(len(v) for v in closed.values())
    w = ["# Engine latency by number of clients", "",
         "The figures behind the charts in `benchmark.md` → *Latency by number of clients*, "
         "drawn by `docs/benchmarks/engine/plot.py`. Milliseconds; \"alone\" is the operation sent "
         "back to back by one client, the other rows the closed-loop mix. "
         + ("Each cell ran once." if repeats == 1 else
            f"Median of {repeats} repeats, lowest–highest in brackets."), ""]
    w.append("| Clients | " + " | ".join(str(n) for n in clients) + " |")
    w.append("|---|" + "---:|" * len(clients))
    w.append("| Operations per second | " + " | ".join(cell_fmt([c["total_ops_per_s"] for c in closed[n]]) for n in clients) + " |")
    w.append("")
    for op in ops:
        w.append(f"**`{op}`** (ms)")
        w.append("")
        w.append("| Clients | " + " | ".join(label for _, label in TABLE_PCTS) + " |")
        w.append("|---:|" + "---:|" * len(TABLE_PCTS))
        if op in single:
            w.append("| alone | " + " | ".join(cell_fmt([c["classes"][op][key] for c in single[op]]) for key, _ in TABLE_PCTS) + " |")
        for n in clients:
            w.append(f"| {n} | " + " | ".join(cell_fmt([c["classes"][op][key] for c in closed[n]]) for key, _ in TABLE_PCTS) + " |")
        w.append("")
    with open(f"{out}/engine_latency.md", "w") as f:
        f.write("\n".join(w).rstrip() + "\n")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    run, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    closed, single, open_ = load(run)
    with tempfile.TemporaryDirectory() as tmp:
        if closed:
            throughput_chart(closed, out, tmp)
            ops = latency_chart(closed, out, tmp)
            tables(closed, single, ops, out)
            print(f"{out}: engine_throughput.png, engine_latency.png, engine_latency.md "
                  f"({len(closed)} client counts, {max(len(v) for v in closed.values())} repeats)")
        if open_:
            ops = open_latency_chart(open_, out, tmp)
            open_tables(open_, ops, out)
            print(f"{out}: engine_open_latency.png, engine_open_latency.md "
                  f"({len(open_)} arrival rates, {max(len(v) for v in open_.values())} repeats)")


if __name__ == "__main__":
    main()
