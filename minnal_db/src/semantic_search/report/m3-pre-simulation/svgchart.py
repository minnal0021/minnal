"""Dependency-free SVG charts for the M3-pre report.

Static SVGs for Markdown: no hover, so every chart labels its key values on the
marks, carries a legend for two or more series, and switches to dark colours
with `prefers-color-scheme`. Series colours come from one fixed slot list, so a
category keeps its colour on every chart (pass `slot=`).
"""
import math
from html import escape

LIGHT = dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df",
             s=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
             ramp=["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"], ref="#8a8984")
DARK = dict(bg="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#383835",
            s=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
            ramp=["#184f95", "#256abf", "#5598e7", "#9ec5f4"], ref="#8f8e88")
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


def _style(n_slots=8):
    def block(p):
        rules = [f".bg{{fill:{p['bg']}}}", f".ink{{fill:{p['ink']}}}", f".ink2{{fill:{p['ink2']}}}",
                 f".grid{{stroke:{p['grid']}}}", f".ring{{stroke:{p['bg']}}}", f".ref{{stroke:{p['ref']}}}", f".reff{{fill:{p['ref']}}}"]
        for i, c in enumerate(p["s"]):
            rules += [f".f{i}{{fill:{c}}}", f".l{i}{{stroke:{c}}}"]
        for i, c in enumerate(p["ramp"]):
            rules += [f".fr{i}{{fill:{c}}}", f".lr{i}{{stroke:{c}}}"]
        return "".join(rules)
    return (f"<style>text{{font-family:{FONT}}}{block(LIGHT)}"
            f"@media (prefers-color-scheme: dark){{{block(DARK)}}}</style>")


class Svg:
    def __init__(self, w, h, title, subtitle=None):
        self.w, self.h, self.parts = w, h, []
        self.add(f'<rect class="bg" x="0" y="0" width="{w}" height="{h}" rx="6"/>')
        self.text(16, 26, title, size=15, weight=600)
        if subtitle:
            self.text(16, 45, subtitle, size=12, cls="ink2")

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=12, cls="ink", anchor="start", weight=400, tip=None):
        t = f"<title>{escape(tip)}</title>" if tip else ""
        self.add(f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
                 f'text-anchor="{anchor}">{t}{escape(str(s))}</text>')

    def save(self, path, desc):
        body = "".join(self.parts)
        open(path, "w").write(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
            f'height="{self.h}" role="img" aria-label="{escape(desc)}"><desc>{escape(desc)}</desc>{_style()}{body}</svg>\n')


def _notes(s, notes, y):
    """Key lines under the legend: (bold term, explanation)."""
    for i, (term, text) in enumerate(notes):
        s.add(f'<text class="ink2" x="16" y="{y + 12 + 17 * i:.1f}" font-size="11">'
              f'<tspan class="ink" font-weight="600">{escape(term)}</tspan> {escape(text)}</text>')


def _ticks(lo, hi, n=5):
    span = hi - lo
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag * 10)
    t = math.ceil(lo / step - 1e-9) * step
    out = []
    while t <= hi + 1e-9:
        out.append(round(t, 10)); t += step
    return out


def line_chart(path, title, subtitle, series, xlabel, ylabel, xlim, ylim, xlog=False, xticks=None,
               yfmt="{:.2f}", xfmt="{:g}", w=720, h=400, legend_cols=3, refs=(), end_labels=True, desc="", notes=()):
    """series: [dict(name, pts=[(x, y)], cls='l0'|'lr2', dash=False, label=True)]"""
    left, right, top = 64, 150 if end_labels else 24, 64
    rows = math.ceil(len(series) / legend_cols)
    bottom = 58 + 20 * rows
    pw, ph = w - left - right, h - top - bottom
    h += 17 * len(notes) + (8 if notes else 0)
    fx = (lambda v: left + (math.log10(v) - math.log10(xlim[0])) / (math.log10(xlim[1]) - math.log10(xlim[0])) * pw) if xlog \
        else (lambda v: left + (v - xlim[0]) / (xlim[1] - xlim[0]) * pw)
    fy = lambda v: top + ph - (v - ylim[0]) / (ylim[1] - ylim[0]) * ph
    s = Svg(w, h, title, subtitle)
    for t in _ticks(*ylim):
        s.add(f'<line class="grid" x1="{left}" x2="{left + pw}" y1="{fy(t):.1f}" y2="{fy(t):.1f}" stroke-width="1"/>')
        s.text(left - 8, fy(t) + 4, yfmt.format(t), size=11, cls="ink2", anchor="end")
    for t in (xticks or _ticks(*xlim)):
        s.add(f'<line class="grid" x1="{fx(t):.1f}" x2="{fx(t):.1f}" y1="{top}" y2="{top + ph}" stroke-width="1"/>')
        s.text(fx(t), top + ph + 16, xfmt.format(t), size=11, cls="ink2", anchor="middle")
    s.text(left + pw / 2, top + ph + 36, xlabel, size=12, cls="ink2", anchor="middle")
    s.add(f'<text class="ink2" font-size="12" text-anchor="middle" transform="translate(16,{top + ph / 2}) rotate(-90)">{escape(ylabel)}</text>')
    for r in refs:   # dict(y=..., label=...)
        s.add(f'<line class="ref" x1="{left}" x2="{left + pw}" y1="{fy(r["y"]):.1f}" y2="{fy(r["y"]):.1f}" stroke-width="1" stroke-dasharray="4 3"/>')
        s.text(left + pw - 4, fy(r["y"]) - 5, r["label"], size=11, cls="ink2", anchor="end")
    labels = []
    for se in series:
        pts = [(x, y) for x, y in se["pts"] if y == y]
        if not pts:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{fx(x):.1f},{fy(y):.1f}" for i, (x, y) in enumerate(pts))
        dash = ' stroke-dasharray="6 4"' if se.get("dash") else ""
        s.add(f'<path class="{se["cls"]}" d="{d}" fill="none" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"{dash}/>')
        fcls = se["cls"].replace("l", "f", 1)
        for x, y in pts:
            s.add(f'<circle class="{fcls} ring" cx="{fx(x):.1f}" cy="{fy(y):.1f}" r="4" stroke-width="2">'
                  f'<title>{escape(se["name"])}: {xfmt.format(x)} → {yfmt.format(y)}</title></circle>')
        if end_labels and se.get("label", True):
            labels.append([fy(pts[-1][1]), f'{se["name"]} {yfmt.format(pts[-1][1])}', fx(pts[-1][0])])
    labels.sort()
    for i in range(1, len(labels)):          # keep end labels 14px apart
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + 14)
    for yv, txt, xv in labels:
        s.text(left + pw + 8, yv + 4, txt, size=11, cls="ink")
    # legend
    ly = h - 20 * rows - 8 - 17 * len(notes) - (8 if notes else 0)
    colw = (w - 32) / legend_cols
    for i, se in enumerate(series):
        x0, y0 = 16 + (i % legend_cols) * colw, ly + (i // legend_cols) * 20
        dash = ' stroke-dasharray="6 4"' if se.get("dash") else ""
        s.add(f'<line class="{se["cls"]}" x1="{x0:.1f}" x2="{x0 + 22:.1f}" y1="{y0:.1f}" y2="{y0:.1f}" stroke-width="2"{dash}/>')
        s.text(x0 + 28, y0 + 4, se["name"], size=11, cls="ink")
    _notes(s, notes, h - 17 * len(notes) - 2)
    s.save(path, desc or title)


def hbar_chart(path, title, subtitle, groups, xlabel, xlim, fmt="{:.3f}", w=720, legend=None, refs=(), desc="",
               bar_h=16, gap=4, group_gap=14, label_w=210, notes=()):
    """groups: [dict(name, bars=[dict(v, cls='f0', label=None)])]; bars within a group share the row label."""
    top, left = 64, label_w
    n_bars = sum(len(g["bars"]) for g in groups)
    legend_rows = []
    if legend:                      # wrap legend items to the chart width
        row, x0 = [], 16
        for cls, txt in legend:
            iw = 30 + 6.4 * len(txt)
            if row and x0 + iw > w - 16:
                legend_rows.append(row); row, x0 = [], 16
            row.append((x0, cls, txt)); x0 += iw
        legend_rows.append(row)
    legend_h = 20 * len(legend_rows) + (8 if legend else 0)
    ph = n_bars * (bar_h + gap) + len(groups) * group_gap
    h = top + ph + 44 + legend_h + 17 * len(notes) + (8 if notes else 0)
    pw = w - left - 70
    fx = lambda v: left + (v - xlim[0]) / (xlim[1] - xlim[0]) * pw
    s = Svg(w, h, title, subtitle)
    for t in _ticks(*xlim):
        s.add(f'<line class="grid" x1="{fx(t):.1f}" x2="{fx(t):.1f}" y1="{top}" y2="{top + ph}" stroke-width="1"/>')
        s.text(fx(t), top + ph + 16, f"{t:g}", size=11, cls="ink2", anchor="middle")
    s.text(left + pw / 2, top + ph + 36, xlabel, size=12, cls="ink2", anchor="middle")
    y = top + group_gap / 2
    base = fx(max(xlim[0], 0) if xlim[0] <= 0 <= xlim[1] else xlim[0])
    for g in groups:
        gy = y
        for b in g["bars"]:
            if b["v"] != b["v"]:            # NaN: no value, say why
                s.text(base + (6 if xlim[0] >= 0 else -6), y + bar_h - 4, b.get("nan", "n/a"), size=11, cls="ink2",
                       anchor="start" if xlim[0] >= 0 else "end")
                y += bar_h + gap
                continue
            x1, x2 = sorted((base, fx(b["v"])))
            r = min(4, (x2 - x1) / 2)
            s.add(f'<rect class="{b["cls"]}" x="{x1:.1f}" y="{y:.1f}" width="{max(x2 - x1, 0.5):.1f}" height="{bar_h}" rx="{r:.1f}">'
                  f'<title>{escape(g["name"])} {escape(b.get("label") or "")}: {fmt.format(b["v"])}</title></rect>')
            tx = (x2 + 6) if b["v"] >= xlim[0] + 0 and fx(b["v"]) >= base else (x1 - 6)
            anchor = "start" if fx(b["v"]) >= base else "end"
            s.text(tx, y + bar_h - 4, fmt.format(b["v"]), size=11, cls="ink", anchor=anchor)
            y += bar_h + gap
        s.text(left - 10, (gy + y - gap) / 2 + 4, g["name"], size=12, cls="ink", anchor="end")
        y += group_gap
    s.add(f'<line class="ref" x1="{base:.1f}" x2="{base:.1f}" y1="{top}" y2="{top + ph}" stroke-width="1"/>')
    for r in refs:
        s.add(f'<line class="ref" x1="{fx(r["x"]):.1f}" x2="{fx(r["x"]):.1f}" y1="{top}" y2="{top + ph}" stroke-width="1" stroke-dasharray="4 3"/>')
        s.text(fx(r["x"]) + 4, top - 6, r["label"], size=11, cls="ink2")
    ly = top + ph + 44
    for i, row in enumerate(legend_rows):
        for x0, cls, txt in row:
            s.add(f'<rect class="{cls}" x="{x0:.1f}" y="{ly + 2 + 20 * i}" width="12" height="12" rx="2"/>')
            s.text(x0 + 18, ly + 12 + 20 * i, txt, size=11)
    ly += legend_h
    _notes(s, notes, ly + 8)
    s.save(path, desc or title)
