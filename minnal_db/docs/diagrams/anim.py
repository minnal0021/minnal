"""Tiny helper for the step-by-step animated SVG diagrams in minnal_db/docs/.

A diagram is a list of steps. Every element that changes is described by its
value at each step; `Anim.states` renders one copy per distinct value and
switches between them with a discrete opacity animation. The diagram is
therefore a plain function of its state model: to change what it shows, change
the model, not the SVG.

The output is self-contained SVG (SMIL animation, no scripts, no external
assets) with light and dark colour schemes.
"""

from html import escape

STYLE = """
<style>
  .surface { fill: #fcfcfb; }
  .frame   { stroke: #e3e2dd; }
  .t1 { fill: #0b0b0b; }
  .t2 { fill: #52514e; }
  .t3 { fill: #86857f; }
  .arrow { fill: #86857f; }
  .line { stroke: #86857f; }
  .grid { stroke: #e8e7e2; }
  .axis { stroke: #cfcec8; }
  .panel { fill: #f2f1ec; stroke: #d8d7d1; }
  .empty { fill: #ffffff; stroke: #dcdbd5; }
  .ghost { fill: #f7f6f2; stroke: #c9c8c2; }
  .pill  { fill: #ffffff; stroke: #d8d7d1; }
  .pill-on { fill: #ffffff; stroke: #0b0b0b; }
  .note-ok  { fill: #dff3ea; }
  .note-bad { fill: #fbe4da; }
  .note-off { fill: #ffffff; stroke: #d8d7d1; }
  .ink-a { fill: #8a5d00; font-weight: 600; }
  .ink-b { fill: #1c5cab; font-weight: 600; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #17171a; }
    .frame   { stroke: #2f2f34; }
    .t1 { fill: #f4f4f2; }
    .t2 { fill: #b9b8b2; }
    .t3 { fill: #85847e; }
    .arrow { fill: #85847e; }
    .line { stroke: #85847e; }
    .grid { stroke: #2a2a2e; }
    .axis { stroke: #3c3c42; }
    .panel { fill: #1f1f24; stroke: #34343a; }
    .empty { fill: #17171a; stroke: #33333a; }
    .ghost { fill: #1b1b1f; stroke: #34343a; }
    .pill  { fill: #17171a; stroke: #34343a; }
    .pill-on { fill: #17171a; stroke: #f4f4f2; }
    .note-ok  { fill: #16382b; }
    .note-bad { fill: #3d2219; }
    .note-off { fill: #17171a; stroke: #34343a; }
    .ink-a { fill: #e0ab3c; font-weight: 600; }
    .ink-b { fill: #7fb2f0; font-weight: 600; }
  }
  text { font-family: ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  .mono { font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace; }
</style>
"""

# Categorical colours, validated for colour-blind separation. They read on
# both the light and the dark surface, so they are not theme-switched.
AMBER = "#eda100"
BLUE = "#3987e5"
GREEN = "#1baf7a"
ORANGE = "#eb6834"
GREY = "#86857f"


def esc(s):
    return escape(str(s), quote=False)


class Anim:
    def __init__(self, n_steps, step_seconds):
        self.n = n_steps
        self.dur = f"{n_steps * step_seconds:.1f}s"
        self.key_times = ";".join(f"{i / n_steps:.5f}" for i in range(n_steps))

    def mask(self, on):
        """An <animate> that shows its parent only on the steps where `on[i]`."""
        if all(on):
            return ""
        values = ";".join("1" if v else "0" for v in on)
        return (f'<animate attributeName="opacity" calcMode="discrete" values="{values}" '
                f'keyTimes="{self.key_times}" dur="{self.dur}" repeatCount="indefinite"/>')

    def states(self, per_step, render):
        """Render `render(value)` once per distinct non-None value in
        `per_step`, each shown only on the steps that hold that value."""
        assert len(per_step) == self.n, (len(per_step), self.n)
        out, seen = [], []
        for v in per_step:
            if v is not None and v not in seen:
                seen.append(v)
        for v in seen:
            body = render(v)
            if body:
                out.append(f"<g>{self.mask([x == v for x in per_step])}{body}</g>")
        return "\n".join(out)


def document(width, height, label, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" aria-label="{escape(label)}">\n'
            f"{STYLE}\n"
            f'<rect class="surface" x="0" y="0" width="{width}" height="{height}" rx="14"/>\n'
            f'<rect class="surface frame" x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="14" fill="none"/>\n'
            f"{body}\n</svg>\n")


def text(x, y, s, cls="t2", size=12, anchor=None, extra=""):
    a = f' text-anchor="{anchor}"' if anchor else ""
    return f'<text class="{cls}" x="{x}" y="{y}" font-size="{size}"{a}{extra}>{s}</text>'


def legend(x, y, items, gap=24):
    """items: [(colour, label)], laid out left to right on one line. Labels
    are measured at ~6.2 px per character (12 px sans), which is generous."""
    out = []
    for colour, label in items:
        out.append(f'<rect x="{x}" y="{y - 10}" width="11" height="11" rx="3" fill="{colour}"/>')
        out.append(text(x + 18, y, esc(label)))
        x += 18 + len(label) * 6.2 + gap
    return "\n".join(out)


def caption(anim, x, y, per_step):
    """per_step: [(dot colour, caption markup)] — one line of narration per step."""
    return anim.states(per_step, lambda v: (
        f'<circle cx="{x + 6}" cy="{y - 4}" r="4" fill="{v[0]}"/>' + text(x + 20, y, v[1], "t1", 12.5)))
