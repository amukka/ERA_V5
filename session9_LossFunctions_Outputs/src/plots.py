"""One matplotlib style for every figure in this session.

Palette and rules come from the data-viz reference instance: categorical slots
1-3 (blue, orange, aqua) which validate on all pairs, a hairline recessive grid,
2px lines, text always in ink rather than in the series colour, a legend whenever
there are two or more series, and -- the rule that matters most here -- never two
y-scales on one plot.  Where the session asks for "the norm and the loss on one
axis", they are put on one axis by being indexed to a common baseline, not by
inventing a second scale.
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

S1 = "#2a78d6"   # blue
S2 = "#eb6834"   # orange
S3 = "#1baf7a"   # aqua
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": ["DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.labelcolor": INK_2,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "figure.dpi": 130,
})


def finish(fig, name: str, note: str | None = None) -> pathlib.Path:
    if note:
        fig.text(0.005, 0.005, note, color=MUTED, fontsize=7.5, va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    path = RESULTS / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def label_end(ax, xs, ys, text, color, dx=0.01, fontsize=8.5):
    """Direct-label the endpoint of a series, in ink, with a colour dot."""
    x, y = xs[-1], ys[-1]
    ax.plot([x], [y], marker="o", ms=5, color=color, zorder=5,
            markeredgecolor=SURFACE, markeredgewidth=2)
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    ax.annotate(text, (x + dx * span, y), color=INK_2, fontsize=fontsize,
                va="center", ha="left")
