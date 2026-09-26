"""E5 — sweep the learning rate at widths 256, 512 and 1,024, and extrapolate.

Standard parameterisation (every matrix N(0, 0.02), one learning rate for every
weight), 4 layers, 64-wide heads, 300 AdamW steps of 4,096 tokens each, 30 steps
of warmup then cosine to 10% of peak. Score: loss on the fixed held-out set.

The grid is powers of two around 1e-3. If the lowest loss sits on the edge of
the grid, the grid is extended on that side until it does not, because a
minimum at the edge of a sweep is not a minimum. The three points around each
minimum are then repeated with a second seed, and two half-octave points are
added either side of it. The minimum is the vertex of a parabola in log2(lr)
through those five points (seed-averaged where two seeds exist).

Every run is cached in ``results/e5_runs.json`` so an interrupted sweep resumes
where it stopped.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from src.report import RESULTS, machine, save_json, save_text, table
from src.train import cosine, pick_device, train, val_batches

WIDTHS = [256, 512, 1024]
STEPS, WARMUP, FLOOR = 300, 30, 0.1
BASE = 1e-3
KS = list(range(-3, 4))          # 1.25e-4 .. 8e-3
SEEDS_AT_MIN = [1]
CACHE = RESULTS / "e5_runs.json"


def lr_of(k):
    return BASE * 2.0 ** k


def load_cache():
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def run_point(cache, width, k, seed, val, verbose):
    key = f"{width}|{k}|{seed}"
    if key not in cache:
        r = train(d_model=width, schedule=cosine(lr_of(k), WARMUP, FLOOR),
                  steps=STEPS, seed=seed, val=val, eval_every=50)
        cache[key] = {"width": width, "k": k, "lr": lr_of(k), "seed": seed,
                      "val": r.val_loss[-1] if not r.diverged else None,
                      "val_curve": list(zip(r.val_steps, r.val_loss)),
                      "train_tail": float(np.mean(r.loss[-20:])),
                      "diverged": r.diverged, "seconds": r.seconds,
                      "n_params": r.n_params}
        CACHE.write_text(json.dumps(cache, indent=1))
        if verbose:
            v = cache[key]["val"]
            print(f"  width {width:>5}  lr {lr_of(k):.2e}  seed {seed}  "
                  f"val {v if v is None else round(v, 4)}  "
                  f"({r.seconds:.0f}s)", flush=True)
    return cache[key]


def loss_of(p):
    return p["val"] if p["val"] is not None else math.inf


def parabola_min(ks, losses):
    """Vertex of a parabola through three (log2 lr, loss) points."""
    a, b, c = np.polyfit(ks, losses, 2)
    if a <= 0:
        return float(ks[int(np.argmin(losses))])
    return float(-b / (2 * a))


def main(verbose: bool = True) -> dict:
    device = pick_device()
    val = val_batches(device)
    cache = load_cache()
    out = {}
    for w in WIDTHS:
        ks = list(KS)
        for k in ks:
            run_point(cache, w, k, 0, val, verbose)
        while True:                               # extend past an edge minimum
            losses = [loss_of(cache[f"{w}|{k}|0"]) for k in ks]
            i = int(np.argmin(losses))
            if i == 0:
                ks.insert(0, ks[0] - 1)
                run_point(cache, w, ks[0], 0, val, verbose)
            elif i == len(ks) - 1:
                ks.append(ks[-1] + 1)
                run_point(cache, w, ks[-1], 0, val, verbose)
            else:
                break
        three = ks[i - 1:i + 2]
        for s in SEEDS_AT_MIN:
            for k in three:
                run_point(cache, w, k, s, val, verbose)
        # refine: half-octave points either side of the grid minimum
        half = [ks[i] - 0.5, ks[i] + 0.5]
        for k in half:
            run_point(cache, w, k, 0, val, verbose)
        five = sorted(three + half)
        seeds = [0] + SEEDS_AT_MIN
        per_seed = [parabola_min(five, [loss_of(cache[f"{w}|{k}|0"])
                                        for k in five])]
        per_seed += [parabola_min(three, [loss_of(cache[f"{w}|{k}|{s}"])
                                          for k in three]) for s in SEEDS_AT_MIN]
        mean_losses = [np.mean([loss_of(cache[f"{w}|{k}|{s}"]) for s in seeds
                                if f"{w}|{k}|{s}" in cache]) for k in five]
        kstar = parabola_min(five, mean_losses)
        allk = sorted(ks + half)
        out[w] = {
            "fit_points": [{"lr": lr_of(k), "mean_val": float(m)}
                           for k, m in zip(five, mean_losses)],
            "grid": [{"lr": lr_of(k), "val": cache[f"{w}|{k}|0"]["val"],
                      "diverged": cache[f"{w}|{k}|0"]["diverged"]} for k in allk],
            "seed1": [{"lr": lr_of(k), "val": cache[f"{w}|{k}|1"]["val"]}
                      for k in three],
            "best_grid_lr": lr_of(ks[i]),
            "best_grid_val": losses[i],
            "lr_star": BASE * 2 ** kstar,
            "lr_star_per_seed": [BASE * 2 ** x for x in per_seed],
            "n_params": cache[f"{w}|{ks[0]}|0"]["n_params"],
            "sec_per_run": float(np.mean([cache[f"{w}|{k}|0"]["seconds"]
                                          for k in ks])),
        }

    # --- extrapolate: log lr* = a + b log width -----------------------------
    lw = np.log2(WIDTHS)
    ll = np.log2([out[w]["lr_star"] for w in WIDTHS])
    b, a = np.polyfit(lw, ll, 1)
    pred = 2 ** (a + b * 12)
    # spread: every combination of per-seed minima
    combos = []
    import itertools
    for pick in itertools.product(*[out[w]["lr_star_per_seed"] for w in WIDTHS]):
        bb, aa = np.polyfit(lw, np.log2(pick), 1)
        combos.append((bb, 2 ** (aa + bb * 12)))
    slopes = [c[0] for c in combos]
    preds = [c[1] for c in combos]
    local = [float((ll[j + 1] - ll[j]) / (lw[j + 1] - lw[j])) for j in range(2)]
    pred_last = 2 ** (ll[2] + local[1] * (12 - lw[2]))
    # the 1/width rule, anchored at our own width-1024 minimum
    inv_width = out[1024]["lr_star"] * 1024 / 4096
    fit = {"slope": float(b), "intercept": float(a), "pred_4096": float(pred),
           "pred_range": [float(min(preds)), float(max(preds))],
           "slope_range": [float(min(slopes)), float(max(slopes))],
           "inv_width_from_1024": float(inv_width),
           "from_256_naive": out[256]["lr_star"],
           "local_slopes": local, "pred_last_segment": float(pred_last)}

    _plot(out, fit)
    _write(out, fit)
    payload = {"config": dict(widths=WIDTHS, steps=STEPS, warmup=WARMUP,
                              floor=FLOOR, tokens_per_step=4096, n_layer=4),
               "widths": {str(w): v for w, v in out.items()}, "fit": fit,
               "machine": machine()}
    save_json("e5_lr_width.json", payload)
    return payload


def _plot(out, fit):
    from src import plots
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2),
                                 gridspec_kw={"width_ratios": [1.6, 1]})
    cols = {256: plots.S1, 512: plots.S2, 1024: plots.S3}
    for w in WIDTHS:
        g = [p for p in out[w]["grid"] if p["val"] is not None]
        xs, ys = [p["lr"] for p in g], [p["val"] for p in g]
        ax.plot(xs, ys, color=cols[w], marker="o", ms=4,
                label=f"d_model {w}  ({out[w]['n_params'] / 1e6:.1f}M)")
        s1 = [p for p in out[w]["seed1"] if p["val"] is not None]
        ax.scatter([p["lr"] for p in s1], [p["val"] for p in s1],
                   facecolor="none", edgecolor=cols[w], s=28, zorder=4)
        ls = out[w]["lr_star"]
        ymin = min(ys)
        ax.plot([ls], [ymin], marker="v", ms=11, color=cols[w],
                markeredgecolor=plots.SURFACE, markeredgewidth=1.5, zorder=6)
        dx = {256: 0, 512: -26, 1024: 26}[w]
        ax.annotate(f"{ls:.1e}", (ls, ymin), xytext=(dx, -17),
                    textcoords="offset points", ha="center",
                    color=plots.INK_2, fontsize=8)
    ax.set_xscale("log")
    ticks = [lr_of(k) for k in range(-3, 4)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.2g}" for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    lo = min(p["val"] for w in WIDTHS for p in out[w]["grid"] if p["val"])
    ax.set_ylim(lo - 0.15, lo + 1.6)
    ax.set_xlabel("peak learning rate")
    ax.set_ylabel("validation loss after 300 steps (nats)")
    ax.set_title("Loss against learning rate, three widths (▼ = fitted minimum)")
    ax.legend(loc="upper center")

    ws = np.array(WIDTHS + [4096])
    ls = [out[w]["lr_star"] for w in WIDTHS]
    bx.plot(WIDTHS, ls, "o", color=plots.INK, ms=6, label="measured minima")
    for w in WIDTHS:
        bx.plot([w, w], out[w]["lr_star_per_seed"], color=plots.MUTED, lw=1)
    xx = np.geomspace(200, 5000, 50)
    a = fit["intercept"]
    bx.plot(xx, 2 ** (a + fit["slope"] * np.log2(xx)), color=plots.S1, lw=1.5,
            label=f"fit, lr* ∝ width^{fit['slope']:.2f}")
    bx.plot(xx, ls[2] * 1024 / xx, color=plots.MUTED, lw=1.2, ls="--",
            label="1/width through width 1,024")
    bx.plot([4096], [fit["pred_4096"]], marker="*", ms=13, color=plots.S2,
            markeredgecolor=plots.SURFACE)
    bx.plot([4096, 4096], fit["pred_range"], color=plots.S2, lw=2)
    bx.annotate(f"4,096: {fit['pred_4096']:.1e}", (4096, fit["pred_4096"]),
                xytext=(-10, -4), textcoords="offset points", ha="right",
                color=plots.INK_2, fontsize=8)
    bx.set_xscale("log", base=2)
    bx.set_yscale("log")
    bx.set_xticks(ws)
    bx.set_xticklabels([f"{w:,}" for w in ws])
    bx.set_xlabel("d_model")
    bx.set_ylabel("best learning rate")
    bx.set_title("Where the minimum moves")
    bx.legend(loc="lower left")
    plots.finish(fig, "e5_lr_width.png",
                 "Filled markers: seed 0 over the whole grid. Hollow: seed 1 at the three points "
                 "around each minimum. Grey bars: minimum per seed.")


def _write(out, fit):
    rows = []
    for w in WIDTHS:
        o = out[w]
        rows.append((f"{w:,}", f"{o['n_params'] / 1e6:.1f}M",
                     f"{o['best_grid_lr']:.2e}", f"{o['best_grid_val']:.4f}",
                     f"**{o['lr_star']:.2e}**",
                     " – ".join(f"{x:.2e}" for x in sorted(o["lr_star_per_seed"]))))
    grid_rows = []
    all_lrs = sorted({p["lr"] for w in WIDTHS for p in out[w]["grid"]})
    for lr in all_lrs:
        cells = [f"{lr:.2e}"]
        for w in WIDTHS:
            m = [p for p in out[w]["grid"] if p["lr"] == lr]
            cells.append("—" if not m else
                         ("diverged" if m[0]["val"] is None else f"{m[0]['val']:.4f}"))
        grid_rows.append(cells)
    md = [
        "# E5 · Learning rate against width\n",
        f"Standard parameterisation, 4 layers, {STEPS} steps × 4,096 tokens, "
        f"{WARMUP}-step warmup, cosine to {FLOOR:.0%} of peak, AdamW (0.9, 0.95), "
        "decay 0.1, clip 1.0. Score: held-out loss over 65,536 tokens.\n",
        "![](e5_lr_width.png)\n",
        "## Every point (seed 0; half-octave points only near each minimum)\n",
        table(grid_rows, ["peak lr"] + [f"width {w:,}" for w in WIDTHS],
              ["r"] * 4),
        "\n## The three minima\n",
        "The minimum is the vertex of a parabola in log₂(lr) through five "
        "points around the lowest loss: the grid minimum, its two neighbours "
        "(two seeds each, averaged) and two half-octave points (one seed). "
        "Per-seed minima refit seed 0 alone on five points and seed 1 alone "
        "on three.\n",
        table(rows, ["width", "params", "best grid lr", "val loss",
                     "fitted minimum", "per-seed minima"], ["r"] * 6),
        "\n## Extrapolating to 4,096\n",
        f"A straight line through the three minima in log–log space has slope "
        f"**{fit['slope']:.2f}** (per-seed combinations: "
        f"{fit['slope_range'][0]:.2f} to {fit['slope_range'][1]:.2f}). "
        f"Section 12's rule of thumb is −1.00.\n",
        table([
            ("fit through the three minima", f"**{fit['pred_4096']:.1e}**"),
            ("range over per-seed minima",
             f"{fit['pred_range'][0]:.1e} – {fit['pred_range'][1]:.1e}"),
            ("1/width, anchored at our width-1,024 minimum",
             f"{fit['inv_width_from_1024']:.1e}"),
            ("the 512 → 1,024 slope alone, continued",
             f"{fit['pred_last_segment']:.1e}"),
            ("carrying width 256's minimum across unchanged",
             f"{fit['from_256_naive']:.1e}"),
        ], ["estimate for d_model 4,096", "learning rate"], ["l", "r"]),
        "\n## The value I would use, and how confident I am\n",
        f"**{fit['pred_4096']:.1e}**, as the centre of a short confirmation "
        "sweep at the real width, not as a value to launch with.\n",
        "**Confidence: low, roughly a factor of two either way.** The "
        "per-seed range above looks tight, but it only measures seed noise "
        "at these three widths. It does not measure the thing that "
        "dominates the error, which is the shape of the curve:\n",
        f"- The minimum does not move as a clean power law. From 256 to 512 "
        f"the local slope is {fit['local_slopes'][0]:.2f}; from 512 to 1,024 "
        f"it is {fit['local_slopes'][1]:.2f}. Three points cannot tell a "
        "power law from a curve that is flattening, and the two readings "
        f"give {fit['pred_4096']:.1e} and {fit['pred_last_segment']:.1e} at "
        "4,096.\n",
        "- Section 12's −1 rule, anchored at our width-1,024 minimum, gives "
        f"{fit['inv_width_from_1024']:.1e}, another factor of two lower. "
        "Our measured slope is about half as steep, and there is a concrete "
        "reason to expect that here. This model initialises every matrix at "
        "a fixed 0.02. Under a textbook 1/√fan-in initialisation the weights "
        "shrink as the model widens, so the same η is a relatively larger "
        "step on a wider model, and that is part of why the optimum moves "
        "left. With a fixed 0.02 the step-to-weight ratio at a given η is the "
        "same at every width (E3 measures it), so that part of the drift is "
        "absent. What remains comes from wider layers summing more "
        "correlated updates. That argument predicts a shallower slope, not "
        "its exact value. A second, untested possibility is the 300-step "
        "budget, which leaves every width far from converged.\n",
        "- 4,096 is four times wider than the widest point measured, and "
        "the vocabulary (10k) and depth (4) would not be the real ones.\n",
        "**What would raise it:** a fourth width at 2,048 to decide between "
        "the power law and the flattening curve, and a longer budget, since "
        "the optimum at 300 steps is not the optimum at 30,000. Or muP, which "
        "Section 12 exists to recommend: under it this whole table collapses "
        "to one column and the question disappears.\n",
    ]
    save_text("e5_lr_width.md", "\n".join(md) + "\n")


if __name__ == "__main__":
    main()
