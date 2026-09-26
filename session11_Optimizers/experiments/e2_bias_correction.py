"""E2 — Adam with bias correction switched off, first twenty steps both ways.

Part A is one weight, the E1 setting extended to 20 gradients drawn from
[0.4, 0.6]. The uncorrected step is exactly ``r(t) = (1 − β₁ᵗ)/√(1 − β₂ᵗ)``
times the corrected one for the same history (divide the two update rules),
so "when does the difference stop mattering" has an exact answer for a single
weight: the step after which r(t) stays within 1% of 1. It is computed for the
session's β₂ = 0.999 and for the β₂ = 0.95 that our training uses.

Part B asks the same question of the real width-256 model, where it no longer
has a closed form: once the first step differs, the two runs see different
gradients. Both are trained 300 steps on identical batches, with and without
warmup, and the difference is judged against the only yardstick that means
anything: how far two *correct* runs differ when only the seed changes.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from src.report import machine, save_json, save_text, table
from src.train import constant, cosine, pick_device, train, val_batches
from src.tuned import lr_256

B1, EPS, LR1 = 0.9, 1e-8, 1e-3
BETA2S = [0.999, 0.95]
STEPS_B, WARMUP_B, EVAL_EVERY = 300, 30, 10


# ------------------------------------------------------------- part A
def single_weight(b2, grads):
    rows, m, v = [], 0.0, 0.0
    for t, g in enumerate(grads, start=1):
        m = B1 * m + (1 - B1) * g
        v = b2 * v + (1 - b2) * g * g
        corr = (m / (1 - B1 ** t)) / (math.sqrt(v / (1 - b2 ** t)) + EPS)
        raw = m / (math.sqrt(v) + EPS)
        rows.append({"t": t, "g": g, "step_corrected": corr,
                     "step_uncorrected": raw, "ratio": raw / corr})
    return rows


def r_of(t, b2):
    return (1 - B1 ** t) / math.sqrt(1 - b2 ** t)


def settle_step(b2, tol):
    """First t after which |r(t) − 1| stays below ``tol`` for good."""
    last_bad = 0
    for t in range(1, 200_000):
        if abs(r_of(t, b2) - 1) >= tol:
            last_bad = t
        elif t > last_bad + 5000 or (t > 50 and r_of(t, b2) < 1 + tol
                                     and b2 ** t < tol / 100):
            break
    return last_bad + 1


# ------------------------------------------------------------- part B
def model_runs(lr, verbose):
    device = pick_device()
    val = val_batches(device)
    out = {}
    for b2 in BETA2S:
        for wu in (0, WARMUP_B):
            sched = cosine(lr, wu, 0.1) if wu else _cos_nowarm(lr)
            cell = {}
            for tag, opt, seed in (("bc", "manual", 0), ("nobc", "manual_nobc", 0),
                                   ("bc_seed1", "manual", 1)):
                r = train(schedule=sched, steps=STEPS_B, seed=seed,
                          betas=(B1, b2), optimizer=opt, val=val,
                          eval_every=EVAL_EVERY, eval_at=(1, 2, 5))
                cell[tag] = {"loss": r.loss, "val_steps": r.val_steps,
                             "val": r.val_loss, "diverged": r.diverged}
                if verbose:
                    print(f"  β₂ {b2}  warmup {wu:>2}  {tag:<8} "
                          f"final val {r.val_loss[-1]:.4f}", flush=True)
            cell.update(_judge(cell))
            out[f"{b2}|{wu}"] = cell
    return out


def _cos_nowarm(lr):
    return cosine(lr, 0, 0.1)


def _judge(cell):
    vs = cell["bc"]["val_steps"]
    a, b, c = (np.array(cell[k]["val"]) for k in ("bc", "nobc", "bc_seed1"))
    gap = b - a
    seed_gap = np.abs(c - a)
    late = [i for i, s in enumerate(vs) if s >= 100]
    noise = float(np.median(seed_gap[late]))
    settled = None
    for i in range(len(vs)):
        if np.all(np.abs(gap[i:]) <= noise):
            settled = vs[i]
            break
    frozen = None                  # gap no longer changing: within noise of its end value
    for i in range(len(vs)):
        if np.all(np.abs(gap[i:] - gap[-1]) <= noise):
            frozen = vs[i]
            break
    at20 = gap[vs.index(20)]
    return {"gap": gap.tolist(), "noise": noise, "settled_step": settled,
            "frozen_step": frozen,
            "gap_at_20": float(at20), "gap_final": float(gap[-1]),
            "max_gap": float(np.max(gap)),
            "max_gap_step": vs[int(np.argmax(gap))],
            "val_final_bc": float(a[-1]), "val_final_nobc": float(b[-1])}


def _from_json():
    """Rebuild part B from a previous run's JSON (``--reuse``), so the write-up
    can be regenerated without forty minutes of training."""
    import json
    from src.report import RESULTS
    old = json.loads((RESULTS / "e2_bias_correction.json").read_text())
    out = {}
    for key, v in old["part_b"].items():
        cell = {tag: {"val_steps": v["val_steps"], "val": v[f"val_{tag}"],
                      "loss": v.get(f"loss_{tag}_first20", [])}
                for tag in ("bc", "nobc", "bc_seed1")}
        cell.update(_judge(cell))
        out[key] = cell
    return out, old["lr"], old["lr_source"]


def main(verbose: bool = True, reuse: bool = False) -> dict:
    rng = np.random.default_rng(0)
    grads = [round(float(x), 3) for x in rng.uniform(0.4, 0.6, 20)]
    part_a = {str(b2): single_weight(b2, grads) for b2 in BETA2S}
    settle = {str(b2): {f"{tol:.0%}": settle_step(b2, tol)
                        for tol in (0.10, 0.01)} for b2 in BETA2S}
    peak = {}
    for b2 in BETA2S:
        ts = np.arange(1, 5001)
        rs = np.array([r_of(t, b2) for t in ts])
        peak[str(b2)] = {"peak": float(rs.max()), "at": int(ts[rs.argmax()])}
    cum = {}
    for b2 in BETA2S:
        rows = part_a[str(b2)]
        cum[str(b2)] = {
            "corrected_20": sum(r["step_corrected"] for r in rows),
            "uncorrected_20": sum(r["step_uncorrected"] for r in rows)}

    if reuse:
        part_b, lr, lr_src = _from_json()
    else:
        lr, lr_src = lr_256()
        part_b = model_runs(lr, verbose)

    payload = {"grads": grads, "part_a": part_a, "settle": settle,
               "peak": peak, "cumulative_20": cum,
               "part_b": {k: {kk: vv for kk, vv in v.items()
                              if kk not in ("bc", "nobc", "bc_seed1")}
                          | {"val_steps": v["bc"]["val_steps"],
                             "val_bc": v["bc"]["val"],
                             "val_nobc": v["nobc"]["val"],
                             "val_bc_seed1": v["bc_seed1"]["val"],
                             "loss_bc_first20": v["bc"]["loss"][:20],
                             "loss_nobc_first20": v["nobc"]["loss"][:20]}
                          for k, v in part_b.items()},
               "lr": lr, "lr_source": lr_src, "machine": machine()}
    save_json("e2_bias_correction.json", payload)
    _plot(part_a, settle, part_b)
    _write(grads, part_a, settle, peak, cum, part_b, lr, lr_src)
    return payload


def _plot(part_a, settle, part_b):
    from src import plots
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.6))
    cols = {"0.999": plots.S1, "0.95": plots.S2}

    ax = axs[0, 0]
    t = np.arange(1, 21)
    for b2, rows in part_a.items():
        ax.plot(t, [r["step_uncorrected"] for r in rows], color=cols[b2],
                marker="o", ms=4, label=f"uncorrected, β₂ = {b2}")
    ax.plot(t, [r["step_corrected"] for r in part_a["0.999"]], color=plots.INK,
            marker="o", ms=4, label="corrected (both β₂)")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("step")
    ax.set_ylabel("step size, in units of η")
    ax.set_title("One weight, first twenty steps")
    ax.legend()

    ax = axs[0, 1]
    ts = np.unique(np.geomspace(1, 20000, 600).astype(int))
    for b2 in ("0.999", "0.95"):
        ax.plot(ts, [r_of(x, float(b2)) for x in ts], color=cols[b2],
                label=f"β₂ = {b2}")
        s = settle[b2]["1%"]
        ax.axvline(s, color=cols[b2], lw=1, ls=":")
        ax.annotate(f"within 1%\nfrom step {s:,}", (s, 4.2), xytext=(4, 0),
                    textcoords="offset points", color=plots.INK_2, fontsize=8)
    ax.axhline(1, color=plots.MUTED, lw=1)
    ax.axvspan(1, 20, color=plots.GRID, alpha=0.6, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("step (log)")
    ax.set_ylabel("uncorrected ÷ corrected step")
    ax.set_title("The same ratio, until it stops mattering (shaded: steps 1–20)")
    ax.legend(loc="upper right")

    for ax, key, title in ((axs[1, 0], "0.999|0", "β₂ = 0.999, no warmup"),
                           (axs[1, 1], f"0.95|{WARMUP_B}",
                            f"β₂ = 0.95, {WARMUP_B}-step warmup (our setting)")):
        c = part_b[key]
        s = np.arange(1, 21)
        ax.plot(s, c["bc"]["loss"][:20], color=plots.INK, marker="o", ms=3.5,
                label="corrected")
        ax.plot(s, c["nobc"]["loss"][:20], color=plots.S2, marker="o", ms=3.5,
                label="uncorrected")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlabel("step")
        ax.set_ylabel("training loss (same batches)")
        ax.set_title(f"Width-256 model: {title}")
        ax.legend()
    plots.finish(fig, "e2_bias_correction.png")


def _write(grads, part_a, settle, peak, cum, part_b, lr, lr_src):
    rows = []
    a, b = part_a["0.999"], part_a["0.95"]
    for i in range(20):
        rows.append((i + 1, f"{grads[i]:.3f}", f"{a[i]['step_corrected']:.4f}",
                     f"{a[i]['step_uncorrected']:.4f}", f"{a[i]['ratio']:.3f}",
                     f"{b[i]['step_uncorrected']:.4f}", f"{b[i]['ratio']:.3f}"))
    srow = [(b2, f"{peak[b2]['peak']:.2f}× at step {peak[b2]['at']}",
             f"{part_a[b2][19]['ratio']:.2f}×",
             f"{settle[b2]['10%']:,}", f"**{settle[b2]['1%']:,}**")
            for b2 in ("0.999", "0.95")]
    brow = []
    for key, c in part_b.items():
        b2, wu = key.split("|")
        brow.append((b2, wu, f"{c['gap_at_20']:+.4f}",
                     f"{c['max_gap']:+.4f} (step {c['max_gap_step']})",
                     f"{c['gap_final']:+.4f}", f"{c['noise']:.4f}",
                     "never" if c["settled_step"] is None
                     else f"**{c['settled_step']}**",
                     "never" if c["frozen_step"] is None
                     else f"**{c['frozen_step']}**"))
    md = [
        "# E2 · Bias correction off, twenty steps both ways\n",
        "![](e2_bias_correction.png)\n",
        "## A · One weight\n",
        f"Twenty gradients drawn uniformly from [0.4, 0.6] (seed 0), β₁ = {B1}. "
        "The step is shown in units of η. Corrected, it sits at ~1.0η "
        "throughout, as in E1. Uncorrected:\n",
        table(rows, ["t", "g", "corrected", "uncorr. β₂=0.999", "ratio",
                     "uncorr. β₂=0.95", "ratio"], ["r"] * 7),
        "\nDividing the two update rules gives the ratio in closed form, "
        "r(t) = (1 − β₁ᵗ) / √(1 − β₂ᵗ), independent of the gradients. So for "
        "a single weight the question has an exact answer:\n",
        table(srow, ["β₂", "largest overshoot", "at step 20",
                     "within 10% from step", "within 1% from step"],
              ["r", "r", "r", "r", "r"]),
        f"\nOver the first twenty steps the uncorrected weight travels "
        f"{abs(cum['0.999']['uncorrected_20']):.1f}η instead of "
        f"{abs(cum['0.999']['corrected_20']):.1f}η at β₂ = 0.999 "
        f"({abs(cum['0.95']['uncorrected_20']):.1f}η at β₂ = 0.95).\n",
        "**The session's step-1 figure of 3.16η is only the beginning.** "
        "With β₂ = 0.999 the uncorrected step keeps *growing* after step 1, "
        "because m fills in a hundred times faster than v. It peaks at "
        f"{peak['0.999']['peak']:.2f}η at step {peak['0.999']['at']}, is still "
        f"{part_a['0.999'][19]['ratio']:.2f}η at step 20, and then takes "
        "thousands of steps to come back: v's correction factor is "
        "√(1 − 0.999ᵗ), which is still 0.80 at step 1,000.\n",
        "**With β₂ = 0.95, the value we train with, the error runs the other "
        f"way first.** Step 1 is only {part_a['0.95'][0]['ratio']:.2f}η. The "
        "square root softens v's bias: uncorrected, √v = √(1 − β₂)·|g| = "
        "0.22|g| while m = (1 − β₁)·g = 0.10g. Step 1 is too small whenever "
        "√(1 − β₂) > 1 − β₁, which is whenever β₂ < 0.99. The "
        f"ratio then overshoots to {peak['0.95']['peak']:.2f}η at step "
        f"{peak['0.95']['at']} and is within 1% from step "
        f"{settle['0.95']['1%']}. The answer to \"after how many steps does "
        "it stop mattering\" is therefore a property of β₂, roughly "
        "4/(1 − β₂) steps for the 1% level, and not a fixed number.\n",
        "## B · The width-256 model\n",
        f"Peak lr {lr:.1e} ({lr_src}), {STEPS_B} steps, cosine to 10%. "
        "Corrected and uncorrected runs use the same seed and the same batches. "
        f"Validation loss is measured every {EVAL_EVERY} steps. The noise yardstick is "
        "the median |Δ val loss| between two corrected runs that differ only in "
        "seed, over steps 100–300.\n",
        table(brow, ["β₂", "warmup", "Δ val at step 20", "largest Δ val",
                     "Δ val at 300", "seed noise", "gap inside noise from",
                     "gap frozen from"],
              ["r"] * 8),
        "\nΔ is uncorrected minus corrected: positive means the uncorrected "
        "run is worse. \"Gap inside noise from\" asks whether the two runs "
        "become indistinguishable. \"Gap frozen from\" asks the weaker question, "
        "whether bias correction has stopped *changing* anything: the first "
        "evaluation after which the gap stays within seed noise of its final "
        "value.\n",
        "## What the model adds to the closed form\n",
        "**The loss never forgets the first few dozen steps.** In none of the "
        "four settings does the uncorrected run come back inside seed noise "
        "within 300 steps. The step rule itself stops differing on schedule "
        f"(from step {settle['0.95']['1%']} at β₂ = 0.95), but the weights "
        "it produced in the meantime are different weights, and a 300-step run "
        "at a decaying learning rate has no time to undo that. What does stop "
        "is the *change*: the gap freezes at the steps in the last column and "
        "is a constant offset after that.\n",
        f"**β₂ = 0.999: it matters for the whole run, in both directions.** "
        f"With warmup, the uncorrected run is ahead by "
        f"{-part_b['0.999|30']['gap_at_20']:.2f} nats at step 20, because a "
        "6× step during warmup is simply a larger learning rate, and larger "
        "is faster at first. It is behind from step ~40 and finishes "
        f"{part_b['0.999|30']['gap_final']:+.2f} worse, because the 3–6× step "
        "persists long after it stopped helping. That matches the closed form: "
        f"the ratio is still {r_of(300, 0.999):.2f} at step 300.\n",
        f"**β₂ = 0.95 without warmup: turning correction off *helped*, by "
        f"{-part_b['0.95|0']['gap_final']:.2f} nats.** The uncorrected first step "
        f"is {part_a['0.95'][0]['ratio']:.2f}η and it climbs to η over about "
        "ten steps, so for any β₂ below 0.99 dropping bias correction is an "
        "accidental warmup. It is a worse warmup than a real one, though: the "
        f"corrected run *with* 30 warmup steps finished at "
        f"{part_b['0.95|30']['val_final_bc']:.3f}, below the uncorrected "
        f"no-warmup run's {part_b['0.95|0']['val_final_nobc']:.3f}.\n",
        f"**β₂ = 0.95 with warmup, our setting: {part_b['0.95|30']['gap_final']:+.2f} "
        f"against seed noise of {part_b['0.95|30']['noise']:.2f}.** Small, "
        "about two noise widths, and frozen early. Warmup had already done the "
        "job bias correction was protecting, so turning it off only made the "
        "first steps timid for no benefit.\n",
    ]
    save_text("e2_bias_correction.md", "\n".join(md))


if __name__ == "__main__":
    main(reuse="--reuse" in sys.argv)
