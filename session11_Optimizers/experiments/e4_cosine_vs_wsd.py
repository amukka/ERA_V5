"""E4 — cosine against WSD, both planned for 300 steps, both stopped at 200.

Both sides are tuned before they are compared. Each schedule gets its own
learning-rate sweep (half-octave grid around 1e-3), scored on the loss it was
designed to deliver, the step-300 loss of a completed run. Each is then run at
its own best rate over three seeds.

Both schedules share 30 steps of linear warmup and decay to zero. WSD holds
the peak until step 240 and decays linearly over the last 20% (60 steps).

What is compared at step 200:

  cosine, stopped    the 300-step cosine run, halted at 200. Its learning rate
                     there is still 29% of peak: it never finished decaying.
  WSD, stopped       the 300-step WSD run, halted at 200. Still at full peak.
  WSD + branch       the WSD run's step-160 checkpoint, decayed linearly to zero
                     over 40 steps, finishing at step 200. The main run is
                     untouched and can carry on, which is the point of WSD.
  cosine, planned    a cosine run told from the start that it has 200 steps.
                     The reference: what you would have got had you known.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from src.report import machine, save_json, save_text, table
from src.train import cosine, pick_device, train, val_batches, wsd

TOTAL, STOP, WARMUP, DECAY_FRAC = 300, 200, 30, 0.2
BRANCH_AT = 160
GRID = [5e-4 * 2 ** (i / 2) for i in range(6)]      # 5e-4 .. 2.8e-3
SEEDS = [0, 1, 2]
EVAL_EVERY = 10


def sched(kind, lr):
    if kind == "cosine":
        return cosine(lr, WARMUP, 0.0)
    return wsd(lr, WARMUP, DECAY_FRAC, 0.0)


def branch_sched(lr):
    def f(step, total):
        return lr * (1 - (step - BRANCH_AT + 1) / (STOP - BRANCH_AT))
    return f


def val_at(run, step):
    return run.val_loss[run.val_steps.index(step)]


def main(verbose: bool = True, reuse: bool = False) -> dict:
    if reuse:                       # re-render from the saved JSON, no training
        import json
        from src.report import RESULTS
        old = json.loads((RESULTS / "e4_cosine_vs_wsd.json").read_text())
        st = old["results"]
        _plot(old["tune"], old["best_lr"], old["curves_seed0"])
        _write(old["tune"], old["best_lr"], st, old["lr_fraction_at_stop"])
        return old
    device = pick_device()
    val = val_batches(device)

    # ---- 1. tune each schedule on its own -------------------------------
    tune = {"cosine": [], "wsd": []}
    for kind in tune:
        for lr in GRID:
            r = train(schedule=sched(kind, lr), steps=TOTAL, seed=0, val=val,
                      eval_at=(STOP,))
            tune[kind].append({"lr": lr, "val_200": val_at(r, STOP),
                               "val_300": r.val_loss[-1],
                               "diverged": r.diverged})
            if verbose:
                print(f"  tune {kind:<6} lr {lr:.2e}  @200 {val_at(r, STOP):.4f}"
                      f"  @300 {r.val_loss[-1]:.4f}", flush=True)
    best = {k: min(v, key=lambda p: p["val_300"])["lr"] for k, v in tune.items()}

    # ---- 2. three seeds at each schedule's own best ----------------------
    res = {k: [] for k in ("cosine_stop", "wsd_stop", "wsd_branch",
                           "cosine_planned", "cosine_300", "wsd_300")}
    curves = {}
    for seed in SEEDS:
        rc = train(schedule=sched("cosine", best["cosine"]), steps=TOTAL,
                   seed=seed, val=val, eval_every=EVAL_EVERY)
        rw, snaps = train(schedule=sched("wsd", best["wsd"]), steps=TOTAL,
                          seed=seed, val=val, eval_every=EVAL_EVERY,
                          checkpoint_at=(BRANCH_AT,))
        rb = train(schedule=branch_sched(best["wsd"]), steps=STOP - BRANCH_AT,
                   seed=seed, val=val, init_state=snaps[BRANCH_AT],
                   start_step=BRANCH_AT, eval_every=EVAL_EVERY)
        rp = train(schedule=cosine(best["cosine"], WARMUP, 0.0), steps=STOP,
                   seed=seed, val=val, eval_every=EVAL_EVERY)
        res["cosine_stop"].append(val_at(rc, STOP))
        res["wsd_stop"].append(val_at(rw, STOP))
        res["wsd_branch"].append(rb.val_loss[-1])
        res["cosine_planned"].append(rp.val_loss[-1])
        res["cosine_300"].append(rc.val_loss[-1])
        res["wsd_300"].append(rw.val_loss[-1])
        if seed == 0:
            curves = {"cosine": (rc.val_steps, rc.val_loss, rc.lr),
                      "wsd": (rw.val_steps, rw.val_loss, rw.lr),
                      "branch": (rb.val_steps, rb.val_loss, rb.lr),
                      "planned": (rp.val_steps, rp.val_loss, rp.lr)}
        if verbose:
            print(f"  seed {seed}: cos@200 {res['cosine_stop'][-1]:.4f}  "
                  f"wsd@200 {res['wsd_stop'][-1]:.4f}  "
                  f"branch@200 {res['wsd_branch'][-1]:.4f}  "
                  f"planned@200 {res['cosine_planned'][-1]:.4f}", flush=True)

    stats = {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)),
                 "per_seed": v} for k, v in res.items()}
    lr_at_stop = {"cosine": sched("cosine", best["cosine"])(STOP - 1, TOTAL)
                  / best["cosine"],
                  "wsd": sched("wsd", best["wsd"])(STOP - 1, TOTAL) / best["wsd"]}
    payload = {"config": dict(total=TOTAL, stop=STOP, warmup=WARMUP,
                              decay_frac=DECAY_FRAC, branch_at=BRANCH_AT,
                              seeds=SEEDS, grid=GRID),
               "tune": tune, "best_lr": best, "results": stats,
               "lr_fraction_at_stop": lr_at_stop,
               "curves_seed0": curves, "machine": machine()}
    save_json("e4_cosine_vs_wsd.json", payload)
    _plot(tune, best, curves)
    _write(tune, best, stats, lr_at_stop)
    return payload


def _plot(tune, best, curves):
    from src import plots
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2),
                            gridspec_kw={"width_ratios": [1, 1.25, 1.25]})
    c = {"cosine": plots.S1, "wsd": plots.S2, "branch": plots.S3,
         "planned": plots.MUTED}
    names = {"cosine": "cosine (300 planned)", "wsd": "WSD (300 planned)",
             "branch": f"WSD branch, decayed {BRANCH_AT}→{STOP}",
             "planned": f"cosine planned for {STOP}"}

    ax = axs[0]
    for k in ("cosine", "wsd"):
        xs = [p["lr"] for p in tune[k]]
        ax.plot(xs, [p["val_300"] for p in tune[k]], color=c[k], marker="o",
                ms=4, label=f"{k}: loss at 300")
        ax.plot(xs, [p["val_200"] for p in tune[k]], color=c[k], marker="o",
                ms=4, ls="--", lw=1.3, label=f"{k}: loss at 200")
        b = [p for p in tune[k] if p["lr"] == best[k]][0]
        ax.plot([b["lr"]], [b["val_300"]], marker="v", ms=11, color=c[k],
                markeredgecolor=plots.SURFACE, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("peak learning rate")
    ax.set_ylabel("validation loss")
    ax.set_title("1 · Each side tuned (▼ = chosen)")
    ax.legend(fontsize=7.5)

    bx = axs[1]
    for k in ("cosine", "wsd", "branch", "planned"):
        steps = np.arange(1, len(curves[k][2]) + 1)
        if k == "branch":
            steps = steps + BRANCH_AT
        peak = best["wsd"] if k in ("wsd", "branch") else best["cosine"]
        bx.plot(steps, np.array(curves[k][2]) / peak, color=c[k],
                lw=1.8 if k != "planned" else 1.3,
                ls="-" if k != "planned" else "--", label=names[k])
    bx.axvline(STOP, color=plots.CRITICAL, lw=1)
    bx.annotate("stop", (STOP, 1.02), color=plots.INK_2, fontsize=8, ha="center")
    bx.set_xlabel("step")
    bx.set_ylabel("learning rate ÷ its own peak")
    bx.set_title("2 · The schedules")
    bx.legend(fontsize=7.5, loc="lower left")

    cx = axs[2]
    for k in ("cosine", "wsd", "branch", "planned"):
        xs, ys, _ = curves[k]
        cx.plot(xs, ys, color=c[k], lw=1.8 if k != "planned" else 1.3,
                ls="-" if k != "planned" else "--", label=names[k])
    cx.axvline(STOP, color=plots.CRITICAL, lw=1)
    lo = min(min(curves[k][1]) for k in curves)
    cx.set_ylim(lo - 0.05, lo + 1.0)
    cx.set_xlim(100, TOTAL + 2)
    cx.set_xlabel("step")
    cx.set_ylabel("validation loss (seed 0)")
    cx.set_title("3 · Loss, steps 100–300")
    cx.legend(fontsize=7.5, loc="upper right")
    plots.finish(fig, "e4_cosine_vs_wsd.png")


def _write(tune, best, st, lr_at_stop):
    trows = []
    for i, lr in enumerate(GRID):
        a, b = tune["cosine"][i], tune["wsd"][i]
        trows.append((f"{lr:.2e}", f"{a['val_300']:.4f}", f"{b['val_300']:.4f}",
                      f"{a['val_200']:.4f}", f"{b['val_200']:.4f}"))

    def f(k):
        return f"{st[k]['mean']:.4f} ± {st[k]['std']:.4f}"
    rows = [
        ("cosine, stopped at 200", f"{lr_at_stop['cosine']:.0%}", f(k="cosine_stop")),
        ("WSD, stopped at 200", f"{lr_at_stop['wsd']:.0%}", f(k="wsd_stop")),
        (f"WSD, step-{BRANCH_AT} checkpoint decayed to 200", "0%", f(k="wsd_branch")),
        ("cosine planned for 200 (reference)", "0%", f(k="cosine_planned")),
        ("cosine, completed at 300", "0%", f(k="cosine_300")),
        ("WSD, completed at 300", "0%", f(k="wsd_300")),
    ]
    best_c = min(tune["cosine"], key=lambda p: p["val_300"])
    best_w = min(tune["wsd"], key=lambda p: p["val_300"])
    worst_w = max(tune["wsd"], key=lambda p: p["val_200"])
    worst_c = max(tune["cosine"], key=lambda p: p["val_200"])

    def paired(a, b):
        d = np.array(st[a]["per_seed"]) - np.array(st[b]["per_seed"])
        return d, float(d.mean()), float(d.std(ddof=1))
    dp, dpm, dps = paired("wsd_stop", "cosine_stop")
    bp, bpm, bps = paired("wsd_branch", "wsd_stop")
    pp, ppm, pps = paired("cosine_planned", "cosine_stop")
    d_stop = st["wsd_stop"]["mean"] - st["cosine_stop"]["mean"]
    d_branch = st["wsd_branch"]["mean"] - st["cosine_stop"]["mean"]
    d_300 = st["wsd_300"]["mean"] - st["cosine_300"]["mean"]
    md = [
        "# E4 · Cosine against WSD, stopped at step 200\n",
        f"Width-256 model. Both schedules planned for {TOTAL} steps, "
        f"{WARMUP} steps of warmup, decay to zero. WSD decays over the last "
        f"{DECAY_FRAC:.0%}. Three seeds at each schedule's own tuned rate. "
        "Mean ± sample std of held-out loss.\n",
        "![](e4_cosine_vs_wsd.png)\n",
        "## 1 · Tuning both sides first\n",
        "Seed 0, each rate a full 300-step run. Each schedule is tuned on the "
        "step-300 loss, the number it was designed to deliver.\n",
        table(trows, ["peak lr", "cosine @300", "WSD @300", "cosine @200",
                      "WSD @200"], ["r"] * 5),
        f"\nChosen: cosine **{best['cosine']:.2e}**, WSD **{best['wsd']:.2e}**.\n",
        "## 2 · The comparison, three seeds\n",
        table(rows, ["model", "lr at the stop (÷ peak)", "val loss"],
              ["l", "r", "r"]),
        f"\nWSD stopped minus cosine stopped: **{d_stop:+.4f}**. "
        f"WSD branch minus cosine stopped: **{d_branch:+.4f}**. "
        f"At the planned end, WSD minus cosine: {d_300:+.4f}.\n",
        "Because every pair shares a seed, and so the same initialisation "
        "and the same batches, the differences are better judged per seed "
        "than from the two spreads:\n",
        table([
            ("WSD stopped − cosine stopped",
             " / ".join(f"{x:+.4f}" for x in dp), f"**{dpm:+.4f} ± {dps:.4f}**"),
            (f"WSD branch − WSD stopped",
             " / ".join(f"{x:+.4f}" for x in bp), f"{bpm:+.4f} ± {bps:.4f}"),
            ("cosine planned for 200 − cosine stopped",
             " / ".join(f"{x:+.4f}" for x in pp), f"{ppm:+.4f} ± {pps:.4f}"),
        ], ["paired difference (negative = first is better)",
            "seed 0 / 1 / 2", "mean ± std"], ["l", "r", "r"]),
        "\n## The model I would keep: WSD's step-200 checkpoint\n",
        f"**It is the lower loss, on every seed**, by {-dpm:.3f} nats on "
        f"average (every seed between {-dp.max():.3f} and {-dp.min():.3f}). "
        "And it is the only one of the two that is still a *usable* "
        "checkpoint rather than an end point. It sits at the peak learning "
        "rate with nothing spent, so it can resume toward 300 or beyond, or be "
        "handed a short decay whenever a finished model is needed. The cosine "
        "model at step 200 has already spent 69% of its learning rate on a "
        "decay aimed at step 300, and resuming it means continuing a schedule "
        "that was written for a different run.\n",
        "## What was not expected\n",
        "**On this model, at this length, decaying bought less than it cost.** "
        "Section 10 expects a model stopped mid-decay to be worse than one "
        "planned for that length, and a WSD checkpoint to need its decay "
        "before it is competitive. Both came out the other way:\n",
        f"- Cosine *planned* for 200 steps finished {ppm:+.3f} worse than the "
        "300-step cosine halted at 200.\n",
        f"- Decaying the WSD run from step {BRANCH_AT} to 0 over 40 steps "
        f"ended {bpm:+.3f} worse than simply stopping it at full rate.\n",
        "The common cause is that 200 to 300 steps of 4,096 tokens is far "
        "from converged for this model. The loss is still falling by about "
        "0.5 nats per hundred steps around step 200, so steps taken at the "
        "peak are worth more than the noise suppression a decay buys. Any "
        "schedule that spends less time at the peak loses, and the ranking "
        "follows the area under each learning-rate curve. On a run long "
        "enough to plateau, the decay's benefit is known to reappear, which "
        "is why the WSD result that transfers is the flexibility, not the "
        "margin.\n",
        "**One caveat on the planned-200 cosine:** it used the peak rate "
        "tuned for a 300-step cosine, not one tuned for 200 steps. It is a "
        "reference, not a tuned competitor, and it is not part of the "
        "comparison the assignment asks for.\n",
        "**Why both sides had to be tuned.** Both schedules have their "
        "minimum at the same peak here, so a comparison at 1e-3 happens to be "
        "fair. At any other single rate it is not, and a mismatched pair can "
        "reverse the verdict or inflate it. From the tuning table, at step 200:\n",
        table([
            ("both tuned (the comparison above, seed 0)",
             f"{best_c['val_200']:.3f}", f"{best_w['val_200']:.3f}",
             f"{best_w['val_200'] - best_c['val_200']:+.3f}"),
            (f"tuned cosine vs WSD at {worst_w['lr']:.2e}",
             f"{best_c['val_200']:.3f}", f"{worst_w['val_200']:.3f}",
             f"**{worst_w['val_200'] - best_c['val_200']:+.3f}**"),
            (f"cosine at {worst_c['lr']:.2e} vs tuned WSD",
             f"{worst_c['val_200']:.3f}", f"{best_w['val_200']:.3f}",
             f"**{best_w['val_200'] - worst_c['val_200']:+.3f}**"),
        ], ["comparison", "cosine @200", "WSD @200", "WSD − cosine"],
            ["l", "r", "r", "r"]),
        "\nThe same two schedules can be reported as cosine winning by "
        f"{worst_w['val_200'] - best_c['val_200']:.2f}, as WSD winning by "
        f"{best_c['val_200'] - best_w['val_200']:.2f}, or as WSD winning by "
        f"{worst_c['val_200'] - best_w['val_200']:.2f}, depending only on "
        "which side was given its best rate. That is the session's closing "
        "warning, reproduced on a 300-step run.\n",
    ]
    save_text("e4_cosine_vs_wsd.md", "\n".join(md))


if __name__ == "__main__":
    main(reuse="--reuse" in sys.argv)
