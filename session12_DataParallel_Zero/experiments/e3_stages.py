"""E3 -- what each stage costs: the bytes, the wire and the clock.

This is the experiment the assignment is actually asking for.  Four
arrangements, thirty-two virtual GPUs, one model, and three questions asked of
each: how much memory does one rank hold, how much goes on the wire per step,
and where does the time go.

Every number here is measured rather than derived.  The memory comes from a
meter that is told about each tensor as it is allocated, so the bytes-per-weight
column is an audit of what the engine did, not an evaluation of the formula the
session notes give.  The two agree, which is the point -- but they are arrived at
from opposite directions, and if the engine had a redundant buffer in it the
audit would say so and the formula would not.

The arrangement of the results follows the arrangement of the sixteen bytes:
what is held, what is sent, what it costs in time, and then the same per-weight
figures carried up to 30 billion parameters, which is where they stop being
interesting and start being decisive.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json

from src.engines import STAGE_ORDER
from src.mesh import NVLINK, INFINIBAND
from src.model import Config, total_params
from src.report import machine, save_json, save_text, table
from src.sim import train_all, project, GiB, V5_PARAMS
from src import plots

WORLD = 32
MICRO = 2
STEPS = 8
MiB = 2 ** 20


def formula(stage: int, w: int) -> float:
    """The bytes per weight each stage is supposed to hold, from the notes."""
    return {0: 16.0, 1: 4 + 12 / w, 2: 2 + 14 / w, 3: 16 / w}[stage]


def wire_formula(stage: int, w: int) -> float:
    """The multiple of P each stage is supposed to send, ring-discounted."""
    return (2.0 if stage < 3 else 3.0) * (w - 1) / w


def _adam_scaling(sizes=(870_656, 8_000_000, 64_000_000), world: int = WORLD,
                  repeats: int = 5) -> list:
    """Time AdamW on a whole parameter vector and on one rank's share of it.

    The step table below finds almost no saving in the optimizer, which is not
    what the design says should happen.  This says why: at 27,000 numbers a
    shard, AdamW is eight tiny tensor operations and the time is the cost of
    launching them, not of doing them.  Walk the size up and the arithmetic
    starts to dominate, and the saving appears where it was always going to be.
    """
    import time
    import torch

    def one(n):
        p32 = torch.randn(n)
        m = torch.zeros(n)
        v = torch.zeros(n)
        g = torch.randn(n)
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            m.mul_(0.9).add_(g, alpha=0.1)
            v.mul_(0.95).addcmul_(g, g, value=0.05)
            p32.addcdiv_(m / 0.1, (v / 0.05).sqrt().add_(1e-8), value=-3e-4)
            best = min(best, time.perf_counter() - t0)
        return best

    rows = []
    for n in sizes:
        shard = n // world
        full_s, shard_s = one(n), one(shard)
        rows.append({"n": n, "shard": shard, "full_seconds": full_s,
                     "shard_seconds": shard_s,
                     "speedup": full_s / max(shard_s, 1e-12),
                     "ideal_speedup": float(world)})
    return rows


def main(verbose: bool = True) -> dict:
    cfg = Config()
    n = total_params(cfg)

    runs = train_all(world_size=WORLD, micro_batch=MICRO, steps=STEPS, cfg=cfg)
    small = train_all(world_size=8, micro_batch=MICRO, steps=STEPS, cfg=cfg)

    rows = []
    for name in STAGE_ORDER:
        r = runs[name]
        rows.append({
            "engine": name,
            "stage": r.stage,
            "bytes_per_weight_measured": r.bytes_per_weight,
            "bytes_per_weight_formula": formula(r.stage, WORLD),
            "resident_bytes": r.resident_bytes,
            "peak_bytes": r.peak_bytes,
            "peak_breakdown": r.peak_breakdown,
            "resident_breakdown": r.resident_breakdown,
            "activation_bytes": r.activation_bytes,
            "wire_in_P_measured": r.wire_in_P,
            "wire_in_P_formula": wire_formula(r.stage, WORLD),
            "collective_calls_per_step": r.collective_calls_per_step,
            "seconds_per_step": r.seconds_per_step,
            "seconds_compute": r.seconds_compute,
            "seconds_collective": r.seconds_collective,
            "seconds_optimizer": r.seconds_optimizer,
            "seconds_other": r.seconds_other,
            "final_loss": r.losses[-1],
            "bytes_per_weight_at_8": small[name].bytes_per_weight,
            "formula_at_8": formula(r.stage, 8),
        })

    exact = all(abs(r["bytes_per_weight_measured"] - r["bytes_per_weight_formula"])
                < 1e-9 for r in rows)
    exact8 = all(abs(r["bytes_per_weight_at_8"] - r["formula_at_8"]) < 1e-9
                 for r in rows)
    wire_exact = all(abs(r["wire_in_P_measured"] - r["wire_in_P_formula"]) < 1e-9
                     for r in rows)

    adam = _adam_scaling()
    v5 = project(runs)
    fig = _figure(rows, v5, n)

    payload = {"machine": machine(), "world_size": WORLD, "micro_batch": MICRO,
               "steps": STEPS, "n_params": n,
               "tokens_per_step": WORLD * MICRO * cfg.max_seq,
               "config": {"n_layer": cfg.n_layer, "d_model": cfg.d_model,
                          "n_head": cfg.n_head, "d_ff": cfg.d_ff,
                          "max_seq": cfg.max_seq, "vocab": cfg.vocab_size,
                          "groups": cfg.n_groups},
               "rows": rows, "adam_scaling": adam,
               "memory_matches_formula": exact,
               "memory_matches_formula_at_8": exact8,
               "wire_matches_formula": wire_exact,
               "v5_projection": v5,
               "losses": {k: r.losses for k, r in runs.items()}}
    save_json("e3_stages.json", payload)

    md = [
        "# E3 · What each stage costs",
        "",
        f"{n:,} parameters ({cfg.n_layer} blocks, d_model {cfg.d_model}, "
        f"{cfg.n_groups} groups), {WORLD} virtual GPUs, {MICRO} sequences each, "
        f"{WORLD * MICRO * cfg.max_seq:,} tokens per step, {STEPS} steps.",
        "",
        "## The memory, audited",
        "",
        "Measured by a meter that is told about every tensor as it is allocated, "
        "then compared against the formula the session notes give. The two columns "
        "are computed from opposite ends and never see each other.",
        "",
        table([(r["engine"], f"{r['bytes_per_weight_measured']:.4f}",
                f"{r['bytes_per_weight_formula']:.4f}",
                f"{r['resident_bytes'] / MiB:.2f} MiB",
                f"{r['peak_bytes'] / MiB:.2f} MiB",
                f"{16.0 / r['bytes_per_weight_measured']:.1f}x")
               for r in rows],
              ["arrangement", "bytes/weight measured", "formula, W=32",
               "resident per rank", "peak per rank", "saving"],
              ["l", "r", "r", "r", "r", "r"]),
        "",
        f"Identical in every row: `memory_matches_formula: {exact}`. At world size 8 "
        f"the same audit reproduces the session's own table — 16.00, 5.50, 3.75, "
        f"2.00 — exactly (`{exact8}`).",
        "",
        "### Where the peak actually goes",
        "",
        "Resident state is not what fills a card. This is the largest the meter ever "
        "saw, broken out by what was holding it.",
        "",
        table([(r["engine"],
                *[f"{r['peak_breakdown'].get(k, 0) / MiB:.2f}"
                  for k in ("weights", "gradients", "optimizer", "bucket",
                            "gathered", "activations")],
                f"{r['peak_bytes'] / MiB:.2f}")
               for r in rows],
              ["arrangement", "weights", "gradients", "optimizer",
               "grad bucket", "gathered", "activations", "peak MiB"],
              ["l"] + ["r"] * 7),
        "",
        "Three things are visible there that the bytes-per-weight column hides.",
        "",
        "**The gradient buffer is 2 bytes a weight and stages 0 and 1 never get it "
        "back.** It is allocated once and zeroed between steps, exactly as "
        "PyTorch's `.grad` is, so it is resident state and not a transient. That is "
        "why ZeRO-1 bottoms out at 4 bytes a weight however many GPUs are added: "
        "2 for the weights plus 2 for the gradients, neither of which it shards.",
        "",
        "**Stages 2 and 3 pay a bucket instead.** They reduce-scatter each group's "
        "gradient as the backward pass produces it and drop the full-size buffer "
        "immediately, so what they hold is one group's worth at a time rather than "
        "the whole model's. The `grad bucket` column is that transient, and it is "
        "the thing DeepSpeed's `bucket_size` names — bigger buckets, fewer and more "
        "efficient transfers, more memory held at once.",
        "",
        "**Activations do not shard at all.** Every arrangement holds the same "
        f"{rows[0]['activation_bytes'] / MiB:.2f} MiB of them, because they belong "
        "to the rank's own sequences and no other rank has a copy to share. ZeRO "
        "says nothing about activation memory; that is what activation "
        "checkpointing is for, and this model uses it — only the tensor entering "
        "each group is kept, and the rest is recomputed in the backward pass.",
        "",
        "## The wire, counted",
        "",
        table([(r["engine"], f"{r['wire_in_P_measured']:.4f}P",
                f"{r['wire_in_P_formula']:.4f}P",
                f"{r['collective_calls_per_step']:.0f}",
                f"{r['wire_in_P_measured'] * 2 * n / 1e6:.2f} MB")
               for r in rows],
              ["arrangement", "per rank per step, measured", "formula",
               "collective calls", "actual bytes"],
              ["l", "r", "r", "r", "r"]),
        "",
        f"`wire_matches_formula: {wire_exact}`. Stages 1 and 2 send precisely what "
        "data parallelism sends. Stage 3 sends half as much again, and the extra "
        "half is one all-gather of the weights in the forward pass and one more in "
        "the backward — "
        f"{rows[3]['collective_calls_per_step']:.0f} calls a step against "
        f"{rows[0]['collective_calls_per_step']:.0f}, because every group is now "
        "fetched twice and returned twice.",
        "",
        "## The clock",
        "",
        "These are threads on one laptop CPU, so the seconds below are not a "
        "prediction about a GPU cluster — 32 ranks share 10 cores, and the "
        "`in collectives` column is barrier and memcpy time, not network time. What "
        "the table *is* good for is the shape: what each arrangement does with a "
        "step, and how much of one it spends not computing.",
        "",
        table([(r["engine"], f"{r['seconds_per_step'] * 1e3:.1f}",
                f"{r['seconds_compute'] * 1e3:.1f}",
                f"{r['seconds_optimizer'] * 1e3:.1f}",
                f"{r['seconds_other'] * 1e3:.1f}",
                f"{r['seconds_collective'] * 1e3:.1f}",
                f"{100 * r['seconds_collective'] / r['seconds_per_step']:.0f}%")
               for r in rows],
              ["arrangement", "ms/step", "forward+backward", "AdamW",
               "casts and copies", "in collectives", "share stopped"],
              ["l", "r", "r", "r", "r", "r", "r"]),
        "",
        "The four middle columns are disjoint and add to the first: time inside a "
        "collective is taken out of the phase it happened in, so stage 2's "
        "reduce-scatters are counted once, in the last column, and not again in the "
        "backward pass they interrupt.",
        "",
        "The AdamW column is the interesting failure. The sharded stages update a "
        f"thirty-second of the weights and it buys them almost nothing: "
        f"{rows[0]['seconds_optimizer'] * 1e3:.1f} ms a step against "
        f"{rows[3]['seconds_optimizer'] * 1e3:.1f}. Timed on its own, away from the "
        "mesh, the same arithmetic behaves the way it should:",
        "",
        table([(f"{a['n']:,}", f"{a['shard']:,}", f"{a['full_seconds'] * 1e3:.3f}",
                f"{a['shard_seconds'] * 1e3:.3f}", f"{a['speedup']:.1f}x")
               for a in adam],
              ["parameters", "one rank's shard (W=32)", "AdamW on all, ms",
               "AdamW on the shard, ms", "speedup"],
              ["r", "r", "r", "r", "r"]),
        "",
        f"{adam[0]['speedup']:.0f}x at this model's size and "
        f"{adam[-1]['speedup']:.0f}x at {adam[-1]['n'] // 10 ** 6}M — past 32x, "
        "because a 2M-element shard also fits in cache and a 64M-element vector "
        f"does not. So the saving is real and it is large. What the {STEPS}-step run "
        "above cannot show is *this machine*: 32 Python threads on 10 cores, where a "
        "tensor operation on 27,000 numbers spends its time in dispatch — holding "
        "the interpreter lock — rather than in arithmetic. The ranks queue behind "
        "each other no matter how small their shards get.",
        "",
        "That is the honest boundary of this simulator and it is worth stating "
        "plainly: **the bytes it reports are exact and the seconds it reports are "
        "about a laptop.** Memory per rank and traffic per step are counted, not "
        "modelled, and they would be the same numbers on 32 H100s. Wall-clock is "
        "not, which is why experiment 5 prices the time from the byte counts and a "
        "stated bandwidth instead of from this machine's clock.",
        "",
        "Thirty-one of every thirty-two of data parallelism's updates are a "
        "computation the other ranks are performing at the same moment on the same "
        "inputs. *That* is the redundancy the **O** in Zero Redundancy Optimizer "
        "names, and it is the one saving in this session that costs nothing at all: "
        "the sharded stages do less arithmetic, not merely less storing.",
        "",
        "## The same per-weight figures, at 30 billion",
        "",
        "Nothing here is re-derived. The measured bytes-per-weight from the first "
        "table, multiplied by 30e9, against a card that holds 74.5 GiB.",
        "",
        table([(name, f"{v['bytes_per_weight']:.4f}", f"{v['state_gib']:,.1f} GiB",
                "fits" if v["fits_80gb_card"] else "does not fit",
                f"{v['wire_bytes'] / 1e9:.0f} GB",
                f"{v['seconds_nvlink']:.2f} s", f"{v['seconds_infiniband']:.2f} s")
               for name, v in v5.items()],
              ["arrangement", "bytes/weight", "state per GPU", "on an 80 GB card",
               "wire per step", "NVLink", "InfiniBand"],
              ["l", "r", "r", "c", "r", "r", "r"]),
        "",
        f"At {WORLD} GPUs, on the training state alone: data parallelism needs "
        f"{v5['data parallel']['state_gib']:,.0f} GiB a card and ZeRO-1 needs "
        f"{v5['ZeRO-1']['state_gib']:,.0f} GiB, and a card has 74.5. ZeRO-2 fits "
        f"with {74.5 - v5['ZeRO-2']['state_gib']:,.1f} GiB to spare, which has to "
        "cover activations, and ZeRO-3 fits with room to spare on far fewer GPUs. "
        "Experiment 4 walks the world size to find where each line crosses.",
        "",
        "![](e3_stages.png)",
        "",
    ]
    save_text("e3_stages.md", "\n".join(md))

    if verbose:
        print(json.dumps({"rows": rows, "v5": v5,
                          "adam_scaling": adam,
                          "memory_matches_formula": exact,
                          "wire_matches_formula": wire_exact},
                         indent=2, default=str))
    return payload


def _figure(rows, v5, n_params, name="e3_stages.png"):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.6, 4.2))
    labels = ["data\nparallel", "ZeRO-1", "ZeRO-2", "ZeRO-3"]
    x = list(range(4))

    # -- what one rank holds, per weight
    cats = [("weights", plots.S1, "bf16 weights"),
            ("gradients", plots.S2, "bf16 gradients"),
            ("optimizer", plots.S3, "fp32 master + 2 moments")]
    bottoms = [0.0] * 4
    for key, colour, label in cats:
        vals = [r["resident_breakdown"].get(key, 0) / n_params for r in rows]
        ax1.bar(x, vals, 0.62, bottom=bottoms, color=colour, label=label)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for i, r in enumerate(rows):
        ax1.annotate(f"{r['bytes_per_weight_measured']:.2f}",
                     (i, r["bytes_per_weight_measured"]), ha="center",
                     va="bottom", fontsize=8.5, color=plots.INK_2)
    ax1.set_axisbelow(True)
    ax1.set_xticks(x, labels)
    ax1.set_ylim(0, 18.4)
    ax1.set_ylabel("bytes held per weight, per GPU")
    ax1.set_title("What one rank keeps (W = 32)", loc="left")
    ax1.legend(loc="upper right")

    # -- what goes on the wire
    ws = [r["wire_in_P_measured"] for r in rows]
    ax2.bar(x, ws, 0.62, color=[plots.S1, plots.S1, plots.S1, plots.S4])
    for i, v in enumerate(ws):
        ax2.annotate(f"{v:.3f}P", (i, v), ha="center", va="bottom", fontsize=8.5,
                     color=plots.INK_2)
    ax2.axhline(2.0, color=plots.MUTED, lw=0.9, ls=(0, (4, 3)))
    ax2.axhline(3.0, color=plots.MUTED, lw=0.9, ls=(0, (4, 3)))
    ax2.annotate("2P", (-0.55, 2.0), color=plots.MUTED, fontsize=8, va="center")
    ax2.annotate("3P", (-0.55, 3.0), color=plots.MUTED, fontsize=8, va="center")
    ax2.set_axisbelow(True)
    ax2.set_xticks(x, labels)
    ax2.set_xlim(-0.75, 3.6)
    ax2.set_ylim(0, 3.5)
    ax2.set_ylabel("bytes per rank per step, in units of P")
    ax2.set_title("What goes on the wire", loc="left")

    # -- the same thing at 30B, against the card
    gib = [v5[k]["state_gib"] for k in
           ("data parallel", "ZeRO-1", "ZeRO-2", "ZeRO-3")]
    colours = [plots.CRITICAL if g > 74.5 else plots.S3 for g in gib]
    ax3.bar(x, gib, 0.62, color=colours)
    ax3.axhline(74.5, color=plots.INK_2, lw=1.4)
    ax3.annotate("one 80 GB card\n= 74.5 GiB", (3.95, 74.5),
                 color=plots.INK_2, fontsize=8.5, va="center", ha="left")
    ax3.set_xlim(-0.6, 5.6)
    for i, g in enumerate(gib):
        ax3.annotate(f"{g:,.0f}", (i, g), ha="center", va="bottom", fontsize=8.5,
                     color=plots.INK_2)
    ax3.set_axisbelow(True)
    ax3.set_yscale("log")
    ax3.set_ylim(3, 1400)
    ax3.set_xticks(x, labels)
    ax3.set_ylabel("GiB of training state per GPU")
    ax3.set_title("The same per-weight cost at 30B (W = 32)", loc="left")

    return plots.finish(fig, name,
                        "Left and centre: measured on 32 virtual GPUs training an "
                        "870,656-parameter model. Right: those measured per-weight "
                        "figures multiplied by 30e9. Log scale.")


if __name__ == "__main__":
    main()
