"""E4 -- the memory wall, walked from one virtual GPU to sixty-four.

Experiment 3 measured the four arrangements at one world size.  This one walks
the world size and watches what each arrangement does with the extra GPUs, and
the answer is not the same for all four.  Two of them get cheaper per rank
without limit; two of them stop.

The stopping is the whole result.  Data parallelism replicates everything, so
adding GPUs does nothing at all to what one of them holds.  ZeRO-1 shards the
twelve bytes of optimizer state and leaves the weights and the gradients
replicated, so it falls towards 4 bytes a weight and cannot go below it however
many machines are bought.  Four bytes is a floor with a size attached: it fills
an 80 GB card at exactly 20 billion parameters, which is why a 30B model has no
data-parallel or ZeRO-1 configuration at any world size.

Everything about the 30B model here is the *measured* bytes-per-weight of a
running engine multiplied by 30e9.  Nothing is quoted.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json

from src.engines import STAGE_ORDER
from src.model import Config
from src.report import machine, save_json, save_text, table
from src.sim import train, project_state, GiB, V5_PARAMS, CARD_BYTES
from src import plots

WORLDS = (1, 2, 4, 8, 16, 32, 64)
STEPS = 2
MICRO = 1


def formula(stage: int, w: int) -> float:
    return {0: 16.0, 1: 4 + 12 / w, 2: 2 + 14 / w, 3: 16 / w}[stage]


def main(verbose: bool = True) -> dict:
    cfg = Config()

    measured = {}
    for name in STAGE_ORDER:
        for w in WORLDS:
            r = train(name, world_size=w, micro_batch=MICRO, steps=STEPS,
                      cfg=cfg, warmup=0)
            measured[(name, w)] = {
                "bytes_per_weight": r.bytes_per_weight,
                "formula": formula(r.stage, w),
                "wire_in_P": r.wire_in_P,
                "activation_bytes": r.activation_bytes,
                "peak_bytes": r.peak_bytes,
                "stage": r.stage,
            }

    exact = all(abs(v["bytes_per_weight"] - v["formula"]) < 1e-9
                for v in measured.values())

    # the ladder: measured bytes per weight, carried to 30B
    ladder = []
    for name in STAGE_ORDER:
        row = {"engine": name, "by_world": {}}
        for w in WORLDS:
            bpw = measured[(name, w)]["bytes_per_weight"]
            state = project_state(bpw, V5_PARAMS)
            row["by_world"][w] = {
                "bytes_per_weight": bpw,
                "gib": state / GiB,
                "fits": state < CARD_BYTES,
                "largest_model_that_fits": CARD_BYTES / bpw,
            }
        fitting = [w for w in WORLDS if row["by_world"][w]["fits"]]
        row["fits_from"] = min(fitting) if fitting else None
        ladder.append(row)

    floors = {
        "data parallel": 16.0,
        "ZeRO-1": 4.0,
        "ZeRO-2": 2.0,
        "ZeRO-3": 0.0,
    }
    boundary = {
        "bytes_per_weight": 4.0,
        "card_gib": CARD_BYTES / GiB,
        "params_that_fill_a_card": CARD_BYTES / 4.0,
        "v5_params": V5_PARAMS,
        "v5_needs_gib": project_state(4.0, V5_PARAMS) / GiB,
    }

    fig = _figure(ladder, measured)

    payload = {"machine": machine(), "worlds": list(WORLDS),
               "measured_matches_formula": exact,
               "measured": {f"{k[0]}|{k[1]}": v for k, v in measured.items()},
               "ladder": ladder, "floors": floors, "boundary": boundary,
               "activation_bytes_per_rank": measured[("ZeRO-3", 32)]["activation_bytes"],
               "config": {"n_layer": cfg.n_layer, "d_model": cfg.d_model,
                          "max_seq": cfg.max_seq, "micro_batch": MICRO}}
    save_json("e4_ladder.json", payload)

    md = [
        "# E4 · The memory wall, one virtual GPU to sixty-four",
        "",
        "Every cell below comes from an engine that ran. The world size on the left "
        "is a real number of threads, each holding real tensors, and the meter "
        "counted them.",
        "",
        "## Bytes per weight, measured",
        "",
        table([(name, *[f"{measured[(name, w)]['bytes_per_weight']:.3f}"
                        for w in WORLDS])
               for name in STAGE_ORDER],
              ["arrangement", *[f"W={w}" for w in WORLDS]],
              ["l"] + ["r"] * len(WORLDS)),
        "",
        f"Against the formula, in all {len(measured)} cells: "
        f"`measured_matches_formula: {exact}`.",
        "",
        "Read the rows rather than the columns and the session's argument appears "
        "on its own:",
        "",
        "- **Data parallelism is flat.** 16.000 at every world size. Buying GPUs "
        "buys throughput and nothing else; each one still holds the entire model, "
        "the entire gradient and the entire optimizer state.",
        "- **ZeRO-1 approaches 4 and stops.** It shards the twelve bytes of "
        "optimizer state and leaves the 2-byte weights and 2-byte gradients "
        "replicated. 4.375 at W=32, 4.188 at W=64, 4.000 in the limit.",
        "- **ZeRO-2 approaches 2 and stops**, for the same reason with one fewer "
        "term: the weights are still replicated.",
        "- **ZeRO-3 has no floor.** 16/W, all the way down. It is the only "
        "arrangement where adding a GPU reduces what every GPU holds.",
        "",
        "## The same rows at 30 billion parameters",
        "",
        "Measured bytes per weight × 30e9, against a card that holds 74.5 GiB. "
        "Cells in bold fit.",
        "",
        table([(row["engine"], *[
                    (f"**{row['by_world'][w]['gib']:,.1f}**"
                     if row["by_world"][w]["fits"]
                     else f"{row['by_world'][w]['gib']:,.1f}")
                    for w in WORLDS])
               for row in ladder],
              ["arrangement", *[f"{w} GPU" + ("s" if w > 1 else "") for w in WORLDS]],
              ["l"] + ["r"] * len(WORLDS)),
        "",
        table([(row["engine"],
                f"{row['fits_from']} GPUs" if row["fits_from"] else "never",
                f"{row['by_world'][32]['largest_model_that_fits'] / 1e9:,.1f}B")
               for row in ladder],
              ["arrangement", "fits an 80 GB card from", "largest model at W=32"],
              ["l", "r", "r"]),
        "",
        "## Why two of the rows never fit",
        "",
        "Not because 30 billion is a large number — because of what stays "
        "replicated. Data parallelism and ZeRO-1 both keep the 2-byte weights and "
        "the 2-byte gradients on every card. That is 4 bytes a weight that no world "
        "size touches, and 4 bytes a weight has a model size attached to it:",
        "",
        table([("a card", f"{boundary['card_gib']:.1f} GiB"),
               ("replicated bytes per weight (weights + gradients)", "4"),
               ("parameters that fill the card",
                f"{boundary['params_that_fill_a_card'] / 1e9:.1f}B"),
               ("our model", f"{boundary['v5_params'] / 1e9:.0f}B"),
               ("what it needs, at 4 bytes a weight",
                f"{boundary['v5_needs_gib']:.1f} GiB")],
              ["", ""], ["l", "r"]),
        "",
        f"**{boundary['params_that_fill_a_card'] / 1e9:.1f} billion parameters is "
        "the boundary**, and it is a property of the arrangement rather than of the "
        "hardware budget. A 20B model sits on the line; ours sits "
        f"{boundary['v5_needs_gib'] / boundary['card_gib']:.1f}x past it. Every "
        "thousand-GPU cluster in the world is still short of a card for it under "
        "ZeRO-1.",
        "",
        "## What the table is not counting",
        "",
        "Training state only. The activations are extra, they belong to the rank's "
        "own sequences, and no ZeRO stage shards them — this run measured "
        f"{payload['activation_bytes_per_rank'] / 2 ** 20:.2f} MiB of them per rank "
        f"for {MICRO} sequence of {cfg.max_seq} tokens, *with* checkpointing "
        "already on. A real V5 micro-batch is far larger and the activation term "
        "with it, so the ZeRO-2 row's 68.1 GiB at 32 GPUs is not a configuration "
        "that fits — it is a configuration with 6.4 GiB left for everything else, "
        "which is not enough. Read the fitted cells as an upper bound on what is "
        "possible, not as a plan.",
        "",
        "![](e4_ladder.png)",
        "",
    ]
    save_text("e4_ladder.md", "\n".join(md))

    if verbose:
        print(json.dumps({"measured_matches_formula": exact, "ladder": ladder,
                          "boundary": boundary}, indent=2, default=str))
    return payload


def _figure(ladder, measured, name="e4_ladder.png"):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.3))
    colours = {"data parallel": plots.CRITICAL, "ZeRO-1": plots.S2,
               "ZeRO-2": plots.S1, "ZeRO-3": plots.S3}

    for row in ladder:
        ws = sorted(row["by_world"])
        ys = [row["by_world"][w]["gib"] for w in ws]
        c = colours[row["engine"]]
        ax1.plot(ws, ys, marker="o", ms=4, color=c)
        fits = [(w, y) for w, y in zip(ws, ys) if y < 74.5]
        if fits:
            ax1.plot([f[0] for f in fits], [f[1] for f in fits], lw=7.5,
                     color=c, alpha=0.45, solid_capstyle="round", zorder=1)
        ax1.annotate(row["engine"], (ws[-1], ys[-1]), color=c, fontsize=8.5,
                     va="center", ha="left", xytext=(7, 0),
                     textcoords="offset points")
    ax1.axhline(74.5, color=plots.INK, lw=1.4)
    ax1.annotate("one 80 GB card = 74.5 GiB", (1, 80), color=plots.INK,
                 fontsize=8.5, va="bottom", ha="left")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(sorted(ladder[0]["by_world"]),
                   [str(w) for w in sorted(ladder[0]["by_world"])])
    ax1.set_xlim(0.9, 260)
    ax1.set_ylim(3, 1500)
    ax1.set_xlabel("GPUs")
    ax1.set_ylabel("GiB of training state per GPU, 30B model")
    ax1.set_title("The memory wall", loc="left")

    for row in ladder:
        ws = sorted(row["by_world"])
        ys = [row["by_world"][w]["largest_model_that_fits"] / 1e9 for w in ws]
        c = colours[row["engine"]]
        ax2.plot(ws, ys, marker="o", ms=4, color=c)
        ax2.annotate(row["engine"], (ws[-1], ys[-1]), color=c, fontsize=8.5,
                     va="center", ha="left", xytext=(7, 0),
                     textcoords="offset points")
    ax2.axhline(30, color=plots.INK, lw=1.4)
    ax2.annotate("our model, 30B", (1, 32), color=plots.INK, fontsize=8.5,
                 va="bottom", ha="left")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks(sorted(ladder[0]["by_world"]),
                   [str(w) for w in sorted(ladder[0]["by_world"])])
    ax2.set_xlim(0.9, 260)
    ax2.set_xlabel("GPUs")
    ax2.set_ylabel("largest model whose state fits one card, billions")
    ax2.set_title("The same wall, read the other way", loc="left")

    return plots.finish(fig, name,
                        "Both panels: measured bytes per weight from a running "
                        "engine at each world size, multiplied out to 30e9. Lines "
                        "are drawn heavy where the arrangement fits. Log-log.")


if __name__ == "__main__":
    main()
