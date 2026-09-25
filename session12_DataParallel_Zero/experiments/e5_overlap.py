"""E5 -- communication that happens during computation costs nothing.

Experiment 3 counted the bytes and said plainly that its seconds are about a
laptop.  This is where the seconds are done properly.  Three inputs, each from
its own place: the per-layer *shape* of a pass, measured on the real model; the
*volume* each arrangement puts on the wire, counted by the ledger in experiment
3; and a *bandwidth*, stated.  Nothing is timed on a thread barrier.

The quantity being measured is the one that decides whether a distributed run is
worth having.  Not how much is sent -- how much of it was sent while the GPU was
busy anyway.  A backward pass runs from the last layer to the first, so the last
layer's gradients are finished long before the first layer's are, and a transfer
started then is free.  A forward pass under ZeRO has the mirror-image problem:
the weights for the next layer have to be fetched before that layer can run, and
a fetch issued early enough is also free.

The four arrangements differ in how their traffic is split between those two
windows, and this is where that turns out to matter:

    arrangement   forward window   backward window   total
    data parallel        —               2P            2P
    ZeRO-1, ZeRO-2       P               P             2P
    ZeRO-3              2P               2P            4P*

    * stage 3 gathers in the forward (P) and again in the backward (P) and
      reduce-scatters the gradients (P), which is the 3P of the notes; the
      forward's share is P and the backward's is 2P.  Stages 1 and 2 all-gather
      the updated weights after the optimizer step, which lands in the next
      step's forward window.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json
import statistics
import time

import torch

from src.mesh import NVLINK, INFINIBAND
from src.model import (Config, LocalParams, forward_backward, group_numel,
                       init_group)
from src.overlap import simulate_pass, best_schedule
from src.report import machine, save_json, save_text, table
from src import plots

PROFILE_LAYERS = 12
V5_PARAMS = 30_000_000_000
V5_LAYERS = 48
WORLD = 32
RING = (WORLD - 1) / WORLD
P_BYTES = 2 * V5_PARAMS

# whole-step compute times from the session notes, for ~1M tokens on 64 cards
CARDS = {"64 x H100": 7.10, "64 x B200": 3.12}
# a step is 6N FLOPs a token: 2N forward, 4N backward
FORWARD_SHARE, BACKWARD_SHARE = 1 / 3, 2 / 3

# how the traffic divides between the two windows, in units of P
TRAFFIC = {
    "data parallel": {"forward": 0.0, "backward": 2.0},
    "ZeRO-1": {"forward": 1.0, "backward": 1.0},
    "ZeRO-2": {"forward": 1.0, "backward": 1.0},
    "ZeRO-3": {"forward": 1.0, "backward": 2.0},
}


class Marks:
    """A stand-in for the memory meter that records times instead of bytes.

    ``forward_backward`` calls ``hold`` when a group's forward output appears
    and ``free`` when that group's backward is finished with it, so passing one
    of these in gets a timestamp at both ends of every group without adding a
    single line of timing code to the model.
    """

    def __init__(self):
        self.held, self.freed = {}, {}

    def hold(self, key, tensor):
        self.held[key] = time.perf_counter()

    def free(self, key):
        self.freed[key] = time.perf_counter()


def _profile(repeats: int = 5) -> dict:
    """Per-group forward and backward time, measured on one thread, alone.

    Timed at the group boundaries rather than around the whole call, because
    timing it around the whole call attributes the forward pass to whichever
    group happens to hand its gradient back first -- which is the last group,
    and which is how a transformer's output head comes to look like a quarter of
    the backward pass when it is nothing of the sort.
    """
    torch.set_num_threads(1)
    cfg = Config(n_layer=PROFILE_LAYERS)
    flats = {g: init_group(cfg, g, seed=7) for g in range(cfg.n_groups)}
    x = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq))
    G = cfg.n_groups
    fwd = {g: [] for g in range(G)}
    bwd = {g: [] for g in range(G)}

    for _ in range(repeats):
        marks = Marks()
        t0 = time.perf_counter()
        forward_backward(cfg, LocalParams(flats), x, y, meter=marks,
                         on_grad=lambda g, grad: None)
        prev = t0
        for g in range(G):                      # forward: groups in order
            now = marks.held[f"activations:boundary{g}"]
            fwd[g].append(now - prev)
            prev = now
        for g in reversed(range(G)):            # backward: groups in reverse
            now = marks.freed[f"activations:boundary{g}"]
            bwd[g].append(now - prev)
            prev = now

    f = {g: statistics.median(v) for g, v in fwd.items()}
    b = {g: statistics.median(v) for g, v in bwd.items()}
    numels = {g: group_numel(cfg, g) for g in range(G)}
    return {
        "n_groups": G,
        "n_layer": cfg.n_layer,
        "repeats": repeats,
        "seconds_forward_by_group": f,
        "seconds_backward_by_group": b,
        "seconds_total_forward": sum(f.values()),
        "seconds_total_backward": sum(b.values()),
        "backward_over_forward": sum(b.values()) / sum(f.values()),
        "params_by_group": numels,
        "compute_share": {g: b[g] / sum(b.values()) for g in b},
        "forward_share": {g: f[g] / sum(f.values()) for g in f},
        "param_share": {g: numels[g] / sum(numels.values()) for g in numels},
    }


def _v5_shape(profile) -> tuple:
    """A 30B model's per-layer compute shares and per-layer byte shares.

    The *shape* is the measured one -- the ends of a transformer do not cost
    what a block costs, and a 30B model has the same two ends -- resampled onto
    48 blocks. The parameters are spread evenly across the blocks, with the
    embedding and head keeping the small share the measured model gives them.
    """
    shares = profile["compute_share"]
    block = statistics.median([shares[g] for g in range(1, profile["n_layer"] + 1)])
    raw = [shares[0]] + [block] * V5_LAYERS + [shares[profile["n_groups"] - 1]]
    compute_share = [r / sum(raw) for r in raw]

    p = profile["param_share"]
    ends = p[0] + p[profile["n_groups"] - 1]
    per_block = (1.0 - ends) / V5_LAYERS
    byte_share = [p[0]] + [per_block] * V5_LAYERS + [p[profile["n_groups"] - 1]]
    return compute_share, byte_share


def _window(compute_share, seconds):
    return [s * seconds for s in compute_share]


def main(verbose: bool = True) -> dict:
    profile = _profile()
    compute_share, byte_share = _v5_shape(profile)

    # -- what it costs with no overlap at all --------------------------------
    baseline = {}
    for card, step in CARDS.items():
        for link in (NVLINK, INFINIBAND):
            for name, split in TRAFFIC.items():
                volume = (split["forward"] + split["backward"]) * P_BYTES * RING
                wire = volume / (link.gb_per_s * 1e9)
                baseline[f"{name} | {card} | {link.name.split(',')[0]}"] = {
                    "arrangement": name, "card": card,
                    "link": link.name.split(",")[0],
                    "volume_in_P": volume / P_BYTES,
                    "compute_seconds": step,
                    "wire_seconds": wire,
                    "ratio": wire / step,
                    "step_serial": step + wire,
                }

    # -- the bucket sweep, on the backward window ----------------------------
    sweeps = {}
    for card, step in CARDS.items():
        for link in (NVLINK, INFINIBAND):
            times = _window(compute_share, step * BACKWARD_SHARE)
            grads = [s * 2.0 * P_BYTES * RING for s in byte_share]   # DP: 2P
            rows = []
            for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 48):
                s = simulate_pass(times, grad_bytes=grads, bucket_layers=k,
                                  gb_per_s=link.gb_per_s,
                                  latency_s=link.latency_us * 1e-6)
                rows.append(s.as_dict())
            sweeps[f"{card} | {link.name.split(',')[0]}"] = {
                "card": card, "link": link.name, "window_seconds": sum(times),
                "rows": rows}

    # -- when does a smaller bucket stop helping? ----------------------------
    sensitivity = []
    times = _window(compute_share, CARDS["64 x B200"] * BACKWARD_SHARE)
    grads = [s * 2.0 * P_BYTES * RING for s in byte_share]
    for fixed_ms in (0.01, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0):
        rows = [simulate_pass(times, grad_bytes=grads, bucket_layers=k,
                              gb_per_s=INFINIBAND.gb_per_s,
                              latency_s=fixed_ms * 1e-3)
                for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 48)]
        b = min(rows, key=lambda s: s.pass_seconds)
        sensitivity.append({"fixed_cost_ms": fixed_ms,
                            "best_bucket": b.bucket_layers,
                            "pass_seconds": b.pass_seconds,
                            "smallest_bucket_seconds": rows[0].pass_seconds})

    # -- the whole step, all four arrangements, both windows -----------------
    steps = {}
    for card, step in CARDS.items():
        for link in (NVLINK, INFINIBAND):
            fwd_times = _window(compute_share, step * FORWARD_SHARE)
            bwd_times = _window(compute_share, step * BACKWARD_SHARE)
            for name, split in TRAFFIC.items():
                fwd = None
                if split["forward"]:
                    gathers = [s * split["forward"] * P_BYTES * RING
                               for s in byte_share]
                    fwd = best_schedule(fwd_times, gather_bytes=gathers,
                                        grad_bytes=None, reverse=False,
                                        gb_per_s=link.gb_per_s,
                                        latency_s=link.latency_us * 1e-6)
                if name == "ZeRO-3":
                    gathers = [s * 1.0 * P_BYTES * RING for s in byte_share]
                    grads_b = [s * 1.0 * P_BYTES * RING for s in byte_share]
                else:
                    gathers = None
                    grads_b = [s * split["backward"] * P_BYTES * RING
                               for s in byte_share]
                bwd = best_schedule(bwd_times, gather_bytes=gathers,
                                    grad_bytes=grads_b, reverse=True,
                                    gb_per_s=link.gb_per_s,
                                    latency_s=link.latency_us * 1e-6)
                total = (fwd.pass_seconds if fwd else step * FORWARD_SHARE) \
                    + bwd.pass_seconds
                moved = ((fwd.transfer_seconds if fwd else 0.0)
                         + bwd.transfer_seconds)
                exposed = (fwd.exposed_seconds if fwd else 0.0) + bwd.exposed_seconds
                steps[f"{name} | {card} | {link.name.split(',')[0]}"] = {
                    "arrangement": name, "card": card,
                    "link": link.name.split(",")[0],
                    "step_seconds": total,
                    "compute_seconds": step,
                    "transfer_seconds": moved,
                    "exposed_seconds": exposed,
                    "hidden_fraction": 1 - exposed / moved if moved else 1.0,
                    "overhead_pct": 100 * (total - step) / step,
                    "forward_bucket": fwd.bucket_layers if fwd else None,
                    "forward_prefetch": fwd.prefetch if fwd else None,
                    "backward_bucket": bwd.bucket_layers,
                    "backward_stalled": bwd.stalled_seconds,
                }

    best = {k: min(v["rows"], key=lambda r: r["pass_seconds"])
            for k, v in sweeps.items()}
    fig = _figure(sweeps, steps)

    payload = {"machine": machine(), "profile": profile,
               "v5": {"params": V5_PARAMS, "layers": V5_LAYERS,
                      "P_bytes": P_BYTES, "world_size": WORLD, "cards": CARDS,
                      "forward_share": FORWARD_SHARE,
                      "backward_share": BACKWARD_SHARE},
               "traffic_split": TRAFFIC, "baseline": baseline, "sweeps": sweeps,
               "best_bucket": best, "sensitivity": sensitivity, "steps": steps}
    save_json("e5_overlap.json", payload)

    ib_h = "64 x H100 | InfiniBand"
    ib_b = "64 x B200 | InfiniBand"
    nv_h = "64 x H100 | NVLink"

    md = [
        "# E5 · Communication that happens during computation costs nothing",
        "",
        "## The profile the simulation runs on",
        "",
        f"A {PROFILE_LAYERS}-block model, one thread, nothing else running, "
        "timestamped at every group boundary in both directions.",
        "",
        table([(f"group {g}" + ("  (embeddings)" if g == 0 else
                                "  (head)" if g == profile["n_groups"] - 1 else
                                "  (block)"),
                f"{profile['params_by_group'][g]:,}",
                f"{profile['seconds_forward_by_group'][g] * 1e3:.2f}",
                f"{profile['seconds_backward_by_group'][g] * 1e3:.2f}",
                f"{100 * profile['compute_share'][g]:.1f}%")
               for g in (0, 1, 2, profile["n_groups"] - 2, profile["n_groups"] - 1)],
              ["group", "parameters", "forward ms", "backward ms",
               "share of the backward"],
              ["l", "r", "r", "r", "r"]),
        "",
        f"The whole backward is {profile['backward_over_forward']:.2f}x the whole "
        "forward here, against the 2x that 6N FLOPs a token predicts, because this "
        "repository's backward recomputes each group's activations before "
        "differentiating them and so carries the forward's work a second time. The "
        "projection below uses the 2x, not the "
        f"{profile['backward_over_forward']:.2f}x, because the cluster step times it "
        "starts from are for a production step and recomputation is a choice that "
        "step may not have made. Assuming the smaller backward is the conservative "
        "direction: a longer backward is a longer window to hide transfers in.",
        "",
        "The blocks cost the same as each other and the two ends do not. That shape "
        f"is what is carried over to a {V5_LAYERS}-block, 30-billion-parameter "
        "model; nothing else about this machine is. The *time* comes from the step "
        "times a real cluster reports, the *volume* from experiment 3's byte "
        "counts, and a step is taken to be one third forward and two thirds "
        "backward, because a step is 6N FLOPs a token and the backward is 4 of "
        "them.",
        "",
        "## Without overlap",
        "",
        "Everything sent after the pass that produced it, at 30B and world size 32.",
        "",
        table([(b["arrangement"], b["card"], b["link"], f"{b['volume_in_P']:.2f}P",
                f"{b['wire_seconds']:.2f} s", f"{100 * b['ratio']:.0f}%",
                f"{b['step_serial']:.2f} s")
               for b in baseline.values()
               if b["arrangement"] in ("data parallel", "ZeRO-3")],
              ["arrangement", "card", "wire", "volume", "wire time",
               "÷ compute", "step, serial"],
              ["l", "l", "l", "r", "r", "r", "r"]),
        "",
        "The 2P-over-InfiniBand rows are the session's own figures — "
        f"{100 * baseline['data parallel | 64 x H100 | InfiniBand']['ratio']:.0f}% "
        "on H100 and "
        f"{100 * baseline['data parallel | 64 x B200 | InfiniBand']['ratio']:.0f}% "
        "on B200. The volume did not change between those two rows. The compute it "
        "has to hide behind got shorter, and that is the entire reason faster cards "
        "make this harder.",
        "",
        "## The bucket sweep",
        "",
        f"Data parallelism's 2P, sent during a backward pass that is "
        f"{BACKWARD_SHARE:.2f} of the step. 48 blocks, so a bucket of 1 is 50 "
        "transfers and a bucket of 48 is two.",
        "",
        table([(f"{r['bucket_layers']}", f"{r['n_transfers']}",
                f"{r['pass_seconds']:.3f} s", f"{r['tail_seconds']:.3f} s",
                f"{100 * r['hidden_fraction']:.0f}%")
               for r in sweeps[ib_h]["rows"]],
              ["bucket, layers", "transfers", "backward pass", "exposed tail",
               "hidden"], ["r", "r", "r", "r", "r"]),
        "",
        f"*(64 × H100 over InfiniBand: a "
        f"{sweeps[ib_h]['window_seconds']:.2f} s backward window carrying "
        f"{sweeps[ib_h]['rows'][0]['transfer_seconds']:.2f} s of traffic.)*",
        "",
        "The cost of a large bucket is not the transfer, it is *when* the transfer "
        "can start. A 48-layer bucket cannot leave until the backward pass has "
        "reached layer 0, so all of it is exposed; a 1-layer bucket leaves as each "
        "layer finishes and only the last one — the embedding, which is small — is "
        "left over at the end.",
        "",
        "## Where the smallest bucket stops being the best one",
        "",
        "At a 10 µs launch cost the smallest bucket always wins, which is not what "
        "production settings look like. The optimum is set by the ratio of the "
        "fixed cost of a transfer to the time the bytes themselves take, so the "
        "honest thing is to sweep the fixed cost and find the crossover rather than "
        "to assert a bucket size. B200, InfiniBand, 2P:",
        "",
        table([(f"{s['fixed_cost_ms']:g} ms", f"{s['best_bucket']}",
                f"{s['pass_seconds']:.3f} s",
                f"{s['smallest_bucket_seconds']:.3f} s")
               for s in sensitivity],
              ["fixed cost per transfer", "best bucket, layers",
               "backward pass", "at bucket = 1"], ["r", "r", "r", "r"]),
        "",
        f"The crossover is at about "
        f"{next((s['fixed_cost_ms'] for s in sensitivity if s['best_bucket'] > 1), None)} "
        "ms of fixed cost per transfer. Below it, start transfers as early as "
        "possible; above it, the launch cost of 50 transfers outweighs the earlier "
        "start and the bucket should grow. A production `bucket_size` of a few "
        "hundred megabytes is a bet about which side of that line the cluster is "
        "on — and the V4 configuration in the session notes bet 2e8 bytes.",
        "",
        "## The whole step, all four arrangements",
        "",
        "Both windows, each with the best bucket and prefetch depth found by "
        "search. `overhead` is how much longer the step is than its own compute.",
        "",
        table([(v["arrangement"], v["card"], v["link"],
                f"{v['transfer_seconds']:.2f} s", f"{v['exposed_seconds']:.2f} s",
                f"{100 * v['hidden_fraction']:.0f}%", f"{v['step_seconds']:.2f} s",
                f"{v['overhead_pct']:+.1f}%")
               for v in steps.values()],
              ["arrangement", "card", "wire", "on the wire", "exposed", "hidden",
               "step", "overhead"],
              ["l", "l", "l", "r", "r", "r", "r", "r"]),
        "",
        "Three readings.",
        "",
        "**Inside a node, everything hides.** Every NVLink row is within a per cent "
        "of its own compute time, stage 3 included. A run whose traffic stays "
        "inside one node barely has a communication cost at all, which is the "
        "answer to the session's second open question: put as much of the traffic "
        "inside a node as will fit.",
        "",
        "**Between nodes, the split matters more than the total.** Data parallelism "
        f"and ZeRO-1 move the same 2P on {ib_b.split(' | ')[0]}, and data "
        f"parallelism pays "
        f"{steps['data parallel | ' + ib_b]['overhead_pct']:.1f}% for it against "
        f"ZeRO-1's {steps['ZeRO-1 | ' + ib_b]['overhead_pct']:.1f}%. The reason is "
        "not volume, it is *windows*: data parallelism has to push all 2P through "
        "the backward pass, while ZeRO-1 and ZeRO-2 put one P in the backward and "
        "the other in the next forward, where there is compute going spare. "
        "Sharding the optimizer state bought a better communication schedule as "
        "well as the memory.",
        "",
        "**Stage 3 is the one that runs out of window.** It has 3P to move and, on "
        f"{ib_b.split(' | ')[0]} over InfiniBand, a step of "
        f"{steps['ZeRO-3 | ' + ib_b]['compute_seconds']:.2f} s to hide "
        f"{steps['ZeRO-3 | ' + ib_b]['transfer_seconds']:.2f} s of traffic in — and "
        f"{steps['ZeRO-3 | ' + ib_b]['backward_stalled']:.2f} s of its backward "
        "pass is spent standing still waiting for weights that have not arrived. "
        "That is the practical form of the 2P-versus-3P difference, and it is the "
        "argument for keeping a stage 3 group inside one node.",
        "",
        "![](e5_overlap.png)",
        "",
    ]
    save_text("e5_overlap.md", "\n".join(md))

    if verbose:
        print(json.dumps({"best_bucket": best, "sensitivity": sensitivity,
                          "steps": steps}, indent=2, default=str))
    return payload


def _figure(sweeps, steps, name="e5_overlap.png"):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3))

    styles = {"64 x H100 | InfiniBand": (plots.S1, "-", "H100, InfiniBand"),
              "64 x B200 | InfiniBand": (plots.S2, "-", "B200, InfiniBand"),
              "64 x H100 | NVLink": (plots.S1, (0, (3, 2)), "H100, NVLink"),
              "64 x B200 | NVLink": (plots.S2, (0, (3, 2)), "B200, NVLink")}
    for key, (colour, ls, label) in styles.items():
        rows = sweeps[key]["rows"]
        xs = [r["bucket_layers"] for r in rows]
        ys = [r["pass_seconds"] for r in rows]
        ax1.plot(xs, ys, color=colour, ls=ls, marker="o", ms=3.5)
        ax1.annotate(label, (xs[-1], ys[-1]), color=colour, fontsize=8,
                     va="center", ha="right", xytext=(-8, 9),
                     textcoords="offset points")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks([1, 2, 4, 8, 16, 48], ["1", "2", "4", "8", "16", "48"])
    ax1.set_xlabel("gradient bucket, layers")
    ax1.set_ylabel("backward pass, seconds")
    ax1.set_title("A big bucket cannot leave early (2P)", loc="left")

    order = ["data parallel", "ZeRO-1", "ZeRO-2", "ZeRO-3"]
    cards = [("64 x H100", "InfiniBand", plots.S1),
             ("64 x B200", "InfiniBand", plots.S2)]
    width = 0.36
    for j, (card, link, colour) in enumerate(cards):
        ys = [steps[f"{a} | {card} | {link}"]["overhead_pct"] for a in order]
        xs = [i + (j - 0.5) * width for i in range(len(order))]
        ax2.bar(xs, ys, width, color=colour, label=f"{card.split()[-1]}, {link}")
        for x, y in zip(xs, ys):
            ax2.annotate(f"{y:.1f}%", (x, y), ha="center", va="bottom",
                         fontsize=8, color=plots.INK_2)
    ax2.set_axisbelow(True)
    ax2.set_xticks(range(len(order)),
                   ["data\nparallel", "ZeRO-1", "ZeRO-2", "ZeRO-3"])
    ax2.set_ylabel("step time above its own compute, %")
    ax2.set_title("What is left exposed after overlapping", loc="left")
    ax2.legend(loc="upper left")

    return plots.finish(fig, name,
                        "Per-layer shape measured; volume counted in E3; "
                        "bandwidth stated. One link, served serially, best "
                        "bucket and prefetch depth by search.")


if __name__ == "__main__":
    main()
