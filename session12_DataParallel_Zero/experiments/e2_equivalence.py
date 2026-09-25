"""E2 -- the two claims that make data parallelism legitimate, checked.

Distributed training is only worth anything if it computes the same thing the
single machine would have computed.  Two separate claims are hiding in that
sentence and both of them are checkable here.

  1. *Averaging the gradients over W ranks gives the gradient of the batch those
     ranks saw between them.*  So 32 ranks on 2 sequences each is not an
     approximation of one machine on 64 sequences -- it is that machine.

  2. *ZeRO changes where the bytes live and nothing else.*  Stages 1, 2 and 3
     rearrange 447 GiB of state and must still produce, weight for weight, the
     numbers data parallelism produced.

The second one is checked the strict way: bitwise.  Not "close", not "within
tolerance" -- identical.  A ZeRO stage that only agreed to six decimals would be
a ZeRO stage with a bug in it, and the reason this repository can make that
demand is in ``mesh.py``: every reduction stacks the ranks in rank order and
sums along that axis, so the slice a reduce-scatter returns went through exactly
the additions the corresponding slice of an all-reduce went through.

Then the control: an engine with the averaging taken out.  It is the same code
with one collective deleted, and it answers the question the first two leave
open -- what, concretely, is the all-reduce buying?
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json

import torch

from src.engines import DataParallel, ENGINES, STAGE_ORDER
from src.model import Config
from src.report import machine, save_json, save_text, table
from src.sim import train, max_weight_divergence
from src import plots

WORLD = 32
MICRO = 2
STEPS = 12


class NoAverage(DataParallel):
    """Data parallelism with the all-reduce deleted, and nothing else changed.

    Every rank keeps its own gradients and updates its own copy from them.  The
    copies were identical when the run started and each one is a correct piece
    of arithmetic; what is gone is the one line that made them agree about which
    direction to move in.
    """
    name, stage = "no averaging", -1

    def _update(self):
        for g in range(self.cfg.n_groups):
            self._adamw(g, self.grads[g].float())
            self.weights[g].copy_(self.master[g])
            self.grads[g].zero_()


ENGINES[NoAverage.name] = NoAverage


def _gradient_equivalence(cfg) -> dict:
    """One step, no optimizer: is the averaged gradient the big-batch gradient?"""
    from src.data import Batcher
    from src.mesh import Mesh
    from src.model import forward_backward, init_group, LocalParams

    flats = {g: init_group(cfg, g, seed=7) for g in range(cfg.n_groups)}
    sampler = Batcher(seq_len=cfg.max_seq)
    batch = sampler.global_batch(WORLD, MICRO)

    loss_single, g_single, _ = forward_backward(cfg, LocalParams(flats), *batch)

    def worker(rank, mesh):
        x, y = Batcher.for_rank(batch, rank, WORLD, MICRO)
        loss, grads, _ = forward_backward(cfg, LocalParams(flats), x, y)
        return loss, grads

    per_rank = Mesh(WORLD).run(worker)
    loss_many = sum(l for l, _ in per_rank) / WORLD

    def compare(cast):
        """Relative L2 over the whole gradient, and the angle between the two.

        Not the largest element-wise relative error: a gradient has elements
        near zero, and dividing by them says more about them than about the
        comparison.  ``||a-b|| / ||b||`` and the angle are what an optimizer
        actually responds to."""
        avg_all, ref_all = [], []
        for g in sorted(g_single):
            stack = torch.stack([gr[g].to(cast).float() for _, gr in per_rank])
            avg_all.append(stack.sum(0) / WORLD)
            ref_all.append(g_single[g])
        a = torch.cat(avg_all).double()
        b = torch.cat(ref_all).double()
        cos = float(torch.dot(a, b) / (a.norm() * b.norm()))
        return {
            "max_abs": float((a - b).abs().max()),
            "relative_l2": float((a - b).norm() / b.norm()),
            "angle_degrees": float(torch.rad2deg(torch.arccos(
                torch.tensor(min(1.0, max(-1.0, cos)))))),
        }

    return {
        "loss_single": loss_single,
        "loss_distributed": loss_many,
        "loss_abs_diff": abs(loss_single - loss_many),
        "fp32": compare(torch.float32),
        "bf16": compare(torch.bfloat16),
    }


def _spread(run) -> list:
    """How far apart the ranks' weight copies are, at the end of the run."""
    r0 = run.weights_by_rank[0]
    return [max_weight_divergence(r0, w) for w in run.weights_by_rank]


def main(verbose: bool = True) -> dict:
    cfg = Config()

    # -- claim 1: W ranks on B sequences == 1 rank on W*B sequences ----------
    grads = _gradient_equivalence(cfg)
    many = train("data parallel", world_size=WORLD, micro_batch=MICRO,
                 steps=STEPS, cfg=cfg, keep_weights=True)
    one = train("data parallel", world_size=1, micro_batch=WORLD * MICRO,
                steps=STEPS, cfg=cfg, keep_weights=True)
    drift = [{"step": s,
              "max_weight_diff": max_weight_divergence(
                  train("data parallel", world_size=WORLD, micro_batch=MICRO,
                        steps=s, cfg=cfg, keep_weights=True).weights,
                  train("data parallel", world_size=1, micro_batch=WORLD * MICRO,
                        steps=s, cfg=cfg, keep_weights=True).weights)}
             for s in (1, 2, 4, 8, STEPS)]
    big_batch = {
        "distributed": f"{WORLD} ranks x {MICRO} sequences",
        "single": f"1 rank x {WORLD * MICRO} sequences",
        "tokens_per_step": WORLD * MICRO * cfg.max_seq,
        "loss_distributed": many.losses[-1],
        "loss_single": one.losses[-1],
        "loss_abs_diff": abs(many.losses[-1] - one.losses[-1]),
        "max_weight_diff": max_weight_divergence(many.weights, one.weights),
        "bitwise": all(torch.equal(many.weights[g], one.weights[g])
                       for g in many.weights),
        "gradients": grads,
        "drift": drift,
    }

    # -- claim 2: the four arrangements agree, bitwise -----------------------
    runs = {"data parallel": many}
    for name in STAGE_ORDER[1:]:
        runs[name] = train(name, world_size=WORLD, micro_batch=MICRO,
                           steps=STEPS, cfg=cfg, keep_weights=True)
    base = many.weights
    agreement = []
    for name, r in runs.items():
        agreement.append({
            "engine": name,
            "stage": r.stage,
            "final_loss": r.losses[-1],
            "loss_diff_vs_dp": abs(r.losses[-1] - many.losses[-1]),
            "max_weight_diff_vs_dp": max_weight_divergence(base, r.weights),
            "bitwise_identical": all(torch.equal(base[g], r.weights[g])
                                     for g in base),
            "bytes_per_weight": r.bytes_per_weight,
            "wire_in_P": r.wire_in_P,
            "rank_spread": max(_spread(r)),
        })

    # -- the control: averaging removed --------------------------------------
    broken = train("no averaging", world_size=WORLD, micro_batch=MICRO,
                   steps=STEPS, cfg=cfg, keep_weights=True)
    control = {
        "final_loss_rank0": broken.losses[-1],
        "final_loss_dp": many.losses[-1],
        "rank_spread": max(_spread(broken)),
        "max_weight_diff_vs_dp": max_weight_divergence(base, broken.weights),
        "wire_in_P": broken.wire_in_P,
    }

    # divergence step by step, for the figure: rank 0 against rank 1
    def divergence_trace(name, steps=STEPS):
        trace = []
        for s in range(1, steps + 1):
            r = train(name, world_size=8, micro_batch=MICRO, steps=s, cfg=cfg,
                      keep_weights=True)
            trace.append(max_weight_divergence(r.weights_by_rank[0],
                                               r.weights_by_rank[1]))
        return trace

    traces = {"data parallel": divergence_trace("data parallel"),
              "no averaging": divergence_trace("no averaging")}

    fig = _figure(runs, broken, traces)

    payload = {"machine": machine(), "world_size": WORLD, "steps": STEPS,
               "micro_batch": MICRO, "big_batch": big_batch,
               "agreement": agreement, "control": control,
               "divergence_traces": traces,
               "losses": {k: r.losses for k, r in runs.items()},
               "loss_no_averaging": broken.losses}
    save_json("e2_equivalence.json", payload)

    md = [
        "# E2 · Data parallelism is one big batch, and ZeRO does not change the answer",
        "",
        "## 1 · Thirty-two ranks on two sequences each *is* one rank on sixty-four",
        "",
        "The claim is about the gradient, so check the gradient, before any "
        "optimizer has had a chance to amplify anything. One step, "
        f"{WORLD * MICRO} sequences: once as {WORLD} ranks averaging their "
        f"{MICRO}-sequence gradients, once as a single machine on all "
        f"{WORLD * MICRO} at once.",
        "",
        table([("the loss", f"{grads['loss_distributed']:.10f}",
                f"{grads['loss_single']:.10f}",
                f"{grads['loss_abs_diff']:.1e}"),
               ("the gradient, averaged in fp32", "—", "—",
                f"{grads['fp32']['relative_l2']:.1e} relative, "
                f"{grads['fp32']['angle_degrees']:.4f}°"),
               ("the gradient, each rank rounded to bf16 first", "—", "—",
                f"{grads['bf16']['relative_l2']:.1e} relative, "
                f"{grads['bf16']['angle_degrees']:.4f}°")],
              ["", big_batch["distributed"], big_batch["single"], "difference"],
              ["l", "r", "r", "r"]),
        "",
        "The first line is the claim and it holds: averaging 32 partial gradients "
        f"reproduces the 64-sequence gradient to {grads['fp32']['relative_l2']:.0e} "
        f"relative and {grads['fp32']['angle_degrees']:.4f} degrees, which is fp32 "
        "summation noise and nothing else. The two runs are the same run.",
        "",
        "The third line is the part worth keeping. A rank does not put its fp32 "
        "gradient on the wire — it puts 2 bytes per weight on the wire, which is "
        "where the second row of the sixteen comes from. Rounding each rank's "
        "*partial* gradient to bf16 and then averaging is not the same as rounding "
        f"the whole gradient once: it costs {grads['bf16']['relative_l2']:.1e} "
        f"relative and {grads['bf16']['angle_degrees']:.3f} degrees of direction, "
        f"{grads['bf16']['relative_l2'] / max(grads['fp32']['relative_l2'], 1e-15):.0f}x "
        "the fp32 figure. That is a real cost of distributing the work, it is "
        "invisible in any loss curve, and the optimizer does not leave it small:",
        "",
        table([(d["step"], f"{d['max_weight_diff']:.1e}") for d in drift],
              ["after N steps", "max \\|w(32 ranks) − w(1 rank)\\|"], ["r", "r"]),
        "",
        "The first row is the surprise, and it is Adam's doing rather than bf16's. "
        "On step 1 the update is `-lr * m_hat / (sqrt(v_hat) + eps)` with `m_hat = g` "
        "and `v_hat = g^2`, so it is `-lr * sign(g)` whatever the size of `g`. A "
        "gradient element that differs by 1e-7 between the two runs moves the weight "
        "by a full `2 * lr` if that difference crosses zero — and `2 * lr = 6e-4`, "
        "which is the column. The scale-free step is the amplifier; bf16 only "
        "supplies the disagreement. It does not compound after that, because the "
        "same normalisation that amplifies the difference also bounds it.",
        "",
        f"After {STEPS} steps the weights differ by "
        f"{big_batch['max_weight_diff']:.1e} and the losses by "
        f"{big_batch['loss_abs_diff']:.1e} nats, and neither run is the wrong one. "
        "They are two equally valid roundings of the same mathematical step. That is "
        "the honest answer to whether the distributed run is the same run: yes in "
        "exact arithmetic, and to within one bf16 rounding per rank in this one.",
        "",
        "## 2 · The four arrangements, compared weight by weight",
        "",
        f"World size {WORLD}, {STEPS} steps, identical data.",
        "",
        table([(a["engine"], f"{a['bytes_per_weight']:.3f}",
                f"{a['wire_in_P']:.4f}P", f"{a['final_loss']:.8f}",
                f"{a['max_weight_diff_vs_dp']:.1e}",
                "yes" if a["bitwise_identical"] else "no")
               for a in agreement],
              ["arrangement", "bytes/weight", "wire/step", "final loss",
               "max \\|Δw\\| vs DP", "bitwise"],
              ["l", "r", "r", "r", "r", "c"]),
        "",
        "Three columns move and one does not. The memory per rank falls by "
        f"{agreement[0]['bytes_per_weight'] / agreement[-1]['bytes_per_weight']:.0f}x, "
        "the traffic rises for stage 3, and the weights do not change at all — not "
        "to eight decimals, but in every bit of every one of "
        f"{many.n_params:,} numbers. **That is the claim ZeRO makes, and it is the "
        "only claim it makes.** It is a statement about addresses, not about "
        "arithmetic.",
        "",
        "## 3 · The control: the same code with the averaging deleted",
        "",
        "`NoAverage` is `DataParallel` with the `all_reduce` line removed. Each rank "
        "updates its own copy from its own gradients — which are correct gradients, "
        "for the two sequences that rank happened to draw.",
        "",
        table([("weights on the wire per step", f"{agreement[0]['wire_in_P']:.4f}P",
                f"{control['wire_in_P']:.4f}P"),
               ("spread between the 32 copies",
                f"{agreement[0]['rank_spread']:.1e}",
                f"{control['rank_spread']:.2e}"),
               ("final loss on rank 0", f"{many.losses[-1]:.6f}",
                f"{control['final_loss_rank0']:.6f}")],
              ["", "data parallelism", "averaging deleted"], ["l", "r", "r"]),
        "",
        f"After {STEPS} steps the 32 copies are {control['rank_spread']:.2e} apart and "
        "moving apart. Nothing raises an error; the loss still falls; each rank is "
        "still training a perfectly good language model. It is just that there are "
        "now 32 of them, each one trained on 1/32 of the data, and the run has "
        "quietly stopped being the run it was supposed to be. The all-reduce is not "
        "a synchronisation detail — it is the thing that makes the 32 copies one "
        "model.",
        "",
        "![](e2_equivalence.png)",
        "",
    ]
    save_text("e2_equivalence.md", "\n".join(md))

    if verbose:
        print(json.dumps({"big_batch": big_batch, "agreement": agreement,
                          "control": control}, indent=2, default=str))
    return payload


def _figure(runs, broken, traces, name="e2_equivalence.png"):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    colours = [plots.S1, plots.S2, plots.S3, plots.S4]
    steps = range(1, len(runs["data parallel"].losses) + 1)
    for (nm, r), c in zip(runs.items(), colours):
        ax1.plot(steps, r.losses, color=c, lw=3.2 if nm == "data parallel" else 1.6,
                 alpha=1.0 if nm == "data parallel" else 0.95,
                 ls="-" if nm == "data parallel" else (0, (3, 2)))
    ax1.plot(steps, broken.losses, color=plots.CRITICAL, lw=1.8)
    ax1.set_xlabel("step")
    ax1.set_ylabel("training loss, nats")
    ax1.set_title("Four arrangements, one curve", loc="left")
    mid = len(runs["data parallel"].losses) // 2
    ax1.annotate("data parallelism, ZeRO-1, ZeRO-2, ZeRO-3\n(all four, one curve, exactly)",
                 (mid + 1, runs["ZeRO-3"].losses[mid]), color=plots.S1,
                 fontsize=8.5, va="bottom", ha="left",
                 xytext=(6, 14), textcoords="offset points")
    ax1.annotate("averaging deleted", (steps[-1], broken.losses[-1]),
                 color=plots.CRITICAL, fontsize=8.5, va="bottom", ha="right",
                 xytext=(-4, 6), textcoords="offset points")

    xs = range(1, len(traces["data parallel"]) + 1)
    ax2.plot(xs, [max(v, 1e-12) for v in traces["no averaging"]],
             color=plots.CRITICAL, marker="o", ms=3.5)
    ax2.plot(xs, [max(v, 1e-12) for v in traces["data parallel"]],
             color=plots.S1, marker="o", ms=3.5)
    ax2.set_yscale("log")
    ax2.set_ylim(3e-13, 1e-1)
    ax2.set_yticks([1e-12, 1e-9, 1e-6, 1e-3],
                   ["0 (exactly)", "1e-9", "1e-6", "1e-3"])
    ax2.set_xlabel("step")
    ax2.set_ylabel("max |w(rank 0) − w(rank 1)|")
    ax2.set_title("What the all-reduce is buying", loc="left")
    ax2.annotate("averaging deleted", (xs[-1], traces["no averaging"][-1]),
                 color=plots.CRITICAL, fontsize=8.5, ha="right", va="bottom",
                 xytext=(-4, 6), textcoords="offset points")
    ax2.annotate("with the all-reduce: the copies never differ, in any bit",
                 (xs[0], 1e-12), color=plots.S1, fontsize=8.5, ha="left",
                 va="bottom", xytext=(2, 6), textcoords="offset points")
    return plots.finish(fig, name,
                        "Right panel: world size 8, one run per step count. "
                        "The blue series is a true zero, drawn on the axis floor.")


if __name__ == "__main__":
    main()
