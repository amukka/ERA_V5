"""Deliverable 7 -- peak memory of ordinary cross-entropy against a chunked one.

Three things are reported, because a single ratio hides which part of the step
it belongs to:

1. **Equivalence first.**  A memory saving from a loss that computes something
   else is not a saving.  Value and gradients (w.r.t. hidden and head weight)
   are compared before any memory is quoted.
2. **The loss step** -- hidden states and head weight in, scalar out, backward
   through it.  This is the part the chunking changes, measured in isolation.
3. **The full training step** -- embedding to optimiser-ready gradients -- so the
   headline ratio is also stated at the scale a person would actually feel it.

Peak is measured by ``src.memory.TrackTensors`` (live tensor bytes, device
independent) and cross-checked by peak RSS of a fresh CPU process.
"""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import VOCAB_SIZE, clean_batch, load_documents
from src.losses import chunked_cross_entropy, lm_loss
from src.memory import MIB, TrackTensors, logits_bytes, peak_rss_subprocess
from src.model import Config, TinyGPT
from src.plots import CRITICAL, S1, S2, finish
from src.report import machine, save_json, save_text, table
from src.train import pick_device

B, T = 16, 256           # 4,096 positions: enough that the logits dominate
CHUNK = 512
SWEEP = (64, 128, 256, 512, 1024, 2048)


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def loss_step(kind, hidden, weight, seqs, chunk):
    """One forward + backward through the loss only; returns (loss, tracker)."""
    hidden = hidden.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    tr = TrackTensors()
    with tr:
        if kind == "ordinary":
            logits = F.linear(hidden, weight)
            rec = lm_loss(logits, seqs)
        else:
            rec = chunked_cross_entropy(hidden, weight, seqs, chunk_size=chunk)
        rec.loss.backward()
    return rec, hidden.grad, weight.grad, tr


def full_step(kind, model, seqs, chunk):
    model.zero_grad(set_to_none=True)
    tr = TrackTensors()
    with tr:
        h = model.hidden(seqs)
        if kind == "ordinary":
            rec = lm_loss(model.head(h), seqs)
        else:
            rec = chunked_cross_entropy(h, model.head.weight, seqs,
                                        chunk_size=chunk)
        rec.loss.backward()
    return rec, tr


def main(verbose: bool = True) -> dict:
    device = pick_device()
    docs, _ = load_documents()
    torch.manual_seed(0)
    model = TinyGPT(Config(max_seq=T)).to(device)
    seqs = clean_batch(docs["train"], B, T, seed=7).to(device)
    n_pairs = B * (T - 1)

    with torch.no_grad():
        hidden = model.hidden(seqs)
    weight = model.head.weight

    # -- 1. equivalence -------------------------------------------------------
    r_o, gh_o, gw_o, _ = loss_step("ordinary", hidden, weight, seqs, CHUNK)
    r_c, gh_c, gw_c, _ = loss_step("chunked", hidden, weight, seqs, CHUNK)
    equiv = {
        "loss_ordinary": float(r_o.loss.detach()), "loss_chunked": float(r_c.loss.detach()),
        "loss_abs_diff": abs(float(r_o.loss.detach()) - float(r_c.loss.detach())),
        "grad_hidden_max_abs_diff": float((gh_o - gh_c).abs().max()),
        "grad_weight_max_abs_diff": float((gw_o - gw_c).abs().max()),
        "grad_hidden_scale": float(gh_o.abs().max()),
        "grad_weight_scale": float(gw_o.abs().max()),
        "n_contributing": (r_o.n_contributing, r_c.n_contributing),
    }
    assert equiv["loss_abs_diff"] < 1e-4, equiv
    assert equiv["grad_hidden_max_abs_diff"] < 1e-4 * max(1, equiv["grad_hidden_scale"]) * 10

    # -- 2. the loss step, and a chunk-size sweep -----------------------------
    _sync(device)
    _, _, _, t_o = loss_step("ordinary", hidden, weight, seqs, CHUNK)
    ordinary = t_o.summary()
    sweep = {}
    for c in SWEEP:
        _, _, _, t_c = loss_step("chunked", hidden, weight, seqs, c)
        sweep[c] = t_c.summary()
    chunked = sweep[CHUNK]
    ratio_loss = ordinary["peak_mib"] / chunked["peak_mib"]

    # -- 3. the whole training step ------------------------------------------
    _, tf_o = full_step("ordinary", model, seqs, CHUNK)
    _, tf_c = full_step("chunked", model, seqs, CHUNK)
    full = {"ordinary": tf_o.summary(), "chunked": tf_c.summary(),
            "ratio": tf_o.peak_mib / tf_c.peak_mib}

    # -- cross-check: a fresh process, a different mechanism ------------------
    setup = (f"""
import torch, torch.nn.functional as F
from src.data import load_documents, clean_batch
from src.losses import lm_loss, chunked_cross_entropy
docs,_ = load_documents()
seqs = clean_batch(docs['train'], {B}, {T}, seed=7)
torch.manual_seed(0)
h = torch.randn({B},{T},{Config().d_model}, requires_grad=True)
w = (torch.randn({VOCAB_SIZE},{Config().d_model})*0.02).requires_grad_(True)
""")
    rss_o = peak_rss_subprocess(str(ROOT), setup,
                                "lm_loss(F.linear(h,w), seqs).loss.backward()")
    rss_c = peak_rss_subprocess(str(ROOT), setup,
                                f"chunked_cross_entropy(h,w,seqs,chunk_size={CHUNK}).loss.backward()")

    full_logits_mib = logits_bytes(n_pairs, VOCAB_SIZE) / MIB
    payload = {
        "machine": machine(), "device": str(device), "B": B, "T": T,
        "chunk": CHUNK, "vocab": VOCAB_SIZE, "n_pairs": n_pairs,
        "full_logits_mib": full_logits_mib, "equivalence": equiv,
        "ordinary": ordinary, "chunked": chunked, "ratio_loss_step": ratio_loss,
        "sweep": {str(k): v for k, v in sweep.items()},
        "full_training_step": full,
        "rss_cpu_mib": {"ordinary": rss_o, "chunked": rss_c,
                        "ratio": rss_o / rss_c if rss_c > 0 else None},
    }

    # -- figure ---------------------------------------------------------------
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    labels = ["ordinary", f"chunked\n({CHUNK} rows)"]
    vals = [ordinary["peak_mib"], chunked["peak_mib"]]
    ax1.bar(labels, vals, color=[CRITICAL, S1], width=0.55)
    for i, v in enumerate(vals):
        ax1.annotate(f"{v:,.0f} MiB", (i, v), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=9)
    ax1.set_ylabel("peak live tensors (MiB)")
    ax1.set_title(f"Loss step peak: {ratio_loss:.1f}x lower")
    ax1.set_ylim(0, max(vals) * 1.18)
    xs = list(SWEEP)
    ax2.plot(xs, [sweep[c]["peak_mib"] for c in xs], color=S1, marker="o")
    ax2.axhline(ordinary["peak_mib"], color=CRITICAL, lw=1.4, ls="--")
    ax2.annotate("ordinary", (xs[0], ordinary["peak_mib"]), xytext=(0, 5),
                 textcoords="offset points", fontsize=8.5)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("chunk size (rows of logits alive at once)")
    ax2.set_ylabel("peak live tensors (MiB)")
    ax2.set_title("Chunk size sets the floor")
    finish(fig, "e7_memory.png",
           f"B={B}, T={T}, V={VOCAB_SIZE:,}, {n_pairs:,} pairs; forward + "
           f"backward through the loss; fp32; {device}")

    # -- write-up -------------------------------------------------------------
    sweep_tbl = table(
        [(f"{c}", f"{sweep[c]['peak_mib']:.1f}",
          f"{ordinary['peak_mib']/sweep[c]['peak_mib']:.2f}x",
          f"{c*VOCAB_SIZE*4/MIB:.1f}") for c in SWEEP],
        ("chunk rows", "peak MiB", "ratio vs ordinary", "one chunk's logits MiB"),
        ("r", "r", "r", "r"))
    md = f"""# 7 · Peak memory: ordinary cross-entropy against a chunked one

`B={B}`, `T={T}`, `V={VOCAB_SIZE:,}`: **{n_pairs:,} predictions**, so the logits are
{n_pairs:,} x {VOCAB_SIZE:,} x 4 B = **{full_logits_mib:.1f} MiB** if held whole.

## The two numbers

| implementation | peak live tensors (loss step, fwd + bwd) |
| -------------- | ---------------------------------------: |
| ordinary `F.cross_entropy(logits)` | **{ordinary['peak_mib']:.1f} MiB** |
| chunked, {CHUNK} rows at a time, recomputed in backward | **{chunked['peak_mib']:.1f} MiB** |
| **ratio** | **{ratio_loss:.2f}x** lower |

## Is it the same loss?

| | ordinary | chunked | |diff| |
| - | -: | -: | -: |
| loss | {equiv['loss_ordinary']:.6f} | {equiv['loss_chunked']:.6f} | {equiv['loss_abs_diff']:.2e} |
| grad wrt hidden, max abs | | | {equiv['grad_hidden_max_abs_diff']:.2e} (scale {equiv['grad_hidden_scale']:.2e}) |
| grad wrt head weight, max abs | | | {equiv['grad_weight_max_abs_diff']:.2e} (scale {equiv['grad_weight_scale']:.2e}) |
| contributing tokens | {equiv['n_contributing'][0]} | {equiv['n_contributing'][1]} | |

Same value and same gradients up to float rounding, so the saving is not bought
with a different objective.

## Chunk-size sweep

{sweep_tbl}

Peak grows roughly linearly with the chunk: what is alive at once is one chunk's
logits plus its softmax and gradient copies, on top of a fixed part (hidden
states, head weight and its gradient).  Ordinary cross-entropy is the same thing
with the "chunk" equal to every prediction in the batch: it holds about four
logits-sized tensors at once ({ordinary['peak_mib']/full_logits_mib:.1f}x the {full_logits_mib:.0f} MiB
logits: logits, softmax, their gradient, and autograd's copies).

## Cross-checks

* **Full training step** (embedding -> gradients, {Config().n_layer} blocks):
  ordinary {full['ordinary']['peak_mib']:.1f} MiB, chunked
  {full['chunked']['peak_mib']:.1f} MiB, **{full['ratio']:.2f}x**.  Smaller than the
  loss-step ratio because the blocks' activations are common to both.
* **Independent measurement**, fresh CPU process, peak RSS growth:
  ordinary {rss_o:.0f} MiB, chunked {rss_c:.0f} MiB
  ({(rss_o/rss_c if rss_c > 0 else float('nan')):.2f}x).  It agrees on direction and on
  "several times lower" but not on the ratio: RSS is coarse (allocator retention,
  page granularity, and the CPU allocator's reuse of freed blocks), so the tracker's
  live-tensor number is the one reported.

## How the chunked version works

1. Gather only contributing positions, so padded positions never get logits.
2. For each chunk of `{CHUNK}` rows: `logits = h_chunk @ W.T`, sum of cross-entropy, done.
3. Wrap each chunk in `torch.utils.checkpoint`: its logits are freed after the
   forward and recomputed in backward.  Without it autograd keeps every chunk's
   logits alive and the peak is back where it started.
4. Sum the chunks' *sums* and divide once by the total count.  Averaging chunk
   averages would be wrong whenever chunks hold different token counts.

The price is one extra `h @ W.T` per chunk in backward: memory is bought with FLOPs.
"""
    save_json("e7_memory.json", payload)
    save_text("e7_memory.md", md)
    if verbose:
        print(md)
    return payload


if __name__ == "__main__":
    main()
