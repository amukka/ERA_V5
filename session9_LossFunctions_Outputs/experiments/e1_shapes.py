"""Deliverable 1 -- every tensor shape, and what each dimension is.

The model and the loss narrate themselves: ``hidden(..., trace=[])`` and
``lm_loss(..., trace=[])`` append a row for every intermediate they build, each
with a sentence naming what one index along each axis *selects*.  The table is
therefore produced by the code that produced the tensors, and cannot drift away
from it the way a hand-written table does.

The part worth reading is the last section.  Nine tensors carry the batch from
token ids to a scalar, and one of them is 39 times larger than the hidden state
it came from.
"""

from __future__ import annotations

import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data import VOCAB_SIZE, load_documents, padded_batch
from src.losses import lm_loss
from src.model import Config, TinyGPT
from src.report import machine, save_json, save_text, table

B, T = 4, 128
MIB = 1024 ** 2


def main(verbose: bool = True) -> dict:
    torch.manual_seed(0)
    docs, _ = load_documents()
    cfg = Config()
    model = TinyGPT(cfg)

    # a padded batch, so the mask in the trace is not trivially all-True
    seqs = padded_batch(docs["train"], B, T, seed=3)

    trace: list = []
    hidden = model.hidden(seqs, trace)
    logits = model.logits(hidden, trace)
    rec = lm_loss(logits, seqs, trace=trace)

    # -- group the trace ----------------------------------------------------
    forward, loss_path = [], []
    seen_loss = False
    for row in trace:
        if row["name"].startswith(("logits", "targets", "flat_", "per_token",
                                   "loss")):
            seen_loss = True
        (loss_path if seen_loss else forward).append(row)

    params = [{"name": n, "shape": tuple(p.shape), "numel": p.numel(),
               "bytes": p.numel() * p.element_size()}
              for n, p in model.named_parameters()]

    def fmt(shape):
        return " x ".join(str(s) for s in shape)

    def rows(entries):
        return [(e["name"], fmt(e["shape"]), f"{e['numel']:,}",
                 f"{e['bytes']/MIB:.2f}", e["meaning"]) for e in entries]

    hdr = ("tensor", "shape", "numbers", "MiB", "what each dimension selects")
    aligns = ("l", "l", "r", "r", "l")

    # block 0 in full; the other blocks are the same shapes with a new prefix
    block0 = [e for e in forward if e["name"].startswith("block0.")]
    preamble = [e for e in forward if not e["name"].startswith("block")]
    later = [e for e in forward if e["name"].startswith(("block1.", "block2.",
                                                         "block3."))]

    by_name = {e["name"]: e for e in trace}
    hidden_bytes = hidden.numel() * hidden.element_size()
    logits_bytes = logits.numel() * logits.element_size()

    payload = {
        "machine": machine(),
        "config": {k: getattr(cfg, k) for k in
                   ("vocab_size", "n_layer", "n_head", "d_model", "d_ff",
                    "max_seq", "tie_embeddings")},
        "batch": {"B": B, "T": T, "valid_tokens": int(seqs.valid.sum()),
                  "padded_positions": int((~seqs.valid).sum())},
        "n_params": model.n_params(),
        "activations": forward + loss_path,
        "parameters": params,
        "n_activation_tensors": len(trace),
        "hidden_bytes": hidden_bytes,
        "logits_bytes": logits_bytes,
        "logits_over_hidden": logits_bytes / hidden_bytes,
        "vocab_over_d_model": VOCAB_SIZE / cfg.d_model,
        "loss": float(rec.loss),
        "n_contributing": rec.n_contributing,
        "n_pairs": rec.n_pairs,
    }

    md = f"""# 1 · Every tensor shape, and what each dimension is

`B={B}` rows, `T={T}` positions, `D={cfg.d_model}` d_model, `V={VOCAB_SIZE:,}` vocab,
`F={cfg.d_ff}` d_ff, `H={cfg.n_head}` heads, `Dh={cfg.head_dim}` per-head width.
The batch is padded on purpose ({payload['batch']['valid_tokens']} real tokens in
{B*T} slots), so the mask below is not trivially all-True.

Every row was emitted by the code that built the tensor, via the `trace`
argument threaded through `model.hidden()` and `lm_loss()`.

## Getting to a hidden state

{table(rows(preamble), hdr, aligns)}

## One block, in full

The other {cfg.n_layer - 1} blocks are these {len(block0)} tensors again with a
different prefix — {len(later)} more tensors, same shapes.

{table(rows(block0), hdr, aligns)}

## From hidden state to one scalar

This is the session. {len(loss_path)} tensors, and the widest thing in the whole
forward pass is in here rather than in attention.

{table(rows(loss_path), hdr, aligns)}

## The parameters

{table([(p["name"], fmt(p["shape"]), f"{p['numel']:,}",
         f"{p['bytes']/MIB:.2f}") for p in params],
       ("parameter", "shape", "numbers", "MiB"), ("l", "l", "r", "r"))}

## What the shapes say

**The logits are the largest tensor in the model, and it is not close.** The
hidden state is `{fmt(tuple(hidden.shape))}` = {hidden_bytes/MIB:.2f} MiB. The
logits are `{fmt(tuple(logits.shape))}` = {logits_bytes/MIB:.2f} MiB — a factor
of **{payload['logits_over_hidden']:.1f}×**, which is exactly `V/D` =
{VOCAB_SIZE:,}/{cfg.d_model} = {payload['vocab_over_d_model']:.1f}. Every axis of
the hidden state survives into the logits; the channel axis is simply replaced
by one {VOCAB_SIZE:,}-wide axis. Nothing about attention is involved, and the
tensor exists only to be collapsed into one number. That is deliverable 7.

**Two axes go into the loss and zero come out.** `logits[:, :-1]` is
`{fmt(by_name['logits[:, :-1]']['shape'])}` and `flat_logits` is
`{fmt(by_name['flat_logits']['shape'])}` — the batch and position axes folded
into one, because cross-entropy has no use for either. It scores rows, and
{B} × {T-1} = {B*(T-1)} of them arrive.

**The denominator is not `B × (T-1)`.** {payload['n_contributing']} of
{payload['n_pairs']} pairs contribute; the rest are padding. That is deliverable 3.
"""

    save_json("e1_shapes.json", payload)
    save_text("e1_shapes.md", md)
    if verbose:
        print(md)
    return payload


if __name__ == "__main__":
    main()
