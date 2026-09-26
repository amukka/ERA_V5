"""A small trainer, because several deliverables only have teeth once trained.

At initialisation almost every bug in this session is invisible.  An untrained
model assigns roughly uniform probability to everything, so padding is not yet
easy, a document boundary is not yet surprising, and a target shifted the wrong
way is not yet trivially copyable.  All three become visible within a few
hundred steps, which is why this file exists.

The loop is deliberately plain -- AdamW, a fixed learning rate, no clipping, no
schedule.  Session 10 is where the loop is the subject.  Here it is a tool, and
the only thing it must do is be identical across the arms of a comparison so
that a difference between them is caused by the thing under test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .data import Seqs, clean_batch, packed_batch, padded_batch
from .losses import lm_loss, loss_mask, align


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


BATCH_MAKERS = {
    "clean": clean_batch,
    "padded": padded_batch,
    "packed": packed_batch,
}


def batch_stream(docs, kind="clean", batch_size=8, width=128, seed=0):
    """An endless, reproducible stream of batches of one kind."""
    make = BATCH_MAKERS[kind]
    i = 0
    while True:
        yield make(docs, batch_size, width, seed=seed * 100_000 + i)
        i += 1


@dataclass
class History:
    steps: list = field(default_factory=list)
    train_loss: list = field(default_factory=list)
    probe_loss: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def train(model, docs, *, steps=300, kind="clean", batch_size=8, width=128,
          lr=3e-4, seed=0, offset=1, mask_padding=True, mask_boundaries=True,
          device=None, probe=None, probe_every=25, loss_hook=None,
          verbose=False):
    """Train ``model`` in place; return a ``History``.

    ``loss_hook(model, seqs) -> (loss, record_dict)`` replaces the standard
    objective when given, which is how the multi-head deliverable trains two
    heads without a second copy of this loop.

    ``probe`` is a fixed batch, scored with the *correct* mask after every
    ``probe_every`` steps.  Where an arm of an experiment trains on a wrong
    objective, its own training loss is not comparable to anything; the probe
    is, because every arm is measured the same way.
    """
    device = device or pick_device()
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    torch.manual_seed(seed)
    stream = batch_stream(docs, kind, batch_size, width, seed)
    probe = probe.to(device) if probe is not None else None
    hist = History()

    for step in range(1, steps + 1):
        seqs = next(stream).to(device)
        if loss_hook is not None:
            loss, record = loss_hook(model, seqs)
        else:
            logits = model(seqs)
            rec = lm_loss(logits, seqs, offset=offset,
                          mask_padding=mask_padding,
                          mask_boundaries=mask_boundaries)
            loss, record = rec.loss, {"n": rec.n_contributing}

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        hist.steps.append(step)
        hist.train_loss.append(float(loss.detach()))
        for k, v in record.items():
            hist.extra.setdefault(k, []).append(v)

        if probe is not None and (step % probe_every == 0 or step == 1):
            hist.probe_loss.append((step, evaluate(model, probe, offset)))
        if verbose and (step % 50 == 0 or step == 1):
            print(f"  step {step:>4}  loss {float(loss.detach()):.4f}")
    return hist


@torch.no_grad()
def evaluate(model, seqs: Seqs, offset: int = 1) -> float:
    """Honest loss on one batch: padding masked, boundaries masked, always."""
    was_training = model.training
    model.eval()
    logits = model(seqs)
    rec = lm_loss(logits, seqs, offset=offset,
                  mask_padding=True, mask_boundaries=True)
    model.train(was_training)
    return float(rec.loss)
