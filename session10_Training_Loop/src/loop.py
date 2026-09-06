"""The five lines, written out honestly.

    logits = model(batch)
    loss   = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

with the two things the session insists on made explicit: gradients accumulate
across micro-batches, and the normalisation is a *choice*.  Both choices are
implemented here so the wrong one can be run on purpose.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .data import Batch


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import numpy as np
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def token_losses(model, batch: Batch, trace=None):
    """Per-token cross entropy and the count of tokens it was computed over.

    Returns ``(loss_sum, n_tokens)``.  ``loss_sum`` is the *sum* over valid
    positions, never a mean -- who divides, and by what, is the argument this
    session is about, so the function that computes the losses refuses to take
    a position on it.
    """
    logits = model(batch.inputs, batch.mask, trace)
    B, T, V = logits.shape
    flat_logits = logits.reshape(B * T, V)
    flat_targets = batch.targets.reshape(B * T)
    per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    per_token = per_token * batch.mask.reshape(B * T).to(per_token.dtype)
    n_tokens = int(batch.mask.sum())
    if trace is not None:
        trace.append({
            "name": "per_token_loss",
            "shape": (B * T,),
            "dtype": str(per_token.dtype).replace("torch.", ""),
            "numel": per_token.numel(),
            "meaning": "B*T=every position in the micro-batch flattened; the "
                       "cross entropy of the true next token at that position, "
                       "zeroed wherever the mask says there is no target",
        })
    return per_token.sum(), n_tokens


def grad_global_norm(model) -> float:
    """One L2 norm over every gradient in the model, concatenated.

    Accumulated on the device and read back exactly once. Calling ``float()``
    inside the loop instead costs one host synchronisation per parameter tensor
    -- 44 of them per step on this model -- and turned the cheapest trace on the
    dashboard into a measurable one for no reason. (``torch._foreach_norm`` is
    the idiomatic fused version and is what to use on CUDA; on the MPS backend
    it measured ten times *slower* than this loop, which is the kind of thing
    you only find out by timing it.)
    """
    total = None
    for p in model.parameters():
        if p.grad is None:
            continue
        sq = p.grad.detach().float().pow(2).sum()
        total = sq if total is None else total + sq
    if total is None:
        return 0.0
    return float(total.sqrt())            # exactly one host synchronisation


@dataclass
class StepRecord:
    step: int
    loss: float           # the number the run would print
    loss_token_wtd: float # the honest number, whatever the run printed
    grad_norm: float      # global L2 over all parameter gradients, pre-clip
    clip_scale: float
    n_tokens: int
    lr: float
    seconds: float


@dataclass
class RunLog:
    mode: str
    steps: list = field(default_factory=list)
    val_steps: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)

    def col(self, name):
        return [getattr(s, name) for s in self.steps]


def accumulate_step(model, micro_batches, mode: str, clip: float | None = 1.0):
    """One optimiser-visible step: several backwards, then one update.

    ``mode='token_weighted'``  every token gets one vote.  The denominator is
        the total number of valid tokens across *all* micro-batches, counted
        before the first backward and handed down -- which is exactly the shape
        the 2024 fix took in the frameworks that had this wrong.

    ``mode='mean_of_means'``  every micro-batch gets one vote regardless of how
        many tokens it holds.  This is the bug from section 8.

    Returns ``(reported_loss, honest_loss, grad_norm, clip_scale, n_tokens)``.
    Gradients are left in ``.grad``; the caller steps and wipes.
    """
    total_tokens = sum(mb.n_tokens for mb in micro_batches)
    n_micro = len(micro_batches)
    sum_loss = 0.0
    reported = 0.0

    for mb in micro_batches:
        loss_sum, n = token_losses(model, mb)
        if mode == "token_weighted":
            scaled = loss_sum / total_tokens
            reported += float(loss_sum.detach()) / total_tokens
        elif mode == "mean_of_means":
            scaled = (loss_sum / n) / n_micro
            reported += float(loss_sum.detach()) / n / n_micro
        else:
            raise ValueError(mode)
        scaled.backward()
        sum_loss += float(loss_sum.detach())

    honest = sum_loss / total_tokens
    norm = grad_global_norm(model)
    scale = 1.0
    if clip is not None and norm > clip:
        scale = clip / (norm + 1e-6)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)
    return reported, honest, norm, scale, total_tokens


@torch.no_grad()
def evaluate(model, batches) -> float:
    """Token-weighted validation loss.  The same batches for every run, and the
    same reduction regardless of what the run was trained with, because a metric
    that changes with the thing it is measuring measures nothing."""
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for mb in batches:
        loss_sum, k = token_losses(model, mb)
        total += float(loss_sum)
        n += k
    if was_training:
        model.train()
    return total / n


def cosine_lr(step: int, total: int, peak: float, warmup: int, floor_frac=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def train(
    model,
    sampler,
    *,
    mode: str = "token_weighted",
    steps: int = 300,
    accum: int = 4,
    lr: float = 3e-4,
    warmup: int = 20,
    clip: float | None = 1.0,
    device=None,
    eval_batches=None,
    eval_every: int = 0,
    batch_hook=None,
    log_every: int = 0,
) -> RunLog:
    device = device or pick_device()
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    log = RunLog(mode=mode)

    for step in range(steps):
        micro = [mb.to(device) for mb in sampler.micro_batches(accum)]
        if batch_hook is not None:
            micro = batch_hook(step, micro)

        cur_lr = cosine_lr(step, steps, lr, warmup)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        reported, honest, norm, scale, ntok = accumulate_step(
            model, micro, mode, clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if device.type == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0

        log.steps.append(StepRecord(step, reported, honest, norm, scale,
                                    ntok, cur_lr, dt))

        if eval_batches and eval_every and (
            step % eval_every == 0 or step == steps - 1
        ):
            log.val_steps.append(step)
            log.val_loss.append(evaluate(model, eval_batches))

        if log_every and step % log_every == 0:
            print(f"  step {step:>4}  loss {reported:.4f}  "
                  f"|g| {norm:7.3f}  tok {ntok:>5}  {dt*1e3:6.1f} ms")

    return log
