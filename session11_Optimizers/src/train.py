"""One training loop, shared by E2 to E5, with the schedule passed in.

Session 14's V5 decisions are applied here and nowhere else:

* AdamW, betas (0.9, 0.95), decoupled weight decay 0.1;
* LayerNorm scales and biases are excluded from decay (Section 7);
* the global grad norm is clipped at 1.0 (Session 10);
* the update-to-weight ratio ``||W_after - W_before|| / ||W_before||`` can be
  logged for every parameter tensor at every step (Section 9).

Data is the Session 10 corpus slice, cut into fixed 128-token windows so every
batch holds the same number of tokens and no masking is involved. Every run with
the same seed sees the same batches in the same order, and every run is scored
on the same held-out validation windows, which is what makes two runs
comparable at all.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .data import Sampler, load_documents
from .model import GPT, Config

SEQ = 128
BATCH = 32            # 4,096 tokens per optimiser step
N_VAL_BATCHES = 16    # 65,536 held-out tokens


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------- schedules
def constant(peak, warmup=0):
    def f(step, total):
        return peak * min(1.0, (step + 1) / warmup) if warmup else peak
    return f


def cosine(peak, warmup, floor=0.0):
    """Warmup, then a cosine from ``peak`` to ``floor*peak`` at step ``total``.
    The run length is baked into every value it returns."""
    def f(step, total):
        if step < warmup:
            return peak * (step + 1) / warmup
        t = (step - warmup) / max(1, total - warmup)
        return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))
    return f


def wsd(peak, warmup, decay_frac=0.2, floor=0.0):
    """Warmup, stable at ``peak``, then a linear decay to ``floor*peak`` over
    the last ``decay_frac`` of the run. Before the decay starts, nothing it
    returns depends on ``total``."""
    def f(step, total):
        if step < warmup:
            return peak * (step + 1) / warmup
        d0 = int(round(total * (1 - decay_frac)))
        if step < d0:
            return peak
        t = (step - d0 + 1) / max(1, total - d0)
        return peak * (1 - (1 - floor) * t)
    return f


# ------------------------------------------------------------------- data
_DOCS = None


def docs():
    global _DOCS
    if _DOCS is None:
        _DOCS, _ = load_documents()
    return _DOCS


def train_stream(seed: int):
    return Sampler(docs()["train"], BATCH, max_len=SEQ, min_len=SEQ,
                   mode="fixed", seed=1000 + seed)


def val_batches(device):
    s = Sampler(docs()["validation"], BATCH, max_len=SEQ, min_len=SEQ,
                mode="fixed", seed=7)
    return [s.batch().to(device) for _ in range(N_VAL_BATCHES)]


@torch.no_grad()
def evaluate(model, batches) -> float:
    model.eval()
    tot = sum(float(model(b.inputs, b.targets)) for b in batches)
    model.train()
    return tot / len(batches)


# ------------------------------------------------------------- optimiser
def param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


class ManualAdamW(torch.optim.Optimizer):
    """AdamW written out, with bias correction switchable.

    With ``bias_correction=True`` this is the same arithmetic as
    ``torch.optim.AdamW`` (E2 checks that it is). With ``False`` it is Adam as
    it would be without Section 6's correction: the raw averages go straight
    into the step, so step t is exactly ``(1 - b1**t) / sqrt(1 - b2**t)`` times
    the corrected one for the same m and v, whatever the gradients were.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, bias_correction=True):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay,
                                      bias_correction=bias_correction))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["t"] += 1
                t, m, v = st["t"], st["m"], st["v"]
                p.mul_(1 - g["lr"] * g["weight_decay"])
                m.lerp_(p.grad, 1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                if g["bias_correction"]:
                    denom = (v.sqrt() / math.sqrt(1 - b2 ** t)).add_(g["eps"])
                    p.addcdiv_(m, denom, value=-g["lr"] / (1 - b1 ** t))
                else:
                    p.addcdiv_(m, v.sqrt().add_(g["eps"]), value=-g["lr"])


def layer_names(model):
    """Every 2-D weight, i.e. every tensor that is a layer rather than a norm."""
    return [n for n, p in model.named_parameters() if p.dim() >= 2]


# ------------------------------------------------------------------- loop
@dataclass
class Run:
    lr: list = field(default_factory=list)
    loss: list = field(default_factory=list)
    val_steps: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    ratio: dict = field(default_factory=dict)     # layer -> [per step]
    seconds: float = 0.0
    n_params: int = 0
    diverged: bool = False
    w_norm: dict = field(default_factory=dict)    # layer -> ||W|| at the end


def train(
    *,
    d_model: int = 256,
    n_layer: int = 4,
    schedule,
    steps: int = 300,
    total: int | None = None,
    seed: int = 0,
    betas=(0.9, 0.95),
    weight_decay: float = 0.1,
    clip: float | None = 1.0,
    eval_every: int = 0,
    eval_at=(),
    log_ratio: bool = False,
    device=None,
    val=None,
    checkpoint_at=(),
    init_state=None,
    start_step: int = 0,
    optimizer: str = "torch",
):
    """Train for ``steps`` optimiser steps of a run whose nominal length is
    ``total`` (defaults to ``steps``). Stopping early, the cosine case, is
    ``steps < total``. Returns a :class:`Run`, and a dict of state snapshots
    for every step in ``checkpoint_at`` (used to branch a WSD decay)."""
    total = total or steps
    device = device or pick_device()
    set_seed(seed)
    model = GPT(Config(d_model=d_model, n_layer=n_layer)).to(device)
    if optimizer == "torch":
        opt = torch.optim.AdamW(param_groups(model, weight_decay), lr=1.0,
                                betas=betas, eps=1e-8)
    else:                         # "manual" or "manual_nobc"
        opt = ManualAdamW(param_groups(model, weight_decay), lr=1.0,
                          betas=betas, eps=1e-8,
                          bias_correction=(optimizer == "manual"))
    stream = train_stream(seed)
    if init_state is not None:
        model.load_state_dict(init_state["model"])
        opt.load_state_dict(init_state["opt"])
        for _ in range(start_step):          # replay the stream to the branch
            stream.batch()
    val = val if val is not None else val_batches(device)
    run = Run(n_params=model.n_params())
    named = [(n, p) for n, p in model.named_parameters() if p.dim() >= 2]
    if log_ratio:
        run.ratio = {n: [] for n, _ in named}
    snaps = {}

    t0 = time.perf_counter()
    for step in range(start_step, start_step + steps):
        b = stream.batch().to(device)
        lr = schedule(step, total)
        for g in opt.param_groups:
            g["lr"] = lr
        loss = model(b.inputs, b.targets)
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if log_ratio:
            before = [p.detach().clone() for _, p in named]
        opt.step()
        opt.zero_grad(set_to_none=True)
        if log_ratio:
            with torch.no_grad():
                r = torch.stack([(p - w0).norm() / w0.norm()
                                 for (_, p), w0 in zip(named, before)]).cpu()
            for (n, _), x in zip(named, r.tolist()):
                run.ratio[n].append(x)
        lv = float(loss.detach())
        run.lr.append(lr)
        run.loss.append(lv)
        if not math.isfinite(lv) or lv > 20:
            run.diverged = True
            break
        s1 = step + 1                                  # steps completed
        if (eval_every and s1 % eval_every == 0) or s1 in eval_at \
                or s1 == start_step + steps:
            run.val_steps.append(s1)
            run.val_loss.append(evaluate(model, val))
        if s1 in checkpoint_at:
            snaps[s1] = {"model": {k: v.detach().clone()
                                   for k, v in model.state_dict().items()},
                         "opt": _clone_opt(opt.state_dict())}
    run.seconds = time.perf_counter() - t0
    with torch.no_grad():
        run.w_norm = {n: float(p.norm()) for n, p in named}
    if checkpoint_at:
        return run, snaps
    return run


def _clone_opt(sd):
    import copy
    return copy.deepcopy(sd)
