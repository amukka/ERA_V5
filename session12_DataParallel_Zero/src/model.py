"""A small GPT cut into groups, because ZeRO-3 needs a model with seams.

The model is an ordinary pre-norm decoder.  What is unusual is how it is held.
Nothing here is an ``nn.Module`` with weights inside it: a group's parameters are
a flat 1-D tensor, and ``layer_forward`` is a function that takes that flat
tensor, views the individual matrices out of it, and returns the group's output.

That shape is forced by the thing being demonstrated.  Under ZeRO-3 a rank does
not have the weights for group ``g`` until it asks the other ranks for them, and
it gives them back immediately afterwards.  A model whose weights live inside its
modules cannot express "I do not have this layer right now"; a model whose
weights arrive as an argument can.  It also makes sharding trivially definable --
a shard is a contiguous slice of the group's flat tensor -- which is what FSDP2
does when it splits each parameter along its first dimension.

The groups are the units ZeRO-3 gathers, in the same sense that
``fully_shard(block)`` makes each block a unit:

    group 0        token and position embeddings
    groups 1..L    one transformer block each
    group L+1      the final norm and the output head
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .data import VOCAB_SIZE


@dataclass
class Config:
    vocab_size: int = VOCAB_SIZE
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    d_ff: int = 512
    max_seq: int = 128

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head

    @property
    def n_groups(self) -> int:
        return self.n_layer + 2


def group_shapes(cfg: Config, g: int) -> list[tuple[str, tuple[int, ...]]]:
    """The named tensors living in group ``g``, in flat-tensor order."""
    D, F_, V, T = cfg.d_model, cfg.d_ff, cfg.vocab_size, cfg.max_seq
    if g == 0:
        return [("tok_emb", (V, D)), ("pos_emb", (T, D))]
    if g == cfg.n_layer + 1:
        return [("ln_f.w", (D,)), ("ln_f.b", (D,)), ("head", (V, D))]
    return [("ln1.w", (D,)), ("ln1.b", (D,)),
            ("qkv", (3 * D, D)), ("proj", (D, D)),
            ("ln2.w", (D,)), ("ln2.b", (D,)),
            ("fc", (F_, D)), ("out", (D, F_))]


def group_numel(cfg: Config, g: int) -> int:
    return sum(math.prod(s) for _, s in group_shapes(cfg, g))


def total_params(cfg: Config) -> int:
    return sum(group_numel(cfg, g) for g in range(cfg.n_groups))


def init_group(cfg: Config, g: int, seed: int = 0) -> torch.Tensor:
    """Initialise group ``g`` as one flat fp32 tensor.  Seeded per group so that
    every rank builds the identical model without having to broadcast it."""
    gen = torch.Generator().manual_seed(seed * 1000 + g)
    parts = []
    for name, shape in group_shapes(cfg, g):
        if name.endswith(".w"):                      # LayerNorm gain
            t = torch.ones(shape)
        elif name.endswith(".b"):                    # LayerNorm bias
            t = torch.zeros(shape)
        else:
            t = torch.empty(shape).normal_(0.0, 0.02, generator=gen)
        parts.append(t.reshape(-1))
    return torch.cat(parts)


def views(cfg: Config, g: int, flat: torch.Tensor) -> dict[str, torch.Tensor]:
    """Cut a group's flat tensor back into its named matrices, without copying."""
    out, off = {}, 0
    for name, shape in group_shapes(cfg, g):
        n = math.prod(shape)
        out[name] = flat[off:off + n].view(shape)
        off += n
    return out


def _layer_norm(x, w, b):
    return F.layer_norm(x, (x.shape[-1],), w, b, eps=1e-5)


def layer_forward(cfg: Config, g: int, flat: torch.Tensor, x: torch.Tensor):
    """One group's forward pass, computed in fp32 from a (possibly bf16) flat
    parameter tensor.  The cast is deliberate and is where mixed precision
    actually lives: the weights are *stored* in 2 bytes and *computed* in 4."""
    p = views(cfg, g, flat.float() if flat.dtype != torch.float32 else flat)

    if g == 0:
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        return p["tok_emb"][x] + p["pos_emb"][pos][None]

    if g == cfg.n_layer + 1:
        h = _layer_norm(x, p["ln_f.w"], p["ln_f.b"])
        return h @ p["head"].t()

    B, T, D = x.shape
    H, Dh = cfg.n_head, cfg.head_dim

    h = _layer_norm(x, p["ln1.w"], p["ln1.b"])
    qkv = h @ p["qkv"].t()
    q, k, v = qkv.split(D, dim=2)
    q = q.view(B, T, H, Dh).transpose(1, 2)
    k = k.view(B, T, H, Dh).transpose(1, 2)
    v = v.view(B, T, H, Dh).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
    causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
    ctx = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(B, T, D)
    x = x + ctx @ p["proj"].t()

    h = _layer_norm(x, p["ln2.w"], p["ln2.b"])
    h = F.gelu(h @ p["fc"].t())
    return x + h @ p["out"].t()


def loss_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


# ---------------------------------------------------------------------------
# forward and backward, one group at a time
# ---------------------------------------------------------------------------

def forward_backward(cfg: Config, provider, x, y, on_grad=None, meter=None):
    """Run the model group by group, then walk back through it group by group.

    ``provider`` is the only way this function can see a weight.  It is called
    as ``provider.acquire(g)`` just before group ``g`` is needed and
    ``provider.release(g)`` the moment it is not needed any more.  Under data
    parallelism and ZeRO stages 1 and 2 acquiring is free, because the rank
    already holds every group.  Under stage 3 it is an all-gather and releasing
    genuinely frees the memory, which is the entire difference between the
    stages and the reason this function takes a provider rather than a model.

    The backward pass here is the recompute kind: the forward stores only the
    tensor entering each group, and each group's activations are rebuilt from
    that boundary when the backward reaches it.  That is activation
    checkpointing, and it is not incidental -- it is what lets stage 3 release a
    group's weights during the forward at all, since the backward re-acquires
    them anyway.  Every arrangement in this session uses the same routine, so
    none of them can win or lose on the strength of a different backward.

    ``on_grad(g, grad_fp32)`` is called the instant group ``g``'s gradient
    exists, which is what lets ZeRO-2 and ZeRO-3 reduce-scatter it and drop the
    full-size buffer before the next group's gradient is computed at all.  With
    no callback the gradients are collected into a dict and returned, which is
    what a single-process run does.

    Returns ``(loss, grads_or_None, activation_bytes)``.
    """
    boundaries = [x]                      # boundaries[g] enters group g
    h = x
    with torch.no_grad():
        for g in range(cfg.n_groups):
            flat = provider.acquire(g)
            h = layer_forward(cfg, g, flat, h)
            provider.release(g)
            boundaries.append(h)
            if meter is not None:
                meter.hold(f"activations:boundary{g}", h)

    activation_bytes = sum(b.numel() * b.element_size() for b in boundaries)

    logits = boundaries[-1].detach().requires_grad_(True)
    loss = loss_from_logits(logits, y)
    loss.backward()
    grad_out = logits.grad

    grads = None if on_grad is not None else {}
    for g in reversed(range(cfg.n_groups)):
        flat = provider.acquire(g).float().detach().requires_grad_(True)
        inp = boundaries[g]
        wants_input_grad = g > 0
        if wants_input_grad:
            inp = inp.detach().requires_grad_(True)
        out = layer_forward(cfg, g, flat, inp)
        targets = [flat] + ([inp] if wants_input_grad else [])
        got = torch.autograd.grad(out, targets, grad_outputs=grad_out)
        grad_out = got[1] if wants_input_grad else None
        provider.release(g)
        if on_grad is not None:
            on_grad(g, got[0])
        else:
            grads[g] = got[0]
        boundaries[g + 1] = None
        if meter is not None:
            meter.free(f"activations:boundary{g}")
    return float(loss.detach()), grads, activation_bytes


class LocalParams:
    """The trivial provider: one process, every weight already in hand."""

    def __init__(self, flats):
        self.flats = flats

    def acquire(self, g):
        return self.flats[g]

    def release(self, g):
        pass
