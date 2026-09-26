"""A ~20M-parameter GPT whose residual stream can be advanced five ways.

Every variant uses the same Block and therefore has exactly the same parameters.
Only the rule that moves the residual stream from one layer to the next changes:

    standard   p[l+1] = p[l] + f_l(p[l])                        stored activations, autograd
    midpoint   p[l+1] = p[l-1] + 2h f_l(p[l])                   exact inverse: p[l-1] = p[l+1] - 2h f_l(p[l])
    euler_fp   p[l+1] = p[l] + h f_l(p[l])                      inverse by fixed-point iteration p <- p[l+1] - h f_l(p)
    momentum   v[l+1] = v[l] + h f_l(x[l]);  x[l+1] = x[l] + h v[l+1]
                                                                exact inverse: x[l] = x[l+1] - h v[l+1], v[l] = v[l+1] - h f_l(x[l])
    revnet     y1 = x1 + A_l(x2);  y2 = x2 + M_l(y1)           exact inverse: x2 = y2 - M_l(y1), x1 = y1 - A_l(x2)

f_l is the whole transformer block minus its input: attention, then the MLP,
    a = attn(ln1(p));  f(p) = a + mlp(ln2(p + a))
so `standard` is exactly a pre-LN GPT-2 block and `euler_fp` with h=1 is the same network.

The four reversible variants run their stack inside one autograd.Function
(`ReversibleStack`). Its forward runs under no_grad and saves only the final
state; its backward walks down the layers, rebuilding each layer's input from its
output and re-running that one layer with grad enabled to get the vector-Jacobian
product. Activation memory is therefore one layer's worth, whatever the depth.
"""
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import memory

VARIANTS = ("standard", "midpoint", "euler_fp", "momentum", "revnet")


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    seq_len: int = 512
    n_layer: int = 10
    n_head: int = 6
    d_model: int = 384
    variant: str = "standard"
    h: float = 1.0          # step size of the integrator (ignored by standard/revnet)
    fp_iters: int = 6       # fixed-point iterations per layer for euler_fp
    loss_chunk: int = 0     # >0: compute the head and loss this many sequences at a time, recomputed in backward
    extra: dict = field(default_factory=dict)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (t.view(B, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    """f(p) = a + mlp(ln2(p + a)),  a = attn(ln1(p)).  standard: p + f(p) is the GPT-2 block."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def f(self, p):
        a = self.attn(self.ln1(p))
        return a + self.mlp(self.ln2(p + a))

    # the two halves, used by revnet
    def f_attn(self, p):
        return self.attn(self.ln1(p))

    def f_mlp(self, p):
        return self.mlp(self.ln2(p))


# ----------------------------------------------------------------------------------------------
# the reversible rules: each defines a forward step on a state tuple, and a backward step that
# rebuilds the input state and returns its adjoint.  Everything in here works on detached tensors.
# ----------------------------------------------------------------------------------------------

def _vjp(fn, inp, params, grad_out):
    """Run fn(inp) with grad, return (fn value detached, d<fn,grad_out>/d inp, d/d params)."""
    with torch.enable_grad():
        x = inp.detach().requires_grad_(True)
        y = fn(x)
        grads = torch.autograd.grad(y, [x] + params, grad_out.to(y.dtype), allow_unused=True)
    return y.detach(), grads[0], grads[1:]


class Rule:
    def __init__(self, cfg):
        self.h = cfg.h
        self.cfg = cfg

    def init(self, x):
        raise NotImplementedError

    def output(self, s):
        raise NotImplementedError

    def output_adjoint(self, s, g):
        raise NotImplementedError

    def input_adjoint(self, g):
        return g[0]

    def step(self, l, blk, s):
        raise NotImplementedError

    def back(self, l, blk, params, s, g, pgrads):
        raise NotImplementedError


class Midpoint(Rule):
    # state after layer l (l >= 0): (p[l], p[l+1]);  layer 0 is one Euler step to get the pair going
    def init(self, x):
        return (x,)

    def output(self, s):
        return s[-1]

    def output_adjoint(self, s, g):
        return (torch.zeros_like(g), g)

    def step(self, l, blk, s):
        h = self.h
        if l == 0:
            (p0,) = s
            return (p0, p0 + h * blk.f(p0))
        a, b = s
        return (b, a + 2 * h * blk.f(b))

    def back(self, l, blk, params, s, g, pgrads):
        h = self.h
        u, w = s               # (p[l], p[l+1])
        gu, gw = g
        if l == 0:
            fz, jx, jp = _vjp(blk.f, u, params, h * gw)
            _acc(pgrads, jp)
            return (u,), (gu + gw + jx,)
        fz, jx, jp = _vjp(blk.f, u, params, 2 * h * gw)
        _acc(pgrads, jp)
        prev = w - 2 * h * fz.to(w.dtype)          # p[l-1]
        return (prev, u), (gw, gu + jx)


class EulerFP(Rule):
    # p[l+1] = p[l] + h f(p[l]); the inverse is the fixed point of p = p[l+1] - h f(p)
    def init(self, x):
        return (x,)

    def output(self, s):
        return s[0]

    def output_adjoint(self, s, g):
        return (g,)

    def step(self, l, blk, s):
        (p,) = s
        return (p + self.h * blk.f(p),)

    def back(self, l, blk, params, s, g, pgrads):
        (y,) = s
        (gy,) = g
        p = y
        with torch.no_grad():
            for _ in range(self.cfg.fp_iters):
                p = y - self.h * blk.f(p).to(y.dtype)
        _, jx, jp = _vjp(blk.f, p, params, self.h * gy)
        _acc(pgrads, jp)
        return (p,), (gy + jx,)


class Momentum(Rule):
    # symplectic (semi-implicit) Euler on a position/velocity pair; velocity starts at zero
    def init(self, x):
        return (x, torch.zeros_like(x))

    def output(self, s):
        return s[0]

    def output_adjoint(self, s, g):
        return (g, torch.zeros_like(g))            # v0 = 0 is a constant, so input_adjoint is g[0]

    def step(self, l, blk, s):
        x, v = s
        v = v + self.h * blk.f(x)
        return (x + self.h * v, v)

    def back(self, l, blk, params, s, g, pgrads):
        h = self.h
        x1, v1 = s
        gx1, gv1 = g
        x0 = x1 - h * v1
        gv = gv1 + h * gx1                         # adjoint of v[l+1] in total
        fz, jx, jp = _vjp(blk.f, x0, params, h * gv)
        _acc(pgrads, jp)
        v0 = v1 - h * fz.to(v1.dtype)
        return (x0, v0), (gx1 + jx, gv)


class RevNet(Rule):
    # additive coupling (Gomez et al. 2017, Reformer 2020): two copies of the stream, attention
    # updates one from the other, the MLP updates the other back; output is their mean
    def init(self, x):
        return (x, x)

    def output(self, s):
        return 0.5 * (s[0] + s[1])

    def output_adjoint(self, s, g):
        return (0.5 * g, 0.5 * g)

    def input_adjoint(self, g):
        return g[0] + g[1]                         # both copies started as x

    def step(self, l, blk, s):
        x1, x2 = s
        y1 = x1 + blk.f_attn(x2)
        y2 = x2 + blk.f_mlp(y1)
        return (y1, y2)

    def back(self, l, blk, params, s, g, pgrads):
        y1, y2 = s
        gy1, gy2 = g
        mz, jm, jpm = _vjp(blk.f_mlp, y1, params, gy2)
        _acc(pgrads, jpm)
        x2 = y2 - mz.to(y2.dtype)
        gy1 = gy1 + jm
        az, ja, jpa = _vjp(blk.f_attn, x2, params, gy1)
        _acc(pgrads, jpa)
        x1 = y1 - az.to(y1.dtype)
        return (x1, x2), (gy1, gy2 + ja)


RULES = {"midpoint": Midpoint, "euler_fp": EulerFP, "momentum": Momentum, "revnet": RevNet}


def _acc(pgrads, new):
    for i, gnew in enumerate(new):
        if gnew is None:
            continue
        pgrads[i] = gnew if pgrads[i] is None else pgrads[i] + gnew


class ReversibleStack(torch.autograd.Function):
    """forward(x) -> stack output, storing only the final state; backward rebuilds the rest."""

    @staticmethod
    def forward(ctx, x, model, *flat_params):
        rule, blocks = model.rule, model.blocks
        s = rule.init(x)
        with torch.no_grad():
            for l, blk in enumerate(blocks):
                s = rule.step(l, blk, s)
                if model.trace_states is not None:
                    model.trace_states.append(tuple(t.detach().clone() for t in s))
        ctx.model = model
        ctx.autocast = (torch.is_autocast_enabled(x.device.type),
                        torch.get_autocast_dtype(x.device.type))
        ctx.save_for_backward(*s)
        return rule.output(s)

    @staticmethod
    def backward(ctx, grad_out):
        model = ctx.model
        rule, blocks = model.rule, model.blocks
        s = tuple(ctx.saved_tensors)
        g = rule.output_adjoint(s, grad_out)
        dev = grad_out.device.type
        pgrads = [None] * len(model.flat_params)
        rebuilt = []
        with torch.autocast(dev, dtype=ctx.autocast[1], enabled=ctx.autocast[0]):
            for l in range(len(blocks) - 1, -1, -1):
                lo, hi = model.param_slices[l]
                local = [None] * (hi - lo)
                s, g = rule.back(l, blocks[l], model.flat_params[lo:hi], s, g, local)
                pgrads[lo:hi] = local
                memory.probe()
                if model.trace_states is not None:
                    rebuilt.append(tuple(t.detach().clone() for t in s))
        if model.trace_states is not None:
            model.rebuilt_states = rebuilt[::-1]
        return (rule.input_adjoint(g), None, *pgrads)


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.variant in VARIANTS, cfg.variant
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))
        self.rule = RULES[cfg.variant](cfg) if cfg.variant != "standard" else None
        self.flat_params, self.param_slices = [], []
        for blk in self.blocks:
            ps = list(blk.parameters())
            self.param_slices.append((len(self.flat_params), len(self.flat_params) + len(ps)))
            self.flat_params += ps
        self.trace_states = None     # set to [] to record forward states (diagnostics only)
        self.rebuilt_states = None

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def n_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel() + self.wpe.weight.numel()
        return n

    def hidden(self, idx):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        if self.rule is None:
            for blk in self.blocks:
                x = x + blk.f(x)
                if x.requires_grad and memory.active():
                    x.register_hook(_probe_hook)
            return x
        return ReversibleStack.apply(x, self, *self.flat_params)

    def forward(self, idx, targets=None):
        x = self.ln_f(self.hidden(idx))
        if targets is not None and self.cfg.loss_chunk and self.training:
            return None, self._chunked_loss(x, targets)
        logits = x @ self.wte.weight.T          # tied head
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def _chunked_loss(self, x, targets):
        """Same loss, but the B x T x vocab logits never exist at once: each chunk of sequences is
        projected, scored and dropped, and recomputed during the backward pass."""
        from torch.utils.checkpoint import checkpoint

        def part(xc, tc):
            lg = xc @ self.wte.weight.T
            return F.cross_entropy(lg.float().view(-1, lg.size(-1)), tc.reshape(-1), reduction="sum")

        total = sum(checkpoint(part, xc, tc, use_reentrant=False)
                    for xc, tc in zip(x.split(self.cfg.loss_chunk), targets.split(self.cfg.loss_chunk)))
        return total / targets.numel()


def _probe_hook(grad):
    memory.probe()
    return grad
