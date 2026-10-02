"""A GPT whose feed-forward blocks are either dense or sparse mixtures of experts, plus the
upcycling step that turns the first into the second.

Dense block:  x + attn(ln1 x), then + mlp(ln2 x)          mlp = proj(gelu(fc(x)))
MoE block:    same attention; the mlp is replaced by n_exp copies of the mlp and a linear router.
              The router scores every expert, keeps the top-k, and softmax-normalises the kept scores,
              so the k gates always sum to 1.  If all experts equal the dense mlp, the layer computes
              exactly the dense mlp, whatever the router says.  That is what makes the swap lossless.
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Cfg:
    vocab_size: int = 8192
    seq_len: int = 256
    n_layer: int = 8
    n_head: int = 6
    d_model: int = 384
    n_exp: int = 0          # 0 = dense
    top_k: int = 2
    aux_coef: float = 0.01  # Switch-style load-balancing loss weight


class Attention(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.nh = c.n_head
        s.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        s.proj = nn.Linear(c.d_model, c.d_model, bias=False)

    def forward(s, x):
        B, T, D = x.shape
        q, k, v = s.qkv(x).split(D, 2)
        q, k, v = (t.view(B, T, s.nh, D // s.nh).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.proj(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.fc = nn.Linear(c.d_model, 4 * c.d_model, bias=False)
        s.proj = nn.Linear(4 * c.d_model, c.d_model, bias=False)

    def forward(s, x):
        return s.proj(F.gelu(s.fc(x), approximate="tanh"))


class MoE(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.k, s.E = c.top_k, c.n_exp
        s.router = nn.Linear(c.d_model, c.n_exp, bias=False)
        s.experts = nn.ModuleList(MLP(c) for _ in range(c.n_exp))
        s.stats = None

    def forward(s, x):
        B, T, D = x.shape
        x = x.reshape(-1, D)
        N = x.shape[0]
        logits = s.router(x).float()
        probs = logits.softmax(-1)
        top_v, top_i = logits.topk(s.k, -1)
        gate = top_v.softmax(-1).to(x.dtype)                       # (N, k), sums to 1
        flat_e = top_i.reshape(-1)                                  # (N*k,) expert of each assignment
        order = flat_e.argsort()
        tok = order // s.k                                          # token of each sorted assignment
        counts = torch.bincount(flat_e, minlength=s.E)
        g = gate.reshape(-1)[order]
        out = torch.zeros_like(x)
        start = 0
        for e, n in enumerate(counts.tolist()):
            if n:
                sl = slice(start, start + n)
                t = tok[sl]
                y = s.experts[e](x[t]) * g[sl, None]
                out.index_add_(0, t, y)
            start += n
        # Switch aux loss: E * sum_e (fraction of assignments to e) * (mean router prob of e)
        frac = counts.float() / (N * s.k)
        s.aux = s.E * (frac * probs.mean(0)).sum()
        s.stats = counts.detach()
        return out.view(B, T, D)


class Block(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        s.attn = Attention(c)
        s.mlp = MoE(c) if c.n_exp else MLP(c)

    def forward(s, x):
        x = x + s.attn(s.ln1(x))
        return x + s.mlp(s.ln2(x))


class GPT(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.c = c
        s.wte = nn.Embedding(c.vocab_size, c.d_model)
        s.wpe = nn.Embedding(c.seq_len, c.d_model)
        s.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        s.ln_f = nn.LayerNorm(c.d_model)
        s.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        s.head.weight = s.wte.weight
        s.apply(s._init)
        for n, p in s.named_parameters():
            if n.endswith("proj.weight"):
                nn.init.normal_(p, 0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(s, idx, tgt=None):
        x = s.wte(idx) + s.wpe(torch.arange(idx.shape[1], device=idx.device))
        for b in s.blocks:
            x = b(x)
        logits = s.head(s.ln_f(x))
        if tgt is None:
            return logits
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.reshape(-1))
        aux = sum(b.mlp.aux for b in s.blocks) / len(s.blocks) if s.c.n_exp else ce.new_zeros(())
        return ce, aux

    def n_params(s):
        return sum(p.numel() for p in s.parameters())

    def n_active(s):
        """parameters a single token touches: everything except the (E - k) unused experts per layer."""
        c = s.c
        if not c.n_exp:
            return s.n_params()
        per_exp = sum(p.numel() for p in s.blocks[0].mlp.experts[0].parameters())
        return s.n_params() - c.n_layer * (c.n_exp - c.top_k) * per_exp


@torch.no_grad()
def upcycle(dense, n_exp, top_k=2, aux_coef=0.01, drop=0.0, router_std=0.02, seed=0):
    """Turn a trained dense GPT into a MoE GPT.

    Every expert starts as a copy of the dense mlp (sparse upcycling, Komatsuzaki et al. 2022).
    drop > 0 is drop-upcycling (Nakamura et al. 2025): in every expert except expert 0, a fraction
    `drop` of the hidden neurons is re-initialised, so the experts start different from each other.
    Expert 0 stays an exact copy, so the function is still preserved when the router sends all mass to it,
    but with drop > 0 the swap is no longer lossless; the measured initial loss shows by how much.
    """
    g = torch.Generator().manual_seed(seed)
    c = Cfg(**{**asdict(dense.c), "n_exp": n_exp, "top_k": top_k, "aux_coef": aux_coef})
    moe = GPT(c).to(next(dense.parameters()).device)
    sd = dense.state_dict()
    new = {}
    for k, v in sd.items():
        if ".mlp." not in k:
            new[k] = v
    moe.load_state_dict(new, strict=False)
    H = 4 * c.d_model
    for lb, lm in zip(dense.blocks, moe.blocks):
        lm.mlp.router.weight.copy_(torch.randn(n_exp, c.d_model, generator=g).to(lb.ln1.weight.device) * router_std)
        for e, ex in enumerate(lm.mlp.experts):
            ex.fc.weight.copy_(lb.mlp.fc.weight)
            ex.proj.weight.copy_(lb.mlp.proj.weight)
            if drop > 0 and e > 0:
                idx = torch.randperm(H, generator=g)[: int(drop * H)].to(ex.fc.weight.device)
                ex.fc.weight[idx] = torch.randn(len(idx), c.d_model, generator=g).to(ex.fc.weight.device) * 0.02
                ex.proj.weight[:, idx] = (torch.randn(c.d_model, len(idx), generator=g).to(ex.fc.weight.device)
                                          * 0.02 / math.sqrt(2 * c.n_layer))
    return moe
