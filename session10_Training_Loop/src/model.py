"""A small GPT, written so that one step can be made to narrate itself.

Nothing here is unusual.  The only addition to a textbook decoder is the
``trace`` argument threaded through ``forward``: when it is a list, every
intermediate tensor is recorded with its shape, its dtype and one line saying
what each of its dimensions means.  That is deliverable 1, and it is worth
having the model produce it rather than a separate document, because a separate
document is a thing that goes stale without telling you.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PAD_ID, VOCAB_SIZE


@dataclass
class Config:
    vocab_size: int = VOCAB_SIZE
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    d_ff: int = 1024
    max_seq: int = 256
    tie_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head


def _note(trace, name, tensor, meaning):
    if trace is not None:
        trace.append(
            {
                "name": name,
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "numel": tensor.numel(),
                "meaning": meaning,
            }
        )
    return tensor


class Attention(nn.Module):
    def __init__(self, cfg: Config, layer: int):
        super().__init__()
        self.cfg, self.layer = cfg, layer
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, attn_bias, trace=None):
        B, T, D = x.shape
        H, Dh = self.cfg.n_head, self.cfg.head_dim
        p = f"block{self.layer}.attn"

        qkv = _note(
            trace, f"{p}.qkv", self.qkv(x),
            "B=rows in the micro-batch, T=positions, 3D=query, key and value "
            "packed into one projection so it is a single matmul",
        )
        q, k, v = qkv.split(D, dim=2)
        q = _note(
            trace, f"{p}.q", q.view(B, T, H, Dh).transpose(1, 2),
            "B=rows, H=attention heads, T=positions asking the question, "
            "Dh=d_model/H, the slice of the channel each head owns",
        )
        k = _note(
            trace, f"{p}.k", k.view(B, T, H, Dh).transpose(1, 2),
            "B=rows, H=heads, T=positions being asked about, Dh=per-head width",
        )
        v = _note(
            trace, f"{p}.v", v.view(B, T, H, Dh).transpose(1, 2),
            "B=rows, H=heads, T=positions whose content gets carried, "
            "Dh=per-head width",
        )

        scores = _note(
            trace, f"{p}.scores", (q @ k.transpose(-2, -1)) / math.sqrt(Dh),
            "B=rows, H=heads, T_q=querying position, T_k=attended position; "
            "entry (b,h,i,j) is how much position i wants position j",
        )
        scores = scores + attn_bias
        weights = _note(
            trace, f"{p}.weights", F.softmax(scores, dim=-1),
            "same axes as scores, now a distribution over T_k for each "
            "(row, head, T_q) -- each slice sums to 1",
        )
        ctx = _note(
            trace, f"{p}.context", weights @ v,
            "B=rows, H=heads, T=positions, Dh=per-head width; the value vectors "
            "averaged with the attention weights",
        )
        ctx = _note(
            trace, f"{p}.merged", ctx.transpose(1, 2).reshape(B, T, D),
            "B=rows, T=positions, D=d_model; the heads concatenated back into "
            "one channel axis",
        )
        return _note(
            trace, f"{p}.out", self.proj(ctx),
            "B=rows, T=positions, D=d_model; attention's contribution to the "
            "residual stream",
        )


class MLP(nn.Module):
    def __init__(self, cfg: Config, layer: int):
        super().__init__()
        self.layer = layer
        self.fc = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.out = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x, trace=None):
        p = f"block{self.layer}.mlp"
        h = _note(
            trace, f"{p}.hidden", F.gelu(self.fc(x)),
            "B=rows, T=positions, F=d_ff, the wide inner width where the "
            "per-position nonlinearity happens",
        )
        return _note(
            trace, f"{p}.out", self.out(h),
            "B=rows, T=positions, D=d_model; the MLP's contribution to the "
            "residual stream",
        )


class Block(nn.Module):
    def __init__(self, cfg: Config, layer: int):
        super().__init__()
        self.layer = layer
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg, layer)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg, layer)

    def forward(self, x, attn_bias, trace=None):
        p = f"block{self.layer}"
        x = x + self.attn(_note(
            trace, f"{p}.ln1", self.ln1(x),
            "B=rows, T=positions, D=d_model; the residual stream normalised "
            "per position before attention reads it",
        ), attn_bias, trace)
        x = _note(
            trace, f"{p}.resid_attn", x,
            "B=rows, T=positions, D=d_model; residual stream after attention "
            "has been added back",
        )
        x = x + self.mlp(_note(
            trace, f"{p}.ln2", self.ln2(x),
            "B=rows, T=positions, D=d_model; residual stream normalised again "
            "before the MLP reads it",
        ), trace)
        return _note(
            trace, f"{p}.resid_mlp", x,
            "B=rows, T=positions, D=d_model; residual stream leaving this block",
        )


class TinyGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # -- parameter accounting ------------------------------------------------
    def n_params(self, embeddings: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters())
        if embeddings:
            return total
        drop = self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        if not self.cfg.tie_embeddings:
            drop += self.head.weight.numel()
        return total - drop

    def _attn_bias(self, mask, T, device, dtype, trace=None):
        """Additive mask: causal, plus -inf on padded key positions."""
        causal = torch.ones(T, T, dtype=torch.bool, device=device).tril()
        keep = causal[None, None] & mask[:, None, None, :]
        bias = torch.zeros(mask.shape[0], 1, T, T, device=device, dtype=dtype)
        bias = bias.masked_fill(~keep, torch.finfo(dtype).min)
        # a fully-masked row would make softmax produce NaN; the diagonal is
        # always legal for a real position, and padded rows are dropped by the
        # loss mask anyway, so re-open the diagonal everywhere.
        eye = torch.eye(T, dtype=torch.bool, device=device)[None, None]
        bias = bias.masked_fill(eye, 0.0)
        return _note(
            trace, "attn_bias", bias,
            "B=rows, 1=broadcast over heads, T_q=querying position, "
            "T_k=attended position; 0 where the edge is allowed and -inf where "
            "it is not (future positions and padding)",
        )

    def forward(self, inputs, mask=None, trace=None):
        B, T = inputs.shape
        device = inputs.device
        if mask is None:
            mask = inputs != PAD_ID

        _note(trace, "inputs", inputs,
              "B=rows in the micro-batch, T=positions; each entry is a token id "
              "in [0, V)")
        _note(trace, "mask", mask,
              "B=rows, T=positions; True where the position holds a real token "
              "rather than padding")

        pos = torch.arange(T, device=device)
        tok = _note(
            trace, "tok_emb", self.tok_emb(inputs),
            "B=rows, T=positions, D=d_model; one learned vector per token id",
        )
        pe = _note(
            trace, "pos_emb", self.pos_emb(pos)[None],
            "1=broadcast over rows, T=positions, D=d_model; one learned vector "
            "per absolute position",
        )
        x = _note(
            trace, "resid_in", tok + pe,
            "B=rows, T=positions, D=d_model; the residual stream as it enters "
            "block 0",
        )

        bias = self._attn_bias(mask, T, device, x.dtype, trace)
        for block in self.blocks:
            x = block(x, bias, trace)

        x = _note(
            trace, "ln_f", self.ln_f(x),
            "B=rows, T=positions, D=d_model; residual stream normalised one "
            "last time before the output head",
        )
        return _note(
            trace, "logits", self.head(x),
            "B=rows, T=positions, V=vocab size; an unnormalised score for every "
            "token the model could name next, at every position",
        )
