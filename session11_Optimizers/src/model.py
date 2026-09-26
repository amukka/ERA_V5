"""The Session 10 GPT, with its width made a parameter.

Same architecture as Session 10 (pre-norm blocks, learned positions, GELU MLP,
untied head, every weight initialised N(0, 0.02)), minus the shape tracing, and
with attention through ``scaled_dot_product_attention`` so the width sweep in
E5 finishes in minutes. Heads are always 64 wide, so widening the model adds
heads rather than making each head fatter.

The initialisation is the *standard parameterisation*: one fixed std for every
matrix whatever its fan-in, and one learning rate for every weight. That is the
setting in which Section 12 says the best learning rate drifts with width, and
it is deliberately what E5 measures.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import VOCAB_SIZE


@dataclass
class Config:
    vocab_size: int = VOCAB_SIZE
    n_layer: int = 4
    d_model: int = 256
    max_seq: int = 256
    head_dim: int = 64

    @property
    def n_head(self) -> int:
        return self.d_model // self.head_dim

    @property
    def d_ff(self) -> int:
        return 4 * self.d_model


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        self.ln2 = nn.LayerNorm(D)
        self.fc = nn.Linear(D, cfg.d_ff, bias=False)
        self.out = nn.Linear(cfg.d_ff, D, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H = self.cfg.n_head
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, H, D // H).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T))
