"""A small decoder, built the way section 2 describes the modern block.

Pre-norm, RMSNorm, SwiGLU.  None of that is what this session is about -- it is
the upstream object the output head needs, and it is written out rather than
imported so the parameter counts in deliverable 6 are counts of code that is
visible here.

Two things are deliberate and both serve later deliverables:

``hidden()`` is separate from the head.  The chunked cross-entropy in
deliverable 7 needs the hidden states and the head's weight matrix but must
never see a full logits tensor, so the head cannot be welded into ``forward``.

``trace`` threads through the forward pass.  When it is a list, every
intermediate is recorded with its shape and one line saying what each axis
selects.  Deliverable 1 asks for exactly that, and having the model produce it
means the table cannot drift away from the code.
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
    d_ff: int = 704          # SwiGLU's three matrices at ~2.75x, a multiple of 64
    max_seq: int = 256
    tie_embeddings: bool = False
    extra_head: bool = False  # deliverable 8: a second head predicting t+2

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head


def note(trace, name, tensor, meaning):
    """Record one tensor, with a sentence naming what each axis selects."""
    if trace is not None:
        trace.append({
            "name": name,
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "numel": tensor.numel(),
            "bytes": tensor.numel() * tensor.element_size(),
            "meaning": meaning,
        })
    return tensor


class RMSNorm(nn.Module):
    """Scale without centring: x / rms(x) * g."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


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

        qkv = note(trace, f"{p}.qkv", self.qkv(x),
                   "B=rows in the batch, T=positions, 3D=query, key and value "
                   "packed into one projection so it is a single matmul")
        q, k, v = qkv.split(D, dim=2)
        q = note(trace, f"{p}.q", q.view(B, T, H, Dh).transpose(1, 2),
                 "B=rows, H=heads, T=the position asking, Dh=d_model/H, the "
                 "slice of the channel this head owns")
        k = note(trace, f"{p}.k", k.view(B, T, H, Dh).transpose(1, 2),
                 "B=rows, H=heads, T=the position being asked about, "
                 "Dh=per-head width")
        v = note(trace, f"{p}.v", v.view(B, T, H, Dh).transpose(1, 2),
                 "B=rows, H=heads, T=the position whose content gets carried, "
                 "Dh=per-head width")

        scores = note(trace, f"{p}.scores",
                      (q @ k.transpose(-2, -1)) / math.sqrt(Dh),
                      "B=rows, H=heads, T_q=querying position, T_k=attended "
                      "position; entry (b,h,i,j) is how much i wants j")
        weights = note(trace, f"{p}.weights", F.softmax(scores + attn_bias, -1),
                       "same axes as scores, now a distribution over T_k for "
                       "each (row, head, T_q) -- each slice sums to 1")
        ctx = note(trace, f"{p}.context", weights @ v,
                   "B=rows, H=heads, T=positions, Dh=per-head width; the value "
                   "vectors averaged with the attention weights")
        ctx = note(trace, f"{p}.merged", ctx.transpose(1, 2).reshape(B, T, D),
                   "B=rows, T=positions, D=d_model; the heads concatenated "
                   "back into one channel axis")
        return note(trace, f"{p}.out", self.proj(ctx),
                    "B=rows, T=positions, D=d_model; attention's contribution "
                    "to the residual stream")


class SwiGLU(nn.Module):
    """down(silu(gate(x)) * up(x)) -- three matrices, section 2's default."""

    def __init__(self, cfg: Config, layer: int):
        super().__init__()
        self.layer = layer
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x, trace=None):
        p = f"block{self.layer}.ffn"
        h = note(trace, f"{p}.hidden", F.silu(self.gate(x)) * self.up(x),
                 "B=rows, T=positions, F=d_ff, the wide inner width; one "
                 "branch decides what to pass, the other how much")
        return note(trace, f"{p}.out", self.down(h),
                    "B=rows, T=positions, D=d_model; the FFN's contribution to "
                    "the residual stream")


class Block(nn.Module):
    def __init__(self, cfg: Config, layer: int):
        super().__init__()
        self.layer = layer
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, layer)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg, layer)

    def forward(self, x, attn_bias, trace=None):
        p = f"block{self.layer}"
        x = x + self.attn(note(trace, f"{p}.norm1", self.norm1(x),
                               "B=rows, T=positions, D=d_model; the residual "
                               "stream normalised per position before "
                               "attention reads it"), attn_bias, trace)
        x = note(trace, f"{p}.resid_attn", x,
                 "B=rows, T=positions, D=d_model; residual stream with "
                 "attention's answer added back -- nothing overwritten")
        x = x + self.ffn(note(trace, f"{p}.norm2", self.norm2(x),
                              "B=rows, T=positions, D=d_model; normalised "
                              "again before the FFN reads it"), trace)
        return note(trace, f"{p}.resid_ffn", x,
                    "B=rows, T=positions, D=d_model; residual stream leaving "
                    "this block")


class TinyGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head2 = (nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
                      if cfg.extra_head else None)
        self.apply(self._init)
        # tie *after* init, or the shared tensor gets initialised twice
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # -- parameter accounting -----------------------------------------------
    def n_params(self, embeddings: bool = True) -> int:
        """Total parameters.  Tied weights are counted once, which is the
        point of counting them this way: ``parameters()`` de-duplicates shared
        tensors, so a tied model really is smaller in memory and not merely on
        a diagram."""
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        if embeddings:
            return total
        drop = self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        if not self.cfg.tie_embeddings:
            drop += self.head.weight.numel()
        return total - drop

    # -- the forward pass ----------------------------------------------------
    def attn_bias(self, seqs, dtype, trace=None):
        """Additive mask: causal, no attending to padding, and -- optionally --
        no attending across a document boundary.

        The diagonal is re-opened everywhere at the end.  A fully masked row
        makes softmax produce NaN, and a padded row is dropped by the loss mask
        anyway, so letting a padded position attend to itself keeps the
        arithmetic finite without letting it reach anything real.
        """
        tokens, valid = seqs.tokens, seqs.valid
        B, T = tokens.shape
        device = tokens.device
        causal = torch.ones(T, T, dtype=torch.bool, device=device).tril()
        keep = causal[None, None] & valid[:, None, None, :]
        bias = torch.zeros(B, 1, T, T, device=device, dtype=dtype)
        bias = bias.masked_fill(~keep, torch.finfo(dtype).min)
        eye = torch.eye(T, dtype=torch.bool, device=device)[None, None]
        bias = bias.masked_fill(eye, 0.0)
        return note(trace, "attn_bias", bias,
                    "B=rows, 1=broadcast over heads, T_q=querying position, "
                    "T_k=attended position; 0 where the edge is allowed and "
                    "-inf where it is not")

    def hidden(self, seqs, trace=None):
        """token ids -> [B, T, D] hidden states.  The head is not applied."""
        tokens = seqs.tokens
        B, T = tokens.shape
        note(trace, "tokens", tokens,
             "B=rows in the batch, T=positions; each entry is a token id in "
             "[0, V).  Unshifted -- this is what the harness is handed")
        note(trace, "valid", seqs.valid,
             "B=rows, T=positions; True where the position holds a real token "
             "rather than padding")

        pos = torch.arange(T, device=tokens.device)
        tok = note(trace, "tok_emb", self.tok_emb(tokens),
                   "B=rows, T=positions, D=d_model; one learned vector per "
                   "token id")
        pe = note(trace, "pos_emb", self.pos_emb(pos)[None],
                  "1=broadcast over rows, T=positions, D=d_model; one learned "
                  "vector per absolute position")
        x = note(trace, "resid_in", tok + pe,
                 "B=rows, T=positions, D=d_model; the residual stream as it "
                 "enters block 0")

        bias = self.attn_bias(seqs, x.dtype, trace)
        for block in self.blocks:
            x = block(x, bias, trace)
        return note(trace, "hidden", self.norm_f(x),
                    "B=rows, T=positions, D=d_model; the final hidden state. "
                    "Every token now has a vector that has seen its context")

    def logits(self, h, trace=None, name="logits"):
        return note(trace, name, self.head(h),
                    "B=rows, T=positions, V=vocab size; one unnormalised score "
                    "for every token the model could name next, at every "
                    "position.  V/D = "
                    f"{self.cfg.vocab_size / self.cfg.d_model:.0f}x larger "
                    "than the hidden state that produced it")

    def forward(self, seqs, trace=None):
        return self.logits(self.hidden(seqs, trace), trace)
