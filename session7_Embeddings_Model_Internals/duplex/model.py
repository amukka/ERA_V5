"""A small GPT whose token-facing paths are swappable.

The trunk is deliberately ordinary and deliberately identical across arms. The
only thing that changes between arms is the object at the front door and the
object at the back door, which is the whole point: if the head-free arm reaches
the same loss as the dense-head arm on the same trunk, the same data and the
same seed, then the D x V head was not carrying information the byte code did
not already have.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec import CodecConfig
from .embedding import DenseTokenPath, DuplexEmbedding, KroneckerInDensOut


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    block_size: int = 128
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_head, dropout=cfg.dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


def build_token_path(arm: str, byte_seqs, cfg: ModelConfig, codec: CodecConfig):
    """The five arms of E3, differing only in how tokens enter and leave."""
    V, D = cfg.vocab_size, cfg.d_model
    if arm == "dense":
        return DenseTokenPath(V, D, tied=False)
    if arm == "dense_tied":
        return DenseTokenPath(V, D, tied=True)
    if arm == "kron_dense":
        return KroneckerInDensOut(byte_seqs, D, codec)
    if arm in ("duplex", "duplex_colce"):
        return DuplexEmbedding(byte_seqs, D, codec, tied=False)
    if arm == "duplex_tied":
        return DuplexEmbedding(byte_seqs, D, codec, tied=True)
    raise ValueError(f"unknown arm {arm!r}")


ARMS = ["dense", "dense_tied", "kron_dense", "duplex", "duplex_tied", "duplex_colce"]

# Arms with no per-token parameters anywhere: the vocabulary enters only as a
# fixed index table built from the tokens' bytes.
HEAD_FREE = {"duplex", "duplex_tied", "duplex_colce"}

# The one arm that is also trained without ever forming a [.., V] tensor.
COLUMN_OBJECTIVE = {"duplex_colce"}


class TinyGPT(nn.Module):
    def __init__(self, arm: str, byte_seqs, cfg: ModelConfig, codec: CodecConfig):
        super().__init__()
        self.arm = arm
        self.cfg = cfg
        self.tokens = build_token_path(arm, byte_seqs, cfg, codec)
        self.pos = nn.Embedding(cfg.block_size, cfg.d_model)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.register_buffer(
            "mask",
            torch.triu(torch.full((cfg.block_size, cfg.block_size), float("-inf")), 1),
        )

    def trunk(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        x = self.tokens(ids) + self.pos(torch.arange(T, device=ids.device))
        m = self.mask[:T, :T]
        for block in self.blocks:
            x = block(x, m)
        return self.ln_f(x)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.trunk(ids)
        if self.arm in COLUMN_OBJECTIVE:
            # Trained as pos_dim independent classifications, so its scores are
            # summed log-probabilities rather than an inner product.
            return self.tokens.factorized_logits(h)
        return self.tokens.logits(h)

    # ------------------------------------------------------------------ loss

    def vocab_loss(self, ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Ordinary next-token cross-entropy over the whole vocabulary.

        Every arm is scored with this, so the arms are comparable. For the
        head-free arms the logits come from the sparse byte-code matmul rather
        than from a weight matrix, but the loss is the same quantity.
        """
        logits = self(ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )

    def column_loss(self, ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy against the code's columns instead of against the vocab.

        The duplex code is exactly one-hot per column, so the prediction is
        pos_dim independent char_dim-way classifications. Nothing in this loss
        mentions V, which is the part that makes a million-token vocabulary
        cost nothing at training time as well as at parameter time.
        """
        y = self.tokens.code_logits(self.trunk(ids))
        P, C = self.tokens.cfg.pos_dim, self.tokens.cfg.char_dim
        grid = y.reshape(-1, P, C)
        labels = self.tokens.code_targets(targets).reshape(-1, P)
        return F.cross_entropy(grid.reshape(-1, C), labels.reshape(-1))

    # --------------------------------------------------------------- costing

    def parameter_report(self) -> dict:
        rep = dict(self.tokens.parameter_report())
        total = sum(p.numel() for p in self.parameters())
        rep["trunk"] = total - rep["total_token_facing"]
        rep["total"] = total
        rep["token_facing_share"] = rep["total_token_facing"] / total
        return rep
