"""A small causal language model that takes the batch seriously.

Written by hand rather than pulled from `transformers` for one reason: the batch
this system produces carries `position_ids` and a **4-D block-diagonal causal
attention mask**, and the model has to honour both or the packing guarantees are
decorative. A packed bin holds several unrelated samples; if the model attends
across them, every claim this submission makes about boundary safety is false.

So the forward signature is:

    forward(input_ids, position_ids, attn_mask)   attn_mask: (B, 1, T, T) additive

and there is no fallback path that silently builds a plain causal mask.

`masked_loss` returns the per-position loss vector as well as the scalar, which
is what makes the token-level perplexity trace and the per-shard learning ledger
possible: the trainer attributes each position's loss back to the sample, the
document and the shard that supplied it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        # attn_mask is additive and already encodes causality *and* the packed
        # segment boundaries, so is_causal stays False -- the mask is the truth.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        return x + self.mlp(self.ln2(x))


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int, n_layer: int, n_head: int, n_embd: int,
                 n_positions: int) -> None:
        super().__init__()
        self.config = {"vocab_size": vocab_size, "n_layer": n_layer, "n_head": n_head,
                       "n_embd": n_embd, "n_positions": n_positions}
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_positions, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.wte.weight          # tied embeddings
        self.apply(self._init)
        for name, p in self.named_parameters():     # GPT-2 style scaled init
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
                attn_mask: torch.Tensor) -> torch.Tensor:
        x = self.wte(input_ids) + self.wpe(position_ids)
        for block in self.blocks:
            x = block(x, attn_mask)
        return self.head(self.ln_f(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def masked_loss(logits: torch.Tensor, input_ids: torch.Tensor,
                loss_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Next-token cross-entropy restricted to positions the mask allows.

    Position j predicts token j from context < j, so logits are shifted by one.
    `loss_mask[j] == 1` already guarantees that j-1 and j belong to the same
    packed sample and that neither is padding.

    Returns (scalar mean loss, per-position loss of shape (B, T), loss-token count).
    """
    b, t, _ = logits.shape
    shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    shift_targets = input_ids[:, 1:].reshape(-1)
    flat = F.cross_entropy(shift_logits, shift_targets, reduction="none").view(b, t - 1)

    per_position = torch.zeros(b, t, dtype=flat.dtype, device=flat.device)
    per_position[:, 1:] = flat                     # loss of position j lives at j
    mask = loss_mask.to(per_position.dtype)
    per_position = per_position * mask

    n = int(mask.sum().item())
    total = per_position.sum()
    return (total / max(1, n)), per_position, n


def build_model(cfg, vocab_size: int, device: str = "cpu") -> TinyCausalLM:
    m = cfg.model
    model = TinyCausalLM(vocab_size=vocab_size, n_layer=int(m.n_layer),
                         n_head=int(m.n_head), n_embd=int(m.n_embd),
                         n_positions=int(m.n_positions))
    return model.to(device)
