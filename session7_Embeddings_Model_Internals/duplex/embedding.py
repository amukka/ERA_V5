"""The duplex input and output paths.

Analysis  (read):   token id -> bytes -> sparse code -> shared projection -> h
Synthesis (write):  h -> shared projection -> code space -> score every token

The two directions use the same fixed codec and the same shape of learned
matrix. The output side holds no per-token parameters at all: the vocabulary
enters only through `idx`, a fixed integer table built from the tokens' bytes.
That is the whole point. A dense head is D x V and grows with the vocabulary;
this head is D x code_dim and does not.

Two implementation notes that matter for the cost claim:

* z-normalisation of the code looks like it should destroy sparsity, since it
  subtracts a constant from all 8,224 entries and makes the code dense. It does
  not, because it is affine. For any vector y,

      <y, (kappa - mu)/sigma>  =  ( <y, kappa> - mu * sum(y) ) / sigma

  and both mu and sigma depend only on the number of nonzeros. So the dense
  correction is one precomputed scalar per token length, and the inner product
  stays a 32-term gather.

* The same identity applied to the analysis side means the input projection is
  a sum of 32 rows of W plus a precomputed column-sum vector, never an 8,224 x D
  matrix multiply.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .codec import CodecConfig, KroneckerCodec


class DuplexEmbedding(nn.Module):
    """Byte-code embedding that can also be run backwards as the output head."""

    def __init__(
        self,
        byte_seqs: list[bytes],
        d_model: int,
        config: CodecConfig,
        tied: bool = False,
    ):
        super().__init__()
        self.codec = KroneckerCodec(config)
        self.cfg = config
        self.d_model = d_model
        self.tied = tied
        self.vocab_size = len(byte_seqs)

        idx, mask, n = self.codec.build_table(byte_seqs)
        self.register_buffer("idx", idx)          # [V, P] flat code indices
        self.register_buffer("mask", mask)        # [V, P]
        self.register_buffer("n", n)              # [V]

        mu, sigma = self.codec.moments(n)
        self.register_buffer("mu", mu.float())
        self.register_buffer("sigma", sigma.float())
        self.register_buffer("inv_sqrt_n", n.float().rsqrt())

        # The whole vocabulary as one sparse [V, code_dim] matrix of raw codes.
        # V * pos_dim nonzeros and no trainable weights: this is the object that
        # stands in for the D x V output head, and scoring is one sparse matmul.
        rows = torch.arange(self.vocab_size).unsqueeze(1).expand_as(idx)[mask]
        cols = idx[mask]
        shape = (self.vocab_size, config.code_dim)
        pos = torch.stack([rows, cols])
        vals = self.inv_sqrt_n.unsqueeze(1).expand_as(idx)[mask]
        self.register_buffer("K", torch.sparse_coo_tensor(pos, vals, shape).coalesce())
        self._K01 = None  # built on demand; see `k01`

        self.W_ana = nn.Parameter(torch.randn(config.code_dim, d_model) * (1.0 / config.code_dim ** 0.5))
        if tied:
            self.W_syn = None
        else:
            self.W_syn = nn.Parameter(torch.randn(d_model, config.code_dim) * (1.0 / d_model ** 0.5))

        # A z-normalised code has unit variance across code_dim entries, so its
        # norm is sqrt(code_dim) ~ 91 by construction. Left alone, the inner
        # product against it starts about ninety times too sharp and the run
        # diverges. This is one scalar, it is learned, and — like everything
        # else on this path — it does not depend on the vocabulary.
        self.logit_scale = nn.Parameter(torch.tensor(config.code_dim ** -0.5).log())

    # ------------------------------------------------------------- analysis

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """[B, T] ids -> [B, T, D]. A 32-row gather, not a matrix multiply."""
        idx = self.idx[token_ids]                       # [B, T, P]
        mask = self.mask[token_ids].unsqueeze(-1)       # [B, T, P, 1]
        rows = self.W_ana[idx] * mask                   # [B, T, P, D]
        summed = rows.sum(dim=2)                        # [B, T, D]

        scale = self.inv_sqrt_n[token_ids].unsqueeze(-1)
        mu = self.mu[token_ids].unsqueeze(-1)
        sigma = self.sigma[token_ids].unsqueeze(-1)
        colsum = self.W_ana.sum(dim=0)                  # [D], the dense correction
        return (summed * scale - mu * colsum) / sigma

    # ------------------------------------------------------------ synthesis

    def code_logits(self, h: torch.Tensor) -> torch.Tensor:
        """[.., D] -> [.., code_dim]. The predicted Kronecker code."""
        W = self.W_ana.t() if self.tied else self.W_syn
        return h @ W

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """[B, T, D] -> [B, T, V], with no per-token parameters.

        The score for token v is the inner product of the predicted code with
        v's z-normalised code, and since z-normalisation is affine that is

            ( <y, kappa_v> - mu_v * sum(y) ) / sigma_v

        where the first term is a sparse matmul against `K`, whose only content
        is where each token's bytes fall. Cost is O(V * P) per position rather
        than O(V * D), and the vocabulary contributes no weights.
        """
        y = self.code_logits(h)                          # [B, T, N]
        flat = y.reshape(-1, y.shape[-1])                # [BT, N]
        acc = torch.sparse.mm(self.K, flat.t()).t()      # [BT, V]
        acc = acc - self.mu.unsqueeze(0) * flat.sum(dim=1, keepdim=True)
        acc = acc * (self.logit_scale.exp() / self.sigma.unsqueeze(0))
        return acc.view(*h.shape[:-1], self.vocab_size)

    @property
    def k01(self) -> torch.Tensor:
        """`K`'s sparsity pattern with unit values, for summing log-probabilities.

        Only the column-objective path needs it, and at a million tokens it is
        another ~0.7 GB of indices, so it is built on first use rather than in
        every model that will never call it.
        """
        if self._K01 is None or self._K01.device != self.K.device:
            k = self.K.coalesce()
            self._K01 = torch.sparse_coo_tensor(
                k.indices(), torch.ones_like(k.values()), k.shape
            ).coalesce()
        return self._K01

    def factorized_logits(self, h: torch.Tensor) -> torch.Tensor:
        """[B, T, D] -> [B, T, V] by summing per-column log-probabilities.

        This is the scoring that matches the column objective. Each column is a
        char_dim-way softmax, so a token's score is the log-probability its own
        byte code receives, summed over the pos_dim columns. A softmax over V
        of these scores renormalises the factorized model onto the vocabulary,
        which is what makes the number comparable to every other arm's.

        Note what is *not* here: nothing in training ever has to compute this.
        The column loss is O(pos_dim * char_dim); only evaluation pays O(V * P).
        """
        y = self.code_logits(h)
        P, C = self.cfg.pos_dim, self.cfg.char_dim
        flat = y.reshape(-1, P, C).log_softmax(dim=-1).reshape(-1, P * C)
        acc = torch.sparse.mm(self.k01, flat.t()).t()    # [BT, V]
        return acc.view(*h.shape[:-1], self.vocab_size)

    def code_targets(self, token_ids: torch.Tensor) -> torch.Tensor:
        """[..] ids -> [.., pos_dim] byte-value labels, one per column.

        The duplex code is one-hot per column, so predicting it is pos_dim
        independent char_dim-way classifications. That is the training signal
        whose cost does not involve V at all.
        """
        return (self.idx[token_ids] % self.cfg.char_dim)

    # -------------------------------------------------------------- costing

    def parameter_report(self) -> dict:
        ana = self.W_ana.numel()
        syn = 0 if self.tied else self.W_syn.numel()
        return {
            "analysis": ana,
            "synthesis": syn,
            "total_token_facing": ana + syn,
            "vocab_size": self.vocab_size,
            "depends_on_vocab": False,
        }


class DenseTokenPath(nn.Module):
    """The control arm: a V x D table and an untied D x V head."""

    def __init__(self, vocab_size: int, d_model: int, tied: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.tied = tied
        self.table = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.table.weight, std=0.02)
        self.head = None if tied else nn.Linear(d_model, vocab_size, bias=False)
        if self.head is not None:
            nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.table(token_ids)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        if self.tied:
            return h @ self.table.weight.t()
        return self.head(h)

    def parameter_report(self) -> dict:
        ana = self.table.weight.numel()
        syn = 0 if self.tied else self.head.weight.numel()
        return {
            "analysis": ana,
            "synthesis": syn,
            "total_token_facing": ana + syn,
            "vocab_size": self.vocab_size,
            "depends_on_vocab": True,
        }


class KroneckerInDensOut(nn.Module):
    """The V5 design as the session commits to it: Kronecker input, dense head.

    This is the arm that isolates what the synthesis side is actually worth. If
    the duplex arm matches this one, the D x V head was redundant.
    """

    def __init__(self, byte_seqs: list[bytes], d_model: int, config: CodecConfig):
        super().__init__()
        self.inp = DuplexEmbedding(byte_seqs, d_model, config, tied=False)
        self.inp.W_syn = None
        self.inp.tied = True  # never used; synthesis comes from the dense head
        self.head = nn.Linear(d_model, len(byte_seqs), bias=False)
        nn.init.normal_(self.head.weight, std=0.02)
        self.vocab_size = len(byte_seqs)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.inp(token_ids)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)

    def parameter_report(self) -> dict:
        ana = self.inp.W_ana.numel()
        syn = self.head.weight.numel()
        return {
            "analysis": ana,
            "synthesis": syn,
            "total_token_facing": ana + syn,
            "vocab_size": self.vocab_size,
            "depends_on_vocab": True,
        }
