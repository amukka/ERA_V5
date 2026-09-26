"""The Kronecker codec, in two versions.

`SHIPPED` is the codec Session 7 released: a 256 x 32 grid, one cell marked per
byte position, scaled by 1/sqrt(L) and z-normalised. It is what Section 7 of the
session describes and what Section 8 warns about.

`DUPLEX` is the version this submission proposes. It changes three things, and
each change exists to buy back a property the shipped codec does not have:

1. A 257th row, INACTIVE, is marked in every column the token's bytes do not
   reach. Every code therefore has exactly one nonzero per column, always 32 of
   them, at a constant scale. The code becomes a clean concatenation of 32
   one-hot vectors, which makes it *self-delimiting*: the length of the token is
   recoverable from the code rather than being lost.

2. The last `tail_bytes` columns hold a digest of the token's entire byte
   string rather than more prefix bytes. This is what removes the silent
   truncation collision of Section 8: two tokens that agree on their first 28
   bytes are separated by their digests instead of being fused forever.

3. Both are consequences of the real goal, which is that the code must be
   recoverable from the embedding. A code that is exactly one-hot per column is
   a 32-way, 257-class classification problem, and that is a decodable object.

Flat index layout is `column * char_dim + row`, so a flat code reshapes to
`[pos_dim, char_dim]` and a per-column argmax reads the bytes straight off. The
shipped module used the other order; the difference is a fixed permutation of
the 8192 axis, which the learned projection absorbs with no effect on anything
measured here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

INACTIVE = 256  # the extra row, meaning "this column is past the end of the token"


@dataclass(frozen=True)
class CodecConfig:
    char_dim: int = 256
    pos_dim: int = 32
    tail_bytes: int = 0
    self_delimiting: bool = False
    normalize: bool = True

    @property
    def head_cols(self) -> int:
        return self.pos_dim - self.tail_bytes

    @property
    def code_dim(self) -> int:
        return self.char_dim * self.pos_dim

    @property
    def label(self) -> str:
        if self.tail_bytes or self.self_delimiting:
            return f"duplex(P={self.pos_dim},T={self.tail_bytes})"
        return f"shipped(P={self.pos_dim})"


SHIPPED = CodecConfig(char_dim=256, pos_dim=32, tail_bytes=0, self_delimiting=False)
DUPLEX = CodecConfig(char_dim=257, pos_dim=32, tail_bytes=4, self_delimiting=True)


class KroneckerCodec:
    """Turns byte strings into fixed sparse codes, and codes back into bytes."""

    def __init__(self, config: CodecConfig = DUPLEX):
        self.cfg = config

    # ------------------------------------------------------------- encoding

    def columns(self, byte_seq: bytes) -> np.ndarray:
        """The row marked in each column, as an int array.

        For the shipped codec this is shorter than pos_dim when the token is
        short: only the columns the bytes reach are marked at all. For the
        duplex codec it is always exactly pos_dim long, because unreached
        columns are marked INACTIVE.
        """
        cfg = self.cfg
        if not cfg.self_delimiting and not cfg.tail_bytes:
            L = min(len(byte_seq), cfg.pos_dim)
            return np.frombuffer(byte_seq[:L], dtype=np.uint8).astype(np.int64)

        head = byte_seq[: cfg.head_cols]
        rows = np.full(cfg.pos_dim, INACTIVE, dtype=np.int64)
        if head:
            rows[: len(head)] = np.frombuffer(head, dtype=np.uint8)
        if cfg.tail_bytes:
            digest = hashlib.blake2b(byte_seq, digest_size=cfg.tail_bytes).digest()
            rows[cfg.head_cols :] = np.frombuffer(digest, dtype=np.uint8)
        return rows

    def flat_indices(self, byte_seq: bytes) -> np.ndarray:
        """Indices into the flattened code vector that carry a nonzero."""
        rows = self.columns(byte_seq)
        cols = np.arange(len(rows), dtype=np.int64)
        return cols * self.cfg.char_dim + rows

    def moments(self, n: int | np.ndarray | torch.Tensor):
        """Mean and standard deviation of a raw code with `n` nonzeros.

        The raw code has n entries equal to 1/sqrt(n) and the rest zero, so its
        sum is sqrt(n) and its sum of squares is exactly 1 regardless of n.
        Both moments therefore depend only on n, never on which cells were
        marked. That is the fact that keeps z-normalisation compatible with
        sparse gathering: see `DuplexEmbedding`.
        """
        N = self.cfg.code_dim
        if isinstance(n, torch.Tensor):
            n = n.to(torch.float32)
            mu = torch.sqrt(n) / N
            sigma = torch.sqrt(torch.clamp(1.0 / N - mu * mu, min=1e-12))
        else:
            n = np.asarray(n, dtype=np.float64)
            mu = np.sqrt(n) / N
            sigma = np.sqrt(np.maximum(1.0 / N - mu * mu, 1e-12))
        return mu, sigma

    def dense(self, byte_seq: bytes) -> np.ndarray:
        """The full code vector. Used for visualisation and for the audits;
        never used on a hot path, where the sparse form is the point."""
        idx = self.flat_indices(byte_seq)
        code = np.zeros(self.cfg.code_dim, dtype=np.float64)
        n = len(idx)
        if n == 0:
            return code
        code[idx] = 1.0 / np.sqrt(n)
        if self.cfg.normalize:
            mu, sigma = self.moments(n)
            code = (code - mu) / sigma
        return code

    # ------------------------------------------------------------- decoding

    def decode_columns(self, rows: np.ndarray) -> tuple[bytes, bytes]:
        """Split a decoded column assignment into (prefix bytes, digest)."""
        cfg = self.cfg
        head = rows[: cfg.head_cols]
        stop = np.nonzero(head == INACTIVE)[0]
        end = int(stop[0]) if len(stop) else len(head)
        prefix = bytes(int(v) for v in head[:end])
        digest = bytes(int(v) for v in rows[cfg.head_cols :]) if cfg.tail_bytes else b""
        return prefix, digest

    def decode(self, code: np.ndarray) -> tuple[bytes, bytes]:
        """Invert a code vector back to (prefix bytes, digest).

        z-normalisation is affine and monotone, so it does not have to be
        undone: the argmax of each column is unchanged by it.
        """
        grid = np.asarray(code).reshape(self.cfg.pos_dim, self.cfg.char_dim)
        return self.decode_columns(grid.argmax(axis=1))

    # ---------------------------------------------------------- vocab table

    def build_table(self, byte_seqs: list[bytes], device="cpu"):
        """Pack a whole vocabulary into gather-ready tensors.

        Returns `idx [V, pos_dim]` flat code indices, `mask [V, pos_dim]`
        marking real entries, and `n [V]` the nonzero count. These three
        tensors are the entire replacement for both the V x D embedding table
        and the D x V output head, and none of them holds a trainable weight.
        """
        V, P = len(byte_seqs), self.cfg.pos_dim
        idx = np.zeros((V, P), dtype=np.int64)
        mask = np.zeros((V, P), dtype=bool)
        for i, seq in enumerate(byte_seqs):
            f = self.flat_indices(seq)
            idx[i, : len(f)] = f
            mask[i, : len(f)] = True
        n = mask.sum(axis=1)
        n[n == 0] = 1  # empty tokens: avoid a divide by zero, code stays all-zero
        return (
            torch.from_numpy(idx).to(device),
            torch.from_numpy(mask).to(device),
            torch.from_numpy(n).to(device),
        )
