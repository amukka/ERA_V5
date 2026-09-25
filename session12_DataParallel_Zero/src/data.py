"""Bytes in, batches out, cut so that no two ranks ever see the same tokens.

The corpus is 538 KB of the mixture built in session 6 -- wikitext, source code
and dialogue, in equal turns -- read as raw UTF-8 bytes.  A byte is a token, so
the vocabulary is 256 and there is no tokenizer to carry between sessions.  That
keeps this directory self-contained and it keeps the embedding small, which
matters here: an embedding table that dominated the parameter count would make
the sharding results a statement about one big matrix rather than about a model.

``Batcher.for_rank`` is the part that makes the run data-parallel.  Step ``s``
draws one contiguous block of ``world_size * micro_batch`` sequences and hands
rank ``r`` the ``r``-th slice of it.  Every rank therefore sees different text,
every step, and the union of what the ranks saw in step ``s`` is exactly the
batch a single machine would have drawn -- which is what experiment 2 checks.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
VOCAB_SIZE = 256


def load_bytes(split: str = "train") -> np.ndarray:
    name = {"train": "corpus.txt", "valid": "valid.txt"}[split]
    raw = (ROOT / "data" / name).read_bytes()
    return np.frombuffer(raw, dtype=np.uint8)


class Batcher:
    """Deterministic sequence sampler shared by every rank and every engine."""

    def __init__(self, seq_len: int = 128, split: str = "train", seed: int = 1234):
        self.tokens = load_bytes(split)
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        # one fixed permutation of legal starting offsets, walked in order, so
        # that "step 7, rank 3" names the same text in every run of every engine
        n_starts = (len(self.tokens) - 1) // seq_len
        self.starts = self.rng.permutation(n_starts) * seq_len
        self.cursor = 0

    def _sequences(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = [self.starts[(self.cursor + i) % len(self.starts)] for i in range(n)]
        self.cursor += n
        x = np.stack([self.tokens[i:i + self.seq_len] for i in idx])
        y = np.stack([self.tokens[i + 1:i + 1 + self.seq_len] for i in idx])
        return (torch.from_numpy(x.astype(np.int64)),
                torch.from_numpy(y.astype(np.int64)))

    def global_batch(self, world_size: int, micro_batch: int):
        """The whole step's batch, as a single machine would have drawn it."""
        return self._sequences(world_size * micro_batch)

    @staticmethod
    def for_rank(batch, rank: int, world_size: int, micro_batch: int):
        x, y = batch
        lo, hi = rank * micro_batch, (rank + 1) * micro_batch
        return x[lo:hi], y[lo:hi]
