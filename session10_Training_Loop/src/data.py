"""The corpus, the tokenizer, and the two batch shapes the session needs.

Everything here is deterministic given a seed.  The two batch modes matter:

``fixed``   every sequence is exactly ``T`` tokens, so every micro-batch holds
            the same number of valid tokens.  This is the *control*: section 8's
            bug is invisible here, which is exactly how it hid for two years.

``varlen``  sequence lengths are drawn from a real length distribution, padded
            to the micro-batch maximum and masked.  Micro-batches now hold
            different token counts, and "the average of the averages" starts
            being wrong.

``bucket``  length-bucketed: every row inside one micro-batch shares a single
            drawn length, so there is no padding at all.  This is what real
            loaders do for throughput, and it is also the setting in which the
            bug bites hardest, because bucketing maximises the spread of token
            counts *between* micro-batches while removing it *within* one.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
SLICE = ROOT / "data" / "corpus_slice.jsonl.gz"
CACHE = ROOT / "data" / "tokens_cache.npz"
TOKENIZER_DIR = ROOT / "tokenizer"

# The tokenizer is byte-level BPE with 10,000 ids (session 2).  We append one
# more id for PAD so that padding is never confusable with a real token; the
# loss mask makes it unreachable anyway, but a distinct id means a bug shows up
# as a bug rather than as slightly-wrong text.
BPE_VOCAB = 10_000
PAD_ID = BPE_VOCAB
VOCAB_SIZE = BPE_VOCAB + 1


def load_tokenizer():
    sys.path.insert(0, str(TOKENIZER_DIR))
    from bpe import Tokenizer  # vendored copy of session 2's shipped tokenizer

    return Tokenizer.load(str(TOKENIZER_DIR / "tokenizer.model"))


def _tokenize_corpus():
    tok = load_tokenizer()
    docs = {"train": [], "validation": []}
    lanes = {"train": [], "validation": []}
    with gzip.open(SLICE, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            ids = tok.encode(row["text"])
            if len(ids) < 64:
                continue
            docs[row["split"]].append(np.asarray(ids, dtype=np.uint16))
            lanes[row["split"]].append(row["lane"])
    return docs, lanes


def load_documents(verbose: bool = False):
    """Return ``{split: [np.uint16 array, ...]}`` of tokenised documents.

    Tokenising the whole slice takes about three seconds, so the result is
    cached next to the slice.  The cache is derived data and is git-ignored.
    """
    if CACHE.exists():
        blob = np.load(CACHE, allow_pickle=True)
        docs = {s: list(blob[s]) for s in ("train", "validation")}
        lanes = {s: list(blob[s + "_lanes"]) for s in ("train", "validation")}
    else:
        docs, lanes = _tokenize_corpus()
        np.savez_compressed(
            CACHE,
            train=np.array(docs["train"], dtype=object),
            validation=np.array(docs["validation"], dtype=object),
            train_lanes=np.array(lanes["train"]),
            validation_lanes=np.array(lanes["validation"]),
        )
    if verbose:
        for split in ("train", "validation"):
            n = sum(len(d) for d in docs[split])
            print(f"{split:<11} {len(docs[split]):>5} docs  {n:>9,} tokens")
    return docs, lanes


@dataclass
class Batch:
    """One micro-batch.

    inputs   (B, T) int64   token ids fed to the model, PAD_ID past the end
    targets  (B, T) int64   the next token at each position, PAD_ID where none
    mask     (B, T) bool    True where ``targets`` is a real token to predict
    lengths  (B,)   int64   valid target count per row
    """

    inputs: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor
    lengths: torch.Tensor

    @property
    def n_tokens(self) -> int:
        return int(self.mask.sum())

    def to(self, device) -> "Batch":
        return Batch(
            self.inputs.to(device),
            self.targets.to(device),
            self.mask.to(device),
            self.lengths.to(device),
        )


class Sampler:
    """Draws micro-batches of document windows.

    ``mode='fixed'``   every row is ``max_len`` tokens.
    ``mode='varlen'``  row lengths are drawn log-uniformly in
                       ``[min_len, max_len]``, which is roughly what a real
                       document-packed loader produces once you stop packing.
    ``mode='bucket'``  one log-uniform length per micro-batch, shared by every
                       row in it -- length bucketing, as a real loader does it.
    """

    def __init__(
        self,
        docs,
        batch_size: int,
        max_len: int = 256,
        min_len: int = 48,
        mode: str = "varlen",
        seed: int = 0,
    ):
        self.docs = [d for d in docs if len(d) >= min_len + 1]
        self.batch_size = batch_size
        self.max_len = max_len
        self.min_len = min_len
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def _draw_len(self):
        lo, hi = np.log(self.min_len), np.log(self.max_len)
        length = int(round(float(np.exp(self.rng.uniform(lo, hi)))))
        return max(self.min_len, min(self.max_len, length))

    def _row(self, length=None):
        if length is None:
            length = self.max_len if self.mode == "fixed" else self._draw_len()
        for _ in range(64):
            doc = self.docs[self.rng.integers(len(self.docs))]
            if len(doc) >= length + 1:
                start = int(self.rng.integers(0, len(doc) - length))
                return np.asarray(doc[start : start + length + 1], dtype=np.int64)
        # every document is shorter than the requested window: take the longest
        doc = max(self.docs, key=len)
        return np.asarray(doc[: len(doc)], dtype=np.int64)

    def batch(self) -> Batch:
        shared = self._draw_len() if self.mode == "bucket" else None
        rows = [self._row(shared) for _ in range(self.batch_size)]
        width = max(len(r) - 1 for r in rows)
        inputs = np.full((self.batch_size, width), PAD_ID, dtype=np.int64)
        targets = np.full((self.batch_size, width), PAD_ID, dtype=np.int64)
        mask = np.zeros((self.batch_size, width), dtype=bool)
        lengths = np.zeros(self.batch_size, dtype=np.int64)
        for i, r in enumerate(rows):
            n = len(r) - 1
            inputs[i, :n] = r[:-1]
            targets[i, :n] = r[1:]
            mask[i, :n] = True
            lengths[i] = n
        return Batch(
            torch.from_numpy(inputs),
            torch.from_numpy(targets),
            torch.from_numpy(mask),
            torch.from_numpy(lengths),
        )

    def micro_batches(self, n: int):
        return [self.batch() for _ in range(n)]
