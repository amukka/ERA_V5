"""The corpus, the tokenizer, and the three batch shapes this session needs.

Session 10's data layer handed out ``inputs`` and ``targets`` already shifted
apart.  That is the wrong shape for session 9, because the shift *is* the
deliverable.  Here a batch carries raw ``tokens`` exactly as the assignment's
harness receives them, and every shift happens inside ``src/losses.py`` where
it can be watched.

Three batch shapes, one per failure the session warns about:

``clean``   one document per row, every row the full width, no padding and no
            boundary.  The control: nothing to mask, so masking changes nothing.

``padded``  one document window per row, row lengths drawn from a real spread,
            padded to the widest.  This is deliverable 3 -- the contributing
            token count is no longer B x (T-1).

``packed``  several documents concatenated into one fixed-width row with an EOS
            between them, session 6 style.  There is no padding at all here and
            the loss still needs a mask, which is the point of deliverable 4.
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

# Session 2's tokenizer is 10,000 byte-level BPE ids with no special tokens, so
# this session adds the two it needs.  EOS is a real token the model predicts
# and learns from; PAD is never predicted and never contributes -- it exists so
# that "there is nothing here" is a distinct id rather than a plausible word.
# If padding shared an id with a real token, deliverable 3's bug would still be
# a bug and would no longer be visible.
BPE_VOCAB = 10_000
EOS_ID = BPE_VOCAB          # 10000
PAD_ID = BPE_VOCAB + 1      # 10001
VOCAB_SIZE = BPE_VOCAB + 2  # 10002


def load_tokenizer():
    sys.path.insert(0, str(TOKENIZER_DIR))
    from bpe import Tokenizer  # vendored copy of session 2's shipped tokenizer

    return Tokenizer.load(str(TOKENIZER_DIR / "tokenizer.model"))


class Vocab:
    """Decoding, including the two ids the base tokenizer has never heard of."""

    def __init__(self):
        self.tok = load_tokenizer()

    def piece(self, token_id: int) -> str:
        """The string for exactly one id, as printable text.

        Byte-level BPE ids are byte sequences, not characters, so a single id
        can be half of a UTF-8 codepoint.  ``errors='replace'`` turns that into
        a visible replacement char instead of an exception -- which is honest:
        the model is predicting bytes, and sometimes one id is not a character.
        """
        if token_id == PAD_ID:
            return "<pad>"
        if token_id == EOS_ID:
            return "<eos>"
        return self.tok.decode([int(token_id)])

    def text(self, ids) -> str:
        return "".join(self.piece(int(i)) for i in ids)


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
    """Return ``({split: [np.uint16 array, ...]}, {split: [lane, ...]})``.

    Tokenising the whole slice takes a few seconds, so the result is cached
    next to the slice.  The cache is derived data and is git-ignored.
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
class Seqs:
    """One batch of sequences, *unshifted*.

    tokens  (B, T) int64  token ids as the model receives them, PAD past the end
    valid   (B, T) bool   True where the position holds a real token
    doc_id  (B, T) int64  which document each position came from; -1 on padding.
                          Two adjacent positions with different doc_ids are a
                          document boundary, which is all deliverable 4 needs.
    """

    tokens: torch.Tensor
    valid: torch.Tensor
    doc_id: torch.Tensor

    @property
    def shape(self):
        return tuple(self.tokens.shape)

    def to(self, device) -> "Seqs":
        return Seqs(self.tokens.to(device), self.valid.to(device),
                    self.doc_id.to(device))


def _window(rng, docs, length):
    """A contiguous ``length``-token window from some document long enough."""
    for _ in range(128):
        doc = docs[rng.integers(len(docs))]
        if len(doc) >= length:
            start = int(rng.integers(0, len(doc) - length + 1))
            return np.asarray(doc[start:start + length], dtype=np.int64)
    doc = max(docs, key=len)
    return np.asarray(doc[:length], dtype=np.int64)


def clean_batch(docs, batch_size=4, width=128, seed=0) -> Seqs:
    """Every row one full-width window of a single document.  No padding."""
    rng = np.random.default_rng(seed)
    rows = [_window(rng, docs, width) for _ in range(batch_size)]
    tokens = np.stack(rows)
    return Seqs(
        torch.from_numpy(tokens),
        torch.ones(tokens.shape, dtype=torch.bool),
        torch.from_numpy(
            np.repeat(np.arange(batch_size)[:, None], width, axis=1)),
    )


def padded_batch(docs, batch_size=4, width=128, min_len=24, seed=0) -> Seqs:
    """Rows of different real lengths, padded out to ``width``.

    The lengths are drawn log-uniformly, which is roughly the spread a real
    loader sees once it stops packing.  ``width`` is fixed rather than set to
    the longest row so that the padded fraction is a number this experiment
    controls rather than one the random draw decides.
    """
    rng = np.random.default_rng(seed)
    lo, hi = np.log(min_len), np.log(width)
    lengths = [int(round(float(np.exp(rng.uniform(lo, hi)))))
               for _ in range(batch_size)]
    lengths = [max(min_len, min(width, n)) for n in lengths]
    lengths[0] = width  # at least one row reaches the full width

    tokens = np.full((batch_size, width), PAD_ID, dtype=np.int64)
    valid = np.zeros((batch_size, width), dtype=bool)
    doc_id = np.full((batch_size, width), -1, dtype=np.int64)
    for i, n in enumerate(lengths):
        tokens[i, :n] = _window(rng, docs, n)
        valid[i, :n] = True
        doc_id[i, :n] = i
    return Seqs(torch.from_numpy(tokens), torch.from_numpy(valid),
                torch.from_numpy(doc_id))


def packed_batch(docs, batch_size=2, width=128, n_docs=2, seed=0) -> Seqs:
    """``n_docs`` document fragments packed into each fixed-width row.

    Each fragment ends with EOS and the next one starts immediately after, so
    the row is full: this is session 6's packing, which exists precisely to
    stop paying for padding.  It buys throughput and it plants deliverable 4's
    trap, because the position holding the last token of one document is asked
    to predict the first token of the next.
    """
    rng = np.random.default_rng(seed)
    tokens = np.full((batch_size, width), PAD_ID, dtype=np.int64)
    valid = np.zeros((batch_size, width), dtype=bool)
    doc_id = np.full((batch_size, width), -1, dtype=np.int64)

    for b in range(batch_size):
        # split the row into n_docs fragments of roughly equal size; each
        # fragment is (text ... EOS), so its text is one token shorter.
        edges = np.linspace(0, width, n_docs + 1).astype(int)
        for d in range(n_docs):
            lo, hi = edges[d], edges[d + 1]
            n = hi - lo
            piece = _window(rng, docs, n - 1)
            tokens[b, lo:hi - 1] = piece
            tokens[b, hi - 1] = EOS_ID
            valid[b, lo:hi] = True
            doc_id[b, lo:hi] = b * n_docs + d
    return Seqs(torch.from_numpy(tokens), torch.from_numpy(valid),
                torch.from_numpy(doc_id))
