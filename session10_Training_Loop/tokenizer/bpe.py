# ---------------------------------------------------------------------------
# VENDORED, UNMODIFIED, from session2_tokenizer/build/out/tokenizer.py.
#
# It is copied here rather than imported across sessions so that this session
# is self-contained: `tokenizer/tokenizer.model` beside it is the same frozen
# 10,000-id byte-level BPE that session 2 shipped, and the two together make
# `data/corpus_slice.jsonl.gz` reproducible from a fresh clone of this
# directory alone.
# ---------------------------------------------------------------------------
"""tokenizer.py — self-contained byte-level BPE tokenizer (minbpe-compatible).

This is the SHIPPED tokenizer. It has real ``encode`` and ``decode`` methods and
guarantees a faithful roundtrip:

    tok = Tokenizer.load("tokenizer.model")
    assert tok.decode(tok.encode(text)) == text        # for ANY text

Why it roundtrips losslessly: encoding maps text -> UTF-8 bytes, then only ever
*merges adjacent tokens*. Every token therefore corresponds to an exact byte
string (``vocab[id]``), and decoding just concatenates those byte strings and
UTF-8 decodes. No character is ever dropped — including whitespace.

Pre-tokenization (``\\s*\\S+|\\s+``) attaches leading whitespace to the following
word (GPT-2 style " word"), so merges never cross a word boundary and there are
no free-standing space tokens inflating the count. The concatenation of all
chunks equals the original text, so the roundtrip stays exact.

No third-party dependencies. Runs on plain CPython 3.8+.
"""

import json
import re
from collections import Counter

# Pre-tokenizer: optional leading whitespace + a run of non-whitespace, OR a
# trailing run of whitespace. findall over this covers the whole string exactly
# (lossless), so decode(encode(text)) == text.
PRETOK = re.compile(r"\s*\S+|\s+")


def pretokenize(text):
    return PRETOK.findall(text)


def word_count(text):
    """Assignment 'words' = whitespace-delimited words (== len(text.split()))."""
    return len(text.split())


def _get_stats(ids, counts, weight=1):
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + weight


def _merge(ids, pair, new_id):
    out, i, n = [], 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    """Byte-level BPE. merges: dict (id,id)->new_id. vocab: dict id->bytes."""

    def __init__(self, merges=None):
        self.merges = dict(merges) if merges else {}
        self.vocab = self._build_vocab()
        self._cache = {}

    # ---- vocab ----
    def _build_vocab(self):
        vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    # ---- encode / decode ----
    def _encode_chunk(self, chunk):
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # find the pair whose merge was learned earliest (lowest new_id).
            best_pair, best_rank = None, None
            for a, b in zip(ids, ids[1:]):
                r = self.merges.get((a, b))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, (a, b)
            if best_pair is None:
                break
            ids = _merge(ids, best_pair, best_rank)
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        out = []
        for chunk in pretokenize(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids):
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    # ---- persistence: minbpe .model format ----
    def save(self, path):
        """Write minbpe v1 .model (loadable by minbpe) + a .vocab preview."""
        with open(path, "w", encoding="utf-8") as f:
            f.write("minbpe v1\n")
            f.write("\n")           # pattern line (empty = byte-level basic)
            f.write("0\n")          # number of special tokens
            # merges in id order: each line is "id_a id_b"
            for (p0, p1), _idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
                f.write(f"{p0} {p1}\n")

    @classmethod
    def load(cls, path):
        merges, idx = {}, 256
        with open(path, "r", encoding="utf-8") as f:
            assert f.readline().strip() == "minbpe v1"
            f.readline()                       # pattern (ignored, byte-level)
            n_special = int(f.readline().strip())
            for _ in range(n_special):
                f.readline()                   # no special tokens in this model
            for line in f:
                line = line.strip()
                if not line:
                    continue
                a, b = map(int, line.split())
                merges[(a, b)] = idx
                idx += 1
        return cls(merges)

    # ---- self-describing JSON (used by the widget + humans) ----
    def to_json(self, extra=None):
        obj = {
            "type": "byte-level-bpe",
            "vocab_size": self.vocab_size,
            "base_alphabet": 256,
            "pretokenizer": r"\s*\S+|\s+",
            "roundtrip": "lossless (decode(encode(text)) == text)",
            # merges listed in id order as [id_a, id_b]; new id = 256 + position
            "merges": [[p0, p1] for (p0, p1), _ in sorted(self.merges.items(), key=lambda kv: kv[1])],
        }
        if extra:
            obj.update(extra)
        return obj

    @classmethod
    def from_json(cls, obj):
        merges, idx = {}, 256
        for p0, p1 in obj["merges"]:
            merges[(p0, p1)] = idx
            idx += 1
        return cls(merges)

    def vocab_tokens(self):
        """Human-readable list of every token (base bytes + merges), id order."""
        out = []
        for i in range(self.vocab_size):
            b = self.vocab[i]
            try:
                s = b.decode("utf-8")
            except UnicodeDecodeError:
                s = "<0x%s>" % b.hex()  # partial-byte token (mid-codepoint)
            out.append(s)
        return out


# --------------------------------------------------------------------------
# Training helpers (used offline by train.py; not needed to run the tokenizer)
# --------------------------------------------------------------------------
def train_language(text, max_merges):
    """Learn up to max_merges byte-level merges on one corpus.

    Returns a list of operand pairs (bytesA, bytesB) in learn order. Operands are
    byte STRINGS so the lists from different languages can be combined afterwards
    into one consistent id space (see combine())."""
    freq = Counter(pretokenize(text))
    words = [list(chunk.encode("utf-8")) for chunk in freq]
    wfreq = [freq[chunk] for chunk in freq]

    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    ops = []
    for _ in range(max_merges):
        counts = {}
        for ids, c in zip(words, wfreq):
            _get_stats(ids, counts, c)
        if not counts:
            break
        # max count; deterministic tie-break by the concatenated bytes
        best = max(counts, key=lambda p: (counts[p], vocab[p[0]] + vocab[p[1]]))
        if counts[best] < 1:
            break
        ops.append((vocab[best[0]], vocab[best[1]]))
        vocab[next_id] = vocab[best[0]] + vocab[best[1]]
        words = [_merge(ids, best, next_id) for ids in words]
        next_id += 1
    return ops


def combine(op_lists):
    """Merge several ordered operand-pair lists into ONE Tokenizer.

    op_lists: iterable of lists of (bytesA, bytesB). Concatenated in order and
    de-duplicated (first occurrence wins). Because each language's list is a
    learn-ordered prefix, every operand is either a base byte or the result of an
    earlier merge, so the combined id space stays consistent."""
    bytes_to_id = {bytes([i]): i for i in range(256)}
    merges = {}
    next_id = 256
    for ops in op_lists:
        for a_bytes, b_bytes in ops:
            merged = a_bytes + b_bytes
            if merged in bytes_to_id:
                continue  # token already exists (shared across languages)
            ia = bytes_to_id.get(a_bytes)
            ib = bytes_to_id.get(b_bytes)
            if ia is None or ib is None:
                continue  # prerequisite missing (shouldn't happen); skip safely
            merges[(ia, ib)] = next_id
            bytes_to_id[merged] = next_id
            next_id += 1
    return Tokenizer(merges)


if __name__ == "__main__":
    # tiny smoke test
    t = train_language("the cat sat on the mat. the cat ran.", 20)
    tok = combine([t])
    for s in ["the cat", "India's population is 1,428,627,663.", "भारत", "\n\ttabs "]:
        assert tok.decode(tok.encode(s)) == s, s
    print("smoke OK  vocab_size=", tok.vocab_size)
