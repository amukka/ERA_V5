"""Vocabularies as byte tables.

The Kronecker path never sees a token id as an opaque integer. It sees the
token's UTF-8 bytes. This module is the only place that knows how to get from
a tokenizer's internal token string back to those bytes, which differs by
tokenizer family, and it is also where we label each token with the script it
belongs to so the injectivity audit can report per script.
"""

from __future__ import annotations

import functools
import importlib.util
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

# The Session 2 tokenizer, which is this course's own vocabulary rather than a
# borrowed one. Session 5 tokenized the proxy corpus with it, so it is also the
# vocabulary E3 trains against.
ERA_V5_TOKENIZER = (
    Path(__file__).resolve().parents[2] / "session2_tokenizer" / "build" / "out"
)

# ---------------------------------------------------------------- byte level


@functools.lru_cache(maxsize=1)
def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode, which maps bytes to printable
    codepoints so a BPE vocabulary can be stored as text."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def _bytes_from_bytelevel(token: str) -> bytes:
    dec = _byte_decoder()
    try:
        return bytes(dec[c] for c in token)
    except KeyError:
        return token.encode("utf-8")


def _bytes_from_sentencepiece(token: str) -> bytes:
    # SentencePiece writes a leading word boundary as U+2581.
    return token.replace("▁", " ").encode("utf-8")


# ------------------------------------------------------------------- scripts

_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0900, 0x097F, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Odia"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0400, 0x04FF, "Cyrillic"),
    (0x0370, 0x03FF, "Greek"),
    (0x0590, 0x05FF, "Hebrew"),
    (0x3040, 0x30FF, "Kana"),
    (0x4E00, 0x9FFF, "Han"),
    (0xAC00, 0xD7AF, "Hangul"),
]

INDIC = {
    "Devanagari",
    "Bengali",
    "Gurmukhi",
    "Gujarati",
    "Odia",
    "Tamil",
    "Telugu",
    "Kannada",
    "Malayalam",
    "Sinhala",
}


def script_of(text: str) -> str:
    """The dominant non-ASCII script in a token, or Latin/Other."""
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        if cp < 0x80:
            name = "ASCII"
        else:
            name = "Other"
            for lo, hi, label in _SCRIPT_RANGES:
                if lo <= cp <= hi:
                    name = label
                    break
            if name == "Other" and unicodedata.category(ch).startswith("L"):
                name = "Latin-ext"
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return "Empty"
    non_ascii = {k: v for k, v in counts.items() if k != "ASCII"}
    pool = non_ascii or counts
    return max(pool.items(), key=lambda kv: kv[1])[0]


# -------------------------------------------------------------------- vocab


@dataclass
class Vocab:
    """A tokenizer flattened into the three things the codec needs."""

    name: str
    tokens: list[str]          # display form, for reporting
    byte_seqs: list[bytes]     # what the codec actually encodes
    scripts: list[str]
    tokenizer: Tokenizer | None = None

    def __len__(self) -> int:
        return len(self.byte_seqs)

    def subset(self, ids: list[int]) -> "Vocab":
        return Vocab(
            name=f"{self.name}[{len(ids)}]",
            tokens=[self.tokens[i] for i in ids],
            byte_seqs=[self.byte_seqs[i] for i in ids],
            scripts=[self.scripts[i] for i in ids],
        )


_FAMILY = {
    "gpt2": _bytes_from_bytelevel,
    "xlm-roberta-base": _bytes_from_sentencepiece,
    "google-bert/bert-base-multilingual-cased": lambda t: t.encode("utf-8"),
}


@functools.lru_cache(maxsize=1)
def load_era_v5_tokenizer():
    """The Session 2 byte-level BPE, loaded from its shipped .model file.

    Its `vocab` is already a dict of id -> bytes, which is exactly what the
    codec wants, so no family-specific byte recovery is needed here.
    """
    src = ERA_V5_TOKENIZER / "tokenizer.py"
    spec = importlib.util.spec_from_file_location("era_v5_tokenizer", src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Tokenizer.load(str(ERA_V5_TOKENIZER / "tokenizer.model"))


def load_era_v5_vocab() -> Vocab:
    """The V5 vocabulary as a byte table, in id order."""
    tok = load_era_v5_tokenizer()
    tokens, byte_seqs, scripts = [], [], []
    for i in range(tok.vocab_size):
        raw = tok.vocab[i]
        display = raw.decode("utf-8", errors="replace")
        tokens.append(display)
        byte_seqs.append(raw)
        scripts.append(script_of(display))
    return Vocab(name="era-v5-s2-bpe", tokens=tokens, byte_seqs=byte_seqs,
                 scripts=scripts, tokenizer=None)


def load_vocab(name: str = "gpt2", tokenizer_file: str | None = None) -> Vocab:
    """Load a tokenizer and flatten it into a byte table."""
    if name == "era-v5":
        return load_era_v5_vocab()
    if tokenizer_file:
        tok = Tokenizer.from_file(tokenizer_file)
    else:
        tok = Tokenizer.from_pretrained(name)

    to_bytes = _FAMILY.get(name, _bytes_from_bytelevel)
    vocab = tok.get_vocab()
    ordered = sorted(vocab.items(), key=lambda kv: kv[1])

    tokens, byte_seqs, scripts = [], [], []
    for token, _id in ordered:
        raw = to_bytes(token)
        display = raw.decode("utf-8", errors="replace")
        tokens.append(display)
        byte_seqs.append(raw)
        scripts.append(script_of(display))

    return Vocab(name=name, tokens=tokens, byte_seqs=byte_seqs,
                 scripts=scripts, tokenizer=tok)
