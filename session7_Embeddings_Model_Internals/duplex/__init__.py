"""Duplex Kronecker embeddings — a byte code that runs in both directions.

Session 7's released Kronecker embedding is a one-way street: a token's bytes
become a fixed sparse code, and a single learned projection turns that code into
a vector. It is forward deterministic and nothing comes back. The assignment
asks what a reverse would look like, and what it would buy.

This package is the answer. It changes the codec just enough to make it
invertible — a 257th "past the end" row so every code is exactly one-hot per
column, and a digest in the last few columns so truncation stops fusing tokens
silently — and then runs the same code backwards to score the vocabulary. The
output head becomes a fixed sparse matrix of where each token's bytes fall,
which holds no trainable weights at all.

    codec.py      the two codecs: SHIPPED (Session 7's) and DUPLEX (proposed)
    vocab.py      tokenizers flattened to byte tables, labelled by script
    embedding.py  the analysis and synthesis paths, plus the dense controls
    model.py      a small GPT whose front and back doors are swappable
"""

from .codec import DUPLEX, SHIPPED, CodecConfig, KroneckerCodec
from .embedding import DenseTokenPath, DuplexEmbedding, KroneckerInDensOut
from .model import ModelConfig, TinyGPT

__all__ = [
    "DUPLEX",
    "SHIPPED",
    "CodecConfig",
    "KroneckerCodec",
    "DenseTokenPath",
    "DuplexEmbedding",
    "KroneckerInDensOut",
    "ModelConfig",
    "TinyGPT",
]
