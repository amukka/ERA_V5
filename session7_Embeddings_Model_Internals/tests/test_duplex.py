"""Invariants the whole submission rests on.

The experiments measure how well the scheme works. These check that the scheme
is the thing it claims to be, which is a different question and a cheaper one.
Run with:  python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from duplex.codec import DUPLEX, INACTIVE, SHIPPED, CodecConfig, KroneckerCodec
from duplex.embedding import DuplexEmbedding

WORDS = [
    b"the", b"a", b"training", b"trainer",
    "भारत".encode(), "अंतर्राष्ट्रीयकरण".encode(), "अंतर्राष्ट्रीयता".encode(),
    "తెలుగు".encode(), "ಕನ್ನಡ".encode(), "தமிழ்".encode(),
    b"", b"\xff\xfe\x00",
]


class TestCodec(unittest.TestCase):
    def test_duplex_code_is_one_hot_per_column(self):
        """Every column carries exactly one mark, so the code is 32 one-hots.

        This is what makes decoding a well-posed classification rather than a
        general sparse-recovery problem.
        """
        codec = KroneckerCodec(DUPLEX)
        for w in WORDS:
            cols = codec.columns(w)
            self.assertEqual(len(cols), DUPLEX.pos_dim, w)
            self.assertTrue(((cols >= 0) & (cols <= INACTIVE)).all(), w)

    def test_shipped_code_is_not_self_delimiting(self):
        """The shipped codec marks only the columns the bytes reach, so a short
        token's code is a prefix of a longer one's. That is the property the
        duplex codec removes."""
        codec = KroneckerCodec(SHIPPED)
        self.assertEqual(len(codec.columns(b"the")), 3)
        self.assertEqual(len(codec.columns(b"a")), 1)

    def test_decode_inverts_encode(self):
        """A duplex code decodes back to the token's leading bytes exactly."""
        codec = KroneckerCodec(DUPLEX)
        for w in WORDS:
            prefix, digest = codec.decode(codec.dense(w))
            self.assertEqual(prefix, w[: DUPLEX.head_cols], w)
            self.assertEqual(len(digest), DUPLEX.tail_bytes, w)

    def test_truncated_tokens_are_separated_by_the_digest(self):
        """Section 8's silent collision: two Hindi words agreeing on their
        first 32 bytes. The shipped codec fuses them; the duplex codec does not."""
        a = "अंतर्राष्ट्रीयकरण".encode()
        b = "अंतर्राष्ट्रीयता".encode()
        self.assertEqual(a[:32], b[:32], "the premise: they agree on 32 bytes")

        shipped = KroneckerCodec(SHIPPED)
        self.assertTrue((shipped.columns(a) == shipped.columns(b)).all())

        duplex = KroneckerCodec(DUPLEX)
        self.assertFalse((duplex.columns(a) == duplex.columns(b)).all())

    def test_znorm_moments_depend_only_on_the_nonzero_count(self):
        """The identity the sparse fast path relies on: mu and sigma are a
        function of how many cells are marked, never of which ones."""
        codec = KroneckerCodec(DUPLEX)
        for w in WORDS:
            code = codec.dense(w)
            n = len(codec.flat_indices(w))
            mu, sigma = codec.moments(n)
            raw = np.zeros(DUPLEX.code_dim)
            raw[codec.flat_indices(w)] = 1.0 / np.sqrt(n)
            np.testing.assert_allclose(code, (raw - mu) / sigma, atol=1e-9)


class TestDuplexEmbedding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.emb = DuplexEmbedding(WORDS, 64, DUPLEX, tied=False)
        self.codec = KroneckerCodec(DUPLEX)

    def test_forward_equals_the_dense_matrix_product(self):
        """The gather fast path and the textbook code-times-matrix agree."""
        ids = torch.arange(len(WORDS)).unsqueeze(0)
        got = self.emb(ids)[0]
        for i, w in enumerate(WORDS):
            want = torch.from_numpy(self.codec.dense(w)).float() @ self.emb.W_ana
            torch.testing.assert_close(got[i], want, atol=2e-4, rtol=2e-4)

    def test_logits_equal_explicit_scoring_against_every_code(self):
        """The sparse head reproduces an explicit inner product against the
        full code table, which is the definition it stands in for."""
        h = torch.randn(1, 3, 64)
        got = self.emb.logits(h)[0]
        codes = torch.stack(
            [torch.from_numpy(self.codec.dense(w)).float() for w in WORDS]
        )
        want = self.emb.code_logits(h)[0] @ codes.t() * self.emb.logit_scale.exp()
        torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)

    def test_no_parameter_is_sized_by_the_vocabulary(self):
        """The claim that makes a 1M vocabulary free: doubling V changes no
        trainable shape anywhere on the path."""
        small = DuplexEmbedding(WORDS, 64, DUPLEX)
        large = DuplexEmbedding(WORDS * 50, 64, DUPLEX)
        self.assertEqual(
            sum(p.numel() for p in small.parameters()),
            sum(p.numel() for p in large.parameters()),
        )
        self.assertFalse(small.parameter_report()["depends_on_vocab"])

    def test_code_targets_match_the_codec(self):
        """The column labels the loss trains against are the codec's own."""
        ids = torch.arange(len(WORDS))
        got = self.emb.code_targets(ids)
        for i, w in enumerate(WORDS):
            np.testing.assert_array_equal(
                got[i].numpy(), self.codec.columns(w)
            )

    def test_factorized_logits_are_summed_log_probabilities(self):
        """The column-trained arm's score is the log-probability of the token's
        own code, so every score must be non-positive and bounded by zero."""
        h = torch.randn(1, 3, 64)
        scores = self.emb.factorized_logits(h)
        self.assertTrue(bool((scores <= 0).all()))
        self.assertEqual(scores.shape, (1, 3, len(WORDS)))


class TestPosDimBudget(unittest.TestCase):
    def test_wider_window_separates_what_32_bytes_fuses(self):
        """Section 8's control: raising pos_dim is the other way to fix the
        collision, and it works — it just costs parameters in proportion."""
        a = "अंतर्राष्ट्रीयकरण".encode()
        b = "अंतर्राष्ट्रीयता".encode()
        narrow = KroneckerCodec(CodecConfig(char_dim=256, pos_dim=32))
        wide = KroneckerCodec(CodecConfig(char_dim=256, pos_dim=48))
        self.assertEqual(
            narrow.columns(a).tobytes(), narrow.columns(b).tobytes(),
            "at 32 bytes the two words are the same object to the model",
        )
        self.assertNotEqual(
            wide.columns(a).tobytes(), wide.columns(b).tobytes(),
            "at 48 bytes they separate",
        )


if __name__ == "__main__":
    unittest.main()
