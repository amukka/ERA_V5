"""Unit tests for the invariants the training stream depends on.

These run against synthetic data and need no artifacts, so they fail fast and
locally when a rule is broken -- before a six-phase demonstration has to notice.

The rules under test are the ones that would silently corrupt training rather
than crash it: a loss mask that crosses a packed boundary, attention that leaks
between two samples sharing a bin, a quota that drifts away from the mixture, a
ledger that can be edited without leaving a mark.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common import (canonical_json, hash_obj, stable_permutation,   # noqa: E402
                             stable_unit)
from pipeline.ledger import Ledger, LedgerSet                                # noqa: E402
from pipeline.mixture import MixtureSchedule                                 # noqa: E402
from pipeline.model import TinyCausalLM, masked_loss                         # noqa: E402
from pipeline.packing import (PAD_SEGMENT, attention_4d, best_fit_decreasing,  # noqa: E402
                              bin_hash, build_bin, compare_policies)

SEQ_LEN = 64
EOS, PAD = 50256, 50256


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def fake_samples(n: int, lengths: list[int] | None = None) -> list[dict]:
    rng = np.random.default_rng(0)
    lengths = lengths or [int(x) for x in rng.integers(8, SEQ_LEN - 1, size=n)]
    out = []
    for i, ln in enumerate(lengths):
        out.append({"sample_id": f"shard-a:{i * 100}-{i * 100 + ln}",
                    "shard_id": "shard-a", "doc_id": f"doc{i}",
                    "token_start": i * 100, "token_end": i * 100 + ln,
                    "n_tokens": int(ln), "lane": "wiki"})
    return out


TOKENS = np.arange(1, 100_000, dtype=np.int64) % 50000


def reader(_shard_id: str, start: int, end: int) -> np.ndarray:
    return TOKENS[start:end]


def make_bin(samples: list[dict], bin_id: str = "b0"):
    return build_bin(bin_id, "wiki", samples, reader, SEQ_LEN, EOS, PAD,
                     eos_between=True, reset_position_ids=True)


# --------------------------------------------------------------------------- #
# packing, masks, position ids
# --------------------------------------------------------------------------- #

class TestPacking(unittest.TestCase):

    def setUp(self) -> None:
        self.samples = fake_samples(40)
        self.bins = best_fit_decreasing(self.samples, SEQ_LEN, True, 8)

    def test_no_bin_overflows_the_context_window(self):
        for items in self.bins:
            used = sum(s["n_tokens"] + 1 for s in items)
            self.assertLessEqual(used, SEQ_LEN)

    def test_every_sample_is_placed_exactly_once(self):
        placed = [s["sample_id"] for b in self.bins for s in b]
        self.assertCountEqual(placed, [s["sample_id"] for s in self.samples])
        self.assertEqual(len(placed), len(set(placed)))

    def test_loss_mask_never_crosses_a_sample_boundary(self):
        for items in self.bins:
            b = make_bin(items)
            seg = np.asarray(b["segment_ids"])
            lm = np.asarray(b["loss_mask"])
            idx = np.flatnonzero(lm == 1)
            self.assertNotIn(0, idx, "position 0 has no context to predict from")
            self.assertTrue(np.all(seg[idx] == seg[idx - 1]),
                            "a loss position was predicted from another sample")

    def test_padding_never_bears_loss(self):
        for items in self.bins:
            b = make_bin(items)
            seg = np.asarray(b["segment_ids"])
            lm = np.asarray(b["loss_mask"])
            self.assertEqual(int(lm[seg == PAD_SEGMENT].sum()), 0)

    def test_position_ids_restart_for_each_packed_sample(self):
        for items in self.bins:
            b = make_bin(items)
            pos = np.asarray(b["position_ids"])
            for s in b["samples"]:
                span = pos[s["bin_offset"]:s["bin_end"]]
                np.testing.assert_array_equal(span, np.arange(len(span)))

    def test_eos_separates_packed_samples(self):
        for items in self.bins:
            b = make_bin(items)
            ids = np.asarray(b["input_ids"])
            for s in b["samples"]:
                self.assertEqual(int(ids[s["bin_end"] - 1]), EOS)

    def test_attention_is_block_diagonal_and_causal(self):
        b = make_bin(self.bins[0])
        seg = torch.as_tensor(np.asarray(b["segment_ids"]))
        mask = attention_4d(seg.unsqueeze(0))[0, 0]
        allowed = mask == 0
        same = seg.unsqueeze(1) == seg.unsqueeze(0)
        causal = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool))
        off_diag = ~torch.eye(SEQ_LEN, dtype=torch.bool)
        leaks = (allowed & off_diag) & ~(same & causal)
        self.assertEqual(int(leaks.sum()), 0,
                         "attention connected two samples that share a bin")

    def test_every_row_of_the_attention_mask_has_something_to_attend_to(self):
        b = make_bin(self.bins[0])
        seg = torch.as_tensor(np.asarray(b["segment_ids"]))
        mask = attention_4d(seg.unsqueeze(0))[0, 0]
        self.assertTrue(bool(((mask == 0).sum(dim=-1) > 0).all()),
                        "an all-masked row would make softmax produce NaN")

    def test_bin_hash_is_stable_and_content_addressed(self):
        b1 = make_bin(self.bins[0])
        b2 = make_bin(self.bins[0])
        self.assertEqual(b1["content_hash"], b2["content_hash"])
        mutated = dict(b2)
        ids = list(mutated["input_ids"])
        ids[3] = (ids[3] + 1) % 50000
        mutated["input_ids"] = ids
        self.assertNotEqual(bin_hash(mutated), b1["content_hash"])

    def test_bin_hash_ignores_nothing_that_reaches_the_model(self):
        b = make_bin(self.bins[0])
        for field in ("input_ids", "loss_mask", "segment_ids", "position_ids"):
            mutated = dict(b)
            arr = list(mutated[field])
            arr[-1] = arr[-1] + 1
            mutated[field] = arr
            self.assertNotEqual(bin_hash(mutated), b["content_hash"],
                                f"{field} does not affect the bin hash")

    def test_best_fit_decreasing_beats_pad_only_and_first_fit(self):
        rows = {r["policy"]: r for r in
                compare_policies(self.samples, SEQ_LEN, True)["rows"]}
        self.assertGreater(rows["best_fit_decreasing"]["utilisation"],
                           rows["pad_only"]["utilisation"])
        self.assertGreaterEqual(rows["best_fit_decreasing"]["utilisation"],
                                rows["first_fit"]["utilisation"])
        self.assertLess(rows["best_fit_decreasing"]["bins"], rows["pad_only"]["bins"])

    def test_packing_is_order_independent_for_the_same_sample_set(self):
        shuffled = [self.samples[i] for i in stable_permutation(len(self.samples), 7)]
        a = [[s["sample_id"] for s in b] for b in
             best_fit_decreasing(self.samples, SEQ_LEN, True, 8)]
        b = [[s["sample_id"] for s in b] for b in
             best_fit_decreasing(shuffled, SEQ_LEN, True, 8)]
        self.assertEqual(a, b, "BFD must sort first, so arrival order cannot matter")


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #

class TestMaskedLoss(unittest.TestCase):

    def test_masked_positions_contribute_nothing(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 8, 17, requires_grad=True)
        ids = torch.randint(0, 17, (2, 8))
        mask = torch.zeros(2, 8, dtype=torch.long)
        mask[:, 3:6] = 1
        loss, per_pos, n = masked_loss(logits, ids, mask)
        per_pos = per_pos.detach()
        self.assertEqual(n, 6)
        self.assertEqual(float(per_pos[:, :3].abs().sum()), 0.0)
        self.assertEqual(float(per_pos[:, 6:].abs().sum()), 0.0)
        self.assertAlmostEqual(float(loss), float(per_pos.sum() / 6), places=6)

    def test_gradient_reaches_only_unmasked_targets(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 6, 11, requires_grad=True)
        ids = torch.randint(0, 11, (1, 6))
        mask = torch.zeros(1, 6, dtype=torch.long)
        mask[0, 4] = 1
        loss, _, _ = masked_loss(logits, ids, mask)
        loss.backward()
        grad = logits.grad[0]
        # only the position that predicts token 4 (i.e. index 3) may have gradient
        nonzero = [i for i in range(6) if float(grad[i].abs().sum()) > 0]
        self.assertEqual(nonzero, [3])

    def test_model_respects_the_segment_mask(self):
        """Changing tokens in one packed sample must not move another's logits."""
        torch.manual_seed(0)
        model = TinyCausalLM(vocab_size=64, n_layer=1, n_head=2, n_embd=32,
                             n_positions=16).eval()
        ids = torch.arange(16).unsqueeze(0) % 64
        seg = torch.tensor([[0] * 8 + [1] * 8])
        pos = torch.cat([torch.arange(8), torch.arange(8)]).unsqueeze(0)
        attn = attention_4d(seg)
        with torch.no_grad():
            a = model(ids, pos, attn)
            other = ids.clone()
            other[0, :8] = (other[0, :8] + 7) % 64          # rewrite segment 0 only
            b = model(other, pos, attn)
        self.assertTrue(torch.allclose(a[0, 8:], b[0, 8:], atol=1e-6),
                        "segment 1 saw segment 0 through the attention mask")
        self.assertFalse(torch.allclose(a[0, :8], b[0, :8], atol=1e-6))


# --------------------------------------------------------------------------- #
# mixture and floors
# --------------------------------------------------------------------------- #

STAGES = [
    {"name": "s1", "step_start": 0, "step_end": 10,
     "shares": {"wiki": 0.60, "dialog": 0.25, "code": 0.15},
     "protected_floors": {"code": 0.10}},
    {"name": "s2", "step_start": 10, "step_end": 20,
     "shares": {"wiki": 0.34, "dialog": 0.33, "code": 0.33},
     "protected_floors": {"dialog": 0.20}},
]


class TestMixture(unittest.TestCase):

    def setUp(self) -> None:
        self.sched = MixtureSchedule(STAGES, batch_size=8, tolerance=0.06)

    def test_quota_always_sums_to_the_batch_size(self):
        for step in range(20):
            self.assertEqual(sum(self.sched.quota_for_step(step).values()), 8)

    def test_quota_is_never_negative(self):
        for step in range(20):
            self.assertTrue(all(v >= 0 for v in self.sched.quota_for_step(step).values()))

    def test_protected_floor_is_met_in_whole_bins(self):
        for step in range(20):
            quota = self.sched.quota_for_step(step)
            for lane, need in self.sched.floor_bins(step).items():
                self.assertGreaterEqual(quota.get(lane, 0), need,
                                        f"lane {lane} fell below its floor at step {step}")

    def test_largest_remainder_does_not_drift(self):
        plan = self.sched.compile(20)
        for lane, share in plan["planned_overall_shares"].items():
            target = np.mean([s["shares"].get(lane, 0.0) for s in STAGES])
            self.assertLess(abs(share - target), 0.06, f"{lane} drifted from its target")

    def test_stage_lookup_is_half_open(self):
        self.assertEqual(self.sched.stage_name(9), "s1")
        self.assertEqual(self.sched.stage_name(10), "s2")

    def test_plan_hash_changes_when_the_plan_changes(self):
        other = MixtureSchedule(
            [{**STAGES[0], "shares": {"wiki": 0.5, "dialog": 0.3, "code": 0.2}},
             STAGES[1]], batch_size=8, tolerance=0.06)
        self.assertNotEqual(self.sched.compile(20)["plan_hash"],
                            other.compile(20)["plan_hash"])

    def test_compliance_flags_a_violated_floor(self):
        plan = self.sched.compile(20)
        starved = [{"wiki": 8, "dialog": 0, "code": 0} for _ in range(20)]
        result = self.sched.compliance(starved, plan)
        self.assertFalse(result["floors_respected"])
        self.assertFalse(result["within_tolerance"])


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #

class TestDeterminism(unittest.TestCase):

    def test_manifest_hash_ignores_wall_clock_but_not_content(self):
        from pipeline.manifest import manifest_hash
        base = {"shard_id": "train-wiki-0000", "content_hash": "abc",
                "tokenizer_hash": "def", "num_tokens": 100,
                "created_at": "2026-01-01T00:00:00Z",
                "admission_decided_at": "2026-01-01T00:00:01Z"}
        later = {**base, "created_at": "2027-06-06T12:00:00Z",
                 "admission_decided_at": "2027-06-06T12:00:01Z"}
        self.assertEqual(manifest_hash(base), manifest_hash(later),
                         "rebuilding the same shard must yield the same manifest hash")
        for field, value in (("content_hash", "zzz"), ("tokenizer_hash", "zzz"),
                             ("num_tokens", 101)):
            self.assertNotEqual(manifest_hash({**base, field: value}),
                                manifest_hash(base), f"{field} must affect the hash")

    def test_permutation_depends_only_on_the_key(self):
        a = stable_permutation(50, "seed", "main", "wiki", 0)
        b = stable_permutation(50, "seed", "main", "wiki", 0)
        c = stable_permutation(50, "seed", "fork_b", "wiki", 0)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertCountEqual(a, range(50))

    def test_unit_is_in_range_and_keyed(self):
        for key in ("a", "b", "c"):
            v = stable_unit(key, 1)
            self.assertGreaterEqual(v, 0.0)
            self.assertLess(v, 1.0)
        self.assertNotEqual(stable_unit("a"), stable_unit("b"))

    def test_canonical_json_is_order_insensitive(self):
        self.assertEqual(canonical_json({"b": 1, "a": 2}), canonical_json({"a": 2, "b": 1}))
        self.assertEqual(hash_obj({"x": [1, 2]}), hash_obj({"x": [1, 2]}))


# --------------------------------------------------------------------------- #
# ledgers
# --------------------------------------------------------------------------- #

class TestLedger(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "l.jsonl")
        self.ledger = Ledger(self.path, "test")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_chain_verifies_for_an_untouched_ledger(self):
        for i in range(10):
            self.ledger.append({"i": i})
        self.assertTrue(self.ledger.verify_chain()["ok"])

    def _rewrite(self, mutate) -> None:
        with open(self.path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        mutate(lines)
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def test_editing_an_event_breaks_the_chain(self):
        for i in range(6):
            self.ledger.append({"i": i})
        self._rewrite(lambda ls: ls.__setitem__(2, ls[2].replace('"i":2', '"i":99')))
        result = Ledger(self.path, "test").verify_chain()
        self.assertFalse(result["ok"])
        self.assertEqual(result["at"], 2)

    def test_deleting_an_event_breaks_the_chain(self):
        for i in range(6):
            self.ledger.append({"i": i})
        self._rewrite(lambda ls: ls.__delitem__(3))
        self.assertFalse(Ledger(self.path, "test").verify_chain()["ok"])

    def test_truncate_restores_an_exact_offset(self):
        for i in range(5):
            self.ledger.append({"i": i})
        mark = self.ledger.offset()
        for i in range(5, 12):
            self.ledger.append({"i": i})
        result = self.ledger.truncate_to(mark)
        self.assertEqual(result["events_discarded"], 7)
        self.assertEqual(self.ledger.count, 5)
        self.assertEqual(self.ledger.tail_hash, mark["tail_hash"])
        self.assertTrue(self.ledger.verify_chain()["ok"])
        self.assertEqual([e["i"] for e in self.ledger], list(range(5)))

    def test_appending_after_truncation_continues_the_chain(self):
        for i in range(5):
            self.ledger.append({"i": i})
        mark = self.ledger.offset()
        for i in range(5, 9):
            self.ledger.append({"i": i})
        self.ledger.truncate_to(mark)
        self.ledger.append({"i": 5, "second_attempt": True})
        self.assertTrue(self.ledger.verify_chain()["ok"])
        self.assertEqual(self.ledger.count, 6)

    def test_truncating_past_the_end_is_refused(self):
        self.ledger.append({"i": 0})
        with self.assertRaises(RuntimeError):
            self.ledger.truncate_to({"bytes": 10 ** 9, "count": 99})

    def test_ledger_set_rolls_back_together(self):
        lset = LedgerSet(self.tmp.name, "branch")
        for i in range(4):
            lset.consumption.append({"i": i})
            lset.learning.append({"i": i})
        marks = lset.offsets()
        for i in range(4, 8):
            lset.consumption.append({"i": i})
            lset.learning.append({"i": i})
        lset.truncate_to(marks)
        self.assertEqual(lset.consumption.count, 4)
        self.assertEqual(lset.learning.count, 4)
        self.assertTrue(lset.verify_all()["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
