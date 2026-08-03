"""Tests that read `submission_artifacts/` and check it against itself.

`run_demo.py` runs these as its second-to-last phase, so they inspect the run
that just happened. They also stand alone: point them at any artifact directory
produced by any earlier run and they will re-derive the same conclusions.

They skip cleanly when there are no artifacts yet, which keeps `python -m
unittest discover -s tests` useful on a fresh clone.

The distinction from `test_units.py` matters: those tests check that the code
*can* be correct; these check that the run that produced these artifacts *was*.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common import file_sha256, hash_obj, load_config, read_json  # noqa: E402
from pipeline.ledger import LedgerSet                                     # noqa: E402
from pipeline.manifest import verify_manifest                             # noqa: E402
from pipeline.packing import PAD_SEGMENT, build_bin                       # noqa: E402

CFG = load_config()
ART = CFG.resolve("artifacts_dir")
HAVE_ARTIFACTS = os.path.exists(os.path.join(ART, "manifests", "shard_index.json"))
needs_artifacts = unittest.skipUnless(
    HAVE_ARTIFACTS, "no submission_artifacts yet -- run `python run_demo.py` first")


def art(*parts: str) -> str:
    return os.path.join(ART, *parts)


@needs_artifacts
class TestShardsAndManifests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = read_json(art("manifests", "shard_index.json"))
        cls.manifests = [read_json(art("manifests", f"{s['shard_id']}.json"))
                         for s in cls.index["shards"]]

    def test_every_shard_matches_its_content_hash(self):
        for m in self.manifests:
            ok, detail = verify_manifest(m, CFG.resolve("shards_dir"))
            self.assertTrue(ok, f"{m['shard_id']} failed verification: {detail}")

    def test_shard_files_are_read_only(self):
        for m in self.manifests:
            path = os.path.join(CFG.resolve("shards_dir"), m["path"])
            self.assertFalse(os.access(path, os.W_OK), f"{m['shard_id']} is writable")

    def test_one_tokenizer_hash_across_every_shard(self):
        freeze = read_json(os.path.join(ROOT, "tokenizer", "tokenizer_freeze.json"))
        self.assertEqual({m["tokenizer_hash"] for m in self.manifests},
                         {freeze["tokenizer_hash"]})

    def test_frozen_tokenizer_file_still_hashes_to_the_frozen_value(self):
        freeze = read_json(os.path.join(ROOT, "tokenizer", "tokenizer_freeze.json"))
        self.assertEqual(file_sha256(os.path.join(ROOT, "tokenizer", "tokenizer.json")),
                         freeze["tokenizer_hash"])

    def test_index_hash_is_reproducible(self):
        self.assertEqual(hash_obj(self.index["shards"]), self.index["index_hash"])

    def test_manifest_self_hash_is_reproducible(self):
        from pipeline.manifest import manifest_hash
        for m in self.manifests:
            self.assertEqual(manifest_hash(m), m["manifest_hash"], m["shard_id"])

    def test_eval_shards_are_never_train(self):
        for m in self.manifests:
            if m["split"] == "eval":
                self.assertTrue(m["never_train"])
                self.assertEqual(m["permission"], "never_train")
                self.assertFalse(m["admitted"])

    def test_validation_shards_are_read_only_for_evaluation(self):
        for m in self.manifests:
            if m["split"] == "validation":
                self.assertEqual(m["permission"], "eval_read_only")
                self.assertFalse(m["admitted"])

    def test_every_admitted_shard_carries_full_lineage(self):
        for m in self.manifests:
            if m["admitted"]:
                self.assertTrue(m["cleaning_pipeline_hash"])
                self.assertEqual(m["contamination_status"], "clean")
                self.assertEqual(m["dedup_status"], "exact_dedup_pass")
                self.assertTrue(m["source"] and m["license"])

    def test_contaminated_shards_were_blocked(self):
        contam = read_json(art("reports", "contamination.json"))
        self.assertTrue(contam["contaminated"],
                        "the planted evaluation leak was never detected")
        for sid in contam["contaminated"]:
            m = next(x for x in self.manifests if x["shard_id"] == sid)
            self.assertFalse(m["admitted"])
            self.assertIn("contamination:eval_overlap_detected", m["block_reasons"])


@needs_artifacts
class TestLedgers(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.branch = CFG.run.branch_id
        cls.ledgers = LedgerSet(CFG.resolve("ledgers_dir"), cls.branch)
        cls.consumption = cls.ledgers.consumption.read_all()
        cls.learning = cls.ledgers.learning.read_all()
        cls.opus = cls.ledgers.opus.read_all()
        cls.micro = [e for e in cls.consumption if e.get("kind") == "microbatch"]
        cls.batches = [e for e in cls.consumption if e.get("kind") == "batch"]

    def test_every_ledger_chain_verifies(self):
        result = self.ledgers.verify_all()
        self.assertTrue(result["ok"], result)

    def test_every_step_appears_exactly_once(self):
        steps = [e["global_step"] for e in self.batches]
        self.assertEqual(sorted(steps), list(range(int(CFG.run.total_steps))))
        self.assertEqual(steps, sorted(steps), "steps are not monotonic")

    def test_microbatch_events_carry_the_required_fields(self):
        need = ("run_id", "branch_id", "global_step", "checkpoint_id", "rank",
                "microbatch_id", "packed_sample_ids", "shard_ids", "token_spans",
                "loss_mask_hash", "attention_policy", "position_policy",
                "mixture_lane", "curriculum_stage", "tokenizer_version",
                "dataloader_version", "opus_decision_ids")
        self.assertTrue(self.micro)
        for e in self.micro:
            for k in need:
                self.assertIn(k, e, f"{e['microbatch_id']} is missing {k}")

    def test_token_spans_are_well_formed(self):
        for e in self.micro:
            for sp in e["token_spans"]:
                self.assertLess(sp["token_start"], sp["token_end"])
                self.assertGreaterEqual(sp["bin_offset"], 0)

    def test_learning_events_attribute_loss_to_shards(self):
        train = [e for e in self.learning if e.get("kind") != "validation"]
        self.assertTrue(train)
        for e in train:
            self.assertTrue(e["shard_losses"])
            self.assertGreater(e["loss_tokens"], 0)
            consumed = {sp["shard_id"] for m in self.micro
                        if m["global_step"] == e["global_step"]
                        for sp in m["token_spans"]}
            self.assertTrue({r["shard_id"] for r in e["shard_losses"]} <= consumed)

    def test_validation_events_are_never_gradient_bearing(self):
        val = [e for e in self.learning if e.get("kind") == "validation"]
        self.assertTrue(val, "validation never ran")
        for e in val:
            self.assertFalse(e["gradient_bearing"])

    def test_opus_recorded_accepted_rejected_and_deferred(self):
        statuses = {e["status"] for e in self.opus}
        self.assertTrue({"accepted", "rejected", "deferred"} <= statuses, statuses)
        self.assertTrue(any(e.get("protected_floor_override") for e in self.opus),
                        "no protected-floor override was ever exercised")

    def test_rejected_candidates_keep_their_reasons_and_features(self):
        rejected = [e for e in self.opus if e["status"] == "rejected"]
        self.assertTrue(rejected)
        for e in rejected:
            self.assertTrue(e["reason"])
            self.assertTrue(e["features"])
            self.assertTrue(e["shard_id"])

    def test_loss_generally_improved(self):
        train = sorted((e for e in self.learning if e.get("kind") != "validation"),
                       key=lambda e: e["global_step"])
        first = np.mean([e["loss"] for e in train[:5]])
        last = np.mean([e["loss"] for e in train[-5:]])
        self.assertLess(last, first, "the model did not learn anything at all")


@needs_artifacts
class TestFirewall(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.fw = read_json(art("reports", "firewall.json"))
        cls.index = read_json(art("manifests", "shard_index.json"))
        cls.ledgers = LedgerSet(CFG.resolve("ledgers_dir"), CFG.run.branch_id)
        cls.consumption = cls.ledgers.consumption.read_all()

    def test_no_attempt_to_train_on_protected_data_succeeded(self):
        self.assertEqual(self.fw["drill"]["leaked"], [])
        self.assertTrue(self.fw["drill"]["all_correct"])

    def test_forged_manifest_was_blocked(self):
        self.assertTrue(self.fw["drill"]["forged_manifest"]["blocked"])

    def test_no_protected_shard_reached_a_loss_bearing_batch(self):
        permitted = {s["shard_id"] for s in self.index["shards"] if s["admitted"]}
        for e in self.consumption:
            if e.get("kind") != "microbatch":
                continue
            for sid in e["shard_ids"]:
                self.assertIn(sid, permitted, f"step {e['global_step']} used {sid}")

    def test_consumption_audit_is_clean(self):
        audit = self.fw["consumption_audit"]
        self.assertTrue(audit["clean"], audit["violations"])
        self.assertEqual(audit["eval_shards_consumed"], [])
        self.assertEqual(audit["validation_shards_consumed"], [])


@needs_artifacts
class TestRecovery(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.rec = read_json(art("reports", "recovery.json"))

    def test_resume_landed_on_the_expected_batch(self):
        crash = self.rec["crash"]
        self.assertTrue(crash["batch_id_match"])
        self.assertTrue(crash["batch_hash_match"])
        self.assertEqual(crash["expected_next_batch_hash"],
                         crash["resumed_next_batch_hash"])

    def test_resume_discarded_the_post_checkpoint_events(self):
        self.assertGreater(self.rec["crash"]["events_discarded"], 0)

    def test_no_batch_was_skipped_or_repeated(self):
        cov = self.rec["coverage"]
        self.assertEqual(cov["duplicated_steps"], [])
        self.assertEqual(cov["missing_steps"], [])
        self.assertTrue(cov["exactly_once"])
        self.assertTrue(cov["monotonic"])

    def test_replay_from_the_ledger_matched(self):
        self.assertTrue(self.rec["replay_ledger"]["all_match"])
        self.assertEqual(self.rec["replay_ledger"]["mismatches"], [])

    def test_replay_by_replanning_matched(self):
        self.assertTrue(self.rec["replay_replan"]["all_match"])
        self.assertEqual(self.rec["replay_replan"]["mismatches"], [])

    def test_every_checkpoint_verifies_and_carries_a_data_position(self):
        self.assertTrue(self.rec["checkpoints"])
        for c in self.rec["checkpoints"]:
            self.assertTrue(c["ok"], c)
            self.assertTrue(c["has_ledger_offsets"])
            self.assertTrue(c["has_planner_state"])

    def test_fork_diverged_from_the_parent_stream(self):
        fork = self.rec["fork"]
        self.assertTrue(fork["diverged"])
        self.assertEqual(fork["divergence_step"], fork["from_step"])
        self.assertEqual(fork["identical_steps"], [])
        self.assertEqual(fork["parent_checkpoint_id"], fork["restored_checkpoint_id"])

    def test_fork_ledger_is_independent_and_intact(self):
        fork = LedgerSet(CFG.resolve("ledgers_dir"), self.rec["fork"]["branch_id"])
        self.assertTrue(fork.verify_all()["ok"])
        created = [e for e in fork.consumption.read_all()
                   if e.get("kind") == "branch_created"]
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["parent_branch_id"], CFG.run.branch_id)


@needs_artifacts
class TestReplayReconstruction(unittest.TestCase):
    """Rebuild bins from the ledger here too, independently of run_demo."""

    def test_bins_rebuild_from_shard_bytes_to_the_recorded_hash(self):
        from pipeline.manifest import read_shard_tokens
        shards_root = CFG.resolve("shards_dir")
        index = read_json(art("manifests", "shard_index.json"))
        by_id = {s["shard_id"]: read_json(art("manifests", f"{s['shard_id']}.json"))
                 for s in index["shards"]}
        cache: dict[str, np.ndarray] = {}

        def reader(sid: str, a: int, b: int) -> np.ndarray:
            if sid not in cache:
                cache[sid] = read_shard_tokens(by_id[sid], shards_root)
            return cache[sid][a:b]

        freeze = read_json(os.path.join(ROOT, "tokenizer", "tokenizer_freeze.json"))
        ledgers = LedgerSet(CFG.resolve("ledgers_dir"), CFG.run.branch_id)
        micro = [e for e in ledgers.consumption.read_all()
                 if e.get("kind") == "microbatch"]
        self.assertTrue(micro)
        checked = 0
        for e in micro[::max(1, len(micro) // 25)][:25]:
            placed = [{"sample_id": f"{s['shard_id']}:{s['token_start']}-{s['token_end']}",
                       "shard_id": s["shard_id"], "doc_id": s["doc_id"],
                       "token_start": s["token_start"], "token_end": s["token_end"],
                       "n_tokens": s["token_end"] - s["token_start"]}
                      for s in sorted(e["token_spans"], key=lambda s: s["bin_offset"])]
            b = build_bin(e["microbatch_id"], e["mixture_lane"], placed, reader,
                          int(CFG.packing.seq_len), int(freeze["eos_token_id"]),
                          int(freeze["pad_token_id"]),
                          bool(CFG.packing.eos_between_samples),
                          bool(CFG.packing.reset_position_ids))
            self.assertEqual(b["content_hash"], e["bin_hash"], e["microbatch_id"])
            seg = np.asarray(b["segment_ids"])
            lm = np.asarray(b["loss_mask"])
            idx = np.flatnonzero(lm == 1)
            self.assertTrue(np.all(seg[idx] == seg[idx - 1]))
            self.assertEqual(int(lm[seg == PAD_SEGMENT].sum()), 0)
            checked += 1
        self.assertGreaterEqual(checked, 5)


@needs_artifacts
class TestEvidenceAndPerformance(unittest.TestCase):

    def test_performance_numbers_reconstruct_from_the_ledger(self):
        from pipeline.performance import packing_efficiency
        perf = read_json(art("performance.json"))
        ledgers = LedgerSet(CFG.resolve("ledgers_dir"), CFG.run.branch_id)
        recomputed = packing_efficiency(ledgers.consumption.read_all(),
                                        int(CFG.packing.seq_len), CFG.run.branch_id)
        self.assertAlmostEqual(recomputed["packing_utilisation"],
                               perf["packing"]["packing_utilisation"], places=9)
        self.assertAlmostEqual(recomputed["loss_utilisation"],
                               perf["packing"]["loss_utilisation"], places=9)

    def test_useful_tokens_are_a_strict_subset_of_raw_tokens(self):
        perf = read_json(art("performance.json"))
        h = perf["headline"]
        self.assertGreater(h["useful_tokens_per_s"], 0)
        self.assertLess(h["useful_tokens_per_s"], h["raw_tokens_per_s"])
        self.assertLess(h["useful_token_fraction"], 1.0)

    def test_chosen_packing_policy_beats_pad_only(self):
        perf = read_json(art("performance.json"))
        self.assertGreater(perf["packing"]["utilisation_vs_pad_only"], 0)

    def test_mixture_compliance_within_tolerance(self):
        comp = read_json(art("reports", "mixture_compliance.json"))
        self.assertTrue(comp["within_tolerance"], comp["max_deviation"])
        self.assertTrue(comp["floors_respected"])

    def test_shard_report_cards_are_reproducible(self):
        cards = read_json(art("reports", "shard_report_cards.json"))
        self.assertEqual(hash_obj(cards["cards"]), cards["report_hash"])
        self.assertGreater(cards["shard_count"], 0)
        self.assertTrue(cards["feedback_for_next_corpus"])

    def test_token_trace_carries_provenance(self):
        trace = read_json(art("reports", "token_perplexity.json"))
        self.assertGreater(trace["traced_tokens"], 0)
        self.assertTrue(trace["hardest_tokens"])
        for h in trace["hardest_tokens"]:
            self.assertTrue(h["shard_id"])
            self.assertTrue(h["doc_id"])

    @unittest.skipUnless(os.path.exists(os.path.join(ART, "evidence.json")),
                         "evidence.json is written after the tests run")
    def test_evidence_bundle_agrees_with_itself(self):
        bundle = read_json(art("evidence.json"))
        self.assertEqual(hash_obj(bundle["requirements"]), bundle["evidence_hash"])
        failed = [r["id"] for r in bundle["requirements"] if r["result"] == "FAIL"]
        self.assertEqual(failed, [], f"failed requirements: {failed}")

    @unittest.skipUnless(os.path.exists(os.path.join(ART, "evidence.md")),
                         "evidence.md is written after the tests run")
    def test_evidence_markdown_was_actually_written(self):
        with open(art("evidence.md"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertGreater(len(text), 2000)
        for heading in ("# Evidence Bundle", "## Required summary",
                        "## Scored areas", "## Headline numbers", "## Every check"):
            self.assertIn(heading, text)
        for row in ("Tokenizer integrity", "Evaluation firewall", "Packing correctness",
                    "Mixture compliance", "OPUS audit trail", "Crash recovery",
                    "Replay", "Learning trace", "Throughput"):
            self.assertIn(row, text)
        self.assertNotIn("FAIL", text.split("## Every check")[0])

    def test_no_artifact_escaped_into_the_project_root(self):
        """A write with a shadowed path variable once landed here. Never again."""
        for name in ("evidence.md", "evidence.json", "performance.json",
                     "run.log", "run_events.jsonl"):
            self.assertFalse(os.path.exists(os.path.join(ROOT, name)),
                             f"{name} was written to the project root, not to "
                             f"submission_artifacts/")


if __name__ == "__main__":
    unittest.main(verbosity=2)
