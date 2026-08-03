"""Stage 14 -- replay: prove that a historical interval can be rebuilt exactly.

Replay is tested two ways, because they fail for different reasons.

**Reconstruction from the ledger.** Take the consumption events for steps
[lo, hi), and rebuild each packed bin from nothing but what the ledger recorded
-- shard ids, token spans, segment order -- plus the sealed shard bytes. Feed
them through the same `build_bin` the original run used and compare
`content_hash` and `loss_mask_hash`. This proves the ledger is a sufficient
description of the stream: no hidden state was needed to reproduce it.

**Reconstruction by replanning.** Restore a checkpoint at or before `lo`, load
the planner state it carries, fast-forward to `lo`, and let the planner produce
steps [lo, hi) again. Compare batch hashes with the ledger. This proves the
*planner* is deterministic: the same seed, branch, config and shards produce the
same selection, the same OPUS decisions and the same packing, in a different
process, with a model that has different weights than the one running at the
time -- because nothing in the data path reads model state.

The first can pass while the second fails (a planner that consults the clock).
The second can pass while the first fails (a ledger that omits a span). The
assignment needs both.
"""

from __future__ import annotations

import numpy as np

from .common import sha256_hex
from .manifest import read_shard_tokens
from .packing import array_bytes, build_bin


# --------------------------------------------------------------------------- #
# 1. rebuild the bins from the ledger records
# --------------------------------------------------------------------------- #

def replay_from_ledger(cfg, registry, consumption_events: list[dict],
                       tokenizer_record: dict, interval: tuple[int, int],
                       branch_id: str, log=None) -> dict:
    lo, hi = interval
    seq_len = int(cfg.packing.seq_len)
    eos_id = int(tokenizer_record["eos_token_id"])
    pad_id = int(tokenizer_record["pad_token_id"])
    eos_between = bool(cfg.packing.eos_between_samples)
    reset_pos = bool(cfg.packing.reset_position_ids)

    tokens: dict[str, np.ndarray] = {}

    def reader(shard_id: str, start: int, end: int) -> np.ndarray:
        if shard_id not in tokens:
            tokens[shard_id] = read_shard_tokens(registry.by_id[shard_id],
                                                 registry.shards_root)
        return tokens[shard_id][start:end]

    rows, mismatches = [], []
    for ev in consumption_events:
        if ev.get("kind") != "microbatch" or ev.get("branch_id") != branch_id:
            continue
        if not (lo <= ev["global_step"] < hi):
            continue
        placed = [{
            "sample_id": f"{sp['shard_id']}:{sp['token_start']}-{sp['token_end']}",
            "shard_id": sp["shard_id"], "doc_id": sp["doc_id"],
            "token_start": sp["token_start"], "token_end": sp["token_end"],
            "n_tokens": sp["token_end"] - sp["token_start"],
        } for sp in sorted(ev["token_spans"], key=lambda s: s["bin_offset"])]

        rebuilt = build_bin(bin_id=ev["microbatch_id"], lane=ev["mixture_lane"],
                            placed=placed, token_reader=reader, seq_len=seq_len,
                            eos_id=eos_id, pad_id=pad_id, eos_between=eos_between,
                            reset_position_ids=reset_pos)
        loss_mask_hash = sha256_hex(array_bytes(rebuilt["loss_mask"]))
        row = {
            "step": ev["global_step"],
            "microbatch_id": ev["microbatch_id"],
            "original_bin_hash": ev["bin_hash"],
            "replay_bin_hash": rebuilt["content_hash"],
            "bin_hash_match": rebuilt["content_hash"] == ev["bin_hash"],
            "original_loss_mask_hash": ev["loss_mask_hash"],
            "replay_loss_mask_hash": loss_mask_hash,
            "loss_mask_match": loss_mask_hash == ev["loss_mask_hash"],
            "sample_ids_match": [s["sample_id"] for s in rebuilt["samples"]] ==
                                list(ev["packed_sample_ids"]),
            "loss_tokens_match": rebuilt["loss_tokens"] == ev["loss_tokens"],
        }
        row["ok"] = all(row[k] for k in ("bin_hash_match", "loss_mask_match",
                                         "sample_ids_match", "loss_tokens_match"))
        rows.append(row)
        if not row["ok"]:
            mismatches.append(row)

    report = {
        "mode": "reconstruct_from_ledger",
        "branch_id": branch_id,
        "interval": [lo, hi],
        "microbatches_replayed": len(rows),
        "steps_replayed": sorted({r["step"] for r in rows}),
        "mismatches": mismatches,
        "all_match": bool(rows) and not mismatches,
        "rows": rows,
    }
    if log:
        log.check("replay_hash_matched", report["all_match"],
                  mode="from_ledger", interval=f"[{lo},{hi})",
                  microbatches=len(rows), mismatches=len(mismatches))
    return report


# --------------------------------------------------------------------------- #
# 2. rebuild the batches by replanning from a checkpoint
# --------------------------------------------------------------------------- #

def replay_by_replan(cfg, schedule, index, registry, tokenizer_record: dict,
                     branch_id: str, seed: int, planner_state: dict,
                     from_step: int, interval: tuple[int, int],
                     ledger_batches: dict[int, dict], log=None) -> dict:
    """Fast-forward a fresh planner from `from_step` and re-produce [lo, hi)."""
    from .batching import BatchPlanner
    from .opus import OpusPolicy

    lo, hi = interval
    planner = BatchPlanner(cfg, schedule, index, OpusPolicy(cfg), registry,
                           tokenizer_record, branch_id, seed)
    planner.load_state_dict(planner_state)
    if planner.state.step != from_step:
        raise RuntimeError(f"planner state is at {planner.state.step}, expected {from_step}")
    planner.fast_forward(lo)

    rows, mismatches = [], []
    for step in range(lo, hi):
        batch = planner.plan_step(step)
        original = ledger_batches.get(step, {})
        row = {
            "step": step,
            "original_batch_id": original.get("batch_id"),
            "replay_batch_id": batch["batch_id"],
            "original_batch_hash": original.get("batch_hash"),
            "replay_batch_hash": batch["batch_hash"],
            "batch_hash_match": original.get("batch_hash") == batch["batch_hash"],
            "bin_hashes_match": list(original.get("bin_hashes", [])) ==
                                [b["content_hash"] for b in batch["bins"]],
            "lanes_match": original.get("actual_bins") == batch["actual_bins"],
            "planner_fingerprint_match": original.get("planner_fingerprint") ==
                                         batch["planner_fingerprint"],
        }
        row["ok"] = all(row[k] for k in ("batch_hash_match", "bin_hashes_match",
                                         "lanes_match", "planner_fingerprint_match"))
        rows.append(row)
        if not row["ok"]:
            mismatches.append(row)

    report = {
        "mode": "replan_from_checkpoint",
        "branch_id": branch_id,
        "from_checkpoint_step": from_step,
        "interval": [lo, hi],
        "steps_replayed": len(rows),
        "mismatches": mismatches,
        "all_match": bool(rows) and not mismatches,
        "rows": rows,
    }
    if log:
        log.check("replay_replan_matched", report["all_match"],
                  mode="replan", from_checkpoint=from_step,
                  interval=f"[{lo},{hi})", steps=len(rows), mismatches=len(mismatches))
    return report


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def ledger_batches_by_step(consumption_events: list[dict], branch_id: str) -> dict[int, dict]:
    return {ev["global_step"]: ev for ev in consumption_events
            if ev.get("kind") == "batch" and ev.get("branch_id") == branch_id}


def coverage(consumption_events: list[dict], branch_id: str,
             expected_steps: int, log=None) -> dict:
    """Every step consumed exactly once: no skips, no repeats, no gaps."""
    steps = [ev["global_step"] for ev in consumption_events
             if ev.get("kind") == "batch" and ev.get("branch_id") == branch_id]
    counts: dict[int, int] = {}
    for s in steps:
        counts[s] = counts.get(s, 0) + 1
    duplicates = sorted(s for s, n in counts.items() if n > 1)
    missing = sorted(set(range(expected_steps)) - set(counts))
    extra = sorted(s for s in counts if s >= expected_steps)
    result = {
        "branch_id": branch_id,
        "expected_steps": expected_steps,
        "recorded_steps": len(steps),
        "unique_steps": len(counts),
        "duplicated_steps": duplicates,
        "missing_steps": missing,
        "out_of_range_steps": extra,
        "exactly_once": not duplicates and not missing and not extra,
        "monotonic": steps == sorted(steps),
    }
    if log:
        log.check("no_skipped_or_repeated_batches",
                  result["exactly_once"] and result["monotonic"],
                  branch=branch_id, expected=expected_steps,
                  recorded=result["recorded_steps"],
                  duplicated=len(duplicates), missing=len(missing))
    return result
