"""Stage 15 -- throughput and packing efficiency.

The number that matters is not tokens/sec. It is **useful loss-bearing tokens
per second at the target mixture**. A loader can look fast while most of what it
delivers is padding, context-only positions, or candidates OPUS will throw away.

So the report always carries three rates side by side:

    raw_tokens_per_s        every position that reached the model
    accepted_tokens_per_s   content tokens OPUS let through
    useful_tokens_per_s     positions that actually produced gradient

and the packing section reports what the chosen policy bought, measured against
the alternatives over the same sample set rather than asserted.

Every figure here is derived from counters the run incremented and from the
ledgers on disk, so `python -m pytest tests/` can recompute them.
"""

from __future__ import annotations

import numpy as np

from .common import hash_obj, write_json


def packing_efficiency(consumption_events: list[dict], seq_len: int,
                       branch_id: str | None = None) -> dict:
    """Recompute packing utilisation from the consumption ledger."""
    rows = [e for e in consumption_events if e.get("kind") == "microbatch"
            and (branch_id is None or e.get("branch_id") == branch_id)]
    if not rows:
        return {"microbatches": 0}
    content = sum(e["content_tokens"] for e in rows)
    pad = sum(e["pad_tokens"] for e in rows)
    loss = sum(e["loss_tokens"] for e in rows)
    positions = len(rows) * seq_len
    samples = sum(len(e["packed_sample_ids"]) for e in rows)
    utils = np.asarray([e["utilisation"] for e in rows], dtype=float)
    return {
        "microbatches": len(rows),
        "positions": positions,
        "content_tokens": content,
        "pad_tokens": pad,
        "loss_tokens": loss,
        "samples_packed": samples,
        "samples_per_bin": round(samples / len(rows), 4),
        "packing_utilisation": round(content / positions, 6),
        "loss_utilisation": round(loss / positions, 6),
        "padding_fraction": round(pad / positions, 6),
        "utilisation_p05": round(float(np.percentile(utils, 5)), 6),
        "utilisation_p50": round(float(np.percentile(utils, 50)), 6),
        "utilisation_p95": round(float(np.percentile(utils, 95)), 6),
        "wasted_positions_vs_full": positions - content,
    }


def opus_throughput(opus_events: list[dict], branch_id: str | None = None) -> dict:
    """Rejection pressure per lane -- the reason accepted tokens/s < raw tokens/s."""
    rows = [e for e in opus_events
            if branch_id is None or e.get("branch_id") == branch_id]
    by_lane: dict[str, dict] = {}
    for e in rows:
        lane = by_lane.setdefault(e["lane"], {"candidates": 0, "accepted": 0,
                                              "rejected": 0, "deferred": 0,
                                              "overrides": 0})
        lane["candidates"] += 1
        lane[e["status"]] = lane.get(e["status"], 0) + 1
        if e.get("protected_floor_override"):
            lane["overrides"] += 1
    for lane in by_lane.values():
        lane["rejection_rate"] = round(lane["rejected"] / max(1, lane["candidates"]), 6)
        lane["acceptance_rate"] = round(lane["accepted"] / max(1, lane["candidates"]), 6)
    total = len(rows)
    return {
        "candidates": total,
        "accepted": sum(1 for e in rows if e["status"] == "accepted"),
        "rejected": sum(1 for e in rows if e["status"] == "rejected"),
        "deferred": sum(1 for e in rows if e["status"] == "deferred"),
        "protected_floor_overrides": sum(1 for e in rows
                                         if e.get("protected_floor_override")),
        "rejection_rate": round(sum(1 for e in rows if e["status"] == "rejected")
                                / max(1, total), 6),
        "by_lane": {k: by_lane[k] for k in sorted(by_lane)},
    }


def build_performance_report(cfg, meters: dict, cache_stats: dict,
                             consumption_events: list[dict], opus_events: list[dict],
                             policy_comparison: dict, phase_timings: dict,
                             recovery_timings: dict, model_params: int,
                             log=None) -> dict:
    seq_len = int(cfg.packing.seq_len)
    packing = packing_efficiency(consumption_events, seq_len, cfg.run.branch_id)
    selector = opus_throughput(opus_events, cfg.run.branch_id)

    chosen = cfg.packing.policy
    rows = {r["policy"]: r for r in policy_comparison["rows"]}
    baseline = rows.get("pad_only", {}).get("utilisation", 0.0)
    achieved = rows.get(chosen, {}).get("utilisation", 0.0)

    report = {
        "run_id": cfg.run.run_id,
        "config_hash": cfg.config_hash,
        "device": cfg.run.device,
        "model_parameters": model_params,
        "sequence_length": seq_len,
        "batch_size_bins": int(cfg.train.batch_size),
        "throughput": meters,
        "shard_cache": cache_stats,
        "packing": {
            **packing,
            "policy": chosen,
            "policy_comparison": policy_comparison["rows"],
            "utilisation_vs_pad_only": round(achieved - baseline, 6),
            "bins_saved_vs_pad_only": (rows.get("pad_only", {}).get("bins", 0)
                                       - rows.get(chosen, {}).get("bins", 0)),
        },
        "opus": selector,
        "phase_seconds": phase_timings,
        "recovery_seconds": recovery_timings,
        "headline": {
            "raw_tokens_per_s": meters.get("main", {}).get("raw_tokens_per_s"),
            "useful_tokens_per_s": meters.get("main", {}).get("useful_tokens_per_s"),
            "accepted_tokens_per_s": meters.get("main", {}).get("accepted_tokens_per_s"),
            "useful_token_fraction": meters.get("main", {}).get("useful_token_fraction"),
            "packing_utilisation": packing.get("packing_utilisation"),
            "opus_rejection_rate": selector.get("rejection_rate"),
            "loader_wait_fraction": meters.get("main", {}).get("loader_wait_fraction"),
            "cache_hit_rate": cache_stats.get("hit_rate"),
        },
    }
    report["performance_hash"] = hash_obj(report["headline"])
    if log:
        h = report["headline"]
        log.ok("performance_measured",
               raw_tok_s=h["raw_tokens_per_s"], useful_tok_s=h["useful_tokens_per_s"],
               accepted_tok_s=h["accepted_tokens_per_s"],
               useful_fraction=h["useful_token_fraction"],
               packing_util=h["packing_utilisation"],
               cache_hit_rate=h["cache_hit_rate"])
        log.info(f"packing: {chosen} reached {achieved:.4f} utilisation vs "
                 f"{baseline:.4f} for pad_only "
                 f"({report['packing']['bins_saved_vs_pad_only']} fewer bins for the "
                 f"same samples)")
    return report


def write_performance(cfg, report: dict) -> str:
    import os
    path = os.path.join(cfg.resolve("artifacts_dir"), "performance.json")
    write_json(path, report)
    return path
