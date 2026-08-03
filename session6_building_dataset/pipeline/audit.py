"""Stage 14b -- audit: what trained this checkpoint, and what preceded that spike?

The ledgers are append-only files. An audit is a query over them. Three questions
from the session notes, answered from the recorded events and nothing else:

    which shards influenced the model between step A and step B?
    which OPUS-selected batches appeared before a loss spike?
    which data trained the checkpoint at step N?

`shards_between` is the reconstruction query: it walks the consumption ledger
and reports tokens per shard, per lane and per stage over an interval. Because
the consumption ledger stores token spans, the answer is exact rather than
approximate -- the same query can hand back the tokens themselves via
`replay.replay_from_ledger`.

`loss_spikes` looks for steps whose loss jumps more than `sigma` standard
deviations above the trailing mean and reports the batches, lanes and OPUS
decisions immediately preceding them. That is the shape of a real incident
investigation: the model did something strange at step N, so what was it eating.
"""

from __future__ import annotations

import numpy as np

from .common import hash_obj


def shards_between(consumption_events: list[dict], branch_id: str,
                   lo: int, hi: int) -> dict:
    """Token accounting per shard/lane/stage for the half-open interval [lo, hi)."""
    by_shard: dict[str, dict] = {}
    by_lane: dict[str, int] = {}
    by_stage: dict[str, int] = {}
    by_doc: dict[str, int] = {}
    steps = set()

    for ev in consumption_events:
        if ev.get("kind") != "microbatch" or ev.get("branch_id") != branch_id:
            continue
        if not (lo <= ev["global_step"] < hi):
            continue
        steps.add(ev["global_step"])
        by_lane[ev["mixture_lane"]] = by_lane.get(ev["mixture_lane"], 0) + ev["loss_tokens"]
        by_stage[ev["curriculum_stage"]] = by_stage.get(
            ev["curriculum_stage"], 0) + ev["loss_tokens"]
        for sp in ev["token_spans"]:
            n = sp["token_end"] - sp["token_start"]
            rec = by_shard.setdefault(sp["shard_id"], {
                "shard_id": sp["shard_id"], "lane": ev["mixture_lane"],
                "content_tokens": 0, "spans": 0, "steps": set()})
            rec["content_tokens"] += n
            rec["spans"] += 1
            rec["steps"].add(ev["global_step"])
            by_doc[sp["doc_id"]] = by_doc.get(sp["doc_id"], 0) + n

    shards = []
    for rec in by_shard.values():
        shards.append({**rec, "steps": sorted(rec["steps"]),
                       "first_step": min(rec["steps"]), "last_step": max(rec["steps"])})
    shards.sort(key=lambda r: -r["content_tokens"])

    return {
        "branch_id": branch_id,
        "interval": [lo, hi],
        "steps_covered": sorted(steps),
        "shard_count": len(shards),
        "shards": shards,
        "content_tokens_by_lane": dict(sorted(by_lane.items())),
        "loss_tokens_by_stage": dict(sorted(by_stage.items())),
        "top_documents": sorted(({"doc_id": d, "content_tokens": n}
                                 for d, n in by_doc.items()),
                                key=lambda r: -r["content_tokens"])[:15],
    }


def loss_spikes(learning_events: list[dict], branch_id: str, sigma: float = 2.0,
                window: int = 8) -> dict:
    """Steps whose loss jumps above the trailing distribution."""
    rows = [e for e in learning_events
            if e.get("branch_id") == branch_id and e.get("kind") != "validation"]
    rows.sort(key=lambda e: e["global_step"])
    losses = np.asarray([e["loss"] for e in rows], dtype=float)
    grads = np.asarray([e["grad_norm"] for e in rows], dtype=float)

    spikes = []
    for i in range(window, len(rows)):
        hist = losses[max(0, i - window):i]
        mu, sd = float(hist.mean()), float(hist.std())
        if sd > 1e-9 and losses[i] > mu + sigma * sd:
            spikes.append({
                "step": rows[i]["global_step"],
                "loss": rows[i]["loss"],
                "trailing_mean": round(mu, 6),
                "trailing_std": round(sd, 6),
                "z": round((losses[i] - mu) / sd, 4),
                "grad_norm": rows[i]["grad_norm"],
                "stage": rows[i]["curriculum_stage"],
                "lanes": rows[i]["actual_bins"],
                "batch_hash": rows[i]["batch_hash"],
                "preceding_steps": [rows[j]["global_step"]
                                    for j in range(max(0, i - 3), i)],
            })
    return {
        "branch_id": branch_id,
        "sigma": sigma,
        "window": window,
        "steps_examined": len(rows),
        "loss_first": round(float(losses[0]), 6) if len(losses) else None,
        "loss_last": round(float(losses[-1]), 6) if len(losses) else None,
        "loss_mean": round(float(losses.mean()), 6) if len(losses) else None,
        "grad_norm_max": round(float(grads.max()), 6) if len(grads) else None,
        "spike_count": len(spikes),
        "spikes": spikes,
    }


def opus_before(opus_events: list[dict], branch_id: str, step: int,
                lookback: int = 3) -> dict:
    """The selector's decisions in the steps leading up to a step of interest."""
    lo = max(0, step - lookback)
    rows = [e for e in opus_events
            if e.get("branch_id") == branch_id and lo <= e.get("global_step", -1) <= step]
    by_status: dict[str, int] = {}
    by_lane: dict[str, dict] = {}
    for e in rows:
        by_status[e["status"]] = by_status.get(e["status"], 0) + 1
        lane = by_lane.setdefault(e["lane"], {})
        lane[e["status"]] = lane.get(e["status"], 0) + 1
    accepted = [e for e in rows if e["status"] == "accepted"]
    return {
        "step": step, "lookback": [lo, step], "decisions": len(rows),
        "by_status": by_status, "by_lane": by_lane,
        "mean_accepted_score": round(
            float(np.mean([e["score"] for e in accepted])), 6) if accepted else None,
        "protected_floor_overrides": sum(1 for e in rows
                                         if e.get("protected_floor_override")),
        "accepted_shards": sorted({e["shard_id"] for e in accepted}),
    }


def checkpoint_provenance(consumption_events: list[dict], learning_events: list[dict],
                          branch_id: str, step: int) -> dict:
    """Everything that trained the model up to the checkpoint at `step`."""
    window = shards_between(consumption_events, branch_id, 0, step)
    learn = [e for e in learning_events
             if e.get("branch_id") == branch_id and e.get("kind") != "validation"
             and e["global_step"] < step]
    return {
        "branch_id": branch_id,
        "checkpoint_step": step,
        "steps_trained": len(learn),
        "loss_tokens": sum(e["loss_tokens"] for e in learn),
        "shards_influencing": window["shard_count"],
        "content_tokens_by_lane": window["content_tokens_by_lane"],
        "loss_first": learn[0]["loss"] if learn else None,
        "loss_last": learn[-1]["loss"] if learn else None,
        "top_shards": [{"shard_id": s["shard_id"], "content_tokens": s["content_tokens"]}
                       for s in window["shards"][:10]],
    }


def run_audit(cfg, consumption_events: list[dict], learning_events: list[dict],
              opus_events: list[dict], branch_id: str, interval: tuple[int, int],
              checkpoint_step: int, log=None) -> dict:
    lo, hi = interval
    window = shards_between(consumption_events, branch_id, lo, hi)
    spikes = loss_spikes(learning_events, branch_id)
    provenance = checkpoint_provenance(consumption_events, learning_events,
                                       branch_id, checkpoint_step)
    focus = spikes["spikes"][0]["step"] if spikes["spikes"] else max(lo, hi - 1)
    selector = opus_before(opus_events, branch_id, focus)

    report = {
        "branch_id": branch_id,
        "questions": {
            f"which shards influenced steps [{lo},{hi})": window,
            "which OPUS decisions preceded the step of interest": selector,
            f"what trained the checkpoint at step {checkpoint_step}": provenance,
        },
        "loss_spikes": spikes,
        "step_of_interest": focus,
    }
    report["audit_hash"] = hash_obj({
        "window": window["shards"], "focus": focus,
        "provenance": provenance["top_shards"]})
    if log:
        log.ok("audit_completed", interval=f"[{lo},{hi})",
               shards_identified=window["shard_count"],
               lanes=len(window["content_tokens_by_lane"]),
               spikes=spikes["spike_count"], step_of_interest=focus,
               audit_hash=report["audit_hash"][:16])
        log.info(f"audit: steps [{lo},{hi}) touched {window['shard_count']} shards; "
                 f"top shard {window['shards'][0]['shard_id']} "
                 f"({window['shards'][0]['content_tokens']} tokens)"
                 if window["shards"] else "audit: no shards in window")
    return report
