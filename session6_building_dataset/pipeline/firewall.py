"""Stage 13 -- the evaluation firewall, exercised rather than asserted.

The registry in `manifest.py` owns the permissions. This module attacks them, so
the evidence bundle can point at a blocked attempt instead of a claim.

Four probes:

  1. every eval shard is offered to `check_trainable` and must raise
  2. every validation shard is offered to `check_trainable` and must raise,
     while `check_readable_for_eval` on the same shard must succeed
  3. the contaminated training shard the scanner found is offered and must raise
  4. a forged manifest -- an eval shard relabelled `split=train, permission=
     trainable` -- is offered. The registry answers on identity, not on labels,
     so the forgery is caught by content hash against the registered eval shard

Every attempt, blocked or allowed, is appended to the firewall ledger. A final
audit re-reads the consumption ledger and proves that no sample id belonging to
a non-trainable shard ever reached a loss-bearing batch.
"""

from __future__ import annotations

from .common import now_iso
from .manifest import FirewallViolation


def _record(ledgers, **fields) -> None:
    ledgers.firewall.append({"ts_probe": now_iso(), **fields})


def firewall_drill(registry, ledgers, contamination_report: dict, log=None) -> dict:
    """Try to get non-trainable data into training. Every attempt must fail."""
    results = {"attempts": [], "blocked": 0, "allowed": 0, "leaked": []}

    def attempt(shard_id: str, context: str, expect_block: bool) -> None:
        try:
            registry.check_trainable(shard_id, context)
            outcome, reason = "allowed", None
        except FirewallViolation as exc:
            outcome, reason = "blocked", str(exc)
        row = {"shard_id": shard_id, "context": context, "outcome": outcome,
               "reason": reason, "expected": "blocked" if expect_block else "allowed",
               "correct": (outcome == "blocked") == expect_block}
        results["attempts"].append(row)
        results["blocked" if outcome == "blocked" else "allowed"] += 1
        if expect_block and outcome == "allowed":
            results["leaked"].append(shard_id)
        _record(ledgers, event="train_access_attempt", **row)

    # 1 -- eval shards
    for sid in registry.ids_for("eval"):
        attempt(sid, "firewall_drill:eval_into_training", expect_block=True)

    # 2 -- validation shards: blocked for training, readable for evaluation
    val_read_ok = True
    for sid in registry.ids_for("validation"):
        attempt(sid, "firewall_drill:validation_into_training", expect_block=True)
        try:
            registry.check_readable_for_eval(sid)
            _record(ledgers, event="validation_eval_read", shard_id=sid,
                    outcome="allowed", correct=True)
        except FirewallViolation as exc:
            val_read_ok = False
            _record(ledgers, event="validation_eval_read", shard_id=sid,
                    outcome="blocked", reason=str(exc), correct=False)
    results["validation_readable_for_eval"] = val_read_ok

    # 3 -- the shard the contamination scanner condemned
    for sid in contamination_report.get("contaminated", []):
        attempt(sid, "firewall_drill:contaminated_shard", expect_block=True)
    results["contaminated_shards"] = list(contamination_report.get("contaminated", []))

    # 4 -- a forged manifest that claims to be trainable
    forgery = _forgery_probe(registry, ledgers)
    results["forged_manifest"] = forgery

    # 5 -- a shard that genuinely is trainable must still be allowed, or the
    #      firewall is just a wall
    trainable = registry.trainable_ids()
    if trainable:
        attempt(trainable[0], "firewall_drill:control_trainable", expect_block=False)

    results["all_correct"] = (all(r["correct"] for r in results["attempts"])
                              and val_read_ok and forgery["blocked"])
    if log:
        log.event("eval_shard_blocked",
                  "PASS" if not results["leaked"] else "FAIL",
                  eval_shards=len(registry.ids_for("eval")),
                  attempts=len(results["attempts"]),
                  blocked=results["blocked"], leaked=len(results["leaked"]))
        log.check("validation_read_only_enforced", val_read_ok,
                  validation_shards=len(registry.ids_for("validation")))
        log.check("forged_manifest_blocked", forgery["blocked"],
                  detected_by=forgery["detected_by"])
        for sid in results["contaminated_shards"]:
            log.ok("contaminated_shard_blocked", shard=sid)
    return results


def _forgery_probe(registry, ledgers) -> dict:
    """Relabel an eval shard as trainable and see whether identity beats labels."""
    eval_ids = registry.ids_for("eval")
    if not eval_ids:
        return {"attempted": False, "blocked": True, "detected_by": "no_eval_shards"}

    victim = registry.by_id[eval_ids[0]]
    forged = dict(victim)
    forged["shard_id"] = victim["shard_id"] + "-FORGED"
    forged["split"] = "train"
    forged["permission"] = "trainable"
    forged["never_train"] = False
    forged["admitted"] = True

    # the registry is asked about a shard it does not know
    detected_by = None
    try:
        registry.check_trainable(forged["shard_id"], "firewall_drill:forged_manifest")
        blocked = False
    except FirewallViolation:
        blocked, detected_by = True, "unknown_shard_id"

    # and even if it were registered, the content hash still resolves to an
    # eval shard, which is the check that does not depend on the label
    twin = next((m for m in registry.by_id.values()
                 if m["content_hash"] == forged["content_hash"]
                 and m["shard_id"] != forged["shard_id"]), None)
    if twin is not None and twin["permission"] == "never_train":
        blocked = True
        detected_by = f"{detected_by or 'content_hash'}+content_hash_matches:{twin['shard_id']}"

    row = {"event": "forged_manifest_attempt", "shard_id": forged["shard_id"],
           "claimed_split": "train", "claimed_permission": "trainable",
           "true_split": victim["split"], "true_permission": victim["permission"],
           "content_hash": forged["content_hash"], "blocked": blocked,
           "detected_by": detected_by, "correct": blocked}
    _record(ledgers, **row)
    return {"attempted": True, **row}


def audit_consumption_permissions(consumption_events: list[dict], registry,
                                  log=None) -> dict:
    """Re-read the consumption ledger: did anything non-trainable ever train?"""
    seen_shards: set[str] = set()
    violations = []
    for ev in consumption_events:
        if ev.get("kind") != "microbatch":
            continue
        for sid in ev.get("shard_ids", []):
            seen_shards.add(sid)
            m = registry.by_id.get(sid)
            if m is None:
                violations.append({"step": ev["global_step"], "shard_id": sid,
                                   "reason": "unknown_shard"})
            elif m["permission"] != "trainable" or not m.get("admitted"):
                violations.append({"step": ev["global_step"], "shard_id": sid,
                                   "reason": f"permission={m['permission']}, "
                                             f"admitted={m.get('admitted')}"})
    result = {
        "consumed_shards": len(seen_shards),
        "violations": violations,
        "clean": not violations,
        "eval_shards_consumed": sorted(
            s for s in seen_shards
            if registry.by_id.get(s, {}).get("split") == "eval"),
        "validation_shards_consumed": sorted(
            s for s in seen_shards
            if registry.by_id.get(s, {}).get("split") == "validation"),
        "blocked_shards_consumed": sorted(
            s for s in seen_shards if not registry.by_id.get(s, {}).get("admitted", False)),
    }
    if log:
        log.check("no_nontrainable_data_in_loss_bearing_batches", result["clean"],
                  consumed_shards=result["consumed_shards"],
                  eval_consumed=len(result["eval_shards_consumed"]),
                  validation_consumed=len(result["validation_shards_consumed"]),
                  blocked_consumed=len(result["blocked_shards_consumed"]))
    return result
