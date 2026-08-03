"""Stage 4/5 -- the mixture schedule, curriculum stages and protected floors.

Session 5 describes a mixture in human terms ("code is 15% early, 40% later").
This module turns that into an integer quota for every optimizer step:

    step 7 -> {wiki: 5, dialog: 2, code: 1}   (batch_size = 8 packed bins)

Two mechanisms:

  largest remainder   turns fractional shares into whole bins without drift, so
                      the realised share over a stage tracks the planned share
                      instead of always rounding the same lane down.

  protected floors    a floor is applied *after* rounding and *after* OPUS has
                      rejected candidates. A lane at its floor cannot be pushed
                      below it: bins are taken back from the largest lane and
                      the rescue is recorded as a protected-floor override.

The compiled plan is written to manifests/mixture_plan.json and hashed, so the
planned schedule is itself an auditable artifact.
"""

from __future__ import annotations

import math
import os

from .common import hash_obj, write_json


class MixtureSchedule:
    def __init__(self, stages: list[dict], batch_size: int, tolerance: float = 0.05) -> None:
        self.stages = [dict(s) for s in stages]
        for s in self.stages:
            s["shares"] = dict(s["shares"])
            s["protected_floors"] = dict(s.get("protected_floors") or {})
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.lanes = sorted({ln for s in self.stages for ln in s["shares"]})

    # -- stage lookup ------------------------------------------------------- #

    def stage_for_step(self, step: int) -> dict:
        for s in self.stages:
            if s["step_start"] <= step < s["step_end"]:
                return s
        return self.stages[-1]

    def stage_name(self, step: int) -> str:
        return self.stage_for_step(step)["name"]

    # -- quotas ------------------------------------------------------------- #

    def quota_for_step(self, step: int) -> dict[str, int]:
        """Integer bins per lane for this step (largest remainder + floors)."""
        stage = self.stage_for_step(step)
        shares = stage["shares"]
        n = self.batch_size

        exact = {ln: shares.get(ln, 0.0) * n for ln in self.lanes}
        quota = {ln: int(math.floor(v)) for ln, v in exact.items()}
        remainder = n - sum(quota.values())
        # hand the leftover bins to the largest fractional parts; ties break on
        # lane name so the result never depends on dict ordering
        ranked = sorted(self.lanes, key=lambda ln: (-(exact[ln] - math.floor(exact[ln])), ln))
        for ln in ranked[:remainder]:
            quota[ln] += 1

        return self.apply_floors(quota, step)[0]

    def floor_bins(self, step: int) -> dict[str, int]:
        stage = self.stage_for_step(step)
        return {ln: int(math.ceil(f * self.batch_size))
                for ln, f in stage["protected_floors"].items()}

    def apply_floors(self, quota: dict[str, int], step: int) -> tuple[dict[str, int], list[dict]]:
        """Raise any lane below its floor, paying for it from the largest lane."""
        quota = dict(quota)
        rescues: list[dict] = []
        floors = self.floor_bins(step)
        for lane, need in sorted(floors.items()):
            while quota.get(lane, 0) < need:
                donor = max((ln for ln in quota if ln != lane and
                             quota[ln] > floors.get(ln, 0)),
                            key=lambda ln: (quota[ln], ln), default=None)
                if donor is None:
                    break
                quota[donor] -= 1
                quota[lane] = quota.get(lane, 0) + 1
                rescues.append({"step": step, "lane": lane, "donor": donor,
                                "floor_bins": need, "reason": "protected_floor_override"})
        return quota, rescues

    def planned_shares(self, step: int) -> dict[str, float]:
        stage = self.stage_for_step(step)
        return {ln: float(stage["shares"].get(ln, 0.0)) for ln in self.lanes}

    # -- compilation -------------------------------------------------------- #

    def compile(self, total_steps: int) -> dict:
        per_step = []
        for step in range(total_steps):
            stage = self.stage_for_step(step)
            quota = self.quota_for_step(step)
            per_step.append({"step": step, "stage": stage["name"], "quota": quota,
                             "planned_shares": self.planned_shares(step),
                             "floor_bins": self.floor_bins(step)})
        totals: dict[str, int] = {ln: 0 for ln in self.lanes}
        for e in per_step:
            for ln, v in e["quota"].items():
                totals[ln] += v
        grand = max(1, sum(totals.values()))
        plan = {
            "batch_size": self.batch_size,
            "total_steps": total_steps,
            "lanes": self.lanes,
            "stages": self.stages,
            "tolerance": self.tolerance,
            "per_step": per_step,
            "planned_bin_totals": totals,
            "planned_overall_shares": {ln: round(v / grand, 6) for ln, v in totals.items()},
        }
        plan["plan_hash"] = hash_obj({k: v for k, v in plan.items() if k != "plan_hash"})
        return plan

    # -- verification ------------------------------------------------------- #

    def compliance(self, actual_bins_by_step: list[dict], plan: dict) -> dict:
        """Planned versus actual lane shares, per stage and overall."""
        by_stage: dict[str, dict] = {}
        for entry, actual in zip(plan["per_step"], actual_bins_by_step):
            st = by_stage.setdefault(entry["stage"], {
                "planned": {ln: 0 for ln in self.lanes},
                "actual": {ln: 0 for ln in self.lanes}, "steps": 0})
            st["steps"] += 1
            for ln in self.lanes:
                st["planned"][ln] += entry["quota"].get(ln, 0)
                st["actual"][ln] += actual.get(ln, 0)

        rows, worst = [], 0.0
        for stage_name, st in sorted(by_stage.items()):
            p_tot = max(1, sum(st["planned"].values()))
            a_tot = max(1, sum(st["actual"].values()))
            for ln in self.lanes:
                p = st["planned"][ln] / p_tot
                a = st["actual"][ln] / a_tot
                floor = next((s["protected_floors"].get(ln, 0.0) for s in self.stages
                              if s["name"] == stage_name), 0.0)
                rows.append({"stage": stage_name, "lane": ln,
                             "planned_share": round(p, 6), "actual_share": round(a, 6),
                             "deviation": round(abs(a - p), 6),
                             "floor": floor, "floor_respected": a + 1e-9 >= floor,
                             "planned_bins": st["planned"][ln],
                             "actual_bins": st["actual"][ln]})
                worst = max(worst, abs(a - p))
        return {
            "tolerance": self.tolerance,
            "max_deviation": round(worst, 6),
            "within_tolerance": worst <= self.tolerance,
            "floors_respected": all(r["floor_respected"] for r in rows),
            "rows": rows,
        }


def supply_check(schedule: MixtureSchedule, registry, plan: dict, seq_len: int,
                 log=None) -> dict:
    """Demand (bins x seq_len) against the tokens actually available per lane."""
    available: dict[str, int] = {}
    samples: dict[str, int] = {}
    for sid in registry.trainable_ids():
        m = registry.by_id[sid]
        available[m["lane"]] = available.get(m["lane"], 0) + m["num_tokens"]
        samples[m["lane"]] = samples.get(m["lane"], 0) + m["num_samples"]

    rows = []
    for lane in schedule.lanes:
        demand = plan["planned_bin_totals"].get(lane, 0) * seq_len
        have = available.get(lane, 0)
        rows.append({"lane": lane, "demand_tokens": demand, "available_tokens": have,
                     "available_samples": samples.get(lane, 0),
                     "repetition_factor": round(demand / max(1, have), 4),
                     "satisfied": have >= demand})
    report = {"seq_len": seq_len, "rows": rows,
              "all_satisfied": all(r["satisfied"] for r in rows)}
    if log:
        for r in rows:
            log.event("lane_supply", "PASS" if r["satisfied"] else "INFO",
                      lane=r["lane"], demand=r["demand_tokens"],
                      available=r["available_tokens"],
                      repetition=r["repetition_factor"])
    return report


def write_plan(cfg, plan: dict, log=None) -> str:
    path = os.path.join(cfg.resolve("manifests_dir"), "mixture_plan.json")
    write_json(path, plan)
    if log:
        log.ok("mixture_compiled", steps=plan["total_steps"],
               lanes=len(plan["lanes"]), plan_hash=plan["plan_hash"][:16],
               overall=plan["planned_overall_shares"])
    return path
