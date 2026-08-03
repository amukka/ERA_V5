"""Stage 8 -- the deterministic batch planner.

This is the component the whole submission turns on. It answers one question,
and it must answer it identically in every process, forever:

    given (branch, step, config, shards, planner state) -> which batch?

Everything it consumes is content-derived: shard bytes, manifest hashes, keyed
permutations, and OPUS features computed from tokens. Nothing depends on model
weights, wall-clock time, worker count or dict iteration order. That is why the
resumed run lands on exactly the batch the crashed run was about to take, and
why replaying steps 5..10 six phases later reproduces the same hashes.

Per step:

  1. `MixtureSchedule` gives an integer bin quota per lane.
  2. For each lane, candidates are drawn in a keyed permutation of that lane's
     samples and passed through OPUS. Accepted samples land in a per-lane
     buffer; rejects and deferrals are recorded but the cursor moves on.
  3. The buffer is packed best-fit-decreasing until it yields `quota` bins.
  4. If a lane cannot reach its quota (OPUS rejected too much, or the lane ran
     dry) and the lane sits on a protected floor, the highest-scoring rejected
     candidates are rescued as protected-floor overrides.
  5. The step's bins are ordered by a keyed permutation and become the batch.

`PlannerState` (cursors, epochs, buffers, OPUS state) is small, serialisable,
and stored inside every checkpoint next to the ledger offsets.
"""

from __future__ import annotations

import os
import time
from typing import Callable

import numpy as np

from .common import hash_obj, sha256_hex, stable_permutation
from .manifest import read_shard_tokens
from .opus import symbol_ratio, token_sketch
from .packing import best_fit_decreasing, bin_hash, build_bin


# --------------------------------------------------------------------------- #
# shard access with a real cache (the throughput report reads its counters)
# --------------------------------------------------------------------------- #

class ShardCache:
    def __init__(self, registry, capacity: int = 24) -> None:
        self.registry = registry
        self.capacity = capacity
        self._cache: dict[str, np.ndarray] = {}
        self._order: list[str] = []
        self.hits = 0
        self.misses = 0
        self.bytes_read = 0
        self.read_seconds = 0.0

    def tokens(self, shard_id: str) -> np.ndarray:
        if shard_id in self._cache:
            self.hits += 1
            self._order.remove(shard_id)
            self._order.append(shard_id)
            return self._cache[shard_id]
        self.misses += 1
        m = self.registry.by_id[shard_id]
        t0 = time.perf_counter()
        arr = read_shard_tokens(m, self.registry.shards_root)
        self.read_seconds += time.perf_counter() - t0
        self.bytes_read += arr.nbytes
        self._cache[shard_id] = arr
        self._order.append(shard_id)
        if len(self._order) > self.capacity:
            self._cache.pop(self._order.pop(0), None)
        return arr

    def read(self, shard_id: str, start: int, end: int) -> np.ndarray:
        return self.tokens(shard_id)[start:end]

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / max(1, total), 6),
                "bytes_read": self.bytes_read,
                "read_seconds": round(self.read_seconds, 6)}


# --------------------------------------------------------------------------- #
# sample index
# --------------------------------------------------------------------------- #

class SampleIndex:
    """Every trainable sample, with the content features OPUS scores."""

    def __init__(self, registry, tok, log=None) -> None:
        self.registry = registry
        self.cache = ShardCache(registry)
        self.by_lane: dict[str, list[dict]] = {}
        self.by_id: dict[str, dict] = {}

        for shard_id in registry.trainable_ids():
            m = registry.by_id[shard_id]
            toks = self.cache.tokens(shard_id)
            spans = [(s["token_start"], s["token_end"]) for s in m["samples"]]
            texts = tok.batch_decode([toks[a:b].tolist() for a, b in spans])
            for s, text in zip(m["samples"], texts):
                rec = {
                    "sample_id": s["sample_id"], "shard_id": shard_id,
                    "doc_id": s["doc_id"], "lane": m["lane"],
                    "token_start": int(s["token_start"]), "token_end": int(s["token_end"]),
                    "n_tokens": int(s["n_tokens"]),
                    "token_hash": s["token_hash"][:16],
                    "sketch": token_sketch(toks[s["token_start"]:s["token_end"]]),
                    "symbol_ratio": round(symbol_ratio(text), 6),
                }
                self.by_lane.setdefault(m["lane"], []).append(rec)
                self.by_id[rec["sample_id"]] = rec

        for lane in self.by_lane:
            self.by_lane[lane].sort(key=lambda r: r["sample_id"])

        self.index_hash = hash_obj({lane: [r["sample_id"] for r in recs]
                                    for lane, recs in sorted(self.by_lane.items())})
        if log:
            log.ok("sample_index_built",
                   samples=sum(len(v) for v in self.by_lane.values()),
                   lanes={k: len(v) for k, v in sorted(self.by_lane.items())},
                   index_hash=self.index_hash[:16])

    def lane_order(self, lane: str, epoch: int, seed: int, branch: str) -> list[int]:
        return stable_permutation(len(self.by_lane[lane]), seed, branch, lane, "epoch", epoch)

    def summary(self) -> dict:
        rows = {}
        for lane, recs in sorted(self.by_lane.items()):
            lens = np.asarray([r["n_tokens"] for r in recs])
            rows[lane] = {
                "samples": len(recs), "tokens": int(lens.sum()),
                "mean_len": round(float(lens.mean()), 3),
                "p50_len": int(np.percentile(lens, 50)),
                "p95_len": int(np.percentile(lens, 95)),
                "mean_symbol_ratio": round(
                    float(np.mean([r["symbol_ratio"] for r in recs])), 6),
            }
        return {"index_hash": self.index_hash, "lanes": rows}


# --------------------------------------------------------------------------- #
# planner state
# --------------------------------------------------------------------------- #

class PlannerState:
    def __init__(self, lanes: list[str]) -> None:
        self.step = 0
        self.cursor = {ln: 0 for ln in lanes}
        self.epoch = {ln: 0 for ln in lanes}
        self.buffer = {ln: [] for ln in lanes}       # accepted sample ids awaiting a bin
        self.deferred = {ln: [] for ln in lanes}     # deferral pool, revisited later

    def to_dict(self, opus_state: dict) -> dict:
        return {"step": self.step, "cursor": dict(self.cursor), "epoch": dict(self.epoch),
                "buffer": {k: list(v) for k, v in self.buffer.items()},
                "deferred": {k: list(v) for k, v in self.deferred.items()},
                "opus": opus_state}

    @classmethod
    def from_dict(cls, d: dict) -> "PlannerState":
        st = cls(list(d["cursor"].keys()))
        st.step = d["step"]
        st.cursor = dict(d["cursor"])
        st.epoch = dict(d["epoch"])
        st.buffer = {k: list(v) for k, v in d["buffer"].items()}
        st.deferred = {k: list(v) for k, v in d["deferred"].items()}
        return st

    def fingerprint(self) -> str:
        return hash_obj({"step": self.step, "cursor": self.cursor, "epoch": self.epoch,
                         "buffer": self.buffer, "deferred": self.deferred})


# --------------------------------------------------------------------------- #
# the planner
# --------------------------------------------------------------------------- #

class BatchPlanner:
    def __init__(self, cfg, schedule, index: SampleIndex, opus, registry,
                 tokenizer_record: dict, branch_id: str, seed: int,
                 state: PlannerState | None = None, log=None) -> None:
        self.cfg = cfg
        self.schedule = schedule
        self.index = index
        self.opus = opus
        self.registry = registry
        self.branch_id = branch_id
        self.seed = seed
        self.log = log
        self.seq_len = int(cfg.packing.seq_len)
        self.eos_id = int(tokenizer_record["eos_token_id"])
        self.pad_id = int(tokenizer_record["pad_token_id"])
        self.eos_between = bool(cfg.packing.eos_between_samples)
        self.reset_pos = bool(cfg.packing.reset_position_ids)
        self.max_open_bins = int(cfg.packing.max_open_bins)
        self.draws_per_bin = int(cfg.opus.draws_per_bin)
        self.min_draws = int(cfg.opus.min_draws_per_lane)
        self.state = state or PlannerState(schedule.lanes)
        self.decisions: list[dict] = []      # OPUS decisions for the current step

    # -- candidate drawing -------------------------------------------------- #

    def _next_candidate(self, lane: str) -> dict | None:
        pool = self.index.by_lane.get(lane)
        if not pool:
            return None
        order = self.index.lane_order(lane, self.state.epoch[lane], self.seed, self.branch_id)
        if self.state.cursor[lane] >= len(order):
            self.state.epoch[lane] += 1                 # a repeated pass over the lane
            self.state.cursor[lane] = 0
            order = self.index.lane_order(lane, self.state.epoch[lane], self.seed,
                                          self.branch_id)
        rec = pool[order[self.state.cursor[lane]]]
        self.state.cursor[lane] += 1
        return rec

    def _fill_lane(self, lane: str, quota: int, step: int, stage: dict) -> list[list[dict]]:
        """Draw, score and pack until this lane owns `quota` bins.

        The draw budget is proportional to what the lane owes, because a real
        loader cannot scan the corpus to fill one batch. That budget is what
        gives the protected floor something to do: a lane the proxy rejects
        heavily runs out of draws before it runs out of quota.
        """
        buf = [self.index.by_id[sid] for sid in self.state.buffer[lane]]
        rejected: list[tuple[dict, dict]] = []
        budget = max(self.min_draws, quota * self.draws_per_bin)
        draws = 0

        def bins_of(items):
            return best_fit_decreasing(items, self.seq_len, self.eos_between,
                                       self.max_open_bins) if items else []

        # in a later stage, give deferred samples a second hearing first
        if stage["name"] != self.schedule.stages[0]["name"] and self.state.deferred[lane]:
            revived = self.state.deferred[lane][:quota]
            self.state.deferred[lane] = self.state.deferred[lane][len(revived):]
            for sid in revived:
                rec = self.index.by_id[sid]
                d = self.opus.decide(rec, step, stage)
                d["reason"] = f"deferred_reconsidered({d['reason']})"
                if d["status"] == "deferred":            # a second look accepts it
                    d["status"], d["reason"] = "accepted", "deferred_reconsidered_accepted"
                self.decisions.append(d)
                if d["status"] == "accepted":
                    self.opus.commit_accepted(rec)
                    buf.append(rec)

        while len(bins_of(buf)) < quota and draws < budget:
            rec = self._next_candidate(lane)
            draws += 1
            if rec is None:
                break
            decision = self.opus.decide(rec, step, stage)
            self.decisions.append(decision)
            if decision["status"] == "accepted":
                self.opus.commit_accepted(rec)
                buf.append(rec)
            elif decision["status"] == "deferred":
                self.state.deferred[lane].append(rec["sample_id"])
            else:
                rejected.append((rec, decision))

        # protected floor: a lane at its floor may not be starved by the proxy
        floors = self.schedule.floor_bins(step)
        if len(bins_of(buf)) < quota and floors.get(lane, 0) > 0:
            for rec, decision in sorted(rejected, key=lambda rd: -rd[1]["score"]):
                if len(bins_of(buf)) >= quota:
                    break
                override = self.opus.override_for_floor(decision, floors[lane])
                self.decisions.append(override)
                self.opus.commit_accepted(rec)
                buf.append(rec)

        bins = bins_of(buf)
        taken, kept_ids = bins[:quota], []
        used = {s["sample_id"] for b in taken for s in b}
        for rec in buf:
            if rec["sample_id"] not in used:
                kept_ids.append(rec["sample_id"])
        self.state.buffer[lane] = kept_ids
        return taken

    # -- one step ----------------------------------------------------------- #

    def plan_step(self, step: int) -> dict:
        if step != self.state.step:
            raise RuntimeError(f"planner is at step {self.state.step}, asked for {step}")
        self.decisions = []
        stage = self.schedule.stage_for_step(step)
        quota = self.schedule.quota_for_step(step)

        placements: list[tuple[str, list[dict]]] = []
        for lane in sorted(quota):
            if quota[lane] <= 0:
                continue
            for b in self._fill_lane(lane, quota[lane], step, stage):
                placements.append((lane, b))

        # deterministic in-batch ordering
        order = stable_permutation(len(placements), self.seed, self.branch_id, "order", step)
        placements = [placements[i] for i in order]

        bins = []
        for i, (lane, items) in enumerate(placements):
            bins.append(build_bin(
                bin_id=f"{self.branch_id}/{step:05d}/bin{i:02d}", lane=lane, placed=items,
                token_reader=self.index.cache.read, seq_len=self.seq_len,
                eos_id=self.eos_id, pad_id=self.pad_id, eos_between=self.eos_between,
                reset_position_ids=self.reset_pos))

        actual = _count_lanes(bins)
        shortfall = {ln: quota[ln] - actual.get(ln, 0) for ln in quota
                     if quota[ln] > actual.get(ln, 0)}
        overrides = [d for d in self.decisions if d.get("protected_floor_override")]

        batch_id = f"{self.branch_id}/step{step:05d}"
        batch = {
            "batch_id": batch_id,
            "branch_id": self.branch_id,
            "step": step,
            "stage": stage["name"],
            "quota": quota,
            "actual_bins": actual,
            "quota_shortfall": shortfall,
            "protected_floor_overrides": len(overrides),
            "override_lanes": sorted({d["lane"] for d in overrides}),
            "bins": bins,
            "batch_hash": sha256_hex("|".join(b["content_hash"] for b in bins)),
            "decisions": list(self.decisions),
            "planner_fingerprint": self.state.fingerprint(),
            "n_bins": len(bins),
            "content_tokens": sum(b["content_tokens"] for b in bins),
            "loss_tokens": sum(b["loss_tokens"] for b in bins),
            "pad_tokens": sum(b["pad_tokens"] for b in bins),
            "positions": len(bins) * self.seq_len,
            "repeat_pass": dict(self.state.epoch),
        }
        self.state.step = step + 1
        return batch

    # -- state -------------------------------------------------------------- #

    def state_dict(self) -> dict:
        return self.state.to_dict(self.opus.state_dict())

    def load_state_dict(self, d: dict) -> None:
        self.state = PlannerState.from_dict(d)
        self.opus.load_state_dict(d["opus"])

    def fast_forward(self, upto_step: int) -> None:
        """Re-derive planner state by replanning from the beginning."""
        while self.state.step < upto_step:
            self.plan_step(self.state.step)


def _count_lanes(bins: list[dict]) -> dict:
    out: dict[str, int] = {}
    for b in bins:
        out[b["lane"]] = out.get(b["lane"], 0) + 1
    return out


def to_tensors(batch: dict, device: str = "cpu"):
    """Stack a planned batch into the tensors the model consumes."""
    import torch
    from .packing import attention_4d
    ids = torch.tensor([b["input_ids"] for b in batch["bins"]], dtype=torch.long, device=device)
    loss_mask = torch.tensor([b["loss_mask"] for b in batch["bins"]], dtype=torch.long,
                             device=device)
    seg = torch.tensor([b["segment_ids"] for b in batch["bins"]], dtype=torch.long,
                       device=device)
    pos = torch.tensor([b["position_ids"] for b in batch["bins"]], dtype=torch.long,
                       device=device)
    attn = attention_4d(seg)
    return ids, loss_mask, seg, pos, attn


def batch_summary(batch: dict) -> dict:
    """The part of a batch that goes into reports (no token arrays)."""
    return {k: v for k, v in batch.items() if k not in ("bins", "decisions")} | {
        "bin_hashes": [b["content_hash"] for b in batch["bins"]],
        "sample_ids": [s["sample_id"] for b in batch["bins"] for s in b["samples"]],
    }
