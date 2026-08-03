"""Stage 7 -- OPUS: the selection policy that sits inside the data path.

OPUS scores every candidate sample before it is allowed into a bin and records
what it decided and why. Four outcomes:

    accept    the sample enters a loss-bearing batch
    reject    hard rule failed, or the proxy score fell below the threshold
    defer     borderline, or a duplicate -- parked in the deferral pool and
              reconsidered in a later curriculum stage
    override  the lane was at its protected floor, so a rejected candidate is
              rescued and the rescue is recorded as a protected-floor override

The critical design constraint: **every feature is derived from content, never
from model state.** A score that depended on the live model could not be
recomputed during replay without re-running training, and the whole replay
guarantee would collapse. So the proxy uses:

    utilisation    how much of a bin the sample fills
    novelty        1 - MinHash overlap with recently accepted samples
    length_fit     penalises stub spans that waste a slot
    lane_affinity  the lane's share in the current curriculum stage

Rejected data is not destroyed. Rejections and deferrals are written to the OPUS
ledger with their reasons, so a sample OPUS found dull at step 4 can be found
again -- by lane, by reason, by score -- when planning the next corpus.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .common import sha256_hex, stable_unit

SKETCH_SIZE = 8          # MinHash signature width
SKETCH_NGRAM = 8         # token n-gram used for the sketch
NOVELTY_WINDOW = 96      # how many recently accepted sketches novelty sees


def token_sketch(tokens: np.ndarray, size: int = SKETCH_SIZE,
                 n: int = SKETCH_NGRAM) -> list[int]:
    """MinHash signature of a token span: the `size` smallest n-gram hashes."""
    if len(tokens) < n:
        n = max(2, len(tokens))
    t = tokens.astype(np.uint64)
    if len(t) < n:
        return []
    h = np.zeros(len(t) - n + 1, dtype=np.uint64)
    for k in range(n):
        h = h * np.uint64(1000003) + t[k:len(t) - n + 1 + k]
    h = np.unique(h)
    return [int(x) for x in np.sort(h)[:size]]


def symbol_ratio(text: str) -> float:
    """Share of characters that are neither alphanumeric nor ordinary space.

    Stands in for the quality/toxicity screen a real OPUS would run: a span that
    is mostly punctuation, markup or table glyphs is not worth a training slot.
    """
    if not text:
        return 1.0
    bad = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in ".,'\"-?!:;()"))
    return bad / len(text)


class OpusPolicy:
    def __init__(self, cfg, log=None) -> None:
        o = cfg.opus
        self.proxy_version = o.proxy_version
        self.weights = dict(o.weights)
        self.accept_threshold = float(o.accept_threshold)
        self.defer_threshold = float(o.defer_threshold)
        self.symbol_reject_ratio = float(o.symbol_reject_ratio)
        self.duplicate_action = o.duplicate_action
        self.seq_len = int(cfg.packing.seq_len)
        self.log = log
        self.reset_state()

    # -- state (part of the dataloader state, restored on resume) ----------- #

    def reset_state(self) -> None:
        self.seen_token_hashes: set[str] = set()
        self.recent: deque[int] = deque(maxlen=NOVELTY_WINDOW * SKETCH_SIZE)
        self._recent_set: set[int] = set()
        self.counters: dict[str, int] = {}

    def state_dict(self) -> dict:
        return {"seen_token_hashes": sorted(self.seen_token_hashes),
                "recent": list(self.recent), "counters": dict(self.counters)}

    def load_state_dict(self, state: dict) -> None:
        self.seen_token_hashes = set(state.get("seen_token_hashes", []))
        self.recent = deque(state.get("recent", []), maxlen=NOVELTY_WINDOW * SKETCH_SIZE)
        self._recent_set = set(self.recent)
        self.counters = dict(state.get("counters", {}))

    def state_hash(self) -> str:
        return sha256_hex("|".join([
            str(len(self.seen_token_hashes)), str(len(self.recent)),
            sha256_hex(",".join(sorted(self.seen_token_hashes)[-64:])),
            sha256_hex(",".join(str(x) for x in list(self.recent)[-64:]))]))

    # -- scoring ------------------------------------------------------------ #

    def novelty(self, sketch: list[int]) -> float:
        if not sketch:
            return 0.0
        overlap = sum(1 for x in sketch if x in self._recent_set)
        return 1.0 - overlap / len(sketch)

    def features(self, sample: dict, stage: dict) -> dict:
        n = int(sample["n_tokens"])
        util = min(1.0, n / (self.seq_len - 1))
        return {
            "utilisation": round(util, 6),
            "novelty": round(self.novelty(sample.get("sketch", [])), 6),
            "length_fit": round(min(1.0, n / 64.0), 6),
            "lane_affinity": round(float(stage["shares"].get(sample["lane"], 0.0)), 6),
        }

    def score(self, features: dict) -> float:
        return round(sum(self.weights[k] * features[k] for k in self.weights), 6)

    # -- decision ----------------------------------------------------------- #

    def decide(self, sample: dict, step: int, stage: dict, quota_pressure: bool = False) -> dict:
        feats = self.features(sample, stage)
        score = self.score(feats)
        candidate_id = f"cand:{step:05d}:{sample['sample_id']}"

        status, reason = "accepted", "score_above_threshold"
        if sample.get("symbol_ratio", 0.0) > self.symbol_reject_ratio:
            status, reason = "rejected", "symbol_heavy_content"
        elif sample.get("token_hash") in self.seen_token_hashes:
            status = "deferred" if self.duplicate_action == "defer" else "rejected"
            reason = "duplicate_token_span"
        elif score < self.defer_threshold:
            status, reason = "rejected", _weakest(feats, self.weights)
        elif score < self.accept_threshold:
            status, reason = "deferred", "borderline_proxy_score"

        decision = {
            "candidate_id": candidate_id,
            "sample_id": sample["sample_id"],
            "shard_id": sample["shard_id"],
            "doc_id": sample["doc_id"],
            "lane": sample["lane"],
            "step": step,
            "stage": stage["name"],
            "proxy_version": self.proxy_version,
            "score": score,
            "features": feats,
            "status": status,
            "reason": reason,
            "protected_floor_override": False,
            "quota_pressure": bool(quota_pressure),
            "effective_tokens": int(sample["n_tokens"]) if status == "accepted" else 0,
        }
        self.counters[status] = self.counters.get(status, 0) + 1
        return decision

    def override_for_floor(self, decision: dict, floor_bins: int) -> dict:
        """Rescue a rejected candidate because its lane sits at its floor."""
        d = dict(decision)
        d["status"] = "accepted"
        d["protected_floor_override"] = True
        d["reason"] = f"protected_floor_override(was:{decision['reason']})"
        d["floor_bins"] = floor_bins
        d["effective_tokens"] = 0  # filled in by the caller once packed
        self.counters["protected_floor_override"] = \
            self.counters.get("protected_floor_override", 0) + 1
        return d

    def commit_accepted(self, sample: dict) -> None:
        """Fold an accepted sample into the novelty and duplicate state."""
        if sample.get("token_hash"):
            self.seen_token_hashes.add(sample["token_hash"])
        for x in sample.get("sketch", []):
            if len(self.recent) == self.recent.maxlen:
                self._recent_set.discard(self.recent[0])
            self.recent.append(x)
            self._recent_set.add(x)

    # -- reporting ---------------------------------------------------------- #

    def summary(self, decisions: list[dict]) -> dict:
        by_status: dict[str, int] = {}
        by_lane: dict[str, dict[str, int]] = {}
        by_reason: dict[str, int] = {}
        scores = []
        for d in decisions:
            by_status[d["status"]] = by_status.get(d["status"], 0) + 1
            lane = by_lane.setdefault(d["lane"], {})
            lane[d["status"]] = lane.get(d["status"], 0) + 1
            by_reason[d["reason"]] = by_reason.get(d["reason"], 0) + 1
            scores.append(d["score"])
        total = max(1, len(decisions))
        return {
            "candidates": len(decisions),
            "by_status": by_status,
            "by_lane": by_lane,
            "by_reason": by_reason,
            "protected_floor_overrides": sum(1 for d in decisions
                                             if d.get("protected_floor_override")),
            "acceptance_rate": round(by_status.get("accepted", 0) / total, 6),
            "rejection_rate": round(by_status.get("rejected", 0) / total, 6),
            "deferral_rate": round(by_status.get("deferred", 0) / total, 6),
            "mean_score": round(float(np.mean(scores)) if scores else 0.0, 6),
            "proxy_version": self.proxy_version,
            "weights": self.weights,
            "thresholds": {"accept": self.accept_threshold, "defer": self.defer_threshold},
        }


def _weakest(features: dict, weights: dict) -> str:
    """Name the feature that cost the candidate the most, for the audit trail."""
    lane = min(features, key=lambda k: features[k] * weights.get(k, 0.0) + 1e-9 * stable_unit(k))
    return f"low_{lane}"
