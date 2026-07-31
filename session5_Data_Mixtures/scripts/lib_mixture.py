"""Shared loader / sampler for the V5 mixture spec.

Everything downstream (supply check, curriculum simulation, proxy training run)
reads the same YAML through this module, so the plan and the code cannot drift
apart. No number in the README is typed by hand; all of it comes from here.
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass, field

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIXTURE_PATH = os.path.join(ROOT, "mixture", "v5_mixture.yaml")
INVENTORY_PATH = os.path.join(ROOT, "inventory", "datasets.csv")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_mixture(path: str = MIXTURE_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def lane_ids(mix: dict) -> list[str]:
    return list(mix["lanes"].keys())


@dataclass
class InventoryRow:
    lane: str
    dataset: str
    source: str
    tokens_b: float
    samples_m: float | None
    license: str
    tier: str
    langs: str
    overlap_frac: float
    keep_frac: float
    confidence: str
    notes: str

    @property
    def planned(self) -> bool:
        return self.confidence == "planned"

    @property
    def net_tokens_b(self) -> float:
        """Usable tokens this row contributes.

        overlap_frac removes what another row already counts; keep_frac is the
        fraction expected to survive our own quality filter. Both are stated per
        row in the inventory so a reviewer can argue with the individual number
        instead of the total.
        """
        return self.tokens_b * (1.0 - self.overlap_frac) * self.keep_frac


def load_inventory(path: str = INVENTORY_PATH) -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(
                InventoryRow(
                    lane=r["lane"].strip(),
                    dataset=r["dataset"].strip(),
                    source=r["source"].strip(),
                    tokens_b=float(r["tokens_b"] or 0),
                    samples_m=float(r["samples_m"]) if r["samples_m"].strip() else None,
                    license=r["license"].strip(),
                    tier=r["tier"].strip(),
                    langs=r["langs"].strip(),
                    overlap_frac=float(r["overlap_frac"] or 0),
                    keep_frac=float(r.get("keep_frac") or 1.0),
                    confidence=r["confidence"].strip(),
                    notes=r["notes"].strip(),
                )
            )
    return rows


# --------------------------------------------------------------------------- #
# curriculum
# --------------------------------------------------------------------------- #
@dataclass
class Curriculum:
    """Stage schedule with linear warmup blending across every transition."""

    mix: dict
    lanes: list[str] = field(init=False)
    stages: list[dict] = field(init=False)
    bounds: list[tuple[float, float]] = field(init=False)

    def __post_init__(self):
        self.lanes = lane_ids(self.mix)
        self.stages = list(self.mix["curriculum"]["stages"])
        anneal = dict(self.mix["anneal"])
        anneal["id"] = "anneal"
        self.stages.append(anneal)
        self.bounds = []
        pos = 0.0
        for st in self.stages:
            self.bounds.append((pos, pos + float(st["tokens_b"])))
            pos += float(st["tokens_b"])
        self.total_b = pos

    def stage_index_at(self, tok_b: float) -> int:
        for i, (lo, hi) in enumerate(self.bounds):
            if lo <= tok_b < hi:
                return i
        return len(self.stages) - 1

    def band_for(self, i: int) -> float:
        """Warmup band (B tokens) for the transition INTO stage i."""
        if i <= 0:
            return 0.0
        key = f"{self.stages[i-1]['id']}->{self.stages[i]['id']}"
        bands = self.mix["curriculum"].get("warmup_bands") or {}
        return float(bands.get(key, self.mix["curriculum"]["warmup_band_tokens_b"]))

    def raw_weights(self, i: int) -> dict[str, float]:
        w = self.stages[i]["weights"]
        return {ln: float(w.get(ln, 0.0)) for ln in self.lanes}

    def weights_at(self, tok_b: float) -> dict[str, float]:
        """Target mixture at a token position, with transition blending applied.

        A transition is spread over `warmup_band_tokens_b`, centred on the stage
        boundary. This is V4's mitigation for the 150x gradient-norm spike: the
        distribution never moves in one hard step.
        """
        i = self.stage_index_at(tok_b)
        band = self.band_for(i)
        lo, _hi = self.bounds[i]
        cur = self.raw_weights(i)
        if i == 0 or band <= 0:
            return cur
        # inside the leading half-band of stage i -> blend from the previous stage
        half = band / 2.0
        if tok_b < lo + half:
            prev = self.raw_weights(i - 1)
            # alpha goes 0.5 -> 1.0 across the leading half band
            alpha = 0.5 + 0.5 * ((tok_b - (lo - half)) / band)
            alpha = min(max(alpha, 0.0), 1.0)
            return {ln: prev[ln] * (1 - alpha) + cur[ln] * alpha for ln in self.lanes}
        # inside the trailing half-band -> blend toward the next stage
        nxt_lo = self.bounds[i][1]
        nxt_band = self.band_for(i + 1) if i + 1 < len(self.stages) else 0.0
        half = nxt_band / 2.0
        if nxt_band > 0 and tok_b > nxt_lo - half and i + 1 < len(self.stages):
            nxt = self.raw_weights(i + 1)
            alpha = ((tok_b - (nxt_lo - half)) / nxt_band)
            alpha = min(max(alpha, 0.0), 1.0) * 0.5
            return {ln: cur[ln] * (1 - alpha) + nxt[ln] * alpha for ln in self.lanes}
        return cur

    def run_average(self, include_anneal: bool = False) -> dict[str, float]:
        """Token-weighted average mixture over the run (blending is mass-neutral)."""
        acc = {ln: 0.0 for ln in self.lanes}
        total = 0.0
        for i, st in enumerate(self.stages):
            if st["id"] == "anneal" and not include_anneal:
                continue
            t = float(st["tokens_b"])
            w = self.raw_weights(i)
            for ln in self.lanes:
                acc[ln] += w[ln] * t
            total += t
        return {ln: acc[ln] / total for ln in self.lanes}

    def demand_b(self, include_anneal: bool = True) -> dict[str, float]:
        """Absolute token demand per lane, in billions."""
        acc = {ln: 0.0 for ln in self.lanes}
        for i, st in enumerate(self.stages):
            if st["id"] == "anneal" and not include_anneal:
                continue
            t = float(st["tokens_b"])
            w = self.raw_weights(i)
            for ln in self.lanes:
                acc[ln] += w[ln] * t
        return acc


# --------------------------------------------------------------------------- #
# floors + selector
# --------------------------------------------------------------------------- #
def floor_map(mix: dict) -> dict[str, float]:
    """Numeric lane floors only (the block also carries boolean policy switches)."""
    return {
        k: float(v)
        for k, v in mix["protected_floors"].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def apply_floors(weights: dict[str, float], floors: dict[str, float]) -> dict[str, float]:
    """Raise any lane below its floor, renormalising the unprotected lanes.

    Protected mass is taken proportionally from lanes that are above their own
    floor, so a floor can never be paid for by another protected lane.
    """
    w = dict(weights)
    deficit = 0.0
    for ln, f in floors.items():
        if w.get(ln, 0.0) < f:
            deficit += f - w.get(ln, 0.0)
            w[ln] = f
    if deficit <= 0:
        return w
    donors = {ln: v for ln, v in w.items() if ln not in floors and v > 0}
    donor_mass = sum(donors.values())
    if donor_mass <= 0:
        raise ValueError("no unprotected mass left to fund the floors")
    for ln, v in donors.items():
        w[ln] = v - deficit * (v / donor_mass)
    return w


@dataclass
class OpusSelector:
    """Simulation of the OPUS batch selector, faithful to the parts that matter.

    Real OPUS scores a candidate batch by estimating how useful its update would
    be against a stable proxy direction. What matters for a mixture decision is
    the failure mode: the proxy is built from what the model is already good at,
    so it systematically under-scores lanes that look unlike its training
    distribution (Indic, agentic). `proxy_bias` models exactly that.
    """

    keep_fraction: float = 0.40
    proxy_bias: dict[str, float] = field(default_factory=dict)
    per_lane_zscore: bool = True
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    def score(self, lane: str) -> float:
        base = self.rng.gauss(0.0, 1.0)
        return base + self.proxy_bias.get(lane, 0.0)

    def select(self, candidates: list[str]) -> list[str]:
        scored = [(self.score(ln), i, ln) for i, ln in enumerate(candidates)]
        if self.per_lane_zscore:
            # normalise within lane: the selector may rank Indic against Indic,
            # but it may not rank Indic against English on a proxy built from English.
            by_lane: dict[str, list[float]] = {}
            for s, _i, ln in scored:
                by_lane.setdefault(ln, []).append(s)
            stats = {}
            for ln, xs in by_lane.items():
                mu = sum(xs) / len(xs)
                var = sum((x - mu) ** 2 for x in xs) / max(len(xs) - 1, 1)
                stats[ln] = (mu, max(var, 1e-9) ** 0.5)
            scored = [((s - stats[ln][0]) / stats[ln][1], i, ln) for s, i, ln in scored]
        scored.sort(key=lambda t: -t[0])
        k = max(1, int(round(self.keep_fraction * len(candidates))))
        return [ln for _s, _i, ln in scored[:k]]


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate(mix: dict) -> list[str]:
    """Return a list of problems. Empty list == the spec is internally consistent."""
    problems: list[str] = []
    lanes = lane_ids(mix)
    floors = floor_map(mix)

    stages = list(mix["curriculum"]["stages"]) + [dict(mix["anneal"], id="anneal")]
    for st in stages:
        s = sum(float(v) for v in st["weights"].values())
        if abs(s - 1.0) > 1e-6:
            problems.append(f"stage {st['id']}: weights sum to {s:.6f}, not 1.0")
        for ln in st["weights"]:
            if ln not in lanes:
                problems.append(f"stage {st['id']}: unknown lane '{ln}'")
        d = sum(float(v) for v in st.get("difficulty", {}).values())
        if st.get("difficulty") and abs(d - 1.0) > 1e-6:
            problems.append(f"stage {st['id']}: difficulty bands sum to {d:.3f}")

    # the curriculum must never plan a lane below its own protected floor:
    # if it did, the floor would be silently rewriting the curriculum.
    for st in mix["curriculum"]["stages"]:
        for ln, f in floors.items():
            if float(st["weights"].get(ln, 0.0)) + 1e-9 < f:
                problems.append(
                    f"stage {st['id']}: {ln}={st['weights'].get(ln, 0.0):.3f} is below its floor {f:.3f}"
                )

    tot_stage = sum(float(s["tokens_b"]) for s in mix["curriculum"]["stages"])
    if abs(tot_stage - mix["budget"]["main_tokens_b"]) > 1e-6:
        problems.append(
            f"stage tokens sum to {tot_stage}B but budget.main_tokens_b is {mix['budget']['main_tokens_b']}B"
        )
    if abs(mix["budget"]["main_tokens_b"] + mix["budget"]["anneal_tokens_b"] - mix["budget"]["total_tokens_b"]) > 1e-6:
        problems.append("main + anneal != total budget")
    if abs(float(mix["anneal"]["tokens_b"]) - float(mix["budget"]["anneal_tokens_b"])) > 1e-6:
        problems.append("anneal.tokens_b disagrees with budget.anneal_tokens_b")

    for key in ("main_run", "anneal"):
        s = sum(float(v) for v in mix["indic_tiers"][key].values())
        if abs(s - 1.0) > 1e-6:
            problems.append(f"indic_tiers.{key} sums to {s:.4f}, not 1.0")
    tr = float(mix["indic_tiers"]["main_run"]["translated"])
    cap = float(mix["indic_tiers"]["caps"]["translated_max_frac_of_lane"])
    if tr > cap + 1e-9:
        problems.append(f"translated Indic share {tr:.3f} exceeds cap {cap:.3f}")
    ls = sum(float(v) for v in mix["indic_tiers"]["language_split"].values())
    if abs(ls - 1.0) > 1e-6:
        problems.append(f"indic language_split sums to {ls:.4f}, not 1.0")

    rb = sum(float(v["share"]) for v in mix["reasoning_bands"].values() if isinstance(v, dict) and "share" in v)
    if abs(rb - 1.0) > 1e-6:
        problems.append(f"reasoning_bands shares sum to {rb:.4f}, not 1.0")
    ash = sum(float(v) for v in mix["reasoning_bands"]["anneal_shift"].values())
    if abs(ash - 1.0) > 1e-6:
        problems.append(f"reasoning_bands.anneal_shift sums to {ash:.4f}, not 1.0")

    # the reserve must actually cover what the anneal plans to spend
    anneal_b = float(mix["anneal"]["tokens_b"])
    for ln, w in mix["anneal"]["weights"].items():
        need = float(w) * anneal_b
        have = float(mix["reserve"]["tokens_b"].get(ln, 0.0))
        if have + 1e-9 < need:
            problems.append(f"reserve.{ln}={have}B does not cover anneal demand {need:.1f}B")
    return problems


if __name__ == "__main__":
    m = load_mixture()
    probs = validate(m)
    if probs:
        print("SPEC PROBLEMS:")
        for p in probs:
            print("  -", p)
        raise SystemExit(1)
    c = Curriculum(m)
    print("spec OK")
    print("run-average (main run):")
    for ln, v in sorted(c.run_average().items(), key=lambda kv: -kv[1]):
        print(f"  {ln:9s} {v*100:5.2f}%")
