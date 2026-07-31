"""Size every lane of the mixture against the data that actually exists.

Demand comes from the curriculum in mixture/v5_mixture.yaml. Supply comes from
inventory/datasets.csv. The point of the script is to make it impossible to hand
a large share to a lane with nothing behind it: each lane is reported as unique
tokens, epochs required, and the synthetic tokens we are committing to generate.

Writes reports/supply_check.md.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from lib_mixture import (
    Curriculum,
    load_inventory,
    load_mixture,
    validate,
    ROOT,
)

# The long-context lane is a VIEW over other lanes: a 32k sequence of repo-level
# code is code tokens. Its demand is therefore charged back to the lanes it is
# drawn from, so nothing is counted twice.
VIEW_LANES = {
    "longctx": {"code": 0.45, "web": 0.35, "math": 0.10, "indic": 0.10},
}

GPU_PEAK_TFLOPS = 989.0    # H100 SXM, BF16 dense
ACHIEVED_MFU = 0.35        # batched inference, realistic rather than peak

# Generating a token is not one price. A rejection-sampled reasoning trace from a
# 32B teacher costs ~16x an 8B translation pass, because you throw most of it away.
GEN_MODEL = {
    #  lane:      (teacher params B, samples generated per token kept)
    "indic":      (8.0, 1.2),
    "reasoning":  (32.0, 4.0),
    "agentic":    (32.0, 5.0),   # only ~1 in 5 rollouts passes its tests
    "longctx":    (8.0, 1.2),
    "instruct":   (8.0, 1.5),
}


def gen_gpu_hours(lane: str, tokens_b: float) -> float:
    """Forward-pass cost of generating `tokens_b` billion KEPT tokens."""
    params_b, oversample = GEN_MODEL.get(lane, (8.0, 1.0))
    flops = 2.0 * params_b * 1e9 * tokens_b * 1e9 * oversample
    return flops / (GPU_PEAK_TFLOPS * 1e12 * ACHIEVED_MFU) / 3600.0


# What the anneal reserve is actually made of, per lane. Stating this stops the
# reserve from being a number with no material behind it.
RESERVE_SOURCE = {
    "web": "top-decile FineWeb-Edu / Nemotron-CC-HQ (Tier B, quality-filtered)",
    "code": "quality-filtered Stack v2 + CommitPackFT patch pairs (Tier A 0.45B)",
    "math": "MegaMath-Pro / FineMath-4plus top decile + competition solutions",
    "reasoning": "self-distilled traces with checkable final answers (planned)",
    "indic": "Sangraha Verified, native-speaker reviewed (Tier A, on hand)",
    "agentic": "self-generated EXECUTION-VERIFIED rollouts (planned) - cannot be bought",
    "longctx": "PG-19 + arXiv full texts + synthesised multi-doc tasks",
    "instruct": "n/a",
}


def main() -> int:
    mix = load_mixture()
    problems = validate(mix)
    inv = load_inventory()
    cur = Curriculum(mix)
    lanes = list(mix["lanes"].keys())

    demand = cur.demand_b(include_anneal=True)
    anneal_demand = {ln: float(w) * float(mix["anneal"]["tokens_b"])
                     for ln, w in mix["anneal"]["weights"].items()}

    # charge view-lane demand back to its parent lanes
    charged = dict(demand)
    charge_detail: dict[str, dict[str, float]] = defaultdict(dict)
    for view, parents in VIEW_LANES.items():
        for parent, frac in parents.items():
            amount = demand[view] * frac
            charged[parent] = charged.get(parent, 0.0) + amount
            charge_detail[parent][view] = amount

    # supply
    existing = defaultdict(float)
    planned = defaultdict(float)
    tier_a = defaultdict(float)
    views = defaultdict(float)
    noncommercial = []
    for r in inv:
        if r.planned:
            planned[r.lane] += r.tokens_b
            continue
        if r.overlap_frac >= 1.0:
            views[r.lane] += r.tokens_b
            continue
        existing[r.lane] += r.net_tokens_b
        if r.tier.startswith("A"):
            tier_a[r.lane] += r.net_tokens_b
        if "NC" in r.license or "non-commercial" in r.license.lower():
            noncommercial.append(r)

    caps = mix["repetition"]["max_epochs_by_tier"]
    hard_cap = float(mix["repetition"]["hard_cap_any_lane"])

    # per-lane epoch cap = supply-weighted average of the tier caps of its sources
    lane_cap: dict[str, float] = {}
    for ln in lanes:
        num = den = 0.0
        for r in inv:
            if r.lane != ln or r.overlap_frac >= 1.0:
                continue
            tier_key = r.tier[0] if r.tier and r.tier[0] in caps else "B"
            w = r.tokens_b if r.planned else r.net_tokens_b
            num += float(caps[tier_key]) * w
            den += w
        lane_cap[ln] = min(num / den if den else hard_cap, hard_cap)

    rows = []
    for ln in lanes:
        d = charged[ln] if ln not in VIEW_LANES else demand[ln]
        sup = existing[ln] + (views[ln] if ln in VIEW_LANES else 0.0)
        pl = planned[ln]
        total_sup = sup + pl
        epochs_existing = d / sup if sup > 0 else float("inf")
        epochs_total = d / total_sup if total_sup > 0 else float("inf")
        cap = lane_cap[ln]
        if epochs_total <= 1.0:
            status = "OK (under one epoch)"
        elif epochs_total <= cap:
            status = f"REPEAT x{epochs_total:.2f} (cap {cap:.1f})"
        else:
            status = f"**OVER CAP x{epochs_total:.2f} > {cap:.1f}**"
        if ln in VIEW_LANES:
            status += " — view lane"
        rows.append({
            "lane": ln,
            "demand_b": d,
            "anneal_b": anneal_demand.get(ln, 0.0),
            "existing_unique_b": sup,
            "planned_synth_b": pl,
            "epochs_existing": epochs_existing,
            "epochs_with_synth": epochs_total,
            "tier_a_b": tier_a[ln],
            "epoch_cap": cap,
            "reserve_b": float(mix["reserve"]["tokens_b"].get(ln, 0.0)),
            "status": status,
        })

    # Indic tier accounting
    indic_main = cur.demand_b(include_anneal=False)["indic"]
    tiers = mix["indic_tiers"]["main_run"]
    indic_rows = []
    tier_supply = {
        "verified": next(r.net_tokens_b for r in inv if r.dataset.startswith("Sangraha Verified")),
        "unverified": next(r.net_tokens_b for r in inv if r.dataset.startswith("Sangraha Unverified")),
        "translated": (next(r.net_tokens_b for r in inv if r.dataset.startswith("Sangraha Synthetic"))
                       + next(r.net_tokens_b for r in inv if r.dataset.startswith("IndicAlign"))),
        "synthetic": next(r.tokens_b for r in inv if r.dataset.startswith("V5 self-generated Indic")),
    }
    reserved_verified = float(mix["reserve"]["tokens_b"]["indic"])
    for t, frac in tiers.items():
        need = indic_main * float(frac)
        avail = tier_supply[t]
        if t == "verified":
            avail = max(avail - reserved_verified, 0.0)   # the anneal reserve is fenced off
        tier_cap = {"verified": caps["A"], "unverified": caps["B"],
                    "translated": caps["C"], "synthetic": caps["C"]}[t]
        indic_rows.append({
            "tier": t,
            "share_of_lane": float(frac),
            "main_run_need_b": need,
            "available_unique_b": avail,
            "epochs": need / avail if avail > 0 else float("inf"),
            "cap": float(tier_cap),
        })

    # sample-vs-token illustration, computed from the inventory
    sample_rows = [r for r in inv if r.samples_m]
    sample_rows.sort(key=lambda r: -(r.samples_m or 0))
    by_samples = sample_rows[:5]
    by_tokens = sorted([r for r in inv if r.samples_m], key=lambda r: -r.tokens_b)[:5]

    total_planned = sum(planned.values())

    # ---------------------------------------------------------------- report
    L = []
    a = L.append
    a("# Supply check — V5 mixture vs. data that exists\n")
    a("_Generated by `scripts/supply_check.py`. Do not edit by hand._\n")
    a(f"Spec validation: **{'PASS' if not problems else 'FAIL'}**"
      + ("" if not problems else " — " + "; ".join(problems)) + "\n")
    a(f"Budget: **{mix['budget']['total_tokens_b']}B tokens** "
      f"({mix['budget']['main_tokens_b']}B main + {mix['budget']['anneal_tokens_b']}B anneal).\n")

    a("## Run-average mixture (derived from the curriculum, not asserted)\n")
    a("| lane | main-run avg | whole-budget avg | anneal |")
    a("|---|---:|---:|---:|")
    ra_main = cur.run_average(include_anneal=False)
    ra_all = cur.run_average(include_anneal=True)
    for ln in sorted(lanes, key=lambda l: -ra_main[l]):
        a(f"| {ln} | {ra_main[ln]*100:.2f}% | {ra_all[ln]*100:.2f}% | "
          f"{float(mix['anneal']['weights'].get(ln,0))*100:.0f}% |")
    a("")

    a("## Lane sizing against real supply\n")
    a("`demand` includes the long-context lane charged back to the lanes it is drawn from "
      f"({', '.join(f'{k} {v:.0%}' for k, v in VIEW_LANES['longctx'].items())}), so no token is counted twice.\n")
    a("| lane | demand (B) | of which anneal | usable supply (B) | planned synthetic (B) | epochs on existing | epochs incl. synthetic | epoch cap | Tier A (B) | reserve (B) | status |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        ee = "inf" if r["epochs_existing"] == float("inf") else f"{r['epochs_existing']:.2f}"
        et = "inf" if r["epochs_with_synth"] == float("inf") else f"{r['epochs_with_synth']:.2f}"
        a(f"| {r['lane']} | {r['demand_b']:.1f} | {r['anneal_b']:.1f} | {r['existing_unique_b']:.1f} | "
          f"{r['planned_synth_b']:.1f} | {ee} | {et} | {r['epoch_cap']:.1f} | {r['tier_a_b']:.2f} | "
          f"{r['reserve_b']:.0f} | {r['status']} |")
    a("")
    a(f"Long-context supply as a *view*: {views['longctx']:.0f}B tokens of parent-lane material already "
      "sits in documents long enough to fill 32k sequences, so the constraint on that lane is the "
      "length distribution of the shards, not the token count.\n")

    a("## Anneal reserve — what is it actually made of?\n")
    a("| lane | reserve (B) | Tier A on hand (B) | planned (B) | material | fundable? |")
    a("|---|---:|---:|---:|---|---|")
    for r in rows:
        if r["reserve_b"] <= 0:
            continue
        ln = r["lane"]
        pl = r["planned_synth_b"]
        if r["tier_a_b"] >= r["reserve_b"]:
            verdict = "yes — on hand"
        elif ln in ("web", "code", "math", "longctx"):
            verdict = "yes — quality-filtered Tier B"
        elif r["tier_a_b"] + pl >= r["reserve_b"]:
            verdict = "**only by building it**"
        else:
            verdict = "**NO — not fundable, cut the anneal share**"
        a(f"| {ln} | {r['reserve_b']:.0f} | {r['tier_a_b']:.2f} | {pl:.0f} | {RESERVE_SOURCE.get(ln,'')} | {verdict} |")
    a("")

    a("## Indic lane, split by provenance tier\n")
    a(f"Main-run Indic demand: **{indic_main:.1f}B tokens** "
      f"(+{anneal_demand['indic']:.0f}B in the anneal, drawn from the fenced reserve).\n")
    a("| tier | share of lane | main-run need (B) | unique supply available (B) | epochs | cap | verdict |")
    a("|---|---:|---:|---:|---:|---:|---|")
    for r in indic_rows:
        ep = "inf" if r["epochs"] == float("inf") else f"{r['epochs']:.2f}"
        ok = "ok" if r["epochs"] <= r["cap"] else "**over cap**"
        a(f"| {r['tier']} | {r['share_of_lane']*100:.1f}% | {r['main_run_need_b']:.1f} | "
          f"{r['available_unique_b']:.1f} | {ep} | {r['cap']:.1f} | {ok} |")
    a("")

    a("## The two currencies: samples vs. tokens\n")
    a("| ranked by samples | samples (M) | tokens (B) | | ranked by tokens | samples (M) | tokens (B) |")
    a("|---|---:|---:|---|---|---:|---:|")
    for i in range(5):
        l1, l2 = by_samples[i], by_tokens[i]
        a(f"| {l1.dataset[:38]} | {l1.samples_m:.2f} | {l1.tokens_b:.2f} | | "
          f"{l2.dataset[:38]} | {l2.samples_m:.3f} | {l2.tokens_b:.1f} |")
    a("")

    a("## Synthetic generation commitment\n")
    a(f"Total planned synthetic: **{total_planned:.1f}B tokens**. Cost is a forward-pass estimate at "
      f"{ACHIEVED_MFU:.0%} MFU on H100 SXM, including the samples thrown away by rejection sampling.\n")
    a("| lane | planned kept (B) | teacher | samples per kept token | est. H100-hours | 8xH100-node-days |")
    a("|---|---:|---:|---:|---:|---:|")
    th = 0.0
    for ln in lanes:
        if planned[ln] <= 0:
            continue
        params_b, over = GEN_MODEL.get(ln, (8.0, 1.0))
        h = gen_gpu_hours(ln, planned[ln])
        th += h
        a(f"| {ln} | {planned[ln]:.1f} | {params_b:.0f}B | {over:.1f}x | {h:,.0f} | {h/8/24:.1f} |")
    a(f"| **total** | **{total_planned:.1f}** | | | **{th:,.0f}** | **{th/8/24:.1f}** |")
    a("")
    a("The agentic figure counts model forward passes only. Each kept trajectory also needs a "
      "container build and a test run, which is CPU-bound and scheduled separately; that, not the "
      "GPU time, is the reason the agentic reserve is the riskiest item in this plan.\n")
    if noncommercial:
        a("## Licence flags\n")
        for r in noncommercial:
            a(f"- `{r.dataset}` — {r.license}. Eval/proxy only; excluded from the production run.")
        a("")

    text = "\n".join(L)
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "supply_check.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(ROOT, "reports", "supply_check.json"), "w", encoding="utf-8") as fh:
        json.dump({"lanes": rows, "indic_tiers": indic_rows,
                   "run_average_main": ra_main, "problems": problems}, fh, indent=2)
    print(text)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
