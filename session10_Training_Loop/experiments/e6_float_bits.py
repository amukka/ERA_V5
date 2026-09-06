"""Deliverable 6 — 0.1 in fp32, bf16 and fp8 E4M3, showing the bits, then
saying which one I would train in and why.

The encodings below are derived from the decimal by hand, in exact rational
arithmetic (``src/floats.py``), and only *then* compared against the bit
pattern torch actually stores.  They agree, which is the point of doing it by
hand: if they had not, one of the two would have been wrong and it would matter
which.

The "which would I train in" answer is not a preference.  It is settled by
measuring the size of a real weight update in this session's own run and
comparing it against each format's spacing at the weight's magnitude.
"""

from __future__ import annotations

import copy
import sys, pathlib
from fractions import Fraction
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from src.data import Sampler, load_documents
from src.floats import (BF16, FP4_E2M1, FP8_E4M3, FP8_E5M2, FP16, FP32,
                        binary_expansion, encode, torch_bits)
from src.loop import pick_device, set_seed, token_losses
from src.model import Config, TinyGPT
from src.plots import CRITICAL, GOOD, INK_2, MUTED, S1, S2, S3, finish, plt
from src.report import machine, save_json, save_text, table

X = 0.1
TORCH_DTYPE = {
    "fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16,
    "fp8 E4M3": torch.float8_e4m3fn, "fp8 E5M2": torch.float8_e5m2,
}


def encode_all():
    out = []
    for fmt in (FP32, FP16, BF16, FP8_E4M3, FP8_E5M2, FP4_E2M1):
        enc = encode(X, fmt)
        hw = torch_bits(X, TORCH_DTYPE[fmt.name]) if fmt.name in TORCH_DTYPE \
            else None
        out.append({
            "format": fmt.name,
            "bits_total": fmt.total_bits,
            "exp_bits": fmt.exp_bits,
            "mant_bits": fmt.mant_bits,
            "bias": fmt.bias,
            "sign": enc.sign,
            "exp_field": enc.exp_field,
            "exp_field_bits": f"{enc.exp_field:0{fmt.exp_bits}b}",
            "unbiased_exp": enc.unbiased_exp,
            "mantissa": enc.mantissa,
            "mantissa_bits": f"{enc.mantissa:0{fmt.mant_bits}b}",
            "significand": enc.significand_binary,
            "bits": enc.bits,
            "hex": enc.hex,
            "value": float(enc.value),
            "value_exact": str(enc.value),
            "abs_error": float(enc.abs_error),
            "rel_error": float(enc.rel_error),
            "ulp_at_0.1": float(enc.ulp),
            "relative_ulp": (float(enc.ulp / enc.value) if enc.value
                             else float("inf")),
            "underflows_to_zero": enc.value == 0,
            "hardware_bits": hw,
            "matches_hardware": (hw == enc.bits_flat) if hw else None,
            "note": fmt.note,
        })
    return out


def real_update_sizes():
    """How big is a weight update, really, in this session's own loop?"""
    set_seed(0)
    device = pick_device()
    cfg = Config()
    model = TinyGPT(cfg).to(device)
    docs, _ = load_documents()
    sampler = Sampler(docs["train"], 8, max_len=256, min_len=32,
                      mode="bucket", seed=5)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                            weight_decay=0.1)

    # warm the optimiser moments so the reported update is a steady-state one,
    # not the first-step transient
    for _ in range(30):
        loss_sum, n = token_losses(model, sampler.batch().to(device))
        (loss_sum / n).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    before = copy.deepcopy(
        {k: v.detach().clone() for k, v in model.named_parameters()})
    loss_sum, n = token_losses(model, sampler.batch().to(device))
    (loss_sum / n).backward()
    opt.step()

    rel, lost_bf16, lost_fp8, n_total = [], 0, 0, 0
    for k, p in model.named_parameters():
        w = before[k].detach().float().cpu()
        d = (p.detach().float().cpu() - w)
        keep = w.abs() > 1e-12
        w, d = w[keep], d[keep]
        rel.append((d.abs() / w.abs()))
        n_total += w.numel()
        # a weight update disappears when it is smaller than half the spacing
        # of the format at that weight's own magnitude
        binade = torch.floor(torch.log2(w.abs()))
        for mant_bits, bucket in ((7, "bf16"), (3, "fp8")):
            ulp = torch.pow(2.0, binade - mant_bits)
            vanished = int((d.abs() < ulp / 2).sum())
            if bucket == "bf16":
                lost_bf16 += vanished
            else:
                lost_fp8 += vanished
    rel = torch.cat(rel)
    q = torch.quantile(rel, torch.tensor([0.05, 0.5, 0.95]))
    return {"median_relative_update": float(q[1]),
            "p05_relative_update": float(q[0]),
            "p95_relative_update": float(q[2]),
            "n_weights": n_total,
            "pct_updates_lost_in_bf16": 100.0 * lost_bf16 / n_total,
            "pct_updates_lost_in_fp8_e4m3": 100.0 * lost_fp8 / n_total}


def stagnation_demo(n_steps=2000, update=1e-5, start=0.1):
    """Add the same small number to a weight, over and over, in each format."""
    curves = {}
    exact = [start + i * update for i in range(n_steps + 1)]

    w = torch.tensor(start, dtype=torch.float32)
    c = [float(w)]
    for _ in range(n_steps):
        w = w + torch.tensor(update, dtype=torch.float32)
        c.append(float(w))
    curves["fp32"] = c

    w = torch.tensor(start, dtype=torch.bfloat16)
    c = [float(w)]
    for _ in range(n_steps):
        w = w + torch.tensor(update, dtype=torch.bfloat16)
        c.append(float(w))
    curves["bf16"] = c

    master = torch.tensor(start, dtype=torch.float32)
    c = [float(master.bfloat16())]
    for _ in range(n_steps):
        master = master + update           # the fp32 master copy accumulates
        c.append(float(master.bfloat16())) # the bf16 shadow is only read from
    curves["bf16 + fp32 master"] = c

    return {"exact": exact, "curves": curves, "update": update,
            "start": start, "steps": n_steps}


def fp16_underflow():
    """Section 10's table, measured."""
    rows = []
    for g in (1e-4, 1e-6, 1e-8, 1e-10):
        f16 = float(torch.tensor(g, dtype=torch.float16))
        b16 = float(torch.tensor(g, dtype=torch.bfloat16))
        scaled = float(torch.tensor(g * 1024, dtype=torch.float16)) / 1024
        rows.append({"gradient": g, "fp16": f16, "bf16": b16,
                     "fp16_with_loss_scaling_1024": scaled,
                     "fp16_is_zero": f16 == 0.0})
    return rows


def main(verbose: bool = True) -> dict:
    encodings = encode_all()
    updates = real_update_sizes()
    stag = stagnation_demo()
    underflow = fp16_underflow()

    by = {e["format"]: e for e in encodings}

    # ---- figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))

    ax = axes[0]
    n = stag["steps"]
    xs = list(range(n + 1))
    ax.plot(xs, stag["exact"], color=MUTED, lw=4.5, solid_capstyle="butt",
            label="exact")
    ax.plot(xs, stag["curves"]["fp32"], color=S1, lw=2.4, label="fp32")
    ax.plot(xs, stag["curves"]["bf16 + fp32 master"], color=S3, lw=1.1,
            label="bf16 + fp32 master")
    ax.plot(xs, stag["curves"]["bf16"], color=S2, lw=2.4, label="bf16")
    ax.set_title(f"Adding {stag['update']:.0e} to a weight of "
                 f"{stag['start']}, {n} times", loc="left")
    ax.set_xlabel("update number")
    ax.set_ylabel("weight value")
    ax.legend(loc="upper left")
    ax.annotate(f"bf16 never moves: {stag['curves']['bf16'][-1]:.6f}\n"
                f"exact answer: {stag['exact'][-1]:.6f}",
                (0.97, 0.10), xycoords="axes fraction",
                ha="right", va="bottom", color=INK_2, fontsize=8)

    ax = axes[1]
    names = ["fp32", "bf16", "fp8 E4M3"]
    vals = [by[k]["relative_ulp"] for k in names]
    ypos = list(range(len(names)))[::-1]
    lo, med, hi = (updates["p05_relative_update"],
                   updates["median_relative_update"],
                   updates["p95_relative_update"])
    ax.axvspan(lo, hi, color=CRITICAL, alpha=0.10, lw=0)
    ax.axvline(med, color=CRITICAL, lw=1.6)
    for y, v in zip(ypos, vals):
        ax.plot([v], [y], marker="o", ms=10, color=S1,
                markeredgecolor="#fcfcfb", markeredgewidth=2, zorder=4)
        ax.annotate(f"{v:.1e}", (v, y), textcoords="offset points",
                    xytext=(11, 0), va="center", color=INK_2, fontsize=8.5)
    ax.set_yticks(ypos); ax.set_yticklabels(names)
    ax.set_ylim(-0.9, len(names) - 0.35)
    ax.set_xscale("log")
    ax.set_xlim(min(vals) / 4, max(vals) * 22)
    ax.grid(axis="y", visible=False)
    ax.annotate(f"real AdamW updates, 5th–95th percentile\n"
                f"median {med:.1e}  ·  a dot to the right of this\n"
                f"line is a format that rounds them to nothing",
                (med, -0.55), textcoords="offset points", xytext=(-9, 0),
                ha="right", va="center", color=INK_2, fontsize=8)
    ax.set_xlabel("spacing between representable values at 0.1, "
                  "relative to the value")
    ax.set_title("A format cannot store an update smaller than its own\n"
                 "spacing", loc="left")

    fig_path = finish(fig, "e6_float_bits.png",
                      "left: 2,000 identical updates.  right: format spacing "
                      "against a real AdamW update, measured at step 31 of a "
                      "run of this session's model")

    # ---- report --------------------------------------------------------------
    p = []
    p.append("# Deliverable 6 — 0.1 in fp32, bf16 and fp8 E4M3\n")

    p.append("## Doing it by hand\n")
    p.append("0.1 is not a binary fraction. Doubling and reading off the "
             "carry gives\n")
    p.append(f"```\n0.1  =  {binary_expansion(X, 28)}  (binary)\n"
             f"     =  1.100110011001100110011…  ×  2⁻⁴\n```\n")
    p.append("so the unbiased exponent is **−4** in every format below, and the "
             "significand to be rounded is always the same repeating "
             "`1.10011001100…`. Only the number of mantissa bits and the "
             "exponent bias change.\n")

    for fmt_key in ("fp32", "bf16", "fp8 E4M3"):
        e = by[fmt_key]
        p.append(f"\n### {fmt_key} — 1 sign, {e['exp_bits']} exponent, "
                 f"{e['mant_bits']} mantissa (bias {e['bias']})\n")
        p.append(f"- exponent field = −4 + {e['bias']} = **{e['exp_field']}** "
                 f"= `{e['exp_field_bits']}`")
        keep = "1" + "0" * 0
        p.append(f"- mantissa = the {e['mant_bits']} bits after the leading 1 "
                 f"of `1.10011001100…`, rounded to nearest "
                 f"(the first discarded bit is a 1 and the tail is non-zero, "
                 f"so it rounds up) = `{e['mantissa_bits']}`")
        p.append(f"- significand = `{e['significand']}`₂ "
                 f"= {float(Fraction(int(e['significand'].replace('.', ''), 2), 1 << e['mant_bits'])):.10f}")
        p.append(f"\n```\n{e['bits']}     {e['hex']}\n"
                 f"^ ^{'-'*(e['exp_bits']-1)} ^{'-'*(e['mant_bits']-1)}\n"
                 f"| exponent    mantissa\nsign\n```\n")
        p.append(f"- stored value = **{e['value']:.20f}**")
        p.append(f"- relative error = **{e['rel_error']:.3e}** "
                 f"({e['rel_error']*100:.4f}%)")
        p.append(f"- torch stores `{e['hardware_bits']}` — "
                 f"**{'identical' if e['matches_hardware'] else 'DIFFERENT'}**\n")

    p.append("\n## All six formats side by side\n")
    p.append(table([
        (e["format"], e["bits_total"], f"{e['exp_bits']}/{e['mant_bits']}",
         e["bits"], e["hex"], f"{e['value']:.12f}", f"{e['rel_error']:.2e}",
         "yes" if e["matches_hardware"] else
         ("—" if e["matches_hardware"] is None else "NO"))
        for e in encodings
    ], ["format", "bits", "E/M", "sign · exponent · mantissa", "hex",
        "value stored for 0.1", "rel. error", "matches torch"],
        ["l", "r", "l", "l", "r", "r", "r", "l"]))
    p.append("\nEvery hand-derived pattern matches the bits the hardware "
             "actually stores. fp4 E2M1 has no torch dtype to check against "
             "and is shown for the shape of the trade only — on its own it "
             "cannot even reach 0.1: it rounds to exactly **zero**, because "
             "the smallest non-zero value fp4 E2M1 can name is 0.5. That is "
             "why NVFP4 never uses the element format on its own, and always "
             "with one shared exponent per block of sixteen.\n")

    p.append("\n## Which one would I train in — bf16, with an fp32 master copy\n")

    p.append(f"""
Not because bf16 is accurate. It is the *least* accurate of the three at
representing 0.1: it stores {by['bf16']['value']:.10f}, a relative error of
{by['bf16']['rel_error']*100:.4f}%, against fp16's
{by['fp16']['rel_error']*100:.4f}% at the same 16 bits. The argument is about
range and about what a weight update actually looks like.

**1. The exponent field is the one that ends runs.** fp16 spends 5 bits on
exponent and cannot hold anything below about 6e-8:
""")
    p.append(table([
        (f"{r['gradient']:.0e}",
         "0 — gone" if r["fp16_is_zero"] else f"{r['fp16']:.3e}",
         f"{r['bf16']:.3e}",
         f"{r['fp16_with_loss_scaling_1024']:.3e}")
        for r in underflow
    ], ["gradient", "in fp16", "in bf16", "in fp16, ×1024 then ÷1024"],
        ["r", "r", "r", "r"]))
    p.append(
        "\nA gradient that becomes exactly zero is a weight that does not move, "
        "and nothing raises an error. Loss scaling patches it and is one more "
        "number to get wrong. bf16 keeps all eight of fp32's exponent bits, so "
        "the floor is never reached and the apparatus is unnecessary.\n")

    p.append(f"""
**2. But bf16 alone cannot hold a weight, and this is measurable.** bf16 keeps
7 mantissa bits, so consecutive bf16 values are 2⁻⁷ of a binade apart — at 0.1
that is {by['bf16']['ulp_at_0.1']:.3e} absolute, or
{by['bf16']['relative_ulp']:.2e} relative. In this session's own run, at step
31, the median AdamW update across all {updates['n_weights']:,} weights is
{updates['median_relative_update']:.2e} of the weight's own size — already
smaller than that spacing. Rounding each real update to the bf16 grid at its own
weight's magnitude, **{updates['pct_updates_lost_in_bf16']:.1f}% of the
{updates['n_weights']:,} updates this optimiser step disappear entirely**
(in fp8 E4M3, {updates['pct_updates_lost_in_fp8_e4m3']:.1f}%).

The left panel is the same fact in slow motion: adding {stag['update']:.0e} to a
bf16 0.1 {stag['steps']} times leaves it at
**{stag['curves']['bf16'][-1]:.6f}** — it never moved once. The exact answer is
{stag['exact'][-1]:.6f}; fp32 reaches {stag['curves']['fp32'][-1]:.6f}. The
weight is not learning slowly, it is not learning at all, and the loss curve
says nothing about it.

That is exactly why section 13's table lists a bf16 weight *and* an fp32 master
copy: 2 bytes for the copy the matmuls read, 4 bytes for the copy the optimiser
adds to. The third line of the figure is that arrangement, and it tracks the
exact answer.

**3. fp8 E4M3 is a matmul input, not a weight.** With 3 mantissa bits its grid
is 2⁻³ of a binade — between 6.3% and 12.5% relative depending on where in the
binade you land, and {by['fp8 E4M3']['relative_ulp']*100:.1f}% at 0.1 exactly.
Errors of that size are survivable where they are averaged across a long
reduction and never accumulated, which is what a forward matmul does. They are
not survivable in a running sum of tens of thousands of updates, which is what a
weight is. The 2026 production recipe is exactly this split: fp8 tensors into
the GEMMs, higher precision for the master weights, the optimiser state, and
attention — softmax amplifies whatever noise you hand it.

**So: bf16 activations and gradients, fp32 master weights and fp32 optimiser
moments, no loss scaling.** For V5 specifically I would take fp8 E4M3 on the
linear layers only once there is a short A/B on the real architecture, because
the thing fp8 costs you is not visible in the loss curve either.
""")

    text = "\n".join(p)
    save_text("e6_float_bits.md", text)
    payload = {"machine": machine(), "x": X, "encodings": encodings,
               "real_update_sizes": updates,
               "stagnation": {k: v for k, v in stag.items()
                              if k != "curves" and k != "exact"}
               | {"final": {k: v[-1] for k, v in stag["curves"].items()},
                  "exact_final": stag["exact"][-1]},
               "fp16_underflow": underflow,
               "all_hand_derivations_match_hardware": all(
                   e["matches_hardware"] for e in encodings
                   if e["matches_hardware"] is not None),
               "figure": fig_path.name}
    save_json("e6_float_bits.json", payload)
    if verbose:
        print(text)
    return payload


if __name__ == "__main__":
    main()
