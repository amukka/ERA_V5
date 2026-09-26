"""E1 — Adam, reproduced by hand for one weight and five gradients.

The hand version is plain Python floats (IEEE float64), written as the five
lines of Section 6 and nothing else. PyTorch's ``torch.optim.Adam`` is then run
on a one-element tensor fed the same five gradients, and every intermediate is
compared:

    m, v        read straight out of ``optimizer.state[w]`` (exp_avg, exp_avg_sq)
    m_hat, v_hat PyTorch never stores these, so they are recovered from its
                 own m and v with its own step count
    step        the change PyTorch actually made to the weight

Done twice: once in float64, where the two should agree to the last few bits,
and once in float32 (the dtype a real master copy uses), where the agreement is
limited by float32 itself. A third pass repeats it for AdamW with decay 0.1, to
check the decoupled term lands where Section 7 says.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from src.report import machine, save_json, save_text, table

W0 = 1.0
GRADS = [0.50, 0.40, 0.60, 0.45, 0.55]     # Section 6's own five gradients
LR, B1, B2, EPS = 1e-3, 0.9, 0.999, 1e-8


def by_hand(w0=W0, grads=GRADS, lr=LR, b1=B1, b2=B2, eps=EPS, wd=0.0):
    w, m, v, rows = w0, 0.0, 0.0, []
    for t, g in enumerate(grads, start=1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        step = -lr * m_hat / (math.sqrt(v_hat) + eps) - lr * wd * w
        w = w + step
        rows.append(dict(t=t, g=g, m=m, v=v, m_hat=m_hat, v_hat=v_hat,
                         step=step, w=w))
    return rows


def by_torch(dtype, w0=W0, grads=GRADS, lr=LR, b1=B1, b2=B2, eps=EPS, wd=0.0):
    w = torch.tensor([w0], dtype=dtype, requires_grad=True)
    cls = torch.optim.AdamW if wd else torch.optim.Adam
    opt = cls([w], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd,
              foreach=False)
    rows = []
    for t, g in enumerate(grads, start=1):
        before = w.detach().clone()
        w.grad = torch.tensor([g], dtype=dtype)
        opt.step()
        st = opt.state[w]
        m = float(st["exp_avg"][0])
        v = float(st["exp_avg_sq"][0])
        t_ = int(st["step"])
        rows.append(dict(t=t_, g=g, m=m, v=v,
                         m_hat=m / (1 - b1 ** t_), v_hat=v / (1 - b2 ** t_),
                         step=float((w.detach() - before)[0]),
                         w=float(w.detach()[0])))
    return rows


KEYS = ["m", "v", "m_hat", "v_hat", "step", "w"]


def compare(hand, tch):
    """Worst relative error per quantity, and the decimal places it implies."""
    out = {}
    for k in KEYS:
        rel = max(abs(a[k] - b[k]) / max(abs(a[k]), 1e-300)
                  for a, b in zip(hand, tch))
        out[k] = {"max_rel_err": rel,
                  "decimals": (math.inf if rel == 0 else -math.log10(rel))}
    return out


def fmt_dec(d):
    return "exact" if d == math.inf else f"{d:.1f}"


def main(verbose: bool = True) -> dict:
    hand = by_hand()
    t64 = by_torch(torch.float64)
    t32 = by_torch(torch.float32)
    c64, c32 = compare(hand, t64), compare(hand, t32)

    hand_w = by_hand(wd=0.1)
    tw64 = by_torch(torch.float64, wd=0.1)
    cw64 = compare(hand_w, tw64)
    # what the decoupled term alone contributed at step 1
    decay_step1 = -LR * 0.1 * W0

    # --- write-up ---------------------------------------------------------
    rows = [(r["t"], f'{r["g"]:.2f}', f'{r["m"]:.6f}', f'{r["v"]:.8f}',
             f'{r["m_hat"]:.6f}', f'{r["v_hat"]:.6f}', f'{r["step"]:+.9f}',
             f'{r["w"]:.9f}') for r in hand]
    hdr = ["t", "g", "m", "v", "m̂", "v̂", "step", "w after"]
    side = []
    for a, b in zip(hand, t64):
        side.append((a["t"], f'{a["step"]:+.15f}', f'{b["step"]:+.15f}',
                     f'{abs(a["step"] - b["step"]):.1e}'))
    agree = [(k, f'{c64[k]["max_rel_err"]:.1e}', fmt_dec(c64[k]["decimals"]),
              f'{c32[k]["max_rel_err"]:.1e}', fmt_dec(c32[k]["decimals"]))
             for k in KEYS]
    min64 = min(c64[k]["decimals"] for k in KEYS)
    min32 = min(c32[k]["decimals"] for k in KEYS)
    ratio_step = [abs(r["step"]) / LR for r in hand]

    md = [
        "# E1 · Adam by hand, checked against PyTorch\n",
        f"One weight starting at w = {W0}, five gradients {GRADS}, "
        f"η = {LR}, β₁ = {B1}, β₂ = {B2}, ε = {EPS:g}. The hand column is "
        "plain Python floats following the five lines of Section 6.\n",
        "## The hand computation\n",
        table(rows, hdr, ["r"] * 8),
        "\nThis reproduces the Section 6 table digit for digit at its printed "
        "precision. Every step is between "
        f"{min(ratio_step) * 100:.2f}% and {max(ratio_step) * 100:.2f}% "
        "of η, although the gradients differ by 50% from smallest to largest. "
        "That is the property the session names: the gradient sets the "
        "direction, η sets the distance.\n",
        "## The step, side by side (float64)\n",
        table(side, ["t", "by hand", "torch.optim.Adam", "|difference|"],
              ["r", "r", "r", "r"]),
        "\n## Agreement for every quantity\n",
        "Worst relative error over the five steps, and the number of decimal "
        "places of agreement it implies (−log₁₀ of the relative error). "
        "PyTorch does not store m̂ or v̂; they are recovered from its own m, v "
        "and step counter.\n",
        table(agree, ["quantity", "float64 rel. err", "float64 decimals",
                      "float32 rel. err", "float32 decimals"],
              ["l", "r", "r", "r", "r"]),
        f"\n**Float64: at least {fmt_dec(min64)} decimal places on every "
        f"quantity. Float32: at least {fmt_dec(min32)}.** In float32, m, v, m̂, "
        "v̂ and w all agree to 7–8 digits, which is float32's own resolution. "
        "The step is the outlier, and not because Adam disagrees: PyTorch never "
        "stores its step, so it is recovered as `w_after − w_before`. Both "
        "weights sit near 1.0, where float32 spacing is 1.2e-7, and the step is "
        "only 1e-3, so the subtraction alone throws away about four digits. "
        "Which is itself a Session 10 lesson: a 1e-3 update to a weight of 1 "
        "keeps only ~4 significant digits in float32, and none at all in bf16.\n",
        "## AdamW, decay 0.1\n",
        "The same five gradients through `torch.optim.AdamW` against the hand "
        "rule with the decoupled `− η·λ·w` term added after the Adam step. "
        f"Worst agreement in float64: {fmt_dec(min(cw64[k]['decimals'] for k in KEYS))} "
        f"decimals. The decay term alone moved the weight by {decay_step1:+.1e} "
        f"at step 1, about {abs(decay_step1) / LR * 100:.0f}% of η, and it is "
        "not divided by √v̂, which is the whole point of Section 7.\n",
        "## A detail found by doing it\n",
        "PyTorch does not compute the step the way Section 6 writes it. It "
        "folds both corrections into a scalar, "
        "`step_size = η / (1 − β₁ᵗ)` and "
        "`denom = √v / √(1 − β₂ᵗ) + ε`. ε is still added *after* the v "
        "correction, as in the formula, so the two routes are the same "
        "algorithm and differ only in rounding order. That is why float64 "
        "agrees to 15–16 digits on m, v, m̂ and v̂ rather than bit for bit. "
        "Step 1 also shows ε at work: the step is 0.99999998·η, not η, "
        "because ε = 1e-8 sits beside √v̂ = 0.5.\n",
    ]
    save_text("e1_adam_by_hand.md", "\n".join(md))
    payload = {"config": dict(w0=W0, grads=GRADS, lr=LR, b1=B1, b2=B2,
                              eps=EPS),
               "hand": hand, "torch_float64": t64, "torch_float32": t32,
               "agreement_float64": c64, "agreement_float32": c32,
               "adamw_hand": hand_w, "adamw_torch_float64": tw64,
               "adamw_agreement_float64": cw64, "machine": machine()}
    save_json("e1_adam_by_hand.json", payload)
    if verbose:
        print("\n".join(md))
    return payload


if __name__ == "__main__":
    main()
