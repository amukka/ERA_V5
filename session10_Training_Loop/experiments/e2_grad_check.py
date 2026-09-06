"""Deliverable 2 — verify one gradient by hand.

Nudge a weight, measure how the loss changed, and compare against what
``backward()`` reported.  Three checks, in increasing order of how much can go
wrong:

  A. the session's own four-link chain, where the answer is 64 exactly
  B. one weight of the real 8.3M-parameter model, in float64
  C. the same weight in float32 -- which *disagrees*, and the reason it
     disagrees is the thing worth understanding

The estimator is the central difference

    dL/dw  ~=  ( L(w+h) - L(w-h) ) / 2h

whose error is O(h^2) from truncation plus O(eps*L/h) from cancellation, so
there is a best h and it is not the smallest one you can type.
"""

from __future__ import annotations

import math
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from src.data import Sampler, load_documents
from src.loop import set_seed, token_losses
from src.model import Config, TinyGPT
from src.plots import CRITICAL, GOOD, INK_2, MUTED, S1, S2, finish, plt
from src.report import machine, save_json, save_text, table


def matching_decimals(a: float, b: float) -> float:
    """How many decimal places two numbers agree to."""
    d = abs(a - b)
    if d == 0:
        return float("inf")
    return -math.log10(d)


# --------------------------------------------------------------------------- A
def toy_chain():
    """x=2, w1=3, w2=4, target=20.  Section 4 says dL/dw1 = 64."""
    x, t = 2.0, 20.0

    def loss_of(w1, w2):
        h = w1 * x
        y = w2 * h
        return (y - t) ** 2

    w1, w2 = 3.0, 4.0
    # by hand, one link at a time
    h = w1 * x                       # 6
    y = w2 * h                       # 24
    dL_dy = 2 * (y - t)              # 8
    dL_dw2 = dL_dy * h               # 48
    dL_dh = dL_dy * w2               # 32
    dL_dw1 = dL_dh * x               # 64

    # by nudging
    eps = 1e-6
    fd = (loss_of(w1 + eps, w2) - loss_of(w1 - eps, w2)) / (2 * eps)

    # by autograd
    a = torch.tensor(w1, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(w2, dtype=torch.float64, requires_grad=True)
    L = (b * (a * x) - t) ** 2
    L.backward()

    return {
        "forward": {"h": h, "y": y, "loss": loss_of(w1, w2)},
        "by_hand": {"dL_dy": dL_dy, "dL_dw2": dL_dw2, "dL_dh": dL_dh,
                    "dL_dw1": dL_dw1},
        "finite_difference_w1": fd,
        "autograd_w1": float(a.grad),
        "autograd_w2": float(b.grad),
        "decimals_hand_vs_autograd": matching_decimals(dL_dw1, float(a.grad)),
        "decimals_hand_vs_nudge": matching_decimals(dL_dw1, fd),
    }


# ------------------------------------------------------------------------- B/C
def real_model_check(dtype, param_name, index, h_values, batch, cfg, seed=0):
    """Central difference on one scalar weight of the real model.

    The forward is deterministic (no dropout anywhere in this model), the batch
    is fixed, and the whole thing runs on CPU so that float64 is available --
    all three are necessary or the difference measures noise instead of slope.
    """
    set_seed(seed)
    model = TinyGPT(cfg).to(dtype)
    model.eval()

    param = dict(model.named_parameters())[param_name]

    def loss_at(value=None):
        with torch.no_grad():
            if value is not None:
                old = param.data[index].item()
                param.data[index] = value
        loss_sum, n = token_losses(model, batch)
        out = float(loss_sum.detach()) / n
        if value is not None:
            with torch.no_grad():
                param.data[index] = old
        return out

    model.zero_grad(set_to_none=True)
    loss_sum, n = token_losses(model, batch)
    (loss_sum / n).backward()
    reported = float(param.grad[index])

    w0 = float(param.data[index])
    base = loss_at()

    rows = []
    for hh in h_values:
        up = loss_at(w0 + hh)
        dn = loss_at(w0 - hh)
        fd = (up - dn) / (2 * hh)
        rows.append({
            "h": hh,
            "loss_plus": up,
            "loss_minus": dn,
            "finite_difference": fd,
            "abs_error": abs(fd - reported),
            "rel_error": abs(fd - reported) / max(abs(reported), 1e-30),
            "decimals": matching_decimals(fd, reported),
        })
    best = min(rows, key=lambda r: r["abs_error"])
    return {
        "dtype": str(dtype).replace("torch.", ""),
        "param": param_name,
        "index": list(index),
        "weight_value": w0,
        "base_loss": base,
        "backward_reported": reported,
        "sweep": rows,
        "best": best,
    }


def main(verbose: bool = True) -> dict:
    cfg = Config(n_layer=4, d_model=256, n_head=4, d_ff=1024)
    docs, _ = load_documents()
    # one small, fixed batch -- small so the sweep is cheap, fixed so the two
    # forwards in the difference see the same data
    batch = Sampler(docs["train"], batch_size=2, max_len=96, min_len=64,
                    mode="fixed", seed=7).batch()

    toy = toy_chain()

    param_name, index = "blocks.0.mlp.fc.weight", (17, 42)
    hs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    f64 = real_model_check(torch.float64, param_name, index, hs, batch, cfg)
    f32 = real_model_check(torch.float32, param_name, index, hs, batch, cfg)

    # a second, structurally different weight, to show it was not a lucky pick
    # pick an embedding row that the batch actually uses -- a row for a token
    # that never appears has gradient exactly zero, which verifies nothing
    live_token = int(batch.inputs[0, 5])
    others = [
        real_model_check(torch.float64, name, idx, [1e-5], batch, cfg)
        for name, idx in [
            ("tok_emb.weight", (live_token, 100)),
            ("blocks.2.attn.qkv.weight", (300, 11)),
            ("head.weight", (4096, 200)),
            ("ln_f.weight", (64,)),
        ]
    ]

    # ---- figure: why float32 cannot do this ---------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for res, colour, name in ((f64, S1, "float64"), (f32, S2, "float32")):
        xs = [r["h"] for r in res["sweep"]]
        ys = [max(r["rel_error"], 1e-17) for r in res["sweep"]]
        ax.plot(xs, ys, color=colour, marker="o", ms=4.5,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("nudge size $h$  (larger  →  smaller)")
    ax.set_ylabel("relative error vs backward()")
    ax.set_title("The nudge agrees with backward() — until the subtraction\n"
                 "runs out of digits", loc="left")
    ax.axhline(1e-6, color=MUTED, lw=0.8, ls="-")
    ax.annotate("agreement to 6 decimals", (1e-1, 4e-7),
                textcoords="offset points", xytext=(4, 0),
                color=MUTED, fontsize=7.5, va="top", ha="left")
    ax.annotate(f"float64 best: {f64['best']['rel_error']:.1e}",
                (f64["best"]["h"], max(f64["best"]["rel_error"], 1e-17)),
                textcoords="offset points", xytext=(6, 10),
                color=INK_2, fontsize=8)
    ax.legend(loc="lower left")
    fig_path = finish(fig, "e2_grad_check.png",
                      f"one weight ({param_name}{list(index)}) of an "
                      f"8.3M-parameter model, central difference on a fixed batch")

    # ---- report --------------------------------------------------------------
    p = []
    p.append("# Deliverable 2 — verify one gradient by hand\n")

    p.append("## A. The session's own chain, where the answer is known\n")
    p.append("`x = 2`, `w1 = 3`, `w2 = 4`, `target = 20`.\n")
    p.append(table([
        ("h = w1·x", "3 × 2", f"{toy['forward']['h']:.0f}"),
        ("y = w2·h", "4 × 6", f"{toy['forward']['y']:.0f}"),
        ("loss = (y−t)²", "(24 − 20)²", f"{toy['forward']['loss']:.0f}"),
    ], ["quantity", "calculation", "value"], ["l", "l", "r"]))
    p.append("\nWalking back one link at a time:\n")
    p.append(table([
        ("∂L/∂y", "2(y − t)", f"{toy['by_hand']['dL_dy']:.0f}"),
        ("∂L/∂w2", "∂L/∂y × h", f"{toy['by_hand']['dL_dw2']:.0f}"),
        ("∂L/∂h", "∂L/∂y × w2", f"{toy['by_hand']['dL_dh']:.0f}"),
        ("∂L/∂w1", "∂L/∂h × x", f"{toy['by_hand']['dL_dw1']:.0f}"),
    ], ["quantity", "calculation", "value"], ["l", "l", "r"]))
    p.append("\n" + table([
        ("by hand (chain rule)", f"{toy['by_hand']['dL_dw1']:.12f}"),
        ("by nudging (h = 1e-6)", f"{toy['finite_difference_w1']:.12f}"),
        ("by backward()", f"{toy['autograd_w1']:.12f}"),
    ], ["∂L/∂w1 obtained", "value"], ["l", "r"]))
    p.append(f"\nHand vs `backward()`: identical to every digit float64 has "
             f"(difference exactly 0). "
             f"Hand vs nudge: {toy['decimals_hand_vs_nudge']:.1f} decimals — "
             "the nudge is the one that is approximate.\n")

    p.append("\n## B. One weight of the real model, in float64\n")
    p.append(f"Weight: `{param_name}[{index[0]}, {index[1]}]` = "
             f"{f64['weight_value']:.10f}, on a fixed batch of "
             f"{batch.inputs.shape[0]}×{batch.inputs.shape[1]} tokens. "
             f"Loss at that point: {f64['base_loss']:.10f}.\n")
    p.append(f"`backward()` reported **{f64['backward_reported']:.12f}**.\n")
    p.append(table([
        (f"{r['h']:.0e}", f"{r['loss_plus']:.12f}", f"{r['loss_minus']:.12f}",
         f"{r['finite_difference']:.10f}", f"{r['rel_error']:.2e}",
         f"{r['decimals']:.1f}")
        for r in f64["sweep"]
    ], ["nudge h", "loss(w+h)", "loss(w−h)", "(Δloss)/2h", "rel. error",
        "decimals agreed"], ["r", "r", "r", "r", "r", "r"]))
    b = f64["best"]
    p.append(f"\nBest at h = {b['h']:.0e}: the nudge says "
             f"{b['finite_difference']:.12f}, `backward()` says "
             f"{f64['backward_reported']:.12f}. They agree to "
             f"**{b['decimals']:.1f} decimal places** "
             f"(relative error {b['rel_error']:.2e}).\n")

    p.append("\nFour more weights, one nudge each at h = 1e-5, to show the "
             "first was not a lucky pick:\n")
    p.append(table([
        (o["param"] + str(o["index"]),
         f"{o['backward_reported']:.10f}",
         f"{o['sweep'][0]['finite_difference']:.10f}",
         f"{o['sweep'][0]['rel_error']:.1e}",
         f"{o['sweep'][0]['decimals']:.1f}")
        for o in others
    ], ["weight", "backward()", "nudge", "rel. error", "decimals"],
        ["l", "r", "r", "r", "r"]))

    p.append("\n## C. The same check in float32, which fails — and why\n")
    bf = f32["best"]
    p.append(f"Identical model, identical batch, identical weight, only the "
             f"dtype changed. `backward()` reports "
             f"{f32['backward_reported']:.10f}; the best nudge manages "
             f"{bf['finite_difference']:.10f} at h = {bf['h']:.0e} — "
             f"**{bf['decimals']:.1f} decimals**, against "
             f"{b['decimals']:.1f} in float64.\n")
    p.append(table([
        (f"{r['h']:.0e}", f"{r['finite_difference']:.8f}",
         f"{r['rel_error']:.2e}", f"{r['decimals']:.1f}")
        for r in f32["sweep"]
    ], ["nudge h", "(Δloss)/2h", "rel. error", "decimals agreed"],
        ["r", "r", "r", "r"]))
    p.append(
        "\nThis is the disagreement the assignment says is worth understanding, "
        "and the thing worth understanding is that **`backward()` is not the "
        "one that is wrong**. The loss here is around "
        f"{f32['base_loss']:.4f}; float32 resolves it to about "
        f"{abs(f32['base_loss']) * 2**-24:.2e}. Dividing that noise by "
        "2h magnifies it as h shrinks, while the truncation error of the "
        "central difference shrinks as h². The two meet at a floor, and no "
        "choice of h gets under it. In float64 the same floor sits eight orders "
        "of magnitude lower, which is why the check has to be run there.\n"
        "\nThe practical rule: a finite-difference gradient check that fails in "
        "float32 has told you nothing. Re-run it in float64 before you go "
        "looking for a bug in the backward pass.\n")

    text = "\n".join(p)
    save_text("e2_grad_check.md", text)
    payload = {"machine": machine(), "toy_chain": toy, "float64": f64,
               "float32": f32, "other_weights": others,
               "figure": fig_path.name}
    save_json("e2_grad_check.json", payload)
    if verbose:
        print(text)
    return payload


if __name__ == "__main__":
    main()
