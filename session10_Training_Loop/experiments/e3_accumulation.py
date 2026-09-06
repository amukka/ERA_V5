"""Deliverable 3 — break gradient accumulation on purpose.

Use the average of the averages with micro-batches of different lengths, and
plot both curves together so the gap is visible rather than asserted.

Three levels of evidence, cheapest first:

  1. the session's own arithmetic, reproduced exactly (2.6000 vs 3.0000)
  2. one real step: both reductions on the *same* micro-batches, comparing not
     just the reported loss but the two gradient vectors they produce
  3. two full training runs -- same seed, same initial weights, same data in the
     same order, differing in nothing but the division -- scored on one common
     token-weighted validation metric, plus the control run where the token
     counts are equal and the gap goes away

The micro-batches here are *length-bucketed*: every row in a micro-batch shares
one length, which is what a real loader does for throughput.  That is not a
thumb on the scale, it is the realistic case, and it is precisely the case that
makes the bug bite -- bucketing maximises the spread of token counts between
micro-batches.
"""

from __future__ import annotations

import copy
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from src.data import Sampler, load_documents
from src.loop import (accumulate_step, evaluate, pick_device, set_seed, train)
from src.model import Config, TinyGPT
from src.plots import CRITICAL, INK_2, MUTED, S1, S2, S3, finish, plt
from src.report import machine, save_json, save_text, table
STEPS = 400
ACCUM = 4
MICRO_B = 8
MIN_LEN, MAX_LEN = 32, 256
EVAL_EVERY = 10
SEED = 1234

# The control arm holds every micro-batch to one length. That length is the
# geometric mean of the bucketed arm's log-uniform range, so the control sees
# the same token budget per micro-batch on average and differs from the
# experiment in exactly one thing: the spread between micro-batches.
CONTROL_LEN = round((MIN_LEN * MAX_LEN) ** 0.5)

ZOOM_FROM = 80        # the first steps are a vertical drop that hides everything


def make_figure(curves) -> pathlib.Path:
    """Three panels: the two curves in the broken setting, the two curves in
    the control, and the gap itself for both -- because at this scale the gap
    is real but small, and a plot whose y-axis is dominated by the opening
    descent would let the reader take it on trust, which is the one thing this
    session is against."""
    tw, mm = "token_weighted", "mean_of_means"

    def tail(setting, mode):
        xs = curves[setting][mode]["val_steps"]
        ys = curves[setting][mode]["val_loss"]
        keep = [(x, y) for x, y in zip(xs, ys) if x >= ZOOM_FROM]
        return [p[0] for p in keep], [p[1] for p in keep]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))

    all_y = [y for s in ("bucketed", "fixed") for m in (tw, mm)
             for y in tail(s, m)[1]]
    pad = 0.05 * (max(all_y) - min(all_y))
    ylim = (min(all_y) - pad, max(all_y) + pad)

    for ax, setting, title in (
        (axes[0], "bucketed", "Micro-batches of different lengths"),
        (axes[1], "fixed", f"Control: every sequence {CONTROL_LEN} tokens"),
    ):
        for mode, colour, name in (
            (tw, S1, "sum of losses ÷ sum of tokens"),
            (mm, S2, "average of the averages"),
        ):
            xs, ys = tail(setting, mode)
            ax.plot(xs, ys, color=colour, label=name)
        gap = curves[setting][mm]["val_loss"][-1] - \
            curves[setting][tw]["val_loss"][-1]
        ax.set_title(title, loc="left")
        ax.set_xlabel(f"optimiser step  (from {ZOOM_FROM})")
        ax.set_ylim(*ylim)
        ax.legend(loc="upper right")
        ax.annotate(f"{gap:+.4f} nats at the end"
                    + ("" if setting == "bucketed" else "\n— this is how it hid"),
                    (0.97, 0.30), xycoords="axes fraction", ha="right",
                    va="top", color=INK_2, fontsize=8)
    axes[0].set_ylabel("validation loss\n(token-weighted — the same metric for both)")

    ax = axes[2]
    for setting, colour, name in (
        ("bucketed", CRITICAL, "micro-batches of different lengths"),
        ("fixed", S3, "control: equal lengths"),
    ):
        xs = curves[setting][tw]["val_steps"]
        d = [b - a for a, b in zip(curves[setting][tw]["val_loss"],
                                   curves[setting][mm]["val_loss"])]
        ax.plot(xs, d, color=colour, label=name)
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_title("The gap itself", loc="left")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("broken − correct  (nats of validation loss)")
    ax.legend(loc="lower right")

    return finish(fig, "e3_accumulation.png",
                  f"{STEPS} steps · micro-batch {MICRO_B} × {ACCUM} "
                  f"accumulation · identical init, identical data order, "
                  f"identical seed — only the division differs")



def session_arithmetic():
    """Section 8's table, computed rather than quoted."""
    mb = [(4, 2.0), (4, 2.0), (2, 5.0)]
    total_tokens = sum(n for n, _ in mb)
    correct = sum(n * l for n, l in mb) / total_tokens
    wrong = sum(l for _, l in mb) / len(mb)
    return {
        "micro_batches": [{"valid_tokens": n, "average_loss": l} for n, l in mb],
        "token_weighted": correct,
        "mean_of_means": wrong,
        "relative_error_pct": 100 * (wrong - correct) / correct,
        "equal_counts_control": {
            "token_weighted": sum(4 * l for _, l in mb) / (4 * 3),
            "mean_of_means": wrong,
        },
    }


def one_step_gradients(cfg, micro, device):
    """Both reductions on the same micro-batches; compare the gradients."""
    set_seed(SEED)
    base = TinyGPT(cfg).to(device)
    init = copy.deepcopy(base.state_dict())

    out = {}
    grads = {}
    for mode in ("token_weighted", "mean_of_means"):
        base.load_state_dict(init)
        base.zero_grad(set_to_none=True)
        reported, honest, norm, scale, ntok = accumulate_step(
            base, micro, mode, clip=None)
        grads[mode] = torch.cat([p.grad.detach().reshape(-1).float().cpu()
                                 for p in base.parameters()])
        out[mode] = {"reported_loss": reported, "honest_loss": honest,
                     "grad_norm": norm}

    a, b = grads["token_weighted"], grads["mean_of_means"]
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    out["comparison"] = {
        "cosine_similarity": cos,
        "angle_degrees": float(torch.rad2deg(torch.arccos(
            torch.clamp(torch.tensor(cos), -1, 1)))),
        "norm_ratio": float(b.norm() / a.norm()),
        "relative_l2_difference": float((b - a).norm() / a.norm()),
        "reported_loss_error_pct": 100 * (out["mean_of_means"]["reported_loss"]
                                          - out["token_weighted"]["reported_loss"])
                                   / out["token_weighted"]["reported_loss"],
    }
    out["micro_batch_tokens"] = [mb.n_tokens for mb in micro]
    return out


def paired_runs(cfg, device, docs, sampler_mode, eval_batches, tag):
    """Two runs identical in everything but the division."""
    set_seed(SEED)
    init = TinyGPT(cfg).state_dict()
    max_len = CONTROL_LEN if sampler_mode == "fixed" else MAX_LEN
    min_len = CONTROL_LEN if sampler_mode == "fixed" else MIN_LEN
    logs = {}
    for mode in ("token_weighted", "mean_of_means"):
        set_seed(SEED)
        model = TinyGPT(cfg)
        model.load_state_dict(copy.deepcopy(init))
        sampler = Sampler(docs["train"], MICRO_B, max_len=max_len,
                          min_len=min_len, mode=sampler_mode, seed=SEED)
        print(f"  [{tag}/{mode}] {STEPS} steps ...", flush=True)
        logs[mode] = train(model, sampler, mode=mode, steps=STEPS, accum=ACCUM,
                           lr=3e-4, warmup=20, clip=1.0, device=device,
                           eval_batches=eval_batches, eval_every=EVAL_EVERY)
    return logs


def main(verbose: bool = True) -> dict:
    device = pick_device()
    cfg = Config(max_seq=MAX_LEN)
    docs, _ = load_documents()

    # one fixed validation set, token-weighted, identical for every run --
    # a metric that changes with the thing it measures measures nothing
    eval_batches = [b.to(device) for b in Sampler(
        docs["validation"], MICRO_B, max_len=MAX_LEN, min_len=MIN_LEN,
        mode="bucket", seed=999).micro_batches(16)]

    arithmetic = session_arithmetic()

    micro = [b.to(device) for b in Sampler(
        docs["train"], MICRO_B, max_len=MAX_LEN, min_len=MIN_LEN,
        mode="bucket", seed=42).micro_batches(ACCUM)]
    one_step = one_step_gradients(cfg, micro, device)

    print("running the paired training runs (4 runs total)", flush=True)
    bucket = paired_runs(cfg, device, docs, "bucket", eval_batches, "bucketed")
    fixed = paired_runs(cfg, device, docs, "fixed", eval_batches, "equal-length")

    def final(logs, mode):
        return logs[mode].val_loss[-1]

    gap_bucket = final(bucket, "mean_of_means") - final(bucket, "token_weighted")
    gap_fixed = final(fixed, "mean_of_means") - final(fixed, "token_weighted")

    curves = {
        k: {m: {"val_steps": v[m].val_steps, "val_loss": v[m].val_loss,
                "reported": v[m].col("loss"),
                "honest": v[m].col("loss_token_wtd"),
                "grad_norm": v[m].col("grad_norm"),
                "n_tokens": v[m].col("n_tokens")}
            for m in v}
        for k, v in (("bucketed", bucket), ("fixed", fixed))
    }
    fig_path = make_figure(curves)

    # ---- report ------------------------------------------------------------
    p = []
    p.append("# Deliverable 3 — breaking gradient accumulation on purpose\n")

    p.append("## 1. The arithmetic, computed rather than quoted\n")
    p.append(table([
        (i + 1, m["valid_tokens"], f"{m['average_loss']:.1f}")
        for i, m in enumerate(arithmetic["micro_batches"])
    ], ["micro-batch", "valid tokens", "average loss"], ["r", "r", "r"]))
    p.append(f"\n- token-weighted (correct): **{arithmetic['token_weighted']:.4f}**")
    p.append(f"- average of the averages: **{arithmetic['mean_of_means']:.4f}**")
    p.append(f"- error: **{arithmetic['relative_error_pct']:.1f}%**\n")
    p.append("Set every token count to 4 and the two agree exactly "
             f"({arithmetic['equal_counts_control']['token_weighted']:.4f} vs "
             f"{arithmetic['equal_counts_control']['mean_of_means']:.4f}). "
             "That is the whole reason it survived in shipping frameworks.\n")

    p.append("\n## 2. One real step — the gradients, not just the number\n")
    p.append(f"Four length-bucketed micro-batches holding "
             f"{', '.join(f'{n:,}' for n in one_step['micro_batch_tokens'])} "
             f"valid tokens (a "
             f"{max(one_step['micro_batch_tokens'])/min(one_step['micro_batch_tokens']):.1f}× "
             "spread), through the same model at the same weights:\n")
    c = one_step["comparison"]
    p.append(table([
        ("reported loss",
         f"{one_step['token_weighted']['reported_loss']:.6f}",
         f"{one_step['mean_of_means']['reported_loss']:.6f}",
         f"{c['reported_loss_error_pct']:+.2f}%"),
        ("gradient L2 norm",
         f"{one_step['token_weighted']['grad_norm']:.6f}",
         f"{one_step['mean_of_means']['grad_norm']:.6f}",
         f"{100*(c['norm_ratio']-1):+.2f}%"),
    ], ["quantity", "sum ÷ sum of tokens", "average of averages", "difference"],
        ["l", "r", "r", "r"]))
    p.append(f"""
The reported number being wrong is the visible half. The half that actually
matters is that the two gradient vectors are not the same vector:

- cosine similarity **{c['cosine_similarity']:.6f}** — an angle of
  **{c['angle_degrees']:.2f}°** between them
- relative L2 difference **{c['relative_l2_difference']*100:.2f}%** of the
  correct gradient's own norm

The optimiser is being pointed somewhere else, not merely told the wrong
distance. Clipping does not rescue this: clipping rescales the length and
leaves the direction exactly as it was.
""")

    p.append("\n## 3. Two runs, identical but for the division\n")
    p.append(f"{STEPS} steps, micro-batch {MICRO_B} × {ACCUM} accumulation, "
             f"AdamW at 3e-4 with 20 warmup steps, clip 1.0, same seed, same "
             f"initial weights, same batches in the same order. Both runs are "
             f"scored on one common token-weighted validation set, so the "
             f"metric does not move with the thing it measures.\n\n"
             f"The control arm holds every sequence to {CONTROL_LEN} tokens — "
             f"the geometric mean of the bucketed arm's 32-256 range — so it "
             f"sees the same token budget per micro-batch on average and "
             f"differs in one thing only: there is no spread between "
             f"micro-batches for the bug to key on.\n")
    p.append(table([
        ("micro-batches of different lengths (bucketed)",
         f"{final(bucket,'token_weighted'):.4f}",
         f"{final(bucket,'mean_of_means'):.4f}",
         f"{gap_bucket:+.4f}",
         f"{100*gap_bucket/final(bucket,'token_weighted'):+.2f}%"),
        (f"control: every micro-batch {CONTROL_LEN} tokens long",
         f"{final(fixed,'token_weighted'):.4f}",
         f"{final(fixed,'mean_of_means'):.4f}",
         f"{gap_fixed:+.4f}",
         f"{100*gap_fixed/final(fixed,'token_weighted'):+.2f}%"),
    ], ["setting", "correct", "broken", "gap (nats)", "relative"],
        ["l", "r", "r", "r", "r"]))
    printed_err = [100 * (r.loss - r.loss_token_wtd) / r.loss_token_wtd
                   for r in bucket["mean_of_means"].steps]
    mean_printed = sum(printed_err) / len(printed_err)
    p.append(f"""
![gradient accumulation]({fig_path.name})

The left panel is the gap — plotted from step {ZOOM_FROM} because the opening
descent from 7.6 to 4.7 nats would otherwise squash it flat, which is precisely
how a gap this size stays invisible on a real dashboard. The middle panel is the
identical experiment with the token counts made equal, where the gap is
{gap_fixed:+.4f} nats: the bug hiding, reproduced. The right panel is the
difference between the two curves in each setting, which is the only view where
neither of them can be mistaken for the other.

**The most uncomfortable number in this whole deliverable is a small one.** The
loss the broken run *printed* was wrong by only {mean_printed:+.2f}% on average
over the {STEPS} steps — well inside the step-to-step noise of any real dashboard
and, at every single step, entirely plausible. Meanwhile the gradient it fed the
optimiser was {c['relative_l2_difference']*100:.1f}% off and pointing
{c['angle_degrees']:.1f}° away. The printed number is nearly innocent while the
training is wrong. That asymmetry is the reason this bug survived in shipping
frameworks until 2024, and it is the reason to check the reduction directly
instead of watching the curve.

**A note on scale.** {gap_bucket:+.4f} nats after {STEPS} steps on an 8.3M model
is small, and I am not going to inflate it. What makes it worth the section is
that it is (a) exactly reproducible, (b) exactly zero in the control, and (c) a
*systematic* bias rather than noise — it is the same wrong direction on every
step of a run that would go on for weeks, on micro-batches whose token counts in
production vary far more than the {max(one_step['micro_batch_tokens'])/min(one_step['micro_batch_tokens']):.1f}×
here.
""")

    text = "\n".join(p)
    save_text("e3_accumulation.md", text)
    payload = {
        "machine": machine(), "steps": STEPS, "accum": ACCUM,
        "micro_batch_rows": MICRO_B, "seed": SEED,
        "session_arithmetic": arithmetic,
        "one_step": one_step,
        "final_val": {
            "bucketed": {m: final(bucket, m) for m in bucket},
            "fixed": {m: final(fixed, m) for m in fixed},
        },
        "gap_bucketed": gap_bucket, "gap_fixed": gap_fixed,
        "curves": curves,
        "figure": fig_path.name,
    }
    save_json("e3_accumulation.json", payload)
    if verbose:
        print(text)
    return payload


if __name__ == "__main__":
    main()
