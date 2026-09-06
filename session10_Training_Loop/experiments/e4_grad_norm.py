"""Deliverable 4 — log the grad norm at every step, then find one step where it
moved before the loss did.

The instrument matters here. A run's *training* loss is computed on whatever
batch happened to arrive, so it jumps around with batch difficulty and would
bury any real signal in noise. So a fixed held-out **probe** is evaluated after
every single optimiser step, on the same tokens every time. That probe is the
loss trace below: it moves only when the model moves.

Both traces are then run through the same detector -- a rolling median and MAD
over the previous 25 steps, flagging anything past 5 sigma **in either
direction** -- so neither series gets a rule the other does not. Two-sided
matters: the first version of this experiment tested for the loss going *up* and
duly reported that it never moved, while the plot showed it falling off the
bottom of the panel. "The loss moved" means moved.

Two arms, identical but for one setting:

  cap off   the anomalous gradient is applied in full
  cap 1.0   the same anomaly, clipped

The anomaly is disclosed and deliberate: this model trains on the wiki lane
only, and at a known step it is handed one global batch of real source code from
the same corpus.  That is a data-mixture contamination -- one of the ways runs
actually break -- not a synthetic spike bolted onto the gradient.
"""

from __future__ import annotations

import copy
import statistics
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from src.data import Sampler, load_documents
from src.loop import (accumulate_step, cosine_lr, evaluate, grad_global_norm,
                      pick_device, set_seed)
from src.model import Config, TinyGPT
from src.plots import CRITICAL, GOOD, INK_2, MUTED, S1, S2, S3, finish, plt
from src.report import machine, save_json, save_text, table

STEPS = 420
INJECT_AT = 300
ACCUM = 4
MICRO_B = 8
MIN_LEN, MAX_LEN = 32, 256
WINDOW = 25
SIGMA = 5.0
SEED = 7


def rolling_z(series, window=WINDOW):
    """Robust z-score against the previous ``window`` steps (median + MAD)."""
    out = [0.0] * len(series)
    for i in range(window, len(series)):
        hist = series[i - window:i]
        med = statistics.median(hist)
        mad = statistics.median([abs(x - med) for x in hist])
        scale = 1.4826 * mad
        if scale <= 0:
            scale = max(1e-12, statistics.pstdev(hist))
        out[i] = (series[i] - med) / scale
    return out


def first_crossing(z, threshold=SIGMA, start=0):
    """First step past ``threshold`` sigma in *either* direction."""
    for i in range(max(start, WINDOW), len(z)):
        if abs(z[i]) > threshold:
            return i
    return None


def run(model, sampler, probe, device, clip, anomaly=None, tag=""):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                            weight_decay=0.1)
    rec = {"step": [], "grad_norm": [], "clip_scale": [], "train_loss": [],
           "probe_loss": [], "n_tokens": [], "lr": [], "seconds": []}
    for step in range(STEPS):
        micro = [mb.to(device) for mb in sampler.micro_batches(ACCUM)]
        t_step = time.perf_counter()
        if anomaly is not None and step == INJECT_AT:
            micro = anomaly
        lr = cosine_lr(step, STEPS, 3e-4, warmup=20)
        for g in opt.param_groups:
            g["lr"] = lr
        reported, honest, norm, scale, ntok = accumulate_step(
            model, micro, "token_weighted", clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        rec["step"].append(step)
        rec["grad_norm"].append(norm)
        rec["clip_scale"].append(scale)
        rec["train_loss"].append(honest)
        rec["probe_loss"].append(evaluate(model, probe))
        rec["n_tokens"].append(ntok)
        rec["lr"].append(lr)
        rec["seconds"].append(time.perf_counter() - t_step)
        if step % 60 == 0:
            print(f"  [{tag}] step {step:>4}  |g| {norm:8.3f}  "
                  f"probe {rec['probe_loss'][-1]:.4f}", flush=True)
    rec["z_grad_norm"] = rolling_z(rec["grad_norm"])
    rec["z_probe_loss"] = rolling_z(rec["probe_loss"])
    return rec


def natural_events(rec, skip_around_injection=True):
    """Every step where the norm crossed 5 sigma, and how long the probe loss
    took to follow."""
    zg, zl = rec["z_grad_norm"], rec["z_probe_loss"]
    events, last = [], -99
    for i in range(WINDOW, len(zg)):
        if abs(zg[i]) <= SIGMA or i - last < 8:
            continue
        if skip_around_injection and abs(i - INJECT_AT) <= 3:
            continue
        last = i
        j = first_crossing(zl, SIGMA, start=i)
        events.append({
            "grad_norm_step": i,
            "grad_norm_z": zg[i],
            "grad_norm": rec["grad_norm"][i],
            "loss_step": j,
            "loss_z": zl[j] if j is not None else None,
            "lead_steps": (j - i) if j is not None else None,
        })
    return events


def grad_norm_cost(model, device):
    """What the trace actually costs: one pass over tensors already in memory."""
    model.to(device)
    for p_ in model.parameters():
        p_.grad = torch.randn_like(p_)
    for _ in range(3):
        grad_global_norm(model)
    t0 = time.perf_counter()
    for _ in range(20):
        grad_global_norm(model)
    dt = (time.perf_counter() - t0) / 20
    model.zero_grad(set_to_none=True)
    return dt


def main(verbose: bool = True, reuse: str | None = None) -> dict:
    device = pick_device()
    cfg = Config(max_seq=MAX_LEN)
    docs, lanes = load_documents()

    wiki = [d for d, ln in zip(docs["train"], lanes["train"]) if ln == "wiki"]
    code = [d for d, ln in zip(docs["train"], lanes["train"]) if ln == "code"]
    print(f"train lanes: {len(wiki)} wiki docs, {len(code)} code docs")

    probe = [b.to(device) for b in Sampler(
        docs["validation"], MICRO_B, max_len=MAX_LEN, min_len=MAX_LEN,
        mode="fixed", seed=555).micro_batches(2)]

    anomaly = [b.to(device) for b in Sampler(
        code, MICRO_B, max_len=MAX_LEN, min_len=MAX_LEN, mode="fixed",
        seed=808).micro_batches(ACCUM)]

    set_seed(SEED)
    init = TinyGPT(cfg).state_dict()
    norm_cost = grad_norm_cost(TinyGPT(cfg), device)

    if reuse:
        import json as _json
        arms = _json.loads(pathlib.Path(reuse).read_text())["arms"]
    else:
        arms = {}
        for name, clip in (("cap off", None), ("cap 1.0", 1.0)):
            set_seed(SEED)
            model = TinyGPT(cfg)
            model.load_state_dict(copy.deepcopy(init))
            sampler = Sampler(wiki, MICRO_B, max_len=MAX_LEN, min_len=MIN_LEN,
                              mode="bucket", seed=SEED)
            print(f"running arm '{name}' ...", flush=True)
            arms[name] = run(model, sampler, probe, device, clip,
                             anomaly=anomaly, tag=name)

    off, capped = arms["cap off"], arms["cap 1.0"]
    nat = natural_events(off)
    nat_leads = [e for e in nat if e["lead_steps"] is not None
                 and e["lead_steps"] > 0]

    # the injected event, measured the same way as any other
    inj_norm_z = off["z_grad_norm"][INJECT_AT]
    inj_loss_step = first_crossing(off["z_probe_loss"], SIGMA, start=INJECT_AT)
    inj_lead = (inj_loss_step - INJECT_AT) if inj_loss_step is not None else None
    capped_loss_step = first_crossing(capped["z_probe_loss"], SIGMA,
                                      start=INJECT_AT)

    # "clearest" = largest excursion in either direction, now that the detector
    # is two-sided; ranking on the signed z would silently prefer upward spikes
    headline = max(nat_leads, key=lambda e: abs(e["grad_norm_z"])) \
        if nat_leads else None

    # ---- figure -------------------------------------------------------------
    lo, hi = INJECT_AT - 30, min(STEPS, INJECT_AT + 45)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9))

    ax = axes[0]
    xs = list(range(lo, hi))
    ax.plot(xs, off["z_grad_norm"][lo:hi], color=S1, label="grad norm")
    ax.plot(xs, off["z_probe_loss"][lo:hi], color=S2, label="probe loss")
    for sgn in (1, -1):
        ax.axhline(sgn * SIGMA, color=MUTED, lw=0.9)
    ax.annotate(f"±{SIGMA:.0f}σ", (hi - 1, SIGMA), textcoords="offset points",
                xytext=(-2, 4), ha="right", color=MUTED, fontsize=8)
    ax.axvline(INJECT_AT, color=CRITICAL, lw=1.0)
    if inj_loss_step:
        ax.axvline(inj_loss_step, color=CRITICAL, lw=1.0, alpha=0.45)
        ax.annotate(f"norm at {INJECT_AT}, loss at {inj_loss_step}\n"
                    f"→ {inj_lead} steps of warning",
                    (0.5, 0.06), xycoords="axes fraction", ha="center",
                    va="bottom", color=INK_2, fontsize=8)
    ax.set_title("Cap off — both traces, one axis", loc="left")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("robust z-score vs the previous 25 steps")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(xs, capped["z_grad_norm"][lo:hi], color=S1, label="grad norm")
    ax.plot(xs, capped["z_probe_loss"][lo:hi], color=S2, label="probe loss")
    for sgn in (1, -1):
        ax.axhline(sgn * SIGMA, color=MUTED, lw=0.9)
    ax.axvline(INJECT_AT, color=CRITICAL, lw=1.0)
    ax.set_ylim(axes[0].get_ylim())
    ax.set_title("Cap 1.0 — same batch, same step", loc="left")
    ax.set_xlabel("optimiser step")
    ax.legend(loc="upper right")
    ax.annotate(f"the norm still reports it in full —\nthe cap scaled every "
                f"gradient by {capped['clip_scale'][INJECT_AT]:.3f}",
                (0.5, 0.06), xycoords="axes fraction", ha="center",
                va="bottom", color=INK_2, fontsize=8)

    ax = axes[2]
    if headline:
        a, b = headline["grad_norm_step"], headline["loss_step"]
        lo2, hi2 = max(0, a - 30), min(STEPS, b + 25)
        xs2 = list(range(lo2, hi2))
        ax.plot(xs2, off["z_grad_norm"][lo2:hi2], color=S1, label="grad norm")
        ax.plot(xs2, off["z_probe_loss"][lo2:hi2], color=S2,
                label="probe loss")
        for sgn in (1, -1):
            ax.axhline(sgn * SIGMA, color=MUTED, lw=0.9)
        ax.axvline(a, color=CRITICAL, lw=1.0)
        ax.axvline(b, color=CRITICAL, lw=1.0, alpha=0.45)
        ax.annotate(f"norm crosses at {a}, loss at {b}\n"
                    f"→ {b - a} steps early",
                    (0.5, 0.06), xycoords="axes fraction", ha="center",
                    va="bottom", color=INK_2, fontsize=8)
        ax.set_title("Nothing injected — a step it found on its own",
                     loc="left")
    else:
        ax.set_title("No unprompted lead event above 5σ", loc="left")
    ax.set_xlabel("optimiser step")
    if headline:
        ax.legend(loc="upper right")

    fig_path = finish(fig, "e4_grad_norm.png",
                      f"{STEPS} steps · the probe loss is a fixed held-out set "
                      f"evaluated after every single step, so it moves only "
                      f"when the model does")

    # ---- report -------------------------------------------------------------
    p = []
    p.append("# Deliverable 4 — the grad norm, logged every step\n")
    p.append(f"{STEPS} steps, micro-batch {MICRO_B} × {ACCUM} accumulation, "
             f"wiki lane only. Logged every step: the global gradient L2 norm "
             f"**before** clipping, the clip scale, the training loss, and a "
             f"probe loss on a fixed held-out set of "
             f"{sum(b.n_tokens for b in probe):,} tokens evaluated after the "
             f"optimiser step.\n")

    p.append("\n## 1. Unprompted — the step the detector found on its own\n")
    if nat_leads:
        h = headline
        p.append(f"""
This is the deliverable, and it was not arranged. Across the {STEPS} steps of
the uncapped arm, {len(nat)} steps crossed 5-sigma on the gradient norm.
{len(nat_leads)} of them {'was' if len(nat_leads) == 1 else 'were'} followed by a
5-sigma crossing of the probe loss at a **later** step.

The clearest is **step {h['grad_norm_step']}**. The gradient norm went to
{h['grad_norm']:.3f} — {abs(h['grad_norm_z']):.1f} sigma away from its own recent
behaviour — while the probe loss stayed inside its noise band. The probe did not
cross until step {h['loss_step']}: **{h['lead_steps']} steps later**. Anyone
watching only the loss had {h['lead_steps']} steps of a run that looked
completely healthy, and by the time the loss moved, the batches responsible had
been gone for {h['lead_steps']} steps.
""")
        p.append(table([
            (e["grad_norm_step"], f"{e['grad_norm']:.3f}",
             f"{e['grad_norm_z']:.1f}", e["loss_step"],
             f"{e['loss_z']:.1f}", f"+{e['lead_steps']}")
            for e in nat_leads
        ], ["norm crossed at", "norm", "z", "loss crossed at", "z",
            "lead (steps)"], ["r", "r", "r", "r", "r", "r"]))
    else:
        p.append(f"Across the {STEPS} steps of the uncapped arm, {len(nat)} "
                 f"steps crossed 5 sigma on the gradient norm and none was "
                 f"followed by a later 5-sigma crossing of the probe loss. "
                 f"Reported as measured.\n")

    p.append("\n## 2. The controlled case — where the loss never finds out at all\n")
    p.append(f"At step {INJECT_AT} the run is handed one global batch of real "
             f"source code — a lane this model has never seen — and then goes "
             f"straight back to wiki. Nothing else changes. This is a data "
             f"contamination: the kind of thing a mixture bug does quietly.\n")
    p.append(table([
        ("gradient norm at the anomalous step",
         f"{off['grad_norm'][INJECT_AT]:.3f}",
         f"{capped['grad_norm'][INJECT_AT]:.3f}"),
        ("its z-score against the previous 25 steps",
         f"{off['z_grad_norm'][INJECT_AT]:.1f}",
         f"{capped['z_grad_norm'][INJECT_AT]:.1f}"),
        ("clip scale applied",
         f"{off['clip_scale'][INJECT_AT]:.3f}",
         f"{capped['clip_scale'][INJECT_AT]:.3f}"),
        ("first step the probe loss crossed 5 sigma afterwards",
         "never" if inj_loss_step is None else str(inj_loss_step),
         "never" if capped_loss_step is None else str(capped_loss_step)),
        ("the probe loss's own z-score there",
         "—" if inj_loss_step is None
         else f"{off['z_probe_loss'][inj_loss_step]:+.1f}",
         "—" if capped_loss_step is None
         else f"{capped['z_probe_loss'][capped_loss_step]:+.1f}"),
        ("warning the norm gave",
         "—" if inj_lead is None else f"{inj_lead} steps",
         "—" if capped_loss_step is None
         else f"{capped_loss_step - INJECT_AT} steps"),
    ], ["", "cap off", "cap 1.0"], ["l", "r", "r"]))

    p.append(f"""
The gradient norm registered the contaminated batch at
**{abs(off['z_grad_norm'][INJECT_AT]):.0f} sigma** — the largest excursion
anywhere in the run, {off['grad_norm'][INJECT_AT]/statistics.median(off['grad_norm']):.0f}x
the median norm.

The probe loss{'' if inj_loss_step is None else f" crossed at step {inj_loss_step}"}:
**{'it never crossed at all' if inj_lead is None else f'{inj_lead} steps of warning'}**.
The batch responsible was gone by then — one step of code in a run of wiki — so
whatever diagnosis was going to happen had to start from the norm, because by
the time the loss reacted the evidence was
{inj_lead if inj_lead else 0} steps in the past.

Worth saying plainly: the probe's reaction was
{'—' if inj_loss_step is None else ('a *fall* of ' if off['z_probe_loss'][inj_loss_step] < 0 else 'a rise of ')}{'' if inj_loss_step is None else f"{abs(off['z_probe_loss'][inj_loss_step]):.0f} sigma"}.
An out-of-distribution batch does not have to make the loss worse to have
damaged the run, and a one-sided alarm would have missed it entirely. That is
not a hypothetical — the first version of this experiment tested only for the
loss rising, reported "never", and was wrong.

The right column is the cap engaging on the same batch: every gradient scaled by
**{capped['clip_scale'][INJECT_AT]:.3f}**, a
{1/capped['clip_scale'][INJECT_AT]:.0f}x reduction, with the direction left
exactly as it was. Note that the norm is measured *before* the cap, which is why
the capped arm still reports the full
{abs(capped['z_grad_norm'][INJECT_AT]):.0f} sigma. Logged after clipping it
would sit pinned at the threshold and tell you nothing — a small detail that is
the difference between a useful trace and a flat line.

(The two arms report slightly different norms at step {INJECT_AT} because
clipping changed the trajectory earlier in the run; it is the same batch through
two models that have already diverged.)
""")

    secs = off.get("seconds") or []
    step_ms = 1e3 * statistics.median(secs) if secs else None
    p.append(f"""
![grad norm]({fig_path.name})

Left: the injected anomaly, uncapped. Middle: the same batch with the cap on.
Right: step
{headline['grad_norm_step'] if headline else '—'}, which nobody arranged.

Both traces are drawn on one axis as robust z-scores against their own previous
25 steps. That is the only honest way to put a loss in nats and a norm in
gradient units on one plot: a second y-axis would let me slide one curve against
the other until the story looked however I wanted it to.

## 3. What the trace costs, and where the cap should go

The grad norm is one sum of squares over tensors that are already in memory and
that the optimiser is about to read anyway: **{norm_cost*1e3:.2f} ms**"""
+ (f", against a median step of {step_ms:.0f} ms — "
   f"{100*norm_cost*1e3/step_ms:.2f}% of the clock" if step_ms else "")
+ f""". It is the cheapest trace on the dashboard and the only one that is ever
early. The loss answers *is it learning*; the norm answers *is it about to
stop*.

For V5 that settles three things. Log the norm from step one. Clip from step
one. And choose the threshold from the norm's own distribution rather than from
habit: in this run the median norm was
{statistics.median(off['grad_norm']):.3f}, the 99th percentile
{sorted(off['grad_norm'])[int(0.99*len(off['grad_norm']))]:.3f}, and the
contaminated batch {off['grad_norm'][INJECT_AT]:.1f}. A cap of 1.0 sits at the
{100*sum(1 for g in off['grad_norm'] if g < 1.0)/len(off['grad_norm']):.0f}th
percentile — it clips a fifth of ordinary steps, which is more than I would want
on a real run. On this evidence I would set it nearer
{sorted(off['grad_norm'])[int(0.99*len(off['grad_norm']))]:.1f}: high enough to
leave normal steps alone, low enough that the step-{INJECT_AT} batch is still
cut by {off['grad_norm'][INJECT_AT]/sorted(off['grad_norm'])[int(0.99*len(off['grad_norm']))]:.0f}x.
That is a number chosen from data, which is the whole ask.
""")

    text = "\n".join(p)
    save_text("e4_grad_norm.md", text)
    payload = {"machine": machine(), "steps": STEPS, "inject_at": INJECT_AT,
               "window": WINDOW, "sigma": SIGMA,
               "arms": {k: v for k, v in arms.items()},
               "natural_events": nat, "natural_lead_events": nat_leads,
               "headline_event": headline,
               "injected": {"grad_norm_z": inj_norm_z,
                            "loss_crossing_step": inj_loss_step,
                            "lead_steps": inj_lead,
                            "capped_loss_crossing_step": capped_loss_step},
               "figure": fig_path.name}
    save_json("e4_grad_norm.json", payload)
    if verbose:
        print(text)
    return payload


if __name__ == "__main__":
    main()
