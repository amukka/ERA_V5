"""Deliverable 3 -- mask padding, and confirm the contributing count changes.

The literal ask is one line of arithmetic: the denominator stops being
`B x (T-1)`.  That part takes a paragraph.

The rest of this file is the reason the ask exists.  Padding is *trivially
predictable* -- after a few dozen steps the model knows that PAD follows PAD --
so counting it does not merely dilute the loss, it feeds it a large number of
free correct answers.  The reported loss improves while the model gets worse at
the only thing anyone wanted.  Two models, identical except for one boolean,
say how much.

The third arm is the session's other quiet bug: mask correctly, then divide by
`B x (T-1)` anyway.  The sum is right and the denominator is not, so the loss is
scaled by whatever fraction of the batch happened to be real -- a fraction that
changes with every batch.
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data import PAD_ID, load_documents, padded_batch
from src.losses import align, lm_loss, loss_mask
from src.model import Config, TinyGPT
from src.plots import CRITICAL, S1, S3, finish, label_end
from src.report import machine, save_json, save_text, table
from src.train import evaluate, pick_device, train

STEPS = 400
B, T = 8, 128


@torch.no_grad()
def pad_mass(model, seqs, device) -> dict:
    """How much probability the model puts on PAD, and how often it wins.

    A model that was never scored on padding has no reason to name it; a model
    that was scored on padding has been rewarded for naming it thousands of
    times. This is where its capacity went.
    """
    seqs = seqs.to(device)
    probs = torch.softmax(model(seqs), dim=-1)
    return {
        "mean_pad_probability": float(probs[..., PAD_ID].mean()),
        "argmax_is_pad_fraction": float(
            (model(seqs).argmax(-1) == PAD_ID).float().mean()),
    }


def main(verbose: bool = True) -> dict:
    device = pick_device()
    docs, _ = load_documents()

    # ---- the count -------------------------------------------------------
    torch.manual_seed(0)
    model = TinyGPT(Config())
    seqs = padded_batch(docs["train"], B, T, seed=5)
    a = align(T, 1)
    logits = model(seqs)

    unmasked = lm_loss(logits, seqs, mask_padding=False, mask_boundaries=False)
    masked = lm_loss(logits, seqs, mask_padding=True, mask_boundaries=False)
    keep = loss_mask(seqs, a, mask_padding=True, mask_boundaries=False)

    row_lengths = seqs.valid.sum(1).tolist()
    counts = {
        "B": B, "T": T,
        "pairs_before_masking": int(unmasked.n_contributing),
        "pairs_after_masking": int(masked.n_contributing),
        "dropped": int(unmasked.n_contributing - masked.n_contributing),
        "row_lengths": row_lengths,
        "padded_fraction": 1 - masked.n_contributing / unmasked.n_contributing,
        "loss_counting_padding": float(unmasked.loss),
        "loss_masked": float(masked.loss),
        "loss_masked_wrong_denominator": masked.loss_over_all_pairs,
    }

    # the same batch, one row at a time, to show the denominator moving
    per_batch = []
    for s in range(6):
        sb = padded_batch(docs["train"], B, T, seed=100 + s)
        m = loss_mask(sb, a, True, False)
        per_batch.append({"seed": 100 + s, "contributing": int(m.sum()),
                          "of": int(m.numel()),
                          "fraction": float(m.float().mean())})

    # ---- two models, one boolean apart -----------------------------------
    probe = padded_batch(docs["validation"], 8, T, seed=7777)
    arms = {}
    for label, mask_padding in (("masks padding", True),
                                ("counts padding", False)):
        torch.manual_seed(0)
        m = TinyGPT(Config())
        hist = train(m, docs["train"], steps=STEPS, kind="padded",
                     batch_size=B, width=T, mask_padding=mask_padding,
                     mask_boundaries=False, device=device, seed=0,
                     probe=probe, probe_every=25)
        arms[label] = {
            "mask_padding": mask_padding,
            "reported_loss": float(sum(hist.train_loss[-20:]) / 20),
            "honest_loss": evaluate(m, probe.to(device), 1),
            "curve": hist.train_loss,
            "probe": hist.probe_loss,
            **pad_mass(m, probe, device),
        }
        if verbose:
            print(f"{label:<16} reported {arms[label]['reported_loss']:.4f}  "
                  f"honest {arms[label]['honest_loss']:.4f}")

    good, bad = arms["masks padding"], arms["counts padding"]

    # ---- figure ----------------------------------------------------------
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1))
    xs = list(range(1, STEPS + 1))
    for label, colour in (("masks padding", S1), ("counts padding", CRITICAL)):
        ax1.plot(xs, arms[label]["curve"], color=colour, label=label)
        label_end(ax1, xs, arms[label]["curve"],
                  f"{label} {arms[label]['curve'][-1]:.2f}", colour)
    ax1.set_title("The loss each one reports")
    ax1.set_xlabel("step"); ax1.set_ylabel("training loss (nats)")
    ax1.set_xlim(0, STEPS * 1.34)

    for label, colour in (("masks padding", S1), ("counts padding", CRITICAL)):
        pts = arms[label]["probe"]
        ax2.plot([p[0] for p in pts], [p[1] for p in pts], color=colour)
        label_end(ax2, [p[0] for p in pts], [p[1] for p in pts],
                  f"{label} {pts[-1][1]:.2f}", colour)
    ax2.set_title("The loss on real tokens only, scored the same way for both")
    ax2.set_xlabel("step"); ax2.set_ylabel("held-out loss (nats)")
    ax2.set_xlim(0, STEPS * 1.34)
    finish(fig, "e3_padding.png",
           f"{STEPS} steps each, identical seed and batches; the only "
           f"difference is whether padded pairs enter the mean")

    payload = {"machine": machine(), "device": str(device), "steps": STEPS,
               "counts": counts, "per_batch": per_batch, "arms": arms}

    md = f"""# 3 · Mask padding, and watch the denominator move

## The count changes

One batch, `B={B}`, `T={T}`. The rows hold {row_lengths} real tokens, so
{counts['padded_fraction']*100:.1f}% of the slots are padding.

{table([
    ("no mask — every pair counts", f"{counts['pairs_before_masking']}",
     f"{counts['loss_counting_padding']:.4f}", "B × (T-1)"),
    ("padding masked out", f"{counts['pairs_after_masking']}",
     f"{counts['loss_masked']:.4f}", "sum of the mask"),
    ("masked, wrong denominator", f"{counts['pairs_before_masking']}",
     f"{counts['loss_masked_wrong_denominator']:.4f}", "B × (T-1), incorrectly"),
], ("what the loss counts", "contributing tokens", "loss (untrained)",
    "denominator"), ("l", "r", "r", "l"))}

**{counts['pairs_before_masking']} → {counts['pairs_after_masking']}**, a drop of
{counts['dropped']} pairs. That is the deliverable, and it is the only part of
this that is visible at initialisation.

The denominator is not a constant you can hard-code, either — it is a property
of the batch:

{table([(str(r["seed"]), f"{r['contributing']}", f"{r['of']}",
         f"{100*r['fraction']:.1f}%") for r in per_batch],
       ("batch", "contributing", "of B × (T-1)", "fraction real"),
       ("r", "r", "r", "r"))}

Divide by `B × (T-1)` and the loss is multiplied by that last column, which
moves every step. The third row of the first table is that bug on this batch:
{counts['loss_masked_wrong_denominator']:.4f} instead of {counts['loss_masked']:.4f},
{100*(counts['loss_masked_wrong_denominator']/counts['loss_masked'] - 1):.1f}%
low, for a sum that was computed correctly.

## Why it matters more than dilution

Two models, {STEPS} steps, same seed, same batches. One boolean apart.

{table([
    (lb, f"{arms[lb]['reported_loss']:.4f}", f"{arms[lb]['honest_loss']:.4f}",
     f"{100*arms[lb]['mean_pad_probability']:.2f}%",
     f"{100*arms[lb]['argmax_is_pad_fraction']:.1f}%")
    for lb in ("masks padding", "counts padding")],
    ("arm", "loss it reports", "loss on real tokens", "mean P(PAD)",
     "top-1 is PAD"), ("l", "r", "r", "r", "r"))}

**The broken arm reports the better number.** {bad['reported_loss']:.4f} against
{good['reported_loss']:.4f} — it looks
{100*(1 - bad['reported_loss']/good['reported_loss']):.0f}% better trained.

**On real tokens it is worse.** {bad['honest_loss']:.4f} against
{good['honest_loss']:.4f}, a gap of
{bad['honest_loss'] - good['honest_loss']:+.4f} nats, both scored identically on
the same held-out batch.

Put those two rows next to each other and the shape of the bug is clear. The
reported loss is wrong by
**{100*(1 - bad['reported_loss']/good['reported_loss']):.0f}%**. The model is
worse by **{100*(bad['honest_loss']/good['honest_loss'] - 1):.1f}%**. The number
on the dashboard is off by roughly
{(1 - bad['reported_loss']/good['reported_loss']) / (bad['honest_loss']/good['honest_loss'] - 1):.0f}×
more than the model is damaged — which is why this survives code review. Nobody
is looking for a bug that makes the loss *better*.

**Because that is where its capacity went.** It puts
{100*bad['mean_pad_probability']:.2f}% of its probability mass on PAD and names
PAD as its top prediction at {100*bad['argmax_is_pad_fraction']:.1f}% of
positions, against {100*good['mean_pad_probability']:.2f}% and
{100*good['argmax_is_pad_fraction']:.1f}% for the arm that never scored it. PAD
is the easiest token in the vocabulary — it is perfectly predictable from
position alone — so the loss falls fastest by learning it, and a loss that falls
is exactly what the run looks like it wants.

Note that attention was masked correctly in **both** arms: no position ever
attended to a padded key. The bug is entirely in the mean, in one line, several
hundred lines away from the attention mask that is doing its job.
"""

    save_json("e3_padding.json", payload)
    save_text("e3_padding.md", md)
    if verbose:
        print(md)
    return payload


if __name__ == "__main__":
    main()
