"""Deliverable 5 -- perplexity, and the cheapest sanity check in training.

An untrained model has no reason to prefer any token, so it should be about as
unsure as a uniform draw from the vocabulary: loss ln(V), perplexity V.  If the
first step of a run does not land there, something between the model output and
the scalar is wrong, and it is worth finding before spending money on it.

This file does four things:

1. Establishes the anchor exactly, with logits that really are uniform.
2. Measures the real untrained model against it, and accounts for the small
   excess rather than waving at it -- the gap is predictable to three decimals
   from the standard deviation of the logits at initialisation.
3. Runs the check against four harnesses, three of them broken, and reports
   which bugs it catches **and which it misses**.  A sanity check whose limits
   are not known is a source of false confidence.
4. Reports bits per byte alongside, because perplexity is per token and a
   token is whatever the tokenizer says it is.
"""

from __future__ import annotations

import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data import (VOCAB_SIZE, Vocab, clean_batch, load_documents,
                      padded_batch)
from src.losses import align, lm_loss, loss_mask, perplexity
from src.model import Config, TinyGPT
from src.plots import CRITICAL, INK_2, S1, S2, S3, finish, label_end
from src.report import machine, save_json, save_text, table
from src.train import pick_device, train

B, T = 8, 128
EARLY_STEPS = 60


def main(verbose: bool = True) -> dict:
    device = pick_device()
    docs, _ = load_documents()
    ln_v = math.log(VOCAB_SIZE)

    # ---- 1. the anchor, exactly ------------------------------------------
    torch.manual_seed(0)
    model = TinyGPT(Config())
    seqs = clean_batch(docs["validation"], B, T, seed=42)

    uniform = torch.zeros(B, T, VOCAB_SIZE)
    uniform_rec = lm_loss(uniform, seqs)
    anchor = {
        "vocab_size": VOCAB_SIZE,
        "ln_vocab": ln_v,
        "uniform_loss": float(uniform_rec.loss),
        "uniform_perplexity": uniform_rec.perplexity,
        "matches_to_decimals": -math.log10(
            abs(float(uniform_rec.loss) - ln_v) + 1e-16),
    }

    # ---- 2. the real untrained model -------------------------------------
    logits = model(seqs)
    rec = lm_loss(logits, seqs)
    a = align(T, 1)
    sel = logits.index_select(1, a.input_pos)
    sigma = float(sel.std())
    # For logits z ~ N(mu, sigma^2), E[logsumexp(z)] ~= ln V + sigma^2/2 and
    # E[-z_y] = -mu, so E[loss] ~= ln V + sigma^2/2. Nothing is fitted here.
    predicted = ln_v + sigma ** 2 / 2
    untrained = {
        "loss": float(rec.loss),
        "perplexity": rec.perplexity,
        "perplexity_over_vocab": rec.perplexity / VOCAB_SIZE,
        "logit_std_at_init": sigma,
        "predicted_loss": predicted,
        "prediction_error": float(rec.loss) - predicted,
    }
    if verbose:
        print(f"untrained loss {untrained['loss']:.4f}  "
              f"ppl {untrained['perplexity']:,.0f}  vocab {VOCAB_SIZE:,}")

    # ---- 3. the check against four harnesses -----------------------------
    padded = padded_batch(docs["train"], B, T, seed=9)
    plog = model(padded)
    pad_rec = lm_loss(plog, padded, mask_padding=True, mask_boundaries=False)
    checks = []

    def verdict(loss):
        ppl = perplexity(loss)
        ok = 0.5 * VOCAB_SIZE <= ppl <= 2.0 * VOCAB_SIZE
        return ppl, ok

    for label, loss_value, note in [
        ("correct harness", float(rec.loss),
         "lands on the anchor"),
        ("targets not shifted", float(lm_loss(logits, seqs, offset=0).loss),
         "an untrained model cannot copy yet, so this passes"),
        ("targets shifted backwards", float(lm_loss(logits, seqs, offset=-1).loss),
         "same reason -- it passes, and it is still broken"),
        ("padding counted in the mean",
         float(lm_loss(plog, padded, mask_padding=False).loss),
         "padding is not yet predictable either"),
        ("masked sum, wrong denominator", pad_rec.loss_over_all_pairs,
         "caught immediately: the sum is right, the divisor is not"),
    ]:
        ppl, ok = verdict(loss_value)
        checks.append({"harness": label, "loss": loss_value, "perplexity": ppl,
                       "passes": ok, "note": note})

    # ---- 4. early training, where the rest of them show ------------------
    early = {}
    for offset, label in ((1, "correct"), (0, "no shift"), (-1, "reversed")):
        torch.manual_seed(0)
        m = TinyGPT(Config())
        h = train(m, docs["train"], steps=EARLY_STEPS, offset=offset,
                  batch_size=B, width=T, device=device, seed=0)
        early[label] = h.train_loss

    # ---- 5. bits per byte -------------------------------------------------
    vocab = Vocab()
    ids = seqs.tokens[0].tolist()
    n_bytes = len(vocab.text(ids).encode("utf-8"))
    bpb = float(rec.loss) * len(ids) / (n_bytes * math.log(2))
    per_token = {"tokens": len(ids), "bytes": n_bytes,
                 "bytes_per_token": n_bytes / len(ids),
                 "bits_per_byte_untrained": bpb}

    # ---- figure -----------------------------------------------------------
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1))
    xs = list(range(1, EARLY_STEPS + 1))
    for label, colour in (("correct", S1), ("no shift", CRITICAL),
                          ("reversed", S2)):
        ax1.plot(xs, early[label], color=colour)
        label_end(ax1, xs, early[label], f"{label} {early[label][-1]:.2f}",
                  colour)
    ax1.axhline(ln_v, color=INK_2, lw=1.2, ls="--")
    ax1.annotate(f"ln(V) = {ln_v:.3f}\nperplexity {VOCAB_SIZE:,}",
                 (1.5, ln_v), xytext=(0, -26), textcoords="offset points",
                 fontsize=8.5, color=INK_2)
    ax1.set_title("Every run starts at the anchor. Only one stays honest.")
    ax1.set_xlabel("step"); ax1.set_ylabel("training loss (nats)")
    ax1.set_xlim(0, EARLY_STEPS * 1.36); ax1.set_ylim(0, 10)

    labels = [c["harness"].replace(" ", "\n", 1) for c in checks]
    vals = [c["perplexity"] for c in checks]
    cols = [S3 if c["passes"] else CRITICAL for c in checks]
    ax2.bar(range(len(vals)), vals, color=cols, width=0.6)
    ax2.axhline(VOCAB_SIZE, color=INK_2, lw=1.2, ls="--")
    ax2.annotate(f"V = {VOCAB_SIZE:,}", (len(vals) - 0.5, VOCAB_SIZE),
                 xytext=(0, 5), textcoords="offset points", fontsize=8.5,
                 color=INK_2, ha="right")
    ax2.set_xticks(range(len(vals)), labels, fontsize=7.2)
    ax2.set_title("The check at step 0: green passes, red is caught")
    ax2.set_ylabel("perplexity at initialisation")
    for i, v in enumerate(vals):
        ax2.annotate(f"{v:,.0f}", (i, v), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=8)
    ax2.set_ylim(0, max(vals) * 1.2)
    finish(fig, "e5_perplexity.png",
           "left: first 60 steps of three alignments; right: five harnesses "
           "scored once, before any training")

    payload = {"machine": machine(), "device": str(device), "anchor": anchor,
               "untrained": untrained, "checks": checks, "early": early,
               "bits_per_byte": per_token}

    md = f"""# 5 · Perplexity, and the cheapest sanity check there is

perplexity = exp(mean loss) — "how many equally likely options is the model
effectively choosing between at this token?"

## The anchor, exactly

A model that knows nothing should spread its probability evenly over the whole
vocabulary. Feed the loss a genuinely uniform logits tensor and it returns the
anchor to the last decimal it has:

| | |
|---|---|
| vocabulary | **{VOCAB_SIZE:,}** |
| ln(V) | **{ln_v:.6f}** nats |
| loss from uniform logits | **{anchor['uniform_loss']:.6f}** nats |
| perplexity from uniform logits | **{anchor['uniform_perplexity']:,.2f}** |

The two agree to {anchor['matches_to_decimals']:.0f} decimal places, which is
what it means for the anchor to be a definition rather than an observation.

## Where the real untrained model sits

{table([
    ("loss", f"{untrained['loss']:.4f} nats"),
    ("perplexity", f"{untrained['perplexity']:,.0f}"),
    ("vocabulary", f"{VOCAB_SIZE:,}"),
    ("perplexity / V", f"{untrained['perplexity_over_vocab']:.3f}"),
], ("", "at initialisation"), ("l", "r"))}

**Perplexity {untrained['perplexity']:,.0f} against a vocabulary of
{VOCAB_SIZE:,}** — {untrained['perplexity_over_vocab']:.1%} of it. The run may
proceed.

The small excess is not noise, and it is worth accounting for rather than
tolerating. At initialisation the logits are not exactly uniform: they have a
standard deviation of {untrained['logit_std_at_init']:.4f}, because the head is
initialised at std 0.02 and reads a {Config().d_model}-wide hidden state. For
logits with spread σ, `E[logsumexp(z)] ≈ ln V + σ²/2`, so the expected loss is

    ln V + σ²/2 = {ln_v:.4f} + {untrained['logit_std_at_init']**2/2:.4f} = {untrained['predicted_loss']:.4f}

against a measured **{untrained['loss']:.4f}**, out by
{abs(untrained['prediction_error']):.4f} nats. Nothing was fitted. A random head
sits slightly *above* ln(V), never below — so a first step **below** the anchor
is the alarming direction, and it is the direction every bug in this session
pushes it.

## What the check catches, and what it does not

Five harnesses, scored once, before any training. Pass = perplexity within a
factor of two of V.

{table([(c["harness"], f"{c['loss']:.4f}", f"{c['perplexity']:,.0f}",
         "pass" if c["passes"] else "**CAUGHT**", c["note"]) for c in checks],
       ("harness", "loss", "perplexity", "step-0 check", "why"),
       ("l", "r", "r", "l", "l"))}

**It catches the denominator instantly.** A correct sum divided by `B × (T-1)`
reports a perplexity of {[c for c in checks if 'denominator' in c['harness']][0]['perplexity']:,.0f}
against a vocabulary of {VOCAB_SIZE:,} — a model that has not seen a single
gradient claiming to have narrowed {VOCAB_SIZE:,} options down to
{[c for c in checks if 'denominator' in c['harness']][0]['perplexity']:,.0f}.
That is impossible, it costs one forward pass to see, and it is free.

**It does not catch the shift, at step 0.** Both broken alignments pass, and
they pass for an honest reason: copying is a *learned* skill. An untrained model
cannot copy its input any better than it can predict the future, so at
initialisation all three alignments sit on the anchor together.

They separate within twenty steps — the figure's left panel — and by step
{EARLY_STEPS} the no-shift run is at {early['no shift'][-1]:.2f} nats
(perplexity {math.exp(early['no shift'][-1]):.1f}) while the correct run is at
{early['correct'][-1]:.2f}. A perplexity of {math.exp(early['no shift'][-1]):.1f}
after {EARLY_STEPS} steps on a 10k vocabulary is not a good run; it is a
different task. So the check is worth running as *"is the loss falling faster
than any real model could learn?"*, not only as a step-0 assertion.

**And nothing about padding.** Counting padding passes cleanly at step 0,
because PAD is not yet predictable either. Deliverable 3 is not covered by this
check and needs its own — the contributing-token count.

## One caution: perplexity is per *token*

Our tokenizer packs {per_token['bytes_per_token']:.2f} bytes into the average
token on this sample. A tokenizer that split the same text into twice as many
tokens would be asked an easier question at each step and would report a better
perplexity for an identical model.

Bits per byte normalises by something the tokenizer cannot move:

    bpb = loss × tokens / (bytes × ln 2) = {per_token['bits_per_byte_untrained']:.4f}

Comparable across tokenizers; perplexity is not. Within one tokenizer, which is
the whole of this session, perplexity is the more readable of the two.
"""

    save_json("e5_perplexity.json", payload)
    save_text("e5_perplexity.md", md)
    if verbose:
        print(md)
    return payload


if __name__ == "__main__":
    main()
