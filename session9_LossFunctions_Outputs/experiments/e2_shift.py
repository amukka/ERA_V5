"""Deliverable 2 -- verify the shift by reading it, then prove why you must.

Part one is the assignment's instruction, done literally: the token *strings*,
inputs beside targets, for the correct alignment and for the two ways of
getting it wrong.  A wall of integers hides an off-by-one.  A column of words
does not: "of" should be followed by "India", and if the table says "of" is
followed by "of" you can see it without knowing anything about the code.

Part two is why the reading matters.  Each of the three alignments trains a
model of its own, from the same seed on the same batches, and the two broken
ones produce the better-looking loss curve.  They are not subtly worse. They
reach a loss the correct objective cannot approach, because they have been
handed the answer, and nothing anywhere raises an exception.

The copy rate is the tell.  For each trained model, how often is its top
prediction at position i just some token it was already given?
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data import VOCAB_SIZE, Vocab, clean_batch, load_documents
from src.losses import align, lm_loss
from src.model import Config, TinyGPT
from src.plots import INK_2 as INK_BAR
from src.plots import CRITICAL, S1, S2, S3, finish, label_end
from src.report import machine, save_json, save_text, table
from src.train import evaluate, pick_device, train

STEPS = 400
N_SHOWN = 16
ARMS = [
    (1, "correct", "position i predicts token i+1"),
    (0, "no shift", "position i predicts token i"),
    (-1, "reversed", "position i predicts token i-1"),
]


def show(piece: str) -> str:
    """A token string, made safe to put in a markdown table cell."""
    out = (piece.replace("\\", "\\\\").replace("\n", "\\n")
           .replace("\r", "\\r").replace("\t", "\\t").replace("|", "\\|"))
    return f"`{out}`" if out else "`` "


@torch.no_grad()
def copy_rate(model, seqs, device):
    """How often the model's top prediction at position i is a token it has
    already been shown at or before i.

    A model trained on the correct objective has to guess.  A model trained on
    a broken alignment has learned to copy, and this number says so.
    """
    seqs = seqs.to(device)
    logits = model(seqs)
    top = logits.argmax(-1)                       # (B, T)
    tokens = seqs.tokens
    same_position = (top == tokens).float().mean()
    prev = torch.zeros_like(top, dtype=torch.bool)
    prev[:, 1:] = top[:, 1:] == tokens[:, :-1]
    return float(same_position), float(prev.float().mean())


def main(verbose: bool = True) -> dict:
    device = pick_device()
    docs, _ = load_documents()
    vocab = Vocab()

    # ---- part one: read the pairs ----------------------------------------
    sample = clean_batch(docs["validation"], 1, 96, seed=11)
    ids = sample.tokens[0]
    tables, alignments = {}, {}
    for offset, label, story in ARMS:
        a = align(len(ids), offset)
        rows = []
        for j in range(N_SHOWN):
            i, t = int(a.input_pos[j]), int(a.target_pos[j])
            rows.append((str(i), show(vocab.piece(int(ids[i]))), "->",
                         str(t), show(vocab.piece(int(ids[t])))))
        tables[label] = table(
            rows, ("pos i", "input token at i", "", "pos", "target token"),
            ("r", "l", "l", "r", "l"))
        alignments[label] = {"offset": offset, "story": story,
                             "n_pairs": int(len(a.input_pos))}

    readable = vocab.text(ids[:48].tolist())

    # ---- part two: train one model per alignment -------------------------
    probe = clean_batch(docs["validation"], 8, 128, seed=999)
    curves, finals = {}, {}
    for offset, label, _ in ARMS:
        torch.manual_seed(0)
        model = TinyGPT(Config())
        hist = train(model, docs["train"], steps=STEPS, offset=offset,
                     device=device, seed=0, probe=probe, probe_every=25)
        same, prev = copy_rate(model, sample, device)
        curves[label] = hist.train_loss
        finals[label] = {
            "offset": offset,
            "final_train_loss": hist.train_loss[-1],
            "mean_last_20": float(sum(hist.train_loss[-20:]) / 20),
            "perplexity": float(torch.exp(torch.tensor(
                sum(hist.train_loss[-20:]) / 20))),
            "honest_next_token_loss": evaluate(model, probe.to(device), 1),
            "copy_rate_same_position": same,
            "copy_rate_previous_token": prev,
            "probe": hist.probe_loss,
        }
        if verbose:
            print(f"{label:<9} train {finals[label]['mean_last_20']:.4f}  "
                  f"honest {finals[label]['honest_next_token_loss']:.4f}")

    # ---- figure -----------------------------------------------------------
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1))
    colours = {"correct": S1, "no shift": CRITICAL, "reversed": S2}
    for label in curves:
        xs = list(range(1, STEPS + 1))
        ys = curves[label]
        ax1.plot(xs, ys, color=colours[label], label=label)
        label_end(ax1, xs, ys, f"{label} {ys[-1]:.2f}", colours[label])
    ax1.set_title("The loss each alignment reports about itself")
    ax1.set_xlabel("step")
    ax1.set_ylabel("training loss (nats)")
    ax1.set_xlim(0, STEPS * 1.28)

    names = [lb for _, lb, _ in ARMS]
    honest = [finals[n]["honest_next_token_loss"] for n in names]
    reported = [finals[n]["mean_last_20"] for n in names]
    y = range(len(names))
    ax2.barh([i + 0.19 for i in y], reported, height=0.36, color=S3,
             label="loss it reported")
    ax2.barh([i - 0.19 for i in y], honest, height=0.36, color=INK_BAR,
             label="loss on next-token prediction, scored honestly")
    ax2.set_yticks(list(y), names)
    ax2.invert_yaxis()
    ax2.set_title("What it reported vs what it can actually do")
    ax2.set_xlabel("loss (nats)")
    ax2.legend(loc="lower right")
    for i, (r, h) in enumerate(zip(reported, honest)):
        ax2.annotate(f"{r:.2f}", (r, i + 0.19), va="center", ha="left",
                     xytext=(4, 0), textcoords="offset points", fontsize=8.5)
        ax2.annotate(f"{h:.2f}", (h, i - 0.19), va="center", ha="left",
                     xytext=(4, 0), textcoords="offset points", fontsize=8.5)
    ax2.set_xlim(0, max(honest + reported) * 1.18)
    path = finish(fig, "e2_shift.png",
                  f"{STEPS} steps per arm, same seed and same batches; "
                  f"honest loss is next-token prediction on a held-out batch")

    payload = {
        "machine": machine(), "device": str(device), "steps": STEPS,
        "sample_text": readable, "alignments": alignments, "arms": finals,
        "curves": curves,
    }

    c, n, r = finals["correct"], finals["no shift"], finals["reversed"]
    md = f"""# 2 · Verify the shift by printing the strings

The sample, decoded, so the tables below can be read as English:

> {readable.strip()[:300]!r}

## The correct alignment: `logits[:, :-1]` against `tokens[:, 1:]`

{alignments['correct']['story']}. Read the two token columns: the target column
is the input column moved up one row. That is the whole of next-token
prediction, and it is checkable by eye.

{tables['correct']}

## Wrong alignment 1: no shift

{alignments['no shift']['story']} — the token it was just handed.

{tables['no shift']}

## Wrong alignment 2: shifted the other way

{alignments['reversed']['story']} — a token already inside its own context.

{tables['reversed']}

## Why this is worth doing by eye

Each alignment trained an identical model from an identical seed on identical
batches for {STEPS} steps.

{table([
    (lb,
     f"{finals[lb]['mean_last_20']:.4f}",
     f"{finals[lb]['perplexity']:,.1f}",
     f"{finals[lb]['honest_next_token_loss']:.4f}",
     f"{100*finals[lb]['copy_rate_same_position']:.1f}%",
     f"{100*finals[lb]['copy_rate_previous_token']:.1f}%")
    for _, lb, _ in ARMS],
    ("alignment", "loss it reports", "perplexity", "honest next-token loss",
     "top-1 = current token", "top-1 = previous token"),
    ("l", "r", "r", "r", "r", "r"))}

**The two broken objectives report the better number.** The correct one settles
at **{c['mean_last_20']:.4f}** nats. No-shift reaches **{n['mean_last_20']:.4f}**
and reversed reaches **{r['mean_last_20']:.4f}** — a loss the honest objective
cannot get near, on a curve with no kink, no spike and no warning in it.

**Nothing was learned.** Scored on the task anyone actually wanted — predict the
next token — the no-shift model is at {n['honest_next_token_loss']:.4f} nats
against the correct model's {c['honest_next_token_loss']:.4f}. It is
{n['honest_next_token_loss'] - c['honest_next_token_loss']:.2f} nats *worse*
while reporting a loss {c['mean_last_20'] - n['mean_last_20']:.2f} nats *better*.

**And you can see what it learned instead.** Its top prediction at position i is
the token at position i **{100*n['copy_rate_same_position']:.1f}%** of the time.
It is an identity function with {VOCAB_SIZE:,} outputs. The reversed model does
the same thing one step over: its top prediction matches the *previous* token
**{100*r['copy_rate_previous_token']:.1f}%** of the time.

The correct model copies the current token {100*c['copy_rate_same_position']:.1f}%
of the time, which is roughly what guessing looks like.

Both bugs are three characters wide. Neither raises anything. The only cheap
defence is the table at the top of this page.
"""

    save_json("e2_shift.json", payload)
    save_text("e2_shift.md", md)
    if verbose:
        print(md)
    return payload




if __name__ == "__main__":
    main()
