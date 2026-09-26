"""Deliverable 4 -- pack two documents into one sequence, mask the join.

Packing exists to stop paying for padding: fill every slot with real text and
the batch is all signal.  It plants a different bug in the same place.  The
position holding the last token of document A is asked to name the first token
of document B, and there is no relationship between them at all.  Both ends of
the pair are real text, so nothing about the *tokens* is wrong.  Only the pair
is.

Three things are measured here.

1. The strings either side of the join, so the pair can be read.
2. The loss before and after masking it, on a trained model -- and the
   per-pair loss at the join against the per-pair loss everywhere else, which
   is the actual explanation of the difference.
3. What happens to the difference as packing gets denser.  At two documents per
   row the join is 1 pair in 128 and the aggregate barely moves; the same bug at
   sixteen documents per row is a different size of number.  Reporting only the
   first would make this look harmless.
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data import EOS_ID, Vocab, load_documents, packed_batch
from src.losses import align, lm_loss, loss_mask
from src.model import Config, TinyGPT
from src.plots import CRITICAL, S1, S3, finish
from src.report import machine, save_json, save_text, table
from src.train import pick_device, train

STEPS = 400
B, T = 8, 128
DENSITIES = (2, 4, 8, 16)


def show(piece: str) -> str:
    out = (piece.replace("\\", "\\\\").replace("\n", "\\n")
           .replace("|", "\\|"))
    return f"`{out}`" if out else "`` "


@torch.no_grad()
def split_by_boundary(model, seqs, device):
    """Mean loss at the join, and mean loss everywhere else, same model."""
    seqs = seqs.to(device)
    a = align(seqs.tokens.shape[1], 1, device)
    logits = model(seqs)
    rec = lm_loss(logits, seqs, mask_padding=True, mask_boundaries=False)
    real = loss_mask(seqs, a, mask_padding=True, mask_boundaries=False)
    same_doc = loss_mask(seqs, a, mask_padding=True, mask_boundaries=True)
    crossing = real & ~same_doc
    per = rec.per_token
    return {
        "n_crossing": int(crossing.sum()),
        "n_interior": int(same_doc.sum()),
        "loss_crossing": float((per * crossing).sum() / max(int(crossing.sum()), 1)),
        "loss_interior": float((per * same_doc).sum() / max(int(same_doc.sum()), 1)),
        "loss_unmasked": float((per * real).sum() / max(int(real.sum()), 1)),
        "loss_masked": float((per * same_doc).sum() / max(int(same_doc.sum()), 1)),
    }


def main(verbose: bool = True) -> dict:
    device = pick_device()
    docs, _ = load_documents()
    vocab = Vocab()

    # ---- read the join ----------------------------------------------------
    sample = packed_batch(docs["validation"], 1, 64, n_docs=2, seed=21)
    ids = sample.tokens[0]
    doc = sample.doc_id[0]
    join = int((doc[1:] != doc[:-1]).nonzero()[0]) + 1   # first position of doc B
    rows = []
    for i in range(join - 4, min(join + 4, len(ids) - 1)):
        crossing = int(doc[i]) != int(doc[i + 1])
        rows.append((str(i), show(vocab.piece(int(ids[i]))),
                     show(vocab.piece(int(ids[i + 1]))),
                     f"doc {int(doc[i])}", f"doc {int(doc[i+1])}",
                     "**masked out**" if crossing else "kept"))
    join_table = table(rows, ("pos i", "input token", "target token",
                              "input's doc", "target's doc", "in the loss?"),
                       ("r", "l", "l", "l", "l", "l"))
    before = vocab.text(ids[max(0, join - 12):join].tolist())
    after = vocab.text(ids[join:join + 12].tolist())

    # ---- an untrained model, for the count --------------------------------
    torch.manual_seed(0)
    fresh = TinyGPT(Config())
    packed = packed_batch(docs["train"], B, T, n_docs=2, seed=31)
    lg = fresh(packed)
    untrained = {
        "unmasked": lm_loss(lg, packed, mask_boundaries=False),
        "masked": lm_loss(lg, packed, mask_boundaries=True),
    }
    untrained_counts = {
        "n_unmasked": untrained["unmasked"].n_contributing,
        "n_masked": untrained["masked"].n_contributing,
        "loss_unmasked": float(untrained["unmasked"].loss),
        "loss_masked": float(untrained["masked"].loss),
    }

    # ---- a trained model ---------------------------------------------------
    torch.manual_seed(0)
    model = TinyGPT(Config())
    hist = train(model, docs["train"], steps=STEPS, kind="packed",
                 batch_size=B, width=T, mask_padding=True,
                 mask_boundaries=True, device=device, seed=0)

    density = {}
    for n_docs in DENSITIES:
        batch = packed_batch(docs["validation"], B, T, n_docs=n_docs, seed=555)
        d = split_by_boundary(model, batch, device)
        d["n_docs"] = n_docs
        d["boundary_fraction"] = d["n_crossing"] / (d["n_crossing"] + d["n_interior"])
        d["delta"] = d["loss_unmasked"] - d["loss_masked"]
        density[n_docs] = d
        if verbose:
            print(f"n_docs={n_docs:>2}  unmasked {d['loss_unmasked']:.4f}  "
                  f"masked {d['loss_masked']:.4f}  "
                  f"crossing {d['loss_crossing']:.4f}")

    two, sixteen = density[2], density[16]

    # ---- figure ------------------------------------------------------------
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1))
    xs = [density[n]["boundary_fraction"] * 100 for n in DENSITIES]
    ax1.plot(xs, [density[n]["loss_unmasked"] for n in DENSITIES],
             color=CRITICAL, marker="o", label="join counted in the loss")
    ax1.plot(xs, [density[n]["loss_masked"] for n in DENSITIES],
             color=S1, marker="o", label="join masked out")
    ax1.set_title("The same batch, scored two ways")
    ax1.set_xlabel("share of pairs that cross a document boundary (%)")
    ax1.set_ylabel("loss (nats)")
    ax1.legend(loc="upper left")
    for n in DENSITIES:
        d = density[n]
        ax1.annotate(f"{n} docs/row", (d["boundary_fraction"]*100,
                                       d["loss_unmasked"]),
                     xytext=(0, 7), textcoords="offset points", fontsize=8,
                     ha="center")

    labels = ["pairs inside\na document", "pairs crossing\na join"]
    vals = [two["loss_interior"], two["loss_crossing"]]
    ax2.bar(labels, vals, color=[S1, CRITICAL], width=0.55)
    for i, v in enumerate(vals):
        ax2.annotate(f"{v:.2f}", (i, v), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=9)
    ax2.set_title("Why: the join is the hardest pair in the batch")
    ax2.set_ylabel("mean loss (nats)")
    ax2.set_ylim(0, max(vals) * 1.2)
    finish(fig, "e4_packing.png",
           f"one model trained {STEPS} steps on packed sequences with joins "
           f"masked; evaluated on held-out packed batches")

    payload = {"machine": machine(), "device": str(device), "steps": STEPS,
               "untrained": untrained_counts, "density": density,
               "join_index": join, "final_train_loss": hist.train_loss[-1]}

    md = f"""# 4 · Two documents in one sequence, and the join between them

## The join, read as text

Document A ends and document B begins inside one row. Decoded either side of
position {join}:

> …{before.strip()[-90:]!r} **`<eos>`** … {after.strip()[:90]!r}…

{join_table}

Exactly one pair crosses the join. Both of its tokens are real text — nothing
here is padding, nothing is malformed — and the pair is still a lie: the last
token of one article does not predict the first token of the next.

## The loss before and after masking it

An untrained model first, so the count is visible on its own:
**{untrained_counts['n_unmasked']} → {untrained_counts['n_masked']}**
contributing pairs, loss {untrained_counts['loss_unmasked']:.4f} →
{untrained_counts['loss_masked']:.4f}. At initialisation the two numbers are
almost identical, because an untrained model finds the join no harder than
anything else. **This is the trap: at step 0 the bug is invisible.**

After {STEPS} steps on packed data:

{table([
    (f"{density[n]['n_docs']}",
     f"{100*density[n]['boundary_fraction']:.2f}%",
     f"{density[n]['loss_unmasked']:.4f}",
     f"{density[n]['loss_masked']:.4f}",
     f"{density[n]['delta']:+.4f}",
     f"{density[n]['loss_crossing']:.4f}")
    for n in DENSITIES],
    ("docs per row", "pairs at a join", "loss, join counted",
     "loss, join masked", "difference", "mean loss *at* the join"),
    ("r", "r", "r", "r", "r", "r"))}

## Explaining the difference

**The difference is small and the cause is large.** At two documents per row the
join is {100*two['boundary_fraction']:.2f}% of the pairs and moves the reported
loss by only {two['delta']:+.4f} nats. Read that as reassuring and you have
misread it. The mean loss *at* the join is **{two['loss_crossing']:.4f}** nats
against **{two['loss_interior']:.4f}** inside a document — the crossing pairs are
{two['loss_crossing']/two['loss_interior']:.1f}× harder, and they are the hardest
pairs in the batch by a wide margin. They are diluted, not benign.

**Dilution is a property of the packing, not of the bug.** Pack sixteen
documents per row instead of two — which is what a 4K or 8K context does to a
corpus of short documents — and the same mistake moves the loss by
{sixteen['delta']:+.4f} nats, {abs(sixteen['delta']/two['delta']):.0f}× more,
because {100*sixteen['boundary_fraction']:.2f}% of pairs now cross a join.
Nothing about the error changed. Only how often it is committed.

(Both columns rise with density for a reason that is not the bug: more
documents in a fixed {T} tokens means shorter fragments, so every position has
less context to work with. That is why the honest comparison is the difference
between the two columns and not the level of either.)

**And the loss is the least of it.** Every crossing pair is a gradient telling
the model that unrelated text follows `<eos>`. The model cannot learn that,
because it is not true, so it learns the next best thing: that after `<eos>`
anything can happen. That is capacity spent on making the model *less* certain,
and it is spent at exactly the position where a document boundary should be the
most informative token in the sequence.

The fix costs one comparison — `doc_id[i] == doc_id[i+1]` — and drops
{two['n_crossing']} pairs out of
{two['n_crossing'] + two['n_interior']}. If the attention mask is also
document-aware (session 6), the two masks must agree: a position that cannot
*see* the previous document must not be scored as though it could.
"""

    save_json("e4_packing.json", payload)
    save_text("e4_packing.md", md)
    if verbose:
        print(md)
    return payload


if __name__ == "__main__":
    main()
