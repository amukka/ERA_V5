"""Deliverable 6 -- tied against untied head parameters, on this configuration.

The arithmetic is one subtraction, so this file does three things around it:
counts the parameters on the configuration actually used here, trains both to
see whether the saving costs anything measurable, and repeats the count at V5's
real width, where the same subtraction is worth half a billion parameters and
is not available.

``n_params`` counts each distinct tensor once. That matters: ``parameters()``
de-duplicates a shared weight, so a tied model is genuinely smaller in memory
and not merely smaller on a diagram.
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data import VOCAB_SIZE, clean_batch, load_documents
from src.losses import lm_loss
from src.model import Config, TinyGPT
from src.plots import CRITICAL, S1, S3, finish, label_end
from src.report import machine, save_json, save_text, table
from src.train import evaluate, pick_device, train

STEPS = 400
B, T = 8, 128
MIB = 1024 ** 2

# Session 7 / 8's target configuration, for the second table.
V5_VOCAB, V5_D = 131_072, 4_096


def main(verbose: bool = True) -> dict:
    device = pick_device()
    docs, _ = load_documents()

    torch.manual_seed(0)
    untied = TinyGPT(Config(tie_embeddings=False))
    torch.manual_seed(0)
    tied = TinyGPT(Config(tie_embeddings=True))

    assert tied.head.weight is tied.tok_emb.weight, "tying did not take"
    assert untied.head.weight is not untied.tok_emb.weight

    cfg = Config()
    head_params = VOCAB_SIZE * cfg.d_model
    counts = {
        "untied_total": untied.n_params(),
        "tied_total": tied.n_params(),
        "saved": untied.n_params() - tied.n_params(),
        "saved_fraction": 1 - tied.n_params() / untied.n_params(),
        "head_params": head_params,
        "untied_non_embedding": untied.n_params(embeddings=False),
        "tied_non_embedding": tied.n_params(embeddings=False),
        "head_share_untied": head_params / untied.n_params(),
    }

    v5 = {
        "vocab": V5_VOCAB, "d_model": V5_D,
        "head_params": V5_VOCAB * V5_D,
        "head_bytes_bf16": V5_VOCAB * V5_D * 2,
        "session7_projection": 33_554_432,
    }
    v5["head_over_session7_input"] = v5["head_params"] / v5["session7_projection"]

    probe = clean_batch(docs["validation"], B, T, seed=999)
    arms = {}
    for label, tie in (("untied", False), ("tied", True)):
        torch.manual_seed(0)
        m = TinyGPT(Config(tie_embeddings=tie))
        hist = train(m, docs["train"], steps=STEPS, batch_size=B, width=T,
                     device=device, seed=0, probe=probe, probe_every=25)
        arms[label] = {
            "tie": tie,
            "params": m.n_params(),
            "final_train": float(sum(hist.train_loss[-20:]) / 20),
            "honest_loss": evaluate(m, probe.to(device), 1),
            "curve": hist.train_loss,
            "probe": hist.probe_loss,
        }
        if verbose:
            print(f"{label:<7} {arms[label]['params']:>10,} params  "
                  f"held-out {arms[label]['honest_loss']:.4f}")

    u, t = arms["untied"], arms["tied"]
    arms_delta = t["honest_loss"] - u["honest_loss"]

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1))
    for label, colour in (("untied", S1), ("tied", S3)):
        pts = arms[label]["probe"]
        ax1.plot([p[0] for p in pts], [p[1] for p in pts], color=colour)
        label_end(ax1, [p[0] for p in pts], [p[1] for p in pts],
                  f"{label} {pts[-1][1]:.3f}", colour)
    ax1.set_title("Held-out loss, tied against untied")
    ax1.set_xlabel("step"); ax1.set_ylabel("loss (nats)")
    ax1.set_xlim(0, STEPS * 1.3)

    parts = ["embeddings\n+ position", "4 blocks", "output head"]
    untied_bars = [untied.tok_emb.weight.numel() + untied.pos_emb.weight.numel(),
                   untied.n_params(embeddings=False),
                   untied.head.weight.numel()]
    tied_bars = [tied.tok_emb.weight.numel() + tied.pos_emb.weight.numel(),
                 tied.n_params(embeddings=False), 0]
    x = range(len(parts))
    ax2.bar([i - 0.2 for i in x], [v/1e6 for v in untied_bars], width=0.4,
            color=S1, label=f"untied - {counts['untied_total']:,}")
    ax2.bar([i + 0.2 for i in x], [v/1e6 for v in tied_bars], width=0.4,
            color=S3, label=f"tied - {counts['tied_total']:,}")
    ax2.set_xticks(list(x), parts, fontsize=8.5)
    ax2.set_ylabel("parameters (millions)")
    ax2.set_title("Where the parameters are")
    ax2.legend(loc="upper right")
    ax2.annotate("the head is the\nsame tensor as\nthe embedding",
                 (2 + 0.2, 0), xytext=(0, 14), textcoords="offset points",
                 ha="center", fontsize=7.6, color=CRITICAL)
    finish(fig, "e6_tying.png",
           f"{STEPS} steps per arm, identical seed and batches")

    payload = {"machine": machine(), "device": str(device),
               "config": {k: getattr(cfg, k) for k in
                          ("vocab_size", "n_layer", "d_model", "d_ff")},
               "counts": counts, "v5": v5, "arms": arms, "steps": STEPS}

    if abs(arms_delta) <= 0.02:
        verdict = "inside the noise of a single short run"
        reading = (f"at this scale the {counts['saved']:,} shared parameters "
                   f"cost nothing measurable")
    else:
        verdict = "a real cost, and larger than this run's step-to-step noise"
        reading = (f"the {counts['saved']:,} parameters tying removes were "
                   f"doing something here, and the {100*counts['saved_fraction']:.1f}% "
                   f"saving is not free")
    direction = "behind" if arms_delta > 0 else "ahead"

    md = f"""# 6 · Tied against untied, on this configuration

`V = {VOCAB_SIZE:,}`, `D = {cfg.d_model}`, {cfg.n_layer} blocks.

{table([
    ("input embedding `tok_emb` [V, D]",
     f"{untied.tok_emb.weight.numel():,}", f"{untied.tok_emb.weight.numel():,}"),
    ("position embedding [max_seq, D]",
     f"{untied.pos_emb.weight.numel():,}", f"{untied.pos_emb.weight.numel():,}"),
    (f"{cfg.n_layer} transformer blocks",
     f"{counts['untied_non_embedding']:,}", f"{counts['untied_non_embedding']:,}"),
    ("output head `head` [V, D]", f"{head_params:,}", "0 - the same tensor"),
    ("**total**", f"**{counts['untied_total']:,}**",
     f"**{counts['tied_total']:,}**"),
], ("parameter group", "untied", "tied"), ("l", "r", "r"))}

**Tying saves {counts['saved']:,} parameters, {100*counts['saved_fraction']:.1f}%
of the model** - {counts['saved']*4/MIB:.1f} MiB of fp32 weights, and three times
that once AdamW's two moments are counted. The head alone is
{100*counts['head_share_untied']:.1f}% of the untied model: the single largest
tensor in it, and {100*head_params/counts['untied_non_embedding']:.0f}% of the
size of all {cfg.n_layer} transformer blocks put together
({head_params:,} against {counts['untied_non_embedding']:,}). One matrix, no
attention in it, weighing nearly as much as the entire depth of the model.

## Does the saving cost anything?

{table([
    (lb, f"{arms[lb]['params']:,}", f"{arms[lb]['final_train']:.4f}",
     f"{arms[lb]['honest_loss']:.4f}") for lb in ("untied", "tied")],
    ("arm", "parameters", "final training loss", "held-out loss"),
    ("l", "r", "r", "r"))}

Same seed, same batches, {STEPS} steps. The tied model is
{100*counts['saved_fraction']:.1f}% smaller and lands
{arms_delta:+.4f} nats {direction} on held-out loss - {verdict}.

Read that as measured rather than as a verdict on tying: {reading}. One pair of
400-step runs at `D={cfg.d_model}` is not an ablation, and the published result
that tying "often helps quality" comes from models where the head is a far
larger share of the budget and the data is far larger relative to it. What this
run does establish is the direction of the trade on *this* configuration: the
{100*counts['saved_fraction']:.1f}% is bought, not found.

The coupling is the real cost and it does not show up in a loss curve: the
vector that means *"this token just arrived"* is forced to be the vector that
means *"predict this token"*. Those are different jobs.

## The same subtraction at V5's width

{table([
    ("vocabulary", f"{V5_VOCAB:,}"),
    ("d_model", f"{V5_D:,}"),
    ("output head, V x D",
     f"**{v5['head_params']:,}** ({v5['head_params']/1e6:.1f}M)"),
    ("as bf16 weights", f"{v5['head_bytes_bf16']/1024**3:.2f} GiB"),
    ("session 7's factored input side",
     f"{v5['session7_projection']:,} ({v5['session7_projection']/1e6:.1f}M)"),
    ("head / input side", f"**{v5['head_over_session7_input']:.1f}x**"),
], ("", "V5"), ("l", "r"))}

At this width tying would save {v5['head_params']/1e6:.1f}M parameters - and it
is not available. Session 7 replaced the dense `[V, D]` input table with a byte
codec plus one {v5['session7_projection']/1e6:.1f}M projection, so there is no
input embedding matrix left to tie *to*. You cannot share rows with a thing that
has no rows.

That is the shape of the problem this session leaves open: the front door was
factored and won 93.75%, and the back door is still a dense
{v5['head_params']/1e6:.1f}M matrix, {v5['head_over_session7_input']:.1f}x the
size of the input side that replaced it. The standard escape is closed, so the
escape has to be a factored head - and that is an ablation nobody has run for a
byte-codec input side.
"""

    save_json("e6_tying.json", payload)
    save_text("e6_tying.md", md)
    if verbose:
        print(md)
    return payload


if __name__ == "__main__":
    main()
