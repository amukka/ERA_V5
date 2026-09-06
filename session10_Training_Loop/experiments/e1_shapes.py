"""Deliverable 1 — print every tensor shape in the step, and say what each
dimension means.

A training step touches four populations of tensors and it is easy to print
only the first:

  1. activations   what flows forward, from token ids to logits
  2. the loss      per-token, then the two scalars the run reports
  3. parameters    and, after backward(), one gradient of identical shape each
  4. optimiser     AdamW's two running numbers per weight -- the reason section
                   13's table says 16 bytes per weight and not 8

Everything below is printed from a real step, not from a diagram.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from src.data import Sampler, VOCAB_SIZE, load_documents
from src.loop import accumulate_step, pick_device, set_seed, token_losses
from src.model import Config, TinyGPT
from src.report import machine, save_json, save_text, table

ACCUM = 4
MICRO_B = 8

AXIS_GLOSSARY = [
    ("B",   "rows in one micro-batch -- what actually fits on the device"),
    ("T",   "positions in the sequence; micro-batch is padded to its own "
            "longest row, so T changes from micro-batch to micro-batch"),
    ("D",   "d_model, the width of the residual stream"),
    ("H",   "attention heads"),
    ("Dh",  "D/H, the slice of the channel one head owns"),
    ("F",   "d_ff, the wide inner width of the MLP"),
    ("V",   "vocabulary size -- one score per token the model could name next"),
    ("T_q", "the position doing the asking, in an attention score matrix"),
    ("T_k", "the position being asked about"),
]


def main(verbose: bool = True) -> dict:
    set_seed(0)
    device = pick_device()
    cfg = Config()
    model = TinyGPT(cfg).to(device)
    docs, _ = load_documents()
    sampler = Sampler(docs["train"], MICRO_B, max_len=cfg.max_seq, seed=1)

    micro = [mb.to(device) for mb in sampler.micro_batches(ACCUM)]

    # ---- 1. activations, from one micro-batch --------------------------------
    trace: list = []
    loss_sum, n_tok = token_losses(model, micro[0], trace)

    act_rows = [
        (t["name"], "×".join(str(d) for d in t["shape"]), t["dtype"],
         f"{t['numel']:,}", t["meaning"])
        for t in trace
    ]

    # ---- 2. the scalars ------------------------------------------------------
    model.zero_grad(set_to_none=True)
    reported, honest, norm, scale, total_tokens = accumulate_step(
        model, micro, "token_weighted", clip=1.0)

    scalar_rows = [
        ("loss_sum (per micro-batch)", "()", "float32",
         "the summed cross entropy of one micro-batch -- a scalar with no "
         "dimensions, which is the whole point: one number for "
         f"{n_tok:,} tokens"),
        ("loss (step, token-weighted)", "()", "float32",
         "loss_sum over all micro-batches divided by the total valid tokens "
         f"in the global batch ({total_tokens:,})"),
        ("grad_norm", "()", "float32",
         "one L2 norm over every gradient in the model concatenated -- the "
         "single number section 12 says to watch"),
    ]

    # ---- 3. parameters and their gradients -----------------------------------
    param_rows, n_params, n_grads = [], 0, 0
    meanings = {
        "tok_emb.weight": "V=one row per token id, D=the vector it maps to",
        "pos_emb.weight": "T_max=one row per absolute position, D=the vector "
                          "added to whatever token sits there",
        "head.weight":    "V=one row per output token, D=the direction in the "
                          "residual stream that votes for it",
        "ln_f.weight":    "D=one gain per channel",
        "ln_f.bias":      "D=one shift per channel",
    }
    for name, p in model.named_parameters():
        meaning = meanings.get(name)
        if meaning is None:
            if name.endswith("qkv.weight"):
                meaning = "3D=query, key and value stacked, D=input width"
            elif name.endswith("attn.proj.weight"):
                meaning = "D=output width back into the residual stream, "\
                          "D=concatenated head width in"
            elif name.endswith("mlp.fc.weight"):
                meaning = "F=d_ff out, D=d_model in"
            elif name.endswith("mlp.out.weight"):
                meaning = "D=d_model out, F=d_ff in"
            elif ".ln" in name and name.endswith("weight"):
                meaning = "D=one gain per channel"
            elif ".ln" in name and name.endswith("bias"):
                meaning = "D=one shift per channel"
            else:
                meaning = "-"
        g = p.grad
        n_params += p.numel()
        n_grads += 0 if g is None else g.numel()
        param_rows.append((
            name,
            "×".join(str(d) for d in p.shape),
            f"{p.numel():,}",
            "×".join(str(d) for d in g.shape) if g is not None else "none",
            "yes" if (g is not None and tuple(g.shape) == tuple(p.shape))
            else "NO",
            meaning,
        ))

    # ---- 4. optimiser state --------------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.step()
    state_rows, state_numel = [], 0
    for name, p in list(model.named_parameters())[:3]:
        st = opt.state[p]
        for key in ("exp_avg", "exp_avg_sq"):
            state_rows.append((
                f"{name} :: {key}",
                "×".join(str(d) for d in st[key].shape),
                f"{st[key].numel():,}",
                "the same shape as the weight -- one running number per weight",
            ))
    for p in model.parameters():
        st = opt.state.get(p, {})
        state_numel += sum(st[k].numel() for k in ("exp_avg", "exp_avg_sq")
                           if k in st)

    # ---- render --------------------------------------------------------------
    parts = []
    parts.append("# Deliverable 1 — every tensor shape in one step\n")
    parts.append(f"Model: {cfg.n_layer} layers, d_model {cfg.d_model}, "
                 f"{cfg.n_head} heads, d_ff {cfg.d_ff}, vocab {VOCAB_SIZE}, "
                 f"{n_params:,} parameters.\n")
    parts.append(f"Step: micro-batch {MICRO_B} rows × {ACCUM} accumulation "
                 f"steps = a global batch of {MICRO_B*ACCUM} sequences and "
                 f"{total_tokens:,} valid tokens.\n")
    parts.append("Micro-batch token counts this step: "
                 + ", ".join(f"{mb.n_tokens:,}" for mb in micro)
                 + " — different, because the sequences are different lengths. "
                   "Deliverable 3 is about what that does.\n")

    parts.append("\n## The axis names\n")
    parts.append(table([(a, m) for a, m in AXIS_GLOSSARY],
                       ["axis", "what one index along it selects"]))

    parts.append("\n## 1. Activations — one micro-batch, forward\n")
    parts.append(f"({len(trace)} tensors, from token ids to logits. "
                 f"This micro-batch is B={micro[0].inputs.shape[0]}, "
                 f"T={micro[0].inputs.shape[1]}.)\n")
    parts.append(table(act_rows,
                       ["tensor", "shape", "dtype", "elements",
                        "what each dimension means"],
                       ["l", "l", "l", "r", "l"]))

    parts.append("\n## 2. The scalars\n")
    parts.append(table(scalar_rows, ["tensor", "shape", "dtype", "meaning"]))

    parts.append("\n## 3. Parameters, and the gradient of each\n")
    parts.append(table(param_rows,
                       ["parameter", "shape", "elements", "grad shape",
                        "same?", "what each dimension means"],
                       ["l", "l", "r", "l", "l", "l"]))
    parts.append(f"\nEvery gradient has the shape of its weight: "
                 f"{n_grads:,} gradient numbers for {n_params:,} weights. "
                 "That is the definition from section 2 — a gradient belongs to "
                 "one weight.\n")

    parts.append("\n## 4. What the optimiser keeps, per weight\n")
    parts.append(table(state_rows, ["state tensor", "shape", "elements",
                                    "meaning"], ["l", "l", "r", "l"]))
    bytes_per_weight = (2 + 2 + 4 + 8)
    parts.append(
        f"\nAdamW holds {state_numel:,} extra numbers for {n_params:,} weights "
        f"— exactly two per weight. With a bf16 weight (2 B), a bf16 gradient "
        f"(2 B), an fp32 master copy (4 B) and these two fp32 moments (8 B), "
        f"that is the {bytes_per_weight} bytes per weight of section 13: "
        f"{n_params * bytes_per_weight / 2**20:.1f} MiB of training state for "
        f"this model, before a single activation is stored.\n")

    text = "\n".join(parts)
    save_text("e1_shapes.md", text)
    payload = {
        "machine": machine(),
        "config": vars(cfg),
        "n_params": n_params,
        "n_params_non_embedding": model.n_params(False),
        "accum": ACCUM,
        "micro_batch_rows": MICRO_B,
        "micro_batch_tokens": [mb.n_tokens for mb in micro],
        "global_batch_tokens": total_tokens,
        "activations": trace,
        "grad_shapes_match": all(r[4] == "yes" for r in param_rows),
        "optimizer_state_numel": state_numel,
        "bytes_per_weight": bytes_per_weight,
        "training_state_mib": n_params * bytes_per_weight / 2**20,
    }
    save_json("e1_shapes.json", payload)
    if verbose:
        print(text)
    return payload


if __name__ == "__main__":
    main()
