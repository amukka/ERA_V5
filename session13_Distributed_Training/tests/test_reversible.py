"""Each reversible backward must give the gradient autograd gives for the same forward rule.

The reference runs rule.step() with grad enabled, so autograd stores every activation. The
reversible path stores only the final state and rebuilds the rest. In float64 on CPU the two
must agree to ~1e-10 for the exact rules (midpoint, momentum, revnet); euler_fp is only as good
as its fixed-point iteration, so it is reported rather than asserted.

    python -m tests.test_reversible
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import GPT, GPTConfig  # noqa: E402


def reference_hidden(model, idx):
    x = model.wte(idx) + model.wpe(torch.arange(idx.shape[1]))
    s = model.rule.init(x)
    for l, blk in enumerate(model.blocks):
        s = model.rule.step(l, blk, s)
    return model.rule.output(s)


def grads(model, idx, tgt, ref):
    model.zero_grad()
    h = reference_hidden(model, idx) if ref else model.hidden(idx)
    loss = torch.nn.functional.cross_entropy((model.ln_f(h) @ model.wte.weight.T).view(-1, model.cfg.vocab_size), tgt.view(-1))
    loss.backward()
    return loss.item(), torch.cat([p.grad.flatten() for p in model.parameters()])


def check(variant, h, fp_iters=6, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=97, seq_len=16, n_layer=6, n_head=2, d_model=32, variant=variant, h=h, fp_iters=fp_iters)
    model = GPT(cfg).to(dtype)
    # larger weights than the training init, so the blocks are far from the identity
    with torch.no_grad():
        for p in model.blocks.parameters():
            if p.dim() == 2:
                p.normal_(0, 0.15)
    idx = torch.randint(0, 97, (3, 16))
    tgt = torch.randint(0, 97, (3, 16))
    l_ref, g_ref = grads(model, idx, tgt, ref=True)
    model.trace_states = []
    l_rev, g_rev = grads(model, idx, tgt, ref=False)
    fwd, rebuilt = model.trace_states, model.rebuilt_states
    # rebuilt[l] is the state entering layer l; fwd[l] the state leaving it, so compare fwd[l-1] with rebuilt[l]
    recon = max((a - b).abs().max().item() for l in range(1, len(fwd)) for a, b in zip(fwd[l - 1], rebuilt[l]))
    rel = ((g_rev - g_ref).norm() / g_ref.norm()).item()
    cos = torch.nn.functional.cosine_similarity(g_rev, g_ref, dim=0).item()
    return {"variant": variant, "h": h, "loss_ref": l_ref, "loss_rev": l_rev,
            "grad_rel_err": rel, "grad_cosine": cos, "max_state_rebuild_err": recon}


if __name__ == "__main__":
    rows = [check("midpoint", 0.5), check("momentum", 0.5), check("revnet", 1.0),
            check("euler_fp", 1.0, fp_iters=6), check("euler_fp", 1.0, fp_iters=30), check("euler_fp", 0.5, fp_iters=6)]
    for r in rows:
        print(json.dumps(r))
    for r in rows[:3]:
        assert r["grad_rel_err"] < 1e-9, r
    print("exact rules OK")
