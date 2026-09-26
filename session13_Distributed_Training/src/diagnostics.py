"""Does a reversible backward still give autograd's gradient once the weights are trained?

The float64 test in tests/ checks the algebra. This checks the thing training depends on: in the
run's own precision, on its own device, at its own weights, how far is the gradient the
reversible backward produces from the gradient plain autograd produces for the same forward rule,
and how far are the rebuilt layer inputs from the ones the forward pass computed.
"""
import torch
import torch.nn.functional as F


def _reference_hidden(model, idx):
    x = model.wte(idx) + model.wpe(torch.arange(idx.shape[1], device=idx.device))
    s = model.rule.init(x)
    for l, blk in enumerate(model.blocks):
        s = model.rule.step(l, blk, s)
    return model.rule.output(s)


def _grad(model, x, y, reference):
    model.zero_grad(set_to_none=True)
    h = _reference_hidden(model, x) if reference else model.hidden(x)
    logits = model.ln_f(h) @ model.wte.weight.T
    loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()
    g = torch.cat([p.grad.flatten().cpu().double() for p in model.parameters()])
    model.zero_grad(set_to_none=True)
    return g


def reversibility_check(model, x, y):
    if model.rule is None:
        return None
    was_training = model.training
    model.eval()
    g_ref = _grad(model, x, y, reference=True)
    model.trace_states = []
    g_rev = _grad(model, x, y, reference=False)
    fwd, rebuilt = model.trace_states, model.rebuilt_states
    model.trace_states = model.rebuilt_states = None
    errs = []
    for l in range(1, len(fwd)):
        e = max(((a - b).abs().max() / a.abs().max().clamp_min(1e-12)).item() for a, b in zip(fwd[l - 1], rebuilt[l]))
        errs.append(e)
    model.train(was_training)
    return {
        "grad_rel_err": ((g_rev - g_ref).norm() / g_ref.norm()).item(),
        "grad_cosine": F.cosine_similarity(g_rev, g_ref, dim=0).item(),
        "state_rebuild_rel_err_per_layer": errs,
        "state_rebuild_rel_err_max": max(errs) if errs else 0.0,
    }
