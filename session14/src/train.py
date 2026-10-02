"""Train a dense GPT, or continue from a checkpoint as a dense or upcycled-MoE model.

    python -m src.train dense   --name dense   --tokens 30_000_000
    python -m src.train dense   --name cont_dense --init results/ckpt/dense.pt --offset-tokens 30_000_000 ...
    python -m src.train moe     --name moe_copy --init results/ckpt/dense.pt --n-exp 8 --drop 0 ...

Every run logs to results/runs/<name>.jsonl (one line per logged step / eval, with per-layer expert load)
and prints the same lines to stdout.  Runs that continue from the dense checkpoint read the rows the dense
run never saw (--offset-tokens), so every continuation sees the same new text in the same order.
"""
import argparse, json, math, time
from dataclasses import asdict
from pathlib import Path

import torch

from .data import Rows, meta
from .model import GPT, Cfg, upcycle

ROOT = Path(__file__).resolve().parents[1]
RUNS, CKPT = ROOT / "results" / "runs", ROOT / "results" / "ckpt"


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def sync(d):
    if d == "mps":
        torch.mps.synchronize()


def lr_at(step, total, peak, warmup, floor=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def evaluate(model, val, d, batch=16, n=40):
    model.eval()
    ls = [model(*val.batch(i * batch, batch, d))[0].item() for i in range(n)]
    model.train()
    return sum(ls) / len(ls)


def load_dense(path, d):
    ck = torch.load(path, map_location=d)
    m = GPT(Cfg(**ck["cfg"])).to(d)
    m.load_state_dict(ck["model"])
    return m


def expert_load(model):
    """per layer: fraction of routed assignments each expert got in the last forward."""
    out = []
    for b in model.blocks:
        s = getattr(b.mlp, "stats", None)
        if s is not None:
            out.append((s.float() / s.sum()).tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["dense", "moe"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--n-exp", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--aux", type=float, default=0.01)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--offset-tokens", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()

    d = device()
    torch.manual_seed(a.seed)
    train, val = Rows("train", a.seq_len), Rows("val", a.seq_len)
    if a.init:
        dense = load_dense(a.init, d)
        model = upcycle(dense, a.n_exp, a.top_k, a.aux, a.drop, seed=a.seed) if a.kind == "moe" else dense
    else:
        model = GPT(Cfg(vocab_size=meta()["vocab_size"], seq_len=a.seq_len, n_layer=a.n_layer,
                        n_head=a.n_head, d_model=a.d_model)).to(d)
    tps = a.batch * a.seq_len
    steps = a.tokens // tps
    first = a.offset_tokens // tps
    assert (first + steps) * a.batch <= len(train), "not enough rows"
    decay = [p for p in model.parameters() if p.ndim >= 2]
    nodecay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    RUNS.mkdir(parents=True, exist_ok=True)
    log = open(RUNS / f"{a.name}.jsonl", "w")

    def emit(rec, echo=True):
        log.write(json.dumps(rec) + "\n"); log.flush()
        if echo:
            print(json.dumps(rec), flush=True)

    emit({"type": "header", "name": a.name, "args": vars(a), "cfg": asdict(model.c), "params": model.n_params(),
          "active_params": model.n_active(), "steps": steps, "tokens_per_step": tps, "device": d,
          "torch": torch.__version__})
    v0 = evaluate(model, val, d)
    emit({"type": "eval", "step": 0, "tokens": 0, "val_loss": v0})
    t_train, t_last, tok_last = 0.0, 0.0, 0
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, a.lr, a.warmup)
        x, y = train.batch((first + step) * a.batch, a.batch, d)
        t0 = time.perf_counter()
        ce, aux = model(x, y)
        (ce + model.c.aux_coef * aux).backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        log_now = step % a.log_every == 0 or step == steps - 1
        if log_now:
            ce_v, gn_v = ce.item(), gn.item()
        sync(d)
        t_train += time.perf_counter() - t0
        done = (step + 1) * tps
        if log_now:
            rec = {"type": "step", "step": step + 1, "tokens": done, "loss": ce_v, "aux": float(aux),
                   "grad_norm": gn_v, "lr": opt.param_groups[0]["lr"], "train_time": round(t_train, 1),
                   "tok_per_s": round((done - tok_last) / (t_train - t_last)) if t_train > t_last else None}
            t_last, tok_last = t_train, done
            ld = expert_load(model)
            if ld:
                rec["max_load"] = [round(max(l), 3) for l in ld]     # even share is 1/n_exp
                rec["dead"] = [sum(1 for v in l if v < 0.01) for l in ld]
            emit(rec, echo=(step % (a.log_every * 5) == 0 or step == steps - 1))
        if (step + 1) % a.eval_every == 0 or step == steps - 1:
            emit({"type": "eval", "step": step + 1, "tokens": done, "val_loss": evaluate(model, val, d)})
    final = evaluate(model, val, d)
    tail = [json.loads(l)["loss"] for l in open(RUNS / f"{a.name}.jsonl") if '"type": "step"' in l][-10:]
    summ = {"type": "summary", "name": a.name, "val_loss_start": v0, "val_loss_final": final,
            "train_loss_last10": sum(tail) / len(tail), "params": model.n_params(),
            "active_params": model.n_active(), "train_seconds": round(t_train, 1),
            "tok_per_s": round(steps * tps / t_train)}
    emit(summ)
    (RUNS / f"{a.name}.json").write_text(json.dumps(summ, indent=2))
    if a.save:
        CKPT.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": asdict(model.c), "model": model.state_dict()}, CKPT / a.save)


if __name__ == "__main__":
    main()
