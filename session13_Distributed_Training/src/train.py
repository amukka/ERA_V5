"""One training run: a fixed token budget, a fixed batch, logged step by step.

    python -m src.train --variant standard --batch 32 --tokens 50_000_000 --name baseline

Writes results/runs/<name>.jsonl (one line per logged step, one per evaluation) and
results/runs/<name>.json (the summary). Throughput counts training tokens per second of wall
time spent in training steps, with the device synchronised at every timing point and the
evaluation passes excluded.
"""
import argparse
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch

from . import memory
from .diagnostics import reversibility_check
from .data import Rows, meta
from .model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def lr_at(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def evaluate(model, val, batch, n_batches, device, amp):
    model.eval()
    losses = []
    for i in range(n_batches):
        x, y = val.batch(i * batch, batch, device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=bool(amp)):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def build(args, device):
    m = meta()
    cfg = GPTConfig(vocab_size=m["vocab_size"], seq_len=args.seq_len, n_layer=args.n_layer,
                    n_head=args.n_head, d_model=args.d_model, variant=args.variant, h=args.h,
                    fp_iters=args.fp_iters, loss_chunk=args.loss_chunk)
    torch.manual_seed(args.seed)
    model = GPT(cfg).to(device)
    return cfg, model


def train_step(model, opt, x, y, device, amp, clip):
    memory.probe()
    with torch.autocast(device, dtype=torch.bfloat16, enabled=bool(amp)):
        _, loss = model(x, y)
    memory.probe()                        # the forward's activations are all live here
    loss.backward()
    memory.probe()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    memory.probe()
    opt.zero_grad(set_to_none=True)
    return loss, gnorm


def run(args):
    device = args.device or pick_device()
    if device == "mps" and args.mem_fraction:
        torch.mps.set_per_process_memory_fraction(args.mem_fraction)
    cfg, model = build(args, device)
    train, val = Rows("train", args.seq_len), Rows("val", args.seq_len)
    tokens_per_step = args.batch * args.seq_len
    steps = args.tokens // tokens_per_step if not args.steps else args.steps
    assert steps * args.batch <= len(train), "not enough training rows for this budget"
    warmup = max(1, int(args.warmup_frac * steps))
    peak_lr = args.lr
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1,
                            fused=(device == "cuda"))
    RUNS.mkdir(parents=True, exist_ok=True)
    log = open(RUNS / f"{args.name}.jsonl", "w") if args.name else None

    def emit(rec):
        if log:
            log.write(json.dumps(rec) + "\n")
            log.flush()

    header = {"name": args.name, "device": device, "config": asdict(cfg), "params": model.n_params(),
              "params_non_embedding": model.n_params(non_embedding=True), "batch": args.batch,
              "tokens_per_step": tokens_per_step, "steps": steps, "peak_lr": peak_lr, "warmup": warmup,
              "amp_bf16": args.amp, "torch": torch.__version__, "machine": platform.machine(),
              "processor": platform.processor()}
    if not args.quiet:
        print(json.dumps(header))
    emit({"type": "header", **header})

    diag_x, diag_y = train.batch(len(train) - 4, 4, device)       # rows the run never trains on
    diag = {"init": reversibility_check(model, diag_x, diag_y)}
    audit = None
    memory.start(device)
    train_time, tokens_done, t_last, tok_last = 0.0, 0, None, 0
    evals, losses, nan_at = [], [], None
    sync(device)
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, peak_lr, warmup)
        x, y = train.batch(step * args.batch, args.batch, device)
        if step == args.audit_step:
            # one step metered after every op, for an exact peak; its time and tokens are not counted
            sync(device)
            with memory.OpLevelPeak(device) as meter:
                loss, gnorm = train_step(model, opt, x, y, device, args.amp, args.clip)
                sync(device)
            audit = {"step": step, "peak_mib": round(meter.peak / 2**20, 1), "peak_op": meter.where, "ops": meter.ops}
            emit({"type": "audit", **audit})
            if not args.quiet:
                print(f"memory audit at step {step}: exact peak {audit['peak_mib']:,.0f} MiB in {meter.where}", flush=True)
            continue
        t0 = time.perf_counter()
        loss, gnorm = train_step(model, opt, x, y, device, args.amp, args.clip)
        if step % args.log_every == 0 or step == steps - 1:
            lv, gv = loss.item(), gnorm.item()        # .item() synchronises
        else:
            lv = None
        if args.sync_every_step or lv is not None:
            sync(device)
        train_time += time.perf_counter() - t0
        tokens_done += tokens_per_step
        if lv is not None:
            if not math.isfinite(lv) and nan_at is None:
                nan_at = step
            now_tps = (tokens_done - tok_last) / (train_time - t_last) if t_last is not None else None
            t_last, tok_last = train_time, tokens_done
            rec = {"type": "step", "step": step, "tokens": tokens_done, "loss": lv, "grad_norm": gv,
                   "lr": opt.param_groups[0]["lr"], "train_time": round(train_time, 3),
                   "tok_per_s": round(now_tps, 1) if now_tps else None,
                   "mem_now_mib": round(memory.now() / 2**20, 1)}
            losses.append((step, lv))
            emit(rec)
            if not args.quiet and (step % (args.log_every * 10) == 0 or step == steps - 1):
                print(f"step {step:5d}/{steps}  tok {tokens_done/1e6:6.2f}M  loss {lv:.4f}  gnorm {gv:.3f}  "
                      f"{rec['tok_per_s'] or 0:,.0f} tok/s  {train_time/60:.1f} min", flush=True)
            if nan_at is not None:
                break
        if args.eval_every and ((step + 1) % args.eval_every == 0 or step == steps - 1):
            vl = evaluate(model, val, args.eval_batch, args.eval_batches, device, args.amp)
            evals.append({"step": step, "tokens": tokens_done, "val_loss": vl})
            emit({"type": "eval", "step": step, "tokens": tokens_done, "val_loss": vl})
            if not args.quiet:
                print(f"            eval @ {tokens_done/1e6:.2f}M tokens: val loss {vl:.4f}", flush=True)
    mem = memory.stop()
    diag["final"] = reversibility_check(model, diag_x, diag_y)
    if diag["final"] is not None and not args.quiet:
        print(f"reversibility at final weights: grad rel err {diag['final']['grad_rel_err']:.2e}, "
              f"cosine {diag['final']['grad_cosine']:.6f}, state rebuild err {diag['final']['state_rebuild_rel_err_max']:.2e}")
    tail = [l for _, l in losses[-max(1, len(losses) // 20):]]
    summary = {**header, "tokens_trained": tokens_done, "train_seconds": round(train_time, 1),
               "tok_per_s": round(tokens_done / train_time, 1),
               "final_train_loss": losses[-1][1] if losses else None,
               "final_train_loss_avg_last5pct": sum(tail) / len(tail) if tail else None,
               "final_val_loss": evals[-1]["val_loss"] if evals else None,
               "diverged_at_step": nan_at,
               "peak_alloc_mib": round(mem["peak_alloc_bytes"] / 2**20, 1),
               "peak_driver_mib": round(mem["peak_driver_bytes"] / 2**20, 1),
               "peak_exact_mib": audit["peak_mib"] if audit else None,
               "peak_exact_op": audit["peak_op"] if audit else None,
               "reversibility": diag}
    emit({"type": "summary", **summary})
    if log:
        log.close()
        (RUNS / f"{args.name}.json").write_text(json.dumps(summary, indent=2))
    if not args.quiet:
        print(json.dumps({k: summary[k] for k in ("tok_per_s", "final_train_loss", "final_val_loss", "peak_exact_mib",
                                                    "peak_alloc_mib", "peak_driver_mib", "train_seconds")}))
    return summary


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--variant", default="standard")
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--fp-iters", type=int, default=6)
    ap.add_argument("--loss-chunk", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--n-layer", type=int, default=10)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--tokens", type=int, default=50_000_000)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", type=int, default=0)              # bf16 autocast; fp32 is faster on the M4
    ap.add_argument("--audit-step", type=int, default=3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--mem-fraction", type=float, default=0.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--sync-every-step", type=int, default=0)
    ap.add_argument("--quiet", type=int, default=0)
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
