"""E3 — Does a language model with no output head actually train?

E1 showed the byte code names every token uniquely. E2 showed the code comes
back out of a d-dimensional embedding by a linear map. Neither of those is a
language model, and the claim the assignment asks to prove is about a language
model: that the D x V output head can be deleted and replaced by scoring the
fixed byte codes, with no loss of quality.

So this trains real transformers. The trunk, the data, the schedule, the seed
and the step count are identical across arms. The only difference is what sits
at the front door and the back door:

  dense        nn.Embedding + nn.Linear head, untied      2 * V * D params
  dense_tied   nn.Embedding, head = table^T                   V * D params
  kron_dense   byte-code input, dense head                 N * D + V * D
  duplex       byte-code input, byte-code output           2 * N * D, no V
  duplex_tied  byte-code input, W_syn = W_ana^T                N * D, no V
  duplex_colce as duplex, but trained on the code's columns  2 * N * D, no V

The last arm is the strongest form of the claim. Its training loss is a sum of
pos_dim independent char_dim-way classifications, so no tensor of size V is
ever formed, in the forward pass or in the loss. It is still evaluated with the
ordinary vocabulary cross-entropy, so its number is comparable to every other
arm's.

Data is the Session 5 proxy corpus, tokenized with the Session 2 BrahmicTokenizer
work. Validation is reported per lane as well as overall, because the failure
this whole line of work is trying to avoid is one that shows up on Indic text
and averages away on English.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from duplex.codec import DUPLEX
from duplex.model import ARMS, COLUMN_OBJECTIVE, HEAD_FREE, ModelConfig, TinyGPT
from duplex.vocab import load_era_v5_vocab

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PROXY = ROOT.parent / "session5_Data_Mixtures" / "data" / "proxy"

# The Session 5 proxy lanes this experiment trains on, and their sampling
# weight. Indic is the majority lane on purpose: it is the case that breaks.
LANES = {"indic": 0.45, "web": 0.30, "code": 0.15, "math": 0.10}


class ProxyData:
    """The Session 5 uint16 token bins, sampled lane by lane."""

    def __init__(self, block_size: int, device: str):
        self.block = block_size
        self.device = device
        self.train = {l: np.fromfile(PROXY / f"{l}.bin", dtype=np.uint16) for l in LANES}
        self.val = {
            l: np.fromfile(PROXY / f"{l}.holdout.bin", dtype=np.uint16) for l in LANES
        }
        self.names = list(LANES)
        self.weights = np.array([LANES[l] for l in self.names], dtype=np.float64)
        self.weights /= self.weights.sum()

    def _draw(self, pool: np.ndarray, batch: int, rng: np.random.Generator):
        starts = rng.integers(0, len(pool) - self.block - 1, size=batch)
        x = np.stack([pool[s : s + self.block] for s in starts]).astype(np.int64)
        y = np.stack([pool[s + 1 : s + 1 + self.block] for s in starts]).astype(np.int64)
        return (
            torch.from_numpy(x).to(self.device),
            torch.from_numpy(y).to(self.device),
        )

    def train_batch(self, batch: int, rng: np.random.Generator):
        """One batch drawn from the mixture, lane by lane in proportion."""
        counts = rng.multinomial(batch, self.weights)
        xs, ys = [], []
        for lane, k in zip(self.names, counts):
            if k:
                x, y = self._draw(self.train[lane], int(k), rng)
                xs.append(x)
                ys.append(y)
        return torch.cat(xs), torch.cat(ys)

    def val_batches(self, lane: str, batch: int, n: int, seed: int = 7):
        rng = np.random.default_rng(seed)
        return [self._draw(self.val[lane], batch, rng) for _ in range(n)]


@torch.no_grad()
def evaluate(model: TinyGPT, data: ProxyData, batch: int, n: int) -> dict:
    """Vocabulary cross-entropy per lane, for every arm on the same footing."""
    model.eval()
    out = {}
    total, weight = 0.0, 0.0
    for lane in data.names:
        losses = [
            model.vocab_loss(x, y).item() for x, y in data.val_batches(lane, batch, n)
        ]
        out[lane] = float(np.mean(losses))
        total += out[lane] * LANES[lane]
        weight += LANES[lane]
    out["mixture"] = total / weight
    out["mixture_ppl"] = float(np.exp(out["mixture"]))
    model.train()
    return out


def run_arm(arm: str, vocab, cfg: ModelConfig, data: ProxyData, args) -> dict:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model = TinyGPT(arm, vocab.byte_seqs, cfg, DUPLEX).to(args.device)
    report = model.parameter_report()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1
    )
    column = arm in COLUMN_OBJECTIVE

    print(
        f"\n--- {arm} --- token-facing {report['total_token_facing']:,} "
        f"({report['token_facing_share']*100:.1f}% of {report['total']:,}), "
        f"objective={'column' if column else 'vocab'}"
    )

    curve, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        x, y = data.train_batch(args.batch, rng)
        loss = model.column_loss(x, y) if column else model.vocab_loss(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, data, args.eval_batch, args.eval_n)
            curve.append({"step": step, "train_loss": loss.item(), **ev})
            print(
                f"  step {step:>5d}  train={loss.item():.4f}  "
                f"val={ev['mixture']:.4f}  ppl={ev['mixture_ppl']:.1f}  "
                f"indic={ev['indic']:.4f}"
            )

    final = evaluate(model, data, args.eval_batch, args.eval_n * 2)
    return {
        "arm": arm,
        "head_free": arm in HEAD_FREE,
        "objective": "column" if column else "vocab",
        "params": report,
        "curve": curve,
        "final": final,
        "seconds": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default="e3_headfree_lm.json")
    args = ap.parse_args()

    vocab = load_era_v5_vocab()
    cfg = ModelConfig(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        block_size=args.block,
    )
    data = ProxyData(args.block, args.device)

    print(
        f"vocab {vocab.name} V={len(vocab)}  code_dim={DUPLEX.code_dim}  "
        f"d_model={cfg.d_model}  device={args.device}  "
        f"tokens/step={args.batch * args.block:,}"
    )

    runs = [run_arm(arm, vocab, cfg, data, args) for arm in args.arms]

    print(f"\n{'arm':<14}{'token-facing':>14}{'V-free':>8}{'val CE':>10}{'ppl':>9}{'indic CE':>10}")
    for r in runs:
        print(
            f"{r['arm']:<14}{r['params']['total_token_facing']:>14,}"
            f"{'yes' if not r['params']['depends_on_vocab'] else 'no':>8}"
            f"{r['final']['mixture']:>10.4f}{r['final']['mixture_ppl']:>9.1f}"
            f"{r['final']['indic']:>10.4f}"
        )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / args.out).write_text(
        json.dumps(
            {
                "vocab": vocab.name,
                "vocab_size": len(vocab),
                "codec": DUPLEX.label,
                "code_dim": DUPLEX.code_dim,
                "model": vars(cfg),
                "lanes": LANES,
                "args": {k: v for k, v in vars(args).items() if k != "arms"},
                "runs": runs,
            },
            indent=2,
        )
    )
    print(f"\nwrote {RESULTS / args.out}")


if __name__ == "__main__":
    main()
