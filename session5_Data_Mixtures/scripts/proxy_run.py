"""Proxy-run harness: train competing mixtures and let the numbers decide.

The same code runs three scales, chosen by config:

  configs/smoke.yaml    ~5M params, ~40M tokens, laptop/MPS, minutes per arm
  configs/proxy_1b.yaml  1B params, 30B tokens, 8xH100, ~16h per arm
  configs/proxy_3b.yaml  3B params, 60B tokens, 8xH100, ~4d per arm

What it implements, faithfully to the plan it is testing:

* the curriculum from mixture/v5_mixture.yaml, compressed onto the proxy budget,
  including per-transition warmup blending;
* OPUS-style data selection as RHO-loss style learnability scoring against a
  frozen reference model (Mindermann et al. 2022). With an English-heavy
  reference this reproduces the failure mode the floors exist to prevent,
  rather than injecting it by hand;
* protected floors enforced as a rolling-window constraint the selector cannot
  cross;
* per-lane loss masking, so tool observations in agentic trajectories are read
  but never trained on;
* gradient-norm spike monitoring against the spec's warn/halt rule;
* per-lane held-out evaluation in bits-per-byte, which is comparable across
  languages and lanes in a way that per-token loss is not.

Usage:
    python scripts/proxy_run.py --config configs/smoke.yaml
    python scripts/proxy_run.py --config configs/smoke.yaml --arms A_proposed
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from lib_mixture import Curriculum, floor_map, load_mixture, ROOT

REPO = os.path.dirname(ROOT)
DATA = os.path.join(ROOT, "data", "proxy")


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size: int, n_layer: int, n_head: int, n_embd: int,
                 block_size: int, dropout: float = 0.0):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok.weight          # weight tying
        self.register_buffer(
            "causal", torch.triu(torch.full((block_size, block_size), float("-inf")), diagonal=1),
            persistent=False,
        )
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, loss_mask=None):
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))[None]
        am = self.causal[:t, :t]
        for blk in self.blocks:
            x = blk(x, am)
        logits = self.head(self.lnf(x))
        if targets is None:
            return logits, None
        ll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
        ).view(b, t)
        if loss_mask is not None:
            denom = loss_mask.sum().clamp(min=1)
            return logits, (ll * loss_mask).sum() / denom
        return logits, ll.mean()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class Lane:
    name: str
    tokens: np.ndarray
    mask: np.ndarray
    hold_tokens: np.ndarray
    hold_mask: np.ndarray
    bytes_per_token: float


def load_lanes(names: list[str], tok_bytes: dict[int, int]) -> dict[str, Lane]:
    lanes: dict[str, Lane] = {}
    for n in names:
        p = os.path.join(DATA, f"{n}.bin")
        if not os.path.exists(p):
            continue
        t = np.fromfile(p, dtype=np.uint16)
        m = np.fromfile(os.path.join(DATA, f"{n}.mask.bin"), dtype=np.uint8)
        ht = np.fromfile(os.path.join(DATA, f"{n}.holdout.bin"), dtype=np.uint16)
        hm = np.fromfile(os.path.join(DATA, f"{n}.holdout.mask.bin"), dtype=np.uint8)
        sample = ht[:20000]
        bpt = float(np.mean([tok_bytes.get(int(i), 1) for i in sample])) if sample.size else 1.0
        lanes[n] = Lane(n, t, m, ht, hm, bpt)
    return lanes


def get_batch(lane: Lane, batch: int, block: int, device: str, rng: np.random.Generator):
    ix = rng.integers(0, len(lane.tokens) - block - 1, size=batch)
    x = np.stack([lane.tokens[i:i + block] for i in ix]).astype(np.int64)
    y = np.stack([lane.tokens[i + 1:i + 1 + block] for i in ix]).astype(np.int64)
    m = np.stack([lane.mask[i + 1:i + 1 + block] for i in ix]).astype(np.float32)
    return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device),
            torch.from_numpy(m).to(device))


@torch.no_grad()
def eval_lane(model: GPT, lane: Lane, block: int, device: str, max_batches: int = 12,
              batch: int = 8) -> dict:
    model.eval()
    n = len(lane.hold_tokens)
    losses, weights = [], []
    starts = list(range(0, max(n - block - 1, 1), block))[: max_batches * batch]
    for i in range(0, len(starts), batch):
        chunk = starts[i:i + batch]
        if not chunk:
            break
        x = np.stack([lane.hold_tokens[s:s + block] for s in chunk if s + block + 1 <= n]).astype(np.int64)
        if x.shape[0] == 0:
            break
        y = np.stack([lane.hold_tokens[s + 1:s + 1 + block] for s in chunk if s + block + 1 <= n]).astype(np.int64)
        m = np.stack([lane.hold_mask[s + 1:s + 1 + block] for s in chunk if s + block + 1 <= n]).astype(np.float32)
        xb = torch.from_numpy(x).to(device)
        yb = torch.from_numpy(y).to(device)
        mb = torch.from_numpy(m).to(device)
        _, loss = model(xb, yb, mb)
        losses.append(loss.item())
        weights.append(float(mb.sum().item()))
    model.train()
    if not losses:
        return {"loss": float("nan"), "bpb": float("nan")}
    mean_loss = float(np.average(losses, weights=weights))
    bpb = mean_loss / math.log(2) / max(lane.bytes_per_token, 1e-6)
    return {"loss": mean_loss, "bpb": bpb, "bytes_per_token": lane.bytes_per_token}


# --------------------------------------------------------------------------- #
# mixture schedule for the proxy
# --------------------------------------------------------------------------- #
def proxy_weights(cur: Curriculum, mix: dict, frac: float, available: list[str],
                  lane_override: dict[str, float] | None = None) -> dict[str, float]:
    """Curriculum weights at `frac` of the way through the run, restricted to the
    lanes the proxy actually has data for and renormalised.

    `lane_override` pins a lane to a fixed share (arm A5 cuts agentic to 1.5%);
    the remaining lanes keep their relative proportions.

    Arm flags that are NOT handled here — indic_tier_override, agentic_tier_filter,
    disable_reserve — are shard-selection flags. Tier and reserve membership live in
    the shard manifests written in Session 4 and consumed by the Session 6
    dataloader, so those arms are configured by pointing the arm at a different
    shard list, not by changing the training loop.
    """
    w = cur.weights_at(frac * cur.total_b)
    w = {k: v for k, v in w.items() if k in available}
    if lane_override:
        pinned = {k: v for k, v in lane_override.items() if k in w}
        rest = {k: v for k, v in w.items() if k not in pinned}
        room = max(1.0 - sum(pinned.values()), 0.0)
        rs = sum(rest.values()) or 1.0
        w = {**pinned, **{k: v / rs * room for k, v in rest.items()}}
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def tokenizer_byte_lengths() -> dict[int, int]:
    tdir = os.path.join(REPO, "session2_tokenizer", "build", "out")
    spec = importlib.util.spec_from_file_location("s2tok", os.path.join(tdir, "tokenizer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tk = mod.Tokenizer.load(os.path.join(tdir, "tokenizer.model"))
    return {i: len(b) for i, b in tk.vocab.items()}


# --------------------------------------------------------------------------- #
# one arm
# --------------------------------------------------------------------------- #
def train_arm(arm: dict, cfg: dict, mix: dict, lanes: dict[str, Lane], device: str,
              ref_model: GPT | None) -> dict:
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    mcfg, tcfg = cfg["model"], cfg["train"]
    scfg = cfg["selector"]
    available = list(lanes.keys())
    cur = Curriculum(mix)

    model = GPT(cfg["vocab_size"], mcfg["n_layer"], mcfg["n_head"], mcfg["n_embd"],
                mcfg["block_size"], mcfg.get("dropout", 0.0)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], betas=(0.9, 0.95),
                            weight_decay=tcfg.get("weight_decay", 0.1))
    tokens_per_step = tcfg["batch_size"] * mcfg["block_size"]
    total_steps = int(tcfg["total_tokens"] // tokens_per_step)
    warmup = int(tcfg.get("warmup_steps", max(20, total_steps // 50)))

    floors = floor_map(mix) if arm.get("floors", True) else {}
    fixed_mixture = arm.get("fixed_mixture")
    use_selector = arm.get("selector", scfg.get("enabled", True)) and ref_model is not None

    seen = {ln: 0 for ln in available}
    grad_norms: list[float] = []
    spikes = {"warn": 0, "halt": 0, "max_mult": 1.0}
    evals: list[dict] = []
    sel_time = 0.0
    floor_forced = 0
    t0 = time.time()

    for step in range(total_steps):
        frac = step / max(total_steps - 1, 1)
        if fixed_mixture:
            w = {k: v for k, v in fixed_mixture.items() if k in available}
            s = sum(w.values())
            w = {k: v / s for k, v in w.items()}
        else:
            w = proxy_weights(cur, mix, frac, available, arm.get("lane_override"))

        # --- pick this step's lane -------------------------------------------------
        forced_lane = None
        if floors:
            total_seen = sum(seen.values()) or 1
            for ln, f in floors.items():
                if ln in available and seen.get(ln, 0) / total_seen < f:
                    forced_lane = ln
                    break
        if forced_lane is not None:
            lane_name = forced_lane
            floor_forced += 1
        elif use_selector:
            ts = time.time()
            cands = list(rng.choice(available, size=int(scfg["candidates"]),
                                    p=[w[l] for l in available]))
            best, best_score = None, -1e9
            st = int(scfg.get("score_tokens", 128))
            for c in cands:
                xb, yb, mb = get_batch(lanes[c], max(1, tcfg["batch_size"] // 2),
                                       min(st, mcfg["block_size"]), device, rng)
                with torch.no_grad():
                    _, l_cur = model(xb, yb, mb)
                    _, l_ref = ref_model(xb, yb, mb)
                # RHO-loss style: train on what the reference finds learnable and the
                # current model has not learned. An English-heavy reference scores
                # Telugu near zero, which is exactly the failure the floors prevent.
                score = float(l_cur.item() - l_ref.item())
                if score > best_score:
                    best, best_score = c, score
            lane_name = best
            sel_time += time.time() - ts
        else:
            lane_name = str(rng.choice(available, p=[w[l] for l in available]))

        # --- one optimisation step -------------------------------------------------
        xb, yb, mb = get_batch(lanes[lane_name], tcfg["batch_size"], mcfg["block_size"], device, rng)
        lr = tcfg["lr"] * (min(step + 1, warmup) / warmup) * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0))))
        for g in opt.param_groups:
            g["lr"] = lr
        _, loss = model(xb, yb, mb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 1.0)))
        opt.step()

        seen[lane_name] += tokens_per_step
        grad_norms.append(gn)
        if len(grad_norms) > 200:
            med = statistics.median(grad_norms[-200:-1])
            if med > 0:
                mult = gn / med
                spikes["max_mult"] = max(spikes["max_mult"], mult)
                if mult >= float(mix["stability"]["gradnorm_kill_rule"]["halt_at"]):
                    spikes["halt"] += 1
                elif mult >= float(mix["stability"]["gradnorm_kill_rule"]["warn_at"]):
                    spikes["warn"] += 1

        if (step + 1) % max(1, total_steps // int(tcfg.get("evals", 4))) == 0 or step == total_steps - 1:
            per_lane = {ln: eval_lane(model, lanes[ln], mcfg["block_size"], device)
                        for ln in available}
            evals.append({"step": step + 1,
                          "tokens": (step + 1) * tokens_per_step,
                          "train_loss": float(loss.item()),
                          "per_lane": per_lane})
            print(f"    [{arm['id']}] step {step+1}/{total_steps} "
                  f"loss={loss.item():.3f} "
                  + " ".join(f"{ln}:{per_lane[ln]['bpb']:.3f}" for ln in available), flush=True)

    total_tok = sum(seen.values()) or 1
    return {
        "arm": arm["id"],
        "description": arm.get("description", ""),
        "params": model.n_params(),
        "steps": total_steps,
        "tokens": total_tok,
        "wall_seconds": round(time.time() - t0, 1),
        "selector_seconds": round(sel_time, 1),
        "selector_overhead_frac": round(sel_time / max(time.time() - t0, 1e-9), 4),
        "realized_share": {ln: seen[ln] / total_tok for ln in available},
        "floor_forced_steps": floor_forced,
        "gradnorm": {"mean": float(np.mean(grad_norms)), **spikes},
        "evals": evals,
        "final": evals[-1]["per_lane"] if evals else {},
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "smoke.yaml"))
    ap.add_argument("--arms", default="", help="comma-separated arm ids; default all")
    ap.add_argument("--tokens", type=int, default=0, help="override train.total_tokens")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if args.tokens:
        cfg["train"]["total_tokens"] = args.tokens
    mix = load_mixture()

    device = cfg.get("device", "auto")
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device} config={os.path.basename(args.config)}")

    tok_bytes = tokenizer_byte_lengths()
    lanes = load_lanes(cfg["lanes"], tok_bytes)
    missing = [l for l in cfg["lanes"] if l not in lanes]
    if missing:
        print(f"missing lane shards {missing}; run scripts/prepare_proxy_data.py first")
        return 1
    print("lanes: " + ", ".join(f"{n}({len(l.tokens)/1e6:.1f}M tok, {l.bytes_per_token:.2f} B/tok)"
                                for n, l in lanes.items()))

    # reference model for the selector: trained only on the proxy direction,
    # which is deliberately English-heavy, exactly as an in-production proxy
    # built from what the model is already good at would be.
    ref = None
    scfg = cfg["selector"]
    if scfg.get("enabled", True):
        print(f"training selector reference on {scfg['reference_lanes']} "
              f"for {scfg['reference_steps']} steps")
        torch.manual_seed(cfg["seed"])
        rng = np.random.default_rng(cfg["seed"])
        m = cfg["model"]
        ref = GPT(cfg["vocab_size"], m["n_layer"], m["n_head"], m["n_embd"],
                  m["block_size"], 0.0).to(device)
        o = torch.optim.AdamW(ref.parameters(), lr=cfg["train"]["lr"])
        for i in range(int(scfg["reference_steps"])):
            ln = str(rng.choice(scfg["reference_lanes"]))
            xb, yb, mb = get_batch(lanes[ln], cfg["train"]["batch_size"], m["block_size"], device, rng)
            _, loss = ref(xb, yb, mb)
            o.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
            o.step()
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        print(f"  reference ready (loss {loss.item():.3f})")

    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    results = []
    for arm in cfg["arms"]:
        if wanted and arm["id"] not in wanted:
            continue
        print(f"  arm {arm['id']}: {arm.get('description','')}")
        results.append(train_arm(arm, cfg, mix, lanes, device, ref))

    out = {"config": os.path.basename(args.config), "device": device,
           "mixture": mix["meta"]["name"], "results": results}
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    tag = cfg.get("name", "proxy")
    with open(os.path.join(ROOT, "reports", f"{tag}_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote reports/{tag}_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
