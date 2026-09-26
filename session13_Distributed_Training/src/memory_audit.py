"""Peak memory of one training step, measured so that it repeats.

For each configuration: build the model, take two ordinary steps (so AdamW's state exists), then
meter two more steps op by op with the GPU synchronised after every op. Both metered steps are
reported, so any disagreement between them is visible. Each configuration runs in a fresh process.

    python -m src.memory_audit
"""
import json
import subprocess
import sys

import torch

from . import memory
from .data import Rows
from .experiments import RESULTS, ROOT, args_for
from .train import build, pick_device, train_step

CONFIGS = [  # (label, variant, h, batch, loss_chunk)
    ("standard b32", "standard", 1.0, 32, 0),
    ("momentum b32", "momentum", 0.5, 32, 0),
    ("momentum b88", "momentum", 0.5, 88, 0),
    ("standard b48, chunked head", "standard", 1.0, 48, 8),
    ("momentum b88, chunked head", "momentum", 0.5, 88, 8),
    ("momentum b256, chunked head", "momentum", 0.5, 256, 8),
]


def one(variant, h, batch, loss_chunk):
    dev = pick_device()
    args = args_for(variant=variant, h=h, batch=batch, loss_chunk=loss_chunk)
    _, model = build(args, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    train = Rows("train", args.seq_len)
    peaks = []
    for step in range(4):
        x, y = train.batch(step * batch, batch, dev)
        if step < 2:
            train_step(model, opt, x, y, dev, False, 1.0)
            continue
        if dev == "mps":
            torch.mps.synchronize()
        with memory.OpLevelPeak(dev, sync=True) as m:
            train_step(model, opt, x, y, dev, False, 1.0)
        peaks.append({"peak_mib": round(m.peak / 2**20, 1), "op": m.where})
    return peaks


if __name__ == "__main__":
    if len(sys.argv) > 1:
        v, h, b, c = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        print(json.dumps(one(v, h, b, c)))
        sys.exit(0)
    out = []
    for label, v, h, b, c in CONFIGS:
        p = subprocess.run([sys.executable, "-m", "src.memory_audit", v, str(h), str(b), str(c)],
                           cwd=ROOT, capture_output=True, text=True)
        peaks = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else None
        row = {"config": label, "variant": v, "batch": b, "loss_chunk": c, "steps": peaks,
               "error": None if peaks else p.stderr[-300:]}
        out.append(row)
        print(label, peaks, flush=True)
    (RESULTS / "memory_audit.json").write_text(json.dumps(out, indent=2))
