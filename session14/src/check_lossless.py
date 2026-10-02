"""Before any MoE training: does the upcycled model compute the same function as the dense one?"""
import json, torch
from .data import Rows
from .train import device, load_dense, CKPT, ROOT
from .model import upcycle

d = device()
dense = load_dense(CKPT / "dense.pt", d)
val = Rows("val", dense.c.seq_len)
x, y = val.batch(0, 16, d)
out = {}
with torch.no_grad():
    ld = dense(x)
    base = dense(x, y)[0].item()
    for name, drop in (("copy (r=0)", 0.0), ("drop-upcycle (r=0.5)", 0.5)):
        m = upcycle(dense, 8, 2, drop=drop)
        lm = m(x)
        out[name] = {"dense_loss": base, "moe_loss": m(x, y)[0].item(),
                     "max_abs_logit_diff": (lm - ld).abs().max().item()}
print(json.dumps(out, indent=2))
(ROOT / "results" / "lossless_check.json").write_text(json.dumps(out, indent=2))
