"""The functions the notebooks call, so every notebook and the command line share one implementation."""
import json
import math
import subprocess
import sys
from pathlib import Path

from .train import RUNS, parser, run

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

BASE_BATCH = 32          # the fixed batch: 32 x 512 = 16,384 tokens a step, fits the standard model
TOKENS = 50_000_000

SCREEN = [                # (variant, h) pairs screened for 4M tokens each at the fixed batch
    ("standard", 1.0),
    ("midpoint", 0.5),
    ("midpoint", 0.25),
    ("momentum", 0.5),
    ("revnet", 1.0),
    ("euler_fp", 1.0),
]


def args_for(**kw):
    a = parser().parse_args([])
    for k, v in kw.items():
        setattr(a, k.replace("-", "_"), v)
    return a


def train(**kw):
    return run(args_for(**kw))


def load(name):
    return json.loads((RUNS / f"{name}.json").read_text())


def load_log(name):
    return [json.loads(l) for l in (RUNS / f"{name}.jsonl").read_text().splitlines()]


def screen_name(variant, h):
    return f"screen_{variant}_h{h:g}"


def pick_variant(rows):
    """Lowest validation loss among reversible rules whose gradient still matches autograd."""
    ok = [r for r in rows if r["variant"] != "standard" and r["final_val_loss"] is not None
          and math.isfinite(r["final_val_loss"]) and r["grad_cosine_final"] > 0.9999]
    return min(ok, key=lambda r: r["final_val_loss"])


def probe(variant, h, batch, steps=4, extra=()):
    """Can this batch train? Runs in a fresh process with the MPS cap at the recommended working set,
    so an out-of-memory is a clean failure instead of the machine starting to swap."""
    name = f"probe_{variant}_b{batch}"
    cmd = [sys.executable, "-m", "src.train", "--variant", variant, "--h", str(h), "--batch", str(batch),
           "--steps", str(steps), "--audit-step", "2", "--mem-fraction", "1.0", "--eval-every", "0",
           "--quiet", "1", "--name", name, "--log-every", "1", *extra]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    s = load(name) if p.returncode == 0 else None
    for ext in (".json", ".jsonl"):
        (RUNS / f"{name}{ext}").unlink(missing_ok=True)
    if s is None:
        err = [l for l in p.stderr.splitlines() if "out of memory" in l.lower() or "Error" in l]
        return {"batch": batch, "fits": False, "error": (err[-1] if err else p.stderr[-300:]).strip()[:300]}
    return {"batch": batch, "fits": True, "peak_exact_mib": s["peak_exact_mib"], "peak_driver_mib": s["peak_driver_mib"],
            "tok_per_s": s["tok_per_s"]}


def max_batch(variant, h, start=32, step=8, verbose=True, extra=()):
    """Double until a batch fails, then bisect to a multiple of `step`."""
    trail = []

    def t(b):
        r = probe(variant, h, b, extra=extra)
        trail.append(r)
        if verbose:
            msg = (f"peak {r['peak_exact_mib']:>8,.0f} MiB  {r['tok_per_s']:>7,.0f} tok/s" if r["fits"]
                   else "does not fit: " + r["error"][:90])
            print(f"  {variant:9s} batch {b:4d}  {msg}", flush=True)
        return r["fits"]

    lo, hi = 0, start
    while t(hi):
        lo, hi = hi, hi * 2
    while hi - lo > step:
        mid = (lo + hi) // 2 // step * step
        if mid <= lo:
            break
        if t(mid):
            lo = mid
        else:
            hi = mid
    return lo, sorted(trail, key=lambda r: r["batch"])
