"""Run every experiment in this session, in order, writing to ``results/``.

    python run_all.py            # everything (about 90 minutes on an Apple M4)
    python run_all.py e1 e2      # just those

E5 runs second because it is also the tuning step: E2 and E3 train the
width-256 model at the best learning rate E5's sweep found, rather than at a
value picked by hand. E4 tunes its two schedules itself.

Each experiment is also runnable on its own (``python experiments/e1_adam_by_hand.py``).
E5 caches every run in ``results/e5_runs.json``; delete that file to re-sweep
from scratch.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
EXPERIMENTS = [
    ("e1", "e1_adam_by_hand",     "Adam by hand, checked against PyTorch"),
    ("e5", "e5_lr_width",         "learning rate sweep at three widths"),
    ("e2", "e2_bias_correction",  "bias correction off, twenty steps both ways"),
    ("e3", "e3_update_ratio",     "update-to-weight ratio per layer, and warmup"),
    ("e4", "e4_cosine_vs_wsd",    "cosine against WSD, stopped at step 200"),
]


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv):
    wanted = set(a.lower() for a in argv) or {k for k, _, _ in EXPERIMENTS}
    for key, name, label in EXPERIMENTS:
        if key not in wanted:
            continue
        print(f"\n{'=' * 72}\n{key.upper()}  {label}\n{'=' * 72}", flush=True)
        t0 = time.perf_counter()
        load(name).main(verbose=True)
        print(f"  -> results/{name}.md  ({time.perf_counter() - t0:.0f}s)")
    print("\ndone. Evidence is in results/.")


if __name__ == "__main__":
    main(sys.argv[1:])
