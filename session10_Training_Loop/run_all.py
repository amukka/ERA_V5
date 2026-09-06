"""Run every experiment in this session, in order, writing to ``results/``.

    python run_all.py            # everything
    python run_all.py e2 e6      # just those

Each experiment is also runnable on its own (``python experiments/e2_grad_check.py``)
and each returns the same dict it writes to ``results/<name>.json``.

Deliverable 5 measures throughput, so it is run last and on its own; do not run
anything else on the machine while it is going or the number it reports will be
a number about your other process.
"""

from __future__ import annotations

import importlib
import sys
import time

EXPERIMENTS = [
    ("e1", "experiments.e1_shapes",      "every tensor shape in one step"),
    ("e2", "experiments.e2_grad_check",  "one gradient, verified by hand"),
    ("e3", "experiments.e3_accumulation","gradient accumulation, broken on purpose"),
    ("e4", "experiments.e4_grad_norm",   "the grad norm, and where it leads the loss"),
    ("e6", "experiments.e6_float_bits",  "0.1 in fp32, bf16 and fp8 E4M3"),
    ("e5", "experiments.e5_mfu",         "MFU, measured and attributed"),
]


def main(argv):
    wanted = set(a.lower() for a in argv) or {k for k, _, _ in EXPERIMENTS}
    for key, module, label in EXPERIMENTS:
        if key not in wanted:
            continue
        print(f"\n{'='*72}\n{key.upper()}  {label}\n{'='*72}", flush=True)
        t0 = time.perf_counter()
        importlib.import_module(module).main(verbose=False)
        print(f"  -> results/{key}_*.json  ({time.perf_counter()-t0:.1f}s)")
    print("\ndone. Evidence is in results/.")


if __name__ == "__main__":
    main(sys.argv[1:])
