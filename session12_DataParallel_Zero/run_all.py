"""Run every experiment in this session, in order, writing to ``results/``.

    python run_all.py            # everything, about four minutes
    python run_all.py e3 e4      # just those

Each experiment is also runnable on its own (``python experiments/e3_stages.py``)
and each returns the same dict it writes to ``results/<name>.json``.

E1 to E4 spawn up to 64 threads and measure bytes, which no other process can
disturb.  E5 measures a per-layer time profile on a single thread, so it is run
last and should have the machine to itself; its *volumes* come from E3's ledger
either way, so a contended profile changes the shape of the simulation and not
the traffic it is simulating.
"""

from __future__ import annotations

import importlib
import sys
import time

EXPERIMENTS = [
    ("e1", "experiments.e1_mesh",        "32 virtual GPUs and three collectives"),
    ("e2", "experiments.e2_equivalence", "the same answer, bitwise, from all four"),
    ("e3", "experiments.e3_stages",      "what each stage costs: bytes, wire, clock"),
    ("e4", "experiments.e4_ladder",      "the memory wall, 1 GPU to 64"),
    ("e5", "experiments.e5_overlap",     "overlapping communication with compute"),
]


def main(argv):
    wanted = set(a.lower() for a in argv) or {k for k, _, _ in EXPERIMENTS}
    for key, module, label in EXPERIMENTS:
        if key not in wanted:
            continue
        print(f"\n{'=' * 72}\n{key.upper()}  {label}\n{'=' * 72}", flush=True)
        t0 = time.perf_counter()
        importlib.import_module(module).main(verbose=False)
        print(f"  -> results/{key}_*.json  ({time.perf_counter() - t0:.1f}s)")
    print("\ndone. Evidence is in results/.")


if __name__ == "__main__":
    main(sys.argv[1:])
