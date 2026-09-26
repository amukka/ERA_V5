"""Run every experiment behind the submission, in order, and write results/.

    python run_all.py            # the full set, about 45 minutes
    python run_all.py --quick    # a reduced sweep, about 5 minutes
    python run_all.py --only e3  # one experiment

Each experiment is independent and writes one JSON file. E1 needs network the
first time (it pulls the GPT-2 and XLM-R vocabularies from the Hub); everything
else runs from files already in this repository.
"""

from __future__ import annotations

import argparse
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXPERIMENTS = {
    "e1": (
        "experiments.e1_injectivity",
        "Does the byte code name every token uniquely?",
        [],
        ["--vocabs", "era-v5", "gpt2"],
    ),
    "e2": (
        "experiments.e2_recovery",
        "Does the code come back out of a d-dimensional embedding?",
        [],
        ["--dims", "128", "256", "512", "--eval-n", "2000",
         "--noise", "0.0", "0.1", "0.4"],
    ),
    "e3": (
        "experiments.e3_headfree_lm",
        "Does a language model with no output head train as well?",
        [],
        ["--steps", "300", "--eval-every", "150", "--eval-n", "4"],
    ),
    "e4": (
        "experiments.e4_scaling",
        "What does the vocabulary cost, and does a 1M-token one work?",
        [],
        ["--million", "100000", "--id-queries", "128"],
    ),
}

# Quick mode is a different measurement, not a cheaper version of the same one,
# so it writes its own files rather than overwriting the results the README
# quotes.
QUICK_SUFFIX = "_quick.json"


def run(key: str, quick: bool) -> float:
    module, blurb, full_args, quick_args = EXPERIMENTS[key]
    print(f"\n{'=' * 74}\n{key.upper()}  {blurb}\n{'=' * 74}", flush=True)
    args = list(quick_args if quick else full_args)
    if quick:
        args += ["--out", module.split(".")[-1] + QUICK_SUFFIX]
    sys.argv = [module] + args
    t0 = time.time()
    runpy.run_module(module, run_name="__main__")
    return time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=sorted(EXPERIMENTS))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    keys = args.only or sorted(EXPERIMENTS)
    timings = {k: run(k, args.quick) for k in keys}

    print(f"\n{'=' * 74}")
    for k, s in timings.items():
        print(f"  {k}  {s:7.1f}s")
    print(f"  results in {ROOT / 'results'}")


if __name__ == "__main__":
    main()
