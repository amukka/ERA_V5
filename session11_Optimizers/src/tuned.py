"""The width-256 learning rate, taken from E5's sweep rather than guessed.

E2 and E3 train the width-256 model at one peak learning rate. That value is
read from E5's result (the best point on its grid), so the experiments that
use it are not running at a number somebody liked. If E5 has not been run yet
the fallback is used and every write-up says so.
"""

from __future__ import annotations

import json

from .report import RESULTS

FALLBACK = 2e-3


def lr_256() -> tuple[float, str]:
    p = RESULTS / "e5_runs.json"          # E5's per-run cache
    if p.exists():
        pts = [r for r in json.loads(p.read_text()).values()
               if r["width"] == 256 and r["seed"] == 0 and r["val"] is not None]
        if len(pts) >= 5:
            best = min(pts, key=lambda r: r["val"])
            return best["lr"], "best point of E5's width-256 sweep"
    return FALLBACK, "fallback; E5 has not been run"
