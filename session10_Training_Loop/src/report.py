"""Small helpers so every experiment leaves the same kind of evidence behind."""

from __future__ import annotations

import json
import pathlib
import platform
import subprocess

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def machine() -> dict:
    chip = ""
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        pass
    return {
        "chip": chip or platform.processor(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": ("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu"),
    }


def save_json(name: str, payload: dict) -> pathlib.Path:
    path = RESULTS / name
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def save_text(name: str, text: str) -> pathlib.Path:
    path = RESULTS / name
    path.write_text(text)
    return path


def table(rows, headers, aligns=None) -> str:
    """Render a markdown table.  ``rows`` is a list of tuples of strings."""
    rows = [[str(c) for c in r] for r in rows]
    aligns = aligns or ["l"] * len(headers)
    widths = [max(len(str(headers[i])), *(len(r[i]) for r in rows)) if rows
              else len(str(headers[i])) for i in range(len(headers))]
    def line(cells):
        out = []
        for i, c in enumerate(cells):
            out.append(c.rjust(widths[i]) if aligns[i] == "r"
                       else c.ljust(widths[i]))
        return "| " + " | ".join(out) + " |"
    sep = "| " + " | ".join(
        ("-" * (widths[i] - 1) + ":") if aligns[i] == "r" else ("-" * widths[i])
        for i in range(len(headers))) + " |"
    return "\n".join([line([str(h) for h in headers]), sep]
                     + [line(r) for r in rows])
