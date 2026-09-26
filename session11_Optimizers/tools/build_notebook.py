"""Assemble session11.ipynb.

The notebook is not a second copy of the code. E1 runs live (it takes a
second). E2 recomputes its closed form live. The four training experiments
take about ninety minutes together, so their cells render the evidence that
``run_all.py`` wrote to ``results/``, and say so. There is one implementation of
every number, in ``experiments/``.

    python tools/build_notebook.py          # write the notebook, unexecuted
    python tools/build_notebook.py --run    # write it and execute it in place
"""

from __future__ import annotations

import pathlib
import sys

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "session11.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


CELLS = [
md("""
# Session 11 — Optimizers and Learning-Rate Schedules

A gradient gives a direction and not a distance. These five experiments measure
the rules that turn one into the other, on the Session 10 model and corpus.

| # | the question | where it is answered |
|---|---|---|
| 1 | Adam by hand for one weight and five gradients, against PyTorch | `experiments/e1_adam_by_hand.py` |
| 2 | bias correction off: twenty steps both ways, and when it stops mattering | `experiments/e2_bias_correction.py` |
| 3 | the update-to-weight ratio per layer, and where warmup stops changing it | `experiments/e3_update_ratio.py` |
| 4 | cosine against WSD, both stopped at step 200 | `experiments/e4_cosine_vs_wsd.py` |
| 5 | learning-rate sweeps at widths 256, 512 and 1,024, extrapolated to 4,096 | `experiments/e5_lr_width.py` |
"""),
md("## 0. Setup"),
code("""
import sys, pathlib, json, importlib.util
ROOT = pathlib.Path.cwd()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from IPython.display import Markdown, Image, display
from src.report import machine

def experiment(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def show(name, image=True):
    display(Markdown((ROOT / "results" / f"{name}.md").read_text()
                     .replace(f"![]({name}.png)", "")))
    if image and (ROOT / "results" / f"{name}.png").exists():
        display(Image(str(ROOT / "results" / f"{name}.png")))

print(json.dumps(machine(), indent=2))
"""),
md("""
## 1. Adam by hand

This runs live. It computes m, v, m̂, v̂ and the step for five gradients in plain
Python floats, then runs `torch.optim.Adam` on the same five gradients and
compares every quantity, in float64 and float32.
"""),
code("""
e1 = experiment("e1_adam_by_hand")
out = e1.main(verbose=False)
for r in out["hand"]:
    print(f"t={r['t']}  g={r['g']:.2f}  m={r['m']:.6f}  v={r['v']:.8f}  "
          f"m̂={r['m_hat']:.6f}  v̂={r['v_hat']:.6f}  step={r['step']:+.9f}")
"""),
code('show("e1_adam_by_hand", image=False)'),
md("""
## 2. Bias correction off

The single-weight half has a closed form, so it is recomputed live: the
uncorrected step is exactly r(t) = (1 − β₁ᵗ)/√(1 − β₂ᵗ) times the corrected one.
"""),
code("""
e2 = experiment("e2_bias_correction")
for b2 in (0.999, 0.95):
    print(f"β₂={b2}:  r(1)={e2.r_of(1,b2):.2f}  r(10)={e2.r_of(10,b2):.2f}  "
          f"r(20)={e2.r_of(20,b2):.2f}  within 10% from step {e2.settle_step(b2,0.10):,}  "
          f"within 1% from step {e2.settle_step(b2,0.01):,}")
"""),
md("The model half trains twelve 300-step runs. Its evidence, from `run_all.py`:"),
code('show("e2_bias_correction")'),
md("## 3. Update-to-weight ratio, per layer"),
code('show("e3_update_ratio")'),
md("## 4. Cosine against WSD"),
code('show("e4_cosine_vs_wsd")'),
md("## 5. Learning rate against width"),
code('show("e5_lr_width")'),
]


def main(run: bool):
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata["kernelspec"] = {"display_name": "Python 3",
                                 "language": "python", "name": "python3"}
    if run:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=600, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.write(nb, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main("--run" in sys.argv)
