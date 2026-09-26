"""Build session9_loss_harness.ipynb from the experiments in ../experiments.

The notebook is generated so its prose cannot drift from the code: every
section is one experiment's ``main()``, and the write-up cells display the
markdown that experiment produced.  Run ``python tools/build_notebook.py`` and
then execute it (``jupyter nbconvert --execute --inplace``).
"""

import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parent.parent
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Session 9 · A loss harness that is correct and observable

The assignment hands us four lines:

```python
hidden = model(tokens)
logits = output_head(hidden)
loss = cross_entropy(logits[:, :-1].reshape(-1, vocab_size),
                     tokens[:, 1:].reshape(-1))
```

They compute *a* number. They do not say which position predicts which token, which pairs are allowed
to count, what the sum is divided by, or whether the logits must exist all at once. This notebook makes
each of those a named, printed, testable decision, then adds a second head that predicts token `t+2`.

Runs top to bottom on Colab (GPU or CPU) or a laptop. Every section calls the matching script in
`experiments/`, so the numbers here are the numbers in `results/` and in the README.

**Part 1**: 1 shapes · 2 shift, as strings · 3 padding · 4 packing · 5 perplexity · 6 tied vs untied ·
7 peak memory. **Part 2**: the `t+2` head.
""")

md("## 0 · Setup")
code("""
import os, sys, subprocess, pathlib
REPO = "https://github.com/amukka/ERA_V5.git"
SUB = "session9_LossFunctions_Outputs"
if not pathlib.Path("src/losses.py").exists():                      # not already inside the folder
    if not pathlib.Path(SUB).exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO, "_repo"], check=True)
        os.replace(f"_repo/{SUB}", SUB)
    os.chdir(SUB)
sys.path.insert(0, os.getcwd())

import torch, importlib
from IPython.display import Markdown, Image, display
from src.train import pick_device
print("cwd   :", os.getcwd())
print("torch :", torch.__version__, "| device:", pick_device())

def show(name, figure=True, md=True):
    \"\"\"Run experiment `name`, then display the write-up it produced.\"\"\"
    mod = importlib.import_module(f"experiments.{name}")
    out = mod.main(verbose=False)
    if md:
        display(Markdown(pathlib.Path(f"results/{name}.md").read_text()))
    if figure and pathlib.Path(f"results/{name}.png").exists():
        display(Image(f"results/{name}.png"))
    return out
""")

md("""
## The harness

`src/losses.py` is the assignment's four lines with every implicit decision written out:

| decision | where it lives |
|---|---|
| which position predicts which token | `align(T, offset)` |
| which pairs may contribute (padding, document joins) | `loss_mask(...)` |
| what the sum is divided by | `LossRecord.n_contributing` |
| whether all the logits must exist at once | `chunked_cross_entropy(...)` |
""")
code("""
import inspect
from src.losses import lm_loss
print(inspect.getsource(lm_loss))
""")

md("## 1 · Every tensor shape, and what each dimension is\n\nThe model threads a `trace` through its forward pass, so each row below was emitted by the code that built the tensor.")
code('r1 = show("e1_shapes", figure=False)')

md("## 2 · The shift, verified with strings\n\nThree alignments, printed as text: correct, no shift, shifted backwards. Then each is trained for 400 steps.")
code('r2 = show("e2_shift")')

md("## 3 · Mask padding, and the count changes")
code('r3 = show("e3_padding")')

md("## 4 · Pack two documents, mask the join")
code('r4 = show("e4_packing")')

md("## 5 · Perplexity of an untrained model ≈ vocabulary size")
code('r5 = show("e5_perplexity")')

md("## 6 · Tied against untied head")
code('r6 = show("e6_tying")')

md("## 7 · Peak memory: ordinary against chunked cross-entropy")
code('r7 = show("e7_memory")')

md("## Part 2 · One extra head, predicting token `t+2`")
code('r8 = show("e8_extra_head", figure=True, md=True)')

md("## The numbers, in one place")
code("""
import json
R = lambda n: json.load(open(f"results/{n}.json"))
e2, e3, e4, e5, e6, e7, e8 = (R(n) for n in ["e2_shift", "e3_padding", "e4_packing", "e5_perplexity", "e6_tying", "e7_memory", "e8_extra_head"])
a = e8["aggregate"]
rows = [
 ("1 shapes",              "logits 4x128x10002 = 39.1x the hidden state (see table above)"),
 ("2 shift",               "reported loss, honest next-token loss: " + "; ".join(f"{k} {v['mean_last_20']:.2f}/{v['honest_next_token_loss']:.2f}" for k, v in e2["arms"].items())),
 ("3 padding",             f"contributing tokens {e3['counts']['pairs_before_masking']} -> {e3['counts']['pairs_after_masking']}"),
 ("4 packing",             f"loss {e4['untrained']['loss_unmasked']:.4f} -> {e4['untrained']['loss_masked']:.4f} untrained; after training {e4['density']['2']['loss_unmasked']:.4f} -> {e4['density']['2']['loss_masked']:.4f}"),
 ("5 perplexity",          f"{e5['untrained']['perplexity']:,.0f} vs V = {e5['anchor']['vocab_size']:,} ({e5['untrained']['perplexity_over_vocab']:.3f}x)"),
 ("6 tied vs untied",      f"{e6['counts']['untied_total']:,} vs {e6['counts']['tied_total']:,} (saves {e6['counts']['saved']:,}, {100*e6['counts']['saved_fraction']:.1f}%)"),
 ("7 peak memory",         f"ordinary {e7['ordinary']['peak_mib']:.0f} MiB vs chunked {e7['chunked']['peak_mib']:.0f} MiB = {e7['ratio_loss_step']:.2f}x"),
 ("Part 2 head 1 (t+1)",   f"{a['end_l1'][0]:.3f} held-out"),
 ("Part 2 head 2 (t+2)",   f"{a['end_l2'][0]:.3f} held-out"),
 ("Part 2 sum",            f"{a['end_sum'][0]:.3f}"),
]
for k, v in rows: print(f"{k:<22} {v}")
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "colab": {"provenance": []}}
out = ROOT / "session9_loss_harness.ipynb"
nbf.write(nb, out)
print("wrote", out)
