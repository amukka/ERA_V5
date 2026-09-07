"""Assemble session10.ipynb from the experiment modules.

The notebook is not a second copy of the code. Every cell either shows a short
piece of arithmetic that is worth watching happen, or calls the same
``experiments/*.py`` module the command line runs, so there is exactly one
implementation of every number in this session and no way for the notebook and
the repository to disagree.

    python tools/build_notebook.py          # write the notebook, unexecuted
    python tools/build_notebook.py --run    # write it and execute it in place
"""

from __future__ import annotations

import pathlib
import sys

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "session10.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


CELLS = [
md("""
# Session 10 — The Training Loop

**Take a small model and a real loop, and make it tell you the truth about itself.**

An 8.3M-parameter GPT, a byte-level BPE tokenizer and a corpus slice — all three
built in earlier sessions of this course — put through one honest training loop
and made to answer six questions about itself.

| # | the question | where it is answered |
|---|---|---|
| 1 | every tensor shape in the step, and what each dimension means | `experiments/e1_shapes.py` |
| 2 | one gradient verified by hand against `backward()` | `experiments/e2_grad_check.py` |
| 3 | gradient accumulation broken on purpose, both curves plotted | `experiments/e3_accumulation.py` |
| 4 | the grad norm at every step, and a step where it moved first | `experiments/e4_grad_norm.py` |
| 5 | my own MFU, and what is costing me the distance to 40% | `experiments/e5_mfu.py` |
| 6 | 0.1 in fp32, bf16 and fp8 E4M3, showing the bits | `experiments/e6_float_bits.py` |

Every number below is computed when this notebook runs. Nothing is quoted.
"""),

md("## 0. Setup"),
code("""
import sys, pathlib, json
ROOT = pathlib.Path.cwd()                # walk up if opened from a subdirectory
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from IPython.display import Markdown, Image, display

import torch
from src.report import machine
from src.loop import pick_device

def show(name):
    "Render an experiment's written-out findings inline."
    display(Markdown((ROOT / "results" / name).read_text()))

print(json.dumps(machine(), indent=2))
print("device the loop will use:", pick_device())
"""),

md("""
### The data, the tokenizer and the model

The corpus slice is 1,052 documents of wikitext-2, source code and dialogue,
committed to this repository so a fresh clone reproduces every number here. The
tokenizer is the 10,000-id byte-level BPE built in session 2. Together they give
about 1.5M training tokens — small, real, and enough.
"""),
code("""
from src.data import load_documents, Sampler, VOCAB_SIZE, PAD_ID
from src.model import TinyGPT, Config

docs, lanes = load_documents(verbose=True)
print("vocab:", VOCAB_SIZE, "(10,000 BPE ids + one PAD)")

cfg = Config()
model = TinyGPT(cfg)
print(f"\\n{cfg.n_layer} layers · d_model {cfg.d_model} · {cfg.n_head} heads · "
      f"d_ff {cfg.d_ff} · context {cfg.max_seq}")
print(f"parameters: {model.n_params():,} total, "
      f"{model.n_params(embeddings=False):,} non-embedding")
"""),

md("""
---
# 1. Every tensor shape in the step

A step touches four populations of tensors and it is easy to print only the
first: the activations, the loss scalars, the parameters *and their gradients*,
and the optimiser's per-weight state. The model narrates its own forward pass —
`forward(..., trace=[])` records every intermediate with a line saying what each
axis selects — so the table cannot drift away from the code that produced it.
"""),
code("""
import experiments.e1_shapes as e1
shapes = e1.main(verbose=False)
print(f"{len(shapes['activations'])} activation tensors traced, "
      f"every gradient shape matches its weight: {shapes['grad_shapes_match']}")
show("e1_shapes.md")
"""),

md("""
---
# 2. One gradient, verified by hand

Nudge a weight, measure how the loss changed, compare against what `backward()`
reported. First on the session's own four-link chain where the answer is 64
exactly, then on one weight of the real 8.3M-parameter model.
"""),
code("""
# the chain from section 4, done the way a person would do it
x, t, w1, w2 = 2.0, 20.0, 3.0, 4.0
h = w1 * x
y = w2 * h
loss = (y - t) ** 2
print(f"h = {h}   y = {y}   loss = {loss}")

dL_dy  = 2 * (y - t)
dL_dw2 = dL_dy * h
dL_dh  = dL_dy * w2
dL_dw1 = dL_dh * x
print(f"dL/dy = {dL_dy}   dL/dw2 = {dL_dw2}   dL/dh = {dL_dh}   dL/dw1 = {dL_dw1}")

eps = 1e-6
nudged = ((w2 * ((w1 + eps) * x) - t) ** 2 - (w2 * ((w1 - eps) * x) - t) ** 2) / (2 * eps)
a = torch.tensor(w1, dtype=torch.float64, requires_grad=True)
((w2 * (a * x)) - t).pow(2).backward()
print(f"\\nby hand      {dL_dw1:.12f}")
print(f"by nudging   {nudged:.12f}")
print(f"by backward  {float(a.grad):.12f}")
"""),
code("""
import experiments.e2_grad_check as e2
grad = e2.main(verbose=False)
best = grad["float64"]["best"]
print(f"real model, float64: backward() {grad['float64']['backward_reported']:.12f}")
print(f"                     nudge      {best['finite_difference']:.12f}")
print(f"                     agreement  {best['decimals']:.1f} decimal places")
show("e2_grad_check.md")
display(Image(filename=str(ROOT / "results" / "e2_grad_check.png")))
"""),

md("""
---
# 3. Gradient accumulation, broken on purpose

Average the averages with micro-batches of different lengths, and plot both
curves together. The micro-batches here are length-bucketed, which is what a
real loader does for throughput and also the setting in which the bug bites
hardest.
"""),
code("""
# section 8's arithmetic, computed
micro = [(4, 2.0), (4, 2.0), (2, 5.0)]      # (valid tokens, average loss)
correct = sum(n * l for n, l in micro) / sum(n for n, _ in micro)
wrong   = sum(l for _, l in micro) / len(micro)
print(f"token-weighted      {correct:.4f}")
print(f"average of averages {wrong:.4f}")
print(f"error               {100*(wrong-correct)/correct:.1f}%")
"""),
code("""
import experiments.e3_accumulation as e3
accum = e3.main(verbose=False)          # four training runs; a few minutes
c = accum["one_step"]["comparison"]
print(f"one step, same weights, same micro-batches:")
print(f"  reported loss differs by {c['reported_loss_error_pct']:+.2f}%")
print(f"  the two gradients differ by {100*c['relative_l2_difference']:.2f}% "
      f"and point {c['angle_degrees']:.2f}deg apart")
print(f"\\nafter {accum['steps']} steps: gap {accum['gap_bucketed']:+.4f} nats, "
      f"control {accum['gap_fixed']:+.4f} nats")
show("e3_accumulation.md")
display(Image(filename=str(ROOT / "results" / "e3_accumulation.png")))
"""),

md("""
---
# 4. The grad norm, at every step

Logged before clipping, every step, alongside a fixed held-out probe loss
evaluated after every step — so the loss trace moves only when the model moves,
not when the batch happens to be hard.
"""),
code("""
import experiments.e4_grad_norm as e4
norms = e4.main(verbose=False)          # two 420-step runs; a few minutes
inj = norms["injected"]
print(f"injected anomaly at step {norms['inject_at']}: "
      f"norm z = {inj['grad_norm_z']:.1f} sigma, "
      f"probe loss crossed at step {inj['loss_crossing_step']} "
      f"({inj['lead_steps']} steps of warning)")
if norms["headline_event"]:
    h = norms["headline_event"]
    print(f"unprompted: norm crossed at step {h['grad_norm_step']} "
          f"({h['grad_norm_z']:.1f} sigma), loss at {h['loss_step']} "
          f"-- {h['lead_steps']} steps early")
show("e4_grad_norm.md")
display(Image(filename=str(ROOT / "results" / "e4_grad_norm.png")))
"""),

md("""
---
# 6. 0.1 in fp32, bf16 and fp8 E4M3

Derived by hand in exact rational arithmetic, then checked against the bits the
hardware actually stores. (Out of order on purpose: deliverable 5 measures
throughput and has to run last, with nothing else competing for the device.)
"""),
code("""
from src.floats import FP32, BF16, FP8_E4M3, encode, binary_expansion, torch_bits

print("0.1 in binary:", binary_expansion(0.1, 28))
print()
for fmt, dt in ((FP32, torch.float32), (BF16, torch.bfloat16),
                (FP8_E4M3, torch.float8_e4m3fn)):
    e = encode(0.1, fmt)
    hw = torch_bits(0.1, dt)
    print(f"{fmt.name:9} {e.bits:34} {e.hex:>10}  "
          f"= {float(e.value):.12f}  rel err {float(e.rel_error):.2e}  "
          f"matches hardware: {hw == e.bits_flat}")
"""),
code("""
import experiments.e6_float_bits as e6
floats = e6.main(verbose=False)
u = floats["real_update_sizes"]
print(f"measured on this session's own run: {u['pct_updates_lost_in_bf16']:.1f}% of "
      f"{u['n_weights']:,} weight updates would round to nothing in bare bf16 "
      f"({u['pct_updates_lost_in_fp8_e4m3']:.1f}% in fp8 E4M3)")
show("e6_float_bits.md")
display(Image(filename=str(ROOT / "results" / "e6_float_bits.png")))
"""),

md("""
---
# 5. MFU, measured and attributed

Run last and alone. The denominator is not a datasheet number: it is the best
sustained matmul throughput this machine is measured to reach, in the same dtype
the loop runs in.

**If you are re-executing this notebook, stop anything else on the machine
before this cell.** Otherwise the number below is a number about your other
process.

**And expect this cell to disagree with the README, by construction.** It
measures the roofline inside a kernel that has just trained six models, so the
device is warm and contended and every matmul in the denominator reads slow. A
depressed denominator does not lower the MFU — it *raises* it. Measured here the
same loop scores about 41%; measured standalone, on an idle machine, it scores
39.2%, and 39.2% is the number the README reports. `run_all.py` therefore runs
E5 last and on its own, and the committed `results/e5_mfu.*` come from that run,
not from this cell.

That gap is the deliverable, not a blemish on it. Nothing about the loop changed
between the two measurements. An MFU is only ever as honest as the peak you
divide by.
"""),
code("""
import experiments.e5_mfu as e5
mfu = e5.main(verbose=False)
b = mfu["baseline"]
print(f"measured roofline : {mfu['peak_tflops']:.2f} TFLOP/s fp32")
print(f"tokens per second : {b['tokens_per_second']:,.0f}")
print(f"achieved          : {b['achieved_tflops']:.3f} TFLOP/s")
print(f"MFU               : {100*b['mfu']:.2f}%")
show("e5_mfu.md")
display(Image(filename=str(ROOT / "results" / "e5_mfu.png")))
"""),

md("""
---
# What the loop said about itself

One table, one line each, every number from the cells above.
"""),
code("""
rows = [
    ("1", "tensors traced in one step",
     f"{len(shapes['activations'])} activations + "
     f"{len(list(model.named_parameters()))} parameters, every gradient the "
     f"shape of its weight"),
    ("1", "training state at 16 bytes/weight",
     f"{shapes['training_state_mib']:.1f} MiB for {shapes['n_params']:,} weights"),
    ("2", "nudge vs backward(), float64",
     f"agree to {grad['float64']['best']['decimals']:.1f} decimals "
     f"(rel. error {grad['float64']['best']['rel_error']:.1e})"),
    ("2", "the same check in float32",
     f"agrees to only {grad['float32']['best']['decimals']:.1f} decimals -- "
     f"cancellation, not a bug in backward()"),
    ("3", "average-of-averages, one step",
     f"gradient {100*accum['one_step']['comparison']['relative_l2_difference']:.1f}% "
     f"off, {accum['one_step']['comparison']['angle_degrees']:.1f}deg away"),
    ("3", f"after {accum['steps']} steps",
     f"{accum['gap_bucketed']:+.4f} nats of validation loss "
     f"({accum['gap_fixed']:+.4f} in the equal-length control)"),
    ("4", "grad norm vs probe loss",
     f"{norms['injected']['lead_steps']} steps of warning on the injected "
     f"anomaly" +
     (f"; {norms['headline_event']['lead_steps']} steps unprompted at step "
      f"{norms['headline_event']['grad_norm_step']}"
      if norms["headline_event"] else "")),
    ("5", "MFU",
     f"{100*mfu['baseline']['mfu']:.2f}% of a measured "
     f"{mfu['peak_tflops']:.2f} TFLOP/s"),
    ("5", "what closes most of the gap",
     f"micro-batch size alone reaches "
     f"{100*mfu['best_measured']['mfu']:.2f}% with no code change"),
    ("6", "0.1, hand-derived vs hardware",
     f"all {sum(1 for e in floats['encodings'] if e['matches_hardware'])} "
     f"checkable formats match bit for bit"),
    ("6", "bare bf16 weights",
     f"{floats['real_update_sizes']['pct_updates_lost_in_bf16']:.1f}% of real "
     f"updates round to nothing -- hence the fp32 master copy"),
]
display(Markdown(
    "| # | what was measured | what it said |\\n|---|---|---|\\n"
    + "\\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)))
"""),

md("""
Every serious training bug in this session was silent. The average of the
averages printed a loss wrong by a fraction of a percent while pointing the
optimiser several degrees off course. Bare bf16 stops a weight moving without
raising anything. A run at 5% MFU draws the same loss curve as a run at 45% and
costs nine times as much. None of them would have been found by watching the
loss.

Print things and check things.
"""),
]


def build():
    nb = nbf.v4.new_notebook(cells=CELLS)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    })
    OUT.write_text(nbf.writes(nb))
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(CELLS)} cells)")
    return OUT


def run(path):
    from nbclient import NotebookClient
    nb = nbf.read(path, as_version=4)
    print("executing (this takes 15-25 minutes: it trains six models) ...")
    NotebookClient(nb, timeout=7200, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.write(nb, path)
    print(f"executed and saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    p = build()
    if "--run" in sys.argv:
        run(p)
