"""Write the session's notebooks, and optionally execute one of them in place.

    python tools/build_notebooks.py                 # write all notebooks, unexecuted
    python tools/build_notebooks.py --run 02        # write them, then execute 02_*.ipynb in place

Each notebook calls src/experiments.py, the same code the command line runs, so the notebooks and
the repository cannot disagree. The training notebooks take one to two hours each on an Apple M4.
"""
import argparse
import pathlib
import time

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parent.parent


def md(t):
    return nbf.v4.new_markdown_cell(t.strip("\n"))


def code(t):
    return nbf.v4.new_code_cell(t.strip("\n"))


SETUP = code('''
import sys, os, json, pathlib, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB and not pathlib.Path("ERA_V5").exists():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/amukka/ERA_V5.git"], check=True)
    os.chdir("ERA_V5/session13_Distributed_Training")
ROOT = pathlib.Path.cwd()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
if not (ROOT / "data" / "train.bin").exists():          # ~5 minutes: stream FineWeb-Edu, train the BPE
    subprocess.run([sys.executable, "tools/prepare_data.py"], check=True)

import torch, platform
from src import experiments as X
from src.train import pick_device
dev = pick_device()
print("device:", dev, "| torch", torch.__version__, "| python", platform.python_version(), "|", platform.platform())
if dev == "mps":
    print(f"MPS recommended working set: {torch.mps.recommended_max_memory()/2**30:.2f} GiB")
if dev == "cuda":
    print(torch.cuda.get_device_name(), f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
print(json.dumps(json.loads((ROOT / "data" / "meta.json").read_text()), indent=1))
''')

MODEL = md('''
**The model** (identical in every run): GPT, 10 layers, d_model 384, 6 heads, MLP 4x, sequence 512,
byte-level BPE vocabulary 8,192 with the head tied to the embedding. **21,053,184 parameters**, of which
17,710,848 are outside the embedding tables. No dropout anywhere, because a reversible backward pass
recomputes each block and a random mask would recompute differently. AdamW (0.9, 0.95, wd 0.1), peak LR
1e-3, 3% warmup then cosine to 10%, gradient clip 1.0, fp32.
''')

NB = {}

NB["01_variant_screen"] = [
md('''
# 01 · Which reversible rule works?

Session 13 §16 gives the midpoint rule. The assignment asks which variant worked, so this notebook
tries five ways of moving the residual stream and keeps the one that trains best **and** still gives the
gradient autograd would. The block `f_l` is the same everywhere, `f(p) = a + mlp(ln2(p + a))` with
`a = attn(ln1(p))`, so every variant has exactly the same 21.05M weights.

| rule | forward | how the backward gets the layer input back |
|---|---|---|
| standard | `p[l+1] = p[l] + f(p[l])` | it doesn't: autograd stores every activation |
| **midpoint** (leapfrog) | `p[l+1] = p[l-1] + 2h f(p[l])` | exact: `p[l-1] = p[l+1] - 2h f(p[l])` |
| **euler_fp** | `p[l+1] = p[l] + h f(p[l])` (the standard net) | fixed-point iteration `p <- p[l+1] - h f(p)`, 6 rounds |
| **momentum** (symplectic Euler) | `v += h f(x); x += h v` | exact: `x = x' - h v'`, then `v = v' - h f(x)` |
| **revnet** (additive coupling) | `y1 = x1 + A(x2); y2 = x2 + M(y1)` | exact: `x2 = y2 - M(y1)`, `x1 = y1 - A(x2)` |

"Euler" is the ordinary residual network. It *can* be run backwards, but only by solving
`p = p[l+1] - h f(p)` for `p`, and that iteration converges only while `h f` is a contraction. The screen
checks whether it stays one.
'''),
SETUP,
MODEL,
md('''
## 1. The algebra, in float64

Before any training: for each rule, the gradient from the reversible backward (which stores only the
final state) against the gradient autograd computes for the same forward rule (which stores everything).
Tiny model, weights deliberately large (std 0.15) so the blocks are far from the identity.
'''),
code('''
from tests.test_reversible import check
rows = [check("midpoint", 0.5), check("momentum", 0.5), check("revnet", 1.0),
        check("euler_fp", 1.0, fp_iters=6), check("euler_fp", 1.0, fp_iters=30), check("euler_fp", 0.5, fp_iters=6)]
print(f"{'rule':10s} {'h':>5s}  {'grad rel err':>13s}  {'grad cosine':>12s}  {'max state rebuild err':>22s}")
for r in rows:
    print(f"{r['variant']:10s} {r['h']:5.2f}  {r['grad_rel_err']:13.2e}  {r['grad_cosine']:12.8f}  {r['max_state_rebuild_err']:22.2e}")
'''),
md('''
## 2. The screen: 4M tokens each, batch 32

Same data order, same seed, same schedule (compressed to 4M tokens). Each run ends with the
reversibility check on four held-out rows: the gradient the reversible backward returns against
autograd's, and the rebuilt layer inputs against the ones the forward computed, both at the trained
weights, in fp32 on the GPU.
'''),
code('''
screen = []
for variant, h in X.SCREEN:
    print(f"\\n=== {variant}  h={h} ===")
    s = X.train(name=X.screen_name(variant, h), variant=variant, h=h, batch=X.BASE_BATCH,
                tokens=4_000_000, eval_every=61, eval_batches=20, log_every=10)
    rv = (s["reversibility"] or {}).get("final") or {}
    screen.append({"variant": variant, "h": h, "final_val_loss": s["final_val_loss"],
                   "final_train_loss": s["final_train_loss"], "tok_per_s": s["tok_per_s"],
                   "peak_exact_mib": s["peak_exact_mib"], "diverged_at_step": s["diverged_at_step"],
                   "grad_cosine_final": rv.get("grad_cosine", 1.0),
                   "grad_rel_err_final": rv.get("grad_rel_err", 0.0),
                   "state_rebuild_err_final": rv.get("state_rebuild_rel_err_max", 0.0)})
(X.RESULTS / "screen.json").write_text(json.dumps(screen, indent=2))
'''),
code('''
print(f"{'rule':9s} {'h':>5s} {'val loss':>9s} {'tok/s':>8s} {'peak MiB':>9s} {'grad cos':>10s} {'grad err':>9s} {'rebuild err':>11s}")
for r in screen:
    print(f"{r['variant']:9s} {r['h']:5.2f} {r['final_val_loss']:9.4f} {r['tok_per_s']:8,.0f} {r['peak_exact_mib']:9,.0f} "
          f"{r['grad_cosine_final']:10.6f} {r['grad_rel_err_final']:9.2e} {r['state_rebuild_err_final']:11.2e}")
best = X.pick_variant(screen)
print("\\nchosen for the 50M-token runs:", best["variant"], "h =", best["h"])
(X.RESULTS / "chosen_variant.json").write_text(json.dumps(best, indent=2))
'''),
code('''
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for variant, h in X.SCREEN:
    log = X.load_log(X.screen_name(variant, h))
    st = [r for r in log if r["type"] == "step"]; ev = [r for r in log if r["type"] == "eval"]
    lab = f"{variant} h={h:g}"
    ax[0].plot([r["tokens"]/1e6 for r in st], [r["loss"] for r in st], label=lab, lw=1)
    ax[1].plot([r["tokens"]/1e6 for r in ev], [r["val_loss"] for r in ev], "o-", label=lab, ms=3)
for a, t in zip(ax, ["training loss", "validation loss"]):
    a.set_xlabel("tokens (M)"); a.set_title(t); a.set_ylim(3.5, 8); a.grid(alpha=.3); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(X.RESULTS / "screen.png", dpi=120); plt.show()
'''),
]

NB["02_baseline"] = [
md('''
# 02 · Baseline: standard GPT, 50M tokens, batch 32

The ordinary residual network, stored activations, plain autograd. Batch 32 x 512 = 16,384 tokens a step,
3,051 steps. Batch 32 is the fixed batch for the first two runs: it is the largest power of two the
**standard** model trains at on this machine (notebook 04 finds the exact ceiling).
'''),
SETUP,
MODEL,
code('''
base = X.train(name="baseline_standard_b32", variant="standard", batch=X.BASE_BATCH, tokens=X.TOKENS,
               eval_every=250, log_every=10)
'''),
code('''
keys = ["tokens_trained", "steps", "final_train_loss", "final_train_loss_avg_last5pct", "final_val_loss",
        "tok_per_s", "train_seconds", "peak_exact_mib", "peak_exact_op", "peak_alloc_mib", "peak_driver_mib"]
print(json.dumps({k: base[k] for k in keys}, indent=1))
'''),
]

NB["03_reversible"] = [
md('''
# 03 · Reversible, same batch: 50M tokens, batch 32

The rule chosen in notebook 01, at exactly the baseline's batch, data order, schedule and seed. The only
difference from notebook 02 is how the residual stream moves between layers and what the backward pass
keeps. This is the like-for-like comparison: same work, what does reversibility cost in time and save in
memory?
'''),
SETUP,
MODEL,
code('''
chosen = json.loads((X.RESULTS / "chosen_variant.json").read_text())
print("rule:", chosen["variant"], "h =", chosen["h"])
rev = X.train(name=f"reversible_{chosen['variant']}_b32", variant=chosen["variant"], h=chosen["h"],
              batch=X.BASE_BATCH, tokens=X.TOKENS, eval_every=250, log_every=10)
'''),
code('''
keys = ["tokens_trained", "steps", "final_train_loss", "final_train_loss_avg_last5pct", "final_val_loss",
        "tok_per_s", "train_seconds", "peak_exact_mib", "peak_exact_op", "peak_alloc_mib", "peak_driver_mib"]
print(json.dumps({k: rev[k] for k in keys}, indent=1))
print(json.dumps(rev["reversibility"], indent=1))
'''),
]

NB["04_reversible_max_batch"] = [
md('''
# 04 · Reversible, pushed to the maximum batch

First the ceiling, for both stacks: each candidate batch trains 4 steps in a fresh process, with the MPS
allocator capped at the recommended working set (so out-of-memory is a clean failure, not swap). The
search doubles until a batch fails and then bisects.

Then the reversible model trains the same 50M tokens at its own ceiling. A bigger batch means fewer
optimizer steps for the same tokens, so the peak learning rate is scaled by sqrt(batch / 32) (the usual
square-root rule for Adam), with 5% warmup. Everything else is unchanged.
'''),
SETUP,
MODEL,
code('''
chosen = json.loads((X.RESULTS / "chosen_variant.json").read_text())
print("searching the standard stack")
max_std, trail_std = X.max_batch("standard", 1.0, start=32, step=4)
print("searching the reversible stack:", chosen["variant"], "h =", chosen["h"])
max_rev, trail_rev = X.max_batch(chosen["variant"], chosen["h"], start=32, step=8)
print(f"\\nmax batch  standard: {max_std}   {chosen['variant']}: {max_rev}   ratio {max_rev/max_std:.2f}x")
(X.RESULTS / "max_batch.json").write_text(json.dumps(
    {"standard": {"max_batch": max_std, "trail": trail_std},
     chosen["variant"]: {"max_batch": max_rev, "trail": trail_rev, "h": chosen["h"]}}, indent=2))
'''),
code('''
import math
lr = 1e-3 * math.sqrt(max_rev / X.BASE_BATCH)
print(f"batch {max_rev} = {max_rev*512:,} tokens a step, {X.TOKENS // (max_rev*512):,} steps, peak LR {lr:.2e}")
big = X.train(name=f"reversible_{chosen['variant']}_maxbatch", variant=chosen["variant"], h=chosen["h"],
              batch=max_rev, tokens=X.TOKENS, lr=lr, warmup_frac=0.05, mem_fraction=1.0,
              eval_every=max(1, 250 * X.BASE_BATCH // max_rev), log_every=2)
'''),
code('''
keys = ["batch", "tokens_trained", "steps", "peak_lr", "final_train_loss", "final_train_loss_avg_last5pct",
        "final_val_loss", "tok_per_s", "train_seconds", "peak_exact_mib", "peak_exact_op", "peak_alloc_mib", "peak_driver_mib"]
print(json.dumps({k: big[k] for k in keys}, indent=1))
print(json.dumps(big["reversibility"], indent=1))
'''),
]


NB["06_head_bottleneck"] = [
md('''
# 06 · After reversibility, the head is the wall

Notebook 04's exact peaks landed in the loss (`_log_softmax_backward_data`) or in one layer's attention
softmax, never in the stack's stored activations, because the reversible stack no longer stores any. What
is left grows with the batch through the output head: one fp32 logits tensor is 512 x 8,192 x 4 bytes =
16 MiB per sequence, and the loss keeps several of them alive at once (logits, log-softmax, and its gradient).

This notebook changes one thing: the head and loss are computed 8 sequences at a time and recomputed in
the backward pass (`--loss-chunk 8`, gradient identical to the plain head to 2e-16 in float64). It then
repeats notebook 04's ceiling search for both stacks. No 50M-token training here, only the 4-step probes.
'''),
SETUP,
code('''
chosen = json.loads((X.RESULTS / "chosen_variant.json").read_text())
ex = ("--loss-chunk", "8")
print("standard, chunked head")
m_std, t_std = X.max_batch("standard", 1.0, start=32, step=4, extra=ex)
print(chosen["variant"] + ", chunked head")
m_rev, t_rev = X.max_batch(chosen["variant"], chosen["h"], start=64, step=16, extra=ex)
old = json.loads((X.RESULTS / "max_batch.json").read_text())
print(f"\\nmax batch with a chunked head: standard {m_std} (was {old['standard']['max_batch']}), "
      f"{chosen['variant']} {m_rev} (was {old[chosen['variant']]['max_batch']}); ratio {m_rev/m_std:.2f}x")
(X.RESULTS / "max_batch_chunked_head.json").write_text(json.dumps(
    {"standard": {"max_batch": m_std, "trail": t_std},
     chosen["variant"]: {"max_batch": m_rev, "trail": t_rev, "h": chosen["h"]}}, indent=2))
'''),
]

NB["05_report"] = [
md('''
# 05 · The three runs side by side

Reads what notebooks 02, 03 and 04 wrote to `results/runs/` and `results/max_batch.json`; trains nothing.
Writes `results/summary.json`, `results/summary.md` and the figures the README shows.
'''),
SETUP,
code('''
import math
import matplotlib.pyplot as plt
chosen = json.loads((X.RESULTS / "chosen_variant.json").read_text())
V = chosen["variant"]
RUNS = {"baseline (standard, b32)": "baseline_standard_b32",
        f"reversible {V} (b32)": f"reversible_{V}_b32",
        f"reversible {V} (max batch)": f"reversible_{V}_maxbatch"}
S = {k: X.load(v) for k, v in RUNS.items()}
L = {k: X.load_log(v) for k, v in RUNS.items()}
mb = json.loads((X.RESULTS / "max_batch.json").read_text())

audit_path = X.RESULTS / "memory_audit.json"
AUD = {(a["variant"], a["batch"], a["loss_chunk"]): max(t["peak_mib"] for t in a["steps"])
       for a in json.loads(audit_path.read_text())} if audit_path.exists() else {}
rows = []
for k, s in S.items():
    rv = (s["reversibility"] or {}).get("final") or {}
    rows.append({"run": k, "batch": s["batch"], "steps": s["steps"], "peak_lr": s["peak_lr"],
                 "final_train_loss": s["final_train_loss_avg_last5pct"], "final_val_loss": s["final_val_loss"],
                 "tok_per_s": s["tok_per_s"], "train_minutes": s["train_seconds"] / 60,
                 "tok_per_s_median_interval": sorted(r["tok_per_s"] for r in L[k] if r["type"] == "step" and r["tok_per_s"])[
                     len([r for r in L[k] if r["type"] == "step" and r["tok_per_s"]]) // 2],
                 "peak_exact_mib": AUD.get((s["config"]["variant"], s["batch"], 0), s["peak_exact_mib"]),
                 "peak_unsynced_audit_mib": s["peak_exact_mib"], "peak_driver_mib": s["peak_driver_mib"],
                 "peak_op": s["peak_exact_op"], "grad_cosine_vs_autograd": rv.get("grad_cosine")})
b = rows[0]
hdr = "| run | batch | steps | final train loss | final val loss | tokens/s | minutes | peak memory (synced audit) | Metal driver peak | vs baseline: speed | vs baseline: memory |"
lines = [hdr, "|" + "---|" * 11]
for r in rows:
    lines.append(f"| {r['run']} | {r['batch']} | {r['steps']:,} | {r['final_train_loss']:.4f} | {r['final_val_loss']:.4f} | "
                 f"{r['tok_per_s']:,.0f} | {r['train_minutes']:.1f} | {r['peak_exact_mib']/1024:.2f} GiB | {r['peak_driver_mib']/1024:.2f} GiB | "
                 f"{r['tok_per_s']/b['tok_per_s']:.2f}x | {r['peak_exact_mib']/b['peak_exact_mib']:.2f}x |")
table = "\\n".join(lines)
(X.RESULTS / "summary.json").write_text(json.dumps({"runs": rows, "max_batch": mb}, indent=2))
(X.RESULTS / "summary.md").write_text(table + "\\n")
from IPython.display import Markdown, display
display(Markdown(table))
'''),
md("## Loss against tokens, and against wall-clock time"),
code('''
def smooth(y, k=25):
    out, acc = [], []
    for v in y:
        acc.append(v); acc = acc[-k:]; out.append(sum(acc) / len(acc))
    return out
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
for k, log in L.items():
    st = [r for r in log if r["type"] == "step"]; ev = [r for r in log if r["type"] == "eval"]
    ax[0].plot([r["tokens"]/1e6 for r in st], smooth([r["loss"] for r in st]), label=k, lw=1.2)
    ax[1].plot([r["tokens"]/1e6 for r in ev], [r["val_loss"] for r in ev], "o-", ms=3, label=k)
    # wall clock of each eval: the train_time of the last step logged before it
    import numpy as np
    mins = np.interp([r["tokens"] for r in ev], [r["tokens"] for r in st], [r["train_time"] for r in st]) / 60
    ax[2].plot(mins, [r["val_loss"] for r in ev], "o-", ms=3, label=k)
ax[0].set(xlabel="tokens (M)", title="training loss (smoothed)", ylim=(3, 7))
ax[1].set(xlabel="tokens (M)", title="validation loss", ylim=(3, 7))
ax[2].set(xlabel="training minutes (evaluation excluded)", title="validation loss vs time", ylim=(3, 7))
for a in ax: a.grid(alpha=.3); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(X.RESULTS / "loss_curves.png", dpi=120); plt.show()
'''),
md('''
## Peak memory, measured so that it repeats

`python -m src.memory_audit` meters two training steps per configuration op by op, with the GPU
synchronised after every op (MPS keeps a freed buffer counted until the GPU has finished with it, so
unsynchronised readings drift by up to ~25% between identical runs). The two steps agree to within 0.3%.
'''),
code('''
aud = json.loads((X.RESULTS / "memory_audit.json").read_text())
fig, ax = plt.subplots(figsize=(10, 3.8))
names = [a["config"] for a in aud]; vals = [max(t["peak_mib"] for t in a["steps"]) / 1024 for a in aud]
bars = ax.barh(names[::-1], vals[::-1], color=["#888" if "standard" in n else "#2a7" for n in names[::-1]])
for b_, v in zip(bars, vals[::-1]):
    ax.text(v + 0.1, b_.get_y() + b_.get_height() / 2, f"{v:.2f} GiB", va="center", fontsize=9)
ax.set_xlabel("peak live tensor memory in one training step (GiB), synced op-level audit")
ax.set_xlim(0, 12.5); ax.grid(alpha=.3, axis="x")
plt.tight_layout(); plt.savefig(X.RESULTS / "memory_audit.png", dpi=120); plt.show()
for a in aud:
    print(f"{a['config']:30s}", [t["peak_mib"] for t in a["steps"]], a["steps"][0]["op"])
'''),
md('''
## Memory against batch size

Every probe from the max-batch search, for both stacks. Peak memory is close to a straight line in the
batch: an intercept (weights, gradients, AdamW state, allocator overhead) plus a slope, the memory one more
sequence of 512 tokens costs. The slope is where reversibility acts.
'''),
code('''
import numpy as np
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
fits = {}
series = dict(mb)
chunk_path = X.RESULTS / "max_batch_chunked_head.json"
if chunk_path.exists():
    series.update({f"{k} + chunked head": v for k, v in json.loads(chunk_path.read_text()).items()})
for name, d in series.items():
    ok = [t for t in d["trail"] if t["fits"]]
    bs = np.array([t["batch"] for t in ok]); pk = np.array([t["peak_exact_mib"] for t in ok])
    slope, icpt = np.polyfit(bs, pk, 1) if len(bs) > 1 else (float("nan"), float("nan"))
    fits[name] = {"slope_mib_per_seq": slope, "intercept_mib": icpt, "max_batch": d["max_batch"]}
    ax[0].plot(bs, pk / 1024, "o-", label=f"{name}: {slope:.1f} MiB / sequence, max batch {d['max_batch']}")
    ax[1].plot(bs, [t["tok_per_s"] for t in ok], "o-", label=name)
cap = torch.mps.recommended_max_memory() / 2**30 if dev == "mps" else None
if cap: ax[0].axhline(cap, color="k", ls="--", lw=1, label=f"MPS cap {cap:.2f} GiB")
ax[0].set(xlabel="batch (sequences of 512)", ylabel="unsynced audit peak (GiB)", title="peak memory vs batch (4-step probes)")
ax[1].set(xlabel="batch (sequences of 512)", ylabel="tokens/s (4-step probe)", title="throughput vs batch")
for a in ax: a.grid(alpha=.3); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(X.RESULTS / "memory_vs_batch.png", dpi=120); plt.show()
logits_mib = 512 * 8192 * 4 / 2**20
print(json.dumps(fits, indent=1))
print(f"one fp32 logits tensor for one sequence: {logits_mib:.0f} MiB (512 x 8192 x 4 bytes)")
json.dump(fits, open(X.RESULTS / "memory_fits.json", "w"), indent=2)
'''),
]


def build(only=None):
    for name, cells in NB.items():
        if only is not None and not any(name.startswith(o) for o in only):
            continue
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, ROOT / f"{name}.ipynb")
        print("wrote", name + ".ipynb")


def execute(prefix):
    from nbclient import NotebookClient
    path = next(ROOT.glob(f"{prefix}*.ipynb"))
    nb = nbf.read(path, as_version=4)
    t0 = time.time()
    NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.write(nb, path)
    print(f"executed {path.name} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", default=[])
    a = ap.parse_args()
    build(a.run or None)          # never overwrite an executed notebook that is not being re-run
    for p in a.run:
        execute(p)
