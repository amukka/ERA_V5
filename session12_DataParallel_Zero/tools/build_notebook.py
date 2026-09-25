"""Assemble session12.ipynb from the experiment modules.

The notebook is not a second copy of the code.  Every cell either shows a short
piece of arithmetic that is worth watching happen -- 32 threads being started,
an all-reduce being taken apart into its two halves -- or calls the same
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
OUT = ROOT / "session12.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


CELLS = [
md("""
# Session 12 — Data Parallel and ZeRO

**Thirty-two virtual GPUs, a demo model that runs on top of them, and ZeRO
stages 1, 2 and 3 simulated to the byte.**

A virtual GPU here is a Python thread that owns a rank, its own slice of the
corpus and its own tensors, and that can reach another rank's tensors *only* by
calling a collective. That restriction is what makes the numbers below
measurements rather than assertions: every byte that crosses between ranks goes
through `all_reduce`, `reduce_scatter` or `all_gather` and is counted on the way
past, and every tensor an engine allocates is registered with a meter.

| # | the question | where it is answered |
|---|---|---|
| 1 | are these really 32 separate workers, and what do the collectives cost? | `experiments/e1_mesh.py` |
| 2 | do the four arrangements compute the same thing? | `experiments/e2_equivalence.py` |
| 3 | what does each stage cost in memory, wire and clock? | `experiments/e3_stages.py` |
| 4 | where is the memory wall, and which arrangements never clear it? | `experiments/e4_ladder.py` |
| 5 | how much of the traffic can be hidden behind the compute? | `experiments/e5_overlap.py` |

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

def show(name):
    "Render an experiment's written-out findings inline."
    display(Markdown((ROOT / "results" / name).read_text()))

print(json.dumps(machine(), indent=2))
"""),

md("""
---
# 1. Thirty-two virtual GPUs

`Mesh(32)` starts 32 threads, gives each one a rank and a barrier, and pins each
to a single torch thread so a rank cannot quietly borrow the machine's other
cores. Nothing is shared except through a collective.
"""),
code("""
import threading
from src.mesh import Mesh

def hello(rank, mesh):
    return (rank, threading.current_thread().name, torch.get_num_threads())

mesh = Mesh(32)
out = mesh.run(hello)
print(f"{len(out)} ranks, {len({o[1] for o in out})} distinct threads, "
      f"{out[0][2]} torch thread each")
print("first four:", out[:4])
"""),

md("""
### The collective identity the whole session rests on

A reduce-scatter followed by an all-gather **is** an all-reduce. Stages 1 and 2
do not send less than data parallelism does — they send the same two phases and
*keep* the slice that appears in between, which data parallelism computes and
throws away. Here it is on four ranks and eight numbers, small enough to read.
"""),
code("""
mesh = Mesh(4)
payload = [torch.tensor([float(r * 10 + i) for i in range(8)]) for r in range(4)]
for r, p in enumerate(payload):
    print(f"rank {r} starts with {p.tolist()}")

def both_ways(rank, mesh):
    direct = mesh.all_reduce(rank, payload[rank], "mean")
    shard  = mesh.reduce_scatter(rank, payload[rank], "mean")
    whole  = mesh.all_gather(rank, shard)
    return direct, shard, whole

res = mesh.run(both_ways)
print()
for r, (direct, shard, whole) in enumerate(res):
    print(f"rank {r}: keeps shard {shard.tolist()}  ->  gathers {whole.tolist()}")
print()
print("all_reduce           ", res[0][0].tolist())
print("gather(scatter(.))   ", res[0][2].tolist())
print("bitwise equal:", all(torch.equal(d, w) for d, _, w in res))
"""),

md("""
### And the ring, counted one send at a time

`ring_all_reduce` does the sends one at a time, each rank only ever handing a
chunk to `rank+1`: `W-1` sends to reduce and `W-1` to gather, each of `N/W`
bytes. That is where `2P` comes from — it is a count, not a convention.
"""),
code("""
import experiments.e1_mesh as e1
mesh_stats = e1.main(verbose=False)
ring = mesh_stats["ring"]
print(f"world size {ring['world_size']}: {ring['sends_per_rank']} sends per rank")
print(f"counted:  {ring['measured_bytes_per_rank']:>10,.0f} bytes per rank")
print(f"2N(W-1)/W:{ring['formula_bytes_per_rank']:>10,.0f} bytes per rank")
print(f"identical: {ring['matches_formula']}   = {ring['in_units_of_N']:.4f}N")
show("e1_mesh.md")
display(Image(filename=str(ROOT / "results" / "e1_mesh.png")))
"""),

md("""
---
# 2. The demo model, and the sixteen bytes

An ordinary pre-norm decoder, but held in an unusual way: a group's parameters
are one flat tensor and `layer_forward` is a *function* that takes it. A model
whose weights live inside its modules cannot express "I do not have this layer
right now"; ZeRO-3 needs one that can.

The four engines differ in exactly one thing — which of the sixteen bytes per
weight a rank keeps and which it asks for when the moment comes.
"""),
code("""
from src.model import Config, total_params, group_numel
cfg = Config()
print(f"{cfg.n_layer} blocks · d_model {cfg.d_model} · {cfg.n_head} heads · "
      f"d_ff {cfg.d_ff} · context {cfg.max_seq} · vocab {cfg.vocab_size}")
print(f"{total_params(cfg):,} parameters in {cfg.n_groups} groups")
for g in range(cfg.n_groups):
    what = ("embeddings" if g == 0 else "norm + head" if g == cfg.n_groups - 1
            else f"block {g}")
    print(f"  group {g}: {group_numel(cfg, g):>9,}  {what}")
"""),
code("""
from src.engines import ENGINES, STAGE_ORDER
from src.mesh import Mesh

W = 8
rows = []
for name in STAGE_ORDER:
    def build(rank, mesh, name=name):
        e = ENGINES[name](rank, mesh, cfg)
        b = e.meter.breakdown()
        n = total_params(cfg)
        return (name, e.stage, b["weights"]/n, b["gradients"]/n,
                b["optimizer"]/n, e.bytes_per_weight())
    rows.append(Mesh(W).run(build)[0])

display(Markdown(
    "| arrangement | stage | bf16 weights | bf16 grads | fp32 master + moments | "
    "total bytes/weight |\\n|---|---|---|---|---|---|\\n" +
    "\\n".join(f"| {a} | {b} | {c:.3f} | {d:.3f} | {e:.3f} | **{f:.3f}** |"
              for a, b, c, d, e, f in rows)))
print(f"world size {W}; the session notes give 16.00, 5.50, 3.75, 2.00")
"""),

md("""
---
# 3. Four arrangements, one answer

Run them. Same data, same order, same optimizer. The memory per rank falls by
32x, the traffic rises for stage 3, and the weights do not change — not to eight
decimals, but in every bit.
"""),
code("""
from src.sim import train, max_weight_divergence

runs = {name: train(name, world_size=8, micro_batch=2, steps=6, cfg=cfg,
                    keep_weights=True)
        for name in STAGE_ORDER}
base = runs["data parallel"].weights

print(f"{'arrangement':<15}{'bytes/wt':>10}{'wire/step':>12}"
      f"{'final loss':>14}{'max |dw| vs DP':>17}")
for name, r in runs.items():
    d = max_weight_divergence(base, r.weights)
    print(f"{name:<15}{r.bytes_per_weight:>10.3f}{r.wire_in_P:>11.4f}P"
          f"{r.losses[-1]:>14.8f}{d:>17.1e}")
print()
print("bitwise identical to data parallelism:",
      all(all(torch.equal(base[g], r.weights[g]) for g in base)
          for r in runs.values()))
"""),
code("""
import experiments.e2_equivalence as e2
eq = e2.main(verbose=False)            # ~1 minute: it trains about 30 models
g = eq["big_batch"]["gradients"]
print("32 ranks x 2 sequences vs 1 rank x 64 sequences, one step:")
print(f"  loss differs by          {eq['big_batch']['loss_abs_diff']:.1e}")
print(f"  gradient, averaged fp32  {g['fp32']['relative_l2']:.1e} relative, "
      f"{g['fp32']['angle_degrees']:.4f} deg")
print(f"  gradient, bf16 per rank  {g['bf16']['relative_l2']:.1e} relative, "
      f"{g['bf16']['angle_degrees']:.4f} deg")
print(f"\\naveraging deleted: the 32 copies drift "
      f"{eq['control']['rank_spread']:.2e} apart and nothing raises an error")
show("e2_equivalence.md")
display(Image(filename=str(ROOT / "results" / "e2_equivalence.png")))
"""),

md("""
---
# 4. What each stage costs

The meter is told about every tensor as it is allocated, so the bytes-per-weight
column is an audit of what the engine did. The formula column is the session
notes' arithmetic. They are computed from opposite ends and never see each other.
"""),
code("""
import experiments.e3_stages as e3
stages = e3.main(verbose=False)
print(f"memory audit matches the formula in every cell: "
      f"{stages['memory_matches_formula']}")
print(f"wire count matches the formula in every cell:   "
      f"{stages['wire_matches_formula']}")
print()
for r in stages["rows"]:
    v = stages["v5_projection"][r["engine"]]
    print(f"{r['engine']:<15}{r['bytes_per_weight_measured']:>8.3f} B/wt"
          f"{r['wire_in_P_measured']:>9.4f}P"
          f"{v['state_gib']:>10,.1f} GiB at 30B  "
          f"{'fits' if v['fits_80gb_card'] else 'does not fit':>14}")
show("e3_stages.md")
display(Image(filename=str(ROOT / "results" / "e3_stages.png")))
"""),

md("""
---
# 5. The memory wall

Walk the world size and watch what each arrangement does with the extra GPUs.
Two of them get cheaper per rank without limit. Two of them stop — and where
they stop has a model size attached to it.
"""),
code("""
import experiments.e4_ladder as e4
ladder = e4.main(verbose=False)
b = ladder["boundary"]
print(f"measured bytes per weight matches the formula in all "
      f"{len(ladder['measured'])} cells: {ladder['measured_matches_formula']}")
print()
print(f"weights + gradients stay replicated under DP and ZeRO-1: 4 bytes/weight")
print(f"4 bytes fills a {b['card_gib']:.1f} GiB card at "
      f"{b['params_that_fill_a_card']/1e9:.1f}B parameters")
print(f"our model is {b['v5_params']/1e9:.0f}B, needing {b['v5_needs_gib']:.1f} GiB "
      f"-- at any world size")
print()
for row in ladder["ladder"]:
    fits = f"from {row['fits_from']} GPUs" if row["fits_from"] else "never"
    print(f"  {row['engine']:<15} fits an 80 GB card {fits}")
show("e4_ladder.md")
display(Image(filename=str(ROOT / "results" / "e4_ladder.png")))
"""),

md("""
---
# 6. Overlapping the communication with the compute

The thread mesh cannot answer the timing question — its wire is a memcpy between
two Python lists — so the timing question is answered with a discrete-event
model instead: measured per-layer shape, counted volume, stated bandwidth.
"""),
code("""
import experiments.e5_overlap as e5
ov = e5.main(verbose=False)
base_h = ov["baseline"]["data parallel | 64 x H100 | InfiniBand"]
base_b = ov["baseline"]["data parallel | 64 x B200 | InfiniBand"]
print(f"2P over InfiniBand, no overlap:")
print(f"  64 x H100: {base_h['wire_seconds']:.2f} s of wire against "
      f"{base_h['compute_seconds']:.2f} s of compute = "
      f"{100*base_h['ratio']:.0f}%")
print(f"  64 x B200: {base_b['wire_seconds']:.2f} s of wire against "
      f"{base_b['compute_seconds']:.2f} s of compute = "
      f"{100*base_b['ratio']:.0f}%  <- same bytes, faster card")
print()
print("with overlap, step time above its own compute:")
for k, v in ov["steps"].items():
    if "InfiniBand" in k:
        print(f"  {k:<40}{v['overhead_pct']:>+7.1f}%   "
              f"{100*v['hidden_fraction']:.0f}% hidden")
show("e5_overlap.md")
display(Image(filename=str(ROOT / "results" / "e5_overlap.png")))
"""),

md("""
---
# What the 32 GPUs said

One table, one line each, every number from the cells above.
"""),
code("""
dp, z1, z2, z3 = stages["rows"]
v5 = stages["v5_projection"]
b = ladder["boundary"]
ib_b = ov["steps"]["data parallel | 64 x B200 | InfiniBand"]
z3_b = ov["steps"]["ZeRO-3 | 64 x B200 | InfiniBand"]

rows = [
    ("1", "32 virtual GPUs, 32 threads, 32 different batches",
     f"{mesh_stats['separateness']['global_batch_tokens']:,} tokens a step, "
     f"{mesh_stats['separateness']['tokens_per_rank']} per rank"),
    ("1", "a ring all-reduce, counted send by send",
     f"{ring['in_units_of_N']:.4f}N per rank = 2N(W-1)/W, exactly"),
    ("2", "averaging 32 partial gradients vs one big batch",
     f"{eq['big_batch']['gradients']['fp32']['relative_l2']:.0e} relative -- "
     "the same run"),
    ("2", "bf16 on the wire, per rank, before averaging",
     f"{eq['big_batch']['gradients']['bf16']['relative_l2']:.0e} relative; "
     "Adam turns it into 2*lr on step 1"),
    ("2", "ZeRO-1, ZeRO-2 and ZeRO-3 against data parallelism",
     "bitwise identical, every weight, every step"),
    ("2", "the same code with the all-reduce deleted",
     f"the 32 copies drift {eq['control']['rank_spread']:.0e} apart, silently"),
    ("3", "bytes per weight, measured at W=32",
     f"{dp['bytes_per_weight_measured']:.3f} / "
     f"{z1['bytes_per_weight_measured']:.3f} / "
     f"{z2['bytes_per_weight_measured']:.3f} / "
     f"{z3['bytes_per_weight_measured']:.3f} -- the formula, in every cell"),
    ("3", "bytes on the wire per step",
     f"{dp['wire_in_P_measured']:.3f}P for stages 0-2, "
     f"{z3['wire_in_P_measured']:.3f}P for stage 3"),
    ("3", "AdamW on a shard vs on everything",
     f"{stages['adam_scaling'][-1]['speedup']:.0f}x at "
     f"{stages['adam_scaling'][-1]['n']//10**6}M parameters"),
    ("4", "the floor DP and ZeRO-1 cannot get under",
     f"4 bytes a weight, which fills a card at "
     f"{b['params_that_fill_a_card']/1e9:.0f}B parameters"),
    ("4", "30B on an 80 GB card",
     f"ZeRO-2 from 32 GPUs ({v5['ZeRO-2']['state_gib']:.1f} GiB), "
     f"ZeRO-3 from 8 ({v5['ZeRO-3']['state_gib']:.1f} GiB)"),
    ("5", "2P over InfiniBand, as a share of compute",
     f"{100*base_h['ratio']:.0f}% on H100, {100*base_b['ratio']:.0f}% on B200 -- "
     "same bytes"),
    ("5", "what overlap leaves exposed, B200 over InfiniBand",
     f"{ib_b['overhead_pct']:+.1f}% for data parallelism, "
     f"{z3_b['overhead_pct']:+.1f}% for ZeRO-3"),
]
display(Markdown(
    "| # | what was measured | what it said |\\n|---|---|---|\\n"
    + "\\n".join(f"| {a} | {b_} | {c} |" for a, b_, c in rows)))
"""),

md("""
ZeRO is a statement about addresses, not about arithmetic. Every stage in this
notebook computed the identical weights, bit for bit, out of a rank that held
between 16 and 0.5 bytes for each of them. What changed was who was holding
which byte at which moment, and how much of the fetching could be arranged to
happen while the machine was busy anyway.

The thing worth carrying out of this session is that both of the numbers that
decide a distributed run — what one rank holds, and what one rank sends — are
countable in advance, on a laptop, before a cluster is booked.
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
    print("executing (three to five minutes: it re-runs every experiment) ...")
    NotebookClient(nb, timeout=7200, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.write(nb, path)
    print(f"executed and saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    p = build()
    if "--run" in sys.argv:
        run(p)
