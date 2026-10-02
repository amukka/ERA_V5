"""Build results/summary.md and the figures from results/runs/*.jsonl."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / "results"
load = lambda n: [json.loads(l) for l in open(R / "runs" / f"{n}.jsonl")]
ev = lambda r: [(x["tokens"], x["val_loss"]) for x in r if x["type"] == "eval"]
st = lambda r: [(x["tokens"], x["loss"]) for x in r if x["type"] == "step"]

dense, cont, copy, drop = (load(n) for n in ("1_dense", "3_dense_continued", "4_moe_copy", "5_moe_drop"))
D = 20_000_000
runs = [("dense, continued (control)", cont, "#555"), ("MoE, copy upcycle", copy, "#d1495b"),
        ("MoE, drop-upcycle r=0.5", drop, "#2a7fba")]

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
ax[0].plot(*zip(*st(dense)), color="#888", lw=1, label="train")
ax[0].plot(*zip(*ev(dense)), "o-", color="k", ms=3, label="val")
ax[0].set(title="Stage 1: dense GPT from scratch", xlabel="tokens", ylabel="loss", ylim=(4, 8))
ax[0].legend()
for n, r, c in runs:
    ax[1].plot(*zip(*ev(r)), "o-", color=c, ms=3, label=n)
    ax[2].plot(*zip(*st(r)), color=c, lw=0.8, alpha=.8, label=n)
ax[1].set(title="Stage 2: continued training, val loss", xlabel="new tokens", ylabel="val loss", ylim=(4.2, 5.1))
ax[2].set(title="Stage 2: train loss", xlabel="new tokens", ylim=(4.1, 4.9))
ax[1].legend(); ax[2].legend()
plt.tight_layout(); plt.savefig(R / "loss_curves.png", dpi=130)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a, (n, r, c) in zip(ax, runs[1:]):
    s = [x for x in r if x["type"] == "step" and "max_load" in x]
    for l in range(len(s[0]["max_load"])):
        a.plot([x["tokens"] for x in s], [x["max_load"][l] for x in s], lw=1, label=f"layer {l}")
    a.axhline(1 / 8, color="k", ls=":"); a.axhline(.5, color="k", ls="--", lw=.6)
    a.set(title=f"{n}: busiest expert's share", xlabel="new tokens", ylabel="share of routed slots", ylim=(0, .55))
ax[0].legend(ncol=2, fontsize=7)
plt.tight_layout(); plt.savefig(R / "expert_load.png", dpi=130)

rows = ["| run | total params | active params / token | val loss at start | val loss at end | tok/s |", "|---|---:|---:|---:|---:|---:|"]
for n in ("1_dense", "3_dense_continued", "4_moe_copy", "5_moe_drop"):
    s = json.loads((R / "runs" / f"{n}.json").read_text())
    rows.append(f"| {n} | {s['params']/1e6:.1f}M | {s['active_params']/1e6:.1f}M | {s['val_loss_start']:.3f} | "
                f"**{s['val_loss_final']:.3f}** | {s['tok_per_s']:,} |")
(R / "summary.md").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
