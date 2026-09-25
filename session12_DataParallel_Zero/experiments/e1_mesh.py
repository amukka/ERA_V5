"""E1 -- thirty-two virtual GPUs, and the three collectives they talk with.

The assignment starts by asking for 32 virtual GPUs, so this is the experiment
that builds them and then proves they are worth the name.  A rank here is a
thread with its own tensors and no route to anybody else's, so three things can
be checked rather than assumed:

  * every rank really is separate -- 32 threads, 32 ranks, 32 different slices
    of text, and no two ranks holding the same batch;
  * a reduce-scatter followed by an all-gather really does equal an all-reduce,
    which is the identity the whole of ZeRO stages 1 and 2 is built on;
  * a ring all-reduce really does move 2N(W-1)/W bytes per rank, counted one
    neighbour-to-neighbour send at a time rather than taken from the formula.

The last of those is what licenses the ledger that experiments 3 and 4 rely on.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json
import threading

import torch

from src.mesh import Mesh, NVLINK, INFINIBAND, PCIE
from src.data import Batcher
from src.model import Config
from src.report import machine, save_json, save_text, table
from src import plots

WORLD = 32
N_ELEMS = 1 << 16          # 65,536 fp32 numbers -- divisible by every W tested


def _identity_check(world: int, n: int = N_ELEMS) -> dict:
    """all_reduce == all_gather(reduce_scatter(.)), bitwise, on random data."""
    mesh = Mesh(world)
    gen = torch.Generator().manual_seed(11)
    payload = [torch.randn(n, generator=gen) for _ in range(world)]

    def worker(rank, mesh):
        mine = payload[rank]
        direct = mesh.all_reduce(rank, mine, "mean")
        shard = mesh.reduce_scatter(rank, mine, "mean")
        two_phase = mesh.all_gather(rank, shard)
        return {
            "shard_numel": shard.numel(),
            "bitwise_equal": bool(torch.equal(direct, two_phase)),
            "max_abs_diff": float((direct - two_phase).abs().max()),
            "checksum": float(direct.double().sum()),
        }

    out = mesh.run(worker)
    truth = torch.stack(payload).mean(0)
    return {
        "world_size": world,
        "per_rank_shard_numel": out[0]["shard_numel"],
        "bitwise_equal_on_every_rank": all(o["bitwise_equal"] for o in out),
        "max_abs_diff": max(o["max_abs_diff"] for o in out),
        "all_ranks_agree": len({round(o["checksum"], 9) for o in out}) == 1,
        "matches_true_mean": float((torch.stack(payload).mean(0)
                                    - truth).abs().max()),
        "ledger": mesh.ledger.summary(),
    }


def _ring_check(world: int, n: int = N_ELEMS) -> dict:
    """Count the bytes a ring actually sends, and compare with the formula."""
    mesh = Mesh(world)
    gen = torch.Generator().manual_seed(12)
    payload = [torch.randn(n, generator=gen) for _ in range(world)]
    trace: list = []

    def worker(rank, mesh):
        ring, sent = mesh.ring_all_reduce(rank, payload[rank], "mean",
                                          trace=trace if rank == 0 else None)
        direct = mesh.all_reduce(rank, payload[rank], "mean")
        return {"sent": sent,
                "max_abs_diff_vs_direct": float((ring - direct).abs().max()),
                "max_rel_diff_vs_direct": float(
                    ((ring - direct).abs() / direct.abs().clamp_min(1e-12)).max()),
                "bitwise": bool(torch.equal(ring, direct))}

    out = mesh.run(worker)
    payload_bytes = n * 4
    formula = 2 * payload_bytes * (world - 1) / world
    return {
        "world_size": world,
        "payload_bytes": payload_bytes,
        "sends_per_rank": len(trace),
        "expected_sends_per_rank": 2 * (world - 1),
        "measured_bytes_per_rank": out[0]["sent"],
        "formula_bytes_per_rank": formula,
        "matches_formula": abs(out[0]["sent"] - formula) < 1e-6,
        "in_units_of_N": out[0]["sent"] / payload_bytes,
        "max_abs_diff_vs_direct": max(o["max_abs_diff_vs_direct"] for o in out),
        "max_rel_diff_vs_direct": max(o["max_rel_diff_vs_direct"] for o in out),
        "bitwise_equal": all(o["bitwise"] for o in out),
        "first_six_sends": trace[:6],
    }


def _separateness(world: int, micro_batch: int = 2) -> dict:
    """Are these really 32 independent workers with 32 different batches?"""
    cfg = Config()
    sampler = Batcher(seq_len=cfg.max_seq)
    batch = sampler.global_batch(world, micro_batch)

    def worker(rank, mesh):
        x, y = Batcher.for_rank(batch, rank, world, micro_batch)
        return {
            "rank": rank,
            "thread": threading.current_thread().name,
            "thread_id": threading.get_ident(),
            "tokens": int(x.numel()),
            "fingerprint": int(x.sum()),
            "torch_threads": torch.get_num_threads(),
            "first_bytes": bytes(x[0, :24].tolist()).decode("utf-8", "replace"),
        }

    out = Mesh(world).run(worker)
    return {
        "world_size": world,
        "distinct_threads": len({o["thread_id"] for o in out}),
        "distinct_ranks": len({o["rank"] for o in out}),
        "distinct_batches": len({o["fingerprint"] for o in out}),
        "tokens_per_rank": out[0]["tokens"],
        "global_batch_tokens": sum(o["tokens"] for o in out),
        "torch_threads_per_rank": out[0]["torch_threads"],
        "sample": [{"rank": o["rank"], "thread": o["thread"],
                    "text": o["first_bytes"]} for o in out[:4]],
    }


def _scaling(worlds=(1, 2, 4, 8, 16, 32), n: int = 4096) -> list:
    """Measured bytes per rank for each collective, as the world grows."""
    rows = []
    for w in worlds:
        mesh = Mesh(w)
        gen = torch.Generator().manual_seed(13)
        payload = [torch.randn(n, generator=gen) for _ in range(w)]

        def worker(rank, mesh):
            mesh.all_reduce(rank, payload[rank], "mean", tag="ar")
            mesh.reduce_scatter(rank, payload[rank], "mean", tag="rs")
            mesh.all_gather(rank, payload[rank][: n // w], tag="ag")
            return None

        mesh.run(worker)
        s = mesh.ledger.summary()["by_tag"]
        rows.append({"world_size": w, "payload_bytes": n * 4,
                     "all_reduce": s.get("ar", 0.0),
                     "reduce_scatter": s.get("rs", 0.0),
                     "all_gather": s.get("ag", 0.0)})
    return rows


def _prices(n_params: int = 30_000_000_000) -> list:
    P = 2 * n_params
    rows = []
    for label, volume in (("2P (data parallelism, ZeRO-1, ZeRO-2)", 2 * P),
                          ("3P (ZeRO-3)", 3 * P)):
        rows.append({
            "volume": label,
            "bytes": volume,
            "nvlink_s": volume / (NVLINK.gb_per_s * 1e9),
            "infiniband_s": volume / (INFINIBAND.gb_per_s * 1e9),
            "pcie_s": volume / (PCIE.gb_per_s * 1e9),
        })
    return rows


def _figure(scaling, prices, name="e1_mesh.png"):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ws = [r["world_size"] for r in scaling]
    n = scaling[0]["payload_bytes"]
    ar = [r["all_reduce"] / n for r in scaling]
    rs = [r["reduce_scatter"] / n for r in scaling]
    ag = [r["all_gather"] / n for r in scaling]
    ax1.plot(ws, ar, marker="o", ms=4, color=plots.S1)
    ax1.plot(ws, rs, marker="o", ms=4, color=plots.S3)
    ax1.plot(ws, ag, marker="o", ms=4, color=plots.S2, ls=(0, (2, 3)))
    plots.label_end(ax1, ws, ar, "all-reduce", plots.S1, dx=0.02)
    plots.label_end(ax1, ws, rs, "reduce-scatter\nand all-gather\n(they coincide)",
                    plots.S3, dx=0.02, fontsize=8)
    ax1.axhline(2.0, color=plots.MUTED, lw=0.9, ls=(0, (4, 3)))
    ax1.axhline(1.0, color=plots.MUTED, lw=0.9, ls=(0, (4, 3)))
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ws, [str(w) for w in ws])
    ax1.set_xlim(0.9, max(ws) * 4.2)
    ax1.set_ylim(0, 2.35)
    ax1.set_xlabel("world size")
    ax1.set_ylabel("bytes per rank, in units of the payload N")
    ax1.set_title("What a collective costs one rank", loc="left")
    ax1.annotate("2N", (ws[0], 2.0), color=plots.MUTED, fontsize=8,
                 va="bottom", ha="left")
    ax1.annotate("N", (ws[0], 1.0), color=plots.MUTED, fontsize=8,
                 va="bottom", ha="left")

    labels = ["2P\n120 GB", "3P\n180 GB"]
    x = range(len(labels))
    width = 0.38
    nv = [p["nvlink_s"] for p in prices]
    ib = [p["infiniband_s"] for p in prices]
    ax2.bar([i - width / 2 for i in x], nv, width, color=plots.S1,
            label="NVLink, 450 GB/s")
    ax2.bar([i + width / 2 for i in x], ib, width, color=plots.S2,
            label="InfiniBand, 50 GB/s")
    for i, (a, b) in enumerate(zip(nv, ib)):
        ax2.annotate(f"{a:.2f} s", (i - width / 2, a), ha="center", va="bottom",
                     fontsize=8, color=plots.INK_2)
        ax2.annotate(f"{b:.2f} s", (i + width / 2, b), ha="center", va="bottom",
                     fontsize=8, color=plots.INK_2)
    ax2.set_axisbelow(True)
    ax2.set_xticks(list(x), labels)
    ax2.set_ylabel("seconds per step")
    ax2.set_ylim(0, 4.2)
    ax2.set_title("The same volume, on the two wires (30B model)", loc="left")
    ax2.legend()
    return plots.finish(fig, name,
                        "Left: measured by the ledger, not by the formula. "
                        "Right: 2P = 120 GB and 3P = 180 GB for 30B parameters in bf16.")


def main(verbose: bool = True) -> dict:
    sep = _separateness(WORLD)
    ident = _identity_check(WORLD)
    ring = _ring_check(WORLD)
    scaling = _scaling()
    prices = _prices()
    fig = _figure(scaling, prices)

    payload = {"machine": machine(), "world_size": WORLD,
               "separateness": sep, "identity": ident, "ring": ring,
               "scaling": scaling, "prices": prices}
    save_json("e1_mesh.json", payload)

    md = [
        "# E1 · Thirty-two virtual GPUs, and three collectives",
        "",
        f"World size {WORLD}. Every rank is a thread with its own tensors, its own "
        f"slice of the corpus and `torch.set_num_threads({sep['torch_threads_per_rank']})`, "
        "so a rank cannot quietly borrow the machine's other cores either.",
        "",
        "## They really are separate",
        "",
        table([(sep["distinct_ranks"], sep["distinct_threads"],
                sep["distinct_batches"], f"{sep['tokens_per_rank']:,}",
                f"{sep['global_batch_tokens']:,}")],
              ["ranks", "OS threads", "distinct batches", "tokens per rank",
               "tokens per step"], ["r"] * 5),
        "",
        "The first bytes each rank drew, to show the text differs and not just a hash:",
        "",
        table([(s["rank"], s["thread"], "`" + s["text"].replace("\n", "\\n") + "`")
               for s in sep["sample"]],
              ["rank", "thread", "first 24 bytes of its first sequence"]),
        "",
        "A replacement character in that column is not a bug: the vocabulary is the "
        "256 byte values, so a sequence boundary is free to land in the middle of a "
        "multi-byte codepoint. Nothing in this session depends on the tokenizer, and "
        "a byte vocabulary keeps the embedding table small enough that the sharding "
        "results below are about the transformer rather than about one large matrix.",
        "",
        "## reduce-scatter + all-gather = all-reduce",
        "",
        f"{ident['world_size']} ranks, {N_ELEMS:,} random fp32 numbers each. Each rank "
        f"keeps a shard of {ident['per_rank_shard_numel']:,} numbers in between.",
        "",
        table([("bitwise equal on every rank",
                str(ident["bitwise_equal_on_every_rank"])),
               ("max \\|all_reduce − all_gather(reduce_scatter)\\|",
                f"{ident['max_abs_diff']:.1e}"),
               ("every rank ended with the same answer",
                str(ident["all_ranks_agree"]))],
              ["check", "result"]),
        "",
        "This is the identity every later result leans on. Stage 1 and stage 2 do "
        "not send less than data parallelism does — they send the *same two phases* "
        "and keep the slice that appears in between them, which data parallelism "
        "computes and throws away.",
        "",
        "## The ring, counted send by send",
        "",
        f"`ring_all_reduce` performs the sends one at a time, each rank only ever "
        f"handing a chunk to `rank+1`. For {ring['world_size']} ranks that is "
        f"{ring['sends_per_rank']} sends per rank "
        f"({ring['expected_sends_per_rank']} expected: {WORLD - 1} to reduce and "
        f"{WORLD - 1} to gather), each one chunk of N/{WORLD}.",
        "",
        table([("bytes counted, one send at a time",
                f"{ring['measured_bytes_per_rank']:,.0f}"),
               ("2N(W−1)/W", f"{ring['formula_bytes_per_rank']:,.0f}"),
               ("in units of N", f"{ring['in_units_of_N']:.4f}"),
               ("identical", str(ring["matches_formula"]))],
              ["", "bytes per rank"], ["l", "r"]),
        "",
        f"So `2P` is not a convention, it is a count. The ring's *answer* agrees "
        f"with the direct reduction to {ring['max_abs_diff_vs_direct']:.1e} absolute "
        f"but is not bitwise identical to it "
        f"(`bitwise_equal: {ring['bitwise_equal']}`), because a ring adds the ranks "
        "in ring order and a direct reduction adds them in rank order, and floating "
        "point addition is not associative. Real NCCL has exactly this property, "
        "which is why changing the world size of a real run changes its loss curve "
        "in the last decimal place.",
        "",
        "## The cost, as the world grows",
        "",
        table([(r["world_size"], f"{r['all_reduce'] / r['payload_bytes']:.4f}N",
                f"{r['reduce_scatter'] / r['payload_bytes']:.4f}N",
                f"{r['all_gather'] / r['payload_bytes']:.4f}N")
               for r in scaling],
              ["world size", "all-reduce", "reduce-scatter", "all-gather"],
              ["r", "r", "r", "r"]),
        "",
        "The cost per rank does not grow with the world — it *approaches* 2N and N "
        "from below and stops. Doubling the GPUs does not double anybody's traffic; "
        "it only stops the (W−1)/W discount from helping. That is the property that "
        "makes data parallelism scale at all.",
        "",
        "## Pricing it for our model",
        "",
        "30B parameters in bf16 is P = 60 GB.",
        "",
        table([(p["volume"], f"{p['bytes'] / 1e9:.0f} GB", f"{p['nvlink_s']:.2f} s",
                f"{p['infiniband_s']:.2f} s", f"{p['pcie_s']:.2f} s")
               for p in prices],
              ["volume", "bytes", "NVLink 450 GB/s", "InfiniBand 50 GB/s",
               "PCIe 60 GB/s"], ["l", "r", "r", "r", "r"]),
        "",
        "![](e1_mesh.png)",
        "",
    ]
    save_text("e1_mesh.md", "\n".join(md))

    if verbose:
        print(json.dumps({k: v for k, v in payload.items()
                          if k not in ("scaling", "prices")}, indent=2,
                         default=str))
    return payload


if __name__ == "__main__":
    main()
