"""E4 — What does the vocabulary actually cost, and does a million-token one work?

The assignment's claim is that a reversible input path lets the output head go,
and that "then we can have a vocab of 1M as well without any issues". E3 shows
the head can go. This experiment is about the second half of the sentence, and
it has two parts because the sentence makes two different kinds of claim.

Part one is arithmetic. Token-facing parameters and their AdamW training state
are computed for every arm across a sweep of vocabulary sizes at V5's reference
width. Nothing is measured here; it is the accounting from Section 3 of the
session, applied to each arm. The point it makes is a slope, not a number: the
dense arms are straight lines through the origin and the duplex arms are flat.

Part two is not arithmetic. A million-token vocabulary is built out of real
words harvested from the Session 4 cleaned corpus — which is Telugu-dominated,
so this is the hostile case rather than the flattering one — and then actually
instantiated. The codec is audited for collisions on it, a real DuplexEmbedding
is constructed over it, its parameters are counted, and a forward pass and a
full-vocabulary scoring pass are run and timed. If any of that had failed, the
claim would be arithmetic that does not survive contact with an allocator.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from duplex.codec import DUPLEX, SHIPPED, KroneckerCodec
from duplex.embedding import DuplexEmbedding
from duplex.vocab import INDIC, script_of

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CORPUS = ROOT.parent / "session4_data_cleaning_dedup" / "cleaned_corpus.jsonl"

BYTES_PER_PARAM = 16  # bf16 weight + bf16 grad + fp32 master + 2 fp32 AdamW moments


def arm_params(arm: str, V: int, D: int, N: int) -> int:
    """Token-facing parameters for each arm. Only two of these mention V."""
    if arm == "dense":
        return 2 * V * D
    if arm == "dense_tied":
        return V * D
    if arm == "kron_dense":
        return N * D + V * D
    if arm == "duplex":
        return 2 * N * D
    if arm == "duplex_tied":
        return N * D
    raise ValueError(arm)


ARMS = ["dense", "dense_tied", "kron_dense", "duplex", "duplex_tied"]
VOCAB_FREE = {"duplex", "duplex_tied"}


def sweep(vocabs, D: int, N: int, card_gb: float) -> list[dict]:
    rows = []
    for V in vocabs:
        rec = {"vocab_size": V, "arms": {}}
        for arm in ARMS:
            p = arm_params(arm, V, D, N)
            gb = p * BYTES_PER_PARAM / 1024 ** 3
            rec["arms"][arm] = {
                "params": p,
                "train_gb": gb,
                "card_share": gb / card_gb,
                "vs_dense": p / arm_params("dense", V, D, N),
                "vocab_free": arm in VOCAB_FREE,
            }
        rows.append(rec)
    return rows


# ---------------------------------------------------------------- the 1M vocab


def harvest(target: int, seed: int = 0) -> list[bytes]:
    """`target` distinct real words from the Session 4 cleaned corpus.

    Real words, not synthetic strings: the collision question is about what
    actual text looks like, and Telugu words with many conjuncts are exactly the
    case the 32-byte window was never sized for.
    """
    if not CORPUS.exists():
        raise SystemExit(f"missing corpus {CORPUS}")
    seen: set[str] = set()
    for line in CORPUS.open():
        seen.update(json.loads(line).get("text", "").split())
        if len(seen) >= target * 2:
            break
    words = sorted(seen)
    if len(words) < target:
        raise SystemExit(f"only {len(words):,} distinct words, need {target:,}")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(words), size=target, replace=False)
    return [words[i].encode("utf-8") for i in sorted(pick)]


def collisions(byte_seqs: list[bytes], cfg) -> dict:
    """Exact collision count, by script, at this vocabulary scale."""
    codec = KroneckerCodec(cfg)
    groups: dict[bytes, list[int]] = defaultdict(list)
    for i, seq in enumerate(byte_seqs):
        groups[codec.columns(seq).astype(np.int16).tobytes()].append(i)

    colliding = [ids for ids in groups.values() if len(ids) > 1]
    n_collided = sum(len(ids) for ids in colliding)

    per_script: Counter = Counter()
    examples = []
    for ids in colliding:
        for i in ids:
            per_script[script_of(byte_seqs[i].decode("utf-8", "replace"))] += 1
        if len(examples) < 8 and len(ids) > 1:
            texts = [byte_seqs[i].decode("utf-8", "replace") for i in ids[:3]]
            if len(set(texts)) > 1:
                examples.append({
                    "tokens": texts,
                    "byte_lengths": [len(byte_seqs[i]) for i in ids[:3]],
                })

    indic = sum(v for k, v in per_script.items() if k in INDIC)
    return {
        "codec": cfg.label,
        "distinct_codes": len(groups),
        "colliding_groups": len(colliding),
        "colliding_tokens": n_collided,
        "collision_rate": n_collided / len(byte_seqs),
        "indic_collided": indic,
        "injective": not colliding,
        "per_script": dict(per_script.most_common(10)),
        "examples": examples,
    }


@torch.no_grad()
def instantiate(byte_seqs: list[bytes], d_model: int, device: str,
                queries: int, chunk: int, seed: int = 0) -> dict:
    """Build the module over the million-token vocabulary and run it for real.

    The identification check is the one that carries weight, so it is done the
    way E2 does it rather than at random initialisation. The analysis matrix is
    random and untrained; the synthesis matrix is its pseudo-inverse, which is
    the minimum-norm linear map from a d-dimensional embedding back to code
    space. Neither holds a per-token parameter. The question asked is then: out
    of a million real words, does the right one score highest?

    Random W_ana is deliberately the pessimistic case. A trained one is at
    least as good, because training can only shape the map toward the codes it
    actually has to separate.
    """
    V = len(byte_seqs)
    torch.manual_seed(seed)
    t0 = time.time()
    emb = DuplexEmbedding(byte_seqs, d_model, DUPLEX, tied=False).to(device)
    build_s = time.time() - t0

    report = emb.parameter_report()
    trainable = sum(p.numel() for p in emb.parameters())
    buffers = sum(b.numel() * b.element_size() for b in emb.buffers()
                  if not b.is_sparse)
    buffers += sum(b._nnz() * (b.values().element_size() + 2 * 8)
                   for b in emb.buffers() if b.is_sparse)

    # W_syn <- pinv(W_ana): the algebraic inverse of the analysis map.
    emb.W_syn.copy_(torch.linalg.pinv(emb.W_ana.float()))

    ids = torch.randint(0, V, (4, 32), device=device, generator=None)
    t0 = time.time()
    h = emb(ids)
    fwd_s = time.time() - t0
    t0 = time.time()
    logits = emb.logits(h)
    score_s = time.time() - t0

    # Identification over the whole vocabulary, in query chunks so the [q, V]
    # score matrix stays a sane size.
    q_ids = torch.randperm(V, device=device)[:queries]
    hits, col_hits, exact = 0, 0.0, 0
    P, C = DUPLEX.pos_dim, DUPLEX.char_dim
    for s in range(0, len(q_ids), chunk):
        block = q_ids[s : s + chunk]
        x = emb(block.unsqueeze(0)).squeeze(0)
        hits += int((emb.logits(x).argmax(-1) == block).sum())
        pred = emb.code_logits(x).view(-1, P, C).argmax(-1)
        true = emb.code_targets(block)
        col_hits += float((pred == true).float().mean()) * len(block)
        exact += int((pred == true).all(dim=1).sum())

    return {
        "vocab_size": V,
        "d_model": d_model,
        "device": device,
        "build_seconds": build_s,
        "trainable_params": trainable,
        "param_report": report,
        "fixed_buffer_bytes": buffers,
        "forward_shape": list(h.shape),
        "logits_shape": list(logits.shape),
        "forward_seconds": fwd_s,
        "score_seconds": score_s,
        "identify_queries": len(q_ids),
        "identify_acc": hits / len(q_ids),
        "column_acc": col_hits / len(q_ids),
        "exact_code_acc": exact / len(q_ids),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=8096, help="V5 reference width")
    ap.add_argument("--card-gb", type=float, default=80.0)
    ap.add_argument("--vocabs", type=int, nargs="+",
                    default=[10_000, 32_000, 50_257, 131_072, 250_002, 1_000_000])
    ap.add_argument("--million", type=int, default=1_000_000)
    ap.add_argument("--live-d-model", type=int, default=512,
                    help="width for the executed 1M demo; params scale linearly in D")
    ap.add_argument("--id-queries", type=int, default=512)
    ap.add_argument("--id-chunk", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="e4_scaling.json")
    args = ap.parse_args()

    N = DUPLEX.code_dim
    print(f"code_dim={N}  D={args.d_model}  card={args.card_gb} GB\n")

    rows = sweep(args.vocabs, args.d_model, N, args.card_gb)
    header = f"{'V':>10}" + "".join(f"{a:>14}" for a in ARMS)
    print(header)
    for r in rows:
        line = f"{r['vocab_size']:>10,}"
        for a in ARMS:
            line += f"{r['arms'][a]['train_gb']:>13.1f}G"
        print(line)
    print("\n(training-state GB at 16 bytes/param; the last two columns are constant)")

    print(f"\n=== executed: a real {args.million:,}-token vocabulary ===")
    words = harvest(args.million)
    lens = np.array([len(w) for w in words])
    scripts = Counter(script_of(w.decode("utf-8", "replace"))
                      for w in words[:: max(1, len(words) // 20000)])
    print(f"harvested {len(words):,} distinct real words  "
          f"mean {lens.mean():.1f} bytes, p99 {np.percentile(lens, 99):.0f}, "
          f"max {lens.max()}")
    print(f"script mix (sampled): {scripts.most_common(6)}")

    audits = []
    for cfg in (SHIPPED, DUPLEX):
        a = collisions(words, cfg)
        audits.append(a)
        print(f"  {a['codec']:<22} collided={a['colliding_tokens']:>8,} "
              f"({a['collision_rate']*100:6.3f}%)  indic={a['indic_collided']:>8,}  "
              f"injective={a['injective']}")

    live = instantiate(words, args.live_d_model, args.device,
                       args.id_queries, args.id_chunk)
    print(f"\n  built in {live['build_seconds']:.1f}s   "
          f"trainable params {live['trainable_params']:,} "
          f"(= 2 x {N} x {args.live_d_model} + 1, no V)")
    print(f"  fixed buffers {live['fixed_buffer_bytes']/1024**3:.2f} GB   "
          f"forward {live['forward_seconds']*1000:.0f} ms   "
          f"score all {len(words):,} {live['score_seconds']*1000:.0f} ms")
    print(f"  from a {args.live_d_model}-d embedding, over {len(words):,} words: "
          f"column {live['column_acc']*100:.2f}%  "
          f"exact code {live['exact_code_acc']*100:.2f}%  "
          f"correct token {live['identify_acc']*100:.2f}%  "
          f"(n={live['identify_queries']}, chance {100/len(words):.5f}%)")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / args.out).write_text(json.dumps({
        "d_model": args.d_model,
        "code_dim": N,
        "card_gb": args.card_gb,
        "bytes_per_param": BYTES_PER_PARAM,
        "sweep": rows,
        "million": {
            "words": len(words),
            "mean_bytes": float(lens.mean()),
            "p99_bytes": float(np.percentile(lens, 99)),
            "max_bytes": int(lens.max()),
            "over_head_window": int((lens > DUPLEX.head_cols).sum()),
            "script_sample": dict(scripts.most_common(12)),
            "audits": audits,
            "live": live,
        },
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {RESULTS / args.out}")


if __name__ == "__main__":
    main()
