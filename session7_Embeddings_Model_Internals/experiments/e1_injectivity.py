"""E1 — Is the codec injective on a real vocabulary?

Nothing downstream matters if it is not. If two tokens produce the same code,
they produce the same embedding forever, and an output head built by inverting
the code can never separate them either. So before claiming the head can be
removed, we have to show the code is a faithful name for the token.

This runs the shipped codec and the duplex codec over real vocabularies and
counts collisions exactly, by script. It is the count Section 8 of the session
asks for, and it is also the precondition for everything in E2 and E3.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from duplex.codec import DUPLEX, SHIPPED, CodecConfig, KroneckerCodec
from duplex.vocab import INDIC, Vocab, load_vocab

RESULTS = Path(__file__).resolve().parent.parent / "results"


def audit(vocab: Vocab, cfg: CodecConfig) -> dict:
    codec = KroneckerCodec(cfg)

    groups: dict[bytes, list[int]] = defaultdict(list)
    truncated = 0
    byte_lens = []

    for i, seq in enumerate(vocab.byte_seqs):
        cols = codec.columns(seq)
        groups[cols.astype(np.int16).tobytes()].append(i)
        byte_lens.append(len(seq))
        if len(seq) > cfg.head_cols:
            truncated += 1

    colliding_groups = {k: v for k, v in groups.items() if len(v) > 1}
    colliding_tokens = sum(len(v) for v in colliding_groups.values())

    per_script_total = Counter(vocab.scripts)
    per_script_collide: Counter = Counter()
    per_script_trunc: Counter = Counter()
    for i, seq in enumerate(vocab.byte_seqs):
        if len(seq) > cfg.head_cols:
            per_script_trunc[vocab.scripts[i]] += 1
    for ids in colliding_groups.values():
        for i in ids:
            per_script_collide[vocab.scripts[i]] += 1

    examples = []
    for ids in list(colliding_groups.values())[:400]:
        toks = [vocab.tokens[i] for i in ids]
        if len(set(toks)) > 1:
            examples.append(
                {
                    "script": vocab.scripts[ids[0]],
                    "tokens": toks[:4],
                    "byte_lengths": [len(vocab.byte_seqs[i]) for i in ids[:4]],
                }
            )
        if len(examples) >= 12:
            break

    lens = np.array(byte_lens)
    scripts = {}
    for s, total in per_script_total.most_common():
        if total < 50:
            continue
        sel = lens[[i for i, x in enumerate(vocab.scripts) if x == s]]
        scripts[s] = {
            "tokens": total,
            "collided": per_script_collide.get(s, 0),
            "collision_rate": per_script_collide.get(s, 0) / total,
            "truncated": per_script_trunc.get(s, 0),
            "truncation_rate": per_script_trunc.get(s, 0) / total,
            "mean_bytes": float(sel.mean()),
            "p99_bytes": float(np.percentile(sel, 99)),
            "indic": s in INDIC,
        }

    indic_tokens = sum(v["tokens"] for v in scripts.values() if v["indic"])
    indic_collided = sum(v["collided"] for v in scripts.values() if v["indic"])

    return {
        "vocab": vocab.name,
        "vocab_size": len(vocab),
        "codec": cfg.label,
        "code_dim": cfg.code_dim,
        "distinct_codes": len(groups),
        "colliding_groups": len(colliding_groups),
        "colliding_tokens": colliding_tokens,
        "collision_rate": colliding_tokens / len(vocab),
        "injective": len(colliding_groups) == 0,
        "truncated_tokens": truncated,
        "indic_tokens": indic_tokens,
        "indic_collided": indic_collided,
        "indic_collision_rate": (indic_collided / indic_tokens) if indic_tokens else 0.0,
        "examples": examples,
        "scripts": scripts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--vocabs",
        nargs="+",
        default=["era-v5", "gpt2", "xlm-roberta-base"],
        help="`era-v5` is this course's own Session 2 tokenizer; the other two "
             "are external vocabularies used to show the effect is not specific "
             "to ours.",
    )
    args = ap.parse_args()

    configs = [
        ("shipped_p32", SHIPPED),
        ("shipped_p48", CodecConfig(char_dim=256, pos_dim=48)),
        ("shipped_p64", CodecConfig(char_dim=256, pos_dim=64)),
        ("duplex_p32_t4", DUPLEX),
    ]

    out = []
    for name in args.vocabs:
        print(f"\n=== {name} ===")
        vocab = load_vocab(name)
        for label, cfg in configs:
            rec = audit(vocab, cfg)
            rec["config_name"] = label
            out.append(rec)
            print(
                f"  {label:14s} distinct={rec['distinct_codes']:>7d}  "
                f"collided={rec['colliding_tokens']:>6d} "
                f"({rec['collision_rate']*100:.3f}%)  "
                f"indic={rec['indic_collided']:>5d}  "
                f"injective={rec['injective']}"
            )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {RESULTS / args.out}")


if __name__ == "__main__":
    main()
