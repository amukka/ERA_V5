"""Measure what the cohort's own Session 4 shard is actually worth, in tokens.

Session 4 reports words. A mixture is denominated in tokens, and for Telugu the
two differ by ~2x. This measures the shard with our OWN Session 2 tokenizer
(byte-level BPE, 10k vocab) rather than assuming a words->tokens constant, then
extrapolates from a sampled fertility to the whole shard.

Writes reports/s4_shard_tokens.json.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
CORPUS = os.path.join(REPO, "session4_data_cleaning_dedup", "cleaned_corpus.jsonl")
TOKDIR = os.path.join(REPO, "session2_tokenizer", "build", "out")
SAMPLE_DOCS = int(os.environ.get("SAMPLE_DOCS", "600"))
SEED = 42


def load_tokenizer():
    spec = importlib.util.spec_from_file_location("s2tok", os.path.join(TOKDIR, "tokenizer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Tokenizer.load(os.path.join(TOKDIR, "tokenizer.model"))


def main() -> int:
    if not os.path.exists(CORPUS):
        print(f"corpus not found: {CORPUS}", file=sys.stderr)
        return 1
    tok = load_tokenizer()

    per_lang: dict[str, dict[str, float]] = {}
    rng = random.Random(SEED)
    reservoir: dict[str, list[str]] = {}
    seen: dict[str, int] = {}

    with open(CORPUS, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            lang = d.get("language", "unk")
            text = d.get("text", "")
            st = per_lang.setdefault(lang, {"docs": 0, "words": 0, "chars": 0})
            st["docs"] += 1
            st["words"] += len(text.split())
            st["chars"] += len(text)
            # reservoir sample per language for the fertility measurement
            seen[lang] = seen.get(lang, 0) + 1
            res = reservoir.setdefault(lang, [])
            if len(res) < SAMPLE_DOCS:
                res.append(text)
            else:
                j = rng.randrange(seen[lang])
                if j < SAMPLE_DOCS:
                    res[j] = text

    t0 = time.time()
    out_langs = {}
    total_tokens = 0.0
    for lang, st in sorted(per_lang.items(), key=lambda kv: -kv[1]["words"]):
        sample = reservoir[lang]
        s_words = sum(len(t.split()) for t in sample)
        s_tokens = sum(len(tok.encode(t)) for t in sample)
        fertility = s_tokens / max(s_words, 1)
        est_tokens = st["words"] * fertility
        total_tokens += est_tokens
        out_langs[lang] = {
            "docs": int(st["docs"]),
            "words": int(st["words"]),
            "chars": int(st["chars"]),
            "sampled_docs": len(sample),
            "sampled_words": s_words,
            "sampled_tokens": s_tokens,
            "measured_fertility_tokens_per_word": round(fertility, 4),
            "estimated_tokens": int(est_tokens),
        }

    out = {
        "corpus": os.path.relpath(CORPUS, REPO),
        "tokenizer": "ERA V5 Session 2 byte-level BPE, vocab 10000",
        "sample_docs_per_language": SAMPLE_DOCS,
        "seed": SEED,
        "languages": out_langs,
        "total_docs": int(sum(v["docs"] for v in per_lang.values())),
        "total_words": int(sum(v["words"] for v in per_lang.values())),
        "total_estimated_tokens": int(total_tokens),
        "total_estimated_tokens_b": round(total_tokens / 1e9, 6),
        "measure_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    dest = os.path.join(ROOT, "reports", "s4_shard_tokens.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "languages"}, indent=2))
    for lang, v in out_langs.items():
        print(f"  {lang:4s} docs={v['docs']:>6} words={v['words']:>10} "
              f"fertility={v['measured_fertility_tokens_per_word']:.3f} tokens~{v['estimated_tokens']:,}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
