"""verify.py — independent reproduction, exactly what a grader does:

  1. load ONLY the shipped tokenizer.model (nothing from training),
  2. assert decode(encode(text)) == text  (the faithful-roundtrip gate),
  3. recompute every fertility + the self score and check they match stats.json.

Run:  python3 verify.py
"""

import json
import os

from tokenizer import Tokenizer, word_count

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DATA = os.path.join(HERE, "data")

tok = Tokenizer.load(os.path.join(OUT, "tokenizer.model"))
stats = json.load(open(os.path.join(OUT, "stats.json"), encoding="utf-8"))
LANGS = ["en", "hi", "te", "kn"]

print(f"loaded tokenizer.model: vocab_size={tok.vocab_size} (base 256 + {tok.vocab_size-256} merges)\n")

# ---- 1. faithful roundtrip gate ----
print("ROUNDTRIP GATE  decode(encode(text)) == text")
gate_ok = True
for s in stats["roundtrip_samples"] + ["India's population is 1,428,627,663."]:
    ok = tok.decode(tok.encode(s)) == s
    gate_ok &= ok
    print(f"  [{'OK' if ok else 'FAIL'}] {s[:48]!r}")
for l in LANGS:
    text = open(os.path.join(DATA, f"{l}.txt"), encoding="utf-8").read()
    ok = tok.decode(tok.encode(text)) == text
    gate_ok &= ok
    print(f"  [{'OK' if ok else 'FAIL'}] full {l}.txt page ({len(text)} chars)")
print(f"  => gate {'PASSED' if gate_ok else 'FAILED'}\n")

# ---- 2. recompute fertilities + score ----
print("FERTILITY  len(encode(page)) / len(page.split())")
fert = {}
match = True
for l in LANGS:
    text = open(os.path.join(DATA, f"{l}.txt"), encoding="utf-8").read()
    w = word_count(text)
    n = len(tok.encode(text))
    fert[l] = n / w
    claimed = stats["languages"][l]["fertility"]
    ok = abs(fert[l] - claimed) < 1e-9
    match &= ok
    print(f"  {l}: words={w:6} tokens={n:6} fertility={fert[l]:.4f}  claimed={claimed:.4f}  {'OK' if ok else 'MISMATCH'}")

order = sorted(LANGS, key=lambda l: fert[l])
spread = fert[order[-1]] - fert[order[0]]
score = 1000 / spread
print(f"\nXmin({order[0]})={fert[order[0]]:.4f}  Xmax({order[-1]})={fert[order[-1]]:.4f}  spread={spread:.4f}")
print(f"score = 1000/{spread:.4f} = {score:.1f}  (claimed {stats['score']:.1f})")
print(f"vocab <= 10000: {tok.vocab_size <= 10000}   English <= 1.2: {fert['en'] <= 1.2}")

good = gate_ok and match and abs(score - stats["score"]) < 1e-6 and tok.vocab_size <= 10000
print("\n" + ("REPRODUCED EXACTLY + ROUNDTRIP PASSES" if good else "PROBLEM"))
