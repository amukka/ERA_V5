"""Build lane-tagged token shards for the proxy run.

The proxy run needs the same LANE STRUCTURE as the real mixture, not the same
scale. Lanes here are filled from:

  web      real English documents from our Session 4 cleaned corpus
  indic    real Telugu documents from the same corpus (Sangraha verified/tel)
  code     real Python source from the local environment
  math     generated worked arithmetic solutions (verifiable answers)
  reasoning generated chain-of-thought at four length bands (R1..R4)
  agentic  generated multi-step tool-use trajectories WITH LOSS MASKS

The generated lanes are toys, and the report says so. Their job is to exercise
the machinery that matters — loss masking on tool observations, reasoning-length
bands, per-lane held-out evaluation — at a scale that fits on a laptop.

Writes data/proxy/<lane>.bin (uint16 token ids), <lane>.mask.bin (uint8, 1 =
token receives loss) and meta.json. Nothing here is committed to git.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
OUT = os.path.join(ROOT, "data", "proxy")
CORPUS = os.path.join(REPO, "session4_data_cleaning_dedup", "cleaned_corpus.jsonl")
TOKDIR = os.path.join(REPO, "session2_tokenizer", "build", "out")

TARGET_TOKENS = {          # per lane, upper bound
    "web": 9_000_000,
    "indic": 22_000_000,
    "code": 6_000_000,
    "math": 3_000_000,
    "reasoning": 3_000_000,
    "agentic": 3_000_000,
}
HOLDOUT_FRAC = 0.02
SEED = 1234


def load_tokenizer():
    spec = importlib.util.spec_from_file_location("s2tok", os.path.join(TOKDIR, "tokenizer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Tokenizer.load(os.path.join(TOKDIR, "tokenizer.model"))


# --------------------------------------------------------------------------- #
# generated lanes
# --------------------------------------------------------------------------- #
TOOLS = ["search_grants", "fetch_record", "list_labs", "get_equipment_budget", "web_search"]
TOPICS = ["cryo-electron microscopy", "mass spectrometry", "confocal imaging",
          "sequencing throughput", "spectroscopy", "flow cytometry"]
INSTS = ["Iowa State", "UT Austin", "Rice", "Purdue", "Emory", "Caltech"]


def make_agentic(rng: random.Random) -> tuple[str, list[tuple[str, bool]]]:
    """Return (text, [(segment, trained)]) for one trajectory.

    trained=False marks tool observations and the user turn: the model may read
    them, and must never be trained to produce them. Training on an observation
    teaches the model to invent tool results instead of calling the tool.
    """
    topic = rng.choice(TOPICS)
    inst = rng.choice(INSTS)
    n_ok = rng.randint(1, 3)
    segs: list[tuple[str, bool]] = []
    segs.append((f"<|user|> Find US grants funding {topic} and say which labs could buy "
                 f"a {rng.randint(200, 900)}k instrument for it.\n", False))
    segs.append((f"<|plan|> I will search grants for {topic}, then pull the awardee records, "
                 f"then check equipment budgets.\n", True))
    for i in range(n_ok):
        tool = rng.choice(TOOLS)
        segs.append((f"<|call|> {tool}({{\"query\": \"{topic}\", \"page\": {i+1}}})\n", True))
        segs.append((f"<|obs|> {{\"results\": [{{\"id\": \"G{rng.randint(10000,99999)}\", "
                     f"\"inst\": \"{inst}\", \"amount\": {rng.randint(200,4000)}000}}]}}\n", False))
    # a failed call and a recovery: the whole point of trajectory data
    segs.append((f"<|call|> fetch_record({{\"id\": \"G{rng.randint(10000,99999)}\"}})\n", True))
    segs.append(("<|obs|> {\"error\": \"record not found\", \"status\": 404}\n", False))
    segs.append(("<|plan|> That record is missing. I will fall back to the awardee index "
                 "and match on institution instead.\n", True))
    segs.append((f"<|call|> list_labs({{\"inst\": \"{inst}\"}})\n", True))
    segs.append((f"<|obs|> {{\"labs\": [\"{inst} Core Facility\", \"{inst} Structural Biology\"]}}\n", False))
    segs.append((f"<|answer|> {inst} Core Facility holds a {topic} award large enough to fund the "
                 f"instrument; {inst} Structural Biology does not.\n", True))
    return "".join(s for s, _ in segs), segs


def make_reasoning(rng: random.Random) -> tuple[str, list[tuple[str, bool]]]:
    """Worked solution at one of four length bands. The answer is checkable."""
    band = rng.choices(["R1", "R2", "R3", "R4"], weights=[0.40, 0.30, 0.20, 0.10])[0]
    a, b, c = rng.randint(2, 40), rng.randint(2, 40), rng.randint(2, 12)
    ans = (a + b) * c
    q = f"<|user|> A depot receives {a} crates on Monday and {b} on Tuesday. Each crate holds {c} units. How many units?\n"
    if band == "R1":
        body = f"<|think|> ({a}+{b})*{c} = {ans}.\n"
    elif band == "R2":
        body = (f"<|think|> Total crates = {a} + {b} = {a+b}. Units per crate = {c}. "
                f"So units = {a+b} * {c} = {ans}.\n")
    elif band == "R3":
        body = (f"<|think|> First the crates: {a} + {b} = {a+b}. Then units: {a+b} * {c}. "
                f"Break it up: {a+b} * {c} = {a+b} * {c//2 if c>1 else 1} * 2 approximately, "
                f"so check directly: {a+b} * {c} = {ans}. Sanity check by the other order: "
                f"{a}*{c} = {a*c} and {b}*{c} = {b*c}, and {a*c} + {b*c} = {ans}. Consistent.\n")
    else:
        body = (f"<|think|> Restate: two deliveries, {a} and {b} crates, {c} units each. "
                f"Attempt one: add crates first, {a} + {b} = {a+b}, times {c} gives {ans}. "
                f"Attempt two: distribute, {a}*{c} = {a*c}, {b}*{c} = {b*c}, sum {ans}. "
                f"The two routes agree, so the multiplication is not where an error would hide. "
                f"Consider whether 'each crate holds {c} units' could mean something else - it "
                f"could mean capacity rather than contents, but the question asks how many units "
                f"the depot received, and no partial-fill information is given, so full crates is "
                f"the reading. Check magnitude: {a+b} crates at {c} units is about {(a+b)*c}. "
                f"Final answer {ans}.\n")
    tail = f"<|answer|> {ans}\n"
    return q + body + tail, [(q, False), (body, True), (tail, True)]


def make_math(rng: random.Random) -> tuple[str, list[tuple[str, bool]]]:
    a, b = rng.randint(11, 999), rng.randint(11, 99)
    q, r = divmod(a, b)
    t = (f"Compute {a} divided by {b}.\nEstimate: {b} * {q} = {b*q}, remainder {a} - {b*q} = {r}.\n"
         f"So {a} = {b} * {q} + {r}, i.e. quotient {q} remainder {r}.\n\n")
    return t, [(t, True)]


# --------------------------------------------------------------------------- #
def encode_segments(tok, segments: list[tuple[str, bool]]) -> tuple[list[int], list[int]]:
    ids: list[int] = []
    mask: list[int] = []
    for text, trained in segments:
        e = tok.encode(text)
        ids.extend(e)
        mask.extend([1 if trained else 0] * len(e))
    return ids, mask


def write_lane(name: str, ids: list[int], mask: list[int], meta: dict) -> None:
    arr = np.array(ids, dtype=np.uint16)
    msk = np.array(mask, dtype=np.uint8)
    n_hold = int(len(arr) * HOLDOUT_FRAC)
    n_hold = max(n_hold, 4096)
    train_ids, hold_ids = arr[:-n_hold], arr[-n_hold:]
    train_msk, hold_msk = msk[:-n_hold], msk[-n_hold:]
    train_ids.tofile(os.path.join(OUT, f"{name}.bin"))
    train_msk.tofile(os.path.join(OUT, f"{name}.mask.bin"))
    hold_ids.tofile(os.path.join(OUT, f"{name}.holdout.bin"))
    hold_msk.tofile(os.path.join(OUT, f"{name}.holdout.mask.bin"))
    meta[name] = {
        "train_tokens": int(train_ids.size),
        "holdout_tokens": int(hold_ids.size),
        "trained_token_frac": float(train_msk.mean()),
    }
    print(f"  {name:10s} train={train_ids.size:>10,} holdout={hold_ids.size:>7,} "
          f"loss-bearing={train_msk.mean():.1%}")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(CORPUS):
        print(f"missing {CORPUS}", file=sys.stderr)
        return 1
    tok = load_tokenizer()
    rng = random.Random(SEED)
    meta: dict = {}

    print("real lanes from the Session 4 corpus:")
    buf = {"web": ([], []), "indic": ([], [])}
    with open(CORPUS, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            lane = "indic" if d.get("language") == "te" else ("web" if d.get("language") == "en" else None)
            if lane is None:
                continue
            if len(buf[lane][0]) >= TARGET_TOKENS[lane]:
                if all(len(buf[l][0]) >= TARGET_TOKENS[l] for l in buf):
                    break
                continue
            e = tok.encode(d["text"] + "\n\n")
            buf[lane][0].extend(e)
            buf[lane][1].extend([1] * len(e))
    for lane in ("web", "indic"):
        write_lane(lane, *buf[lane], meta=meta)

    print("code lane from local Python sources:")
    code_ids, code_mask = [], []
    roots = [os.path.join(ROOT, ".venv", "lib"), ROOT]
    files: list[str] = []
    for r in roots:
        files.extend(glob.glob(os.path.join(r, "**", "*.py"), recursive=True))
    rng.shuffle(files)
    for f in files:
        if len(code_ids) >= TARGET_TOKENS["code"]:
            break
        try:
            with open(f, encoding="utf-8") as fh:
                src = fh.read()
        except Exception:
            continue
        if len(src) < 200 or len(src) > 60000:
            continue
        e = tok.encode(src + "\n\n")
        code_ids.extend(e)
        code_mask.extend([1] * len(e))
    write_lane("code", code_ids, code_mask, meta)

    print("generated lanes:")
    for name, maker in (("math", make_math), ("reasoning", make_reasoning), ("agentic", make_agentic)):
        ids: list[int] = []
        mask: list[int] = []
        while len(ids) < TARGET_TOKENS[name]:
            _text, segs = maker(rng)
            i, m = encode_segments(tok, segs)
            ids.extend(i)
            mask.extend(m)
        write_lane(name, ids, mask, meta)

    # one readable example of each generated lane, for the write-up
    samples = {}
    for name, maker in (("reasoning", make_reasoning), ("agentic", make_agentic)):
        text, segs = maker(random.Random(7))
        samples[name] = {"text": text,
                         "segments": [{"trained": t, "text": s} for s, t in segs]}
    doc = {"lanes": meta, "tokenizer": "ERA V5 S2 byte-level BPE (10k)",
           "seed": SEED, "samples": samples}
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    # data/ is gitignored, so the loss-mask evidence is also written under reports/,
    # where a reviewer can read it without rebuilding the shards.
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "proxy_data_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print(f"wrote {OUT}/meta.json and reports/proxy_data_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
