"""Build the committed corpus slice that session 10 trains on.

The full session-6 corpus (12 MB of wikitext-2 + code + dialog) is rebuilt from
Hugging Face by that session and is not tracked in git.  Session 10 needs a
*fixed* slice of it so a fresh clone reproduces the same numbers, so this script
takes a deterministic subset, small enough to commit gzipped, and writes it to
``data/corpus_slice.jsonl.gz``.

Run from anywhere:  python tools/build_slice.py
"""

import gzip
import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent.parent
SRC = HERE.parent / "session6_building_dataset" / "corpus"
OUT = HERE / "data" / "corpus_slice.jsonl.gz"

# chars per lane per split.  wiki dominates because it is the only lane with
# long documents; code and dialog are kept because their length distribution is
# very different from wiki's, and that difference is the whole point of the
# gradient-accumulation experiment.
BUDGET = {
    ("train", "wiki"): 8_600_000,
    ("train", "code"): 500_000,
    ("train", "dialog"): 320_000,
    ("validation", "wiki"): 690_000,
    ("validation", "code"): 40_000,
    ("validation", "dialog"): 25_000,
}


def main() -> None:
    if not SRC.exists():
        raise SystemExit(
            f"session-6 corpus not found at {SRC}.\n"
            "Rebuild it with:  cd ../session6_building_dataset && "
            "python run_demo.py --rebuild-corpus"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows, totals = [], {}
    for (split, lane), budget in BUDGET.items():
        path = SRC / split / f"{lane}.jsonl"
        used = 0
        with path.open() as fh:
            for line in fh:                      # file order == deterministic
                if used >= budget:
                    break
                doc = json.loads(line)
                text = doc["text"]
                if len(text) < 200:              # too short to make a sequence
                    continue
                rows.append(
                    {
                        "split": split,
                        "lane": lane,
                        "doc_id": doc["doc_id"],
                        "text": text,
                    }
                )
                used += len(text)
        totals[f"{split}/{lane}"] = used

    with gzip.open(OUT, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"wrote {OUT.relative_to(HERE)}  {OUT.stat().st_size/1e6:.2f} MB  "
          f"{len(rows)} docs")
    for k, v in totals.items():
        print(f"  {k:<20} {v:>9,} chars")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
