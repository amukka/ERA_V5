"""Download a FineWeb-Edu slice, train an 8,192-token BPE on it, write uint16 token files.

    python tools/prepare_data.py            # ~56M train tokens + ~1M validation tokens

Outputs (all in data/):
    tokenizer.json   byte-level BPE, vocab 8,192, <|eos|> = id 0
    train.bin        uint16 token ids, documents separated by <|eos|>
    val.bin          same, from documents the training file never sees
    meta.json        counts, source, and the chars-per-token ratio
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VOCAB = 8192
EOS = "<|eos|>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tokens", type=int, default=56_000_000)
    ap.add_argument("--val-tokens", type=int, default=1_000_000)
    ap.add_argument("--tokenizer-docs", type=int, default=60_000)
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    t0 = time.time()

    stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    it = iter(stream)

    # 1. tokenizer, trained on the first documents of the stream
    docs = [next(it)["text"] for _ in range(args.tokenizer_docs)]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=[EOS],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(docs, trainer=trainer)
    tok.save(str(DATA / "tokenizer.json"))
    eos_id = tok.token_to_id(EOS)
    print(f"tokenizer trained on {len(docs):,} docs in {time.time()-t0:.0f}s, eos={eos_id}")

    # 2. validation first (from the documents the tokenizer was trained on is fine; the model is not),
    #    then training, both as one long stream with <|eos|> between documents
    def fill(target, source):
        out, n, chars, ndocs = [], 0, 0, 0
        batch = []
        for d in source:
            batch.append(d)
            if len(batch) == 512:
                for e, t in zip(tok.encode_batch(batch), batch):
                    out.append(np.asarray(e.ids + [eos_id], dtype=np.uint16))
                    n += len(e.ids) + 1
                    chars += len(t)
                    ndocs += 1
                batch = []
                if n >= target:
                    break
        arr = np.concatenate(out)[:target]
        return arr, chars, ndocs

    val, vchars, vdocs = fill(args.val_tokens, iter(docs))
    rest = (row["text"] for row in it)
    train, tchars, tdocs = fill(args.train_tokens, rest)
    val.tofile(DATA / "val.bin")
    train.tofile(DATA / "train.bin")
    meta = {
        "source": "HuggingFaceFW/fineweb-edu, config sample-10BT, streamed in order",
        "vocab_size": VOCAB, "eos_id": eos_id,
        "train_tokens": int(train.size), "train_docs": tdocs,
        "val_tokens": int(val.size), "val_docs": vdocs,
        "chars_per_token": round(tchars / max(1, train.size), 3),
        "note": "val.bin comes from the tokenizer's training documents; train.bin from the documents after them",
        "seconds": round(time.time() - t0, 1),
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
