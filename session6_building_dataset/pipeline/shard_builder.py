"""Stage 1 -- documents become immutable tokenized shards.

Each (split, lane) group is tokenized with the frozen tokenizer and written as a
flat uint16 token blob plus a manifest. A shard is sealed on write: the manifest
records SHA256 of the exact bytes, and every later read verifies it. Changing a
shard means writing a new shard with a new id, a new hash and a parent link --
never editing one in place.

Inside a shard, each document is cut into *samples* of at most `seq_len - 1`
tokens (one slot is kept for the EOS separator that packing inserts). A sample
is the unit that packing, OPUS and the consumption ledger all talk about, and it
is addressed by an absolute token span into its shard:

    sample_id = "<shard_id>:<token_start>-<token_end>"

That span is what makes replay possible: given the ledger, the manifests and the
shard bytes, the exact tokens of any historical batch can be read back.
"""

from __future__ import annotations

import os

import numpy as np

from .common import (ensure_dir, hash_obj, now_iso, sha256_hex, write_json)
from .manifest import SPLIT_PERMISSION, finalise, write_manifest

CLEANING_PIPELINE = {
    "name": "s6-admission-v1",
    "steps": ["hf_source_fetch", "min_max_char_filter", "exact_dedup_by_content_hash",
              "canary_tagging", "frozen_tokenizer_encode", "span_chunking"],
    "version": "1.0.0",
}
CLEANING_PIPELINE_HASH = hash_obj(CLEANING_PIPELINE)


def _encode_documents(tok, docs: list[dict], max_sample: int) -> tuple[np.ndarray, list, list]:
    """Tokenize a group of documents into one contiguous token blob."""
    texts = [d["text"] for d in docs]
    encoded = tok(texts, add_special_tokens=False)["input_ids"]

    tokens: list[int] = []
    samples: list[dict] = []
    doc_records: list[dict] = []
    for d, ids in zip(docs, encoded):
        doc_start = len(tokens)
        tokens.extend(int(i) for i in ids)
        doc_end = len(tokens)
        doc_records.append({
            "doc_id": d["doc_id"], "token_start": doc_start, "token_end": doc_end,
            "token_count": doc_end - doc_start, "content_hash": d["content_hash"],
            "char_len": d["char_len"],
        })
        # cut the document into packable spans
        for s in range(doc_start, doc_end, max_sample):
            e = min(s + max_sample, doc_end)
            if e - s < 8:            # a stub shorter than this is not worth a slot
                continue
            samples.append({"doc_id": d["doc_id"], "token_start": s, "token_end": e,
                            "n_tokens": e - s})
    return np.asarray(tokens, dtype=np.uint16), samples, doc_records


def build_shards(cfg, tok, tokenizer_hash: str, docs_by_split: dict[str, list[dict]],
                 log=None) -> list[dict]:
    """Write every shard and its manifest. Returns the manifests."""
    shards_root = cfg.resolve("shards_dir")
    manifests_dir = cfg.resolve("manifests_dir")
    ensure_dir(shards_root)
    ensure_dir(manifests_dir)
    per_shard = cfg.shards.docs_per_shard
    max_sample = cfg.packing.seq_len - 1
    dtype = cfg.shards.dtype

    manifests: list[dict] = []
    for split, docs in docs_by_split.items():
        lanes = sorted({d["lane"] for d in docs})
        for lane in lanes:
            lane_docs = [d for d in docs if d["lane"] == lane]
            for shard_ix, start in enumerate(range(0, len(lane_docs), per_shard)):
                group = lane_docs[start:start + per_shard]
                tokens, samples, doc_records = _encode_documents(tok, group, max_sample)
                shard_id = f"{split}-{lane}-{shard_ix:04d}"
                rel = os.path.join(split, f"{shard_id}.bin")
                path = os.path.join(shards_root, rel)
                ensure_dir(os.path.dirname(path))
                raw = tokens.astype(dtype).tobytes()
                with open(path, "wb") as fh:      # sealed on write
                    fh.write(raw)
                os.chmod(path, 0o444)             # immutable in the ordinary sense

                for s in samples:
                    s["sample_id"] = f"{shard_id}:{s['token_start']}-{s['token_end']}"
                    s["token_hash"] = sha256_hex(
                        tokens[s["token_start"]:s["token_end"]].astype(np.uint32).tobytes())

                manifest = {
                    "shard_id": shard_id,
                    "path": rel,
                    "split": split,
                    "lane": lane,
                    "permission": SPLIT_PERMISSION[split],
                    "never_train": split == "eval",
                    "dtype": dtype,
                    "num_documents": len(group),
                    "num_samples": len(samples),
                    "num_tokens": int(tokens.size),
                    "content_hash": sha256_hex(raw),
                    "tokenizer_hash": tokenizer_hash,
                    "tokenizer_id": cfg.tokenizer.hf_id,
                    "source": group[0]["source"],
                    "license": group[0]["license"],
                    "cleaning_pipeline": CLEANING_PIPELINE["name"],
                    "cleaning_pipeline_hash": CLEANING_PIPELINE_HASH,
                    "dedup_status": "exact_dedup_pass",
                    "contamination_status": "unscanned",
                    "parent_shard_ids": [],
                    "created_at": now_iso(),
                    "doc_ids": [d["doc_id"] for d in group],
                    "documents": doc_records,
                    "samples": samples,
                    "benchmark_ids": sorted({d["benchmark_id"] for d in group
                                             if "benchmark_id" in d}),
                    "canaries": sorted({d["canary"] for d in group if "canary" in d}),
                }
                manifests.append(finalise(manifest))

    if log:
        by_split: dict[str, dict] = {}
        for m in manifests:
            b = by_split.setdefault(m["split"], {"shards": 0, "tokens": 0, "samples": 0})
            b["shards"] += 1
            b["tokens"] += m["num_tokens"]
            b["samples"] += m["num_samples"]
        for split in sorted(by_split):
            log.ok("shards_created", split=split, **by_split[split])
    return manifests


def persist_manifests(cfg, manifests: list[dict], log=None) -> dict:
    """Write every manifest plus an index whose hash covers all of them."""
    manifests_dir = cfg.resolve("manifests_dir")
    for m in manifests:
        finalise(m)
        write_manifest(manifests_dir, m)
    index = {
        "created_at": now_iso(),
        "shard_count": len(manifests),
        "token_total": sum(m["num_tokens"] for m in manifests),
        "shards": [{"shard_id": m["shard_id"], "split": m["split"], "lane": m["lane"],
                    "num_tokens": m["num_tokens"], "num_samples": m["num_samples"],
                    "content_hash": m["content_hash"],
                    "manifest_hash": m["manifest_hash"],
                    "admitted": m.get("admitted"),
                    "permission": m["permission"]} for m in sorted(
                        manifests, key=lambda x: x["shard_id"])],
    }
    index["index_hash"] = hash_obj(index["shards"])
    write_json(os.path.join(manifests_dir, "shard_index.json"), index)
    if log:
        log.ok("manifests_written", shards=index["shard_count"],
               index_hash=index["index_hash"][:16])
    return index
