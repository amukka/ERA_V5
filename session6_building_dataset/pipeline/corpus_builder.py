"""Stage 0 -- documents.

Pulls three real Hugging Face corpora into `corpus/{train,validation,eval}/`
as JSONL documents carrying provenance, licence, split and a content hash.

    wiki    Salesforce/wikitext (wikitext-2-raw-v1)     encyclopedic prose
    dialog  pixelsandpointers/better_daily_dialog       DailyDialog, parquet
    code    code-search-net/code_search_net (python)    real Python functions

Why the DailyDialog mirror: the canonical `daily_dialog` repo ships a loading
*script*, and datasets>=3 refuses to execute dataset scripts, so `load_dataset
("daily_dialog")` raises. The mirror is the same corpus converted to parquet.

Three splits with three different permissions downstream:

    train       admitted into loss-bearing batches
    validation  readable during training for evaluation, never gradient-bearing
    eval        never readable by training at all (the firewall's job)

Every eval document is prefixed with a unique canary string. The builder also
plants exactly one leaked training document that quotes an eval document
verbatim. Nothing downstream is told which document that is; the contamination
scanner has to find it by computing over the token streams.

Documents are written once and reused on later runs (the corpus is committed to
the repository so the demonstration runs offline); pass --rebuild-corpus to
re-download.
"""

from __future__ import annotations

import os
import re
from typing import Iterable

from .common import (canonical_json, ensure_dir, hash_obj, read_json, sha256_hex,
                     stable_permutation, write_json, write_jsonl)

WIKI_HEADER = re.compile(r"^\s*=\s[^=].*\s=\s*$")


# --------------------------------------------------------------------------- #
# hugging face access
# --------------------------------------------------------------------------- #

def _rows(hf_id: str, hf_config: str | None, split: str, limit: int,
          streaming: bool = False) -> list[dict]:
    """Fetch up to `limit` rows, tolerating datasets that lack a given split."""
    from datasets import load_dataset
    kwargs = {"path": hf_id, "streaming": streaming}
    if hf_config:
        kwargs["name"] = hf_config
    try:
        if streaming:
            ds = load_dataset(split=split, **kwargs)
            return [r for _, r in zip(range(limit), ds)]
        ds = load_dataset(split=f"{split}[:{limit}]", **kwargs)
        return list(ds)
    except Exception:
        # fall back to the train split when validation/test are absent
        if split == "train":
            raise
        ds = load_dataset(split="train", **kwargs) if not streaming else \
            load_dataset(split="train", **kwargs)
        offset = {"validation": 100_000, "test": 200_000}.get(split, 0)
        out = []
        for i, r in enumerate(ds):
            if i < offset:
                continue
            out.append(r)
            if len(out) >= limit:
                break
        return out


# --------------------------------------------------------------------------- #
# per-lane document assembly
# --------------------------------------------------------------------------- #

def _wiki_docs(rows: Iterable[dict], min_chars: int, max_chars: int) -> list[str]:
    """Rebuild wikitext articles from line rows, then chunk on blank lines."""
    articles, current = [], []
    for r in rows:
        line = r.get("text", "")
        if WIKI_HEADER.match(line):
            if current:
                articles.append("".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        articles.append("".join(current))

    docs = []
    for art in articles:
        art = art.strip()
        if not art:
            continue
        buf = ""
        for para in art.split("\n \n"):
            para = para.strip()
            if not para:
                continue
            if len(buf) + len(para) > max_chars and len(buf) >= min_chars:
                docs.append(buf.strip())
                buf = ""
            buf += para + "\n\n"
        if len(buf.strip()) >= min_chars:
            docs.append(buf.strip())
    return docs


def _dialog_docs(rows: Iterable[dict], min_chars: int, max_chars: int) -> list[str]:
    """Group utterance rows back into whole conversations."""
    convos: dict[int, list[str]] = {}
    order: list[int] = []
    for r in rows:
        did = int(r.get("dialog_id", 0))
        if did not in convos:
            convos[did] = []
            order.append(did)
        utt = (r.get("utterance") or "").strip()
        if utt:
            speaker = "A" if len(convos[did]) % 2 == 0 else "B"
            convos[did].append(f"{speaker}: {utt}")
    docs = []
    for did in order:
        text = "\n".join(convos[did]).strip()
        if min_chars <= len(text):
            docs.append(text[:max_chars])
    return docs


def _code_docs(rows: Iterable[dict], min_chars: int, max_chars: int) -> list[str]:
    docs = []
    for r in rows:
        src = (r.get("whole_func_string") or r.get("func_code_string") or "").strip()
        if len(src) < min_chars:
            continue
        path = r.get("func_path_in_repository", "")
        repo = r.get("repository_name", "")
        header = f"# repo: {repo}\n# path: {path}\n"
        docs.append((header + src)[:max_chars])
    return docs


BUILDERS = {"wiki": _wiki_docs, "dialog": _dialog_docs, "code": _code_docs}
# rows needed per document differ wildly by lane (wikitext rows are single lines)
ROW_FACTOR = {"wiki": 60, "dialog": 9, "code": 3}
STREAMING = {"wiki": False, "dialog": False, "code": True}


def _make_doc(lane: str, split: str, idx: int, text: str, src: dict) -> dict:
    doc = {
        "doc_id": f"{lane}.{split}.{idx:05d}",
        "lane": lane,
        "split": split,
        "source": src["hf_id"] + (f":{src['hf_config']}" if src.get("hf_config") else ""),
        "license": src["license"],
        "text": text,
        "char_len": len(text),
    }
    if split == "eval":
        doc["benchmark_id"] = f"{lane}-eval-v1"
        doc["canary"] = "CANARY-S6-" + sha256_hex(doc["doc_id"])[:16].upper()
        doc["text"] = doc["canary"] + "\n" + text
        doc["char_len"] = len(doc["text"])
        doc["never_train"] = True
    doc["content_hash"] = sha256_hex(doc["text"])
    return doc


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def build_corpus(cfg, log, rebuild: bool = False) -> dict:
    corpus_dir = cfg.resolve("corpus_dir")
    manifest_path = os.path.join(corpus_dir, "corpus_manifest.json")
    spec_hash = hash_obj({"sources": dict(cfg.corpus.sources),
                          "min_chars": cfg.corpus.min_chars,
                          "max_chars": cfg.corpus.max_chars,
                          "plant_leak": cfg.corpus.plant_leak})

    if os.path.exists(manifest_path) and not rebuild:
        man = read_json(manifest_path)
        # The manifest is committed but the documents are not, so a fresh clone
        # has the description without the data. Reuse requires both.
        missing = [f"{split}/{lane}.jsonl"
                   for split in ("train", "validation", "eval")
                   for lane in cfg.corpus.sources
                   if not os.path.exists(os.path.join(corpus_dir, split, f"{lane}.jsonl"))]
        if man.get("spec_hash") == spec_hash and not missing:
            log.ok("corpus_reused", documents=man["document_count"],
                   chars=man["char_total"], spec_hash=spec_hash[:12])
            return man
        if missing:
            log.note("corpus_documents_absent", missing=len(missing),
                     example=missing[0], action="re-fetching from Hugging Face")
        else:
            log.note("corpus_spec_changed", old=man.get("spec_hash", "")[:12],
                     new=spec_hash[:12])

    min_c, max_c = cfg.corpus.min_chars, cfg.corpus.max_chars
    by_split: dict[str, list[dict]] = {"train": [], "validation": [], "eval": []}
    source_info = {}

    for lane, src in cfg.corpus.sources.items():
        src = dict(src)
        wanted = {"train": src["train_docs"], "validation": src["valid_docs"],
                  "eval": src["eval_docs"]}
        hf_split = {"train": "train", "validation": "validation", "eval": "test"}
        for split, n_docs in wanted.items():
            rows = _rows(src["hf_id"], src.get("hf_config"), hf_split[split],
                         n_docs * ROW_FACTOR[lane], streaming=STREAMING[lane])
            texts = BUILDERS[lane](rows, min_c, max_c)
            # deterministic subsample so a bigger fetch never changes the pick
            order = stable_permutation(len(texts), "corpus", lane, split, spec_hash)
            texts = [texts[i] for i in order[:n_docs]]
            for i, t in enumerate(texts):
                by_split[split].append(_make_doc(lane, split, i, t, src))
            log.info(f"{lane:<7} {split:<10} rows={len(rows):<6} docs={len(texts)}")
        source_info[lane] = {k: src[k] for k in ("hf_id", "hf_config", "license")}

    # ---- plant exactly one leaked training document ----------------------- #
    leak_doc_id = None
    if cfg.corpus.plant_leak:
        victim = by_split["eval"][0]
        host = by_split["train"][0]
        leaked = dict(host)
        leaked["doc_id"] = f"{host['lane']}.train.90000"
        leaked["text"] = host["text"] + "\n\n" + victim["text"]
        leaked["char_len"] = len(leaked["text"])
        leaked["content_hash"] = sha256_hex(leaked["text"])
        by_split["train"].append(leaked)
        leak_doc_id = leaked["doc_id"]

    # ---- exact dedup across the whole corpus ------------------------------ #
    seen: dict[str, str] = {}
    removed = []
    for split in ("train", "validation", "eval"):
        kept = []
        for d in by_split[split]:
            if d["content_hash"] in seen:
                removed.append({"doc_id": d["doc_id"], "duplicate_of": seen[d["content_hash"]]})
                continue
            seen[d["content_hash"]] = d["doc_id"]
            d["dedup_status"] = "exact_dedup_pass"
            kept.append(d)
        by_split[split] = kept

    counts = {}
    for split, docs in by_split.items():
        ensure_dir(os.path.join(corpus_dir, split))
        for lane in cfg.corpus.sources:
            lane_docs = sorted([d for d in docs if d["lane"] == lane],
                               key=lambda d: d["doc_id"])
            write_jsonl(os.path.join(corpus_dir, split, f"{lane}.jsonl"), lane_docs)
            counts[f"{split}/{lane}"] = len(lane_docs)

    all_docs = [d for s in ("train", "validation", "eval") for d in by_split[s]]
    manifest = {
        "corpus_version": "s6-corpus-v1",
        "spec_hash": spec_hash,
        "sources": source_info,
        "document_count": len(all_docs),
        "char_total": sum(d["char_len"] for d in all_docs),
        "counts": counts,
        "documents_hash": sha256_hex("".join(sorted(d["content_hash"] for d in all_docs))),
        "duplicates_removed": removed,
        "planted_leak_doc": leak_doc_id,
        "canary_count": sum(1 for d in all_docs if "canary" in d),
    }
    write_json(manifest_path, manifest)
    log.ok("corpus_built", documents=manifest["document_count"],
           chars=manifest["char_total"], lanes=len(source_info),
           dupes_removed=len(removed))
    log.note("corpus_leak_planted", doc=leak_doc_id or "none",
             canaries=manifest["canary_count"])
    return manifest


def load_split(cfg, split: str) -> list[dict]:
    from .common import read_jsonl
    corpus_dir = cfg.resolve("corpus_dir")
    docs = []
    for lane in cfg.corpus.sources:
        path = os.path.join(corpus_dir, split, f"{lane}.jsonl")
        if os.path.exists(path):
            docs.extend(read_jsonl(path))
    return sorted(docs, key=lambda d: d["doc_id"])
