"""Stage 2 -- manifests, the admission gate and the evaluation firewall.

A manifest is the only thing the rest of the system is allowed to know about a
shard. It carries identity (content hash), meaning (tokenizer hash), lineage
(source, licence, cleaning status) and permission (split, never_train,
contamination status).

Three separate gates live here:

  verify_manifest    the bytes on disk still hash to what the manifest claims
  admit              licence / tokenizer / dedup / contamination / permission
  scan_contamination finds training shards that overlap evaluation content by
                     comparing 12-gram token hashes -- the leaked document the
                     corpus builder planted is found here, not declared

Permissions by split:

    train       trainable
    validation  readable for evaluation, never gradient-bearing
    eval        never readable by training  (never_train)
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np

from .common import (canonical_json, ensure_dir, hash_obj, now_iso, read_json,
                     sha256_hex, write_json)

NGRAM = 12                  # token n-gram width used for contamination matching
NGRAM_BASE = np.uint64(1000003)

# What counts as contamination.
#
# Counting matching n-grams does not work: a 100k-token shard of ordinary prose
# or code shares thousands of 12-grams with any other corpus in the same
# language -- section headers, licence boilerplate, `for i in range(len(`. A
# threshold low enough to catch a leak condemns the whole corpus.
#
# Verbatim reproduction has a different shape. A leaked document matches in one
# long *unbroken run* of windows, because the tokens really are the same tokens
# in the same order. Incidental overlap matches in short scattered bursts. So
# the test is on run length, per document, not on hit count per shard.
MIN_VERBATIM_RUN = 48       # consecutive matching windows == ~59 verbatim tokens
MIN_DOC_HIT_RATIO = 0.30    # or this much of one document matches at all
MIN_DOC_WINDOWS = 64        # documents shorter than this are not judged by ratio

SPLIT_PERMISSION = {
    "train": "trainable",
    "validation": "eval_read_only",
    "eval": "never_train",
}


class FirewallViolation(RuntimeError):
    """Raised when non-trainable data is offered to a loss-bearing batch."""


# --------------------------------------------------------------------------- #
# manifest construction and verification
# --------------------------------------------------------------------------- #

# Wall-clock fields are recorded in the manifest but excluded from its hash.
# Rebuilding the same shard from the same documents with the same tokenizer must
# produce the same manifest hash, or the shard is not content-addressed and
# "immutable object" means nothing. `checkpoint.py` excludes `created_at` from
# `checkpoint_id` for the same reason.
UNHASHED_FIELDS = ("manifest_hash", "created_at", "admission_decided_at")


def manifest_hash(manifest: dict) -> str:
    """Hash of the manifest's content and lineage, ignoring wall-clock fields."""
    return hash_obj({k: v for k, v in manifest.items() if k not in UNHASHED_FIELDS})


def finalise(manifest: dict) -> dict:
    manifest["manifest_hash"] = manifest_hash(manifest)
    return manifest


def write_manifest(manifests_dir: str, manifest: dict) -> str:
    path = os.path.join(ensure_dir(manifests_dir), f"{manifest['shard_id']}.json")
    write_json(path, manifest)
    return path


def load_manifests(manifests_dir: str) -> list[dict]:
    out = []
    for name in sorted(os.listdir(manifests_dir)):
        if name.endswith(".json") and name not in ("shard_index.json",
                                                   "mixture_plan.json",
                                                   "corpus_manifest.json"):
            out.append(read_json(os.path.join(manifests_dir, name)))
    return out


def verify_manifest(manifest: dict, shards_root: str) -> tuple[bool, dict]:
    """Recompute both hashes from bytes on disk."""
    path = os.path.join(shards_root, manifest["path"])
    detail = {"shard_id": manifest["shard_id"]}
    if not os.path.exists(path):
        detail["error"] = "missing_shard_file"
        return False, detail
    with open(path, "rb") as fh:
        raw = fh.read()
    content_ok = sha256_hex(raw) == manifest["content_hash"]
    size_ok = len(raw) == manifest["num_tokens"] * np.dtype(manifest["dtype"]).itemsize
    mh_ok = manifest_hash(manifest) == manifest["manifest_hash"]
    detail.update({"content_hash_ok": content_ok, "size_ok": size_ok,
                   "manifest_hash_ok": mh_ok})
    return bool(content_ok and size_ok and mh_ok), detail


def read_shard_tokens(manifest: dict, shards_root: str) -> np.ndarray:
    path = os.path.join(shards_root, manifest["path"])
    return np.fromfile(path, dtype=manifest["dtype"])


def read_span(manifest: dict, shards_root: str, start: int, end: int) -> np.ndarray:
    """Read one token span without loading the whole shard (replay uses this)."""
    itemsize = np.dtype(manifest["dtype"]).itemsize
    path = os.path.join(shards_root, manifest["path"])
    with open(path, "rb") as fh:
        fh.seek(start * itemsize)
        raw = fh.read((end - start) * itemsize)
    return np.frombuffer(raw, dtype=manifest["dtype"])


# --------------------------------------------------------------------------- #
# contamination scanning
# --------------------------------------------------------------------------- #

def _ngram_hashes(tokens: np.ndarray, n: int = NGRAM) -> np.ndarray:
    """Rolling polynomial hash of every length-n token window."""
    if len(tokens) < n:
        return np.zeros(0, dtype=np.uint64)
    t = tokens.astype(np.uint64)
    out = np.zeros(len(tokens) - n + 1, dtype=np.uint64)
    for k in range(n):
        out = out * NGRAM_BASE + t[k:len(tokens) - n + 1 + k]
    return out


def _longest_run(flags: np.ndarray) -> int:
    """Length of the longest unbroken run of True in a boolean array."""
    if flags.size == 0 or not flags.any():
        return 0
    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int((edges[1::2] - edges[0::2]).max())


def scan_contamination(train_manifests: list[dict], eval_manifests: list[dict],
                       shards_root: str, log=None) -> dict:
    """Flag training documents that reproduce evaluation content verbatim.

    Nothing here is told where the planted leak is. The leak is whatever shows
    up as an unbroken run of matching 12-gram windows -- the signature of the
    same tokens in the same order, which incidental phrase overlap does not
    produce.
    """
    eval_hashes = []
    for m in eval_manifests:
        eval_hashes.append(_ngram_hashes(read_shard_tokens(m, shards_root)))
    fingerprint = np.unique(np.concatenate(eval_hashes)) if eval_hashes else \
        np.zeros(0, dtype=np.uint64)

    report = {
        "ngram": NGRAM,
        "fingerprint_size": int(fingerprint.size),
        "rule": {"min_verbatim_run": MIN_VERBATIM_RUN,
                 "min_doc_hit_ratio": MIN_DOC_HIT_RATIO,
                 "min_doc_windows": MIN_DOC_WINDOWS},
        "shards": [], "contaminated": [], "contaminated_documents": [],
    }

    for m in train_manifests:
        toks = read_shard_tokens(m, shards_root)
        hits = np.isin(_ngram_hashes(toks), fingerprint, assume_unique=False)

        flagged, doc_rows = [], []
        for d in m["documents"]:
            lo = d["token_start"]
            hi = max(lo, d["token_end"] - NGRAM + 1)
            span = hits[lo:hi]
            windows = int(span.size)
            n_hits = int(span.sum())
            run = _longest_run(span)
            ratio = n_hits / max(1, windows)
            verbatim = run >= MIN_VERBATIM_RUN
            saturated = windows >= MIN_DOC_WINDOWS and ratio >= MIN_DOC_HIT_RATIO
            row = {"doc_id": d["doc_id"], "windows": windows, "hits": n_hits,
                   "hit_ratio": round(ratio, 6), "longest_verbatim_run": run,
                   "flagged": bool(verbatim or saturated),
                   "reason": "verbatim_run" if verbatim else
                             ("saturated_overlap" if saturated else None)}
            doc_rows.append(row)
            if row["flagged"]:
                flagged.append(row)

        entry = {
            "shard_id": m["shard_id"],
            "windows": int(hits.size),
            "hits": int(hits.sum()),
            "hit_ratio": round(int(hits.sum()) / max(1, int(hits.size)), 6),
            "max_verbatim_run": max((r["longest_verbatim_run"] for r in doc_rows),
                                    default=0),
            "flagged_documents": flagged,
            "documents": [r["doc_id"] for r in flagged],
        }
        if flagged:
            m["contamination_status"] = "eval_overlap_detected"
            m["eval_overlap"] = entry
            report["contaminated"].append(m["shard_id"])
            report["contaminated_documents"].extend(r["doc_id"] for r in flagged)
        else:
            m["contamination_status"] = "clean"
            m["eval_overlap"] = {"hits": entry["hits"], "hit_ratio": entry["hit_ratio"],
                                 "max_verbatim_run": entry["max_verbatim_run"]}
        report["shards"].append(entry)

    if log:
        log.event("contamination_scan", "PASS",
                  fingerprint_ngrams=report["fingerprint_size"],
                  scanned=len(train_manifests),
                  contaminated_shards=len(report["contaminated"]),
                  contaminated_docs=len(report["contaminated_documents"]),
                  rule=f"run>={MIN_VERBATIM_RUN} windows")
        clean_max = max((e["max_verbatim_run"] for e in report["shards"]
                         if e["shard_id"] not in report["contaminated"]), default=0)
        log.info(f"longest verbatim run in a clean shard: {clean_max} windows "
                 f"(threshold {MIN_VERBATIM_RUN}) -- incidental overlap does not "
                 f"reach the rule")
        for e in report["shards"]:
            if e["flagged_documents"]:
                log.ok("eval_overlap_blocked", shard=e["shard_id"],
                       documents=e["documents"],
                       longest_run=max(r["longest_verbatim_run"]
                                       for r in e["flagged_documents"]))
    return report


# --------------------------------------------------------------------------- #
# admission gate
# --------------------------------------------------------------------------- #

BLOCKED_LICENSES = {"unknown", "restricted", "noncommercial_restricted"}


def admit(manifest: dict, tokenizer_hash: str) -> tuple[bool, list[str]]:
    """Decide whether a shard may enter a loss-bearing batch."""
    reasons = []
    if manifest.get("tokenizer_hash") != tokenizer_hash:
        reasons.append("tokenizer_hash_mismatch")
    if manifest.get("split") != "train":
        reasons.append(f"split_not_trainable:{manifest.get('split')}")
    if manifest.get("permission") != "trainable":
        reasons.append(f"permission:{manifest.get('permission')}")
    if manifest.get("never_train"):
        reasons.append("never_train_flag")
    if manifest.get("license", "unknown") in BLOCKED_LICENSES:
        reasons.append("license_not_admissible")
    if manifest.get("dedup_status") != "exact_dedup_pass":
        reasons.append("dedup_status_missing")
    if manifest.get("contamination_status") not in ("clean",):
        reasons.append(f"contamination:{manifest.get('contamination_status')}")
    if not manifest.get("cleaning_pipeline_hash"):
        reasons.append("unknown_cleaning_lineage")
    return (not reasons), reasons


def run_admission(manifests: list[dict], tokenizer_hash: str, log=None) -> dict:
    """Apply the gate to every shard and record the outcome in the manifest."""
    admitted, blocked = [], []
    for m in manifests:
        ok, reasons = admit(m, tokenizer_hash)
        m["admitted"] = ok
        m["block_reasons"] = reasons
        m["admission_decided_at"] = now_iso()
        finalise(m)
        (admitted if ok else blocked).append(m)
    summary = {
        "admitted": [m["shard_id"] for m in admitted],
        "blocked": [{"shard_id": m["shard_id"], "split": m["split"],
                     "reasons": m["block_reasons"]} for m in blocked],
        "admitted_tokens": sum(m["num_tokens"] for m in admitted),
        "blocked_tokens": sum(m["num_tokens"] for m in blocked),
    }
    if log:
        log.ok("admission_gate", admitted=len(admitted), blocked=len(blocked),
               admitted_tokens=summary["admitted_tokens"])
        by_reason: dict[str, int] = {}
        for b in summary["blocked"]:
            for r in b["reasons"]:
                by_reason[r.split(":")[0]] = by_reason.get(r.split(":")[0], 0) + 1
        for reason, n in sorted(by_reason.items()):
            log.ok("shard_blocked", reason=reason, shards=n)
    return summary


# --------------------------------------------------------------------------- #
# the registry the dataloader must ask before serving anything
# --------------------------------------------------------------------------- #

class ShardRegistry:
    """Every shard, its permission, and a log of who touched it."""

    def __init__(self, manifests: Iterable[dict], shards_root: str) -> None:
        self.shards_root = shards_root
        self.by_id: dict[str, dict] = {m["shard_id"]: m for m in manifests}
        self.access_log: list[dict] = []

    # -- permission checks -------------------------------------------------- #

    def trainable_ids(self) -> list[str]:
        return sorted(sid for sid, m in self.by_id.items()
                      if m.get("admitted") and m["permission"] == "trainable")

    def check_trainable(self, shard_id: str, context: str) -> dict:
        """Gate every loss-bearing read. Raises rather than returning False."""
        m = self.by_id.get(shard_id)
        if m is None:
            raise FirewallViolation(f"unknown shard {shard_id}")
        self.access_log.append({"ts": now_iso(), "shard_id": shard_id,
                                "context": context, "permission": m["permission"]})
        if m["permission"] != "trainable" or not m.get("admitted"):
            raise FirewallViolation(
                f"{shard_id} is {m['permission']} "
                f"(admitted={m.get('admitted')}, reasons={m.get('block_reasons')})")
        return m

    def check_readable_for_eval(self, shard_id: str) -> dict:
        m = self.by_id[shard_id]
        self.access_log.append({"ts": now_iso(), "shard_id": shard_id,
                                "context": "validation_eval",
                                "permission": m["permission"]})
        if m["permission"] == "never_train" or m["split"] == "eval":
            raise FirewallViolation(f"{shard_id} may not be read during training")
        return m

    def ids_for(self, split: str) -> list[str]:
        return sorted(sid for sid, m in self.by_id.items() if m["split"] == split)
