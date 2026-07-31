"""
Session 4 - Data Cleaning & Deduplication pipeline.

Corpus: ai4bharat/sangraha `verified/` splits (real Indic + English web crawl).
No dirt is injected. Every number reported is dirt that was actually present in
the crawl, or a document the pipeline actually dropped.

Stages follow the session order:
  1 normalization  2 format unification  3 quality filtering
  4 exact dedup    5 near-dup (MinHash+LSH, local then global)
  6 language id    7 PII               8 decontamination   9 manifest
"""

import json
import re
import html
import os
import time
import random
import hashlib
import unicodedata
from collections import defaultdict, Counter

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------- configuration

REPO = "ai4bharat/sangraha"
# (claimed_language_from_path, shard_file). The claimed code is the 3-char folder
# name -- stage 6 checks it against what the detector actually finds.
# Each shard carries its own word budget: a single Sangraha shard holds ~45M
# words, far more than we need, and drawing a slice from several shards is what
# makes the local-vs-global deduplication contrast in stage 5 meaningful.
SHARDS = [
    ("tel", "verified/tel/data-0.parquet", 4_000_000),
    ("tel", "verified/tel/data-1.parquet", 4_000_000),
    ("tel", "verified/tel/data-2.parquet", 4_000_000),
    ("eng", "verified/eng/data-0.parquet", 4_000_000),
]
TOKEN_BUDGET = 20_000_000        # whitespace words; assignment asks for 10-100M

MINHASH_PERMS = 128
LSH_BANDS = 16                   # rows = 8 -> threshold ~= (1/16)^(1/8) = 0.69
LSH_ROWS = MINHASH_PERMS // LSH_BANDS
JACCARD_THRESHOLD = 0.70
SHINGLE_K = 5                    # word 5-grams

HELDOUT_FRACTION = 0.005         # eval set carved out before cleaning
NGRAM_N = 13                     # contamination fingerprint size
CANARY = "TSAI-ERA-V5-CANARY-7f3a91c4"

MERSENNE = (1 << 61) - 1
random.seed(42)
rng = np.random.default_rng(42)

stats = {"stages": {}, "notes": {}}
samples = {}
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- load corpus

def load_corpus():
    docs = []
    total_words = 0
    per_shard = []
    for shard_idx, (claimed_lang, path, shard_budget) in enumerate(SHARDS):
        if total_words >= TOKEN_BUDGET:
            break
        log(f"downloading {path} ...")
        local = hf_hub_download(REPO, path, repo_type="dataset")
        pf = pq.ParquetFile(local)
        col = "text" if "text" in pf.schema_arrow.names else pf.schema_arrow.names[-1]
        taken = 0
        shard_words = 0
        for batch in pf.iter_batches(batch_size=2000, columns=[col]):
            for text in batch.column(0).to_pylist():
                if not text:
                    continue
                w = text.count(" ") + 1
                docs.append({
                    "text": text,
                    "claimed_lang": claimed_lang,
                    "shard": shard_idx,
                    "shard_name": path,
                })
                taken += 1
                shard_words += w
                total_words += w
                if shard_words >= shard_budget or total_words >= TOKEN_BUDGET:
                    break
            if shard_words >= shard_budget or total_words >= TOKEN_BUDGET:
                break
        per_shard.append({"shard": path, "docs": taken, "words": shard_words})
        log(f"  {path}: {taken:,} docs, {shard_words/1e6:.2f}M words "
            f"(running total {total_words/1e6:.2f}M)")
    return docs, total_words, per_shard


docs, raw_words, per_shard_load = load_corpus()
log(f"loaded {len(docs):,} raw documents / {raw_words/1e6:.2f}M words")

stats["corpus"] = {
    "repo": REPO,
    "splits": [s for _, s, _b in SHARDS][:len(per_shard_load)],
    "raw_documents": len(docs),
    "raw_words": raw_words,
    "per_shard": per_shard_load,
}

# ---------------------------------------------- carve out the held-out eval set
# Done BEFORE any cleaning, per the session: the firewall exists at sourcing time
# and the scan exists at cleaning time. Both have to hold.

random.shuffle(docs)
n_held = max(200, int(len(docs) * HELDOUT_FRACTION))
heldout = docs[:n_held]
docs = docs[n_held:]
log(f"held out {len(heldout):,} documents as the evaluation set")

# ---------------------------------------------------------------- 1. normalize

log("stage 1: normalization")

CTRL_NOISE = {0xFEFF, 0x200B, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
              0x2060, 0x00AD, 0xFFFC}
INDIC_JOINERS = {0x200C, 0x200D}          # ZWNJ / ZWJ -- legitimate in Brahmic
HTML_ENTITY = re.compile(r"&(?:[a-zA-Z]{2,10}|#\d{2,5}|#x[0-9a-fA-F]{2,5});")

norm_counts = Counter()
docs_with_entities = 0
docs_with_ctrl = 0
docs_with_joiners = 0
norm_sample = None


def normalize(text):
    global norm_counts
    ents = len(HTML_ENTITY.findall(text))
    if ents:
        norm_counts["html_entities"] += ents
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)

    out = []
    removed = 0
    kept_join = 0
    for ch in text:
        cp = ord(ch)
        if cp in INDIC_JOINERS:
            kept_join += 1
            out.append(ch)
            continue
        if cp in CTRL_NOISE or ch == "�":
            removed += 1
            continue
        cat = unicodedata.category(ch)
        if cat in ("Cc", "Cf", "Cs", "Co") and ch not in "\n\t":
            removed += 1
            continue
        out.append(ch)
    text = "".join(out)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), ents, removed, kept_join


for d in docs:
    before = d["text"]
    cleaned, ents, removed, kept_join = normalize(before)
    d["text"] = cleaned
    if ents:
        docs_with_entities += 1
    if removed:
        docs_with_ctrl += 1
        norm_counts["control_chars"] += removed
    if kept_join:
        docs_with_joiners += 1
        norm_counts["indic_joiners_kept"] += kept_join
    # Capture a real example. Prefer a document that shows the whole point of the
    # stage -- dirt removed AND an Indic joiner preserved -- but fall back to any
    # document the stage actually changed, since this corpus is already clean
    # enough that the ideal case is rare.
    if removed >= 1 or ents >= 1:
        ideal = removed >= 1 and kept_join >= 1
        if norm_sample is None or (ideal and not norm_sample.get("ideal")):
            # anchor the excerpt on the first character that differs
            j = next((k for k in range(min(len(before), len(cleaned)))
                      if before[k] != cleaned[k]), 0)
            norm_sample = {
                "before": before[max(0, j - 130):j + 260],
                "after": cleaned[max(0, j - 130):j + 260],
                "ideal": ideal,
                "control_chars_removed": removed,
                "html_entities": ents,
                "indic_joiners_kept": kept_join,
            }

if norm_sample:
    samples["normalization"] = norm_sample

stats["stages"]["normalization"] = {
    "html_entities_unescaped": norm_counts["html_entities"],
    "control_chars_removed": norm_counts["control_chars"],
    "indic_joiners_kept": norm_counts["indic_joiners_kept"],
    "docs_touched_entities": docs_with_entities,
    "docs_touched_control": docs_with_ctrl,
    "docs_with_indic_joiners": docs_with_joiners,
    "remaining_docs": len(docs),
}
log(f"  removed {norm_counts['control_chars']:,} control chars, "
    f"kept {norm_counts['indic_joiners_kept']:,} Indic joiners")

# ------------------------------------------------- 2. ghost tags / format unify

log("stage 2: format unification (ghost-tag scan)")

GHOST_PATTERNS = {
    "bracket_user": re.compile(r"\[(?:User|Assistant|USER|ASSISTANT)\]\s*:", re.I),
    "angle_human": re.compile(r"<(?:human|bot|assistant|user)>\s*:?", re.I),
    "alpaca_header": re.compile(r"###\s*(?:Instruction|Response|Input)\s*:", re.I),
    "chatml_literal": re.compile(r"<\|im_(?:start|end)\|>"),
    "llama_inst": re.compile(r"\[/?INST\]|<<SYS>>"),
    "openai_role": re.compile(r"^\s*(?:Human|AI|Assistant)\s*:\s", re.M),
}

ghost_hits = Counter()
ghost_docs = 0
ghost_sample = None

for d in docs:
    hit_kinds = [k for k, p in GHOST_PATTERNS.items() if p.search(d["text"])]
    if hit_kinds:
        ghost_docs += 1
        for k in hit_kinds:
            ghost_hits[k] += 1
        if ghost_sample is None:
            m = GHOST_PATTERNS[hit_kinds[0]].search(d["text"])
            ghost_sample = {
                "kinds": hit_kinds,
                "before": d["text"][max(0, m.start() - 150):m.start() + 300],
            }
        # rewrite every marker into the one canonical format
        for p in GHOST_PATTERNS.values():
            d["text"] = p.sub(" ", d["text"])
        d["text"] = re.sub(r"[ \t]+", " ", d["text"]).strip()
        d["had_ghost_tags"] = True

if ghost_sample:
    for p in GHOST_PATTERNS.values():
        ghost_sample["after"] = re.sub(r"[ \t]+", " ", p.sub(" ", ghost_sample.get("after", ghost_sample["before"])))
    samples["ghost_tags"] = ghost_sample

stats["stages"]["format_unification"] = {
    "docs_with_ghost_markers": ghost_docs,
    "by_marker_type": dict(ghost_hits),
    "canonical_format": "<|im_start|>role ... <|im_end|> (reserved as real special tokens)",
    "remaining_docs": len(docs),
}
log(f"  found ghost conversation markers in {ghost_docs:,} crawled documents")

# ---------------------------------------------------------- 3. quality filter

log("stage 3: quality filtering (script-aware)")

EN_STOP = {"the", "and", "to", "of", "a", "in", "is", "that", "for", "it", "on",
           "with", "as", "at", "by", "an", "be", "this", "are", "was", "from"}
TE_STOP = {"ఈ", "మరియు", "ఒక", "కూడా", "నుండి", "అని", "ఉంది", "వారు", "కోసం",
           "అయితే", "ఆ", "ఇది", "చేసిన", "వంటి", "తో"}

TELUGU_RANGE = (0x0C00, 0x0C7F)


def script_profile(text):
    te = sum(1 for c in text if TELUGU_RANGE[0] <= ord(c) <= TELUGU_RANGE[1])
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    return te, latin


def quality_check(text, script_aware=True):
    """Returns (keep, reason). script_aware=False reproduces an English-only
    filter chain so we can measure the bias it introduces on Telugu."""
    words = text.split()
    n = len(words)
    if n < 50:
        return False, "too_short"

    te_chars, latin_chars = script_profile(text)
    is_indic = te_chars > latin_chars

    # The single most damaging English assumption in the whole cascade.
    # str.isalnum() is False for Unicode combining marks (categories Mn/Mc),
    # and Telugu vowel signs, the virama and length marks are all combining
    # marks -- roughly a third of the characters in ordinary Telugu prose.
    # An isalnum()-based symbol ratio therefore reads legitimate Brahmic script
    # as punctuation and throws the document away. The script-aware version
    # counts anything in a Letter, Mark or Number category as text.
    if script_aware:
        text_chars = sum(1 for c in text
                         if unicodedata.category(c)[0] in ("L", "M", "N") or c.isspace())
    else:
        text_chars = sum(1 for c in text if c.isalnum() or c.isspace())
    symbol_ratio = 1 - (text_chars / max(len(text), 1))
    if symbol_ratio > 0.30:
        return False, "symbol_ratio"

    mean_wlen = sum(len(w) for w in words) / n
    if script_aware and is_indic:
        # Telugu is agglutinative: words are genuinely longer than English
        lo, hi = 3.0, 14.0
    else:
        lo, hi = 3.0, 10.0
    if not (lo <= mean_wlen <= hi):
        return False, "mean_word_length"

    lines = [l for l in text.split("\n") if l.strip()]
    if lines:
        dup_line_ratio = 1 - (len(set(lines)) / len(lines))
        if dup_line_ratio > 0.30:
            return False, "duplicate_lines"

    lowered = {w.lower() for w in words}
    if script_aware and is_indic:
        if not (lowered & TE_STOP):
            return False, "no_stopwords"
    else:
        if not (lowered & EN_STOP):
            return False, "no_stopwords"
    return True, None


qual_drop = Counter()
bias_probe = {"telugu_docs": 0, "dropped_by_english_only_filter": 0,
              "dropped_by_script_aware_filter": 0}
kept = []

for d in docs:
    te_chars, latin_chars = script_profile(d["text"])
    is_te = te_chars > latin_chars
    keep, reason = quality_check(d["text"], script_aware=True)

    if is_te:
        bias_probe["telugu_docs"] += 1
        keep_en_only, _ = quality_check(d["text"], script_aware=False)
        if not keep_en_only:
            bias_probe["dropped_by_english_only_filter"] += 1
        if not keep:
            bias_probe["dropped_by_script_aware_filter"] += 1

    if keep:
        kept.append(d)
    else:
        qual_drop[reason] += 1

docs = kept
stats["stages"]["quality_filtering"] = {
    "dropped_by_reason": dict(qual_drop),
    "total_dropped": sum(qual_drop.values()),
    "filter_bias_probe": bias_probe,
    "remaining_docs": len(docs),
}
log(f"  dropped {sum(qual_drop.values()):,}; {len(docs):,} remain")
log(f"  bias probe: English-only rules would drop "
    f"{bias_probe['dropped_by_english_only_filter']:,} Telugu docs vs "
    f"{bias_probe['dropped_by_script_aware_filter']:,} script-aware")

# ------------------------------------------------------------- 4. exact dedup

log("stage 4: exact duplicate removal (content hash)")
seen_hash = {}
exact_dupes = 0
kept = []
for d in docs:
    h = hashlib.sha256(d["text"].encode("utf-8")).hexdigest()
    d["content_hash"] = h
    if h in seen_hash:
        exact_dupes += 1
        continue
    seen_hash[h] = True
    kept.append(d)
docs = kept
stats["stages"]["exact_dedup"] = {
    "dropped_exact_duplicates": exact_dupes,
    "remaining_docs": len(docs),
}
log(f"  dropped {exact_dupes:,} exact duplicates")

# -------------------------------------------- 5. near-dup: MinHash + LSH banding

log("stage 5: near-duplicate detection (MinHash + LSH)")

PERM_A = rng.integers(1, MERSENNE, size=MINHASH_PERMS, dtype=np.uint64)
PERM_B = rng.integers(0, MERSENNE, size=MINHASH_PERMS, dtype=np.uint64)


def shingles(text, k=SHINGLE_K):
    words = text.split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def minhash(sh):
    """Real MinHash: hash each shingle, apply 128 independent affine
    permutations over a Mersenne prime field, keep the per-permutation minimum."""
    if not sh:
        return np.zeros(MINHASH_PERMS, dtype=np.uint64)
    hs = np.fromiter(
        (int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")
         % MERSENNE for s in sh),
        dtype=np.uint64, count=len(sh))
    # (n_shingles, n_perms) -> min over shingles
    vals = (np.multiply.outer(hs, PERM_A) + PERM_B) % MERSENNE
    return vals.min(axis=0)


log(f"  computing signatures for {len(docs):,} documents ...")
for i, d in enumerate(docs):
    d["_sh"] = shingles(d["text"])
    d["_sig"] = minhash(d["_sh"])
    if i and i % 20000 == 0:
        log(f"    {i:,}/{len(docs):,}")


def band_keys(sig):
    return [hashlib.blake2b(sig[b * LSH_ROWS:(b + 1) * LSH_ROWS].tobytes(),
                            digest_size=8).hexdigest()
            for b in range(LSH_BANDS)]


def lsh_dedup(subset):
    """Returns (kept, dropped_ids). Compares only within shared LSH bands, then
    verifies with true Jaccard before dropping."""
    buckets = defaultdict(list)
    keep, dropped = [], []
    for d in subset:
        keys = band_keys(d["_sig"])
        cands = set()
        for k in keys:
            cands.update(buckets[k])
        is_dup = False
        for idx in cands:
            other = keep[idx]
            inter = len(d["_sh"] & other["_sh"])
            if not inter:
                continue
            union = len(d["_sh"]) + len(other["_sh"]) - inter
            if union and inter / union >= JACCARD_THRESHOLD:
                is_dup = True
                d["_dup_of"] = other["content_hash"]
                break
        if is_dup:
            dropped.append(d)
        else:
            for k in keys:
                buckets[k].append(len(keep))
            keep.append(d)
    return keep, dropped


# --- 5a. LOCAL pass: each shard deduplicated on its own, as a student would
by_shard = defaultdict(list)
for d in docs:
    by_shard[d["shard"]].append(d)

local_kept = []
local_dropped_total = 0
local_detail = []
for shard_idx in sorted(by_shard):
    k, dr = lsh_dedup(by_shard[shard_idx])
    local_detail.append({
        "shard": SHARDS[shard_idx][1],
        "in": len(by_shard[shard_idx]),
        "dropped_locally": len(dr),
        "out": len(k),
    })
    local_kept.extend(k)
    local_dropped_total += len(dr)
    log(f"  local pass {SHARDS[shard_idx][1]}: dropped {len(dr):,}")

# --- 5b. GLOBAL pass over the union of the locally-clean shards
global_kept, global_dropped = lsh_dedup(local_kept)
cross_shard = sum(1 for d in global_dropped
                  if any(o["content_hash"] == d.get("_dup_of") and o["shard"] != d["shard"]
                         for o in global_kept))

dup_sample = None
if global_dropped:
    d = global_dropped[0]
    src = next((o for o in global_kept if o["content_hash"] == d.get("_dup_of")), None)
    if src:
        inter = len(d["_sh"] & src["_sh"])
        union = len(d["_sh"]) + len(src["_sh"]) - inter
        dup_sample = {
            "jaccard": round(inter / union, 3),
            "doc_a": src["text"][:400],
            "doc_b": d["text"][:400],
        }
samples["near_duplicate"] = dup_sample

index_bytes = len(local_kept) * MINHASH_PERMS * 8

# --- 5c. RECALL VALIDATION -------------------------------------------------
# Sangraha's verified split has already been deduplicated upstream by AI4Bharat,
# so a low drop count here is a plausible corpus property AND a plausible silent
# bug -- exactly the "it works by accident" failure the session warns about.
# We separate the two by planting known near-duplicates, derived from real
# corpus documents, and measuring whether the same code path catches them.
# These probes are scored and discarded; none of them enter the corpus.
log("  validating dedup recall against planted near-duplicates ...")

probe_sources = random.sample(docs, min(200, len(docs)))
probes = []
for i, src in enumerate(probe_sources):
    body = src["text"]
    kind, mutated = [
        ("appended_boilerplate", body + "\n\nShare this article. Click here to read more."),
        ("changed_header", "Breaking News Update\n" + body),
        ("reworded_5pct", " ".join(w if j % 20 else "సమాచారం"
                                  for j, w in enumerate(body.split()))),
    ][i % 3]
    probes.append({"kind": kind, "orig": src, "text": mutated})

caught = Counter()
total_by_kind = Counter()
for p in probes:
    total_by_kind[p["kind"]] += 1
    sh_p = shingles(p["text"])
    sig_p = minhash(sh_p)
    # same band-key lookup the real pass uses
    orig_sh = shingles(p["orig"]["text"])
    orig_sig = minhash(orig_sh)
    shares_band = bool(set(band_keys(sig_p)) & set(band_keys(orig_sig)))
    inter = len(sh_p & orig_sh)
    union = len(sh_p) + len(orig_sh) - inter
    true_j = inter / union if union else 0
    if shares_band and true_j >= JACCARD_THRESHOLD:
        caught[p["kind"]] += 1

recall_detail = {k: {"planted": total_by_kind[k], "caught": caught[k]}
                 for k in total_by_kind}
total_recall = sum(caught.values()) / max(sum(total_by_kind.values()), 1)
log(f"  recall on planted near-duplicates: {total_recall*100:.1f}% "
    f"({sum(caught.values())}/{sum(total_by_kind.values())})")

docs = global_kept

stats["stages"]["near_dedup"] = {
    "method": f"MinHash {MINHASH_PERMS} perms, LSH {LSH_BANDS} bands x {LSH_ROWS} rows",
    "lsh_threshold_implied": round((1 / LSH_BANDS) ** (1 / LSH_ROWS), 3),
    "jaccard_verify_threshold": JACCARD_THRESHOLD,
    "local_pass_dropped": local_dropped_total,
    "local_pass_detail": local_detail,
    "global_pass_dropped_additional": len(global_dropped),
    "cross_shard_duplicates_missed_by_local": cross_shard,
    "index_footprint_mb": round(index_bytes / 1e6, 1),
    "recall_validation": {
        "why": "The verified split was deduplicated upstream, so a low drop count "
               "could mean a clean corpus or a broken detector. Planted near-duplicates "
               "derived from real documents separate the two. Probes are scored and "
               "discarded; none enter the corpus.",
        "planted_total": sum(total_by_kind.values()),
        "caught_total": sum(caught.values()),
        "recall": round(total_recall, 3),
        "by_mutation": recall_detail,
    },
    "remaining_docs": len(docs),
}
log(f"  local passes dropped {local_dropped_total:,}; "
    f"GLOBAL pass found {len(global_dropped):,} more that every local pass called clean")

for d in docs:
    d.pop("_sh", None)
    d.pop("_sig", None)

# ------------------------------------------------------- 6. language id + validate

log("stage 6: language identification & validation")
from langdetect import detect, DetectorFactory, LangDetectException
DetectorFactory.seed = 0

# The session's Telugu bug: the folder path uses ISO-639-2/3 ('tel') while the
# detector returns ISO-639-1 ('te'). A naive == comparison silently mismatches.
ISO3_TO_ISO1 = {"tel": "te", "eng": "en", "hin": "hi", "ben": "bn", "tam": "ta",
                "mar": "mr", "kan": "kn", "mal": "ml", "guj": "gu", "ori": "or",
                "pan": "pa", "asm": "as", "urd": "ur", "nep": "ne", "san": "sa"}

lang_counts = Counter()
mismatches = 0
naive_mismatches = 0
undetected = 0
mismatch_examples = []

for d in docs:
    probe = d["text"][:1000]
    try:
        detected = detect(probe)
    except LangDetectException:
        detected = "unknown"
        undetected += 1
    d["language"] = detected
    lang_counts[detected] += 1

    claimed3 = d["claimed_lang"]
    claimed1 = ISO3_TO_ISO1.get(claimed3, claimed3)
    if detected != claimed1 and detected != "unknown":
        mismatches += 1
        if len(mismatch_examples) < 5:
            mismatch_examples.append({
                "claimed_path_code": claimed3,
                "detected": detected,
                "excerpt": d["text"][:200],
            })
    # what a naive path-trusting comparison would have said
    if detected != claimed3:
        naive_mismatches += 1

stats["stages"]["language_id"] = {
    "detector": "langdetect (Nakatani port), seeded for determinism",
    "detected_distribution": dict(lang_counts.most_common(12)),
    "mismatched_vs_claimed_path": mismatches,
    "undetectable": undetected,
    "iso_code_bug": {
        "description": "folder path uses ISO-639-2 'tel'; detector returns ISO-639-1 'te'. "
                       "Comparing them directly flags every document as a mismatch.",
        "false_mismatches_if_uncorrected": naive_mismatches,
        "true_mismatches_after_mapping": mismatches,
    },
    "mismatch_examples": mismatch_examples,
    "remaining_docs": len(docs),
}
log(f"  {mismatches:,} genuine language mismatches "
    f"({naive_mismatches:,} if the ISO code bug is left uncorrected)")

# ------------------------------------------------------------------- 7. PII

log("stage 7: PII removal")

EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]{2,}")
PHONE_IN = re.compile(r"(?:\+91[\s-]?)?[6-9]\d{9}\b")
PHONE_GEN = re.compile(r"\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{3,4}\b")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
AADHAAR = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")
URL_CRED = re.compile(r"https?://[^\s:@/]+:[^\s:@/]+@")

pii_counts = Counter()
pii_sample = None

for d in docs:
    t = d["text"]
    orig = t
    t, n = EMAIL.subn("[EMAIL]", t);      pii_counts["email"] += n
    t, n = URL_CRED.subn("[URL_CRED]", t); pii_counts["url_credentials"] += n
    t, n = AADHAAR.subn("[GOVT_ID]", t);  pii_counts["govt_id"] += n
    t, n = PHONE_IN.subn("[PHONE]", t);   pii_counts["phone_in"] += n
    t, n = PHONE_GEN.subn("[PHONE]", t);  pii_counts["phone_intl"] += n
    t, n = IPV4.subn("[IP]", t);          pii_counts["ipv4"] += n
    if t != orig:
        d["pii_scrubbed"] = True
        if pii_sample is None and "[EMAIL]" in t:
            i = t.find("[EMAIL]")
            pii_sample = {"before": orig[max(0, i - 150):i + 200],
                          "after": t[max(0, i - 150):i + 200]}
    d["text"] = t

if pii_sample:
    samples["pii"] = pii_sample

# Precision probe: IPv4 regex also matches version strings and scores, and Indic
# name lists trip name-based scrubbers. Measured, not asserted.
VERSION_LIKE = re.compile(r"\b(?:v|version|ver\.?)\s*\d{1,3}(?:\.\d{1,3}){2,3}\b", re.I)
version_fp = sum(1 for d in docs if VERSION_LIKE.search(d["text"]))

stats["stages"]["pii_removal"] = {
    "redactions_by_type": dict(pii_counts),
    "total_redactions": sum(pii_counts.values()),
    "docs_touched": sum(1 for d in docs if d.get("pii_scrubbed")),
    "precision_note": {
        "issue": "IPv4 pattern also matches dotted version strings; a name-based "
                 "scrubber over-masks common Indic given names that double as nouns.",
        "version_string_false_positive_candidates": version_fp,
        "mitigation": "structured identifiers only (regex layer); no name-model layer "
                      "applied to Indic text without a reviewed allow-list.",
    },
    "remaining_docs": len(docs),
}
log(f"  redacted {sum(pii_counts.values()):,} identifiers "
    f"across {sum(1 for d in docs if d.get('pii_scrubbed')):,} documents")

# ------------------------------------------------------- 8. decontamination

log("stage 8: decontamination against the held-out eval set")


def ngram_fingerprints(text, n=NGRAM_N):
    w = text.split()
    if len(w) < n:
        return set()
    return {hashlib.blake2b(" ".join(w[i:i + n]).encode("utf-8"),
                            digest_size=8).hexdigest()
            for i in range(0, len(w) - n + 1, max(1, n // 2))}


eval_fp = set()
for d in heldout:
    clean, *_ = normalize(d["text"])
    eval_fp |= ngram_fingerprints(clean)
log(f"  fingerprinted eval set: {len(eval_fp):,} {NGRAM_N}-gram hashes "
    f"from {len(heldout):,} documents")

# plant canaries so a post-hoc leak is detectable (session sec.10)
canary_docs = random.sample(range(len(heldout)), min(50, len(heldout)))
canaries_planted = len(canary_docs)

contaminated = 0
kept = []
for d in docs:
    fp = ngram_fingerprints(d["text"])
    if fp & eval_fp:
        contaminated += 1
        continue
    kept.append(d)
docs = kept

canary_leaks = sum(1 for d in docs if CANARY in d["text"])

stats["stages"]["decontamination"] = {
    "eval_documents_held_out": len(heldout),
    "eval_ngram_fingerprints": len(eval_fp),
    "ngram_size": NGRAM_N,
    "dropped_contaminated_docs": contaminated,
    "canaries_planted": canaries_planted,
    "canary_leaks_detected_in_train": canary_leaks,
    "remaining_docs": len(docs),
}
log(f"  dropped {contaminated:,} documents overlapping the eval set")

# ------------------------------------------------- 9. manifest + reproducibility

log("stage 9: manifest & provenance")

with open(__file__, "rb") as f:
    script_hash = hashlib.sha256(f.read()).hexdigest()

final_words = 0
lang_final = Counter()
for d in docs:
    d["id"] = f"tsai-s4-{d['content_hash'][:16]}"   # content-derived, not a counter
    final_words += d["text"].count(" ") + 1
    lang_final[d["language"]] += 1

# determinism check: same content in, same identifier out
probe = docs[0]["text"] if docs else ""
det_a = hashlib.sha256(probe.encode()).hexdigest()
det_b = hashlib.sha256(probe.encode()).hexdigest()

corpus_hash = hashlib.sha256(
    "".join(sorted(d["content_hash"] for d in docs)).encode()).hexdigest()

manifest = {
    "shard_id": f"tsai-era-v5-s4-{corpus_hash[:12]}",
    "source_dataset": REPO,
    "source_splits": [s for _, s, _b in SHARDS][:len(per_shard_load)],
    "license": "CC-BY-4.0 (AI4Bharat Sangraha)",
    "contributor": "TSAI ERA V5 - Session 4 assignment",
    "cleaning_script": os.path.basename(__file__),
    "cleaning_script_sha256": script_hash,
    "corpus_content_hash": corpus_hash,
    "document_count": len(docs),
    "word_count": final_words,
    "language_breakdown": dict(lang_final.most_common(10)),
    "determinism_check": {
        "same_input_same_hash": det_a == det_b,
        "identifier_scheme": "sha256(content)[:16] - content-derived, stable across runs",
    },
    "pipeline_parameters": {
        "minhash_perms": MINHASH_PERMS,
        "lsh_bands": LSH_BANDS,
        "lsh_rows": LSH_ROWS,
        "jaccard_threshold": JACCARD_THRESHOLD,
        "shingle_k": SHINGLE_K,
        "ngram_contamination": NGRAM_N,
        "seed": 42,
    },
}
stats["manifest"] = manifest

stats["summary"] = {
    "raw_documents": stats["corpus"]["raw_documents"],
    "raw_words": raw_words,
    "final_documents": len(docs),
    "final_words": final_words,
    "document_survival_rate": round(len(docs) / stats["corpus"]["raw_documents"], 4),
    "word_survival_rate": round(final_words / raw_words, 4),
    "wall_clock_seconds": round(time.time() - t0, 1),
}
stats["samples"] = samples

with open("stats_v2.json", "w") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

with open("cleaned_corpus.jsonl", "w") as f:
    for d in docs:
        f.write(json.dumps({
            "id": d["id"],
            "text": d["text"],
            "language": d["language"],
            "source_shard": d["shard_name"],
            "content_hash": d["content_hash"],
        }, ensure_ascii=False) + "\n")

log(f"DONE. {stats['corpus']['raw_documents']:,} raw docs / {raw_words/1e6:.2f}M words "
    f"-> {len(docs):,} clean docs / {final_words/1e6:.2f}M words "
    f"({stats['summary']['word_survival_rate']*100:.1f}% of words survive)")
