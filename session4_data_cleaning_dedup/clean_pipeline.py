import json
import re
import html
import unicodedata
import hashlib
from datasets import load_dataset

# Configuration
DATASET_NAME = "tatsu-lab/alpaca"
OUTPUT_STATS_FILE = "stats.json"

print(f"Loading dataset: {DATASET_NAME}...")
try:
    dataset = load_dataset(DATASET_NAME, split="train")
    # Take first 10,000 records to make near-deduplication extremely fast while preserving statistical significance
    raw_data = [
        {
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item["output"]
        }
        for item in dataset
    ][:10000]
except Exception as e:
    print(f"Could not download dataset from HF: {e}. Generating high-fidelity mock dataset instead.")
    raw_data = []
    for i in range(5000):
        raw_data.append({
            "instruction": f"Solve this math problem: {i} + {i}",
            "input": "",
            "output": f"The answer is {i*2}."
        })

print(f"Loaded {len(raw_data)} raw records.")

# To showcase all strategies, we will inject some dirt/issues into 20% of the dataset
import random
random.seed(42)

injected_html = 0
injected_control = 0
injected_zwj = 0
injected_ghost_tags = 0
injected_pii = 0
injected_near_dups = 0
injected_mislabeled = 0
injected_contaminated = 0

dirty_records = []
for i, item in enumerate(raw_data):
    inst = item["instruction"]
    inp = item["input"]
    out = item["output"]
    
    # 1. Inject HTML Entities & Control Characters
    if i % 15 == 0:
        inst = inst + " &amp; check if it is &lt; 100 &gt; &quot;true&quot;"
        injected_html += 1
    if i % 20 == 0:
        # Add byte-order mark (\ufeff), zero-width space (\u200b), bi-directional override (\u202e)
        inst = "\ufeff\u200b" + inst + " \u202e reverse text"
        injected_control += 1
        
    # 2. Inject Indic Joiners (ZWJ/ZWNJ)
    if i % 25 == 0:
        # We add some Telugu/Hindi text with ZWJ and ZWNJ
        inst = inst + " (Examples: లక్ష్మి \u200d and నిశ్శబ్ద \u200c)"
        injected_zwj += 1
        
    # 3. Inject Ghost Conversation Tags
    if i % 18 == 0:
        # Mix incompatible formats
        formats = [
            f"[User]: {inst}\n[Assistant]: {out}",
            f"<human>: {inst}\n<bot>: {out}",
            f"### Instruction:\n{inst}\n### Response:\n{out}"
        ]
        inst = random.choice(formats)
        injected_ghost_tags += 1
        
    # 4. Inject PII (Emails, Phones, IP Addresses)
    if i % 22 == 0:
        inst = inst + f" Contact developer at contact{i}@ai-school.edu or call +1-555-019-{i%100:02d}."
        injected_pii += 1
        
    # 5. Inject Near Duplicates
    if i % 30 == 0 and len(dirty_records) > 0:
        # Copy a previous record but change only a tiny boilerplate at the end
        prev = dirty_records[-1]
        inst = prev["instruction"] + " (Please double-check)."
        inp = prev["input"]
        out = prev["output"]
        injected_near_dups += 1
        
    # 6. Inject Mislabeled Language
    if i % 40 == 0:
        # Add purely non-English characters but keep claimed language as English
        inst = "తెలుగు భాషలో సమాధానం ఇవ్వండి: " + inst
        injected_mislabeled += 1
        
    # 7. Inject Contamination (Evaluation Questions)
    if i % 50 == 0:
        inst = "What is the capital of France? " + inst
        injected_contaminated += 1
        
    dirty_records.append({
        "instruction": inst,
        "input": inp,
        "output": out
    })

print(f"Dirt injection complete.")
print(f" - HTML Entities: {injected_html}")
print(f" - Control Characters: {injected_control}")
print(f" - Indic ZWJ/ZWNJ: {injected_zwj}")
print(f" - Ghost Chat Tags: {injected_ghost_tags}")
print(f" - PII Patterns: {injected_pii}")
print(f" - Near-duplicates: {injected_near_dups}")
print(f" - Mislabeled Lang: {injected_mislabeled}")
print(f" - Contamination: {injected_contaminated}")

stats = {
    "total_raw": len(dirty_records),
    "stages": {}
}

samples = {}

# --- Stage 1: Text Normalization ---
print("Stage 1: Normalizing Text...")
stage1_data = []
cleaned_html_count = 0
cleaned_control_count = 0
kept_indic_joiners = 0

def clean_text_normalization(text):
    global cleaned_html_count, cleaned_control_count, kept_indic_joiners
    if not text:
        return text
    
    # Count HTML entities
    if "&amp;" in text or "&lt;" in text or "&gt;" in text or "&quot;" in text:
        cleaned_html_count += 1
    # Unescape HTML
    text = html.unescape(text)
    
    # Normalize Unicode Form NFC
    text = unicodedata.normalize("NFC", text)
    
    # Strip invisible control chars except ZWJ (\u200d) and ZWNJ (\u200c)
    cleaned_text = ""
    for char in text:
        cp = ord(char)
        if cp in (0x200c, 0x200d):
            kept_indic_joiners += 1
            cleaned_text += char
        elif unicodedata.category(char) in ("Cc", "Cs"):
            cleaned_control_count += 1
            continue
        elif cp in (0xfeff, 0x202e, 0x200b):
            cleaned_control_count += 1
            continue
        else:
            cleaned_text += char
            
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

sample_before = dirty_records[0]["instruction"]

for item in dirty_records:
    c_inst = clean_text_normalization(item["instruction"])
    c_inp = clean_text_normalization(item["input"])
    c_out = clean_text_normalization(item["output"])
    stage1_data.append({"instruction": c_inst, "input": c_inp, "output": c_out})

samples["normalization"] = {
    "before": sample_before,
    "after": stage1_data[0]["instruction"]
}
stats["stages"]["normalization"] = {
    "removed_html_entities": cleaned_html_count,
    "removed_control_chars": cleaned_control_count,
    "kept_indic_joiners": kept_indic_joiners,
    "remaining_docs": len(stage1_data)
}

# --- Stage 2: Ghost-Tag / Format Unification ---
print("Stage 2: Unifying Conversation Formats...")
stage2_data = []
unified_count = 0

user_patterns = [
    re.compile(r"^\[User\]:\s*", re.I),
    re.compile(r"^<human>:\s*", re.I),
    re.compile(r"^### Instruction:\s*", re.I)
]
assistant_patterns = [
    re.compile(r"\[Assistant\]:\s*", re.I),
    re.compile(r"<bot>:\s*", re.I),
    re.compile(r"### Response:\s*", re.I)
]

def unify_format(item):
    global unified_count
    inst = item["instruction"]
    inp = item["input"]
    out = item["output"]
    dirty = False
    
    for pat in user_patterns:
        if pat.search(inst):
            inst = pat.sub("", inst)
            dirty = True
            
    for pat in assistant_patterns:
        if pat.search(out):
            out = pat.sub("", out)
            dirty = True
        if pat.search(inst):
            parts = pat.split(inst, 1)
            inst = parts[0].strip()
            out = parts[1].strip() + "\n" + out
            dirty = True
            
    if dirty:
        unified_count += 1
        
    canonical_text = f"<|im_start|>user\n{inst}\n{inp if inp else ''}".strip() + f"<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
    return {
        "text": canonical_text,
        "instruction": inst,
        "input": inp,
        "output": out
    }

sample_before_format = stage1_data[18]["instruction"]
for item in stage1_data:
    stage2_data.append(unify_format(item))

samples["format_unification"] = {
    "before": sample_before_format,
    "after": stage2_data[18]["text"]
}
stats["stages"]["format_unification"] = {
    "unified_formats": unified_count,
    "remaining_docs": len(stage2_data)
}

# --- Stage 3: Quality Filtering ---
print("Stage 3: Quality Filtering...")
stage3_data = []
dropped_short = 0
dropped_high_symbol = 0
dropped_no_stopword = 0

STOPWORDS = {"the", "and", "to", "of", "a", "in", "is", "that", "for", "it", "on", "with", "as", "at", "by", "an", "be", "this", "are", "ఈ", "మరియు"}

for item in stage2_data:
    inst = item["instruction"]
    out = item["output"]
    text = inst + " " + out
    words = text.lower().split()
    
    if len(words) < 5:
        dropped_short += 1
        continue
        
    non_alphanum = len([c for c in text if not c.isalnum() and not c.isspace()])
    symbol_ratio = non_alphanum / max(len(text), 1)
    if symbol_ratio > 0.35:
        dropped_high_symbol += 1
        continue
        
    has_stopword = any(w in STOPWORDS for w in words)
    is_indic = any(0x0c00 <= ord(c) <= 0x0c7f for c in text)
    if not has_stopword and not is_indic:
        dropped_no_stopword += 1
        continue
        
    stage3_data.append(item)

stats["stages"]["quality_filtering"] = {
    "dropped_too_short": dropped_short,
    "dropped_high_symbol_ratio": dropped_high_symbol,
    "dropped_no_stopwords": dropped_no_stopword,
    "remaining_docs": len(stage3_data)
}

# --- Stage 4 & 5: Near-Deduplication & Global Deduplication ---
print("Stage 4 & 5: Deduplicating with LSH-bucketing...")
stage5_data = []
dropped_duplicates = 0

def get_shingles(text, k=5):
    text = re.sub(r'\s+', '', text.lower())
    return set(text[i:i+k] for i in range(len(text) - k + 1))

# Fast LSH-like bucketing:
# We map each document to its 3 most frequent shingles or standard prefix hashes.
# We only perform full Jaccard comparison within the same bucket.
buckets = {}
for item in stage3_data:
    sh = get_shingles(item["instruction"])
    if not sh:
        stage5_data.append(item)
        continue
        
    # Generate signature keys (LSH bands)
    # Sort shingles to be deterministic and pick top 2
    sorted_sh = sorted(list(sh))
    keys = sorted_sh[:2] if len(sorted_sh) >= 2 else sorted_sh
    
    is_dup = False
    # Check Jaccard only with items sharing the same keys
    candidates = {}
    for key in keys:
        if key in buckets:
            for prev_item, prev_sh in buckets[key]:
                candidates[prev_item["instruction"]] = (prev_item, prev_sh)
            
    for prev_item, prev_sh in candidates.values():
        intersection = len(sh.intersection(prev_sh))
        union = len(sh.union(prev_sh))
        jaccard = intersection / union if union > 0 else 0
        if jaccard > 0.80:
            is_dup = True
            break
            
    if is_dup:
        dropped_duplicates += 1
    else:
        # Add to buckets
        for key in keys:
            if key not in buckets:
                buckets[key] = []
            buckets[key].append((item, sh))
        stage5_data.append(item)

stats["stages"]["deduplication"] = {
    "dropped_near_duplicates": dropped_duplicates,
    "remaining_docs": len(stage5_data)
}

# --- Stage 6: Language ID & Validation ---
print("Stage 6: Language ID & Validation...")
stage6_data = []
mismatched_language = 0

for item in stage5_data:
    inst = item["instruction"]
    telugu_chars = len([c for c in inst if 0x0c00 <= ord(c) <= 0x0c7f])
    english_chars = len([c for c in inst if c.isascii()])
    
    if telugu_chars > 5 and english_chars / len(inst) < 0.5:
        mismatched_language += 1
        item["language"] = "te"
    else:
        item["language"] = "en"
        
    stage6_data.append(item)

stats["stages"]["language_id"] = {
    "mismatched_or_routed_languages": mismatched_language,
    "remaining_docs": len(stage6_data)
}

# --- Stage 7: PII Scrubbing ---
print("Stage 7: Removing PII...")
stage7_data = []
scrubbed_emails = 0
scrubbed_phones = 0

email_regex = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
phone_regex = re.compile(r'\+?\d{1,4}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}')

def scrub_pii(text):
    global scrubbed_emails, scrubbed_phones
    if not text:
        return text
    
    text, email_subs = email_regex.subn("[EMAIL]", text)
    scrubbed_emails += email_subs
    
    text, phone_subs = phone_regex.subn("[PHONE]", text)
    scrubbed_phones += phone_subs
    
    return text

sample_before_pii = stage6_data[22]["instruction"] if len(stage6_data) > 22 else ""

for item in stage6_data:
    item["text"] = scrub_pii(item["text"])
    item["instruction"] = scrub_pii(item["instruction"])
    item["input"] = scrub_pii(item["input"])
    item["output"] = scrub_pii(item["output"])
    stage7_data.append(item)

samples["pii"] = {
    "before": sample_before_pii,
    "after": stage7_data[22]["instruction"] if len(stage7_data) > 22 else ""
}
stats["stages"]["pii_removal"] = {
    "scrubbed_emails": scrubbed_emails,
    "scrubbed_phones": scrubbed_phones,
    "remaining_docs": len(stage7_data)
}

# --- Stage 8: Decontamination ---
print("Stage 8: Decontamination...")
stage8_data = []
decontaminated_count = 0

EVAL_BENCHMARKS = [
    "what is the capital of france",
    "solve this math problem"
]

for item in stage7_data:
    text_lower = item["text"].lower()
    is_contaminated = False
    for eval_q in EVAL_BENCHMARKS:
        if eval_q in text_lower:
            is_contaminated = True
            break
            
    if is_contaminated:
        decontaminated_count += 1
    else:
        stage8_data.append(item)

stats["stages"]["decontamination"] = {
    "dropped_contaminated_docs": decontaminated_count,
    "remaining_docs": len(stage8_data)
}

# --- Stage 9: Provenance & Reproducibility ---
print("Stage 9: Creating Manifest...")
stage9_data = []

manifest = {
    "dataset_source": DATASET_NAME,
    "cleaning_script_hash": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
    "provenance": {
        "license": "Apache 2.0 (Alpaca / HuggingFace)",
        "creator": "Tatsu Lab & TSAI Student pipeline",
        "clean_ratio": len(stage8_data) / len(dirty_records)
    },
    "shards": []
}

for item in stage8_data:
    content_hash = hashlib.sha256(item["text"].encode('utf-8')).hexdigest()
    item["id"] = f"tsai-clean-{content_hash[:16]}"
    item["hash"] = content_hash
    stage9_data.append(item)

manifest["shards"].append({
    "file_path": "cleaned_dataset.json",
    "token_count": sum(len(item["text"].split()) for item in stage9_data),
    "document_count": len(stage9_data),
    "language_breakdown": {
        "en": sum(1 for item in stage9_data if item["language"] == "en"),
        "te": sum(1 for item in stage9_data if item["language"] == "te")
    }
})

stats["manifest"] = manifest
stats["samples"] = samples

with open(OUTPUT_STATS_FILE, "w") as f:
    json.dump(stats, f, indent=2)

with open("cleaned_dataset.json", "w") as f:
    json.dump(stage9_data, f, indent=2)

print("Pipeline completed successfully! Dumps written to stats.json and cleaned_dataset.json.")
