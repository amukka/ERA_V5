#!/usr/bin/env python3
"""Fetch India's Wikipedia page in en/hi/te/kn at a pinned revision.
Saves plain-text article + meta (title, revid, word count). Stdlib only."""
import json, urllib.parse, os, re, subprocess

LANGS = {
    "en": "India",
    "hi": "भारत",
    "te": "భారతదేశం",
    "kn": "ಭಾರತ",
}
OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(OUT, "data")
os.makedirs(DATA, exist_ok=True)

UA = "ERA-V5-tokenizer-assignment/1.0 (educational)"

def api(lang, params):
    params = {**params, "format": "json"}
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    out = subprocess.run(["curl", "-sS", "--fail", "-A", UA, url],
                         capture_output=True, timeout=90)
    if out.returncode != 0:
        raise RuntimeError(f"curl failed for {lang}: {out.stderr.decode('utf-8','ignore')}")
    return json.loads(out.stdout.decode("utf-8"))

def wordcount(text):
    # whitespace-delimited tokens on the cleaned article text
    return len([w for w in text.split() if w.strip()])

meta = {}
for lang, title in LANGS.items():
    # resolve current revision id (to pin), then fetch plain-text extract
    info = api(lang, {"action": "query", "prop": "revisions", "titles": title,
                      "rvprop": "ids", "rvlimit": 1})
    pages = info["query"]["pages"]
    page = next(iter(pages.values()))
    revid = page["revisions"][0]["revid"]
    pageid = page["pageid"]

    ext = api(lang, {"action": "query", "prop": "extracts", "explaintext": 1,
                     "pageids": pageid, "exlimit": 1})
    ptext = ext["query"]["pages"][str(pageid)]["extract"]
    # light cleanup: collapse blank lines
    ptext = re.sub(r"\n{2,}", "\n", ptext).strip()

    fn = os.path.join(DATA, f"{lang}.txt")
    with open(fn, "w", encoding="utf-8") as f:
        f.write(ptext)
    wc = wordcount(ptext)
    meta[lang] = {"title": title, "pageid": pageid, "revid": revid,
                  "chars": len(ptext), "words": wc,
                  "url": f"https://{lang}.wikipedia.org/w/index.php?oldid={revid}"}
    print(f"{lang}: title={title!r} revid={revid} chars={len(ptext)} words={wc}")

with open(os.path.join(DATA, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("saved ->", DATA)
