# How Attention Works Now — a chronological timeline

**ERA V5 · Session 8 · Modern Attention Variants**

Every attention mechanism from the session, in the order it was **actually launched** — 2014 to
2026 — with an honest trade-off for each and a stated source for every date.

- **Live app:** **https://era-v5-attention-timeline.netlify.app**
- **Repo:** https://github.com/amukka/ERA_V5/tree/main/session8_ModernAttentionVariants
- **Timeline page:** `site/index.html` — the submission
- **Interactive companion:** `site/session8.html` — 15 live widgets that build the mechanisms from
  first principles, every number computed in the browser

---

## Why chronological

Vanilla attention was never wrong. It was **expensive**. Everything that follows is somebody
looking at that bill and trying to pay less of it, and each one pays with something different.
Laid out by date, the field visibly changes its mind four times:

| Act | Span | What it wants |
|---|---|---|
| I | 2014–2017 | **exactness** — attention is invented, exact, and nobody minds the cost |
| II | 2019–2021 | **the bill** — read fewer keys, or delete softmax for a fixed state |
| III | 2021–2024 | **length** — position becomes the wall; rotations, then rescaled rotations |
| IV | 2023–2026 | **memory back** — serving cost binds; compress heads, features, sequence |

Two things only visible from a timeline:

1. **Teaching order ≠ historical order.** Segment recurrence (Jan 2019) predates RoPE (Apr 2021).
   MQA (Nov 2019) predates everything in Acts II–III and was then ignored for **four years** until
   serving costs made it urgent.
2. **The field reverses twice.** It trades exactness for cost in 2019–2021, then FlashAttention
   (2022) makes exactness cheap again and retires a chunk of that work.

---

## Date methodology

**One rule: the date is the arXiv v1 submission date**, read off the abstract page. Not the
conference date, not the journal date, **not a later revision**, not a blog post about the paper.

- For non-paper releases (Qwen3-Next, DeepSeek-V3.2-Exp, NTK-aware scaling) the public release or
  post date is used, the source is named, and the entry is flagged if the exact day is uncertain.
- **Every date was fetched individually from its primary source.** None came from memory.
- Where an abstract page did **not** confirm a mechanism detail, the entry says so in its
  `verified` / `caveat` fields rather than quietly asserting it.
- **Ordering is checked by code**, not by eye — `site/js/timeline.js` asserts ascending date order
  at load and prints the result on the page.

Run the verification harness:

```bash
npm install          # jsdom, dev-only
npm test             # exercises the interactive widgets (495 interactions)
node /tmp/tl-smoke.mjs   # see "Checks" below
```

---

## Sources for every date

`v1` = arXiv v1 submission date, confirmed from the arXiv abstract page unless noted.

| # | Date | Mechanism | Source |
|---|---|---|---|
| 1 | 2014-09-01 | Additive (Bahdanau) attention | [arXiv:1409.0473](https://arxiv.org/abs/1409.0473) |
| 2 | 2017-05-08 | Learned absolute positions (ConvS2S) | [arXiv:1705.03122](https://arxiv.org/abs/1705.03122) |
| 3 | 2017-06-12 | Scaled dot-product + multi-head + sinusoidal | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| 4 | 2019-01-09 | Transformer-XL segment recurrence | [arXiv:1901.02860](https://arxiv.org/abs/1901.02860) |
| 5 | 2019-04-23 | Sparse Transformer | [arXiv:1904.10509](https://arxiv.org/abs/1904.10509) |
| 6 | 2019-11-06 | Multi-Query Attention (MQA) | [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) |
| 7 | 2019-11-13 | Compressive Transformer | [arXiv:1911.05507](https://arxiv.org/abs/1911.05507) |
| 8 | 2020-01-13 | Reformer (LSH) | [arXiv:2001.04451](https://arxiv.org/abs/2001.04451) |
| 9 | 2020-04-10 | Sliding window (Longformer) | [arXiv:2004.05150](https://arxiv.org/abs/2004.05150) |
| 10 | 2020-06-08 | Linformer (low-rank) | [arXiv:2006.04768](https://arxiv.org/abs/2006.04768) |
| 11 | 2020-06-29 | Linear attention ("Transformers are RNNs") | [arXiv:2006.16236](https://arxiv.org/abs/2006.16236) |
| 12 | 2020-07-28 | BigBird | [arXiv:2007.14062](https://arxiv.org/abs/2007.14062) |
| 13 | 2020-09-30 | Performer (FAVOR+) | [arXiv:2009.14794](https://arxiv.org/abs/2009.14794) |
| 14 | 2021-02-22 | Delta rule for fast weights | [arXiv:2102.11174](https://arxiv.org/abs/2102.11174) |
| 15 | 2021-04-20 | RoPE (RoFormer) | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| 16 | 2021-08-27 | ALiBi | [arXiv:2108.12409](https://arxiv.org/abs/2108.12409) |
| 17 | 2022-05-27 | FlashAttention | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |
| 18 | 2023-05-22 | Grouped-Query Attention (GQA) | [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) |
| 19 | 2023-06-27 | Position Interpolation | [arXiv:2306.15595](https://arxiv.org/abs/2306.15595) |
| 20 | 2023-06-28 **≈** | NTK-aware scaled RoPE | [r/LocalLLaMA, bloc97](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) — **no paper**, see caveats |
| 21 | 2023-08-31 | YaRN | [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) — **v1, not the Nov v2** |
| 22 | 2023-09-29 | Attention sinks (StreamingLLM) | [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) |
| 23 | 2023-10-03 | Ring Attention | [arXiv:2310.01889](https://arxiv.org/abs/2310.01889) |
| 24 | 2023-10-10 | Sliding window at scale (Mistral 7B) | [arXiv:2310.06825](https://arxiv.org/abs/2310.06825) |
| 25 | 2023-12-01 | Mamba (selective SSM) | [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) |
| 26 | 2024-02-21 | LongRoPE | [arXiv:2402.13753](https://arxiv.org/abs/2402.13753) |
| 27 | 2024-05-07 | Multi-head Latent Attention (MLA) | [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) |
| 28 | 2024-06-10 | DeltaNet parallelised over sequence | [arXiv:2406.06484](https://arxiv.org/abs/2406.06484) |
| 29 | 2024-10-07 | Differential attention | [arXiv:2410.05258](https://arxiv.org/abs/2410.05258) |
| 30 | 2024-12-09 | Gated DeltaNet | [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) |
| 31 | 2025-01-14 | Lightning attention at scale (MiniMax-01) | [arXiv:2501.08313](https://arxiv.org/abs/2501.08313) |
| 32 | 2025-02-16 | Native Sparse Attention (NSA) | [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) |
| 33 | 2025-09-11 **≈** | Qwen3-Next 3:1 hybrid | [HF model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) · [vLLM day-0](https://vllm.ai/blog/2025-09-11-qwen3-next) |
| 34 | 2025-09-29 | DeepSeek Sparse Attention (DSA) | [DeepSeek announcement](https://api-docs.deepseek.com/news/news250929/) · [HF](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) |
| 35 | 2025-12-13 | **DroPE** | [arXiv:2512.12167](https://arxiv.org/abs/2512.12167) · [Sakana AI](https://sakana.ai/drope/) |
| 36 | 2026-04-03 | Olmo Hybrid | [arXiv:2604.03444](https://arxiv.org/abs/2604.03444) |
| 37 | 2026-05-27 | Periodic RoPE | [arXiv:2605.27980](https://arxiv.org/abs/2605.27980) |

**≈** = day approximate, month reliable.

### Beyond the required list

The assignment named 18 mechanisms. These 19 were added because the story does not work without
them: Bahdanau attention, Transformer-XL, Compressive Transformer, Reformer, Linformer, BigBird,
Performer, **FlashAttention**, Position Interpolation, Ring Attention, Mistral 7B SWA, **Mamba**,
LongRoPE, DeltaNet parallelisation, **Differential attention**, MiniMax-01, **Qwen3-Next**, Olmo
Hybrid, Periodic RoPE.

Three earn their place emphatically:

- **FlashAttention (2022)** is the only entry that costs nothing. It made *exact* attention so much
  cheaper that it retired years of approximate work. Its lesson: check whether a mechanism is
  actually bound by the thing you are reducing — three years of approximate-attention research
  targeted FLOPs when the bottleneck was memory traffic.
- **Position Interpolation (Jun 2023)** is the missing link between RoPE and NTK/YaRN. Without it
  the extension thread starts mid-sentence.
- **Differential attention (2024)** attacks a *quality* problem, not a cost problem — a useful
  reminder that not every issue with attention is a bill.

---

## Three date corrections

### 1 · "DroPE" is three different things

| | What | Relevance |
|---|---|---|
| ✅ | [arXiv:2512.12167](https://arxiv.org/abs/2512.12167) — Gelberg, Eguchi, Akiba, Cetin (**Sakana AI**), 13 Dec 2025. "Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings" | **This is the one Session 8 meant.** Pretrain with RoPE, remove positional embeddings, briefly recalibrate at the original context length. |
| ❌ | [arXiv:2503.15029](https://arxiv.org/abs/2503.15029) — 19 Mar 2025. "**DRoPE**: Directional Rotary Position Embedding for Efficient Agent Interaction Modeling" | Autonomous driving. Encodes *vehicle headings*. Nothing to do with context length. A name collision that will mislead a careless search. |
| ❌ | [arXiv:2606.07404](https://arxiv.org/abs/2606.07404) — "Reversible Foundations: Training a 120B Sparse MoE" | A **web search told me this paper describes DroPE**, with a plausible mechanism summary. I fetched it. It **does not mention DroPE at all.** |

That third row is the exact failure mode from the Session 7 warning — and it came from a *search
summary*, which is worse than a model guess because it reads as sourced.

### 2 · YaRN is August 2023, not November

arXiv `2309.00071` submission history: **v1 31 Aug 2023**, v2 1 Nov 2023, v3 6 Feb 2026. Secondary
sources routinely cite November — the revision. This matters for **ordering**: v1 puts YaRN *before*
StreamingLLM (29 Sep 2023). Using v2 would have mis-sequenced two entries in a submission whose
entire premise is the sequence. Hence the v1-only rule.

### 3 · One date genuinely cannot be pinned

**NTK-aware scaled RoPE has no paper.** It is an `r/LocalLLaMA` post by **bloc97**, with a
Dynamic-NTK follow-up by **emozilla**, confirmed as influencing CodeLlama and Qwen. There is no v1
stamp. Placed at 2023-06-28 and flagged **≈**; its ordering against PI (27 Jun) and YaRN (31 Aug) is
reliable, the exact day is not. Qwen3-Next's day is likewise taken from vLLM's day-0 post.

---

## Two things for the instructor to check

1. **The V4 → DroPE → 256K claim has no public source I could find.** DroPE is real and published
   (above). But `arXiv:2606.07404`, the only LightningLM paper I found, reports **8K context** and
   mentions **no** DroPE, DeltaNet layers, DDDGDDDG, Memory Stream, or sparse-attention budgets.
   The *mechanism* is verified; the *combination* is not.
2. **A suspicious number coincidence.** That paper reports "a released training loss of **1.78** at
   120B scale". The session notes describe a "**1.78B** seed model". Same number, two very different
   roles.

## One thing the timeline found

**The 3:1 hybrid ratio may be converging.** V4's `DDDGDDDG` is 6 D and 2 G per 8 = **3:1**.
Qwen3-Next ships 48 layers as 36 Gated DeltaNet + 12 Gated Attention = **3:1**. Two independent
teams, one exact-access layer per four. Not an ablation, but stronger support than either team's own
report — and it partially answers the §16 open question about whether that ratio was arbitrary.

---

## Repo layout

```
session8_ModernAttentionVariants/
├── README.md                  ← this file: dates and sources
├── package.json               ← npm test, npm run deploy
├── session8.txt               ← session transcript (§17 truncated in capture)
└── site/                      ← Netlify publish root
    ├── index.html             ← THE SUBMISSION: chronological timeline
    ├── session8.html          ← interactive companion, 15 widgets
    ├── netlify.toml
    ├── data/mechanisms.js     ← all 37 entries: dates, sources, trade-offs
    ├── css/app.css
    ├── smoke.mjs              ← headless harness for the widgets
    └── js/
        ├── timeline.js        ← renders the timeline, asserts date order
        ├── lib.js             ← DOM + matrix helpers
        ├── costs.js           ← shared cache/compute cost model
        └── widgets/           ← s02 … s16
```

## Checks

| Check | Result |
|---|---|
| Chronological order (asserted in code) | **PASS** — 37 entries ascending |
| All 18 required mechanisms present | **PASS** |
| Every entry has problem / answer / pros / cons / pick / source | **PASS** |
| Widget harness (`npm test`) | **PASS** — 15 widgets, 495 interactions |
| Console errors on either page | **none** |

## Honest limits

I verified **titles, authors, dates and whatever the abstract explicitly stated**. I did *not*
re-read every full paper, so the trade-off write-ups draw on abstracts, project pages and prior
knowledge. **If a trade-off here contradicts a paper's body, the paper wins** — that is the half of
this I could not mechanically check, and I would rather say so than imply a uniform depth of
verification I did not achieve.

Section 17 of the source transcript was truncated mid-sentence during capture and is not fully
represented.
