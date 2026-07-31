# V5 Data Mixture and Curriculum

**ERA V5 · Session 5 submission.** A mixture-and-curriculum specification for V5: a
coding/agentic model with a controllable reasoning dial and native Indic fluency,
trained on a fixed budget of **2.5T tokens**.

Every number below is derived from two files — [`mixture/v5_mixture.yaml`](mixture/v5_mixture.yaml)
(the plan) and [`inventory/datasets.csv`](inventory/datasets.csv) (what exists) — by scripts
in [`scripts/`](scripts/) that regenerate the reports in [`reports/`](reports/). Nothing in this
README is typed by hand. If you disagree with a number, change it in the YAML and re-run;
the consistency checks will tell you what else breaks.

```
python scripts/supply_check.py           # demand vs. real supply -> reports/supply_check.md
python scripts/curriculum_sim.py         # 2.5T-token simulation  -> reports/curriculum_realized.md
python scripts/prepare_proxy_data.py     # lane-tagged shards from the S2 tokenizer + S4 corpus
python scripts/proxy_run.py --config configs/smoke.yaml   # the harness, end to end
python scripts/make_smoke_report.py      # -> reports/smoke_run.md
```

---

## 1. The mixture

Composed backward from [`inventory/benchmarks.yaml`](inventory/benchmarks.yaml): every lane
exists because a benchmark we intend to win requires it, and each lane's share is argued
against what that benchmark needs and what data exists to feed it.

| lane | main run | whole budget | anneal | what it buys | why this size |
|---|---:|---:|---:|---|---|
| web | 39.3% | 38.2% | 12% | MMLU, MMLU-Pro, breadth | the residual; large because breadth is cheap and abundant, and shrinking under the curriculum |
| code | 19.9% | 19.9% | 20% | HumanEval+, MBPP+, LiveCodeBench, SWE-bench | the primary capability; 0.91 epochs of the permissive Stack v2 slice, so it is bought outright |
| math | 13.1% | 13.2% | 15% | GSM8K, MATH-500, GPQA | plentiful (518B usable) and it is where reasoning *shape* is cheapest to buy |
| indic | 12.5% | 12.8% | 20% | MILU, IndicGenBench, FLORES-200 | the differentiator; 1.5x V4's protected 8%, sized to what the four tiers can actually fund |
| longctx | 7.1% | 7.0% | 5% | RULER-128k, LongBench, SWE-bench | a *view* over other lanes, not new tokens; the cost is charged back to its parents |
| agentic | 4.0% | 4.4% | 13% | SWE-bench, Terminal-Bench, BFCL, tau-bench | cut from 5% because the supply behind it is mostly unverified (§5) |
| reasoning | 2.8% | 3.3% | 15% | AIME, MATH-500, LiveCodeBench | small in pretraining *by design*: reasoning is taught later (§6) |
| instruct | 1.3% | 1.2% | 0% | IFEval, format compliance | deliberately tiny; SFT re-teaches this and pretraining tokens spent here are wasted |

The run average is **computed from the curriculum**, not asserted alongside it
([`scripts/lib_mixture.py`](scripts/lib_mixture.py) `Curriculum.run_average`). The stage
schedule is the primary object:

| stage | tokens | seq len | web | code | math | reasoning | indic | agentic | longctx | instruct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s0 seed | 50B | 4k | 72 | 8 | 4 | 2 | **10** | **3** | 0 | 1 |
| s1 foundation | 850B | 4k | 56 | 15 | 9 | 2 | 11 | 3 | 3 | 1 |
| s2 capability | 800B | 8k | 35 | 24 | 18 | 3 | 12 | 4 | 3 | 1 |
| s3 long context | 450B | 32k | 23 | 22 | 12 | 3 | 13 | 5 | 20 | 2 |
| s4 pre-cool | 250B | 32k | 19 | 22 | 15 | 5 | 19 | 6 | 12 | 2 |
| **anneal** | **100B** | 128k | 12 | 20 | 15 | **15** | **20** | **13** | 5 | 0 |

Three deliberate choices in that table:

- **Indic starts at the floor in stage 0 and never enters mid-run.** V4's 150x gradient-norm
  event came from raising a language share into a model that had not been seeing it. A
  capability that is protected at the end must be present at the start.
- **Web falls 72% → 12%, mirroring V4's 70% → 18%.** The trajectory is the point, not the
  endpoint: breadth first, then capability.
- **The anneal is 4% of the budget and carries 15% reasoning and 13% agentic**, against 2.8%
  and 4.0% in the main run. That concentration is only possible if the material survives the
  main run, which is what the reserve (§8) is for.

---

## 2. Supply: what exists, and where the plan is writing cheques

Full table: [`reports/supply_check.md`](reports/supply_check.md). Demand includes the
long-context lane charged back to the lanes it is drawn from (code 45%, web 35%, math 10%,
indic 10%), so no token is counted twice. Supply is `tokens x (1 - overlap) x keep_frac`,
with both fractions stated per row in the inventory so you can argue with the individual
number rather than the total.

| lane | demand | usable supply | planned synthetic | epochs | cap | verdict |
|---|---:|---:|---:|---:|---:|---|
| web | 1016B | 4930B | — | 0.21 | 3.0 | bought outright |
| code | 576B | 633B | — | 0.91 | 3.0 | bought outright, but only just |
| math | 346B | 518B | — | 0.67 | 3.0 | bought outright |
| indic | 338B | 237B | 72B | 1.09 | 2.2 | needs repetition **and** generation |
| agentic | 110B | 75B | 13B | 1.25 | 1.9 | tightest lane in the plan |
| reasoning | 83B | 13B | 55B | 1.22 | 2.7 | **80% of this lane must be generated** |
| longctx | 175B | 439B (view) | 20B | 0.38 | 2.3 | length distribution, not token count, is the constraint |
| instruct | 31B | 6B | 18B | 1.32 | 1.8 | cheap to cut if generation slips |

Epoch caps follow Muennighoff et al. 2023: up to ~4 epochs is close to fresh data for Tier A,
and we tighten to 3.0 for Tier B and 1.5 for Tier C, because repeating machine-generated text
compounds its errors rather than reinforcing facts.

**Two lanes failed this check on the first pass, and the plan changed rather than the
accounting.** Reasoning started at 4.2% of the main run, which needed 5.41 epochs of the
world's open long-CoT supply — over every cap. Agentic started at 5.1% resting on 112B tokens
of execution-free rollouts. Both were cut, and the freed budget went to math, where worked
solutions are abundant and buy much of the same structure.

**Total synthetic commitment: 178B tokens ≈ 16,400 H100-hours ≈ 85 8xH100-node-days**, costed
per lane at the teacher size and rejection-sampling rate each actually needs
([`scripts/supply_check.py`](scripts/supply_check.py) `GEN_MODEL`). This is the number a
reviewer should push hardest on, because it is the largest unfunded item in the plan:

| lane | kept tokens | teacher | samples per kept token | H100-hours |
|---|---:|---:|---:|---:|
| reasoning | 55B | 32B | 4.0x | 11,299 |
| agentic | 13B | 32B | 5.0x | 3,338 |
| indic | 72B | 8B | 1.2x | 1,109 |
| longctx | 20B | 8B | 1.2x | 308 |
| instruct | 18B | 8B | 1.5x | 347 |

The agentic figure counts GPU forward passes only. Each kept trajectory also needs a container
build and a test run; that CPU-side cost, not the GPU cost, is why the agentic reserve is the
riskiest line in this plan.

**Licence exclusion:** `xLAM-function-calling-60k` is CC-BY-NC-4.0 and is excluded from the
production run. It is used only for proxy evaluation. Books3 is excluded from the long-context
lane for the same class of reason.

---

## 3. The Indic lane, in four tiers

**12.5% of the main run = 300B tokens**, plus 20B in the anneal. A headline Indic percentage
without this table is the exact wishful accounting this session exists to prevent.

| tier | share of lane | need | unique supply | epochs | cap | source |
|---|---:|---:|---:|---:|---:|---|
| verified | 15.9% | 47.8B | 29.5B | 1.62 | 4.0 | Sangraha Verified (51.5B Indic, minus 22B fenced for the anneal) |
| unverified | 16.4% | 49.3B | 21.9B | 2.25 | 3.0 | Sangraha Unverified, perplexity-filtered, keep 0.90 |
| translated | 44.0% | 132.2B | 154.2B | 0.86 | 1.5 | Sangraha Synthetic + IndicAlign, keep 0.85 |
| synthetic | 23.7% | 71.2B | 72.0B | 0.99 | 1.5 | generated by us |

Real numbers behind this, from the Sangraha release: **64.3B verified** (51.5B after removing
its 12.8B English), **24.3B unverified**, **162.7B translated/romanised**, 251.3B total across
22 languages. Telugu alone: 3.7B verified, 0.65B unverified, 11.9B translated.

Three things this table forces into the open:

1. **Translated data is 44% of the lane and is capped at 45%.** That cap is a guess about
   translationese, and it is the Indic number I am least confident in — arm A4 of the proxy
   run exists to test it (§9).
2. **Verified data is only 16% of the lane, and it is the tier MILU actually needs.** MILU is
   built from regional and state-level examinations; translated English Wikipedia does not
   contain those answers. If A4 shows verified-heavy wins, the fix is not more translation but
   more OCR and transcription — which is slow, and is the cohort's ongoing cleaning work.
3. **The anneal Indic mixture is 80% verified / 15% synthetic / 5% unverified / 0% translated.**
   Translated text has no business in the highest-leverage 4% of the run.

**Language split within the lane:** hin 24, ben 12, tel 12, tam 11, mar 9, guj 7, kan 7,
mal 6, ori 3, pan 3, urd 3, asm 2, npi 1. Telugu is held one notch above its speaker share
because our cleaned shards are Telugu and we have native-speaker review capacity there.

### A measured correction to Indic token accounting

Session 2 reported our tokenizer's Telugu fertility as **1.99 tokens/word** on clean Wikipedia.
Measured on the real Session 4 web corpus it is **3.62**
([`reports/s4_shard_tokens.json`](reports/s4_shard_tokens.json), 600-doc sample per language).
English moves 1.20 → 2.15 on the same corpus.

That is an 82% error in the direction that matters: **Indic budgets denominated in
Wikipedia-measured tokens understate the real cost by nearly half.** Every Indic number in this
plan is denominated in tokens measured on real corpus text. It also means published
token counts from other tokenizers are not directly comparable to ours, and the supply table
carries that as an unquantified risk.

### What the cohort has actually contributed

Our Session 4 shard: 50,867 documents, 15.4M words, **50.0M tokens measured with our own
tokenizer**. Against a 300B Indic lane that is **0.017%** — and because it is a cleaned subset
of Sangraha `verified/tel`, its net new supply is **zero**. What it adds is provenance: a
documented, decontaminated, PII-scrubbed, deduplicated slice with a reproducible manifest.
The honest read is that the cohort's cleaning work buys *trust in a tier*, not *tokens in a
lane*, and the verified tier is exactly where the plan is thinnest.

---

## 4. The agentic lane, and the difference between having data and having good data

**4.0% of the main run = 97B tokens**, 13B more in the anneal.

The supply looks comfortable and is not. Of ~150B nominal tokens of open agent traces, the
largest single source — `SWE-ZERO-12M-trajectories`, 112B tokens over 12.3M trajectories — is
**execution-free rollouts sampled from a 1.7B model at temperature 1.0**. No container was
built and no test was ever run. It is Tier C by construction: real trajectory *shape*, unknown
trajectory *correctness*. We keep 50% of it after filtering and cap it at 1.5 epochs.

Execution-verified public data is roughly **1.65B tokens** (SWE-Gym, R2E-Gym,
Nemotron-RL-Agentic-SWE-Pivot). The anneal wants 13B. **The agentic reserve cannot be bought
at any price; it must be built** — ~1.5M rollouts in containerised repos with the repo's own
tests as the verifier. That is the single largest engineering commitment in this plan and the
first thing to cut if it slips (fallback: anneal agentic 13% → 7%, backfilled with code).

### The masking rule

Loss is applied to model turns only — plan, tool call, final answer. Never to the user turn,
and **never to tool observations**. Training on an observation teaches the model to invent tool
results instead of calling the tool, which is the one failure that makes an agent useless.

In our generated trajectories, **54.4% of tokens carry loss** and 45.6% are context-only
(measured, [`reports/proxy_data_meta.json`](reports/proxy_data_meta.json), built by
[`scripts/prepare_proxy_data.py`](scripts/prepare_proxy_data.py)). A budget line that counts
agentic tokens without counting the mask overstates the training signal by nearly half. From
a real sample in this repo:

```
mask  <|user|> Find US grants funding confocal imaging and say which labs could buy a 866k instrument...
LOSS  <|plan|> I will search grants for confocal imaging, then pull the awardee records...
LOSS  <|call|> search_grants({"query": "confocal imaging", "page": 1})
mask  <|obs|> {"results": [{"id": "G19494", "inst": "UT Austin", "amount": 3563000}]}
LOSS  <|call|> fetch_record({"id": "G86387"})
mask  <|obs|> {"error": "record not found", "status": 404}
LOSS  <|plan|> That record is missing. I will fall back to the awardee index instead.
LOSS  <|answer|> UT Austin Core Facility holds a confocal imaging award large enough...
```

The failed call and the recovery are the part that cannot be bought from function-calling
datasets: `glaive-function-calling-v2` has 113k samples but only ~0.1B tokens, because a single
call is short. **Sample count and token count rank these datasets differently**, which is why
the inventory carries both (see the two-currency table in
[`reports/supply_check.md`](reports/supply_check.md): IndicAlign is #1 by samples at 74.7M and
#2 by tokens; SWE-ZERO is the reverse).

---

## 5. The reasoning lane and the effort dial

**2.8% of the main run, 15% of the anneal.** Small in pretraining on purpose: long traces mixed
into pretraining do not produce a reasoning model. The base model learns the *shape* of careful
work; the capability is taught in SFT and then in RL with verifiable rewards (Sessions 17-18).
What Session 5 decides is whether the material for those stages will still exist — hence the
16B anneal reserve and the 55B generation commitment.

A reasoning-effort dial only works if the model has seen the whole range, so the lane reserves
a *distribution of trace lengths*, not a token count:

| band | trace budget | share (pretrain) | share (anneal) | example |
|---|---:|---:|---:|---|
| R0 | 0 | 30% | 15% | direct answer, no trace |
| R1 | ≤256 | 30% | 20% | `(11+27)*12 = 456.` |
| R2 | ≤1024 | 22% | 25% | total crates 11+27 = 38; units 38*12 = 456 |
| R3 | ≤4096 | 13% | 25% | solves it, then re-derives by distributing (11*12 + 27*12) and checks the two agree |
| R4 | ≤16384 | 5% | 15% | restates the problem, tries two routes, questions whether "holds 12 units" means capacity or contents, checks magnitude, then answers |

Those are the four bands our own generator emits
([`scripts/prepare_proxy_data.py`](scripts/prepare_proxy_data.py) `make_reasoning`), so the
band definitions are executable rather than illustrative. The anneal shifts mass toward R3/R4
because that is where self-correction lives, and the model can only benefit from it once the
rest of the capability exists.

**The trap:** buying reasoning only in mathematics. The lane requires traces across maths,
code and general problem-solving, or the dial works in one domain and nowhere else. Our 55B
generation budget is allocated 50% maths, 30% code, 20% general, with a 10% Indic-language
slice so reasoning is not English-only.

---

## 6. Difficulty ladder

Within each stage, data is ordered easiest-first. Bands are assigned per document by classifier
score at shard-build time (Session 6 work); the per-stage distribution is declared in
[`mixture/v5_mixture.yaml`](mixture/v5_mixture.yaml) alongside each stage's lane weights, so
difficulty is scheduled with the same explicitness as the mixture:

| stage | D1 | D2 | D3 | D4 |
|---|---:|---:|---:|---:|
| s0 seed | 85% | 15% | — | — |
| s1 foundation | 45% | 40% | 15% | — |
| s2 capability | 15% | 40% | 35% | 10% |
| s3 long context | 5% | 30% | 45% | 20% |
| s4 pre-cool | — | 20% | 45% | 35% |
| anneal | — | 10% | 40% | 50% |

D1 never reappears after s3 and D4 never appears before s2: the ladder is monotone, so a
document's band determines *when* it can be sampled, not just how it is labelled.

| band | web | code | math | indic | agentic |
|---|---|---|---|---|---|
| D1 | encyclopaedia lead paragraph | single function, no imports | two-step arithmetic | children's story, news brief | one tool call, one observation |
| D2 | how-to article | module with tests | GSM8K-style word problem | editorial, exam prep text | three calls, no failures |
| D3 | technical documentation | multi-file change with an API contract | competition algebra | legal/administrative prose, OCR'd textbook | ten calls with one recovery |
| D4 | research paper | repo-scale refactor across languages | olympiad problem needing a lemma | classical literature, code-switched speech | long horizon, dead ends, tool version drift |

---

## 7. Protected floors and OPUS

V4 ran one always-on lane: Indic at 8% of every batch. V5 protects three:

```
indic 10%   agentic 3%   reasoning 2%     = 15% of every batch, outside selector control
```

Selection still operates *within* a protected lane — the best Indic beats the worst Indic — it
simply cannot drive the lane to zero. The curriculum never plans a lane below its own floor
(asserted in `validate()`), so the floor constrains the selector, never the plan.

[`scripts/curriculum_sim.py`](scripts/curriculum_sim.py) simulates all 2,500 steps of 1B tokens,
scoring 1,500 candidate batches per step with an English-heavy proxy. Results
([`reports/curriculum_realized.md`](reports/curriculum_realized.md)):

| lane | planned | floors on + per-lane scoring | floors OFF + V4-style global scoring | floors ON + global scoring |
|---|---:|---:|---:|---:|
| indic | 12.82% | 12.85% | **3.56%** | 10.01% |
| agentic | 4.38% | 4.47% | **0.91%** | 3.05% |
| reasoning | 3.32% | 3.39% | **2.09%** | 2.65% |
| web | 38.20% | 38.03% | 51.38% | 42.56% |

Two findings, one of which changed the design:

1. **Unprotected, an English-heavy selector destroys the project's reason to exist**: Indic
   collapses 72% (232B tokens), agentic 80% (89B tokens). This is not a hypothetical — it is
   what an aggressive selector does when its proxy is built from what the model is already good
   at.
2. **Floors alone are not enough.** With floors on but V4's global scoring, Indic lands at
   10.01% — *exactly* the floor. The selector eats everything above it, so the floor silently
   becomes the ceiling, and the plan's 12.5% quietly becomes 10%. V5 therefore adds **per-lane
   score normalisation**: the selector ranks Indic against Indic, not against English. With
   both, the realised mixture matches the plan to within 0.05 points.

The cost of per-lane normalisation is real and worth stating: it removes the selector's ability
to *reallocate budget between* lanes, keeping only its ability to pick the best data within
one. We are trading some of OPUS's adaptivity for a mixture we can actually predict, and arm
A3 measures what that trade costs.

Floor violations across the simulated run: **0**.

---

## 8. The anneal reserve

104B tokens fenced by manifest flag (`reserve=anneal`) and excluded from main-run sampling, so
"never spent early" is structural rather than a matter of discipline.

| lane | reserve | material | fundable today? |
|---|---:|---|---|
| indic | 22B | Sangraha Verified, native-speaker reviewed | yes — 53B Tier A on hand |
| code | 21B | quality-filtered Stack v2 + CommitPackFT patch pairs | yes |
| math | 16B | MegaMath-Pro / FineMath-4+ top decile | yes |
| reasoning | 16B | self-distilled traces with checkable answers | **only by building it** |
| agentic | 14B | execution-verified rollouts | **only by building it** |
| web | 13B | top-decile FineWeb-Edu / Nemotron-CC-HQ | yes |
| longctx | 6B | PG-19, arXiv full texts, synthesised multi-doc tasks | yes |

Reserve selection rule: highest quality-score decile within Tier A, deduplicated against the
main run, decontaminated against every eval in `benchmarks.yaml`, and never sampled before the
anneal. Arm A6 of the proxy run tests whether holding it back beats spending it uniformly — if
it does not, this entire mechanism should be deleted rather than defended.

---

## 9. Keeping the run alive through mixture transitions

V4: a sudden Hindi-share increase into frozen embeddings raised the gradient norm ~150x. We use
that single event to calibrate a toy model (`spike ≈ 1 + K · ΔL1 · amp / band`) and then *size
each warmup band from the size of its own transition* rather than using a constant:

| transition | L1 change | largest move | band | hard step, frozen emb. | with band, live emb. |
|---|---:|---|---:|---:|---:|
| s0 → s1 | 0.320 | web 72→56% | 25B | 299x | 1.99x |
| s1 → s2 | 0.420 | web 56→35% | 35B | 392x | 1.93x |
| s2 → s3 | 0.400 | longctx 3→20% | 35B | 374x | 1.89x |
| s3 → s4 | 0.240 | longctx 20→12% | 20B | 225x | 1.93x |
| s4 → anneal | 0.360 | reasoning 5→15% | 30B | 336x | 1.93x |

Kill rule: gradient norm above **4x** the 1000-step rolling median → halt, roll back, re-enter
the transition with a 2x longer band. Warn at 2x. Embeddings stay live through every transition.
A constant 20B band — the first version of this spec — put the s1→s2 transition at 4.26x,
tripping our own halt rule; that is why the bands are per-transition.

---

## 10. This is a hypothesis. Here is the experiment that could refute it.

**1B parameters, 30B tokens, six arms** ([`configs/proxy_1b.yaml`](configs/proxy_1b.yaml)).
Cost is computed, not guessed: `6ND = 6 · 1e9 · 3e10 = 1.8e20` FLOPs/arm; at ~400 TFLOP/s
achieved on 8×H100 that is **~15.6 h/arm, ~4 node-days for the set**.

| arm | isolates | pre-registered decision rule |
|---|---|---|
| A1 proposed | the plan as written | reference arm |
| A2 web-heavy 70% | the naive baseline | A1 must beat A2 by **≥3.0 pts MILU** while losing **≤1.5 pts MMLU** and **≤1.0 pt HumanEval+** |
| A3 no floors | what the floors buy | if A3 ≈ A1 on Indic/agentic, the floors are ceremony and should go |
| A4 verified-heavy Indic (40/25/20/15) | the translated-vs-verified question | if A4 beats A1 by **>2.0 chrF++** on FLORES en→te, shift the tier split and buy OCR instead of translation |
| A5 agentic Tier A only (1.5%) | whether 112B of execution-free rollouts earn their budget | if A5 ≥ A1 on BFCL, cut the agentic lane and re-plan the reserve |
| A6 no reserve | whether holding back the anneal reserve is worth it | A1 − A6 must be **≥2.0 pts** averaged over MILU/HumanEval+/GSM8K, else delete the reserve mechanism |

Then **3B/60B for the two survivors only** ([`configs/proxy_3b.yaml`](configs/proxy_3b.yaml),
~4 node-days each): the question there is not which mixture wins but whether the 1B ranking
reproduces. A flip means the 1B round was measuring noise.

Kill criteria for any arm: gradient norm >4x rolling median for >200 steps; any lane's realised
share drifting >2 points from target for 1B tokens; held-out loss rising on a protected lane
for two consecutive evals.

### What we ran here

A full 1B run needs a GPU node this submission does not have. What *is* here is the harness
that would run it, executed end to end at laptop scale on real data
([`configs/smoke.yaml`](configs/smoke.yaml), results in
[`reports/smoke_run.md`](reports/smoke_run.md)): ~5.8M parameters, 12M tokens per arm, on Telugu
and English from our own Session 4 corpus plus code from the local environment and generated
math/reasoning/agentic lanes. It exercises the parts that carry the argument — curriculum
blending, RHO-loss-style selection against a deliberately English-heavy reference, protected
floors, loss masking on tool observations, per-lane bits-per-byte evaluation, gradient-norm
monitoring — at a scale where the answer is about the machinery, not about V5. The same file
runs `proxy_1b.yaml` unchanged.

Three arms, 2,929 steps each, ~10 minutes per arm on an M-series GPU. Held-out bits per byte,
lower is better:

| arm | web | code | math | reasoning | indic | agentic | realised indic | realised agentic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A proposed (floors on) | 2.449 | 2.779 | **2.000** | **1.151** | **1.151** | **0.471** | **10.3%** | **3.0%** |
| B no floors | 2.370 | 2.630 | 2.783 | 2.484 | 1.260 | 2.713 | 4.1% | 0.5% |
| C web-heavy, no floors | 2.363 | 2.691 | 3.474 | 3.158 | 1.327 | 3.418 | 1.8% | 0.2% |

**The floors survived contact with a real training loop.** Arm A lands at 10.3% Indic and 3.0%
agentic — its floors, to a tenth of a point — while the same curriculum and the same selector
with floors removed collapses to 4.1% and 0.5%. This is the §7 simulation reproduced on a model
that actually trained: an English-heavy proxy direction does not merely *under-weight* the scarce
lanes, it drives them toward zero, and arm C shows a naive web-heavy preset takes agentic to 0.2%.

**The trade-off is real, and arm A pays for it.** A is worse than B on web (+0.079 bpb) and code
(+0.149) — the tokens spent on protected lanes came out of the abundant ones, exactly as §2
argues they must. What the plan claims is that the exchange rate is favourable: A gives up 0.08
bpb of web to gain **2.24 bpb of agentic** and 1.33 of reasoning. At this scale that trade is
lopsided enough to be worth stating; at 1B it is the thing arms A2 and A3 exist to measure
properly.

**One result went against the spec, and it is the useful one.** The §9 kill rule — halt above 4x
the 1000-step rolling median — fired 12 times in arm A and 22 in arm B, with peaks of 7.1x and
12.0x. Nothing diverged; the runs finished cleanly. The rule is simply too tight for 4,096-token
steps, where the rolling median is dominated by noise rather than by distribution shift. **The
kill rule as written would have halted a healthy run**, so it needs a minimum-batch qualifier and
a persistence requirement (the 1B config's ">200 consecutive steps") before it is trusted to stop
anything expensive. A monitoring rule is a hypothesis too, and this one failed its first test.

Two caveats a reviewer should hold against all of the above. The realised code share reaches
~40% against a planned ~20%: with only six lanes and four candidate batches per step, the
selector distorts the mixture far more than it would at production candidate counts, and the math
lane is starved to ~0.5% as a result. And selector overhead is 34–37% here against the 3% assumed
in the plan, because a scoring pass is a fixed cost per step and these steps are tiny.

---

## 11. Where this plan is weakest

Stated plainly, because a reviewer will find them anyway:

1. **178B tokens of generation is the plan's largest unfunded commitment**, and 55B of it is
   rejection-sampled reasoning from a 32B teacher (~11.3k H100-hours). If that budget is not
   granted, reasoning drops to 1.5% of the main run and the anneal reasoning share halves.
2. **The agentic anneal reserve cannot be bought.** 1.65B tokens of execution-verified public
   data against a 14B need. If the container/test harness is not ready, anneal agentic goes
   13% → 7%.
3. **Reasoning-trace token counts in the inventory are low-confidence.** They are sample counts
   times an assumed tokens-per-sample; the whole reasoning row could be off by 2x in either
   direction. It is flagged `confidence: low` in the CSV and is the first thing to measure.
4. **Cross-tokenizer token counts are not comparable.** Published counts (Sangraha, MegaMath,
   Stack v2) come from other tokenizers; our own fertility measurement moved 82% between clean
   and real text. Treat every supply number as ±20% until re-counted with the production
   tokenizer.
5. **The 45% translated cap is a judgement call**, not a measurement. Arm A4 exists because of
   it.
6. **The gradient-norm model has one calibration point.** It sizes warmup bands; it does not
   predict dynamics. The first real transition either confirms it or replaces it.
7. **The 4x gradient-norm kill rule is mis-specified**, and the smoke run proved it: it fired 34
   times across three healthy arms that all finished cleanly (§10). It needs a minimum-batch-size
   qualifier and the ">200 consecutive steps" persistence requirement that only the 1B config
   currently carries. Until that is fixed the rule would halt good runs, which is the expensive
   direction to be wrong in.

---

## 12. Layout

```
mixture/v5_mixture.yaml     the plan: budget, stages, floors, reserve, tiers, bands
inventory/datasets.csv      every dataset, with tokens, samples, licence, tier, overlap, keep-rate
inventory/benchmarks.yaml   what each benchmark measures, its training shape, and where loss goes
configs/                    smoke.yaml (runs here), proxy_1b.yaml, proxy_3b.yaml (the real experiment)
scripts/lib_mixture.py      loader, curriculum with warmup blending, floors, OPUS selector, validation
scripts/supply_check.py     demand vs supply, epochs, generation cost
scripts/curriculum_sim.py   2.5T-token simulation, floors on/off, transition sizing
scripts/proxy_run.py        the training harness (torch), all three scales
scripts/prepare_proxy_data.py   builds lane-tagged shards incl. masked agentic trajectories
scripts/make_smoke_report.py    turns a run's results.json into reports/<tag>_run.md
scripts/measure_s4_shard.py     measures our Session 4 corpus in real tokens
reports/                    everything above, generated
```

Session 4 (cleaning, deduplication, provenance) is in
[`../session4_data_cleaning_dedup/`](../session4_data_cleaning_dedup/); this plan consumes its
manifest and its measured tokens.
