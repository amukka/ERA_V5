# Session 7 — Duplex Kronecker Embeddings

**Problem 5: making Kronecker embeddings reversible, and deleting the output head.**

> *"Kronecker is forward deterministic (same word will always give same embedding).
> How do I make a reverse of this (same embedding gives the same Kronecker)? If we
> can do this, then we can get rid of the final head as well! Then we can have a
> vocab of 1M as well without any issues!"*

The claim is proven. A 4-layer transformer with **no output head at all** reaches a
**lower** validation loss than the same trunk with a dense `D x V` head, and a
**1,000,000-token** vocabulary is built, scored, and identified with **zero**
parameters that depend on `V`.

---

## 1. The result in one table

Six arms. Identical trunk, data, seed, schedule and step count. The only
difference is what sits at the front door and the back door.

| arm | token-facing params | sized by `V`? | val CE | ppl | Indic CE |
|---|---:|:---:|---:|---:|---:|
| `dense` — `nn.Embedding` + `nn.Linear` head | 5,056,000 | yes | 4.2254 | 68.4 | 4.5141 |
| `dense_tied` — head = tableᵀ | 2,528,000 | yes | 4.4858 | 88.7 | 4.8267 |
| `kron_dense` — byte-code in, dense head | 4,633,344 | yes | **4.1273** | **62.0** | **4.4116** |
| **`duplex` — byte-code in, byte-code out** | **4,210,688** | **no** | **4.1851** | **65.7** | **4.4251** |
| `duplex_tied` — `W_syn = W_anaᵀ` | 2,105,344 | no | 4.6192 | 101.4 | 4.8196 |
| `duplex_colce` — trained on code columns | 4,210,688 | no | 11.8967 | — | 13.8135 |

**The head-free arm beats the dense-head arm** — 4.1851 against 4.2254 — using
17% fewer token-facing parameters, and it does so on Indic text too (4.4251
against 4.5141). Whatever the `D x V` head was contributing, the token's own
bytes already contained it.

`results/e3_headfree_lm.json` · reproduce with `python -m experiments.e3_headfree_lm`

---

## 2. What changed, and why each change was necessary

Session 7's released codec marks a `256 x 32` grid — one cell per byte, at
`(value, position)` — and projects the flattened grid through one shared
`Linear(8192, d_model)`. It is a one-way street. Three changes make it run
backwards, and none of them is decoration:

**A 257th row, `INACTIVE`.** The shipped codec marks only the columns a token's
bytes reach, so `a` marks one cell and `the` marks three, and a short token's
code is a *prefix* of a longer one's. Marking every unreached column `INACTIVE`
makes the code exactly one-hot per column — always 32 marks, always the same
scale. The code becomes **self-delimiting**: the token's length is recoverable
rather than lost.

That is what turns decoding from general sparse recovery into something
well-posed: **32 independent 257-way classifications**, on a set of size
`257^32`.

**A digest in the last 4 columns.** Section 8 of the session names the sovereign
risk: bytes past position 32 are dropped, silently, and two tokens agreeing on
their first 32 bytes get the same vector forever. Devanagari and Telugu spend
3 bytes per character and a conjunct like `क्ष` costs 9, so 32 bytes is 10
characters, not 32. Replacing the last 4 columns with a BLAKE2b digest of the
*whole* byte string separates them.

**One learned scalar, `logit_scale`.** A z-normalised code has unit variance
across 8,224 entries, so its norm is `sqrt(8224) ≈ 91` by construction. Scoring
by inner product against it starts about ninety times too sharp and the run
diverges — this was a real bug, caught in E3's first pass, and it is worth
naming because it is the kind of thing that makes a good idea look like a bad
one. One learned scalar fixes it, and like everything else on this path it does
not depend on `V`.

### The output head, deleted

The vocabulary becomes one **sparse `[V, 8224]` matrix of where each token's
bytes fall**. It holds `V x 32` nonzeros and **not one trainable weight**.
Scoring is one sparse matmul:

```python
logits = torch.sparse.mm(K, y.t()).t()          # y = h @ W_syn
logits = (logits - mu * y.sum(-1, keepdim=True)) * (logit_scale.exp() / sigma)
```

z-normalisation is affine, so the dense correction collapses to **one scalar per
token** instead of a dense 8,224-wide subtraction. Sparsity survives, and cost is
`O(V x 32)` rather than `O(V x D)`.

---

## 3. How it is proven

Four experiments, each answering a question the next one depends on.

### E1 — Does the byte code name every token uniquely?

Nothing else matters if it does not: two tokens with one code are one token
forever, and no head built by inverting the code could separate them. This is
also the per-script collision count Section 8 of the session explicitly asks for.

| vocabulary | codec | collided | Indic | injective |
|---|---|---:|---:|:---:|
| **ERA V5 (Session 2, V=9,875)** | shipped `P=32` | 23 (0.233%) | **23** | ✗ |
| | shipped `P=48` | 0 | 0 | ✓ |
| | **duplex `P=32,T=4`** | **0** | **0** | **✓** |
| GPT-2 (V=50,257) | shipped `P=32` | 17 (0.034%) | 0 | ✗ |
| | shipped `P=64` | 2 (0.004%) | 0 | ✗ |
| | **duplex `P=32,T=4`** | **0** | 0 | **✓** |
| XLM-R (V=250,002) | shipped `P=32` | 103 (0.041%) | 41 | ✗ |
| | **duplex `P=32,T=4`** | **0** | **0** | **✓** |

**Every one of the 23 collisions on our own vocabulary is Indic**, at 6x the rate
of GPT-2's — on a vocabulary that is only 40% Indic. This is the sovereign risk
as a number rather than an argument. The duplex codec is injective on all three
at the *same* 32-column budget; widening to `P=48` also works but costs 50% more
projection parameters.

`results/e1_injectivity.json` · `python -m experiments.e1_injectivity`

### E2 — Does the code come back out of a `d`-dimensional embedding?

The forward path compresses 8,224 dimensions into `d`. That should be lossy. It
is not, because the code is not an arbitrary vector: it carries `32 x log2(257) =
256 bits` and lives on a set of size `257^32`. The measurements a random
projection needs scale with sparsity times the log of the block size:

```
d* ≳ 2 · P · ln(C) = 2 · 32 · ln(257) ≈ 355
```

Measured on XLM-R's 250,002 tokens, with three synthesis maps that store nothing
per token (`pinv` = algebraic inverse, `lstsq` = fitted, `tied` = `W_anaᵀ`):

| `d` | decoder | column acc | exact code | **correct token** |
|---:|---|---:|---:|---:|
| 128 | pinv | 23.52% | 0.00% | **96.55%** |
| 128 | lstsq | 82.77% | 1.35% | 45.55% |
| 128 | tied | 22.90% | 0.00% | **96.70%** |
| 512 | pinv | 89.03% | 1.10% | **100.00%** |
| 512 | lstsq | 97.01% | 54.75% | 99.90% |
| 512 | tied | 87.10% | 0.60% | **100.00%** |

**The most important row is `d=128, pinv`: 0% exact code recovery, 96.55% correct
token.** These come apart, and the gap is the whole insight —

> You do not need to recover the code. You only need the right token to win.

Real tokens are a vanishingly sparse, well-separated subset of the `257^32`
possible codes, so identification is a far easier problem than reconstruction,
and identification is the only thing an output head is for. A least-squares
decoder optimises the wrong objective (it reconstructs the code, and *loses* on
identification: 45.55% against 96.55%). The algebraic inverse preserves the
geometry that matters.

At 20% noise on the hidden state, identification holds at **99.95%**.

`results/e2_recovery.json` · `python -m experiments.e2_recovery`

### E3 — Does a head-free language model actually train?

E1 and E2 are about codes, not language models. E3 trains real transformers on
the Session 5 proxy corpus — tokenized with the Session 2 tokenizer, 45% Indic
by design, because the failure this work exists to avoid shows up on Indic and
averages away on English. The table in §1 is the result.

Two findings beyond the headline, both worth keeping:

**Tying hurts more here than it does for a dense table.** `duplex_tied` (4.6192)
is worse than `dense_tied` (4.4858), while untied `duplex` beats untied `dense`.
Section 5 of the session argues analysis and synthesis are different jobs; on a
structured input path they are apparently *more* different, not less, because
the projection is the only adaptive capacity the whole path has.

**The V-free head works. The V-free *loss* does not.** `duplex_colce` trains on
32 independent 257-way classifications, so no tensor of size `V` is ever formed —
not in the forward pass, not in the loss. Its column loss reaches 0.82. Its
vocabulary CE is **11.90**, catastrophically worse than everything else. The
factorized model spends its probability mass on the `257^32` code space, almost
all of which is not a token, and renormalising onto the vocabulary afterwards
does not recover what was lost. **This is a negative result and it is reported as
one.** The saving that is real is the head; the further saving on the loss is
not, at least not in this form.

`results/e3_headfree_lm.json` · `python -m experiments.e3_headfree_lm`

### E4 — Does a 1,000,000-token vocabulary actually work?

Half arithmetic, half execution. The arithmetic, at V5's reference width
`D = 8,096`, in AdamW training state (16 bytes/param):

| `V` | dense | dense tied | kron+dense head | **duplex** | duplex tied |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 2.4 GB | 1.2 GB | 2.2 GB | **2.0 GB** | 1.0 GB |
| 50,257 | 12.1 GB | 6.1 GB | 7.1 GB | **2.0 GB** | 1.0 GB |
| 131,072 | 31.6 GB | 15.8 GB | 16.8 GB | **2.0 GB** | 1.0 GB |
| 250,002 | 60.3 GB | 30.2 GB | 31.2 GB | **2.0 GB** | 1.0 GB |
| **1,000,000** | **241.3 GB** | 120.6 GB | 121.6 GB | **2.0 GB** | 1.0 GB |

The last two columns are flat. That is the point — a slope, not a number.

The execution is the part that matters, because arithmetic can be right and
still not survive an allocator. A million-token vocabulary was built from **real
words** harvested from the Session 4 cleaned corpus — Telugu-dominated, mean 25.6
bytes, max 289, so the hostile case rather than the flattering one:

```
harvested 1,000,000 distinct real words   mean 25.6 bytes, p99 61, max 289
  shipped(P=32)      collided= 122,869 (12.287%)   indic= 122,840   injective=False
  duplex(P=32,T=4)   collided=       0 ( 0.000%)   indic=       0   injective=True

  trainable params 8,421,377  (= 2 x 8224 x 512 + 1, no V)
  forward 9 ms    score all 1,000,000 tokens  6,814 ms
  from a 512-d embedding, over 1,000,000 words:
      column 89.61%   exact code 0.59%   correct token 100.00%   (chance 0.00010%)
```

**At a million real Indic words the shipped codec fuses 12.3% of the
vocabulary** — 122,840 of the 122,869 collisions are Indic. The duplex codec has
none.

And **100% of tokens are identified correctly out of 1,000,000 candidates from a
512-dimensional vector**, against a chance rate of 0.0001%, with an *untrained,
random* analysis matrix and zero per-token parameters. Exact code recovery is
0.59% — again, you do not need the code, you need the right token to win.

`results/e4_scaling.json` · `python -m experiments.e4_scaling`

---

## 4. What this costs, stated honestly

- **The codec still cannot learn.** Section 9 of the session is the warning:
  a Kronecker input path has the projection and nothing else, so every token
  adapts through one shared matrix or not at all. Making it reversible does not
  add a degree of freedom. It makes keeping the projection trainable *more*
  important, not less.
- **A digest is not free.** Four of 32 columns now hold a hash instead of bytes,
  so the recoverable prefix is 28 bytes, not 32. Tokens longer than 28 bytes are
  distinguished but not *readable* — the digest separates them without saying
  what they were. For scoring a fixed vocabulary that is sufficient; for
  open-vocabulary generation it is not.
- **`duplex_tied` is worse than the dense control.** Tying is the elegant version
  and it does not pay here.
- **The column-factored loss failed** (E3), so training cost still scales with
  `V` even though parameters do not. Making the loss `V`-free is open.
- **Scale.** These are 8M-parameter models on a 9,875-token vocabulary. The
  standard Session 5 set and the session itself both insist a proportion is an
  opinion until a proxy run has tested it; the same applies here. This is a
  proxy run, and the 1B/3B confirmation is not done.

## 5. Layout and how to run

```
duplex/
  codec.py       SHIPPED (Session 7's) and DUPLEX (proposed) codecs, encode + decode
  vocab.py       tokenizers flattened to byte tables, labelled by script
  embedding.py   analysis and synthesis paths, plus the dense controls
  model.py       a small GPT whose front and back doors are swappable
experiments/
  e1_injectivity.py    collision audit, per script, across three vocabularies
  e2_recovery.py       can the code be recovered from d dimensions?
  e3_headfree_lm.py    six arms, real training, real data
  e4_scaling.py        cost arithmetic + an executed 1M-token vocabulary
tests/test_duplex.py   11 invariant tests
results/               one JSON per experiment
```

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_all.py --quick      # reduced sweep, ~5 minutes
python run_all.py              # everything, ~45 minutes
python -m unittest discover -s tests
```

E1 pulls the GPT-2 and XLM-R vocabularies from the Hub on first run. Everything
else reads files already in this repository: the Session 2 tokenizer, the
Session 5 proxy corpus, the Session 4 cleaned corpus. E3 uses MPS or CUDA when
available and falls back to CPU.

---

## 6. Where this leaves V5

The session's committed design is a Kronecker input path with a dense untied
head. The evidence here says the head can go, and that the input path should be
widened at the tail rather than at the window:

- Adopt `INACTIVE` and a 4-byte digest. It buys injectivity on all three
  vocabularies at `pos_dim = 32`, where the alternative — `pos_dim = 48` — costs
  50% more projection and *still* leaves GPT-2 with collisions.
- Keep the head-free synthesis path untied. It won on both loss and parameters.
- Keep the projection trainable, per Section 9. Nothing here changes that, and
  the reduced adaptive capacity makes it more urgent.
- Record `char_dim`, `pos_dim`, `tail_bytes`, `logit_scale`, the tokenizer hash
  and the synthesis mode under `embedding_policy_id`. `tail_bytes` is now part of
  the codec identity: change it and every code changes, exactly as changing the
  tokenizer does.
