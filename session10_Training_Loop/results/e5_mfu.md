# Deliverable 5 — my own MFU, reported honestly

Machine: **Apple M4**, torch 2.13.0, device `mps`. One machine, no distributed anything.


## The denominator, measured

| fp32 square matmul | TFLOP/s |
| ------------------ | ------: |
| 1024×1024          |    1.90 |
| 2048×2048          |    2.16 |
| 3072×3072          |    1.13 |
| 4096×4096          |    1.25 |

Best sustained: **2.16 TFLOP/s fp32**. (bf16 on the same device reaches 1.43 TFLOP/s, which is the ceiling a bf16 loop would be measured against — the loop below runs in fp32, so fp32 is the matched denominator.)

This is deliberately the strictest available denominator: a throughput this exact machine has been observed to sustain, on the one operation a transformer is almost entirely made of.


## The numerator

| quantity                  |      FLOPs | what it is                                                                                         |
| ------------------------- | ---------: | -------------------------------------------------------------------------------------------------- |
| parameters in matmuls (N) |  5,710,592 | everything except the two embedding tables; the D×V output head is included because it is a matmul |
| 6N                        | 34,263,552 | 2N forward + 4N backward, per token                                                                |
| 12·L·D·T                  |  3,145,728 | the QKᵀ and attention·V products at T=256; no weights, so not counted in N                         |
| total per token           | 37,409,280 | attention is 8.4% of it at this sequence length                                                    |

## The number

| quantity            |                      value |
| ------------------- | -------------------------: |
| micro-batch         |             8 × 256 tokens |
| accumulation steps  |                          4 |
| global batch        | 32 sequences, 8,192 tokens |
| tokens per second   |                     17,494 |
| achieved            |              0.654 TFLOP/s |
| machine measured at |               2.16 TFLOP/s |
| **MFU**             |                 **30.32%** |

**30.3%**, which is inside the 35-50% band the session calls
healthy. I did not expect that and spent a while trying to find the mistake, so
the rest of this section is mostly about why the number is high and what it is
still hiding.


## Why this is not the triumph it looks like


**MFU is a ratio, and its denominator is a choice.** The numerator was measured.
So was the denominator — but *which* denominator is a decision, and it moves the
answer by more than any change to the loop would. Three defensible ones for this
same run:

| denominator                          | TFLOP/s | MFU it gives | why                                                                                                                                          |
| ------------------------------------ | ------: | -----------: | -------------------------------------------------------------------------------------------------------------------------------------------- |
| best sustained fp32 matmul, measured |    2.16 |        30.3% | the loop runs in fp32, so this is the matched one — and the number reported above                                                            |
| best sustained bf16 matmul, measured |    1.43 |        45.8% | what the same loop would be scored against once moved to bf16, without getting any faster on the day it moves                                |
| fp32 matmul at 1024×1024, measured   |    1.90 |        34.4% | what the device does on matrices the size this model actually uses — a denominator that would flatter the run and is therefore the wrong one |

The session's own worked example — a 9B model at 8.2% on eight H100s — is
measured against a **datasheet** peak, and that is the second reason this run
scores well. A datasheet peak is what the hardware does under conditions the
loop may never meet: the right dtype, the right kernel, the right shapes. My
denominator was produced by `torch.matmul` in fp32 through the same framework on
the same device — the same path the model's own matmuls take — so the numerator
and the denominator are not measuring different machines. That makes the ratio
honest, and it also makes it flattering, and both are worth saying.

Put the other way round: **this loop would not score 40% on an H100.** Nothing
about it is written for one — no `torch.compile`, no fused optimiser, no
FlashAttention, no bf16 — and every one of those is worth more on hardware whose
peak assumes them.


## What is still on the table, measured


![mfu](e5_mfu.png)

Rather than guess at the remaining gap, both plausible causes were varied and
the measurements disagree with the story I expected to tell.

**1. Width, not batch size, is where the headroom is.** The right panel holds
the micro-batch fixed and widens the model: MFU goes 27.6% → 34.8%
→ 37.0% at d_model 256 → 512 → 768. That is
9 points from one knob. A 256×256 weight matrix cannot
saturate a unit that only reaches its own peak at 4096×4096 — the roofline table
shows the identical effect on bare matmuls, climbing
1.90 → 2.16 TFLOP/s over the same range. The
model is too small for the machine, and no amount of loop engineering fixes
that.

**2. Micro-batch size helps, but far less than I assumed.** The left panel holds
the model fixed and grows the micro-batch: 27.3% →
30.9% from
8×256 to
64×256 — a 8× increase
in work per kernel buying about 4
points. Real, and much smaller than the width effect.

**3. Long sequences cost throughput even though attention is only
8.4% of the counted FLOPs.** At matched token
counts the shorter sequence wins every time:

| tokens per micro-batch |  shape |    MFU |
| ---------------------: | -----: | -----: |
|                  2,048 |  8×256 | 27.75% |
|                  4,096 | 16×256 | 27.28% |
|                  4,096 |  8×512 | 26.84% |
|                  8,192 | 32×256 | 30.88% |
|                  8,192 | 16×512 | 27.40% |
|                  8,192 | 8×1024 | 25.72% |
|                 16,384 | 64×256 | 29.88% |

Attention is counted in the numerator and grows with T, so a longer sequence is
credited with *more* FLOPs and still scores lower. That is the unfused attention
path being expensive: this model materialises a full B×H×T×T score matrix and a
second one for the softmax output, and both are memory traffic that no FLOP
count sees. FlashAttention exists for exactly this.

**4. Accumulation is free throughput, which I had not expected to be
measurable.** The baseline runs 4 micro-batches per optimiser
step and reaches 30.32%; the same
8×256 micro-batch with `accum=1` in the
sweep reaches 27.75%.
The optimiser is 2.4% of the baseline
step; running it once per 4 micro-batches instead of once per
one is where the difference comes from. Gradient accumulation was adopted in
section 7 to buy a bigger global batch than memory allows — it also amortises an
update that is almost pure overhead.

**5. Where the step time actually goes.** Forward
37.4%, backward
59.8%, optimiser
2.4%. Backward at roughly twice
forward is the textbook ratio and is the one part of this that looks exactly as
it should.

**6. The loader is not the problem, but it would be if left inline.** Building
one step's micro-batches costs 3.5 ms
against a 468.3 ms step —
1% of a step. It is prebuilt and
timed separately above rather than quietly counted as compute, because counting
it as compute is a way to report a throughput number that is not true.

## What I would actually do

Widen the model. On measurement it is worth
9 points and everything else on this list is worth less. Then
raise the micro-batch until memory objects, keep accumulating, and only then
reach for `torch.compile`, a fused optimiser and a fused attention kernel — in
that order, because that is the order the measurements put them in and not the
order I would have guessed.

And the part that does not change with any of it: at
30.32% and at 37% this
model draws the same loss curve. The loss tells you whether it is learning. Only
this number tells you what you are paying to find out.


## Every configuration measured

| what varied        | d_model | micro-batch | tokens/s | TFLOP/s |    MFU |
| ------------------ | ------: | ----------: | -------: | ------: | -----: |
| baseline (accum 4) |     256 |       8×256 |   17,494 |   0.654 | 30.32% |
| batch/seq          |     256 |       8×256 |   16,008 |   0.599 | 27.75% |
| batch/seq          |     256 |      16×256 |   15,737 |   0.589 | 27.28% |
| batch/seq          |     256 |      32×256 |   17,813 |   0.666 | 30.88% |
| batch/seq          |     256 |      64×256 |   17,234 |   0.645 | 29.88% |
| batch/seq          |     256 |       8×512 |   14,283 |   0.579 | 26.84% |
| batch/seq          |     256 |      16×512 |   14,580 |   0.591 | 27.40% |
| batch/seq          |     256 |      8×1024 |   11,849 |   0.555 | 25.72% |
| d_model            |     256 |      16×512 |   14,710 |   0.597 | 27.64% |
| d_model            |     512 |      16×512 |    6,320 |   0.751 | 34.81% |
| d_model            |     768 |      16×512 |    3,395 |   0.798 | 36.96% |