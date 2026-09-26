# 5 · Perplexity, and the cheapest sanity check there is

perplexity = exp(mean loss) — "how many equally likely options is the model
effectively choosing between at this token?"

## The anchor, exactly

A model that knows nothing should spread its probability evenly over the whole
vocabulary. Feed the loss a genuinely uniform logits tensor and it returns the
anchor to the last decimal it has:

| | |
|---|---|
| vocabulary | **10,002** |
| ln(V) | **9.210540** nats |
| loss from uniform logits | **9.210541** nats |
| perplexity from uniform logits | **10,002.00** |

The two agree to 6 decimal places, which is
what it means for the anchor to be a definition rather than an observation.

## Where the real untrained model sits

|                | at initialisation |
| -------------- | ----------------: |
| loss           |       9.2563 nats |
| perplexity     |            10,470 |
| vocabulary     |            10,002 |
| perplexity / V |             1.047 |

**Perplexity 10,470 against a vocabulary of
10,002** — 104.7% of it. The run may
proceed.

The small excess is not noise, and it is worth accounting for rather than
tolerating. At initialisation the logits are not exactly uniform: they have a
standard deviation of 0.3184, because the head is
initialised at std 0.02 and reads a 256-wide hidden state. For
logits with spread σ, `E[logsumexp(z)] ≈ ln V + σ²/2`, so the expected loss is

    ln V + σ²/2 = 9.2105 + 0.0507 = 9.2612

against a measured **9.2563**, out by
0.0049 nats. Nothing was fitted. A random head
sits slightly *above* ln(V), never below — so a first step **below** the anchor
is the alarming direction, and it is the direction every bug in this session
pushes it.

## What the check catches, and what it does not

Five harnesses, scored once, before any training. Pass = perplexity within a
factor of two of V.

| harness                       |   loss | perplexity | step-0 check | why                                                      |
| ----------------------------- | -----: | ---------: | ------------ | -------------------------------------------------------- |
| correct harness               | 9.2563 |     10,470 | pass         | lands on the anchor                                      |
| targets not shifted           | 9.2443 |     10,345 | pass         | an untrained model cannot copy yet, so this passes       |
| targets shifted backwards     | 9.2491 |     10,395 | pass         | same reason -- it passes, and it is still broken         |
| padding counted in the mean   | 9.3214 |     11,175 | pass         | padding is not yet predictable either                    |
| masked sum, wrong denominator | 6.5420 |        694 | **CAUGHT**   | caught immediately: the sum is right, the divisor is not |

**It catches the denominator instantly.** A correct sum divided by `B × (T-1)`
reports a perplexity of 694
against a vocabulary of 10,002 — a model that has not seen a single
gradient claiming to have narrowed 10,002 options down to
694.
That is impossible, it costs one forward pass to see, and it is free.

**It does not catch the shift, at step 0.** Both broken alignments pass, and
they pass for an honest reason: copying is a *learned* skill. An untrained model
cannot copy its input any better than it can predict the future, so at
initialisation all three alignments sit on the anchor together.

They separate within twenty steps — the figure's left panel — and by step
60 the no-shift run is at 4.27 nats
(perplexity 71.4) while the correct run is at
6.01. A perplexity of 71.4
after 60 steps on a 10k vocabulary is not a good run; it is a
different task. So the check is worth running as *"is the loss falling faster
than any real model could learn?"*, not only as a step-0 assertion.

**And nothing about padding.** Counting padding passes cleanly at step 0,
because PAD is not yet predictable either. Deliverable 3 is not covered by this
check and needs its own — the contributing-token count.

## One caution: perplexity is per *token*

Our tokenizer packs 2.85 bytes into the average
token on this sample. A tokenizer that split the same text into twice as many
tokens would be asked an easier question at each step and would report a better
perplexity for an identical model.

Bits per byte normalises by something the tokenizer cannot move:

    bpb = loss × tokens / (bytes × ln 2) = 4.6831

Comparable across tokenizers; perplexity is not. Within one tokenizer, which is
the whole of this session, perplexity is the more readable of the two.
