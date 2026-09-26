# Part 2 · One extra head, predicting token t+2

One trunk, two untied heads on the same final hidden state. Head 1 scores
position `i` against token `i+1`; head 2 against token `i+2` (alignment printed in
experiment 2's style: the last two positions of each row have no t+2 target and are
dropped). Training loss = `L1 + L2`. 1500 steps, `B=8`, `T=128`, AdamW
lr 0.0003, 3 seeds (mean ± std across seeds). Held-out = 32 fresh
validation rows, honest mask.

## The numbers

| | head 1 (t+1) | head 2 (t+2) | sum L1 + L2 |
| - | -: | -: | -: |
| untrained (held-out) | 9.244 ± 0.011 | 9.252 ± 0.032 | 18.496 |
| **trained, held-out** | **3.755 ± 0.017** | **4.726 ± 0.006** | **8.481 ± 0.020** |
| trained, train stream (last 100 steps) | 3.588 ± 0.042 | 4.617 ± 0.039 | 8.205 |

Reference points: ln V = 9.211; unigram entropy of the data =
6.310 nats (a model that ignores context can do no better).

Head 2 finishes **0.971 ± 0.015 nats above head 1**
(held-out).

## Does the second head change the first?

| head 1 held-out loss | |
| - | -: |
| trained with head 2 (sum objective) | 3.755 ± 0.017 |
| trained alone | 3.715 ± 0.012 |

Difference +0.040 nats against a seed spread of
about 0.017.

## What happens to the second head's loss, and why

* **Identical for the first ~60 steps.** Both heads start at ≈ ln V (an untrained
  model cannot tell t+1 from t+2) and fall together to about the unigram entropy
  (6.31 nats). What is learned there (token frequencies)
  helps both targets equally, so the two curves lie on top of each other.
* **Then they separate, and the gap keeps growing.** Past the unigram level, head 1
  keeps falling steadily while head 2 flattens. L2 − L1 opens from 0 at step 50 to
  ≈ 0.6 by step 400, ≈ 0.9 by step 800 and 0.97 at step 1500
  (held-out), still widening slowly. In the second half of training head 1 gains far
  more per step than head 2.
* **Why head 2 is harder: it is predicting one more token blind.** After the trunk
  has seen tokens `0..i`, token `i+1` is constrained by grammar and local context.
  Token `i+2` depends on the same context *and* on the unobserved token `i+1`, so
  the uncertainty compounds. For a stationary source
  `H(x_{i+2} | x_{≤i}) ≥ H(x_{i+1} | x_{≤i})`, so a persistent gap is
  expected; it is not a bug. What head 1 learns from local structure (syntax,
  word completion after sub-word pieces) is exactly what head 2 cannot use.
* **The gap is not a floor yet.** At this model size and step count neither head is
  near its entropy, so the ≈ 1.0 nat gap is a snapshot, not the
  irreducible difference.
* **The extra head slightly taxes head 1**: it ends +0.040 nats
  worse than when trained alone. Small (but larger than the seed spread), because
  the trunk's capacity is now shared with a second objective.
* **Read the sum with care.** `L1 + L2` is dominated by the harder head, and it is
  not comparable with any single-head loss. That is why the two are reported
  separately.

![e8](e8_extra_head.png)
