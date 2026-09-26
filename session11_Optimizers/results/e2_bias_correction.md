# E2 · Bias correction off, twenty steps both ways

![](e2_bias_correction.png)

## A · One weight

Twenty gradients drawn uniformly from [0.4, 0.6] (seed 0), β₁ = 0.9. The step is shown in units of η. Corrected, it sits at ~1.0η throughout, as in E1. Uncorrected:

|  t |     g | corrected | uncorr. β₂=0.999 | ratio | uncorr. β₂=0.95 | ratio |
| -: | ----: | --------: | ---------------: | ----: | --------------: | ----: |
|  1 | 0.527 |    1.0000 |           3.1623 | 3.162 |          0.4472 | 0.447 |
|  2 | 0.454 |    0.9934 |           4.2214 | 4.250 |          0.6056 | 0.608 |
|  3 | 0.408 |    0.9856 |           4.8790 | 4.950 |          0.7103 | 0.718 |
|  4 | 0.403 |    0.9820 |           5.3439 | 5.442 |          0.7888 | 0.798 |
|  5 | 0.563 |    0.9930 |           5.7565 | 5.797 |          0.8541 | 0.861 |
|  6 | 0.583 |    1.0015 |           6.0657 | 6.057 |          0.9065 | 0.910 |
|  7 | 0.521 |    1.0036 |           6.2671 | 6.245 |          0.9476 | 0.950 |
|  8 | 0.546 |    1.0071 |           6.4240 | 6.379 |          0.9816 | 0.982 |
|  9 | 0.509 |    1.0064 |           6.5112 | 6.470 |          1.0074 | 1.007 |
| 10 | 0.587 |    1.0126 |           6.6099 | 6.528 |          1.0314 | 1.028 |
| 11 | 0.563 |    1.0154 |           6.6599 | 6.559 |          1.0499 | 1.045 |
| 12 | 0.401 |    0.9986 |           6.5592 | 6.569 |          1.0521 | 1.058 |
| 13 | 0.571 |    1.0050 |           6.5937 | 6.561 |          1.0668 | 1.069 |
| 14 | 0.407 |    0.9906 |           6.4780 | 6.539 |          1.0655 | 1.077 |
| 15 | 0.546 |    0.9961 |           6.4809 | 6.507 |          1.0759 | 1.084 |
| 16 | 0.435 |    0.9869 |           6.3804 | 6.465 |          1.0750 | 1.089 |
| 17 | 0.573 |    0.9961 |           6.3908 | 6.416 |          1.0844 | 1.092 |
| 18 | 0.508 |    0.9962 |           6.3378 | 6.362 |          1.0874 | 1.095 |
| 19 | 0.460 |    0.9902 |           6.2413 | 6.303 |          1.0852 | 1.096 |
| 20 | 0.485 |    0.9884 |           6.1686 | 6.241 |          1.0852 | 1.097 |

Dividing the two update rules gives the ratio in closed form, r(t) = (1 − β₁ᵗ) / √(1 − β₂ᵗ), independent of the gradients. So for a single weight the question has an exact answer:

|    β₂ | largest overshoot | at step 20 | within 10% from step | within 1% from step |
| ----: | ----------------: | ---------: | -------------------: | ------------------: |
| 0.999 |  6.57× at step 12 |      6.24× |                1,751 |           **3,925** |
|  0.95 |  1.10× at step 20 |      1.10× |                    6 |              **76** |

Over the first twenty steps the uncorrected weight travels 119.5η instead of 19.9η at β₂ = 0.999 (19.0η at β₂ = 0.95).

**The session's step-1 figure of 3.16η is only the beginning.** With β₂ = 0.999 the uncorrected step keeps *growing* after step 1, because m fills in a hundred times faster than v. It peaks at 6.57η at step 12, is still 6.24η at step 20, and then takes thousands of steps to come back: v's correction factor is √(1 − 0.999ᵗ), which is still 0.80 at step 1,000.

**With β₂ = 0.95, the value we train with, the error runs the other way first.** Step 1 is only 0.45η. The square root softens v's bias: uncorrected, √v = √(1 − β₂)·|g| = 0.22|g| while m = (1 − β₁)·g = 0.10g. Step 1 is too small whenever √(1 − β₂) > 1 − β₁, which is whenever β₂ < 0.99. The ratio then overshoots to 1.10η at step 20 and is within 1% from step 76. The answer to "after how many steps does it stop mattering" is therefore a property of β₂, roughly 4/(1 − β₂) steps for the 1% level, and not a fixed number.

## B · The width-256 model

Peak lr 1.0e-03 (best point of E5's width-256 sweep), 300 steps, cosine to 10%. Corrected and uncorrected runs use the same seed and the same batches. Validation loss is measured every 10 steps. The noise yardstick is the median |Δ val loss| between two corrected runs that differ only in seed, over steps 100–300.

|    β₂ | warmup | Δ val at step 20 |      largest Δ val | Δ val at 300 | seed noise | gap inside noise from | gap frozen from |
| ----: | -----: | ---------------: | -----------------: | -----------: | ---------: | --------------------: | --------------: |
| 0.999 |      0 |          +0.1280 | +0.7124 (step 150) |      +0.6741 |     0.0648 |                 never |         **100** |
| 0.999 |     30 |          -0.7575 | +0.6828 (step 300) |      +0.6828 |     0.0125 |                 never |         **270** |
|  0.95 |      0 |          -0.0209 |   +0.2981 (step 5) |      -0.3353 |     0.0654 |                 never |         **160** |
|  0.95 |     30 |          -0.0366 | +0.1614 (step 190) |      +0.1389 |     0.0755 |                 never |          **60** |

Δ is uncorrected minus corrected: positive means the uncorrected run is worse. "Gap inside noise from" asks whether the two runs become indistinguishable. "Gap frozen from" asks the weaker question, whether bias correction has stopped *changing* anything: the first evaluation after which the gap stays within seed noise of its final value.

## What the model adds to the closed form

**The loss never forgets the first few dozen steps.** In none of the four settings does the uncorrected run come back inside seed noise within 300 steps. The step rule itself stops differing on schedule (from step 76 at β₂ = 0.95), but the weights it produced in the meantime are different weights, and a 300-step run at a decaying learning rate has no time to undo that. What does stop is the *change*: the gap freezes at the steps in the last column and is a constant offset after that.

**β₂ = 0.999: it matters for the whole run, in both directions.** With warmup, the uncorrected run is ahead by 0.76 nats at step 20, because a 6× step during warmup is simply a larger learning rate, and larger is faster at first. It is behind from step ~40 and finishes +0.68 worse, because the 3–6× step persists long after it stopped helping. That matches the closed form: the ratio is still 1.96 at step 300.

**β₂ = 0.95 without warmup: turning correction off *helped*, by 0.34 nats.** The uncorrected first step is 0.45η and it climbs to η over about ten steps, so for any β₂ below 0.99 dropping bias correction is an accidental warmup. It is a worse warmup than a real one, though: the corrected run *with* 30 warmup steps finished at 4.216, below the uncorrected no-warmup run's 4.265.

**β₂ = 0.95 with warmup, our setting: +0.14 against seed noise of 0.08.** Small, about two noise widths, and frozen early. Warmup had already done the job bias correction was protecting, so turning it off only made the first steps timid for no benefit.
