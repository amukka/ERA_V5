# 3 · Mask padding, and watch the denominator move

## The count changes

One batch, `B=8`, `T=128`. The rows hold [128, 93, 57, 39, 26, 46, 48, 26] real tokens, so
55.2% of the slots are padding.

| what the loss counts        | contributing tokens | loss (untrained) | denominator            |
| --------------------------- | ------------------: | ---------------: | ---------------------- |
| no mask — every pair counts |                1016 |           9.2625 | B × (T-1)              |
| padding masked out          |                 455 |           9.2497 | sum of the mask        |
| masked, wrong denominator   |                1016 |           4.1423 | B × (T-1), incorrectly |

**1016 → 455**, a drop of
561 pairs. That is the deliverable, and it is the only part of
this that is visible at initialisation.

The denominator is not a constant you can hard-code, either — it is a property
of the batch:

| batch | contributing | of B × (T-1) | fraction real |
| ----: | -----------: | -----------: | ------------: |
|   100 |          637 |         1016 |         62.7% |
|   101 |          616 |         1016 |         60.6% |
|   102 |          615 |         1016 |         60.5% |
|   103 |          531 |         1016 |         52.3% |
|   104 |          520 |         1016 |         51.2% |
|   105 |          575 |         1016 |         56.6% |

Divide by `B × (T-1)` and the loss is multiplied by that last column, which
moves every step. The third row of the first table is that bug on this batch:
4.1423 instead of 9.2497,
-55.2%
low, for a sum that was computed correctly.

## Why it matters more than dilution

Two models, 400 steps, same seed, same batches. One boolean apart.

| arm            | loss it reports | loss on real tokens | mean P(PAD) | top-1 is PAD |
| -------------- | --------------: | ------------------: | ----------: | -----------: |
| masks padding  |          4.5673 |              4.8850 |       0.00% |         0.0% |
| counts padding |          2.7219 |              4.9542 |      49.54% |        49.3% |

**The broken arm reports the better number.** 2.7219 against
4.5673 — it looks
40% better trained.

**On real tokens it is worse.** 4.9542 against
4.8850, a gap of
+0.0692 nats, both scored identically on
the same held-out batch.

Put those two rows next to each other and the shape of the bug is clear. The
reported loss is wrong by
**40%**. The model is
worse by **1.4%**. The number
on the dashboard is off by roughly
29×
more than the model is damaged — which is why this survives code review. Nobody
is looking for a bug that makes the loss *better*.

**Because that is where its capacity went.** It puts
49.54% of its probability mass on PAD and names
PAD as its top prediction at 49.3% of
positions, against 0.00% and
0.0% for the arm that never scored it. PAD
is the easiest token in the vocabulary — it is perfectly predictable from
position alone — so the loss falls fastest by learning it, and a loss that falls
is exactly what the run looks like it wants.

Note that attention was masked correctly in **both** arms: no position ever
attended to a padded key. The bug is entirely in the mean, in one line, several
hundred lines away from the attention mask that is doing its job.
