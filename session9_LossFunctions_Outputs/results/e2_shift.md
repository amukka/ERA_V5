# 2 · Verify the shift by printing the strings

The sample, decoded, so the tables below can be read as English:

> 'rains from the storm triggered significant flooding along the Sabana River in Acapulco , killing four people . However , the overall effects of Beat'

## The correct alignment: `logits[:, :-1]` against `tokens[:, 1:]`

position i predicts token i+1. Read the two token columns: the target column
is the input column moved up one row. That is the whole of next-token
prediction, and it is checkable by eye.

| pos i | input token at i |    | pos | target token   |
| ----: | ---------------- | -- | --: | -------------- |
|     0 | ` ra`            | -> |   1 | `ins`          |
|     1 | `ins`            | -> |   2 | ` from`        |
|     2 | ` from`          | -> |   3 | ` the`         |
|     3 | ` the`           | -> |   4 | ` st`          |
|     4 | ` st`            | -> |   5 | `orm`          |
|     5 | `orm`            | -> |   6 | ` t`           |
|     6 | ` t`             | -> |   7 | `rig`          |
|     7 | `rig`            | -> |   8 | `g`            |
|     8 | `g`              | -> |   9 | `ered`         |
|     9 | `ered`           | -> |  10 | ` significant` |
|    10 | ` significant`   | -> |  11 | ` flo`         |
|    11 | ` flo`           | -> |  12 | `od`           |
|    12 | `od`             | -> |  13 | `ing`          |
|    13 | `ing`            | -> |  14 | ` along`       |
|    14 | ` along`         | -> |  15 | ` the`         |
|    15 | ` the`           | -> |  16 | ` S`           |

## Wrong alignment 1: no shift

position i predicts token i — the token it was just handed.

| pos i | input token at i |    | pos | target token   |
| ----: | ---------------- | -- | --: | -------------- |
|     0 | ` ra`            | -> |   0 | ` ra`          |
|     1 | `ins`            | -> |   1 | `ins`          |
|     2 | ` from`          | -> |   2 | ` from`        |
|     3 | ` the`           | -> |   3 | ` the`         |
|     4 | ` st`            | -> |   4 | ` st`          |
|     5 | `orm`            | -> |   5 | `orm`          |
|     6 | ` t`             | -> |   6 | ` t`           |
|     7 | `rig`            | -> |   7 | `rig`          |
|     8 | `g`              | -> |   8 | `g`            |
|     9 | `ered`           | -> |   9 | `ered`         |
|    10 | ` significant`   | -> |  10 | ` significant` |
|    11 | ` flo`           | -> |  11 | ` flo`         |
|    12 | `od`             | -> |  12 | `od`           |
|    13 | `ing`            | -> |  13 | `ing`          |
|    14 | ` along`         | -> |  14 | ` along`       |
|    15 | ` the`           | -> |  15 | ` the`         |

## Wrong alignment 2: shifted the other way

position i predicts token i-1 — a token already inside its own context.

| pos i | input token at i |    | pos | target token   |
| ----: | ---------------- | -- | --: | -------------- |
|     1 | `ins`            | -> |   0 | ` ra`          |
|     2 | ` from`          | -> |   1 | `ins`          |
|     3 | ` the`           | -> |   2 | ` from`        |
|     4 | ` st`            | -> |   3 | ` the`         |
|     5 | `orm`            | -> |   4 | ` st`          |
|     6 | ` t`             | -> |   5 | `orm`          |
|     7 | `rig`            | -> |   6 | ` t`           |
|     8 | `g`              | -> |   7 | `rig`          |
|     9 | `ered`           | -> |   8 | `g`            |
|    10 | ` significant`   | -> |   9 | `ered`         |
|    11 | ` flo`           | -> |  10 | ` significant` |
|    12 | `od`             | -> |  11 | ` flo`         |
|    13 | `ing`            | -> |  12 | `od`           |
|    14 | ` along`         | -> |  13 | `ing`          |
|    15 | ` the`           | -> |  14 | ` along`       |
|    16 | ` S`             | -> |  15 | ` the`         |

## Why this is worth doing by eye

Each alignment trained an identical model from an identical seed on identical
batches for 400 steps.

| alignment | loss it reports | perplexity | honest next-token loss | top-1 = current token | top-1 = previous token |
| --------- | --------------: | ---------: | ---------------------: | --------------------: | ---------------------: |
| correct   |          4.2748 |       71.9 |                 4.1085 |                  4.2% |                   5.2% |
| no shift  |          0.1947 |        1.2 |                12.7611 |                 97.9% |                   2.1% |
| reversed  |          1.2460 |        3.5 |                 9.1554 |                  4.2% |                  79.2% |

**The two broken objectives report the better number.** The correct one settles
at **4.2748** nats. No-shift reaches **0.1947**
and reversed reaches **1.2460** — a loss the honest objective
cannot get near, on a curve with no kink, no spike and no warning in it.

**Nothing was learned.** Scored on the task anyone actually wanted — predict the
next token — the no-shift model is at 12.7611 nats
against the correct model's 4.1085. It is
8.65 nats *worse*
while reporting a loss 4.08 nats *better*.

**And you can see what it learned instead.** Its top prediction at position i is
the token at position i **97.9%** of the time.
It is an identity function with 10,002 outputs. The reversed model does
the same thing one step over: its top prediction matches the *previous* token
**79.2%** of the time.

The correct model copies the current token 4.2%
of the time, which is roughly what guessing looks like.

Both bugs are three characters wide. Neither raises anything. The only cheap
defence is the table at the top of this page.
