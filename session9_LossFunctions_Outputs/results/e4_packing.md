# 4 · Two documents in one sequence, and the join between them

## The join, read as text

Document A ends and document B begins inside one row. Decoded either side of
position 32:

> …'-@ the @-@ counter pain<eos>' **`<eos>`** … 'rack , composed by chiptune musician Y'…

| pos i | input token | target token | input's doc | target's doc | in the loss?   |
| ----: | ----------- | ------------ | ----------- | ------------ | -------------- |
|    28 | `ter`       | ` p`         | doc 0       | doc 0        | kept           |
|    29 | ` p`        | `ain`        | doc 0       | doc 0        | kept           |
|    30 | `ain`       | `<eos>`      | doc 0       | doc 0        | kept           |
|    31 | `<eos>`     | `rac`        | doc 0       | doc 1        | **masked out** |
|    32 | `rac`       | `k`          | doc 1       | doc 1        | kept           |
|    33 | `k`         | ` ,`         | doc 1       | doc 1        | kept           |
|    34 | ` ,`        | ` composed`  | doc 1       | doc 1        | kept           |
|    35 | ` composed` | ` by`        | doc 1       | doc 1        | kept           |

Exactly one pair crosses the join. Both of its tokens are real text — nothing
here is padding, nothing is malformed — and the pair is still a lie: the last
token of one article does not predict the first token of the next.

## The loss before and after masking it

An untrained model first, so the count is visible on its own:
**1016 → 1008**
contributing pairs, loss 9.2833 →
9.2832. At initialisation the two numbers are
almost identical, because an untrained model finds the join no harder than
anything else. **This is the trap: at step 0 the bug is invisible.**

After 400 steps on packed data:

| docs per row | pairs at a join | loss, join counted | loss, join masked | difference | mean loss *at* the join |
| -----------: | --------------: | -----------------: | ----------------: | ---------: | ----------------------: |
|            2 |           0.79% |             4.4037 |            4.3867 |    +0.0170 |                  6.5495 |
|            4 |           2.36% |             5.0088 |            4.9610 |    +0.0479 |                  6.9867 |
|            8 |           5.51% |             5.3450 |            5.2618 |    +0.0832 |                  6.7710 |
|           16 |          11.81% |             5.6081 |            5.4850 |    +0.1231 |                  6.5276 |

## Explaining the difference

**The difference is small and the cause is large.** At two documents per row the
join is 0.79% of the pairs and moves the reported
loss by only +0.0170 nats. Read that as reassuring and you have
misread it. The mean loss *at* the join is **6.5495** nats
against **4.3867** inside a document — the crossing pairs are
1.5× harder, and they are the hardest
pairs in the batch by a wide margin. They are diluted, not benign.

**Dilution is a property of the packing, not of the bug.** Pack sixteen
documents per row instead of two — which is what a 4K or 8K context does to a
corpus of short documents — and the same mistake moves the loss by
+0.1231 nats, 7× more,
because 11.81% of pairs now cross a join.
Nothing about the error changed. Only how often it is committed.

(Both columns rise with density for a reason that is not the bug: more
documents in a fixed 128 tokens means shorter fragments, so every position has
less context to work with. That is why the honest comparison is the difference
between the two columns and not the level of either.)

**And the loss is the least of it.** Every crossing pair is a gradient telling
the model that unrelated text follows `<eos>`. The model cannot learn that,
because it is not true, so it learns the next best thing: that after `<eos>`
anything can happen. That is capacity spent on making the model *less* certain,
and it is spent at exactly the position where a document boundary should be the
most informative token in the sequence.

The fix costs one comparison — `doc_id[i] == doc_id[i+1]` — and drops
8 pairs out of
1016. If the attention mask is also
document-aware (session 6), the two masks must agree: a position that cannot
*see* the previous document must not be scored as though it could.
