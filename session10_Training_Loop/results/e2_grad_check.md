# Deliverable 2 — verify one gradient by hand

## A. The session's own chain, where the answer is known

`x = 2`, `w1 = 3`, `w2 = 4`, `target = 20`.

| quantity      | calculation | value |
| ------------- | ----------- | ----: |
| h = w1·x      | 3 × 2       |     6 |
| y = w2·h      | 4 × 6       |    24 |
| loss = (y−t)² | (24 − 20)²  |    16 |

Walking back one link at a time:

| quantity | calculation | value |
| -------- | ----------- | ----: |
| ∂L/∂y    | 2(y − t)    |     8 |
| ∂L/∂w2   | ∂L/∂y × h   |    48 |
| ∂L/∂h    | ∂L/∂y × w2  |    32 |
| ∂L/∂w1   | ∂L/∂h × x   |    64 |

| ∂L/∂w1 obtained       |           value |
| --------------------- | --------------: |
| by hand (chain rule)  | 64.000000000000 |
| by nudging (h = 1e-6) | 64.000000008058 |
| by backward()         | 64.000000000000 |

Hand vs `backward()`: identical to every digit float64 has (difference exactly 0). Hand vs nudge: 8.1 decimals — the nudge is the one that is approximate.


## B. One weight of the real model, in float64

Weight: `blocks.0.mlp.fc.weight[17, 42]` = -0.0124816364, on a fixed batch of 2×96 tokens. Loss at that point: 9.3029963979.

`backward()` reported **-0.004117679113**.

| nudge h |      loss(w+h) |      loss(w−h) |    (Δloss)/2h | rel. error | decimals agreed |
| ------: | -------------: | -------------: | ------------: | ---------: | --------------: |
|   1e-01 | 9.302611659054 | 9.303429750377 | -0.0040904566 |   6.61e-03 |             4.6 |
|   1e-02 | 9.302955468124 | 9.303037816216 | -0.0041174046 |   6.67e-05 |             6.6 |
|   1e-03 | 9.302992282696 | 9.303000518048 | -0.0041176764 |   6.67e-07 |             8.6 |
|   1e-04 | 9.302995986186 | 9.302996809722 | -0.0041176791 |   6.50e-09 |            10.6 |
|   1e-05 | 9.302996356753 | 9.302996439106 | -0.0041176792 |   1.51e-08 |            10.2 |
|   1e-06 | 9.302996393812 | 9.302996402047 | -0.0041176786 |   1.14e-07 |             9.3 |
|   1e-07 | 9.302996397518 | 9.302996398341 | -0.0041176751 |   9.77e-07 |             8.4 |
|   1e-08 | 9.302996397888 | 9.302996397971 | -0.0041176840 |   1.18e-06 |             8.3 |

Best at h = 1e-04: the nudge says -0.004117679087, `backward()` says -0.004117679113. They agree to **10.6 decimal places** (relative error 6.50e-09).


Four more weights, one nudge each at h = 1e-5, to show the first was not a lucky pick:

| weight                            |    backward() |         nudge | rel. error | decimals |
| --------------------------------- | ------------: | ------------: | ---------: | -------: |
| tok_emb.weight[293, 100]          | -0.0154110461 | -0.0154110462 |    4.5e-09 |     10.2 |
| blocks.2.attn.qkv.weight[300, 11] |  0.0000276442 |  0.0000276443 |    2.3e-06 |     10.2 |
| head.weight[4096, 200]            | -0.0000171628 | -0.0000171628 |    4.4e-08 |     12.1 |
| ln_f.weight[64]                   | -0.0047538565 | -0.0047538564 |    1.5e-08 |     10.2 |

## C. The same check in float32, which fails — and why

Identical model, identical batch, identical weight, only the dtype changed. `backward()` reports -0.0041176705; the best nudge manages -0.0041325887 at h = 1e-02 — **4.8 decimals**, against 10.6 in float64.

| nudge h |  (Δloss)/2h | rel. error | decimals agreed |
| ------: | ----------: | ---------: | --------------: |
|   1e-01 | -0.00409126 |   6.41e-03 |             4.6 |
|   1e-02 | -0.00413259 |   3.62e-03 |             4.8 |
|   1e-03 | -0.00381470 |   7.36e-02 |             3.5 |
|   1e-04 | -0.00635783 |   5.44e-01 |             2.6 |
|   1e-05 |  0.00000000 |   1.00e+00 |             2.4 |
|   1e-06 | -0.31789144 |   7.62e+01 |             0.5 |
|   1e-07 |  0.00000000 |   1.00e+00 |             2.4 |
|   1e-08 |  0.00000000 |   1.00e+00 |             2.4 |

This is the disagreement the assignment says is worth understanding, and the thing worth understanding is that **`backward()` is not the one that is wrong**. The loss here is around 9.3030; float32 resolves it to about 5.55e-07. Dividing that noise by 2h magnifies it as h shrinks, while the truncation error of the central difference shrinks as h². The two meet at a floor, and no choice of h gets under it. In float64 the same floor sits eight orders of magnitude lower, which is why the check has to be run there.

The practical rule: a finite-difference gradient check that fails in float32 has told you nothing. Re-run it in float64 before you go looking for a bug in the backward pass.
