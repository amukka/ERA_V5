# E1 · Adam by hand, checked against PyTorch

One weight starting at w = 1.0, five gradients [0.5, 0.4, 0.6, 0.45, 0.55], η = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 1e-08. The hand column is plain Python floats following the five lines of Section 6.

## The hand computation

| t |    g |        m |          v |       m̂ |       v̂ |         step |     w after |
| : | ---: | -------: | ---------: | -------: | -------: | -----------: | ----------: |
| 1 | 0.50 | 0.050000 | 0.00025000 | 0.500000 | 0.250000 | -0.001000000 | 0.999000000 |
| 2 | 0.40 | 0.085000 | 0.00040975 | 0.447368 | 0.204977 | -0.000988126 | 0.998011874 |
| 3 | 0.60 | 0.136500 | 0.00076934 | 0.503690 | 0.256703 | -0.000994140 | 0.997017734 |
| 4 | 0.45 | 0.167850 | 0.00097107 | 0.488078 | 0.243132 | -0.000989847 | 0.996027887 |
| 5 | 0.55 | 0.206065 | 0.00127260 | 0.503199 | 0.255030 | -0.000996425 | 0.995031463 |

This reproduces the Section 6 table digit for digit at its printed precision. Every step is between 98.81% and 100.00% of η, although the gradients differ by 50% from smallest to largest. That is the property the session names: the gradient sets the direction, η sets the distance.

## The step, side by side (float64)

| t |            by hand |   torch.optim.Adam | |difference| |
| : | -----------------: | -----------------: | -----------: |
| 1 | -0.000999999980000 | -0.000999999980000 |      1.1e-18 |
| 2 | -0.000988125782298 | -0.000988125782298 |      3.3e-17 |
| 3 | -0.000994140046704 | -0.000994140046704 |      1.5e-17 |
| 4 | -0.000989846693133 | -0.000989846693133 |      4.6e-17 |
| 5 | -0.000996424710595 | -0.000996424710595 |      4.8e-17 |

## Agreement for every quantity

Worst relative error over the five steps, and the number of decimal places of agreement it implies (−log₁₀ of the relative error). PyTorch does not store m̂ or v̂; they are recovered from its own m, v and step counter.

| quantity | float64 rel. err | float64 decimals | float32 rel. err | float32 decimals |
| -------- | ---------------: | ---------------: | ---------------: | ---------------: |
| m        |          2.0e-16 |             15.7 |          1.6e-08 |              7.8 |
| v        |          1.7e-16 |             15.8 |          5.7e-08 |              7.2 |
| m_hat    |          2.3e-16 |             15.6 |          1.6e-08 |              7.8 |
| v_hat    |          2.2e-16 |             15.7 |          5.7e-08 |              7.2 |
| step     |          4.8e-14 |             13.3 |          1.4e-05 |              4.9 |
| w        |          0.0e+00 |            exact |          1.3e-08 |              7.9 |

**Float64: at least 13.3 decimal places on every quantity. Float32: at least 4.9.** In float32, m, v, m̂, v̂ and w all agree to 7–8 digits, which is float32's own resolution. The step is the outlier, and not because Adam disagrees: PyTorch never stores its step, so it is recovered as `w_after − w_before`. Both weights sit near 1.0, where float32 spacing is 1.2e-7, and the step is only 1e-3, so the subtraction alone throws away about four digits. Which is itself a Session 10 lesson: a 1e-3 update to a weight of 1 keeps only ~4 significant digits in float32, and none at all in bf16.

## AdamW, decay 0.1

The same five gradients through `torch.optim.AdamW` against the hand rule with the decoupled `− η·λ·w` term added after the Adam step. Worst agreement in float64: 13.1 decimals. The decay term alone moved the weight by -1.0e-04 at step 1, about 10% of η, and it is not divided by √v̂, which is the whole point of Section 7.

## A detail found by doing it

PyTorch does not compute the step the way Section 6 writes it. It folds both corrections into a scalar, `step_size = η / (1 − β₁ᵗ)` and `denom = √v / √(1 − β₂ᵗ) + ε`. ε is still added *after* the v correction, as in the formula, so the two routes are the same algorithm and differ only in rounding order. That is why float64 agrees to 15–16 digits on m, v, m̂ and v̂ rather than bit for bit. Step 1 also shows ε at work: the step is 0.99999998·η, not η, because ε = 1e-8 sits beside √v̂ = 0.5.
