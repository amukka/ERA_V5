# E5 · Learning rate against width

Standard parameterisation, 4 layers, 300 steps × 4,096 tokens, 30-step warmup, cosine to 10% of peak, AdamW (0.9, 0.95), decay 0.1, clip 1.0. Score: held-out loss over 65,536 tokens.

![](e5_lr_width.png)

## Every point (seed 0; half-octave points only near each minimum)

|  peak lr | width 256 | width 512 | width 1,024 |
| -------: | --------: | --------: | ----------: |
| 1.25e-04 |    5.2357 |    4.6513 |      4.3654 |
| 2.50e-04 |    4.6842 |    4.3202 |      4.1523 |
| 3.54e-04 |         — |    4.2133 |      4.1023 |
| 5.00e-04 |    4.3461 |    4.1548 |      4.0853 |
| 7.07e-04 |    4.2502 |    4.2277 |      4.0589 |
| 1.00e-03 |    4.2155 |    4.2850 |      4.1544 |
| 1.41e-03 |    4.3146 |         — |           — |
| 2.00e-03 |    4.5099 |    4.4093 |      4.3236 |
| 4.00e-03 |    4.5543 |    4.6313 |      4.6645 |
| 8.00e-03 |    4.8621 |    4.9552 |      5.1031 |

## The three minima

The minimum is the vertex of a parabola in log₂(lr) through five points around the lowest loss: the grid minimum, its two neighbours (two seeds each, averaged) and two half-octave points (one seed). Per-seed minima refit seed 0 alone on five points and seed 1 alone on three.

| width | params | best grid lr | val loss | fitted minimum |     per-seed minima |
| ----: | -----: | -----------: | -------: | -------------: | ------------------: |
|   256 |   8.3M |     1.00e-03 |   4.2155 |   **8.96e-04** | 8.76e-04 – 9.36e-04 |
|   512 |  23.0M |     5.00e-04 |   4.1548 |   **5.11e-04** | 5.11e-04 – 5.15e-04 |
| 1,024 |  71.1M |     5.00e-04 |   4.0853 |   **5.05e-04** | 4.69e-04 – 5.17e-04 |

## Extrapolating to 4,096

A straight line through the three minima in log–log space has slope **-0.41** (per-seed combinations: -0.50 to -0.38). Section 12's rule of thumb is −1.00.

| estimate for d_model 4,096                    |     learning rate |
| --------------------------------------------- | ----------------: |
| fit through the three minima                  |       **2.6e-04** |
| range over per-seed minima                    | 2.2e-04 – 2.8e-04 |
| 1/width, anchored at our width-1,024 minimum  |           1.3e-04 |
| the 512 → 1,024 slope alone, continued        |           4.9e-04 |
| carrying width 256's minimum across unchanged |           9.0e-04 |

## The value I would use, and how confident I am

**2.6e-04**, as the centre of a short confirmation sweep at the real width, not as a value to launch with.

**Confidence: low, roughly a factor of two either way.** The per-seed range above looks tight, but it only measures seed noise at these three widths. It does not measure the thing that dominates the error, which is the shape of the curve:

- The minimum does not move as a clean power law. From 256 to 512 the local slope is -0.81; from 512 to 1,024 it is -0.02. Three points cannot tell a power law from a curve that is flattening, and the two readings give 2.6e-04 and 4.9e-04 at 4,096.

- Section 12's −1 rule, anchored at our width-1,024 minimum, gives 1.3e-04, another factor of two lower. Our measured slope is about half as steep, and there is a concrete reason to expect that here. This model initialises every matrix at a fixed 0.02. Under a textbook 1/√fan-in initialisation the weights shrink as the model widens, so the same η is a relatively larger step on a wider model, and that is part of why the optimum moves left. With a fixed 0.02 the step-to-weight ratio at a given η is the same at every width (E3 measures it), so that part of the drift is absent. What remains comes from wider layers summing more correlated updates. That argument predicts a shallower slope, not its exact value. A second, untested possibility is the 300-step budget, which leaves every width far from converged.

- 4,096 is four times wider than the widest point measured, and the vocabulary (10k) and depth (4) would not be the real ones.

**What would raise it:** a fourth width at 2,048 to decide between the power law and the flattening curve, and a longer budget, since the optimum at 300 steps is not the optimum at 30,000. Or muP, which Section 12 exists to recommend: under it this whole table collapses to one column and the question disappears.

