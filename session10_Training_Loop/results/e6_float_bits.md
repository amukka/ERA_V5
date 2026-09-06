# Deliverable 6 — 0.1 in fp32, bf16 and fp8 E4M3

## Doing it by hand

0.1 is not a binary fraction. Doubling and reading off the carry gives

```
0.1  =  0.0001100110011001100110011001…  (binary)
     =  1.100110011001100110011…  ×  2⁻⁴
```

so the unbiased exponent is **−4** in every format below, and the significand to be rounded is always the same repeating `1.10011001100…`. Only the number of mantissa bits and the exponent bias change.


### fp32 — 1 sign, 8 exponent, 23 mantissa (bias 127)

- exponent field = −4 + 127 = **123** = `01111011`
- mantissa = the 23 bits after the leading 1 of `1.10011001100…`, rounded to nearest (the first discarded bit is a 1 and the tail is non-zero, so it rounds up) = `10011001100110011001101`
- significand = `1.10011001100110011001101`₂ = 1.6000000238

```
0 01111011 10011001100110011001101     0x3DCCCCCD
^ ^------- ^----------------------
| exponent    mantissa
sign
```

- stored value = **0.10000000149011611938**
- relative error = **1.490e-08** (0.0000%)
- torch stores `00111101110011001100110011001101` — **identical**


### bf16 — 1 sign, 8 exponent, 7 mantissa (bias 127)

- exponent field = −4 + 127 = **123** = `01111011`
- mantissa = the 7 bits after the leading 1 of `1.10011001100…`, rounded to nearest (the first discarded bit is a 1 and the tail is non-zero, so it rounds up) = `1001101`
- significand = `1.1001101`₂ = 1.6015625000

```
0 01111011 1001101     0x3DCD
^ ^------- ^------
| exponent    mantissa
sign
```

- stored value = **0.10009765625000000000**
- relative error = **9.766e-04** (0.0977%)
- torch stores `0011110111001101` — **identical**


### fp8 E4M3 — 1 sign, 4 exponent, 3 mantissa (bias 7)

- exponent field = −4 + 7 = **3** = `0011`
- mantissa = the 3 bits after the leading 1 of `1.10011001100…`, rounded to nearest (the first discarded bit is a 1 and the tail is non-zero, so it rounds up) = `101`
- significand = `1.101`₂ = 1.6250000000

```
0 0011 101     0x1D
^ ^--- ^--
| exponent    mantissa
sign
```

- stored value = **0.10156250000000000000**
- relative error = **1.562e-02** (1.5625%)
- torch stores `00011101` — **identical**


## All six formats side by side

| format   | bits | E/M  | sign · exponent · mantissa         |        hex | value stored for 0.1 | rel. error | matches torch |
| -------- | ---: | ---- | ---------------------------------- | ---------: | -------------------: | ---------: | ------------- |
| fp32     |   32 | 8/23 | 0 01111011 10011001100110011001101 | 0x3DCCCCCD |       0.100000001490 |   1.49e-08 | yes           |
| fp16     |   16 | 5/10 | 0 01011 1001100110                 |     0x2E66 |       0.099975585938 |   2.44e-04 | yes           |
| bf16     |   16 | 8/7  | 0 01111011 1001101                 |     0x3DCD |       0.100097656250 |   9.77e-04 | yes           |
| fp8 E4M3 |    8 | 4/3  | 0 0011 101                         |       0x1D |       0.101562500000 |   1.56e-02 | yes           |
| fp8 E5M2 |    8 | 5/2  | 0 01011 10                         |       0x2E |       0.093750000000 |   6.25e-02 | yes           |
| fp4 E2M1 |    4 | 2/1  | 0 00 0                             |        0x0 |       0.000000000000 |   1.00e+00 | —             |

Every hand-derived pattern matches the bits the hardware actually stores. fp4 E2M1 has no torch dtype to check against and is shown for the shape of the trade only — on its own it cannot even reach 0.1: it rounds to exactly **zero**, because the smallest non-zero value fp4 E2M1 can name is 0.5. That is why NVFP4 never uses the element format on its own, and always with one shared exponent per block of sixteen.


## Which one would I train in — bf16, with an fp32 master copy


Not because bf16 is accurate. It is the *least* accurate of the three at
representing 0.1: it stores 0.1000976562, a relative error of
0.0977%, against fp16's
0.0244% at the same 16 bits. The argument is about
range and about what a weight update actually looks like.

**1. The exponent field is the one that ends runs.** fp16 spends 5 bits on
exponent and cannot hold anything below about 6e-8:

| gradient |   in fp16 |   in bf16 | in fp16, ×1024 then ÷1024 |
| -------: | --------: | --------: | ------------------------: |
|    1e-04 | 1.000e-04 | 1.001e-04 |                 1.000e-04 |
|    1e-06 | 1.013e-06 | 9.984e-07 |                 1.000e-06 |
|    1e-08 |  0 — gone | 1.001e-08 |                 1.001e-08 |
|    1e-10 |  0 — gone | 1.000e-10 |                 1.164e-10 |

A gradient that becomes exactly zero is a weight that does not move, and nothing raises an error. Loss scaling patches it and is one more number to get wrong. bf16 keeps all eight of fp32's exponent bits, so the floor is never reached and the apparatus is unnecessary.


**2. But bf16 alone cannot hold a weight, and this is measurable.** bf16 keeps
7 mantissa bits, so consecutive bf16 values are 2⁻⁷ of a binade apart — at 0.1
that is 4.883e-04 absolute, or
4.88e-03 relative. In this session's own run, at step
31, the median AdamW update across all 8,336,384 weights is
2.48e-03 of the weight's own size — already
smaller than that spacing. Rounding each real update to the bf16 grid at its own
weight's magnitude, **51.6% of the
8,336,384 updates this optimiser step disappear entirely**
(in fp8 E4M3, 91.4%).

The left panel is the same fact in slow motion: adding 1e-05 to a
bf16 0.1 2000 times leaves it at
**0.100098** — it never moved once. The exact answer is
0.120000; fp32 reaches 0.119997. The
weight is not learning slowly, it is not learning at all, and the loss curve
says nothing about it.

That is exactly why section 13's table lists a bf16 weight *and* an fp32 master
copy: 2 bytes for the copy the matmuls read, 4 bytes for the copy the optimiser
adds to. The third line of the figure is that arrangement, and it tracks the
exact answer.

**3. fp8 E4M3 is a matmul input, not a weight.** With 3 mantissa bits its grid
is 2⁻³ of a binade — between 6.3% and 12.5% relative depending on where in the
binade you land, and 7.7% at 0.1 exactly.
Errors of that size are survivable where they are averaged across a long
reduction and never accumulated, which is what a forward matmul does. They are
not survivable in a running sum of tens of thousands of updates, which is what a
weight is. The 2026 production recipe is exactly this split: fp8 tensors into
the GEMMs, higher precision for the master weights, the optimiser state, and
attention — softmax amplifies whatever noise you hand it.

**So: bf16 activations and gradients, fp32 master weights and fp32 optimiser
moments, no loss scaling.** For V5 specifically I would take fp8 E4M3 on the
linear layers only once there is a short A/B on the real architecture, because
the thing fp8 costs you is not visible in the loss curve either.
