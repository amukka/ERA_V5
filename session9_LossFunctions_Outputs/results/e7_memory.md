# 7 · Peak memory: ordinary cross-entropy against a chunked one

`B=16`, `T=256`, `V=10,002`: **4,080 predictions**, so the logits are
4,080 x 10,002 x 4 B = **155.7 MiB** if held whole.

## The two numbers

| implementation | peak live tensors (loss step, fwd + bwd) |
| -------------- | ---------------------------------------: |
| ordinary `F.cross_entropy(logits)` | **624.5 MiB** |
| chunked, 512 rows at a time, recomputed in backward | **86.1 MiB** |
| **ratio** | **7.25x** lower |

## Is it the same loss?

| | ordinary | chunked | |diff| |
| - | -: | -: | -: |
| loss | 9.261488 | 9.261488 | 0.00e+00 |
| grad wrt hidden, max abs | | | 7.28e-12 (scale 2.46e-05) |
| grad wrt head weight, max abs | | | 2.09e-07 (scale 2.90e-01) |
| contributing tokens | 4080 | 4080 | |

Same value and same gradients up to float rounding, so the saving is not bought
with a different objective.

## Chunk-size sweep

| chunk rows | peak MiB | ratio vs ordinary | one chunk's logits MiB |
| ---------: | -------: | ----------------: | ---------------------: |
|         64 |     37.4 |            16.71x |                    2.4 |
|        128 |     42.2 |            14.80x |                    4.9 |
|        256 |     56.8 |            10.99x |                    9.8 |
|        512 |     86.1 |             7.25x |                   19.5 |
|       1024 |    144.8 |             4.31x |                   39.1 |
|       2048 |    262.0 |             2.38x |                   78.1 |

Peak grows roughly linearly with the chunk: what is alive at once is one chunk's
logits plus its softmax and gradient copies, on top of a fixed part (hidden
states, head weight and its gradient).  Ordinary cross-entropy is the same thing
with the "chunk" equal to every prediction in the batch: it holds about four
logits-sized tensors at once (4.0x the 156 MiB
logits: logits, softmax, their gradient, and autograd's copies).

## Cross-checks

* **Full training step** (embedding -> gradients, 4 blocks):
  ordinary 816.4 MiB, chunked
  434.3 MiB, **1.88x**.  Smaller than the
  loss-step ratio because the blocks' activations are common to both.
* **Independent measurement**, fresh CPU process, peak RSS growth:
  ordinary 637 MiB, chunked 202 MiB
  (3.15x).  It agrees on direction and on
  "several times lower" but not on the ratio: RSS is coarse (allocator retention,
  page granularity, and the CPU allocator's reuse of freed blocks), so the tracker's
  live-tensor number is the one reported.

## How the chunked version works

1. Gather only contributing positions, so padded positions never get logits.
2. For each chunk of `512` rows: `logits = h_chunk @ W.T`, sum of cross-entropy, done.
3. Wrap each chunk in `torch.utils.checkpoint`: its logits are freed after the
   forward and recomputed in backward.  Without it autograd keeps every chunk's
   logits alive and the peak is back where it started.
4. Sum the chunks' *sums* and divide once by the total count.  Averaging chunk
   averages would be wrong whenever chunks hold different token counts.

The price is one extra `h @ W.T` per chunk in backward: memory is bought with FLOPs.
