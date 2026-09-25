# E1 · Thirty-two virtual GPUs, and three collectives

World size 32. Every rank is a thread with its own tensors, its own slice of the corpus and `torch.set_num_threads(1)`, so a rank cannot quietly borrow the machine's other cores either.

## They really are separate

| ranks | OS threads | distinct batches | tokens per rank | tokens per step |
| ----: | ---------: | ---------------: | --------------: | --------------: |
|    32 |         32 |               32 |             256 |           8,192 |

The first bytes each rank drew, to show the text differs and not just a hash:

| rank | thread | first 24 bytes of its first sequence |
| ---- | ------ | ------------------------------------ |
| 0    | gpu00  | ` at home against Wake Fo`           |
| 1    | gpu01  | `ified Chinese : 陈星�`                |
| 2    | gpu02  | `      'sigma_8': 0.8159,`           |
| 3    | gpu03  | `ndations from Bob Dowie `           |

A replacement character in that column is not a bug: the vocabulary is the 256 byte values, so a sequence boundary is free to land in the middle of a multi-byte codepoint. Nothing in this session depends on the tokenizer, and a byte vocabulary keeps the embedding table small enough that the sharding results below are about the transformer rather than about one large matrix.

## reduce-scatter + all-gather = all-reduce

32 ranks, 65,536 random fp32 numbers each. Each rank keeps a shard of 2,048 numbers in between.

| check                                           | result  |
| ----------------------------------------------- | ------- |
| bitwise equal on every rank                     | True    |
| max \|all_reduce − all_gather(reduce_scatter)\| | 0.0e+00 |
| every rank ended with the same answer           | True    |

This is the identity every later result leans on. Stage 1 and stage 2 do not send less than data parallelism does — they send the *same two phases* and keep the slice that appears in between them, which data parallelism computes and throws away.

## The ring, counted send by send

`ring_all_reduce` performs the sends one at a time, each rank only ever handing a chunk to `rank+1`. For 32 ranks that is 62 sends per rank (62 expected: 31 to reduce and 31 to gather), each one chunk of N/32.

|                                   | bytes per rank |
| --------------------------------- | -------------: |
| bytes counted, one send at a time |        507,904 |
| 2N(W−1)/W                         |        507,904 |
| in units of N                     |         1.9375 |
| identical                         |           True |

So `2P` is not a convention, it is a count. The ring's *answer* agrees with the direct reduction to 2.4e-07 absolute but is not bitwise identical to it (`bitwise_equal: False`), because a ring adds the ranks in ring order and a direct reduction adds them in rank order, and floating point addition is not associative. Real NCCL has exactly this property, which is why changing the world size of a real run changes its loss curve in the last decimal place.

## The cost, as the world grows

| world size | all-reduce | reduce-scatter | all-gather |
| ---------: | ---------: | -------------: | ---------: |
|          1 |    0.0000N |        0.0000N |    0.0000N |
|          2 |    1.0000N |        0.5000N |    0.5000N |
|          4 |    1.5000N |        0.7500N |    0.7500N |
|          8 |    1.7500N |        0.8750N |    0.8750N |
|         16 |    1.8750N |        0.9375N |    0.9375N |
|         32 |    1.9375N |        0.9688N |    0.9688N |

The cost per rank does not grow with the world — it *approaches* 2N and N from below and stops. Doubling the GPUs does not double anybody's traffic; it only stops the (W−1)/W discount from helping. That is the property that makes data parallelism scale at all.

## Pricing it for our model

30B parameters in bf16 is P = 60 GB.

| volume                                |  bytes | NVLink 450 GB/s | InfiniBand 50 GB/s | PCIe 60 GB/s |
| ------------------------------------- | -----: | --------------: | -----------------: | -----------: |
| 2P (data parallelism, ZeRO-1, ZeRO-2) | 120 GB |          0.27 s |             2.40 s |       2.00 s |
| 3P (ZeRO-3)                           | 180 GB |          0.40 s |             3.60 s |       3.00 s |

![](e1_mesh.png)
