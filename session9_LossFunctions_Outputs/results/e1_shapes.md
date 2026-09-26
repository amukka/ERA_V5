# 1 · Every tensor shape, and what each dimension is

`B=4` rows, `T=128` positions, `D=256` d_model, `V=10,002` vocab,
`F=704` d_ff, `H=4` heads, `Dh=64` per-head width.
The batch is padded on purpose (320 real tokens in
512 slots), so the mask below is not trivially all-True.

Every row was emitted by the code that built the tensor, via the `trace`
argument threaded through `model.hidden()` and `lm_loss()`.

## Getting to a hidden state

| tensor    | shape             | numbers |  MiB | what each dimension selects                                                                                                        |
| --------- | ----------------- | ------: | ---: | ---------------------------------------------------------------------------------------------------------------------------------- |
| tokens    | 4 x 128           |     512 | 0.00 | B=rows in the batch, T=positions; each entry is a token id in [0, V).  Unshifted -- this is what the harness is handed             |
| valid     | 4 x 128           |     512 | 0.00 | B=rows, T=positions; True where the position holds a real token rather than padding                                                |
| tok_emb   | 4 x 128 x 256     | 131,072 | 0.50 | B=rows, T=positions, D=d_model; one learned vector per token id                                                                    |
| pos_emb   | 1 x 128 x 256     |  32,768 | 0.12 | 1=broadcast over rows, T=positions, D=d_model; one learned vector per absolute position                                            |
| resid_in  | 4 x 128 x 256     | 131,072 | 0.50 | B=rows, T=positions, D=d_model; the residual stream as it enters block 0                                                           |
| attn_bias | 4 x 1 x 128 x 128 |  65,536 | 0.25 | B=rows, 1=broadcast over heads, T_q=querying position, T_k=attended position; 0 where the edge is allowed and -inf where it is not |
| hidden    | 4 x 128 x 256     | 131,072 | 0.50 | B=rows, T=positions, D=d_model; the final hidden state. Every token now has a vector that has seen its context                     |

## One block, in full

The other 3 blocks are these 15 tensors again with a
different prefix — 45 more tensors, same shapes.

| tensor              | shape             | numbers |  MiB | what each dimension selects                                                                                   |
| ------------------- | ----------------- | ------: | ---: | ------------------------------------------------------------------------------------------------------------- |
| block0.norm1        | 4 x 128 x 256     | 131,072 | 0.50 | B=rows, T=positions, D=d_model; the residual stream normalised per position before attention reads it         |
| block0.attn.qkv     | 4 x 128 x 768     | 393,216 | 1.50 | B=rows in the batch, T=positions, 3D=query, key and value packed into one projection so it is a single matmul |
| block0.attn.q       | 4 x 4 x 128 x 64  | 131,072 | 0.50 | B=rows, H=heads, T=the position asking, Dh=d_model/H, the slice of the channel this head owns                 |
| block0.attn.k       | 4 x 4 x 128 x 64  | 131,072 | 0.50 | B=rows, H=heads, T=the position being asked about, Dh=per-head width                                          |
| block0.attn.v       | 4 x 4 x 128 x 64  | 131,072 | 0.50 | B=rows, H=heads, T=the position whose content gets carried, Dh=per-head width                                 |
| block0.attn.scores  | 4 x 4 x 128 x 128 | 262,144 | 1.00 | B=rows, H=heads, T_q=querying position, T_k=attended position; entry (b,h,i,j) is how much i wants j          |
| block0.attn.weights | 4 x 4 x 128 x 128 | 262,144 | 1.00 | same axes as scores, now a distribution over T_k for each (row, head, T_q) -- each slice sums to 1            |
| block0.attn.context | 4 x 4 x 128 x 64  | 131,072 | 0.50 | B=rows, H=heads, T=positions, Dh=per-head width; the value vectors averaged with the attention weights        |
| block0.attn.merged  | 4 x 128 x 256     | 131,072 | 0.50 | B=rows, T=positions, D=d_model; the heads concatenated back into one channel axis                             |
| block0.attn.out     | 4 x 128 x 256     | 131,072 | 0.50 | B=rows, T=positions, D=d_model; attention's contribution to the residual stream                               |
| block0.resid_attn   | 4 x 128 x 256     | 131,072 | 0.50 | B=rows, T=positions, D=d_model; residual stream with attention's answer added back -- nothing overwritten     |
| block0.norm2        | 4 x 128 x 256     | 131,072 | 0.50 | B=rows, T=positions, D=d_model; normalised again before the FFN reads it                                      |
| block0.ffn.hidden   | 4 x 128 x 704     | 360,448 | 1.38 | B=rows, T=positions, F=d_ff, the wide inner width; one branch decides what to pass, the other how much        |
| block0.ffn.out      | 4 x 128 x 256     | 131,072 | 0.50 | B=rows, T=positions, D=d_model; the FFN's contribution to the residual stream                                 |
| block0.resid_ffn    | 4 x 128 x 256     | 131,072 | 0.50 | B=rows, T=positions, D=d_model; residual stream leaving this block                                            |

## From hidden state to one scalar

This is the session. 8 tensors, and the widest thing in the whole
forward pass is in here rather than in attention.

| tensor         | shape           |   numbers |   MiB | what each dimension selects                                                                                                                                                      |
| -------------- | --------------- | --------: | ----: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| logits         | 4 x 128 x 10002 | 5,121,024 | 19.54 | B=rows, T=positions, V=vocab size; one unnormalised score for every token the model could name next, at every position.  V/D = 39x larger than the hidden state that produced it |
| logits[:, :-1] | 4 x 127 x 10002 | 5,081,016 | 19.38 | B=rows, P=positions that have something to predict, V=vocab; the last positions are dropped because nothing follows them                                                         |
| targets        | 4 x 127         |       508 |  0.00 | B=rows, P=pairs; the token id each position must name. Position i of the row is token i+1 of the sequence                                                                        |
| flat_logits    | 508 x 10002     | 5,081,016 | 19.38 | N=B*P, one row per prediction the batch will be scored on, V=vocab; cross_entropy has no use for the batch and position axes, so they are folded into one                        |
| flat_targets   | 508             |       508 |  0.00 | N=B*P; one correct id per row of flat_logits, in the same order                                                                                                                  |
| per_token_loss | 4 x 127         |       508 |  0.00 | B=rows, P=pairs; -log(probability the model gave the true token) at each position, before anything is masked                                                                     |
| loss_mask      | 4 x 127         |       508 |  0.00 | B=rows, P=pairs; True where the pair is allowed to contribute. Its sum is the denominator, and it is not B*P                                                                     |
| loss           |                 |         1 |  0.00 | a scalar in nats: the mean surprise per contributing token. This is the one number the whole forward pass exists to produce                                                      |

## The parameters

| parameter                 | shape       |   numbers |  MiB |
| ------------------------- | ----------- | --------: | ---: |
| tok_emb.weight            | 10002 x 256 | 2,560,512 | 9.77 |
| pos_emb.weight            | 256 x 256   |    65,536 | 0.25 |
| blocks.0.norm1.weight     | 256         |       256 | 0.00 |
| blocks.0.attn.qkv.weight  | 768 x 256   |   196,608 | 0.75 |
| blocks.0.attn.proj.weight | 256 x 256   |    65,536 | 0.25 |
| blocks.0.norm2.weight     | 256         |       256 | 0.00 |
| blocks.0.ffn.gate.weight  | 704 x 256   |   180,224 | 0.69 |
| blocks.0.ffn.up.weight    | 704 x 256   |   180,224 | 0.69 |
| blocks.0.ffn.down.weight  | 256 x 704   |   180,224 | 0.69 |
| blocks.1.norm1.weight     | 256         |       256 | 0.00 |
| blocks.1.attn.qkv.weight  | 768 x 256   |   196,608 | 0.75 |
| blocks.1.attn.proj.weight | 256 x 256   |    65,536 | 0.25 |
| blocks.1.norm2.weight     | 256         |       256 | 0.00 |
| blocks.1.ffn.gate.weight  | 704 x 256   |   180,224 | 0.69 |
| blocks.1.ffn.up.weight    | 704 x 256   |   180,224 | 0.69 |
| blocks.1.ffn.down.weight  | 256 x 704   |   180,224 | 0.69 |
| blocks.2.norm1.weight     | 256         |       256 | 0.00 |
| blocks.2.attn.qkv.weight  | 768 x 256   |   196,608 | 0.75 |
| blocks.2.attn.proj.weight | 256 x 256   |    65,536 | 0.25 |
| blocks.2.norm2.weight     | 256         |       256 | 0.00 |
| blocks.2.ffn.gate.weight  | 704 x 256   |   180,224 | 0.69 |
| blocks.2.ffn.up.weight    | 704 x 256   |   180,224 | 0.69 |
| blocks.2.ffn.down.weight  | 256 x 704   |   180,224 | 0.69 |
| blocks.3.norm1.weight     | 256         |       256 | 0.00 |
| blocks.3.attn.qkv.weight  | 768 x 256   |   196,608 | 0.75 |
| blocks.3.attn.proj.weight | 256 x 256   |    65,536 | 0.25 |
| blocks.3.norm2.weight     | 256         |       256 | 0.00 |
| blocks.3.ffn.gate.weight  | 704 x 256   |   180,224 | 0.69 |
| blocks.3.ffn.up.weight    | 704 x 256   |   180,224 | 0.69 |
| blocks.3.ffn.down.weight  | 256 x 704   |   180,224 | 0.69 |
| norm_f.weight             | 256         |       256 | 0.00 |
| head.weight               | 10002 x 256 | 2,560,512 | 9.77 |

## What the shapes say

**The logits are the largest tensor in the model, and it is not close.** The
hidden state is `4 x 128 x 256` = 0.50 MiB. The
logits are `4 x 128 x 10002` = 19.54 MiB — a factor
of **39.1×**, which is exactly `V/D` =
10,002/256 = 39.1. Every axis of
the hidden state survives into the logits; the channel axis is simply replaced
by one 10,002-wide axis. Nothing about attention is involved, and the
tensor exists only to be collapsed into one number. That is deliverable 7.

**Two axes go into the loss and zero come out.** `logits[:, :-1]` is
`4 x 127 x 10002` and `flat_logits` is
`508 x 10002` — the batch and position axes folded
into one, because cross-entropy has no use for either. It scores rows, and
4 × 127 = 508 of them arrive.

**The denominator is not `B × (T-1)`.** 316 of
508 pairs contribute; the rest are padding. That is deliverable 3.
