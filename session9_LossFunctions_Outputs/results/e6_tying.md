# 6 · Tied against untied, on this configuration

`V = 10,002`, `D = 256`, 4 blocks.

| parameter group                  |        untied |                tied |
| -------------------------------- | ------------: | ------------------: |
| input embedding `tok_emb` [V, D] |     2,560,512 |           2,560,512 |
| position embedding [max_seq, D]  |        65,536 |              65,536 |
| 4 transformer blocks             |     3,213,568 |           3,213,568 |
| output head `head` [V, D]        |     2,560,512 | 0 - the same tensor |
| **total**                        | **8,400,128** |       **5,839,616** |

**Tying saves 2,560,512 parameters, 30.5%
of the model** - 9.8 MiB of fp32 weights, and three times
that once AdamW's two moments are counted. The head alone is
30.5% of the untied model: the single largest
tensor in it, and 80% of the
size of all 4 transformer blocks put together
(2,560,512 against 3,213,568). One matrix, no
attention in it, weighing nearly as much as the entire depth of the model.

## Does the saving cost anything?

| arm    | parameters | final training loss | held-out loss |
| ------ | ---------: | ------------------: | ------------: |
| untied |  8,400,128 |              4.2748 |        4.1085 |
| tied   |  5,839,616 |              4.3986 |        4.2360 |

Same seed, same batches, 400 steps. The tied model is
30.5% smaller and lands
+0.1275 nats behind on held-out loss - a real cost, and larger than this run's step-to-step noise.

Read that as measured rather than as a verdict on tying: the 2,560,512 parameters tying removes were doing something here, and the 30.5% saving is not free. One pair of
400-step runs at `D=256` is not an ablation, and the published result
that tying "often helps quality" comes from models where the head is a far
larger share of the budget and the data is far larger relative to it. What this
run does establish is the direction of the trade on *this* configuration: the
30.5% is bought, not found.

The coupling is the real cost and it does not show up in a loss curve: the
vector that means *"this token just arrived"* is forced to be the vector that
means *"predict this token"*. Those are different jobs.

## The same subtraction at V5's width

|                                 |                       V5 |
| ------------------------------- | -----------------------: |
| vocabulary                      |                  131,072 |
| d_model                         |                    4,096 |
| output head, V x D              | **536,870,912** (536.9M) |
| as bf16 weights                 |                 1.00 GiB |
| session 7's factored input side |       33,554,432 (33.6M) |
| head / input side               |                **16.0x** |

At this width tying would save 536.9M parameters - and it
is not available. Session 7 replaced the dense `[V, D]` input table with a byte
codec plus one 33.6M projection, so there is no
input embedding matrix left to tie *to*. You cannot share rows with a thing that
has no rows.

That is the shape of the problem this session leaves open: the front door was
factored and won 93.75%, and the back door is still a dense
536.9M matrix, 16.0x the
size of the input side that replaced it. The standard escape is closed, so the
escape has to be a factored head - and that is an ablation nobody has run for a
byte-codec input side.
