# Deliverable 1 — every tensor shape in one step

Model: 4 layers, d_model 256, 4 heads, d_ff 1024, vocab 10001, 8,336,384 parameters.

Step: micro-batch 8 rows × 4 accumulation steps = a global batch of 32 sequences and 4,055 valid tokens.

Micro-batch token counts this step: 899, 940, 995, 1,221 — different, because the sequences are different lengths. Deliverable 3 is about what that does.


## The axis names

| axis | what one index along it selects                                                                                       |
| ---- | --------------------------------------------------------------------------------------------------------------------- |
| B    | rows in one micro-batch -- what actually fits on the device                                                           |
| T    | positions in the sequence; micro-batch is padded to its own longest row, so T changes from micro-batch to micro-batch |
| D    | d_model, the width of the residual stream                                                                             |
| H    | attention heads                                                                                                       |
| Dh   | D/H, the slice of the channel one head owns                                                                           |
| F    | d_ff, the wide inner width of the MLP                                                                                 |
| V    | vocabulary size -- one score per token the model could name next                                                      |
| T_q  | the position doing the asking, in an attention score matrix                                                           |
| T_k  | the position being asked about                                                                                        |

## 1. Activations — one micro-batch, forward

(69 tensors, from token ids to logits. This micro-batch is B=8, T=192.)

| tensor              | shape       | dtype   |   elements | what each dimension means                                                                                                                                         |
| ------------------- | ----------- | ------- | ---------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| inputs              | 8×192       | int64   |      1,536 | B=rows in the micro-batch, T=positions; each entry is a token id in [0, V)                                                                                        |
| mask                | 8×192       | bool    |      1,536 | B=rows, T=positions; True where the position holds a real token rather than padding                                                                               |
| tok_emb             | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; one learned vector per token id                                                                                                   |
| pos_emb             | 1×192×256   | float32 |     49,152 | 1=broadcast over rows, T=positions, D=d_model; one learned vector per absolute position                                                                           |
| resid_in            | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; the residual stream as it enters block 0                                                                                          |
| attn_bias           | 8×1×192×192 | float32 |    294,912 | B=rows, 1=broadcast over heads, T_q=querying position, T_k=attended position; 0 where the edge is allowed and -inf where it is not (future positions and padding) |
| block0.ln1          | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; the residual stream normalised per position before attention reads it                                                             |
| block0.attn.qkv     | 8×192×768   | float32 |  1,179,648 | B=rows in the micro-batch, T=positions, 3D=query, key and value packed into one projection so it is a single matmul                                               |
| block0.attn.q       | 8×4×192×64  | float32 |    393,216 | B=rows, H=attention heads, T=positions asking the question, Dh=d_model/H, the slice of the channel each head owns                                                 |
| block0.attn.k       | 8×4×192×64  | float32 |    393,216 | B=rows, H=heads, T=positions being asked about, Dh=per-head width                                                                                                 |
| block0.attn.v       | 8×4×192×64  | float32 |    393,216 | B=rows, H=heads, T=positions whose content gets carried, Dh=per-head width                                                                                        |
| block0.attn.scores  | 8×4×192×192 | float32 |  1,179,648 | B=rows, H=heads, T_q=querying position, T_k=attended position; entry (b,h,i,j) is how much position i wants position j                                            |
| block0.attn.weights | 8×4×192×192 | float32 |  1,179,648 | same axes as scores, now a distribution over T_k for each (row, head, T_q) -- each slice sums to 1                                                                |
| block0.attn.context | 8×4×192×64  | float32 |    393,216 | B=rows, H=heads, T=positions, Dh=per-head width; the value vectors averaged with the attention weights                                                            |
| block0.attn.merged  | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; the heads concatenated back into one channel axis                                                                                 |
| block0.attn.out     | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; attention's contribution to the residual stream                                                                                   |
| block0.resid_attn   | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; residual stream after attention has been added back                                                                               |
| block0.ln2          | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; residual stream normalised again before the MLP reads it                                                                          |
| block0.mlp.hidden   | 8×192×1024  | float32 |  1,572,864 | B=rows, T=positions, F=d_ff, the wide inner width where the per-position nonlinearity happens                                                                     |
| block0.mlp.out      | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; the MLP's contribution to the residual stream                                                                                     |
| block0.resid_mlp    | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; residual stream leaving this block                                                                                                |
| block1.ln1          | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; the residual stream normalised per position before attention reads it                                                             |
| block1.attn.qkv     | 8×192×768   | float32 |  1,179,648 | B=rows in the micro-batch, T=positions, 3D=query, key and value packed into one projection so it is a single matmul                                               |
| block1.attn.q       | 8×4×192×64  | float32 |    393,216 | B=rows, H=attention heads, T=positions asking the question, Dh=d_model/H, the slice of the channel each head owns                                                 |
| block1.attn.k       | 8×4×192×64  | float32 |    393,216 | B=rows, H=heads, T=positions being asked about, Dh=per-head width                                                                                                 |
| block1.attn.v       | 8×4×192×64  | float32 |    393,216 | B=rows, H=heads, T=positions whose content gets carried, Dh=per-head width                                                                                        |
| block1.attn.scores  | 8×4×192×192 | float32 |  1,179,648 | B=rows, H=heads, T_q=querying position, T_k=attended position; entry (b,h,i,j) is how much position i wants position j                                            |
| block1.attn.weights | 8×4×192×192 | float32 |  1,179,648 | same axes as scores, now a distribution over T_k for each (row, head, T_q) -- each slice sums to 1                                                                |
| block1.attn.context | 8×4×192×64  | float32 |    393,216 | B=rows, H=heads, T=positions, Dh=per-head width; the value vectors averaged with the attention weights                                                            |
| block1.attn.merged  | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; the heads concatenated back into one channel axis                                                                                 |
| block1.attn.out     | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; attention's contribution to the residual stream                                                                                   |
| block1.resid_attn   | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; residual stream after attention has been added back                                                                               |
| block1.ln2          | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; residual stream normalised again before the MLP reads it                                                                          |
| block1.mlp.hidden   | 8×192×1024  | float32 |  1,572,864 | B=rows, T=positions, F=d_ff, the wide inner width where the per-position nonlinearity happens                                                                     |
| block1.mlp.out      | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; the MLP's contribution to the residual stream                                                                                     |
| block1.resid_mlp    | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; residual stream leaving this block                                                                                                |
| block2.ln1          | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; the residual stream normalised per position before attention reads it                                                             |
| block2.attn.qkv     | 8×192×768   | float32 |  1,179,648 | B=rows in the micro-batch, T=positions, 3D=query, key and value packed into one projection so it is a single matmul                                               |
| block2.attn.q       | 8×4×192×64  | float32 |    393,216 | B=rows, H=attention heads, T=positions asking the question, Dh=d_model/H, the slice of the channel each head owns                                                 |
| block2.attn.k       | 8×4×192×64  | float32 |    393,216 | B=rows, H=heads, T=positions being asked about, Dh=per-head width                                                                                                 |
| block2.attn.v       | 8×4×192×64  | float32 |    393,216 | B=rows, H=heads, T=positions whose content gets carried, Dh=per-head width                                                                                        |
| block2.attn.scores  | 8×4×192×192 | float32 |  1,179,648 | B=rows, H=heads, T_q=querying position, T_k=attended position; entry (b,h,i,j) is how much position i wants position j                                            |
| block2.attn.weights | 8×4×192×192 | float32 |  1,179,648 | same axes as scores, now a distribution over T_k for each (row, head, T_q) -- each slice sums to 1                                                                |
| block2.attn.context | 8×4×192×64  | float32 |    393,216 | B=rows, H=heads, T=positions, Dh=per-head width; the value vectors averaged with the attention weights                                                            |
| block2.attn.merged  | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; the heads concatenated back into one channel axis                                                                                 |
| block2.attn.out     | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; attention's contribution to the residual stream                                                                                   |
| block2.resid_attn   | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; residual stream after attention has been added back                                                                               |
| block2.ln2          | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; residual stream normalised again before the MLP reads it                                                                          |
| block2.mlp.hidden   | 8×192×1024  | float32 |  1,572,864 | B=rows, T=positions, F=d_ff, the wide inner width where the per-position nonlinearity happens                                                                     |
| block2.mlp.out      | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; the MLP's contribution to the residual stream                                                                                     |
| block2.resid_mlp    | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; residual stream leaving this block                                                                                                |
| block3.ln1          | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; the residual stream normalised per position before attention reads it                                                             |
| block3.attn.qkv     | 8×192×768   | float32 |  1,179,648 | B=rows in the micro-batch, T=positions, 3D=query, key and value packed into one projection so it is a single matmul                                               |
| block3.attn.q       | 8×4×192×64  | float32 |    393,216 | B=rows, H=attention heads, T=positions asking the question, Dh=d_model/H, the slice of the channel each head owns                                                 |
| block3.attn.k       | 8×4×192×64  | float32 |    393,216 | B=rows, H=heads, T=positions being asked about, Dh=per-head width                                                                                                 |
| block3.attn.v       | 8×4×192×64  | float32 |    393,216 | B=rows, H=heads, T=positions whose content gets carried, Dh=per-head width                                                                                        |
| block3.attn.scores  | 8×4×192×192 | float32 |  1,179,648 | B=rows, H=heads, T_q=querying position, T_k=attended position; entry (b,h,i,j) is how much position i wants position j                                            |
| block3.attn.weights | 8×4×192×192 | float32 |  1,179,648 | same axes as scores, now a distribution over T_k for each (row, head, T_q) -- each slice sums to 1                                                                |
| block3.attn.context | 8×4×192×64  | float32 |    393,216 | B=rows, H=heads, T=positions, Dh=per-head width; the value vectors averaged with the attention weights                                                            |
| block3.attn.merged  | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; the heads concatenated back into one channel axis                                                                                 |
| block3.attn.out     | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; attention's contribution to the residual stream                                                                                   |
| block3.resid_attn   | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; residual stream after attention has been added back                                                                               |
| block3.ln2          | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; residual stream normalised again before the MLP reads it                                                                          |
| block3.mlp.hidden   | 8×192×1024  | float32 |  1,572,864 | B=rows, T=positions, F=d_ff, the wide inner width where the per-position nonlinearity happens                                                                     |
| block3.mlp.out      | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; the MLP's contribution to the residual stream                                                                                     |
| block3.resid_mlp    | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; residual stream leaving this block                                                                                                |
| ln_f                | 8×192×256   | float32 |    393,216 | B=rows, T=positions, D=d_model; residual stream normalised one last time before the output head                                                                   |
| logits              | 8×192×10001 | float32 | 15,361,536 | B=rows, T=positions, V=vocab size; an unnormalised score for every token the model could name next, at every position                                             |
| per_token_loss      | 1536        | float32 |      1,536 | B*T=every position in the micro-batch flattened; the cross entropy of the true next token at that position, zeroed wherever the mask says there is no target      |

## 2. The scalars

| tensor                      | shape | dtype   | meaning                                                                                                                         |
| --------------------------- | ----- | ------- | ------------------------------------------------------------------------------------------------------------------------------- |
| loss_sum (per micro-batch)  | ()    | float32 | the summed cross entropy of one micro-batch -- a scalar with no dimensions, which is the whole point: one number for 899 tokens |
| loss (step, token-weighted) | ()    | float32 | loss_sum over all micro-batches divided by the total valid tokens in the global batch (4,055)                                   |
| grad_norm                   | ()    | float32 | one L2 norm over every gradient in the model concatenated -- the single number section 12 says to watch                         |

## 3. Parameters, and the gradient of each

| parameter                 | shape     |  elements | grad shape | same? | what each dimension means                                                            |
| ------------------------- | --------- | --------: | ---------- | ----- | ------------------------------------------------------------------------------------ |
| tok_emb.weight            | 10001×256 | 2,560,256 | 10001×256  | yes   | V=one row per token id, D=the vector it maps to                                      |
| pos_emb.weight            | 256×256   |    65,536 | 256×256    | yes   | T_max=one row per absolute position, D=the vector added to whatever token sits there |
| blocks.0.ln1.weight       | 256       |       256 | 256        | yes   | D=one gain per channel                                                               |
| blocks.0.ln1.bias         | 256       |       256 | 256        | yes   | D=one shift per channel                                                              |
| blocks.0.attn.qkv.weight  | 768×256   |   196,608 | 768×256    | yes   | 3D=query, key and value stacked, D=input width                                       |
| blocks.0.attn.proj.weight | 256×256   |    65,536 | 256×256    | yes   | D=output width back into the residual stream, D=concatenated head width in           |
| blocks.0.ln2.weight       | 256       |       256 | 256        | yes   | D=one gain per channel                                                               |
| blocks.0.ln2.bias         | 256       |       256 | 256        | yes   | D=one shift per channel                                                              |
| blocks.0.mlp.fc.weight    | 1024×256  |   262,144 | 1024×256   | yes   | F=d_ff out, D=d_model in                                                             |
| blocks.0.mlp.out.weight   | 256×1024  |   262,144 | 256×1024   | yes   | D=d_model out, F=d_ff in                                                             |
| blocks.1.ln1.weight       | 256       |       256 | 256        | yes   | D=one gain per channel                                                               |
| blocks.1.ln1.bias         | 256       |       256 | 256        | yes   | D=one shift per channel                                                              |
| blocks.1.attn.qkv.weight  | 768×256   |   196,608 | 768×256    | yes   | 3D=query, key and value stacked, D=input width                                       |
| blocks.1.attn.proj.weight | 256×256   |    65,536 | 256×256    | yes   | D=output width back into the residual stream, D=concatenated head width in           |
| blocks.1.ln2.weight       | 256       |       256 | 256        | yes   | D=one gain per channel                                                               |
| blocks.1.ln2.bias         | 256       |       256 | 256        | yes   | D=one shift per channel                                                              |
| blocks.1.mlp.fc.weight    | 1024×256  |   262,144 | 1024×256   | yes   | F=d_ff out, D=d_model in                                                             |
| blocks.1.mlp.out.weight   | 256×1024  |   262,144 | 256×1024   | yes   | D=d_model out, F=d_ff in                                                             |
| blocks.2.ln1.weight       | 256       |       256 | 256        | yes   | D=one gain per channel                                                               |
| blocks.2.ln1.bias         | 256       |       256 | 256        | yes   | D=one shift per channel                                                              |
| blocks.2.attn.qkv.weight  | 768×256   |   196,608 | 768×256    | yes   | 3D=query, key and value stacked, D=input width                                       |
| blocks.2.attn.proj.weight | 256×256   |    65,536 | 256×256    | yes   | D=output width back into the residual stream, D=concatenated head width in           |
| blocks.2.ln2.weight       | 256       |       256 | 256        | yes   | D=one gain per channel                                                               |
| blocks.2.ln2.bias         | 256       |       256 | 256        | yes   | D=one shift per channel                                                              |
| blocks.2.mlp.fc.weight    | 1024×256  |   262,144 | 1024×256   | yes   | F=d_ff out, D=d_model in                                                             |
| blocks.2.mlp.out.weight   | 256×1024  |   262,144 | 256×1024   | yes   | D=d_model out, F=d_ff in                                                             |
| blocks.3.ln1.weight       | 256       |       256 | 256        | yes   | D=one gain per channel                                                               |
| blocks.3.ln1.bias         | 256       |       256 | 256        | yes   | D=one shift per channel                                                              |
| blocks.3.attn.qkv.weight  | 768×256   |   196,608 | 768×256    | yes   | 3D=query, key and value stacked, D=input width                                       |
| blocks.3.attn.proj.weight | 256×256   |    65,536 | 256×256    | yes   | D=output width back into the residual stream, D=concatenated head width in           |
| blocks.3.ln2.weight       | 256       |       256 | 256        | yes   | D=one gain per channel                                                               |
| blocks.3.ln2.bias         | 256       |       256 | 256        | yes   | D=one shift per channel                                                              |
| blocks.3.mlp.fc.weight    | 1024×256  |   262,144 | 1024×256   | yes   | F=d_ff out, D=d_model in                                                             |
| blocks.3.mlp.out.weight   | 256×1024  |   262,144 | 256×1024   | yes   | D=d_model out, F=d_ff in                                                             |
| ln_f.weight               | 256       |       256 | 256        | yes   | D=one gain per channel                                                               |
| ln_f.bias                 | 256       |       256 | 256        | yes   | D=one shift per channel                                                              |
| head.weight               | 10001×256 | 2,560,256 | 10001×256  | yes   | V=one row per output token, D=the direction in the residual stream that votes for it |

Every gradient has the shape of its weight: 8,336,384 gradient numbers for 8,336,384 weights. That is the definition from section 2 — a gradient belongs to one weight.


## 4. What the optimiser keeps, per weight

| state tensor                      | shape     |  elements | meaning                                                       |
| --------------------------------- | --------- | --------: | ------------------------------------------------------------- |
| tok_emb.weight :: exp_avg         | 10001×256 | 2,560,256 | the same shape as the weight -- one running number per weight |
| tok_emb.weight :: exp_avg_sq      | 10001×256 | 2,560,256 | the same shape as the weight -- one running number per weight |
| pos_emb.weight :: exp_avg         | 256×256   |    65,536 | the same shape as the weight -- one running number per weight |
| pos_emb.weight :: exp_avg_sq      | 256×256   |    65,536 | the same shape as the weight -- one running number per weight |
| blocks.0.ln1.weight :: exp_avg    | 256       |       256 | the same shape as the weight -- one running number per weight |
| blocks.0.ln1.weight :: exp_avg_sq | 256       |       256 | the same shape as the weight -- one running number per weight |

AdamW holds 16,672,768 extra numbers for 8,336,384 weights — exactly two per weight. With a bf16 weight (2 B), a bf16 gradient (2 B), an fp32 master copy (4 B) and these two fp32 moments (8 B), that is the 16 bytes per weight of section 13: 127.2 MiB of training state for this model, before a single activation is stored.
