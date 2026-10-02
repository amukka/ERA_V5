#!/bin/bash
# Full pipeline. ~1.5 h on an Apple M4. Logs: results/runs/*.jsonl and results/logs/*.log
set -e
cd "$(dirname "$0")"
mkdir -p results/logs
P=python3
D=20_000_000   # dense pre-training tokens (rows 0..)
C=6_000_000    # continuation tokens, identical rows for all three continuations
$P -m src.train dense --name 1_dense --tokens $D --lr 1e-3 --warmup 100 --save dense.pt 2>&1 | tee results/logs/1_dense.log
$P -m src.check_lossless 2>&1 | tee results/logs/2_check_lossless.log
$P -m src.train dense --name 3_dense_continued --init results/ckpt/dense.pt --offset-tokens $D --tokens $C --lr 5e-4 --warmup 50 2>&1 | tee results/logs/3_dense_continued.log
$P -m src.train moe --name 4_moe_copy --init results/ckpt/dense.pt --offset-tokens $D --tokens $C --lr 5e-4 --warmup 50 --drop 0 --n-exp 8 --top-k 2 2>&1 | tee results/logs/4_moe_copy.log
$P -m src.train moe --name 5_moe_drop --init results/ckpt/dense.pt --offset-tokens $D --tokens $C --lr 5e-4 --warmup 50 --drop 0.5 --n-exp 8 --top-k 2 2>&1 | tee results/logs/5_moe_drop.log
