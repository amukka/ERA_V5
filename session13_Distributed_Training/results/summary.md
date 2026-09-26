| run | batch | steps | final train loss | final val loss | tokens/s | minutes | peak memory (synced audit) | Metal driver peak | vs baseline: speed | vs baseline: memory |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (standard, b32) | 32 | 3,051 | 3.9759 | 3.9801 | 10,983 | 75.8 | 9.15 GiB | 11.56 GiB | 1.00x | 1.00x |
| reversible momentum (b32) | 32 | 3,051 | 4.0136 | 4.0206 | 8,925 | 93.3 | 2.65 GiB | 6.07 GiB | 0.81x | 0.29x |
| reversible momentum (max batch) | 88 | 1,109 | 4.1624 | 4.1808 | 7,776 | 107.0 | 5.93 GiB | 11.28 GiB | 0.71x | 0.65x |
