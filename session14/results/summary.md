| run | total params | active params / token | val loss at start | val loss at end | tok/s |
|---|---:|---:|---:|---:|---:|
| 1_dense | 17.4M | 17.4M | 9.083 | **4.409** | 13,346 |
| 3_dense_continued | 17.4M | 17.4M | 4.409 | **4.279** | 13,974 |
| 4_moe_copy | 83.5M | 26.9M | 4.409 | **4.261** | 5,899 |
| 5_moe_drop | 83.5M | 26.9M | 4.996 | **4.280** | 4,983 |
