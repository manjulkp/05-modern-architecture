# Modern architecture results

Model: 3 layers, 4 heads, 96 embedding size, 64 block size. 2000 iterations, 2 seeds.

| Variant | Parameters | Best validation loss | Seed range | Seconds per iteration | Tokens/s, no cache | Tokens/s, KV cache |
| --- | --- | --- | --- | --- | --- | --- |
| RoPE | 352,416 | 1.7364 | 1.7327 to 1.7401 | 0.079 | 585 | 813 |
| All upgrades | 324,096 | 1.7469 | 1.7431 to 1.7508 | 0.093 | 356 | 683 |
| Baseline (Project 4 design) | 358,560 | 1.8238 | 1.8180 to 1.8296 | 0.106 | 365 | 548 |
| Flash Attention | 358,560 | 1.8238 | 1.8180 to 1.8296 | 0.071 | 840 | 1065 |
| RMSNorm | 357,888 | 1.8242 | 1.8202 to 1.8281 | 0.092 | 288 | 408 |
| Grouped-query attention | 330,912 | 1.8360 | 1.8256 to 1.8465 | 0.071 | 360 | 435 |
