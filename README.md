# Project 5: Modern architecture upgrade

Upgrades the Tiny GPT from Project 4 with features used in modern models, testing
each one separately to see what it actually changes.

| Feature | What it does |
| --- | --- |
| Flash Attention | Uses PyTorch's optimized attention, with the same math |
| RMSNorm | A simpler normalization than LayerNorm |
| RoPE | Encodes position by rotating keys and queries, replacing the learned position table |
| Grouped-query attention | Several query heads share fewer key/value heads, which saves parameters |
| KV cache | Reuses earlier keys and values, so generation only processes the newest token |

## Results

RoPE gave the best validation loss, 1.7364 against 1.8238 for the baseline. Flash
Attention matched the baseline's loss and took less time per iteration. The KV cache
increased generation speed for every variant.

| Variant | Parameters | Best validation loss | Seconds per iteration | Tokens/s, no cache | Tokens/s, KV cache |
| --- | --- | --- | --- | --- | --- |
| RoPE | 352,416 | 1.7364 | 0.079 | 585 | 813 |
| All upgrades | 324,096 | 1.7469 | 0.093 | 356 | 683 |
| Baseline (Project 4 design) | 358,560 | 1.8238 | 0.106 | 365 | 548 |
| Flash Attention | 358,560 | 1.8238 | 0.071 | 840 | 1065 |
| RMSNorm | 357,888 | 1.8242 | 0.092 | 288 | 408 |
| Grouped-query attention | 330,912 | 1.8360 | 0.071 | 360 | 435 |

The model has 3 layers, 4 heads, a 96 embedding size and a 64 block size, trained for
2,000 iterations on 2 seeds. With only 2 seeds, treat small differences as ties. Speeds
come from one machine and can vary between runs.

Seed ranges are in `outputs\results.md`, and the chart is `outputs\comparison.png`.

## Run it

This uses the data from Project 4, so run its `prepare_data.py` first.

```powershell
uv sync
uv run python experiment.py
uv run pytest
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--iters` | `2000` | Training iterations per run |
| `--seeds` | `2` | Seeds per variant |
| `--layers` | `3` | Transformer blocks |
| `--heads` | `4` | Attention heads |
| `--embd` | `96` | Embedding size |
| `--block` | `64` | Longest context, in characters |

## Files

| File | What it does |
| --- | --- |
| `modern_gpt.py` | The model, with each feature behind a switch |
| `experiment.py` | Trains every variant and measures quality and speed |
| `tests/` | Includes checks that cached and uncached outputs match, and that Flash and manual attention agree |
