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

**Not available yet.** The experiment hasn't finished. When it has, results will be in
`outputs\results.md` and `outputs\comparison.png`.

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
