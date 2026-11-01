"""Train the baseline and each upgrade, then measure quality, speed and KV cache gains."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from modern_gpt import ModernConfig, ModernGPT
from tinygpt.data import load_splits
from tinygpt.plots import INK_SECONDARY, SERIES, plt, style_axes
from tinygpt.training import TrainConfig, train

VARIANTS = {
    "Baseline (Project 4 design)": {},
    "Flash Attention": {"use_flash": True},
    "RMSNorm": {"use_rmsnorm": True},
    "RoPE": {"use_rope": True},
    "Grouped-query attention": {"n_kv_head": 2},
    "All upgrades": {"use_flash": True, "use_rmsnorm": True, "use_rope": True, "n_kv_head": 2},
}


def measure_generation(model, length, repeats=3):
    """Return tokens per second without and with the KV cache."""
    prompt = torch.zeros(1, 1, dtype=torch.long)
    speeds = []
    for method in (model.generate, model.generate_cached):
        best = float("inf")
        for _ in range(repeats):
            started = time.perf_counter()
            method(prompt, length)
            best = min(best, time.perf_counter() - started)
        speeds.append(length / best)
    return speeds


def plot(rows, path):
    rows = sorted(rows, key=lambda row: -row["val_mean"])
    y = range(len(rows))
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(rows) + 1.5))
    ax.hlines(y, [r["val_min"] for r in rows], [r["val_max"] for r in rows], color=SERIES[0], lw=1.6, alpha=0.45)
    ax.scatter([r["val_mean"] for r in rows], y, color=SERIES[0], s=46, edgecolors="#fcfcfb", linewidths=1.5, zorder=3)
    ax.set_yticks(list(y), [r["name"] for r in rows])
    style_axes(ax, grid="x")
    ax.margins(x=0.15, y=0.1)
    ax.set_xlabel("Best validation loss (lower is better)", color=INK_SECONDARY)
    ax.set_title("Upgrades compared: dot is the mean, line is the range across seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="../04-tiny-gpt/data")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--embd", type=int, default=96)
    parser.add_argument("--block", type=int, default=64)
    args = parser.parse_args(argv)

    if not (Path(args.data) / "train.txt").exists():
        parser.error(f"{args.data}/train.txt not found. Run prepare_data.py in Project 4 first.")

    tokenizer, train_ids, val_ids = load_splits(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, switches in VARIANTS.items():
        config = ModernConfig(
            tokenizer.vocab_size, args.block, args.layers, args.heads, args.embd, **switches
        )
        val_losses, per_iter, model = [], [], None
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = ModernGPT(config)
            train_config = TrainConfig(
                max_iters=args.iters, seed=seed, eval_every=max(1, args.iters // 8), eval_iters=30
            )
            history = train(model, train_ids, val_ids, train_config, log=None)
            val_losses.append(min(row["val_loss"] for row in history))
            per_iter.append(history[-1]["seconds"] / max(1, args.iters))
            print(f"{name:<30} seed {seed}  val {val_losses[-1]:.4f}  {history[-1]['seconds']:.0f}s", flush=True)
        slow, fast = measure_generation(model, args.block - 1)
        rows.append(
            {
                "name": name,
                "parameters": model.num_parameters(),
                "val_mean": statistics.mean(val_losses),
                "val_min": min(val_losses),
                "val_max": max(val_losses),
                "seconds_per_iter": statistics.mean(per_iter),
                "tokens_per_second": slow,
                "tokens_per_second_cached": fast,
            }
        )

    lines = [
        "# Modern architecture results",
        "",
        f"Model: {args.layers} layers, {args.heads} heads, {args.embd} embedding size, {args.block} block size. "
        f"{args.iters} iterations, {args.seeds} seeds.",
        "",
        "| Variant | Parameters | Best validation loss | Seed range | Seconds per iteration | Tokens/s, no cache | Tokens/s, KV cache |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(rows, key=lambda r: r["val_mean"]):
        lines.append(
            f"| {row['name']} | {row['parameters']:,} | {row['val_mean']:.4f} | {row['val_min']:.4f} to {row['val_max']:.4f} | "
            f"{row['seconds_per_iter']:.3f} | {row['tokens_per_second']:.0f} | {row['tokens_per_second_cached']:.0f} |"
        )
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(rows, indent=1))
    plot(rows, out / "comparison.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
