"""
Plot PPO Reward vs. Training Step
==================================
Reads ppo_metrics.json from all 3 seed checkpoints and produces:
  • Individual seed traces (faint)
  • Mean ± 1 std shading across seeds
  • Rolling-average smoothed mean line
  • Horizontal dashed lines for before/after mean rewards

Output: results/c1/reward_curve.png
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Config ────────────────────────────────────────────────────

SEED_DIRS = {
    "seed=42":  "checkpoints/ppo/ppo_metrics.json",
    "seed=123": "checkpoints/ppo_seed123/ppo_metrics.json",
    "seed=456": "checkpoints/ppo_seed456/ppo_metrics.json",
}
OUT_DIR  = "results/c1"
OUT_FILE = os.path.join(OUT_DIR, "reward_curve.png")
SMOOTH_WINDOW = 10      # rolling-average window size


def rolling_mean(arr, w):
    """Simple causal rolling mean (pads start by repeating first value)."""
    arr = np.array(arr, dtype=float)
    out = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - w + 1)
        out[i] = arr[start : i + 1].mean()
    return out


def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load curves ───────────────────────────────────────────
    curves   = {}
    befores  = {}
    afters   = {}

    for label, path in SEED_DIRS.items():
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping {label}")
            continue
        m = load_metrics(path)
        curves[label]  = np.array(m["rewards_curve"], dtype=float)
        befores[label] = m["mean_reward_before"]
        afters[label]  = m["mean_reward_after"]
        print(f"  Loaded {label}: {len(curves[label])} steps, "
              f"before={befores[label]:.3f}, after={afters[label]:.3f}")

    if not curves:
        print("ERROR: No metrics files found. Did PPO training complete?")
        return

    # ── Align curve lengths (truncate to shortest) ────────────
    min_len = min(len(c) for c in curves.values())
    aligned = {k: v[:min_len] for k, v in curves.items()}
    steps   = np.arange(1, min_len + 1)

    stack = np.stack(list(aligned.values()))   # (n_seeds, steps)
    mean  = stack.mean(axis=0)
    std   = stack.std(axis=0)
    smooth_mean = rolling_mean(mean, SMOOTH_WINDOW)

    # ── Plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    COLORS = ["#4C72B0", "#DD8452", "#55A868"]

    # Individual seed traces
    for (label, curve), color in zip(aligned.items(), COLORS):
        ax.plot(steps, curve, alpha=0.18, color=color, linewidth=0.8)

    # Std shading
    ax.fill_between(steps, mean - std, mean + std,
                    alpha=0.15, color="#4C72B0", label="Mean ± 1 std")

    # Raw mean
    ax.plot(steps, mean, color="#4C72B0", linewidth=0.9, alpha=0.4)

    # Smoothed mean
    ax.plot(steps, smooth_mean, color="#1a1aff", linewidth=2.2,
            label=f"Smoothed mean (w={SMOOTH_WINDOW})")

    # Before / after dashed reference lines
    grand_before = np.mean(list(befores.values()))
    grand_after  = np.mean(list(afters.values()))
    ax.axhline(grand_before, color="#c0392b", linestyle="--", linewidth=1.4,
               label=f"Mean reward before PPO ({grand_before:.2f})")
    ax.axhline(grand_after,  color="#27ae60", linestyle="--", linewidth=1.4,
               label=f"Mean reward after PPO ({grand_after:.2f})")

    # Seed-coloured dots in legend (proxy artists)
    for (label, _), color in zip(aligned.items(), COLORS):
        ax.plot([], [], color=color, alpha=0.5, linewidth=2, label=label)

    ax.set_xlabel("PPO Step", fontsize=12)
    ax.set_ylabel("Reward Score", fontsize=12)
    ax.set_title("PPO Training: Reward vs. Step  (3 seeds, β=0.2)", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", alpha=0.15)

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=150)
    print(f"\nSaved: {OUT_FILE}")

    # ── Print summary table ───────────────────────────────────
    print("\n── Reward Summary ──────────────────────────────────────")
    print(f"{'Seed':<12} {'Before':>10} {'After':>10} {'Δ%':>10}")
    print("-" * 46)
    for label in curves:
        b, a = befores[label], afters[label]
        pct = (a - b) / abs(b) * 100
        print(f"{label:<12} {b:>10.3f} {a:>10.3f} {pct:>+9.1f}%")
    grand_pct = (grand_after - grand_before) / abs(grand_before) * 100
    print("-" * 46)
    print(f"{'Mean':<12} {grand_before:>10.3f} {grand_after:>10.3f} {grand_pct:>+9.1f}%")


if __name__ == "__main__":
    main()
