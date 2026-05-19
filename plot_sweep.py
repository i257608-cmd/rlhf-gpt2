"""
Day 7 — Stability Sweep Analysis & Gao et al. (2023) Reward Hacking Plot
=========================================================================
Reads results/stability_sweep/sweep_results.csv and produces:

  results/stability_sweep/reward_vs_kl.png   — reward vs KL coloured by beta
  results/stability_sweep/beta_summary.png   — mean reward ± std per beta
  results/stability_sweep/hacking_onset.txt  — text report of hacking onset

Reward hacking onset criterion (Gao et al. 2023):
  "Reward hacking occurs when the proxy reward increases while the true
   quality (or KL divergence) degrades beyond a threshold."
  We operationalise this as: collapse_detected=True OR max_kl > 15.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

SWEEP_DIR   = "results/stability_sweep"
RESULTS_CSV = os.path.join(SWEEP_DIR, "sweep_results.csv")


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    df = df[df["status"] == "ok"].copy()
    df["beta"] = df["beta"].astype(float)
    return df


def plot_reward_vs_kl(df: pd.DataFrame):
    """Scatter: final reward vs final KL, one point per (beta, seed) run."""
    fig, ax = plt.subplots(figsize=(7, 5))

    betas = sorted(df["beta"].unique())
    colours = cm.viridis(np.linspace(0.15, 0.85, len(betas)))

    for beta, col in zip(betas, colours):
        sub = df[df["beta"] == beta]
        ax.scatter(
            sub["mean_kl_final_20"],
            sub["mean_reward_after"],
            color=col, s=80, zorder=3,
            label=f"β = {beta}",
        )
        # Annotate collapsed runs
        for _, row in sub.iterrows():
            if row.get("collapse_detected", False):
                ax.annotate(
                    "⚠ hack",
                    (row["mean_kl_final_20"], row["mean_reward_after"]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, color="red",
                )

    # Hacking threshold line
    ax.axvline(x=15, color="red", linestyle="--", linewidth=1.2,
               label="KL hacking threshold (15)")

    ax.set_xlabel("Mean KL divergence (final 20 steps)", fontsize=12)
    ax.set_ylabel("Mean reward after training", fontsize=12)
    ax.set_title("Reward vs KL per β — Gao et al. (2023) reward hacking onset",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(SWEEP_DIR, "reward_vs_kl.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_beta_summary(df: pd.DataFrame):
    """Bar chart: mean reward ± std across seeds, per beta."""
    summary = df.groupby("beta").agg(
        mean_r=("mean_reward_after", "mean"),
        std_r=("mean_reward_after", "std"),
        collapse_rate=("collapse_detected", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(summary))
    bars = ax.bar(
        x, summary["mean_r"], yerr=summary["std_r"],
        capsize=6, color=["#2196F3", "#4CAF50", "#FF9800", "#F44336"],
        edgecolor="black", linewidth=0.8, alpha=0.85,
    )

    # Mark collapsed betas in red
    for i, row in summary.iterrows():
        if row["collapse_rate"] > 0:
            bars[i].set_edgecolor("red")
            bars[i].set_linewidth(2.5)
            ax.text(i, row["mean_r"] + row["std_r"] + 0.15, "⚠",
                    ha="center", fontsize=12, color="red")

    ax.set_xticks(x)
    ax.set_xticklabels([f"β={b}" for b in summary["beta"]], fontsize=11)
    ax.set_ylabel("Mean reward after training", fontsize=12)
    ax.set_title("PPO stability across KL penalty strengths (mean ± std, 3 seeds)",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(SWEEP_DIR, "beta_summary.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def report_hacking_onset(df: pd.DataFrame):
    """Print and save a text report identifying reward hacking onset."""
    lines = []
    lines.append("=" * 60)
    lines.append("REWARD HACKING ONSET — Gao et al. (2023) analysis")
    lines.append("=" * 60)
    lines.append(
        "Criterion: collapse_detected=True OR max_kl > 15\n"
        "(Gao et al. 2023: 'proxy reward rises while true quality declines')\n"
    )

    summary = df.groupby("beta").agg(
        mean_r=("mean_reward_after", "mean"),
        std_r=("mean_reward_after", "std"),
        mean_kl=("mean_kl_final_20", "mean"),
        max_kl=("max_kl", "max"),
        collapse_rate=("collapse_detected", "mean"),
        n=("seed", "count"),
    ).reset_index()

    lines.append(f"{'β':>6}  {'mean_r':>8}  {'std_r':>6}  {'mean_kl':>8}  "
                 f"{'max_kl':>7}  {'collapse%':>10}  {'verdict':}")
    lines.append("-" * 65)

    onset_beta = None
    for _, row in summary.iterrows():
        hacking = row["collapse_rate"] > 0 or row["max_kl"] > 15
        verdict = "REWARD HACKING" if hacking else "stable"
        if hacking and onset_beta is None:
            onset_beta = row["beta"]
        lines.append(
            f"{row['beta']:>6.2f}  {row['mean_r']:>8.3f}  {row['std_r']:>6.3f}  "
            f"{row['mean_kl']:>8.3f}  {row['max_kl']:>7.3f}  "
            f"{row['collapse_rate']*100:>9.0f}%  {verdict}"
        )

    lines.append("")
    if onset_beta is not None:
        lines.append(f">> Reward hacking onset detected at β = {onset_beta}")
        lines.append(f"   Lower β = weaker KL penalty = less constraint on policy.")
        lines.append(f"   This matches Gao et al. (2023) Eq.4: reward scales as")
        lines.append(f"   r̂ ∝ √KL when KL is unconstrained, then collapses.")
    else:
        lines.append(">> No reward hacking detected across all β values.")
        lines.append("   All runs remained within the stable KL regime (KL < 15).")

    report = "\n".join(lines)
    print("\n" + report)

    out = os.path.join(SWEEP_DIR, "hacking_onset.txt")
    with open(out, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved: {out}")


def main():
    if not os.path.exists(RESULTS_CSV):
        print(f"ERROR: {RESULTS_CSV} not found. Run stability_sweep.py first.")
        return

    df = load_results()
    print(f"Loaded {len(df)} successful runs from {RESULTS_CSV}\n")

    os.makedirs(SWEEP_DIR, exist_ok=True)
    plot_reward_vs_kl(df)
    plot_beta_summary(df)
    report_hacking_onset(df)
    print("\nDay 7 analysis complete.")


if __name__ == "__main__":
    main()
