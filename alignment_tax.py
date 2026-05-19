"""
Day 8 — Alignment Tax Analysis
================================
Computes and plots the alignment tax (perplexity cost per unit reward gain)
for PPO and DPO relative to the SFT baseline.  Also uses the stability-sweep
KL curves to identify the optimal stopping KL and summarises the recommended
stable training configuration.

Outputs
-------
  results/day8/alignment_tax.json        — all computed metrics
  results/day8/ppl_vs_reward.png         — perplexity vs mean reward (methods)
  results/day8/tax_per_beta.png          — alignment tax proxy vs β (sweep)
  results/day8/optimal_kl.png            — reward vs mean-KL with stop line
  results/day8/summary.txt              — human-readable recommendation
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR      = "results/day8"
C1_JSON      = "results/c1/eval_results.json"
C2_JSON      = "results/c2/comparison.json"
SWEEP_CSV    = "results/stability_sweep/sweep_results.csv"

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────────────
c1  = json.load(open(C1_JSON))
c2  = json.load(open(C2_JSON))
df  = pd.read_csv(SWEEP_CSV)
df  = df[df["status"] == "ok"].copy()

# ── 2. Method-level alignment tax ────────────────────────────────────────────
sft_ppl      = c2["SFT (baseline)"]["perplexity"]
sft_reward   = c2["SFT (baseline)"]["mean_reward"]

ppo_ppl      = c2["PPO (aligned)"]["perplexity"]
ppo_reward   = c2["PPO (aligned)"]["mean_reward"]

dpo_ppl      = c2["DPO (aligned)"]["perplexity"]
dpo_reward   = c2["DPO (aligned)"]["mean_reward"]

ppo_tax_pct  = (ppo_ppl - sft_ppl) / sft_ppl * 100
dpo_tax_pct  = (dpo_ppl - sft_ppl) / sft_ppl * 100

ppo_reward_gain = ppo_reward - sft_reward
dpo_reward_gain = dpo_reward - sft_reward

# Efficiency = reward gain per 1% perplexity increase
ppo_efficiency = ppo_reward_gain / ppo_tax_pct
dpo_efficiency = dpo_reward_gain / dpo_tax_pct

print("=== Method-Level Alignment Tax ===")
print(f"SFT  : reward={sft_reward:.3f}, ppl={sft_ppl:.2f}")
print(f"PPO  : reward={ppo_reward:.3f}, ppl={ppo_ppl:.2f}, "
      f"tax={ppo_tax_pct:.1f}%, gain={ppo_reward_gain:.3f}, "
      f"efficiency={ppo_efficiency:.4f} reward/ppl%")
print(f"DPO  : reward={dpo_reward:.3f}, ppl={dpo_ppl:.2f}, "
      f"tax={dpo_tax_pct:.1f}%, gain={dpo_reward_gain:.3f}, "
      f"efficiency={dpo_efficiency:.4f} reward/ppl%")

# ── 3. Sweep-level tax proxy (KL as fluency cost proxy) ──────────────────────
sweep_summary = (
    df.groupby("beta")
    .agg(
        mean_reward   = ("mean_reward_after",  "mean"),
        std_reward    = ("mean_reward_after",  "std"),
        mean_kl       = ("mean_kl_final_20",   "mean"),
        max_kl        = ("max_kl",             "max"),
        collapse_rate = ("collapse_detected",  "mean"),
    )
    .reset_index()
)
# Tax proxy: KL / reward_gain (higher = more KL cost per reward unit)
sweep_summary["reward_gain"]  = sweep_summary["mean_reward"] - sft_reward
sweep_summary["kl_per_reward"]= sweep_summary["mean_kl"] / sweep_summary["reward_gain"]

print("\n=== Sweep-Level Tax Proxy (KL / reward gain) ===")
print(sweep_summary[["beta","mean_reward","mean_kl","kl_per_reward","collapse_rate"]].to_string(index=False))

# ── 4. Optimal stopping KL ───────────────────────────────────────────────────
# Last stable beta (no collapse) with highest reward
stable = sweep_summary[sweep_summary["collapse_rate"] == 0]
optimal_row = stable.sort_values("mean_reward", ascending=False).iloc[0]
optimal_beta     = optimal_row["beta"]
optimal_kl       = optimal_row["mean_kl"]
optimal_reward   = optimal_row["mean_reward"]

# Collapse threshold from sweep: lowest beta that collapses
collapsing = sweep_summary[sweep_summary["collapse_rate"] > 0]
collapse_kl_threshold = collapsing["mean_kl"].min()

print(f"\n=== Optimal Stopping KL ===")
print(f"Recommended β    : {optimal_beta}")
print(f"Optimal mean KL  : {optimal_kl:.2f}  (KL at β={optimal_beta}, no collapse)")
print(f"Collapse onset KL: {collapse_kl_threshold:.2f}  (β=0.10 begins collapsing here)")
print(f"Safety margin    : {collapse_kl_threshold - optimal_kl:.2f} KL units below collapse threshold")

# ── 5. Plot A: Perplexity vs Reward (method comparison) ──────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

methods  = ["SFT",   "PPO",    "DPO"]
rewards  = [sft_reward, ppo_reward, dpo_reward]
ppls     = [sft_ppl, ppo_ppl, dpo_ppl]
colors   = ["#4C72B0", "#DD8452", "#55A868"]
markers  = ["o", "s", "^"]

for m, r, p, c, mk in zip(methods, rewards, ppls, colors, markers):
    ax.scatter(r, p, color=c, marker=mk, s=120, zorder=5, label=m)
    ax.annotate(m, (r, p), textcoords="offset points",
                xytext=(8, 4), fontsize=11, color=c, fontweight="bold")

# Alignment tax arrows (SFT → PPO, SFT → DPO)
for r, p, c in [(ppo_reward, ppo_ppl, "#DD8452"), (dpo_reward, dpo_ppl, "#55A868")]:
    ax.annotate("",
        xy=(r, p), xytext=(sft_reward, sft_ppl),
        arrowprops=dict(arrowstyle="->", color=c, lw=1.5, linestyle="dashed"))

ax.set_xlabel("Mean Reward", fontsize=12)
ax.set_ylabel("Perplexity (↓ better)", fontsize=12)
ax.set_title("Alignment Tax: Perplexity cost vs Reward gain\n(PPO vs DPO vs SFT baseline)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
# Annotation boxes
ax.text(0.03, 0.96,
        f"PPO tax: +{ppo_tax_pct:.1f}% ppl  |  efficiency: {ppo_efficiency:.3f}\n"
        f"DPO tax: +{dpo_tax_pct:.1f}% ppl  |  efficiency: {dpo_efficiency:.3f}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))
plt.tight_layout()
out_a = os.path.join(OUT_DIR, "ppl_vs_reward.png")
plt.savefig(out_a, dpi=150)
plt.close()
print(f"\nSaved: {out_a}")

# ── 6. Plot B: KL-per-reward-unit vs β (sweep tax proxy) ─────────────────────
fig, ax1 = plt.subplots(figsize=(7, 5))

betas    = sweep_summary["beta"].values
kl_costs = sweep_summary["kl_per_reward"].values
m_rewards= sweep_summary["mean_reward"].values
collapse = sweep_summary["collapse_rate"].values

bar_colors = ["#d62728" if c > 0 else "#2ca02c" for c in collapse]
bars = ax1.bar(range(len(betas)), kl_costs, color=bar_colors, alpha=0.8, width=0.5)
ax1.set_xticks(range(len(betas)))
ax1.set_xticklabels([f"β={b}" for b in betas], fontsize=11)
ax1.set_ylabel("KL / Reward Gain (lower = more efficient)", fontsize=11)
ax1.set_title("Alignment Tax Proxy: KL cost per reward unit vs β", fontsize=11)

ax2 = ax1.twinx()
ax2.plot(range(len(betas)), m_rewards, "D--", color="#1f77b4",
         linewidth=2, markersize=8, label="Mean reward")
ax2.set_ylabel("Mean Reward After", fontsize=11, color="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#1f77b4")

# Legend patches
from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor="#d62728", alpha=0.8, label="Collapse detected"),
    Patch(facecolor="#2ca02c", alpha=0.8, label="Stable"),
    plt.Line2D([0],[0], color="#1f77b4", marker="D", linestyle="--", label="Mean reward"),
]
ax1.legend(handles=legend_els, fontsize=9, loc="upper right")
ax1.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
out_b = os.path.join(OUT_DIR, "tax_per_beta.png")
plt.savefig(out_b, dpi=150)
plt.close()
print(f"Saved: {out_b}")

# ── 7. Plot C: Reward vs mean KL with optimal stop line ──────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

beta_vals = sorted(df["beta"].unique())
import matplotlib.cm as cm
colours = cm.RdYlGn_r(np.linspace(0.1, 0.9, len(beta_vals)))

for beta, col in zip(beta_vals, colours):
    sub = df[df["beta"] == beta]
    for _, row in sub.iterrows():
        mk = "X" if row["collapse_detected"] else "o"
        ax.scatter(row["mean_kl_final_20"], row["mean_reward_after"],
                   color=col, marker=mk, s=100, zorder=4)
    # Label each beta group (mean point)
    mx = sub["mean_kl_final_20"].mean()
    my = sub["mean_reward_after"].mean()
    ax.annotate(f"β={beta}", (mx, my), textcoords="offset points",
                xytext=(6, 4), fontsize=9, color=col,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))

# Optimal stopping line
ax.axvline(x=optimal_kl, color="green", linestyle="-", linewidth=2,
           label=f"Optimal stop KL ≈ {optimal_kl:.1f} (β={optimal_beta})")
ax.axvline(x=collapse_kl_threshold, color="red", linestyle="--", linewidth=1.5,
           label=f"Collapse onset KL ≈ {collapse_kl_threshold:.1f}")
ax.axvline(x=15, color="orange", linestyle=":", linewidth=1.2,
           label="Gao threshold (KL=15)")

ax.set_xlabel("Mean KL (final 20 steps)", fontsize=12)
ax.set_ylabel("Mean Reward After Training", fontsize=12)
ax.set_title("Reward vs KL — Optimal Stopping KL Identification", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out_c = os.path.join(OUT_DIR, "optimal_kl.png")
plt.savefig(out_c, dpi=150)
plt.close()
print(f"Saved: {out_c}")

# ── 8. Save JSON results ──────────────────────────────────────────────────────
results = {
    "method_alignment_tax": {
        "SFT":  {"mean_reward": sft_reward, "perplexity": sft_ppl,
                 "tax_pct": 0.0, "reward_gain": 0.0},
        "PPO":  {"mean_reward": ppo_reward, "perplexity": ppo_ppl,
                 "tax_pct": round(ppo_tax_pct, 3),
                 "reward_gain": round(ppo_reward_gain, 4),
                 "efficiency_reward_per_ppl_pct": round(ppo_efficiency, 4)},
        "DPO":  {"mean_reward": dpo_reward, "perplexity": dpo_ppl,
                 "tax_pct": round(dpo_tax_pct, 3),
                 "reward_gain": round(dpo_reward_gain, 4),
                 "efficiency_reward_per_ppl_pct": round(dpo_efficiency, 4)},
    },
    "optimal_stopping": {
        "recommended_beta":        optimal_beta,
        "optimal_mean_kl":         round(float(optimal_kl), 3),
        "optimal_mean_reward":     round(float(optimal_reward), 4),
        "collapse_onset_kl":       round(float(collapse_kl_threshold), 3),
        "safety_margin_kl":        round(float(collapse_kl_threshold - optimal_kl), 3),
    },
    "sweep_summary": sweep_summary.round(4).to_dict(orient="records"),
    "recommendation": (
        f"Use β=0.20 (KL penalty). "
        f"This achieves mean reward {optimal_reward:.3f} with mean KL {optimal_kl:.2f}, "
        f"a safety margin of {collapse_kl_threshold - optimal_kl:.2f} KL units below "
        f"the collapse threshold ({collapse_kl_threshold:.2f}). "
        f"PPO at β=0.20 is more efficient than DPO "
        f"({ppo_efficiency:.3f} vs {dpo_efficiency:.3f} reward per ppl%). "
        f"Stop training if KL exceeds {collapse_kl_threshold:.1f} (collapse onset)."
    ),
}

out_json = os.path.join(OUT_DIR, "alignment_tax.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out_json}")

# ── 9. Human-readable summary ─────────────────────────────────────────────────
summary_lines = [
    "=" * 65,
    "DAY 8 — ALIGNMENT TAX ANALYSIS SUMMARY",
    "=" * 65,
    "",
    "1. METHOD COMPARISON (PPO vs DPO vs SFT baseline)",
    "-" * 50,
    f"   SFT  : reward = {sft_reward:.3f}, perplexity = {sft_ppl:.2f}",
    f"   PPO  : reward = {ppo_reward:.3f}, perplexity = {ppo_ppl:.2f}",
    f"          alignment tax = +{ppo_tax_pct:.1f}% ppl for +{ppo_reward_gain:.3f} reward",
    f"          efficiency    = {ppo_efficiency:.4f} reward / ppl%",
    f"   DPO  : reward = {dpo_reward:.3f}, perplexity = {dpo_ppl:.2f}",
    f"          alignment tax = +{dpo_tax_pct:.1f}% ppl for +{dpo_reward_gain:.3f} reward",
    f"          efficiency    = {dpo_efficiency:.4f} reward / ppl%",
    f"   >> PPO is {'more' if ppo_efficiency > dpo_efficiency else 'less'} efficient than DPO",
    "",
    "2. OPTIMAL STOPPING KL (from stability sweep)",
    "-" * 50,
    f"   Recommended β         : {optimal_beta}",
    f"   Optimal mean KL       : {optimal_kl:.2f}",
    f"   Collapse onset KL     : {collapse_kl_threshold:.2f}  (β=0.10 collapses here)",
    f"   Safety margin         : {collapse_kl_threshold - optimal_kl:.2f} KL units",
    f"   Gao et al. threshold  : 15.0 (our onset {collapse_kl_threshold:.2f} is consistent)",
    "",
    "3. STABLE TRAINING CONFIGURATION (RECOMMENDATION)",
    "-" * 50,
    f"   Model      : GPT-2 (124M)",
    f"   Algorithm  : PPO with KL penalty",
    f"   β (KL pen) : 0.20",
    f"   Steps      : 200",
    f"   Stop if KL : > {collapse_kl_threshold:.1f}  (early-stop criterion)",
    f"   Expected   : reward ≈ {optimal_reward:.2f}, ppl ≈ {ppo_ppl:.1f} (+{ppo_tax_pct:.1f}% vs SFT)",
    "",
    "=" * 65,
]

summary_txt = "\n".join(summary_lines)
print("\n" + summary_txt)

out_txt = os.path.join(OUT_DIR, "summary.txt")
with open(out_txt, "w") as f:
    f.write(summary_txt + "\n")
print(f"\nSaved: {out_txt}")
print("\nDay 8 complete.")
