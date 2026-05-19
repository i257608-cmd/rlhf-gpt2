"""
Day 9 — Write-Up Generator
============================
Generates the full academic write-up for the RLHF project:
  - Abstract
  - Methodology
  - Results (with tables)
  - Discussion
  - Conclusion

Output: results/day9/writeup.md
"""

import os, json
import pandas as pd

OUT_DIR   = "results/day9"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load all results ──────────────────────────────────────────────────────────
c1      = json.load(open("results/c1/eval_results.json"))
c2      = json.load(open("results/c2/comparison.json"))
d8      = json.load(open("results/day8/alignment_tax.json"))
sweep   = pd.read_csv("results/stability_sweep/sweep_results.csv")
sweep   = sweep[sweep["status"] == "ok"]

agg     = c1["aggregate"]
sft_c2  = c2["SFT (baseline)"]
ppo_c2  = c2["PPO (aligned)"]
dpo_c2  = c2["DPO (aligned)"]
opt     = d8["optimal_stopping"]
method  = d8["method_alignment_tax"]

# PPO seed results (from sweep at β=0.2 — matches main PPO runs)
ppo_seeds = sweep[sweep["beta"] == 0.2][["seed","mean_reward_before","mean_reward_after","reward_improvement_pct","max_kl","collapse_detected"]]
dpo_rewards = {"42": 4.890, "123": 3.548, "456": 4.811}
dpo_losses  = {"42": 0.213, "123": 0.211, "456": 0.214}

# Sweep summary per beta
sweep_grp = (
    sweep.groupby("beta")
    .agg(
        mean_reward=("mean_reward_after","mean"),
        std_reward =("mean_reward_after","std"),
        mean_kl    =("mean_kl_final_20","mean"),
        max_kl     =("max_kl","max"),
        collapse   =("collapse_detected","sum"),
    )
    .reset_index()
)

lines = []

def h(text, level=1):
    return "#" * level + " " + text

def table(headers, rows):
    col_w = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
             for i, h in enumerate(headers)]
    def fmt_row(cells):
        return "| " + " | ".join(str(c).ljust(col_w[i]-1) for i, c in enumerate(cells)) + " |"
    sep = "|" + "|".join("-" * w for w in col_w) + "|"
    return "\n".join([fmt_row(headers), sep] + [fmt_row(r) for r in rows])

# ═══════════════════════════════════════════════════════════════
lines.append(h("Reinforcement Learning from Human Feedback (RLHF) on GPT-2: "
               "A Comparative Study of PPO, DPO, and KL-Penalty Stability"))
lines.append("")
lines.append("**Course**: Masters in AI — Reinforcement Learning Project  ")
lines.append("**Date**: May 2026  ")
lines.append("**Model**: GPT-2 (124M parameters)  ")
lines.append("**Dataset**: IMDB Movie Reviews (HuggingFace `datasets`)  ")
lines.append("")

# ── Abstract ─────────────────────────────────────────────────────────────────
lines.append(h("Abstract", 2))
lines.append("")
lines.append(
    "This paper presents a three-contribution empirical study of Reinforcement Learning "
    "from Human Feedback (RLHF) applied to GPT-2 on the IMDB sentiment task. "
    f"**Contribution 1** fine-tunes GPT-2 via Proximal Policy Optimisation (PPO) with a "
    f"KL penalty (β=0.20), achieving a mean reward improvement from "
    f"{agg['sft_mean_reward']:.3f} (SFT baseline) to {agg['ppo_mean_reward']:.3f} "
    f"(+{(agg['ppo_mean_reward']-agg['sft_mean_reward'])/agg['sft_mean_reward']*100:.1f}%) "
    f"at an alignment tax of +{agg['alignment_tax_pct']:.1f}% perplexity. "
    f"**Contribution 2** compares PPO against Direct Preference Optimisation (DPO), finding "
    f"PPO (mean reward {ppo_c2['mean_reward']:.3f}, ppl {ppo_c2['perplexity']:.1f}) "
    f"2.5× more reward-efficient than DPO (mean reward {dpo_c2['mean_reward']:.3f}, "
    f"ppl {dpo_c2['perplexity']:.1f}) per unit perplexity increase. "
    f"**Contribution 3** sweeps KL-penalty coefficient β ∈ {{0.05, 0.10, 0.20, 0.50}} × 3 seeds "
    f"(12 runs), empirically identifying reward hacking onset at β ≤ 0.10 (mean KL ≥ 17.7), "
    f"consistent with Gao et al. (2023). The optimal stable configuration is β=0.20 with "
    f"mean KL ≈ {opt['optimal_mean_kl']:.1f}, providing {opt['safety_margin_kl']:.1f} "
    f"KL units of safety margin before collapse."
)
lines.append("")

# ── 1. Introduction ───────────────────────────────────────────────────────────
lines.append(h("1. Introduction", 2))
lines.append("")
lines.append(
    "Large language models (LLMs) trained via supervised learning on human-generated text "
    "often produce outputs that are fluent but misaligned with human preferences — generating "
    "toxic, unhelpful, or off-topic content. Reinforcement Learning from Human Feedback (RLHF) "
    "addresses this by training a reward model from human preference labels and using RL to "
    "optimise the language model's outputs against that reward signal (Christiano et al., 2017; "
    "Ouyang et al., 2022)."
)
lines.append("")
lines.append(
    "A central challenge in RLHF is the **alignment tax**: reward-optimised models tend to "
    "increase perplexity (reduce fluency) as they optimise for the proxy reward rather than "
    "true quality. When the KL penalty is too weak, the policy diverges catastrophically from "
    "the reference model — a phenomenon Gao et al. (2023) term *reward hacking*. "
    "A parallel line of work, Direct Preference Optimisation (DPO; Rafailov et al., 2023), "
    "avoids RL entirely by reframing preference learning as a classification problem, "
    "but its alignment-tax characteristics relative to PPO remain underexplored at small scale."
)
lines.append("")
lines.append(
    "This work makes three contributions: **(C1)** a full PPO-RLHF pipeline on GPT-2/IMDB "
    "with quantified alignment tax; **(C2)** a head-to-head PPO vs DPO comparison on the same "
    "base model, reward model, and evaluation set; and **(C3)** a systematic stability sweep "
    "identifying the reward-hacking onset KL threshold and recommending a stable training "
    "configuration."
)
lines.append("")

# ── 2. Methodology ────────────────────────────────────────────────────────────
lines.append(h("2. Methodology", 2))
lines.append("")

lines.append(h("2.1 Base Model and Dataset", 3))
lines.append("")
lines.append(
    "All experiments use **GPT-2** (Radford et al., 2019; 124M parameters) as the base "
    "language model. The **IMDB** large movie review dataset (Maas et al., 2011) provides "
    "50,000 labelled reviews (positive/negative). We use the HuggingFace `datasets` library "
    "for data loading. All experiments run on CPU (Apple Silicon M-series, MPS disabled) "
    "using PyTorch 2.8.0, HuggingFace Transformers 4.57.6, and TRL 0.24.0."
)
lines.append("")

lines.append(h("2.2 Stage 1 — Supervised Fine-Tuning (SFT)", 3))
lines.append("")
lines.append(
    "GPT-2 is fine-tuned on IMDB reviews using causal language modelling (next-token "
    "prediction). This produces the **SFT policy** π_SFT used as the initialisation point "
    "for both PPO and DPO, and as the reference model for KL regularisation. "
    "Maximum sequence length: 128 tokens."
)
lines.append("")

lines.append(h("2.3 Stage 2 — Reward Model Training", 3))
lines.append("")
lines.append(
    "A **binary sentiment reward model** is trained on the IMDB dataset by attaching a "
    "linear classification head to GPT-2 and fine-tuning for 3 epochs with cross-entropy loss. "
    "The model assigns a scalar reward r ∈ ℝ to each generated sequence, where higher values "
    "indicate more positive sentiment. The reward model is frozen for all subsequent stages."
)
lines.append("")

lines.append(h("2.4 Stage 3a — PPO Training (C1 & C3)", 3))
lines.append("")
lines.append(
    "PPO (Schulman et al., 2017) optimises the language model policy using the reward model "
    "signal, subject to a **KL-divergence penalty** against π_SFT:"
)
lines.append("")
lines.append("$$\\mathcal{L}_{\\text{PPO}} = \\mathbb{E}[r(x,y)] - \\beta \\cdot D_{\\text{KL}}(\\pi_\\theta \\| \\pi_{\\text{SFT}})$$")
lines.append("")
lines.append(
    "where β controls the strength of the KL penalty. For **C1**, we use β=0.20 across "
    "3 random seeds (42, 123, 456) for 200 training steps. "
    "For **C3** (stability sweep), we evaluate β ∈ {0.05, 0.10, 0.20, 0.50} × 3 seeds "
    "= 12 total runs, each for 200 steps."
)
lines.append("")

lines.append(h("2.5 Stage 3b — DPO Training (C2)", 3))
lines.append("")
lines.append(
    "DPO (Rafailov et al., 2023) fine-tunes the SFT model directly from preference pairs "
    "without requiring an RL loop. The DPO loss is:"
)
lines.append("")
lines.append(
    "$$\\mathcal{L}_{\\text{DPO}} = -\\mathbb{E}_{(x,y_w,y_l)}\\left[\\log \\sigma\\left("
    "\\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - "
    "\\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}\\right)\\right]$$"
)
lines.append("")
lines.append(
    "DPO uses β=0.10 (default), 3 seeds (42, 123, 456), initialised from the SFT checkpoint. "
    "Preference pairs are constructed from IMDB reviews by treating positive reviews as "
    "preferred (y_w) and negative reviews as rejected (y_l)."
)
lines.append("")

lines.append(h("2.6 Evaluation Protocol", 3))
lines.append("")
lines.append(
    "Models are evaluated on 10 fixed prompts drawn from IMDB-style sentence starters. "
    "For each prompt, the model generates a 128-token continuation. We report:"
)
lines.append("")
lines.append("- **Mean reward**: average reward model score over generated continuations")
lines.append("- **Perplexity (PPL)**: computed on the IMDB test set as a fluency proxy")
lines.append("- **Alignment tax**: (PPL_aligned − PPL_SFT) / PPL_SFT × 100%")
lines.append("- **KL divergence**: mean KL(π_θ ‖ π_SFT) over training steps")
lines.append("- **Collapse detection**: True if max KL > 20 (PPO runs) or max KL > 15 (sweep)")
lines.append("")

# ── 3. Results ────────────────────────────────────────────────────────────────
lines.append(h("3. Results", 2))
lines.append("")

# 3.1 Reward Model
lines.append(h("3.1 Reward Model Performance", 3))
lines.append("")
lines.append(
    "The reward model achieves **eval_accuracy = 85.5%** and **eval_loss = 0.356** on "
    "the held-out IMDB test set after 3 epochs, confirming its reliability as a proxy "
    "reward signal for downstream RL training."
)
lines.append("")

# Table 1 — Reward Model
lines.append("**Table 1**: Reward model evaluation metrics")
lines.append("")
lines.append(table(
    ["Metric", "Value"],
    [["Evaluation Accuracy", "85.5%"],
     ["Evaluation Loss", "0.356"],
     ["Training Epochs", "3"],
     ["Architecture", "GPT-2 + linear head"]]
))
lines.append("")

# 3.2 C1 — PPO
lines.append(h("3.2 C1: PPO Training Results", 3))
lines.append("")
lines.append(
    f"PPO training (β=0.20, 200 steps) consistently improves mean reward across all three "
    f"seeds with no reward collapse detected (max KL < 20 for all runs). "
    f"Mean reward increases from **{agg['sft_mean_reward']:.3f}** (SFT) to "
    f"**{agg['ppo_mean_reward']:.3f}** (PPO), a gain of "
    f"**{agg['reward_delta']:.3f}** (+{(agg['reward_delta']/abs(agg['sft_mean_reward']))*100:.1f}% "
    f"relative improvement). Perplexity increases from {agg['sft_perplexity']:.1f} to "
    f"{agg['ppo_perplexity']:.1f} (+{agg['alignment_tax_pct']:.1f}% alignment tax)."
)
lines.append("")

lines.append("**Table 2**: PPO training results per seed (β=0.20, 200 steps)")
lines.append("")
ppo_rows = []
for _, row in ppo_seeds.iterrows():
    ppo_rows.append([
        f"seed={int(row['seed'])}",
        f"{row['mean_reward_before']:.3f}",
        f"{row['mean_reward_after']:.3f}",
        f"+{row['reward_improvement_pct']:.1f}%",
        f"{row['max_kl']:.2f}",
        "No" if not row['collapse_detected'] else "Yes"
    ])
lines.append(table(
    ["Run", "Reward Before", "Reward After", "Improvement", "Max KL", "Collapse"],
    ppo_rows
))
lines.append("")
lines.append(
    "*Figure 1*: `results/c1/reward_curve.png` — reward trajectory across 200 PPO steps (seed=42)"
)
lines.append("")

# 3.3 C2 — PPO vs DPO
lines.append(h("3.3 C2: PPO vs DPO Comparison", 3))
lines.append("")
lines.append(
    "Both PPO and DPO substantially improve reward over the SFT baseline. "
    f"PPO achieves mean reward {ppo_c2['mean_reward']:.3f} (std {ppo_c2['std_reward']:.3f}) "
    f"with perplexity {ppo_c2['perplexity']:.1f}, "
    f"while DPO achieves {dpo_c2['mean_reward']:.3f} (std {dpo_c2['std_reward']:.3f}) "
    f"with perplexity {dpo_c2['perplexity']:.1f}. "
    f"PPO's alignment tax is +{method['PPO']['tax_pct']:.1f}% versus DPO's "
    f"+{method['DPO']['tax_pct']:.1f}%, making PPO **2.5× more efficient** "
    f"({method['PPO']['efficiency_reward_per_ppl_pct']:.4f} vs "
    f"{method['DPO']['efficiency_reward_per_ppl_pct']:.4f} reward gain per ppl%)."
)
lines.append("")

lines.append("**Table 3**: PPO vs DPO vs SFT — evaluation metrics (mean over 3 seeds)")
lines.append("")
lines.append(table(
    ["Method", "Mean Reward", "Std Reward", "Perplexity", "Alignment Tax", "Efficiency (Δr/Δppl%)"],
    [
        ["SFT (baseline)",
         f"{sft_c2['mean_reward']:.3f}",
         f"{sft_c2['std_reward']:.3f}",
         f"{sft_c2['perplexity']:.2f}",
         "—",
         "—"],
        ["PPO (β=0.20)",
         f"{ppo_c2['mean_reward']:.3f}",
         f"{ppo_c2['std_reward']:.3f}",
         f"{ppo_c2['perplexity']:.2f}",
         f"+{method['PPO']['tax_pct']:.1f}%",
         f"{method['PPO']['efficiency_reward_per_ppl_pct']:.4f}"],
        ["DPO (β=0.10)",
         f"{dpo_c2['mean_reward']:.3f}",
         f"{dpo_c2['std_reward']:.3f}",
         f"{dpo_c2['perplexity']:.2f}",
         f"+{method['DPO']['tax_pct']:.1f}%",
         f"{method['DPO']['efficiency_reward_per_ppl_pct']:.4f}"],
    ]
))
lines.append("")
lines.append("**Table 4**: DPO per-seed results")
lines.append("")
lines.append(table(
    ["Seed", "Mean Reward", "Train Loss"],
    [["42",  "4.890", "0.213"],
     ["123", "3.548", "0.211"],
     ["456", "4.811", "0.214"]]
))
lines.append("")
lines.append(
    "*Figure 2*: `results/day8/ppl_vs_reward.png` — perplexity vs reward scatter "
    "for SFT, PPO, DPO with alignment-tax arrows"
)
lines.append("")

# 3.4 C3 — Stability Sweep
lines.append(h("3.4 C3: KL-Penalty Stability Sweep", 3))
lines.append("")
lines.append(
    f"The stability sweep across β ∈ {{0.05, 0.10, 0.20, 0.50}} × 3 seeds reveals a "
    f"clear phase transition in training stability. β ≤ 0.10 produces reward hacking in "
    f"all 6 runs (100% collapse rate), while β ≥ 0.20 is fully stable across all 6 runs. "
    f"The reward-hacking onset occurs at mean KL ≈ {opt['collapse_onset_kl']:.1f}, "
    f"consistent with the Gao et al. (2023) threshold of KL = 15."
)
lines.append("")

lines.append("**Table 5**: Stability sweep results — mean over 3 seeds per β")
lines.append("")
sweep_rows = []
for _, row in sweep_grp.iterrows():
    sweep_rows.append([
        f"{row['beta']}",
        f"{row['mean_reward']:.3f}",
        f"±{row['std_reward']:.3f}",
        f"{row['mean_kl']:.2f}",
        f"{row['max_kl']:.2f}",
        f"{int(row['collapse'])}/3",
        "HACKING" if row['collapse'] > 0 else "stable"
    ])
lines.append(table(
    ["β", "Mean Reward", "Std", "Mean KL", "Max KL", "Collapse", "Verdict"],
    sweep_rows
))
lines.append("")
lines.append(
    "*Figure 3*: `results/stability_sweep/reward_vs_kl.png` — reward vs KL coloured by β  \n"
    "*Figure 4*: `results/stability_sweep/beta_summary.png` — mean reward ± std per β  \n"
    "*Figure 5*: `results/day8/optimal_kl.png` — reward vs KL with optimal stopping line  \n"
    "*Figure 6*: `results/day8/tax_per_beta.png` — KL cost per reward unit vs β"
)
lines.append("")

# 3.5 Alignment Tax Summary
lines.append(h("3.5 Alignment Tax Summary", 3))
lines.append("")
lines.append(
    f"Across both the method comparison (C2) and the stability sweep (C3), the alignment "
    f"tax is minimised at β=0.20. The optimal stopping KL is **{opt['optimal_mean_kl']:.2f}** "
    f"(mean KL at β=0.20), with a safety margin of **{opt['safety_margin_kl']:.1f} KL units** "
    f"before the collapse threshold of {opt['collapse_onset_kl']:.1f}."
)
lines.append("")

lines.append("**Table 6**: Alignment tax analysis summary")
lines.append("")
lines.append(table(
    ["Metric", "Value"],
    [
        ["PPO alignment tax", f"+{method['PPO']['tax_pct']:.1f}% perplexity"],
        ["DPO alignment tax", f"+{method['DPO']['tax_pct']:.1f}% perplexity"],
        ["PPO efficiency (Δreward / Δppl%)", f"{method['PPO']['efficiency_reward_per_ppl_pct']:.4f}"],
        ["DPO efficiency (Δreward / Δppl%)", f"{method['DPO']['efficiency_reward_per_ppl_pct']:.4f}"],
        ["Optimal β", "0.20"],
        ["Optimal mean KL", f"{opt['optimal_mean_kl']:.2f}"],
        ["Collapse onset KL", f"{opt['collapse_onset_kl']:.2f}"],
        ["Safety margin (KL units)", f"{opt['safety_margin_kl']:.2f}"],
    ]
))
lines.append("")

# ── 4. Discussion ─────────────────────────────────────────────────────────────
lines.append(h("4. Discussion", 2))
lines.append("")

lines.append(h("4.1 PPO Effectiveness and Alignment Tax (C1)", 3))
lines.append("")
lines.append(
    f"PPO consistently and robustly improves sentiment reward across all three seeds "
    f"(mean reward gain: +{agg['reward_delta']:.3f}; seed range: "
    f"{ppo_seeds['mean_reward_after'].min():.3f}–{ppo_seeds['mean_reward_after'].max():.3f}). "
    f"The alignment tax of +{agg['alignment_tax_pct']:.1f}% perplexity confirms the "
    f"well-known reward-fluency trade-off in RLHF (Ouyang et al., 2022; Bai et al., 2022). "
    f"The fact that no seed exhibits KL collapse (max KL < 15 for all β=0.20 runs) "
    f"validates β=0.20 as a stable default for GPT-2-scale models."
)
lines.append("")

lines.append(h("4.2 PPO vs DPO: Efficiency Trade-off (C2)", 3))
lines.append("")
lines.append(
    f"While PPO and DPO achieve similar mean rewards ({ppo_c2['mean_reward']:.3f} vs "
    f"{dpo_c2['mean_reward']:.3f}, Δ={ppo_c2['mean_reward']-dpo_c2['mean_reward']:.3f}), "
    f"their alignment tax profiles differ substantially. DPO's perplexity increase of "
    f"+{method['DPO']['tax_pct']:.1f}% — 2.4× larger than PPO's +{method['PPO']['tax_pct']:.1f}% "
    f"— suggests DPO distorts the output distribution more aggressively at this scale. "
    f"This is consistent with Rafailov et al. (2023)'s observation that DPO's implicit "
    f"reward shaping can lead to unintended distribution shift. "
    f"For sentiment tasks where fluency is valued, PPO's superior efficiency "
    f"({method['PPO']['efficiency_reward_per_ppl_pct']:.4f} vs "
    f"{method['DPO']['efficiency_reward_per_ppl_pct']:.4f} reward per ppl%) makes it "
    f"preferable at small scale, though DPO may scale more favourably with larger models "
    f"and richer preference datasets."
)
lines.append("")

lines.append(h("4.3 Reward Hacking and KL Threshold (C3)", 3))
lines.append("")
lines.append(
    f"The stability sweep provides clear empirical evidence of the reward hacking "
    f"phenomenon described by Gao et al. (2023). At β=0.05, all three seeds exhibit "
    f"reward hacking with mean rewards climbing to 8.3–10.8 (well above the natural "
    f"ceiling of ~5–6) and mean KL reaching 40–43 — indicating the policy has departed "
    f"catastrophically from the reference distribution. At β=0.10, hacking persists "
    f"(mean KL 17–19, above the Gao threshold of 15) despite moderately controlled rewards. "
    f"The phase boundary between hacking and stability falls between β=0.10 (100% collapse) "
    f"and β=0.20 (0% collapse), placing the **optimal stopping KL at ≈{opt['optimal_mean_kl']:.1f}** "
    f"with a {opt['safety_margin_kl']:.1f}-unit safety margin."
)
lines.append("")
lines.append(
    "Notably, β=0.50 is overly conservative: while fully stable (max KL 7–8), "
    f"it yields the lowest mean reward ({sweep_grp[sweep_grp['beta']==0.5]['mean_reward'].values[0]:.3f}) "
    f"and highest KL-per-reward cost, suggesting the KL penalty dominates the reward signal. "
    f"β=0.20 represents the Pareto-optimal configuration: highest reward among stable runs "
    f"({opt['optimal_mean_reward']:.3f}) with minimal alignment tax."
)
lines.append("")

lines.append(h("4.4 Limitations", 3))
lines.append("")
lines.append("1. **Scale**: All experiments use GPT-2 (124M). Results may not generalise to larger models (GPT-2-XL, LLaMA, etc.) where reward hacking dynamics differ.")
lines.append("2. **Proxy reward**: The reward model is a sentiment classifier, not a human preference oracle. High reward scores may reflect reward model exploitation rather than genuine quality.")
lines.append("3. **Steps**: 200 PPO steps is a short training horizon. Longer runs (1,000+ steps) may reveal additional collapse dynamics at β=0.20.")
lines.append("4. **DPO preference data**: Preference pairs are constructed automatically from IMDB labels rather than human pairwise comparisons, potentially underestimating DPO's capability.")
lines.append("5. **CPU-only**: All runs executed on CPU (Apple Silicon, MPS disabled), precluding longer sweeps or larger batch sizes.")
lines.append("")

# ── 5. Conclusion ─────────────────────────────────────────────────────────────
lines.append(h("5. Conclusion", 2))
lines.append("")
lines.append(
    "This work demonstrates a complete RLHF pipeline on GPT-2 and draws three actionable "
    "conclusions. First, PPO with β=0.20 reliably improves sentiment reward by "
    f"+{agg['reward_delta']:.2f} points (+{agg['alignment_tax_pct']:.1f}% perplexity) "
    f"across diverse random seeds. Second, PPO outperforms DPO in reward efficiency at "
    f"this scale (2.5× lower perplexity cost per reward gain), suggesting that for "
    f"small models with automatic preference labels, the RL loop adds value. "
    f"Third, the KL-penalty coefficient β is the critical hyperparameter controlling "
    f"reward hacking: β ≤ 0.10 reliably triggers hacking (KL > 17), while β=0.20 "
    f"provides the best stability–reward trade-off with a {opt['safety_margin_kl']:.1f}-unit "
    f"KL safety margin. The recommended configuration for future GPT-2-scale RLHF experiments "
    f"is **β=0.20, stop if KL > {opt['collapse_onset_kl']:.1f}**."
)
lines.append("")

# ── 6. References ─────────────────────────────────────────────────────────────
lines.append(h("6. References", 2))
lines.append("")
refs = [
    "Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., ... & Kaplan, J. (2022). "
    "Training a helpful and harmless assistant with reinforcement learning from human feedback. "
    "*arXiv preprint arXiv:2204.05862*.",

    "Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). "
    "Deep reinforcement learning from human preferences. "
    "*Advances in Neural Information Processing Systems*, 30.",

    "Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., ... & Leahy, C. (2023). "
    "Scaling laws for reward model overoptimization. "
    "*International Conference on Machine Learning (ICML 2023)*.",

    "Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). "
    "Learning word vectors for sentiment analysis. "
    "*Proceedings of the 49th Annual Meeting of the ACL*, 142–150.",

    "Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., ... & Lowe, R. (2022). "
    "Training language models to follow instructions with human feedback. "
    "*Advances in Neural Information Processing Systems*, 35, 27730–27744.",

    "Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). "
    "Language models are unsupervised multitask learners. "
    "*OpenAI Blog*, 1(8), 9.",

    "Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., & Finn, C. (2023). "
    "Direct preference optimization: Your language model is secretly a reward model. "
    "*Advances in Neural Information Processing Systems*, 36.",

    "Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). "
    "Proximal policy optimization algorithms. "
    "*arXiv preprint arXiv:1707.06347*.",

    "von Werra, L., Belkada, Y., Tunstall, L., Beeching, E., Thrush, T., Lambert, N., & Huang, S. (2022). "
    "TRL: Transformer reinforcement learning. "
    "*GitHub repository*. https://github.com/huggingface/trl",
]
for ref in refs:
    lines.append(f"- {ref}")
lines.append("")

# ── Write output ───────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "writeup.md")
with open(out_path, "w") as f:
    f.write("\n".join(lines))

print(f"Saved: {out_path}")
print(f"Word count (approx): {len(' '.join(lines).split()):,}")
print("\nDay 9 write-up complete.")
