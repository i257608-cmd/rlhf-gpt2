# RLHF on GPT-2: PPO vs DPO with KL-Penalty Stability Analysis

**Masters in AI — Reinforcement Learning Project**  
**Model**: GPT-2 (124M) · **Dataset**: IMDB · **Framework**: HuggingFace TRL 0.24.0

A complete, reproducible RLHF pipeline with three empirical contributions:

| Contribution | Description | Key Result |
|---|---|---|
| **C1** | PPO-RLHF pipeline | +3.98 reward, +18.3% perplexity alignment tax |
| **C2** | PPO vs DPO comparison | PPO is 2.5× more reward-efficient than DPO |
| **C3** | KL-penalty stability sweep (12 runs) | β ≤ 0.10 causes reward hacking; β = 0.20 is optimal |

---

## Repository Structure

```
rlhf-gpt2/
├── config.py                  # Shared hyperparameters and paths
├── data_utils.py              # Tokenizer and dataset helpers
│
├── train_sft.py               # Stage 1: Supervised Fine-Tuning
├── train_reward_model.py      # Stage 2: Reward model (binary classifier)
├── train_ppo.py               # Stage 3a: PPO training (--beta, --seed, --steps)
├── train_dpo.py               # Stage 3b: DPO training (--seed)
│
├── evaluate.py                # C1 and C2 evaluation (--mode c1 / c2)
├── stability_sweep.py         # C3: β × seed grid sweep (12 runs)
├── plot_sweep.py              # Gao et al. reward-vs-KL plots
├── alignment_tax.py           # Alignment tax analysis + optimal stopping
├── writeup.py                 # Generates results/day9/writeup.md
│
├── demo.py                    # Interactive CLI demo (SFT vs PPO vs DPO)
│
├── checkpoints/
│   ├── sft/                   # SFT checkpoint
│   ├── reward_model/          # Reward model checkpoint
│   ├── ppo/                   # PPO checkpoint (seed=42)
│   ├── ppo_seed123/           # PPO checkpoint (seed=123)
│   ├── ppo_seed456/           # PPO checkpoint (seed=456)
│   ├── dpo/                   # DPO checkpoint (seed=42)
│   ├── dpo_seed123/           # DPO checkpoint (seed=123)
│   └── dpo_seed456/           # DPO checkpoint (seed=456)
│
└── results/
    ├── c1/                    # C1 eval results + reward curve plot
    ├── c2/                    # C2 comparison (SFT vs PPO vs DPO)
    ├── stability_sweep/       # Sweep CSV + Gao et al. plots
    ├── day8/                  # Alignment tax analysis outputs
    ├── day9/                  # Final write-up (writeup.md)
    └── day10/                 # Final demo output (demo_final.txt)
```

---

## Setup

**Requirements**: Python 3.9, macOS or Linux (CPU-only; MPS disabled throughout)

```bash
# Clone and enter the repo
git clone https://github.com/i257608-cmd/rlhf-gpt2.git
cd rlhf-gpt2

# Create virtual environment
python3.9 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch==2.8.0 transformers==4.57.6 trl==0.24.0 \
            datasets accelerate pandas matplotlib
```

---

## Running the Pipeline

Run stages **in order**. Each stage saves to `checkpoints/` and is a prerequisite for the next.

### Stage 1 — Supervised Fine-Tuning (SFT)
```bash
python train_sft.py
# Output: checkpoints/sft/
```

### Stage 2 — Reward Model
```bash
python train_reward_model.py
# Output: checkpoints/reward_model/
# eval_accuracy = 85.5%, eval_loss = 0.356
```

### Stage 3a — PPO Training (3 seeds)
```bash
python train_ppo.py --beta 0.2 --seed 42  --steps 200
python train_ppo.py --beta 0.2 --seed 123 --steps 200 --output_dir checkpoints/ppo_seed123
python train_ppo.py --beta 0.2 --seed 456 --steps 200 --output_dir checkpoints/ppo_seed456
# Output: checkpoints/ppo/, checkpoints/ppo_seed123/, checkpoints/ppo_seed456/
```

### Stage 3b — DPO Training (3 seeds)
```bash
python train_dpo.py --seed 42
python train_dpo.py --seed 123 --output_dir checkpoints/dpo_seed123
python train_dpo.py --seed 456 --output_dir checkpoints/dpo_seed456
# Output: checkpoints/dpo/, checkpoints/dpo_seed123/, checkpoints/dpo_seed456/
```

### Evaluation
```bash
# C1: SFT vs PPO (reward + perplexity + alignment tax)
python evaluate.py --mode c1

# C2: SFT vs PPO vs DPO comparison
python evaluate.py --mode c2
```

### Stability Sweep (C3 — ~20 hours CPU total)
```bash
python stability_sweep.py          # 12 runs: β∈{0.05,0.10,0.20,0.50} × seed∈{42,123,456}
python plot_sweep.py               # Generate Gao et al. plots
python alignment_tax.py            # Alignment tax + optimal stopping analysis
```

### Generate Write-Up
```bash
python writeup.py                  # Output: results/day9/writeup.md
```

### Interactive Demo
```bash
# Interactive mode
python demo.py

# Single prompt
python demo.py --prompt "This movie was absolutely"

# Run all preset prompts and save results
python demo.py --all-prompts       # Output: results/day10/demo_final.txt
```

---

## Key Results

### Reward Model
| Metric | Value |
|---|---|
| Evaluation Accuracy | **85.5%** |
| Evaluation Loss | 0.356 |

### C1 — PPO Training (β = 0.20, 200 steps, 3 seeds)
| Seed | Reward Before | Reward After | Improvement | Max KL | Collapse |
|---|---|---|---|---|---|
| 42  | 3.578 | 5.648 | +57.9% | 14.23 | No |
| 123 | 3.301 | 4.753 | +44.0% | 14.62 | No |
| 456 | 3.222 | 5.457 | +69.4% | 14.62 | No |

### C2 — PPO vs DPO vs SFT
| Method | Mean Reward | Perplexity | Alignment Tax | Efficiency |
|---|---|---|---|---|
| SFT (baseline) | 2.498 | 37.79 | — | — |
| PPO (β=0.20) | **4.942** | 44.69 | +18.3% | **0.1339** |
| DPO (β=0.10) | 4.816 | 54.39 | +43.9% | 0.0528 |

### C3 — KL-Penalty Stability Sweep
| β | Mean Reward | Mean KL | Collapse Rate | Verdict |
|---|---|---|---|---|
| 0.05 | 9.4 | ~40 | 3/3 | **REWARD HACKING** |
| 0.10 | 6.0 | ~18 | 3/3 | **REWARD HACKING** |
| **0.20** | **5.3** | **9.6** | **0/3** | **✓ stable (optimal)** |
| 0.50 | 4.2 | 4.5 | 0/3 | ✓ stable (conservative) |

---

## Configuration

All key hyperparameters are in `config.py`:

| Parameter | Value |
|---|---|
| Base model | `gpt2` (124M) |
| Max sequence length | 128 tokens |
| PPO KL penalty (β) | 0.20 |
| DPO β | 0.10 |
| PPO training steps | 200 |
| Sweep β values | {0.05, 0.10, 0.20, 0.50} |
| Sweep seeds | {42, 123, 456} |

---

## Report

The full academic write-up (Methodology + Results + Discussion + Appendices) is at:
```
results/day9/writeup.md
```

Figures are in `results/c1/`, `results/stability_sweep/`, and `results/day8/`.

---

## References

- Ouyang et al. (2022). *Training language models to follow instructions with human feedback.* NeurIPS.
- Gao et al. (2023). *Scaling laws for reward model overoptimization.* ICML.
- Rafailov et al. (2023). *Direct preference optimization.* NeurIPS.
- Schulman et al. (2017). *Proximal policy optimization algorithms.* arXiv:1707.06347.
- von Werra et al. (2022). *TRL: Transformer reinforcement learning.* GitHub.

---

## Project Structure

```
rlhf-gpt2/
├── config.py               # all hyperparameters (edit here)
├── data_utils.py           # dataset loaders for every stage
├── train_sft.py            # C1 Stage 1 — Supervised Fine-Tuning
├── train_reward_model.py   # C1 Stage 2 — Reward Model
├── train_ppo.py            # C1 Stage 3 — PPO   (also used by C3)
├── train_dpo.py            # C2          — DPO
├── evaluate.py             # shared evaluation (reward, perplexity)
├── stability_sweep.py      # C3          — beta × seed grid sweep
├── demo.py                 # interactive CLI demo
└── requirements.txt
```

Checkpoints and results are written here:

```
checkpoints/sft/
checkpoints/reward_model/
checkpoints/ppo/
checkpoints/dpo/
results/c1/eval_results.json
results/c2/comparison.json
results/stability_sweep/sweep_results.csv
results/stability_sweep/summary.csv
```

---

## 1 — Setup

```bash
# create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies  (~2 min)
pip3 install -r requirements.txt
```

GPU (T4 / RTX 3060 or better) is strongly recommended.
All scripts fall back to CPU but will be much slower.

---

## 2 — Contribution 1: Reproducible RLHF Pipeline (SFT → RM → PPO)

Run the three stages **in order**:

### Stage 1 — Supervised Fine-Tuning (~20–30 min GPU)

Fine-tunes GPT-2 on 5 000 positive IMDB reviews.

```bash
python train_sft.py
```

Checkpoint: `checkpoints/sft/`

---

### Stage 2 — Reward Model (~15–20 min GPU)

Trains `GPT2ForSequenceClassification` on 4 000 IMDB reviews
(positive = preferred, negative = rejected).
Reward signal used in PPO: `logit[positive] − logit[negative]`

```bash
python train_reward_model.py
```

Checkpoint: `checkpoints/reward_model/`

---

### Stage 3 — PPO Fine-Tuning (~30–60 min GPU)

Loads the SFT model as policy + reference, runs 200 PPO steps
with a fixed KL penalty β = 0.2.

```bash
python train_ppo.py
# optional overrides:
python train_ppo.py --beta 0.2 --seed 42 --steps 200
```

Checkpoint: `checkpoints/ppo/`

---

### Evaluate C1

```bash
python evaluate.py --mode c1
```

Reports:
- Qualitative generation examples (SFT vs PPO)
- Mean reward before / after alignment
- Perplexity on held-out IMDB test set
- Alignment tax % = (ppl_PPO − ppl_SFT) / ppl_SFT × 100

Output: `results/c1/eval_results.json`

---

## 3 — Contribution 2: PPO vs DPO Comparison

### Train DPO (~10–20 min GPU)

Trains the SFT model with Direct Preference Optimisation on
IMDB preference pairs (no PPO loop, no reward model needed at training time).

```bash
python train_dpo.py
# optional:
python train_dpo.py --seed 42
```

Checkpoint: `checkpoints/dpo/`

---

### Evaluate C2

```bash
python evaluate.py --mode c2
```

Prints a side-by-side table: **SFT vs PPO vs DPO**
(mean reward ± std, perplexity)

Output: `results/c2/comparison.json`

---

## 4 — Contribution 3: Stability & Alignment-Tax Sweep

Runs PPO 12 times (4 β values × 3 seeds).
Each run is independent so you can interrupt and re-run.

```bash
python stability_sweep.py
```

β values: `[0.05, 0.1, 0.2, 0.5]`
Seeds: `[42, 123, 456]`

Output:
```
results/stability_sweep/sweep_results.csv   — one row per run
results/stability_sweep/summary.csv         — aggregated by beta
```

---

## 5 — Interactive Demo

Loads all available checkpoints and lets you type prompts interactively.
Missing checkpoints are skipped gracefully.

```bash
python demo.py
# or single shot:
python demo.py --prompt "This movie was"
```

---

## Hyperparameter Reference

All defaults live in `config.py`.

| Stage | Key param | Default |
|-------|-----------|---------|
| SFT   | `SFT_LR`  | 1e-5    |
| SFT   | `SFT_EPOCHS` | 2   |
| RM    | `RM_LR`   | 1e-5    |
| PPO   | `PPO_KL_BETA` | 0.2 |
| PPO   | `PPO_STEPS`   | 200 |
| DPO   | `DPO_BETA`    | 0.1 |

---

## Notes

- GPT-2 has no pad token; the code sets `pad_token = eos_token` automatically.
- PPO uses **left-padding** for generation; the reward model uses **right-padding** for classification.
- The PPO script exposes `run_ppo()` so `stability_sweep.py` can call it directly.
- Each training run saves a `*_metrics.json` alongside the checkpoint.

