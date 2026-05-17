# rlhf-gpt2

RLHF fine-tuning of GPT-2 with PPO and DPO using HuggingFace TRL.
Three contributions: reproducible pipeline, PPO vs DPO comparison, stability & alignment-tax sweep.

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

