"""
Central configuration — all hyperparameters and paths in one place.
Edit this file; every training script imports from here.
"""

# ── Base model ────────────────────────────────────────────────
BASE_MODEL   = "gpt2"      # 124 M parameters
MAX_LENGTH   = 128         # maximum token length across all stages

# ── Dataset ───────────────────────────────────────────────────
DATASET_NAME     = "imdb"
SFT_TRAIN_SIZE   = 5_000   # positive reviews for SFT
SFT_EVAL_SIZE    = 500
RM_TRAIN_SIZE    = 4_000   # pos+neg pairs for reward model
RM_EVAL_SIZE     = 400
PPO_QUERY_SIZE   = 1_000   # short prefixes as PPO prompts
DPO_TRAIN_SIZE   = 4_000   # preference pairs for DPO
DPO_EVAL_SIZE    = 400

PROMPT_MAX_LEN   = 64      # tokens for PPO query prefix
RESPONSE_MAX_LEN = 64      # tokens to generate during PPO rollouts

# ── SFT (Stage 1) ─────────────────────────────────────────────
SFT_LR         = 1e-5
SFT_BATCH_SIZE = 16
SFT_EPOCHS     = 2
SFT_OUTPUT_DIR = "checkpoints/sft"

# ── Reward Model (Stage 2) ────────────────────────────────────
RM_LR         = 1e-5
RM_BATCH_SIZE = 16
RM_EPOCHS     = 3
RM_OUTPUT_DIR = "checkpoints/reward_model"

# ── PPO (Stage 3 / C1 + C3) ──────────────────────────────────
PPO_LR          = 1.41e-5
PPO_BATCH_SIZE  = 16
PPO_MINI_BATCH  = 4
PPO_EPOCHS      = 4        # inner PPO epochs per rollout batch
PPO_STEPS       = 200      # total rollout steps
PPO_KL_BETA     = 0.2      # KL penalty coefficient β
PPO_OUTPUT_DIR  = "checkpoints/ppo"

# ── DPO (C2) ─────────────────────────────────────────────────
DPO_LR         = 1e-5
DPO_BATCH_SIZE = 16
DPO_EPOCHS     = 3
DPO_BETA       = 0.1       # DPO temperature β
DPO_OUTPUT_DIR = "checkpoints/dpo"

# ── Stability Sweep (C3) ──────────────────────────────────────
SWEEP_BETAS      = [0.05, 0.1, 0.2, 0.5]
SWEEP_SEEDS      = [42, 123, 456]
SWEEP_STEPS      = 200
SWEEP_OUTPUT_DIR = "results/stability_sweep"

# ── Evaluation ────────────────────────────────────────────────
EVAL_PROMPTS = [
    "This movie was",
    "The film was",
    "I thought this movie",
    "The acting in this film",
    "The plot of the movie",
    "Overall, the film",
    "The director did",
    "The story was",
    "The cinematography was",
    "This was one of the",
]

RESULTS_DIR = "results"
