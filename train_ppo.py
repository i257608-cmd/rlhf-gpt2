"""
Contribution 1 — Stage 3: PPO Policy Optimisation
===================================================
Rewritten for TRL 0.24.0.  The new PPOTrainer uses a `trainer.train()` API
instead of a manual rollout loop.  Key differences from the old 0.9.x API:

  • PPOConfig is now a TrainingArguments subclass; use `kl_coef`, `num_ppo_epochs`,
    `num_mini_batches`, `total_episodes`, `response_length`.
  • PPOTrainer takes `model` (policy), `ref_model`, `value_model`, `reward_model`
    as separate objects and calls `trainer.train()`.
  • TRL's `get_reward()` helper accesses `model.base_model_prefix` + `model.score`,
    so the reward_model must expose the same interface as
    GPT2ForSequenceClassification — see RewardModelForPPO below.

Usage:
    python train_ppo.py
    python train_ppo.py --beta 0.2 --seed 42 --steps 200
    python train_ppo.py --beta 0.05 --seed 123 --output_dir results/run_A

Output:
    checkpoints/ppo/           (default model + tokenizer)
    <output_dir>/ppo_metrics.json
"""

import os
import json
import argparse

# ── Force CPU-only: patch MPS out before TRL/accelerate ever check it ────────
# MPS causes nan/inf in torch.multinomial (Apple Silicon numerical instability)
import torch
torch.backends.mps.is_available = lambda: False   # type: ignore[assignment]
torch.backends.mps.is_built     = lambda: False   # type: ignore[assignment]
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    GPT2ForSequenceClassification,
)
from trl import PPOTrainer, PPOConfig

from data_utils import load_ppo_prompts, get_tokenizer
from config import (
    SFT_OUTPUT_DIR, RM_OUTPUT_DIR, PPO_OUTPUT_DIR, RESULTS_DIR,
    PPO_LR, PPO_BATCH_SIZE, PPO_MINI_BATCH, PPO_EPOCHS,
    PPO_KL_BETA, PPO_STEPS,
    RESPONSE_MAX_LEN, EVAL_PROMPTS,
)


# ── Reward model wrapper ──────────────────────────────────────────────────────

class RewardModelForPPO(nn.Module):
    """
    Adapts our trained GPT2ForSequenceClassification (num_labels=2) to the
    interface that TRL 0.24 PPOTrainer.get_reward() requires.

    TRL's get_reward() does:
        lm_backbone = getattr(model, model.base_model_prefix)
        output = lm_backbone(..., output_hidden_states=True)
        reward_logits = model.score(output.hidden_states[-1])  # [B, seq, K]
        score = reward_logits[:, last_token, :].squeeze(-1)    # [B] if K==1

    We reparametrise the 2-class head into a 1-scalar head:
        score(h) = (w_pos − w_neg) · h  ≡  logit_positive − logit_negative
    This is exactly the original reward signal, just folded into one linear layer.
    The reward model weights are frozen throughout PPO.
    """

    base_model_prefix = "transformer"   # tells get_reward() which attr is the LM backbone

    def __init__(self, gpt2_seq_clf: GPT2ForSequenceClassification):
        super().__init__()
        # Expose the GPT-2 backbone under the canonical name
        self.transformer = gpt2_seq_clf.transformer
        self.config = gpt2_seq_clf.config

        # Merge the 2-class head into a single-scalar head:
        #   score(h) = (w_pos − w_neg) · h
        w_pos = gpt2_seq_clf.score.weight[1].detach()   # [hidden_size]
        w_neg = gpt2_seq_clf.score.weight[0].detach()   # [hidden_size]
        self.score = nn.Linear(gpt2_seq_clf.config.n_embd, 1, bias=False)
        with torch.no_grad():
            self.score.weight.copy_((w_pos - w_neg).unsqueeze(0))

        # Freeze — reward model must never be updated during PPO
        for p in self.parameters():
            p.requires_grad_(False)


# ── Evaluation helper ─────────────────────────────────────────────────────────

def _eval_mean_reward(policy_model, tokenizer, rm_raw, device, n: int = 5) -> float:
    """
    Generate n completions from EVAL_PROMPTS and return their mean reward.
    Uses the raw GPT2ForSequenceClassification (num_labels=2) for clarity.
    """
    policy_model.eval()
    rm_raw.eval()
    # Each model may be on a different device (TRL may move policy to MPS)
    policy_device = next(policy_model.parameters()).device
    rm_device = next(rm_raw.parameters()).device
    gen_kwargs = dict(
        max_new_tokens=RESPONSE_MAX_LEN,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    rewards = []
    for prompt in EVAL_PROMPTS[:n]:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(policy_device)
        with torch.no_grad():
            out = policy_model.generate(ids, **gen_kwargs)
            text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            enc = tokenizer(
                prompt + " " + text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(rm_device)
            logits = rm_raw(**enc).logits   # [1, 2]
            r = (logits[0, 1] - logits[0, 0]).item()
        rewards.append(r)
    return float(np.mean(rewards))


# ── Main PPO training function ────────────────────────────────────────────────

def run_ppo(
    kl_beta:    float = PPO_KL_BETA,
    seed:       int   = 42,
    steps:      int   = PPO_STEPS,
    output_dir: str   = None,
    sft_path:   str   = None,
    rm_path:    str   = None,
) -> dict:
    """
    Run one full PPO training run and return a metrics dict:
        beta, seed, steps,
        mean_reward_before, mean_reward_after, reward_improvement_pct,
        max_kl, mean_kl_final_20, collapse_detected,
        rewards_curve, kl_curve
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = output_dir or PPO_OUTPUT_DIR
    sft_path   = sft_path   or SFT_OUTPUT_DIR
    rm_path    = rm_path    or RM_OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"  # MPS disabled — causes nan/inf in multinomial sampling
    print(f"\n[PPO] device={device}  beta={kl_beta}  seed={seed}  steps={steps}")

    # Explicitly disable MPS so TRL doesn't auto-dispatch to it
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = get_tokenizer(sft_path)

    # ── Policy model (AutoModelForCausalLM for generation + log-probs) ───────
    print("Loading policy model ...")
    policy_model = AutoModelForCausalLM.from_pretrained(sft_path).to(device)

    # ── Reference model (frozen SFT copy — defines the KL baseline) ──────────
    print("Loading reference model ...")
    ref_model = AutoModelForCausalLM.from_pretrained(sft_path).to(device)

    # ── Value model (GPT-2 sequence classifier, num_labels=1) ─────────────────
    # Must implement the get_reward() interface used by TRL internally:
    #   • model.base_model_prefix  → attribute name of the LM backbone
    #   • model.score              → linear head  [hidden_size → 1]
    # AutoModelForSequenceClassification("gpt2", num_labels=1) gives exactly this.
    print("Loading value model ...")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        sft_path, num_labels=1,
    ).to(device)

    # ── Reward model (trained RM wrapped for TRL get_reward() interface) ──────
    print("Loading reward model ...")
    rm_raw = GPT2ForSequenceClassification.from_pretrained(
        rm_path, num_labels=2,
    ).to(device)
    reward_model = RewardModelForPPO(rm_raw).to(device)

    # ── Prompt dataset (must have 'input_ids' column, 1D int lists per item) ──
    print("Loading PPO prompts ...")
    dataset = load_ppo_prompts(tokenizer)

    # ── Baseline reward — evaluated BEFORE PPOTrainer is created ─────────────
    # Must be done here: PPOTrainer.__init__ moves reward_model (which shares
    # rm_raw.transformer by reference) to MPS, leaving rm_raw.score on CPU.
    print("Computing baseline reward ...")
    mean_reward_before = _eval_mean_reward(policy_model, tokenizer, rm_raw, device)
    print(f"  Baseline mean reward: {mean_reward_before:.4f}")

    # ── PPO config ─────────────────────────────────────────────────────────────
    # total_episodes = number of individual prompts seen total
    #   = steps × batch_size  (so that steps = num_total_batches)
    total_episodes = steps * PPO_BATCH_SIZE
    ppo_config = PPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=PPO_BATCH_SIZE,
        num_ppo_epochs=PPO_EPOCHS,
        num_mini_batches=PPO_BATCH_SIZE // PPO_MINI_BATCH,
        total_episodes=total_episodes,
        response_length=RESPONSE_MAX_LEN,
        kl_coef=kl_beta,
        temperature=1.0,
        learning_rate=PPO_LR,
        report_to="none",
        seed=seed,
        logging_steps=10,
        save_strategy="no",
        use_mps_device=False,  # MPS causes nan/inf in multinomial sampling
        bf16=False,            # bf16 unsupported on CPU
        fp16=False,            # fp16 unsupported on CPU
        eval_strategy="no",   # disable eval completions logging (no eval_dataset needed)
    )

    # ── PPOTrainer ─────────────────────────────────────────────────────────────
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=ref_model,
        value_model=value_model,
        reward_model=reward_model,
        train_dataset=dataset,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    # TRL 0.24.0 unconditionally calls generate_completions() at each log step,
    # which crashes when eval_dataloader is None. Patch it out.
    ppo_trainer.generate_completions = lambda *args, **kwargs: None
    print("Starting PPO training ...")
    ppo_trainer.train()

    # ── Post-training reward ───────────────────────────────────────────────────
    # Re-sync rm_raw to same device as policy (PPOTrainer may have moved things)
    post_policy = ppo_trainer.policy_model
    post_device = next(post_policy.parameters()).device
    rm_raw = rm_raw.to(post_device)
    print("Computing post-training reward ...")
    mean_reward_after = _eval_mean_reward(
        post_policy, tokenizer, rm_raw, str(post_device)
    )
    print(f"  Post-PPO mean reward: {mean_reward_after:.4f}")

    # ── Extract training curves from log history ───────────────────────────────
    log_history   = ppo_trainer.state.log_history
    rewards_curve: list = []
    kl_curve:      list = []
    for entry in log_history:
        r = (entry.get("objective/scores")
             or entry.get("objective/rlhf_reward")
             or entry.get("train/reward")
             or entry.get("rewards"))
        k = (entry.get("train/kl")
             or entry.get("objective/kl")
             or entry.get("kl"))
        if r is not None:
            rewards_curve.append(float(r))
        if k is not None:
            kl_curve.append(float(k))

    # ── Aggregate statistics ───────────────────────────────────────────────────
    max_kl = float(np.max(kl_curve)) if kl_curve else 0.0
    mean_kl_final_20 = (
        float(np.mean(kl_curve[-20:])) if len(kl_curve) >= 20
        else float(np.mean(kl_curve)) if kl_curve
        else 0.0
    )
    reward_improvement_pct = (
        100.0 * (mean_reward_after - mean_reward_before) / (abs(mean_reward_before) + 1e-8)
    )
    collapse_detected = max_kl > 20.0

    # ── Save policy model + tokenizer ──────────────────────────────────────────
    ppo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ── Build and persist metrics ──────────────────────────────────────────────
    metrics = {
        "beta":                    kl_beta,
        "seed":                    seed,
        "steps":                   steps,
        "mean_reward_before":      mean_reward_before,
        "mean_reward_after":       mean_reward_after,
        "reward_improvement_pct":  reward_improvement_pct,
        "max_kl":                  max_kl,
        "mean_kl_final_20":        mean_kl_final_20,
        "collapse_detected":       collapse_detected,
        "rewards_curve":           rewards_curve,
        "kl_curve":                kl_curve,
    }
    metrics_path = os.path.join(output_dir, "ppo_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nPPO checkpoint saved to:  {output_dir}")
    print(f"Metrics saved to:         {metrics_path}")
    return metrics


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PPO fine-tuning (C1 Stage 3)")
    parser.add_argument("--beta",       type=float, default=PPO_KL_BETA)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--steps",      type=int,   default=PPO_STEPS)
    parser.add_argument("--output_dir", type=str,   default=None)
    parser.add_argument("--sft_path",   type=str,   default=None)
    parser.add_argument("--rm_path",    type=str,   default=None)
    args = parser.parse_args()

    metrics = run_ppo(
        kl_beta=args.beta,
        seed=args.seed,
        steps=args.steps,
        output_dir=args.output_dir,
        sft_path=args.sft_path,
        rm_path=args.rm_path,
    )

    print("\n── Final metrics ──")
    for k, v in metrics.items():
        if not isinstance(v, list):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
