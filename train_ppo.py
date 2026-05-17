"""
Contribution 1 — Stage 3: PPO Policy Optimisation
===================================================
Loads the SFT model as the starting policy, initialises a frozen reference
copy, then runs PPO with the trained reward model as the scalar signal.

KL penalty keeps the policy close to the reference to prevent reward hacking.
This script is also imported by stability_sweep.py (C3) via run_ppo().

Usage:
    python train_ppo.py
    python train_ppo.py --beta 0.2 --seed 42 --steps 200
    python train_ppo.py --beta 0.05 --seed 123 --steps 200 --output_dir results/run_A

Output:
    checkpoints/ppo/           (default model + tokenizer)
    <output_dir>/ppo_metrics.json
"""

import os
import json
import argparse

import numpy as np
import torch
from tqdm import tqdm
from transformers import GPT2ForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from torch.utils.data import DataLoader

from data_utils import load_ppo_prompts, get_tokenizer
from config import (
    SFT_OUTPUT_DIR, RM_OUTPUT_DIR, PPO_OUTPUT_DIR, RESULTS_DIR,
    PPO_LR, PPO_BATCH_SIZE, PPO_MINI_BATCH, PPO_EPOCHS,
    RESPONSE_MAX_LEN, EVAL_PROMPTS,
)


# ── Reward function ───────────────────────────────────────────

def build_reward_fn(rm_model, rm_tokenizer, device):
    """
    Returns a callable:
        reward_fn(texts: list[str]) -> list[torch.Tensor]
    Each tensor is a scalar = logit[positive] - logit[negative].
    Higher value = more positive-sentiment text.
    """
    rm_model.eval()

    def reward_fn(texts):
        rewards = []
        for text in texts:
            inputs = rm_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            ).to(device)
            with torch.no_grad():
                logits = rm_model(**inputs).logits   # shape [1, 2]
            r = (logits[0, 1] - logits[0, 0]).item()
            rewards.append(torch.tensor(r, dtype=torch.float32))
        return rewards

    return reward_fn


# ── Baseline / evaluation helper ─────────────────────────────

def _eval_mean_reward(model, tokenizer, reward_fn, device, gen_kwargs, n=5):
    """Generate from EVAL_PROMPTS and return mean reward (no grad)."""
    rewards = []
    for prompt in EVAL_PROMPTS[:n]:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.pretrained_model.generate(ids, **gen_kwargs)
        response = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        r = reward_fn([prompt + " " + response])
        rewards.append(r[0].item())
    return float(np.mean(rewards))


# ── Main PPO training function ────────────────────────────────

def run_ppo(
    kl_beta:    float = 0.2,
    seed:       int   = 42,
    steps:      int   = 200,
    output_dir: str   = None,
    sft_path:   str   = None,
    rm_path:    str   = None,
) -> dict:
    """
    Run a full PPO training run.

    Returns a metrics dict with:
        beta, seed, steps,
        mean_reward_before, mean_reward_after, reward_improvement_pct,
        max_kl, mean_kl_final_20,
        collapse_detected,
        rewards_curve, kl_curve
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = output_dir or PPO_OUTPUT_DIR
    sft_path   = sft_path   or SFT_OUTPUT_DIR
    rm_path    = rm_path    or RM_OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[PPO] device={device}  beta={kl_beta}  seed={seed}  steps={steps}")

    # ── Tokenizer (left-padding for generation) ───────────────
    tokenizer = get_tokenizer(sft_path)

    # ── Policy model (value head added by TRL) ────────────────
    model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_path)
    model.to(device)

    # ── Reference model (frozen SFT copy) ────────────────────
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_path)
    ref_model.to(device)

    # ── Reward model ──────────────────────────────────────────
    rm_tokenizer = get_tokenizer(rm_path)
    rm_tokenizer.padding_side = "right"
    rm_model = GPT2ForSequenceClassification.from_pretrained(
        rm_path, num_labels=2
    ).to(device)
    reward_fn = build_reward_fn(rm_model, rm_tokenizer, device)

    # ── PPO config ────────────────────────────────────────────
    ppo_config = PPOConfig(
        model_name=sft_path,
        learning_rate=PPO_LR,
        ppo_epochs=PPO_EPOCHS,
        mini_batch_size=PPO_MINI_BATCH,
        batch_size=PPO_BATCH_SIZE,
        init_kl_coef=kl_beta,
        kl_penalty="kl",
        adap_kl_ctrl=False,   # fixed β — we are sweeping it in C3
        seed=seed,
        log_with=None,
    )

    # ── PPO dataset ───────────────────────────────────────────
    dataset = load_ppo_prompts(tokenizer)

    def collate_fn(batch):
        return {
            "input_ids": [item["input_ids"] for item in batch],
            "query":     [item["query"]     for item in batch],
        }

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collate_fn,
    )

    generation_kwargs = dict(
        min_new_tokens=4,
        max_new_tokens=RESPONSE_MAX_LEN,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    # ── Baseline reward (before any PPO update) ───────────────
    print("Computing baseline reward ...")
    mean_reward_before = _eval_mean_reward(model, tokenizer, reward_fn, device, generation_kwargs)
    print(f"  Baseline mean reward: {mean_reward_before:.4f}")

    # ── PPO training loop ─────────────────────────────────────
    all_rewards:  list[float] = []
    all_kl:       list[float] = []
    peak_reward   = -float("inf")
    collapse_detected = False

    step_count = 0
    pbar = tqdm(ppo_trainer.dataloader, desc="PPO steps", total=steps)

    for batch in pbar:
        if step_count >= steps:
            break

        # query_tensors must be a list of 1-D tensors
        query_tensors = batch["input_ids"]
        if isinstance(query_tensors, torch.Tensor):
            query_tensors = [query_tensors[i] for i in range(len(query_tensors))]

        # Generate responses (return new tokens only)
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **generation_kwargs,
        )

        queries   = tokenizer.batch_decode(query_tensors,   skip_special_tokens=True)
        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # Score each (query + response) pair
        rewards = reward_fn([q + " " + r for q, r in zip(queries, responses)])

        # PPO gradient update
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        mean_r = float(np.mean([r.item() for r in rewards]))
        kl     = float(stats.get("objective/kl",
                       stats.get("ppo/mean_non_score_reward", 0.0)))

        all_rewards.append(mean_r)
        all_kl.append(kl)

        # Collapse detection: reward drops >30 % below running peak
        if mean_r > peak_reward:
            peak_reward = mean_r
        if peak_reward > 0 and mean_r < peak_reward * 0.70:
            collapse_detected = True

        step_count += 1
        pbar.set_postfix(reward=f"{mean_r:.3f}", kl=f"{kl:.4f}")

        if step_count % 50 == 0:
            print(f"  Step {step_count:>3}: mean_reward={mean_r:.4f}  kl={kl:.5f}")

    # ── Final reward (after PPO) ──────────────────────────────
    print("Computing final reward ...")
    mean_reward_after = _eval_mean_reward(model, tokenizer, reward_fn, device, generation_kwargs)
    improvement = (mean_reward_after - mean_reward_before) / (abs(mean_reward_before) + 1e-8) * 100
    print(f"  Final reward: {mean_reward_after:.4f}  "
          f"(Δ = {mean_reward_after - mean_reward_before:+.4f}, "
          f"+{improvement:.1f}%)")

    if collapse_detected:
        print("  WARNING: reward collapse detected during training!")

    # ── Save model ────────────────────────────────────────────
    ppo_trainer.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ── Save metrics ──────────────────────────────────────────
    metrics = {
        "beta":                   kl_beta,
        "seed":                   seed,
        "steps":                  step_count,
        "mean_reward_before":     mean_reward_before,
        "mean_reward_after":      mean_reward_after,
        "reward_improvement_pct": improvement,
        "max_kl":                 float(max(all_kl)) if all_kl else 0.0,
        "mean_kl_final_20":       float(np.mean(all_kl[-20:])) if len(all_kl) >= 20
                                  else float(np.mean(all_kl)) if all_kl else 0.0,
        "collapse_detected":      collapse_detected,
        "rewards_curve":          all_rewards,
        "kl_curve":               all_kl,
    }

    metrics_path = os.path.join(output_dir, "ppo_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nPPO checkpoint saved to:  {output_dir}")
    print(f"Metrics saved to:         {metrics_path}")
    print("Next step: python evaluate.py --mode c1")
    return metrics


# ── CLI entry point ───────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO fine-tuning (C1 Stage 3)")
    parser.add_argument("--beta",       type=float, default=0.2,
                        help="KL penalty coefficient (default: 0.2)")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--steps",      type=int,   default=200,
                        help="Number of PPO rollout steps (default: 200)")
    parser.add_argument("--output_dir", type=str,   default=None,
                        help="Where to save the model (default: checkpoints/ppo)")
    args = parser.parse_args()

    run_ppo(
        kl_beta=args.beta,
        seed=args.seed,
        steps=args.steps,
        output_dir=args.output_dir,
    )
