"""
Contribution 2 — DPO Training
==============================
Trains the SFT model using Direct Preference Optimisation (DPO)
on IMDB preference pairs (positive reviews = chosen, negative = rejected).
No reward model or PPO loop needed — DPO optimises the policy directly
from human preference data.

Usage:
    python train_dpo.py
    python train_dpo.py --seed 42

Output:
    checkpoints/dpo/           (model + tokenizer)
    checkpoints/dpo/dpo_metrics.json
"""

import os
import json
import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, GPT2ForSequenceClassification
from trl import DPOTrainer, DPOConfig

from data_utils import load_dpo_dataset, get_tokenizer
from config import (
    SFT_OUTPUT_DIR, RM_OUTPUT_DIR, DPO_OUTPUT_DIR,
    DPO_LR, DPO_BATCH_SIZE, DPO_EPOCHS, DPO_BETA,
    MAX_LENGTH, EVAL_PROMPTS,
)


def run_dpo(seed: int = 42, output_dir: str = None) -> dict:
    """
    Run a DPO training run and return a metrics dict with:
        seed, mean_reward_after, train_loss
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = output_dir or DPO_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs/dpo", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[DPO] device={device}  seed={seed}")

    # ── Tokenizer ────────────────────────────────────────────
    tokenizer = get_tokenizer(SFT_OUTPUT_DIR)

    # ── Policy model (starts from SFT) ────────────────────────
    print("Loading SFT model as DPO policy ...")
    model = AutoModelForCausalLM.from_pretrained(SFT_OUTPUT_DIR)
    model.to(device)

    # ── Reference model (frozen SFT — defines the prior) ──────
    print("Loading SFT model as DPO reference ...")
    ref_model = AutoModelForCausalLM.from_pretrained(SFT_OUTPUT_DIR)
    ref_model.to(device)

    # ── Preference dataset ───────────────────────────────────
    print("Building IMDB preference pairs ...")
    train_ds, eval_ds = load_dpo_dataset(
        train_size=4_000,
        eval_size=400,
    )
    print(f"  Train pairs: {len(train_ds)} | Eval pairs: {len(eval_ds)}")

    # ── DPO config ────────────────────────────────────────────
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=DPO_EPOCHS,
        per_device_train_batch_size=DPO_BATCH_SIZE,
        per_device_eval_batch_size=DPO_BATCH_SIZE,
        learning_rate=DPO_LR,
        weight_decay=0.01,
        warmup_steps=50,
        beta=DPO_BETA,
        max_length=MAX_LENGTH,
        max_prompt_length=64,
        logging_dir="logs/dpo",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        seed=seed,
    )

    # ── Train ────────────────────────────────────────────────
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    print("Starting DPO training ...")
    train_result = trainer.train()

    # ── Save ──────────────────────────────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ── Post-training reward scoring ──────────────────────────
    # Load the reward model to measure how well DPO improved sentiment
    rm_tokenizer = get_tokenizer(RM_OUTPUT_DIR)
    rm_tokenizer.padding_side = "right"
    rm_model = GPT2ForSequenceClassification.from_pretrained(
        RM_OUTPUT_DIR, num_labels=2
    ).to(device)
    rm_model.eval()

    generation_kwargs = dict(
        max_new_tokens=64,
        do_sample=True,
        top_p=0.9,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    model.eval()
    rewards_after = []
    with torch.no_grad():
        for prompt in EVAL_PROMPTS[:5]:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            out = model.generate(ids, **generation_kwargs)
            response = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

            rm_inputs = rm_tokenizer(
                prompt + " " + response,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(device)
            logits = rm_model(**rm_inputs).logits
            r = (logits[0, 1] - logits[0, 0]).item()
            rewards_after.append(r)

    mean_reward = float(np.mean(rewards_after))
    print(f"\nDPO mean reward on eval prompts: {mean_reward:.4f}")

    # ── Save metrics ──────────────────────────────────────────
    metrics = {
        "seed":               seed,
        "mean_reward_after":  mean_reward,
        "train_loss":         train_result.training_loss,
    }
    metrics_path = os.path.join(output_dir, "dpo_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDPO checkpoint saved to:  {output_dir}")
    print(f"Metrics saved to:         {metrics_path}")
    print("Next step: python evaluate.py --mode c2")
    return metrics


# ── CLI entry point ───────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO fine-tuning (C2)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save the model (default: checkpoints/dpo)")
    args = parser.parse_args()

    run_dpo(seed=args.seed, output_dir=args.output_dir)
