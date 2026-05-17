"""
Contribution 1 — Stage 2: Reward Model Training
================================================
Trains GPT2ForSequenceClassification on the full IMDB dataset
(positive label = 1, negative label = 0).
The trained model is then used as the reward function during PPO.

Reward signal during PPO:
    reward = logits[positive_class] - logits[negative_class]
           = logit[1] - logit[0]    (higher = more positive)

Usage:
    python train_reward_model.py

Output:
    checkpoints/reward_model/   (model + tokenizer)
"""

import os
import numpy as np
import torch
from transformers import (
    GPT2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from data_utils import load_rm_dataset, get_tokenizer
from config import BASE_MODEL, MAX_LENGTH, RM_LR, RM_BATCH_SIZE, RM_EPOCHS, RM_OUTPUT_DIR


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = float((preds == labels).mean())
    return {"accuracy": accuracy}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Tokenizer (right-padding for classification) ──────────
    print("Loading tokenizer ...")
    tokenizer = get_tokenizer(BASE_MODEL)
    tokenizer.padding_side = "right"

    # ── Dataset ───────────────────────────────────────────────
    print("Loading IMDB (pos + neg) ...")
    train_ds, eval_ds = load_rm_dataset(
        tokenizer,
        train_size=4_000,
        eval_size=400,
        max_length=MAX_LENGTH,
    )
    print(f"  Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # ── Model ─────────────────────────────────────────────────
    print("Loading GPT-2ForSequenceClassification ...")
    model = GPT2ForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    model.config.pad_token_id = tokenizer.pad_token_id

    # ── Training config ───────────────────────────────────────
    os.makedirs(RM_OUTPUT_DIR, exist_ok=True)
    os.makedirs("logs/rm", exist_ok=True)

    training_args = TrainingArguments(
        output_dir=RM_OUTPUT_DIR,
        num_train_epochs=RM_EPOCHS,
        per_device_train_batch_size=RM_BATCH_SIZE,
        per_device_eval_batch_size=RM_BATCH_SIZE,
        learning_rate=RM_LR,
        weight_decay=0.01,
        warmup_steps=100,
        logging_dir="logs/rm",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    # ── Train ─────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    print("Training reward model ...")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────
    trainer.save_model(RM_OUTPUT_DIR)
    tokenizer.save_pretrained(RM_OUTPUT_DIR)
    print(f"\nReward model saved to: {RM_OUTPUT_DIR}")
    print("Next step: python train_ppo.py")


if __name__ == "__main__":
    main()
