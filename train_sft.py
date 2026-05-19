"""
Contribution 1 — Stage 1: Supervised Fine-Tuning (SFT)
=======================================================
Fine-tunes GPT-2 on positive IMDB reviews so the model learns
to generate positive-sentiment movie review text.
The saved checkpoint becomes the starting point for both PPO and DPO.

Usage:
    python train_sft.py

Output:
    checkpoints/sft/   (model + tokenizer)
"""

import os
import torch
from transformers import AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

from data_utils import load_sft_dataset, get_tokenizer
from config import BASE_MODEL, MAX_LENGTH, SFT_LR, SFT_BATCH_SIZE, SFT_EPOCHS, SFT_OUTPUT_DIR


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Tokenizer ────────────────────────────────────────────
    print("Loading tokenizer ...")
    tokenizer = get_tokenizer(BASE_MODEL)

    # ── Model ────────────────────────────────────────────────
    print("Loading GPT-2 ...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.config.pad_token_id = tokenizer.pad_token_id

    # ── Dataset ──────────────────────────────────────────────
    print("Loading IMDB (positive reviews) ...")
    train_ds, eval_ds = load_sft_dataset(
        train_size=5_000,
        eval_size=500,
    )
    print(f"  Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # ── Training config ───────────────────────────────────────
    os.makedirs(SFT_OUTPUT_DIR, exist_ok=True)
    os.makedirs("logs/sft", exist_ok=True)

    training_args = SFTConfig(
        output_dir=SFT_OUTPUT_DIR,
        num_train_epochs=SFT_EPOCHS,
        per_device_train_batch_size=SFT_BATCH_SIZE,
        per_device_eval_batch_size=SFT_BATCH_SIZE,
        learning_rate=SFT_LR,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="logs/sft",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        max_length=MAX_LENGTH,
        dataset_text_field="text",    # column name in our dataset
    )

    # ── Train ────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("Starting SFT training ...")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────
    trainer.save_model(SFT_OUTPUT_DIR)
    tokenizer.save_pretrained(SFT_OUTPUT_DIR)
    print(f"\nSFT checkpoint saved to: {SFT_OUTPUT_DIR}")
    print("Next step: python train_reward_model.py")


if __name__ == "__main__":
    main()
