"""
Interactive CLI Demo
====================
Loads SFT, PPO, and DPO checkpoints, then lets you compare
their generation outputs side-by-side with reward scores.

Usage:
    python demo.py                              # interactive loop
    python demo.py --prompt "This movie was"   # single prompt

Missing checkpoints are skipped gracefully — you can run the
demo after just Stage 1 (SFT) if you haven't trained PPO/DPO yet.
"""

import os
import argparse

import torch
from transformers import AutoModelForCausalLM, GPT2ForSequenceClassification

from data_utils import get_tokenizer
from config import SFT_OUTPUT_DIR, PPO_OUTPUT_DIR, DPO_OUTPUT_DIR, RM_OUTPUT_DIR


# ── Helpers ───────────────────────────────────────────────────

def load_causal_model(path: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(path)
    model.to(device).eval()
    return model


def generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 80) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def reward_score(rm_model, rm_tokenizer, text: str, device: str) -> float:
    inputs = rm_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256
    ).to(device)
    with torch.no_grad():
        logits = rm_model(**inputs).logits
    return (logits[0, 1] - logits[0, 0]).item()


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RLHF Demo: SFT vs PPO vs DPO")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to evaluate (omit for interactive mode)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── Load generation models ───────────────────────────────
    tokenizer = get_tokenizer(SFT_OUTPUT_DIR)

    available = {}
    for name, path in [
        ("SFT (baseline)", SFT_OUTPUT_DIR),
        ("PPO (aligned)",  PPO_OUTPUT_DIR),
        ("DPO (aligned)",  DPO_OUTPUT_DIR),
    ]:
        if os.path.isdir(path):
            available[name] = load_causal_model(path, device)
            print(f"  Loaded {name} from {path}")
        else:
            print(f"  Skipping {name} — {path} not found")

    if not available:
        print("\nNo checkpoints found. Run the training scripts first.")
        return

    # ── Load reward model ────────────────────────────────────
    has_rm = os.path.isdir(RM_OUTPUT_DIR)
    if has_rm:
        rm_tokenizer = get_tokenizer(RM_OUTPUT_DIR)
        rm_tokenizer.padding_side = "right"
        rm_model = GPT2ForSequenceClassification.from_pretrained(
            RM_OUTPUT_DIR, num_labels=2
        ).to(device).eval()
        print(f"  Loaded reward model from {RM_OUTPUT_DIR}")
    else:
        print(f"  Skipping reward model — {RM_OUTPUT_DIR} not found")

    print("\nReady! (type 'quit' to exit)\n")

    # ── Generation loop ───────────────────────────────────────
    while True:
        if args.prompt:
            prompt = args.prompt
        else:
            try:
                prompt = input("Prompt: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

        if prompt.lower() in ("q", "quit", "exit", ""):
            break

        print()
        for name, model in available.items():
            response = generate(model, tokenizer, prompt, device)
            if has_rm:
                r = reward_score(rm_model, rm_tokenizer, prompt + " " + response, device)
                print(f"[{name}]  reward={r:+.3f}")
            else:
                print(f"[{name}]")
            print(f"  {response}\n")

        print("-" * 60)

        if args.prompt:
            break


if __name__ == "__main__":
    main()
