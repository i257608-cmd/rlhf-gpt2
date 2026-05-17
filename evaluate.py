"""
Shared Evaluation Script
========================
Computes reward scores, perplexity, and alignment tax for
all three contributions.

Usage:
    python evaluate.py              # runs all modes
    python evaluate.py --mode c1    # SFT vs PPO (Contribution 1)
    python evaluate.py --mode c2    # SFT vs PPO vs DPO (Contribution 2)

Output:
    results/c1/eval_results.json
    results/c2/comparison.json
"""

import os
import json
import argparse

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, GPT2ForSequenceClassification

from data_utils import get_tokenizer
from config import (
    SFT_OUTPUT_DIR, PPO_OUTPUT_DIR, DPO_OUTPUT_DIR, RM_OUTPUT_DIR,
    RESULTS_DIR, EVAL_PROMPTS, MAX_LENGTH,
)


# ── Shared helpers ────────────────────────────────────────────

def load_rm(rm_path: str, device: str):
    """Load reward model and its tokenizer."""
    tokenizer = get_tokenizer(rm_path)
    tokenizer.padding_side = "right"
    model = GPT2ForSequenceClassification.from_pretrained(rm_path, num_labels=2)
    model.to(device).eval()
    return model, tokenizer


def score_text(rm_model, rm_tokenizer, text: str, device: str) -> float:
    """Scalar reward for a single text string."""
    inputs = rm_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256
    ).to(device)
    with torch.no_grad():
        logits = rm_model(**inputs).logits
    return (logits[0, 1] - logits[0, 0]).item()


def generate_response(model, tokenizer, prompt: str, device: str,
                       max_new_tokens: int = 64) -> str:
    """Generate one response from a model given a prompt."""
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


def compute_perplexity(model, tokenizer, texts: list[str], device: str,
                        max_length: int = MAX_LENGTH) -> float:
    """
    Average perplexity of a causal LM on a list of texts.
    Perplexity = exp(average NLL loss).
    Used to measure the alignment tax:
        alignment_tax_% = (ppl_ppo - ppl_sft) / ppl_sft * 100
    """
    model.eval()
    total_loss = 0.0
    count = 0

    for text in tqdm(texts, desc="Perplexity", leave=False):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)
        if inputs["input_ids"].shape[1] < 2:
            continue
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / max(count, 1)
    return float(np.exp(avg_loss))


# ── Contribution 1: SFT vs PPO ────────────────────────────────

def run_c1_eval(device: str) -> dict:
    """
    Evaluates SFT baseline vs PPO-aligned model.
    Reports:
      • qualitative generation examples for all EVAL_PROMPTS
      • mean reward per model
      • perplexity per model → alignment tax %
    """
    print("\n" + "=" * 60)
    print("CONTRIBUTION 1 EVALUATION: SFT vs PPO")
    print("=" * 60)

    c1_dir = os.path.join(RESULTS_DIR, "c1")
    os.makedirs(c1_dir, exist_ok=True)

    rm_model, rm_tokenizer = load_rm(RM_OUTPUT_DIR, device)

    sft_tokenizer = get_tokenizer(SFT_OUTPUT_DIR)
    ppo_tokenizer = get_tokenizer(PPO_OUTPUT_DIR)
    sft_model = AutoModelForCausalLM.from_pretrained(SFT_OUTPUT_DIR).to(device).eval()
    ppo_model = AutoModelForCausalLM.from_pretrained(PPO_OUTPUT_DIR).to(device).eval()

    results = {"prompts": []}
    sft_rewards, ppo_rewards = [], []

    print("\n--- Qualitative Examples ---")
    for prompt in EVAL_PROMPTS:
        sft_resp = generate_response(sft_model, sft_tokenizer, prompt, device)
        ppo_resp = generate_response(ppo_model, ppo_tokenizer, prompt, device)

        sft_r = score_text(rm_model, rm_tokenizer, prompt + " " + sft_resp, device)
        ppo_r = score_text(rm_model, rm_tokenizer, prompt + " " + ppo_resp, device)

        sft_rewards.append(sft_r)
        ppo_rewards.append(ppo_r)

        results["prompts"].append({
            "prompt":       prompt,
            "sft_response": sft_resp, "sft_reward": sft_r,
            "ppo_response": ppo_resp, "ppo_reward": ppo_r,
        })

        print(f"\nPrompt: {prompt!r}")
        print(f"  SFT  (reward={sft_r:+.3f}): {sft_resp[:120]}")
        print(f"  PPO  (reward={ppo_r:+.3f}): {ppo_resp[:120]}")

    # ── Perplexity (alignment tax) ────────────────────────────
    print("\nComputing perplexity on held-out IMDB test set ...")
    ppl_texts = (load_dataset("imdb", split="test")
                 .shuffle(seed=0).select(range(200))["text"])

    sft_ppl = compute_perplexity(sft_model, sft_tokenizer, ppl_texts, device)
    ppo_ppl = compute_perplexity(ppo_model, ppo_tokenizer, ppl_texts, device)
    alignment_tax = (ppo_ppl - sft_ppl) / max(sft_ppl, 1e-8) * 100

    results["aggregate"] = {
        "sft_mean_reward":   float(np.mean(sft_rewards)),
        "ppo_mean_reward":   float(np.mean(ppo_rewards)),
        "reward_delta":      float(np.mean(ppo_rewards) - np.mean(sft_rewards)),
        "sft_perplexity":    sft_ppl,
        "ppo_perplexity":    ppo_ppl,
        "alignment_tax_pct": alignment_tax,
    }

    print("\n--- Aggregate Results ---")
    print(f"  SFT mean reward  : {results['aggregate']['sft_mean_reward']:+.4f}")
    print(f"  PPO mean reward  : {results['aggregate']['ppo_mean_reward']:+.4f}")
    print(f"  Reward delta     : {results['aggregate']['reward_delta']:+.4f}")
    print(f"  SFT perplexity   : {sft_ppl:.2f}")
    print(f"  PPO perplexity   : {ppo_ppl:.2f}")
    print(f"  Alignment tax    : {alignment_tax:+.1f}%")

    out_path = os.path.join(c1_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_path}")
    return results


# ── Contribution 2: SFT vs PPO vs DPO ────────────────────────

def run_c2_eval(device: str) -> dict:
    """
    Head-to-head comparison: SFT (baseline), PPO, and DPO.
    Reports mean reward and perplexity for each model.
    """
    print("\n" + "=" * 60)
    print("CONTRIBUTION 2 EVALUATION: SFT vs PPO vs DPO")
    print("=" * 60)

    c2_dir = os.path.join(RESULTS_DIR, "c2")
    os.makedirs(c2_dir, exist_ok=True)

    rm_model, rm_tokenizer = load_rm(RM_OUTPUT_DIR, device)

    models_cfg = [
        ("SFT (baseline)", SFT_OUTPUT_DIR),
        ("PPO (aligned)",  PPO_OUTPUT_DIR),
        ("DPO (aligned)",  DPO_OUTPUT_DIR),
    ]

    ppl_texts = (load_dataset("imdb", split="test")
                 .shuffle(seed=0).select(range(100))["text"])

    aggregate = {}

    for label, path in models_cfg:
        if not os.path.isdir(path):
            print(f"  WARNING: {path} not found — skipping {label}")
            continue

        tokenizer = get_tokenizer(path)
        model = AutoModelForCausalLM.from_pretrained(path).to(device).eval()

        rewards = [
            score_text(
                rm_model, rm_tokenizer,
                prompt + " " + generate_response(model, tokenizer, prompt, device),
                device,
            )
            for prompt in EVAL_PROMPTS
        ]
        ppl = compute_perplexity(model, tokenizer, ppl_texts, device)

        aggregate[label] = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward":  float(np.std(rewards)),
            "perplexity":  ppl,
        }
        print(f"  {label:<22} reward={np.mean(rewards):+.4f} ± {np.std(rewards):.4f}  "
              f"ppl={ppl:.2f}")

    out_path = os.path.join(c2_dir, "comparison.json")
    with open(out_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nSaved to: {out_path}")
    return aggregate


# ── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate RLHF models")
    parser.add_argument("--mode", choices=["c1", "c2", "all"], default="all",
                        help="Which evaluation to run (default: all)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.mode in ("c1", "all"):
        run_c1_eval(device)
    if args.mode in ("c2", "all"):
        run_c2_eval(device)


if __name__ == "__main__":
    main()
