"""
Interactive CLI Demo — RLHF on GPT-2
======================================
Loads SFT, PPO, and DPO checkpoints and compares their
generation outputs side-by-side with reward scores.

Usage:
    python demo.py                              # interactive loop
    python demo.py --prompt "This movie was"   # single prompt
    python demo.py --all-prompts                # run all preset prompts, save to results/day10/
    python demo.py --max-new-tokens 120        # control output length

Missing checkpoints are skipped gracefully.
"""

import os
import sys
import argparse
import textwrap

import torch
from transformers import AutoModelForCausalLM, GPT2ForSequenceClassification

from data_utils import get_tokenizer
from config import SFT_OUTPUT_DIR, PPO_OUTPUT_DIR, DPO_OUTPUT_DIR, RM_OUTPUT_DIR

# Preset prompts used for the final evaluation run
PRESET_PROMPTS = [
    "This movie was absolutely",
    "The film had incredible",
    "I was disappointed by",
    "One of the best performances I have ever",
    "The director managed to create",
]

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║         RLHF Demo: SFT  vs  PPO  vs  DPO  (GPT-2)          ║
║         Masters in AI — Reinforcement Learning Project       ║
╚══════════════════════════════════════════════════════════════╝
"""

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


def print_and_log(msg: str, fh=None):
    print(msg)
    if fh:
        fh.write(msg + "\n")


def run_prompt(prompt, available, tokenizer, has_rm, rm_model, rm_tokenizer,
               device, max_new_tokens, fh=None):
    """Run one prompt through all models, print results, return {name: reward}."""
    rewards = {}
    print_and_log(f'\n  Prompt: "{prompt}"', fh)
    print_and_log("  " + "─" * 58, fh)
    for name, model in available.items():
        response = generate(model, tokenizer, prompt, device, max_new_tokens)
        wrapped = textwrap.fill(response, width=70,
                                initial_indent="    ", subsequent_indent="    ")
        if has_rm:
            r = reward_score(rm_model, rm_tokenizer, prompt + " " + response, device)
            rewards[name] = r
            print_and_log(f"  [{name}]  reward={r:+.3f}", fh)
        else:
            print_and_log(f"  [{name}]", fh)
        print_and_log(wrapped, fh)
        print_and_log("", fh)
    print_and_log("  " + "═" * 58, fh)
    return rewards


def print_summary(all_rewards, fh=None):
    """Print mean reward per model across all prompts."""
    if not all_rewards:
        return
    model_names = list(next(iter(all_rewards.values())).keys())
    print_and_log("\n  ┌─────────────────────────────────────────────────┐", fh)
    print_and_log("  │             SUMMARY — Mean Rewards               │", fh)
    print_and_log("  ├─────────────────────────┬───────────────────────┤", fh)
    print_and_log("  │ Model                   │ Mean Reward           │", fh)
    print_and_log("  ├─────────────────────────┼───────────────────────┤", fh)
    for name in model_names:
        vals = [r[name] for r in all_rewards.values() if name in r]
        if vals:
            mean = sum(vals) / len(vals)
            print_and_log(f"  │ {name:<23} │ {mean:>+10.3f}              │", fh)
    print_and_log("  └─────────────────────────┴───────────────────────┘", fh)


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RLHF Demo: SFT vs PPO vs DPO side-by-side with reward scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python demo.py                          # interactive mode
              python demo.py --prompt "This movie"   # single prompt
              python demo.py --all-prompts            # run all presets, save log
        """),
    )
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to evaluate")
    parser.add_argument("--all-prompts", action="store_true",
                        help="Run all preset prompts and save results to results/day10/")
    parser.add_argument("--max-new-tokens", type=int, default=80,
                        help="Max new tokens per generation (default: 80)")
    args = parser.parse_args()

    # Disable MPS
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    if hasattr(torch.backends, "mps"):
        torch.backends.mps.enabled = False
    device = "cpu"

    print(BANNER)
    print(f"  Device : {device}")
    print(f"  Tokens : {args.max_new_tokens} max new tokens\n")

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
            print(f"  ✓ Loaded {name:<20} ← {path}")
        else:
            print(f"  ✗ Skipping {name:<18} — {path} not found")

    if not available:
        print("\nNo checkpoints found. Run the training scripts first.")
        sys.exit(1)

    # ── Load reward model ────────────────────────────────────
    has_rm = os.path.isdir(RM_OUTPUT_DIR)
    if has_rm:
        rm_tokenizer = get_tokenizer(RM_OUTPUT_DIR)
        rm_tokenizer.padding_side = "right"
        rm_model = GPT2ForSequenceClassification.from_pretrained(
            RM_OUTPUT_DIR, num_labels=2
        ).to(device).eval()
        print(f"  ✓ Loaded reward model       ← {RM_OUTPUT_DIR}")
    else:
        rm_model = rm_tokenizer = None
        print(f"  ✗ Reward model not found — {RM_OUTPUT_DIR}")

    print()

    # ── All-prompts mode ──────────────────────────────────────
    if args.all_prompts:
        out_dir = "results/day10"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "demo_final.txt")
        all_rewards = {}
        with open(out_path, "w") as fh:
            print_and_log(BANNER.strip(), fh)
            print_and_log(f"  Preset prompts: {len(PRESET_PROMPTS)}", fh)
            print_and_log("", fh)
            for i, prompt in enumerate(PRESET_PROMPTS, 1):
                print_and_log(f"{'═'*62}", fh)
                print_and_log(f"  PROMPT {i}/{len(PRESET_PROMPTS)}", fh)
                rewards = run_prompt(
                    prompt, available, tokenizer, has_rm,
                    rm_model, rm_tokenizer, device, args.max_new_tokens, fh
                )
                all_rewards[prompt] = rewards
            print_summary(all_rewards, fh)
        print(f"\n  Results saved to: {out_path}\n")
        print_summary(all_rewards)
        return

    # ── Single-prompt mode ────────────────────────────────────
    if args.prompt:
        run_prompt(
            args.prompt, available, tokenizer, has_rm,
            rm_model, rm_tokenizer, device, args.max_new_tokens
        )
        return

    # ── Interactive mode ──────────────────────────────────────
    print("  Interactive mode — type a prompt and press Enter.")
    print("  Type 'quit' or press Ctrl+C to exit.\n")
    all_rewards = {}
    while True:
        try:
            prompt = input("  Prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in ("q", "quit", "exit", ""):
            break
        rewards = run_prompt(
            prompt, available, tokenizer, has_rm,
            rm_model, rm_tokenizer, device, args.max_new_tokens
        )
        all_rewards[prompt] = rewards

    if len(all_rewards) > 1:
        print_summary(all_rewards)


if __name__ == "__main__":
    main()
