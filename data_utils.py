"""
Dataset loading utilities for all RLHF training stages.
Each function returns exactly the format expected by its training script.
"""

import torch
from datasets import load_dataset, Dataset as HFDataset


# ── Shared tokenizer helper ───────────────────────────────────

def get_tokenizer(model_name_or_path: str):
    """Load tokenizer; GPT-2 has no pad token so we set it to eos."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ── Stage 1: SFT ─────────────────────────────────────────────

def load_sft_dataset(train_size: int = 5_000, eval_size: int = 500):
    """
    Positive IMDB reviews for Supervised Fine-Tuning.
    Returns two HuggingFace datasets with a single 'text' column —
    SFTTrainer handles tokenisation internally.
    """
    ds = load_dataset("imdb", split="train")
    positives = ds.filter(lambda x: x["label"] == 1).shuffle(seed=42)
    train = positives.select(range(train_size)).select_columns(["text"])
    val   = positives.select(range(train_size, train_size + eval_size)).select_columns(["text"])
    return train, val


# ── Stage 2: Reward Model ─────────────────────────────────────

def load_rm_dataset(tokenizer, train_size: int = 4_000, eval_size: int = 400,
                    max_length: int = 128):
    """
    Full IMDB dataset (pos + neg) for reward model training.
    Returns two torch-formatted datasets with input_ids, attention_mask, labels.
    """
    tokenizer.padding_side = "right"   # RM needs right-padding

    train_ds = load_dataset("imdb", split="train").shuffle(seed=42).select(range(train_size))
    eval_ds  = load_dataset("imdb", split="test").shuffle(seed=42).select(range(eval_size))

    def tokenize(examples):
        enc = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        enc["labels"] = examples["label"]
        return enc

    cols_to_remove = ["text", "label"]
    train_tok = train_ds.map(tokenize, batched=True, remove_columns=cols_to_remove)
    eval_tok  = eval_ds.map(tokenize,  batched=True, remove_columns=cols_to_remove)
    train_tok.set_format("torch")
    eval_tok.set_format("torch")
    return train_tok, eval_tok


# ── Stage 3 / C2: PPO query prompts ──────────────────────────

def load_ppo_prompts(tokenizer, size: int = 1_000, prompt_max_len: int = 64):
    """
    Short IMDB review prefixes used as PPO rollout queries.
    Returns a dataset with 'input_ids' (list of int) and 'query' (str).
    """
    ds = load_dataset("imdb", split="test").shuffle(seed=42).select(range(size))

    def process(examples):
        queries = []
        for text in examples["text"]:
            # Take the first sentence up to 200 chars
            first = text[:200].split(". ")[0].strip()
            queries.append(first + "." if not first.endswith(".") else first)
        enc = tokenizer(
            queries,
            truncation=True,
            max_length=prompt_max_len,
            padding=False,
        )
        return {"input_ids": enc["input_ids"], "query": queries}

    return ds.map(process, batched=True, remove_columns=["text", "label"])


# ── C2: DPO preference pairs ──────────────────────────────────

def load_dpo_dataset(train_size: int = 4_000, eval_size: int = 400):
    """
    IMDB-derived preference pairs for DPO training.
    Format per row: {"prompt": str, "chosen": str, "rejected": str}
      - prompt:   shared fixed prefix
      - chosen:   a positive review (preferred completion)
      - rejected: a negative review (dispreferred completion)
    DPOTrainer internally computes: prompt+chosen vs prompt+rejected.
    """
    PROMPT = "Write a movie review: "

    def split_by_label(ds, n_pos, n_neg):
        pos = [x["text"] for x in ds if x["label"] == 1][:n_pos]
        neg = [x["text"] for x in ds if x["label"] == 0][:n_neg]
        return pos, neg

    train_ds = load_dataset("imdb", split="train").shuffle(seed=42)
    test_ds  = load_dataset("imdb", split="test").shuffle(seed=42)

    half_train = train_size // 2
    half_eval  = eval_size  // 2

    pos_train, neg_train = split_by_label(train_ds, half_train, half_train)
    pos_eval,  neg_eval  = split_by_label(test_ds,  half_eval,  half_eval)

    def make_pairs(positives, negatives):
        return [
            {
                "prompt":   PROMPT,
                "chosen":   p[:400],
                "rejected": n[:400],
            }
            for p, n in zip(positives, negatives)
        ]

    return (
        HFDataset.from_list(make_pairs(pos_train, neg_train)),
        HFDataset.from_list(make_pairs(pos_eval,  neg_eval)),
    )
