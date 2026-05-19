"""One-shot: compute DPO eval reward from saved checkpoint and write dpo_metrics.json."""
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, GPT2ForSequenceClassification
from data_utils import get_tokenizer
from config import DPO_OUTPUT_DIR, RM_OUTPUT_DIR, EVAL_PROMPTS

TRAIN_LOSS = 0.21286743958791096
SEED = 42
DEVICE = "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)

print("Loading DPO model from checkpoint ...")
tokenizer = get_tokenizer(DPO_OUTPUT_DIR)
model = AutoModelForCausalLM.from_pretrained(DPO_OUTPUT_DIR).to(DEVICE)
model.eval()

print("Loading reward model ...")
rm_tokenizer = get_tokenizer(RM_OUTPUT_DIR)
rm_tokenizer.padding_side = "right"
rm_model = GPT2ForSequenceClassification.from_pretrained(
    RM_OUTPUT_DIR, num_labels=2
).to(DEVICE)
rm_model.eval()

generation_kwargs = dict(
    max_new_tokens=64,
    do_sample=True,
    top_p=0.9,
    temperature=1.0,
    pad_token_id=tokenizer.eos_token_id,
)

rewards = []
with torch.no_grad():
    for prompt in EVAL_PROMPTS[:5]:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        out = model.generate(ids, **generation_kwargs)
        response = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

        rm_inputs = rm_tokenizer(
            prompt + " " + response,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(DEVICE)
        logits = rm_model(**rm_inputs).logits
        r = (logits[0, 1] - logits[0, 0]).item()
        rewards.append(r)
        print(f"  reward={r:.4f}  response[:60]={response[:60]!r}")

mean_reward = float(np.mean(rewards))
print(f"\nDPO mean reward: {mean_reward:.4f}")

metrics = {
    "seed": SEED,
    "mean_reward_after": mean_reward,
    "train_loss": TRAIN_LOSS,
}
import os
path = os.path.join(DPO_OUTPUT_DIR, "dpo_metrics.json")
with open(path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved: {path}")
