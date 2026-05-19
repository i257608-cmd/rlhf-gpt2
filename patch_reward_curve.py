"""Extract rewards_curve from ppo_training.log and patch ppo_metrics.json."""
import re
import json

LOG_PATH = "ppo_training.log"
METRICS_PATH = "checkpoints/ppo/ppo_metrics.json"

rewards_curve = []
pattern = re.compile(r"'objective/scores':\s*([-\d.eE+]+)")

with open(LOG_PATH, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            rewards_curve.append(float(m.group(1)))

print(f"Found {len(rewards_curve)} reward entries")
print(f"  First 5: {rewards_curve[:5]}")
print(f"  Last  5: {rewards_curve[-5:]}")

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

metrics["rewards_curve"] = rewards_curve

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Patched {METRICS_PATH}")
