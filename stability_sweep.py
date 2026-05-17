"""
Contribution 3 — Stability Benchmarking & Alignment Tax
========================================================
Grid-sweeps PPO over KL beta values and random seeds.
For each (beta, seed) pair the script calls run_ppo() from train_ppo.py,
records the metrics, and produces:

  results/stability_sweep/sweep_results.csv   — one row per run
  results/stability_sweep/summary.csv         — aggregated by beta

Key metrics reported per beta value:
  • mean_reward_after   — average final reward
  • std_reward          — variation across seeds (stability)
  • collapse_rate       — fraction of seeds where collapse was detected
  • mean_kl_final       — average KL divergence in last 20 steps
  • max_kl              — worst-case KL observed

Usage:
    python stability_sweep.py
"""

import os
import time

import pandas as pd

from config import SWEEP_BETAS, SWEEP_SEEDS, SWEEP_STEPS, SWEEP_OUTPUT_DIR
from train_ppo import run_ppo


def main():
    os.makedirs(SWEEP_OUTPUT_DIR, exist_ok=True)

    total_runs = len(SWEEP_BETAS) * len(SWEEP_SEEDS)
    records: list[dict] = []

    print(f"Starting stability sweep: {len(SWEEP_BETAS)} betas × "
          f"{len(SWEEP_SEEDS)} seeds = {total_runs} runs\n")

    run_idx = 0
    for beta in SWEEP_BETAS:
        for seed in SWEEP_SEEDS:
            run_idx += 1
            print(f"\n{'='*60}")
            print(f"Run {run_idx}/{total_runs}  beta={beta}  seed={seed}")
            print(f"{'='*60}")

            out_dir = os.path.join(SWEEP_OUTPUT_DIR, f"beta{beta}_seed{seed}")
            t0 = time.time()

            try:
                metrics = run_ppo(
                    kl_beta=beta,
                    seed=seed,
                    steps=SWEEP_STEPS,
                    output_dir=out_dir,
                )
                elapsed = time.time() - t0
                metrics["elapsed_seconds"] = elapsed
                metrics["status"] = "ok"

            except Exception as exc:
                elapsed = time.time() - t0
                print(f"ERROR: {exc}")
                metrics = {
                    "beta":     beta,
                    "seed":     seed,
                    "status":   "error",
                    "error":    str(exc),
                    "elapsed_seconds": elapsed,
                }

            # Remove the long curve lists before saving to CSV
            for key in ("rewards_curve", "kl_curve"):
                metrics.pop(key, None)

            records.append(metrics)

            # Checkpoint results after every run
            df = pd.DataFrame(records)
            df.to_csv(
                os.path.join(SWEEP_OUTPUT_DIR, "sweep_results.csv"),
                index=False,
            )

    # ── Summary table ─────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("STABILITY SWEEP SUMMARY")
    print("=" * 70)

    ok_records = [r for r in records if r.get("status") == "ok"]

    if ok_records:
        df_ok = pd.DataFrame(ok_records)

        summary = df_ok.groupby("beta").agg(
            mean_reward_after=("mean_reward_after", "mean"),
            std_reward=("mean_reward_after",  "std"),
            collapse_rate=("collapse_detected", "mean"),
            mean_kl_final=("mean_kl_final_20",  "mean"),
            max_kl=("max_kl",                   "mean"),
        ).round(4)

        print(summary.to_string())
        summary.to_csv(os.path.join(SWEEP_OUTPUT_DIR, "summary.csv"))

        print(f"\nFull results : {SWEEP_OUTPUT_DIR}/sweep_results.csv")
        print(f"Summary      : {SWEEP_OUTPUT_DIR}/summary.csv")
    else:
        print("No successful runs to summarise.")

    failed = total_runs - len(ok_records)
    if failed:
        print(f"\nWARNING: {failed}/{total_runs} run(s) failed. Check sweep_results.csv.")


if __name__ == "__main__":
    main()
