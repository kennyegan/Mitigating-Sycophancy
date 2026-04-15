#!/usr/bin/env python3
"""
Aggregate DPO Evaluation Results Across Seeds.

Reads the eval JSON from each DPO seed run and computes mean ± SD for:
  - Behavioral: opinion sycophancy, overall sycophancy
  - Capability: MMLU accuracy, GSM8k accuracy
  - Probe decomposition (best layer per seed): social compliance, belief
    corruption, robust tracking, other

Output: results/dpo_seed_summary.json

Usage:
    python scripts/aggregate_dpo_seeds.py
    python scripts/aggregate_dpo_seeds.py --output results/dpo_seed_summary.json
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path


# Seed → eval result file path
SEED_FILES = {
    100: "results/dpo_eval_results.json",
    200: "results/dpo_eval_seed200.json",
    300: "results/dpo_eval_seed300.json",
}


def load_seed_result(seed: int, path: str) -> dict:
    """Load and validate a single seed's eval results."""
    p = Path(path)
    if not p.exists():
        print(f"  [MISSING] Seed {seed}: {path}")
        return None
    with open(p) as f:
        data = json.load(f)
    print(f"  [OK] Seed {seed}: {path}")
    return data


def extract_metrics(data: dict, seed: int) -> dict:
    """Extract the key metrics from a single seed's eval JSON."""
    behavioral = data.get("behavioral", {})
    overall = behavioral.get("overall", {})
    per_source = behavioral.get("per_source", {})
    capabilities = data.get("capabilities", {})
    probes = data.get("probes", {})

    # Behavioral
    opinion_syc = per_source.get("anthropic_opinion", {}).get("sycophancy_rate")
    overall_syc = overall.get("sycophancy_rate")

    # Capability
    mmlu = capabilities.get("mmlu", {}).get("accuracy")
    gsm8k = capabilities.get("gsm8k", {}).get("accuracy")

    # Probe decomposition — use the best layer from the summary
    summary = probes.get("summary", {})
    best_layer = summary.get("best_layer")
    per_layer = probes.get("per_layer", {})

    if best_layer is not None and str(best_layer) in per_layer:
        layer_data = per_layer[str(best_layer)]
        social_compliance = layer_data.get("social_compliance_rate")
        belief_corruption = layer_data.get("belief_corruption_rate")
        robust_tracking = layer_data.get("robust_rate")
        other = layer_data.get("other_rate")
    else:
        social_compliance = summary.get("best_social_compliance_rate")
        belief_corruption = summary.get("best_belief_corruption_rate")
        robust_tracking = summary.get("best_robust_rate")
        other = None
        if all(v is not None for v in [social_compliance, belief_corruption, robust_tracking]):
            other = 1.0 - social_compliance - belief_corruption - robust_tracking

    return {
        "seed": seed,
        "best_probe_layer": best_layer,
        "opinion_sycophancy": opinion_syc,
        "overall_sycophancy": overall_syc,
        "mmlu_accuracy": mmlu,
        "gsm8k_accuracy": gsm8k,
        "social_compliance": social_compliance,
        "belief_corruption": belief_corruption,
        "robust_tracking": robust_tracking,
        "other": other,
    }


def compute_summary_stats(values: list) -> dict:
    """Compute mean and SD for a list of numeric values, skipping None."""
    valid = [v for v in values if v is not None]
    if not valid:
        return {"mean": None, "sd": None, "n": 0}
    arr = np.array(valid)
    return {
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "n": len(arr),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate DPO eval results across seeds"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/dpo_seed_summary.json",
        help="Output path (default: results/dpo_seed_summary.json)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("AGGREGATING DPO RESULTS ACROSS SEEDS")
    print("=" * 70)

    # Load all seed results
    seed_data = {}
    for seed, path in SEED_FILES.items():
        result = load_seed_result(seed, path)
        if result is not None:
            seed_data[seed] = result

    if not seed_data:
        print("\nERROR: No seed results found. Run DPO training/eval first.")
        return 1

    print(f"\nLoaded {len(seed_data)}/{len(SEED_FILES)} seed results")

    if len(seed_data) < len(SEED_FILES):
        missing = [s for s in SEED_FILES if s not in seed_data]
        print(f"WARNING: Missing seeds: {missing}")

    # Extract metrics per seed
    per_seed = []
    for seed in sorted(seed_data.keys()):
        metrics = extract_metrics(seed_data[seed], seed)
        per_seed.append(metrics)
        print(f"\n  Seed {seed}:")
        print(f"    Opinion syc:       {metrics['opinion_sycophancy']:.1%}" if metrics['opinion_sycophancy'] is not None else "    Opinion syc:       N/A")
        print(f"    Overall syc:       {metrics['overall_sycophancy']:.1%}" if metrics['overall_sycophancy'] is not None else "    Overall syc:       N/A")
        print(f"    MMLU:              {metrics['mmlu_accuracy']:.1%}" if metrics['mmlu_accuracy'] is not None else "    MMLU:              N/A")
        print(f"    GSM8k:             {metrics['gsm8k_accuracy']:.1%}" if metrics['gsm8k_accuracy'] is not None else "    GSM8k:             N/A")
        print(f"    Social compliance: {metrics['social_compliance']:.1%}" if metrics['social_compliance'] is not None else "    Social compliance: N/A")
        print(f"    Belief corruption: {metrics['belief_corruption']:.1%}" if metrics['belief_corruption'] is not None else "    Belief corruption: N/A")
        print(f"    Robust tracking:   {metrics['robust_tracking']:.1%}" if metrics['robust_tracking'] is not None else "    Robust tracking:   N/A")
        print(f"    Other:             {metrics['other']:.1%}" if metrics['other'] is not None else "    Other:             N/A")
        print(f"    Best probe layer:  {metrics['best_probe_layer']}")

    # Compute summary statistics
    metric_keys = [
        ("opinion_sycophancy", "Opinion Sycophancy"),
        ("overall_sycophancy", "Overall Sycophancy"),
        ("mmlu_accuracy", "MMLU Accuracy"),
        ("gsm8k_accuracy", "GSM8k Accuracy"),
        ("social_compliance", "Social Compliance"),
        ("belief_corruption", "Belief Corruption"),
        ("robust_tracking", "Robust Tracking"),
        ("other", "Other"),
    ]

    summary = {}
    print("\n" + "=" * 70)
    print("SUMMARY (Mean ± SD)")
    print("=" * 70)

    for key, label in metric_keys:
        values = [m[key] for m in per_seed]
        stats = compute_summary_stats(values)
        summary[key] = stats
        if stats["mean"] is not None:
            print(f"  {label:25s}: {stats['mean']:.4f} ± {stats['sd']:.4f}  (n={stats['n']})")
        else:
            print(f"  {label:25s}: N/A")

    # Assemble output
    output = {
        "description": "Multi-seed DPO evaluation summary (seeds 100, 200, 300)",
        "seeds": sorted(seed_data.keys()),
        "n_seeds": len(seed_data),
        "per_seed": per_seed,
        "summary": summary,
        "source_files": {seed: str(path) for seed, path in SEED_FILES.items()},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSummary saved to {out_path}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
