#!/usr/bin/env python3
"""
Aggregate DPO Size-Sensitivity Evaluation Results.

Reads eval JSONs for each training-set size (N=100, 200, 400, 800) and builds
a summary table with columns: N, opinion_sycophancy, overall_sycophancy,
MMLU, GSM8k.

The N=400 condition uses the original DPO eval (--reference-n400) since that
checkpoint already exists from the primary experiment (seed=100, N=400).

Usage:
    python scripts/aggregate_size_sensitivity.py \
        --dir results/dpo_size_sensitivity/ \
        --reference-n400 results/dpo_eval_results.json \
        --output results/dpo_size_sensitivity/summary.json
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path


def load_eval(path: str, label: str) -> dict | None:
    """Load and validate an eval JSON."""
    p = Path(path)
    if not p.exists():
        print(f"  [MISSING] {label}: {path}")
        return None
    with open(p) as f:
        data = json.load(f)
    print(f"  [OK] {label}: {path}")
    return data


def extract_metrics(data: dict, n: int) -> dict:
    """Extract key metrics from an eval JSON."""
    behavioral = data.get("behavioral", {})
    overall = behavioral.get("overall", {})
    per_source = behavioral.get("per_source", {})
    capabilities = data.get("capabilities", {})

    return {
        "N": n,
        "opinion_sycophancy": per_source.get("anthropic_opinion", {}).get("sycophancy_rate"),
        "overall_sycophancy": overall.get("sycophancy_rate"),
        "MMLU": capabilities.get("mmlu", {}).get("accuracy"),
        "GSM8k": capabilities.get("gsm8k", {}).get("accuracy"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate DPO size-sensitivity eval results"
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default="results/dpo_size_sensitivity",
        help="Directory containing N{size}_eval.json files",
    )
    parser.add_argument(
        "--reference-n400",
        type=str,
        default="results/dpo_eval_results.json",
        help="Path to existing N=400 eval results (default DPO run)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/dpo_size_sensitivity/summary.json",
        help="Output path for summary JSON",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("AGGREGATING DPO SIZE-SENSITIVITY RESULTS")
    print("=" * 70)

    base_dir = Path(args.dir)

    # Define sources: N -> eval path
    size_files = {
        100: str(base_dir / "N100_eval.json"),
        200: str(base_dir / "N200_eval.json"),
        400: args.reference_n400,
        800: str(base_dir / "N800_eval.json"),
    }

    # Load all eval results
    results = {}
    for n, path in sorted(size_files.items()):
        data = load_eval(path, f"N={n}")
        if data is not None:
            results[n] = data

    if not results:
        print("\nERROR: No eval results found.")
        return 1

    print(f"\nLoaded {len(results)}/{len(size_files)} conditions")

    # Extract metrics per size
    rows = []
    for n in sorted(results.keys()):
        metrics = extract_metrics(results[n], n)
        rows.append(metrics)

    # Print table
    print("\n" + "=" * 70)
    print(f"{'N':>5}  {'Opinion Syc':>12}  {'Overall Syc':>12}  {'MMLU':>8}  {'GSM8k':>8}")
    print("-" * 55)
    for row in rows:
        def fmt(v):
            return f"{v:.4f}" if v is not None else "N/A"
        print(f"{row['N']:>5}  {fmt(row['opinion_sycophancy']):>12}  "
              f"{fmt(row['overall_sycophancy']):>12}  "
              f"{fmt(row['MMLU']):>8}  {fmt(row['GSM8k']):>8}")
    print("=" * 70)

    # Compute trend statistics if we have enough data
    trend = {}
    if len(rows) >= 3:
        ns = np.array([r["N"] for r in rows if r["opinion_sycophancy"] is not None])
        syc = np.array([r["opinion_sycophancy"] for r in rows if r["opinion_sycophancy"] is not None])
        if len(ns) >= 3:
            correlation = float(np.corrcoef(ns, syc)[0, 1])
            trend["opinion_sycophancy_vs_N_correlation"] = correlation
            print(f"\nCorrelation(N, opinion_sycophancy) = {correlation:.4f}")

    # Assemble output
    output = {
        "description": "DPO size-sensitivity analysis (N=100, 200, 400, 800)",
        "sizes": sorted(results.keys()),
        "n_conditions": len(results),
        "per_size": rows,
        "trend": trend,
        "source_files": {str(n): path for n, path in size_files.items()},
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
