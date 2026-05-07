#!/usr/bin/env python3
"""
Free-form sycophancy evaluation: score aggregation and comparison.

Aggregates LLM-as-judge scores into summary statistics:
  - Per-domain means and SDs for each scoring dimension
  - Overall means and SDs
  - Composite pushback_resistance (average of turn2 and turn3)
  - Optional: baseline vs DPO comparison with bootstrap CIs

Usage:
    # Single condition summary:
    python src/eval/freeform_aggregate.py \
        --scores results/freeform/llama3_base_scores.jsonl \
        --output results/freeform/llama3_base_summary.json

    # Baseline vs DPO comparison:
    python src/eval/freeform_aggregate.py \
        --baseline results/freeform/llama3_base_scores.jsonl \
        --dpo results/freeform/llama3_dpo_scores.jsonl \
        --output results/freeform/comparison_summary.json

    # Custom bootstrap iterations:
    python src/eval/freeform_aggregate.py \
        --baseline results/freeform/llama3_base_scores.jsonl \
        --dpo results/freeform/llama3_dpo_scores.jsonl \
        --output results/freeform/comparison_summary.json \
        --n-bootstrap 10000 --seed 42
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DIMENSIONS = [
    "truthfulness",
    "sycophancy",
    "helpfulness",
]

COMPOSITE_DIMENSIONS = [
    "hedging",              # average of hedging_turn1/2/3
    "pushback_resistance",  # average of pushback_resistance_turn2/3
]


def load_scores(path: str) -> list[dict]:
    """Load scored entries from a JSONL file, skipping failures."""
    entries = []
    with open(path, "r") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("scores") is None:
                    continue  # Skip failed scoring
                entries.append(entry)
            except json.JSONDecodeError as e:
                logger.warning("Skipping line %d in %s: %s", i, path, e)
    logger.info("Loaded %d scored entries from %s", len(entries), path)
    return entries


def extract_scores_by_domain(entries: list[dict]) -> dict[str, dict[str, list[float]]]:
    """Extract score arrays organized by domain and dimension.

    Returns: {domain: {dimension: [scores]}}
    Also includes an "overall" pseudo-domain with all scores.
    """
    by_domain: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for entry in entries:
        domain = entry.get("domain", "unknown")
        scores = entry["scores"]

        def _val(v):
            """Extract numeric value from either plain int/float or {"score": N}."""
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, dict):
                return float(v["score"]) if "score" in v else None
            return None

        for dim in DIMENSIONS:
            if dim in scores:
                val = _val(scores[dim])
                if val is not None:
                    by_domain[domain][dim].append(val)
                    by_domain["overall"][dim].append(val)

        # Composite: hedging (average of per-turn scores)
        h_vals = []
        for key in ("hedging_turn1", "hedging_turn2", "hedging_turn3"):
            if key in scores:
                v = _val(scores[key])
                if v is not None:
                    h_vals.append(v)
        if h_vals:
            composite_h = sum(h_vals) / len(h_vals)
            by_domain[domain]["hedging"].append(composite_h)
            by_domain["overall"]["hedging"].append(composite_h)

        # Composite: pushback_resistance (average of per-turn scores)
        p_vals = []
        for key in ("pushback_resistance_turn2", "pushback_resistance_turn3"):
            if key in scores:
                v = _val(scores[key])
                if v is not None:
                    p_vals.append(v)
        if p_vals:
            composite_p = sum(p_vals) / len(p_vals)
            by_domain[domain]["pushback_resistance"].append(composite_p)
            by_domain["overall"]["pushback_resistance"].append(composite_p)

    return dict(by_domain)


def compute_summary(score_arrays: dict[str, list[float]]) -> dict:
    """Compute mean, SD, N for each dimension."""
    summary = {}
    for dim, values in score_arrays.items():
        arr = np.array(values)
        summary[dim] = {
            "mean": round(float(np.mean(arr)), 3),
            "sd": round(float(np.std(arr, ddof=1)), 3) if len(arr) > 1 else 0.0,
            "n": len(arr),
        }
    return summary


def compute_comparison(
    baseline_arrays: dict[str, list[float]],
    dpo_arrays: dict[str, list[float]],
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict:
    """Compute deltas between baseline and DPO with bootstrap CIs."""
    rng = np.random.RandomState(seed)

    all_dims = set(baseline_arrays.keys()) | set(dpo_arrays.keys())
    comparison = {}

    for dim in sorted(all_dims):
        base_vals = np.array(baseline_arrays.get(dim, []))
        dpo_vals = np.array(dpo_arrays.get(dim, []))

        if len(base_vals) == 0 or len(dpo_vals) == 0:
            comparison[dim] = {
                "baseline_mean": round(float(np.mean(base_vals)), 3) if len(base_vals) > 0 else None,
                "dpo_mean": round(float(np.mean(dpo_vals)), 3) if len(dpo_vals) > 0 else None,
                "delta": None,
                "ci_95": None,
                "note": "Insufficient data for comparison",
            }
            continue

        base_mean = float(np.mean(base_vals))
        dpo_mean = float(np.mean(dpo_vals))
        delta = dpo_mean - base_mean

        # Bootstrap CI for the difference in means
        boot_deltas = []
        for _ in range(n_bootstrap):
            b_sample = rng.choice(base_vals, size=len(base_vals), replace=True)
            d_sample = rng.choice(dpo_vals, size=len(dpo_vals), replace=True)
            boot_deltas.append(float(np.mean(d_sample) - np.mean(b_sample)))

        boot_deltas = np.array(boot_deltas)
        ci_low = float(np.percentile(boot_deltas, 2.5))
        ci_high = float(np.percentile(boot_deltas, 97.5))

        comparison[dim] = {
            "baseline_mean": round(base_mean, 3),
            "dpo_mean": round(dpo_mean, 3),
            "delta": round(delta, 3),
            "ci_95": [round(ci_low, 3), round(ci_high, 3)],
            "baseline_n": len(base_vals),
            "dpo_n": len(dpo_vals),
        }

    return comparison


def build_single_summary(entries: list[dict]) -> dict:
    """Build a complete summary for a single condition."""
    by_domain = extract_scores_by_domain(entries)

    result = {
        "n_conversations": len(entries),
        "domains": {},
    }

    for domain in sorted(by_domain.keys()):
        result["domains"][domain] = compute_summary(by_domain[domain])

    return result


def build_comparison_summary(
    baseline_entries: list[dict],
    dpo_entries: list[dict],
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict:
    """Build a comparison summary between baseline and DPO conditions."""
    base_by_domain = extract_scores_by_domain(baseline_entries)
    dpo_by_domain = extract_scores_by_domain(dpo_entries)

    all_domains = sorted(set(base_by_domain.keys()) | set(dpo_by_domain.keys()))

    result = {
        "baseline_n": len(baseline_entries),
        "dpo_n": len(dpo_entries),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "domains": {},
    }

    for domain in all_domains:
        base_arrays = base_by_domain.get(domain, {})
        dpo_arrays = dpo_by_domain.get(domain, {})

        result["domains"][domain] = {
            "baseline": compute_summary(base_arrays),
            "dpo": compute_summary(dpo_arrays),
            "comparison": compute_comparison(
                base_arrays, dpo_arrays, n_bootstrap, seed
            ),
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate free-form sycophancy judge scores into summary statistics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Either --scores (single) or --baseline + --dpo (comparison)
    parser.add_argument(
        "--scores", default=None,
        help="Path to scores JSONL for single-condition summary",
    )
    parser.add_argument(
        "--baseline", default=None,
        help="Path to baseline scores JSONL for comparison",
    )
    parser.add_argument(
        "--dpo", default=None,
        help="Path to DPO scores JSONL for comparison",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON file for summary statistics",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=5000,
        help="Number of bootstrap iterations for CIs (comparison mode only)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for bootstrap",
    )

    args = parser.parse_args()

    # Validate args
    if args.scores and (args.baseline or args.dpo):
        logger.error("Use either --scores OR (--baseline + --dpo), not both")
        sys.exit(1)
    if not args.scores and not (args.baseline and args.dpo):
        logger.error("Provide either --scores or both --baseline and --dpo")
        sys.exit(1)

    if args.scores:
        # Single condition
        entries = load_scores(args.scores)
        if not entries:
            logger.error("No valid scores found")
            sys.exit(1)
        result = build_single_summary(entries)
        result["mode"] = "single"
    else:
        # Comparison
        baseline_entries = load_scores(args.baseline)
        dpo_entries = load_scores(args.dpo)
        if not baseline_entries or not dpo_entries:
            logger.error("Need valid scores in both baseline and DPO files")
            sys.exit(1)
        result = build_comparison_summary(
            baseline_entries, dpo_entries,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        result["mode"] = "comparison"

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Summary written to %s", args.output)

    # Print a quick summary to stdout
    if args.scores:
        overall = result["domains"].get("overall", {})
        print("\n=== Single Condition Summary ===")
        print(f"N = {result['n_conversations']}")
        for dim in DIMENSIONS + COMPOSITE_DIMENSIONS:
            stats = overall.get(dim, {})
            if stats:
                print(f"  {dim}: {stats['mean']:.3f} ± {stats['sd']:.3f}")
    else:
        overall = result["domains"].get("overall", {})
        print("\n=== Comparison Summary (DPO - Baseline) ===")
        print(f"Baseline N = {result['baseline_n']}, DPO N = {result['dpo_n']}")
        comp = overall.get("comparison", {})
        for dim in DIMENSIONS + COMPOSITE_DIMENSIONS:
            stats = comp.get(dim, {})
            if stats and stats.get("delta") is not None:
                ci = stats.get("ci_95", [None, None])
                print(
                    f"  {dim}: Δ = {stats['delta']:+.3f} "
                    f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
                )


if __name__ == "__main__":
    main()
