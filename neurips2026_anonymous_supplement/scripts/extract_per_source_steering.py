#!/usr/bin/env python3
"""
Extract per-source steering analysis from the existing checkpoint.

The checkpoint already contains per-source sycophancy rates for every
steering condition. This script extracts and formats the opinion-domain
results to determine whether steering reduces sycophancy in the domain
where it actually exists (opinion: 83% baseline).

No GPU required — pure offline analysis of existing checkpoint data.
"""

import json
import sys
from pathlib import Path

CHECKPOINT_PATH = "results/steering_results.json.checkpoint.json"
OUTPUT_PATH = "results/steering_per_source_analysis.json"


def main():
    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        return 1

    with open(ckpt_path) as f:
        ckpt = json.load(f)

    conditions = ckpt.get("conditions", {})
    print(f"Loaded checkpoint with {len(conditions)} conditions")
    print(f"Checkpoint state: {ckpt.get('checkpoint_state', 'unknown')}")
    print()

    # Extract baseline opinion rate
    baseline = conditions.get("baseline", {})
    baseline_syc = baseline.get("sycophancy", {})
    baseline_opinion = baseline_syc.get("per_source", {}).get("anthropic_opinion", {})
    baseline_opinion_rate = baseline_opinion.get("sycophancy_rate", 0)
    baseline_opinion_ci = baseline_opinion.get("sycophancy_rate_ci", [0, 0])
    baseline_opinion_n = baseline_opinion.get("total", 0)

    print("=" * 90)
    print("OPINION-DOMAIN STEERING ANALYSIS")
    print(f"Baseline opinion sycophancy: {baseline_opinion_rate:.1%}  "
          f"CI: [{baseline_opinion_ci[0]:.1%}, {baseline_opinion_ci[1]:.1%}]  "
          f"N={baseline_opinion_n}")
    print("=" * 90)
    print()

    # Collect all conditions with opinion-domain results
    results = []
    for key, cond in conditions.items():
        if key == "baseline":
            continue
        syc = cond.get("sycophancy", {})
        opinion = syc.get("per_source", {}).get("anthropic_opinion", {})
        if not opinion:
            continue

        opinion_rate = opinion.get("sycophancy_rate", 0)
        opinion_ci = opinion.get("sycophancy_rate_ci", [0, 0])
        opinion_n = opinion.get("total", 0)
        reduction_pp = (baseline_opinion_rate - opinion_rate) * 100

        # Check if reduction is outside baseline CI
        outside_ci = opinion_rate < baseline_opinion_ci[0]

        results.append({
            "key": key,
            "description": cond.get("description", key),
            "layers": cond.get("layers", []),
            "alpha": cond.get("alpha", 0),
            "opinion_sycophancy_rate": opinion_rate,
            "opinion_sycophancy_ci": opinion_ci,
            "opinion_n": opinion_n,
            "reduction_pp": reduction_pp,
            "outside_baseline_ci": outside_ci,
            "overall_rate": syc.get("overall_sycophancy_rate", 0),
            "factual_rate": syc.get("per_source", {}).get("truthfulqa_factual", {}).get("sycophancy_rate", 0),
            "reasoning_rate": syc.get("per_source", {}).get("gsm8k_reasoning", {}).get("sycophancy_rate", 0),
        })

    # Sort by reduction (best first)
    results.sort(key=lambda x: x["reduction_pp"], reverse=True)

    # Print table
    print(f"{'Condition':<45} {'Opinion Rate':>12} {'Reduction':>10} {'CI':>24} {'Outside CI':>11}")
    print("-" * 105)
    for r in results:
        ci_str = f"[{r['opinion_sycophancy_ci'][0]:.1%}, {r['opinion_sycophancy_ci'][1]:.1%}]"
        marker = " ***" if r["outside_baseline_ci"] else ""
        print(f"{r['description']:<45} {r['opinion_sycophancy_rate']:>11.1%} "
              f"{r['reduction_pp']:>+9.1f}pp {ci_str:>24} "
              f"{'YES' if r['outside_baseline_ci'] else 'no':>10}{marker}")

    print()
    print("=" * 90)

    # Summary: best conditions
    significant = [r for r in results if r["outside_baseline_ci"]]
    print(f"\nConditions with opinion rate BELOW baseline CI [{baseline_opinion_ci[0]:.1%}]: "
          f"{len(significant)} / {len(results)}")

    if significant:
        print("\nSignificant reductions (opinion sycophancy below baseline CI):")
        for r in significant:
            print(f"  {r['description']}: {r['opinion_sycophancy_rate']:.1%} "
                  f"({r['reduction_pp']:+.1f}pp)")
            print(f"    Factual: {r['factual_rate']:.1%}, Reasoning: {r['reasoning_rate']:.1%}")
    else:
        print("\nNO steering condition reduces opinion sycophancy below baseline CI.")
        print("Redundancy hypothesis confirmed even within high-sycophancy domain.")
        print("→ Decision: Skip Phase 2, proceed to Phase 4 (DPO).")

    # Top 5 best reductions regardless of significance
    print(f"\nTop 5 opinion-domain reductions:")
    for r in results[:5]:
        print(f"  {r['description']}: {r['opinion_sycophancy_rate']:.1%} "
              f"({r['reduction_pp']:+.1f}pp)")

    # Top 5 worst (increases)
    print(f"\nTop 5 opinion-domain increases (sycophancy got worse):")
    for r in results[-5:]:
        print(f"  {r['description']}: {r['opinion_sycophancy_rate']:.1%} "
              f"({r['reduction_pp']:+.1f}pp)")

    # Save structured output
    output = {
        "analysis": "per_source_steering_opinion_domain",
        "baseline": {
            "opinion_sycophancy_rate": baseline_opinion_rate,
            "opinion_sycophancy_ci": baseline_opinion_ci,
            "opinion_n": baseline_opinion_n,
        },
        "n_conditions": len(results),
        "n_significant": len(significant),
        "decision_gate": "proceed_to_phase2" if significant else "skip_to_phase4_dpo",
        "conditions": results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
