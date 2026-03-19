#!/usr/bin/env python3
"""
Analyze steering results checkpoint: compute per-source sycophancy rates
for every steering condition and identify best (layer, alpha) for reducing
opinion-domain sycophancy.
"""

import json
import math
import sys
from pathlib import Path

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "results" / "steering_results.json.checkpoint.json"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "results" / "steering_per_source_analysis.json"


def wilson_ci(successes: int, n: int, z: float = 1.96):
    """Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = successes / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def main():
    with open(CHECKPOINT_PATH) as f:
        data = json.load(f)

    conditions = data["conditions"]

    # Extract baseline opinion sycophancy rate
    baseline = conditions["baseline"]
    baseline_opinion = baseline["sycophancy"]["per_source"]["anthropic_opinion"]
    baseline_opinion_rate = baseline_opinion["sycophancy_rate"]
    baseline_overall = baseline["sycophancy"]["overall_sycophancy_rate"]

    print(f"Baseline opinion sycophancy rate: {baseline_opinion_rate*100:.1f}%")
    print(f"Baseline overall sycophancy rate: {baseline_overall*100:.1f}%")
    print(f"Baseline opinion N: {baseline_opinion['total']}")
    print()

    # Build rows for all conditions
    rows = []
    for cond_name, cond in conditions.items():
        syc = cond["sycophancy"]
        per_source = syc.get("per_source", {})

        opinion = per_source.get("anthropic_opinion", {})
        factual = per_source.get("truthfulqa_factual", {})
        reasoning = per_source.get("gsm8k_reasoning", {})

        opinion_rate = opinion.get("sycophancy_rate", None)
        opinion_count = opinion.get("sycophantic_count", 0)
        opinion_n = opinion.get("total", 0)
        opinion_ci = wilson_ci(opinion_count, opinion_n)

        factual_rate = factual.get("sycophancy_rate", None)
        factual_count = factual.get("sycophantic_count", 0)
        factual_n = factual.get("total", 0)

        reasoning_rate = reasoning.get("sycophancy_rate", None)
        reasoning_count = reasoning.get("sycophantic_count", 0)
        reasoning_n = reasoning.get("total", 0)

        overall_rate = syc["overall_sycophancy_rate"]

        opinion_reduction_pp = (baseline_opinion_rate - opinion_rate) * 100 if opinion_rate is not None else None

        # Get layer/alpha info
        layers = cond.get("layers", [])
        alpha = cond.get("alpha", 0.0)

        # Capability retention
        mmlu_retained = cond.get("mmlu_retained", None)
        gsm8k_retained = cond.get("gsm8k_retained", None)

        rows.append({
            "condition": cond_name,
            "layers": layers,
            "alpha": alpha,
            "opinion_syc_rate": opinion_rate,
            "opinion_syc_count": opinion_count,
            "opinion_n": opinion_n,
            "opinion_wilson_ci": list(opinion_ci),
            "opinion_reduction_pp": opinion_reduction_pp,
            "factual_syc_rate": factual_rate,
            "factual_n": factual_n,
            "reasoning_syc_rate": reasoning_rate,
            "reasoning_n": reasoning_n,
            "overall_syc_rate": overall_rate,
            "mmlu_retained": mmlu_retained,
            "gsm8k_retained": gsm8k_retained,
        })

    # Sort by opinion reduction (descending = best reduction first)
    rows.sort(key=lambda r: -(r["opinion_reduction_pp"] or -999))

    # Print table
    header = f"{'Condition':<28} {'OpnSyc%':>8} {'OpnCI95':>18} {'Reduct_pp':>10} {'FctSyc%':>8} {'RsnSyc%':>8} {'OvrlSyc%':>9} {'MMLU_ret':>9} {'GSM_ret':>9}"
    print(header)
    print("-" * len(header))

    for r in rows:
        op = f"{r['opinion_syc_rate']*100:.1f}" if r['opinion_syc_rate'] is not None else "N/A"
        ci_lo, ci_hi = r['opinion_wilson_ci']
        ci_str = f"[{ci_lo*100:.1f}, {ci_hi*100:.1f}]"
        red = f"{r['opinion_reduction_pp']:+.1f}" if r['opinion_reduction_pp'] is not None else "N/A"
        fct = f"{r['factual_syc_rate']*100:.1f}" if r['factual_syc_rate'] is not None else "N/A"
        rsn = f"{r['reasoning_syc_rate']*100:.1f}" if r['reasoning_syc_rate'] is not None else "N/A"
        ovr = f"{r['overall_syc_rate']*100:.1f}"
        mmlu = f"{r['mmlu_retained']*100:.1f}" if r['mmlu_retained'] is not None else "N/A"
        gsm = f"{r['gsm8k_retained']*100:.1f}" if r['gsm8k_retained'] is not None else "N/A"

        print(f"{r['condition']:<28} {op:>8} {ci_str:>18} {red:>10} {fct:>8} {rsn:>8} {ovr:>9} {mmlu:>9} {gsm:>9}")

    # Identify best conditions: meaningful reduction (>5pp) with reasonable capability
    print("\n" + "=" * 80)
    print("CONDITIONS WITH >5pp OPINION SYCOPHANCY REDUCTION (sorted by reduction)")
    print("=" * 80)
    meaningful = [r for r in rows if r['opinion_reduction_pp'] is not None and r['opinion_reduction_pp'] > 5]
    if not meaningful:
        print("  No conditions found with >5pp opinion sycophancy reduction.")
    else:
        for r in meaningful:
            ci_lo, ci_hi = r['opinion_wilson_ci']
            cap_note = ""
            if r['mmlu_retained'] is not None and r['mmlu_retained'] < 0.9:
                cap_note = " [MMLU degraded]"
            if r['gsm8k_retained'] is not None and r['gsm8k_retained'] < 0.9:
                cap_note += " [GSM8K degraded]"
            print(f"  {r['condition']:<26} opinion={r['opinion_syc_rate']*100:.1f}% "
                  f"CI=[{ci_lo*100:.1f},{ci_hi*100:.1f}] "
                  f"reduction={r['opinion_reduction_pp']:+.1f}pp "
                  f"overall={r['overall_syc_rate']*100:.1f}%{cap_note}")

    # Save full results
    output = {
        "baseline": {
            "opinion_sycophancy_rate": baseline_opinion_rate,
            "opinion_n": baseline_opinion["total"],
            "overall_sycophancy_rate": baseline_overall,
        },
        "conditions_sorted_by_opinion_reduction": rows,
        "meaningful_reductions": [r for r in rows if r['opinion_reduction_pp'] is not None and r['opinion_reduction_pp'] > 5],
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
