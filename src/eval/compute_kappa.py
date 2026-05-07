#!/usr/bin/env python3
"""
Compute Cohen's kappa (judge vs human) per rubric dimension on the audit sample.

Reads results/freeform/audit_sample.jsonl after manual scoring is complete and
writes results/freeform/agreement.json with weighted-linear kappa per dimension,
plus exact-agreement % and the 95% CI computed by bootstrap resampling.

Usage:
    python src/eval/compute_kappa.py \
        --audit results/freeform/audit_sample.jsonl \
        --out results/freeform/agreement.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

DIMENSIONS = [
    "truthfulness",
    "sycophancy",
    "hedging",
    "helpfulness",
    "pushback_resistance_turn2",
    "pushback_resistance_turn3",
]


def kappa_weight_for(dim):
    """Linear-weighted kappa for ordinal scales; unweighted for binary."""
    if dim.startswith("pushback_resistance"):
        return None  # binary -> unweighted
    return "linear"


def landis_koch(k):
    if k < 0:    return "poor (worse than chance)"
    if k < 0.20: return "slight"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate"
    if k < 0.80: return "substantial"
    return "almost perfect"


def bootstrap_ci(judge, human, dim, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    weights = kappa_weight_for(dim)
    n = len(judge)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            k = cohen_kappa_score(np.array(judge)[idx], np.array(human)[idx], weights=weights)
        except ValueError:
            continue
        boots.append(k)
    if not boots:
        return None, None
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def extract_pairs(items, dim):
    judge_vals, human_vals = [], []
    for it in items:
        j = it.get("judge_scores", {}).get(dim)
        h = it.get("manual_scores", {}).get(dim, {}).get("score")
        if j is None or h is None:
            continue
        judge_vals.append(j)
        human_vals.append(h)
    return judge_vals, human_vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    items = [json.loads(line) for line in open(args.audit) if line.strip()]
    print(f"Loaded {len(items)} audit items from {args.audit}")

    results = {"n_items": len(items), "n_boot": args.n_boot, "per_dimension": {}}

    print(f"\n{'Dimension':<32} {'N':>4}  {'kappa':>7}  {'95% CI':>18}  {'exact%':>8}  agreement")
    print("-" * 95)

    for dim in DIMENSIONS:
        judge, human = extract_pairs(items, dim)
        n = len(judge)
        if n < 2:
            print(f"{dim:<32} {n:>4}  (insufficient paired data)")
            results["per_dimension"][dim] = {"n": n, "kappa": None}
            continue

        weights = kappa_weight_for(dim)
        try:
            k = cohen_kappa_score(judge, human, weights=weights)
        except ValueError as e:
            print(f"{dim:<32} {n:>4}  ERROR: {e}")
            results["per_dimension"][dim] = {"n": n, "kappa": None, "error": str(e)}
            continue

        lo, hi = bootstrap_ci(judge, human, dim, n_boot=args.n_boot)
        exact = sum(1 for j, h in zip(judge, human) if j == h) / n

        ci_str = f"[{lo:+.2f}, {hi:+.2f}]" if lo is not None else "  (n/a)"
        print(f"{dim:<32} {n:>4}  {k:>+7.3f}  {ci_str:>18}  {exact*100:>7.1f}%  {landis_koch(k)}")
        results["per_dimension"][dim] = {
            "n": n,
            "kappa": float(k),
            "kappa_type": "linear-weighted" if weights == "linear" else "unweighted",
            "ci_95_low": lo,
            "ci_95_high": hi,
            "exact_agreement": float(exact),
            "landis_koch": landis_koch(k),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
