#!/usr/bin/env python3
"""
Targeted capability evaluation for specific steering conditions.

Loads the model, computes steering vectors, then evaluates MMLU and GSM8k
for specified (layer, alpha) pairs. Appends results to the existing
steering_results.json and updates the checkpoint.

Usage:
    python scripts/eval_steering_capability.py \
        --conditions "layer15_alpha2.0,layer20_alpha2.0" \
        --mmlu-samples 500 --gsm8k-samples 200
"""

import json
import sys
import os
import random
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import SycophancyModel

import importlib.util
spec = importlib.util.spec_from_file_location(
    "steering", os.path.join(os.path.dirname(__file__), "05_representation_steering.py")
)
steering_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(steering_mod)


def main():
    parser = argparse.ArgumentParser(description="Targeted capability eval for steering conditions")
    parser.add_argument("--conditions", type=str, required=True,
                        help="Comma-separated condition keys, e.g. layer15_alpha2.0,layer20_alpha2.0")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data", type=str, default="data/processed/master_sycophancy.jsonl")
    parser.add_argument("--steering-results", type=str, default="results/steering_results.json")
    parser.add_argument("--checkpoint", type=str, default="results/steering_results.json.checkpoint.json")
    parser.add_argument("--mmlu-samples", type=int, default=500)
    parser.add_argument("--gsm8k-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    condition_keys = [k.strip() for k in args.conditions.split(",")]

    # Load existing results
    with open(args.steering_results) as f:
        results = json.load(f)
    conditions = results["conditions"]

    # Validate requested conditions exist
    for key in condition_keys:
        if key not in conditions:
            print(f"ERROR: condition '{key}' not found in {args.steering_results}")
            print(f"Available: {sorted(conditions.keys())}")
            return 1

    # Check which conditions already have capability data
    needed = []
    for key in condition_keys:
        cond = conditions[key]
        if "mmlu" in cond and "gsm8k" in cond:
            print(f"SKIP: {key} already has MMLU ({cond['mmlu']['accuracy']:.1%}) "
                  f"and GSM8k ({cond['gsm8k']['accuracy']:.1%})")
        else:
            needed.append(key)

    if not needed:
        print("All requested conditions already have capability data. Nothing to do.")
        return 0

    print(f"Need capability eval for: {needed}")
    print()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load data and model
    dataset = steering_mod.load_data(args.data)
    print(f"Loaded {len(dataset)} samples")

    model = SycophancyModel(args.model)

    # Compute steering vectors (needed for hooks)
    layers = results["metadata"]["layers_tested"]
    shuffled = list(dataset)
    random.Random(args.seed).shuffle(shuffled)
    n_steering = results["metadata"]["n_steering_samples"]
    steering_data = shuffled[:n_steering]

    print(f"Computing steering vectors for layers {layers} using {n_steering} samples...")
    steering_vectors, vector_norms = steering_mod.compute_steering_vectors(
        model, steering_data, layers=layers, n_samples=n_steering
    )

    # Evaluate each condition
    baseline_cond = conditions["baseline"]

    for key in needed:
        cond = conditions[key]
        cond_layers = cond["layers"]
        alpha = cond["alpha"]

        hooks = steering_mod.build_steering_hooks(
            steering_vectors, cond_layers, alpha
        ) if cond_layers else None

        print(f"\n{'='*70}")
        print(f"Capability eval: {cond['description']}")
        print(f"  Layers: {cond_layers}, Alpha: {alpha}")
        print(f"  Opinion sycophancy: {cond['sycophancy']['per_source']['anthropic_opinion']['sycophancy_rate']:.1%}")
        print(f"{'='*70}")

        # MMLU
        print(f"\nEvaluating MMLU ({args.mmlu_samples} samples)...")
        mmlu_result = steering_mod.evaluate_mmlu_with_steering(
            model=model,
            n_samples=args.mmlu_samples,
            hooks=hooks,
            seed=args.seed,
        )
        cond["mmlu"] = mmlu_result
        print(f"  MMLU accuracy: {mmlu_result['accuracy']:.1%}")

        # GSM8k
        print(f"Evaluating GSM8k ({args.gsm8k_samples} samples)...")
        gsm8k_result = steering_mod.evaluate_gsm8k_with_steering(
            model=model,
            n_samples=args.gsm8k_samples,
            hooks=hooks,
            seed=args.seed,
        )
        cond["gsm8k"] = gsm8k_result
        print(f"  GSM8k accuracy: {gsm8k_result['accuracy']:.1%}")

        # Compute retention metrics
        baseline_mmlu = baseline_cond.get("mmlu", {})
        baseline_gsm8k = baseline_cond.get("gsm8k", {})

        if baseline_mmlu:
            base_acc = baseline_mmlu.get("accuracy", 0)
            cond_acc = mmlu_result.get("accuracy", 0)
            cond["mmlu_retained"] = float(cond_acc / base_acc) if base_acc > 0 else 0.0
            base_flags, cond_flags = steering_mod._paired_flags(baseline_mmlu, mmlu_result)
            cond["mmlu_retained_ci"] = steering_mod.bootstrap_retention_ci(
                base_flags, cond_flags, seed=args.seed + 1
            )
            print(f"  MMLU retained: {cond['mmlu_retained']:.1%}")

        if baseline_gsm8k:
            base_acc = baseline_gsm8k.get("accuracy", 0)
            cond_acc = gsm8k_result.get("accuracy", 0)
            cond["gsm8k_retained"] = float(cond_acc / base_acc) if base_acc > 0 else 0.0
            base_flags, cond_flags = steering_mod._paired_flags(baseline_gsm8k, gsm8k_result)
            cond["gsm8k_retained_ci"] = steering_mod.bootstrap_retention_ci(
                base_flags, cond_flags, seed=args.seed + 2
            )
            print(f"  GSM8k retained: {cond['gsm8k_retained']:.1%}")

        # Drop per_example arrays to keep compact
        if "per_example" in cond.get("mmlu", {}):
            cond["mmlu"].pop("per_example", None)
        if "per_example" in cond.get("gsm8k", {}):
            cond["gsm8k"].pop("per_example", None)

    # Save updated results
    results["metadata"]["capability_eval_updated"] = datetime.now(timezone.utc).isoformat()
    with open(args.steering_results, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nUpdated: {args.steering_results}")

    # Update checkpoint too
    if Path(args.checkpoint).exists():
        with open(args.checkpoint) as f:
            ckpt = json.load(f)
        for key in needed:
            if key in ckpt.get("conditions", {}):
                ckpt["conditions"][key] = conditions[key]
        with open(args.checkpoint, "w") as f:
            json.dump(ckpt, f, indent=2)
        print(f"Updated: {args.checkpoint}")

    # Print summary
    print(f"\n{'='*70}")
    print("CAPABILITY EVAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Condition':<35} {'Opinion':>8} {'MMLU':>8} {'GSM8k':>8} {'MMLU Ret':>10} {'GSM8k Ret':>10}")
    print("-" * 85)
    for key in condition_keys:
        cond = conditions[key]
        op = cond["sycophancy"]["per_source"]["anthropic_opinion"]["sycophancy_rate"]
        mmlu = cond.get("mmlu", {}).get("accuracy", 0)
        gsm = cond.get("gsm8k", {}).get("accuracy", 0)
        mr = cond.get("mmlu_retained", 0)
        gr = cond.get("gsm8k_retained", 0)
        print(f"{cond['description']:<35} {op:>7.1%} {mmlu:>7.1%} {gsm:>7.1%} {mr:>9.1%} {gr:>9.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
