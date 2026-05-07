#!/usr/bin/env python3
"""
Bootstrap stability of the patching-derived head ranking.

Wraps scripts/03_activation_patching.py in a 5-fold bootstrap resampling
loop to answer whether the top-k sycophancy heads (L4H28, L4H5, L5H31
on Llama-3-8B) are stable or artifacts of N=100 sampling noise.

For each of N_BOOTSTRAP resamples, a fresh random subset of
N_SAMPLES prompts is drawn with a distinct seed, the full
layer-scan + head-scan patching pipeline is run, and the per-head
recovery scores + top-k rankings are recorded. We then aggregate:

    - pairwise Jaccard similarity of top-{3,5,10} head sets
    - per-head rank mean and standard deviation across resamples
    - per-head recovery mean and 95% percentile bootstrap CI

Output: results/patching_bootstrap.json
Paper slot: stability footnote in section 5.4 / appendix table.
"""

import os
import sys
import json
import argparse
import importlib.util
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

# Project root on path so "from src.models import SycophancyModel" works.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import SycophancyModel  # noqa: E402

# Import the patching script as a module. Its filename starts with a
# digit so we load it by path rather than a normal "from scripts ..."
# import.
_PATCHING_PATH = PROJECT_ROOT / "scripts" / "03_activation_patching.py"
_spec = importlib.util.spec_from_file_location("patching_module", _PATCHING_PATH)
patching = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patching)  # type: ignore[union-attr]


DEFAULT_SEEDS = [42, 123, 456, 789, 1011]
DEFAULT_N_SAMPLES = 100
DEFAULT_HEAD_TOP_K = 5  # number of critical layers to scan heads on
TOP_K_FOR_JACCARD = (3, 5, 10)
N_BOOTSTRAP_CI = 2000  # inner bootstrap for per-head recovery CI


def get_git_hash() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
        ).decode().strip()
    except Exception:
        return ""


def run_one_resample(model, data_path, seed, n_samples, max_positions, head_top_k):
    """Run layer-scan + head-scan on a seeded N-sample resample."""
    # load_data reads patching.RANDOM_SEED to pick its subset. Mutate it
    # so each bootstrap round draws a distinct sample.
    patching.RANDOM_SEED = seed
    patching.HEAD_ANALYSIS_TOP_K = head_top_k
    patching.set_seeds(seed)

    dataset = patching.load_data(data_path, n_samples)
    layers = list(range(model.n_layers))

    # --- Layer-level scan ---
    layer_results = []
    for item in dataset:
        r = patching.run_layer_patching_single(
            model, item, layers, max_positions=max_positions
        )
        if r is not None:
            layer_results.append(r)

    if len(layer_results) == 0:
        return {
            "seed": seed,
            "n_valid": 0,
            "critical_layers": [],
            "head_scores": {},
            "top_10_heads": [],
            "error": "no valid layer-patching samples",
        }

    layer_agg = patching.aggregate_patching_results(layer_results, layers)
    critical_layers = layer_agg["critical_layers"]

    # --- Head-level scan ---
    head_results = []
    for item in dataset:
        r = patching.run_head_patching_single(model, item, critical_layers)
        if r is not None:
            head_results.append(r)

    head_agg = patching.aggregate_head_results(head_results)

    return {
        "seed": seed,
        "n_valid_layer": len(layer_results),
        "n_valid_head": len(head_results),
        "critical_layers": list(map(int, critical_layers)),
        "layer_importance": layer_agg["layer_importance"],
        "head_scores": {
            name: stats["mean_recovery"]
            for name, stats in head_agg["per_head"].items()
        },
        "top_10_heads": [h["head"] for h in head_agg["top_10_heads"]],
    }


def pairwise_jaccard(head_lists, k):
    """Mean Jaccard over all unordered pairs of top-k head sets."""
    if len(head_lists) < 2:
        return None
    sets = [set(lst[:k]) for lst in head_lists if lst]
    if len(sets) < 2:
        return None
    pairs = []
    for a, b in combinations(sets, 2):
        union = a | b
        pairs.append(len(a & b) / len(union) if union else 0.0)
    return {
        "mean": float(np.mean(pairs)),
        "min": float(np.min(pairs)),
        "max": float(np.max(pairs)),
        "n_pairs": len(pairs),
    }


def per_head_stability(per_resample):
    """For every head seen across resamples, compute rank + recovery stats.

    A head absent from a resample's top-10 ranking is treated as having
    rank = 11 (sentinel beyond top-10) for that resample.
    """
    all_heads = set()
    for r in per_resample:
        all_heads.update(r.get("head_scores", {}).keys())

    stats = {}
    for head in sorted(all_heads):
        ranks = []
        recoveries = []
        appearances_in_top10 = 0
        for r in per_resample:
            scores = r.get("head_scores", {})
            if head in scores:
                recoveries.append(scores[head])
            top10 = r.get("top_10_heads", [])
            if head in top10:
                ranks.append(top10.index(head) + 1)
                appearances_in_top10 += 1
            else:
                ranks.append(11)  # sentinel for "outside top-10"

        if recoveries:
            rec_arr = np.asarray(recoveries, dtype=np.float64)
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(rec_arr), size=(N_BOOTSTRAP_CI, len(rec_arr)))
            boot_means = rec_arr[idx].mean(axis=1)
            ci_lo, ci_hi = np.quantile(boot_means, [0.025, 0.975])
            recovery_stats = {
                "mean": float(rec_arr.mean()),
                "sd": float(rec_arr.std(ddof=0)),
                "ci_95": [float(ci_lo), float(ci_hi)],
                "n_resamples_present": int(len(rec_arr)),
            }
        else:
            recovery_stats = None

        stats[head] = {
            "rank_mean": float(np.mean(ranks)),
            "rank_sd": float(np.std(ranks, ddof=0)),
            "top10_appearance_rate": appearances_in_top10 / len(per_resample),
            "recovery": recovery_stats,
        }
    return stats


def aggregate(per_resample):
    top_lists = [r.get("top_10_heads", []) for r in per_resample]
    jaccards = {
        f"top{k}": pairwise_jaccard(top_lists, k) for k in TOP_K_FOR_JACCARD
    }
    head_stats = per_head_stability(per_resample)

    # Critical-layer stability
    crit_lists = [tuple(sorted(r.get("critical_layers", []))) for r in per_resample]
    layer_counts = {}
    for lst in crit_lists:
        for layer in lst:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
    critical_layer_frequency = {
        str(layer): count / len(per_resample)
        for layer, count in sorted(layer_counts.items())
    }

    # Rank the heads that appear most consistently at the top
    top3_counts = {}
    for lst in top_lists:
        for head in lst[:3]:
            top3_counts[head] = top3_counts.get(head, 0) + 1
    top3_head_frequency = {
        head: count / len(per_resample)
        for head, count in sorted(top3_counts.items(), key=lambda x: -x[1])
    }

    return {
        "pairwise_jaccard": jaccards,
        "critical_layer_frequency": critical_layer_frequency,
        "top3_head_frequency": top3_head_frequency,
        "per_head_stats": head_stats,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", "-m",
                   default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--data", "-d",
                   default="data/processed/master_sycophancy_balanced.jsonl")
    p.add_argument("--n-bootstrap", type=int, default=len(DEFAULT_SEEDS))
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--max-positions", type=int, default=50)
    p.add_argument("--head-top-k", type=int, default=DEFAULT_HEAD_TOP_K)
    p.add_argument("--seeds", default=None,
                   help="Comma-separated seeds. Overrides --n-bootstrap.")
    p.add_argument("--output", "-o",
                   default="results/patching_bootstrap.json")
    p.add_argument("--device", default=None,
                   choices=["cpu", "cuda", "mps"])
    return p.parse_args()


def main():
    args = parse_args()
    started_at = datetime.now(timezone.utc)

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = DEFAULT_SEEDS[: args.n_bootstrap]

    print("=" * 70)
    print("PATCHING BOOTSTRAP — head-ranking stability")
    print("=" * 70)
    print(f"Model:          {args.model}")
    print(f"Data:           {args.data}")
    print(f"N bootstrap:    {len(seeds)} seeds={seeds}")
    print(f"N samples/fold: {args.n_samples}")
    print(f"Head top-K:     {args.head_top_k}")
    print(f"Git hash:       {get_git_hash() or 'N/A'}")
    print()

    model = SycophancyModel(args.model, device=args.device)

    per_resample = []
    for i, seed in enumerate(seeds, start=1):
        print(f"[{i}/{len(seeds)}] seed={seed}")
        result = run_one_resample(
            model=model,
            data_path=args.data,
            seed=seed,
            n_samples=args.n_samples,
            max_positions=args.max_positions,
            head_top_k=args.head_top_k,
        )
        print(f"    valid_layer={result.get('n_valid_layer')} "
              f"valid_head={result.get('n_valid_head')} "
              f"critical_layers={result.get('critical_layers')}")
        print(f"    top10 heads: {result.get('top_10_heads')}")
        per_resample.append(result)

    aggregate_stats = aggregate(per_resample)

    output = {
        "schema_version": "1.0",
        "metadata": {
            "model_name": args.model,
            "data_path": args.data,
            "n_bootstrap": len(seeds),
            "n_samples_per_resample": args.n_samples,
            "seeds": seeds,
            "head_top_k": args.head_top_k,
            "max_positions": args.max_positions,
            "git_hash": get_git_hash(),
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now(timezone.utc).isoformat(),
        },
        "per_resample": per_resample,
        "aggregate": aggregate_stats,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")

    # --- Manifest (matches existing manifest convention) ---
    manifest = {
        "script": "scripts/12_patching_bootstrap.py",
        "status": "completed",
        "command": " ".join(sys.argv),
        "git_hash": get_git_hash(),
        "model": args.model,
        "data": args.data,
        "seeds": seeds,
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": [str(out_path)],
    }
    manifest_dir = Path("results") / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = manifest_dir / f"patching_bootstrap_{ts}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # --- Human-readable summary ---
    print("\n" + "=" * 70)
    print("STABILITY SUMMARY")
    print("=" * 70)
    for k, v in aggregate_stats["pairwise_jaccard"].items():
        if v is not None:
            print(f"  {k} mean Jaccard: {v['mean']:.3f} "
                  f"(min {v['min']:.3f}, max {v['max']:.3f}, n_pairs={v['n_pairs']})")
    print("  Top-3 head frequency:")
    for head, freq in list(aggregate_stats["top3_head_frequency"].items())[:10]:
        print(f"    {head}: {freq:.1%}")
    print("  Critical-layer frequency:")
    for layer, freq in aggregate_stats["critical_layer_frequency"].items():
        print(f"    L{layer}: {freq:.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
