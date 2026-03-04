#!/usr/bin/env python3
"""
Probe-Control runner aligned to 02_train_probes neutral-transfer semantics.

This script preserves the dedicated probe-control entrypoint while delegating
core logic to the unified probe pipeline implementation.
"""

import sys
import json
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime, timezone

PROBE_SCRIPT_PATH = Path(__file__).parent / "02_train_probes.py"
DEFAULT_OUTPUT = "results/probe_control_results.json"


def load_probe_module():
    spec = importlib.util.spec_from_file_location("train_probes_v2", PROBE_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe-control run (neutral_transfer) using unified probe pipeline."
    )
    parser.add_argument("--model", "-m", type=str, default="gpt2-medium",
                        help="HuggingFace model name")
    parser.add_argument("--data", "-d", type=str, default="data/processed/master_sycophancy.jsonl",
                        help="Path to JSONL dataset")
    parser.add_argument("--max-samples", "-n", type=int, default=None,
                        help="Maximum samples to evaluate")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices, or 'all'")
    parser.add_argument("--probe-position", type=str, default="final",
                        choices=["final", "answer_token", "both"],
                        help="Token position for probe")
    parser.add_argument("--probe-type", type=str, default="logistic",
                        choices=["logistic", "ridge"],
                        help="Probe architecture")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for activation extraction")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output JSON path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda", "mps"],
                        help="Device (default: auto-detect)")
    return parser.parse_args()


def write_manifest(
    probe_module,
    output_path: str,
    model_name: str,
    data_path: str,
    seed: int,
    started_at: datetime,
    ended_at: datetime,
) -> None:
    manifest = {
        "script": "scripts/02b_probe_control.py",
        "status": "completed",
        "command": " ".join(sys.argv),
        "git_hash": probe_module.get_git_hash(),
        "model": model_name,
        "data": data_path,
        "seed": seed,
        "analysis_mode": "neutral_transfer",
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "artifacts": [output_path],
    }

    output = Path(output_path)
    manifest_dir = output.parent / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"probe_control_{ended_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    probe_module = load_probe_module()
    probe_module.set_seeds(args.seed)

    print("=" * 80)
    print("PROBE CONTROL (UNIFIED PIPELINE)")
    print("=" * 80)
    print(f"Model:          {args.model}")
    print(f"Data:           {args.data}")
    print(f"Max Samples:    {args.max_samples or 'all'}")
    print(f"Probe Position: {args.probe_position}")
    print(f"Probe Type:     {args.probe_type}")
    print("Analysis Mode:  neutral_transfer")
    print(f"CV Folds:       {args.n_folds}")
    print(f"Seed:           {args.seed}")
    print(f"Git Hash:       {probe_module.get_git_hash() or 'N/A'}")
    print()

    dataset = probe_module.load_data(args.data, args.max_samples)
    print(f"Loaded {len(dataset)} samples\n")

    model = probe_module.SycophancyModel(args.model, device=args.device)
    if args.layers is None or args.layers == "all":
        layers = list(range(model.n_layers))
    else:
        layers = [int(l.strip()) for l in args.layers.split(",") if l.strip()]
    print(f"Probing {len(layers)} layers: {layers[0]}..{layers[-1]}\n")

    results = probe_module.run_probe_pipeline(
        model=model,
        dataset=dataset,
        layers=layers,
        probe_position=args.probe_position,
        probe_type=args.probe_type,
        analysis_mode="neutral_transfer",
        batch_size=args.batch_size,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    if "error" in results:
        print(f"Error: {results['error']}")
        return 1

    output = {
        "metadata": {
            "model_name": args.model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_path": args.data,
            "probe_position": args.probe_position,
            "probe_type": args.probe_type,
            "analysis_mode": "neutral_transfer",
            "script_variant": "probe_control",
            "n_folds": args.n_folds,
            "random_seed": args.seed,
            "git_hash": probe_module.get_git_hash(),
            "environment": probe_module.get_environment_info(),
        },
        **results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")

    for pos in output["probe_positions"]:
        s = output["per_position_summary"][pos]
        print(f"\n--- Position: {pos} ---")
        print(f"Best layer: {s['best_layer']}")
        print(f"Best transfer accuracy: {s['best_biased_transfer_accuracy']:.1%}")
        ci = s["best_biased_transfer_accuracy_ci"]
        print(f"Transfer 95% CI: [{ci[0]:.1%}, {ci[1]:.1%}]")
        print(f"Social compliance: {s['best_social_compliance_rate']:.1%}")
        print(f"Belief corruption: {s['best_belief_corruption_rate']:.1%}")
        print(f"Dominant pattern: {s['dominant_pattern_best_layer']}")

    ended_at = datetime.now(timezone.utc)
    write_manifest(
        probe_module=probe_module,
        output_path=args.output,
        model_name=args.model,
        data_path=args.data,
        seed=args.seed,
        started_at=started_at,
        ended_at=ended_at,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
