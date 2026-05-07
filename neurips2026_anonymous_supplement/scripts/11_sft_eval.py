#!/usr/bin/env python3
"""
SFT Baseline Evaluation Script — Thin Wrapper Around DPO Eval Pipeline.

Runs the IDENTICAL evaluation suite as 07_dpo_eval.py but loading the SFT adapter:
  1. Behavioral eval: sycophancy rates on full benchmark (per-source breakdown)
  2. Capability eval: MMLU (500 samples) + GSM8k (200 samples)
  3. Probe re-analysis: neutral-transfer probes on layers 0-5

All evaluation logic is imported from 07_dpo_eval.py — no duplication.
The only difference is the adapter path and output file.

Usage:
    python scripts/11_sft_eval.py
    python scripts/11_sft_eval.py --adapter-path results/sft_model/ --output results/sft_eval_results.json
    python scripts/11_sft_eval.py --skip-probes
"""

import sys
import os
import json
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime, timezone

import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all evaluation functions from 07_dpo_eval.py
_spec = importlib.util.spec_from_file_location(
    "dpo_eval",
    os.path.join(os.path.dirname(__file__), "07_dpo_eval.py"),
)
_eval_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_module)

# Pull out the functions we need
set_seeds = _eval_module.set_seeds
get_git_hash = _eval_module.get_git_hash
get_environment_info = _eval_module.get_environment_info
load_data = _eval_module.load_data
load_pre_dpo_baseline = _eval_module.load_pre_dpo_baseline
load_dpo_model_for_behavioral = _eval_module.load_dpo_model_for_behavioral
load_dpo_model_for_probes = _eval_module.load_dpo_model_for_probes
run_behavioral_eval = _eval_module.run_behavioral_eval
run_capability_eval = _eval_module.run_capability_eval
run_probe_analysis = _eval_module.run_probe_analysis
compute_comparison = _eval_module.compute_comparison

# Defaults — same as DPO eval except adapter/output paths
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_ADAPTER_PATH = "results/sft_model"
DEFAULT_DATA_PATH = "data/processed/master_sycophancy.jsonl"
DEFAULT_OUTPUT = "results/sft_eval_results.json"
DEFAULT_MMLU_SAMPLES = 500
DEFAULT_GSM8K_SAMPLES = 200
DEFAULT_PROBE_LAYERS = "0,1,2,3,4,5"
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SFT baseline model: behavioral + capability + mechanistic probe analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/11_sft_eval.py
  python scripts/11_sft_eval.py --adapter-path results/sft_model/ --output results/sft_eval_results.json
  python scripts/11_sft_eval.py --skip-probes
        """
    )
    parser.add_argument('--model', '-m', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--adapter-path', type=str, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument('--data', '-d', type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument('--output', '-o', type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--mmlu-samples', type=int, default=DEFAULT_MMLU_SAMPLES)
    parser.add_argument('--gsm8k-samples', type=int, default=DEFAULT_GSM8K_SAMPLES)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--skip-probes', action='store_true')
    parser.add_argument('--probe-layers', type=str, default=DEFAULT_PROBE_LAYERS)
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--n-folds', type=int, default=5)
    return parser.parse_args()


def write_manifest(output_path: str, model_name: str, adapter_path: str,
                    data_path: str, seed: int,
                    started_at: datetime, ended_at: datetime):
    manifest = {
        'script': 'scripts/11_sft_eval.py',
        'status': 'completed',
        'command': ' '.join(sys.argv),
        'git_hash': get_git_hash(),
        'model': model_name,
        'adapter_path': adapter_path,
        'data': data_path,
        'seed': seed,
        'started_at': started_at.isoformat(),
        'ended_at': ended_at.isoformat(),
        'artifacts': [output_path],
    }
    output = Path(output_path)
    manifest_dir = output.parent / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"sft_eval_{ended_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def main():
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    set_seeds(args.seed)

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    probe_layers = [int(l.strip()) for l in args.probe_layers.split(",")]

    print("=" * 80)
    print("SFT BASELINE MODEL EVALUATION")
    print("=" * 80)
    print(f"Base Model:     {args.model}")
    print(f"Adapter Path:   {args.adapter_path}")
    print(f"Data:           {args.data}")
    print(f"Output:         {args.output}")
    print(f"Device:         {device}")
    print(f"MMLU Samples:   {args.mmlu_samples}")
    print(f"GSM8k Samples:  {args.gsm8k_samples}")
    print(f"Skip Probes:    {args.skip_probes}")
    print(f"Probe Layers:   {probe_layers}")
    print(f"Seed:           {args.seed}")
    print(f"Git Hash:       {get_git_hash() or 'N/A'}")
    print()

    # Load data
    dataset = load_data(args.data)
    print(f"Loaded {len(dataset)} samples\n")

    # Load pre-DPO baselines for comparison (same baselines as DPO eval)
    pre_dpo_baseline = load_pre_dpo_baseline("results/baseline_llama3_summary.json")
    pre_dpo_probes = load_pre_dpo_baseline("results/probe_control_balanced_results.json")

    # ---- Behavioral + Capability (HF model with SFT adapter) ----
    print("Loading SFT model for behavioral + capability evaluation...")
    hf_model, tokenizer = load_dpo_model_for_behavioral(args.model, args.adapter_path, device)

    behavioral = run_behavioral_eval(hf_model, tokenizer, dataset, device)
    capabilities = run_capability_eval(hf_model, tokenizer, args.mmlu_samples,
                                        args.gsm8k_samples, args.seed, device)

    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Mechanistic Probe Analysis ----
    probes = None
    if not args.skip_probes:
        hooked_model = load_dpo_model_for_probes(args.model, args.adapter_path, device)
        probes = run_probe_analysis(hooked_model, dataset, probe_layers,
                                     seed=args.seed, n_folds=args.n_folds)
        del hooked_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Comparison to pre-training baseline ----
    comparison = compute_comparison(behavioral, capabilities, probes,
                                     pre_dpo_baseline, pre_dpo_probes)

    # ---- Assemble output ----
    ended_at = datetime.now(timezone.utc)

    output = {
        'schema_version': '1.0',
        'training_method': 'sft',
        'metadata': {
            'model_name': args.model,
            'adapter_path': args.adapter_path,
            'timestamp': ended_at.isoformat(),
            'data_path': args.data,
            'random_seed': args.seed,
            'git_hash': get_git_hash(),
            'environment': get_environment_info(),
            'runtime_seconds': (ended_at - started_at).total_seconds(),
        },
        'behavioral': behavioral,
        'capabilities': capabilities,
        'probes': probes if probes else {'skipped': True},
        'comparison': comparison,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")

    # Final summary
    print("\n" + "=" * 80)
    print("SFT BASELINE EVALUATION SUMMARY")
    print("=" * 80)

    overall = behavioral.get('overall', {})
    print(f"  Sycophancy rate (post-SFT): {overall.get('sycophancy_rate', 0):.1%}")
    print(f"  Mean compliance gap:        {overall.get('mean_compliance_gap', 0):+.4f}")

    mmlu = capabilities.get('mmlu', {})
    gsm8k = capabilities.get('gsm8k', {})
    if mmlu.get('accuracy') is not None:
        print(f"  MMLU accuracy:              {mmlu['accuracy']:.1%}")
    if gsm8k.get('accuracy') is not None:
        print(f"  GSM8k accuracy:             {gsm8k['accuracy']:.1%}")

    if probes and 'summary' in probes:
        ps = probes['summary']
        print(f"  Best probe layer:           {ps.get('best_layer')}")
        print(f"  Social compliance (best):   {ps.get('best_social_compliance_rate', 0):.1%}")
        print(f"  Belief corruption (best):   {ps.get('best_belief_corruption_rate', 0):.1%}")
        print(f"  Robust tracking (best):     {ps.get('best_robust_rate', 0):.1%}")

    print("=" * 80)

    # Write manifest
    write_manifest(
        output_path=args.output,
        model_name=args.model,
        adapter_path=args.adapter_path,
        data_path=args.data,
        seed=args.seed,
        started_at=started_at,
        ended_at=ended_at,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
