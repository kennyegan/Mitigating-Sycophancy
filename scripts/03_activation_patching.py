#!/usr/bin/env python3
"""
Causal Activation Patching for Sycophancy Circuit Discovery

Phase 4.2-4.3: Identify the causal circuit responsible for sycophancy via
activation patching (also called causal tracing).

Method:
    1. Clean run:    Forward pass on neutral prompt → cache activations
    2. Corrupted run: Forward pass on biased prompt → sycophantic output
    3. Patching:     For each (layer L, token position T), substitute corrupted
                     activations with clean activations → measure output recovery

    Recovery metric:
        log P(honest | patched) - log P(honest | corrupted)
        Positive = patching at (L, T) restores truthful behavior

    After identifying critical layers, do head-level patching to pinpoint
    specific "sycophancy heads."

Usage:
    python scripts/03_activation_patching.py --model gpt2 --max-samples 10
    python scripts/03_activation_patching.py --model meta-llama/Meta-Llama-3-8B-Instruct --max-samples 100

Outputs:
    - results/patching_heatmap.json: Layer × position recovery scores
    - results/head_importance.json: Per-head effect scores at critical layers
"""

import sys
import os
import json
import math
import argparse
import subprocess
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime, timezone
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from functools import partial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import SycophancyModel


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DATA_PATH = "data/processed/master_sycophancy.jsonl"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_MODEL_NAME = "gpt2-medium"
RANDOM_SEED = 42

# How many top layers to analyze at head level
HEAD_ANALYSIS_TOP_K = 5


# =============================================================================
# Reproducibility Utilities
# =============================================================================

def get_git_hash() -> Optional[str]:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return None


def get_environment_info() -> Dict:
    return {
        'python_version': sys.version.split()[0],
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def set_seeds(seed: int = RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Data Loading
# =============================================================================

def load_data(data_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data not found at {data_path}. "
            "Run 'make data' first to download the multi-dataset benchmark."
        )

    with open(data_path, 'r') as f:
        dataset = [json.loads(line) for line in f]

    if max_samples is not None:
        # Random sample for representativeness
        rng = np.random.RandomState(RANDOM_SEED)
        indices = rng.choice(len(dataset), size=min(max_samples, len(dataset)), replace=False)
        dataset = [dataset[i] for i in sorted(indices)]

    return dataset


# =============================================================================
# Core Patching Functions
# =============================================================================

def compute_honest_log_prob(
    model_instance,
    prompt_tokens: torch.Tensor,
    honest_tokens: torch.Tensor,
    prompt_len: int,
) -> float:
    """Compute log P(honest_target | prompt) using the model."""
    seq = torch.cat([prompt_tokens, honest_tokens], dim=1)
    logits = model_instance(seq)
    log_probs = F.log_softmax(logits[0], dim=-1)
    total = sum(
        log_probs[prompt_len - 1 + i, honest_tokens[0, i].item()].item()
        for i in range(honest_tokens.shape[1])
    )
    return total


def patch_residual_stream(
    model: SycophancyModel,
    biased_prompt_tokens: torch.Tensor,
    clean_cache: Dict[str, torch.Tensor],
    layer: int,
    position: int,
    honest_tokens: torch.Tensor,
    biased_prompt_len: int,
) -> float:
    """
    Run biased prompt but patch in clean activation at (layer, position).

    Returns log P(honest | patched).
    """
    _model = model.model
    hook_name = f"blocks.{layer}.hook_resid_post"

    def patch_hook(activation, hook, clean_act, pos):
        activation[0, pos, :] = clean_act[0, pos, :]
        return activation

    clean_act = clean_cache[hook_name]

    seq = torch.cat([biased_prompt_tokens, honest_tokens], dim=1)

    hook_fn = partial(patch_hook, clean_act=clean_act, pos=position)

    with _model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        logits = _model(seq)

    log_probs = F.log_softmax(logits[0], dim=-1)
    total = sum(
        log_probs[biased_prompt_len - 1 + i, honest_tokens[0, i].item()].item()
        for i in range(honest_tokens.shape[1])
    )
    return total


def patch_attention_head(
    model: SycophancyModel,
    biased_prompt_tokens: torch.Tensor,
    clean_cache: Dict[str, torch.Tensor],
    layer: int,
    head: int,
    honest_tokens: torch.Tensor,
    biased_prompt_len: int,
) -> float:
    """
    Run biased prompt but patch in clean attention output for a specific head.

    Returns log P(honest | patched).
    """
    _model = model.model
    hook_name = f"blocks.{layer}.attn.hook_result"
    d_head = _model.cfg.d_head

    def head_patch_hook(activation, hook, clean_act, h):
        activation[0, :, h, :] = clean_act[0, :, h, :]
        return activation

    clean_act = clean_cache[hook_name]

    seq = torch.cat([biased_prompt_tokens, honest_tokens], dim=1)

    hook_fn = partial(head_patch_hook, clean_act=clean_act, h=head)

    with _model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        logits = _model(seq)

    log_probs = F.log_softmax(logits[0], dim=-1)
    total = sum(
        log_probs[biased_prompt_len - 1 + i, honest_tokens[0, i].item()].item()
        for i in range(honest_tokens.shape[1])
    )
    return total


# =============================================================================
# Layer-Level Patching Pipeline
# =============================================================================

@torch.no_grad()
def run_layer_patching_single(
    model: SycophancyModel,
    item: Dict,
    layers: List[int],
    max_positions: int = 50,
) -> Optional[Dict]:
    """
    Run layer × position patching for a single sample.

    Returns a dict with recovery scores, or None if the sample is invalid.
    """
    _model = model.model

    neutral_prompt = item.get('neutral_prompt')
    biased_prompt = item.get('biased_prompt')
    honest_target = item.get('honest_target', item.get('non_sycophantic_target'))
    syc_target = item.get('sycophantic_target')

    if not all([neutral_prompt, biased_prompt, honest_target, syc_target]):
        return None

    try:
        honest_tokens = _model.to_tokens(honest_target, prepend_bos=False)
        neutral_tokens = _model.to_tokens(neutral_prompt, prepend_bos=True)
        biased_tokens = _model.to_tokens(biased_prompt, prepend_bos=True)
    except Exception:
        return None

    if honest_tokens.shape[1] == 0:
        return None

    neutral_len = neutral_tokens.shape[1]
    biased_len = biased_tokens.shape[1]

    # Clean run: cache all activations from neutral prompt
    hook_names_resid = [f"blocks.{l}.hook_resid_post" for l in layers]

    neutral_seq = torch.cat([neutral_tokens, honest_tokens], dim=1)
    _, clean_cache = _model.run_with_cache(
        neutral_seq,
        names_filter=lambda name: name in hook_names_resid
    )

    # Corrupted baseline: log P(honest | biased, no patching)
    corrupted_log_prob = compute_honest_log_prob(
        _model, biased_tokens, honest_tokens, biased_len
    )

    # Clean baseline: log P(honest | neutral, no patching)
    clean_log_prob = compute_honest_log_prob(
        _model, neutral_tokens, honest_tokens, neutral_len
    )

    # If model is already honest on biased prompt, skip
    # (no sycophancy to patch)
    if corrupted_log_prob >= clean_log_prob - 0.1:
        return None

    total_effect = clean_log_prob - corrupted_log_prob

    # Determine token positions to patch
    # Use the biased prompt length (we patch positions in the biased sequence)
    n_positions = min(biased_len, max_positions)
    # Sample positions evenly if too many
    if biased_len > max_positions:
        positions = np.linspace(0, biased_len - 1, max_positions, dtype=int).tolist()
    else:
        positions = list(range(biased_len))

    # Patch each (layer, position) and measure recovery
    recovery_matrix = np.zeros((len(layers), len(positions)))

    for li, layer in enumerate(layers):
        for pi, pos in enumerate(positions):
            # Need clean activation at this position
            # If position exceeds neutral sequence length, skip
            hook_key = f"blocks.{layer}.hook_resid_post"
            if pos >= clean_cache[hook_key].shape[1]:
                recovery_matrix[li, pi] = 0.0
                continue

            patched_log_prob = patch_residual_stream(
                model, biased_tokens, dict(clean_cache),
                layer, pos, honest_tokens, biased_len
            )

            # Recovery: how much patching restores honest probability
            recovery = patched_log_prob - corrupted_log_prob
            recovery_matrix[li, pi] = recovery

    return {
        'recovery_matrix': recovery_matrix,
        'total_effect': total_effect,
        'clean_log_prob': clean_log_prob,
        'corrupted_log_prob': corrupted_log_prob,
        'n_layers': len(layers),
        'positions': positions,
        'biased_prompt_len': biased_len,
        'source': item.get('metadata', {}).get('source', 'unknown'),
    }


# =============================================================================
# Head-Level Patching Pipeline
# =============================================================================

@torch.no_grad()
def run_head_patching_single(
    model: SycophancyModel,
    item: Dict,
    critical_layers: List[int],
) -> Optional[Dict]:
    """
    Run head-level patching within critical layers for a single sample.

    Returns dict with per-head recovery scores.
    """
    _model = model.model

    neutral_prompt = item.get('neutral_prompt')
    biased_prompt = item.get('biased_prompt')
    honest_target = item.get('honest_target', item.get('non_sycophantic_target'))

    if not all([neutral_prompt, biased_prompt, honest_target]):
        return None

    try:
        honest_tokens = _model.to_tokens(honest_target, prepend_bos=False)
        neutral_tokens = _model.to_tokens(neutral_prompt, prepend_bos=True)
        biased_tokens = _model.to_tokens(biased_prompt, prepend_bos=True)
    except Exception:
        return None

    if honest_tokens.shape[1] == 0:
        return None

    neutral_len = neutral_tokens.shape[1]
    biased_len = biased_tokens.shape[1]

    # Cache clean activations (need hook_result for head-level patching)
    hook_names = [f"blocks.{l}.attn.hook_result" for l in critical_layers]

    neutral_seq = torch.cat([neutral_tokens, honest_tokens], dim=1)
    _, clean_cache = _model.run_with_cache(
        neutral_seq,
        names_filter=lambda name: name in hook_names
    )

    # Corrupted baseline
    corrupted_log_prob = compute_honest_log_prob(
        _model, biased_tokens, honest_tokens, biased_len
    )

    n_heads = model.n_heads
    head_scores = {}

    for layer in critical_layers:
        for head in range(n_heads):
            patched_log_prob = patch_attention_head(
                model, biased_tokens, dict(clean_cache),
                layer, head, honest_tokens, biased_len
            )
            recovery = patched_log_prob - corrupted_log_prob
            head_scores[f"L{layer}H{head}"] = recovery

    return {
        'head_scores': head_scores,
        'corrupted_log_prob': corrupted_log_prob,
        'source': item.get('metadata', {}).get('source', 'unknown'),
    }


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_patching_results(
    results: List[Dict],
    layers: List[int],
) -> Dict:
    """Aggregate layer-level patching across samples."""
    # Collect recovery matrices (they may have different position counts)
    # Normalize each to [0, n_layers] × [0, 1] grid

    n_layers = len(layers)

    # Use max positions across all results for the heatmap
    all_recovery = []
    total_effects = []

    for r in results:
        # Normalize recovery by total effect
        if r['total_effect'] > 0:
            normalized = r['recovery_matrix'] / r['total_effect']
        else:
            normalized = r['recovery_matrix']
        all_recovery.append(normalized)
        total_effects.append(r['total_effect'])

    # Average across samples (pad shorter position arrays)
    max_pos = max(m.shape[1] for m in all_recovery)
    padded = np.zeros((len(all_recovery), n_layers, max_pos))
    counts = np.zeros((n_layers, max_pos))

    for i, m in enumerate(all_recovery):
        n_pos = m.shape[1]
        padded[i, :, :n_pos] = m
        counts[:, :n_pos] += 1

    # Avoid division by zero
    counts = np.maximum(counts, 1)
    mean_recovery = padded.sum(axis=0) / counts

    # Per-layer importance: sum recovery across positions
    layer_importance = mean_recovery.sum(axis=1)

    # Find critical layers (top-k by importance)
    top_indices = np.argsort(layer_importance)[::-1][:HEAD_ANALYSIS_TOP_K]
    critical_layers = [layers[i] for i in top_indices]

    return {
        'mean_recovery_heatmap': mean_recovery.tolist(),
        'layer_importance': {str(layers[i]): float(layer_importance[i]) for i in range(n_layers)},
        'critical_layers': critical_layers,
        'n_samples': len(results),
        'mean_total_effect': float(np.mean(total_effects)),
        'std_total_effect': float(np.std(total_effects)),
    }


def aggregate_head_results(
    results: List[Dict],
) -> Dict:
    """Aggregate head-level patching across samples."""
    all_heads = set()
    for r in results:
        all_heads.update(r['head_scores'].keys())

    head_scores = {}
    for head_name in sorted(all_heads):
        scores = [r['head_scores'].get(head_name, 0.0) for r in results]
        head_scores[head_name] = {
            'mean_recovery': float(np.mean(scores)),
            'std_recovery': float(np.std(scores)),
            'n_samples': len([s for s in scores if s != 0.0]),
        }

    # Rank heads
    ranked = sorted(
        head_scores.items(),
        key=lambda x: x[1]['mean_recovery'],
        reverse=True,
    )

    return {
        'per_head': head_scores,
        'top_10_heads': [
            {'head': name, **scores}
            for name, scores in ranked[:10]
        ],
        'n_samples': len(results),
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal activation patching for sycophancy circuit discovery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/03_activation_patching.py --model gpt2 --max-samples 10
  python scripts/03_activation_patching.py --model meta-llama/Meta-Llama-3-8B-Instruct --max-samples 100
        """
    )

    parser.add_argument('--model', '-m', type=str, default=DEFAULT_MODEL_NAME,
                        help=f"HuggingFace model name (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument('--data', '-d', type=str, default=DEFAULT_DATA_PATH,
                        help=f"Path to JSONL dataset (default: {DEFAULT_DATA_PATH})")
    parser.add_argument('--max-samples', '-n', type=int, default=100,
                        help="Maximum samples for patching (default: 100)")
    parser.add_argument('--layers', type=str, default=None,
                        help="Comma-separated layer indices, or 'all' (default: all)")
    parser.add_argument('--max-positions', type=int, default=50,
                        help="Max token positions to patch per sample (default: 50)")
    parser.add_argument('--head-top-k', type=int, default=HEAD_ANALYSIS_TOP_K,
                        help=f"Number of top layers for head analysis (default: {HEAD_ANALYSIS_TOP_K})")
    parser.add_argument('--skip-head-analysis', action='store_true',
                        help="Skip head-level patching (faster, layer-only)")
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f"Random seed (default: {RANDOM_SEED})")
    parser.add_argument('--output-dir', '-o', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda', 'mps'],
                        help="Device (default: auto-detect)")

    return parser.parse_args()


def main():
    args = parse_args()

    global RANDOM_SEED, HEAD_ANALYSIS_TOP_K
    RANDOM_SEED = args.seed
    HEAD_ANALYSIS_TOP_K = args.head_top_k
    set_seeds(args.seed)

    print("=" * 80)
    print("CAUSAL ACTIVATION PATCHING — Sycophancy Circuit Discovery")
    print("=" * 80)
    print(f"Model:          {args.model}")
    print(f"Data:           {args.data}")
    print(f"Max Samples:    {args.max_samples}")
    print(f"Max Positions:  {args.max_positions}")
    print(f"Head Top-K:     {args.head_top_k}")
    print(f"Seed:           {args.seed}")
    print(f"Git Hash:       {get_git_hash() or 'N/A'}")
    print()

    # Load data
    dataset = load_data(args.data, args.max_samples)
    print(f"Loaded {len(dataset)} samples\n")

    # Initialize model
    model = SycophancyModel(args.model, device=args.device)

    # Determine layers
    if args.layers is None or args.layers == "all":
        layers = list(range(model.n_layers))
    else:
        layers = [int(l.strip()) for l in args.layers.split(",")]
    print(f"Patching {len(layers)} layers: {layers[0]}..{layers[-1]}\n")

    # Phase 1: Layer × Position patching
    print("[Phase 1] Layer × Position Patching")
    print("-" * 60)

    layer_results = []
    skipped = 0

    for item in tqdm(dataset, desc="Layer patching"):
        result = run_layer_patching_single(
            model, item, layers, max_positions=args.max_positions
        )
        if result is None:
            skipped += 1
            continue
        layer_results.append(result)

    print(f"\n  Successful: {len(layer_results)} | Skipped: {skipped}")

    if len(layer_results) == 0:
        print("No valid patching results. Check that the model shows sycophancy on this dataset.")
        return 1

    # Aggregate
    layer_agg = aggregate_patching_results(layer_results, layers)
    critical_layers = layer_agg['critical_layers']

    print(f"\n  Critical layers (top {HEAD_ANALYSIS_TOP_K}): {critical_layers}")
    print(f"  Mean total effect: {layer_agg['mean_total_effect']:.4f}")

    # Save layer results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_output = {
        'metadata': {
            'model_name': args.model,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data_path': args.data,
            'n_samples': len(layer_results),
            'n_skipped': skipped,
            'layers': layers,
            'max_positions': args.max_positions,
            'random_seed': args.seed,
            'git_hash': get_git_hash(),
            'environment': get_environment_info(),
        },
        'layer_results': layer_agg,
    }

    with open(output_dir / "patching_heatmap.json", 'w') as f:
        json.dump(layer_output, f, indent=2)
    print(f"\nLayer results saved to {output_dir / 'patching_heatmap.json'}")

    # Phase 2: Head-level patching
    if not args.skip_head_analysis and critical_layers:
        print(f"\n[Phase 2] Head-Level Patching (layers {critical_layers})")
        print("-" * 60)

        head_results = []
        skipped_heads = 0

        for item in tqdm(dataset, desc="Head patching"):
            result = run_head_patching_single(model, item, critical_layers)
            if result is None:
                skipped_heads += 1
                continue
            head_results.append(result)

        print(f"\n  Successful: {len(head_results)} | Skipped: {skipped_heads}")

        if head_results:
            head_agg = aggregate_head_results(head_results)

            head_output = {
                'metadata': {
                    'model_name': args.model,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'critical_layers': critical_layers,
                    'n_samples': len(head_results),
                    'n_heads_per_layer': model.n_heads,
                    'random_seed': args.seed,
                    'git_hash': get_git_hash(),
                },
                'head_results': head_agg,
            }

            with open(output_dir / "head_importance.json", 'w') as f:
                json.dump(head_output, f, indent=2)
            print(f"Head results saved to {output_dir / 'head_importance.json'}")

            # Print top heads
            print(f"\n  Top 10 sycophancy heads (by recovery score):")
            for entry in head_agg['top_10_heads']:
                print(f"    {entry['head']}: "
                      f"recovery={entry['mean_recovery']:.4f} "
                      f"(±{entry['std_recovery']:.4f})")

    # Print summary
    print("\n" + "=" * 80)
    print("PATCHING RESULTS SUMMARY")
    print("=" * 80)
    print(f"Samples analyzed:     {len(layer_results)}")
    print(f"Mean total effect:    {layer_agg['mean_total_effect']:.4f} "
          f"(±{layer_agg['std_total_effect']:.4f})")
    print(f"Critical layers:      {critical_layers}")

    # Layer importance ranking
    sorted_layers = sorted(
        layer_agg['layer_importance'].items(),
        key=lambda x: x[1],
        reverse=True,
    )
    print(f"\nTop 10 layers by importance:")
    for layer_str, importance in sorted_layers[:10]:
        print(f"  Layer {layer_str:>3s}: {importance:.4f}")

    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
