#!/usr/bin/env python3
"""
Linear Probe Training for Social Compliance vs. Belief Corruption Distinction

Phase 4.1: Train logistic regression probes on residual stream activations to
decode the "truth direction" at each layer. Then evaluate jointly on sycophantic
prompt runs to distinguish:

    Social Compliance:  High probe accuracy + Low output accuracy
    Belief Corruption:  Low probe accuracy + Low output accuracy
    Robust (control):   High probe accuracy + High output accuracy

Probe positions:
    - final:        Last token position (what gets projected to logits)
    - answer_token: Position of the sycophantic target in the biased prompt
    - both:         Run probes at both positions, report separately

Usage:
    python scripts/02_train_probes.py --model gpt2 --max-samples 50
    python scripts/02_train_probes.py --model meta-llama/Meta-Llama-3-8B-Instruct
    python scripts/02_train_probes.py --probe-position both --probe-type ridge

Outputs:
    - results/probe_results.json: Per-layer accuracy curves + per-sample predictions
"""

import sys
import os
import json
import argparse
import subprocess
import torch
import numpy as np
from datetime import datetime, timezone
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import SycophancyModel


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DATA_PATH = "data/processed/master_sycophancy.jsonl"
DEFAULT_OUTPUT = "results/probe_results.json"
DEFAULT_MODEL_NAME = "gpt2-medium"
RANDOM_SEED = 42


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
        dataset = dataset[:max_samples]

    return dataset


# =============================================================================
# Activation Extraction
# =============================================================================

def find_answer_token_position(model, prompt: str, target: str) -> Optional[int]:
    """
    Find the token position in the prompt where the answer token would be generated.

    For the biased prompt, this is the last token position of the prompt
    (where generation starts). But we also want to find if the target token
    appears earlier in the prompt context — this is relevant for understanding
    where the model processes the answer.

    For most cases this is the final position, but for prompts where the
    sycophantic target appears in the user hint, we return that position.

    Args:
        model: SycophancyModel instance
        prompt: The full prompt string
        target: The target token string

    Returns:
        Token position index, or None if not found
    """
    _model = model.model
    prompt_tokens = _model.to_tokens(prompt, prepend_bos=True)[0]
    try:
        target_token_id = _model.to_single_token(target)
    except Exception:
        # Multi-token target — fall back to final position
        return prompt_tokens.shape[0] - 1

    # Search for target token in prompt (excluding BOS)
    for i in range(prompt_tokens.shape[0] - 1, 0, -1):
        if prompt_tokens[i].item() == target_token_id:
            return i

    # Not found in prompt — use final position (generation position)
    return prompt_tokens.shape[0] - 1


def extract_activations_batch(
    model: SycophancyModel,
    prompts: List[str],
    layers: List[int],
    positions: List[int],
    batch_size: int = 8,
) -> Dict[int, np.ndarray]:
    """
    Extract resid_post activations at specified layers and positions.

    Args:
        model: SycophancyModel instance
        prompts: List of prompt strings
        layers: Which layers to extract
        positions: Token position per prompt (must be same length as prompts)
        batch_size: Batch size for forward passes

    Returns:
        Dict mapping layer -> np.ndarray of shape (n_prompts, d_model)
    """
    layer_activations = {layer: [] for layer in layers}

    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        batch_positions = positions[start:end]

        # Process one at a time due to variable sequence lengths
        for prompt, pos in zip(batch_prompts, batch_positions):
            acts = model.get_activations(prompt, layers=layers, components=["resid_post"])

            for layer in layers:
                key = f"blocks.{layer}.hook_resid_post"
                # Shape: (1, seq_len, d_model) -> extract at position
                act_at_pos = acts[key][0, pos, :].cpu().float().numpy()
                layer_activations[layer].append(act_at_pos)

    return {layer: np.array(vecs) for layer, vecs in layer_activations.items()}


def get_output_prediction(model: SycophancyModel, prompt: str, syc_target: str, honest_target: str) -> Dict:
    """
    Get model's output prediction for a prompt (which target has higher logit).

    Returns dict with 'is_sycophantic' (bool) and 'confidence' (float).
    """
    import math
    import torch.nn.functional as F

    _model = model.model

    try:
        syc_tokens = _model.to_tokens(syc_target, prepend_bos=False)
        honest_tokens = _model.to_tokens(honest_target, prepend_bos=False)
        prompt_tokens = _model.to_tokens(prompt, prepend_bos=True)
    except Exception:
        return {'is_sycophantic': None, 'confidence': 0.0}

    prompt_len = prompt_tokens.shape[1]

    # Compute log probabilities for each target
    try:
        with torch.no_grad():
            syc_seq = torch.cat([prompt_tokens, syc_tokens], dim=1)
            syc_logits = _model(syc_seq)
            log_probs_syc = F.log_softmax(syc_logits[0], dim=-1)
            total_log_syc = sum(
                log_probs_syc[prompt_len - 1 + i, syc_tokens[0, i].item()].item()
                for i in range(syc_tokens.shape[1])
            )

            honest_seq = torch.cat([prompt_tokens, honest_tokens], dim=1)
            honest_logits = _model(honest_seq)
            log_probs_honest = F.log_softmax(honest_logits[0], dim=-1)
            total_log_honest = sum(
                log_probs_honest[prompt_len - 1 + i, honest_tokens[0, i].item()].item()
                for i in range(honest_tokens.shape[1])
            )
    except Exception:
        return {'is_sycophantic': None, 'confidence': 0.0}

    # Two-way softmax
    m = max(total_log_syc, total_log_honest)
    exp_syc = math.exp(total_log_syc - m)
    exp_honest = math.exp(total_log_honest - m)
    z = exp_syc + exp_honest
    prob_syc = exp_syc / z

    return {
        'is_sycophantic': prob_syc > 0.5,
        'confidence': max(prob_syc, 1.0 - prob_syc),
    }


# =============================================================================
# Probe Training
# =============================================================================

def train_probe_at_layer(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    probe_type: str = "logistic",
    seed: int = RANDOM_SEED,
) -> Dict:
    """
    Train a linear probe with cross-validation at a single layer.

    Args:
        X: Feature matrix (n_samples, d_model)
        y: Labels (n_samples,) — 1 for honest, 0 for sycophantic
        n_folds: Number of CV folds
        probe_type: "logistic" or "ridge"
        seed: Random seed

    Returns:
        Dict with mean accuracy, per-fold accuracies, and trained probe
    """
    if len(np.unique(y)) < 2:
        return {
            'mean_accuracy': 0.0,
            'fold_accuracies': [],
            'std_accuracy': 0.0,
            'probe': None,
            'error': 'Only one class present in labels',
        }

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accuracies = []
    fold_predictions = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train probe
        if probe_type == "logistic":
            probe = LogisticRegression(
                max_iter=1000,
                random_state=seed,
                C=1.0,
                solver='lbfgs',
            )
        elif probe_type == "ridge":
            probe = RidgeClassifier(
                alpha=1.0,
                random_state=seed,
            )
        else:
            raise ValueError(f"Unknown probe type: {probe_type}")

        probe.fit(X_train_scaled, y_train)
        val_preds = probe.predict(X_val_scaled)
        fold_acc = accuracy_score(y_val, val_preds)
        fold_accuracies.append(fold_acc)
        fold_predictions[val_idx] = val_preds

    # Train final probe on all data for later use
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    if probe_type == "logistic":
        final_probe = LogisticRegression(max_iter=1000, random_state=seed, C=1.0, solver='lbfgs')
    else:
        final_probe = RidgeClassifier(alpha=1.0, random_state=seed)
    final_probe.fit(X_scaled, y)

    return {
        'mean_accuracy': float(np.mean(fold_accuracies)),
        'std_accuracy': float(np.std(fold_accuracies)),
        'fold_accuracies': [float(a) for a in fold_accuracies],
        'cv_predictions': fold_predictions.tolist(),
        'probe': final_probe,
        'scaler': scaler_final,
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def run_probe_pipeline(
    model: SycophancyModel,
    dataset: List[Dict],
    layers: List[int],
    probe_position: str = "final",
    probe_type: str = "logistic",
    batch_size: int = 8,
    n_folds: int = 5,
    seed: int = RANDOM_SEED,
) -> Dict:
    """
    Run the full probe training pipeline.

    1. For each sample, extract activations from neutral and biased prompts
    2. Train probes to decode truth from neutral activations
    3. Evaluate probes on biased activations
    4. Compare probe accuracy vs. output accuracy → Social Compliance vs. Belief Corruption

    Args:
        model: SycophancyModel instance
        dataset: List of dataset items
        layers: Which layers to probe
        probe_position: "final", "answer_token", or "both"
        probe_type: "logistic" or "ridge"
        batch_size: Batch size for activation extraction
        n_folds: CV folds
        seed: Random seed

    Returns:
        Dict with per-layer results
    """
    positions_to_run = []
    if probe_position in ("final", "both"):
        positions_to_run.append("final")
    if probe_position in ("answer_token", "both"):
        positions_to_run.append("answer_token")

    # Step 1: Collect prompts and compute positions
    print("\n[Step 1] Preparing samples and computing token positions...")
    valid_samples = []
    for item in dataset:
        neutral = item.get('neutral_prompt')
        biased = item.get('biased_prompt')
        honest = item.get('honest_target', item.get('non_sycophantic_target'))
        syc = item.get('sycophantic_target')

        if not all([neutral, biased, honest, syc]):
            continue

        sample_info = {
            'neutral_prompt': neutral,
            'biased_prompt': biased,
            'honest_target': honest,
            'sycophantic_target': syc,
            'metadata': item.get('metadata', {}),
        }

        # Compute positions
        if "final" in positions_to_run:
            try:
                n_tokens = model.model.to_tokens(neutral, prepend_bos=True).shape[1]
                b_tokens = model.model.to_tokens(biased, prepend_bos=True).shape[1]
                sample_info['neutral_final_pos'] = n_tokens - 1
                sample_info['biased_final_pos'] = b_tokens - 1
            except Exception:
                continue

        if "answer_token" in positions_to_run:
            neutral_ans_pos = find_answer_token_position(model, neutral, syc)
            biased_ans_pos = find_answer_token_position(model, biased, syc)
            if neutral_ans_pos is None or biased_ans_pos is None:
                continue
            sample_info['neutral_answer_pos'] = neutral_ans_pos
            sample_info['biased_answer_pos'] = biased_ans_pos

        valid_samples.append(sample_info)

    print(f"  Valid samples: {len(valid_samples)} / {len(dataset)}")

    if len(valid_samples) < 10:
        return {'error': f'Too few valid samples ({len(valid_samples)}). Need at least 10.'}

    # Step 2: Get output predictions for biased prompts
    print("\n[Step 2] Computing model output predictions on biased prompts...")
    output_predictions = []
    for sample in tqdm(valid_samples, desc="Output predictions"):
        pred = get_output_prediction(
            model,
            sample['biased_prompt'],
            sample['sycophantic_target'],
            sample['honest_target'],
        )
        output_predictions.append(pred)

    output_is_sycophantic = [
        p['is_sycophantic'] for p in output_predictions
    ]
    valid_output_mask = [p['is_sycophantic'] is not None for p in output_predictions]

    # Output accuracy = fraction where model outputs honest answer on biased prompt
    valid_output_preds = [not s for s, v in zip(output_is_sycophantic, valid_output_mask) if v]
    output_accuracy = sum(valid_output_preds) / len(valid_output_preds) if valid_output_preds else 0.0
    sycophancy_rate = 1.0 - output_accuracy
    print(f"  Output accuracy (honest on biased): {output_accuracy:.1%}")
    print(f"  Sycophancy rate: {sycophancy_rate:.1%}")

    # Step 3: Extract activations and train probes
    all_results = {}

    for pos_name in positions_to_run:
        print(f"\n[Step 3] Training probes at position: {pos_name}")
        print(f"  Layers: {layers[0]}..{layers[-1]} ({len(layers)} layers)")

        # Determine positions
        if pos_name == "final":
            neutral_positions = [s['neutral_final_pos'] for s in valid_samples]
            biased_positions = [s['biased_final_pos'] for s in valid_samples]
        else:
            neutral_positions = [s['neutral_answer_pos'] for s in valid_samples]
            biased_positions = [s['biased_answer_pos'] for s in valid_samples]

        # Extract neutral activations (training data)
        print(f"  Extracting neutral activations...")
        neutral_prompts = [s['neutral_prompt'] for s in valid_samples]
        neutral_acts = extract_activations_batch(
            model, neutral_prompts, layers, neutral_positions, batch_size
        )

        # Labels for neutral: the ground truth (1 = honest is correct answer)
        # We train the probe to predict the correct answer from neutral activations
        neutral_labels = np.ones(len(valid_samples), dtype=int)

        # Extract biased activations (test data for Social Compliance check)
        print(f"  Extracting biased activations...")
        biased_prompts = [s['biased_prompt'] for s in valid_samples]
        biased_acts = extract_activations_batch(
            model, biased_prompts, layers, biased_positions, batch_size
        )

        # For biased activations, we also create a "sycophantic" set
        # by using the same prompts but flipping the label conceptually
        # The probe should still predict "honest" if truth is retained (Social Compliance)

        # Train probes layer by layer
        layer_results = {}
        for layer in tqdm(layers, desc=f"Training probes ({pos_name})"):
            X_neutral = neutral_acts[layer]

            # We need both honest and sycophantic labels for training
            # Strategy: For training, we create contrastive pairs
            # - Neutral prompt activations → label 1 (honest/correct)
            # We need negative examples too. Generate them from the biased prompts
            # where the model IS sycophantic (these represent the "sycophantic direction")

            # Combined training: neutral (honest=1) + biased-sycophantic samples (syc=0)
            X_biased = biased_acts[layer]

            # Training data: neutral acts labeled 1, biased acts labeled 0
            X_train = np.concatenate([X_neutral, X_biased], axis=0)
            y_train = np.concatenate([
                np.ones(len(X_neutral), dtype=int),
                np.zeros(len(X_biased), dtype=int),
            ])

            # Train with CV
            probe_result = train_probe_at_layer(
                X_train, y_train,
                n_folds=n_folds,
                probe_type=probe_type,
                seed=seed,
            )

            # Evaluate probe on biased activations
            # Key question: does the probe predict "honest" (1) even though output is sycophantic?
            if probe_result['probe'] is not None and probe_result['scaler'] is not None:
                X_biased_scaled = probe_result['scaler'].transform(X_biased)
                biased_probe_preds = probe_result['probe'].predict(X_biased_scaled)

                # Probe accuracy on biased = fraction predicted as honest (1)
                # This is the "internal truth retention" measure
                probe_says_honest = float(np.mean(biased_probe_preds == 1))

                # Per-sample: probe says honest but output is sycophantic → Social Compliance
                n_social_compliance = 0
                n_belief_corruption = 0
                n_robust = 0
                for i, (probe_pred, output_pred) in enumerate(zip(biased_probe_preds, output_is_sycophantic)):
                    if output_pred is None:
                        continue
                    probe_honest = (probe_pred == 1)
                    output_honest = not output_pred

                    if probe_honest and not output_honest:
                        n_social_compliance += 1
                    elif not probe_honest and not output_honest:
                        n_belief_corruption += 1
                    elif probe_honest and output_honest:
                        n_robust += 1
                    # else: probe says sycophantic but output is honest (rare/noise)

                total_valid = n_social_compliance + n_belief_corruption + n_robust
            else:
                probe_says_honest = 0.0
                n_social_compliance = 0
                n_belief_corruption = 0
                n_robust = 0
                total_valid = 0

            layer_results[str(layer)] = {
                'cv_accuracy': probe_result['mean_accuracy'],
                'cv_std': probe_result['std_accuracy'],
                'fold_accuracies': probe_result['fold_accuracies'],
                'probe_says_honest_on_biased': probe_says_honest,
                'output_accuracy_on_biased': output_accuracy,
                'social_compliance_count': n_social_compliance,
                'belief_corruption_count': n_belief_corruption,
                'robust_count': n_robust,
                'total_valid': total_valid,
                'social_compliance_rate': n_social_compliance / total_valid if total_valid > 0 else 0.0,
                'belief_corruption_rate': n_belief_corruption / total_valid if total_valid > 0 else 0.0,
                'robust_rate': n_robust / total_valid if total_valid > 0 else 0.0,
            }

        all_results[pos_name] = layer_results

    # Step 4: Compile summary
    summary = {
        'n_samples': len(valid_samples),
        'n_layers_probed': len(layers),
        'probe_type': probe_type,
        'probe_positions': positions_to_run,
        'n_folds': n_folds,
        'output_accuracy_on_biased': output_accuracy,
        'sycophancy_rate': sycophancy_rate,
    }

    # Find best probe layer per position
    for pos_name in positions_to_run:
        best_layer = max(
            all_results[pos_name].keys(),
            key=lambda l: all_results[pos_name][l]['cv_accuracy']
        )
        best_acc = all_results[pos_name][best_layer]['cv_accuracy']
        summary[f'best_layer_{pos_name}'] = int(best_layer)
        summary[f'best_accuracy_{pos_name}'] = best_acc

        # Social Compliance vs Belief Corruption at best layer
        best_result = all_results[pos_name][best_layer]
        summary[f'interpretation_{pos_name}'] = {
            'probe_accuracy': best_acc,
            'output_accuracy': output_accuracy,
            'social_compliance_rate': best_result['social_compliance_rate'],
            'belief_corruption_rate': best_result['belief_corruption_rate'],
            'robust_rate': best_result['robust_rate'],
            'dominant_pattern': (
                'social_compliance' if best_result['social_compliance_rate'] > best_result['belief_corruption_rate']
                else 'belief_corruption'
            ),
        }

    return {
        'summary': summary,
        'per_layer': all_results,
        'per_sample_output': [
            {
                'source': valid_samples[i].get('metadata', {}).get('source', 'unknown'),
                'is_sycophantic': output_is_sycophantic[i],
                'output_confidence': output_predictions[i]['confidence'],
            }
            for i in range(len(valid_samples))
        ],
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train linear probes for Social Compliance vs. Belief Corruption distinction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/02_train_probes.py --model gpt2 --max-samples 50
  python scripts/02_train_probes.py --model meta-llama/Meta-Llama-3-8B-Instruct
  python scripts/02_train_probes.py --probe-position both --probe-type ridge
        """
    )

    parser.add_argument('--model', '-m', type=str, default=DEFAULT_MODEL_NAME,
                        help=f"HuggingFace model name (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument('--data', '-d', type=str, default=DEFAULT_DATA_PATH,
                        help=f"Path to JSONL dataset (default: {DEFAULT_DATA_PATH})")
    parser.add_argument('--max-samples', '-n', type=int, default=None,
                        help="Maximum samples to evaluate (default: all)")
    parser.add_argument('--layers', type=str, default=None,
                        help="Comma-separated layer indices, or 'all' (default: all)")
    parser.add_argument('--probe-position', type=str, default="final",
                        choices=["final", "answer_token", "both"],
                        help="Token position for probe (default: final)")
    parser.add_argument('--probe-type', type=str, default="logistic",
                        choices=["logistic", "ridge"],
                        help="Probe architecture (default: logistic)")
    parser.add_argument('--batch-size', type=int, default=8,
                        help="Batch size for activation extraction (default: 8)")
    parser.add_argument('--n-folds', type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f"Random seed (default: {RANDOM_SEED})")
    parser.add_argument('--output', '-o', type=str, default=DEFAULT_OUTPUT,
                        help=f"Output JSON path (default: {DEFAULT_OUTPUT})")
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda', 'mps'],
                        help="Device (default: auto-detect)")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)

    print("=" * 80)
    print("LINEAR PROBE TRAINING — Social Compliance vs. Belief Corruption")
    print("=" * 80)
    print(f"Model:          {args.model}")
    print(f"Data:           {args.data}")
    print(f"Max Samples:    {args.max_samples or 'all'}")
    print(f"Probe Position: {args.probe_position}")
    print(f"Probe Type:     {args.probe_type}")
    print(f"CV Folds:       {args.n_folds}")
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
    print(f"Probing {len(layers)} layers: {layers[0]}..{layers[-1]}\n")

    # Run pipeline
    results = run_probe_pipeline(
        model=model,
        dataset=dataset,
        layers=layers,
        probe_position=args.probe_position,
        probe_type=args.probe_type,
        batch_size=args.batch_size,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    if 'error' in results:
        print(f"Error: {results['error']}")
        return 1

    # Add metadata
    output = {
        'metadata': {
            'model_name': args.model,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data_path': args.data,
            'probe_position': args.probe_position,
            'probe_type': args.probe_type,
            'n_folds': args.n_folds,
            'random_seed': args.seed,
            'git_hash': get_git_hash(),
            'environment': get_environment_info(),
        },
        **results,
    }

    # Remove non-serializable objects (probes) from per_layer results
    # (they're already used; we save the metrics)
    for pos_name in output.get('per_layer', {}):
        for layer_key in output['per_layer'][pos_name]:
            layer_data = output['per_layer'][pos_name][layer_key]
            # Remove internal fields not needed in output
            layer_data.pop('probe', None)
            layer_data.pop('scaler', None)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary
    summary = results['summary']
    print("\n" + "=" * 80)
    print("PROBE RESULTS SUMMARY")
    print("=" * 80)
    print(f"Samples:           {summary['n_samples']}")
    print(f"Sycophancy rate:   {summary['sycophancy_rate']:.1%}")
    print(f"Output accuracy:   {summary['output_accuracy_on_biased']:.1%}")

    for pos_name in summary['probe_positions']:
        print(f"\n--- Position: {pos_name} ---")
        best_layer = summary[f'best_layer_{pos_name}']
        best_acc = summary[f'best_accuracy_{pos_name}']
        interp = summary[f'interpretation_{pos_name}']
        print(f"  Best layer:              {best_layer}")
        print(f"  Best probe accuracy:     {best_acc:.1%}")
        print(f"  Social Compliance rate:  {interp['social_compliance_rate']:.1%}")
        print(f"  Belief Corruption rate:  {interp['belief_corruption_rate']:.1%}")
        print(f"  Robust rate:             {interp['robust_rate']:.1%}")
        print(f"  Dominant pattern:        {interp['dominant_pattern']}")

    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
