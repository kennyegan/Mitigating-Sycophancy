#!/usr/bin/env python3
"""
Linear Probe Training for Social Compliance vs Belief Corruption Analysis.

This script supports two modes:

1) neutral_transfer (default, claim-bearing)
   - Train probes on neutral activations only (answer-identity labels)
   - Evaluate transfer on biased activations
   - Compare probe correctness vs model output to estimate:
     social compliance, belief corruption, and robustness

2) mixed_diagnostic (diagnostic only)
   - Train probes on mixed neutral+biased activations
   - Label = prompt condition (neutral vs biased)
   - Uses leakage-safe GroupKFold by sample_id
   - Intended only to quantify format-confounding risk
"""

import sys
import os
import json
import argparse
import subprocess
import math
import hashlib
import random
import torch
import numpy as np
from datetime import datetime, timezone
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import SycophancyModel


DEFAULT_DATA_PATH = "data/processed/master_sycophancy.jsonl"
DEFAULT_OUTPUT = "results/probe_results.json"
DEFAULT_MODEL_NAME = "gpt2-medium"
DEFAULT_SCHEMA_VERSION = "2.0"
RANDOM_SEED = 42


def get_git_hash() -> Optional[str]:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(data_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data not found at {data_path}. "
            "Run 'make data' first to download the benchmark."
        )

    with open(data_path, 'r') as f:
        dataset = [json.loads(line) for line in f]

    if max_samples is not None:
        dataset = dataset[:max_samples]

    return dataset


def bootstrap_ci_binary(values: List[int], confidence: float = 0.95, n_bootstrap: int = 2000, seed: int = 42) -> Tuple[float, float]:
    """Bootstrap percentile CI for binary outcomes."""
    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))

    rng = np.random.RandomState(seed)
    n = len(arr)
    boot = []
    for _ in range(n_bootstrap):
        sample = arr[rng.randint(0, n, size=n)]
        boot.append(float(np.mean(sample)))

    alpha = 1 - confidence
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return (lo, hi)


def get_sample_id(item: Dict, fallback_idx: int) -> str:
    sid = item.get('sample_id') or item.get('metadata', {}).get('sample_id')
    if sid:
        return str(sid)

    core = {
        'neutral_prompt': item.get('neutral_prompt', ''),
        'biased_prompt': item.get('biased_prompt', ''),
        'sycophantic_target': item.get('sycophantic_target', ''),
        'honest_target': item.get('honest_target', item.get('non_sycophantic_target', '')),
    }
    payload = json.dumps(core, sort_keys=True, ensure_ascii=True).encode('utf-8')
    digest = hashlib.sha1(payload).hexdigest()[:12]
    return f"fallback-{fallback_idx:06d}-{digest}"


def find_answer_token_position(model: SycophancyModel, prompt: str, target: str) -> int:
    _model = model.model
    prompt_tokens = _model.to_tokens(prompt, prepend_bos=True)[0]

    try:
        target_token_id = _model.to_single_token(target)
    except Exception:
        return prompt_tokens.shape[0] - 1

    for i in range(prompt_tokens.shape[0] - 1, 0, -1):
        if prompt_tokens[i].item() == target_token_id:
            return i
    return prompt_tokens.shape[0] - 1


def extract_activations_batch(
    model: SycophancyModel,
    prompts: List[str],
    layers: List[int],
    positions: List[int],
    batch_size: int = 8,
) -> Dict[int, np.ndarray]:
    layer_activations = {layer: [] for layer in layers}

    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        batch_positions = positions[start:end]

        for prompt, pos in zip(batch_prompts, batch_positions):
            acts = model.get_activations(prompt, layers=layers, components=["resid_post"])
            for layer in layers:
                key = f"blocks.{layer}.hook_resid_post"
                seq_len = acts[key].shape[1]
                actual_pos = min(max(pos, 0), seq_len - 1)
                act_at_pos = acts[key][0, actual_pos, :].cpu().float().numpy()
                layer_activations[layer].append(act_at_pos)

    return {layer: np.array(vecs) for layer, vecs in layer_activations.items()}


def get_output_prediction(model: SycophancyModel, prompt: str, syc_target: str, honest_target: str) -> Dict:
    import torch.nn.functional as F

    _model = model.model
    try:
        syc_tokens = _model.to_tokens(syc_target, prepend_bos=False)
        honest_tokens = _model.to_tokens(honest_target, prepend_bos=False)
        prompt_tokens = _model.to_tokens(prompt, prepend_bos=True)
    except Exception:
        return {'is_sycophantic': None, 'confidence': 0.0}

    prompt_len = prompt_tokens.shape[1]

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

    m = max(total_log_syc, total_log_honest)
    exp_syc = math.exp(total_log_syc - m)
    exp_honest = math.exp(total_log_honest - m)
    z = exp_syc + exp_honest
    prob_syc = exp_syc / z

    return {
        'is_sycophantic': prob_syc > 0.5,
        'confidence': max(prob_syc, 1.0 - prob_syc),
    }


def prepare_valid_samples(model: SycophancyModel, dataset: List[Dict], probe_position: str) -> List[Dict]:
    positions_to_run = []
    if probe_position in ("final", "both"):
        positions_to_run.append("final")
    if probe_position in ("answer_token", "both"):
        positions_to_run.append("answer_token")

    valid_samples = []
    for idx, item in enumerate(dataset):
        neutral = item.get('neutral_prompt')
        biased = item.get('biased_prompt')
        honest = item.get('honest_target', item.get('non_sycophantic_target'))
        syc = item.get('sycophantic_target')

        if not all([neutral, biased, honest, syc]):
            continue

        sample = {
            'sample_id': get_sample_id(item, idx),
            'neutral_prompt': neutral,
            'biased_prompt': biased,
            'honest_target': honest,
            'sycophantic_target': syc,
            'metadata': item.get('metadata', {}),
        }

        try:
            n_tokens = model.model.to_tokens(neutral, prepend_bos=True).shape[1]
            b_tokens = model.model.to_tokens(biased, prepend_bos=True).shape[1]
            sample['neutral_final_pos'] = n_tokens - 1
            sample['biased_final_pos'] = b_tokens - 1
        except Exception:
            continue

        if "answer_token" in positions_to_run:
            try:
                sample['neutral_answer_pos'] = find_answer_token_position(model, neutral, syc)
                sample['biased_answer_pos'] = find_answer_token_position(model, biased, syc)
            except Exception:
                continue

        valid_samples.append(sample)

    return valid_samples


def create_answer_identity_labels(valid_samples: List[Dict]) -> Tuple[np.ndarray, Dict]:
    labels = []
    per_source = {}

    for sample in valid_samples:
        meta = sample.get('metadata', {})
        honest = sample['honest_target']
        syc = sample['sycophantic_target']
        source = meta.get('source', 'unknown')

        honest_opt = str(meta.get('honest_option', '')).strip().upper()
        syc_opt = str(meta.get('sycophantic_option', '')).strip().upper()
        if honest_opt in {'A', 'B'}:
            # Label denotes which option is honest: 0 => A, 1 => B.
            label = 0 if honest_opt == 'A' else 1
        elif syc_opt in {'A', 'B'}:
            label = 1 if syc_opt == 'A' else 0
        else:
            # Fallback for legacy records without explicit option metadata.
            label = 0 if honest < syc else 1

        labels.append(label)
        per_source.setdefault(source, []).append(label)

    labels = np.asarray(labels, dtype=int)
    overall_majority = float(max(np.mean(labels == 0), np.mean(labels == 1))) if len(labels) > 0 else 0.0

    source_stats = {}
    for source, vals in per_source.items():
        arr = np.asarray(vals, dtype=int)
        source_stats[source] = {
            'n_total': int(len(arr)),
            'n_class_0': int(np.sum(arr == 0)),
            'n_class_1': int(np.sum(arr == 1)),
            'majority_fraction': float(max(np.mean(arr == 0), np.mean(arr == 1))) if len(arr) > 0 else 0.0,
        }

    return labels, {
        'overall': {
            'n_total': int(len(labels)),
            'n_class_0': int(np.sum(labels == 0)),
            'n_class_1': int(np.sum(labels == 1)),
            'majority_fraction': overall_majority,
        },
        'per_source': source_stats,
    }


def make_probe(probe_type: str, seed: int):
    if probe_type == "logistic":
        return LogisticRegression(max_iter=1000, random_state=seed, C=1.0, solver='lbfgs')
    if probe_type == "ridge":
        return RidgeClassifier(alpha=1.0, random_state=seed)
    raise ValueError(f"Unknown probe_type: {probe_type}")


def run_neutral_transfer_position(
    model: SycophancyModel,
    valid_samples: List[Dict],
    layers: List[int],
    position_name: str,
    probe_type: str,
    batch_size: int,
    n_folds: int,
    seed: int,
) -> Tuple[Dict, Dict]:
    labels, class_balance = create_answer_identity_labels(valid_samples)

    output_predictions = []
    for sample in tqdm(valid_samples, desc=f"Output predictions ({position_name})"):
        output_predictions.append(
            get_output_prediction(model, sample['biased_prompt'], sample['sycophantic_target'], sample['honest_target'])
        )
    output_is_sycophantic = [p['is_sycophantic'] for p in output_predictions]

    if position_name == "final":
        neutral_positions = [s['neutral_final_pos'] for s in valid_samples]
        biased_positions = [s['biased_final_pos'] for s in valid_samples]
    else:
        neutral_positions = [s['neutral_answer_pos'] for s in valid_samples]
        biased_positions = [s['biased_answer_pos'] for s in valid_samples]

    neutral_prompts = [s['neutral_prompt'] for s in valid_samples]
    biased_prompts = [s['biased_prompt'] for s in valid_samples]

    neutral_acts = extract_activations_batch(model, neutral_prompts, layers, neutral_positions, batch_size)
    biased_acts = extract_activations_batch(model, biased_prompts, layers, biased_positions, batch_size)

    per_layer = {}

    for layer in tqdm(layers, desc=f"Neutral-transfer probes ({position_name})"):
        X_neutral = neutral_acts[layer]
        X_biased = biased_acts[layer]
        y = labels

        if len(np.unique(y)) < 2:
            per_layer[str(layer)] = {
                'error': 'Only one class present in labels',
                'neutral_cv_accuracy': 0.0,
                'biased_transfer_accuracy': 0.0,
            }
            continue

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        neutral_fold_acc = []
        biased_fold_acc = []
        oof_biased_preds = np.full(len(y), -1, dtype=int)

        for train_idx, val_idx in skf.split(X_neutral, y):
            X_train = X_neutral[train_idx]
            y_train = y[train_idx]

            X_val_neutral = X_neutral[val_idx]
            X_val_biased = X_biased[val_idx]
            y_val = y[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_neutral_scaled = scaler.transform(X_val_neutral)
            X_val_biased_scaled = scaler.transform(X_val_biased)

            probe = make_probe(probe_type, seed)
            probe.fit(X_train_scaled, y_train)

            neutral_pred = probe.predict(X_val_neutral_scaled)
            biased_pred = probe.predict(X_val_biased_scaled)

            neutral_fold_acc.append(float(accuracy_score(y_val, neutral_pred)))
            biased_fold_acc.append(float(accuracy_score(y_val, biased_pred)))
            oof_biased_preds[val_idx] = biased_pred

        neutral_cv_accuracy = float(np.mean(neutral_fold_acc))
        biased_transfer_accuracy = float(np.mean(biased_fold_acc))

        transfer_correct = [int(oof_biased_preds[i] == y[i]) for i in range(len(y)) if oof_biased_preds[i] != -1]
        transfer_ci = bootstrap_ci_binary(transfer_correct, seed=seed)

        # Cross-tabulate probe correctness vs model output
        social_flags = []
        belief_flags = []
        robust_flags = []
        n_other = 0

        for i in range(len(valid_samples)):
            if oof_biased_preds[i] == -1 or output_is_sycophantic[i] is None:
                continue
            probe_correct = (oof_biased_preds[i] == y[i])
            model_honest = (not output_is_sycophantic[i])

            social = int(probe_correct and not model_honest)
            belief = int((not probe_correct) and (not model_honest))
            robust = int(probe_correct and model_honest)
            other = int((not probe_correct) and model_honest)

            if other:
                n_other += 1
            social_flags.append(social)
            belief_flags.append(belief)
            robust_flags.append(robust)

        total_classified = len(social_flags)
        social_rate = float(np.mean(social_flags)) if social_flags else 0.0
        belief_rate = float(np.mean(belief_flags)) if belief_flags else 0.0
        robust_rate = float(np.mean(robust_flags)) if robust_flags else 0.0

        per_layer[str(layer)] = {
            'neutral_cv_accuracy': neutral_cv_accuracy,
            'neutral_cv_std': float(np.std(neutral_fold_acc)),
            'neutral_fold_accuracies': neutral_fold_acc,
            'biased_transfer_accuracy': biased_transfer_accuracy,
            'biased_transfer_std': float(np.std(biased_fold_acc)),
            'biased_transfer_fold_accuracies': biased_fold_acc,
            'biased_transfer_accuracy_ci': [float(transfer_ci[0]), float(transfer_ci[1])],
            'accuracy_drop': float(neutral_cv_accuracy - biased_transfer_accuracy),
            'social_compliance_rate': social_rate,
            'social_compliance_rate_ci': list(bootstrap_ci_binary(social_flags, seed=seed + 1)) if social_flags else [0.0, 0.0],
            'belief_corruption_rate': belief_rate,
            'belief_corruption_rate_ci': list(bootstrap_ci_binary(belief_flags, seed=seed + 2)) if belief_flags else [0.0, 0.0],
            'robust_rate': robust_rate,
            'robust_rate_ci': list(bootstrap_ci_binary(robust_flags, seed=seed + 3)) if robust_flags else [0.0, 0.0],
            'other_count': int(n_other),
            'total_classified': int(total_classified),
            'dominant_pattern': 'social_compliance' if social_rate >= belief_rate else 'belief_corruption',
        }

    # Output accuracy (honest on biased prompts)
    valid_mask = [x is not None for x in output_is_sycophantic]
    if any(valid_mask):
        output_honest = [int(not x) for x in output_is_sycophantic if x is not None]
        output_acc = float(np.mean(output_honest))
    else:
        output_acc = 0.0

    best_layer = max(per_layer.keys(), key=lambda l: per_layer[l].get('biased_transfer_accuracy', 0.0))
    best = per_layer[best_layer]

    summary = {
        'analysis_mode': 'neutral_transfer',
        'position': position_name,
        'n_samples': int(len(valid_samples)),
        'output_accuracy_on_biased': output_acc,
        'sycophancy_rate': float(1.0 - output_acc),
        'best_layer': int(best_layer),
        'best_neutral_cv_accuracy': float(best.get('neutral_cv_accuracy', 0.0)),
        'best_biased_transfer_accuracy': float(best.get('biased_transfer_accuracy', 0.0)),
        'best_biased_transfer_accuracy_ci': best.get('biased_transfer_accuracy_ci', [0.0, 0.0]),
        'best_social_compliance_rate': float(best.get('social_compliance_rate', 0.0)),
        'best_belief_corruption_rate': float(best.get('belief_corruption_rate', 0.0)),
        'best_robust_rate': float(best.get('robust_rate', 0.0)),
        'dominant_pattern_best_layer': best.get('dominant_pattern', 'unknown'),
        'class_balance': class_balance,
    }

    return per_layer, summary


def run_mixed_diagnostic_position(
    model: SycophancyModel,
    valid_samples: List[Dict],
    layers: List[int],
    position_name: str,
    probe_type: str,
    batch_size: int,
    n_folds: int,
    seed: int,
) -> Tuple[Dict, Dict]:
    if position_name == "final":
        neutral_positions = [s['neutral_final_pos'] for s in valid_samples]
        biased_positions = [s['biased_final_pos'] for s in valid_samples]
    else:
        neutral_positions = [s['neutral_answer_pos'] for s in valid_samples]
        biased_positions = [s['biased_answer_pos'] for s in valid_samples]

    neutral_prompts = [s['neutral_prompt'] for s in valid_samples]
    biased_prompts = [s['biased_prompt'] for s in valid_samples]
    sample_ids = [s['sample_id'] for s in valid_samples]

    neutral_acts = extract_activations_batch(model, neutral_prompts, layers, neutral_positions, batch_size)
    biased_acts = extract_activations_batch(model, biased_prompts, layers, biased_positions, batch_size)

    per_layer = {}

    for layer in tqdm(layers, desc=f"Mixed-diagnostic probes ({position_name})"):
        X_neutral = neutral_acts[layer]
        X_biased = biased_acts[layer]

        X = np.concatenate([X_neutral, X_biased], axis=0)
        y = np.concatenate([
            np.zeros(len(X_neutral), dtype=int),
            np.ones(len(X_biased), dtype=int),
        ])
        groups = np.array(sample_ids + sample_ids)

        n_splits = min(n_folds, len(np.unique(groups)))
        if n_splits < 2:
            per_layer[str(layer)] = {
                'error': 'Insufficient unique groups for GroupKFold',
                'mixed_cv_accuracy': 0.0,
            }
            continue

        gkf = GroupKFold(n_splits=n_splits)
        fold_accs = []
        oof_preds = np.full(len(y), -1, dtype=int)

        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            probe = make_probe(probe_type, seed)
            probe.fit(X_train_scaled, y_train)

            pred = probe.predict(X_val_scaled)
            fold_accs.append(float(accuracy_score(y_val, pred)))
            oof_preds[val_idx] = pred

        oof_correct = [int(oof_preds[i] == y[i]) for i in range(len(y)) if oof_preds[i] != -1]
        mixed_ci = bootstrap_ci_binary(oof_correct, seed=seed)
        mixed_acc = float(np.mean(fold_accs)) if fold_accs else 0.0

        per_layer[str(layer)] = {
            'mixed_cv_accuracy': mixed_acc,
            'mixed_cv_std': float(np.std(fold_accs)) if fold_accs else 0.0,
            'mixed_fold_accuracies': fold_accs,
            'mixed_cv_accuracy_ci': [float(mixed_ci[0]), float(mixed_ci[1])],
            'group_leakage_safe': True,
            'splitter': 'GroupKFold',
            'format_confounded_risk': mixed_acc >= 0.80,
            'diagnostic_note': 'High mixed accuracy may indicate prompt-format discrimination, not truth tracking.',
        }

    best_layer = max(per_layer.keys(), key=lambda l: per_layer[l].get('mixed_cv_accuracy', 0.0))
    best = per_layer[best_layer]

    summary = {
        'analysis_mode': 'mixed_diagnostic',
        'position': position_name,
        'n_samples': int(len(valid_samples)),
        'best_layer': int(best_layer),
        'best_mixed_cv_accuracy': float(best.get('mixed_cv_accuracy', 0.0)),
        'best_mixed_cv_accuracy_ci': best.get('mixed_cv_accuracy_ci', [0.0, 0.0]),
        'format_confounded_risk': bool(best.get('format_confounded_risk', False)),
        'split_definition': 'GroupKFold by sample_id (paired neutral/biased samples stay in same fold)',
    }

    return per_layer, summary


def run_probe_pipeline(
    model: SycophancyModel,
    dataset: List[Dict],
    layers: List[int],
    probe_position: str = "final",
    probe_type: str = "logistic",
    analysis_mode: str = "neutral_transfer",
    batch_size: int = 8,
    n_folds: int = 5,
    seed: int = RANDOM_SEED,
) -> Dict:
    positions_to_run = []
    if probe_position in ("final", "both"):
        positions_to_run.append("final")
    if probe_position in ("answer_token", "both"):
        positions_to_run.append("answer_token")

    valid_samples = prepare_valid_samples(model, dataset, probe_position)
    if len(valid_samples) < 20:
        return {'error': f'Too few valid samples ({len(valid_samples)}). Need at least 20.'}

    all_layer_results = {}
    all_position_summary = {}

    for pos in positions_to_run:
        if analysis_mode == 'neutral_transfer':
            per_layer, summary = run_neutral_transfer_position(
                model=model,
                valid_samples=valid_samples,
                layers=layers,
                position_name=pos,
                probe_type=probe_type,
                batch_size=batch_size,
                n_folds=n_folds,
                seed=seed,
            )
        elif analysis_mode == 'mixed_diagnostic':
            per_layer, summary = run_mixed_diagnostic_position(
                model=model,
                valid_samples=valid_samples,
                layers=layers,
                position_name=pos,
                probe_type=probe_type,
                batch_size=batch_size,
                n_folds=n_folds,
                seed=seed,
            )
        else:
            return {'error': f'Unknown analysis_mode: {analysis_mode}'}

        all_layer_results[pos] = per_layer
        all_position_summary[pos] = summary

    return {
        'schema_version': DEFAULT_SCHEMA_VERSION,
        'analysis_mode': analysis_mode,
        'split_definition': (
            'neutral_transfer: StratifiedKFold on neutral only; transfer evaluated on biased holdout indices'
            if analysis_mode == 'neutral_transfer'
            else 'mixed_diagnostic: GroupKFold on mixed neutral+biased with grouping by sample_id'
        ),
        'n_samples': int(len(valid_samples)),
        'n_layers_probed': int(len(layers)),
        'probe_type': probe_type,
        'probe_positions': positions_to_run,
        'n_folds': n_folds,
        'per_position_summary': all_position_summary,
        'per_layer': all_layer_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train linear probes for sycophancy mechanism analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    parser.add_argument('--analysis-mode', type=str, default='neutral_transfer',
                        choices=['neutral_transfer', 'mixed_diagnostic'],
                        help=(
                            "neutral_transfer: train on neutral, test transfer on biased (claim-bearing); "
                            "mixed_diagnostic: condition classifier with group-safe folds (diagnostic only)"
                        ))
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


def write_manifest(
    output_path: str,
    model_name: str,
    data_path: str,
    seed: int,
    started_at: datetime,
    ended_at: datetime,
):
    manifest = {
        'script': 'scripts/02_train_probes.py',
        'status': 'completed',
        'command': ' '.join(sys.argv),
        'git_hash': get_git_hash(),
        'model': model_name,
        'data': data_path,
        'seed': seed,
        'started_at': started_at.isoformat(),
        'ended_at': ended_at.isoformat(),
        'artifacts': [output_path],
    }

    output = Path(output_path)
    manifest_dir = output.parent / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"probe_train_{ended_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def main():
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    set_seeds(args.seed)

    print("=" * 80)
    print("LINEAR PROBE TRAINING")
    print("=" * 80)
    print(f"Model:          {args.model}")
    print(f"Data:           {args.data}")
    print(f"Max Samples:    {args.max_samples or 'all'}")
    print(f"Probe Position: {args.probe_position}")
    print(f"Probe Type:     {args.probe_type}")
    print(f"Analysis Mode:  {args.analysis_mode}")
    print(f"CV Folds:       {args.n_folds}")
    print(f"Seed:           {args.seed}")
    print(f"Git Hash:       {get_git_hash() or 'N/A'}")
    print()

    dataset = load_data(args.data, args.max_samples)
    print(f"Loaded {len(dataset)} samples\n")

    model = SycophancyModel(args.model, device=args.device)

    if args.layers is None or args.layers == "all":
        layers = list(range(model.n_layers))
    else:
        layers = [int(l.strip()) for l in args.layers.split(",")]

    print(f"Probing {len(layers)} layers: {layers[0]}..{layers[-1]}\n")

    results = run_probe_pipeline(
        model=model,
        dataset=dataset,
        layers=layers,
        probe_position=args.probe_position,
        probe_type=args.probe_type,
        analysis_mode=args.analysis_mode,
        batch_size=args.batch_size,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    if 'error' in results:
        print(f"Error: {results['error']}")
        return 1

    output = {
        'metadata': {
            'model_name': args.model,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data_path': args.data,
            'probe_position': args.probe_position,
            'probe_type': args.probe_type,
            'analysis_mode': args.analysis_mode,
            'n_folds': args.n_folds,
            'random_seed': args.seed,
            'git_hash': get_git_hash(),
            'environment': get_environment_info(),
        },
        **results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {args.output}")

    for pos in output['probe_positions']:
        s = output['per_position_summary'][pos]
        print(f"\n--- Position: {pos} ---")
        if args.analysis_mode == 'neutral_transfer':
            print(f"Best layer: {s['best_layer']}")
            print(f"Best transfer accuracy: {s['best_biased_transfer_accuracy']:.1%}")
            ci = s['best_biased_transfer_accuracy_ci']
            print(f"Transfer 95% CI: [{ci[0]:.1%}, {ci[1]:.1%}]")
            print(f"Social compliance: {s['best_social_compliance_rate']:.1%}")
            print(f"Belief corruption: {s['best_belief_corruption_rate']:.1%}")
            print(f"Dominant pattern: {s['dominant_pattern_best_layer']}")
        else:
            print(f"Best layer: {s['best_layer']}")
            print(f"Best mixed CV accuracy: {s['best_mixed_cv_accuracy']:.1%}")
            ci = s['best_mixed_cv_accuracy_ci']
            print(f"Mixed 95% CI: [{ci[0]:.1%}, {ci[1]:.1%}]")
            print(f"Format-confounded risk: {s['format_confounded_risk']}")

    ended_at = datetime.now(timezone.utc)
    write_manifest(
        output_path=args.output,
        model_name=args.model,
        data_path=args.data,
        seed=args.seed,
        started_at=started_at,
        ended_at=ended_at,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
