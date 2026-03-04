# Mitigating Sycophancy in LLMs

Mechanistic sycophancy research pipeline with reproducible baselines, probes, patching, ablation, and steering.

## Status

As of March 3, 2026, the codebase has been upgraded to a stricter methodology:

- Length-normalized confidence reporting in baseline evaluation
- Deterministic `sample_id` and randomized answer-position support (`--randomize-positions`)
- Probe redesign with explicit modes:
  - `neutral_transfer` (default, claim-bearing)
  - `mixed_diagnostic` (format-confound diagnostic)
- Leakage-safe grouped folds in mixed mode
- Steering checkpoint/resume:
  - `--checkpoint-path`
  - `--resume-from-checkpoint`
  - `--save-every-condition`
- Capability scoring upgrades:
  - MMLU tokenization-variant robust scoring
  - GSM8k strict normalized numeric extraction from generated completion
  - confidence intervals for capability metrics/retention
- SLURM artifact contracts and consolidated rerun manifest (`scripts/99_collect_result_manifest.py`)

Numerical claims in `paper.md` are currently conservative/provisional until full corrected reruns complete.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Core Pipeline

```bash
# 1) Build benchmark data
python scripts/00_data_setup.py --samples 500

# Optional: balanced A/B randomization for synthetic domains
python scripts/00_data_setup.py --samples 500 --randomize-positions \
  --output data/processed/master_sycophancy_balanced.jsonl

# 2) Baseline
python scripts/01_run_baseline.py --model meta-llama/Meta-Llama-3-8B-Instruct

# 3) Probes (claim-bearing)
python scripts/02_train_probes.py \
  --analysis-mode neutral_transfer \
  --probe-position both \
  --output results/probe_results_neutral_transfer.json

# 4) Probes (diagnostic)
python scripts/02_train_probes.py \
  --analysis-mode mixed_diagnostic \
  --probe-position both \
  --output results/probe_results_mixed_diagnostic.json

# 5) Probe-control entrypoint (aligned neutral-transfer semantics)
python scripts/02b_probe_control.py --output results/probe_control_results.json

# 6) Patching
python scripts/03_activation_patching.py --output-dir results

# 7) Head ablation
python scripts/04_head_ablation.py --eval-capabilities --output results/head_ablation_results.json

# 8) Steering (checkpointable)
python scripts/05_representation_steering.py \
  --eval-capabilities \
  --checkpoint-path results/steering_results.json.checkpoint.json \
  --save-every-condition \
  --output results/steering_results.json
```

## SLURM

Use static-safe job scripts with resource settings passed at submit time:

```bash
bash slurm/submit_all.sh
```

This submits Jobs 1–13 (including consolidated manifest generation).

## Results Manifest

After runs finish:

```bash
python scripts/99_collect_result_manifest.py --output results/full_rerun_manifest.json
```

The manifest validates expected artifacts and records key metrics from available outputs.
