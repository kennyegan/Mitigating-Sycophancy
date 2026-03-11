# Quick Start

## 1) Environment

```bash
pip install -r requirements.txt
pip install -e .
```

## 2) Data

```bash
# Full benchmark
python scripts/00_data_setup.py --samples 500

# Balanced A/B randomized benchmark (recommended for probe reruns)
python scripts/00_data_setup.py --samples 500 --randomize-positions \
  --output data/processed/master_sycophancy_balanced.jsonl
```

## 3) Baseline

```bash
python scripts/01_run_baseline.py \
  --data data/processed/master_sycophancy.jsonl \
  --output-json results/baseline_llama3_summary.json \
  --output-csv results/baseline_llama3_detailed.csv
```

## 4) Probes

```bash
# Claim-bearing
python scripts/02_train_probes.py \
  --analysis-mode neutral_transfer \
  --probe-position both \
  --output results/probe_results_neutral_transfer.json

# Diagnostic
python scripts/02_train_probes.py \
  --analysis-mode mixed_diagnostic \
  --probe-position both \
  --output results/probe_results_mixed_diagnostic.json
```

## 5) Steering With Resume

```bash
python scripts/05_representation_steering.py \
  --eval-capabilities \
  --checkpoint-path results/steering_results.json.checkpoint.json \
  --save-every-condition \
  --output results/steering_results.json
```

If interrupted:

```bash
python scripts/05_representation_steering.py \
  --resume-from-checkpoint \
  --checkpoint-path results/steering_results.json.checkpoint.json \
  --output results/steering_results.json
```

## 6) SLURM Full Matrix

```bash
bash slurm/submit_all.sh
```

## 7) Consolidated Artifact Manifest

```bash
python scripts/99_collect_result_manifest.py --output results/full_rerun_manifest.json
```
