# Quick Start Guide

## Complete Research Pipeline

### 1. Setup (One-time)
```bash
make setup
```

### 2. Download Data
```bash
# Full dataset (1500 samples: 500 per type)
make data

# OR small dataset for testing (150 samples: 50 per type)
make data-small
```

### 3. Run Baseline Evaluation
```bash
make baseline
```

---

## Direct Script Usage

### Generate Master Dataset
```bash
# 500 samples per dataset type
python scripts/00_data_setup.py --samples 500

# Custom configuration
python scripts/00_data_setup.py --samples 100 --output data/custom.jsonl
```

### Run Baseline Evaluation
```bash
python scripts/01_run_baseline.py
```

To customize, edit `scripts/01_run_baseline.py`:
- `MODEL_NAME` - Change to `"gpt2"` for CPU testing
- `MAX_SAMPLES` - Number of samples to evaluate

---

## Expected Workflow

```bash
# First time setup
make setup

# Download data (only need to do once)
make data

# Run experiments
make baseline

# View results
cat results/baseline_results.csv
```

---

## Dataset Types

The master dataset contains three types of sycophancy:

| Type | Source | Description |
|------|--------|-------------|
| **Opinion** | Anthropic/model-written-evals | Tests if model agrees with user beliefs |
| **Factual** | TruthfulQA | Tests if model agrees with misconceptions |
| **Reasoning** | GSM8k | Tests if model follows flawed logic |

---

## Output Files

After running the pipeline:

```
data/processed/
├── master_sycophancy.jsonl         # Unified dataset
└── master_sycophancy_metadata.json # Statistics

results/
└── baseline_results.csv            # Per-sample evaluation results
```

---

## Quick Testing (No GPU Required)

```bash
# Generate small dataset
make data-small

# Edit scripts/01_run_baseline.py:
#   MODEL_NAME = "gpt2"
#   MAX_SAMPLES = 30

# Run
make baseline
```

---

## Troubleshooting

**Dataset not found?**
```bash
make data  # Downloads the dataset
```

**Out of memory?**
- Use `MODEL_NAME = "gpt2"` for testing
- Reduce `MAX_SAMPLES` in baseline script

**Want to re-download data?**
```bash
rm data/processed/master_sycophancy.jsonl
make data
```
