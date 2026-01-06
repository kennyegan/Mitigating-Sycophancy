# Experiment Scripts

This directory contains the experiment pipeline scripts for sycophancy research.

## Script Execution Order

### Phase 1: Data Preparation

#### `00_data_setup.py` - Multi-Dataset Download & Processing
**Purpose:** Download and process samples from all three sycophancy datasets.

**Usage:**
```bash
# Default: 500 samples per dataset (1500 total)
python scripts/00_data_setup.py

# Custom sample count
python scripts/00_data_setup.py --samples 100

# Save individual dataset files too
python scripts/00_data_setup.py --save-individual

# Custom output path
python scripts/00_data_setup.py --output data/custom_path.jsonl
```

**Output:**
- `data/processed/master_sycophancy.jsonl` - Unified dataset
- `data/processed/master_sycophancy_metadata.json` - Dataset statistics

**Datasets Processed:**
1. **Opinion Sycophancy** (Anthropic/model-written-evals)
2. **Factual Sycophancy** (TruthfulQA)
3. **Reasoning Sycophancy** (GSM8k)

---

### Phase 2: Baseline Evaluation

#### `01_run_baseline.py` - Compliance Gap Baseline Evaluation
**Purpose:** Evaluate sycophancy using the industry-standard **Compliance Gap** metric.

**Key Metric:**
```
Delta = P(Sycophantic | Biased) - P(Sycophantic | Neutral)
```
This measures how much user bias shifts the model toward sycophantic responses, distinguishing true sycophancy from baseline model tendencies.

**Usage:**
```bash
# Run with default settings (150 samples, Llama-3-8B)
python scripts/01_run_baseline.py
```

**Configuration** (edit the script):
```python
DATA_PATH = "data/processed/master_sycophancy.jsonl"
CSV_OUTPUT = "results/detailed_results.csv"
JSON_OUTPUT = "results/baseline_summary.json"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # or "gpt2" for testing
MAX_SAMPLES = 150  # 50 from each dataset type
```

**Outputs:**
- `results/detailed_results.csv` - Per-sample data with all probabilities
- `results/baseline_summary.json` - Aggregated stats and top sycophantic prompts

**CSV Columns:**
- `source` - Dataset type (anthropic_opinion/truthfulqa_factual/gsm8k_reasoning)
- `prompt_preview` - First 100 chars of biased prompt
- `sycophantic_target` / `honest_target` - Token targets
- `neutral_prob_syc` / `neutral_prob_honest` - Probabilities without user bias
- `biased_prob_syc` / `biased_prob_honest` - Probabilities with user bias
- `compliance_gap` - The key metric (biased_prob_syc - neutral_prob_syc)
- `is_sycophantic` - Boolean (biased_prob_syc > biased_prob_honest)

**JSON Summary Contains:**
- Model metadata and timestamp
- Overall sycophancy rate and mean compliance gap
- Per-source breakdown with standard errors
- Top 5 most sycophantic prompts (highest compliance gap)

---

## Complete Workflow

```bash
# Step 1: Download data (only need to run once)
python scripts/00_data_setup.py --samples 500

# Step 2: Run baseline evaluation
python scripts/01_run_baseline.py

# Step 3: Analyze results
cat results/baseline_summary.json | python -m json.tool
head results/detailed_results.csv
```

---

## Quick Testing

For rapid iteration without GPU:

```bash
# Generate small dataset
python scripts/00_data_setup.py --samples 10

# Edit 01_run_baseline.py:
#   MODEL_NAME = "gpt2"
#   MAX_SAMPLES = 30

# Run baseline
python scripts/01_run_baseline.py
```

---

## Future Scripts (Planned)

- `02_train_probes.py` - Linear probe training for compliance detection
- `03_activation_patching.py` - Circuit discovery via activation patching
- `04_steering_vectors.py` - Compute and test steering vectors
- `05_safety_evaluation.py` - MMLU + refusal rate checks

---

## Notes

- All scripts use `seed=42` for reproducibility
- Data download only needs to be run once
- Master dataset format is standardized via `SycophancySample` dataclass
- All prompts use Llama-3 Instruct chat template
