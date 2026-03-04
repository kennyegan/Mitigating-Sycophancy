#!/bin/bash
#SBATCH --job-name=syc-base
#SBATCH --output=slurm/logs/base_%j.out
#SBATCH --error=slurm/logs/base_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 5: Base vs. Instruct Comparison (RLHF Hypothesis)
#
# Runs baseline + probes on Llama-3-8B (base, no RLHF).
# Hypothesis: base model shows Belief Corruption or no sycophancy effect,
# while Instruct model shows Social Compliance.
#
# Expected runtime: ~8-10 hours on A100 (baseline + probes)
# Output: results/baseline_llama3_base_*.json, results/probe_results_llama3_base_*.json
# =============================================================================

set -euo pipefail

source "/home/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"



module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export TOKENIZERS_PARALLELISM=false

mkdir -p results slurm/logs

check_artifact() {
    local path="$1"
    if [ ! -s "${path}" ]; then
        echo "ERROR: expected artifact missing or empty: ${path}"
        exit 2
    fi
}

echo "============================================"
echo "SLURM Job: Base vs. Instruct Comparison"
echo "Model: ${BASE_MODEL}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

# Step 1: Baseline on base model
echo "[1/3] Running baseline on base model..."
python scripts/01_run_baseline.py \
    --model "${BASE_MODEL}" \
    --data data/processed/master_sycophancy.jsonl \
    --output-json results/baseline_llama3_base_summary.json \
    --output-csv results/baseline_llama3_base_detailed.csv \
    --seed 42

check_artifact results/baseline_llama3_base_summary.json
check_artifact results/baseline_llama3_base_detailed.csv

# Step 2: Probes on base model
echo "[2/3] Running logistic probes on base model..."
python scripts/02_train_probes.py \
    --model "${BASE_MODEL}" \
    --data data/processed/master_sycophancy.jsonl \
    --probe-position both \
    --probe-type logistic \
    --analysis-mode neutral_transfer \
    --batch-size 8 \
    --n-folds 5 \
    --seed 42 \
    --output results/probe_results_llama3_base_neutral_transfer.json

check_artifact results/probe_results_llama3_base_neutral_transfer.json

# Step 3: Patching on base model (smaller sample for comparison)
echo "[3/3] Running activation patching on base model..."
python scripts/03_activation_patching.py \
    --model "${BASE_MODEL}" \
    --data data/processed/master_sycophancy.jsonl \
    --max-samples 50 \
    --max-positions 50 \
    --head-top-k 5 \
    --seed 42 \
    --output-dir results/base_model

check_artifact results/base_model/patching_heatmap.json
check_artifact results/base_model/head_importance.json

echo "============================================"
echo "Base comparison complete: $(date)"
echo "============================================"
