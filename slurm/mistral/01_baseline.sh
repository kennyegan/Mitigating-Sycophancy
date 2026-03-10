#!/bin/bash
#SBATCH --job-name=mistral-baseline
#SBATCH --output=slurm/logs/mistral_baseline_%j.out
#SBATCH --error=slurm/logs/mistral_baseline_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Mistral Job M1: Baseline Evaluation
# Output: results/mistral/baseline_summary.json, results/mistral/baseline_detailed.csv
# =============================================================================

set -euo pipefail
source "/home/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"

module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export TOKENIZERS_PARALLELISM=false

mkdir -p results/mistral slurm/logs

check_artifact() {
    local path="$1"
    if [ ! -s "${path}" ]; then
        echo "ERROR: expected artifact missing or empty: ${path}"
        exit 2
    fi
}

echo "============================================"
echo "SLURM Job: Mistral Baseline Evaluation"
echo "Model: ${REPLICATION_MODEL}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

python scripts/01_run_baseline.py \
    --model "${REPLICATION_MODEL}" \
    --data data/processed/master_sycophancy.jsonl \
    --output-json results/mistral/baseline_summary.json \
    --output-csv results/mistral/baseline_detailed.csv \
    --seed 42 \
    --run-sanity-checks

check_artifact results/mistral/baseline_summary.json
check_artifact results/mistral/baseline_detailed.csv

echo "============================================"
echo "Mistral baseline complete: $(date)"
echo "============================================"
