#!/bin/bash
#SBATCH --job-name=syc-baseline
#SBATCH --output=slurm/logs/baseline_%j.out
#SBATCH --error=slurm/logs/baseline_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
# GPU/CPU/MEM are set dynamically below

# =============================================================================
# Job 1: Baseline Evaluation — Llama-3-8B-Instruct, full 1,500 samples
#
# Expected runtime: ~2-3 hours on A100
# Output: results/baseline_llama3_summary.json, results/baseline_llama3_detailed.csv
# =============================================================================

set -euo pipefail

# Load cluster config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# Apply SLURM resource settings from config
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --gres=gpu:${GPUS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}

# Account/QOS (only if set)
if [ -n "${SLURM_ACCOUNT}" ] && [ "${SLURM_ACCOUNT}" != "TODO_ACCOUNT" ]; then
    export SBATCH_ACCOUNT="${SLURM_ACCOUNT}"
fi

# Setup environment
module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

# Cache settings
export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export TOKENIZERS_PARALLELISM=false

# Create output dirs
mkdir -p results slurm/logs

echo "============================================"
echo "SLURM Job: Baseline Evaluation"
echo "Model: ${PRIMARY_MODEL}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

# Run baseline on full 1,500-sample dataset
python scripts/01_run_baseline.py \
    --model "${PRIMARY_MODEL}" \
    --data data/processed/master_sycophancy.jsonl \
    --output-json results/baseline_llama3_summary.json \
    --output-csv results/baseline_llama3_detailed.csv \
    --seed 42 \
    --run-sanity-checks

echo "============================================"
echo "Baseline complete: $(date)"
echo "============================================"
