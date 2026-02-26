#!/bin/bash
#SBATCH --job-name=syc-probes
#SBATCH --output=slurm/logs/probes_%j.out
#SBATCH --error=slurm/logs/probes_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 3: Linear Probes — Social Compliance vs. Belief Corruption
#
# Extracts resid_post at all 32 layers for 1,500 samples (neutral + biased),
# then trains logistic regression probes with 5-fold CV at each layer.
#
# This is the most memory-intensive job: ~40GB activations cached.
# Requests 80GB RAM to be safe.
#
# Expected runtime: ~4-6 hours on A100
# Output: results/probe_results_llama3.json
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --gres=gpu:${GPUS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=80G

if [ -n "${SLURM_ACCOUNT}" ] && [ "${SLURM_ACCOUNT}" != "TODO_ACCOUNT" ]; then
    export SBATCH_ACCOUNT="${SLURM_ACCOUNT}"
fi

module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export TOKENIZERS_PARALLELISM=false

mkdir -p results slurm/logs

echo "============================================"
echo "SLURM Job: Linear Probes"
echo "Model: ${PRIMARY_MODEL}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

# Run probes at both positions with logistic regression
python scripts/02_train_probes.py \
    --model "${PRIMARY_MODEL}" \
    --data data/processed/master_sycophancy.jsonl \
    --probe-position both \
    --probe-type logistic \
    --batch-size 8 \
    --n-folds 5 \
    --seed 42 \
    --output results/probe_results_llama3_logistic.json

echo "--- Logistic probes done, now running Ridge ---"

# Also run with Ridge for comparison (proposal says test both)
python scripts/02_train_probes.py \
    --model "${PRIMARY_MODEL}" \
    --data data/processed/master_sycophancy.jsonl \
    --probe-position both \
    --probe-type ridge \
    --batch-size 8 \
    --n-folds 5 \
    --seed 42 \
    --output results/probe_results_llama3_ridge.json

echo "============================================"
echo "Probes complete: $(date)"
echo "============================================"
