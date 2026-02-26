#!/bin/bash
#SBATCH --job-name=syc-patching
#SBATCH --output=slurm/logs/patching_%j.out
#SBATCH --error=slurm/logs/patching_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 4: Causal Activation Patching — Sycophancy Circuit Discovery
#
# Runs layer × position patching on 100 samples, then head-level patching
# on the top-5 critical layers.
#
# This is the slowest job: hundreds of forward passes per sample.
# 100 samples × 32 layers × ~50 positions = ~160,000 forward passes.
#
# Expected runtime: ~6-10 hours on A100
# Output: results/patching_heatmap.json, results/head_importance.json
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --gres=gpu:${GPUS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}

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
echo "SLURM Job: Activation Patching"
echo "Model: ${PRIMARY_MODEL}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

python scripts/03_activation_patching.py \
    --model "${PRIMARY_MODEL}" \
    --data data/processed/master_sycophancy.jsonl \
    --max-samples 100 \
    --max-positions 50 \
    --head-top-k 5 \
    --seed 42 \
    --output-dir results

echo "============================================"
echo "Patching complete: $(date)"
echo "============================================"
