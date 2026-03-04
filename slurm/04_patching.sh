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

check_artifact results/patching_heatmap.json
check_artifact results/head_importance.json

echo "============================================"
echo "Patching complete: $(date)"
echo "============================================"
