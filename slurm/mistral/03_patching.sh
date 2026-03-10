#!/bin/bash
#SBATCH --job-name=mistral-patching
#SBATCH --output=slurm/logs/mistral_patching_%j.out
#SBATCH --error=slurm/logs/mistral_patching_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Mistral Job M3: Causal Activation Patching
# Output: results/mistral/patching_heatmap.json, results/mistral/head_importance.json
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
echo "SLURM Job: Mistral Activation Patching"
echo "Model: ${REPLICATION_MODEL}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

python scripts/03_activation_patching.py \
    --model "${REPLICATION_MODEL}" \
    --data data/processed/master_sycophancy.jsonl \
    --max-samples 100 \
    --max-positions 50 \
    --head-top-k 5 \
    --seed 42 \
    --output-dir results/mistral

check_artifact results/mistral/patching_heatmap.json
check_artifact results/mistral/head_importance.json

echo "============================================"
echo "Mistral patching complete: $(date)"
echo "============================================"
