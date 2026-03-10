#!/bin/bash
#SBATCH --job-name=mistral-probes
#SBATCH --output=slurm/logs/mistral_probes_%j.out
#SBATCH --error=slurm/logs/mistral_probes_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Mistral Job M2: Balanced Probe Control (neutral-transfer)
# Output: results/mistral/probe_control_balanced_results.json
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
echo "SLURM Job: Mistral Probe Control (Balanced)"
echo "Model: ${REPLICATION_MODEL}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

# Use the already-generated balanced dataset
python scripts/02b_probe_control.py \
    --model "${REPLICATION_MODEL}" \
    --data data/processed/master_sycophancy_balanced.jsonl \
    --output results/mistral/probe_control_balanced_results.json \
    --seed 42

check_artifact results/mistral/probe_control_balanced_results.json

echo "============================================"
echo "Mistral probes complete: $(date)"
echo "============================================"
