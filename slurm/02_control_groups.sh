#!/bin/bash
#SBATCH --job-name=syc-controls
#SBATCH --output=slurm/logs/controls_%j.out
#SBATCH --error=slurm/logs/controls_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 2: Control Groups — Filter with Llama-3-8B-Instruct
#
# Generates:
#   data/processed/control_groups/fictional_entities.jsonl
#   data/processed/control_groups/uncertain_knowledge.jsonl
#   data/processed/control_groups/adversarially_true.jsonl
#
# Expected runtime: ~1-2 hours on A100 (forward passes on 1,500 samples)
# =============================================================================

set -euo pipefail

source "/work/pi_larsonj_wit_edu/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"



module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export TOKENIZERS_PARALLELISM=false

mkdir -p data/processed/control_groups slurm/logs

check_artifact() {
    local path="$1"
    if [ ! -s "${path}" ]; then
        echo "ERROR: expected artifact missing or empty: ${path}"
        exit 2
    fi
}

echo "============================================"
echo "SLURM Job: Control Groups"
echo "Model: ${PRIMARY_MODEL}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

python scripts/00_data_setup.py \
    --samples 500 \
    --seed 42 \
    --control-groups \
    --control-model "${PRIMARY_MODEL}"

check_artifact data/processed/control_groups/fictional_entities.jsonl
check_artifact data/processed/control_groups/uncertain_knowledge.jsonl
check_artifact data/processed/control_groups/adversarially_true.jsonl
check_artifact data/processed/control_groups/control_groups_metadata.json

echo "============================================"
echo "Control groups complete: $(date)"
echo "============================================"
