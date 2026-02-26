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

mkdir -p data/processed/control_groups slurm/logs

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

echo "============================================"
echo "Control groups complete: $(date)"
echo "============================================"
