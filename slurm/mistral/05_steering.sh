#!/bin/bash
#SBATCH --job-name=mistral-steer
#SBATCH --output=slurm/logs/mistral_steering_%j.out
#SBATCH --error=slurm/logs/mistral_steering_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Mistral Job M5: Representation Steering
# Output: results/mistral/steering_results.json
# =============================================================================

set -euo pipefail
source "/work/pi_larsonj_wit_edu/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"

module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export TOKENIZERS_PARALLELISM=false

mkdir -p results/mistral slurm/logs

STEERING_MMLU_SAMPLES="${STEERING_MMLU_SAMPLES:-500}"
STEERING_GSM8K_SAMPLES="${STEERING_GSM8K_SAMPLES:-200}"

check_artifact() {
    local path="$1"
    if [ ! -s "${path}" ]; then
        echo "ERROR: expected artifact missing or empty: ${path}"
        exit 2
    fi
}

STEERING_OUT="results/mistral/steering_results.json"
STEERING_CKPT="${STEERING_OUT}.checkpoint.json"
RESUME_FLAG=()
if [ -s "${STEERING_CKPT}" ]; then
    echo "Existing checkpoint detected, resuming: ${STEERING_CKPT}"
    RESUME_FLAG=(--resume-from-checkpoint)
fi

echo "============================================"
echo "SLURM Job: Mistral Representation Steering"
echo "Model: ${REPLICATION_MODEL}"
echo "Layers: 1,2,3,4,5,10,15,20"
echo "Alphas: 0.5,1.0,2.0,5.0,10.0,20.0,50.0"
echo "MMLU samples: ${STEERING_MMLU_SAMPLES}"
echo "GSM8k samples: ${STEERING_GSM8K_SAMPLES}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

python scripts/05_representation_steering.py \
    --model "${REPLICATION_MODEL}" \
    --data data/processed/master_sycophancy.jsonl \
    --layers "1,2,3,4,5,10,15,20" \
    --alphas "0.5,1.0,2.0,5.0,10.0,20.0,50.0" \
    --n-steering-samples 200 \
    --eval-capabilities \
    --mmlu-samples "${STEERING_MMLU_SAMPLES}" \
    --gsm8k-samples "${STEERING_GSM8K_SAMPLES}" \
    --seed 42 \
    --checkpoint-path "${STEERING_CKPT}" \
    "${RESUME_FLAG[@]}" \
    --save-every-condition \
    --output "${STEERING_OUT}"

check_artifact "${STEERING_OUT}"
check_artifact "${STEERING_CKPT}"

echo "============================================"
echo "Mistral steering complete: $(date)"
echo "============================================"
