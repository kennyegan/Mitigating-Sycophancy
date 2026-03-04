#!/bin/bash
#SBATCH --job-name=syc-steer
#SBATCH --output=slurm/logs/steering_%j.out
#SBATCH --error=slurm/logs/steering_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 12: Representation Steering
#
# Computes contrastive steering vectors (biased - neutral activations) at
# multiple layers and evaluates sycophancy reduction across an alpha sweep.
# Uses seed-controlled shuffle to split data: first 200 samples for steering
# vector computation, remaining ~1300 for evaluation.
#
# Conditions: 8 layers x 7 alphas (single-layer) + 7 alphas (multi-layer L1-5)
#             + baseline = ~64 sycophancy evals + selective capability evals
#
# Expected runtime: ~4-8 hours on A100
# Output: results/steering_results.json
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

STEERING_OUT="results/steering_results.json"
STEERING_CKPT="${STEERING_OUT}.checkpoint.json"
RESUME_FLAG=()
if [ -s "${STEERING_CKPT}" ]; then
    echo "Existing checkpoint detected, resuming: ${STEERING_CKPT}"
    RESUME_FLAG=(--resume-from-checkpoint)
fi

echo "============================================"
echo "SLURM Job: Representation Steering"
echo "Model: ${PRIMARY_MODEL}"
echo "Layers: 1,2,3,4,5,10,15,20"
echo "Alphas: 0.5,1.0,2.0,5.0,10.0,20.0,50.0"
echo "Steering samples: 200"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

python scripts/05_representation_steering.py \
    --model "${PRIMARY_MODEL}" \
    --data data/processed/master_sycophancy.jsonl \
    --layers "1,2,3,4,5,10,15,20" \
    --alphas "0.5,1.0,2.0,5.0,10.0,20.0,50.0" \
    --n-steering-samples 200 \
    --eval-capabilities \
    --mmlu-samples 500 \
    --gsm8k-samples 1319 \
    --seed 42 \
    --checkpoint-path "${STEERING_CKPT}" \
    "${RESUME_FLAG[@]}" \
    --save-every-condition \
    --output "${STEERING_OUT}"

check_artifact "${STEERING_OUT}"
check_artifact "${STEERING_CKPT}"

echo "============================================"
echo "Representation steering complete: $(date)"
echo "============================================"
