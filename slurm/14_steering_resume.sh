#!/bin/bash
#SBATCH --job-name=syc-steer-resume
#SBATCH --output=slurm/logs/steering_resume_%j.out
#SBATCH --error=slurm/logs/steering_resume_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 14: Steering Resume — Finish Capability Evaluations
#
# The checkpoint has all 64 sycophancy conditions completed. This job resumes
# from checkpoint to finish the remaining capability evaluations (MMLU + GSM8k)
# for the selected conditions (~5 conditions × MMLU + GSM8k).
#
# Uses 200 GSM8k samples (not 1319) to fit within wall time.
# The checkpoint's resume logic skips already-computed conditions.
#
# Expected runtime: ~4-8 hours on A100
# Output: results/steering_results.json (final, not checkpoint)
# =============================================================================

set -euo pipefail

source "/work/pi_larsonj_wit_edu/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"

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

# Verify checkpoint exists
check_artifact "${STEERING_CKPT}"

echo "============================================"
echo "SLURM Job: Steering Resume (Capability Evals)"
echo "Model: ${PRIMARY_MODEL}"
echo "Resuming from: ${STEERING_CKPT}"
echo "MMLU samples: 500"
echo "GSM8k samples: 200"
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
    --gsm8k-samples 200 \
    --seed 42 \
    --checkpoint-path "${STEERING_CKPT}" \
    --resume-from-checkpoint \
    --save-every-condition \
    --output "${STEERING_OUT}"

check_artifact "${STEERING_OUT}"

echo "============================================"
echo "Steering resume complete: $(date)"
echo "============================================"
