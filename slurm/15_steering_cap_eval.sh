#!/bin/bash
#SBATCH --job-name=syc-steer-cap
#SBATCH --output=slurm/logs/steer_cap_%j.out
#SBATCH --error=slurm/logs/steer_cap_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 15: Targeted Capability Eval for L15/L20 alpha=2.0 Steering
#
# These two conditions showed the best opinion-domain sycophancy reduction
# (-6.9pp and -5.7pp) without catastrophic side-effects in other domains.
# This job evaluates MMLU (500) and GSM8k (200) retention for each.
#
# Expected runtime: ~1-2 hours on A100
# Output: updates results/steering_results.json in-place
# =============================================================================

set -euo pipefail

source "/home/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"

module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export TOKENIZERS_PARALLELISM=false

mkdir -p slurm/logs

echo "============================================"
echo "SLURM Job: Steering Capability Eval (Targeted)"
echo "Model: ${PRIMARY_MODEL}"
echo "Conditions: layer15_alpha2.0, layer20_alpha2.0"
echo "MMLU samples: 500"
echo "GSM8k samples: 200"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

python scripts/eval_steering_capability.py \
    --model "${PRIMARY_MODEL}" \
    --data data/processed/master_sycophancy.jsonl \
    --conditions "layer15_alpha2.0,layer20_alpha2.0" \
    --steering-results results/steering_results.json \
    --checkpoint results/steering_results.json.checkpoint.json \
    --mmlu-samples 500 \
    --gsm8k-samples 200 \
    --seed 42

echo "============================================"
echo "Steering capability eval complete: $(date)"
echo "============================================"
