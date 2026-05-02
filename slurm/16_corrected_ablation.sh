#!/bin/bash
#SBATCH --job-name=syc-abl-corrected
#SBATCH --output=slurm/logs/abl_corrected_%j.out
#SBATCH --error=slurm/logs/abl_corrected_%j.err
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 16: CORRECTED Head Ablation — Validated Top-3 Heads
#
# The original ablation (Jobs 7, 9) targeted L1H20, L5H5, L4H28 based on a
# stale patching run. The validated rerun (Mar 4) identifies different top heads:
#
#   Validated Top-3:  L4H28 (0.443), L4H5 (0.302), L5H31 (0.256)
#   Stale Top-3:      L1H20 (0.569→actual 0.040), L5H5 (0.567→actual -0.237)
#
# L4H28 was already ablated (no effect: 28.1% vs 28.0%). This job tests L4H5
# and L5H31 individually and all three together.
#
# THIS IS THE MOST IMPORTANT EXPERIMENT IN THE PROJECT.
# If ablation of the correct top-3 still shows no sycophancy reduction,
# the patching-to-ablation dissociation is validated with the right heads.
# If sycophancy drops, the entire paper's conclusion changes.
#
# Expected runtime: ~4-6 hours on A100
# Output: results/corrected_ablation_results.json
# =============================================================================

set -euo pipefail

source "/work/pi_larsonj_wit_edu/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"

module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

mkdir -p results slurm/logs

echo "============================================"
echo "SLURM Job: CORRECTED Head Ablation"
echo "Model: ${PRIMARY_MODEL}"
echo "Heads: L4H28, L4H5, L5H31 (validated top-3)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

python scripts/04_head_ablation.py \
    --model "${PRIMARY_MODEL}" \
    --data data/processed/master_sycophancy.jsonl \
    --heads "L4H28,L4H5,L5H31" \
    --eval-capabilities \
    --mmlu-samples 500 \
    --gsm8k-samples 200 \
    --seed 42 \
    --output results/corrected_ablation_results.json

echo "============================================"
echo "Corrected ablation complete: $(date)"
echo "============================================"
