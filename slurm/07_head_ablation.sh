#!/bin/bash
#SBATCH --job-name=syc-ablation
#SBATCH --output=slurm/logs/ablation_%j.out
#SBATCH --error=slurm/logs/ablation_%j.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 7: Head Ablation — Causal Intervention on Sycophancy Heads
#
# Ablates L1H20, L5H5, L4H28 (zero + mean ablation) and measures:
# - Effect on sycophancy rate (primary metric)
# - MMLU accuracy (capability preservation)
# - GSM8k accuracy (reasoning preservation)
#
# 9 conditions × (1500 syc + 500 MMLU + 200 GSM8k) evaluations
#
# Expected runtime: ~12-16 hours on A100
# Output: results/head_ablation_results.json
# =============================================================================

set -euo pipefail

source "/home/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"

#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --gres=gpu:${GPUS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=80G

if [ -n "${SLURM_ACCOUNT}" ] && [ "${SLURM_ACCOUNT}" != "TODO_ACCOUNT" ]; then
    export SBATCH_ACCOUNT="${SLURM_ACCOUNT}"
fi

module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export TOKENIZERS_PARALLELISM=false

mkdir -p results slurm/logs

echo "============================================"
echo "SLURM Job: Head Ablation Experiment"
echo "Model: ${PRIMARY_MODEL}"
echo "Heads: L1H20, L5H5, L4H28"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

python scripts/04_head_ablation.py \
    --model "${PRIMARY_MODEL}" \
    --data data/processed/master_sycophancy.jsonl \
    --heads "L1H20,L5H5,L4H28" \
    --eval-capabilities \
    --mmlu-samples 500 \
    --gsm8k-samples 200 \
    --seed 42 \
    --output results/head_ablation_results.json

echo "============================================"
echo "Head ablation complete: $(date)"
echo "============================================"
