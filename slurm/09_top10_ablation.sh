#!/bin/bash
#SBATCH --job-name=syc-top10-abl
#SBATCH --output=slurm/logs/top10_abl_%j.out
#SBATCH --error=slurm/logs/top10_abl_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --account=pi_larsonj_wit_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

# =============================================================================
# Job 9: Top-10 Head Ablation — Circuit Redundancy Test
#
# Tests whether ablating the top 10 patching-identified heads (instead of 3)
# reduces sycophancy. If still null, confirms distributed/redundant circuit.
#
# Heads: L1H20, L5H5, L4H28, L5H17, L3H17, L5H4, L5H19, L5H24, L4H5, L3H0
# Uses --all-only flag to skip single/pairwise conditions (only baseline + all-10)
#
# Expected runtime: ~3-4 hours on A100
# Output: results/top10_ablation_results.json
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

echo "============================================"
echo "SLURM Job: Top-10 Head Ablation"
echo "Model: ${PRIMARY_MODEL}"
echo "Heads: L1H20,L5H5,L4H28,L5H17,L3H17,L5H4,L5H19,L5H24,L4H5,L3H0"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

python scripts/04_head_ablation.py \
    --model "${PRIMARY_MODEL}" \
    --data data/processed/master_sycophancy.jsonl \
    --heads "L1H20,L5H5,L4H28,L5H17,L3H17,L5H4,L5H19,L5H24,L4H5,L3H0" \
    --all-only \
    --eval-capabilities \
    --mmlu-samples 500 \
    --gsm8k-samples 200 \
    --seed 42 \
    --output results/top10_ablation_results.json

echo "============================================"
echo "Top-10 ablation complete: $(date)"
echo "============================================"
