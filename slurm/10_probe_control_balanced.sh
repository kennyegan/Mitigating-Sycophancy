#!/bin/bash
#SBATCH --job-name=syc-probe-bal
#SBATCH --output=slurm/logs/probe_bal_%j.out
#SBATCH --error=slurm/logs/probe_bal_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --account=pi_larsonj_wit_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

# =============================================================================
# Job 10: Balanced Probe Control — Class Balance Fix
#
# Regenerates data with randomized (A)/(B) positions for TruthfulQA and GSM8k,
# then reruns probe control experiment. This fixes the degenerate label
# distribution (100% class 1) for these sources.
#
# Phase 1: Data regeneration (~5 min, CPU only)
# Phase 2: Probe control with balanced data (~1-2h, GPU)
#
# Output:
#   data/processed/master_sycophancy_balanced.jsonl
#   results/probe_control_balanced_results.json
# =============================================================================

set -euo pipefail

source "/home/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"

module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export TOKENIZERS_PARALLELISM=false

mkdir -p data/processed results slurm/logs

echo "============================================"
echo "SLURM Job: Balanced Probe Control"
echo "Model: ${PRIMARY_MODEL}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

echo ""
echo "[Phase 1] Regenerating data with randomized answer positions..."
echo ""

python scripts/00_data_setup.py \
    --randomize-positions \
    --output data/processed/master_sycophancy_balanced.jsonl \
    --seed 42

echo ""
echo "[Phase 2] Running probe control on balanced data..."
echo ""

python scripts/02b_probe_control.py \
    --model "${PRIMARY_MODEL}" \
    --data data/processed/master_sycophancy_balanced.jsonl \
    --output results/probe_control_balanced_results.json \
    --seed 42

echo "============================================"
echo "Balanced probe control complete: $(date)"
echo "============================================"
