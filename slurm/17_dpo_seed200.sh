#!/bin/bash
#SBATCH --job-name=syc-dpo-s200
#SBATCH --output=slurm/logs/dpo_seed200_%j.out
#SBATCH --error=slurm/logs/dpo_seed200_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --account=pi_larsonj_wit_edu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# =============================================================================
# Job: DPO Training + Eval — Seed 200
#
# Mirrors the seed=100 DPO pipeline (Job 53801949 train, Job 55240703 eval)
# with seed=200 for multi-seed robustness.
#
# Step 1: DPO fine-tune with 400 opinion pairs (seed=200, disjoint from benchmark)
#         LoRA rank 16, alpha 32, q/k/v/o projections
#         DPO beta=0.1, lr=5e-5, cosine schedule, 3 epochs
#         Output: results/dpo_model_seed200/
#
# Step 2: Full evaluation (behavioral + capability + probe re-analysis)
#         Behavioral: sycophancy rate, compliance gap, per-source
#         Capability: MMLU (500), GSM8k (200)
#         Probes: neutral-transfer on layers 0-5
#         Output: results/dpo_eval_seed200.json
#
# Expected runtime: ~12-20 hours on A100
# =============================================================================

set -euo pipefail

source "/home/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"

module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p slurm/logs results

echo "============================================"
echo "SLURM Job: DPO Seed 200 — Training"
echo "Model: ${PRIMARY_MODEL}"
echo "Train pairs: 400"
echo "Seed: 200 (disjoint from benchmark seed=42)"
echo "Epochs: 3"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

# ---- Step 1: DPO Training ----
python scripts/06_dpo_training.py \
    --model "${PRIMARY_MODEL}" \
    --n-train-pairs 400 \
    --seed 200 \
    --output-dir results/dpo_model_seed200 \
    --epochs 3 \
    --lr 5e-5 \
    --lora-rank 16 \
    --batch-size 4 \
    --beta 0.1 \
    --eval-split 0.1

echo "============================================"
echo "DPO training (seed=200) complete: $(date)"
echo "Starting evaluation..."
echo "============================================"

# ---- Step 2: Full Evaluation ----
python scripts/07_dpo_eval.py \
    --model "${PRIMARY_MODEL}" \
    --adapter-path results/dpo_model_seed200 \
    --data data/processed/master_sycophancy.jsonl \
    --output results/dpo_eval_seed200.json \
    --mmlu-samples 500 \
    --gsm8k-samples 200 \
    --probe-layers "0,1,2,3,4,5" \
    --seed 42

echo "============================================"
echo "DPO seed=200 pipeline complete: $(date)"
echo "Results: results/dpo_eval_seed200.json"
echo "============================================"
