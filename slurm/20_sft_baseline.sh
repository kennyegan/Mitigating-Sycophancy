#!/bin/bash
#SBATCH --job-name=syc-sft-base
#SBATCH --output=slurm/logs/sft_baseline_%j.out
#SBATCH --error=slurm/logs/sft_baseline_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --account=pi_larsonj_wit_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

# =============================================================================
# Job: SFT Baseline Training + Eval
#
# Same data, same LoRA config, same model as DPO — but trained with standard
# supervised fine-tuning on chosen (non-sycophantic) responses only.
# This is the simplest serious training-time baseline for comparison.
#
# Step 1: SFT fine-tune with 400 chosen responses (seed=100, same data as DPO)
#         LoRA rank 16, alpha 32, q/k/v/o projections
#         lr=5e-5, cosine schedule, 3 epochs
#         Output: results/sft_model/
#
# Step 2: Full evaluation (behavioral + capability + probe re-analysis)
#         Behavioral: sycophancy rate, compliance gap, per-source
#         Capability: MMLU (500), GSM8k (200)
#         Probes: neutral-transfer on layers 0-5
#         Output: results/sft_eval_results.json
#
# Expected runtime: ~10-16 hours on A100
# =============================================================================

set -euo pipefail

source "/work/pi_larsonj_wit_edu/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"

module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p slurm/logs results

echo "============================================"
echo "SLURM Job: SFT Baseline — Training"
echo "Model: ${PRIMARY_MODEL}"
echo "Train pairs: 400 (chosen responses only)"
echo "Seed: 100 (same data as DPO)"
echo "Epochs: 3"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

# ---- Step 1: SFT Training ----
python scripts/10_sft_training.py \
    --model "${PRIMARY_MODEL}" \
    --n-pairs 400 \
    --seed 100 \
    --output-dir results/sft_model \
    --epochs 3 \
    --lr 5e-5 \
    --lora-rank 16 \
    --batch-size 4 \
    --eval-split 0.1

echo "============================================"
echo "SFT training complete: $(date)"
echo "Starting evaluation..."
echo "============================================"

# ---- Step 2: Full Evaluation ----
python scripts/11_sft_eval.py \
    --model "${PRIMARY_MODEL}" \
    --adapter-path results/sft_model \
    --data data/processed/master_sycophancy.jsonl \
    --output results/sft_eval_results.json \
    --mmlu-samples 500 \
    --gsm8k-samples 200 \
    --probe-layers "0,1,2,3,4,5" \
    --seed 42

echo "============================================"
echo "SFT baseline pipeline complete: $(date)"
echo "Results: results/sft_eval_results.json"
echo "============================================"
