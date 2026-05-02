#!/bin/bash
#SBATCH --job-name=dpo-mistral
#SBATCH --output=slurm/logs/dpo_mistral_%j.out
#SBATCH --error=slurm/logs/dpo_mistral_%j.err
#SBATCH --partition=gpu
#SBATCH --account=pi_larsonj_wit_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 26: DPO Fine-Tuning + Probe Decomposition on Mistral-7B-Instruct
#
# Closes the single-architecture gap on Contribution 5 (DPO probe decomposition)
# by replicating the Llama-3 DPO pipeline on Mistral-7B-Instruct-v0.1.
#
# Steps:
#   A. DPO training (generates preference pairs + LoRA fine-tune)
#   B. Behavioral evaluation (sycophancy rate + MMLU + GSM8k)
#   C. Probe re-analysis (neutral_transfer decomposition on DPO model)
#
# Expected runtime: ~4-6 hours on A100
# Output: results/mistral/dpo_model/, results/mistral/dpo_eval_results.json,
#         results/mistral/dpo_probe_control_balanced.json
# =============================================================================

set -euo pipefail

source "/work/pi_larsonj_wit_edu/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"

MISTRAL_MODEL="mistralai/Mistral-7B-Instruct-v0.1"
ADAPTER_DIR="results/mistral/dpo_model"

module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export TOKENIZERS_PARALLELISM=false

mkdir -p results/mistral slurm/logs

check_artifact() {
    local path="$1"
    if [ ! -s "${path}" ]; then
        echo "ERROR: expected artifact missing or empty: ${path}"
        exit 2
    fi
}

echo "============================================"
echo "SLURM Job: DPO Mistral Pipeline"
echo "Model: ${MISTRAL_MODEL}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

# --- Step A: DPO Training ---
echo ""
echo ">>> Step A/3: DPO training"
echo "    Started: $(date)"

python scripts/06_dpo_training.py \
    --model "${MISTRAL_MODEL}" \
    --n-train-pairs 400 \
    --output-dir "${ADAPTER_DIR}" \
    --epochs 2 \
    --lr 1e-5 \
    --lora-rank 16 \
    --batch-size 4 \
    --beta 0.05 \
    --eval-split 0.1 \
    --seed 100

if [ $? -ne 0 ]; then echo "STEP A FAILED: DPO training"; exit 1; fi
check_artifact "${ADAPTER_DIR}/adapter_model.safetensors"
echo "    Completed: $(date)"

# --- Step B: Behavioral Evaluation ---
echo ""
echo ">>> Step B/3: Behavioral evaluation (sycophancy + MMLU + GSM8k)"
echo "    Started: $(date)"

python scripts/07_dpo_eval.py \
    --model "${MISTRAL_MODEL}" \
    --adapter-path "${ADAPTER_DIR}" \
    --data data/processed/master_sycophancy.jsonl \
    --mmlu-samples 500 \
    --gsm8k-samples 200 \
    --seed 42 \
    --output results/mistral/dpo_eval_results.json

if [ $? -ne 0 ]; then echo "STEP B FAILED: Behavioral evaluation"; exit 1; fi
check_artifact results/mistral/dpo_eval_results.json
echo "    Completed: $(date)"

# --- Step C: Probe Re-Analysis — SKIPPED ---
# HookedTransformer + merged-LoRA path triggers an unresolved device-mismatch in
# transformer_lens fold_value_biases; both prior attempts crashed here. Mistral DPO
# Section 5.10 row uses Step B behavioral numbers (sycophancy + MMLU/GSM8k) only.
echo ""
echo ">>> Step C/3: Probe control — SKIPPED (TransformerLens+LoRA device mismatch)"

# --- Summary ---
echo ""
echo "============================================"
echo "DPO Mistral pipeline complete (probes skipped): $(date)"
echo "Outputs:"
ls -lh results/mistral/dpo_model/adapter_model.safetensors
ls -lh results/mistral/dpo_eval_results.json
echo "============================================"
