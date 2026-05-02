#!/bin/bash
#SBATCH --job-name=syc-gsm8k-full
#SBATCH --output=slurm/logs/fullN_gsm8k_%j.out
#SBATCH --error=slurm/logs/fullN_gsm8k_%j.err
#SBATCH --partition=gpu
#SBATCH --account=pi_larsonj_wit_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 24: Full-N=1319 GSM8k eval on DPO seeds 200 + 300, and SFT baseline
#
# Fixes paper_todo.md bug B7: baseline is N=1319 but post-DPO and SFT
# evals were N=200. Seed 100 already done Apr 13 —
# results/dpo_gsm8k_full_results.json shows 36.8%.
#
# Runs scripts/07_dpo_eval.py with --gsm8k-samples 1319 --skip-probes
# on each of: dpo_model_seed200, dpo_model_seed300, sft_model.
# Expected ~1.5-2h per adapter on A100 (~5h total).
#
# Outputs:
#   results/dpo_gsm8k_full_seed200.json
#   results/dpo_gsm8k_full_seed300.json
#   results/sft_gsm8k_full.json
# =============================================================================

set -euo pipefail

source "/work/pi_larsonj_wit_edu/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"

module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export TOKENIZERS_PARALLELISM=false

mkdir -p slurm/logs results

check_artifact() {
    local path="$1"
    if [ ! -s "${path}" ]; then
        echo "ERROR: expected artifact missing or empty: ${path}"
        exit 2
    fi
}

echo "============================================"
echo "SLURM Job: Full-N GSM8k on DPO seeds 200+300"
echo "Model: ${PRIMARY_MODEL}"
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time:  $(date)"
echo "============================================"

# --- Seed 200 ---
echo ""
echo ">>> Seed 200: full-N GSM8k eval"
echo "    Started: $(date)"

python scripts/07_dpo_eval.py \
    --model "${PRIMARY_MODEL}" \
    --adapter-path results/dpo_model_seed200 \
    --data data/processed/master_sycophancy.jsonl \
    --gsm8k-samples 1319 \
    --mmlu-samples 500 \
    --skip-probes \
    --seed 42 \
    --output results/dpo_gsm8k_full_seed200.json

check_artifact results/dpo_gsm8k_full_seed200.json
echo "    Completed: $(date)"

# --- Seed 300 ---
echo ""
echo ">>> Seed 300: full-N GSM8k eval"
echo "    Started: $(date)"

python scripts/07_dpo_eval.py \
    --model "${PRIMARY_MODEL}" \
    --adapter-path results/dpo_model_seed300 \
    --data data/processed/master_sycophancy.jsonl \
    --gsm8k-samples 1319 \
    --mmlu-samples 500 \
    --skip-probes \
    --seed 42 \
    --output results/dpo_gsm8k_full_seed300.json

check_artifact results/dpo_gsm8k_full_seed300.json
echo "    Completed: $(date)"

# --- SFT baseline ---
echo ""
echo ">>> SFT baseline: full-N GSM8k eval"
echo "    Started: $(date)"

python scripts/07_dpo_eval.py \
    --model "${PRIMARY_MODEL}" \
    --adapter-path results/sft_model \
    --data data/processed/master_sycophancy.jsonl \
    --gsm8k-samples 1319 \
    --mmlu-samples 500 \
    --skip-probes \
    --seed 42 \
    --output results/sft_gsm8k_full.json

check_artifact results/sft_gsm8k_full.json
echo "    Completed: $(date)"

echo ""
echo "============================================"
echo "Full-N GSM8k complete: $(date)"
ls -lh results/dpo_gsm8k_full_seed*.json results/sft_gsm8k_full.json
echo "============================================"
