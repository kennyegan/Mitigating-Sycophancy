#!/bin/bash
#SBATCH --job-name=syc-ff-gen
#SBATCH --output=slurm/logs/freeform_gen_%j.out
#SBATCH --error=slurm/logs/freeform_gen_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --account=pi_larsonj_wit_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# =============================================================================
# Job: Free-Form Sycophancy Transcript Generation
#
# Generates multi-turn conversational transcripts for the 150-prompt free-form
# sycophancy benchmark (data/freeform/). Each prompt runs a 3-turn conversation
# with escalating user pushback.
#
# Step 1: Baseline (no adapter) — 150 prompts x 3 turns x max 512 tokens
#         Output: results/freeform/llama3_base_transcripts.jsonl
#
# Step 2: DPO model (with LoRA adapter) — same 150 prompts
#         Output: results/freeform/llama3_dpo_transcripts.jsonl
#
# Expected runtime: ~3-5 hours on A100 (150 prompts x 3 generations x 2 models)
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

mkdir -p slurm/logs results/freeform

echo "============================================"
echo "SLURM Job: Free-Form Transcript Generation"
echo "Model: ${PRIMARY_MODEL}"
echo "Prompts: data/freeform/"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

# ---- Step 1: Baseline (no adapter) ----
echo ""
echo "============================================"
echo "Step 1: Baseline generation (no adapter)"
echo "Time: $(date)"
echo "============================================"

python src/eval/freeform_generate.py \
    --model "${PRIMARY_MODEL}" \
    --prompts data/freeform/ \
    --output results/freeform/llama3_base_transcripts.jsonl \
    --seed 42

echo "Baseline generation complete: $(date)"

# ---- Step 2: DPO model ----
echo ""
echo "============================================"
echo "Step 2: DPO model generation"
echo "Time: $(date)"
echo "============================================"

python src/eval/freeform_generate.py \
    --model "${PRIMARY_MODEL}" \
    --prompts data/freeform/ \
    --output results/freeform/llama3_dpo_transcripts.jsonl \
    --adapter-path results/dpo_model \
    --seed 42

echo ""
echo "============================================"
echo "Free-form generation complete: $(date)"
echo "Results:"
echo "  Baseline: results/freeform/llama3_base_transcripts.jsonl"
echo "  DPO:      results/freeform/llama3_dpo_transcripts.jsonl"
echo "============================================"
