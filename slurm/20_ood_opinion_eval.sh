#!/bin/bash
#SBATCH --job-name=syc-ood-eval
#SBATCH --output=slurm/logs/ood_opinion_eval_%j.out
#SBATCH --error=slurm/logs/ood_opinion_eval_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 20: Out-of-Distribution Opinion Sycophancy Evaluation
#
# Evaluates the DPO model on opinion sycophancy prompts that differ from
# the Anthropic model-written-evals format used in training and evaluation.
#
# Three OOD conditions (~450 total samples):
#   1. New Anthropic samples (seed=200) — same template, different questions
#   2. Rephrased templates — 4 diverse prompt formats on Anthropic questions
#   3. Manual diverse opinions — 50 hand-crafted questions on new topics
#
# Both pre-DPO and post-DPO models are evaluated for comparison.
#
# Addresses reviewer W6: the 23.8pp reduction is in-distribution only.
# This test measures whether DPO generalizes across prompt formats/topics.
#
# Expected runtime: ~4-6 hours on A100 (two full model loads + ~900 forward passes)
# Output: results/ood_opinion_eval_results.json
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

mkdir -p slurm/logs

echo "============================================"
echo "SLURM Job: OOD Opinion Sycophancy Evaluation"
echo "Model: ${PRIMARY_MODEL}"
echo "Adapter: results/dpo_model"
echo "Seed: 200"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

python scripts/08_ood_opinion_eval.py \
    --model "${PRIMARY_MODEL}" \
    --adapter-path results/dpo_model \
    --output results/ood_opinion_eval_results.json \
    --seed 200 \
    --n-new-anthropic 200 \
    --n-rephrased 200

echo "============================================"
echo "OOD opinion evaluation complete: $(date)"
echo "============================================"
