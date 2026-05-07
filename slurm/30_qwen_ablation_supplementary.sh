#!/bin/bash
#SBATCH --job-name=syc-qwen-abl
#SBATCH --output=slurm/logs/qwen_abl_supp_%j.out
#SBATCH --error=slurm/logs/qwen_abl_supp_%j.err
#SBATCH --partition=gpu
#SBATCH --account=pi_larsonj_wit_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Supplementary job: Qwen-14B head ablation — missing conditions only.
# Main job 56532117 timed out at 48h after completing baseline + 3 single-zero +
# 1 partial pair condition. This supplementary runs the conditions critical for
# §5.14 cross-scale ablation null comparison to Llama-3:
#   - baseline (re-run for consistency, ~30min sycophancy + capabilities)
#   - all_zero (top-3 heads L0H0+L0H17+L0H11 zero-ablated together)
#   - all_mean (top-3 heads mean-ablated together)
#
# Heads come from main job's head_importance.json: top-3 by recovery score.
# Expected runtime: ~15-18h total.
# =============================================================================

set -euo pipefail

source "/work/pi_larsonj_wit_edu/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"

STRONGER_MODEL="Qwen/Qwen2.5-14B-Instruct"
DATA="data/processed/master_sycophancy_balanced.jsonl"
OUTDIR="results/stronger"

module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export TOKENIZERS_PARALLELISM=false

mkdir -p ${OUTDIR} slurm/logs

# Extract top-3 heads from prior patching run
TOP3_HEADS=$(python -c "
import json
with open('${OUTDIR}/patching/head_importance.json') as f:
    data = json.load(f)
top3 = data['head_results']['top_10_heads'][:3]
print(','.join(h['head'] for h in top3))
")

if [ -z "${TOP3_HEADS}" ]; then
    echo "ERROR: failed to extract top-3 heads"
    exit 3
fi

echo "============================================"
echo "SLURM Job: Qwen-14B Ablation Supplementary"
echo "Model: ${STRONGER_MODEL}"
echo "Top-3 heads (from main job): ${TOP3_HEADS}"
echo "Conditions: baseline, all_zero, all_mean"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

python scripts/04_head_ablation.py \
    --model "${STRONGER_MODEL}" \
    --data "${DATA}" \
    --heads "${TOP3_HEADS}" \
    --eval-capabilities \
    --mmlu-samples 500 \
    --gsm8k-samples 200 \
    --seed 42 \
    --all-only \
    --include-all-mean \
    --output "${OUTDIR}/head_ablation_supplementary.json"

if [ ! -s "${OUTDIR}/head_ablation_supplementary.json" ]; then
    echo "ERROR: supplementary output missing or empty"
    exit 2
fi

echo ""
echo "============================================"
echo "Qwen ablation supplementary complete: $(date)"
ls -lh ${OUTDIR}/head_ablation_supplementary.json
echo "============================================"
