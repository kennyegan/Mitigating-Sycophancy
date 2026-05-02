#!/bin/bash
#SBATCH --job-name=syc-qwen-diag
#SBATCH --output=slurm/logs/qwen_diag_%j.out
#SBATCH --error=slurm/logs/qwen_diag_%j.err
#SBATCH --partition=gpu
#SBATCH --account=pi_larsonj_wit_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 28: Qwen2.5-14B "No samples evaluated" diagnostic
#
# Two previous full-pipeline attempts (55633705, 55649560) failed at
# Step 1 with "Error: No samples evaluated". Before we burn another
# 20-hour A100 slot on the full stronger-model pipeline, run the
# baseline on just 10 samples with --debug to see *why* every sample is
# being filtered. Likely causes:
#   - compliance-gap filter rejects Qwen outputs (model refuses the
#     biased format, so honest and sycophantic logits come out equal)
#   - tokenizer incompatibility producing empty honest_tokens
#   - prompt length truncation during SycophancyModel.to_tokens
#
# The --debug flag in 01_run_baseline.py prints details for the first 5
# skipped samples, which will pinpoint the root cause in one run.
#
# Expected runtime: ~15-30 min on A100 (~10 samples).
# Output: results/stronger/qwen_diagnostic.json + stdout with skip reasons.
# =============================================================================

set -euo pipefail

source "/work/pi_larsonj_wit_edu/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"

QWEN_MODEL="Qwen/Qwen2.5-14B-Instruct"
OUTDIR="results/stronger"

module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export TOKENIZERS_PARALLELISM=false

mkdir -p ${OUTDIR} slurm/logs

echo "============================================"
echo "SLURM Job: Qwen-14B diagnostic (N=10, --debug)"
echo "Model: ${QWEN_MODEL}"
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time:  $(date)"
echo "============================================"

python scripts/01_run_baseline.py \
    --model "${QWEN_MODEL}" \
    --data data/processed/master_sycophancy_balanced.jsonl \
    --max-samples 10 \
    --debug \
    --output-json "${OUTDIR}/qwen_diagnostic.json" \
    --output-csv "${OUTDIR}/qwen_diagnostic.csv" \
    --seed 42 || echo "Exit code: $? (expected if No samples evaluated)"

echo ""
echo "============================================"
echo "Qwen diagnostic complete: $(date)"
echo "Inspect the stdout above for per-sample skip reasons."
echo "If JSON was written:"
ls -lh ${OUTDIR}/qwen_diagnostic.json 2>/dev/null || echo "  (no JSON — check stdout for error)"
echo "============================================"
