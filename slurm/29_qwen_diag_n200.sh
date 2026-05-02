#!/bin/bash
#SBATCH --job-name=syc-qwen-n200
#SBATCH --output=slurm/logs/qwen_diag_n200_%j.out
#SBATCH --error=slurm/logs/qwen_diag_n200_%j.err
#SBATCH --partition=gpu
#SBATCH --account=pi_larsonj_wit_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 29: Qwen2.5-14B extended diagnostic — N=200
#
# Follow-up to job 55853891 (N=10, completed OK — 90% sycophancy but 9/10
# samples had compliance gaps near 0.000). We need to know whether that
# near-zero-gap pattern is representative of the full dataset or an
# artifact of the small random subset.
#
# Resolves decision D3 in SESSION_HANDOFF.md: cut §5.14 / reframe as
# heterogeneity / run the full stronger-model pipeline.
#
# Runs scripts/01_run_baseline.py --max-samples 200 --debug on the
# balanced dataset. Expected runtime: ~30-60 min on A100.
#
# Outputs: results/stronger/qwen_n200_baseline.{json,csv}
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
echo "SLURM Job: Qwen-14B extended diagnostic (N=200, --debug)"
echo "Model: ${QWEN_MODEL}"
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time:  $(date)"
echo "============================================"

python scripts/01_run_baseline.py \
    --model "${QWEN_MODEL}" \
    --data data/processed/master_sycophancy_balanced.jsonl \
    --max-samples 200 \
    --debug \
    --output-json "${OUTDIR}/qwen_n200_baseline.json" \
    --output-csv "${OUTDIR}/qwen_n200_baseline.csv" \
    --seed 42 || echo "Exit code: $? (expected if No samples evaluated)"

echo ""
echo "============================================"
echo "Qwen N=200 diagnostic complete: $(date)"
echo ""
echo "Key stats to inspect in the JSON:"
echo "  - overall.sycophancy_rate          (was 90% at N=10)"
echo "  - overall.mean_compliance_gap      (was +0.017 at N=10)"
echo "  - overall.samples_evaluated        (was 10/10 at N=10)"
echo "  - overall.samples_skipped          (was 0 at N=10)"
echo "  - overall.confidence_filtered.*    (N=10 had only 4 confident)"
echo ""
ls -lh ${OUTDIR}/qwen_n200_baseline.json 2>/dev/null || echo "  (no JSON — check stdout for error)"
echo "============================================"
