#!/bin/bash
#SBATCH --job-name=syc-boot
#SBATCH --output=slurm/logs/patching_bootstrap_%j.out
#SBATCH --error=slurm/logs/patching_bootstrap_%j.err
#SBATCH --partition=gpu
#SBATCH --account=pi_larsonj_wit_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 23: Patching Bootstrap — Head-Ranking Stability
#
# Addresses Experiment 5 (Mechanism stability checks) from neurips-plan.md.
# Reviewer question: are the top-3 heads (L4H28, L4H5, L5H31) stable
# or N=100 sampling noise?
#
# Runs scripts/12_patching_bootstrap.py: 5 bootstrap resamples of 100
# prompts each, full layer-scan + head-scan per resample, then aggregates
# pairwise Jaccard + per-head rank stability + recovery CIs.
#
# Expected runtime: ~18-22 hours on A100 (5 resamples × ~4h each); first
# attempt at 6h wallclock (job 55757250) timed out before a single
# resample finished. Layer-scan dominates: 32 layers × 50 positions × 100
# samples per resample, so each resample is ~4 GPU-hours.
# Output: results/patching_bootstrap.json + manifest
# =============================================================================

set -euo pipefail

source "/work/pi_larsonj_wit_edu/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"

OUTPUT="results/patching_bootstrap.json"
DATA="data/processed/master_sycophancy_balanced.jsonl"

module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export TOKENIZERS_PARALLELISM=false

mkdir -p slurm/logs results results/manifests

check_artifact() {
    local path="$1"
    if [ ! -s "${path}" ]; then
        echo "ERROR: expected artifact missing or empty: ${path}"
        exit 2
    fi
}

echo "============================================"
echo "SLURM Job: Patching Bootstrap (head stability)"
echo "Model: ${PRIMARY_MODEL}"
echo "Data:  ${DATA}"
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time:  $(date)"
echo "============================================"

python scripts/12_patching_bootstrap.py \
    --model "${PRIMARY_MODEL}" \
    --data "${DATA}" \
    --n-bootstrap 5 \
    --n-samples 100 \
    --head-top-k 5 \
    --max-positions 50 \
    --output "${OUTPUT}"

check_artifact "${OUTPUT}"

echo ""
echo "============================================"
echo "Patching bootstrap complete: $(date)"
ls -lh "${OUTPUT}"
echo "============================================"
