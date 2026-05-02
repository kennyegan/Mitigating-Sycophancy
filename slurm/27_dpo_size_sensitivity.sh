#!/bin/bash
#SBATCH --job-name=dpo_size
#SBATCH --output=slurm/logs/dpo_size_%j.out
#SBATCH --error=slurm/logs/dpo_size_%j.err
#SBATCH --partition=gpu
#SBATCH --account=pi_larsonj_wit_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 27: DPO Size-Sensitivity Analysis
#
# Trains DPO at three dataset sizes (N=100, 200, 800) and evaluates each.
# N=400 already exists from the primary DPO run (results/dpo_model, seed=100).
#
# Tests whether the DPO sycophancy-reduction effect is robust to training
# set size — addresses reviewer concern about data quantity.
#
# Expected runtime: ~5 hours on A100-80GB
# Output: results/dpo_size_sensitivity/
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

OUTDIR="results/dpo_size_sensitivity"
DATA="data/processed/master_sycophancy.jsonl"
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

mkdir -p ${OUTDIR} slurm/logs

check_artifact() {
    local path="$1"
    if [ ! -s "${path}" ]; then
        echo "ERROR: expected artifact missing or empty: ${path}"
        exit 2
    fi
}

echo "============================================"
echo "SLURM Job: DPO Size-Sensitivity Analysis"
echo "Sizes: N=100, N=200, N=800 (N=400 reused)"
echo "Model: ${MODEL}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

# =============================================================================
# N=100
# =============================================================================
echo ""
echo ">>> [1/6] Training DPO with N=100"
echo "    Started: $(date)"

python scripts/06_dpo_training.py \
    --model "${MODEL}" \
    --n-train-pairs 100 \
    --seed 100 \
    --output-dir "${OUTDIR}/N100" \
    --epochs 3 \
    --lr 5e-5 \
    --lora-rank 16 \
    --batch-size 4 \
    --beta 0.1 \
    --eval-split 0.1

check_artifact "${OUTDIR}/N100/adapter_model.safetensors"
echo "    Completed: $(date)"

echo ""
echo ">>> [2/6] Evaluating DPO N=100"
echo "    Started: $(date)"

python scripts/07_dpo_eval.py \
    --model "${MODEL}" \
    --adapter-path "${OUTDIR}/N100" \
    --data "${DATA}" \
    --output "${OUTDIR}/N100_eval.json" \
    --seed 42

check_artifact "${OUTDIR}/N100_eval.json"
echo "    Completed: $(date)"

# =============================================================================
# N=200
# =============================================================================
echo ""
echo ">>> [3/6] Training DPO with N=200"
echo "    Started: $(date)"

python scripts/06_dpo_training.py \
    --model "${MODEL}" \
    --n-train-pairs 200 \
    --seed 100 \
    --output-dir "${OUTDIR}/N200" \
    --epochs 3 \
    --lr 5e-5 \
    --lora-rank 16 \
    --batch-size 4 \
    --beta 0.1 \
    --eval-split 0.1

check_artifact "${OUTDIR}/N200/adapter_model.safetensors"
echo "    Completed: $(date)"

echo ""
echo ">>> [4/6] Evaluating DPO N=200"
echo "    Started: $(date)"

python scripts/07_dpo_eval.py \
    --model "${MODEL}" \
    --adapter-path "${OUTDIR}/N200" \
    --data "${DATA}" \
    --output "${OUTDIR}/N200_eval.json" \
    --seed 42

check_artifact "${OUTDIR}/N200_eval.json"
echo "    Completed: $(date)"

# =============================================================================
# N=800
# =============================================================================
echo ""
echo ">>> [5/6] Training DPO with N=800"
echo "    Started: $(date)"

python scripts/06_dpo_training.py \
    --model "${MODEL}" \
    --n-train-pairs 800 \
    --seed 100 \
    --output-dir "${OUTDIR}/N800" \
    --epochs 3 \
    --lr 5e-5 \
    --lora-rank 16 \
    --batch-size 4 \
    --beta 0.1 \
    --eval-split 0.1

check_artifact "${OUTDIR}/N800/adapter_model.safetensors"
echo "    Completed: $(date)"

echo ""
echo ">>> [6/6] Evaluating DPO N=800"
echo "    Started: $(date)"

python scripts/07_dpo_eval.py \
    --model "${MODEL}" \
    --adapter-path "${OUTDIR}/N800" \
    --data "${DATA}" \
    --output "${OUTDIR}/N800_eval.json" \
    --seed 42

check_artifact "${OUTDIR}/N800_eval.json"
echo "    Completed: $(date)"

# =============================================================================
# Aggregate results (includes N=400 from primary DPO run)
# =============================================================================
echo ""
echo ">>> Aggregating size-sensitivity results"
echo "    Started: $(date)"

python scripts/aggregate_size_sensitivity.py \
    --dir "${OUTDIR}" \
    --reference-n400 results/dpo_eval_results.json \
    --output "${OUTDIR}/summary.json"

check_artifact "${OUTDIR}/summary.json"
echo "    Completed: $(date)"

# --- Summary ---
echo ""
echo "============================================"
echo "DPO size-sensitivity analysis complete: $(date)"
echo "Outputs in: ${OUTDIR}/"
ls -lh ${OUTDIR}/
echo "============================================"
