#!/bin/bash
#SBATCH --job-name=syc-stronger
#SBATCH --output=slurm/logs/stronger_%j.out
#SBATCH --error=slurm/logs/stronger_%j.err
#SBATCH --partition=gpu
#SBATCH --account=pi_larsonj_wit_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 25: Stronger-Model Pipeline — Qwen2.5-14B-Instruct
#
# Runs the full sycophancy analysis pipeline on a larger model:
#   Step 1: Baseline sycophancy evaluation
#   Step 2: Probe control (neutral-only training)
#   Step 3: Activation patching (100 samples)
#   Step 4: Head ablation (top-3 heads from patching)
#
# Expected runtime: ~16-24 hours on A100-80GB
# Output: results/stronger/
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

check_artifact() {
    local path="$1"
    if [ ! -s "${path}" ]; then
        echo "ERROR: expected artifact missing or empty: ${path}"
        exit 2
    fi
}

echo "============================================"
echo "SLURM Job: Stronger-Model Pipeline"
echo "Model: ${STRONGER_MODEL}"
echo "Data: ${DATA}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

# --- Step 1: Baseline sycophancy evaluation ---
# Reusing existing artifact from prior successful run (Apr 30): 1498/1500 evaluated,
# syc_rate 28.2%, mean compliance gap -0.12. The baseline path is non-deterministic
# at full N (intermittent "No samples evaluated"); skipping rerun here.
echo ""
echo ">>> Step 1/4: Baseline evaluation — REUSING existing artifact"
if [ ! -s "${OUTDIR}/baseline_summary.json" ]; then
    echo "ERROR: expected reused artifact missing: ${OUTDIR}/baseline_summary.json"
    exit 2
fi
ls -lh ${OUTDIR}/baseline_summary.json ${OUTDIR}/baseline_details.csv
echo "    Skipped (reused): $(date)"

# --- Step 2: Probe control experiment ---
echo ""
echo ">>> Step 2/4: Probe control"
echo "    Started: $(date)"

python scripts/02b_probe_control.py \
    --model "${STRONGER_MODEL}" \
    --data "${DATA}" \
    --probe-type logistic \
    --batch-size 8 \
    --n-folds 5 \
    --seed 42 \
    --output "${OUTDIR}/probe_control_balanced.json"

check_artifact "${OUTDIR}/probe_control_balanced.json"
echo "    Completed: $(date)"

# --- Step 3: Activation patching (100 samples) ---
echo ""
echo ">>> Step 3/4: Activation patching"
echo "    Started: $(date)"

python scripts/03_activation_patching.py \
    --model "${STRONGER_MODEL}" \
    --data "${DATA}" \
    --max-samples 100 \
    --seed 42 \
    --output-dir "${OUTDIR}/patching"

check_artifact "${OUTDIR}/patching/patching_heatmap.json"
echo "    Completed: $(date)"

# --- Extract top-3 heads from patching results ---
echo ""
echo ">>> Extracting top-3 heads from patching results..."

TOP3_HEADS=$(python -c "
import json, sys
with open('${OUTDIR}/patching/patching_heatmap.json') as f:
    data = json.load(f)
# Handle both possible output formats
if 'head_rankings' in data:
    rankings = data['head_rankings']
elif 'top_heads' in data:
    rankings = data['top_heads']
else:
    # Fall back to finding heads from the heatmap matrix
    import numpy as np
    heatmap = np.array(data.get('heatmap', data.get('patching_results', [])))
    flat = [(abs(heatmap[l][h]), l, h) for l in range(len(heatmap)) for h in range(len(heatmap[l]))]
    flat.sort(reverse=True)
    rankings = [{'layer': l, 'head': h} for _, l, h in flat[:3]]

top3 = rankings[:3]
heads_str = ','.join(f\"L{h['layer']}H{h['head']}\" for h in top3)
print(heads_str)
")

echo "    Top-3 heads: ${TOP3_HEADS}"

# --- Step 4: Head ablation on top-3 heads ---
echo ""
echo ">>> Step 4/4: Head ablation"
echo "    Started: $(date)"

python scripts/04_head_ablation.py \
    --model "${STRONGER_MODEL}" \
    --data "${DATA}" \
    --heads "${TOP3_HEADS}" \
    --eval-capabilities \
    --mmlu-samples 500 \
    --gsm8k-samples 200 \
    --seed 42 \
    --output "${OUTDIR}/head_ablation.json"

check_artifact "${OUTDIR}/head_ablation.json"
echo "    Completed: $(date)"

# --- Summary ---
echo ""
echo "============================================"
echo "Stronger-model pipeline complete: $(date)"
echo "Outputs in: ${OUTDIR}/"
ls -lh ${OUTDIR}/
echo "============================================"
