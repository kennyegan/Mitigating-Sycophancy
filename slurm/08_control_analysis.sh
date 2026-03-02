#!/bin/bash
#SBATCH --job-name=syc-controls
#SBATCH --output=slurm/logs/controls_%j.out
#SBATCH --error=slurm/logs/controls_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 8: Control Group Analysis
#
# Runs existing scripts on control group subsets:
# - fictional_entities.jsonl (100 samples) — baseline + probes + patching
# - uncertain_knowledge.jsonl (68 samples) — baseline only (too small for CV)
# - adversarially_true.jsonl (387 samples) — baseline + probes
#
# Expected runtime: ~6-8 hours on A100
# Output: results/control_groups/
# =============================================================================

set -euo pipefail

source "/home/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"

#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --gres=gpu:${GPUS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=80G

if [ -n "${SLURM_ACCOUNT}" ] && [ "${SLURM_ACCOUNT}" != "TODO_ACCOUNT" ]; then
    export SBATCH_ACCOUNT="${SLURM_ACCOUNT}"
fi

module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export TOKENIZERS_PARALLELISM=false

mkdir -p results/control_groups slurm/logs

echo "============================================"
echo "SLURM Job: Control Group Analysis"
echo "Model: ${PRIMARY_MODEL}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

# --- Step 1: Baseline on fictional entities (100 samples) ---
echo ""
echo "[1/6] Baseline on fictional entities..."
python scripts/01_run_baseline.py \
    --model "${PRIMARY_MODEL}" \
    --data data/processed/control_groups/fictional_entities.jsonl \
    --output-json results/control_groups/baseline_fictional.json \
    --output-csv results/control_groups/baseline_fictional.csv \
    --seed 42

# --- Step 2: Baseline on uncertain knowledge (68 samples) ---
echo ""
echo "[2/6] Baseline on uncertain knowledge (descriptive stats only)..."
python scripts/01_run_baseline.py \
    --model "${PRIMARY_MODEL}" \
    --data data/processed/control_groups/uncertain_knowledge.jsonl \
    --output-json results/control_groups/baseline_uncertain.json \
    --output-csv results/control_groups/baseline_uncertain.csv \
    --seed 42

# --- Step 3: Baseline on adversarially true (387 samples) ---
echo ""
echo "[3/6] Baseline on adversarially true hints..."
python scripts/01_run_baseline.py \
    --model "${PRIMARY_MODEL}" \
    --data data/processed/control_groups/adversarially_true.jsonl \
    --output-json results/control_groups/baseline_adversarial.json \
    --output-csv results/control_groups/baseline_adversarial.csv \
    --seed 42

# --- Step 4: Probes on fictional entities (100 samples) ---
echo ""
echo "[4/6] Probes on fictional entities..."
python scripts/02_train_probes.py \
    --model "${PRIMARY_MODEL}" \
    --data data/processed/control_groups/fictional_entities.jsonl \
    --probe-position final \
    --probe-type logistic \
    --n-folds 5 \
    --seed 42 \
    --output results/control_groups/probe_fictional.json

# --- Step 5: Probes on adversarially true (387 samples) ---
echo ""
echo "[5/6] Probes on adversarially true hints..."
python scripts/02_train_probes.py \
    --model "${PRIMARY_MODEL}" \
    --data data/processed/control_groups/adversarially_true.jsonl \
    --probe-position final \
    --probe-type logistic \
    --n-folds 5 \
    --seed 42 \
    --output results/control_groups/probe_adversarial.json

# --- Step 6: Patching on fictional entities (50 samples) ---
echo ""
echo "[6/6] Activation patching on fictional entities..."
python scripts/03_activation_patching.py \
    --model "${PRIMARY_MODEL}" \
    --data data/processed/control_groups/fictional_entities.jsonl \
    --max-samples 50 \
    --max-positions 50 \
    --head-top-k 5 \
    --seed 42 \
    --output-dir results/control_groups/patching_fictional

echo ""
echo "============================================"
echo "Control group analysis complete: $(date)"
echo "============================================"
