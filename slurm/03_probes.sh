#!/bin/bash
#SBATCH --job-name=syc-probes
#SBATCH --output=slurm/logs/probes_%j.out
#SBATCH --error=slurm/logs/probes_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 3: Linear Probes — Social Compliance vs. Belief Corruption
#
# Extracts resid_post at all 32 layers for 1,500 samples (neutral + biased),
# then trains logistic regression probes with 5-fold CV at each layer.
#
# This is the most memory-intensive job: ~40GB activations cached.
# Requests 80GB RAM to be safe.
#
# Expected runtime: ~4-6 hours on A100
# Output: results/probe_results_neutral_transfer.json, results/probe_results_mixed_diagnostic.json
# =============================================================================

set -euo pipefail

source "/work/pi_larsonj_wit_edu/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"



module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export TOKENIZERS_PARALLELISM=false

mkdir -p results slurm/logs

check_artifact() {
    local path="$1"
    if [ ! -s "${path}" ]; then
        echo "ERROR: expected artifact missing or empty: ${path}"
        exit 2
    fi
}

echo "============================================"
echo "SLURM Job: Linear Probes"
echo "Model: ${PRIMARY_MODEL}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

# Claim-bearing run: neutral_transfer
python scripts/02_train_probes.py \
    --model "${PRIMARY_MODEL}" \
    --data data/processed/master_sycophancy.jsonl \
    --probe-position both \
    --probe-type logistic \
    --analysis-mode neutral_transfer \
    --batch-size 8 \
    --n-folds 5 \
    --seed 42 \
    --output results/probe_results_neutral_transfer.json

check_artifact results/probe_results_neutral_transfer.json

echo "--- Neutral-transfer complete, now running mixed diagnostic ---"

# Diagnostic run: mixed_diagnostic (format-confound risk estimate)
python scripts/02_train_probes.py \
    --model "${PRIMARY_MODEL}" \
    --data data/processed/master_sycophancy.jsonl \
    --probe-position both \
    --probe-type logistic \
    --analysis-mode mixed_diagnostic \
    --batch-size 8 \
    --n-folds 5 \
    --seed 42 \
    --output results/probe_results_mixed_diagnostic.json

check_artifact results/probe_results_mixed_diagnostic.json

echo "============================================"
echo "Probes complete: $(date)"
echo "============================================"
