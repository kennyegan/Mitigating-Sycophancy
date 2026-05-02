#!/bin/bash
#SBATCH --job-name=syc-probe-ctrl
#SBATCH --output=slurm/logs/probe_ctrl_%j.out
#SBATCH --error=slurm/logs/probe_ctrl_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 6: Probe Control Experiment — Neutral-Only Training
#
# Validates that probe accuracy reflects genuine truth-representation tracking
# rather than prompt-format classification. Trains probes on neutral prompts
# only, then tests on biased prompt activations.
#
# Expected runtime: ~4-6 hours on A100
# Output: results/probe_control_results.json
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
echo "SLURM Job: Probe Control Experiment"
echo "Model: ${PRIMARY_MODEL}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

python scripts/02b_probe_control.py \
    --model "${PRIMARY_MODEL}" \
    --data data/processed/master_sycophancy.jsonl \
    --probe-type logistic \
    --batch-size 8 \
    --n-folds 5 \
    --seed 42 \
    --output results/probe_control_results.json

check_artifact results/probe_control_results.json

echo "============================================"
echo "Probe control complete: $(date)"
echo "============================================"
