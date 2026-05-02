#!/bin/bash
#SBATCH --job-name=syc-top10-gsm
#SBATCH --output=slurm/logs/top10_gsm_%j.out
#SBATCH --error=slurm/logs/top10_gsm_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 11: Top-10 Head Ablation with Full GSM8k (1319 samples)
#
# Reruns the top-10 ablation experiment with the complete GSM8k test set
# instead of the 200-sample subset. This resolves the noisy 12.5% → 8.0%
# drop observed in Job 9 by providing sufficient statistical power.
#
# Expected runtime: up to ~1-2 days on A100 for full 1319-sample GSM8k eval
# Output: results/top10_ablation_full_gsm8k.json
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
echo "SLURM Job: Top-10 Ablation + Full GSM8k"
echo "Model: ${PRIMARY_MODEL}"
echo "Heads: L1H20,L5H5,L4H28,L5H17,L3H17,L5H4,L5H19,L5H24,L4H5,L3H0"
echo "GSM8k samples: 1319 (full test set)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

python scripts/04_head_ablation.py \
    --model "${PRIMARY_MODEL}" \
    --data data/processed/master_sycophancy.jsonl \
    --heads "L1H20,L5H5,L4H28,L5H17,L3H17,L5H4,L5H19,L5H24,L4H5,L3H0" \
    --all-only \
    --eval-capabilities \
    --mmlu-samples 500 \
    --gsm8k-samples 1319 \
    --seed 42 \
    --output results/top10_ablation_full_gsm8k.json

check_artifact results/top10_ablation_full_gsm8k.json

echo "============================================"
echo "Top-10 full GSM8k ablation complete: $(date)"
echo "============================================"
