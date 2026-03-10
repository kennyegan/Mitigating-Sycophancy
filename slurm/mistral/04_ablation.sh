#!/bin/bash
#SBATCH --job-name=mistral-ablation
#SBATCH --output=slurm/logs/mistral_ablation_%j.out
#SBATCH --error=slurm/logs/mistral_ablation_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Mistral Job M4: Top-10 Head Ablation
#
# NOTE: This script reads head_importance.json from M3 to pick the top 10 heads
# dynamically. If M3 hasn't run, it will fail.
#
# Output: results/mistral/top10_ablation_full_gsm8k.json
# =============================================================================

set -euo pipefail
source "/home/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"

module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export TOKENIZERS_PARALLELISM=false

mkdir -p results/mistral slurm/logs

check_artifact() {
    local path="$1"
    if [ ! -s "${path}" ]; then
        echo "ERROR: expected artifact missing or empty: ${path}"
        exit 2
    fi
}

# Dynamically extract top 10 heads from patching results
check_artifact results/mistral/head_importance.json

TOP_HEADS=$(python -c "
import json
with open('results/mistral/head_importance.json') as f:
    data = json.load(f)
# Handle both possible JSON structures
if 'head_results' in data and 'top_10_heads' in data['head_results']:
    heads = data['head_results']['top_10_heads']
    print(','.join(h['head'] for h in heads[:10]))
else:
    heads = data.get('head_rankings', data.get('heads', []))
    if isinstance(heads[0], dict):
        ranked = sorted(heads, key=lambda h: h.get('mean_recovery', h.get('recovery', 0)), reverse=True)[:10]
        print(','.join(f\"L{h['layer']}H{h['head']}\" for h in ranked))
    else:
        print(','.join(str(h) for h in heads[:10]))
")

echo "============================================"
echo "SLURM Job: Mistral Top-10 Ablation"
echo "Model: ${REPLICATION_MODEL}"
echo "Heads: ${TOP_HEADS}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

python scripts/04_head_ablation.py \
    --model "${REPLICATION_MODEL}" \
    --data data/processed/master_sycophancy.jsonl \
    --heads "${TOP_HEADS}" \
    --all-only \
    --eval-capabilities \
    --mmlu-samples 500 \
    --gsm8k-samples 1319 \
    --seed 42 \
    --output results/mistral/top10_ablation_full_gsm8k.json

check_artifact results/mistral/top10_ablation_full_gsm8k.json

echo "============================================"
echo "Mistral ablation complete: $(date)"
echo "============================================"
