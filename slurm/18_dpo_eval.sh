#!/bin/bash
#SBATCH --job-name=syc-dpo-eval
#SBATCH --output=slurm/logs/dpo_eval_%j.out
#SBATCH --error=slurm/logs/dpo_eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 18: DPO Model Evaluation — Behavioral + Capability + Mechanistic Probes
#
# Evaluates the DPO-trained model (base + LoRA from results/dpo_model/) on:
#   1. Full behavioral evaluation (sycophancy rate, compliance gap, per-source)
#   2. Capability retention (MMLU 500, GSM8k 200)
#   3. Neutral-transfer probe re-analysis at specified layers
#
# Compares to pre-DPO baselines and computes 2x2 decomposition deltas.
#
# Expected runtime: ~8-12 hours on A100 (probes dominate)
# Output: results/dpo_eval_results.json
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

mkdir -p slurm/logs

echo "============================================"
echo "SLURM Job: DPO Model Evaluation"
echo "Model: ${PRIMARY_MODEL}"
echo "Adapter: results/dpo_model"
echo "MMLU samples: 500"
echo "GSM8k samples: 200"
echo "Probe layers: 0,1,2,3,4,5"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

python scripts/07_dpo_eval.py \
    --model "${PRIMARY_MODEL}" \
    --adapter-path results/dpo_model \
    --data data/processed/master_sycophancy.jsonl \
    --output results/dpo_eval_results.json \
    --mmlu-samples 500 \
    --gsm8k-samples 200 \
    --probe-layers "0,1,2,3,4,5" \
    --seed 42

echo "============================================"
echo "DPO evaluation complete: $(date)"
echo "============================================"
