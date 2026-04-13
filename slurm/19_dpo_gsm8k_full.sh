#!/bin/bash
#SBATCH --job-name=syc-dpo-gsm8k
#SBATCH --output=slurm/logs/dpo_gsm8k_full_%j.out
#SBATCH --error=slurm/logs/dpo_gsm8k_full_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 19: DPO Model — Full GSM8k Evaluation (N=1319)
#
# Reruns 07_dpo_eval.py with --gsm8k-samples 1319 and --skip-probes.
# MMLU is also re-run (N=500) for completeness; probes are skipped since
# those results are already in dpo_eval_results.json.
#
# Addresses reviewer W9: the N=200 GSM8k comparison is uninformative
# (13pp CI width). N=1319 gives a ~5pp CI width.
#
# Expected runtime: ~2-3 hours on A100
# Output: results/dpo_gsm8k_full_results.json
# =============================================================================

set -euo pipefail

source "/home/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"

module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p slurm/logs

echo "============================================"
echo "SLURM Job: DPO Full GSM8k Evaluation"
echo "Model: ${PRIMARY_MODEL}"
echo "Adapter: results/dpo_model"
echo "GSM8k samples: 1319 (full test set)"
echo "MMLU samples: 500"
echo "Probes: SKIPPED"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

python scripts/07_dpo_eval.py \
    --model "${PRIMARY_MODEL}" \
    --adapter-path results/dpo_model \
    --data data/processed/master_sycophancy.jsonl \
    --output results/dpo_gsm8k_full_results.json \
    --mmlu-samples 500 \
    --gsm8k-samples 1319 \
    --skip-probes \
    --seed 42

echo "============================================"
echo "DPO full GSM8k evaluation complete: $(date)"
echo "============================================"
