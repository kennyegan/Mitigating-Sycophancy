#!/bin/bash
#SBATCH --job-name=syc-ood-fix
#SBATCH --output=slurm/logs/ood_eval_fix_%j.out
#SBATCH --error=slurm/logs/ood_eval_fix_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --account=pi_larsonj_wit_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# =============================================================================
# Job: OOD Eval Fix — Run Both Conditions in One Pass
#
# The original run (slurm/19_ood_eval.sh) passed --skip-baseline to both
# invocations, so baseline_results was always empty. This fix runs the script
# ONCE without --skip-baseline, producing a single output file with both
# baseline and DPO results plus the comparison delta table.
#
# Expected runtime: ~2-3 hours on A100
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

mkdir -p slurm/logs results

echo "============================================"
echo "SLURM Job: OOD Eval Fix (both conditions)"
echo "Model: ${PRIMARY_MODEL}"
echo "Adapter: results/dpo_model"
echo "Data: data/ood_prompts/"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

# Verify OOD data exists
if [ ! -f data/ood_prompts/nlp_survey.jsonl ] || [ ! -f data/ood_prompts/political_typology.jsonl ]; then
    echo "ERROR: OOD benchmark data not found in data/ood_prompts/"
    exit 1
fi

# Single run — evaluates BOTH baseline (no adapter) and DPO (with adapter)
# in sequence, computes comparison deltas, saves everything to one file.
python scripts/09_ood_eval.py \
    --model "${PRIMARY_MODEL}" \
    --adapter-path results/dpo_model \
    --data-dir data/ood_prompts \
    --output results/ood_eval_results.json \
    --seed 42

echo "============================================"
echo "OOD eval fix complete: $(date)"
echo "Results: results/ood_eval_results.json"
echo "============================================"
