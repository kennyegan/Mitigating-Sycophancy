#!/bin/bash
#SBATCH --job-name=syc-ood-bench
#SBATCH --output=slurm/logs/ood_eval_%j.out
#SBATCH --error=slurm/logs/ood_eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --account=pi_larsonj_wit_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# =============================================================================
# Job: OOD Benchmark Sycophancy Evaluation
#
# Evaluates sycophancy on two published Anthropic sycophancy subcategories
# (NLP survey, political typology) that were never seen during DPO training.
#
# Step 1: Baseline evaluation (no adapter)
#         Output: results/ood_eval_baseline.json
#
# Step 2: DPO model evaluation (with LoRA adapter merged)
#         Output: results/ood_eval_dpo.json
#
# Data: data/ood_prompts/ (prepared by scripts/prepare_ood_benchmarks.py)
# Expected runtime: ~2-3 hours on A100 (1000 samples x 2 models x 4 fwd passes)
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
echo "SLURM Job: OOD Benchmark Evaluation"
echo "Model: ${PRIMARY_MODEL}"
echo "Data: data/ood_prompts/"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

# Verify OOD data exists
if [ ! -f data/ood_prompts/nlp_survey.jsonl ] || [ ! -f data/ood_prompts/political_typology.jsonl ]; then
    echo "ERROR: OOD benchmark data not found in data/ood_prompts/"
    echo "Run 'python scripts/prepare_ood_benchmarks.py' first (requires internet)."
    exit 1
fi

echo "OOD data files:"
wc -l data/ood_prompts/*.jsonl

# ---- Single run: baseline + DPO in one invocation ----
echo ""
echo "============================================"
echo "Running baseline (no adapter) + DPO evaluation"
echo "Time: $(date)"
echo "============================================"

python scripts/09_ood_eval.py \
    --model "${PRIMARY_MODEL}" \
    --adapter-path results/dpo_model \
    --data-dir data/ood_prompts \
    --output results/ood_eval_results.json \
    --seed 42

echo ""
echo "============================================"
echo "OOD benchmark evaluation complete: $(date)"
echo "Results: results/ood_eval_results.json"
echo "============================================"
