#!/bin/bash
#SBATCH --job-name=syc-dpo-train
#SBATCH --output=slurm/logs/dpo_train_%j.out
#SBATCH --error=slurm/logs/dpo_train_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# Job 17: DPO Fine-Tuning for Sycophancy Reduction
#
# Fine-tunes Llama-3-8B-Instruct with DPO + LoRA to prefer honest responses
# over sycophantic ones. Generates 400 training pairs from the Anthropic
# opinion dataset using seed=100 (disjoint from the seed=42 benchmark).
#
# Expected runtime: ~2-6 hours on A100 (depends on pair count and epochs)
# Output: results/dpo_model/ (LoRA adapter)
#         results/dpo_training_metrics.json
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
echo "SLURM Job: DPO Fine-Tuning for Sycophancy"
echo "Model: ${PRIMARY_MODEL}"
echo "Train pairs: 400"
echo "Seed: 100 (disjoint from benchmark seed=42)"
echo "Epochs: 3"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time: $(date)"
echo "============================================"

python scripts/06_dpo_training.py \
    --model "${PRIMARY_MODEL}" \
    --n-train-pairs 400 \
    --seed 100 \
    --output-dir results/dpo_model \
    --epochs 3 \
    --lr 5e-5 \
    --lora-rank 16 \
    --batch-size 4 \
    --beta 0.1 \
    --eval-split 0.1

echo "============================================"
echo "DPO training complete: $(date)"
echo "============================================"
