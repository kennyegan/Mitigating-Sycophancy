#!/bin/bash
#SBATCH --job-name=smoke-14b
#SBATCH --output=slurm/logs/smoke_14b_%j.out
#SBATCH --error=slurm/logs/smoke_14b_%j.err
#SBATCH --partition=gpu
#SBATCH --account=pi_larsonj_wit_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

set -euo pipefail
source /work/pi_larsonj_wit_edu/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh
module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}
export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"

python -c "
import torch
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained(
    'Qwen/Qwen2.5-14B-Instruct',
    dtype=torch.float16,
    device='cuda'
)
print(f'Load VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB')
with torch.no_grad():
    tokens = model.to_tokens('The capital of France is')
    logits, cache = model.run_with_cache(tokens)
print(f'Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB')
print(f'Layers: {model.cfg.n_layers}, d_model: {model.cfg.d_model}')
print(f'Cache entries: {len(cache)}')
"
