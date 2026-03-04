#!/bin/bash
#SBATCH --job-name=syc-manifest
#SBATCH --output=slurm/logs/manifest_%j.out
#SBATCH --error=slurm/logs/manifest_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

source "/home/egank2_wit_edu/Mitigating-Sycophancy/slurm/config.sh"

module load ${CONDA_MODULE}
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}

mkdir -p results slurm/logs

python scripts/99_collect_result_manifest.py --output results/full_rerun_manifest.json

if [ ! -s results/full_rerun_manifest.json ]; then
    echo "ERROR: consolidated manifest missing: results/full_rerun_manifest.json"
    exit 2
fi

echo "Consolidated manifest complete: $(date)"
