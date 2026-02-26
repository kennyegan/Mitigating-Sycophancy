#!/bin/bash
# =============================================================================
# Submit all SLURM jobs with dependency chains
#
# Submission order:
#   Job 1 (baseline) + Job 2 (controls) → run in parallel
#   Job 3 (probes)                      → depends on Job 1 completing
#   Job 4 (patching)                    → depends on Job 1 completing
#   Job 5 (base comparison)             → independent, can run in parallel
#
# Usage:
#   cd /path/to/Mitigating-Sycophancy
#   bash slurm/submit_all.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# Validate config
if [ "${SLURM_PARTITION}" = "TODO_PARTITION" ]; then
    echo "ERROR: Edit slurm/config.sh first — SLURM_PARTITION is still TODO"
    exit 1
fi
if [ "${PROJECT_DIR}" = "TODO_PROJECT_DIR" ]; then
    echo "ERROR: Edit slurm/config.sh first — PROJECT_DIR is still TODO"
    exit 1
fi

mkdir -p slurm/logs

echo "Submitting SLURM jobs..."
echo "Partition: ${SLURM_PARTITION}"
echo "Project:   ${PROJECT_DIR}"
echo ""

# Build common sbatch flags
COMMON_FLAGS="--partition=${SLURM_PARTITION} --gres=gpu:${GPU_TYPE}:${GPUS_PER_NODE} --cpus-per-task=${CPUS_PER_TASK}"
if [ -n "${SLURM_ACCOUNT}" ] && [ "${SLURM_ACCOUNT}" != "TODO_ACCOUNT" ]; then
    COMMON_FLAGS="${COMMON_FLAGS} --account=${SLURM_ACCOUNT}"
fi
if [ -n "${SLURM_QOS}" ]; then
    COMMON_FLAGS="${COMMON_FLAGS} --qos=${SLURM_QOS}"
fi

# Job 1: Baseline (no dependencies)
JOB1=$(sbatch ${COMMON_FLAGS} --mem=${MEM} --parsable slurm/01_baseline.sh)
echo "[Job 1] Baseline:         ${JOB1}"

# Job 2: Control groups (no dependencies, parallel with Job 1)
JOB2=$(sbatch ${COMMON_FLAGS} --mem=${MEM} --parsable slurm/02_control_groups.sh)
echo "[Job 2] Control groups:   ${JOB2}"

# Job 3: Probes (depends on baseline completing — needs the model cached)
JOB3=$(sbatch ${COMMON_FLAGS} --mem=80G --dependency=afterok:${JOB1} --parsable slurm/03_probes.sh)
echo "[Job 3] Probes:           ${JOB3}  (after Job 1)"

# Job 4: Patching (depends on baseline completing)
JOB4=$(sbatch ${COMMON_FLAGS} --mem=${MEM} --dependency=afterok:${JOB1} --parsable slurm/04_patching.sh)
echo "[Job 4] Patching:         ${JOB4}  (after Job 1)"

# Job 5: Base comparison (independent)
JOB5=$(sbatch ${COMMON_FLAGS} --mem=80G --parsable slurm/05_base_comparison.sh)
echo "[Job 5] Base comparison:  ${JOB5}"

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f slurm/logs/baseline_${JOB1}.out"
echo ""
echo "Dependency chain:"
echo "  Job 1 (baseline) ─┬─→ Job 3 (probes)"
echo "                     └─→ Job 4 (patching)"
echo "  Job 2 (controls)     → independent"
echo "  Job 5 (base)         → independent"
