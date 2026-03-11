#!/bin/bash
# =============================================================================
# Submit Mistral-7B-Instruct replication pipeline.
#
# Runs the core experiments (baseline, probes, patching, ablation, steering)
# on Mistral-7B-Instruct-v0.3 to validate cross-architecture generalization.
#
# All outputs go to results/mistral/ to avoid overwriting Llama-3 results.
#
# Usage:
#   bash slurm/submit_mistral.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

if [ "${SLURM_PARTITION}" = "TODO_PARTITION" ]; then
    echo "ERROR: Edit slurm/config.sh first — SLURM_PARTITION is still TODO"
    exit 1
fi

mkdir -p "${PROJECT_DIR}/slurm/logs"
mkdir -p "${PROJECT_DIR}/results/mistral"
cd "${PROJECT_DIR}"

COMMON_FLAGS=(
    "--partition=${SLURM_PARTITION}"
    "--gres=gpu:${GPU_TYPE}:${GPUS_PER_NODE}"
    "--cpus-per-task=${CPUS_PER_TASK}"
)
if [ -n "${SLURM_ACCOUNT}" ] && [ "${SLURM_ACCOUNT}" != "TODO_ACCOUNT" ]; then
    COMMON_FLAGS+=("--account=${SLURM_ACCOUNT}")
fi
if [ -n "${SLURM_QOS}" ]; then
    COMMON_FLAGS+=("--qos=${SLURM_QOS}")
fi

submit_gpu_job() {
    local script_path="$1"
    local mem="$2"
    local dependency="${3:-}"
    local cmd=(sbatch "${COMMON_FLAGS[@]}" "--mem=${mem}" --parsable)
    if [ -n "${dependency}" ]; then
        cmd+=("--dependency=afterok:${dependency}")
    fi
    cmd+=("${script_path}")
    "${cmd[@]}"
}

echo "============================================"
echo "Submitting Mistral-7B replication pipeline"
echo "Model: ${REPLICATION_MODEL}"
echo "Partition: ${SLURM_PARTITION}"
echo "Project: ${PROJECT_DIR}"
echo "============================================"
echo ""

# Job M1: Baseline
JOB_M1=$(submit_gpu_job slurm/mistral/01_baseline.sh "${MEM}")
echo "[Job M1] baseline:       ${JOB_M1}"

# Job M2: Probes (after baseline)
JOB_M2=$(submit_gpu_job slurm/mistral/02_probes.sh "80G" "${JOB_M1}")
echo "[Job M2] probes:         ${JOB_M2} (after M1)"

# Job M3: Patching (after baseline)
JOB_M3=$(submit_gpu_job slurm/mistral/03_patching.sh "80G" "${JOB_M1}")
echo "[Job M3] patching:       ${JOB_M3} (after M1)"

# Job M4: Ablation (after patching — needs head_importance.json to pick heads)
JOB_M4=$(submit_gpu_job slurm/mistral/04_ablation.sh "80G" "${JOB_M3}")
echo "[Job M4] ablation:       ${JOB_M4} (after M3)"

# Job M5: Steering (after patching)
JOB_M5=$(submit_gpu_job slurm/mistral/05_steering.sh "80G" "${JOB_M3}")
echo "[Job M5] steering:       ${JOB_M5} (after M3)"

echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f slurm/logs/mistral_steering_\${JOB_M5}.out"
echo ""
echo "Results will be in: results/mistral/"
