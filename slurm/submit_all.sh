#!/bin/bash
# =============================================================================
# Submit full SLURM rerun matrix with explicit artifact-oriented dependencies.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

if [ "${SLURM_PARTITION}" = "TODO_PARTITION" ]; then
    echo "ERROR: Edit slurm/config.sh first — SLURM_PARTITION is still TODO"
    exit 1
fi
if [ "${PROJECT_DIR}" = "TODO_PROJECT_DIR" ]; then
    echo "ERROR: Edit slurm/config.sh first — PROJECT_DIR is still TODO"
    exit 1
fi

mkdir -p "${PROJECT_DIR}/slurm/logs"
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

echo "Submitting full rerun matrix..."
echo "Partition: ${SLURM_PARTITION}"
echo "Project:   ${PROJECT_DIR}"
echo ""

JOB1=$(submit_gpu_job slurm/01_baseline.sh "${MEM}")
echo "[Job 1]  baseline:                ${JOB1}"

JOB2=$(submit_gpu_job slurm/02_control_groups.sh "${MEM}")
echo "[Job 2]  control group generation:${JOB2}"

JOB3=$(submit_gpu_job slurm/03_probes.sh "80G" "${JOB1}")
echo "[Job 3]  probes:                  ${JOB3} (after Job 1)"

JOB4=$(submit_gpu_job slurm/04_patching.sh "${MEM}" "${JOB1}")
echo "[Job 4]  patching:                ${JOB4} (after Job 1)"

JOB5=$(submit_gpu_job slurm/05_base_comparison.sh "80G")
echo "[Job 5]  base comparison:         ${JOB5}"

JOB6=$(submit_gpu_job slurm/06_probe_control.sh "80G" "${JOB1}")
echo "[Job 6]  probe control:           ${JOB6} (after Job 1)"

JOB7=$(submit_gpu_job slurm/07_head_ablation.sh "80G" "${JOB4}")
echo "[Job 7]  head ablation:           ${JOB7} (after Job 4)"

JOB8=$(submit_gpu_job slurm/08_control_analysis.sh "80G" "${JOB2}")
echo "[Job 8]  control analysis:        ${JOB8} (after Job 2)"

JOB9=$(submit_gpu_job slurm/09_top10_ablation.sh "80G" "${JOB7}")
echo "[Job 9]  top-10 ablation:         ${JOB9} (after Job 7)"

JOB10=$(submit_gpu_job slurm/10_probe_control_balanced.sh "80G" "${JOB1}")
echo "[Job 10] balanced probe control:  ${JOB10} (after Job 1)"

JOB11=$(submit_gpu_job slurm/11_top10_full_gsm8k.sh "80G" "${JOB9}")
echo "[Job 11] top-10 full gsm8k:      ${JOB11} (after Job 9)"

JOB12=$(submit_gpu_job slurm/12_steering.sh "80G" "${JOB4}")
echo "[Job 12] steering:                ${JOB12} (after Job 4)"

FINAL_DEP="${JOB1}:${JOB2}:${JOB3}:${JOB4}:${JOB5}:${JOB6}:${JOB7}:${JOB8}:${JOB9}:${JOB10}:${JOB11}:${JOB12}"
JOB13=$(sbatch "${COMMON_FLAGS[@]}" "--mem=4G" "--time=00:30:00" \
    --dependency="afterok:${FINAL_DEP}" --parsable slurm/13_collect_manifest.sh)
echo "[Job 13] consolidated manifest:   ${JOB13} (after Jobs 1-12)"

echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f slurm/logs/steering_${JOB12}.out"
echo ""
echo "Final manifest path:"
echo "  results/full_rerun_manifest.json"
