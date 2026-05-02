#!/bin/bash
# =============================================================================
# SLURM Configuration — EDIT THESE FOR YOUR CLUSTER
# =============================================================================
# This file is sourced by all SLURM job scripts.
# Fill in the TODOs below before submitting jobs.

# --- Cluster-specific settings ---
export SLURM_PARTITION="gpu"                   # Unity cluster GPU partition
export SLURM_ACCOUNT="pi_larsonj_wit_edu"     # PI account on Unity
export SLURM_QOS=""                            # e.g. "high" (leave empty if not needed)

# --- Conda environment ---
export CONDA_MODULE="conda/latest"             # loads miniforge3-24.7.1 on Unity
export CONDA_ENV="sycophancy-lab"              # your conda env name

# --- Project paths ---
export PROJECT_DIR="/work/pi_larsonj_wit_edu/egank2_wit_edu/Mitigating-Sycophancy"
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"  # HuggingFace cache (avoids re-downloads)
export HF_HUB_OFFLINE=1                              # use cached weights; skip HF.co license check (fixes 401 on gated repos)
export TORCH_HOME="${PROJECT_DIR}/.cache/torch"

# --- GPU settings ---
export GPU_TYPE="a100"                         # Request A100 (40/80GB VRAM) for TransformerLens
export GPUS_PER_NODE=1                         # 1 GPU is sufficient for 8B models
export CPUS_PER_TASK=8                         # CPU cores for data loading
export MEM="64G"                               # RAM (64G safe for Llama-3-8B)

# --- Models ---
export PRIMARY_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
export BASE_MODEL="meta-llama/Meta-Llama-3-8B"
export REPLICATION_MODEL="mistralai/Mistral-7B-Instruct-v0.1"

# --- HuggingFace token (needed for Llama-3 gated models) ---
export HF_TOKEN="${HF_TOKEN:-}"
