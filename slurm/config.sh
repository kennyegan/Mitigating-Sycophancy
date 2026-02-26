#!/bin/bash
# =============================================================================
# SLURM Configuration — EDIT THESE FOR YOUR CLUSTER
# =============================================================================
# This file is sourced by all SLURM job scripts.
# Fill in the TODOs below before submitting jobs.

# --- Cluster-specific settings ---
export SLURM_PARTITION="TODO_PARTITION"       # e.g. "gpu", "a100", "compute"
export SLURM_ACCOUNT="TODO_ACCOUNT"           # e.g. "mylab" (remove if not needed)
export SLURM_QOS=""                            # e.g. "high" (leave empty if not needed)

# --- Conda environment ---
export CONDA_MODULE="anaconda3"                # module to load (e.g. "anaconda3/2023.09")
export CONDA_ENV="sycophancy-lab"              # your conda env name

# --- Project paths ---
export PROJECT_DIR="TODO_PROJECT_DIR"          # e.g. "/home/kenegan/Mitigating-Sycophancy"
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"  # HuggingFace cache (avoids re-downloads)
export TORCH_HOME="${PROJECT_DIR}/.cache/torch"

# --- GPU settings ---
export GPUS_PER_NODE=1                         # 1 GPU is sufficient for 8B models
export CPUS_PER_TASK=8                         # CPU cores for data loading
export MEM="64G"                               # RAM (64G safe for Llama-3-8B)

# --- Models ---
export PRIMARY_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
export BASE_MODEL="meta-llama/Meta-Llama-3-8B"
export REPLICATION_MODEL="mistralai/Mistral-7B-Instruct-v0.3"

# --- HuggingFace token (needed for Llama-3 gated models) ---
# Option 1: Set here (not recommended for shared systems)
# export HF_TOKEN="hf_xxxxx"
# Option 2: Run `huggingface-cli login` once in your conda env before submitting
