# SLURM Jobs for Sycophancy Research

## Quick Start

### 1. Edit `slurm/config.sh`

Fill in the `TODO` placeholders:

```bash
export SLURM_PARTITION="your-gpu-partition"    # e.g. "gpu", "a100"
export SLURM_ACCOUNT="your-account"            # or leave empty
export CONDA_MODULE="anaconda3"                # your cluster's module name
export CONDA_ENV="sycophancy-lab"              # your conda env
export PROJECT_DIR="/path/to/Mitigating-Sycophancy"
```

### 2. Setup on the cluster (one-time)

```bash
# Clone repo
git clone https://github.com/kennyegan/Mitigating-Sycophancy.git
cd Mitigating-Sycophancy

# Create conda env
module load anaconda3  # or your module name
conda create -n sycophancy-lab python=3.10 -y
conda activate sycophancy-lab
pip install -r requirements.txt
pip install -e .

# Login to HuggingFace (needed for Llama-3 gated models)
huggingface-cli login

# Generate the 1,500-sample dataset
python scripts/00_data_setup.py --samples 500
```

### 3. Submit all jobs

```bash
bash slurm/submit_all.sh
```

Or submit individually:

```bash
sbatch slurm/01_baseline.sh
sbatch slurm/02_control_groups.sh
sbatch slurm/03_probes.sh      # after baseline finishes
sbatch slurm/04_patching.sh    # after baseline finishes
sbatch slurm/05_base_comparison.sh
```

## Job Summary

| Job | Script | Time | Depends On | Output |
|-----|--------|------|-----------|--------|
| 1 | `01_baseline.sh` | ~2-3h | — | `results/baseline_llama3_*` |
| 2 | `02_control_groups.sh` | ~1-2h | — | `data/processed/control_groups/` |
| 3 | `03_probes.sh` | ~4-6h | Job 1 | `results/probe_results_llama3_*` |
| 4 | `04_patching.sh` | ~6-10h | Job 1 | `results/patching_heatmap.json`, `results/head_importance.json` |
| 5 | `05_base_comparison.sh` | ~8-10h | — | `results/*_base_*` |

**Total GPU-hours: ~25-35h** (Jobs 1, 2, 5 run in parallel; 3, 4 wait for 1)

## Dependency Chain

```
Job 1 (baseline) ─┬─→ Job 3 (probes)
                   └─→ Job 4 (patching)
Job 2 (controls)     → independent
Job 5 (base)         → independent
```

## Monitoring

```bash
# Check job status
squeue -u $USER

# Watch output in real-time
tail -f slurm/logs/baseline_JOBID.out

# Check GPU usage
srun --jobid=JOBID nvidia-smi
```

## Troubleshooting

**OOM on GPU:** Llama-3-8B in float16 needs ~16GB VRAM. If your GPU has less, try adding `--device cpu` to the python commands (will be very slow).

**HuggingFace download fails:** Run `huggingface-cli login` in your conda env. Llama-3 requires accepting Meta's license at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct.

**Module not found:** Make sure `pip install -e .` was run in the conda env so `src/` is importable.

**SLURM partition/account errors:** Check available partitions with `sinfo` and accounts with `sacctmgr show associations user=$USER`.
