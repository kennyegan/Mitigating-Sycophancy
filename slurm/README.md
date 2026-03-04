# SLURM Orchestration

This directory uses static-safe job scripts and submit-time resource control.

## Configure

Edit `slurm/config.sh`:

- `SLURM_PARTITION`
- `SLURM_ACCOUNT`
- `CONDA_MODULE`
- `CONDA_ENV`
- `PROJECT_DIR`
- GPU/CPU/MEM defaults

Do not hardcode Hugging Face tokens in repo files. Set `HF_TOKEN` in your shell environment.

## Submit Full Matrix

```bash
bash slurm/submit_all.sh
```

This submits Jobs 1–13:

- 1–12: data/baseline/probes/patching/ablation/steering tracks
- 13: consolidated artifact manifest (`results/full_rerun_manifest.json`)

## Reliability Contracts

- Each job exits non-zero if expected artifacts are missing/empty.
- Steering job validates both:
  - `results/steering_results.json`
  - `results/steering_results.json.checkpoint.json`
- Steering job auto-resumes when checkpoint file already exists.

## Monitor

```bash
squeue -u $USER
tail -f slurm/logs/steering_JOBID.out
```
