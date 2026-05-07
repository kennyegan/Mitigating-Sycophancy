# Compute Environment Record

This file documents the hardware and software environment used to produce the
cached artifacts in `results/`.  Most fields below are recovered from the
`metadata.environment` block stored in each result JSON; fields that were not
recorded at run time are explicitly marked **not measured**.

## 1. Software environment

The following block is identical across every cached `metadata.environment`
field in the supplement:

```
python_version : 3.10.19
torch_version  : 2.10.0+cu128
cuda_available : true
cuda_version   : 12.8
device_count   : 1
```

Other libraries (versions not stamped in cached JSONs):

| Library | Approximate version | Notes |
|---|---|---|
| transformer_lens | 1.x (matches torch 2.10) | not measured exactly |
| trl | 0.x | DPO trainer |
| peft | 0.x | LoRA adapters |
| accelerate | 0.x | training |
| transformers | 4.x | inference / training |
| datasets | 2.x | TruthfulQA, GSM8k, MMLU |
| scikit-learn | 1.x | logistic probes |
| matplotlib | 3.x | figures |
| numpy | 1.24+ | |

## 2. Hardware

| | |
|---|---|
| GPU class | A100-class (40 GB / 80 GB variants) |
| Mixed precision | bfloat16 throughout (training and inference) |
| CPU / RAM | not measured at run time; jobs ran on a shared HPC partition |
| Disk | ≈ 50 GB peak per training job (model + adapters + checkpoints) |
| Network | required only to download HuggingFace models / datasets |

## 3. Approximate compute breakdown

The 120 GPU-hour total reported in the paper aggregates the following
**estimated** per-stage costs.  Per-stage figures were not recorded as
wall-clock fields and should be treated as estimates:

| Stage | Estimated GPU-hours |
|---|---|
| Llama-3-8B-Instruct baseline + probes | ~3 |
| Activation patching heatmap | ~6 |
| Patching bootstrap (5 resamples) | ~20 |
| Head ablation (top-3 / top-10) | ~5 |
| Top-10 ablation w/ full GSM8k | ~5 |
| DPO training × 3 seeds | ~12 |
| DPO eval w/ full GSM8k × 3 seeds | ~5 |
| SFT training | ~4 |
| SFT eval w/ full GSM8k | ~2 |
| Mistral cross-architecture (baseline + DPO + ablation) | ~25 |
| Qwen-2.5-14B replication | ~25 |
| OOD opinion + Anthropic OOD | ~2 |
| Free-form generation (base + DPO) | ~2 |
| DPO size-sensitivity (N=100/200/800) | ~4 |
| **Total (estimated)** | **~120** |

## 4. API costs

The free-form judge uses an LLM-as-judge; total estimated cost ≈ **$8** for
both base and DPO transcripts on the 150-prompt free-form benchmark.

## 5. Cached verification vs. full rerun

| | Cached verification | Full rerun |
|---|---|---|
| Hardware | CPU only | A100-class GPU |
| Wall time | < 5 min | ~120 GPU-hours |
| Disk | ~13 MB | ~50 GB |
| RAM | ~4 GB | ~64 GB recommended |
| Network | none | HuggingFace + datasets |
| Coverage | every numerical paper claim | full pipeline |

## 6. Result-file timestamp range

Earliest cached artifact: 2026-03-04 17:09 UTC
Latest cached artifact:   2026-05-05 02:37 UTC

These are run timestamps; they do not encode any author-identifying
information.

## 7. Outstanding TODOs (compute side)

- **C1** — Per-stage measured wall-clock and GPU-hour breakdown.  Per-stage
  figures above are estimates; the paper-level 120 GPU-hour total is
  reported but not derivable from cached JSON timestamps because jobs
  shared a slurm queue.
