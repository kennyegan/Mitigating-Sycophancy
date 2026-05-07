# Reproduction Guide

Two reproduction paths are supported.  **Path A** is intended for reviewers and
takes < 5 minutes on a laptop.  **Path B** rebuilds every cached artifact from
scratch and requires ≈120 A100-GPU-hours.

All commands below are intended to be run from inside this folder.

```
cd neurips2026_anonymous_supplement
```

---

## Path A — Fast verification from cached artifacts

### A.0  Requirements

- Python ≥ 3.10
- (optional, for figure regeneration) `pip install matplotlib numpy pandas`
- No GPU, no model downloads, no network access

### A.1  Anonymization check

```
python tools/check_anonymization.py
```

Scans the entire supplement folder for institution names, e-mail addresses,
absolute personal paths, GitHub usernames, API-key-shaped strings, WandB URLs,
and a configurable blocklist.  Prints `PASS` if clean.

### A.2  Manifest check

```
python scripts/check_manifest.py
```

Parses `MANIFEST.md`, asserts that every file referenced exists, and reports
any missing or empty artifact.

### A.3  Claim verification

```
python scripts/verify_claims.py
```

Reads numbers directly from cached JSON files, compares them to the paper's
reported values within tolerance, and prints a table:

```
claim                           expected   observed   tol     status
Llama-3-8B baseline overall     0.280      0.280      0.005   PASS
…
```

Numbers are **never hard-coded.**  Each row is recomputed from a cached
result file, and the file path is printed in the table footer.

### A.4  Tables

```
python scripts/make_table1.py    # baseline sycophancy by source
python scripts/make_table3.py    # DPO vs SFT preservation
python scripts/make_table4.py    # cross-model generality bounds
```

Each script writes both Markdown and CSV under `results/derived/`.

### A.5  Figures

```
python scripts/make_figures.py
```

Regenerates whichever figures are reproducible from cached JSON (e.g. seed
robustness, OOD retention, free-form 5-dimensional plot) using matplotlib.
Reports any figure for which a regenerator is not available; the original
PDF/PNG copies remain in `figures/` for those cases.

---

## Path B — Full rerun

This requires the full upstream Python environment (`requirements.txt`, full
list commented out for safety) and the original models.  Per-script
compute estimates are reproduced in `README.md` and at the top of each script.

### B.0  Environment

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Uncomment the GPU stack lines in requirements.txt before running:
# pip install torch>=2.0 transformers>=4.40 transformer_lens>=1.14 \
#             trl>=0.8 peft>=0.10 accelerate>=0.27 datasets>=2.14 \
#             scikit-learn>=1.3 einops>=0.7 jaxtyping>=0.2
```

Models used: `meta-llama/Meta-Llama-3-8B-Instruct`,
`meta-llama/Meta-Llama-3-8B`, `mistralai/Mistral-7B-Instruct-v0.1`,
`Qwen/Qwen2.5-14B-Instruct`.

### B.1  Prepare data

```
python scripts/00_data_setup.py            # builds data/processed/master_sycophancy*.jsonl
python scripts/prepare_ood_benchmarks.py   # builds data/processed/ood_opinion_benchmark.jsonl
```

### B.2  Baseline

```
python scripts/01_run_baseline.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --output results/baseline_llama3_summary.json
```
*GPU required, ≈1 h.*

### B.3  Probes

```
python scripts/02_train_probes.py        # full neutral-transfer (paper Layer-10 alt.)
python scripts/02b_probe_control.py      # final-position control (Layer-1 best)
```

### B.4  Patching + bootstrap

```
python scripts/03_activation_patching.py
python scripts/12_patching_bootstrap.py --n-resamples 5
```

### B.5  Head ablation

```
python scripts/04_head_ablation.py --top-k 10 \
    --output results/top10_ablation_full_gsm8k.json --gsm8k-samples 1319
```

### B.6  DPO + SFT (3 seeds DPO, 1 seed SFT)

```
for s in 100 200 300; do
  python scripts/06_dpo_training.py --seed $s \
    --output results/dpo_model_seed${s}/
  python scripts/07_dpo_eval.py --adapter results/dpo_model_seed${s}/ \
    --gsm8k-samples 1319 \
    --output results/dpo_gsm8k_full_seed${s}.json
done

python scripts/10_sft_training.py
python scripts/11_sft_eval.py --gsm8k-samples 1319 \
    --output results/sft_gsm8k_full.json
python scripts/aggregate_dpo_seeds.py    # builds results/dpo_seed_summary.json
```

### B.7  Cross-model

```
# Mistral
python scripts/01_run_baseline.py --model mistralai/Mistral-7B-Instruct-v0.1 \
    --output results/mistral/baseline_summary.json
python scripts/06_dpo_training.py --model mistralai/Mistral-7B-Instruct-v0.1 \
    --output results/mistral/dpo_model/
python scripts/07_dpo_eval.py --adapter results/mistral/dpo_model/ \
    --output results/mistral/dpo_eval_results.json
python scripts/04_head_ablation.py --model mistralai/Mistral-7B-Instruct-v0.1 \
    --top-k 10 --gsm8k-samples 1319 \
    --output results/mistral/top10_ablation_full_gsm8k.json

# Qwen
python scripts/01_run_baseline.py --model Qwen/Qwen2.5-14B-Instruct \
    --output results/stronger/baseline_summary.json
python scripts/04_head_ablation.py --model Qwen/Qwen2.5-14B-Instruct \
    --top-k 3 --output results/stronger/head_ablation_supplementary.json
```

### B.8  OOD + free-form

```
python scripts/09_ood_eval.py
python scripts/08_ood_opinion_eval.py
python -m src.eval.freeform_generate
python -m src.eval.freeform_judge       # requires JUDGE_API_KEY in environment
python -m src.eval.freeform_aggregate
```

### B.9  Final manifest

```
python scripts/99_collect_result_manifest.py    # rebuilds results/full_rerun_manifest.json
```

After a full rerun all cached files in `results/` will be regenerated; rerun
Path A's verification scripts to confirm the regenerated numbers still match
the paper.

---

## Per-script header convention

Each pipeline script in `scripts/` documents in its module-level docstring:

- **Compute type** — CPU-only, GPU-required, or API-required
- **Estimated runtime** on the original hardware
- **Expected input files**
- **Expected output files**

Every cached JSON also carries a `metadata.environment` block recording the
exact Python / Torch / CUDA versions used during the original run.  These are
aggregated in [`compute/COMPUTE.md`](compute/COMPUTE.md).
