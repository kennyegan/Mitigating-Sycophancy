# NeurIPS 2026 Anonymous Supplement

**Paper:** *Mechanisms and Limits of Sycophancy Mitigation in Instruction-Tuned LLMs*

This is the **anonymous code/results supplement** that accompanies the
submission.  It contains the minimum code, configs, prompts, rubrics, and
cached experimental outputs needed to verify the paper's main tables and
figures **from cached artifacts** without rerunning ≈120 GPU-hours of
experiments.

## Anonymity statement

All author names, affiliations, e-mail addresses, GitHub usernames, cluster
account identifiers, project paths, and any other identifying metadata have
been removed.  Personal absolute paths in scripts have been replaced with
relative paths.  Manuscript files are **not** included.  See
[`ANONYMIZATION.md`](ANONYMIZATION.md) for a full record of what was checked
and what was removed, and run `python tools/check_anonymization.py` to verify.

## What the supplement supports

The cached outputs in `results/` support the following paper claims (all
traceable line-items in [`MANIFEST.md`](MANIFEST.md)):

1. **Llama-3-8B-Instruct baseline sycophancy.** Overall 28.0%, opinion
   82.4%, factual 1.6%, reasoning 0.0%.
2. **Probe decomposition.** Layer 1 best-transfer; robust tracking 59.9%,
   social compliance 18.0%, belief corruption 10.1%; neutral-transfer probe
   design with randomized answer positions.
3. **Patching → ablation dissociation.** Patching identifies early-layer
   carriers, but top-10 zero-ablation gives no sycophancy reduction
   (Llama Δ +0.5 pp; Mistral Δ +1.0 pp); top-3 head-set Jaccard 0.09.
4. **DPO vs. SFT on Llama-3.** DPO opinion 82.4 → 57.1 ± 2.8% (3 seeds),
   GSM8k preserved 33.2% → 40.2 ± 2.9%; SFT on identical data collapses
   GSM8k to 5.8%.  Seeds 100, 200, 300.
5. **Cross-model bounds.** Mistral DPO collapses GSM8k to 0.0% with
   factual sycophancy → 100%.  Qwen-2.5-14B opinion sycophancy 75.3% with
   near-zero compliance gap (+0.006), and top-3 ablation **increases**
   sycophancy by +20.3 pp.
6. **OOD / free-form.** OOD opinion sycophancy retention by condition;
   free-form judge-scored sycophancy 2.66 → 2.43 (CI crosses zero).

## Folder layout

```
README.md                ← this file
REPRODUCE.md             ← exact reproduction commands and compute requirements
MANIFEST.md              ← table mapping every paper claim to a cached file + verifier
ANONYMIZATION.md         ← what was removed and how to re-check
PREFLIGHT_REPORT.md      ← record of preflight inspection performed before build
requirements.txt         ← dependencies (stdlib-only verification; matplotlib for figures)

configs/                 ← YAML configs distilled from training scripts
scripts/                 ← pipeline scripts (sanitized) + thin verification wrappers
src/                     ← Python package (data loading, evaluation, judge, utils)
results/                 ← cached JSON / CSV outputs (no model weights)
results/mistral/         ← cross-architecture replication
results/stronger/        ← Qwen-2.5-14B results
results/freeform/        ← free-form judge scores + 10-row audit sample
results/dpo_size_sensitivity/  ← DPO size ablation
results/manifests/       ← per-experiment provenance manifests
figures/                 ← 12 generated figures (PDF + PNG)
prompts/                 ← prompt JSONLs used for free-form generation
rubrics/                 ← judge rubric (5-dimension, anchored)
data/                    ← processed dataset, OOD prompts, free-form prompts
examples/                ← small audit-sample of free-form transcripts (10 rows)
release/                 ← zip-creation script (used by us, runnable by reviewers)
tools/                   ← anonymity checker
compute/                 ← hardware / software environment record (`COMPUTE.md`)
```

## Fast verification (recommended for reviewers)

This path is **CPU-only** and reads only cached JSON / CSV files; it does not
download models, run inference, train probes, or do GPU work.

```
cd neurips2026_anonymous_supplement
python tools/check_anonymization.py     # PASS expected
python scripts/check_manifest.py        # PASS expected
python scripts/verify_claims.py         # prints claim table; PASS / WARN / FAIL per row
python scripts/make_table1.py           # writes results/derived/table1.{md,csv}
python scripts/make_table3.py           # DPO vs SFT
python scripts/make_table4.py           # cross-model generality bounds
python scripts/make_figures.py          # regenerates figures from cached JSON when possible
```

All five scripts work with **only the Python standard library** plus
`matplotlib`/`numpy`/`pandas` for figure regeneration.

## Compute Requirements

### A — Fast cached-artifact verification (this is what reviewers run)

| | |
|--|--|
| Hardware | CPU only (any modern x86_64 / ARM64) |
| RAM | ≤ 4 GB; comfortably runs on a 8-GB laptop |
| Disk | ~13 MB unpacked |
| GPU | not required |
| Network | not required (everything is in the zip) |
| Wall time | < 5 minutes including figures |
| Python | 3.10 or later |
| Dependencies | none for verification; `matplotlib`+`numpy`+`pandas` for figures |
| Coverage | All numerical claims listed in §"What the supplement supports" |

### B — Full rerun

| Stage | Hardware | Approx. wall time |
|-------|----------|-------------------|
| Baseline behavioral evaluation (Llama-3-8B) | 1× A100 40/80 GB | ~1 h |
| Probe training + neutral-transfer eval | 1× A100 | ~2 h |
| Activation patching (heatmap) | 1× A100 | ~6 h |
| Patching bootstrap (5 resamples) | 1× A100 | ~20 h |
| Head ablation (top-3 / top-10) | 1× A100 | ~3 h |
| Top-10 ablation w/ full GSM8k 1319 | 1× A100 | ~5 h |
| DPO training (3 seeds) | 1× A100 80 GB | ~4 h × 3 |
| DPO evaluation w/ full GSM8k (3 seeds) | 1× A100 | ~1.7 h × 3 |
| SFT training | 1× A100 80 GB | ~4 h |
| SFT evaluation w/ full GSM8k | 1× A100 | ~1.7 h |
| Mistral cross-model replication | 1× A100 | ~12 h |
| Qwen-2.5-14B replication | 1× A100 80 GB | ~12 h |
| OOD opinion + Anthropic OOD | 1× A100 | ~1 h |
| Free-form generation (Llama base + DPO) | 1× A100 | ~1 h |
| Free-form judge scoring (LLM-as-judge) | API | ~$8 (estimated) |
| **Total compute (estimated)** | A100-class | **~120 GPU-hours** |
| GPU memory | 40 GB sufficient for inference; 80 GB recommended for DPO/SFT |
| Mixed precision | bfloat16 throughout |
| Model downloads required | `meta-llama/Meta-Llama-3-8B-Instruct`, `meta-llama/Meta-Llama-3-8B`, `mistralai/Mistral-7B-Instruct-v0.1`, `Qwen/Qwen2.5-14B-Instruct` |
| Datasets | Anthropic `model-written-evals` (sycophancy, NLP survey, political typology), GSM8k (test, 1319), TruthfulQA (`truthful_qa`), MMLU (subset, 500) |

The 120 GPU-hour estimate aggregates all stages above; per-stage figures are
**estimated, not measured wall-clock.**  See [`compute/COMPUTE.md`](compute/COMPUTE.md)
for the package-version block recovered from each result file's `metadata.environment`.

### Per-script compute summary

| Script | Type | Notes |
|--------|------|-------|
| `tools/check_anonymization.py` | CPU, stdlib | < 5 s |
| `scripts/check_manifest.py` | CPU, stdlib | < 1 s |
| `scripts/verify_claims.py` | CPU, stdlib | < 5 s |
| `scripts/make_table1.py` | CPU, stdlib | < 1 s |
| `scripts/make_table3.py` | CPU, stdlib | < 1 s |
| `scripts/make_table4.py` | CPU, stdlib | < 1 s |
| `scripts/make_figures.py` | CPU, matplotlib | < 1 min |
| `scripts/01_run_baseline.py` | GPU required | ~1 h on 1× A100 |
| `scripts/02_train_probes.py` | GPU required | ~2 h |
| `scripts/03_activation_patching.py` | GPU required | ~6 h |
| `scripts/04_head_ablation.py` | GPU required | ~3-5 h |
| `scripts/06_dpo_training.py` | GPU required, 80 GB | ~4 h |
| `scripts/07_dpo_eval.py` | GPU required | ~1.7 h |
| `scripts/10_sft_training.py` | GPU required, 80 GB | ~4 h |
| `scripts/11_sft_eval.py` | GPU required | ~1.7 h |
| `scripts/12_patching_bootstrap.py` | GPU required | ~20 h (5 resamples) |
| `src/eval/freeform_generate.py` | GPU required | ~1 h |
| `src/eval/freeform_judge.py` | API key required | ~$8 |

## Caveats and limitations

- **Adapter weights are not bundled.** LoRA adapters for the DPO/SFT runs
  total ≈ 2 GB and are excluded.  Configs and training scripts are included
  so they can be re-trained.
- **Free-form transcripts are not bundled.**  Aggregate scores and a
  10-row example are included; the full transcript JSONLs would have added
  ≈ 100 MB without changing any aggregate number.
- **Some paper numbers are approximate ratios** (OOD retention; Protocol-A /
  Protocol-B labels).  The verifier reports the per-condition source numbers
  and the closest-match retention; it does not assert label identity.
- **`results/full_rerun_manifest.json`** lists artifact existence with
  per-file status; it is mirrored at `results/full_rerun_manifest.json` in this
  package and is consumed by `scripts/check_manifest.py`.
