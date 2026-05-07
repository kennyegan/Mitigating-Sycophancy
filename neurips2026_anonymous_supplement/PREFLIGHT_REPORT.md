# Preflight Report — Anonymous Supplement Build

This report is generated **before** any files are copied or sanitized into the
supplement.  It documents the inspection performed on the source repository,
candidate artifacts, planned inclusions, and the compatibility of paper claims
with cached results.

The supplement build is **safe to proceed**.  No ABORT condition was triggered.
Several **PARTIAL** items are recorded below and tracked in `MANIFEST.md`.

---

## 1. Repo areas inspected

| Area | Purpose | Notes |
|------|---------|-------|
| `scripts/` | numbered pipeline scripts (`00_…12_`) and aggregators | Will be copied (sanitized). |
| `src/` | Python package (`data/`, `eval/`, `analysis/`, `models/`, `utils/`) | Will be copied (sanitized). |
| `slurm/` | cluster launch scripts | **Excluded** — contain account names, absolute paths.  A neutral example will be hand-written. |
| `results/` | cached JSON/CSV results | **Selectively** copied — only files needed to verify paper claims. |
| `results/dpo_model*`, `results/sft_model`, `results/mistral/dpo_model` | LoRA adapter checkpoints | **Excluded** (≈415 MB each, ~2 GB total). |
| `figures/` | generated PDF/PNG figures | Will be copied. |
| `data/freeform/`, `data/ood_prompts/` | prompt datasets | Will be copied. |
| `data/processed/master_sycophancy*.jsonl` | merged eval dataset | Will be copied. |
| `notebooks/`, `outputs/`, `docs/`, `paper_todo.md`, `SESSION_HANDOFF.md`, `Research_Proposal.md`, `context.md` | drafts, internal notes | **Excluded** — contain author/institution references. |
| `*.tex`, `paper.pdf`, `paper.bbl`, `paper.aux`, `*.bak`, `paper.md*` | manuscript and build artifacts | **Excluded** — handled by the main paper submission. |
| `.git/`, `.cache/`, `.pytest_cache/`, `__pycache__/`, `*.egg-info` | metadata/caches | **Excluded**. |
| `uv.lock` | full pinned lockfile | Excluded; supplied a minimal `requirements.txt`. |

---

## 2. Paper-claim → cached-artifact traceability

Every paper number planned for `verify_claims.py` was traced to a concrete
cached JSON file.  The verifier reads from these files; it does not hard-code
numbers and does not invent values.

| Claim | Expected | Source file | Field path | Observed |
|-------|----------|-------------|-----------|----------|
| Llama overall sycophancy | 28.0% | `results/baseline_llama3_summary.json` | `overall.sycophancy_rate` | 0.2800 |
| Llama opinion sycophancy | 82.4% | same | `per_source.anthropic_opinion.sycophancy_rate` | 0.8249 |
| Llama factual sycophancy | 1.6% | same | `per_source.truthfulqa_factual.sycophancy_rate` | 0.0160 |
| Llama reasoning sycophancy | 0.0% | same | `per_source.gsm8k_reasoning.sycophancy_rate` | 0.0000 |
| Probe best layer | 1 | `results/probe_control_balanced_results.json` | `per_position_summary.final.best_layer` | 1 |
| Robust tracking | 59.9% | same | `…best_robust_rate` | 0.5987 |
| Social compliance | 18.0% | same | `…best_social_compliance_rate` | 0.1800 |
| Belief corruption | 10.1% | same | `…best_belief_corruption_rate` | 0.1007 |
| Top-3 patching Jaccard | 0.09 | `results/patching_bootstrap.json` | `aggregate.pairwise_jaccard.top3.mean` | 0.09 |
| Llama top-10 zero-ablation Δ | +0.5 pp | `results/top10_ablation_full_gsm8k.json` | conditions.{baseline,all_zero}.sycophancy.overall_sycophancy_rate | +0.47 pp |
| Mistral top-10 zero-ablation Δ | +1.0 pp | `results/mistral/top10_ablation_full_gsm8k.json` | same | +1.00 pp |
| DPO opinion (3-seed mean) | 57.1 ± 2.8% | `results/dpo_seed_summary.json` | `summary.opinion_sycophancy.mean/sd` | 0.5707 ± 0.0283 |
| DPO GSM8k full (3-seed mean) | 40.2 ± 2.9% | `results/dpo_gsm8k_full_*.json` | `capabilities.gsm8k.accuracy` | 0.4023 ± 0.0294 |
| Baseline GSM8k accuracy | 33.2% | `results/top10_ablation_full_gsm8k.json` | `conditions.baseline.gsm8k.accuracy` | 0.3321 |
| SFT GSM8k full | 5.8% | `results/sft_gsm8k_full.json` | `capabilities.gsm8k.accuracy` | 0.0584 |
| Mistral DPO factual sycophancy | 100% | `results/mistral/dpo_eval_results.json` | `behavioral.per_source.truthfulqa_factual.sycophancy_rate` | 1.000 |
| Mistral DPO GSM8k accuracy | 0.0% | same | `capabilities.gsm8k.accuracy` | 0.000 |
| Qwen baseline opinion | 75.3% | `results/stronger/baseline_summary.json` | `per_source.anthropic_opinion.sycophancy_rate` | 0.753 |
| Qwen baseline opinion mean comp. gap | +0.006 | same | `per_source.anthropic_opinion.mean_compliance_gap` | 0.0059 |
| Qwen top-3 zero-ablation Δ | +20.3 pp | `results/stronger/head_ablation_supplementary.json` | `conditions.{baseline,all_zero}.sycophancy.overall_sycophancy_rate` | +20.33 pp |
| Free-form overall sycophancy (baseline → DPO) | 2.66 → 2.43 | `results/freeform/comparison_summary.json` | `domains.overall.{baseline,dpo}.sycophancy.mean` | 2.66 → 2.433 |
| OOD retention (Protocol A ≈ 77%) | ~0.77 | `results/ood_opinion_eval_results.json` | derived from `comparison.condition_*.delta_pp` | **PARTIAL** (see below) |
| OOD retention (Protocol B ≈ 20%) | ~0.20 | same | derived | **PARTIAL** |

Notes on PARTIAL items:
- The OOD retention numbers in the paper are approximate ratios of out-of-distribution
  Δ-sycophancy to in-distribution Δ-sycophancy.  The cached file reports the
  per-condition Δs but does not name which condition is "Protocol A" vs.
  "Protocol B"; the verifier reports the closest matching condition rather than
  asserting the exact label.

---

## 3. Files planned for inclusion

The following will be copied (and sanitized where applicable):

```
configs/                     ← hand-written YAML extracted from training scripts (no secrets)
scripts/00_data_setup.py
scripts/01_run_baseline.py
scripts/02_train_probes.py
scripts/02b_probe_control.py
scripts/03_activation_patching.py
scripts/04_head_ablation.py
scripts/05_representation_steering.py
scripts/06_dpo_training.py
scripts/07_dpo_eval.py
scripts/08_ood_opinion_eval.py
scripts/09_ood_eval.py
scripts/10_sft_training.py
scripts/11_sft_eval.py
scripts/12_patching_bootstrap.py
scripts/aggregate_dpo_seeds.py
scripts/aggregate_size_sensitivity.py
scripts/parse_qwen_ablation_log.py
scripts/prepare_ood_benchmarks.py
scripts/99_collect_result_manifest.py
src/__init__.py + analysis/, data/, eval/, models/, utils/

results/baseline_llama3_summary.json + .csv
results/baseline_llama3_base_summary.json + .csv
results/probe_results_neutral_transfer.json
results/probe_results_llama3_base_neutral_transfer.json
results/probe_results_mixed_diagnostic.json
results/probe_control_results.json
results/probe_control_balanced_results.json
results/patching_heatmap.json
results/patching_bootstrap.json
results/head_importance.json
results/head_ablation_results.json
results/top10_ablation_results.json
results/top10_ablation_full_gsm8k.json
results/corrected_ablation_results.json
results/dpo_eval_results.json
results/dpo_eval_seed200.json
results/dpo_eval_seed300.json
results/dpo_gsm8k_full_results.json
results/dpo_gsm8k_full_seed200.json
results/dpo_gsm8k_full_seed300.json
results/dpo_seed_summary.json
results/dpo_training_metrics.json
results/sft_eval_results.json
results/sft_gsm8k_full.json
results/sft_training_metrics.json
results/steering_results.json (heavy; small subset taken)
results/steering_per_source_analysis.json
results/ood_eval_baseline.json
results/ood_eval_dpo.json
results/ood_eval_results.json
results/ood_opinion_eval_results.json
results/full_rerun_manifest.json
results/dpo_size_sensitivity/{N100,N200,N800}_eval.json + summary.json

results/mistral/baseline_summary.json + detailed.csv
results/mistral/dpo_eval_results.json
results/mistral/probe_control_balanced_results.json
results/mistral/top10_ablation_full_gsm8k.json
results/mistral/head_importance.json
results/mistral/patching_heatmap.json
results/mistral/manifest.json
(adapter weights excluded)

results/stronger/baseline_summary.json + details.csv
results/stronger/qwen_n200_baseline.json + .csv
results/stronger/qwen_diagnostic.json + .csv
results/stronger/probe_control_balanced.json
results/stronger/head_ablation_supplementary.json
results/stronger/head_ablation_partial.json

results/freeform/comparison_summary.json
results/freeform/llama3_base_scores.jsonl
results/freeform/llama3_dpo_scores.jsonl
results/freeform/audit_sample.jsonl
(transcripts excluded — large; if requested, trim a small sample)

results/manifests/*.json (small audit JSONs)

figures/*.pdf and *.png (12 figures)

data/processed/master_sycophancy.jsonl
data/processed/master_sycophancy_balanced.jsonl
data/processed/master_sycophancy_metadata.json
data/processed/master_sycophancy_balanced_metadata.json
data/processed/ood_opinion_benchmark.jsonl
data/freeform/*.jsonl + README.md
data/ood_prompts/*.jsonl + metadata.json + README.md

prompts/freeform_*.jsonl   (mirror of data/freeform/*)
rubrics/freeform_rubric.json (from src/eval/rubric.json)

examples/freeform_audit_sample.jsonl (small sample for review)
```

## 4. Files **excluded** and why

| Path / pattern | Reason |
|---------------|--------|
| `slurm/*.sh` | SBATCH `--account=…`, absolute project path, cluster username. Reproduction guidance written from scratch instead. |
| `results/*/dpo_model*/*.safetensors`, `*/sft_model*/*.safetensors` | Model adapters: ≈415 MB each, ≈2 GB total. Configs and training scripts retained instead. |
| `results/*/checkpoints/` | Intermediate optimizer state. Not needed for verification. |
| `results/*/tokenizer*.json`, `chat_template.jinja`, `special_tokens_map.json` | Reproduce from upstream HuggingFace model. |
| `results/dpo_model*/dpo_training_pairs.json`, `results/sft_model/sft_training_examples.json` | Large; the user-message text is regenerable from `data/processed/master_sycophancy.jsonl`. |
| `results/freeform/*_transcripts.jsonl` | Verbose; rubric scores already in `*_scores.jsonl`. A 10-row example is included under `examples/`. |
| `paper*.tex*`, `paper.pdf`, `paper.aux`, `paper.bbl`, `paper.log`, `paper.out`, `paper.md*`, `references.bib`, `neurips_2026.sty`, `checklist.tex*` | Manuscript artifacts. Submitted separately. |
| `paper_todo.md`, `SESSION_HANDOFF.md`, `Research_Proposal.md`, `context.md`, `research.md`, `sycophancy-mech-interp-research.md`, `sycophancy-mechinterp-research.md`, `neurips-execution-plan.md`, `neurips-plan.md`, `notebooks/01_baseline_colab.ipynb`, `outputs/*.md`, `docs/*.md`, `PROJECT_OVERVIEW.md`, `QUICKSTART.md`, `README.md` | Drafts / planning notes; may contain author/institution mentions. |
| `.git/`, `.cache/`, `.pytest_cache/`, `__pycache__/`, `*.egg-info`, `uv.lock` | Metadata/caches. |

## 5. Anonymity risk inventory

The following identifying patterns were detected in the source repo and will
either be removed (file excluded) or sanitized (file copied with substitutions).

| Pattern | Locations | Action |
|--------|-----------|--------|
| `pi_larsonj_wit_edu` | `slurm/*.sh`, `scripts/generate_figures_neurips.py:40` | slurm scripts excluded; figure script's hard-coded path replaced with `Path(__file__).resolve().parent.parent`. |
| `egank2_wit_edu` | `slurm/*.sh`, `scripts/generate_figures_neurips.py` | same |
| `/work/pi_larsonj_wit_edu/egank2_wit_edu/Mitigating-Sycophancy` | hard-coded absolute paths | replaced with relative paths |
| `wandb` Python dep | `requirements.txt` | dropped; nothing in supplement code calls it |
| Author names | manuscript files | manuscript excluded entirely |
| Email addresses | not detected in copied files | n/a |
| API keys / tokens | none detected | n/a |
| GitHub usernames | none detected in copied files | n/a |

A scan with `tools/check_anonymization.py` will be re-run after copying
to confirm no leak made it into the supplement.

## 6. Missing artifacts

| Item | Note |
|------|------|
| Mistral baseline GSM8k accuracy as a single field | Not stored in baseline summary; full-rerun would record it.  Paper's "9.3% → 0.0%" GSM8k claim is partially backed: the post-DPO 0.0% is in `results/mistral/dpo_eval_results.json:capabilities.gsm8k.accuracy`; the 9.3% pre-DPO baseline is recorded only in slurm log text and is not present in any cached JSON in this repo.  Will be flagged **PARTIAL** with a TODO. |
| OOD "Protocol A" / "Protocol B" labelling | Not labelled in cached JSON; only `condition_1`, `condition_2`, `condition_3` exist.  Verifier reports per-condition retention. |
| Free-form judge transcripts (full) | Excluded due to size; a 10-row example sample is provided. |
| Mistral sanity check (chat template, A/B mapping, label inversion check) | No dedicated cached JSON; chat template is documented in `src/eval/freeform_generate.py` and `slurm/mistral/*.sh`.  Sanity is implicitly covered by the agreement of post-DPO behavioral numbers with the manifest.  Marked **PARTIAL**. |
| Original hardware/CUDA/PyTorch version traceability | Each result JSON's `metadata.environment` block is preserved verbatim and aggregated in `compute/COMPUTE.md`. |

## 7. Conflicting result files

None of the conflicts below affect any paper claim; the verifier always reads
the canonical files listed in §2.

| Files | Difference | Resolution |
|-------|-----------|-----------|
| `results/dpo_eval_results.json` (n=200 GSM8k subset, 38.5%) vs `results/dpo_gsm8k_full_results.json` (full 1319-sample, 36.85%) | Different sample sizes | Paper uses full; verifier reads `dpo_gsm8k_full_*` only. |
| `results/top10_ablation_results.json` (n=200 GSM8k) vs `results/top10_ablation_full_gsm8k.json` (n=1319) | Same | Verifier uses `…_full_gsm8k.json`. |
| `results/head_ablation_results.json` (3-head L1H20/L5H5/L4H28) vs `results/corrected_ablation_results.json` (revised 3-head set) | Different head sets — both cached for transparency | Both included; paper's top-10 claim uses `top10_…_full_gsm8k.json`. |
| `results/steering_results.json` and `results/steering_results.json.checkpoint.json` | Identical content (checkpoint) | Only `steering_results.json` copied. |

## 8. Decision

**Safe to proceed.**

Build will continue with conservative copying, sanitization, and verification.
Any number that cannot be recomputed from a cached file is documented as
**PARTIAL** in `MANIFEST.md` and is not asserted in `verify_claims.py`.
