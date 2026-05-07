# Supplement Manifest

This file maps each paper claim and table/figure to:

- the cached result file it is computed from (`source`),
- the script that verifies or regenerates it (`verifier`),
- a status: **FOUND** (file exists and contains the asserted field), **PARTIAL**
  (file exists but the paper number is approximate / not directly stored), or
  **MISSING** (file is not in this supplement).

The verification scripts read the **observed** value from each `source` file
and compare to the paper's reported value within tolerance.  Numbers are
**never hard-coded**; if a value cannot be recomputed from a cached file the
row is marked PARTIAL/MISSING and a TODO is recorded below the table.

The full machine-readable manifest produced by the original pipeline is at
`results/full_rerun_manifest.json` (consumed by `scripts/check_manifest.py`).

---

## Section 1 — Llama-3-8B-Instruct baseline

| paper number | source | field | verifier | status |
|---|---|---|---|---|
| Overall sycophancy 28.0% | `results/baseline_llama3_summary.json` | `overall.sycophancy_rate` | `scripts/verify_claims.py`, `scripts/make_table1.py` | FOUND |
| Opinion sycophancy 82.4% | same | `per_source.anthropic_opinion.sycophancy_rate` | same | FOUND |
| Factual sycophancy 1.6% | same | `per_source.truthfulqa_factual.sycophancy_rate` | same | FOUND |
| Reasoning sycophancy 0.0% | same | `per_source.gsm8k_reasoning.sycophancy_rate` | same | FOUND |
| Llama-3-8B base (non-instruct) baseline | `results/baseline_llama3_base_summary.json` | various | `scripts/make_table1.py` | FOUND |

## Section 2 — Probe decomposition

| paper number | source | field | verifier | status |
|---|---|---|---|---|
| Best-transfer layer 1 | `results/probe_control_balanced_results.json` | `per_position_summary.final.best_layer` | `scripts/verify_claims.py` | FOUND |
| Robust tracking 59.9% | same | `…best_robust_rate` | same | FOUND |
| Social compliance 18.0% | same | `…best_social_compliance_rate` | same | FOUND |
| Belief corruption 10.1% | same | `…best_belief_corruption_rate` | same | FOUND |
| Other 12.1% | derived: `1 − robust − social − belief` | (verifier computes) | same | FOUND |
| Neutral-transfer probe design (randomized answer positions) | `src/data/base.py`, `src/eval/`, `data/processed/master_sycophancy_balanced_metadata.json` | `randomization` field in metadata | n/a (design check) | FOUND |
| Reference: Llama base probe results | `results/probe_results_llama3_base_neutral_transfer.json` | full | `scripts/make_figures.py` | FOUND |
| Reference: mixed-diagnostic probes | `results/probe_results_mixed_diagnostic.json` | full | `scripts/make_figures.py` | FOUND |

## Section 3 — Patching → ablation dissociation

| paper number | source | field | verifier | status |
|---|---|---|---|---|
| Patching layer-scan heatmap | `results/patching_heatmap.json` | `layer_results.mean_recovery_heatmap` | `scripts/make_figures.py` (fig1) | FOUND |
| Top-3 head Jaccard 0.09 | `results/patching_bootstrap.json` | `aggregate.pairwise_jaccard.top3.mean` | `scripts/verify_claims.py` | FOUND |
| Top-5 / Top-10 Jaccard | same | `aggregate.pairwise_jaccard.top5.mean`, `…top10.mean` | `scripts/make_table4.py` | FOUND |
| Llama top-10 zero-ablation Δ +0.5 pp | `results/top10_ablation_full_gsm8k.json` | `conditions.{baseline,all_zero}.sycophancy.overall_sycophancy_rate` | `scripts/verify_claims.py` | FOUND |
| Mistral top-10 zero-ablation Δ +1.0 pp | `results/mistral/top10_ablation_full_gsm8k.json` | same | same | FOUND |
| 3-head ablation control | `results/head_ablation_results.json`, `results/corrected_ablation_results.json` | `conditions.*` | `scripts/make_figures.py` (fig5) | FOUND |
| Patching head importance | `results/head_importance.json`, `results/mistral/head_importance.json` | full | `scripts/make_figures.py` | FOUND |

## Section 4 — DPO vs. SFT (Llama-3-8B-Instruct)

| paper number | source | field | verifier | status |
|---|---|---|---|---|
| DPO opinion 82.4 → 57.1 ± 2.8% | `results/dpo_seed_summary.json` | `summary.opinion_sycophancy.{mean,sd}` | `scripts/verify_claims.py`, `scripts/make_table3.py` | FOUND |
| DPO GSM8k full 40.2 ± 2.9% (3 seeds) | `results/dpo_gsm8k_full_results.json`, `…seed200.json`, `…seed300.json` | `capabilities.gsm8k.accuracy` | same | FOUND |
| Baseline GSM8k 33.2% | `results/top10_ablation_full_gsm8k.json` | `conditions.baseline.gsm8k.accuracy` | same | FOUND |
| SFT GSM8k 5.8% | `results/sft_gsm8k_full.json` | `capabilities.gsm8k.accuracy` | same | FOUND |
| DPO seeds 100, 200, 300 | `results/dpo_seed_summary.json` | `seeds` | same | FOUND |
| Per-seed DPO eval | `results/dpo_eval_results.json`, `…seed200.json`, `…seed300.json` | full | `scripts/make_figures.py` (fig7) | FOUND |
| DPO training metrics | `results/dpo_training_metrics.json` | full | `scripts/make_figures.py` | FOUND |
| SFT training metrics | `results/sft_training_metrics.json` | full | `scripts/make_figures.py` (fig11) | FOUND |
| DPO size sensitivity (N=100/200/800) | `results/dpo_size_sensitivity/{N100,N200,N800}_eval.json` + `summary.json` | full | `scripts/make_figures.py` | FOUND |

## Section 5 — Cross-model bounds

| paper number | source | field | verifier | status |
|---|---|---|---|---|
| Mistral baseline opinion 50.8% | `results/mistral/baseline_summary.json` | `per_source.anthropic_opinion.sycophancy_rate` | `scripts/make_table4.py` | FOUND |
| Mistral DPO factual sycophancy 100% | `results/mistral/dpo_eval_results.json` | `behavioral.per_source.truthfulqa_factual.sycophancy_rate` | `scripts/verify_claims.py` | FOUND |
| Mistral DPO GSM8k → 0.0% | same | `capabilities.gsm8k.accuracy` | same | FOUND |
| Mistral baseline GSM8k 9.3% | not stored as a single field in any cached JSON | derived from slurm log only | n/a | **PARTIAL** — TODO M1 |
| Mistral chat template / A-B mapping / decoding config | `src/eval/freeform_generate.py`, `src/data/base.py` | constants | n/a | **PARTIAL** — TODO M2 |
| Mistral label-inversion sanity check | not present as cached JSON | n/a | n/a | **PARTIAL** — TODO M3 |
| Qwen-2.5-14B opinion 75.3% | `results/stronger/baseline_summary.json` | `per_source.anthropic_opinion.sycophancy_rate` | `scripts/verify_claims.py` | FOUND |
| Qwen mean compliance gap +0.006 | same | `per_source.anthropic_opinion.mean_compliance_gap` | same | FOUND |
| Qwen top-3 ablation Δ +20.3 pp | `results/stronger/head_ablation_supplementary.json` | `conditions.{baseline,all_zero}.sycophancy.overall_sycophancy_rate` | same | FOUND |
| Qwen partial ablation conditions | `results/stronger/head_ablation_partial.json` | full | `scripts/make_figures.py` | FOUND |
| Qwen N=200 baseline (JSON) | `results/stronger/qwen_n200_baseline.json` | full | reference only | FOUND |
| Qwen N=200 baseline (CSV) | `results/stronger/qwen_n200_baseline.csv` | full | reference only | FOUND |
| Qwen 10-sample diagnostic (JSON) | `results/stronger/qwen_diagnostic.json` | full | reference only | FOUND |
| Qwen 10-sample diagnostic (CSV) | `results/stronger/qwen_diagnostic.csv` | full | reference only | FOUND |

## Section 6 — OOD / free-form

| paper number | source | field | verifier | status |
|---|---|---|---|---|
| In-distribution DPO Δ on opinion (−23.8 pp) | `results/ood_opinion_eval_results.json` | `comparison.in_distribution_reference.delta_pp` | `scripts/verify_claims.py`, `scripts/make_table4.py` | FOUND |
| Protocol A retention ≈ 77% | derived from `comparison.condition_*.delta_pp` | (label A↔condition mapping not stored in JSON) | `scripts/verify_claims.py` reports per-condition retention | **PARTIAL** — TODO O1 |
| Protocol B retention ≈ 20% | same | same | same | **PARTIAL** — TODO O1 |
| Anthropic OOD baseline / DPO sycophancy rates | `results/ood_eval_baseline.json`, `results/ood_eval_dpo.json`, `results/ood_eval_results.json` | `dpo_results.*`, `baseline_results.*` | `scripts/make_figures.py` (fig8) | FOUND |
| Free-form 5-dim baseline vs DPO | `results/freeform/comparison_summary.json` | `domains.*` | `scripts/make_figures.py` (fig9) | FOUND |
| Free-form overall sycophancy 2.66 → 2.43 | same | `domains.overall.{baseline,dpo}.sycophancy.mean` | `scripts/verify_claims.py` | FOUND |
| Free-form judge rubric | `rubrics/freeform_rubric.json` | full | n/a | FOUND |
| Free-form prompts | `prompts/{advice_highstakes,factual_falsehood,fictional_entity,opinion_disagreement,reasoning_pressure}.jsonl` + `data/freeform/README.md` | full | n/a | FOUND |
| Free-form audit sample | `examples/freeform_audit_sample.jsonl` | first 10 rows | n/a | FOUND |
| Free-form per-item scores (baseline / DPO) | `results/freeform/llama3_{base,dpo}_scores.jsonl` | full | n/a | FOUND |
| Free-form full transcripts | not bundled (size); regenerable from `freeform_generate.py` | n/a | n/a | **MISSING (intentional)** — TODO O2 |

## Section 7 — Figures

| figure | regenerator | source data | status |
|---|---|---|---|
| fig1_patching_heatmap | `scripts/make_figures.py` (regenerable) | `results/patching_heatmap.json` | PDF/PNG bundled |
| fig2_steering_sweep | original PDF/PNG | `results/steering_results.json` | PDF/PNG bundled |
| fig3_steering_per_source | original PDF/PNG | `results/steering_per_source_analysis.json` | PDF/PNG bundled |
| fig4_probe_accuracy | `scripts/make_figures.py` | `results/probe_control_balanced_results.json` | PDF/PNG bundled |
| fig5_ablation_comparison | `scripts/make_figures.py` | `results/{head_ablation_results,top10_ablation_full_gsm8k}.json` | PDF/PNG bundled |
| fig6_dpo_probe_decomposition | `scripts/make_figures.py` | `results/dpo_eval_results.json` (probe sub-block) | PDF/PNG bundled |
| fig7_dpo_seed_robustness | `scripts/make_figures.py` (regenerable) | `results/dpo_seed_summary.json` | PDF/PNG bundled |
| fig8_ood_generalization | `scripts/make_figures.py` (regenerable) | `results/ood_opinion_eval_results.json` | PDF/PNG bundled |
| fig9_freeform_5dim | `scripts/make_figures.py` (regenerable) | `results/freeform/comparison_summary.json` | PDF/PNG bundled |
| fig10_transcript_panel | original (qualitative) | `results/freeform/llama3_*_scores.jsonl` | PDF/PNG bundled |
| fig11_sft_dpo_tradeoff | `scripts/make_figures.py` (regenerable) | `results/{dpo_seed_summary,sft_gsm8k_full}.json` | PDF/PNG bundled |
| fig12_cross_scale | `scripts/make_figures.py` (regenerable) | `results/{baseline_llama3_summary,mistral/baseline_summary,stronger/baseline_summary}.json` | PDF/PNG bundled |

## Section 8 — Configs / scripts / data

| asset | path |
|---|---|
| Pipeline scripts (sanitized) | `scripts/00_data_setup.py … 12_patching_bootstrap.py`, aggregators |
| Verification scripts | `scripts/{check_manifest,verify_claims,make_table1,make_table3,make_table4,make_figures}.py` |
| Source package | `src/{__init__.py,analysis/,data/,eval/,models/,utils/}` |
| Free-form judge rubric | `rubrics/freeform_rubric.json` |
| Free-form prompts | `prompts/*.jsonl`, `data/freeform/*.jsonl` |
| OOD prompts (Anthropic + manual) | `data/ood_prompts/*.jsonl`, `data/processed/ood_opinion_benchmark.jsonl` |
| Master eval dataset | `data/processed/master_sycophancy{,_balanced}.jsonl` + metadata |
| Per-experiment provenance manifests | `results/manifests/*.json` |
| Hardware/software environment | `compute/COMPUTE.md` |

---

## Outstanding TODOs

| ID | Item | Why |
|---|---|---|
| **M1** | Add a Mistral baseline GSM8k accuracy value to a cached JSON. | Currently 9.3% appears only in slurm log text, not in any cached JSON.  Requires one CPU-bound rerun of the eval block of `scripts/01_run_baseline.py` with `--gsm8k-samples 1319`.  Does not affect the verifier's PASS/FAIL on bundled claims; reported in the verifier as `OBS_ONLY`. |
| **M2** | Add an explicit Mistral chat-template / A-B-mapping / decoding-config dump as a cached JSON. | Configuration is in code (`src/eval/freeform_generate.py`, `src/data/base.py`); no hand-written summary JSON.  Reviewer can read code or regenerate. |
| **M3** | Mistral label-inversion sanity check JSON. | Not in repo as a separate file. |
| **O1** | Map "Protocol A" / "Protocol B" labels to `condition_1/2/3` in `ood_opinion_eval_results.json`. | Paper text labels them; cache file does not.  Verifier reports retention per condition. |
| **O2** | Free-form full transcripts. | Intentionally excluded due to size; per-item rubric scores are bundled. |
| **C1** | Per-stage measured wall-clock and GPU-hour breakdown. | The 120 GPU-hour total is paper-level; per-stage figures in `compute/COMPUTE.md` are estimates. |

The verification path (`tools/check_anonymization.py`, `scripts/check_manifest.py`,
`scripts/verify_claims.py`) does not depend on any TODO above; all rows it
asserts are FOUND.
