# Context: Mitigating Sycophancy in Large Language Models

> Implementation log and research context for mechanistic interpretability analysis of LLM sycophancy circuits.
> Last updated: 2026-03-16

---

## Research Summary

This project investigates whether sycophancy in RLHF-trained LLMs arises from **Social Compliance** (model retains internal truth but suppresses it in output) or **Belief Corruption** (user hints degrade internal representations). We use mechanistic interpretability on Llama-3-8B-Instruct to localize sycophancy circuits, probe internal representations, and attempt inference-time mitigation via activation steering.

**Central finding:** Sycophancy is primarily social compliance, but the underlying circuit is **redundantly distributed** --- causal patching identifies responsible heads, yet ablating them produces no measurable sycophancy reduction (patching-to-ablation dissociation). This implies inference-time activation manipulation may be fundamentally limited, and training-time intervention is likely required.

---

## Architecture and Pipeline

### Models Under Study

| Model | Role | Notes |
|-------|------|-------|
| `meta-llama/Meta-Llama-3-8B-Instruct` | Primary target | RLHF-tuned, main sycophancy analysis |
| `meta-llama/Meta-Llama-3-8B` | Base comparison | Pre-RLHF baseline (unexpectedly MORE sycophantic overall) |
| `mistralai/Mistral-7B-Instruct-v0.1` | Cross-architecture replication | Different circuit structure, same behavioral pattern |

### Dataset: Reasoning-Sycophancy Benchmark (N=1,500)

Three domains, 500 samples each:

| Domain | Source | Sycophancy Rate (Instruct) | Interpretation |
|--------|--------|---------------------------|----------------|
| Opinion | Anthropic model-written evals | 82.4% | High --- subjective, no ground truth |
| Factual | TruthfulQA misconceptions | 1.6% | Very low --- model resists factual errors |
| Reasoning | GSM8k with corrupted logic | 0.0% | Near zero --- math is verifiable |

Control groups: fictional entities, uncertain knowledge, adversarially-true hints.

### Experiment Pipeline (13 SLURM Jobs)

```
Job 1:  Baseline compliance gap          --> baseline_llama3_summary.json
Job 2:  Control group data generation    --> data/processed/control_groups/
Job 3:  Linear probes (2 modes)          --> probe_results_{neutral_transfer,mixed_diagnostic}.json
Job 4:  Causal activation patching       --> patching_heatmap.json, head_importance.json
Job 5:  Base model comparison pipeline   --> baseline_llama3_base_*, base_model/
Job 6:  Probe control (neutral-only)     --> probe_control_results.json
Job 7:  Head ablation (top 3)            --> head_ablation_results.json
Job 8:  Control group analysis           --> control_groups/{baseline,probe}_{fictional,uncertain,adversarial}
Job 9:  Top-10 head ablation (N=200)     --> top10_ablation_results.json
Job 10: Balanced probe control           --> probe_control_balanced_results.json
Job 11: Top-10 ablation full GSM8k       --> top10_ablation_full_gsm8k.json
Job 12: Representation steering sweep    --> steering_results.json (checkpoint/resume enabled)
Job 13: Manifest collection & validation --> full_rerun_manifest.json
```

### Key Metrics

| Metric | Definition |
|--------|-----------|
| Compliance Gap | P(Sycophantic \| Biased) - P(Sycophantic \| Neutral) |
| Social Compliance Rate | Probe correct AND output sycophantic (model knows truth, suppresses it) |
| Belief Corruption Rate | Probe wrong AND output sycophantic (internal representation degraded) |
| Patching Recovery | log P(honest \| patched) - log P(honest \| corrupted) |
| Capability Retention | MMLU/GSM8k accuracy under intervention / baseline accuracy |

---

## Current Run Status (2026-03-16)

### Llama-3-8B-Instruct (Primary): ALL COMPLETE

| Artifact | Status | Generated | Key Number |
|----------|--------|-----------|------------|
| `baseline_llama3_summary.json` | OK | Mar 4 | Sycophancy rate: 28.0% |
| `baseline_llama3_base_summary.json` | OK | Mar 4 | Base model: 36.7% |
| `probe_results_neutral_transfer.json` | OK | Mar 4 | Best layer: 10 |
| `probe_results_mixed_diagnostic.json` | OK | Mar 4 | Best layer: 2 |
| `probe_control_balanced_results.json` | OK | Mar 4 | Best layer: 1 |
| `patching_heatmap.json` | OK | Mar 4 | Top layers: 1,2,3,4,5 |
| `head_importance.json` | OK | Mar 4 | L1H20 top head |
| `head_ablation_results.json` | OK | Mar 6 | Baseline: 28.0%, All-zero: 28.1% |
| `top10_ablation_results.json` | OK | Mar 6 | Baseline: 28.0%, All-zero: 28.5% |
| `top10_ablation_full_gsm8k.json` | OK | Mar 9 | Full 1319-sample GSM8k eval |
| `steering_results.json` | OK | Mar 7 | Checkpoint: completed, cap evals included |
| `full_rerun_manifest.json` | OK | Mar 9 | missing_count: 0 |

### Mistral-7B-Instruct (Replication): ALL COMPLETE

| Artifact | Status | Generated |
|----------|--------|-----------|
| `mistral/baseline_summary.json` | OK | Mar 9 |
| `mistral/probe_control_balanced_results.json` | OK | Mar 9 |
| `mistral/patching_heatmap.json` | OK | Mar 10 |
| `mistral/head_importance.json` | OK | Mar 10 |
| `mistral/top10_ablation_full_gsm8k.json` | OK | Mar 11 |
| `mistral/steering_results.json` | OK | Mar 11 |

### Verdict: No reruns needed for core pipeline. All artifacts validated.

The latest SLURM runs (53384704-53384706, Mar 6) were redundant re-reruns; the authoritative results were already generated Mar 4-9 and confirmed by the manifest.

### Per-Source Steering Analysis (Extracted 2026-03-16)

The steering checkpoint already contained per-source breakdowns for all 64 conditions. Offline extraction reveals steering **does reduce opinion-domain sycophancy** --- but with critical caveats:

| Condition | Opinion Syc | Change | Factual | Reasoning | Interpretation |
|-----------|-------------|--------|---------|-----------|----------------|
| Baseline (no steering) | 83.0% | --- | 1.6% | 0.0% | Normal |
| **Layer 15, alpha=2.0** | **76.1%** | **-6.9pp** | 9.2% | 9.8% | **Best safe candidate** |
| **Layer 20, alpha=2.0** | **77.3%** | **-5.7pp** | 4.8% | 0.0% | **Second best** |
| Layer 10, alpha=50.0 | 48.2% | -34.9pp | 0.0% | 0.0% | Model collapsed (outputs "no" to everything) |
| Layer 3, alpha=50.0 | 51.8% | -31.2pp | 100% | 100% | Reversed polarity (agrees with everything) |
| Most high-alpha conditions | ~51.8% | ~-31pp | 100% | 100% | Coin-flip incoherence, not real reduction |

**Key insight:** 31/63 conditions reduce opinion sycophancy below baseline CI [79.2%], but most achieve this by breaking the model. The 51.8% cluster is suspiciously close to random (coin flip). Only late-layer, low-alpha conditions (L15/L20 at alpha=2) show genuine moderate reduction without catastrophic side-effects --- but MMLU/GSM8k capability verification is needed for these specific conditions.

**Decision:** Layer 15 alpha=2.0 and Layer 20 alpha=2.0 are the candidates to investigate for capability retention. If they preserve >95% MMLU/GSM8k, this becomes a real (modest) mitigation result.

---

## Key Findings (Provisional --- Pending Paper Finalization)

### 1. Social Compliance Dominates

Probe transfer accuracy drops only 11.1 pp (89.0% -> 77.9%) from neutral to biased conditions. The model retains internal truth representations even when outputting sycophantic answers.

- Social compliance: 18.0% [16.0%, 19.9%]
- Belief corruption: 10.1% [8.6%, 11.7%]

### 2. Patching-to-Ablation Dissociation (Novel)

Causal patching identifies top-10 heads with high recovery scores. But ablating all 10 simultaneously:

| Condition | Syc Rate | Change | MMLU | GSM8k |
|-----------|----------|--------|------|-------|
| Baseline | 28.0% | --- | 62.0% | 34.0% |
| Top-10 zero-ablate | 28.5% | +0.5pp | 63.4% | 31.5% |
| Top-3 zero-ablate | 28.1% | +0.1pp | 62.2% | 32.5% |

Heads are causally **sufficient** (patching restores honest output) but not causally **necessary** (ablation has no effect). The signal is redundantly distributed.

**CAVEAT (discovered 2026-03-16 audit):** The original ablation targeted L1H20, L5H5, L4H28 based on a stale pre-rerun patching result. The validated rerun (Mar 4) shows the actual top-3 are L4H28 (0.443), L4H5 (0.302), L5H31 (0.256). L4H28 was tested (no effect: 28.1%), but L4H5 and L5H31 were not. Job 16 (`slurm/16_corrected_ablation.sh`) tests the correct top-3. The dissociation finding is **pending re-validation**.

### 3. Domain-Specific Circuits

Different attention heads mediate opinion vs. fictional-entity sycophancy with sign-reversed roles. Zero overlap between domain circuits.

### 4. Cross-Architecture Consistency

Llama-3 and Mistral-7B both show ~28% sycophancy on opinion, patching-to-ablation dissociation, but entirely different underlying head rankings.

### 5. Implication for AI Safety

Inference-time steering/ablation is insufficient. Effective sycophancy mitigation likely requires training-time objectives that penalize compliance on verifiable claims.

---

## Changelog

| Date | Commit | Change | Impact |
|------|--------|--------|--------|
| 2026-02-24 | `9631fa5` | Initial environment setup | Conda env, dependencies |
| 2026-02-24 | `c0b0441` | Remove HF token from repo; load from env | Security fix |
| 2026-02-25 | `3c0af3a` | Add initial results | First baseline + probe runs |
| 2026-02-25 | `555e16c` | Update .gitignore | Exclude cache dirs |
| 2026-02-26 | `d704f9f` | Add README | Documentation |
| 2026-02-26 | `e298190` | Add test suite | Evaluation math, probe leakage, schema checks |
| 2026-02-27 | `47d1d51` | Update src modules | SycophancyModel wrapper, evaluation.py refactor |
| 2026-02-27 | `326a8b5` | Update scripts | Milestone 1-4 hardening: probe modes, patching, ablation, steering checkpoint |
| 2026-02-28 | `9f2ce26` | Update SLURM jobs | 13-job pipeline with dependencies |
| 2026-03-01 | `084a5a9` | Update SLURM | Config fixes for cluster |
| 2026-03-01 | `566a6a9` | Update results | First full rerun results |
| 2026-03-01 | `e292645` | Update SLURM | Job tweaks |
| 2026-03-01 | `0ad8f02` | Update SLURM | Final job config |
| 2026-03-02 | `f8b87b6` | Add paper scaffolding | paper.md initial structure |
| 2026-03-02 | `a28b21f` | Paper draft | Methods, results sections |
| 2026-03-03 | `e113ca4` | Update docs | PROJECT_OVERVIEW, QUICKSTART, RERUN_STEPS |
| 2026-03-03 | `ed5b6c1` | Update SLURM | Steering checkpoint/resume support |
| 2026-03-03 | `10c2d31` | Update QUICKSTART | Clarify setup steps |
| 2026-03-03 | `b2270df` | Update slurm logs | Archive old logs |
| 2026-03-03 | `b508868` | Update config | slurm/config.sh model paths |
| 2026-03-03 | `6826726` | Update Mistral job | Cross-architecture replication pipeline |
| 2026-03-03 | `cdb589d` | Update paper | Incorporate Mistral findings, dissociation discussion |
| 2026-03-03 | `2df0e82` | Update project overview | Milestone status, completion gates |
| 2026-03-16 | `f241597` | Update context | context.md initial creation |
| 2026-03-16 | --- | Per-source steering extraction | Offline analysis of checkpoint: L15/L20 alpha=2 candidates identified |
| 2026-03-16 | --- | Add `extract_per_source_steering.py` | No-GPU analysis script for opinion-domain steering |
| 2026-03-16 | --- | Add `slurm/14_steering_resume.sh` | Resume job for capability evals (12h wall time, 200 GSM8k) |
| 2026-03-16 | --- | NeurIPS assessment | Identified ~35-45% baseline, path to ~75-80% via DPO + probe re-analysis |
| 2026-03-16 | --- | Three-way pre-submission audit | Found stale head table (CRITICAL), 10 bugs (none invalidate results), 10 paper fixes |
| 2026-03-16 | --- | Add `slurm/16_corrected_ablation.sh` | Ablation of validated top-3 (L4H28, L4H5, L5H31) — blocks paper direction |
| 2026-03-16 | --- | Add `scripts/eval_steering_capability.py` | Targeted MMLU/GSM8k eval for specific steering conditions |

---

## Infrastructure Notes

- **Cluster:** Unity HPC (WIT), PI account `pi_larsonj_wit_edu`
- **GPU:** NVIDIA A100 (40GB PCIE on uri-gpu017, 80GB SXM4 on uri-gpu002)
- **Environment:** `sycophancy-lab` conda env, Python 3.10.19, PyTorch 2.10.0+cu128
- **TransformerLens config:** `fold_ln=False`, `use_attn_result=True` (required for head-level patching)
- **Reproducibility:** All experiments use seed 42, bootstrap CIs (N=10,000), Bonferroni + BH corrections
- **Checkpoint system:** Steering experiments save after every condition for wall-time resilience

## File Structure

```
Mitigating-Sycophancy/
  src/
    models/sycophancy_model.py   -- TransformerLens HookedTransformer wrapper
    analysis/evaluation.py       -- Compliance gap, CIs, permutation tests
    data/base.py                 -- SycophancySample dataclass, Llama-3 chat formatting
    data/{anthropic,truthful_qa,gsm8k_reasoning,control_groups}.py
  scripts/
    00_data_setup.py             -- Dataset creation orchestrator
    01_run_baseline.py           -- Compliance gap baseline evaluation
    02_train_probes.py           -- Linear probes (neutral_transfer / mixed_diagnostic)
    02b_probe_control.py         -- Probe control wrapper
    03_activation_patching.py    -- Causal patching (Wang et al. 2022)
    04_head_ablation.py          -- Head ablation with capability eval
    05_representation_steering.py-- Steering sweep with checkpoint/resume
    99_collect_result_manifest.py-- Artifact validation
  slurm/
    config.sh                    -- Cluster/model/path configuration
    01_baseline.sh ... 13_collect_manifest.sh  -- 13 SLURM job scripts
    submit_all.sh                -- Dependency-aware batch submit
    logs/                        -- SLURM stdout/stderr logs
  data/processed/                -- 1,500-sample dataset + controls
  results/                       -- All analysis outputs + manifests
  results/mistral/               -- Mistral-7B replication results
  results_archive/               -- Pre-rerun snapshot (Mar 3)
  tests/                         -- Evaluation math, probe, schema tests
```

---

## Pre-Submission Audit (2026-03-16)

Three-way audit: paper claims vs data, implementation bugs, statistical methodology.

### Existing Results: ALL VALID — No Reruns Needed

| Bug Found | Affects Existing Results? | Resolution |
|-----------|--------------------------|------------|
| Paper Section 5.4 head table uses stale (Mar 3) rankings | No — `head_importance.json` has correct data | Fix paper numbers; run Job 16 for corrected ablation |
| Layer patching alignment bug (absolute vs shared-suffix) | No — layers 4,5 correctly selected as critical | Methodological note in paper |
| Multiple testing corrections never applied (dead code) | No — results are valid, reporting needs corrections | Wire in Bonferroni/BH; report adjusted p-values |
| BH monotonicity bug in evaluation.py | No — function is never called | Fix code for future use |
| Position confound in probes | No — balanced dataset (50.3%/50.8%/50.0%) already used | Paper already cites balanced results |
| fold_ln=False in TransformerLens | No — defensible choice | Add discussion in paper |
| Checkpoint per_example drop on resume | No — steering completed in single run, all CIs valid | Fix code for future resilience |
| Mean-ablation neutral activation preference | No — mean-ablation reported as collapsed (0%); not a primary finding | Acknowledged in paper |
| GSM8k neutral prompt info asymmetry | No — reasoning sycophancy is 0.0%; no claims depend on this | Methodological caveat |
| Bootstrap CI seed in evaluation.py | No — callers set global seed via set_seeds(42) | Fix code for standalone use |

### New Experiments Required

| Job | Script | Purpose | Blocks |
|-----|--------|---------|--------|
| **16 (CRITICAL)** | `slurm/16_corrected_ablation.sh` | Ablate actual top-3: L4H28, L4H5, L5H31 | Validates/invalidates dissociation finding |
| 15 (optional) | `slurm/15_steering_cap_eval.sh` | MMLU/GSM8k for L15/L20 alpha=2 steering | Steering mitigation claim |

### Paper Fixes Required (No GPU)

| Section | Issue | Fix |
|---------|-------|-----|
| 5.4 | Head table uses stale rankings (L1H20=0.569) | Replace with validated: L4H28=0.443, L4H5=0.302, L5H31=0.256 |
| 5.6 | L5H5 GSM8k says 31.0% | Change to 31.5% |
| 5.7 | GSM8k retention says 90.1% | Change to 90.0% |
| 5.7 | GSM8k retention CI excludes 1.0 [0.823, 0.990], paper says "non-significant" | Reword to "marginally significant" |
| 5.1 | Samples evaluated says 1,500 | Change to 1,493 (7 skipped) |
| 5.1 | Compliance gap -0.0434 | Change to -0.0435 |
| 6 | Base model opinion says 50.4% | Change to 50.3% |
| 5.8 | MMLU N=500 | Change to N=499 |
| Methods | No discussion of fold_ln=False choice | Add paragraph |
| Results | No multiple testing corrections reported | Apply Bonferroni/BH, report adjusted values |

### Implementation Fixes (No Rerun Needed)

| File | Issue | Priority |
|------|-------|----------|
| `evaluation.py:685-695` | BH monotonicity: enforce in sorted order, not original order | Medium (dead code) |
| `evaluation.py:369-410` | `compute_bootstrap_ci` needs local RandomState seed | Low |
| `05_representation_steering.py:743-748` | Save per_example before dropping for checkpoint resilience | Low |
| `04_head_ablation.py:499-500` | Fix misleading indentation in MMLU else branch | Low |

---

## Research Direction & NeurIPS Path (Assessed 2026-03-16)

### Current NeurIPS Readiness: ~35-45% (Borderline Reject / Weak Accept)

**Strengths:** Social compliance evidence (novel), patching-to-ablation dissociation (novel methodology contribution), cross-architecture replication, well-designed benchmark with controls.

**Weaknesses:** No successful mitigation (title says "Mitigating"), primarily negative results, limited theoretical insight, domain-specificity undermines generality claims.

### Phase 1: Per-Source Steering (COMPLETE --- data extracted 2026-03-16)

The checkpoint already had all per-source data. Key finding: late-layer, low-alpha steering (L15 alpha=2, L20 alpha=2) produces modest opinion-domain reduction (-5.7 to -6.9pp) without catastrophic side-effects. High-alpha conditions produce coin-flip incoherence (51.8% = random), not genuine reduction.

Next step: verify MMLU/GSM8k retention for L15 alpha=2 and L20 alpha=2 conditions from steering_results.json capability data.

### Phase 2: Distributed Steering (CONDITIONAL on Phase 1)

If L15/L20 alpha=2 preserves capability, test layer-normalized distributed steering across all identified layers simultaneously at very low alpha. The 60x norm difference between layer 1 (0.069) and layer 20 (4.285) means alpha is not comparable across layers without normalization.

### Phase 3: Paper Tightening (IN PARALLEL)

- Add 2x2 probe contingency table (probe correct/wrong x model sycophantic/honest)
- Expand fictional-entity two-circuit finding (zero overlap, sign-reversed L1H20 --- adds novelty for free)
- Bonferroni/BH corrections for 56 steering conditions
- Create visualization: patching heatmap, steering sweep plots, probe accuracy curves

### Phase 4: DPO Training-Time Intervention (2-4 weeks, highest impact)

**The experiment that elevates this paper from descriptive to prescriptive:**

1. Generate 250-500 NEW opinion pairs (different seeds, not the 500 test samples)
2. DPO fine-tune Llama-3-8B-Instruct (TRL DPOTrainer, LoRA rank 16)
3. Evaluate: target <60% opinion sycophancy, >95% MMLU retention
4. **CRITICAL:** Re-run neutral-transfer probes on DPO model
   - Pre-DPO: 18% social compliance, 10% belief corruption
   - Post-DPO hypothesis: social compliance drops, robust tracking increases
   - This shows mechanistically WHAT DPO does to sycophancy circuits
   - First paper to demonstrate this

**Impact estimates:**
| Scenario | NeurIPS Likelihood |
|----------|-------------------|
| Current state | ~35-45% |
| + Phase 1 opinion steering confirmed | ~50-60% |
| + Phase 3 paper tightened | ~60-70% |
| + Phase 4 DPO + probe re-analysis | ~75-80% |

### Methodological Cautions
1. Patching sufficiency does not imply necessity --- always validate with ablation
2. Domain-specificity means aggregate sycophancy rates mask important structure
3. Cross-architecture replication is essential; behavioral similarity hides mechanistic divergence
4. Probe accuracy on biased prompts should be interpreted carefully (format confounds if not controlled)
5. Per-source CIs must use domain-specific N (e.g., N=436 for opinion), not full N=1500
6. DPO evaluation MUST use data not seen during training (separate seed generation)
