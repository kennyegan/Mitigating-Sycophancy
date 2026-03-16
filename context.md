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

## Future Work and Open Questions

### Immediate (Paper Completion)
- Finalize paper.md numerical claims with validated rerun numbers
- Add Mistral cross-architecture comparison tables
- Complete discussion of patching-to-ablation dissociation implications

### Near-Term Extensions
- **Logit lens analysis** on identified sycophancy heads (what do they write to residual stream?)
- **Attention pattern visualization** (do sycophancy heads attend to user-hint tokens?)
- **Layer-by-layer probe accuracy curves** to map where social compliance emerges
- **Ablation on opinion-only subset** (where sycophancy is 82.4%) rather than full dataset

### Medium-Term Research Directions
- **Training-time interventions:** DPO/RLHF objective modifications that penalize compliance on verifiable claims
- **Activation addition** (Turner et al. 2023) as alternative to subtraction steering
- **Larger models:** Scale analysis (does circuit redundancy increase with model size?)
- **Multi-turn sycophancy:** Extend benchmark to conversational settings where sycophancy compounds
- **Causal scrubbing** (Chan et al. 2022) for more rigorous circuit validation

### Methodological Cautions
1. Patching sufficiency does not imply necessity --- always validate with ablation
2. Domain-specificity means aggregate sycophancy rates mask important structure
3. Cross-architecture replication is essential; behavioral similarity hides mechanistic divergence
4. Probe accuracy on biased prompts should be interpreted carefully (format confounds if not controlled)
