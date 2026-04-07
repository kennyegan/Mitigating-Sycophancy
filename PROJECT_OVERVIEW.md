# Project Overview

## Objective

Mechanistic interpretability analysis of sycophancy in RLHF-trained LLMs, with the goal of understanding and mitigating the behavior. Target venue: NeurIPS 2026.

## Current State (April 7, 2026) — ALL EXPERIMENTS COMPLETE

### Milestones 1-5: COMPLETE

All methodology hardening, probe redesign, reproducibility, capability evaluation, and test expansion milestones are complete. See `context.md` for detailed changelog.

### Milestone 6: Full Corrected Rerun Matrix — COMPLETE

- All 13 SLURM jobs complete, manifest `missing_count: 0`
- Mistral cross-architecture replication complete
- Corrected ablation (Job 16) complete — dissociation confirmed with validated top-3 heads
- Steering capability eval (Job 15) complete — L20 alpha=2: 96.9% MMLU, 87.3% GSM8k

### Milestone 7: Paper Synchronization — IN PROGRESS

- Paper numbers audited and corrected (6 discrepancies fixed Mar 17)
- Section 5.4 head table replaced with validated rankings
- Section 5.6.1 added with corrected ablation results
- Section 5.8 updated with per-source opinion steering signal and L15/L20 capability data
- Figure references (Fig 1-5) added to paper sections
- Publication figures generated (5 figures, PDF + PNG in `figures/`)
- Still needed: 2x2 probe table, multiple testing corrections, fictional-entity section expansion

### Milestone 8: DPO Training-Time Intervention — IN PROGRESS

- DPO training COMPLETE (Job 53801949, Mar 22, 2026)
  - Train loss: 0.69 → 0.16, rewards accuracy: 0.44 → 0.95
  - Eval loss: 0.42 (stable, no overfitting)
  - Adapter saved: `results/dpo_model/` (LoRA rank 16, 400 training pairs, seed=100)
- DPO eval COMPLETE (Job 55240703, Apr 7, 2026)
  - Behavioral: opinion syc 82.4%→58.6% (-23.8pp), MMLU 62.8% (+0.8pp), GSM8k 38.5% (+5.3pp)
  - Probe re-analysis: social compliance 18.0%→11.4% (-6.6pp), robust tracking 59.9%→75.5% (+15.6pp)
  - First mechanistic evidence of what DPO does to sycophancy circuits
  - Artifact: `results/dpo_eval_results.json`

## Completion Gates

- Gate A: Core refactors + tests — **COMPLETE**
- Gate B: Schema-valid outputs — **COMPLETE**
- Gate C: Full SLURM reruns + manifests — **COMPLETE**
- Gate D: Paper numbers confirmed — **COMPLETE** (audit Mar 16, fixes Mar 17-19)
- Gate E: Publication figures — **COMPLETE** (5 figures generated Mar 20)
- Gate F: DPO intervention + probe re-analysis — **COMPLETE** (Apr 7: SC 18%→11.4%, robust 60%→75.5%)
- Gate G: Final paper draft with all results — **COMPLETE** (DPO section, abstract, discussion, conclusion updated Apr 7)

## Key Results

| Finding | Status | Impact |
|---------|--------|--------|
| Social compliance > belief corruption (1.8:1) | Confirmed | Core contribution |
| Patching-to-ablation dissociation | Confirmed with correct heads (Mar 19) | Novel methodology finding |
| Domain-specific circuits (zero overlap) | Confirmed | Adds novelty |
| Cross-architecture replication (Mistral) | Confirmed | Strengthens claims |
| Opinion steering L20 alpha=2: -5.7pp, 96.9% MMLU | Confirmed | Modest positive result |
| DPO training converged (loss 0.69→0.16, accuracy 95%) | Complete | Training works |
| DPO behavioral: opinion syc 82.4%→58.6%, MMLU preserved | Complete | -23.8pp reduction |
| DPO probe: SC 18%→11.4%, robust 60%→75.5% | **Complete** | **First mechanistic DPO evidence** |
| Paper updated: Sec 5.11, Abstract, Discussion, Conclusion | **Complete** | Full story in paper |

## NeurIPS Readiness

| Milestone | Likelihood |
|-----------|-----------|
| Full pipeline + paper updated (all sections written) | ~80-85% |
| + Final polish pass (formatting, notation table) | ~82-85% |
