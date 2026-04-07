# Project Overview

## Objective

Mechanistic interpretability analysis of sycophancy in RLHF-trained LLMs, with the goal of understanding and mitigating the behavior. Target venue: NeurIPS 2026.

## Current State (March 22, 2026)

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
- DPO eval SUBMITTED (Job 53811183) — behavioral eval + MMLU/GSM8k + probe re-analysis
- Awaiting eval results to complete Phase 4

## Completion Gates

- Gate A: Core refactors + tests — **COMPLETE**
- Gate B: Schema-valid outputs — **COMPLETE**
- Gate C: Full SLURM reruns + manifests — **COMPLETE**
- Gate D: Paper numbers confirmed — **COMPLETE** (audit Mar 16, fixes Mar 17-19)
- Gate E: Publication figures — **COMPLETE** (5 figures generated Mar 20)
- Gate F: DPO intervention + probe re-analysis — **IN PROGRESS** (training done, eval submitted Mar 22)
- Gate G: Final paper draft with all results — **BLOCKED on Gate F**

## Key Results

| Finding | Status | Impact |
|---------|--------|--------|
| Social compliance > belief corruption (1.8:1) | Confirmed | Core contribution |
| Patching-to-ablation dissociation | Confirmed with correct heads (Mar 19) | Novel methodology finding |
| Domain-specific circuits (zero overlap) | Confirmed | Adds novelty |
| Cross-architecture replication (Mistral) | Confirmed | Strengthens claims |
| Opinion steering L20 alpha=2: -5.7pp, 96.9% MMLU | Confirmed | Modest positive result |
| DPO training converged (loss 0.69→0.16, accuracy 95%) | Complete | Training works |
| DPO eval + probe re-analysis | **Running (Job 53811183)** | **Paper-making experiment** |

## NeurIPS Readiness

| Milestone | Likelihood |
|-----------|-----------|
| Current (all inference-time experiments done) | ~55-65% |
| + Figures + statistical corrections | ~60-70% |
| + DPO mitigation + mechanistic probe analysis | ~75-80% |
