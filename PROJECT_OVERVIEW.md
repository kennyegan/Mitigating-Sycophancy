# Sycophancy is Belief Corruption, Not Social Compliance:
## A Mechanistic Interpretability Analysis of LLM Sycophancy Circuits

**Kenneth Egan** · kenegan2005@gmail.com · [github.com/kennyegan/Mitigating-Sycophancy](https://github.com/kennyegan/Mitigating-Sycophancy)

**Current Status:** Phases 1–4 Complete | Phase 5 In Progress (head ablation + probe control + control groups) | **Key Finding: Belief Corruption dominates at 99.8%**

**Last Updated:** 2026-03-02

---

## Research Goals

- **Distinguish Mechanisms:** Mechanistically differentiate **Social Compliance** from **Belief Corruption**. **Result: Belief Corruption dominates at 99.8%.**
- **Reasoning Benchmark:** Measure sycophancy across opinion, factual, and reasoning domains via the Reasoning-Sycophancy benchmark (1,500 samples). ✅ Complete.
- **Circuit Discovery:** Causally identify specific attention heads responsible for sycophancy via activation patching. **Result: L1H20, L5H5, L4H28 in layers 1–5.** ✅ Complete.
- **Causal Intervention:** Head ablation targeting the identified circuit to reduce sycophancy with <5% capability degradation on MMLU and GSM8k. 🔄 In Progress.
- **Artifacts:** Peer-reviewed paper (NeurIPS/ICLR target), public TransformerLens codebase, HuggingFace benchmark.

---

## Models

| Model | Role | Rationale |
|---|---|---|
| **Llama-3-8B-Instruct** | Primary | RLHF-trained; expected to show strong sycophancy |
| **Llama-3-8B-Base** | Comparison | No RLHF; tests whether RLHF introduces the sycophancy circuit |
| **Mistral-7B-Instruct** | Replication | Validates circuit generalization across model families |

> **Note:** gpt2-medium was used for initial dev/testing only. All paper results use Llama-3-8B-Instruct.

---

## Phase 1 — Infrastructure & Tech Stack ✅ COMPLETE

**Deliverables**
- [x] `SycophancyModel` class wrapper with TransformerLens hooks (`src/models/sycophancy_model.py`)
- [x] Activation caching pipeline (`get_activations`, `get_attention_patterns`)
- [x] Steering vector computation and hook-based generation (`compute_steering_vector`, `generate_with_steering`)
- [x] Device auto-detection (CUDA → MPS → CPU), float16, `fold_ln=False`

---

## Phase 2 — Reasoning-Sycophancy Benchmark ✅ COMPLETE

**1,500 samples across three sycophancy domains:**

| Domain | Source | N | Sycophancy Type |
|---|---|---|---|
| Math Reasoning | GSM8k + Logic Corruptor | 500 | Following corrupted arithmetic |
| Factual QA | TruthfulQA | 500 | Agreement with misconceptions |
| Opinion | Anthropic model-written-evals | 500 | Agreement with false user opinions |

**Deliverables**
- [x] Dataset processors: `src/data/anthropic.py`, `gsm8k_reasoning.py`, `truthful_qa.py`
- [x] Master dataset: `data/processed/master_sycophancy.jsonl` (1,500 samples)
- [x] Orchestrator: `scripts/00_data_setup.py`

**Control Groups** ✅ Data generated, 🔄 Analysis pending
- [x] **Uncertain Knowledge:** 68 samples filtered (<60% confidence on neutral prompt). Data at `data/processed/control_groups/uncertain_knowledge.jsonl`.
- [x] **Fictional Entities:** 100 samples about non-existent objects. Data at `data/processed/control_groups/fictional_entities.jsonl`.
- [x] **Adversarially-True Hints:** 387 samples where user asserts correct answer. Data at `data/processed/control_groups/adversarially_true.jsonl`.
- [ ] Run baseline + probes + patching on control group subsets (SLURM job pending).

---

## Phase 3 — Baseline Evaluation ✅ COMPLETE

**Primary Metric: Compliance Gap**
$$\Delta = P(\text{Sycophantic} \mid \text{Biased Prompt}) - P(\text{Sycophantic} \mid \text{Neutral Prompt})$$

**Key Results:**
- Overall sycophancy rate: **28.0%** [95% CI: 25.8%–30.3%]
- Opinion (anthropic): **82.4%** | Factual (TruthfulQA): **1.6%** | Reasoning (GSM8k): **0.0%**
- Base model: **36.7%** overall (higher than instruct) — RLHF redistributes, doesn't introduce sycophancy

**Deliverables**
- [x] Evaluation script: `scripts/01_run_baseline.py`
- [x] Statistics module: `src/analysis/evaluation.py` (Wilson CIs, Cohen's d, permutation tests, bootstrap)
- [x] Baseline on Llama-3-8B-Instruct, full 1,500-sample dataset
- [x] Per-domain breakdown: math vs. factual vs. opinion
- [x] Base vs. Instruct comparison (RLHF hypothesis)

---

## Phase 4 — Mechanistic Analysis ✅ COMPLETE

### 4.1 Linear Probes — **Result: Belief Corruption (99.8%)**

Logistic and ridge probes trained on `resid_post` at all 32 layers with 5-fold CV.

- Best logistic probe: **99.47%** accuracy (Layer 6)
- Best ridge probe: **99.60%** accuracy (Layer 2)
- **Belief Corruption rate: 99.76%** (logistic) / **99.29%** (ridge)
- Social Compliance rate: 0.24% — effectively zero

**Deliverables**
- [x] `scripts/02_train_probes.py` — 5-fold CV, layer 0 through 31
- [x] Results: `results/probe_results_llama3_logistic.json`, `probe_results_llama3_ridge.json`

### 4.2 Causal Activation Patching — **Result: Layers 1–5 critical**

- Mean total effect: **2.1050** (±2.7278)
- Critical layers: **1, 2, 3, 4, 5** (top 5 by importance score)

**Deliverables**
- [x] `scripts/03_activation_patching.py` — layer × token position heatmap
- [x] Results: `results/patching_heatmap.json`

### 4.3 Attention Head Analysis — **Result: L1H20, L5H5, L4H28**

Top 3 heads by recovery score:
1. **L1H20**: 0.5690 (±1.2114)
2. **L5H5**: 0.5669 (±0.6947)
3. **L4H28**: 0.5062 (±0.6719)

**Deliverables**
- [x] Head-level importance scores: `results/head_importance.json`
- [x] 100/100 samples successfully patched

### 4.4 Base vs. Instruct Comparison — **Result: RLHF redistributes sycophancy**

- Base model sycophancy: **36.7%** (higher than instruct 28.0%)
- Base model critical layers: 0–4 with lower effect size (0.93 vs 2.10)
- RLHF suppresses factual/reasoning sycophancy but concentrates it in opinion domain

**Deliverables**
- [x] Full pipeline on Llama-3-8B-Base
- [x] Results: `results/baseline_llama3_base_summary.json`, `results/base_model/`

---

## Phase 5 — Causal Intervention 🔄 IN PROGRESS

Since sycophancy is Belief Corruption (not Social Compliance), the intervention targets the early-layer circuit where corruption enters the residual stream. Starting with direct head ablation (simpler, faster, cleaner causal result) before full steering vector pipeline.

### 5.1 Head Ablation (primary)
Ablate L1H20, L5H5, L4H28 directly (zero-ablation + mean-ablation). Measure effect on opinion sycophancy rate + MMLU/GSM8k capability retention.

### 5.2 Probe Control Experiment (probe validity)
Train probes on neutral prompts only, test on biased prompts. Validates that 99.5% probe accuracy reflects genuine truth tracking, not prompt format classification.

### 5.3 Control Group Analysis
Run fictional entities, uncertain knowledge, adversarially-true subsets through baseline + probes + patching pipelines.

**Safety Validation**

| Metric | Requirement | Method |
|---|---|---|
| MMLU Accuracy | ≥95% of baseline | 500-question MMLU subset |
| GSM8k Accuracy | ≥95% of baseline | 200-question GSM8k subset |
| Sycophancy Rate | Reduction vs. baseline | Full 1,500-sample benchmark |

**Deliverables**
- [ ] `scripts/04_head_ablation.py` — zero + mean ablation, 9 conditions
- [ ] `scripts/02b_probe_control.py` — neutral-only probe training control
- [ ] SLURM jobs: `06_probe_control.sh`, `07_head_ablation.sh`, `08_control_analysis.sh`

---

## Phase 6 — Mistral-7B Replication ⏳ PENDING

Repeat Phases 3–5 on Mistral-7B-Instruct. Goal: validate circuit generalization across model families. Report differences as model-family specificity, not failure.

---

## Phase 7 — Statistical Validation & Reproducibility ⏳ PENDING

- Bootstrap resampling (N=10,000) for all CIs
- Cohen's *d* effect sizes for probe separation
- Bonferroni + Benjamini-Hochberg corrections for multiple comparisons
- Fixed seeds throughout; git hash captured in all result files

> Core infrastructure already implemented in `src/analysis/evaluation.py`

---

## Phase 8 — Manuscript & Release ⏳ PENDING

**Target Venues**
- **Primary:** NeurIPS / ICLR Main Track
- **Secondary:** ACL / EMNLP
- **Workshops:** AI Alignment / SoLaR

**Final Outputs**
- arXiv preprint: "Sycophancy is Belief Corruption, Not Social Compliance"
- Public GitHub repo: fully reproducible
- HuggingFace dataset: Reasoning-Sycophancy benchmark

---

## Pre-Committed Null Results (Section 7 of Proposal)

**Null result realized:** The Social Compliance hypothesis was falsified — Belief Corruption dominates at 99.8%. This is itself a significant contribution.

Remaining pre-committed conditions:
- If steering/ablation degrades capabilities >5% → report full results without cherry-picking
- If Mistral-7B shows a different causal structure → report as model-family specificity

---

## Critical Risks & Mitigations

| Risk | Mitigation |
|---|---|
| "It's just hallucination" | Fictional Entities control group |
| Probes are noisy | 5-fold CV; test Ridge vs. Logistic probe architectures |
| Steering breaks capabilities | Sweep α; report Pareto frontier; target circuit components only |
| Circuit doesn't replicate | Report Mistral-7B results honestly; model-family specificity is a valid finding |

---

## Timeline

| Milestone | Status |
|---|---|
| Infrastructure & `SycophancyModel` | ✅ Complete |
| Reasoning-Sycophancy Benchmark (1,500 samples) | ✅ Complete |
| Control group data generation | ✅ Complete (555 samples across 3 groups) |
| Baseline evaluation (Llama-3-8B-Instruct + Base) | ✅ Complete — 28.0% instruct, 36.7% base |
| Linear probes: Belief Corruption vs Social Compliance | ✅ Complete — 99.8% Belief Corruption |
| Causal tracing + head-level patching | ✅ Complete — L1H20, L5H5, L4H28 |
| Base vs. Instruct comparison | ✅ Complete — RLHF redistributes sycophancy |
| Probe control experiment (probe validity) | 🔄 In Progress |
| Head ablation intervention | 🔄 In Progress |
| Control group analysis (baseline + probes + patching) | 🔄 In Progress |
| Mistral-7B replication | ⏳ Pending |
| Statistical validation & reproducibility | ⏳ Pending |
| Manuscript submission-ready | ⏳ Pending |

---

## Infrastructure Reference

**TransformerLens Hook Names**
```
blocks.{layer}.hook_resid_post    # Main probing target
blocks.{layer}.hook_resid_pre
blocks.{layer}.attn.hook_pattern  # Attention weights
blocks.{layer}.hook_attn_out
blocks.{layer}.hook_mlp_out
```

**Key Flags**
- `fold_ln=False` — unfolded layer norms for interpretability
- `dtype=float16` — memory efficiency on A100
- Seed=42 throughout; git hash in all result files

**Quick Commands**
```bash
make setup          # Install dependencies
make data           # Download full benchmark (500/type)
make data-small     # Dev dataset (50/type)
make baseline       # Compliance gap evaluation
make test           # Pytest suite
```
