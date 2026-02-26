# Sycophancy is Social Compliance, Not Belief Corruption:
## A Mechanistic Interpretability Analysis of LLM Sycophancy Circuits

**Kenneth Egan** · kenegan2005@gmail.com · [github.com/kennyegan/Mitigating-Sycophancy](https://github.com/kennyegan/Mitigating-Sycophancy)

**Current Status:** Phase 1-2 Complete | Phase 3 In Progress (baseline on Llama-3-8B pending) | **Next: Control groups + Phase 4**

**Last Updated:** 2026-02-20

---

## Research Goals

- **Distinguish Mechanisms:** Mechanistically differentiate **Social Compliance** (outputting falsehoods while retaining truth internally) from **Belief Corruption** (internal truth representations degraded by user hint).
- **Reasoning Benchmark:** Measure sycophancy in Chain-of-Thought reasoning via the Reasoning-Sycophancy benchmark (1,500 samples).
- **Circuit Discovery:** Causally identify specific attention heads responsible for sycophantic output suppression via activation patching.
- **Inference-Time Mitigation:** Develop steering vectors targeting only the identified circuit components, with <5% capability degradation on MMLU and GSM8k.
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

**Control Groups — ⚠️ NOT YET IMPLEMENTED (Required for paper)**
- [ ] **Uncertain Knowledge:** Filter for questions where Llama-3-8B-Instruct has <60% confidence on neutral prompt. Isolates sycophancy from uncertainty-driven agreement.
- [ ] **Fictional Entities:** Synthetic questions about non-existent objects. Distinguishes sycophancy from hallucination.
- [ ] **Adversarially-True Hints:** User provides a false hint that matches model's pre-existing bias. Distinguishes genuine persuasion from bias reinforcement.

---

## Phase 3 — Baseline Evaluation 🔄 IN PROGRESS

**Primary Metric: Compliance Gap**
$$\Delta = P(\text{Sycophantic} \mid \text{Biased Prompt}) - P(\text{Sycophantic} \mid \text{Neutral Prompt})$$

**Deliverables**
- [x] Evaluation script: `scripts/01_run_baseline.py`
- [x] Statistics module: `src/analysis/evaluation.py` (Wilson CIs, Cohen's d, permutation tests, bootstrap)
- [x] Dev results (gpt2-medium, 50 samples): 60% sycophancy rate — for development only, not paper results
- [ ] **Baseline on Llama-3-8B-Instruct, full 1,500-sample dataset** ← next immediate task
- [ ] Per-domain breakdown: math vs. factual vs. opinion
- [ ] Base vs. Instruct comparison (RLHF hypothesis)

---

## Phase 4 — Mechanistic Analysis ⏳ PENDING

### 4.1 Linear Probes (Social Compliance vs. Belief Corruption)

Train logistic regression probes on `resid_post` activations at each layer from **neutral prompt** forward passes. Decode the "truth direction."

Then evaluate on **sycophantic prompt** runs:

| Pattern | Probe Accuracy | Output Accuracy | Interpretation |
|---|---|---|---|
| Social Compliance | High | Low | Model retains truth but suppresses it |
| Belief Corruption | Low | Low | User hint degrades internal truth |
| Robust (control) | High | High | No sycophancy |

**Deliverables**
- [ ] `scripts/02_train_probes.py` — 5-fold CV, layer 0 through n_layers
- [ ] Layer-wise probe accuracy curves (probe acc vs. output acc per layer)

### 4.2 Causal Activation Patching

1. **Clean run:** Neutral prompt → cache activations at all layers/positions
2. **Corrupted run:** Sycophantic prompt → record sycophantic output
3. **Patch:** For each (layer L, position T), swap corrupted → clean activations; measure output recovery

**Deliverables**
- [ ] `scripts/03_activation_patching.py` — layer × token position heatmap
- [ ] Identification of critical layers responsible for sycophancy

### 4.3 Attention Head Analysis

Within critical layers from 4.2, perform head-level patching:
- Do sycophancy heads preferentially attend to user-hint tokens?
- Do they suppress attention to evidence tokens?
- Logit lens analysis: what do sycophancy heads write to the residual stream?

**Deliverables**
- [ ] Head-level importance scores
- [ ] Logit lens visualization on sycophancy heads

### 4.4 Base vs. Instruct Comparison

Run all of 4.1–4.3 on Llama-3-8B-Base. Hypothesis: RLHF introduces the sycophancy circuit; base model shows Belief Corruption or no effect.

---

## Phase 5 — Inference-Time Steering ⏳ PENDING

Grounded in mechanistically-identified circuit from Phase 4. Targets **only** the identified components, not the full residual stream.

1. Compute sycophancy vector: `mean(activations | sycophantic) − mean(activations | neutral)` at critical layers
2. Subtract `α × sycophancy_vector` at identified layers during generation
3. Sweep `α ∈ {0.1, 0.5, 1.0, 2.0, 5.0}`

**Safety Validation**

| Metric | Requirement | Method |
|---|---|---|
| MMLU Accuracy | ≥95% of baseline | 500-question MMLU subset |
| GSM8k Accuracy | ≥95% of baseline | Full GSM8k test set |
| Refusal Rate | No increase | 500 neutral prompts audit |
| Compliance Gap Δ | Reduction vs. baseline | Full 1,500-sample benchmark |

**Deliverables**
- [ ] `scripts/04_steering_vectors.py` — compute, save, apply, alpha sweep
- [ ] `scripts/05_safety_evaluation.py` — MMLU, GSM8k, refusal audit
- [ ] Pareto frontier plot: sycophancy reduction vs. capability retention

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
- arXiv preprint: "Sycophancy is Social Compliance, Not Belief Corruption"
- Public GitHub repo: fully reproducible
- HuggingFace dataset: Reasoning-Sycophancy benchmark

---

## Pre-Committed Null Results (Section 7 of Proposal)

We will fully report any of the following regardless of outcome:
- If probe accuracy is low in both conditions → linear probes insufficient; report implications
- If causal tracing shows no localized critical layers → sycophancy is distributed; report as evidence against circuit localization
- If steering degrades capabilities >5% → report full Pareto frontier without cherry-picking
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

| Milestone | Target | Status |
|---|---|---|
| Infrastructure & `SycophancyModel` | Week 2 | ✅ Complete |
| Reasoning-Sycophancy Benchmark (1,500 samples) | Week 4 | ✅ Complete |
| **Control groups** (uncertain, fictional, adversarial) | Week 5 | ⚠️ Missing |
| Baseline evaluation (Llama-3-8B-Instruct, full dataset) | Week 6 | 🔄 In Progress |
| Linear probes: layer-wise truth direction | Week 8 | ⏳ Pending |
| Causal tracing heatmaps: critical layer identification | Week 10 | ⏳ Pending |
| Head-level patching: sycophancy circuit pinpointed | Week 13 | ⏳ Pending |
| Base vs. Instruct comparison | Week 14 | ⏳ Pending |
| Mistral-7B replication | Week 16 | ⏳ Pending |
| Steering vectors: compute, apply, alpha sweep | Week 18 | ⏳ Pending |
| Safety evaluation: MMLU, GSM8k, refusal audit | Week 19 | ⏳ Pending |
| Statistical validation & reproducibility package | Week 21 | ⏳ Pending |
| Manuscript submission-ready | Week 22 | ⏳ Pending |

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
